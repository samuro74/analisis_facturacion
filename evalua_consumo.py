from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- CONFIGURACIÓN ---------------------------

RUTA_ARCHIVO = Path("suministro_insumos_y_medicamentos.csv")
CARPETA_SALIDA = RUTA_ARCHIVO.parent / "auditoria_consumos"
ARCHIVO_EXCEL = CARPETA_SALIDA / "reporte_auditoria_consumos.xlsx"

# Mínimo de días con registros requerido para comparar un día contra el patrón
# histórico del mismo paciente y producto.
MIN_DIAS_BASELINE = 2

# Se considera señal de incremento si la cantidad diaria supera simultáneamente:
# 1) 2 veces la mediana histórica, y 2) la mediana + 3 desviaciones absolutas medianas.
# Para productos con mediana histórica cero, se aplica una regla distinta.
FACTOR_INCREMENTO = 2.0
MULTIPLICADOR_MAD = 3.0

# --------------------------- FUNCIONES ---------------------------

def normalizar_texto(serie):
    return serie.astype("string").str.strip().replace("", pd.NA)


def nombre_hoja_excel(nombre):
    nombre = re.sub(r"[\[\]:*?/\\]", "_", str(nombre))
    return nombre[:31]


def preparar_datos(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}\n"
            "Revise RUTA_ARCHIVO y confirme que el CSV esté cerrado o disponible."
        )

    # El archivo de ejemplo usa tabulaciones y Windows-1252.
    df = pd.read_csv(
        ruta,
        sep="\t",
        encoding="cp1252",
        dtype="string",
        low_memory=False
    )
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    columnas_requeridas = [
        "identificacion", "nombre_paciente", "estacion_enfermeria",
        "codigo_producto", "medicamento", "costo", "sw_pos",
        "cantidad_suministrada", "cantidad_perdidas",
        "cantidad_aprovechada", "fecha_registro_control",
        "observacion", "usuario_suministra"
    ]
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(
            "No se encontraron estas columnas requeridas: "
            + ", ".join(faltantes)
            + "\nColumnas detectadas: " + ", ".join(df.columns)
        )

    # Normalización de campos de texto.
    for col in [
        "identificacion", "nombre_paciente", "estacion_enfermeria",
        "codigo_producto", "medicamento", "observacion", "usuario_suministra",
        "sw_pos"
    ]:
        df[col] = normalizar_texto(df[col])

    # Conversión numérica tolerante a valores vacíos o mal formados.
    columnas_numericas = [
        "costo", "cantidad_suministrada", "cantidad_perdidas",
        "cantidad_aprovechada"
    ]
    for col in columnas_numericas:
        texto = df[col].astype("string").str.strip()
        # No se cambian separadores de miles de forma automática para evitar
        # interpretar mal valores. Los valores no convertibles quedan como NA.
        df[col] = pd.to_numeric(texto, errors="coerce")

    df["fecha_registro_control"] = pd.to_datetime(
        df["fecha_registro_control"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["fecha"] = df["fecha_registro_control"].dt.date
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora"] = df["fecha_registro_control"].dt.time

    # Clave clínica-operativa del producto. Si no existe código, se usa el nombre.
    df["producto_analisis"] = df["codigo_producto"].fillna(
        df["medicamento"].fillna("PRODUCTO SIN IDENTIFICAR")
    )
    df["producto_descripcion"] = df["medicamento"].fillna(
        df["codigo_producto"].fillna("PRODUCTO SIN IDENTIFICAR")
    )

    # Costo total estimado del suministro cuando costo representa costo unitario.
    # Validar esta interpretación con el diccionario de datos institucional.
    df["valor_suministrado_estimado"] = (
        df["costo"] * df["cantidad_suministrada"]
    )

    # Indicadores de calidad de datos.
    df["alerta_fecha_invalida"] = df["fecha_registro_control"].isna()
    df["alerta_identificacion_vacia"] = df["identificacion"].isna()
    df["alerta_producto_vacio"] = (
        df["codigo_producto"].isna() & df["medicamento"].isna()
    )
    df["alerta_cantidad_suministrada_invalida"] = (
        df["cantidad_suministrada"].isna()
        | (df["cantidad_suministrada"] < 0)
    )
    df["alerta_cantidad_perdida_invalida"] = (
        df["cantidad_perdidas"].notna() & (df["cantidad_perdidas"] < 0)
    )
    df["alerta_cantidad_aprovechada_invalida"] = (
        df["cantidad_aprovechada"].notna() & (df["cantidad_aprovechada"] < 0)
    )

    # Se compara la suma de cantidades aprovechadas y perdidas con lo suministrado.
    # Solo se genera alerta cuando los tres campos están diligenciados.
    cantidades_completas = df[
        ["cantidad_suministrada", "cantidad_perdidas", "cantidad_aprovechada"]
    ].notna().all(axis=1)
    df["diferencia_suministro_vs_aprovechado_perdido"] = (
        df["cantidad_suministrada"]
        - df["cantidad_perdidas"]
        - df["cantidad_aprovechada"]
    )
    df["alerta_conciliacion_cantidades"] = (
        cantidades_completas
        & (df["diferencia_suministro_vs_aprovechado_perdido"] != 0)
    )

    return df


def construir_resumen_diario(df):
    validos = df[
        df["fecha"].notna()
        & df["identificacion"].notna()
        & df["producto_analisis"].notna()
        & df["cantidad_suministrada"].notna()
    ].copy()

    claves = ["identificacion", "nombre_paciente", "producto_analisis",
              "producto_descripcion", "fecha"]
    diario = (
        validos.groupby(claves, dropna=False)
        .agg(
            cantidad_suministrada=("cantidad_suministrada", "sum"),
            cantidad_perdidas=("cantidad_perdidas", "sum"),
            cantidad_aprovechada=("cantidad_aprovechada", "sum"),
            numero_registros=("cantidad_suministrada", "size"),
            costo_unitario_registrado=("costo", "median"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
            estaciones=("estacion_enfermeria", lambda s: " | ".join(
                sorted(set(s.dropna().astype(str)))
            )),
            usuarios=("usuario_suministra", lambda s: " | ".join(
                sorted(set(s.dropna().astype(str)))
            )),
        )
        .reset_index()
    )
    diario = diario.sort_values(
        ["identificacion", "producto_analisis", "fecha"]
    )
    return diario


def detectar_tendencias_y_picos(diario):
    if diario.empty:
        diario["mediana_historica"] = pd.Series(dtype=float)
        diario["alerta_pico_consumo"] = pd.Series(dtype=bool)
        diario["motivo_alerta_tendencia"] = pd.Series(dtype="string")
        return diario

    partes = []
    for _, grupo in diario.groupby(
        ["identificacion", "producto_analisis"], dropna=False, sort=False
    ):
        grupo = grupo.sort_values("fecha").copy()

        # Baseline excluye el día evaluado y utiliza solamente días anteriores,
        # evitando que el dato actual contamine su propia referencia.
        cantidades = grupo["cantidad_suministrada"].astype(float).to_numpy()
        medianas, mads, n_hist, alertas, motivos = [], [], [], [], []

        for i, valor in enumerate(cantidades):
            historial = cantidades[:i]
            historial = historial[np.isfinite(historial)]
            n = len(historial)
            n_hist.append(n)

            if n < MIN_DIAS_BASELINE:
                medianas.append(np.nan)
                mads.append(np.nan)
                alertas.append(False)
                motivos.append(
                    f"Sin alerta estadística: historial insuficiente ({n} días previos)"
                )
                continue

            mediana = float(np.median(historial))
            mad = float(np.median(np.abs(historial - mediana)))
            medianas.append(mediana)
            mads.append(mad)

            if mediana > 0:
                umbral_mad = mediana + MULTIPLICADOR_MAD * 1.4826 * mad
                umbral = max(FACTOR_INCREMENTO * mediana, umbral_mad)
                es_alerta = valor > umbral
                if es_alerta:
                    motivos.append(
                        f"Pico frente a historial: {valor:g} unidades; "
                        f"mediana previa {mediana:g}; umbral aproximado {umbral:.2f}"
                    )
                else:
                    motivos.append("")
            else:
                # Si antes no se registraba suministro, cualquier cantidad positiva
                # se marca como cambio a revisar, no como error confirmado.
                es_alerta = valor > 0
                motivos.append(
                    "Suministro positivo después de historial con mediana cero; "
                    "verificar inicio de tratamiento, cambio clínico o registro"
                    if es_alerta else ""
                )
            alertas.append(bool(es_alerta))

        grupo["dias_previos_disponibles"] = n_hist
        grupo["mediana_historica"] = medianas
        grupo["MAD_historica"] = mads
        grupo["alerta_pico_consumo"] = alertas
        grupo["motivo_alerta_tendencia"] = motivos
        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)


def construir_alertas_registro(df):
    columnas_alerta = [
        "alerta_fecha_invalida", "alerta_identificacion_vacia",
        "alerta_producto_vacio", "alerta_cantidad_suministrada_invalida",
        "alerta_cantidad_perdida_invalida",
        "alerta_cantidad_aprovechada_invalida",
        "alerta_conciliacion_cantidades"
    ]
    mascara = df[columnas_alerta].any(axis=1)
    alertas = df.loc[mascara].copy()

    def explicar(fila):
        causas = []
        if fila["alerta_fecha_invalida"]:
            causas.append("Fecha/hora ausente o no interpretable")
        if fila["alerta_identificacion_vacia"]:
            causas.append("Identificación del paciente vacía")
        if fila["alerta_producto_vacio"]:
            causas.append("Producto sin código y sin descripción")
        if fila["alerta_cantidad_suministrada_invalida"]:
            causas.append("Cantidad suministrada vacía, no numérica o negativa")
        if fila["alerta_cantidad_perdida_invalida"]:
            causas.append("Cantidad perdida negativa")
        if fila["alerta_cantidad_aprovechada_invalida"]:
            causas.append("Cantidad aprovechada negativa")
        if fila["alerta_conciliacion_cantidades"]:
            causas.append(
                "Suministrado no coincide con aprovechado + perdido; "
                "confirmar definición de los campos"
            )
        return "; ".join(causas)

    alertas["motivo_alerta_calidad"] = alertas.apply(explicar, axis=1)
    return alertas


def crear_resumenes(df, diario, tendencias):
    resumen_paciente = (
        df.groupby(["identificacion", "nombre_paciente"], dropna=False)
        .agg(
            registros=("identificacion", "size"),
            productos_distintos=("producto_analisis", "nunique"),
            unidades_suministradas=("cantidad_suministrada", "sum"),
            unidades_perdidas=("cantidad_perdidas", "sum"),
            unidades_aprovechadas=("cantidad_aprovechada", "sum"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
            primera_fecha=("fecha_registro_control", "min"),
            ultima_fecha=("fecha_registro_control", "max"),
            estaciones=("estacion_enfermeria", lambda s: " | ".join(
                sorted(set(s.dropna().astype(str)))
            )),
        )
        .reset_index()
        .sort_values("unidades_suministradas", ascending=False)
    )

    resumen_producto = (
        df.groupby(["codigo_producto", "medicamento"], dropna=False)
        .agg(
            registros=("medicamento", "size"),
            pacientes_distintos=("identificacion", "nunique"),
            unidades_suministradas=("cantidad_suministrada", "sum"),
            unidades_perdidas=("cantidad_perdidas", "sum"),
            unidades_aprovechadas=("cantidad_aprovechada", "sum"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
            estaciones=("estacion_enfermeria", "nunique"),
        )
        .reset_index()
        .sort_values("unidades_suministradas", ascending=False)
    )

    resumen_estacion = (
        df.groupby("estacion_enfermeria", dropna=False)
        .agg(
            registros=("identificacion", "size"),
            pacientes_distintos=("identificacion", "nunique"),
            productos_distintos=("producto_analisis", "nunique"),
            unidades_suministradas=("cantidad_suministrada", "sum"),
            unidades_perdidas=("cantidad_perdidas", "sum"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
        )
        .reset_index()
        .sort_values("unidades_suministradas", ascending=False)
    )

    resumen_usuario = (
        df.groupby("usuario_suministra", dropna=False)
        .agg(
            registros=("identificacion", "size"),
            pacientes_distintos=("identificacion", "nunique"),
            productos_distintos=("producto_analisis", "nunique"),
            unidades_suministradas=("cantidad_suministrada", "sum"),
            unidades_perdidas=("cantidad_perdidas", "sum"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
        )
        .reset_index()
        .sort_values("registros", ascending=False)
    )

    resumen_mensual = (
        df[df["fecha_registro_control"].notna()]
        .assign(mes=lambda x: x["fecha_registro_control"].dt.to_period("M").astype(str))
        .groupby(["mes", "producto_analisis", "producto_descripcion"], dropna=False)
        .agg(
            pacientes_distintos=("identificacion", "nunique"),
            registros=("identificacion", "size"),
            unidades_suministradas=("cantidad_suministrada", "sum"),
            unidades_perdidas=("cantidad_perdidas", "sum"),
            valor_suministrado_estimado=("valor_suministrado_estimado", "sum"),
        )
        .reset_index()
        .sort_values(["mes", "unidades_suministradas"], ascending=[True, False])
    )

    picos = tendencias[tendencias["alerta_pico_consumo"]].copy()
    return resumen_paciente, resumen_producto, resumen_estacion, resumen_usuario, resumen_mensual, picos


def crear_graficos(diario, tendencias, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    if diario.empty:
        return

    # Tendencia total diaria.
    total_diario = diario.groupby("fecha", as_index=False)["cantidad_suministrada"].sum()
    if not total_diario.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(total_diario["fecha"], total_diario["cantidad_suministrada"], marker="o")
        plt.title("Unidades suministradas por día - todos los pacientes")
        plt.xlabel("Fecha")
        plt.ylabel("Unidades registradas")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(carpeta / "tendencia_total_diaria.png", dpi=150)
        plt.close()

    # Gráficos para los 10 productos con mayor volumen.
    top_productos = (
        diario.groupby("producto_descripcion")["cantidad_suministrada"]
        .sum().nlargest(10).index
    )
    for i, producto in enumerate(top_productos, start=1):
        serie = diario[diario["producto_descripcion"] == producto].groupby(
            "fecha", as_index=False
        )["cantidad_suministrada"].sum()
        if serie.empty:
            continue
        plt.figure(figsize=(11, 4))
        plt.plot(serie["fecha"], serie["cantidad_suministrada"], marker="o")
        plt.title(f"Tendencia diaria: {str(producto)[:90]}")
        plt.xlabel("Fecha")
        plt.ylabel("Unidades suministradas")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(carpeta / f"tendencia_producto_{i:02d}.png", dpi=150)
        plt.close()


def exportar_excel(hojas, ruta_excel):
    ruta_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        for nombre, tabla in hojas.items():
            tabla.to_excel(
                writer, sheet_name=nombre_hoja_excel(nombre), index=False
            )
            ws = writer.sheets[nombre_hoja_excel(nombre)]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            # Ancho de columna práctico, limitado para no crear hojas excesivas.
            for columna in ws.columns:
                letra = columna[0].column_letter
                max_len = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in columna[:200]
                )
                ws.column_dimensions[letra].width = min(max(max_len + 2, 10), 42)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo y validando archivo...")
    df = preparar_datos(RUTA_ARCHIVO)

    print("Construyendo consumo diario por paciente y producto...")
    diario = construir_resumen_diario(df)
    tendencias = detectar_tendencias_y_picos(diario)
    alertas_calidad = construir_alertas_registro(df)

    (resumen_paciente, resumen_producto, resumen_estacion,
     resumen_usuario, resumen_mensual, picos) = crear_resumenes(
        df, diario, tendencias
    )

    # Indicadores ejecutivos.
    indicadores = pd.DataFrame([
        ["Registros totales", len(df)],
        ["Pacientes identificados", df["identificacion"].nunique(dropna=True)],
        ["Productos distintos", df["producto_analisis"].nunique(dropna=True)],
        ["Estaciones de enfermería", df["estacion_enfermeria"].nunique(dropna=True)],
        ["Usuarios que registran", df["usuario_suministra"].nunique(dropna=True)],
        ["Registros con alertas de calidad", len(alertas_calidad)],
        ["Días-paciente-producto con pico estadístico", len(picos)],
        ["Fecha inicial", df["fecha_registro_control"].min()],
        ["Fecha final", df["fecha_registro_control"].max()],
        ["Suma de cantidades suministradas", df["cantidad_suministrada"].sum()],
        ["Suma de cantidades perdidas", df["cantidad_perdidas"].sum()],
        ["Suma de cantidades aprovechadas", df["cantidad_aprovechada"].sum()],
        ["Valor estimado (costo x cantidad suministrada)", df["valor_suministrado_estimado"].sum()],
    ], columns=["indicador", "valor"])

    diccionario = pd.DataFrame([
        ["cantidad_suministrada", "Cantidad registrada como suministrada en el sistema."],
        ["cantidad_perdidas", "Cantidad registrada como perdida."],
        ["cantidad_aprovechada", "Cantidad registrada como aprovechada; validar definición local."],
        ["valor_suministrado_estimado", "costo multiplicado por cantidad suministrada; válido solo si costo es unitario."],
        ["alerta_pico_consumo", "Variación estadística respecto de días anteriores del mismo paciente-producto; requiere revisión clínica."],
        ["alerta_conciliacion_cantidades", "Suministrado difiere de aprovechado + perdido; validar si esa igualdad aplica al proceso institucional."],
    ], columns=["campo", "interpretacion"])

    hojas = {
        "Indicadores": indicadores,
        "Consumo_diario": diario,
        "Tendencias_y_picos": tendencias,
        "Alertas_calidad": alertas_calidad,
        "Resumen_paciente": resumen_paciente,
        "Resumen_producto": resumen_producto,
        "Resumen_estacion": resumen_estacion,
        "Resumen_usuario": resumen_usuario,
        "Resumen_mensual": resumen_mensual,
        "Diccionario": diccionario,
        "Datos_limpios": df,
    }

    print("Generando gráficos...")
    crear_graficos(diario, tendencias, CARPETA_SALIDA / "graficos")

    print("Exportando reporte Excel...")
    exportar_excel(hojas, ARCHIVO_EXCEL)

    print("\nPROCESO TERMINADO")
    print(f"Reporte Excel: {ARCHIVO_EXCEL}")
    print(f"Gráficos: {CARPETA_SALIDA / 'graficos'}")
    print("\nInterpretación:")
    print("- Revise primero 'Alertas_calidad' para errores de diligenciamiento o inconsistencias.")
    print("- Revise 'Tendencias_y_picos' para cambios frente al historial individual.")
    print("- Cruce las alertas con órdenes médicas, administración efectiva, dispensación,")
    print("  devoluciones, suspensión de tratamientos, procedimientos y estancia clínica.")
    print("- Un pico no demuestra error: puede corresponder a dosis de carga, cambio de")
    print("  tratamiento, procedimiento, deterioro clínico o suministro acumulado.")


if __name__ == "__main__":
    main()
