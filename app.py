"""Visualizador del proyecto final de Machine Learning: RSU Saltillo."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

st.set_page_config(page_title="RSU Saltillo | Machine Learning", page_icon="🗺️", layout="wide")

CLUSTER_COLORS = ["#F205CB", "#0487D9", "#F25C54", "#04BF33", "#D2D904"]
ZONE_COLORS = {"Norte": "#0487D9", "Sur": "#F205CB"}  # Archivo final del notebook

@st.cache_resource(show_spinner="Cargando datos, entrenando modelos y evaluando clústeres...")
def load_analysis():
    import analysis_core
    return analysis_core

try:
    a = load_analysis()
except Exception as exc:
    st.error("No se pudieron cargar los archivos o ejecutar los cálculos.")
    st.exception(exc)
    st.stop()


def fmt(value, decimals=0):
    return f"{value:,.{decimals}f}"


def base_map(ax):
    a.ASENTAMIENTOS_PROYECTADOS.plot(
        ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.25
    )
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_clusters(ax, k):
    """Same background/label logic as notebook's Chunk 16."""
    base_map(ax)
    model = a.MODELOS_CLUSTER[k]
    geo = a.DATOS_CLUSTER.copy()
    geo["Cluster"] = model.labels_ + 1
    for num in range(1, k + 1):
        subset = geo[geo["Cluster"] == num]
        subset.plot(
            ax=ax, color=CLUSTER_COLORS[num - 1],
            edgecolor="white", linewidth=0.35, label=f"Clúster {num}"
        )
    ax.scatter(
        model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
        marker="X", s=140, c="black", edgecolors="white", linewidths=0.8
    )
    ax.set_title(f"K-Means: {k} clústeres")
    ax.legend(loc="best", fontsize=8, title="Clúster")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_north_south():
    fig, ax = plt.subplots(figsize=(10, 9))
    base_map(ax)
    for name in ("Norte", "Sur"):
        sub = a.ZONAS_CLUSTER[a.ZONAS_CLUSTER["Zona_Propuesta"] == name]
        sub.plot(ax=ax, color=ZONE_COLORS[name], edgecolor="white", linewidth=0.35, label=name)
    ax.scatter(
        a.MODELO_NORTE_SUR.cluster_centers_[:, 0],
        a.MODELO_NORTE_SUR.cluster_centers_[:, 1],
        marker="X", s=170, c="black", edgecolors="white", linewidth=1,
        label="Centros K-Means"
    )
    ax.legend(loc="best")
    ax.set_title("Propuesta Norte-Sur · turno vespertino")
    return fig


st.title("Clusterización de asentamientos para la recolección de residuos en Saltillo")
st.caption("Proyecto final de Machine Learning · Noviembre de 2026 · Análisis del turno vespertino")

with st.sidebar:
    st.header("Secciones")
    section = st.radio(
        "Ir a", ["Resumen", "Predicción poblacional", "Residuos", "Análisis geográfico", "K-Means"],
        label_visibility="collapsed"
    )
    st.divider()

if section == "Resumen":
    st.subheader("Resumen")
    st.markdown("""
    **¿De qué trata este proyecto?**

    Este proyecto analiza la distribución geográfica de los asentamientos
    de Saltillo para proponer una división territorial que facilite la
    planificación de la recolección de residuos sólidos urbanos durante
    el turno vespertino.

    Se utilizan registros históricos, información geográfica y técnicas
    de Machine Learning para desarrollar cuatro etapas:

    1. **Predicción poblacional:** estimar la población de Saltillo para 2025
       mediante modelos de regresión.
    2. **Estimación de residuos:** calcular la demanda de recolección
       a partir de los registros históricos.
    3. **Análisis geográfico:** obtener puntos representativos
       de los asentamientos.
    4. **Clusterización:** utilizar K-Means para proponer y evaluar
       una división Norte-Sur.

    El análisis se concentra en **396 asentamientos del turno vespertino**,
    con una demanda estimada para dos días de acumulación.
    Los resultados constituyen una propuesta exploratoria,
    no rutas definitivas de recolección.
    """)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asentamientos totales", fmt(len(a.ASENTAMIENTOS_PROYECTADOS)))
    c2.metric("Asentamientos vespertinos", fmt(len(a.DATOS_CLUSTER)))
    c3.metric("Demanda de 2 días", f"{fmt(a.PESOS_CLUSTER.sum(), 2)} t")
    c4.metric("Tasa local", f"{a.TASA_GLOBAL:.4f} kg/hab/día")
    st.markdown("**Objetivo:** estudiar la demanda vespertina de recolección y una posible división Norte-Sur mediante aprendizaje no supervisado.")
    left, right = st.columns([1.3, 1])
    with left:
        fig = plot_north_south()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Distribución de demanda")
        st.dataframe(a.RESUMEN_ZONAS.round(2), hide_index=True, use_container_width=True)
        st.metric("Diferencia Norte-Sur", f"{a.DIFERENCIA_TONELADAS:.2f} t")
        st.metric("Desequilibrio respecto al total", f"{a.DESEQUILIBRIO_ZONAS:.2f} %")

elif section == "Predicción poblacional":
    st.subheader("1. Predicción de población para 2025")
    st.markdown("Se entrenaron seis variantes de regresión y se evaluaron sobre años reservados mediante validación temporal.")
    col1, col2 = st.columns(2)
    col1.metric("Modelo con menor MAE", a.Mejor_Modelo)
    col2.metric("Población predicha para 2025", fmt(a.Prediccion_Final))

    show_models = st.multiselect("Modelos a visualizar", a.Nombres_Modelos, default=a.Nombres_Modelos)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(a.Datos["Año"], a.Datos["Población"], c="black", s=40, label="Población histórica", zorder=3)
    years = np.linspace(a.Datos["Año"].min(), 2025, 300)
    x_values = ((years - 2000) / 10).reshape(-1, 1)
    for model_name in show_models:
        ax.plot(years, a.predecir_modelo(model_name, x_values), linewidth=1.4, label=model_name)
    ax.axvline(2025, linestyle="--", linewidth=1, color="gray")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set(xlabel="Año", ylabel="Población (habitantes)", title="Comparación de modelos poblacionales")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("#### Evaluación con años reservados")
    st.dataframe(a.Resultados_Modelos.round(2), hide_index=True, use_container_width=True)
    st.markdown("#### Predicciones para 2025")
    st.dataframe(a.Tabla_Predicciones.round(0), hide_index=True, use_container_width=True)
    st.caption("Selección por menor MAE. Ridge utiliza regularización y selección de alfa dentro de los datos de entrenamiento.")

elif section == "Residuos":
    st.subheader("2. Estimación de recolección vespertina")
    st.markdown("Se relaciona la población asignada a cada sector con las toneladas registradas durante 2025. Los asentamientos compartidos reparten su población entre sectores, sin duplicarla.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sectores", fmt(len(a.DATOS_SECTORES)))
    c2.metric("Población asignada", fmt(a.POBLACION_TOTAL))
    c3.metric("Toneladas registradas", f"{fmt(a.TONELADAS_TOTALES, 2)} t")
    c4.metric("Recolección diaria por habitante", f"{a.TASA_GLOBAL:.4f} kg")
    st.caption(f"Periodo de registros: {a.FECHA_INICIO.date()} al {a.FECHA_FIN.date()} ({a.DIAS_PERIODO} días calendario).")
    st.markdown("**Criterio para clusterización:** demanda estimada de cada asentamiento = población × tasa local × 2 días / 1,000.")
    st.dataframe(
        a.DATOS_SECTORES[["Zona", "Sector", "Poblacion_Estimada", "Toneladas_Recolectadas", "Kg_Hab_Dia"]].round(4),
        hide_index=True, use_container_width=True
    )
    fig, ax = plt.subplots(figsize=(12, 4.2))
    plot_data = a.DATOS_SECTORES.sort_values("Sector")
    ax.bar(plot_data["Sector"].astype(str), plot_data["Kg_Hab_Dia"], color="#0487D9", alpha=0.85)
    ax.axhline(a.TASA_GLOBAL, linestyle="--", color="#F205CB", label=f"Tasa global: {a.TASA_GLOBAL:.3f}")
    ax.set(xlabel="Sector", ylabel="kg/hab/día", title="Recolección registrada por habitante y sector")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("La tasa mide recolección del turno vespertino, no necesariamente la totalidad de residuos generados. Supone cobertura suficiente de registros.")

elif section == "Análisis geográfico":
    st.subheader("3. Geometrías y puntos representativos")
    n_outside = int((~a.CENTROIDES["Centroide_Dentro"]).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Polígonos del KML", len(a.CENTROIDES))
    c2.metric("Centroides fuera del polígono", n_outside)
    c3.metric("CRS métrico", str(a.CRS_PROYECTADO))
    fig, ax = plt.subplots(figsize=(11, 9))
    base_map(ax)
    a.PUNTOS_GEO.plot(ax=ax, color="#F205CB", markersize=4, label="Puntos representativos") if hasattr(a, "PUNTOS_GEO") else ax.scatter(
        a.CENTROIDES["Coordenada_X"], a.CENTROIDES["Coordenada_Y"], c="#F205CB", s=4, label="Puntos representativos"
    )
    ax.set_title("862 polígonos y sus puntos representativos")
    ax.legend(loc="best")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Cuando el centroide cae fuera del polígono, se utiliza representative_point() para elegir un punto interior.")
    st.dataframe(a.CENTROIDES[["Name", "Coordenada_X", "Coordenada_Y", "Centroide_Dentro"]].head(30), hide_index=True, use_container_width=True)

elif section == "K-Means":
    st.subheader("4. Clusterización ponderada por demanda")
    st.markdown("Se comparan configuraciones **k = 2, 3, 4, 5**. Los asentamientos matutinos se muestran en gris y no intervienen en el algoritmo.")
    st.dataframe(a.RESULTADOS_CLUSTER.round(3), hide_index=True, use_container_width=True)

    k = st.select_slider("Número de clústeres a visualizar", options=[2, 3, 4, 5], value=2)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plot_clusters(ax, k)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if st.checkbox("Mostrar comparación de los cuatro escenarios", value=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 13))
        for ax, num in zip(axes.flat, range(2, 6)):
            plot_clusters(ax, num)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Propuesta Norte-Sur (k = 2)")
    left, right = st.columns([1.2, 1])
    with left:
        fig = plot_north_south()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.dataframe(a.RESUMEN_ZONAS.round(2), hide_index=True, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = a.RESUMEN_ZONAS
        bars = ax.bar(vals["Zona_Propuesta"], vals["Demanda_Toneladas"], color=[ZONE_COLORS[z] for z in vals["Zona_Propuesta"]])
        ax.bar_label(bars, fmt="%.2f t", padding=4)
        ax.axhline(vals["Demanda_Toneladas"].sum() / 2, linestyle="--", color="black", label="Carga ideal")
        ax.set_ylabel("Toneladas / 2 días")
        ax.set_ylim(0, vals["Demanda_Toneladas"].max() * 1.2)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown("#### Continuidad entre polígonos vespertinos")
    st.dataframe(a.CONTINUIDAD_ZONAS, hide_index=True, use_container_width=True)
    st.warning("K-Means no garantiza zonas continuas ni una asignación equilibrada de toneladas. Los componentes detectados reflejan conexiones por frontera entre polígonos vespertinos, no conectividad por vialidades.")
    export_data = a.ZONAS_CLUSTER[["CVEASEN", "Poblacion_Original", "Demanda_Toneladas", "Cluster", "Zona_Propuesta", "Coordenada_X", "Coordenada_Y"]].copy()
    export_data["Cluster"] += 1
    st.download_button("Descargar asignación Norte-Sur (CSV)", export_data.to_csv(index=False).encode("utf-8-sig"), "zonas_vespertinas.csv", "text/csv")

st.divider()
