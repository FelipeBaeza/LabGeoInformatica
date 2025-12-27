import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import pickle
import io
import base64

st.set_page_config(page_title="Descargas - Isla de Pascua", layout="wide")

st.title("Centro de Descargas")
st.markdown("---")

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

st.markdown("""
Descarga los datos procesados, modelos entrenados y resultados del análisis territorial de Isla de Pascua.
""")

# ============================================================================
# SECCIÓN 1: DATOS GEOESPACIALES
# ============================================================================
st.header("Datos Geoespaciales")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos Vectoriales")
    
    # GeoPackage con grilla
    gpkg_file = DATA_PROCESSED / "prepared.gpkg"
    if gpkg_file.exists():
        with open(gpkg_file, "rb") as f:
            st.download_button(
                label="Descargar GeoPackage Completo",
                data=f,
                file_name="isla_pascua_datos.gpkg",
                mime="application/octet-stream",
                help="Contiene todas las capas vectoriales procesadas"
            )
    
    # Grilla con topografía
    grid_topo = DATA_PROCESSED / "grid_with_topography.gpkg"
    if grid_topo.exists():
        with open(grid_topo, "rb") as f:
            st.download_button(
                label="Grilla con Variables Topograficas",
                data=f,
                file_name="grid_topografia.gpkg",
                mime="application/octet-stream"
            )

with col2:
    st.subheader("Datos Raster")
    
    # DEM
    dem_file = DATA_PROCESSED / "dem_isla_pascua_clipped.tif"
    if dem_file.exists():
        with open(dem_file, "rb") as f:
            st.download_button(
                label="DEM (Modelo Digital Elevación)",
                data=f,
                file_name="dem_isla_pascua.tif",
                mime="image/tiff"
            )
    
    # Hillshade
    hillshade_file = DATA_PROCESSED / "hillshade_isla_pascua.tif"
    if hillshade_file.exists():
        with open(hillshade_file, "rb") as f:
            st.download_button(
                label="Hillshade (Sombreado)",
                data=f,
                file_name="hillshade_isla_pascua.tif",
                mime="image/tiff"
            )

st.markdown("---")

# ============================================================================
# SECCIÓN 2: DATOS TABULARES
# ============================================================================
st.header("Datos Tabulares")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Features Topográficos")
    topo_csv = DATA_PROCESSED / "grid_topographic_features.csv"
    if topo_csv.exists():
        df_topo = pd.read_csv(topo_csv)
        csv_data = df_topo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV",
            data=csv_data,
            file_name="features_topograficos.csv",
            mime="text/csv"
        )
        st.caption(f"{len(df_topo)} registros")

with col2:
    st.subheader("Estadísticas ESDA")
    st.info("Generar desde notebooks de análisis")
    # Placeholder para estadísticas

with col3:
    st.subheader("Resultados ML")
    st.info("Generar desde notebook de ML")
    # Placeholder para resultados

st.markdown("---")

# ============================================================================
# SECCIÓN 3: MODELOS ENTRENADOS
# ============================================================================
st.header("Modelos de Machine Learning")

st.markdown("""
Descarga los modelos entrenados para predicción de densidad edificatoria.
""")

models_dir = OUTPUTS / "models"
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl"))
    
    if model_files:
        cols = st.columns(min(3, len(model_files)))
        for idx, model_file in enumerate(model_files):
            with cols[idx % 3]:
                model_name = model_file.stem.replace("_", " ").title()
                with open(model_file, "rb") as f:
                    st.download_button(
                        label=f"{model_name}",
                        data=f,
                        file_name=model_file.name,
                        mime="application/octet-stream"
                    )
    else:
        st.info("No hay modelos guardados. Ejecuta el notebook de ML primero.")
else:
    st.info("Directorio de modelos no encontrado.")

st.markdown("---")

# ============================================================================
# SECCIÓN 4: VISUALIZACIONES
# ============================================================================
st.header("Visualizaciones y Mapas")

figures_dir = OUTPUTS / "figures"
if figures_dir.exists():
    image_files = list(figures_dir.glob("*.png"))
    
    if image_files:
        st.markdown(f"**{len(image_files)} visualizaciones disponibles**")
        
        # Selector de imagen
        selected_img = st.selectbox(
            "Selecciona una visualización:",
            options=image_files,
            format_func=lambda x: x.stem.replace("_", " ").title()
        )
        
        if selected_img:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(str(selected_img), use_container_width=True)
            
            with col2:
                st.markdown(f"**Archivo:** `{selected_img.name}`")
                st.markdown(f"**Tamaño:** {selected_img.stat().st_size / 1024:.1f} KB")
                
                with open(selected_img, "rb") as f:
                    st.download_button(
                        label="Descargar Imagen",
                        data=f,
                        file_name=selected_img.name,
                        mime="image/png"
                    )
    else:
        st.info("No hay visualizaciones guardadas.")
else:
    st.info("Directorio de figuras no encontrado.")

st.markdown("---")

# ============================================================================
# SECCIÓN 5: REPORTES
# ============================================================================
st.header("Reportes y Documentacion")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Documentación Técnica")
    
    docs_files = [
        ("README.md", "Guía General del Proyecto"),
        ("docs/arquitectura.md", "Arquitectura del Sistema"),
        ("docs/guia_usuario.md", "Guía de Usuario"),
        ("docs/topographic_integration.md", "Integración DEM")
    ]
    
    for file_path, description in docs_files:
        full_path = BASE_DIR / file_path
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                st.download_button(
                    label=f"{description}",
                    data=content,
                    file_name=full_path.name,
                    mime="text/markdown"
                )

with col2:
    st.subheader("Informe Técnico")
    
    informe_tex = BASE_DIR / "docs" / "informe_tecnico.tex"
    if informe_tex.exists():
        with open(informe_tex, "r", encoding="utf-8") as f:
            st.download_button(
                label="Informe LaTeX",
                data=f,
                file_name="informe_tecnico.tex",
                mime="text/plain"
            )
    
    st.info("Compila el archivo .tex con pdflatex para generar el PDF")

st.markdown("---")

# ============================================================================
# SECCIÓN 6: PAQUETE COMPLETO
# ============================================================================
st.header("Descarga Completa")

st.markdown("""
**Nota:** Para descargar el proyecto completo, clona el repositorio Git o descarga el ZIP desde GitHub.
""")

st.code("""
# Clonar repositorio
git clone [URL_DEL_REPOSITORIO]

# O descargar como ZIP
# Desde GitHub: Code → Download ZIP
""", language="bash")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Sistema de Análisis Territorial - Isla de Pascua | Laboratorio Integrador 2025")
