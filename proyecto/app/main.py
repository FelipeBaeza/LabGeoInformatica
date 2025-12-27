#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aplicación Web de Análisis Geoespacial con Streamlit.

Esta aplicación permite visualizar y analizar datos geoespaciales
de comunas de Chile.
"""

import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Analisis Territorial - Isla de Pascua",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ===== Funciones de conexión =====

@st.cache_resource
def get_db_engine():
    """Crear conexión a la base de datos."""
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'postgis'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'geodb'),
        'user': os.getenv('POSTGRES_USER', 'geouser'),
        'password': os.getenv('POSTGRES_PASSWORD', 'geopass123')
    }
    
    connection_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    return create_engine(connection_string)


def test_connection():
    """Probar conexión a la base de datos."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT PostGIS_Version();"))
            version = result.fetchone()[0]
            return True, version
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=300)
def get_available_tables():
    """Obtener lista de tablas disponibles."""
    try:
        engine = get_db_engine()
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'geoanalisis'
            ORDER BY table_name
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        return []


@st.cache_data(ttl=300)
def load_geodata(table_name: str, schema: str = 'geoanalisis'):
    """Cargar datos geoespaciales desde PostGIS."""
    try:
        engine = get_db_engine()
        query = f"SELECT * FROM {schema}.{table_name}"
        gdf = gpd.read_postgis(query, engine, geom_col='geometry')
        return gdf
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


def load_geojson(filepath: str):
    """Cargar datos desde archivo GeoJSON."""
    try:
        return gpd.read_file(filepath)
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None


# ===== Funciones de visualización =====

def create_map(gdf, column=None, style='default'):
    """Crear mapa interactivo con Folium."""
    if gdf is None or gdf.empty:
        return None
    
    # Calcular centroide
    centroid = gdf.geometry.unary_union.centroid
    
    # Crear mapa base
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=13,
        tiles='CartoDB positron'
    )
    
    # Determinar tipo de geometría
    geom_type = gdf.geometry.iloc[0].geom_type
    
    if column and column in gdf.columns:
        # Mapa coroplético
        if geom_type in ['Polygon', 'MultiPolygon']:
            folium.Choropleth(
                geo_data=gdf.to_json(),
                data=gdf,
                columns=[gdf.index.name or 'index', column],
                key_on='feature.properties.index',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=column
            ).add_to(m)
    else:
        # Mapa simple
        style_function = lambda x: {
            'fillColor': '#3388ff',
            'color': '#000000',
            'fillOpacity': 0.5,
            'weight': 1
        }
        
        folium.GeoJson(
            gdf.to_json(),
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=list(gdf.columns[:5]),
                aliases=list(gdf.columns[:5])
            )
        ).add_to(m)
    
    return m


def create_statistics_charts(gdf, column):
    """Crear gráficos estadísticos."""
    if column not in gdf.columns:
        return None, None
    
    # Histograma
    fig_hist = px.histogram(
        gdf,
        x=column,
        title=f'Distribucion de {column}',
        template='plotly_white'
    )
    
    # Box plot
    fig_box = px.box(
        gdf,
        y=column,
        title=f'Box Plot de {column}',
        template='plotly_white'
    )
    
    return fig_hist, fig_box


# ===== Páginas de la aplicación =====

def page_home():
    """Página principal."""
    st.markdown('<h1 class="main-header">Analisis Territorial - Isla de Pascua</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Sistema de Analisis Geoespacial
    
    Esta aplicacion permite:
    - **Visualizar** datos geoespaciales de la comuna
    - **Analizar** patrones espaciales
    - **Explorar** diferentes capas de informacion
    - **Generar** estadisticas y reportes
    
    ---
    """)
    
    # Estado de la conexión
    col1, col2, col3 = st.columns(3)
    
    connected, info = test_connection()
    
    with col1:
        if connected:
            st.success(f"Conectado a PostGIS v{info}")
        else:
            st.error("Sin conexion a PostGIS")
    
    with col2:
        tables = get_available_tables()
        st.info(f"{len(tables)} capas disponibles")
    
    with col3:
        st.info("Ultima actualizacion: Hoy")
    
    # Mostrar tablas disponibles
    if tables:
        st.markdown("### Capas de datos disponibles")
        cols = st.columns(3)
        for i, table in enumerate(tables):
            with cols[i % 3]:
                st.markdown(f"- `{table}`")


def page_map_viewer():
    """Página de visualización de mapas."""
    st.header("Visor de Mapas")
    
    # Sidebar para controles
    with st.sidebar:
        st.subheader("Configuracion del Mapa")
        
        # Selección de fuente de datos
        data_source = st.radio(
            "Fuente de datos",
            ["PostGIS"]
        )
        
        tables = get_available_tables()
        if tables:
            selected_table = st.selectbox("Seleccionar capa", tables)
        else:
            st.warning("No hay tablas disponibles")
            return
    
    # Cargar datos
    gdf = None
    
    if 'selected_table' in locals():
        with st.spinner("Cargando datos..."):
            gdf = load_geodata(selected_table)
    
    if gdf is not None and not gdf.empty:
        # Información de la capa
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros", len(gdf))
        with col2:
            st.metric("Tipo de geometria", gdf.geometry.iloc[0].geom_type)
        with col3:
            st.metric("CRS", str(gdf.crs))
        
        # Selección de columna para colorear
        color_column = st.selectbox(
            "Colorear por:",
            ["Ninguno"]
        )
        
        # Crear y mostrar mapa
        selected_col = None if color_column == "Ninguno" else color_column
        m = create_map(gdf, selected_col)
        
        if m:
            st_folium(m, width=None, height=600)
        
        # Mostrar tabla de atributos
        with st.expander("Ver tabla de atributos"):
            # Excluir columna de geometría para mostrar
            display_df = gdf.drop(columns=['geometry'])
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Selecciona una capa de datos para visualizar")


def page_spatial_analysis():
    """Página de análisis espacial."""
    st.header("Analisis Espacial")
    
    tables = get_available_tables()
    
    if not tables:
        st.warning("No hay datos disponibles para analisis")
        return
    
    selected_table = st.selectbox("Seleccionar capa para analisis", tables)
    
    if selected_table:
        with st.spinner("Cargando datos..."):
            gdf = load_geodata(selected_table)
        
        if gdf is not None:
            st.subheader("Estadisticas Descriptivas")
            
            # Estadísticas generales
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Informacion geometrica**")
                
                # Calcular áreas si es polígono
                if gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
                    # Reproyectar para cálculo de área
                    gdf_proj = gdf.to_crs(epsg=32719)  # UTM 19S para Chile
                    gdf['area_m2'] = gdf_proj.geometry.area
                    gdf['area_ha'] = gdf['area_m2'] / 10000
                    
                    st.metric("Area total (ha)", f"{gdf['area_ha'].sum():,.2f}")
                    st.metric("Area promedio (ha)", f"{gdf['area_ha'].mean():,.2f}")
                
                st.metric("Total de entidades", len(gdf))
            
            with col2:
                st.markdown("**Columnas numericas**")
                numeric_cols = gdf.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                if numeric_cols:
                    selected_col = st.selectbox("Variable a analizar", numeric_cols)
                    
                    st.metric("Media", f"{gdf[selected_col].mean():,.2f}")
                    st.metric("Desv. Estandar", f"{gdf[selected_col].std():,.2f}")
                    st.metric("Min - Max", f"{gdf[selected_col].min():,.2f} - {gdf[selected_col].max():,.2f}")
            
            # Gráficos
            if numeric_cols and 'selected_col' in locals():
                st.subheader("Visualizaciones")
                
                fig_hist, fig_box = create_statistics_charts(gdf, selected_col)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                with col2:
                    st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})


def page_about():
    """Página de información."""
    st.header("Acerca del Proyecto")
    
    st.markdown("""
    ### Laboratorio de Geoinformatica
    
    Este proyecto forma parte del curso de Geoinformatica y tiene como objetivo
    desarrollar habilidades en:
    
    - **Manejo de datos geoespaciales** con Python
    - **Bases de datos espaciales** con PostGIS
    - **Analisis espacial** y geoestadistica
    - **Visualizacion** de datos geograficos
    - **Desarrollo web** con Streamlit
    
    ---
    
    ### Tecnologias utilizadas
    
    | Tecnologia | Uso |
    |------------|-----|
    | Python | Lenguaje principal |
    | GeoPandas | Manejo de datos espaciales |
    | PostGIS | Base de datos espacial |
    | Folium | Mapas interactivos |
    | Streamlit | Aplicacion web |
    | Docker | Contenedorizacion |
    
    ---
    
    **Universidad de Santiago de Chile** | Geoinformatica 2025
    """)


def page_visualizaciones():
    """Página de visualizaciones y mapas."""
    from pathlib import Path
    
    st.header("Visualizaciones y Mapas")
    
    # Definir rutas
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUTS = BASE_DIR / "outputs"
    
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
                st.image(str(selected_img), use_container_width=True)

                # Descripciones para cada visualización (clave: nombre del archivo sin extensión)
                explanations = {
                    'pca_biplot': (
                        "El PCA Biplot muestra la proyección de observaciones y variables en los "
                        "dos primeros componentes principales. En este proyecto ayuda a identificar "
                        "cómo las variables topográficas y de servicios contribuyen a la variación "
                        "en la densidad edificatoria y permite detectar agrupamientos espaciales."
                    ),
                    'pca_scree_plot': (
                        "El Scree Plot de PCA muestra la varianza explicada por cada componente "
                        "principal. Se usa para decidir cuántos componentes retener en el análisis "
                        "de reducción de dimensionalidad aplicado a las características espaciales."
                    ),
                    '01_overview_datasets': (
                        "Vista general de los datasets principales (edificaciones, calles, límites). "
                        "Útil para entender la cobertura y calidad de los datos antes del análisis."
                    ),
                    '05_density_map': (
                        "Mapa de densidad de edificaciones: muestra la concentración espacial de "
                        "construcciones por celdas. Es clave para identificar zonas urbanizadas y "
                        "priorizar intervenciones."
                    ),
                    'network_centrality_analysis': (
                        "Análisis de centralidad de la red vial que identifica calles críticas "
                        "para la conectividad. Ayuda a evaluar vulnerabilidades y accesibilidad."
                    )
                }

                key = selected_img.stem.lower().replace(' ', '_')
                desc = explanations.get(key)
                if desc:
                    st.markdown("**Descripción:**")
                    st.write(desc)
                else:
                    # Mostrar una explicación genérica si no existe una específica
                    st.info("Visualización seleccionada: muestra un resultado gráfico generado durante el análisis.")
        else:
            st.info("No hay visualizaciones guardadas.")
    else:
        st.info("Directorio de figuras no encontrado.")


# ===== Main =====

def main():
    """Función principal de la aplicación."""
    
    # Sidebar - Navegación
    st.sidebar.title("Navegacion")
    
    pages = {
        "Inicio": page_home,
        "Visor de Mapas": page_map_viewer,
        "Analisis Espacial": page_spatial_analysis,
        "Visualizaciones": page_visualizaciones,
        "Acerca de": page_about
    }
    
    selected_page = st.sidebar.radio("Ir a", list(pages.keys()))
    
    # Mostrar página seleccionada
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
