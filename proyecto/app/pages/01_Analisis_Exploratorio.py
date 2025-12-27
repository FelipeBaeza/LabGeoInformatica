"""
Página de Análisis Exploratorio Espacial (ESDA)
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os
from sqlalchemy import create_engine

# Configuracion
st.set_page_config(page_title="Analisis Exploratorio", page_icon=None, layout="wide")
st.title("Analisis Exploratorio de Datos Espaciales")
st.markdown("---")

CRS_UTM = 'EPSG:32719'

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '55432'),
    'database': os.getenv('POSTGRES_DB', 'geodatabase'),
    'user': os.getenv('POSTGRES_USER', 'geouser'),
    'password': os.getenv('POSTGRES_PASSWORD', 'geopass123'),
}


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


@st.cache_data
def load_data():
    """Cargar datos desde PostGIS"""
    data = {}
    tables = {
        'boundary': 'limite_administrativa',
        'buildings': 'area_construcciones',
        'amenities': 'punto_interes',
        'streets': 'linea_calles',
        'green_areas': 'area_naturaleza_playas',
    }

    engine = get_engine()
    for key, table in tables.items():
        try:
            gdf = gpd.read_postgis(
                f"SELECT * FROM geoanalisis.{table}", engine, geom_col='geometry'
            )
            if len(gdf) > 0:
                data[key] = gdf
        except Exception as e:
            st.warning(f"Error cargando {table}: {e}")

    return data

# Cargar datos
try:
    data = load_data()
    
    if not data:
        st.error("No se encontraron datos. Ejecute primero el script de descarga.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Opciones de Analisis")
    
    layer_options = list(data.keys())
    selected_layer = st.sidebar.selectbox(
        "Seleccionar capa para análisis:",
        layer_options,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    gdf = data[selected_layer]
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["Estadisticas", "Mapa", "Visualizaciones"])
    
    with tab1:
        st.subheader(f"Estadísticas: {selected_layer.replace('_', ' ').title()}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de features", len(gdf))
        with col2:
            st.metric("Tipo de geometría", gdf.geometry.geom_type.mode().iloc[0] if len(gdf) > 0 else "N/A")
        with col3:
            if gdf.crs:
                st.metric("CRS", str(gdf.crs.to_epsg()))
            else:
                st.metric("CRS", "No definido")
        
        st.markdown("---")
        
        # Columnas disponibles
        st.subheader("Columnas del Dataset")
        cols_df = pd.DataFrame({
            'Columna': gdf.columns.tolist(),
            'Tipo': [str(gdf[col].dtype) for col in gdf.columns],
            'No Nulos': [gdf[col].notna().sum() for col in gdf.columns]
        })
        st.dataframe(cols_df, use_container_width=True)
        
        # Muestra de datos
        st.subheader("Muestra de Datos")
        n_samples = st.slider("Número de registros", 5, min(50, len(gdf)), 10)
        st.dataframe(gdf.drop(columns=['geometry']).head(n_samples), use_container_width=True)
        
        # Estadísticas numéricas
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Estadisticas Numericas")
            st.dataframe(gdf[numeric_cols].describe(), use_container_width=True)
        
        # Estadísticas categóricas
        categorical_cols = gdf.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c != 'geometry']
        
        if categorical_cols:
            st.subheader("Variables Categoricas")
            selected_cat = st.selectbox("Seleccionar variable:", categorical_cols)
            if selected_cat:
                value_counts = gdf[selected_cat].value_counts().head(15)
                st.bar_chart(value_counts)
    
    with tab2:
        st.subheader(f"Mapa: {selected_layer.replace('_', ' ').title()}")
        
        # Centro del mapa
        if 'boundary' in data:
            center = data['boundary'].to_crs('EPSG:4326').geometry.centroid.iloc[0]
            center_lat, center_lon = center.y, center.x
        else:
            gdf_wgs = gdf.to_crs('EPSG:4326')
            bounds = gdf_wgs.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        
        # Crear mapa
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, 
                      tiles='CartoDB positron')
        
        # Convertir a WGS84
        gdf_plot = gdf.to_crs('EPSG:4326')
        
        # Color por tipo de geometría
        geom_type = gdf_plot.geometry.geom_type.iloc[0] if len(gdf_plot) > 0 else None
        
        if geom_type:
            if 'Polygon' in geom_type:
                folium.GeoJson(
                    gdf_plot,
                    style_function=lambda x: {
                        'fillColor': '#3388ff',
                        'color': '#3388ff',
                        'weight': 1,
                        'fillOpacity': 0.5
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=gdf_plot.columns[:3].tolist() if len(gdf_plot.columns) > 3 else gdf_plot.columns.tolist(),
                        aliases=gdf_plot.columns[:3].tolist() if len(gdf_plot.columns) > 3 else gdf_plot.columns.tolist()
                    )
                ).add_to(m)
            elif 'Line' in geom_type:
                folium.GeoJson(
                    gdf_plot,
                    style_function=lambda x: {
                        'color': '#e41a1c',
                        'weight': 2
                    }
                ).add_to(m)
            elif 'Point' in geom_type:
                for idx, row in gdf_plot.iterrows():
                    if row.geometry:
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=5,
                            color='#4daf4a',
                            fill=True,
                            fillColor='#4daf4a',
                            fillOpacity=0.7
                        ).add_to(m)
        
        # Mostrar mapa
        st_folium(m, width=None, height=600)
    
    with tab3:
        st.subheader("Visualizaciones")
        
        # Área (para polígonos)
        gdf_utm = gdf.to_crs(CRS_UTM)
        
        if 'Polygon' in str(gdf_utm.geometry.geom_type.iloc[0]):
            gdf_utm['area_m2'] = gdf_utm.geometry.area
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribución de Áreas**")
                fig, ax = plt.subplots(figsize=(8, 5))
                gdf_utm['area_m2'].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
                ax.set_xlabel('Área (m²)')
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'Histograma de Áreas - {selected_layer}')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Estadísticas de Área**")
                stats = gdf_utm['area_m2'].describe()
                st.dataframe(pd.DataFrame(stats).T.round(2))
        
        elif 'Line' in str(gdf_utm.geometry.geom_type.iloc[0]):
            gdf_utm['length_m'] = gdf_utm.geometry.length
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribución de Longitudes**")
                fig, ax = plt.subplots(figsize=(8, 5))
                gdf_utm['length_m'].hist(bins=30, ax=ax, color='coral', edgecolor='white')
                ax.set_xlabel('Longitud (m)')
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'Histograma de Longitudes - {selected_layer}')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Estadísticas de Longitud**")
                stats = gdf_utm['length_m'].describe()
                st.dataframe(pd.DataFrame(stats).T.round(2))
        
        else:
            st.info("Para puntos, el análisis de distribución espacial está disponible en la página de Hot Spots.")
        
        # Mapa estático de densidad
        st.markdown("---")
        st.markdown("**Mapa de Distribución**")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if 'boundary' in data:
            data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                   edgecolor='black', linewidth=2)
        
        gdf_utm.plot(ax=ax, alpha=0.6, color='steelblue', markersize=5)
        ax.set_title(f'Distribución Espacial - {selected_layer.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_axis_off()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.info("Asegúrese de haber ejecutado el script de descarga de datos primero.")
