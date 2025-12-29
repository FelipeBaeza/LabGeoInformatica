"""
Pagina de Análisis Exploratorio de Datos Espaciales (ESDA)
Muestra estadísticas, mapas y patrones de distribución
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

st.set_page_config(page_title="Análisis Exploratorio", layout="wide")

st.title("Análisis Exploratorio de Datos Espaciales")

st.markdown("""
Este módulo permite explorar los datos geográficos de Isla de Pascua. El análisis exploratorio 
es el primer paso en cualquier estudio territorial, ya que nos permite entender como están 
distribuidos los elementos en el espacio antes de aplicar técnicas más avanzadas.
""")
st.markdown("---")

# Configuracion BD
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
    """Cargar datos desde PostGIS."""
    data = {}
    tables = {
        'boundary': 'isla_de_pascua_boundary',
        'buildings': 'isla_de_pascua_buildings',
        'amenities': 'isla_de_pascua_amenities',
        'streets': 'isla_de_pascua_streets',
    }
    
    try:
        engine = get_engine()
        for key, table in tables.items():
            try:
                gdf = gpd.read_postgis(
                    f"SELECT * FROM geoanalisis.{table}", 
                    engine, geom_col='geometry'
                )
                data[key] = gdf.to_crs("EPSG:4326")
            except Exception as e:
                st.warning(f"No se pudo cargar {table}")
    except:
        st.error("Error conectando a la base de datos")
    
    return data


# Cargar datos
data = load_data()

if not data or 'buildings' not in data:
    st.error("No se pudieron cargar los datos. Verifique la conexion a PostGIS.")
    st.stop()

# ============================================================================
# SECCION 1: ESTADISTICAS DESCRIPTIVAS
# ============================================================================

st.header("1. Estadísticas Descriptivas")

st.markdown("""
Las estadísticas descriptivas nos dan una visión general de la cantidad y características 
de los elementos geográficos. Esto nos permite dimensionar el alcance del estudio.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Edificaciones", len(data.get('buildings', [])))
with col2:
    st.metric("Puntos de Interés", len(data.get('amenities', [])))
with col3:
    st.metric("Segmentos Viales", len(data.get('streets', [])))
with col4:
    if 'buildings' in data:
        # Calcular area en proyeccion UTM
        buildings_utm = data['buildings'].to_crs("EPSG:32706")
        area_total = buildings_utm.geometry.area.sum() / 10000
        st.metric("Area Construida", f"{area_total:.1f} ha")

st.markdown("""
**Interpretación:** Isla de Pascua tiene una cantidad relativamente pequeña de edificaciones 
concentradas principalmente en Hanga Roa. Los puntos de interés incluyen restaurantes, 
hoteles y servicios turísticos que atienden a los visitantes de la isla.
""")

# ============================================================================
# SECCION 2: MAPA INTERACTIVO
# ============================================================================

st.header("2. Mapa de Distribución Espacial")

st.markdown("""
Este mapa interactivo muestra la ubicación de todas las edificaciones y puntos de interés.
Permite identificar visualmente dónde se concentran los elementos y detectar patrones espaciales.
""")

# Calcular centro
if 'boundary' in data and len(data['boundary']) > 0:
    bounds = data['boundary'].total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
else:
    center_lat = -27.1167
    center_lon = -109.3667

# Crear mapa
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

# Agregar edificaciones
if 'buildings' in data:
    for idx, row in data['buildings'].head(500).iterrows():
        centroid = row.geometry.centroid
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=3,
            color='#FF6B35',
            fill=True,
            fillOpacity=0.7
        ).add_to(m)

# Agregar amenidades
if 'amenities' in data:
    for idx, row in data['amenities'].iterrows():
        if row.geometry.geom_type == 'Point':
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='#2E86AB',
                fill=True,
                fillOpacity=0.8
            ).add_to(m)

st_folium(m, width=900, height=500)

st.markdown("""
**Interpretación:** El mapa revela que las edificaciones (naranja) se concentran casi 
exclusivamente en el sector oeste de la isla, correspondiente al pueblo de Hanga Roa. 
Los puntos de interés (azul) siguen el mismo patrón, lo que indica que los servicios 
turísticos están ubicados cerca de la zona habitada.
""")

# ============================================================================
# SECCION 3: HISTOGRAMA DE AREAS
# ============================================================================

st.header("3. Distribución de Tamanos de Edificaciones")

st.markdown("""
Este gráfico muestra cómo se distribuyen las edificaciones según su tamaño (área en metros cuadrados).
Nos permite entender si predominan construcciones pequeñas o grandes.
""")

if 'buildings' in data:
    buildings_utm = data['buildings'].to_crs("EPSG:32706")
    buildings_utm['area_m2'] = buildings_utm.geometry.area
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Filtrar areas razonables
    areas = buildings_utm['area_m2']
    areas = areas[(areas > 10) & (areas < 2000)]
    
    ax.hist(areas, bins=50, color='#FF6B35', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Área (m2)')
    ax.set_ylabel('Cantidad de edificaciones')
    ax.set_title('Distribución de áreas de edificaciones')
    ax.axvline(areas.median(), color='red', linestyle='--', label=f'Mediana: {areas.median():.0f} m2')
    ax.legend()
    
    st.pyplot(fig)
    plt.close()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Área minima", f"{areas.min():.0f} m2")
    with col2:
        st.metric("Área mediana", f"{areas.median():.0f} m2")
    with col3:
        st.metric("Área maxima", f"{areas.max():.0f} m2")

st.markdown("""
**Interpretación:** La mayoria de las edificaciones son de tamaño pequeño a mediano, 
típicas de viviendas familiares. La distribución sesgada hacia la izquierda indica 
que las construcciones grandes (hoteles, edificios públicos) son escasas.
""")