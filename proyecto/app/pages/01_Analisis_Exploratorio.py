"""
Pagina de Analisis Exploratorio de Datos Espaciales (ESDA)
Muestra estadisticas, mapas y patrones de distribucion
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

st.set_page_config(page_title="Analisis Exploratorio", layout="wide")

st.title("Analisis Exploratorio de Datos Espaciales")

st.markdown("""
Este modulo permite explorar los datos geograficos de Isla de Pascua. El analisis exploratorio 
es el primer paso en cualquier estudio territorial, ya que nos permite entender como estan 
distribuidos los elementos en el espacio antes de aplicar tecnicas mas avanzadas.
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
        'boundary': 'limite_administrativa',
        'buildings': 'area_construcciones',
        'amenities': 'punto_interes',
        'streets': 'linea_calles',
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

st.header("1. Estadisticas Descriptivas")

st.markdown("""
Las estadisticas descriptivas nos dan una vision general de la cantidad y caracteristicas 
de los elementos geograficos. Esto nos permite dimensionar el alcance del estudio.
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Edificaciones", len(data.get('buildings', [])))
with col2:
    st.metric("Puntos de Interes", len(data.get('amenities', [])))
with col3:
    st.metric("Segmentos Viales", len(data.get('streets', [])))
with col4:
    if 'buildings' in data:
        # Calcular area en proyeccion UTM
        buildings_utm = data['buildings'].to_crs("EPSG:32706")
        area_total = buildings_utm.geometry.area.sum() / 10000
        st.metric("Area Construida", f"{area_total:.1f} ha")

st.markdown("""
**Interpretacion:** Isla de Pascua tiene una cantidad relativamente pequena de edificaciones 
concentradas principalmente en Hanga Roa. Los puntos de interes incluyen restaurantes, 
hoteles y servicios turisticos que atienden a los visitantes de la isla.
""")

# ============================================================================
# SECCION 2: MAPA INTERACTIVO
# ============================================================================

st.header("2. Mapa de Distribucion Espacial")

st.markdown("""
Este mapa interactivo muestra la ubicacion de todas las edificaciones y puntos de interes.
Permite identificar visualmente donde se concentran los elementos y detectar patrones espaciales.
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
**Interpretacion:** El mapa revela que las edificaciones (naranja) se concentran casi 
exclusivamente en el sector oeste de la isla, correspondiente al pueblo de Hanga Roa. 
Los puntos de interes (azul) siguen el mismo patron, lo que indica que los servicios 
turisticos estan ubicados cerca de la zona habitada.
""")

# ============================================================================
# SECCION 3: HISTOGRAMA DE AREAS
# ============================================================================

st.header("3. Distribucion de Tamanos de Edificaciones")

st.markdown("""
Este grafico muestra como se distribuyen las edificaciones segun su tamano (area en metros cuadrados).
Nos permite entender si predominan construcciones pequenas o grandes.
""")

if 'buildings' in data:
    buildings_utm = data['buildings'].to_crs("EPSG:32706")
    buildings_utm['area_m2'] = buildings_utm.geometry.area
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Filtrar areas razonables
    areas = buildings_utm['area_m2']
    areas = areas[(areas > 10) & (areas < 2000)]
    
    ax.hist(areas, bins=50, color='#FF6B35', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Area (m2)')
    ax.set_ylabel('Cantidad de edificaciones')
    ax.set_title('Distribucion de areas de edificaciones')
    ax.axvline(areas.median(), color='red', linestyle='--', label=f'Mediana: {areas.median():.0f} m2')
    ax.legend()
    
    st.pyplot(fig)
    plt.close()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Area minima", f"{areas.min():.0f} m2")
    with col2:
        st.metric("Area mediana", f"{areas.median():.0f} m2")
    with col3:
        st.metric("Area maxima", f"{areas.max():.0f} m2")

st.markdown("""
**Interpretacion:** La mayoria de las edificaciones son de tamano pequeno a mediano, 
tipicas de viviendas familiares. La distribucion sesgada hacia la izquierda indica 
que las construcciones grandes (hoteles, edificios publicos) son escasas.
""")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("Analisis Exploratorio - Isla de Pascua | Laboratorio Integrador 2025")
