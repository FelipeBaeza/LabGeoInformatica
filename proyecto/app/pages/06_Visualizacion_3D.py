"""
Modelo 3D de Densidad Urbana - Isla de Pascua
Visualizacion de patrones de ocupacion territorial
"""
import streamlit as st
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Modelo 3D - Isla de Pascua", layout="wide")

st.title("Modelo 3D de Densidad Urbana")

st.markdown("""
Este modelo tridimensional representa la **densidad de edificaciones** en Isla de Pascua.
Cada columna del mapa indica la cantidad de construcciones en esa zona: columnas mas altas 
significan mayor concentracion de edificios. Esta visualizacion permite identificar rapidamente 
las areas de mayor desarrollo urbano, lo cual es clave para la planificacion territorial.
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
def load_buildings():
    """Cargar edificios desde PostGIS."""
    try:
        engine = get_engine()
        gdf = gpd.read_postgis(
            "SELECT * FROM geoanalisis.area_construcciones", 
            engine, geom_col='geometry'
        )
        return gdf
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


# Cargar datos
buildings = load_buildings()

if buildings is None or len(buildings) == 0:
    st.error("No se pudieron cargar los edificios desde la base de datos.")
    st.stop()

# ============================================================================
# CONFIGURACION
# ============================================================================

st.sidebar.header("Configuracion")

# Parametros
cell_size = st.sidebar.selectbox(
    "Tamano de celda:",
    [100, 200, 300, 400],
    index=1,
    format_func=lambda x: f"{x} metros"
)

height_multiplier = st.sidebar.slider(
    "Escala de altura:",
    min_value=5,
    max_value=50,
    value=20
)

# ============================================================================
# PROCESAR DATOS
# ============================================================================

# Convertir a CRS proyectado para calculos de area
# Usar UTM zona 12S para Isla de Pascua (Pacifico Sur)
buildings_proj = buildings.to_crs("EPSG:32712")

# Calcular area de cada edificio
buildings_proj['area_m2'] = buildings_proj.geometry.area

# Obtener centroides en WGS84
buildings_wgs = buildings.to_crs("EPSG:4326")
centroids = buildings_wgs.geometry.centroid

# Crear dataframe con los datos
points_data = pd.DataFrame({
    'lon': centroids.x.values,
    'lat': centroids.y.values,
    'area': buildings_proj['area_m2'].values
})

# Filtrar puntos invalidos
points_data = points_data.dropna()

if len(points_data) == 0:
    st.error("No hay datos validos para visualizar.")
    st.stop()

# Calcular centro del mapa
center_lon = points_data['lon'].mean()
center_lat = points_data['lat'].mean()

# ============================================================================
# CREAR VISUALIZACION 3D
# ============================================================================

st.subheader(f"Mapa de Densidad ({len(points_data)} edificios)")

# Vista inicial
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=13,
    pitch=45,
    bearing=0
)

# Capa hexagonal de densidad
hex_layer = pdk.Layer(
    "HexagonLayer",
    data=points_data,
    get_position=['lon', 'lat'],
    radius=cell_size,
    elevation_scale=height_multiplier,
    elevation_range=[0, 300],
    extruded=True,
    pickable=True,
    coverage=0.8,
    color_range=[
        [255, 255, 178],
        [254, 217, 118],
        [254, 178, 76],
        [253, 141, 60],
        [240, 59, 32],
        [189, 0, 38]
    ]
)

# Tooltip
tooltip = {
    "html": "<b>Zona</b><br/>Edificios en celda: {elevationValue}",
    "style": {"backgroundColor": "#2c3e50", "color": "white"}
}

# Renderizar mapa
st.pydeck_chart(pdk.Deck(
    layers=[hex_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/dark-v10"
))

# ============================================================================
# ESTADISTICAS
# ============================================================================

st.markdown("---")
st.subheader("Estadisticas")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total edificios", len(points_data))
with col2:
    st.metric("Area promedio", f"{points_data['area'].mean():.0f} m2")
with col3:
    st.metric("Area total", f"{points_data['area'].sum()/10000:.1f} ha")

# ============================================================================
# INTERPRETACION
# ============================================================================

st.markdown("---")
st.subheader("Interpretacion")

st.markdown("""
**Lectura del modelo:**
- Las columnas mas altas (rojo) indican zonas con mayor concentracion de edificios
- Las columnas bajas (amarillo) representan areas con menos construcciones
- La ausencia de columnas indica zonas sin edificaciones

**Patrones observados:**
- La mayor densidad se concentra en **Hanga Roa**, el unico centro urbano de la isla
- Las zonas perifericas muestran una densidad constructiva mucho menor
- Este patron refleja la concentracion historica de la poblacion en un solo asentamiento
""")

st.markdown("---")
st.caption("Modelo 3D de Densidad - Isla de Pascua | Laboratorio Integrador 2025")
