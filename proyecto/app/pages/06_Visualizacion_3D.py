import streamlit as st
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import rasterio

st.set_page_config(page_title="Visualización 3D - Isla de Pascua", layout="wide", page_icon="🏔️")

st.title("🏔️ Visualización 3D del Terreno")
st.markdown("---")

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"

st.markdown("""
Explora el terreno de Isla de Pascua en 3D con elevación real del DEM.
""")

# ============================================================================
# CARGAR DATOS
# ============================================================================

@st.cache_data
def load_grid_data():
    """Cargar grilla con datos topográficos"""
    grid_file = DATA_PROCESSED / "grid_with_topography.gpkg"
    
    if not grid_file.exists():
        grid_file = DATA_PROCESSED / "prepared.gpkg"
    
    if grid_file.exists():
        try:
            grid = gpd.read_file(grid_file)
        except:
            import fiona
            layers = fiona.listlayers(grid_file)
            grid = gpd.read_file(grid_file, layer=layers[0])
        
        # Convertir a WGS84 para pydeck
        if grid.crs != "EPSG:4326":
            grid = grid.to_crs("EPSG:4326")
        
        # Extraer coordenadas del centroide
        grid['lon'] = grid.geometry.centroid.x
        grid['lat'] = grid.geometry.centroid.y
        
        return grid
    return None

grid = load_grid_data()

if grid is None:
    st.error("No se pudieron cargar los datos. Ejecuta primero los scripts de procesamiento.")
    st.stop()

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

st.sidebar.header("⚙️ Configuración 3D")

# Seleccionar variable a visualizar
numeric_cols = grid.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['lon', 'lat', 'cell_id', 'index', 'fid']
available_vars = [col for col in numeric_cols if col not in exclude]

if not available_vars:
    available_vars = ['elevation_mean'] if 'elevation_mean' in grid.columns else numeric_cols[:1]

color_var = st.sidebar.selectbox(
    "Variable para color:",
    options=available_vars,
    index=0 if 'elevation_mean' in available_vars else 0
)

# Configuración de elevación
elevation_var = st.sidebar.selectbox(
    "Variable para elevación:",
    options=available_vars,
    index=0 if 'elevation_mean' in available_vars else 0
)

elevation_scale = st.sidebar.slider(
    "Escala de elevación:",
    min_value=1,
    max_value=100,
    value=20,
    help="Multiplica la elevación para exageración vertical"
)

# Configuración de vista
view_pitch = st.sidebar.slider(
    "Ángulo de vista (pitch):",
    min_value=0,
    max_value=90,
    value=45
)

view_zoom = st.sidebar.slider(
    "Zoom:",
    min_value=10,
    max_value=15,
    value=12
)

# ============================================================================
# PREPARAR DATOS PARA PYDECK
# ============================================================================

# Asegurar que las variables existen
if color_var not in grid.columns:
    color_var = numeric_cols[0]
if elevation_var not in grid.columns:
    elevation_var = numeric_cols[0]

# Normalizar variable de color para RGB
color_values = grid[color_var].fillna(0)
color_normalized = (color_values - color_values.min()) / (color_values.max() - color_values.min())

# Crear colores RGB (gradiente azul -> verde -> rojo)
def value_to_color(value):
    """Convertir valor normalizado [0,1] a color RGB"""
    if value < 0.5:
        # Azul a verde
        r = 0
        g = int(255 * (value * 2))
        b = int(255 * (1 - value * 2))
    else:
        # Verde a rojo
        r = int(255 * ((value - 0.5) * 2))
        g = int(255 * (1 - (value - 0.5) * 2))
        b = 0
    return [r, g, b, 200]

grid['color'] = color_normalized.apply(value_to_color)

# Elevación
grid['elevation_3d'] = grid[elevation_var].fillna(0) * elevation_scale

# ============================================================================
# CREAR VISUALIZACIÓN 3D
# ============================================================================

# Calcular centro
center_lat = grid['lat'].mean()
center_lon = grid['lon'].mean()

# Configurar vista inicial
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=view_zoom,
    pitch=view_pitch,
    bearing=0
)

# Capa de columnas 3D
column_layer = pdk.Layer(
    "ColumnLayer",
    data=grid,
    get_position=["lon", "lat"],
    get_elevation="elevation_3d",
    elevation_scale=1,
    radius=100,
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

# Capa de polígonos (base)
polygon_layer = pdk.Layer(
    "GeoJsonLayer",
    data=grid.__geo_interface__,
    get_fill_color="color",
    get_line_color=[255, 255, 255, 100],
    line_width_min_pixels=1,
    pickable=True,
    auto_highlight=True,
)

# Tooltip
tooltip = {
    "html": f"<b>Celda</b><br/>"
            f"<b>{color_var}:</b> {{{color_var}}}<br/>"
            f"<b>{elevation_var}:</b> {{{elevation_var}}}<br/>",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

# Renderizar mapa
st.pydeck_chart(pdk.Deck(
    layers=[column_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/satellite-v9"
))

# ============================================================================
# ESTADÍSTICAS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"{color_var} (promedio)",
        value=f"{grid[color_var].mean():.2f}",
        delta=f"±{grid[color_var].std():.2f}"
    )

with col2:
    st.metric(
        label=f"{color_var} (mínimo)",
        value=f"{grid[color_var].min():.2f}"
    )

with col3:
    st.metric(
        label=f"{color_var} (máximo)",
        value=f"{grid[color_var].max():.2f}"
    )

# ============================================================================
# INFORMACIÓN
# ============================================================================

with st.expander("ℹ️ Información sobre la visualización"):
    st.markdown("""
    ### Controles del Mapa 3D
    
    - **Rotar**: Click izquierdo + arrastrar
    - **Inclinar**: Click derecho + arrastrar
    - **Zoom**: Rueda del ratón o doble click
    - **Pan**: Shift + click izquierdo + arrastrar
    
    ### Variables Disponibles
    
    - **Elevación**: Altura sobre el nivel del mar (DEM)
    - **Pendiente (slope)**: Inclinación del terreno en grados
    - **Aspecto (aspect)**: Orientación de la pendiente (0-360°)
    - **Otras variables**: Según análisis realizados
    
    ### Interpretación
    
    - **Colores fríos (azul)**: Valores bajos
    - **Colores cálidos (rojo)**: Valores altos
    - **Altura de columnas**: Proporcional a la variable de elevación seleccionada
    """)

st.markdown("---")
st.caption("Visualización 3D - Isla de Pascua | Elemento de Excelencia")
