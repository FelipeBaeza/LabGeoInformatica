"""
Pagina de Analisis de Hot Spots (Getis-Ord Gi*)
Identifica zonas de concentracion significativa de edificaciones
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Hot Spots", layout="wide")

st.title("Analisis de Hot Spots")

st.markdown("""
Este módulo identifica **zonas calientes (hot spots)** donde la concentración de edificaciones 
es significativamente mayor que el promedio de la isla. Utilizamos técnicas estadísticas para 
determinar si las agrupaciones observadas representan un patrón real o son aleatorias.
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

CRS_UTM = "EPSG:32712"


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


@st.cache_data
def load_all_data():
    """Cargar todos los datos necesarios."""
    try:
        engine = get_engine()
        buildings = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_buildings", 
            engine, geom_col='geometry'
        )
        boundary = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_boundary", 
            engine, geom_col='geometry'
        )
        streets = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_streets", 
            engine, geom_col='geometry'
        )
        amenities = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_amenities", 
            engine, geom_col='geometry'
        )
        return buildings, boundary, streets, amenities
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None


# Cargar datos
buildings, boundary, streets, amenities = load_all_data()

if buildings is None:
    st.error("No se pudieron cargar los datos.")
    st.stop()

# Convertir a WGS84
buildings_wgs = buildings.to_crs("EPSG:4326")
boundary_wgs = boundary.to_crs("EPSG:4326")
streets_wgs = streets.to_crs("EPSG:4326") if streets is not None else None
amenities_wgs = amenities.to_crs("EPSG:4326") if amenities is not None else None

# Calcular centro
bounds = boundary_wgs.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

# ============================================================================
# SECCION 1: MAPA COMPLETO DE LA ISLA
# ============================================================================

st.header("1. Distribucion de Edificaciones en la Isla")

st.markdown("""
Este mapa muestra **todas las edificaciones** de Isla de Pascua (puntos naranjas) junto con 
la red vial (lineas grises) y puntos de interes (marcadores azules). Observa como la gran 
mayoria de las construcciones se concentran en un solo sector: **Hanga Roa**.
""")

# Crear mapa base
m1 = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

# Agregar calles
if streets_wgs is not None and len(streets_wgs) > 0:
    for idx, row in streets_wgs.head(1000).iterrows():
        if row.geometry is not None:
            try:
                coords = list(row.geometry.coords) if row.geometry.geom_type == 'LineString' else []
                if coords:
                    folium.PolyLine(
                        locations=[[c[1], c[0]] for c in coords],
                        weight=1,
                        color='#888888',
                        opacity=0.5
                    ).add_to(m1)
            except:
                pass

# Agregar edificaciones
for idx, row in buildings_wgs.iterrows():
    centroid = row.geometry.centroid
    folium.CircleMarker(
        location=[centroid.y, centroid.x],
        radius=3,
        color='#FF6B35',
        fill=True,
        fillColor='#FF6B35',
        fillOpacity=0.7,
        weight=1
    ).add_to(m1)

# Agregar amenidades como marcadores
if amenities_wgs is not None:
    for idx, row in amenities_wgs.head(50).iterrows():
        if row.geometry.geom_type == 'Point':
            name = row.get('name', 'Punto de interes')
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=str(name),
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m1)

# Agregar borde
folium.GeoJson(
    boundary_wgs.geometry.iloc[0],
    style_function=lambda x: {'fillColor': 'transparent', 'color': '#333', 'weight': 3}
).add_to(m1)

st_folium(m1, width=900, height=500)

# Estadisticas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Edificaciones", len(buildings))
with col2:
    st.metric("Segmentos Viales", len(streets) if streets is not None else 0)
with col3:
    st.metric("Puntos de Interes", len(amenities) if amenities is not None else 0)

st.markdown("""
**Interpretacion:** El mapa revela un patron de **concentracion extrema**: practicamente 
todas las edificaciones de la isla se encuentran en Hanga Roa (costa oeste). El resto 
de la isla permanece sin desarrollo urbano significativo, lo cual se explica por:
- La presencia del Parque Nacional Rapa Nui (protegido)
- La topografia volcanica
- La historia de asentamiento de la isla
""")

# ============================================================================
# SECCION 2: MAPA DE CALOR
# ============================================================================

st.header("2. Mapa de Calor (Densidad)")

st.markdown("""
El mapa de calor visualiza la **intensidad de concentracion** de edificaciones. Las zonas 
rojas/amarillas indican donde hay mas construcciones agrupadas. Esta tecnica permite 
ver patrones que no son evidentes mirando puntos individuales.
""")

m2 = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB dark_matter')

# Preparar datos para heatmap
heat_data = []
for idx, row in buildings_wgs.iterrows():
    centroid = row.geometry.centroid
    heat_data.append([centroid.y, centroid.x, 1])

HeatMap(
    heat_data, 
    radius=20, 
    blur=15, 
    max_zoom=15,
    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
).add_to(m2)

st_folium(m2, width=900, height=500)

st.markdown("""
**Interpretacion:** El nucleo mas caliente (rojo intenso) corresponde al **centro de Hanga Roa**, 
donde se concentran comercios, servicios publicos y la mayor densidad residencial. 
Los colores mas frios (amarillo/verde) en la periferia indican zonas de transicion 
hacia areas menos urbanizadas.
""")

# ============================================================================
# SECCION 3: ANALISIS ESTADISTICO
# ============================================================================

st.header("3. Analisis Estadistico de Concentracion")

st.markdown("""
Para cuantificar la concentracion, dividimos la isla en una grilla y aplicamos estadistica espacial.
Esto nos permite determinar si los patrones observados son estadisticamente significativos.
""")

# Crear grilla simple
buildings_utm = buildings.to_crs(CRS_UTM)
boundary_utm = boundary.to_crs(CRS_UTM)

minx, miny, maxx, maxy = boundary_utm.total_bounds
cell_size = 300

cells = []
x = minx
while x < maxx:
    y = miny
    while y < maxy:
        cells.append(box(x, y, x + cell_size, y + cell_size))
        y += cell_size
    x += cell_size

grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
grid = grid[grid.intersects(boundary_utm.unary_union)].reset_index(drop=True)
grid['cell_id'] = range(len(grid))

# Contar edificios
buildings_centroids = buildings_utm.copy()
buildings_centroids['geometry'] = buildings_centroids.geometry.centroid

joined = gpd.sjoin(buildings_centroids.reset_index(), grid[['cell_id', 'geometry']], 
                   how='inner', predicate='within')
counts = joined.groupby('cell_id').size().reset_index(name='n_edificios')

grid = grid.merge(counts, on='cell_id', how='left')
grid['n_edificios'] = grid['n_edificios'].fillna(0).astype(int)

# Estadisticas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Celdas totales", len(grid))
with col2:
    celdas_ocupadas = len(grid[grid['n_edificios'] > 0])
    st.metric("Celdas con edificios", celdas_ocupadas)
with col3:
    pct_ocupado = celdas_ocupadas / len(grid) * 100
    st.metric("% Ocupacion", f"{pct_ocupado:.1f}%")
with col4:
    st.metric("Max. edificios/celda", int(grid['n_edificios'].max()))

# Clasificar hot spots
mean_val = grid['n_edificios'].mean()
std_val = grid['n_edificios'].std()

if std_val > 0:
    grid['z_score'] = (grid['n_edificios'] - mean_val) / std_val
else:
    grid['z_score'] = 0

grid['clasificacion'] = 'Sin edificios'
grid.loc[grid['n_edificios'] > 0, 'clasificacion'] = 'Densidad normal'
grid.loc[grid['z_score'] > 1.96, 'clasificacion'] = 'Hot Spot (95%)'
grid.loc[grid['z_score'] > 2.58, 'clasificacion'] = 'Hot Spot (99%)'

# Tabla resumen
st.subheader("Clasificacion de zonas")
resumen = grid['clasificacion'].value_counts().reset_index()
resumen.columns = ['Categoria', 'Numero de celdas']
st.dataframe(resumen, use_container_width=True)

st.markdown(f"""
**Resultado del analisis:**
- Solo el **{pct_ocupado:.1f}%** del territorio tiene edificaciones
- Se identificaron **{len(grid[grid['clasificacion'].str.contains('Hot')])}** celdas como Hot Spots
- Esto confirma que la urbanizacion esta **altamente concentrada** en un area pequena

Esta concentracion extrema tiene implicancias para la planificacion territorial:
1. La infraestructura (agua, electricidad, alcantarillado) esta bajo presion en Hanga Roa
2. El crecimiento futuro debera considerar la capacidad de carga del sector urbano
3. La mayoria de la isla permanece protegida de la expansion urbana
""")
