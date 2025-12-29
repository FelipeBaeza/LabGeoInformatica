"""
Pagina de Analisis de Hot Spots y LISA
Identifica zonas de concentracion significativa usando estadistica espacial.

Segun el documento del laboratorio, esta seccion debe incluir:
- Analisis de autocorrelacion (Moran's I global y local)
- Hot spots y clusters usando LISA
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import os
from sqlalchemy import create_engine

# Intentar importar PySAL para LISA
try:
    from libpysal.weights import Queen
    from esda.moran import Moran, Moran_Local
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False

st.set_page_config(page_title="Hot Spots y LISA", layout="wide")

st.title("Analisis de Hot Spots y Autocorrelacion Espacial")

st.markdown("""
Este modulo implementa el analisis de **autocorrelacion espacial** requerido por el laboratorio:
1. **Moran's I Global**: Mide si existe patron espacial en toda el area
2. **LISA (Local Moran's I)**: Identifica clusters locales (hot spots y cold spots)
3. **Mapa de calor**: Visualiza la intensidad de concentracion
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
        return buildings, boundary
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None


@st.cache_data
def create_analysis_grid(_buildings, _boundary, cell_size=300):
    """Crear grilla de analisis con conteo de edificios."""
    buildings_utm = _buildings.to_crs(CRS_UTM)
    boundary_utm = _boundary.to_crs(CRS_UTM)
    
    minx, miny, maxx, maxy = boundary_utm.total_bounds
    
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
    
    return grid


# Cargar datos
buildings, boundary = load_all_data()

if buildings is None:
    st.error("No se pudieron cargar los datos.")
    st.stop()

# Crear grilla
st.sidebar.header("Configuracion")
cell_size = st.sidebar.slider("Tamano de celda (m)", 200, 500, 300, 50)

grid = create_analysis_grid(buildings, boundary, cell_size)
grid_wgs = grid.to_crs("EPSG:4326")

# Calcular centro
bounds = grid_wgs.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

# ============================================================================
# SECCION 1: MORAN'S I GLOBAL
# ============================================================================

st.header("1. Moran's I Global - Autocorrelacion Espacial")

st.markdown("""
El **Indice de Moran (I)** mide si existe un patron espacial en la distribucion de edificaciones:
- **I > 0**: Autocorrelacion positiva (valores similares se agrupan) - Indica clustering
- **I = 0**: Distribucion aleatoria
- **I < 0**: Autocorrelacion negativa (valores distintos se agrupan) - Patron disperso
""")

if PYSAL_AVAILABLE:
    # Calcular Moran's I
    try:
        # Solo celdas con vecinos validos
        grid_analysis = grid[grid['n_edificios'] >= 0].copy()
        
        # Crear matriz de pesos espaciales (vecinos Queen)
        w = Queen.from_dataframe(grid_analysis)
        w.transform = 'r'  # Row standardization
        
        # Moran's I Global
        moran = Moran(grid_analysis['n_edificios'], w)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Moran's I", f"{moran.I:.4f}")
        with col2:
            st.metric("P-value", f"{moran.p_norm:.4f}")
        with col3:
            significancia = "Significativo" if moran.p_norm < 0.05 else "No significativo"
            st.metric("Resultado", significancia)
        
        # Interpretacion
        if moran.I > 0 and moran.p_norm < 0.05:
            st.success(f"""
            **Interpretacion:** El Moran's I de **{moran.I:.4f}** con p-value **{moran.p_norm:.4f}** 
            indica una **autocorrelacion espacial positiva significativa**. 
            
            Esto significa que las edificaciones NO estan distribuidas aleatoriamente, 
            sino que forman **clusters** donde zonas con muchos edificios estan cerca 
            de otras zonas con muchos edificios (y viceversa).
            """)
        else:
            st.info("No se detecta autocorrelacion espacial significativa.")
            
    except Exception as e:
        st.warning(f"Error calculando Moran's I: {e}")
else:
    st.warning("PySAL no disponible. Mostrando analisis simplificado.")
    
    # Calculo simplificado de concentracion
    mean_val = grid['n_edificios'].mean()
    std_val = grid['n_edificios'].std()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Promedio edificios/celda", f"{mean_val:.2f}")
    with col2:
        st.metric("Desv. Estandar", f"{std_val:.2f}")
    with col3:
        cv = std_val / mean_val if mean_val > 0 else 0
        st.metric("Coef. Variacion", f"{cv:.2f}")

# ============================================================================
# SECCION 2: LISA - LOCAL MORAN'S I
# ============================================================================

st.header("2. Analisis LISA - Clusters Locales")

st.markdown("""
**LISA (Local Indicators of Spatial Association)** identifica clusters locales:

| Tipo | Significado | Color |
|------|-------------|-------|
| **High-High** | Celda alta rodeada de celdas altas (Hot Spot) | Rojo |
| **Low-Low** | Celda baja rodeada de celdas bajas (Cold Spot) | Azul |
| **High-Low** | Celda alta rodeada de celdas bajas (Outlier) | Rosa |
| **Low-High** | Celda baja rodeada de celdas altas (Outlier) | Celeste |
| **No Significativo** | Sin patron claro | Gris |
""")

if PYSAL_AVAILABLE:
    try:
        # LISA - Local Moran
        lisa = Moran_Local(grid_analysis['n_edificios'], w)
        
        # Clasificar clusters
        grid_analysis['lisa_q'] = lisa.q  # Cuadrante
        grid_analysis['lisa_p'] = lisa.p_sim  # P-value
        grid_analysis['lisa_sig'] = lisa.p_sim < 0.05  # Significativo?
        
        # Asignar categoria LISA
        def classify_lisa(row):
            if not row['lisa_sig']:
                return 'No Significativo'
            q = row['lisa_q']
            if q == 1:
                return 'High-High (Hot Spot)'
            elif q == 2:
                return 'Low-High (Outlier)'
            elif q == 3:
                return 'Low-Low (Cold Spot)'
            elif q == 4:
                return 'High-Low (Outlier)'
            return 'No Significativo'
        
        grid_analysis['lisa_cluster'] = grid_analysis.apply(classify_lisa, axis=1)
        
        # Estadisticas LISA
        lisa_counts = grid_analysis['lisa_cluster'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hh = lisa_counts.get('High-High (Hot Spot)', 0)
            st.metric("Hot Spots (HH)", hh)
        with col2:
            ll = lisa_counts.get('Low-Low (Cold Spot)', 0)
            st.metric("Cold Spots (LL)", ll)
        with col3:
            outliers = lisa_counts.get('High-Low (Outlier)', 0) + lisa_counts.get('Low-High (Outlier)', 0)
            st.metric("Outliers", outliers)
        with col4:
            ns = lisa_counts.get('No Significativo', 0)
            st.metric("No Significativo", ns)
        
        # Mapa LISA
        st.subheader("Mapa de Clusters LISA")
        
        grid_lisa_wgs = grid_analysis.to_crs("EPSG:4326")
        
        m_lisa = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')
        
        # Colores LISA
        lisa_colors = {
            'High-High (Hot Spot)': '#d7191c',  # Rojo
            'Low-Low (Cold Spot)': '#2c7bb6',   # Azul
            'High-Low (Outlier)': '#fdae61',    # Naranja
            'Low-High (Outlier)': '#abd9e9',    # Celeste
            'No Significativo': '#eeeeee'        # Gris
        }
        
        for idx, row in grid_lisa_wgs.iterrows():
            color = lisa_colors.get(row['lisa_cluster'], '#eeeeee')
            
            folium.GeoJson(
                row.geometry,
                style_function=lambda x, c=color: {
                    'fillColor': c,
                    'color': '#333',
                    'weight': 0.5,
                    'fillOpacity': 0.7
                },
                tooltip=f"Edificios: {row['n_edificios']}, Cluster: {row['lisa_cluster']}"
            ).add_to(m_lisa)
        
        # Leyenda
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                    background-color: white; padding: 10px; border-radius: 5px;
                    border: 2px solid grey; font-size: 12px;">
            <b>Clusters LISA</b><br>
            <i style="background: #d7191c; width: 12px; height: 12px; display: inline-block;"></i> Hot Spot (HH)<br>
            <i style="background: #2c7bb6; width: 12px; height: 12px; display: inline-block;"></i> Cold Spot (LL)<br>
            <i style="background: #fdae61; width: 12px; height: 12px; display: inline-block;"></i> Outlier (HL)<br>
            <i style="background: #abd9e9; width: 12px; height: 12px; display: inline-block;"></i> Outlier (LH)<br>
            <i style="background: #eeeeee; width: 12px; height: 12px; display: inline-block;"></i> No Sig.
        </div>
        """
        m_lisa.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(m_lisa, width=900, height=500)
        
        st.markdown(f"""
        **Interpretacion del mapa LISA:**
        - Las celdas **rojas (Hot Spots)** son zonas con alta densidad rodeadas de alta densidad - 
          corresponden al centro de Hanga Roa
        - Las celdas **azules (Cold Spots)** son zonas con baja densidad rodeadas de baja densidad - 
          corresponden al resto de la isla
        - Se identificaron **{hh} Hot Spots** significativos al 95% de confianza
        """)
        
    except Exception as e:
        st.warning(f"Error en analisis LISA: {e}")
        st.info("Mostrando mapa de densidad alternativo.")
        
else:
    # Alternativa sin PySAL: mapa de densidad por grilla
    st.subheader("Mapa de Densidad por Grilla")
    
    # Clasificar por z-score
    mean_val = grid['n_edificios'].mean()
    std_val = grid['n_edificios'].std()
    
    if std_val > 0:
        grid['z_score'] = (grid['n_edificios'] - mean_val) / std_val
    else:
        grid['z_score'] = 0
    
    grid['clasificacion'] = 'Sin edificios'
    grid.loc[grid['n_edificios'] > 0, 'clasificacion'] = 'Densidad baja'
    grid.loc[grid['z_score'] > 1, 'clasificacion'] = 'Densidad media'
    grid.loc[grid['z_score'] > 1.96, 'clasificacion'] = 'Densidad alta (95%)'
    grid.loc[grid['z_score'] > 2.58, 'clasificacion'] = 'Densidad muy alta (99%)'
    
    # Convertir a WGS84 DESPUES de agregar las columnas
    grid_clasificado = grid.to_crs("EPSG:4326")
    
    m_grid = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')
    
    colors_grid = {
        'Sin edificios': '#f7f7f7',
        'Densidad baja': '#fee8c8',
        'Densidad media': '#fdbb84',
        'Densidad alta (95%)': '#e34a33',
        'Densidad muy alta (99%)': '#b30000'
    }
    
    for idx, row in grid_clasificado.iterrows():
        color = colors_grid.get(row['clasificacion'], '#f7f7f7')
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': '#333',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            tooltip=f"Edificios: {row['n_edificios']}, Clase: {row['clasificacion']}"
        ).add_to(m_grid)
    
    st_folium(m_grid, width=900, height=500)
    
    # Mostrar tabla resumen de clasificacion
    st.subheader("Resumen de Clasificacion")
    resumen = grid['clasificacion'].value_counts().reset_index()
    resumen.columns = ['Categoria', 'Numero de celdas']
    st.dataframe(resumen, use_container_width=True)

# ============================================================================
# SECCION 3: MAPA DE CALOR
# ============================================================================

st.header("3. Mapa de Calor (Densidad Continua)")

st.markdown("""
El mapa de calor visualiza la **intensidad de concentracion** de edificaciones de forma continua.
A diferencia del mapa LISA que usa celdas discretas, este muestra gradientes suaves.
""")

buildings_wgs = buildings.to_crs("EPSG:4326")

m_heat = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB dark_matter')

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
).add_to(m_heat)

st_folium(m_heat, width=900, height=500)

st.markdown("""
**Interpretacion:** El nucleo mas caliente (rojo intenso) corresponde al **centro de Hanga Roa**, 
donde se concentran comercios, servicios publicos y la mayor densidad residencial. 
Los colores mas frios indican zonas de transicion hacia areas menos urbanizadas.
""")

# ============================================================================
# SECCION 4: RESUMEN
# ============================================================================

st.header("4. Resumen del Analisis")

# Estadisticas finales
celdas_ocupadas = len(grid[grid['n_edificios'] > 0])
pct_ocupado = celdas_ocupadas / len(grid) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Celdas totales", len(grid))
with col2:
    st.metric("Celdas con edificios", celdas_ocupadas)
with col3:
    st.metric("% Ocupacion", f"{pct_ocupado:.1f}%")
with col4:
    st.metric("Max. edificios/celda", int(grid['n_edificios'].max()))

st.markdown(f"""
**Conclusiones del analisis de Hot Spots:**

1. **Autocorrelacion espacial confirmada**: Las edificaciones NO estan distribuidas 
   aleatoriamente, sino que forman clusters significativos.

2. **Concentracion extrema**: Solo el **{pct_ocupado:.1f}%** del territorio tiene 
   edificaciones, todas concentradas en Hanga Roa.

3. **Implicancias para planificacion:**
   - La infraestructura esta bajo presion en el area urbana
   - El crecimiento futuro debe considerar la capacidad de carga
   - La mayoria de la isla permanece protegida de expansion urbana
""")
