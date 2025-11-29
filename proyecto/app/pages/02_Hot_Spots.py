"""
Página de Análisis de Hot Spots (Getis-Ord Gi*)
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from shapely.geometry import box
import os

# Intentar importar librerías de análisis espacial
try:
    from libpysal.weights import Queen, KNN
    from esda.getisord import G_Local
    SPATIAL_LIBS = True
except ImportError:
    SPATIAL_LIBS = False

# Configuración
st.set_page_config(page_title="Análisis Hot Spots", page_icon="🔥", layout="wide")
st.title("🔥 Análisis de Hot Spots")
st.markdown("---")

# Rutas
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data', 'raw', 'isla_de_pascua')
CRS_UTM = 'EPSG:32719'

@st.cache_data
def load_data():
    """Cargar datos geoespaciales"""
    data = {}
    files = {
        'boundary': 'isla_de_pascua_boundary.geojson',
        'buildings': 'isla_de_pascua_buildings.geojson',
        'amenities': 'isla_de_pascua_amenities.geojson',
        'streets': 'isla_de_pascua_streets.geojson',
    }
    
    for key, filename in files.items():
        filepath = os.path.join(DATA_PATH, filename)
        if os.path.exists(filepath):
            try:
                data[key] = gpd.read_file(filepath)
            except Exception as e:
                pass
    
    return data

@st.cache_data
def create_grid(boundary_gdf, cell_size=200):
    """Crear grilla de análisis"""
    boundary_utm = boundary_gdf.to_crs(CRS_UTM)
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
    
    return grid

def count_features_in_cells(grid, features_gdf, column_name='count'):
    """Contar features en cada celda de la grilla"""
    features_utm = features_gdf.to_crs(CRS_UTM)
    
    # Para polígonos, usar centroide
    if 'Polygon' in str(features_utm.geometry.geom_type.iloc[0]):
        features_utm['point'] = features_utm.geometry.centroid
        features_points = gpd.GeoDataFrame(features_utm, geometry='point', crs=CRS_UTM)
    else:
        features_points = features_utm
    
    grid[column_name] = 0
    
    for idx, cell in grid.iterrows():
        count = len(features_points[features_points.geometry.within(cell.geometry)])
        grid.loc[idx, column_name] = count
    
    return grid

# Cargar datos
try:
    data = load_data()
    
    if not data or 'boundary' not in data:
        st.error("No se encontraron datos. Ejecute primero el script de descarga.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔥 Opciones de Hot Spots")
    
    # Seleccionar capa
    available_layers = [k for k in data.keys() if k != 'boundary']
    if not available_layers:
        st.error("No hay capas disponibles para análisis")
        st.stop()
    
    selected_layer = st.sidebar.selectbox(
        "Capa a analizar:",
        available_layers,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    cell_size = st.sidebar.slider("Tamaño de celda (metros)", 100, 500, 200, 50)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Mapa de Calor", "📊 Getis-Ord Gi*", "📈 Estadísticas"])
    
    with tab1:
        st.subheader("Mapa de Calor")
        
        gdf = data[selected_layer].to_crs('EPSG:4326')
        center = data['boundary'].to_crs('EPSG:4326').geometry.centroid.iloc[0]
        
        # Crear mapa base
        m = folium.Map(
            location=[center.y, center.x],
            zoom_start=13,
            tiles='CartoDB dark_matter'
        )
        
        # Obtener puntos para heatmap
        heat_data = []
        for idx, row in gdf.iterrows():
            if row.geometry:
                geom = row.geometry
                if geom.geom_type == 'Point':
                    heat_data.append([geom.y, geom.x])
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = geom.centroid
                    heat_data.append([centroid.y, centroid.x])
        
        if heat_data:
            # Añadir heatmap
            HeatMap(
                heat_data,
                radius=20,
                blur=15,
                max_zoom=18,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'}
            ).add_to(m)
            
            st_folium(m, width=None, height=600)
        else:
            st.warning("No hay datos de puntos para visualizar")
    
    with tab2:
        st.subheader("Análisis Getis-Ord Gi* (Hot Spot Analysis)")
        
        if not SPATIAL_LIBS:
            st.warning("⚠️ Las librerías de análisis espacial (libpysal, esda) no están instaladas.")
            st.info("Instale con: pip install libpysal esda")
        else:
            # Crear grilla
            with st.spinner("Creando grilla de análisis..."):
                grid = create_grid(data['boundary'], cell_size)
                grid = count_features_in_cells(grid, data[selected_layer], 'n_features')
            
            st.info(f"Grilla creada: {len(grid)} celdas de {cell_size}m x {cell_size}m")
            
            # Filtrar celdas con features
            grid_analysis = grid[grid['n_features'] > 0].copy()
            
            if len(grid_analysis) < 5:
                st.warning("No hay suficientes celdas con datos para el análisis")
            else:
                # Calcular Gi*
                with st.spinner("Calculando estadístico Getis-Ord Gi*..."):
                    try:
                        w = Queen.from_dataframe(grid_analysis)
                        if w.n == 0:
                            w = KNN.from_dataframe(grid_analysis, k=4)
                        
                        gi = G_Local(grid_analysis['n_features'].values, w, star=True)
                        grid_analysis['gi_z'] = gi.Zs
                        grid_analysis['gi_p'] = gi.p_sim
                        
                        # Clasificar hot/cold spots
                        grid_analysis['hotspot_type'] = 'No significativo'
                        grid_analysis.loc[(grid_analysis['gi_z'] > 1.96) & (grid_analysis['gi_p'] < 0.05), 'hotspot_type'] = 'Hot Spot (95%)'
                        grid_analysis.loc[(grid_analysis['gi_z'] > 2.58) & (grid_analysis['gi_p'] < 0.01), 'hotspot_type'] = 'Hot Spot (99%)'
                        grid_analysis.loc[(grid_analysis['gi_z'] < -1.96) & (grid_analysis['gi_p'] < 0.05), 'hotspot_type'] = 'Cold Spot (95%)'
                        grid_analysis.loc[(grid_analysis['gi_z'] < -2.58) & (grid_analysis['gi_p'] < 0.01), 'hotspot_type'] = 'Cold Spot (99%)'
                        
                        # Visualización
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Mapa de Z-scores (Gi*)**")
                            fig, ax = plt.subplots(figsize=(10, 10))
                            data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                                   edgecolor='black', linewidth=2)
                            grid_analysis.plot(column='gi_z', ax=ax, cmap='RdBu_r', 
                                             legend=True, vmin=-3, vmax=3,
                                             legend_kwds={'label': 'Z-score (Gi*)'})
                            ax.set_title('Estadístico Getis-Ord Gi*', fontsize=12, fontweight='bold')
                            ax.set_axis_off()
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("**Clasificación de Hot/Cold Spots**")
                            fig, ax = plt.subplots(figsize=(10, 10))
                            data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                                   edgecolor='black', linewidth=2)
                            
                            colors = {
                                'No significativo': '#cccccc',
                                'Cold Spot (99%)': '#2166ac',
                                'Cold Spot (95%)': '#92c5de',
                                'Hot Spot (95%)': '#f4a582',
                                'Hot Spot (99%)': '#b2182b'
                            }
                            
                            for htype, color in colors.items():
                                subset = grid_analysis[grid_analysis['hotspot_type'] == htype]
                                if len(subset) > 0:
                                    subset.plot(ax=ax, color=color, edgecolor='gray', 
                                              linewidth=0.5, label=htype)
                            
                            ax.legend(loc='lower right', title='Tipo')
                            ax.set_title('Clasificación Hot/Cold Spots', fontsize=12, fontweight='bold')
                            ax.set_axis_off()
                            st.pyplot(fig)
                        
                        # Resumen
                        st.markdown("---")
                        st.markdown("**📊 Resumen de Hot/Cold Spots**")
                        summary = grid_analysis['hotspot_type'].value_counts()
                        st.dataframe(summary, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error en análisis: {e}")
    
    with tab3:
        st.subheader("Estadísticas de Densidad")
        
        # Crear grilla
        grid = create_grid(data['boundary'], cell_size)
        grid = count_features_in_cells(grid, data[selected_layer], 'n_features')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribución de Densidad por Celda**")
            fig, ax = plt.subplots(figsize=(8, 5))
            grid['n_features'].hist(bins=30, ax=ax, color='coral', edgecolor='white')
            ax.set_xlabel('Número de features por celda')
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Histograma de Densidad - {selected_layer}')
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Estadísticas**")
            stats = grid['n_features'].describe()
            st.dataframe(pd.DataFrame(stats).round(2))
        
        # Mapa de densidad
        st.markdown("---")
        st.markdown("**Mapa de Densidad por Celda**")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                               edgecolor='black', linewidth=2)
        grid.plot(column='n_features', ax=ax, cmap='YlOrRd', legend=True,
                 legend_kwds={'label': 'N° features'})
        ax.set_title(f'Densidad de {selected_layer.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_axis_off()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Verifique que los datos estén disponibles.")
