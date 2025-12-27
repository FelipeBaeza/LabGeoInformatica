"""
Página de Machine Learning Espacial - Predicciones
Optimizado para rendimiento con operaciones vectorizadas
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import os
from sqlalchemy import create_engine

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# Configuración
st.set_page_config(page_title="ML Espacial", page_icon=None, layout="wide")
st.title("Machine Learning Espacial")
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
    }

    engine = get_engine()
    for key, table in tables.items():
        try:
            data[key] = gpd.read_postgis(
                f"SELECT * FROM geoanalisis.{table}", engine, geom_col='geometry'
            )
        except Exception as e:
            pass  # Silently skip missing tables
    return data


@st.cache_data
def create_grid(boundary_wkt, cell_size=200):
    """Crear grilla base (cacheada)"""
    from shapely import wkt
    boundary_geom = wkt.loads(boundary_wkt)
    boundary = gpd.GeoDataFrame(geometry=[boundary_geom], crs=CRS_UTM)
    
    minx, miny, maxx, maxy = boundary.total_bounds
    
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size
    
    grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
    grid = grid[grid.intersects(boundary.unary_union)].reset_index(drop=True)
    grid['cell_id'] = range(len(grid))
    
    return grid


@st.cache_data
def create_grid_with_features_optimized(boundary_wkt, buildings_wkt_list, amenities_wkt_list, 
                                         streets_wkt_list, cell_size=200):
    """
    Crear grilla con todas las features usando operaciones vectorizadas.
    Los datos se pasan como WKT para permitir caching.
    """
    from shapely import wkt
    
    # Reconstruir GeoDataFrames desde WKT
    boundary_geom = wkt.loads(boundary_wkt)
    boundary = gpd.GeoDataFrame(geometry=[boundary_geom], crs=CRS_UTM)
    
    # Crear grilla
    minx, miny, maxx, maxy = boundary.total_bounds
    
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size
    
    grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_UTM)
    grid = grid[grid.intersects(boundary.unary_union)].reset_index(drop=True)
    grid['cell_id'] = range(len(grid))
    
    # Feature 1: Número de edificios (usando spatial join vectorizado)
    if buildings_wkt_list:
        buildings_geoms = [wkt.loads(w) for w in buildings_wkt_list]
        buildings = gpd.GeoDataFrame(geometry=buildings_geoms, crs=CRS_UTM)
        buildings['centroid'] = buildings.geometry.centroid
        buildings_points = gpd.GeoDataFrame(geometry=buildings['centroid'], crs=CRS_UTM)
        
        # Spatial join vectorizado
        joined = gpd.sjoin(buildings_points.reset_index(), grid[['cell_id', 'geometry']], 
                          how='inner', predicate='within')
        counts = joined.groupby('cell_id').size().reset_index(name='n_buildings')
        grid = grid.merge(counts, on='cell_id', how='left')
        grid['n_buildings'] = grid['n_buildings'].fillna(0).astype(int)
        
        # Área total de edificios por celda
        buildings['area'] = buildings.geometry.area
        buildings_with_cell = gpd.sjoin(buildings.reset_index(), grid[['cell_id', 'geometry']], 
                                        how='inner', predicate='intersects')
        area_sum = buildings_with_cell.groupby('cell_id')['area'].sum().reset_index(name='total_building_area')
        grid = grid.merge(area_sum, on='cell_id', how='left')
        grid['total_building_area'] = grid['total_building_area'].fillna(0)
    else:
        grid['n_buildings'] = 0
        grid['total_building_area'] = 0.0
    
    # Feature 2: Amenidades
    if amenities_wkt_list:
        amenities_geoms = [wkt.loads(w) for w in amenities_wkt_list]
        amenities = gpd.GeoDataFrame(geometry=amenities_geoms, crs=CRS_UTM)
        # Para puntos, usar directamente; para polígonos, usar centroide
        if 'Polygon' in str(amenities.geometry.geom_type.iloc[0] if len(amenities) > 0 else ''):
            amenities['point'] = amenities.geometry.centroid
            amenities_points = gpd.GeoDataFrame(geometry=amenities['point'], crs=CRS_UTM)
        else:
            amenities_points = amenities
        
        joined = gpd.sjoin(amenities_points.reset_index(), grid[['cell_id', 'geometry']], 
                          how='inner', predicate='within')
        counts = joined.groupby('cell_id').size().reset_index(name='n_amenities')
        grid = grid.merge(counts, on='cell_id', how='left')
        grid['n_amenities'] = grid['n_amenities'].fillna(0).astype(int)
    else:
        grid['n_amenities'] = 0
    
    # Feature 3: Distancia al centro (vectorizado)
    island_center = boundary.geometry.centroid.values[0]
    grid['centroid'] = grid.geometry.centroid
    grid['dist_to_center'] = grid['centroid'].distance(island_center)
    
    # Feature 4: Longitud de calles (simplificado - conteo de intersecciones)
    if streets_wkt_list:
        streets_geoms = [wkt.loads(w) for w in streets_wkt_list]
        streets = gpd.GeoDataFrame(geometry=streets_geoms, crs=CRS_UTM)
        
        # Contar calles que intersectan cada celda
        joined = gpd.sjoin(streets.reset_index(), grid[['cell_id', 'geometry']], 
                          how='inner', predicate='intersects')
        counts = joined.groupby('cell_id').size().reset_index(name='n_streets')
        grid = grid.merge(counts, on='cell_id', how='left')
        grid['n_streets'] = grid['n_streets'].fillna(0).astype(int)
        # Usar n_streets * cell_size como proxy de longitud
        grid['street_length'] = grid['n_streets'] * cell_size * 0.5
    else:
        grid['n_streets'] = 0
        grid['street_length'] = 0.0
    
    # Feature 5: Coordenadas normalizadas (vectorizado)
    grid['x_norm'] = (grid.geometry.centroid.x - minx) / (maxx - minx)
    grid['y_norm'] = (grid.geometry.centroid.y - miny) / (maxy - miny)
    
    return grid


# Cargar datos
try:
    data = load_data()
    
    if not data or 'boundary' not in data:
        st.error("No se encontraron datos")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Opciones de ML")
    
    cell_size = st.sidebar.slider("Tamaño de celda (m)", 100, 500, 200, 50)
    
    task_type = st.sidebar.selectbox(
        "Tipo de tarea:",
        ["Regresión", "Clasificación"]
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Entrenar Modelo", "Predicciones", "Evaluación"])
    
    with tab1:
        st.subheader("Entrenar Modelo de Machine Learning")
        
        st.markdown("""
        Este modelo predice la **densidad de edificaciones** en cada celda 
        basándose en características espaciales como:
        - Distancia al centro de la isla
        - Número de amenidades cercanas
        - Conectividad vial
        - Posición geográfica
        """)
        
        if st.button("Crear Dataset y Entrenar", type="primary"):
            progress = st.progress(0, "Iniciando...")
            
            # Preparar datos para cache (convertir a WKT)
            progress.progress(10, "Preparando datos...")
            
            boundary_utm = data['boundary'].to_crs(CRS_UTM)
            boundary_wkt = boundary_utm.geometry.unary_union.wkt
            
            buildings_wkt = []
            if 'buildings' in data:
                buildings_utm = data['buildings'].to_crs(CRS_UTM)
                buildings_wkt = [g.wkt for g in buildings_utm.geometry]
            
            amenities_wkt = []
            if 'amenities' in data:
                amenities_utm = data['amenities'].to_crs(CRS_UTM)
                amenities_wkt = [g.wkt for g in amenities_utm.geometry]
            
            streets_wkt = []
            if 'streets' in data:
                streets_utm = data['streets'].to_crs(CRS_UTM)
                streets_wkt = [g.wkt for g in streets_utm.geometry]
            
            # Crear grilla con features
            progress.progress(30, "Creando features espaciales...")
            
            grid = create_grid_with_features_optimized(
                boundary_wkt, buildings_wkt, amenities_wkt, streets_wkt, cell_size
            )
            
            st.session_state['grid'] = grid
            progress.progress(50, "Dataset creado...")
            
            st.success(f"Dataset creado: {len(grid)} celdas")
            
            # Preparar datos para ML
            feature_cols = ['n_amenities', 'dist_to_center', 'street_length', 'x_norm', 'y_norm', 'n_streets']
            available_cols = [c for c in feature_cols if c in grid.columns]
            
            X = grid[available_cols].fillna(0)
            
            if task_type == "Regresión":
                y = grid['n_buildings']
            else:
                # Clasificación: Alta densidad (>5 edificios)
                y = (grid['n_buildings'] > 5).astype(int)
            
            progress.progress(60, "Dividiendo datos...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            progress.progress(70, "Entrenando modelo...")
            
            if task_type == "Regresión":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                progress.progress(100, "¡Completado!")
                
                st.success("Modelo entrenado!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, y_pred)
                
                progress.progress(100, "¡Completado!")
                
                st.success("Modelo entrenado!")
                st.metric("Accuracy", f"{acc:.4f}")
            
            # Guardar en session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_cols'] = available_cols
            st.session_state['task_type'] = task_type
            
            # Feature importance
            st.markdown("---")
            st.markdown("**Importancia de Features**")
            
            importance = pd.DataFrame({
                'Feature': available_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            importance.plot(kind='barh', x='Feature', y='Importance', ax=ax, 
                          color='steelblue', legend=False)
            ax.set_xlabel('Importancia')
            ax.set_title('Importancia de Features - Random Forest')
            st.pyplot(fig)
            plt.close(fig)
        
        elif 'model' in st.session_state:
            st.info("Modelo ya entrenado. Vaya a la pestaña 'Predicciones' para ver resultados.")
    
    with tab2:
        st.subheader("Mapa de Predicciones")
        
        if 'model' not in st.session_state:
            st.warning("Primero entrene un modelo en la pestaña anterior")
        else:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            grid = st.session_state['grid']
            feature_cols = st.session_state['feature_cols']
            task_type = st.session_state['task_type']
            
            # Predecir para toda la grilla
            X_all = grid[feature_cols].fillna(0)
            X_all_scaled = scaler.transform(X_all)
            
            predictions = model.predict(X_all_scaled)
            grid['prediction'] = predictions
            
            if task_type == "Regresión":
                grid['error'] = grid['n_buildings'] - grid['prediction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Valores Reales**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    grid.plot(column='n_buildings', ax=ax, cmap='YlOrRd', legend=True,
                             legend_kwds={'label': 'N° Edificios'})
                    ax.set_title('Valores Reales', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.markdown("**Predicciones del Modelo**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    grid.plot(column='prediction', ax=ax, cmap='YlOrRd', legend=True,
                             legend_kwds={'label': 'N° Predicho'})
                    ax.set_title('Predicciones', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                    plt.close(fig)
            
            else:
                grid['class_label'] = grid['prediction'].map({0: 'Baja Densidad', 1: 'Alta Densidad'})
                
                fig, ax = plt.subplots(figsize=(10, 8))
                data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                       edgecolor='black', linewidth=2)
                
                colors = {'Baja Densidad': '#3288bd', 'Alta Densidad': '#d53e4f'}
                for label, color in colors.items():
                    subset = grid[grid['class_label'] == label]
                    if len(subset) > 0:
                        subset.plot(ax=ax, color=color, edgecolor='gray', 
                                   linewidth=0.5, label=label, alpha=0.7)
                
                ax.legend(loc='lower right', title='Predicción')
                ax.set_title('Clasificación de Densidad Predicha', fontweight='bold')
                ax.set_axis_off()
                st.pyplot(fig)
                plt.close(fig)
    
    with tab3:
        st.subheader("Evaluación del Modelo")
        
        if 'model' not in st.session_state:
            st.warning("Primero entrene un modelo")
        else:
            grid = st.session_state['grid']
            task_type = st.session_state['task_type']
            
            if 'prediction' not in grid.columns:
                st.warning("Primero vea las predicciones en la pestaña anterior")
            elif task_type == "Regresión":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Dispersión: Real vs Predicho**")
                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.scatter(grid['n_buildings'], grid['prediction'], alpha=0.5, s=20)
                    max_val = max(grid['n_buildings'].max(), grid['prediction'].max())
                    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Línea perfecta')
                    ax.set_xlabel('Valor Real')
                    ax.set_ylabel('Predicción')
                    ax.legend()
                    ax.set_title('Comparación Real vs Predicho')
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    st.markdown("**Distribución de Errores**")
                    fig, ax = plt.subplots(figsize=(7, 7))
                    grid['error'] = grid['n_buildings'] - grid['prediction']
                    grid['error'].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    ax.set_xlabel('Error (Real - Predicho)')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Histograma de Errores')
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Métricas
                st.markdown("---")
                st.markdown("**Métricas de Evaluación**")
                
                r2 = r2_score(grid['n_buildings'], grid['prediction'])
                rmse = np.sqrt(mean_squared_error(grid['n_buildings'], grid['prediction']))
                mae = np.mean(np.abs(grid['n_buildings'] - grid['prediction']))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("MAE", f"{mae:.2f}")
            
            else:
                y_true = (grid['n_buildings'] > 5).astype(int)
                y_pred = grid['prediction']
                
                acc = accuracy_score(y_true, y_pred)
                
                st.metric("Accuracy", f"{acc:.4f}")
                
                # Matriz de confusión simple
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(cm, cmap='Blues')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Baja', 'Alta'])
                ax.set_yticklabels(['Baja', 'Alta'])
                ax.set_xlabel('Predicho')
                ax.set_ylabel('Real')
                ax.set_title('Matriz de Confusión')
                
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm[i, j], ha='center', va='center', 
                               color='white' if cm[i, j] > cm.max()/2 else 'black')
                
                plt.colorbar(im)
                st.pyplot(fig)
                plt.close(fig)

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
