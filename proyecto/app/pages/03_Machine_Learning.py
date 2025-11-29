"""
Página de Machine Learning Espacial - Predicciones
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from shapely.geometry import box
import os
import joblib

# ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# Configuración
st.set_page_config(page_title="ML Espacial", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Espacial")
st.markdown("---")

# Rutas
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data', 'raw', 'isla_de_pascua')
MODELS_PATH = os.path.join(os.path.dirname(BASE_PATH), 'outputs', 'models')
CRS_UTM = 'EPSG:32719'

@st.cache_data
def load_data():
    """Cargar datos"""
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
            except:
                pass
    return data

def create_grid_with_features(data, cell_size=200):
    """Crear grilla con todas las features"""
    boundary = data['boundary'].to_crs(CRS_UTM)
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
    
    # Feature 1: Número de edificios
    buildings = data['buildings'].to_crs(CRS_UTM)
    buildings['centroid_geom'] = buildings.geometry.centroid
    
    grid['n_buildings'] = 0
    grid['total_building_area'] = 0.0
    
    for idx, cell in grid.iterrows():
        buildings_in = buildings[buildings['centroid_geom'].within(cell.geometry)]
        grid.loc[idx, 'n_buildings'] = len(buildings_in)
        if len(buildings_in) > 0:
            grid.loc[idx, 'total_building_area'] = buildings_in.geometry.area.sum()
    
    # Feature 2: Amenidades
    if 'amenities' in data:
        amenities = data['amenities'].to_crs(CRS_UTM)
        grid['n_amenities'] = 0
        for idx, cell in grid.iterrows():
            amen_in = amenities[amenities.geometry.within(cell.geometry) | 
                               amenities.geometry.centroid.within(cell.geometry)]
            grid.loc[idx, 'n_amenities'] = len(amen_in)
    
    # Feature 3: Distancia al centro
    island_center = boundary.geometry.centroid.values[0]
    grid['centroid'] = grid.geometry.centroid
    grid['dist_to_center'] = grid['centroid'].distance(island_center)
    
    # Feature 4: Longitud de calles
    if 'streets' in data:
        streets = data['streets'].to_crs(CRS_UTM)
        grid['street_length'] = 0.0
        for idx, cell in grid.iterrows():
            streets_in = streets[streets.geometry.intersects(cell.geometry)]
            if len(streets_in) > 0:
                clipped = streets_in.geometry.intersection(cell.geometry)
                grid.loc[idx, 'street_length'] = clipped.length.sum()
    
    # Feature 5: Coordenadas normalizadas
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
    st.sidebar.header("🤖 Opciones de ML")
    
    cell_size = st.sidebar.slider("Tamaño de celda (m)", 100, 500, 200, 50)
    
    task_type = st.sidebar.selectbox(
        "Tipo de tarea:",
        ["Regresión", "Clasificación"]
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Entrenar Modelo", "🗺️ Predicciones", "📈 Evaluación"])
    
    with tab1:
        st.subheader("Entrenar Modelo de Machine Learning")
        
        if st.button("🔄 Crear Dataset y Entrenar", type="primary"):
            with st.spinner("Creando features espaciales..."):
                grid = create_grid_with_features(data, cell_size)
                st.session_state['grid'] = grid
            
            st.success(f"✅ Dataset creado: {len(grid)} celdas")
            
            # Preparar datos
            feature_cols = ['n_amenities', 'dist_to_center', 'street_length', 'x_norm', 'y_norm']
            available_cols = [c for c in feature_cols if c in grid.columns]
            
            X = grid[available_cols].fillna(0)
            
            if task_type == "Regresión":
                y = grid['n_buildings']
            else:
                # Clasificación: Alta densidad (>5 edificios)
                y = (grid['n_buildings'] > 5).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            with st.spinner("Entrenando modelo..."):
                if task_type == "Regresión":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    st.success(f"✅ Modelo entrenado!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.2f}")
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Modelo entrenado!")
                    st.metric("Accuracy", f"{acc:.4f}")
            
            # Guardar en session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_cols'] = available_cols
            st.session_state['task_type'] = task_type
            
            # Feature importance
            st.markdown("---")
            st.markdown("**📊 Importancia de Features**")
            
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
        
        elif 'model' in st.session_state:
            st.info("✅ Modelo ya entrenado. Vaya a la pestaña 'Predicciones' para ver resultados.")
    
    with tab2:
        st.subheader("Mapa de Predicciones")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Primero entrene un modelo en la pestaña anterior")
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
            
            if task_type == "Regresión":
                grid['prediction'] = predictions
                grid['error'] = grid['n_buildings'] - grid['prediction']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Valores Reales**")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    grid.plot(column='n_buildings', ax=ax, cmap='YlOrRd', legend=True,
                             legend_kwds={'label': 'N° Edificios'})
                    ax.set_title('Valores Reales', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**Predicciones del Modelo**")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                           edgecolor='black', linewidth=2)
                    grid.plot(column='prediction', ax=ax, cmap='YlOrRd', legend=True,
                             legend_kwds={'label': 'N° Predicho'})
                    ax.set_title('Predicciones', fontweight='bold')
                    ax.set_axis_off()
                    st.pyplot(fig)
                
                # Mapa de errores
                st.markdown("---")
                st.markdown("**Mapa de Errores (Real - Predicho)**")
                fig, ax = plt.subplots(figsize=(12, 10))
                data['boundary'].to_crs(CRS_UTM).plot(ax=ax, facecolor='none', 
                                                       edgecolor='black', linewidth=2)
                grid.plot(column='error', ax=ax, cmap='RdBu_r', legend=True,
                         legend_kwds={'label': 'Error'})
                ax.set_title('Mapa de Errores', fontweight='bold')
                ax.set_axis_off()
                st.pyplot(fig)
            
            else:
                grid['prediction'] = predictions
                grid['class_label'] = grid['prediction'].map({0: 'Baja Densidad', 1: 'Alta Densidad'})
                
                fig, ax = plt.subplots(figsize=(12, 10))
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
    
    with tab3:
        st.subheader("Evaluación del Modelo")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Primero entrene un modelo")
        else:
            grid = st.session_state['grid']
            task_type = st.session_state['task_type']
            
            if task_type == "Regresión" and 'prediction' in grid.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Dispersión: Real vs Predicho**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(grid['n_buildings'], grid['prediction'], alpha=0.5, s=20)
                    max_val = max(grid['n_buildings'].max(), grid['prediction'].max())
                    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Línea perfecta')
                    ax.set_xlabel('Valor Real')
                    ax.set_ylabel('Predicción')
                    ax.legend()
                    ax.set_title('Comparación Real vs Predicho')
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**Distribución de Errores**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    grid['error'].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    ax.set_xlabel('Error (Real - Predicho)')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Histograma de Errores')
                    st.pyplot(fig)
                
                # Métricas
                st.markdown("---")
                st.markdown("**📊 Métricas de Evaluación**")
                
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
            
            elif task_type == "Clasificación" and 'prediction' in grid.columns:
                y_true = (grid['n_buildings'] > 5).astype(int)
                y_pred = grid['prediction']
                
                acc = accuracy_score(y_true, y_pred)
                
                st.metric("Accuracy", f"{acc:.4f}")
                
                # Matriz de confusión simple
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
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

except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
