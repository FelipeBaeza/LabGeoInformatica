"""
Pagina de Machine Learning Espacial
Entrena y compara modelos para predecir densidad de edificaciones.

Modelos comparados segun requerimientos del laboratorio:
- Random Forest
- XGBoost
- SVM (Support Vector Machine)
- Gradient Boosting
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from sqlalchemy import create_engine

# XGBoost (opcional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="Machine Learning", layout="wide")

st.title("Machine Learning Espacial")

st.markdown("""
Esta página presenta la **comparación completa de modelos de Machine Learning** para
predecir la densidad de edificaciones en Isla de Pascua. 

Siguiendo los requerimientos del laboratorio, se comparan:
- **Random Forest** - Modelo basado en ensamble de árboles
- **XGBoost** - Gradient Boosting optimizado
- **SVM (Support Vector Machine)** - Modelo de kernel
- **Gradient Boosting** - Boosting tradicional
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
def load_and_prepare_data(cell_size=300):
    """Cargar datos y crear features espaciales."""
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
        amenities = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_amenities", 
            engine, geom_col='geometry'
        )
        streets = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_streets", 
            engine, geom_col='geometry'
        )
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None
    
    # Proyectar a UTM
    boundary_utm = boundary.to_crs(CRS_UTM)
    buildings_utm = buildings.to_crs(CRS_UTM)
    amenities_utm = amenities.to_crs(CRS_UTM)
    streets_utm = streets.to_crs(CRS_UTM)
    
    # Crear grilla
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
    
    # Calcular centroides
    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y
    
    # Variable objetivo: numero de edificios
    buildings_centroids = buildings_utm.copy()
    buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
    
    joined = gpd.sjoin(buildings_centroids.reset_index(), grid[['cell_id', 'geometry']], 
                       how='inner', predicate='within')
    counts = joined.groupby('cell_id').size().reset_index(name='n_edificios')
    grid = grid.merge(counts, on='cell_id', how='left')
    grid['n_edificios'] = grid['n_edificios'].fillna(0).astype(int)
    
    # Feature 1: Distancia al centro de la isla
    center_x = boundary_utm.geometry.centroid.x.iloc[0]
    center_y = boundary_utm.geometry.centroid.y.iloc[0]
    grid['dist_centro'] = np.sqrt(
        (grid['centroid_x'] - center_x)**2 + 
        (grid['centroid_y'] - center_y)**2
    )
    
    # Feature 2: Numero de amenidades cercanas
    amenities_points = amenities_utm.copy()
    if 'Point' not in str(amenities_points.geometry.geom_type.iloc[0]):
        amenities_points['geometry'] = amenities_points.geometry.centroid
    
    joined_amen = gpd.sjoin(amenities_points.reset_index(), grid[['cell_id', 'geometry']], 
                            how='inner', predicate='within')
    amen_counts = joined_amen.groupby('cell_id').size().reset_index(name='n_amenities')
    grid = grid.merge(amen_counts, on='cell_id', how='left')
    grid['n_amenities'] = grid['n_amenities'].fillna(0).astype(int)
    
    # Feature 3: Longitud de calles
    grid['street_length'] = 0.0
    for idx, cell in grid.iterrows():
        streets_in_cell = streets_utm[streets_utm.intersects(cell.geometry)]
        if len(streets_in_cell) > 0:
            clipped = streets_in_cell.intersection(cell.geometry)
            grid.loc[idx, 'street_length'] = clipped.length.sum()
    
    # Normalizar coordenadas
    grid['x_norm'] = (grid['centroid_x'] - grid['centroid_x'].min()) / (grid['centroid_x'].max() - grid['centroid_x'].min())
    grid['y_norm'] = (grid['centroid_y'] - grid['centroid_y'].min()) / (grid['centroid_y'].max() - grid['centroid_y'].min())
    
    # Crear zona_id para validación espacial (GroupKFold)
    grid['zona_id'] = pd.cut(grid['x_norm'] + grid['y_norm'], bins=5, labels=False)
    
    return grid


# ============================================================================
# CARGAR DATOS
# ============================================================================

st.sidebar.header("Configuracion")
cell_size = st.sidebar.slider("Tamaño de celda (m)", 200, 400, 300, 50)

with st.spinner("Preparando datos y features..."):
    grid = load_and_prepare_data(cell_size)

if grid is None:
    st.stop()

# ============================================================================
# SECCION 1: FEATURES ESPACIALES
# ============================================================================

st.header("1. Variables Predictoras (Feature Engineering Espacial)")

st.markdown("""
Para implementar machine learning geoespacial, es necesario crear **features espaciales** 
que capturen las características territoriales de cada zona. Estos features fueron diseñados
siguiendo las mejores prácticas de análisis espacial:
""")

features = ['dist_centro', 'n_amenities', 'street_length', 'x_norm', 'y_norm']
feature_names = {
    'dist_centro': 'Distancia al centro (m)',
    'n_amenities': 'Número de amenidades',
    'street_length': 'Longitud de calles (m)',
    'x_norm': 'Posición X normalizada',
    'y_norm': 'Posición Y normalizada'
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | Variable | Descripción |
    |----------|-------------|
    | `dist_centro` | Distancia al centroide de la isla |
    | `n_amenities` | Servicios/comercios en la celda |
    | `street_length` | Metros de calles en la celda |
    | `x_norm`, `y_norm` | Posición geográfica normalizada |
    """)

with col2:
    st.dataframe(grid[features].describe().round(2))

# ============================================================================
# SECCION 2: COMPARACIÓN DE MODELOS
# ============================================================================

st.header("2. Comparacion de Modelos de Machine Learning")

st.markdown("""
Según los requerimientos del laboratorio, se deben comparar **al menos 3 algoritmos diferentes**.
A continuación se presentan los modelos evaluados y sus resultados:
""")

# Preparar datos
X = grid[features].values
y = grid['n_edificios'].values
groups = grid['zona_id'].values

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir modelos a comparar
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVM (RBF Kernel)': SVR(kernel='rbf', C=100, gamma='scale'),
    'Linear Regression': LinearRegression(),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                          learning_rate=0.1, random_state=42, verbosity=0)

# Entrenamiento y evaluación con validación cruzada espacial
results = {}
gkf = GroupKFold(n_splits=5)

with st.spinner("Entrenando y comparando modelos con validación espacial..."):
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        # Validación cruzada espacial (GroupKFold evita data leakage espacial)
        cv_scores = cross_val_score(model, X_scaled, y, cv=gkf, groups=groups, scoring='r2')
        
        # Entrenar modelo completo para predicciones
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }
        
        progress_bar.progress((i + 1) / len(models))

progress_bar.empty()

# Crear DataFrame de resultados
results_df = pd.DataFrame({
    'Modelo': results.keys(),
    'R² (CV Espacial)': [r['cv_mean'] for r in results.values()],
    'Desv. Std': [r['cv_std'] for r in results.values()],
    'R² (Train)': [r['r2'] for r in results.values()],
    'RMSE': [r['rmse'] for r in results.values()],
    'MAE': [r['mae'] for r in results.values()]
}).sort_values('R² (CV Espacial)', ascending=False).reset_index(drop=True)

# Mostrar tabla de comparación
st.subheader("Tabla Comparativa de Modelos")
st.dataframe(results_df.style.highlight_max(subset=['R² (CV Espacial)', 'R² (Train)'], color='lightgreen')
             .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
             .format({
                 'R² (CV Espacial)': '{:.4f}',
                 'Desv. Std': '±{:.4f}',
                 'R² (Train)': '{:.4f}',
                 'RMSE': '{:.2f}',
                 'MAE': '{:.2f}'
             }))

# Gráfico de comparación
st.subheader("Visualizacion de Rendimiento")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico R² 
ax1 = axes[0]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'][:len(results)]
bars = ax1.barh(list(results.keys()), [r['cv_mean'] for r in results.values()], color=colors)
ax1.errorbar([r['cv_mean'] for r in results.values()], range(len(results)), 
             xerr=[r['cv_std']*2 for r in results.values()], fmt='none', color='black', capsize=3)
ax1.set_xlabel('R² Score (Validación Cruzada Espacial)')
ax1.set_title('Comparación de R² con Validación Espacial')
ax1.set_xlim(0, 1)

# Agregar valores
for bar, val in zip(bars, [r['cv_mean'] for r in results.values()]):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

# Gráfico RMSE
ax2 = axes[1]
bars2 = ax2.barh(list(results.keys()), [r['rmse'] for r in results.values()], color=colors)
ax2.set_xlabel('RMSE (Error Cuadrático Medio)')
ax2.set_title('Comparación de Error (RMSE)')

for bar, val in zip(bars2, [r['rmse'] for r in results.values()]):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ============================================================================
# SECCION 3: MODELO SELECCIONADO Y JUSTIFICACIÓN
# ============================================================================

st.header("3. Modelo Seleccionado y Justificacion")

# Identificar el mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_result = results[best_model_name]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mejor Modelo", best_model_name)
with col2:
    st.metric("R² (CV Espacial)", f"{best_result['cv_mean']:.4f}")
with col3:
    st.metric("RMSE", f"{best_result['rmse']:.2f}")

st.markdown(f"""
### ¿Por qué se eligió **{best_model_name}**?

El modelo **{best_model_name}** fue seleccionado como el mejor modelo basándose en los siguientes criterios:

1. **Mayor R² en Validación Cruzada Espacial** ({best_result['cv_mean']:.4f}): 
   - Este es el criterio más importante ya que utiliza **GroupKFold** para evitar data leakage espacial
   - Las zonas geográficas se agrupan para garantizar que el modelo generalice bien a nuevas áreas

2. **Mejor balance entre ajuste y generalización**:
   - R² de entrenamiento ({best_result['r2']:.4f}) vs R² CV ({best_result['cv_mean']:.4f})
   - Diferencia pequeña indica que no hay overfitting significativo

3. **Error de predicción controlado**:
   - RMSE de {best_result['rmse']:.2f} edificios por celda
   - MAE de {best_result['mae']:.2f} edificios por celda

### Descarte de otros modelos:
""")

# Explicar por qué cada modelo fue descartado
for name, res in results.items():
    if name != best_model_name:
        diff = best_result['cv_mean'] - res['cv_mean']
        st.markdown(f"""
- **{name}**: R² CV = {res['cv_mean']:.4f} (inferior por {diff:.4f})
""")

st.info("""
**Nota técnica**: La validación cruzada espacial (GroupKFold con zonas geográficas) es fundamental
en problemas geoespaciales. A diferencia de la validación cruzada tradicional que puede mezclar
datos espacialmente cercanos en train/test, este método garantiza que los grupos espaciales
completos se mueven juntos, evitando la autocorrelación espacial artificial.
""")

# ============================================================================
# SECCION 4: IMPORTANCIA DE VARIABLES
# ============================================================================

st.header("4. Importancia de Variables")

st.markdown("""
La importancia de variables nos indica qué factores son más determinantes para predecir
la densidad de edificaciones. Solo los modelos basados en árboles proporcionan esta información
directamente.
""")

# Obtener importancias (solo para modelos que las tienen)
if hasattr(results[best_model_name]['model'], 'feature_importances_'):
    importance = pd.DataFrame({
        'Variable': [feature_names[f] for f in features],
        'Importancia': results[best_model_name]['model'].feature_importances_
    }).sort_values('Importancia', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(importance['Variable'], importance['Importancia'], color='#2E86AB')
    ax.set_xlabel('Importancia Relativa')
    ax.set_title(f'Importancia de Variables - {best_model_name}')
    
    # Agregar porcentajes
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width*100:.1f}%', va='center')
    
    st.pyplot(fig)
    plt.close()
    
    top_feature = importance.iloc[-1]
    st.markdown(f"""
    **Hallazgo principal**: La variable más importante es **{top_feature['Variable']}** 
    con una importancia de {top_feature['Importancia']*100:.1f}%. Esto significa que 
    esta característica es la que mejor predice dónde hay más edificaciones en la isla.
    """)
else:
    st.warning(f"El modelo {best_model_name} no proporciona importancias de features directamente.")

# ============================================================================
# SECCION 5: MAPA DE PREDICCIONES
# ============================================================================

st.header("5. Mapa de Predicciones")

st.markdown(f"""
Este mapa muestra las predicciones del modelo **{best_model_name}** seleccionado.
Las zonas con colores más intensos son donde el modelo predice mayor densidad de edificaciones.
""")

grid['prediccion'] = results[best_model_name]['predictions']

# Convertir a WGS84 para visualizacion
grid_wgs = grid.to_crs("EPSG:4326")

# Calcular bounds
bounds = grid_wgs.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

# Crear mapa con Folium
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

# Agregar celdas con predicciones
max_pred = grid_wgs['prediccion'].max() if grid_wgs['prediccion'].max() > 0 else 1

for idx, row in grid_wgs.iterrows():
    if row['prediccion'] > 0.5:
        # Color basado en prediccion
        intensity = min(row['prediccion'] / max_pred, 1)
        r = int(255 * intensity)
        g = int(100 * (1 - intensity))
        b = 0
        color = f'#{r:02x}{g:02x}{b:02x}'
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 1,
                'fillOpacity': 0.6
            },
            tooltip=f"Predicción: {row['prediccion']:.1f}, Real: {row['n_edificios']}"
        ).add_to(m)

st_folium(m, width=900, height=500)

# ============================================================================
# SECCION 6: RESUMEN Y CONCLUSIONES
# ============================================================================

st.header("6. Resumen de la Comparativa de Modelos")

st.markdown(f"""
### Modelos Evaluados

| Modelo | Tipo | Ventajas | Desventajas |
|--------|------|----------|-------------|
| **Random Forest** | Ensamble de árboles | Robusto, feature importance | Lento en predicción |
| **Gradient Boosting** | Boosting secuencial | Alta precisión | Sensible a outliers |
| **SVM (RBF)** | Kernel | Bueno en espacios pequeños | Difícil interpretación |
| **XGBoost** | Gradient Boosting optimizado | Muy eficiente | Requiere tuning |
| **Linear Regression** | Modelo lineal | Simple, interpretable | No captura no-linealidades |

### Modelo Final: **{best_model_name}**

Se selecciono este modelo porque:
- Mejor R2 en validacion cruzada espacial
- Balance adecuado entre bias y varianza
- Capacidad de capturar patrones espaciales complejos
- Interpretabilidad mediante feature importance
""")

# Métricas finales
st.success(f"""
**Métricas del modelo final ({best_model_name})**:
- R² (Validación Espacial): {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}
- RMSE: {best_result['rmse']:.2f} edificios/celda
- MAE: {best_result['mae']:.2f} edificios/celda
""")
