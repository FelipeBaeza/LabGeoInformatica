"""
Pagina de Machine Learning Espacial
Entrena modelos para predecir densidad de edificaciones
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="Machine Learning", layout="wide")

st.title("Machine Learning Espacial")

st.markdown("""
El modelo aprende de los datos existentes para 
identificar que factores determinan dónde se construye más en la isla.
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
    
    return grid


# ============================================================================
# CARGAR DATOS
# ============================================================================

st.sidebar.header("Configuracion")
cell_size = st.sidebar.slider("Tamano de celda (m)", 200, 400, 300, 50)

with st.spinner("Preparando datos y features..."):
    grid = load_and_prepare_data(cell_size)

if grid is None:
    st.stop()

# ============================================================================
# SECCION 1: FEATURES ESPACIALES
# ============================================================================

st.header("1. Variables Predictoras (Features)")

st.markdown("""
Para que el modelo de inteligencia artificial pueda hacer predicciones, necesitamos 
definir **variables predictoras** que describan las caracteristicas de cada zona. 
Estas son las variables que calculamos para cada celda:
""")

features = ['dist_centro', 'n_amenities', 'street_length', 'x_norm', 'y_norm']
feature_names = {
    'dist_centro': 'Distancia al centro (m)',
    'n_amenities': 'Numero de amenidades',
    'street_length': 'Longitud de calles (m)',
    'x_norm': 'Posicion X normalizada',
    'y_norm': 'Posicion Y normalizada'
}

st.markdown("""
| Variable | Descripcion | Hipotesis |
|----------|-------------|-----------|
| Distancia al centro | Metros desde el centroide de la isla | Mas lejos = menos edificios |
| Amenidades | Servicios cercanos (comercios, etc.) | Mas servicios = mas edificios |
| Longitud calles | Metros de calles en la celda | Mas calles = mas edificios |
| Posicion X, Y | Ubicacion geografica | El oeste tiene mas edificios |
""")

# Mostrar estadisticas de features
st.dataframe(grid[features].describe().round(2))

# ============================================================================
# SECCION 2: ENTRENAR MODELO
# ============================================================================

st.header("2. Entrenamiento del Modelo")

st.markdown("""
Utilizamos el algoritmo **Random Forest**, que combina multiples arboles de decision 
para hacer predicciones robustas. Este metodo es muy efectivo para datos geograficos 
porque puede capturar relaciones no lineales entre las variables.
""")

# Preparar datos
X = grid[features].values
y = grid['n_edificios'].values

# Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Validacion cruzada
with st.spinner("Entrenando modelo..."):
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    model.fit(X_scaled, y)
    predictions = model.predict(X_scaled)

# Metricas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("R2 Promedio (CV)", f"{cv_scores.mean():.3f}")
with col2:
    st.metric("Desv. Estandar", f"{cv_scores.std():.3f}")
with col3:
    rmse = np.sqrt(np.mean((y - predictions)**2))
    st.metric("RMSE", f"{rmse:.2f}")

st.markdown(f"""
**Interpretacion de metricas:**
- **R2 = {cv_scores.mean():.2f}** significa que el modelo explica el {cv_scores.mean()*100:.0f}% de la 
  variabilidad en la densidad de edificaciones. Un valor cercano a 1 indica buen ajuste.
- **RMSE = {rmse:.2f}** es el error promedio en numero de edificios por celda.
""")

# ============================================================================
# SECCION 3: IMPORTANCIA DE VARIABLES
# ============================================================================

st.header("3. Importancia de Variables")

st.markdown("""
Este grafico muestra cuanto contribuye cada variable a las predicciones del modelo.
Las variables mas importantes son las que mejor explican donde hay mas edificaciones.
""")

importance = pd.DataFrame({
    'Variable': [feature_names[f] for f in features],
    'Importancia': model.feature_importances_
}).sort_values('Importancia', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importance['Variable'], importance['Importancia'], color='#2E86AB')
ax.set_xlabel('Importancia Relativa')
ax.set_title('Importancia de Variables en el Modelo')

st.pyplot(fig)
plt.close()

st.markdown(f"""
**Interpretacion:** La variable mas importante es **{importance.iloc[-1]['Variable']}** 
con una importancia de {importance.iloc[-1]['Importancia']*100:.1f}%. Esto significa que 
esta caracteristica es la que mejor predice donde hay mas edificaciones en la isla.
""")

# ============================================================================
# SECCION 4: MAPA DE PREDICCIONES
# ============================================================================

st.header("4. Mapa de Predicciones")

st.markdown("""
Este mapa compara los valores reales (izquierda) con las predicciones del modelo (derecha).
Si el modelo funciona bien, ambos mapas deberian verse similares.
""")

grid['prediccion'] = predictions

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
max_pred = grid_wgs['prediccion'].max()
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
            tooltip=f"Prediccion: {row['prediccion']:.1f}, Real: {row['n_edificios']}"
        ).add_to(m)

st_folium(m, width=900, height=500)

st.markdown("""
**Interpretacion:** El mapa muestra las predicciones del modelo. Las zonas con colores 
mas intensos son donde el modelo predice mayor densidad de edificaciones. La coincidencia 
con la ubicacion real de Hanga Roa confirma que el modelo ha aprendido correctamente 
los patrones espaciales de la isla.
""")
