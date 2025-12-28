"""
Página de Descargas - Sistema de Análisis Territorial
Permite descargar datos, mapas y reportes del proyecto.
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import os
import json
from pathlib import Path
from io import BytesIO
from sqlalchemy import create_engine

st.set_page_config(page_title="Descargas", layout="wide")

st.title("📥 Centro de Descargas")
st.markdown("""
Descargue los datos, mapas y resultados del análisis territorial de Isla de Pascua.
""")

# Configuración BD
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


# =============================================================================
# SECCIÓN 1: DATOS ESPACIALES
# =============================================================================

st.header("1. Datos Espaciales")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Datos desde PostGIS")
    
    try:
        engine = get_engine()
        
        # Lista de tablas disponibles
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'geoanalisis'
            ORDER BY table_name
        """
        tables_df = pd.read_sql(tables_query, engine)
        
        if len(tables_df) > 0:
            selected_table = st.selectbox(
                "Seleccione tabla:",
                tables_df['table_name'].tolist()
            )
            
            format_option = st.radio(
                "Formato de descarga:",
                ["GeoJSON", "CSV (sin geometría)"]
            )
            
            if st.button("Preparar descarga", key="postgis_download"):
                with st.spinner("Cargando datos..."):
                    gdf = gpd.read_postgis(
                        f'SELECT * FROM geoanalisis."{selected_table}"',
                        engine, geom_col='geometry'
                    )
                    
                    if format_option == "GeoJSON":
                        geojson_str = gdf.to_json()
                        st.download_button(
                            label=f"⬇️ Descargar {selected_table}.geojson",
                            data=geojson_str,
                            file_name=f"{selected_table}.geojson",
                            mime="application/json"
                        )
                    else:
                        # CSV sin geometría
                        df = pd.DataFrame(gdf.drop(columns=['geometry']))
                        df['centroid_x'] = gdf.geometry.centroid.x
                        df['centroid_y'] = gdf.geometry.centroid.y
                        csv_str = df.to_csv(index=False)
                        st.download_button(
                            label=f"⬇️ Descargar {selected_table}.csv",
                            data=csv_str,
                            file_name=f"{selected_table}.csv",
                            mime="text/csv"
                        )
                    
                    st.success(f"✓ {len(gdf)} registros listos para descarga")
        else:
            st.warning("No hay tablas disponibles en PostGIS")
            
    except Exception as e:
        st.error(f"Error conectando a PostGIS: {e}")
        st.info("Asegúrese de que el contenedor postgis esté corriendo")

with col2:
    st.subheader("📁 Datos Locales")
    
    data_path = Path("/data/raw/isla_de_pascua")
    if not data_path.exists():
        data_path = Path("../data/raw/isla_de_pascua")
    
    if data_path.exists():
        geojson_files = list(data_path.glob("*.geojson"))
        
        if geojson_files:
            selected_file = st.selectbox(
                "Seleccione archivo:",
                [f.name for f in geojson_files]
            )
            
            filepath = data_path / selected_file
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            st.download_button(
                label=f"⬇️ Descargar {selected_file}",
                data=content,
                file_name=selected_file,
                mime="application/json"
            )
            
            # Mostrar info del archivo
            gdf = gpd.read_file(filepath)
            st.caption(f"📊 {len(gdf)} registros | {filepath.stat().st_size/1024:.1f} KB")
        else:
            st.warning("No hay archivos GeoJSON en data/raw")
    else:
        st.warning("Directorio de datos no encontrado")


# =============================================================================
# SECCIÓN 2: MAPAS Y FIGURAS
# =============================================================================

st.markdown("---")
st.header("2. Mapas y Figuras")

outputs_path = Path("/outputs")
if not outputs_path.exists():
    outputs_path = Path("../outputs")

if outputs_path.exists():
    # Buscar imágenes
    image_files = list(outputs_path.glob("*.png")) + list(outputs_path.glob("*.jpg"))
    
    if image_files:
        cols = st.columns(3)
        
        for idx, img_file in enumerate(image_files[:9]):  # Máximo 9 imágenes
            col_idx = idx % 3
            
            with cols[col_idx]:
                st.image(str(img_file), caption=img_file.name, use_container_width=True)
                
                with open(img_file, 'rb') as f:
                    img_bytes = f.read()
                
                st.download_button(
                    label=f"⬇️ {img_file.name}",
                    data=img_bytes,
                    file_name=img_file.name,
                    mime="image/png",
                    key=f"img_{idx}"
                )
    else:
        st.info("No hay imágenes en el directorio outputs/")
else:
    st.warning("Directorio outputs/ no encontrado")


# =============================================================================
# SECCIÓN 3: REPORTES Y ESTADÍSTICAS
# =============================================================================

st.markdown("---")
st.header("3. Reportes y Estadísticas")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Estadísticas Generales")
    
    if st.button("Generar Reporte de Estadísticas"):
        try:
            # Cargar datos
            buildings = gpd.read_file(data_path / "isla_de_pascua_buildings.geojson")
            buildings_utm = buildings.to_crs("EPSG:32712")
            buildings_utm['area_m2'] = buildings_utm.geometry.area
            
            streets = gpd.read_file(data_path / "isla_de_pascua_streets.geojson")
            
            # Crear reporte
            stats = {
                "Metrica": [
                    "Total Edificaciones",
                    "Área Total Construida (ha)",
                    "Área Promedio Edificio (m²)",
                    "Total Calles",
                    "Red Vial (km)"
                ],
                "Valor": [
                    len(buildings_utm),
                    round(buildings_utm['area_m2'].sum() / 10000, 2),
                    round(buildings_utm['area_m2'].mean(), 2),
                    len(streets),
                    "N/A"  # Calcular si es necesario
                ]
            }
            
            df_stats = pd.DataFrame(stats)
            st.dataframe(df_stats)
            
            # Descargar
            csv_stats = df_stats.to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar estadisticas.csv",
                data=csv_stats,
                file_name="estadisticas_isla_pascua.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error generando reporte: {e}")

with col2:
    st.subheader("📄 Archivos Adicionales")
    
    # Buscar CSVs y otros archivos
    csv_files = list(outputs_path.glob("*.csv")) if outputs_path.exists() else []
    json_files = list(outputs_path.glob("*.json")) if outputs_path.exists() else []
    
    all_files = csv_files + json_files
    
    if all_files:
        for f in all_files[:5]:
            with open(f, 'r') as file:
                content = file.read()
            
            st.download_button(
                label=f"⬇️ {f.name}",
                data=content,
                file_name=f.name,
                mime="text/csv" if f.suffix == ".csv" else "application/json",
                key=f"file_{f.name}"
            )
    else:
        st.info("No hay archivos CSV/JSON adicionales")


# =============================================================================
# SECCIÓN 4: API REST
# =============================================================================

st.markdown("---")
st.header("4. Acceso API REST")

st.markdown("""
El proyecto incluye una API REST para acceso programático a los datos.

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/tables` | Lista todas las tablas |
| GET | `/api/data/{tabla}` | Obtiene datos de una tabla |
| GET | `/api/stats/{tabla}` | Estadísticas de una tabla |
| POST | `/api/predict` | Predicción de densidad |
| GET | `/api/health` | Estado de la API |

**Ejemplo de uso con Python:**
```python
import requests

# Listar tablas
response = requests.get("http://localhost:8000/api/tables")
tables = response.json()

# Obtener datos
response = requests.get("http://localhost:8000/api/data/isla_de_pascua_buildings?limit=100")
data = response.json()

# Predicción
response = requests.post("http://localhost:8000/api/predict", 
                         json={"x": -109.43, "y": -27.15})
prediction = response.json()
```

**Documentación interactiva:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
""")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Centro de Descargas - Sistema de Análisis Territorial | Laboratorio Integrador 2025")
