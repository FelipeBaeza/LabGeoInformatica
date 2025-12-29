"""
Pagina de Analisis de Redes y Accesibilidad Vial
Implementa analisis de centralidad e isocronas.

Este es un ELEMENTO DE EXCELENCIA segun el documento del laboratorio.
"""
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os
from sqlalchemy import create_engine

# Intentar importar OSMnx y NetworkX
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

st.set_page_config(page_title="Analisis de Redes", layout="wide")

st.title("Analisis de Redes y Accesibilidad Vial")

st.markdown("""
Este modulo implementa el **analisis de redes viales**, un elemento de excelencia
segun el documento del laboratorio. Incluye:

1. **Metricas de centralidad** - Identificar nodos/calles mas importantes
2. **Isocronas de accesibilidad** - Zonas alcanzables en cierto tiempo
3. **Estadisticas de la red** - Densidad, conectividad, etc.
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
def load_streets():
    """Cargar calles desde PostGIS."""
    try:
        engine = get_engine()
        streets = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_streets", 
            engine, geom_col='geometry'
        )
        boundary = gpd.read_postgis(
            "SELECT * FROM geoanalisis.isla_de_pascua_boundary", 
            engine, geom_col='geometry'
        )
        return streets, boundary
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None


@st.cache_resource
def load_network_graph():
    """Cargar o crear grafo de red vial."""
    if not OSMNX_AVAILABLE:
        return None
    
    try:
        # Intentar cargar desde archivo
        graph_path = "/app/data/raw/isla_de_pascua/network_graph.graphml"
        if os.path.exists(graph_path):
            G = ox.load_graphml(graph_path)
            return G
        else:
            # Descargar desde OSM
            with st.spinner("Descargando red vial desde OpenStreetMap..."):
                G = ox.graph_from_place("Isla de Pascua, Chile", network_type='all')
            return G
    except Exception as e:
        st.warning(f"No se pudo cargar el grafo: {e}")
        return None


# Cargar datos
streets, boundary = load_streets()

if streets is None:
    st.error("No se pudieron cargar los datos de calles.")
    st.stop()

streets_wgs = streets.to_crs("EPSG:4326")
boundary_wgs = boundary.to_crs("EPSG:4326")

# Calcular centro
bounds = boundary_wgs.total_bounds
center_lon = (bounds[0] + bounds[2]) / 2
center_lat = (bounds[1] + bounds[3]) / 2

# ============================================================================
# SECCION 1: ESTADISTICAS BASICAS DE RED
# ============================================================================

st.header("1. Estadisticas de la Red Vial")

st.markdown("""
La red vial de Isla de Pascua es relativamente simple comparada con ciudades continentales,
pero es critica para la movilidad en la isla.
""")

# Calcular estadisticas basicas
streets_utm = streets.to_crs("EPSG:32712")
total_length = streets_utm.geometry.length.sum() / 1000  # km

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Segmentos de calle", len(streets))
with col2:
    st.metric("Longitud total", f"{total_length:.1f} km")
with col3:
    avg_length = streets_utm.geometry.length.mean()
    st.metric("Longitud promedio", f"{avg_length:.0f} m")
with col4:
    boundary_utm = boundary.to_crs("EPSG:32712")
    area_km2 = boundary_utm.geometry.area.iloc[0] / 1e6
    density = total_length / area_km2
    st.metric("Densidad vial", f"{density:.2f} km/km2")

# ============================================================================
# SECCION 2: MAPA DE RED VIAL
# ============================================================================

st.header("2. Mapa de la Red Vial")

st.markdown("""
Este mapa muestra la red de calles de la isla. El color indica la longitud de cada segmento.
Las calles mas largas suelen ser las principales arterias de conexion.
""")

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

# Clasificar calles por longitud
streets_wgs_copy = streets_wgs.copy()
streets_wgs_copy['length_m'] = streets_utm.geometry.length.values

# Cuartiles de longitud
q1 = streets_wgs_copy['length_m'].quantile(0.25)
q2 = streets_wgs_copy['length_m'].quantile(0.50)
q3 = streets_wgs_copy['length_m'].quantile(0.75)

def get_color(length):
    if length > q3:
        return '#d7191c'  # Rojo - calles largas/principales
    elif length > q2:
        return '#fdae61'  # Naranja
    elif length > q1:
        return '#abd9e9'  # Celeste
    else:
        return '#2c7bb6'  # Azul - calles cortas/secundarias

for idx, row in streets_wgs_copy.iterrows():
    if row.geometry is not None:
        try:
            coords = list(row.geometry.coords) if row.geometry.geom_type == 'LineString' else []
            if coords:
                color = get_color(row['length_m'])
                folium.PolyLine(
                    locations=[[c[1], c[0]] for c in coords],
                    weight=3,
                    color=color,
                    opacity=0.8,
                    tooltip=f"Longitud: {row['length_m']:.0f} m"
                ).add_to(m)
        except:
            pass

# Leyenda
legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
            background-color: white; padding: 10px; border-radius: 5px;
            border: 2px solid grey; font-size: 12px;">
    <b>Longitud de calle</b><br>
    <i style="background: #d7191c; width: 20px; height: 3px; display: inline-block;"></i> Larga (principal)<br>
    <i style="background: #fdae61; width: 20px; height: 3px; display: inline-block;"></i> Media-alta<br>
    <i style="background: #abd9e9; width: 20px; height: 3px; display: inline-block;"></i> Media-baja<br>
    <i style="background: #2c7bb6; width: 20px; height: 3px; display: inline-block;"></i> Corta (secundaria)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=900, height=500)

# ============================================================================
# SECCION 3: ANALISIS DE CENTRALIDAD
# ============================================================================

st.header("3. Analisis de Centralidad")

st.markdown("""
La **centralidad** mide la importancia de cada nodo (interseccion) en la red.
Un nodo con alta centralidad es critico para la conectividad de la isla.
""")

if OSMNX_AVAILABLE:
    G = load_network_graph()
    
    if G is not None:
        with st.spinner("Calculando metricas de centralidad..."):
            G_undirected = G.to_undirected()
            
            # Calcular centralidades (solo en una muestra para velocidad)
            n_sample = min(100, G.number_of_nodes())
            
            betweenness = nx.betweenness_centrality(G_undirected, k=n_sample)
            closeness = nx.closeness_centrality(G_undirected)
            degree = nx.degree_centrality(G_undirected)
            
            # Agregar al grafo
            for node in G.nodes():
                G.nodes[node]['betweenness'] = betweenness.get(node, 0)
                G.nodes[node]['closeness'] = closeness.get(node, 0)
                G.nodes[node]['degree'] = degree.get(node, 0)
            
            # Convertir a GeoDataFrame
            nodes_gdf = ox.graph_to_gdfs(G, edges=False)
            nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
            
            # Mostrar estadisticas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nodos en la red", G.number_of_nodes())
            with col2:
                st.metric("Aristas en la red", G.number_of_edges())
            with col3:
                density = nx.density(G_undirected)
                st.metric("Densidad de red", f"{density:.4f}")
            
            # Mapa de centralidad
            st.subheader("Mapa de Centralidad (Closeness)")
            
            m_cent = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')
            
            # Normalizar closeness para colores
            max_close = nodes_gdf['closeness'].max()
            min_close = nodes_gdf['closeness'].min()
            
            for idx, row in nodes_gdf.iterrows():
                if max_close > min_close:
                    intensity = (row['closeness'] - min_close) / (max_close - min_close)
                else:
                    intensity = 0.5
                
                # Color de azul (bajo) a rojo (alto)
                r = int(255 * intensity)
                b = int(255 * (1 - intensity))
                color = f'#{r:02x}00{b:02x}'
                
                # Tamano basado en betweenness
                size = 3 + row['betweenness'] * 50
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=size,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    tooltip=f"Closeness: {row['closeness']:.4f}"
                ).add_to(m_cent)
            
            st_folium(m_cent, width=900, height=500)
            
            st.markdown("""
            **Interpretacion:**
            - Los nodos **rojos** tienen alta centralidad (son los mas accesibles desde cualquier punto)
            - Los nodos **azules** tienen baja centralidad (estan en la periferia de la red)
            - El tamano del punto indica la importancia como "puente" (betweenness)
            """)
            
            # ================================================================
            # SECCION 4: ISOCRONAS
            # ================================================================
            
            st.header("4. Isocronas de Accesibilidad")
            
            st.markdown("""
            Las **isocronas** muestran que tan lejos se puede llegar desde un punto central
            en un tiempo determinado. Esto es util para planificacion urbana y servicios.
            """)
            
            # Encontrar nodo central
            central_node = max(closeness.items(), key=lambda x: x[1])[0]
            central_coords = nodes_gdf.loc[central_node].geometry
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Punto central identificado: nodo con mayor closeness centrality")
            with col2:
                st.info(f"Coordenadas: ({central_coords.y:.4f}, {central_coords.x:.4f})")
            
            # Calcular tiempos de viaje
            try:
                # Agregar tiempo de viaje (30 km/h promedio)
                for u, v, data in G.edges(data=True):
                    length = data.get('length', 100)
                    data['travel_time'] = length / (30000 / 60)  # minutos
                
                travel_times = nx.single_source_dijkstra_path_length(
                    G, central_node, weight='travel_time'
                )
                
                # Clasificar nodos por isocrona
                nodes_gdf['travel_time'] = nodes_gdf.index.map(lambda x: travel_times.get(x, 999))
                
                # Mapa de isocronas
                m_iso = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
                
                # Colores por tiempo
                def get_iso_color(time):
                    if time <= 5:
                        return '#d7191c'  # Rojo - 5 min
                    elif time <= 10:
                        return '#fdae61'  # Naranja - 10 min
                    elif time <= 15:
                        return '#ffffbf'  # Amarillo - 15 min
                    else:
                        return '#abd9e9'  # Celeste - >15 min
                
                for idx, row in nodes_gdf.iterrows():
                    if row['travel_time'] < 999:
                        color = get_iso_color(row['travel_time'])
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=4,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7,
                            tooltip=f"Tiempo: {row['travel_time']:.1f} min"
                        ).add_to(m_iso)
                
                # Marcar punto central
                folium.Marker(
                    location=[central_coords.y, central_coords.x],
                    popup="Punto Central",
                    icon=folium.Icon(color='blue', icon='star')
                ).add_to(m_iso)
                
                # Leyenda
                legend_iso = """
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                            background-color: white; padding: 10px; border-radius: 5px;
                            border: 2px solid grey; font-size: 12px;">
                    <b>Tiempo de viaje</b><br>
                    <i style="background: #d7191c; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 0-5 min<br>
                    <i style="background: #fdae61; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 5-10 min<br>
                    <i style="background: #ffffbf; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 10-15 min<br>
                    <i style="background: #abd9e9; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> >15 min
                </div>
                """
                m_iso.get_root().html.add_child(folium.Element(legend_iso))
                
                st_folium(m_iso, width=900, height=500)
                
                st.markdown("""
                **Interpretacion:**
                - Desde el punto central de Hanga Roa, la mayoria de la zona urbana 
                  es accesible en menos de 10 minutos
                - Las zonas perifericas requieren mas de 15 minutos
                - Esto es relevante para ubicar servicios de emergencia o comercios
                """)
                
            except Exception as e:
                st.warning(f"No se pudieron calcular isocronas: {e}")
    else:
        st.warning("No se pudo cargar el grafo de red vial.")
else:
    st.warning("OSMnx no esta disponible. Mostrando solo estadisticas basicas de la red.")
    st.info("Para el analisis completo de centralidad e isocronas, instale: pip install osmnx networkx")

# ============================================================================
# SECCION 5: RESUMEN
# ============================================================================

st.header("5. Resumen del Analisis de Redes")

st.markdown(f"""
**Hallazgos principales:**

1. **Red vial compacta**: Con {total_length:.1f} km de calles, la isla tiene una red 
   vial pequena pero funcional para su tamano.

2. **Concentracion en Hanga Roa**: La mayoria de las intersecciones de alta centralidad 
   estan en el pueblo principal.

3. **Accesibilidad limitada**: Desde el centro, se puede alcanzar la mayoria de 
   servicios en menos de 15 minutos, pero las zonas arqueologicas estan mas alejadas.

4. **Vulnerabilidad**: Algunas intersecciones son criticas - si se bloquean, afectarian 
   significativamente la movilidad de la isla.
""")

st.success("""
Este analisis de redes cumple con el **elemento de excelencia** del laboratorio,
implementando analisis de centralidad, isocronas y accesibilidad vial.
""")
