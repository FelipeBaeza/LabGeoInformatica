#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para descargar datos geoespaciales de múltiples fuentes.

Fuentes soportadas:
- OSM (OpenStreetMap): Datos vectoriales
- INE (Instituto Nacional de Estadísticas): Datos censales
- Sentinel: Imágenes satelitales (requiere cuenta Copernicus)

Uso:
    python download_data.py --comuna "La Florida" --year 2024
    python download_data.py --comuna "Providencia" --year 2024 --sources osm
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import pandas as pd
import osmnx as ox
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


def setup_directories():
    """Crear directorios necesarios si no existen."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorios configurados en {DATA_RAW_DIR}")


def download_osm_data(comuna: str, output_dir: Path) -> dict:
    """
    Descargar datos de OpenStreetMap para una comuna.
    
    Args:
        comuna: Nombre de la comuna
        output_dir: Directorio de salida
    
    Returns:
        dict: Diccionario con los GeoDataFrames descargados
    """
    print(f"\n📥 Descargando datos OSM para {comuna}...")
    
    # Definir el lugar de búsqueda - intentar diferentes formatos
    place_name = f"{comuna}, Chile"
    
    try:
        # Obtener límite de la comuna
        print("  → Obteniendo límite administrativo...")
        boundary = ox.geocode_to_gdf(place_name)
        boundary.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_boundary.geojson", driver="GeoJSON")
        
        # Obtener red de calles
        print("  → Descargando red de calles...")
        G = ox.graph_from_place(place_name, network_type='all')
        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_streets.geojson", driver="GeoJSON")
        
        # Obtener edificios
        print("  → Descargando edificios...")
        buildings = ox.features_from_place(place_name, tags={'building': True})
        if not buildings.empty:
            buildings = buildings.reset_index()
            # Filtrar solo geometrías Polygon/MultiPolygon
            buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            buildings.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_buildings.geojson", driver="GeoJSON")
        
        # Obtener amenidades (servicios)
        print("  → Descargando amenidades...")
        amenities = ox.features_from_place(place_name, tags={'amenity': True})
        if not amenities.empty:
            amenities = amenities.reset_index()
            amenities.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_amenities.geojson", driver="GeoJSON")
        
        # Obtener áreas verdes
        print("  → Descargando áreas verdes...")
        green_areas = ox.features_from_place(place_name, tags={'leisure': ['park', 'garden', 'playground']})
        if not green_areas.empty:
            green_areas = green_areas.reset_index()
            green_areas.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_green_areas.geojson", driver="GeoJSON")
        
        # Obtener transporte público
        print("  → Descargando paradas de transporte...")
        transport = ox.features_from_place(place_name, tags={'public_transport': True})
        if not transport.empty:
            transport = transport.reset_index()
            transport.to_file(output_dir / f"{comuna.lower().replace(' ', '_')}_transport.geojson", driver="GeoJSON")
        
        print(f"  ✓ Datos OSM descargados exitosamente")
        
        return {
            'boundary': boundary,
            'streets': edges,
            'buildings': buildings if not buildings.empty else None,
            'amenities': amenities if not amenities.empty else None,
            'green_areas': green_areas if not green_areas.empty else None,
            'transport': transport if not transport.empty else None
        }
        
    except Exception as e:
        print(f"  ✗ Error al descargar datos OSM: {e}")
        return {}


def download_ine_data(comuna: str, year: int, output_dir: Path) -> dict:
    """
    Descargar datos del INE (Instituto Nacional de Estadísticas).
    
    Nota: Esta función simula la descarga ya que los datos del INE
    requieren acceso a su portal o API específica.
    
    Args:
        comuna: Nombre de la comuna
        year: Año de los datos
        output_dir: Directorio de salida
    
    Returns:
        dict: Diccionario con los datos descargados
    """
    print(f"\n📥 Descargando datos INE para {comuna} ({year})...")
    
    # Nota: En un caso real, aquí se conectaría a la API del INE
    # o se descargarían los datos desde su portal
    print("  ⚠ Los datos del INE deben descargarse manualmente desde:")
    print("    https://www.ine.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda")
    print("    https://geoine-ine-chile.opendata.arcgis.com/")
    
    # Crear archivo de placeholder con instrucciones
    readme_path = output_dir / f"{comuna.lower().replace(' ', '_')}_ine_README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""Datos INE para {comuna} - {year}
{'='*50}

Para descargar datos del INE:

1. Censo 2017:
   - Portal: https://geoine-ine-chile.opendata.arcgis.com/
   - Buscar: Manzanas Censales, Zonas Censales
   
2. Proyecciones de Población:
   - Portal: https://www.ine.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion
   
3. Variables disponibles:
   - Población total
   - Viviendas
   - Hogares
   - Nivel educacional
   - Actividad económica

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    print(f"  ✓ Instrucciones guardadas en {readme_path}")
    return {}


def download_sentinel_data(comuna: str, year: int, output_dir: Path) -> dict:
    """
    Descargar imágenes Sentinel.
    
    Nota: Requiere cuenta en Copernicus Open Access Hub.
    
    Args:
        comuna: Nombre de la comuna
        year: Año de las imágenes
        output_dir: Directorio de salida
    
    Returns:
        dict: Diccionario con los datos descargados
    """
    print(f"\n📥 Preparando descarga Sentinel para {comuna} ({year})...")
    
    print("  ⚠ Las imágenes Sentinel requieren:")
    print("    1. Cuenta en Copernicus: https://scihub.copernicus.eu/dhus/#/self-registration")
    print("    2. O usar Google Earth Engine: https://earthengine.google.com/")
    
    # Crear archivo de placeholder con instrucciones
    readme_path = output_dir / f"{comuna.lower().replace(' ', '_')}_sentinel_README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""Imágenes Sentinel para {comuna} - {year}
{'='*50}

Opciones para obtener imágenes satelitales:

1. Copernicus Open Access Hub (Sentinel-2):
   - URL: https://scihub.copernicus.eu/
   - Producto: S2A_MSIL2A (Level-2A, atmosféricamente corregido)
   - Bandas útiles: B2, B3, B4 (RGB), B8 (NIR)
   
2. Google Earth Engine:
   - Colección: COPERNICUS/S2_SR
   - Más fácil de procesar
   
3. Sentinel Hub:
   - URL: https://www.sentinel-hub.com/
   - API REST disponible

Índices calculables:
   - NDVI: (B8 - B4) / (B8 + B4) - Vegetación
   - NDBI: (B11 - B8) / (B11 + B8) - Áreas construidas
   - NDWI: (B3 - B8) / (B3 + B8) - Agua

Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    print(f"  ✓ Instrucciones guardadas en {readme_path}")
    return {}


def create_summary_report(comuna: str, year: int, data: dict, output_dir: Path):
    """Crear reporte resumen de los datos descargados."""
    print("\n📊 Generando reporte de datos...")
    
    report_path = output_dir / f"{comuna.lower().replace(' ', '_')}_data_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""
{'='*60}
REPORTE DE DESCARGA DE DATOS GEOESPACIALES
{'='*60}

Comuna: {comuna}
Año: {year}
Fecha de descarga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATOS DESCARGADOS
{'-'*60}
""")
        
        for source, datasets in data.items():
            f.write(f"\n[{source.upper()}]\n")
            if datasets:
                for name, gdf in datasets.items():
                    if gdf is not None and hasattr(gdf, 'shape'):
                        f.write(f"  - {name}: {gdf.shape[0]} registros\n")
            else:
                f.write("  - Sin datos descargados\n")
        
        f.write(f"""
{'-'*60}
PRÓXIMOS PASOS
{'-'*60}
1. Cargar datos en PostGIS usando el script load_to_postgis.py
2. Validar calidad de datos
3. Comenzar análisis exploratorio

{'='*60}
""")
    
    print(f"  ✓ Reporte guardado en {report_path}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Descargar datos geoespaciales para análisis comunal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python download_data.py --comuna "La Florida" --year 2024
  python download_data.py --comuna "Providencia" --year 2024 --sources osm
  python download_data.py --comuna "Santiago" --year 2024 --sources osm ine
        """
    )
    
    parser.add_argument(
        '--comuna', '-c',
        type=str,
        required=True,
        help='Nombre de la comuna a analizar'
    )
    
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=2024,
        help='Año de los datos (default: 2024)'
    )
    
    parser.add_argument(
        '--sources', '-s',
        nargs='+',
        choices=['osm', 'ine', 'sentinel', 'all'],
        default=['all'],
        help='Fuentes de datos a descargar (default: all)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║     DESCARGA DE DATOS GEOESPACIALES - GEOINFORMÁTICA        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Comuna: {args.comuna}")
    print(f"Año: {args.year}")
    print(f"Fuentes: {', '.join(args.sources)}")
    
    # Configurar directorios
    setup_directories()
    
    # Crear subdirectorio para la comuna
    comuna_dir = DATA_RAW_DIR / args.comuna.lower().replace(' ', '_')
    comuna_dir.mkdir(exist_ok=True)
    
    # Determinar qué fuentes descargar
    sources = args.sources
    if 'all' in sources:
        sources = ['osm', 'ine', 'sentinel']
    
    # Almacenar datos descargados
    all_data = {}
    
    # Descargar de cada fuente
    if 'osm' in sources:
        all_data['osm'] = download_osm_data(args.comuna, comuna_dir)
    
    if 'ine' in sources:
        all_data['ine'] = download_ine_data(args.comuna, args.year, comuna_dir)
    
    if 'sentinel' in sources:
        all_data['sentinel'] = download_sentinel_data(args.comuna, args.year, comuna_dir)
    
    # Generar reporte
    create_summary_report(args.comuna, args.year, all_data, comuna_dir)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   DESCARGA COMPLETADA                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"Datos guardados en: {comuna_dir}")
    print("\nPróximo paso: python scripts/load_to_postgis.py")


if __name__ == "__main__":
    main()
