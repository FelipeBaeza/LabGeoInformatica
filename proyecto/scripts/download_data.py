#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para descargar datos geoespaciales de múltiples fuentes.

Fuentes soportadas:
- OSM (OpenStreetMap): Datos vectoriales
- INE (Instituto Nacional de Estadísticas): Datos censales (simulados)
- Sentinel: Imágenes satelitales (requiere cuenta Copernicus)
- DEM: Modelo de elevación digital

Uso:
    python download_data.py --comuna "La Florida" --year 2024
    python download_data.py --comuna "Providencia" --year 2024 --sources osm

Estructura de clase según requerimientos del laboratorio.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"


class DataDownloader:
    """
    Clase para descargar datos geoespaciales de múltiples fuentes.
    
    Implementa la estructura requerida por el laboratorio con métodos para:
    - Descarga de límites administrativos
    - Red vial desde OpenStreetMap
    - Imágenes Sentinel-2 (placeholder para GEE)
    - DEM (placeholder)
    - Datos censales/socioeconómicos
    
    Attributes:
        comuna: Nombre de la comuna a analizar
        output_dir: Directorio de salida para los datos
    """
    
    def __init__(self, comuna_name: str, output_dir: str = '../data'):
        """
        Inicializar el descargador de datos.
        
        Args:
            comuna_name: Nombre de la comuna
            output_dir: Directorio de salida (default: '../data')
        """
        self.comuna = comuna_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Subdirectorios
        self.raw_dir = self.output_dir / 'raw' / comuna_name.lower().replace(' ', '_')
        self.processed_dir = self.output_dir / 'processed'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self._downloaded_data = {}
        print(f"✓ DataDownloader inicializado para {comuna_name}")
        print(f"  Directorio de salida: {self.raw_dir}")
    
    def download_administrative_boundaries(self) -> Optional[gpd.GeoDataFrame]:
        """
        Descarga límites administrativos desde OpenStreetMap/Nominatim.
        
        En un caso real, se conectaría a IDE Chile WFS para obtener
        límites oficiales.
        
        Returns:
            GeoDataFrame con el límite de la comuna
        """
        print(f"\n→ Descargando límites administrativos para {self.comuna}...")
        
        place_name = f"{self.comuna}, Chile"
        
        try:
            boundary = ox.geocode_to_gdf(place_name)
            output_path = self.raw_dir / f"{self.comuna.lower().replace(' ', '_')}_boundary.geojson"
            boundary.to_file(output_path, driver="GeoJSON")
            
            self._downloaded_data['boundary'] = boundary
            print(f"  ✓ Límite guardado: {output_path}")
            return boundary
            
        except Exception as e:
            print(f"  ✗ Error al descargar límites: {e}")
            return None
    
    def download_osm_network(self, network_type: str = 'all') -> Optional[gpd.GeoDataFrame]:
        """
        Descarga red vial desde OpenStreetMap usando OSMnx.
        
        Args:
            network_type: Tipo de red ('all', 'drive', 'walk', 'bike')
        
        Returns:
            GeoDataFrame con las calles y GeoPackage con el grafo
        """
        print(f"\n→ Descargando red vial ({network_type})...")
        
        place_name = f"{self.comuna}, Chile"
        
        try:
            # Descargar grafo de calles
            G = ox.graph_from_place(place_name, network_type=network_type)
            
            # Guardar como GeoPackage
            output_gpkg = self.raw_dir / 'red_vial.gpkg'
            ox.save_graph_geopackage(G, filepath=output_gpkg)
            
            # Convertir a GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)
            output_geojson = self.raw_dir / f"{self.comuna.lower().replace(' ', '_')}_streets.geojson"
            edges.to_file(output_geojson, driver="GeoJSON")
            
            # Guardar grafo en formato GraphML
            output_graphml = self.raw_dir / 'network_graph.graphml'
            ox.save_graphml(G, filepath=output_graphml)
            
            self._downloaded_data['streets'] = edges
            self._downloaded_data['network_graph'] = G
            
            print(f"  ✓ Red vial guardada: {len(edges)} segmentos")
            print(f"    - GeoPackage: {output_gpkg}")
            print(f"    - GraphML: {output_graphml}")
            return edges
            
        except Exception as e:
            print(f"  ✗ Error al descargar red vial: {e}")
            return None
    
    def download_sentinel2(self, start_date: str, end_date: str, 
                           cloud_cover_max: int = 20) -> Dict[str, Any]:
        """
        Descarga imágenes Sentinel-2 desde Google Earth Engine.
        
        Nota: Requiere autenticación en GEE. Esta implementación es
        un placeholder que crea instrucciones para descarga manual.
        
        Args:
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            cloud_cover_max: Máximo porcentaje de nubes
        
        Returns:
            Diccionario con información de descarga
        """
        print(f"\n→ Preparando descarga Sentinel-2 ({start_date} a {end_date})...")
        
        # Crear archivo de instrucciones y código GEE
        gee_code = f'''
// ========================================
// Código para Google Earth Engine
// Copiar y pegar en https://code.earthengine.google.com/
// ========================================

// Definir área de interés
var comuna = ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq('ADM2_NAME', '{self.comuna}'));

// Alternativamente, definir manualmente el bounding box
// var geometry = ee.Geometry.Rectangle([-109.5, -27.3, -109.1, -27.0]);

// Colección Sentinel-2 Surface Reflectance
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(comuna)
    .filterDate('{start_date}', '{end_date}')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', {cloud_cover_max}));

print('Imágenes disponibles:', s2.size());

// Crear mosaico (mediana para reducir nubes)
var mosaic = s2.median().clip(comuna);

// Calcular NDVI
var ndvi = mosaic.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Calcular NDBI (Índice de Área Construida)
var ndbi = mosaic.normalizedDifference(['B11', 'B8']).rename('NDBI');

// Visualizar
Map.centerObject(comuna, 12);
Map.addLayer(mosaic, {{bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}}, 'RGB');
Map.addLayer(ndvi, {{min: -0.5, max: 0.8, palette: ['red', 'yellow', 'green']}}, 'NDVI');

// Exportar NDVI
Export.image.toDrive({{
    image: ndvi,
    description: '{self.comuna.replace(" ", "_")}_NDVI',
    scale: 10,
    region: comuna,
    maxPixels: 1e13
}});

// Exportar NDBI
Export.image.toDrive({{
    image: ndbi,
    description: '{self.comuna.replace(" ", "_")}_NDBI', 
    scale: 10,
    region: comuna,
    maxPixels: 1e13
}});
'''
        
        # Guardar código GEE
        gee_path = self.raw_dir / 'sentinel2_gee_code.js'
        with open(gee_path, 'w', encoding='utf-8') as f:
            f.write(gee_code)
        
        # Crear README con instrucciones
        readme_path = self.raw_dir / 'sentinel2_README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Descarga de Imágenes Sentinel-2 para {self.comuna}

## Opción 1: Google Earth Engine (Recomendado)

1. Ir a [Google Earth Engine](https://code.earthengine.google.com/)
2. Copiar el código de `sentinel2_gee_code.js`
3. Ejecutar y exportar a Google Drive
4. Descargar los archivos GeoTIFF

## Opción 2: Copernicus Open Access Hub

1. Ir a [Copernicus Hub](https://scihub.copernicus.eu/)
2. Buscar: {self.comuna}, Chile
3. Fechas: {start_date} a {end_date}
4. Producto: S2A_MSIL2A (Level-2A)
5. Nubes: < {cloud_cover_max}%

## Índices Calculables

| Índice | Fórmula | Uso |
|--------|---------|-----|
| NDVI | (B8-B4)/(B8+B4) | Vegetación |
| NDBI | (B11-B8)/(B11+B8) | Áreas construidas |
| NDWI | (B3-B8)/(B3+B8) | Agua |

Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")
        
        print(f"  ✓ Código GEE guardado: {gee_path}")
        print(f"  ✓ Instrucciones: {readme_path}")
        
        return {
            'gee_code_path': str(gee_path),
            'readme_path': str(readme_path),
            'start_date': start_date,
            'end_date': end_date
        }
    
    def download_dem(self, source: str = 'ALOS') -> Dict[str, Any]:
        """
        Descarga DEM (Digital Elevation Model) desde ALOS PALSAR o SRTM.
        
        Args:
            source: Fuente del DEM ('ALOS' o 'SRTM')
        
        Returns:
            Diccionario con información de descarga
        """
        print(f"\n→ Preparando descarga DEM ({source})...")
        
        dem_code = f'''
// ========================================
// Código GEE para descargar DEM
// ========================================

var comuna = ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq('ADM2_NAME', '{self.comuna}'));

// ALOS DEM (30m resolución)
var alos = ee.Image('JAXA/ALOS/AW3D30/V2_2')
    .select('AVE_DSM')
    .clip(comuna);

// SRTM DEM (30m resolución)
var srtm = ee.Image('USGS/SRTMGL1_003')
    .select('elevation')
    .clip(comuna);

// Calcular pendiente
var slope = ee.Terrain.slope(alos);

// Visualizar
Map.addLayer(alos, {{min: 0, max: 500, palette: ['green', 'yellow', 'red']}}, 'ALOS DEM');
Map.addLayer(slope, {{min: 0, max: 45, palette: ['green', 'yellow', 'red']}}, 'Slope');

// Exportar DEM
Export.image.toDrive({{
    image: alos,
    description: '{self.comuna.replace(" ", "_")}_DEM',
    scale: 30,
    region: comuna,
    maxPixels: 1e13
}});

// Exportar Pendiente
Export.image.toDrive({{
    image: slope,
    description: '{self.comuna.replace(" ", "_")}_SLOPE',
    scale: 30,
    region: comuna,
    maxPixels: 1e13
}});
'''
        
        dem_code_path = self.raw_dir / 'dem_gee_code.js'
        with open(dem_code_path, 'w', encoding='utf-8') as f:
            f.write(dem_code)
        
        print(f"  ✓ Código GEE para DEM guardado: {dem_code_path}")
        
        return {
            'gee_code_path': str(dem_code_path),
            'source': source
        }
    
    def download_census_data(self, year: int = 2017) -> Optional[gpd.GeoDataFrame]:
        """
        Descarga o simula datos censales/socioeconómicos.
        
        En un caso real, se conectaría a la API del INE o se descargarían
        los datos desde GeoINE. Esta implementación crea datos simulados
        basados en la estructura de edificios.
        
        Args:
            year: Año del censo (default: 2017)
        
        Returns:
            GeoDataFrame con datos socioeconómicos por manzana/zona
        """
        print(f"\n→ Generando datos socioeconómicos (basados en Censo {year})...")
        
        # Primero necesitamos el límite
        if 'boundary' not in self._downloaded_data:
            self.download_administrative_boundaries()
        
        if 'boundary' not in self._downloaded_data:
            print("  ✗ No se pudo obtener el límite de la comuna")
            return None
        
        boundary = self._downloaded_data['boundary']
        
        # Crear grilla de "manzanas censales" simuladas
        from shapely.geometry import box as shapely_box
        
        bounds = boundary.total_bounds  # [minx, miny, maxx, maxy]
        cell_size = 0.005  # ~500m en grados
        
        cells = []
        x = bounds[0]
        while x < bounds[2]:
            y = bounds[1]
            while y < bounds[3]:
                cell = shapely_box(x, y, x + cell_size, y + cell_size)
                if cell.intersects(boundary.unary_union):
                    cells.append(cell)
                y += cell_size
            x += cell_size
        
        # Crear GeoDataFrame con datos simulados
        np.random.seed(42)  # Para reproducibilidad
        n_cells = len(cells)
        
        census_data = gpd.GeoDataFrame({
            'geometry': cells,
            'manzana_id': [f'MZ_{i:04d}' for i in range(n_cells)],
            # Datos demográficos simulados
            'poblacion': np.random.poisson(150, n_cells),
            'viviendas': np.random.poisson(50, n_cells),
            'hogares': np.random.poisson(45, n_cells),
            # Datos socioeconómicos simulados
            'ingreso_promedio': np.random.normal(500000, 150000, n_cells).clip(200000, 2000000).astype(int),
            'escolaridad_promedio': np.random.normal(12, 3, n_cells).clip(4, 18).round(1),
            'tasa_desempleo': np.random.beta(2, 10, n_cells).round(3),
            # Infraestructura
            'acceso_agua': np.random.beta(9, 1, n_cells).round(2),
            'acceso_electricidad': np.random.beta(9.5, 0.5, n_cells).round(2),
            'acceso_internet': np.random.beta(7, 3, n_cells).round(2),
        }, crs="EPSG:4326")
        
        # Filtrar solo celdas dentro del límite
        census_data = census_data[census_data.intersects(boundary.unary_union)].reset_index(drop=True)
        
        output_path = self.raw_dir / f"{self.comuna.lower().replace(' ', '_')}_censo_{year}.geojson"
        census_data.to_file(output_path, driver="GeoJSON")
        
        self._downloaded_data['census'] = census_data
        
        print(f"  ✓ Datos censales simulados: {len(census_data)} manzanas")
        print(f"    Guardado en: {output_path}")
        
        # Crear README explicativo
        readme_path = self.raw_dir / 'censo_README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Datos Censales para {self.comuna}

## Fuente Original
Los datos reales del Censo deben descargarse desde:
- [GeoINE](https://geoine-ine-chile.opendata.arcgis.com/)
- [Portal INE](https://www.ine.cl/estadisticas/sociales/censos-de-poblacion-y-vivienda)

## Datos Simulados
Este archivo contiene datos **simulados** para propósitos de demostración.
Las variables incluyen:

| Variable | Descripción | Rango |
|----------|-------------|-------|
| poblacion | Población total | Poisson(150) |
| viviendas | Número de viviendas | Poisson(50) |
| hogares | Número de hogares | Poisson(45) |
| ingreso_promedio | Ingreso promedio (CLP) | 200k-2M |
| escolaridad_promedio | Años de escolaridad | 4-18 |
| tasa_desempleo | Tasa de desempleo | 0-1 |
| acceso_agua | Proporción con agua potable | 0-1 |
| acceso_electricidad | Proporción con electricidad | 0-1 |
| acceso_internet | Proporción con internet | 0-1 |

**Nota**: Para análisis real, reemplace estos datos con los del Censo 2017.

Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")
        
        return census_data
    
    def download_buildings(self) -> Optional[gpd.GeoDataFrame]:
        """Descarga edificios desde OpenStreetMap."""
        print(f"\n→ Descargando edificios...")
        
        place_name = f"{self.comuna}, Chile"
        
        try:
            buildings = ox.features_from_place(place_name, tags={'building': True})
            if not buildings.empty:
                buildings = buildings.reset_index()
                buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                
                output_path = self.raw_dir / f"{self.comuna.lower().replace(' ', '_')}_buildings.geojson"
                buildings.to_file(output_path, driver="GeoJSON")
                
                self._downloaded_data['buildings'] = buildings
                print(f"  ✓ Edificios descargados: {len(buildings)}")
                return buildings
        except Exception as e:
            print(f"  ✗ Error: {e}")
        return None
    
    def download_amenities(self) -> Optional[gpd.GeoDataFrame]:
        """Descarga amenidades desde OpenStreetMap."""
        print(f"\n→ Descargando amenidades...")
        
        place_name = f"{self.comuna}, Chile"
        
        try:
            amenities = ox.features_from_place(place_name, tags={'amenity': True})
            if not amenities.empty:
                amenities = amenities.reset_index()
                
                output_path = self.raw_dir / f"{self.comuna.lower().replace(' ', '_')}_amenities.geojson"
                amenities.to_file(output_path, driver="GeoJSON")
                
                self._downloaded_data['amenities'] = amenities
                print(f"  ✓ Amenidades descargadas: {len(amenities)}")
                return amenities
        except Exception as e:
            print(f"  ✗ Error: {e}")
        return None
    
    def download_all(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Descarga todos los datos disponibles.
        
        Args:
            start_date: Fecha inicio para imágenes satelitales
            end_date: Fecha fin para imágenes satelitales
        
        Returns:
            Diccionario con todos los datos descargados
        """
        print("\n" + "="*60)
        print(f"DESCARGA COMPLETA DE DATOS: {self.comuna}")
        print("="*60)
        
        # Datos vectoriales
        self.download_administrative_boundaries()
        self.download_osm_network()
        self.download_buildings()
        self.download_amenities()
        
        # Datos censales
        self.download_census_data()
        
        # Imágenes satelitales (instrucciones)
        if start_date and end_date:
            self.download_sentinel2(start_date, end_date)
        else:
            self.download_sentinel2('2024-01-01', '2024-12-31')
        
        # DEM
        self.download_dem()
        
        # Generar reporte
        self._generate_summary_report()
        
        print("\n" + "="*60)
        print("DESCARGA COMPLETADA")
        print("="*60)
        print(f"Datos guardados en: {self.raw_dir}")
        
        return self._downloaded_data
    
    def _generate_summary_report(self):
        """Genera reporte resumen de los datos descargados."""
        report_path = self.raw_dir / 'data_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""
{'='*60}
REPORTE DE DESCARGA DE DATOS GEOESPACIALES
{'='*60}

Comuna: {self.comuna}
Fecha de descarga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATOS DESCARGADOS
{'-'*60}
""")
            for name, data in self._downloaded_data.items():
                if hasattr(data, 'shape'):
                    f.write(f"  - {name}: {data.shape[0]} registros\n")
                else:
                    f.write(f"  - {name}: disponible\n")
            
            f.write(f"""
{'-'*60}
PRÓXIMOS PASOS
{'-'*60}
1. Cargar datos en PostGIS: python load_to_postgis.py
2. Ejecutar notebooks de análisis
3. Iniciar aplicación web

{'='*60}
""")
        
        print(f"\n✓ Reporte guardado: {report_path}")


def setup_directories():
    """Crear directorios necesarios si no existen."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorios configurados en {DATA_RAW_DIR}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Descargar datos geoespaciales para análisis comunal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python download_data.py --comuna "La Florida" --year 2024
  python download_data.py --comuna "Isla de Pascua" --sources osm
  python download_data.py --comuna "Santiago" --year 2024 --sources all
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
        choices=['osm', 'census', 'sentinel', 'dem', 'all'],
        default=['all'],
        help='Fuentes de datos a descargar (default: all)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Fecha inicio para imágenes satelitales (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='Fecha fin para imágenes satelitales (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║     DESCARGA DE DATOS GEOESPACIALES - GEOINFORMÁTICA        ║
║          Usando clase DataDownloader                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Comuna: {args.comuna}")
    print(f"Año: {args.year}")
    print(f"Fuentes: {', '.join(args.sources)}")
    
    # Crear instancia del descargador
    downloader = DataDownloader(args.comuna, str(DATA_RAW_DIR.parent))
    
    # Determinar qué descargar
    sources = args.sources
    if 'all' in sources:
        downloader.download_all(args.start_date, args.end_date)
    else:
        if 'osm' in sources:
            downloader.download_administrative_boundaries()
            downloader.download_osm_network()
            downloader.download_buildings()
            downloader.download_amenities()
        
        if 'census' in sources:
            downloader.download_census_data(args.year)
        
        if 'sentinel' in sources:
            downloader.download_sentinel2(args.start_date, args.end_date)
        
        if 'dem' in sources:
            downloader.download_dem()
    
    print("\n✓ Proceso completado!")
    print(f"Datos guardados en: {downloader.raw_dir}")


if __name__ == "__main__":
    main()
