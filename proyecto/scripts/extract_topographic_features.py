#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para extraer valores topográficos para la grilla de análisis.

Extrae valores de:
- Elevación (DEM)
- Pendiente (slope)
- Orientación (aspect)

Para cada celda de la grilla de 200m x 200m usada en el análisis ML.

Uso:
    python extract_topographic_features.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

print("="*70)
print("EXTRACCIÓN DE VARIABLES TOPOGRÁFICAS PARA GRILLA DE ANÁLISIS")
print("="*70)

# ============================================================================
# 1. CARGAR GRILLA DE ANÁLISIS
# ============================================================================
print("\n[1/4] Cargando grilla de análisis...")

# Intentar cargar desde GeoPackage procesado
grid_file = DATA_PROCESSED / "prepared.gpkg"

if not grid_file.exists():
    print(f"  ✗ No se encontró grilla en {grid_file}")
    print("  → Ejecutar primero los notebooks de análisis para generar la grilla")
    sys.exit(1)

# Cargar la capa de grilla
try:
    grid = gpd.read_file(grid_file, layer='grid_analysis')
    print(f"  ✓ Grilla cargada: {len(grid)} celdas")
except Exception as e:
    print(f"  ⚠ No se pudo cargar capa 'grid_analysis', intentando con primera capa...")
    try:
        import fiona
        layers = fiona.listlayers(grid_file)
        print(f"  → Capas disponibles: {layers}")
        grid = gpd.read_file(grid_file, layer=layers[0])
        print(f"  ✓ Grilla cargada desde capa '{layers[0]}': {len(grid)} celdas")
    except Exception as e2:
        print(f"  ✗ Error cargando grilla: {e2}")
        sys.exit(1)

# ============================================================================
# 2. CARGAR RASTERS TOPOGRÁFICOS
# ============================================================================
print("\n[2/4] Cargando rasters topográficos...")

rasters = {
    'elevation': DATA_PROCESSED / "dem_isla_pascua_clipped.tif",
    'slope': DATA_PROCESSED / "slope_isla_pascua.tif",
    'aspect': DATA_PROCESSED / "aspect_isla_pascua.tif"
}

# Verificar que existan
for name, path in rasters.items():
    if not path.exists():
        print(f"  ✗ No se encontró {name}: {path}")
        sys.exit(1)
    else:
        print(f"  ✓ {name:12s} → {path.name}")

# ============================================================================
# 3. EXTRAER ESTADÍSTICAS ZONALES
# ============================================================================
print("\n[3/4] Extrayendo estadísticas zonales...")
print("  (Esto puede tomar varios minutos...)")

# Reproyectar grilla al CRS del DEM si es necesario
with rasterio.open(rasters['elevation']) as src:
    dem_crs = src.crs

if grid.crs != dem_crs:
    print(f"  → Reproyectando grilla de {grid.crs} a {dem_crs}")
    grid = grid.to_crs(dem_crs)

# Extraer estadísticas para cada raster
for var_name, raster_path in tqdm(rasters.items(), desc="  Procesando rasters"):
    stats = zonal_stats(
        grid.geometry,
        str(raster_path),
        stats=['mean', 'min', 'max', 'std'],
        nodata=-9999,
        all_touched=True
    )
    
    # Convertir a DataFrame y agregar a grid
    stats_df = pd.DataFrame(stats)
    grid[f'{var_name}_mean'] = stats_df['mean']
    grid[f'{var_name}_min'] = stats_df['min']
    grid[f'{var_name}_max'] = stats_df['max']
    grid[f'{var_name}_std'] = stats_df['std']

print(f"  ✓ Estadísticas extraídas para {len(rasters)} variables")

# ============================================================================
# 4. GUARDAR RESULTADOS
# ============================================================================
print("\n[4/4] Guardando resultados...")

# Guardar grilla con variables topográficas
output_file = DATA_PROCESSED / "grid_with_topography.gpkg"
grid.to_file(output_file, driver="GPKG")
print(f"  ✓ Grilla guardada en {output_file}")

# Guardar también como CSV para fácil integración en ML
csv_file = DATA_PROCESSED / "grid_topographic_features.csv"
topo_features = grid[[
    'cell_id', 
    'elevation_mean', 'elevation_std',
    'slope_mean', 'slope_std',
    'aspect_mean'
]].copy()

# Manejar geometrías
if 'cell_id' not in topo_features.columns and grid.index.name:
    topo_features['cell_id'] = grid.index

topo_features.to_csv(csv_file, index=False)
print(f"  ✓ Features CSV guardado en {csv_file}")

# ============================================================================
# RESUMEN DE VARIABLES GENERADAS
# ============================================================================
print("\n" + "="*70)
print("VARIABLES TOPOGRÁFICAS GENERADAS")
print("="*70)

topo_cols = [col for col in grid.columns if any(x in col for x in ['elevation', 'slope', 'aspect'])]

print(f"\nTotal de variables: {len(topo_cols)}")
print("\nEstadísticas descriptivas:")
print(grid[topo_cols].describe().round(2))

# Estadísticas por variable
print("\n" + "-"*70)
print("RESUMEN POR VARIABLE")
print("-"*70)

for var in ['elevation', 'slope', 'aspect']:
    mean_col = f'{var}_mean'
    if mean_col in grid.columns:
        data = grid[mean_col].dropna()
        print(f"\n{var.upper()}:")
        print(f"  Rango:    {data.min():.2f} - {data.max():.2f}")
        print(f"  Promedio: {data.mean():.2f}")
        print(f"  Mediana:  {data.median():.2f}")
        print(f"  Std Dev:  {data.std():.2f}")

print("\n" + "="*70)
print("EXTRACCIÓN COMPLETADA")
print("="*70)
print("\nArchivos generados:")
print(f"  1. {output_file}")
print(f"  2. {csv_file}")
print("\nPróximos pasos:")
print("  1. Integrar features topográficos en notebook de ML")
print("  2. Re-entrenar modelos con nuevas variables")
print("  3. Evaluar mejora en performance")
