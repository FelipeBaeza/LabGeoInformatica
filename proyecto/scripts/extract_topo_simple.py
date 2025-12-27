#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script simplificado para extraer valores topográficos usando rasterio.

Extrae valores promedio de:
- Elevación (DEM)
- Pendiente (slope)
- Orientación (aspect)

Para cada celda de la grilla de 200m x 200m.

Uso:
    python extract_topo_simple.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

print("="*70)
print("EXTRACCIÓN DE VARIABLES TOPOGRÁFICAS - MÉTODO SIMPLIFICADO")
print("="*70)

# ============================================================================
# 1. CARGAR GRILLA
# ============================================================================
print("\n[1/4] Cargando grilla de análisis...")

grid_file = DATA_PROCESSED / "prepared.gpkg"
if not grid_file.exists():
    print(f"  ✗ No se encontró {grid_file}")
    print("  → Creando grilla básica desde límite...")
    
    # Cargar límite
    boundary_file = DATA_RAW / "isla_de_pascua" / "isla_de_pascua_boundary.geojson"
    if not boundary_file.exists():
        print(f"  ✗ No se encontró límite: {boundary_file}")
        sys.exit(1)
    
    boundary = gpd.read_file(boundary_file)
    
    # Crear grilla simple
    from shapely.geometry import box
    minx, miny, maxx, maxy = boundary.total_bounds
    cell_size = 200
    
    cells = []
    cell_ids = []
    i = 0
    for x in np.arange(minx, maxx, cell_size):
        for y in np.arange(miny, maxy, cell_size):
            cell = box(x, y, x + cell_size, y + cell_size)
            if boundary.unary_union.intersects(cell):
                cells.append(cell)
                cell_ids.append(i)
                i += 1
    
    grid = gpd.GeoDataFrame({'cell_id': cell_ids, 'geometry': cells}, crs=boundary.crs)
    print(f"  ✓ Grilla creada: {len(grid)} celdas")
else:
    try:
        grid = gpd.read_file(grid_file)
        print(f"  ✓ Grilla cargada: {len(grid)} celdas")
    except:
        import fiona
        layers = fiona.listlayers(grid_file)
        grid = gpd.read_file(grid_file, layer=layers[0])
        print(f"  ✓ Grilla cargada: {len(grid)} celdas")

# Asegurar que tenga cell_id
if 'cell_id' not in grid.columns:
    grid['cell_id'] = range(len(grid))

# ============================================================================
# 2. CARGAR RASTERS
# ============================================================================
print("\n[2/4] Cargando rasters topográficos...")

rasters = {
    'elevation': DATA_PROCESSED / "dem_isla_pascua_clipped.tif",
    'slope': DATA_PROCESSED / "slope_isla_pascua.tif",
    'aspect': DATA_PROCESSED / "aspect_isla_pascua.tif"
}

for name, path in rasters.items():
    if not path.exists():
        print(f"  ✗ No encontrado: {path}")
        sys.exit(1)
    print(f"  ✓ {name:12s} → {path.name}")

# ============================================================================
# 3. EXTRAER VALORES
# ============================================================================
print("\n[3/4] Extrayendo valores topográficos...")

# Reproyectar grilla si es necesario
with rasterio.open(rasters['elevation']) as src:
    dem_crs = src.crs

if grid.crs != dem_crs:
    print(f"  → Reproyectando grilla a {dem_crs}")
    grid = grid.to_crs(dem_crs)

# Extraer para cada raster
for var_name, raster_path in rasters.items():
    print(f"\n  Procesando {var_name}...")
    
    with rasterio.open(raster_path) as src:
        # Leer datos
        data = src.read(1)
        transform = src.transform
        
        # Extraer valores para cada celda
        values_mean = []
        values_std = []
        values_min = []
        values_max = []
        
        for idx, row in tqdm(grid.iterrows(), total=len(grid), desc=f"    {var_name}"):
            # Crear máscara para esta celda
            geom = [mapping(row.geometry)]
            
            try:
                # Obtener ventana de la celda
                window = rasterio.features.geometry_window(src, geom)
                
                # Leer datos de la ventana
                window_data = src.read(1, window=window)
                
                # Crear máscara para la geometría
                mask = rasterio.features.geometry_mask(
                    geom,
                    out_shape=window_data.shape,
                    transform=rasterio.windows.transform(window, transform),
                    invert=True
                )
                
                # Aplicar máscara
                masked_data = np.ma.masked_array(window_data, ~mask)
                
                # Calcular estadísticas
                if masked_data.count() > 0:
                    values_mean.append(float(masked_data.mean()))
                    values_std.append(float(masked_data.std()))
                    values_min.append(float(masked_data.min()))
                    values_max.append(float(masked_data.max()))
                else:
                    values_mean.append(np.nan)
                    values_std.append(np.nan)
                    values_min.append(np.nan)
                    values_max.append(np.nan)
            except Exception as e:
                values_mean.append(np.nan)
                values_std.append(np.nan)
                values_min.append(np.nan)
                values_max.append(np.nan)
        
        # Agregar a grid
        grid[f'{var_name}_mean'] = values_mean
        grid[f'{var_name}_std'] = values_std
        grid[f'{var_name}_min'] = values_min
        grid[f'{var_name}_max'] = values_max
    
    print(f"  ✓ {var_name} completado")

# ============================================================================
# 4. GUARDAR RESULTADOS
# ============================================================================
print("\n[4/4] Guardando resultados...")

# Guardar GeoPackage
output_gpkg = DATA_PROCESSED / "grid_with_topography.gpkg"
grid.to_file(output_gpkg, driver="GPKG")
print(f"  ✓ GeoPackage: {output_gpkg}")

# Guardar CSV
topo_cols = ['cell_id', 'elevation_mean', 'elevation_std', 'slope_mean', 'slope_std', 'aspect_mean']
csv_file = DATA_PROCESSED / "grid_topographic_features.csv"

# Preparar DataFrame sin geometría
df_export = grid[topo_cols].copy()
df_export.to_csv(csv_file, index=False)
print(f"  ✓ CSV: {csv_file}")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "="*70)
print("ESTADÍSTICAS DE VARIABLES TOPOGRÁFICAS")
print("="*70)

summary_cols = [col for col in grid.columns if '_mean' in col]
print(f"\nVariables generadas: {len(summary_cols)}")
print("\nEstadísticas descriptivas:")
print(grid[summary_cols].describe().round(2))

print("\n" + "="*70)
print("EXTRACCIÓN COMPLETADA")
print("="*70)
print(f"\nCeldas procesadas: {len(grid)}")
print(f"Variables por celda: {len(topo_cols)}")
print("\nArchivos generados:")
print(f"  1. {output_gpkg}")
print(f"  2. {csv_file}")
