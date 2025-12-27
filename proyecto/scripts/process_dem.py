#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para procesar DEM y extraer variables topográficas.

Genera:
- DEM recortado al límite de la isla
- Slope (pendiente en grados)
- Aspect (orientación en grados)
- Hillshade (sombreado del relieve)
- Extracción de valores para grilla de análisis

Uso:
    python process_dem.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import subprocess
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DEM_SOURCE = Path("/home/felipe/Documentos/LabGeoInformatica/Datos GeoJSON/Dem Isla de pascua 4m.tif")

# Crear directorios
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PROCESAMIENTO DE DEM - ISLA DE PASCUA")
print("="*70)

# ============================================================================
# 1. COPIAR DEM AL PROYECTO
# ============================================================================
print("\n[1/6] Copiando DEM al proyecto...")
dem_raw = DATA_RAW / "dem_isla_pascua_4m.tif"

if not dem_raw.exists():
    import shutil
    shutil.copy2(DEM_SOURCE, dem_raw)
    print(f"  ✓ DEM copiado a {dem_raw}")
else:
    print(f"  ✓ DEM ya existe en {dem_raw}")

# ============================================================================
# 2. RECORTAR DEM AL LÍMITE DE LA ISLA
# ============================================================================
print("\n[2/6] Recortando DEM al límite de la isla...")

# Cargar límite de la isla
boundary_file = DATA_RAW / "isla_de_pascua" / "isla_de_pascua_boundary.geojson"
if not boundary_file.exists():
    print(f"  ⚠ Límite no encontrado en {boundary_file}")
    print("  → Usando DEM completo sin recorte")
    dem_clipped = dem_raw
else:
    boundary = gpd.read_file(boundary_file)
    
    # Reproyectar límite al CRS del DEM
    with rasterio.open(dem_raw) as src:
        dem_crs = src.crs
    
    if boundary.crs != dem_crs:
        boundary = boundary.to_crs(dem_crs)
        print(f"  → Límite reproyectado a {dem_crs}")
    
    # Recortar DEM
    dem_clipped = DATA_PROCESSED / "dem_isla_pascua_clipped.tif"
    
    with rasterio.open(dem_raw) as src:
        out_image, out_transform = mask(src, boundary.geometry, crop=True)
        out_meta = src.meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        with rasterio.open(dem_clipped, "w", **out_meta) as dest:
            dest.write(out_image)
    
    print(f"  ✓ DEM recortado guardado en {dem_clipped}")

# ============================================================================
# 3. CALCULAR SLOPE (PENDIENTE)
# ============================================================================
print("\n[3/6] Calculando slope (pendiente)...")
slope_file = DATA_PROCESSED / "slope_isla_pascua.tif"

cmd_slope = [
    "gdaldem", "slope",
    str(dem_clipped),
    str(slope_file),
    "-compute_edges",
    "-co", "COMPRESS=LZW"
]

result = subprocess.run(cmd_slope, capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✓ Slope calculado: {slope_file}")
else:
    print(f"  ✗ Error calculando slope: {result.stderr}")

# ============================================================================
# 4. CALCULAR ASPECT (ORIENTACIÓN)
# ============================================================================
print("\n[4/6] Calculando aspect (orientación)...")
aspect_file = DATA_PROCESSED / "aspect_isla_pascua.tif"

cmd_aspect = [
    "gdaldem", "aspect",
    str(dem_clipped),
    str(aspect_file),
    "-compute_edges",
    "-zero_for_flat",
    "-co", "COMPRESS=LZW"
]

result = subprocess.run(cmd_aspect, capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✓ Aspect calculado: {aspect_file}")
else:
    print(f"  ✗ Error calculando aspect: {result.stderr}")

# ============================================================================
# 5. CALCULAR HILLSHADE (SOMBREADO)
# ============================================================================
print("\n[5/6] Calculando hillshade (sombreado)...")
hillshade_file = DATA_PROCESSED / "hillshade_isla_pascua.tif"

cmd_hillshade = [
    "gdaldem", "hillshade",
    str(dem_clipped),
    str(hillshade_file),
    "-z", "2",  # Exageración vertical
    "-az", "315",  # Azimut de la luz (noroeste)
    "-alt", "45",  # Altitud de la luz
    "-compute_edges",
    "-co", "COMPRESS=LZW"
]

result = subprocess.run(cmd_hillshade, capture_output=True, text=True)
if result.returncode == 0:
    print(f"  ✓ Hillshade calculado: {hillshade_file}")
else:
    print(f"  ✗ Error calculando hillshade: {result.stderr}")

# ============================================================================
# 6. EXTRAER ESTADÍSTICAS
# ============================================================================
print("\n[6/6] Extrayendo estadísticas del DEM...")

with rasterio.open(dem_clipped) as src:
    dem_data = src.read(1, masked=True)
    
    stats = {
        'min_elevation': float(dem_data.min()),
        'max_elevation': float(dem_data.max()),
        'mean_elevation': float(dem_data.mean()),
        'std_elevation': float(dem_data.std()),
        'resolution': src.res[0],
        'crs': str(src.crs),
        'bounds': src.bounds,
        'shape': dem_data.shape
    }

print("\n" + "="*70)
print("ESTADÍSTICAS DEL DEM")
print("="*70)
print(f"  Elevación mínima:  {stats['min_elevation']:.2f} m")
print(f"  Elevación máxima:  {stats['max_elevation']:.2f} m")
print(f"  Elevación promedio: {stats['mean_elevation']:.2f} m")
print(f"  Desviación estándar: {stats['std_elevation']:.2f} m")
print(f"  Resolución:        {stats['resolution']} m")
print(f"  Sistema de coordenadas: {stats['crs']}")
print(f"  Dimensiones:       {stats['shape']}")

# ============================================================================
# RESUMEN DE ARCHIVOS GENERADOS
# ============================================================================
print("\n" + "="*70)
print("ARCHIVOS GENERADOS")
print("="*70)

files_generated = [
    ("DEM recortado", dem_clipped),
    ("Slope (pendiente)", slope_file),
    ("Aspect (orientación)", aspect_file),
    ("Hillshade (sombreado)", hillshade_file)
]

for name, filepath in files_generated:
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name:25s} {filepath.name:35s} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {name:25s} NO GENERADO")

print("\n" + "="*70)
print("PROCESAMIENTO COMPLETADO")
print("="*70)
print("\nPróximos pasos:")
print("  1. Ejecutar script de extracción de valores para grilla")
print("  2. Integrar variables topográficas en modelos ML")
print("  3. Actualizar visualizaciones con hillshade")
