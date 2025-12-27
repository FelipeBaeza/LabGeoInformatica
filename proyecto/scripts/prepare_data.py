#!/usr/bin/env python3
"""prepare_data.py

Normaliza y prepara los GeoJSON del proyecto para análisis:
- Detecta y asigna CRS si falta (usa heurística para manzanas)
- Reproyecta a EPSG:32712 (UTM zone 12S)
- Valida y arregla geometrías
- Exporta capas a un GeoPackage en proyecto/data/processed/prepared.gpkg
- Genera un resumen JSON en proyecto/outputs/geojson_processing_summary.json

Uso: python proyecto/scripts/prepare_data.py
"""
import json
from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'Datos GeoJSON'
OUT_DIR = ROOT / 'proyecto' / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)
GPKG = OUT_DIR / 'prepared.gpkg'
SUMMARY = ROOT / 'proyecto' / 'outputs' / 'geojson_processing_summary.json'

FILES = [
    'manzanas_isla_de_pascua.geojson',
    'limite_administrativa.geojson',
    'linea_calles.geojson',
    'mapa_comuna.geojson',
    'punto_interes.geojson',
]

TARGET_EPSG = 32712


def guess_and_set_crs(gdf, filename):
    """Si gdf.crs es None, intentar una heurística para asignar CRS.
    Para `manzanas_isla_de_pascua.geojson` asumimos EPSG:3857 si las
    coordenadas parecen grandes (magnitud > 1e6)."""
    if gdf.crs is not None:
        return gdf, str(gdf.crs)

    # Heurística: mirar primer punto
    try:
        geom = gdf.geometry.values[0]
        if geom is None:
            return gdf, None
        coords = list(geom.representative_point().coords)[0]
        x, y = float(coords[0]), float(coords[1])
        # si las coordenadas están en rango de WebMercator/metros grandes
        if abs(x) > 1e6 or abs(y) > 1e6:
            gdf = gdf.set_crs(epsg=3857, allow_override=True)
            return gdf, 'EPSG:3857'
        else:
            # probable lon/lat
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
            return gdf, 'EPSG:4326'
    except Exception:
        return gdf, None


def process_file(fname):
    path = DATA_DIR / fname
    result = {
        'file': str(path),
        'exists': path.exists(),
        'original_crs': None,
        'assigned_crs': None,
        'reprojected': False,
        'n_features': 0,
        'n_invalid': 0,
        'errors': None,
    }
    if not path.exists():
        return result

    try:
        gdf = gpd.read_file(path)
        result['n_features'] = len(gdf)
        result['original_crs'] = str(gdf.crs) if gdf.crs else None

        if gdf.crs is None:
            gdf, assigned = guess_and_set_crs(gdf, fname)
            result['assigned_crs'] = assigned

        # Reproject to target
        if gdf.crs is None:
            # fallback: assume EPSG:4326
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
            result['assigned_crs'] = result.get('assigned_crs') or 'EPSG:4326'

        if gdf.crs.to_epsg() != TARGET_EPSG:
            gdf = gdf.to_crs(epsg=TARGET_EPSG)
            result['reprojected'] = True

        # validar geometrías
        invalid_mask = ~gdf.is_valid
        n_invalid = int(invalid_mask.sum())
        result['n_invalid'] = n_invalid
        if n_invalid > 0:
            # intentar reparar con buffer(0)
            gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
            # recalcular
            invalid_mask2 = ~gdf.is_valid
            result['n_invalid_after_fix'] = int(invalid_mask2.sum())

        # exportar a GeoPackage (una capa por archivo base name)
        layer_name = Path(fname).stem
        gdf.to_file(GPKG, layer=layer_name, driver='GPKG')
        result['layer'] = str(GPKG) + ':' + layer_name
    except Exception as e:
        result['errors'] = repr(e)

    return result


def main():
    summary = {
        'target_epsg': TARGET_EPSG,
        'files': []
    }

    for f in FILES:
        print('Processing', f)
        r = process_file(f)
        summary['files'].append(r)

    # guardar resumen
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print('Wrote summary to', SUMMARY)


if __name__ == '__main__':
    main()
