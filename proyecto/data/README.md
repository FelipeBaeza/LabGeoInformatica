# Data README

Este archivo describe los datos del proyecto y cómo obtener los datasets faltantes necesarios para completar el laboratorio.

Estructura esperada (local):

- proyecto/data/raw/    # datos crudos descargados (shapefiles, GeoTIFF, CSV)
- proyecto/data/processed/  # datos recortados y procesados (GeoPackage, GeoJSON, TIFF recortados)

Datasets que ya están en el repo (vectoriales):
- `Datos GeoJSON/manzana_metadato.geojson` -> manzanas censales (atributos censales)
- `Datos GeoJSON/mapa_comuna.geojson` -> límite de comuna
- `Datos GeoJSON/linea_calles.geojson` -> red vial (vector)
- `Datos GeoJSON/*` -> varias capas de POI y áreas (construcciones, playas, reservas, transporte, etc.)

Datasets faltantes (necesarios y cómo obtenerlos):
1. DEM (SRTM / ALOS)
   - Por qué: calcular elevación, slope y aspect
   - Fuentes: USGS EarthExplorer, AWS public datasets (SRTM), CGIAR-CSI
   - Comando ejemplo (recorte):
     gdalwarp -of GTiff -cutline proyecto/data/processed/manzanas_isla_de_pascua.geojson -crop_to_cutline input_srtm.tif proyecto/data/processed/dem_isla_de_pascua.tif

2. Sentinel-2 (o Landsat) para NDVI
   - Por qué: índices espectrales para modelos
   - Fuentes: Google Earth Engine (recomendado), Copernicus SciHub
   - Nota: GEE es la opción recomendada para composites y filtrado por nubes.

3. Tabla INE oficial (CSV / DBF) de manzanas (Censo 2017)
   - Por qué: verificación y trazabilidad de variables censales
   - Fuentes: https://www.ine.cl/ (Censo 2017) o IDE Chile

4. Metadatos y licencias
   - Agregar un `proyecto/docs/data_dictionary.md` con la documentación de variables.

Archivos útiles en `proyecto/scripts/`:
- `generate_unmatched_manzanas.py` - script que compara CSV INE con GeoJSON de manzanas y genera `proyecto/outputs/unmatched_manzanas.csv`.

Si quieres que descargue automáticamente la red vial con OSMnx o prepare un script de GEE para NDVI, dime y lo creo.
