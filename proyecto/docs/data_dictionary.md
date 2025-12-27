# Data Dictionary (plantilla)

Este documento debe completarse con el diccionario de variables para cada dataset.

Formato recomendado:
- dataset: nombre del archivo (ruta)
- field: nombre del campo
- type: tipo (integer, float, string, boolean)
- description: breve descripción
- source: origen (INE, OSM, Copernicus, etc.)
- notes: observaciones (valores nulos, codificación especial)

Ejemplo (manzanas):

Dataset: `proyecto/data/processed/manzanas_isla_de_pascua.geojson`
- MANZENT (string): identificador único de la manzana (INE) - source: INE
- MANZANA (integer): número de manzana dentro del distrito - source: INE
- TOTAL_PERS (integer): población total - source: INE, Censo 2017
- TOTAL_HOMB (integer): hombres - source: INE
- TOTAL_MUJE (integer): mujeres - source: INE
- VIV_OCUPA_ (integer): viviendas ocupadas - source: INE
- VIV_AGUA_R (integer): viviendas con agua - source: INE
- PUEBLOS_IN (integer): población indígena - source: INE

Acción pendiente:
- Completar este archivo extrayendo campos desde `Datos GeoJSON/manzana_metadato.geojson` o desde la tabla INE oficial.
