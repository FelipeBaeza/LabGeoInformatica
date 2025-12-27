#!/usr/bin/env bash
set -euo pipefail

# prepare_data.sh
# Reproyecta y prepara los GeoJSON listados hacia EPSG:32712 usando ogr2ogr
# Crea un GeoPackage en proyecto/data/processed/prepared.gpkg
# Genera un resumen JSON en proyecto/outputs/geojson_processing_summary.json

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$ROOT/Datos GeoJSON"
OUT_DIR="$ROOT/proyecto/data/processed"
OUT_DIR_PLAIN="$ROOT/proyecto/data/processed"
GPKG="$OUT_DIR/prepared.gpkg"
SUMMARY="$ROOT/proyecto/outputs/geojson_processing_summary.json"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$SUMMARY")"

FILES=(
  "manzanas_isla_de_pascua.geojson"
  "limite_administrativa.geojson"
  "linea_calles.geojson"
  "mapa_comuna.geojson"
  "punto_interes.geojson"
)

TARGET_EPSG=32712

TMP_ENTRIES="$(mktemp)"
echo "" > "$TMP_ENTRIES"
first=true
first_layer=true

for f in "${FILES[@]}"; do
  path="$DATA_DIR/$f"
  info_exists=false
  original_srs=null
  assigned_srs=null
  reprojected=false

  if [ -f "$path" ]; then
    info_exists=true
    # intentar detectar SRS desde ogrinfo
    ogr_out=$(ogrinfo -ro -so "$path" 2>&1 || true)
    if echo "$ogr_out" | grep -q 'AUTHORITY\["EPSG"'; then
      epsg=$(echo "$ogr_out" | grep -oP 'AUTHORITY\["EPSG"\s*,\s*"\K[0-9]+' || true)
      if [ -n "$epsg" ]; then
        original_srs="EPSG:$epsg"
      fi
    fi

    # si no hay SRS, usar heurística por Extent o, si falla, extraer una muestra directa
    if [ "$original_srs" = null ]; then
      extent_line=$(echo "$ogr_out" | grep -i 'Extent' || true)
      if [ -n "$extent_line" ]; then
        # extraer números: (minx,miny) - (maxx,maxy)
        nums=$(echo "$extent_line" | grep -oP '\([-0-9\.]+,\s*[-0-9\.]+\)\s*-\s*\([-0-9\.]+,\s*[-0-9\.]+\)' || true)
        if [ -n "$nums" ]; then
          # strip parentheses
          coords=$(echo "$nums" | tr -d '()')
          minx=$(echo "$coords" | awk -F' - ' '{print $1}' | awk -F',' '{print $1}')
          miny=$(echo "$coords" | awk -F' - ' '{print $1}' | awk -F',' '{print $2}')
          maxx=$(echo "$coords" | awk -F' - ' '{print $2}' | awk -F',' '{print $1}')
          maxy=$(echo "$coords" | awk -F' - ' '{print $2}' | awk -F',' '{print $2}')
        fi
      fi

      # si no obtuvimos extent numérico válido, intentar leer una muestra desde el GeoJSON
    if [ -z "${minx:-}" ] || [ -z "${miny:-}" ] || [ -z "${maxx:-}" ] || [ -z "${maxy:-}" ]; then
    # extraer primera coordenada disponible usando grep (formato [x,y])
    sample_coord=$(grep -oP '\[\s*-?[0-9]+(?:\.[0-9]+)?\s*,\s*-?[0-9]+(?:\.[0-9]+)?\s*\]' "$path" | head -n1 | tr -d '[]' | tr -s ' ' | sed 's/,/ /') || true
    if [ -n "$sample_coord" ]; then
      # split into x/y
      minx=$(echo $sample_coord | awk '{print $1}')
      miny=$(echo $sample_coord | awk '{print $2}')
      maxx=$minx
      maxy=$miny
    fi
    fi

      # decide SRS usando reglas heurísticas basadas en magnitudes
      if [ -n "${minx:-}" ] && awk "BEGIN{print ($minx >= -180 && $maxx <= 180 && $miny >= -90 && $maxy <= 90)}" | grep -q 1; then
        assigned_srs="EPSG:4326"
      elif [ -n "${minx:-}" ] && (awk "BEGIN{print (sqrt(($minx*$minx)) > 10000000 || sqrt(($miny*$miny)) > 10000000)}" | grep -q 1); then
        # coordenadas de gran magnitud -> probable WebMercator
        assigned_srs="EPSG:3857"
      elif [ -n "${minx:-}" ] && awk "BEGIN{print ($minx > -1000000 && $maxx < 10000000 && $maxy > 1000000 && $maxy < 10000000)}" | grep -q 1; then
        # UTM-like
        assigned_srs="EPSG:32712"
      else
        assigned_srs="EPSG:3857"
      fi
    fi

    # construir comando ogr2ogr
    layer_name=$(basename "$f" .geojson)
    # si assigned_srs está presente, usamos -a_srs para asignarlo primero
    if [ "$assigned_srs" != null ]; then
      echo "Assigning $assigned_srs to $f and reprojecting to EPSG:$TARGET_EPSG"
      # use -s_srs to specify source SRS explicitly
      if [ "$first_layer" = true ]; then
        ogr2ogr -f GPKG "$GPKG" "$path" -nln "$layer_name" -s_srs "$assigned_srs" -t_srs "EPSG:$TARGET_EPSG" -overwrite -skipfailures
        first_layer=false
      else
        ogr2ogr -f GPKG "$GPKG" "$path" -nln "$layer_name" -s_srs "$assigned_srs" -t_srs "EPSG:$TARGET_EPSG" -append -skipfailures
      fi
      reprojected=true
    else
      # si original_srs detectado, simplemente reproyectar
      echo "Reprojecting $f (detected $original_srs) to EPSG:$TARGET_EPSG"
      if [ "$first_layer" = true ]; then
        ogr2ogr -f GPKG "$GPKG" "$path" -nln "$layer_name" -t_srs "EPSG:$TARGET_EPSG" -overwrite -skipfailures
        first_layer=false
      else
        ogr2ogr -f GPKG "$GPKG" "$path" -nln "$layer_name" -t_srs "EPSG:$TARGET_EPSG" -append -skipfailures
      fi
      reprojected=true
    fi
  fi

  # agregar entrada al JSON
  if [ "$first" = true ]; then
    first=false
  else
    echo ',' >> "$SUMMARY"
  fi

  # Build JSON entry safely in bash
  j_orig=null
  j_assigned=null
  if [ "$original_srs" != null ]; then
    j_orig="\"$original_srs\""
  fi
  if [ "$assigned_srs" != null ]; then
    j_assigned="\"$assigned_srs\""
  fi

  echo "{\"file\": \"$path\", \"exists\": $info_exists, \"original_srs\": $j_orig, \"assigned_srs\": $j_assigned, \"reprojected\": $reprojected, \"layer\": \"$GPKG:$layer_name\" }" >> "$TMP_ENTRIES"

done

# Construir JSON final usando las entradas temporales
echo -n "{" > "$SUMMARY"
echo -n "\"target_epsg\":$TARGET_EPSG, \"files\":[" >> "$SUMMARY"

first_out=true
while IFS= read -r line; do
  if [ -z "$line" ]; then
    continue
  fi
  if [ "$first_out" = true ]; then
    echo -n "$line" >> "$SUMMARY"
    first_out=false
  else
    echo -n ",${line}" >> "$SUMMARY"
  fi
done < "$TMP_ENTRIES"

echo -n "]}" >> "$SUMMARY"

rm -f "$TMP_ENTRIES"

echo "Prepared GeoPackage: $GPKG"
echo "Summary: $SUMMARY"

exit 0
