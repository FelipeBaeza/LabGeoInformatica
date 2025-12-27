# API Reference - Sistema de Análisis Territorial

## Descripción General

La API REST proporciona acceso programático a los datos geoespaciales y modelos
predictivos del Sistema de Análisis Territorial de Isla de Pascua.

**URL Base:** `http://localhost:8000`

**Documentación interactiva:** 
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

---

## Autenticación

Actualmente la API no requiere autenticación para uso local.

---

## Endpoints

### GET /

**Descripción:** Información general de la API.

**Respuesta:**
```json
{
  "name": "API Análisis Territorial - Isla de Pascua",
  "version": "1.0.0",
  "endpoints": {
    "tables": "/api/tables",
    "data": "/api/data/{table_name}",
    "stats": "/api/stats/{table_name}",
    "predict": "/api/predict (POST)",
    "docs": "/api/docs"
  }
}
```

---

### GET /api/tables

**Descripción:** Lista todas las tablas disponibles en el schema `geoanalisis`.

**Respuesta:**
```json
[
  {
    "name": "area_construcciones",
    "schema": "geoanalisis",
    "row_count": 4045,
    "geometry_type": "MULTIPOLYGON"
  },
  {
    "name": "linea_calles",
    "schema": "geoanalisis",
    "row_count": 4139,
    "geometry_type": "LINESTRING"
  }
]
```

---

### GET /api/data/{table_name}

**Descripción:** Obtiene datos de una tabla específica.

**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `table_name` | string | requerido | Nombre de la tabla |
| `limit` | int | 100 | Máximo de registros (máx 1000) |
| `format` | string | "geojson" | Formato: "geojson" o "json" |

**Ejemplo:**
```
GET /api/data/area_construcciones?limit=10&format=geojson
```

**Respuesta (GeoJSON):**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "MultiPolygon",
        "coordinates": [...]
      },
      "properties": {
        "id": 1,
        "building": "yes",
        "name": null
      }
    }
  ]
}
```

---

### GET /api/stats/{table_name}

**Descripción:** Obtiene estadísticas de una tabla.

**Parámetros:**

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `table_name` | string | Nombre de la tabla |

**Respuesta:**
```json
{
  "table": "area_construcciones",
  "row_count": 4045,
  "columns": ["id", "building", "geometry"],
  "geometry_type": "MultiPolygon",
  "bounds": {
    "minx": -109.5,
    "miny": -27.2,
    "maxx": -109.2,
    "maxy": -27.0
  }
}
```

---

### POST /api/predict

**Descripción:** Realiza una predicción de densidad para una ubicación geográfica.

**Body (JSON):**
```json
{
  "x": -109.4295,
  "y": -27.1499,
  "features": {}
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `x` | float | Longitud (WGS84) |
| `y` | float | Latitud (WGS84) |
| `features` | object | Features adicionales (opcional) |

**Respuesta:**
```json
{
  "x": -109.4295,
  "y": -27.1499,
  "prediction": 85.5,
  "prediction_class": "Alta densidad",
  "confidence": 0.85
}
```

**Clases de predicción:**
- `Alta densidad`: prediction > 50
- `Media densidad`: prediction > 20
- `Baja densidad`: prediction > 5
- `Sin urbanización`: prediction <= 5

---

### GET /api/health

**Descripción:** Verifica el estado de la API y conexión a la base de datos.

**Respuesta:**
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

---

## Ejemplos de Uso

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# Listar tablas
response = requests.get(f"{BASE_URL}/api/tables")
tables = response.json()
print(f"Tablas disponibles: {len(tables)}")

# Obtener datos
response = requests.get(
    f"{BASE_URL}/api/data/area_construcciones",
    params={"limit": 50, "format": "geojson"}
)
geojson = response.json()
print(f"Features: {len(geojson['features'])}")

# Predicción
response = requests.post(
    f"{BASE_URL}/api/predict",
    json={"x": -109.4295, "y": -27.1499}
)
prediction = response.json()
print(f"Predicción: {prediction['prediction_class']}")
```

### cURL

```bash
# Listar tablas
curl http://localhost:8000/api/tables

# Obtener datos
curl "http://localhost:8000/api/data/area_construcciones?limit=10"

# Estadísticas
curl http://localhost:8000/api/stats/area_construcciones

# Predicción
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"x": -109.4295, "y": -27.1499}'

# Health check
curl http://localhost:8000/api/health
```

### JavaScript (fetch)

```javascript
// Listar tablas
const tables = await fetch('http://localhost:8000/api/tables')
  .then(res => res.json());

// Obtener datos
const data = await fetch('http://localhost:8000/api/data/area_construcciones?limit=100')
  .then(res => res.json());

// Predicción
const prediction = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({x: -109.4295, y: -27.1499})
}).then(res => res.json());
```

---

## Ejecución de la API

### Con Docker

```bash
# La API se ejecuta como parte del stack Docker
docker-compose up -d
# API disponible en http://localhost:8000
```

### Desarrollo Local

```bash
cd proyecto/app
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Códigos de Error

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 404 | Tabla no encontrada |
| 422 | Error de validación |
| 500 | Error interno del servidor |

---

## Notas

- Los datos GeoJSON usan CRS WGS84 (EPSG:4326)
- El límite máximo de registros es 1000
- La API usa FastAPI con documentación automática
