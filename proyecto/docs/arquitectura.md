# Arquitectura del Sistema

## Descripción General

El Sistema de Análisis Territorial Integral de Isla de Pascua es una aplicación geoespacial completa que integra múltiples servicios containerizados para procesamiento, almacenamiento y visualización de datos espaciales.

---

## Diagrama de Arquitectura

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           USUARIO                                          │
│                    ┌─────────────────┐                                     │
│                    │   Navegador     │                                     │
│                    └────────┬────────┘                                     │
└─────────────────────────────┼─────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE PRESENTACIÓN                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              STREAMLIT (geo_streamlit:8501)                          │    │
│  │  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐            │    │
│  │  │  Inicio  │ │   Análisis   │ │ Hot Spots│ │   ML     │            │    │
│  │  └──────────┘ └──────────────┘ └──────────┘ └──────────┘            │    │
│  │  ┌──────────┐ ┌──────────────┐                                      │    │
│  │  │ Modelos  │ │  Descargas   │                                      │    │
│  │  └──────────┘ └──────────────┘                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE DESARROLLO                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              JUPYTER LAB (geo_jupyter:8888)                          │    │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐     │    │
│  │  │ 01_ESDA.ipynb    │ │ 02_Hotspots.ipynb│ │ 03_ML.ipynb      │     │    │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘     │    │
│  │  ┌──────────────────┐ ┌──────────────────┐                          │    │
│  │  │ 04_Geostats.ipynb│ │ 05_Sintesis.ipynb│                          │    │
│  │  └──────────────────┘ └──────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CAPA DE DATOS                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              POSTGIS (geo_postgis:5432)                              │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │                  Schema: geoanalisis                            │ │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │ │    │
│  │  │  │ buildings  │ │  streets   │ │ amenities  │ │  boundary  │  │ │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │ │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                 │ │    │
│  │  │  │green_areas │ │ transport  │ │   ndvi     │                 │ │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘                 │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VOLÚMENES PERSISTENTES                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ data/raw/   │  │data/processed│  │  outputs/   │  │  notebooks/ │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Componentes del Sistema

### 1. PostGIS (geo_postgis)

**Propósito**: Base de datos espacial para almacenamiento y consultas geoespaciales.

| Propiedad | Valor |
|-----------|-------|
| Imagen | `postgis/postgis:15-3.3` |
| Puerto externo | 55432 |
| Puerto interno | 5432 |
| Base de datos | `geodatabase` |
| Usuario | `geouser` |
| Schema | `geoanalisis` |

**Extensiones habilitadas**:
- `postgis`
- `postgis_topology`
- `postgis_raster`
- `fuzzystrmatch`
- `postgis_tiger_geocoder`

### 2. Jupyter Lab (geo_jupyter)

**Propósito**: Entorno de desarrollo para análisis exploratorio y modelado.

| Propiedad | Valor |
|-----------|-------|
| Puerto | 8888 |
| Token | Configurado en `.env` |
| Kernel | Python 3.11 |

**Bibliotecas principales**:
- GeoPandas, Shapely, Fiona
- PySAL, ESDA, splot
- scikit-learn, XGBoost
- Matplotlib, Seaborn, Plotly

### 3. Streamlit (geo_streamlit)

**Propósito**: Aplicación web interactiva para visualización de resultados.

| Propiedad | Valor |
|-----------|-------|
| Puerto | 8501 |
| Framework | Streamlit 1.28+ |

**Páginas**:
1. `main.py` - Dashboard principal
2. `01_Analisis_Exploratorio.py` - ESDA
3. `02_Hot_Spots.py` - Análisis de clusters
4. `03_Machine_Learning.py` - Predicciones ML
5. `04_Modelos_ML.py` - Dashboard de modelos
6. `05_Descargas.py` - Centro de descargas

---

## Flujo de Datos

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OSM / STAC    │────▶│  download_data  │────▶│    data/raw/    │
│ (Fuentes)       │     │     .py         │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │  prepare_data   │────▶│ data/processed/ │
│                 │     │     .py         │     │                 │
│                 │     └─────────────────┘     └────────┬────────┘
│                 │                                      │
│   PostGIS       │◀─────────────────────────────────────┘
│   Database      │
│                 │     ┌─────────────────┐     ┌─────────────────┐
│                 │────▶│   Notebooks     │────▶│    outputs/     │
│                 │     │  (análisis)     │     │  figures/       │
└─────────────────┘     └─────────────────┘     │  models/        │
                                                │  reports/       │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Streamlit     │
                                                │   (Dashboard)   │
                                                └─────────────────┘
```

---

## Módulos Python

### scripts/spatial_analysis.py

Funciones para análisis espacial:
- `create_queen_weights()` - Matriz de pesos Queen
- `create_distance_weights()` - Matriz de distancias
- `morans_i()` - Índice de Moran global
- `local_morans_i()` - LISA (Local Moran)
- `getis_ord_gi_star()` - Hot spot analysis

### scripts/geostatistics.py

Funciones para geoestadística:
- `empirical_semivariogram()` - Semivariograma experimental
- `fit_variogram_model()` - Ajuste de modelo teórico
- `ordinary_kriging()` - Interpolación Kriging
- `idw_interpolation()` - Interpolación IDW
- `cross_validation_loo()` - Validación cruzada

---

## Configuración Docker

### docker-compose.yml

```yaml
services:
  postgis:
    image: postgis/postgis:15-3.3
    ports: "55432:5432"
    
  jupyter:
    build: ./docker/jupyter
    ports: "8888:8888"
    depends_on: postgis
    
  streamlit:
    build: ./docker/web
    ports: "8501:8501"
    depends_on: postgis
```

---

## Seguridad

- Credenciales en archivo `.env` (no versionado)
- PostGIS accesible solo localmente (puerto 55432)
- Variables de entorno para configuración sensible

---

## Escalabilidad

El sistema está diseñado para ser escalable:

1. **Horizontal**: Agregar más workers de Streamlit
2. **Vertical**: Aumentar recursos de contenedores
3. **Datos**: PostGIS soporta terabytes de datos espaciales

---

## Monitoreo

```bash
# Estado de contenedores
docker-compose ps

# Logs en tiempo real
docker-compose logs -f

# Recursos
docker stats
```
