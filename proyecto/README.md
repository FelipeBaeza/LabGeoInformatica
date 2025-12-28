# Laboratorio Integrador - Análisis Geoespacial de Isla de Pascua

## 📍 Descripción
Sistema de Análisis Territorial Integral de la **Isla de Pascua (Rapa Nui)**, Chile.
Este proyecto implementa un análisis geoespacial completo incluyendo ESDA, geoestadística,
machine learning espacial y una aplicación web interactiva.

## 👥 Equipo
- Felipe Baeza
- Catalina López

---

## 🚀 Despliegue Rápido

### Requisitos
- Docker y Docker Compose
- Git
- ~10GB de espacio en disco

### 1. Clonar y Configurar

```bash
# Clonar repositorio
git clone [URL_DE_TU_REPO]
cd laboratorio_geoinformatica/proyecto

# El archivo .env ya está configurado
```

### 2. Levantar Servicios

```bash
# Limpiar contenedores previos (opcional)
sudo docker compose down -v

# Construir e iniciar servicios
sudo docker compose up -d --build

# Verificar que todo esté corriendo
sudo docker compose ps
```

### 3. Cargar Datos a PostGIS

```bash
# Ejecutar desde contenedor Jupyter
sudo docker exec geo_jupyter python /home/jovyan/scripts/load_to_postgis.py \
    --source /home/jovyan/data/raw/isla_de_pascua

# Debería mostrar: 6 tablas cargadas, 8,439 registros totales
```

### 4. Verificar Servicios

| Servicio | URL | Puerto |
|----------|-----|--------|
| **Streamlit** | http://localhost:8501 | 8501 |
| **API REST** | http://localhost:8002 | 8002 |
| **API Docs** | http://localhost:8002/api/docs | 8002 |
| **Jupyter** | http://localhost:8888 | 8888 |
| **PostGIS** | localhost:55432 | 55432 |

---

## 📊 Datos Cargados

| Tabla | Registros | Geometría |
|-------|-----------|-----------|
| isla_de_pascua_buildings | 4,045 | POLYGON |
| isla_de_pascua_streets | 4,139 | LINESTRING |
| isla_de_pascua_amenities | 241 | POINT |
| isla_de_pascua_green_areas | 12 | POINT |
| isla_de_pascua_boundary | 1 | POLYGON |
| isla_de_pascua_transport | 1 | POINT |

---

## 📁 Estructura del Proyecto

```
proyecto/
├── app/                    # Aplicación web Streamlit
│   ├── main.py             # Página principal
│   ├── api.py              # API REST FastAPI
│   └── pages/              # Páginas adicionales (6)
├── data/
│   ├── raw/isla_de_pascua/ # Datos GeoJSON
│   └── processed/          # Datos procesados
├── docker/                 # Configuraciones Docker
│   ├── jupyter/Dockerfile
│   ├── postgis/init.sql
│   └── web/Dockerfile
├── docs/                   # Documentación
│   ├── arquitectura.md
│   ├── guia_usuario.md
│   └── api_reference.md
├── notebooks/              # Análisis (5 notebooks)
│   ├── 01_ESDA_Analisis_Exploratorio.ipynb
│   ├── 02_Hot_Spots_Analysis.ipynb
│   ├── 03_Machine_Learning_Espacial.ipynb
│   ├── 04_Geoestadistica.ipynb
│   └── 05_Sintesis_Resultados.ipynb
├── outputs/                # Resultados (mapas, modelos)
├── scripts/                # Scripts Python
│   ├── download_data.py
│   ├── load_to_postgis.py
│   ├── spatial_analysis.py
│   ├── geostatistics.py
│   └── network_analysis_advanced.py
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## 🔧 Comandos Útiles

```bash
# Ver logs de todos los servicios
sudo docker compose logs -f

# Ver logs de un servicio específico
sudo docker compose logs -f streamlit

# Reiniciar un servicio
sudo docker compose restart api

# Detener todos los servicios
sudo docker compose down

# Limpiar todo (incluyendo datos)
sudo docker compose down -v --remove-orphans
```

---

## 🔬 Componentes del Análisis

### 1. ESDA (Análisis Exploratorio)
- Estadísticas descriptivas
- Moran's I global y local (LISA)
- Getis-Ord Gi* para Hot Spots

### 2. Geoestadística
- Semivariogramas experimentales
- Modelos teóricos (esférico, exponencial, gaussiano)
- Kriging ordinario e IDW
- Validación cruzada

### 3. Machine Learning
- Random Forest y XGBoost
- Validación espacial (GroupKFold)
- SHAP para interpretabilidad
- Predicción de densidad urbana

### 4. Elementos de Excelencia
- ✅ Visualización 3D (PyDeck)
- ✅ Análisis de Redes (NetworkX)
- ✅ API REST (FastAPI)

---

## 📚 Referencias

- OSMnx: Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.
- Ley 21.070 (2018): Regulación de residencia en Isla de Pascua
- PostGIS: Extensión espacial para PostgreSQL

---

**Universidad de Santiago de Chile - Desarrollo de Aplicaciones Geoinformáticas - 2025**
