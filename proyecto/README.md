# Laboratorio Integrador - Análisis Geoespacial de Isla de Pascua

## 📍 Descripción
Sistema de Análisis Territorial Integral de la **Isla de Pascua (Rapa Nui)**, Chile.
Este proyecto implementa un análisis geoespacial completo incluyendo ESDA, geoestadística,
machine learning espacial y una aplicación web interactiva.

## 👥 Equipo
- Integrante 1: [Tu Nombre]
- Integrante 2: [Nombre del compañero]

## 🚀 Inicio Rápido

### Requisitos Previos
- Docker y Docker Compose instalados
- Git
- 10GB de espacio en disco

### Instalación

1. **Clonar el repositorio:**
```bash
git clone [URL_DE_TU_REPO]
cd laboratorio_geoinformatica/proyecto
```

2. **Configurar variables de entorno:**
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

3. **Levantar servicios Docker:**
```bash
docker-compose up -d
```

4. **Verificar servicios:**
```bash
docker-compose ps
# Deberías ver: postgis, jupyter, streamlit corriendo
```

### Acceso a Servicios

| Servicio | URL | Descripción |
|----------|-----|-------------|
| Jupyter Lab | http://localhost:8888 | Notebooks de análisis |
| Streamlit | http://localhost:8501 | Aplicación web |
| PostGIS | localhost:55432 | Base de datos espacial |

## 📁 Estructura del Proyecto

```
proyecto/
├── app/                    # Aplicación web Streamlit
│   ├── main.py
│   └── pages/             # Páginas adicionales
├── data/
│   ├── raw/               # Datos originales
│   └── processed/         # Datos procesados
├── docker/                # Configuraciones Docker
│   ├── jupyter/
│   ├── postgis/
│   └── web/
├── docs/                  # Documentación
├── notebooks/             # Análisis Jupyter
│   ├── 01_ESDA_Analisis_Exploratorio.ipynb
│   ├── 02_Geoestadistica_Hotspots.ipynb
│   ├── 03_Machine_Learning_Espacial.ipynb
│   ├── 04_Geoestadistica.ipynb
│   └── 05_Sintesis_Resultados.ipynb
├── outputs/
│   ├── figures/           # Mapas y gráficos
│   ├── models/            # Modelos entrenados
│   └── reports/           # Informes
├── scripts/               # Scripts Python
├── docker-compose.yml
└── requirements.txt
```

## 📊 Componentes del Análisis

1. **ESDA (Análisis Exploratorio Espacial)**
   - Estadísticas descriptivas
   - Mapas temáticos
   - Autocorrelación espacial (Moran's I, LISA)

2. **Geoestadística**
   - Semivariogramas
   - Interpolación Kriging vs IDW
   - Validación cruzada

3. **Machine Learning Espacial**
   - Feature engineering espacial
   - Random Forest, XGBoost
   - Validación espacial (GroupKFold)
   - Interpretabilidad (SHAP)

4. **Aplicación Web**
   - Mapas interactivos
   - Dashboard de estadísticas
   - Descarga de resultados

## 🔧 Uso

### Ejecutar Análisis
```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar notebooks en orden
jupyter lab notebooks/
```

### Ejecutar Aplicación Web
```bash
# Con Docker
docker-compose up streamlit

# Sin Docker
streamlit run app/main.py
```

## 📚 Fuentes de Datos

- **Límites y edificios**: OpenStreetMap (OSMnx)
- **Índices espectrales**: Sentinel-2 (STAC)
- **Variables censales**: INE Chile

## 📝 Licencia
Proyecto académico - Universidad de Santiago de Chile
