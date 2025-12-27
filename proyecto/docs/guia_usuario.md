# Guia de Usuario - Sistema de Analisis Territorial

## Introduccion

Esta aplicacion web permite explorar y analizar datos geoespaciales de la Isla de Pascua
de forma interactiva.

## Acceso

Navegar a: **http://localhost:8501**

## Paginas Disponibles

### Inicio
Resumen general del proyecto con estadisticas clave y mapa de ubicacion.

### Visor de Mapas
- Seleccionar capas de datos (edificios, calles, amenidades)
- Cambiar estilos de visualizacion
- Zoom y navegacion interactiva

### Analisis Exploratorio
- Ver estadisticas descriptivas por variable
- Histogramas y boxplots interactivos
- Analisis de autocorrelacion espacial

### Hot Spots
- Visualizacion de clusters espaciales
- Mapa de Getis-Ord Gi*
- Identificacion de zonas de alta/baja densidad

### Machine Learning
- Entrenar modelos predictivos
- Visualizacion de predicciones en mapa
- Importancia de variables

### Modelos ML
- Seleccion de modelo predictivo
- Metricas de rendimiento
- Validacion espacial

### Descargas
- Exportar datos en GeoJSON/CSV
- Descargar mapas en PNG
- Obtener reportes en PDF

## Solucion de Problemas

**Error de conexion a la base de datos:**
```bash
docker-compose restart postgis
```

**Graficos no cargan:**
- Refrescar la pagina (F5)
- Verificar que PostGIS este corriendo

## Contacto
Para reportar errores: francisco.parra.o@usach.cl
