#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API REST para el Sistema de Análisis Territorial.

Endpoints:
- GET /api/tables - Lista tablas disponibles
- GET /api/data/{table} - Datos de una tabla (GeoJSON)
- GET /api/stats/{table} - Estadísticas de una tabla
- POST /api/predict - Predicción ML simple

Uso:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine, text

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

app = FastAPI(
    title="API Análisis Territorial - Isla de Pascua",
    description="API REST para acceso a datos geoespaciales y modelos predictivos",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS para permitir acceso desde Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de base de datos
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '55432'),
    'database': os.getenv('POSTGRES_DB', 'geodatabase'),
    'user': os.getenv('POSTGRES_USER', 'geouser'),
    'password': os.getenv('POSTGRES_PASSWORD', 'geopass123'),
}


def get_engine():
    """Crear conexión a la base de datos."""
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class TableInfo(BaseModel):
    name: str
    schema: str
    row_count: int
    geometry_type: Optional[str] = None


class StatsResponse(BaseModel):
    table: str
    row_count: int
    columns: List[str]
    geometry_type: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None


class PredictionRequest(BaseModel):
    x: float  # Coordenada X (longitud)
    y: float  # Coordenada Y (latitud)
    features: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    x: float
    y: float
    prediction: float
    prediction_class: str
    confidence: Optional[float] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Endpoint raíz con información de la API."""
    return {
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


@app.get("/api/tables", response_model=List[TableInfo])
def list_tables():
    """
    Lista todas las tablas disponibles en el schema geoanalisis.
    
    Returns:
        Lista de tablas con nombre, schema y conteo de filas.
    """
    try:
        engine = get_engine()
        
        query = text("""
            SELECT 
                table_name,
                table_schema
            FROM information_schema.tables 
            WHERE table_schema = 'geoanalisis'
            ORDER BY table_name
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            tables = []
            
            for row in result:
                table_name = row[0]
                schema = row[1]
                
                # Obtener conteo de filas
                count_query = text(f'SELECT COUNT(*) FROM geoanalisis."{table_name}"')
                count_result = conn.execute(count_query)
                row_count = count_result.scalar()
                
                # Intentar obtener tipo de geometría
                geom_type = None
                try:
                    geom_query = text(f"""
                        SELECT GeometryType(geometry) 
                        FROM geoanalisis."{table_name}" 
                        WHERE geometry IS NOT NULL 
                        LIMIT 1
                    """)
                    geom_result = conn.execute(geom_query)
                    geom_row = geom_result.fetchone()
                    if geom_row:
                        geom_type = geom_row[0]
                except:
                    pass
                
                tables.append(TableInfo(
                    name=table_name,
                    schema=schema,
                    row_count=row_count,
                    geometry_type=geom_type
                ))
            
            return tables
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(e)}")


@app.get("/api/data/{table_name}")
def get_table_data(
    table_name: str,
    limit: int = Query(default=100, le=1000, description="Número máximo de registros"),
    format: str = Query(default="geojson", description="Formato: geojson o json")
):
    """
    Obtiene datos de una tabla en formato GeoJSON o JSON.
    
    Args:
        table_name: Nombre de la tabla
        limit: Límite de registros (max 1000)
        format: Formato de salida (geojson o json)
    
    Returns:
        Datos de la tabla en el formato especificado.
    """
    try:
        engine = get_engine()
        
        # Cargar datos con GeoPandas
        query = f'SELECT * FROM geoanalisis."{table_name}" LIMIT {limit}'
        gdf = gpd.read_postgis(query, engine, geom_col='geometry')
        
        if format == "geojson":
            # Convertir a GeoJSON
            return json.loads(gdf.to_json())
        else:
            # Convertir a JSON simple (sin geometría compleja)
            df = pd.DataFrame(gdf.drop(columns=['geometry']))
            df['centroid_x'] = gdf.geometry.centroid.x
            df['centroid_y'] = gdf.geometry.centroid.y
            return df.to_dict(orient='records')
            
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Tabla no encontrada o error: {str(e)}")


@app.get("/api/stats/{table_name}", response_model=StatsResponse)
def get_table_stats(table_name: str):
    """
    Obtiene estadísticas de una tabla.
    
    Args:
        table_name: Nombre de la tabla
    
    Returns:
        Estadísticas incluyendo conteo, columnas, tipo de geometría y bounds.
    """
    try:
        engine = get_engine()
        
        # Cargar muestra de datos
        query = f'SELECT * FROM geoanalisis."{table_name}" LIMIT 1000'
        gdf = gpd.read_postgis(query, engine, geom_col='geometry')
        
        # Obtener conteo total
        with engine.connect() as conn:
            count_query = text(f'SELECT COUNT(*) FROM geoanalisis."{table_name}"')
            total_count = conn.execute(count_query).scalar()
        
        # Calcular bounds
        bounds = None
        if len(gdf) > 0:
            total_bounds = gdf.total_bounds
            bounds = {
                "minx": float(total_bounds[0]),
                "miny": float(total_bounds[1]),
                "maxx": float(total_bounds[2]),
                "maxy": float(total_bounds[3])
            }
        
        # Tipo de geometría
        geom_type = None
        if len(gdf) > 0:
            geom_type = gdf.geometry.geom_type.iloc[0]
        
        return StatsResponse(
            table=table_name,
            row_count=total_count,
            columns=list(gdf.columns),
            geometry_type=geom_type,
            bounds=bounds
        )
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Tabla no encontrada o error: {str(e)}")


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Realiza una predicción de densidad para una ubicación.
    
    Este endpoint usa un modelo simplificado basado en distancia al centro
    de Hanga Roa para predecir la densidad de edificaciones.
    
    Args:
        request: Coordenadas (x, y) en WGS84
    
    Returns:
        Predicción de densidad y clasificación.
    """
    try:
        # Centro aproximado de Hanga Roa
        CENTER_X = -109.4295
        CENTER_Y = -27.1499
        
        # Calcular distancia al centro (en grados, aprox)
        dist = np.sqrt((request.x - CENTER_X)**2 + (request.y - CENTER_Y)**2)
        
        # Modelo simple: densidad inversamente proporcional a distancia
        # D = max_density * exp(-k * dist)
        max_density = 100
        k = 50  # Factor de decaimiento
        prediction = max_density * np.exp(-k * dist)
        
        # Clasificación
        if prediction > 50:
            pred_class = "Alta densidad"
            confidence = 0.85
        elif prediction > 20:
            pred_class = "Media densidad"
            confidence = 0.70
        elif prediction > 5:
            pred_class = "Baja densidad"
            confidence = 0.75
        else:
            pred_class = "Sin urbanización"
            confidence = 0.90
        
        return PredictionResponse(
            x=request.x,
            y=request.y,
            prediction=round(prediction, 2),
            prediction_class=pred_class,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/api/health")
def health_check():
    """Verificar estado de la API y conexión a BD."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
