#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para cargar datos GeoJSON a PostGIS.

Uso:
    python load_to_postgis.py --comuna "La Florida"
    python load_to_postgis.py --source ../data/raw/la_florida/
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Configuración de base de datos (valores por defecto)
# Primero intenta leer de variables de entorno (útil para contenedores)
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '55432'),
    'database': os.environ.get('POSTGRES_DB', 'geodatabase'),
    'user': os.environ.get('POSTGRES_USER', 'geouser'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'geopass123')
}

# Si no hay variables de entorno, intentar cargar desde .env
if DB_CONFIG['host'] == 'localhost':
    try:
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'POSTGRES_HOST':
                            DB_CONFIG['host'] = value
                        elif key == 'POSTGRES_PORT':
                            DB_CONFIG['port'] = value
                        elif key == 'POSTGRES_DB':
                            DB_CONFIG['database'] = value
                        elif key == 'POSTGRES_USER':
                            DB_CONFIG['user'] = value
                        elif key == 'POSTGRES_PASSWORD':
                            DB_CONFIG['password'] = value
    except Exception:
        pass  # Usar valores por defecto



def get_db_engine():
    """Crear conexión a la base de datos."""
    connection_string = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    print(f"  Conectando a: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    return create_engine(connection_string)


def test_connection(engine):
    """Probar conexión a la base de datos."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT PostGIS_Version();"))
            version = result.fetchone()[0]
            print(f"✓ Conectado a PostGIS versión: {version}")
            return True
    except Exception as e:
        print(f"✗ Error de conexión: {e}")
        return False


def load_geojson_to_postgis(filepath: Path, table_name: str, engine, schema: str = 'geoanalisis'):
    """
    Cargar un archivo GeoJSON a PostGIS.
    
    Args:
        filepath: Ruta al archivo GeoJSON
        table_name: Nombre de la tabla en PostGIS
        engine: Engine de SQLAlchemy
        schema: Schema de la base de datos
    """
    try:
        # Leer GeoJSON
        gdf = gpd.read_file(filepath)
        
        if gdf.empty:
            print(f"  ⚠ Archivo vacío: {filepath.name}")
            return False
        
        # Asegurar que tenga CRS
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        
        # Reproyectar a WGS84 si es necesario
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Limpiar nombre de columnas
        gdf.columns = [c.lower().replace(' ', '_').replace('-', '_') for c in gdf.columns]
        
        # Cargar a PostGIS
        gdf.to_postgis(
            table_name,
            engine,
            schema=schema,
            if_exists='replace',
            index=True,
            index_label='gid'
        )
        
        print(f"  ✓ {table_name}: {len(gdf)} registros cargados")
        return True
        
    except Exception as e:
        print(f"  ✗ Error cargando {filepath.name}: {e}")
        return False


def load_all_geojson_from_dir(source_dir: Path, engine, schema: str = 'geoanalisis'):
    """
    Cargar todos los archivos GeoJSON de un directorio.
    
    Args:
        source_dir: Directorio con archivos GeoJSON
        engine: Engine de SQLAlchemy
        schema: Schema de la base de datos
    """
    geojson_files = list(source_dir.glob("*.geojson"))
    
    if not geojson_files:
        print(f"⚠ No se encontraron archivos GeoJSON en {source_dir}")
        return
    
    print(f"\n Cargando {len(geojson_files)} archivos desde {source_dir}")
    
    loaded = 0
    failed = 0
    
    for filepath in tqdm(geojson_files, desc="Cargando archivos"):
        # Generar nombre de tabla desde el nombre del archivo
        table_name = filepath.stem.lower().replace(' ', '_').replace('-', '_')
        
        if load_geojson_to_postgis(filepath, table_name, engine, schema):
            loaded += 1
        else:
            failed += 1
    
    print(f"\n Resumen: {loaded} cargados, {failed} fallidos")


def create_spatial_indexes(engine, schema: str = 'geoanalisis'):
    """Crear índices espaciales en todas las tablas."""
    print("\n🔧 Creando índices espaciales...")
    
    try:
        with engine.connect() as conn:
            # Obtener todas las tablas con geometría
            query = text(f"""
                SELECT table_name 
                FROM information_schema.columns 
                WHERE table_schema = '{schema}' 
                AND column_name = 'geometry'
            """)
            result = conn.execute(query)
            tables = [row[0] for row in result.fetchall()]
            
            for table in tables:
                try:
                    # Crear índice espacial
                    idx_query = text(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table}_geom 
                        ON {schema}.{table} USING GIST (geometry)
                    """)
                    conn.execute(idx_query)
                    conn.commit()
                    print(f"  ✓ Índice creado para {table}")
                except Exception as e:
                    print(f"  ⚠ Error creando índice para {table}: {e}")
        
        print("✓ Índices espaciales creados")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def list_tables(engine, schema: str = 'geoanalisis'):
    """Listar todas las tablas en el schema."""
    print(f"\n Tablas en schema '{schema}':")
    
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM {schema}."" || table_name || "") as rows
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
                ORDER BY table_name
            """)
            
            # Consulta simplificada
            query = text(f"""
                SELECT table_name
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
                ORDER BY table_name
            """)
            result = conn.execute(query)
            
            for row in result.fetchall():
                print(f"  - {row[0]}")
                
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Cargar datos GeoJSON a PostGIS'
    )
    
    parser.add_argument(
        '--comuna', '-c',
        type=str,
        help='Nombre de la comuna (busca en data/raw/{comuna}/)'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        help='Ruta específica al directorio con GeoJSON'
    )
    
    parser.add_argument(
        '--schema',
        type=str,
        default='geoanalisis',
        help='Schema de destino en PostgreSQL'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Listar tablas existentes'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║           CARGA DE DATOS A POSTGIS - GEOINFORMÁTICA          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Conectar a la base de datos
    print(" Conectando a PostGIS...")
    engine = get_db_engine()
    
    if not test_connection(engine):
        print("\n No se pudo conectar a la base de datos.")
        print("   Verifica que el contenedor Docker esté corriendo:")
        print("   docker-compose up -d postgis")
        sys.exit(1)
    
    # Solo listar tablas
    if args.list:
        list_tables(engine, args.schema)
        return
    
    # Determinar directorio fuente
    if args.source:
        source_dir = Path(args.source)
    elif args.comuna:
        source_dir = DATA_RAW_DIR / args.comuna.lower().replace(' ', '_')
    else:
        # Cargar datos del proyecto principal (../Datos GeoJSON)
        source_dir = BASE_DIR.parent / "Datos GeoJSON"
        if not source_dir.exists():
            print(" Debe especificar --comuna o --source")
            sys.exit(1)
    
    if not source_dir.exists():
        print(f" Directorio no encontrado: {source_dir}")
        sys.exit(1)
    
    # Cargar datos
    load_all_geojson_from_dir(source_dir, engine, args.schema)
    
    # Crear índices espaciales
    create_spatial_indexes(engine, args.schema)
    
    # Mostrar tablas cargadas
    list_tables(engine, args.schema)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                   CARGA COMPLETADA                           ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
