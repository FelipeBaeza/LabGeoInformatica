-- Activar extensiones espaciales
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Crear schema para el proyecto
CREATE SCHEMA IF NOT EXISTS geoanalisis;

-- Otorgar permisos
GRANT ALL PRIVILEGES ON SCHEMA geoanalisis TO geouser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA geoanalisis TO geouser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA geoanalisis TO geouser;

-- Mensaje de confirmación
DO $$
BEGIN
    RAISE NOTICE 'PostGIS configurado correctamente para el proyecto!';
END $$;

-- Load data from SQL files
\i '/data/Datos SQL/area_construcciones.sql'
\i '/data/Datos SQL/area_estacionamiento.sql'
\i '/data/Datos SQL/area_interes.sql'
\i '/data/Datos SQL/area_naturaleza_playas.sql'
\i '/data/Datos SQL/area_religiosa.sql'
\i '/data/Datos SQL/area_reserva_agua.sql'
\i '/data/Datos SQL/area_transporte.sql'
\i '/data/Datos SQL/area_uso_de_tierra_agua.sql'
\i '/data/Datos SQL/linea_calles.sql'
\i '/data/Datos SQL/linea_flujo_agua.sql'
\i '/data/Datos SQL/mapa_comuna.sql'
\i '/data/Datos SQL/punto_atraccion_turistica.sql'
\i '/data/Datos SQL/punto_interes.sql'
\i '/data/Datos SQL/punto_naturaleza.sql'
\i '/data/Datos SQL/punto_religioso.sql'
\i '/data/Datos SQL/punto_trafico.sql'
\i '/data/Datos SQL/punto_transporte.sql'