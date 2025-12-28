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

-- Load data from SQL files (usando ruta del volumen montado)
\i '/datos_sql/area_construcciones.sql'
\i '/datos_sql/area_estacionamiento.sql'
\i '/datos_sql/area_interes.sql'
\i '/datos_sql/area_naturaleza_playas.sql'
\i '/datos_sql/area_religiosa.sql'
\i '/datos_sql/area_reserva_agua.sql'
\i '/datos_sql/area_transporte.sql'
\i '/datos_sql/area_uso_de_tierra_agua.sql'
\i '/datos_sql/linea_calles.sql'
\i '/datos_sql/linea_flujo_agua.sql'
\i '/datos_sql/mapa_comuna.sql'
\i '/datos_sql/punto_atraccion_turistica.sql'
\i '/datos_sql/punto_interes.sql'
\i '/datos_sql/punto_naturaleza.sql'
\i '/datos_sql/punto_religioso.sql'
\i '/datos_sql/punto_trafico.sql'
\i '/datos_sql/punto_transporte.sql'

-- Mensaje final
DO $$
BEGIN
    RAISE NOTICE 'Datos SQL cargados correctamente!';
END $$;