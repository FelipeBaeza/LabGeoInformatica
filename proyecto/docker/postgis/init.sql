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
