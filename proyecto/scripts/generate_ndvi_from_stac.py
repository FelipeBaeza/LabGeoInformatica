import argparse
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from pystac_client import Client
from sqlalchemy import create_engine
from dotenv import load_dotenv


def get_db_config():
    load_dotenv()
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "55432"),
        "db": os.getenv("POSTGRES_DB", "geodatabase"),
        "user": os.getenv("POSTGRES_USER", "geouser"),
        "password": os.getenv("POSTGRES_PASSWORD", "geopass123"),
    }


def get_engine(cfg):
    return create_engine(
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['db']}"
    )


def get_boundary(engine):
    sql = "SELECT geometry FROM geoanalisis.limite_administrativa"
    gdf = gpd.read_postgis(sql, engine, geom_col="geometry")
    if gdf.empty:
        raise RuntimeError("No se encontró limite_administrativa en geoanalisis")
    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.unary_union
    return gdf, geom


def _find_asset_href(item, desired_common, fallback_keys):
    for key, asset in item.assets.items():
        common_name = None
        bands = asset.extra_fields.get("eo:bands") if hasattr(asset, "extra_fields") else None
        if bands:
            common_name = bands[0].get("common_name")
        if common_name == desired_common or key in fallback_keys:
            return asset.href
    raise KeyError(f"No se encontró asset para {desired_common} en item {item.id}")


def fetch_ndvi(bounds, geom, start_date, end_date, out_tif):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 30}},
        limit=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No se encontraron escenas Sentinel-2 para el rango dado")
    item = items[0]
    b04_href = _find_asset_href(item, "red", {"B04", "red"})
    b08_href = _find_asset_href(item, "nir", {"B08", "nir"})

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(b04_href) as red_src, rasterio.open(b08_href) as nir_src:
            geom_src = (
                gpd.GeoSeries([geom], crs="EPSG:4326")
                .to_crs(red_src.crs)
                .iloc[0]
            )
            red, red_transform = mask(red_src, [geom_src], crop=True)
            nir, nir_transform = mask(nir_src, [geom_src], crop=True)

            red = red.astype("float32")
            nir = nir.astype("float32")
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = np.clip(ndvi, -1.0, 1.0)

            profile = red_src.profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                transform=red_transform,
                height=ndvi.shape[1],
                width=ndvi.shape[2],
                nodata=None,
                compress="lzw",
            )

            out_path = Path(out_tif)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(ndvi[0], 1)
    return out_tif


def load_raster_to_postgis(tif_path, schema, table, cfg):
    if not shutil.which("raster2pgsql"):
        raise RuntimeError("raster2pgsql no está instalado")
    env = os.environ.copy()
    env["PGPASSWORD"] = cfg["password"]
    cmd = (
        f"raster2pgsql -s 4326 -I -C -M {tif_path} {schema}.{table} | "
        f"psql -h {cfg['host']} -p {cfg['port']} -U {cfg['user']} -d {cfg['db']}"
    )
    subprocess.run(cmd, shell=True, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Descargar NDVI Sentinel-2 y cargar a PostGIS")
    parser.add_argument("--start", default="2024-06-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-06-30", help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--schema", default="geoanalisis", help="Schema destino")
    parser.add_argument("--table", default="ndvi_isla_de_pascua", help="Tabla destino")
    parser.add_argument("--out", default="data/processed/ndvi.tif", help="Ruta GeoTIFF salida")
    args = parser.parse_args()

    cfg = get_db_config()
    engine = get_engine(cfg)
    gdf, geom = get_boundary(engine)
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy

    tif_path = fetch_ndvi(bounds, geom, args.start, args.end, args.out)
    load_raster_to_postgis(tif_path, args.schema, args.table, cfg)
    print(f"NDVI cargado en {args.schema}.{args.table} desde {tif_path}")


if __name__ == "__main__":
    try:
        import shutil
    except ImportError:
        pass
    main()
