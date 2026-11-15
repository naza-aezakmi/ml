"""Download small source DEM windows and prepare comparable metric grids."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/terrain"
MOLA = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
SITES = [
    {"id": "jezero", "body": "Mars", "lat": 18.44, "lon": 77.45},
    {"id": "gale", "body": "Mars", "lat": -5.4, "lon": 137.8},
    {"id": "elysium", "body": "Mars", "lat": 4.5, "lon": 135.9},
    {"id": "atacama", "body": "Earth", "lat": -24.5, "lon": -69.5},
    {"id": "namib", "body": "Earth", "lat": -24.5, "lon": 15.5},
    {"id": "mojave", "body": "Earth", "lat": 35.5, "lon": -116.5},
    {"id": "ladakh", "body": "Earth", "lat": 34.5, "lon": 77.5},
    {"id": "iceland", "body": "Earth", "lat": 64.5, "lon": -18.5},
    {"id": "sahara", "body": "Earth", "lat": 22.5, "lon": 5.5},
]
PIXEL_M = 1000
SIZE = 46  # 40 km analysis window + 3 km halo on each side
TRANSFORM = from_origin(-23000, 23000, PIXEL_M, PIXEL_M)


def source_url(site: dict) -> str:
    if site["body"] == "Mars":
        return MOLA
    lat, lon = math.floor(site["lat"]), math.floor(site["lon"])
    tile = (f"Copernicus_DSM_COG_30_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM")
    return f"https://copernicus-dem-90m.s3.amazonaws.com/{tile}/{tile}.tif"


def local_crs(site: dict) -> CRS:
    body = "+a=3396190 +b=3396190" if site["body"] == "Mars" else "+datum=WGS84"
    return CRS.from_string(f"+proj=aeqd +lat_0={site['lat']} +lon_0={site['lon']} {body} +units=m +no_defs")


def prepare_site(site: dict, download: bool) -> dict:
    native = DATA / "native" / f"{site['id']}.tif"
    metadata_path = native.with_suffix(".json")
    destination = DATA / "aligned" / f"{site['id']}.tif"
    crs = local_crs(site)
    if not native.exists() or not metadata_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {native}; run terrain_data.py --download first")
        url = source_url(site)
        print(f"Reading {site['body']} {site['id']} ...", flush=True)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="60",
                          GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="2"):
            with rasterio.open(url) as src:
                bounds = transform_bounds(crs, src.crs, -23000, -23000, 23000, 23000, densify_pts=41)
                raw = from_bounds(*bounds, transform=src.transform)
                col, row = math.floor(raw.col_off) - 2, math.floor(raw.row_off) - 2
                right, bottom = math.ceil(raw.col_off + raw.width) + 2, math.ceil(raw.row_off + raw.height) + 2
                window = Window(col, row, right-col, bottom-row)
                if col < 0 or row < 0 or right > src.width or bottom > src.height:
                    raise ValueError(f"{site['id']}: analysis window crosses source tile; mosaic required")
                z = src.read(1, window=window, masked=True).astype("float32")
                if not np.isfinite(z.filled(np.nan)).any():
                    raise ValueError(f"{site['id']}: no valid elevation data")
                z = z * src.scales[0] + src.offsets[0]
                native.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(native, "w", driver="GTiff", width=z.shape[1], height=z.shape[0],
                                   count=1, dtype="float32", crs=src.crs,
                                   transform=src.window_transform(window), nodata=-9999,
                                   compress="deflate") as dst:
                    dst.write(z.filled(-9999), 1)
                metadata = {**site, "source_url": url,
                            "catalog_url": "https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m"
                            if site["body"] == "Mars" else "https://copernicus-dem-90m.s3.amazonaws.com/readme.html",
                            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                            "source_crs_wkt": src.crs.to_wkt(), "source_pixel_size_crs_units": list(src.res),
                            "source_shape": list(src.shape), "source_nodata": src.nodata,
                            "source_scale": src.scales[0], "source_offset": src.offsets[0],
                            "source_window_col_row_width_height": [col, row, right-col, bottom-row],
                            "vertical_units": "m", "native_crop_shape": list(z.shape),
                            "source_notes": "MOLA interpolated altimetry, heights above areoid" if site["body"] == "Mars"
                            else "Copernicus GLO-90 DSM, EGM2008 heights; may include vegetation/buildings",
                            "native_crop_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(native.read_bytes()).hexdigest() != metadata["native_crop_sha256"]:
        raise ValueError(f"Checksum mismatch for {native}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(native) as src:
        aligned = np.full((SIZE, SIZE), -9999, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=aligned,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=TRANSFORM, dst_crs=crs, dst_nodata=-9999,
                  resampling=Resampling.average)
    with rasterio.open(destination, "w", driver="GTiff", width=SIZE, height=SIZE, count=1,
                       dtype="float32", crs=crs, transform=TRANSFORM, nodata=-9999,
                       compress="deflate") as dst:
        dst.write(aligned, 1)
        dst.update_tags(site_id=site["id"], body=site["body"], source_url=metadata["source_url"],
                        resampling="average", analysis_pixel_m=str(PIXEL_M))
    return {**metadata, "aligned_file": str(destination.relative_to(ROOT)),
            "analysis_crs_wkt": crs.to_wkt(), "analysis_pixel_m": PIXEL_M,
            "resampling": "average", "analysis_extent_km": 40, "halo_km": 3,
            "valid_fraction_including_halo": float((aligned != -9999).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Allow online download of missing native crops")
    args = parser.parse_args()
    records = [prepare_site(site, args.download) for site in SITES]
    (DATA / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(records)} regions at {PIXEL_M} m/pixel; {DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()
