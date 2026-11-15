"""Scan all global 10-degree tiles using local metric 10 km grids and area weights."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from scipy.ndimage import minimum_filter

from global_terrain_data import DATA, ROOT, MARS_RADIUS_M
from terrain_eda import features, slopes

OUT = ROOT / "reports/global_terrain"
STEP = 10000
HALO = 6  # derivatives + enough room to build 100 km windows across tile edges


def prepare_land() -> None:
    """Mask bathymetry BEFORE averaging elevations; preserve below-sea-level land."""
    output = DATA / "earth_land.tif"
    if output.exists():
        return
    with rasterio.open(DATA / "earth_etopo2022_60s.tif") as source:
        records = []
        for name, value in [("land.geojson", 1), ("lakes.geojson", 0)]:
            records.extend((f["geometry"], value) for f in json.loads((DATA/name).read_text())["features"]
                           if f["geometry"] is not None)
        print("Rasterizing full global land and lake mask ...", flush=True)
        mask = rasterize(records, out_shape=source.shape, transform=source.transform, dtype="uint8")
        profile = source.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", nodata=-99999, dtype="float32")
        with rasterio.open(output, "w", **profile) as dst:
            for row in range(0, source.height, 256):
                height = min(256, source.height-row)
                window = Window(0, row, source.width, height)
                z = source.read(1, window=window, masked=True).astype("float32").filled(-99999)
                z[mask[row:row+height] == 0] = -99999
                dst.write(z, 1, window=window)
            dst.update_tags(source="NOAA ETOPO 2022 v1 60s ice surface",
                            mask="Natural Earth 1:10m land minus lakes; applied at native ETOPO grid")
        # Also save the independently defined eligibility mask for coverage reporting.
        profile.update(dtype="uint8", nodata=None)
        with rasterio.open(DATA / "earth_land_mask.tif", "w", **profile) as dst:
            dst.write(mask, 1)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_w = weights[order]
    centers = (np.cumsum(sorted_w)-.5*sorted_w)/sorted_w.sum()
    return float(np.interp(q, centers, values[order]))


def tile_grid(body: str, south: int, west: int):
    lat, lon = south+5, west+5
    geo = "+proj=longlat +datum=WGS84" if body == "Earth" else f"+proj=longlat +a={MARS_RADIUS_M} +b={MARS_RADIUS_M}"
    ellipsoid = "+datum=WGS84" if body == "Earth" else f"+a={MARS_RADIUS_M} +b={MARS_RADIUS_M}"
    local = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} {ellipsoid} +units=m")
    left, bottom, right, top = transform_bounds(geo, local, west, south, west+10, south+10, densify_pts=41)
    # Snap to 100 km so patch blocks stay 10 x 10 pixels.
    left = math.floor(left/100000)*100000-HALO*STEP
    right = math.ceil(right/100000)*100000+HALO*STEP
    bottom = math.floor(bottom/100000)*100000-HALO*STEP
    top = math.ceil(top/100000)*100000+HALO*STEP
    width, height = int(round((right-left)/STEP)), int(round((top-bottom)/STEP))
    transform = from_origin(left, top, STEP, STEP)
    x = left+(np.arange(width)+.5)*STEP
    y = top-(np.arange(height)+.5)*STEP
    xx, yy = np.meshgrid(x, y)
    longitude, latitude = pyproj.Transformer.from_crs(local, geo, always_xy=True).transform(xx, yy)
    # Membership uses half-open geographic cells; centers on polar singularity have zero measure.
    wrapped = ((longitude-west) % 360)
    member = (wrapped < 10) & (latitude >= south) & (latitude < south+10)
    factors = pyproj.Proj(local).get_factors(longitude, latitude)
    area = STEP**2 / np.asarray(factors.areal_scale)
    return local, transform, longitude, latitude, member, area


def extract_body(body: str) -> None:
    destination = DATA / f"{body.lower()}_cells.npz"
    cache_metadata = destination.with_suffix(".json")
    config = {"step_m": STEP, "mars_radius_m": MARS_RADIUS_M, "version": 1}
    if (destination.exists() and (OUT / f"{body.lower()}_patches.csv").exists()
            and cache_metadata.exists() and json.loads(cache_metadata.read_text()) == config):
        print(f"Cached global features for {body}", flush=True)
        return
    patches, tiles = [], []
    cells = {key: [] for key in ["lat", "lon", "slope_deg", "area_km2", "elevation_m"]}
    source_path = DATA / ("earth_land.tif" if body == "Earth" else "mars_mola32.tif")
    with rasterio.open(source_path) as source:
        mask_source = rasterio.open(DATA / "earth_land_mask.tif") if body == "Earth" else None
        try:
            for south in range(-90, 90, 10):
                for west in range(-180, 180, 10):
                    local, transform, lon, lat, member, area = tile_grid(body, south, west)
                    shape = member.shape
                    z = np.full(shape, -99999, dtype="float32")
                    reproject(rasterio.band(source, 1), z, src_transform=source.transform, src_crs=source.crs,
                              src_nodata=source.nodata, dst_transform=transform, dst_crs=local,
                              dst_nodata=-99999, resampling=Resampling.average, num_threads=2)
                    z[z == -99999] = np.nan
                    if mask_source:
                        eligible = np.zeros(shape, dtype="uint8")
                        reproject(rasterio.band(mask_source, 1), eligible,
                                  src_transform=mask_source.transform, src_crs=mask_source.crs,
                                  dst_transform=transform, dst_crs=local,
                                  resampling=Resampling.nearest, dst_nodata=0)
                        eligible = eligible.astype(bool)
                    else:
                        eligible = np.ones(shape, dtype=bool)
                    z[~eligible] = np.nan
                    slope = slopes(z, STEP)
                    # Slopes require a neighborhood on land, avoiding coastal/bathymetric transitions.
                    valid = np.isfinite(slope) & minimum_filter(eligible.astype('uint8'), size=3).astype(bool)
                    take = member & valid & np.isfinite(area) & (area > 0)
                    selected_area = area[take]/1e6
                    tile_id = f"{body}_{south:+03d}_{west:+04d}"
                    for key, array in [("lat", lat), ("lon", lon), ("slope_deg", slope),
                                       ("area_km2", area/1e6), ("elevation_m", z)]:
                        cells[key].append(array[take].astype('float32'))
                    domain = member & eligible
                    valid_z = domain & np.isfinite(z)
                    tile = {"tile_id": tile_id, "body": body, "south": south, "west": west,
                            "domain_cells": int(domain.sum()), "valid_elevation_cells": int(valid_z.sum()),
                            "valid_slope_cells": int(take.sum()),
                            "domain_area_km2": float(area[domain].sum()/1e6),
                            "valid_elevation_area_km2": float(area[valid_z].sum()/1e6),
                            "valid_slope_area_km2": float(selected_area.sum()),
                            "median_slope_deg": weighted_quantile(slope[take], selected_area, .5) if take.any() else np.nan}
                    tiles.append(tile)
                    # Enumerate every 100 km block of this local grid. Retain full domain blocks only.
                    # Partial coast/tile-edge blocks are counted as exclusions, never filled with zeros.
                    for row in range(HALO, shape[0]-HALO-9, 10):
                        for col in range(HALO, shape[1]-HALO-9, 10):
                            ys, xs = slice(row, row+10), slice(col, col+10)
                            if not member[ys, xs].all() or take[ys, xs].mean() < .95:
                                continue
                            patch_z, patch_s = z[ys, xs], slope[ys, xs].copy()
                            patch_s[~take[ys, xs]] = np.nan
                            if np.isfinite(patch_z).mean() < .95:
                                continue
                            f = features(patch_z, patch_s, STEP)
                            patches.append({"patch_id": f"{tile_id}_{row}_{col}", "body": body,
                                            "tile_id": tile_id, "lat": float(lat[ys, xs].mean()),
                                            "lon": float(lon[ys, xs].mean()), "pixel_m": STEP,
                                            "window_km": 100,
                                            "area_km2": float(area[ys, xs][take[ys, xs]].sum()/1e6), **f})
                print(f"{body}: completed latitude {south:+d}..{south+10:+d}; {len(patches)} patches", flush=True)
        finally:
            if mask_source:
                mask_source.close()
    np.savez_compressed(destination, **{key: np.concatenate(value) for key, value in cells.items()})
    pd.DataFrame(patches).to_csv(OUT / f"{body.lower()}_patches.csv", index=False)
    pd.DataFrame(tiles).to_csv(OUT / f"{body.lower()}_coverage.csv", index=False)
    cache_metadata.write_text(json.dumps(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", choices=["Earth", "Mars", "both"], default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.body in ("Earth", "both"):
        prepare_land()
        extract_body("Earth")
    if args.body in ("Mars", "both"):
        extract_body("Mars")
    (DATA / "analysis_config.json").write_text(json.dumps({"pixel_m": STEP, "tile_degrees": 10,
        "patch_km": 100, "patch_valid_fraction_min": .95, "resampling": "average",
        "Earth_crs": "WGS84, local AEQD per 10 degree tile",
        "Mars_radius_m": MARS_RADIUS_M, "area_weights": "pixel area / local projection areal_scale",
        "mask": "Natural Earth 1:10m land minus lakes, applied before resampling; eroded for slopes",
        "coastal_and_tile_edge_patch_policy": "exclude incomplete 100 km windows"}, indent=2))


if __name__ == "__main__":
    main()
