"""Global candidate retrieval, cached higher-resolution DEMs and sensitivity experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from global_terrain_data import ROOT, DATA as GLOBAL, MARS_RADIUS_M, sha256
from terrain_data import MOLA
from terrain_eda import features, slopes

DATA = ROOT / "data/candidates"
OUT = ROOT / "reports/global_terrain"
FIELDS = ["median_slope_deg", "p90_slope_deg", "plane_residual_sd_m"]
WEIGHTS = np.array([.25, .25, .5])
TARGETS = [dict(id="jezero", lat=18.44, lon=77.45, body="Mars"),
           dict(id="gale", lat=-5.4, lon=137.8, body="Mars"),
           dict(id="elysium", lat=4.5, lon=135.9, body="Mars")]
GRID = from_origin(-80000, 80000, 1000, 1000)
SHAPE = (160, 160)
SCALES = (1000, 2000, 5000, 10000)
SHIFTS = list(itertools.product((-10000, 0, 10000), repeat=2))


def crs_for(site, detailed=True):
    radius = 3396190 if detailed else MARS_RADIUS_M
    body = "+datum=WGS84" if site["body"] == "Earth" else f"+a={radius} +b={radius}"
    return rasterio.crs.CRS.from_string(f"+proj=aeqd +lat_0={site['lat']} +lon_0={site['lon']} {body} +units=m")


def warp(source, site, step=1000, detailed=True):
    shape = (160000//step, 160000//step)
    z = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(source) as src:
        reproject(rasterio.band(src, 1), z, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=from_origin(-80000, 80000, step, step),
                  dst_crs=crs_for(site, detailed), dst_nodata=np.nan, resampling=Resampling.average)
    return z


def score(frame, target, tolerances):
    x = frame[FIELDS].to_numpy(dtype=float)
    q = np.array([target[f] for f in FIELDS], dtype=float)
    contribution = WEIGHTS*((np.log1p(x)-np.log1p(q))/tolerances)**2
    return np.sqrt(contribution.sum(axis=1)), contribution


def geodesic_km(a, b):
    return pyproj.Geod(ellps="WGS84").inv(a["lon"], a["lat"], b["lon"], b["lat"])[2]/1000


def coarse_search():
    DATA.mkdir(parents=True, exist_ok=True)
    earth = pd.read_csv(OUT / "earth_patches.csv")
    x = np.log1p(earth[FIELDS].to_numpy())
    tolerances = np.quantile(x, .75, axis=0)-np.quantile(x, .25, axis=0)
    if not np.isfinite(tolerances).all() or (tolerances <= 0).any():
        raise ValueError("Nonpositive feature IQR")
    all_rows, shortlist, targets = [], [], []
    for target in TARGETS:
        z = warp(GLOBAL / "mars_mola32.tif", target, 10000, detailed=False)
        f = features(z[3:13, 3:13], slopes(z, 10000)[3:13, 3:13], 10000)
        targets.append({**target, **f})
        distances, contributions = score(earth, f, tolerances)
        ranked = earth.copy()
        ranked["distance"] = distances
        ranked["target_id"] = target["id"]
        for j, field in enumerate(FIELDS):
            ranked["distance_squared__"+field] = contributions[:, j]
        ranked = ranked.sort_values(["distance", "patch_id"]).reset_index(drop=True)
        ranked["global_rank"] = np.arange(1, len(ranked)+1)
        all_rows.append(ranked)
        selected = []
        for record in ranked.to_dict("records"):
            if all(geodesic_km(record, prior) >= 300 for prior in selected):
                selected.append(record)
            if len(selected) == 6:
                break
        for i, record in enumerate(selected, 1):
            shortlist.append({**record, "id": f"{target['id']}_E{i:02d}", "selection_rank": i,
                              "body": "Earth", "target_id": target["id"]})
        print(target["id"], [(r["global_rank"], round(r["lat"], 2), round(r["lon"], 2)) for r in selected], flush=True)
    pd.concat(all_rows).to_csv(OUT / "candidate_global_rankings.csv", index=False)
    pd.DataFrame(shortlist).to_csv(OUT / "candidate_shortlist.csv", index=False)
    plan = {"targets": targets, "candidates": shortlist, "fields": FIELDS,
            "weights": WEIGHTS.tolist(), "log1p_iqr": tolerances.tolist(),
            "scenario": "EXPLORATORY_REGIONAL_TERRAIN_NOT_EXPERT_APPROVED",
            "candidate_pool_size_per_target": len(earth), "shortlist_per_target": 6,
            "diversity_min_km": 300, "window_km": 100,
            "expert_review": {"status": "pending", "reviewer": None,
                              "scope": "Regional terrain resemblance only; no rover suitability labels"},
            "notes": "Slope group weight 0.5 split between median and p90; residual roughness group 0.5. "
                     "Log1p IQR scales fitted once on all coarse Earth candidates and frozen for refinement."}
    (DATA / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def earth_tiles(site):
    local = crs_for(site)
    bounds = transform_bounds(local, "EPSG:4326", -80000, -80000, 80000, 80000, densify_pts=41)
    left, bottom, right, top = bounds
    if right-left > 180:
        raise ValueError("Dateline-crossing crop needs explicit tile splitting")
    names = []
    for lat in range(math.floor(bottom), math.ceil(top)):
        for lon in range(math.floor(left), math.ceil(right)):
            tile = f"Copernicus_DSM_COG_30_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
            names.append(tile)
    return names


def fetch_tile(tile):
    path = DATA / "earth_tiles" / f"{tile}.tif"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch {path}")
        return record
    url = f"https://copernicus-dem-90m.s3.amazonaws.com/{tile}/{tile}.tif"
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5,
            status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
        for attempt in range(3):
            try:
                response = session.get(url, timeout=(30, 90))
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
    if response.status_code == 404:
        return {"tile": tile, "url": url, "status": "missing_404"}
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(response.content)
    with rasterio.open(part) as src:
        if src.count != 1 or src.crs is None:
            raise ValueError(f"Unexpected raster {tile}")
    part.replace(path)
    record = {"tile": tile, "file": str(path.relative_to(ROOT)), "url": url,
              "sha256": sha256(path), "bytes": path.stat().st_size,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
              "status": "downloaded"}
    meta.write_text(json.dumps(record, indent=2))
    print(f"Downloaded {tile}", flush=True)
    return record


def fetch_mars(site):
    path = DATA / "mars_native" / f"{site['id']}.tif"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch {path}")
        return record
    print(f"Reading detailed Mars crop {site['id']} ...", flush=True)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="2"):
        with rasterio.open(MOLA) as src:
            bounds = transform_bounds(crs_for(site), src.crs, -80000, -80000, 80000, 80000, densify_pts=41)
            raw = from_bounds(*bounds, transform=src.transform)
            col, row = math.floor(raw.col_off)-2, math.floor(raw.row_off)-2
            width = math.ceil(raw.col_off+raw.width)+2-col
            height = math.ceil(raw.row_off+raw.height)+2-row
            window = Window(col, row, width, height)
            z = src.read(1, window=window, masked=True).astype('float32')
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                               dtype="float32", crs=src.crs, transform=src.window_transform(window),
                               nodata=-99999, compress="deflate") as dst:
                dst.write(z.filled(-99999), 1)
            record = {"site_id": site["id"], "url": MOLA, "file": str(path.relative_to(ROOT)),
                      "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256(path),
                      "source_crs_wkt": src.crs.to_wkt(), "source_nodata": src.nodata,
                      "source_scale": src.scales[0], "source_offset": src.offsets[0],
                      "window": [col, row, width, height], "bytes": path.stat().st_size}
            if src.scales[0] != 1 or src.offsets[0] != 0:
                raise ValueError("Source scale/offset requires explicit application")
    meta.write_text(json.dumps(record, indent=2))
    print(f"Downloaded Mars {site['id']}", flush=True)
    return record


def download(plan):
    tiles = sorted(set(itertools.chain.from_iterable(earth_tiles(s) for s in plan["candidates"])))
    print(f"Downloading {len(tiles)} unique Earth tiles and 3 Mars crops", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        earth_records = list(pool.map(fetch_tile, tiles))
    mars_records = [fetch_mars(s) for s in plan["targets"]]
    (DATA / "sources.json").write_text(json.dumps({"earth": earth_records, "mars": mars_records}, indent=2))


def detailed_earth(site):
    z = np.full(SHAPE, np.nan, dtype="float32")
    with rasterio.open(GLOBAL / "earth_land_mask.tif") as mask:
        for tile in earth_tiles(site):
            path = DATA / "earth_tiles" / f"{tile}.tif"
            if not path.exists():
                continue  # provenance records every 404; valid-area threshold decides usability
            with rasterio.open(path) as src:
                native = src.read(1, masked=True).astype("float32").filled(np.nan)
                eligible = np.zeros(src.shape, dtype='uint8')
                reproject(rasterio.band(mask, 1), eligible, src_transform=mask.transform, src_crs=mask.crs,
                          dst_transform=src.transform, dst_crs=src.crs, resampling=Resampling.nearest)
                native[eligible == 0] = np.nan
                reproject(native, z, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                          dst_transform=GRID, dst_crs=crs_for(site), dst_nodata=np.nan,
                          resampling=Resampling.average, init_dest_nodata=False)
        eligible = np.zeros(SHAPE, dtype='uint8')
        reproject(rasterio.band(mask, 1), eligible, src_transform=mask.transform, src_crs=mask.crs,
                  dst_transform=GRID, dst_crs=crs_for(site), resampling=Resampling.nearest)
        z[eligible == 0] = np.nan
    return z


def coarse_grid(z, step):
    factor = step//1000
    blocks = z.reshape(SHAPE[0]//factor, factor, SHAPE[1]//factor, factor)
    count = np.isfinite(blocks).sum(axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count >= .95*factor**2)


def window_features(z, step, east=0, north=0):
    left = int((30000+east)//step)
    top = int((30000-north)//step)
    size = 100000//step
    s = slopes(z, step)
    return features(z[top:top+size, left:left+size], s[top:top+size, left:left+size], step)


def analyze(plan):
    records, failures, grids = [], [], {}
    sites = [*plan["targets"], *plan["candidates"]]
    for site in sites:
        ident = site["id"]
        z = (warp(DATA/"mars_native"/f"{ident}.tif", site) if site["body"] == "Mars"
             else detailed_earth(site))
        grids[ident] = z
        aligned = DATA / "aligned" / f"{ident}.tif"
        aligned.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(aligned, "w", driver="GTiff", count=1, width=160, height=160,
                           crs=crs_for(site), transform=GRID, dtype="float32", nodata=-99999,
                           compress="deflate") as dst:
            dst.write(np.where(np.isfinite(z), z, -99999), 1)
        for step in SCALES:
            reduced = coarse_grid(z, step)
            for east, north in SHIFTS:
                try:
                    f = window_features(reduced, step, east, north)
                    records.append({"site_id": ident, "body": site["body"], "step_m": step,
                                    "east_m": east, "north_m": north, **f})
                except ValueError as error:
                    failures.append({"site_id": ident, "step_m": step, "east_m": east,
                                     "north_m": north, "reason": str(error)})
        print(f"Refined {ident}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "candidate_detailed_features.csv", index=False)
    pd.DataFrame(failures, columns=["site_id", "step_m", "east_m", "north_m", "reason"]).to_csv(OUT / "candidate_invalid_windows.csv", index=False)
    tol = np.array(plan["log1p_iqr"])
    scenarios, baseline, source_changes = [], [], []
    valid_pool = {}
    for target in plan["targets"]:
        tid = target["id"]
        candidates = [s for s in plan["candidates"] if s["target_id"] == tid]
        subset = frame[frame.site_id.isin([s["id"] for s in candidates])]
        target_rows = frame[frame.site_id == tid]
        center = subset[(subset.step_m == 1000)&(subset.east_m == 0)&(subset.north_m == 0)].copy()
        query = target_rows[(target_rows.step_m == 1000)&(target_rows.east_m == 0)&(target_rows.north_m == 0)]
        if query.empty or center.empty:
            failures.append({"site_id": tid, "reason": "No baseline query/candidates"})
            continue
        dist, parts = score(center, query.iloc[0], tol)
        center["distance"] = dist
        for j, field in enumerate(FIELDS):
            center["distance_squared__"+field] = parts[:, j]
        center = center.sort_values(["distance", "site_id"])
        center["rank_1km"] = np.arange(1, len(center)+1)
        center["target_id"] = tid
        baseline.append(center)
        # Fixed common pool: every compared candidate must be valid for ALL 36 settings.
        counts = subset.groupby("site_id").size()
        pool = counts[counts == len(SCALES)*len(SHIFTS)].index.tolist()
        valid_pool[tid] = pool
        for step in SCALES:
            for qe, qn in SHIFTS:
                q = target_rows[(target_rows.step_m == step)&(target_rows.east_m == qe)&(target_rows.north_m == qn)]
                if q.empty:
                    continue
                for ee, en in SHIFTS:
                    sub = subset[(subset.step_m == step)&(subset.east_m == ee)&(subset.north_m == en)&subset.site_id.isin(pool)].copy()
                    if sub.empty:
                        continue
                    sub["distance"], _ = score(sub, q.iloc[0], tol)
                    sub = sub.sort_values(["distance", "site_id"])
                    for rank, row in enumerate(sub.itertuples(), 1):
                        scenarios.append({"target_id": tid, "site_id": row.site_id, "step_m": step,
                                          "query_east_m": qe, "query_north_m": qn,
                                          "earth_east_m": ee, "earth_north_m": en,
                                          "rank": rank, "distance": row.distance, "pool_size": len(pool)})
        for site in [target, *candidates]:
            global_source = GLOBAL/("mars_mola32.tif" if site["body"] == "Mars" else "earth_land.tif")
            same_grid = warp(global_source, site, 10000, detailed=False)
            try:
                original = window_features(same_grid, 10000)
                detailed = frame[(frame.site_id == site["id"])&(frame.step_m == 10000)&(frame.east_m == 0)&(frame.north_m == 0)]
                if detailed.empty:
                    continue
                for field in FIELDS:
                    source_changes.append({"site_id": site["id"], "target_id": tid, "feature": field,
                                           "global_source_10km": original[field],
                                           "detailed_source_10km": detailed.iloc[0][field]})
            except ValueError:
                pass
    if not baseline:
        raise ValueError("No valid refinement baselines; inspect missing DEM coverage")
    base = pd.concat(baseline)
    trials = pd.DataFrame(scenarios)
    base.to_csv(OUT / "candidate_refined_ranking.csv", index=False)
    trials.to_csv(OUT / "candidate_sensitivity_trials.csv", index=False)
    pd.DataFrame(source_changes).to_csv(OUT / "candidate_source_changes.csv", index=False)
    stability = trials.groupby(["target_id", "site_id"]).agg(
        experiments=("rank", "size"), best_rank=("rank", "min"), median_rank=("rank", "median"),
        worst_rank=("rank", "max"), top1_frequency=("rank", lambda s: (s == 1).mean()),
        top3_frequency=("rank", lambda s: (s <= 3).mean())).reset_index()
    stability.to_csv(OUT / "candidate_rank_stability.csv", index=False)
    metadata = {"valid_pool_by_target": valid_pool, "scales_m": list(SCALES), "shifts_m": SHIFTS,
                "independent_query_earth_shifts": True, "possible_trials_per_target": 324,
                "sensitivity_note": "Finite deterministic settings; frequencies are not confidence or probabilities",
                "expert_review": plan["expert_review"], "invalid_feature_windows": len(failures),
                "status": "exploratory terrain candidates; not confirmed analogs"}
    (DATA / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(stability.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["search", "download", "analyze"])
    args = parser.parse_args()
    if args.stage == "search":
        coarse_search()
    else:
        plan = json.loads((DATA/"plan.json").read_text())
        if args.stage == "download":
            download(plan)
        else:
            analyze(plan)


if __name__ == "__main__":
    main()
