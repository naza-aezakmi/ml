"""Fetch complete global source models, preserve provenance, prepare readable MOLA."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/global_terrain"
MARS_RADIUS_M = 3396000  # PDS4 MEG032 label, not the different USGS 463 m mosaic sphere
PDS = "https://pds-geosciences.wustl.edu/mgs/urn-nasa-pds-mgs_mola_topography_derived/meg032/"
URLS = {
    "earth_etopo2022_60s.tif": "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif",
    "megt90n000fb.img": PDS + "megt90n000fb.img",
    "megt90n000fb.xml": PDS + "megt90n000fb.xml",
    "land.geojson": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson",
    "lakes.geojson": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_lakes.geojson",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item: tuple[str, str]) -> dict:
    name, url = item
    path = DATA / name
    meta = path.with_suffix(path.suffix + ".json")
    if path.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if record["sha256"] != sha256(path):
            raise ValueError(f"Checksum mismatch: {path}")
        print(f"Cached {name}", flush=True)
        return record
    part = path.with_suffix(path.suffix + ".part")
    # A complete decoded GeoJSON can remain after a transport-length mismatch.
    # Validate its full structure before recovering; do not append a byte range to decoded data.
    if name.endswith(".geojson") and part.exists():
        try:
            obj = json.loads(part.read_text())
            if obj.get("type") == "FeatureCollection" and isinstance(obj.get("features"), list):
                record = {"file": name, "url": url, "bytes": part.stat().st_size,
                          "retrieved_at_utc": datetime.fromtimestamp(part.stat().st_mtime, timezone.utc).isoformat(),
                          "sha256": sha256(part),
                          "recovery_note": "Complete parsed GeoJSON recovered after HTTP decoded-length mismatch; original HTTP headers unavailable"}
                part.replace(path)
                meta.write_text(json.dumps(record, indent=2))
                print(f"Recovered complete JSON {name}", flush=True)
                return record
        except (ValueError, UnicodeDecodeError):
            pass
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, stream=True, headers=headers, timeout=(30, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if not append:
                    offset = 0
                read = offset
                next_log = (read // (32*1024*1024)+1)*32*1024*1024
                with part.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
                        read += len(chunk)
                        if read >= next_log:
                            print(f"{name}: {read/1e6:.0f} MB", flush=True)
                            next_log += 32*1024*1024
                expected = response.headers.get("Content-Length")
                if expected and not response.headers.get("Content-Encoding") and read-offset != int(expected):
                    raise ValueError(f"Truncated response for {name}")
                record = {"file": name, "url": url, "bytes": read,
                          "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                          "etag": response.headers.get("ETag"),
                          "last_modified": response.headers.get("Last-Modified"),
                          "sha256": sha256(part)}
            part.replace(path)
            meta.write_text(json.dumps(record, indent=2))
            print(f"Complete {name}: {read/1e6:.1f} MB", flush=True)
            return record
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def prepare_mars() -> None:
    path = DATA / "mars_mola32.tif"
    if path.exists():
        with rasterio.open(path) as src:
            import pyproj
            if abs(pyproj.CRS(src.crs).ellipsoid.semi_major_metre-MARS_RADIUS_M) < .01:
                return
    # Validate raw binary layout against the downloaded PDS4 label, never guess endian/shape.
    tree = ET.parse(DATA / "megt90n000fb.xml")
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1"}
    image = tree.find(".//p:Array_2D_Image", ns)
    if image is None:
        raise ValueError("PDS4 label has no Array_2D_Image")
    axes = {e.findtext("p:axis_name", namespaces=ns): int(e.findtext("p:elements", namespaces=ns))
            for e in image.findall("p:Axis_Array", ns)}
    dtype = image.findtext("p:Element_Array/p:data_type", namespaces=ns)
    offset = int(image.findtext("p:offset", namespaces=ns))
    if axes != {"Line": 5760, "Sample": 11520} or dtype != "SignedMSB2" or offset != 0:
        raise ValueError(f"Unexpected MOLA layout: {axes}, {dtype}, offset={offset}")
    raw = DATA / "megt90n000fb.img"
    if raw.stat().st_size != 5760*11520*2:
        raise ValueError("Incorrect full MOLA file size")
    data = np.memmap(raw, dtype=">i2", mode="r", shape=(5760, 11520))
    profile = dict(driver="GTiff", width=11520, height=5760, count=1, dtype="int16",
                   crs=f"+proj=longlat +a={MARS_RADIUS_M} +b={MARS_RADIUS_M} +no_defs",
                   transform=from_origin(-180, 90, 1/32, 1/32),
                   compress="deflate", tiled=True, nodata=-32768)
    with rasterio.open(path, "w", **profile) as dst:
        for row in range(0, 5760, 256):
            block = np.roll(np.asarray(data[row:row+256], dtype="int16"), 5760, axis=1)
            dst.write(block, 1, window=Window(0, row, 11520, len(block)))
        dst.update_tags(source_url=URLS["megt90n000fb.img"],
                        longitude_conversion="0..360 to -180..180 by half-width roll",
                        vertical_units="m", vertical_datum="Mars areoid")
    print(f"Prepared {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.download:
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(fetch, URLS.items()))
    else:
        records = []
        for name in URLS:
            path = DATA / name
            meta = path.with_suffix(path.suffix + ".json")
            if not path.exists() or not meta.exists():
                raise FileNotFoundError(f"Run global_terrain_data.py --download; missing {name}")
            record = json.loads(meta.read_text())
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
            records.append(record)
    prepare_mars()
    (DATA / "manifest.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
