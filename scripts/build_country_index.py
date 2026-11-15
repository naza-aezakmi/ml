"""Pack Natural Earth country boundaries for offline candidate labels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geojson", type=Path)
    args = parser.parse_args()
    polygons = []
    for feature in json.loads(args.geojson.read_text())["features"]:
        properties, geometry = feature["properties"], feature["geometry"]
        if not geometry:
            continue
        country = properties["ADMIN"]
        code = properties.get("ISO_A2_EH") or properties.get("ISO_A2")
        code = None if code == "-99" else code
        parts = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for rings in parts:
            outer = rings[0]
            bounds = [min(p[0] for p in outer), min(p[1] for p in outer), max(p[0] for p in outer), max(p[1] for p in outer)]
            polygons.append({"country": country, "code": code, "bounds": bounds, "rings": rings})
    payload = {"source": SOURCE, "source_sha256": hashlib.sha256(args.geojson.read_bytes()).hexdigest(), "scale": "Natural Earth 1:50m", "polygons": polygons}
    data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0)
    for directory in [ROOT / "artifacts/environment", ROOT.parent / "api/app/data/environment"]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "countries.json.gz").write_bytes(data)
    print(f"Packed {len(polygons)} polygons, {len(data)} bytes")


if __name__ == "__main__":
    main()
