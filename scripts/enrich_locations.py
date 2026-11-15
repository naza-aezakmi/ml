"""Attach administrative labels from a local Natural Earth admin-1 GeoJSON."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_admin_1_states_provinces.geojson"
NAMES = ["Nearly flat · low relief", "Nearly flat · moderate relief", "Gentle terrain · moderate relief", "Gentle terrain · pronounced relief", "Gentle terrain · pronounced relief", "Rolling terrain · pronounced relief", "Rolling terrain · high relief", "Hilly terrain · high relief", "Hilly terrain · high relief", "Rugged terrain · high relief"]


def inside(x, y, ring):
    result = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            result = not result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geojson", type=Path)
    args = parser.parse_args()
    regions = []
    for feature in json.loads(args.geojson.read_text())["features"]:
        geometry = feature["geometry"]
        if not geometry:
            continue
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for rings in polygons:
            outer = rings[0]
            regions.append((min(p[0] for p in outer), min(p[1] for p in outer), max(p[0] for p in outer), max(p[1] for p in outer), rings, feature["properties"]))
    path = ROOT / "artifacts/mars/mars_types.json"
    catalog = json.loads(path.read_text())
    for terrain in catalog["types"]:
        terrain["display_name"] = NAMES[terrain["id"]]
        for site in terrain["earth_candidates"]:
            x, y = site["lon"], site["lat"]
            hits = [p for west, south, east, north, rings, p in regions if west <= x <= east and south <= y <= north and inside(x, y, rings[0]) and not any(inside(x, y, hole) for hole in rings[1:])]
            if not hits:
                raise ValueError(f"No administrative region for {site['id']}: {x}, {y}")
            p = hits[0]
            region, country = p.get("name_en") or p.get("name") or "", p.get("admin") or ""
            site["location"] = {"label": f"{region}, {country}" if region and region != country else country, "region": region, "country": country, "source": SOURCE}
            print(site["id"], site["location"]["label"])
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
