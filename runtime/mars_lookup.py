"""Portable inference from the trained Mars map. No ML runtime or network required."""
import json
import math
from functools import lru_cache
from pathlib import Path

ANALOG_NOTE = "Regional terrain screening. Class prototype values are not exact measurements at the clicked point. Soil properties and rover passability are not predicted."


def normalize_longitude(lon):
    return (lon + 180) % 360 - 180


@lru_cache(maxsize=1)
def load_model(directory):
    directory = Path(directory)
    catalog = json.loads((directory / "mars_types.json").read_text())
    labels = (directory / "mars_types_grid.bin").read_bytes()
    grid = catalog["grid"]
    ids = {t["id"] for t in catalog["types"]}
    if len(labels) != grid["width"] * grid["height"] or not set(labels) <= ids:
        raise ValueError("Invalid Mars model grid")
    return catalog, labels


def lookup_mars(lat, lon, directory):
    if not math.isfinite(lat) or not math.isfinite(lon) or abs(lat) > 90:
        raise ValueError("Invalid Mars coordinates")
    catalog, labels = load_model(str(directory))
    grid = catalog["grid"]
    lon = normalize_longitude(lon)
    row = min(grid["height"] - 1, math.floor((90 - lat) / grid["step_degrees"]))
    col = min(grid["width"] - 1, math.floor((lon + 180) / grid["step_degrees"]))
    terrain = next(t for t in catalog["types"] if t["id"] == labels[row * grid["width"] + col])
    climate = None
    c = catalog.get("mars_climate")
    if c:
        y = min(range(len(c["latitudes"])), key=lambda i: abs(c["latitudes"][i] - lat))
        x = min(range(len(c["longitudes"])), key=lambda i: abs(normalize_longitude(c["longitudes"][i] - lon)))
        climate = {"context": c["context"], "source": "MCD 6.2 model snapshot", "grid_lat": c["latitudes"][y], "grid_lon": c["longitudes"][x], "relative_humidity": None, "precipitation": None}
        climate.update({name: values[y][x] for name, values in c["fields"].items()})
    return {
        "body": "mars", "modelVersion": str(catalog["version"]), "point": {"lat": lat, "lon": lon},
        "terrainType": {k: v for k, v in terrain.items() if k != "earth_candidates"},
        "candidates": terrain["earth_candidates"], "climate": climate,
        "scale": {"gridStepDegrees": grid["step_degrees"], "descriptorRadiusKm": 50}, "note": ANALOG_NOTE,
    }
