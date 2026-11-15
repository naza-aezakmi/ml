"""Vendor one portable ML release into sibling web/api repositories; offline and deterministic."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["mars_types.json", "mars_types_grid.bin"]
CONTEXT_FILES = ["countries.json.gz", "mars_climate.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", type=Path)
    parser.add_argument("--api", type=Path)
    args = parser.parse_args()
    artifacts = ROOT / "artifacts/mars"
    manifest = {"model": "Mars terrain analogs", "format": 1, "files": {name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest() for name in FILES}}
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for target, subdir in [(args.web, "public/ml/mars"), (args.api, "app/data/mars")]:
        if target:
            destination = target.resolve() / subdir
            destination.mkdir(parents=True, exist_ok=True)
            for name in FILES + ["manifest.json"]:
                shutil.copy2(artifacts / name, destination / name)
            print("Exported", destination)
    if args.web:
        shutil.copy2(ROOT / "runtime/marsAnalog.ts", args.web / "shared/marsAnalog.ts")
    if args.api:
        shutil.copy2(ROOT / "runtime/mars_lookup.py", args.api / "app/mars_lookup.py")
        context = ROOT / "artifacts/environment"
        if all((context / name).exists() for name in CONTEXT_FILES):
            context_manifest = {
                "purpose": "Country labels and separate environmental passports; outside terrain ranking",
                "files": {name: hashlib.sha256((context / name).read_bytes()).hexdigest() for name in CONTEXT_FILES},
            }
            (context / "manifest.json").write_text(json.dumps(context_manifest, indent=2) + "\n")
            destination = args.api / "app/data/environment"
            destination.mkdir(parents=True, exist_ok=True)
            for name in CONTEXT_FILES + ["manifest.json"]:
                shutil.copy2(context / name, destination / name)
            print("Exported contextual data", destination)


if __name__ == "__main__":
    main()
