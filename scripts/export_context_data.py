"""Export the already documented Mars Climate Database snapshot for passports."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    climate = json.loads(args.catalog.read_text())["mars_climate"]
    for directory in [ROOT / "artifacts/environment", ROOT.parent / "api/app/data/environment"]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mars_climate.json").write_text(json.dumps(climate, separators=(",", ":")) + "\n")
    print("Exported MCD 6.2 snapshot")


if __name__ == "__main__":
    main()
