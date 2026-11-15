"""Check the complete shipped map and release integrity without sklearn."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_environment_manifest(self):
        directory = ROOT / "artifacts/environment"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)

    def test_manifest(self):
        directory = ROOT / "artifacts/mars"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)

    def test_complete_global_map(self):
        directory = ROOT / "artifacts/mars"
        catalog = json.loads((directory / "mars_types.json").read_text())
        labels = (directory / "mars_types_grid.bin").read_bytes()
        self.assertEqual(len(labels), catalog["grid"]["width"] * catalog["grid"]["height"])
        self.assertEqual(set(labels), set(range(10)))
        for terrain in catalog["types"]:
            self.assertEqual(len(terrain["earth_candidates"]), 3)
            for site in terrain["earth_candidates"]:
                self.assertTrue(site["location"]["label"])
                self.assertEqual(len(site["match_explanation"]["comparisons"]), 3)


if __name__ == "__main__":
    unittest.main()
