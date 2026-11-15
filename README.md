# Mars → Earth terrain analog model

Trained MiniBatchKMeans model, complete global Mars type map, 30 Earth candidates, exact distance explanations, and portable browser/Python inference. The website can answer a click without a Python server or a new nearest-neighbor search.

## What is shipped

- `models/mars_terrain_types.joblib`: trained `scaler`, `model`, and type-ID `remap` (scikit-learn 1.6.1). Load only this trusted artifact. Not required by the website.
- `artifacts/mars/mars_types_grid.bin`: all 720 × 360 type IDs, one byte per 0.5° cell, north to south and west to east from −180°.
- `artifacts/mars/mars_types.json`: ten prototypes, three Earth places per type, regional administrative labels, terrain comparisons, land cover, and separate climate context.
- `artifacts/mars/manifest.json`: SHA-256 checksums of the inference assets.
- `runtime/`: matching TypeScript and Python inference implementations.
- `training/`: original training, extraction and explanation scripts. Large source DEMs, caches and reports are excluded from Git.

## Integrate with the sibling repositories

From this repository, with `web`, `api`, and `ml` in the same parent folder:

```bash
python3 scripts/export.py --web ../web --api ../api
python3 -m unittest discover -s tests -v
```

This copies one release into `web/public/ml/mars`, `web/shared/marsAnalog.ts`, `api/app/data/mars`, and `api/app/mars_lookup.py`. Commit those exported files to their respective repositories. Neither Docker build needs a sibling checkout or network access to ML assets. Edit inference in `ml/runtime` and export again instead of maintaining separate copies.

Web: `loadMarsModel()` fetches the two assets once per page, then `model.lookup(lat, lon)` returns the type, named Earth candidates, feature evidence, and climate context. The existing globe already emits degrees north/east; its local `−Z` axis points east. Do not apply a different 3D conversion to these emitted coordinates.

API: `POST /terrain/analogs` accepts `{"planet":"mars","lat":18.44,"lon":77.45}`. It reads the same assets locally with no sklearn dependency. The current web card performs inference in the browser, so it does not depend on deploying this endpoint first. The existing `/terrain/search` method and lunar flow are retained.

## Model and selection method

Mars terrain comes from NASA MOLA MEGDR32; Earth terrain from NOAA ETOPO2022 60s. Regional descriptors use 81 neighboring cells of the common approximately 10 km sampling grid, typically about 50 km radius, excluding insufficient neighborhoods. Three features: median slope, 90th percentile slope, and p95−p05 elevation relief. These are regional descriptors, not wheel-scale slopes or obstacles.

Features are transformed with `log1p` and a Mars-fitted RobustScaler. MiniBatchKMeans uses seed 42 and 150,000 area-sampled Mars observations. Candidate k values were 6, 10 and 14; k=10 is the demo's chosen granularity, not a discovery of ten natural terrain classes. All cells in one type share its fixed Earth catalog.

Earth retrieval computes exact Euclidean distance to the **trained type centroid** in this shared feature space across 1,323,499 eligible Earth locations. Three geographically separated candidates are chosen with separation over 500 km from the nearest 20,000. Within that fixed shortlist, candidates with ≥50% bare/sparse cover are displayed first. Display position is not the raw terrain rank. The catalog includes both.

Per-feature squared differences sum to squared distance exactly. Explanation bars show contributions to remaining mismatch, not confidence or causal importance. No SHAP model is used. The comparison table uses the inverse-transformed trained centroid; it does not substitute the median of class members.

## Sources and limits

- [MOLA MEGDR](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html); [ETOPO2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model).
- [ESA WorldCover](https://esa-worldcover.org/en): 2021 cover, approximately 100 m screening in the catalog; some polar points lack cover data.
- [NASA POWER](https://power.larc.nasa.gov/): Earth long-term climatology; air temperature at 2 m, humidity and precipitation.
- [Mars Climate Database](https://www-mars.lmd.jussieu.fr/): MCD 6.2 coarse snapshot at Ls=0°, local noon, average solar scenario; atmospheric fields at 10 m. Not current weather or an annual mean. Missing Mars humidity and precipitation remain null.
- [Natural Earth admin-1 boundaries](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_admin_1_states_provinces.geojson): regional labels, not exact addresses or current legal boundary assertions. `scripts/enrich_locations.py` rebuilds labels from a downloaded source GeoJSON.

Climate is displayed separately and does not affect terrain matching. Some top terrain matches are forested or icy. Neither material composition nor rover passability has been learned; sites are candidates for regional screening, not approved testing grounds. The 0.5° click map approximates the original sampling grid. No trained Moon model is included.

## Training source

For the existing workspace, the processed training inputs remain in the parent `data/` directory. To rerun the copied pipeline, put the required source/cache folders under `training/data/` (or link that directory locally), install `training/requirements-terrain.txt` in an isolated environment, and run the scripts from `training/`:

```bash
python global_terrain_data.py --help
python global_terrain_extract.py --help
python mars_types.py
```

`mars_types.py` consumes `data/global_terrain/{mars,earth}_cells.npz` and writes raw model/catalog outputs. Climate and cover enrichment additionally require their downloaded inputs, documented in the scripts. The checked-in inference release already includes those results. Do not silently overwrite a published release with an unenriched raw catalog; validate and export after enrichment and explanation generation.
