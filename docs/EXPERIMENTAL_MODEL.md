# The KMeans experiment

We tried a fast alternative for arbitrary clicks on Mars: group terrain into types, then attach a fixed Earth catalogue to each type. Every click in one type returns the same three Earth candidates.

The trained model and global lookup map are included here. The current website uses the separate [four-criterion terrain search](MODEL_CARD.md), which supports both the Moon and Mars.

## Training data and features

We used NASA MOLA MEGDR32 for Mars and NOAA ETOPO 2022 for Earth. Each descriptor summarises 81 nearby cells on the common approximately 10 km grid, typically within about 50 km.

The three features were median slope, 90th-percentile slope and p95−p05 elevation relief. These describe regional morphology, not wheel-scale obstacles.

We transformed the features with **log1p**, fitted **RobustScaler on Mars**, then trained **MiniBatchKMeans** on **150,000 area-sampled Mars vectors**, with seed 42.

| Number of clusters | Silhouette score |
|---|---:|
| 6 | 0.410 |
| 10 | 0.356 |
| 14 | 0.312 |

The diagnostics used 2,000 sampled training vectors. Six clusters had the highest silhouette score. We kept ten to show more terrain detail in the prototype; this does not establish ten natural terrain classes. There was no held-out field-site validation.

## How Earth places were chosen

We transformed Earth descriptors with the same Mars-fitted scaler and calculated Euclidean distance to each **trained cluster centroid** across **1,323,499 eligible Earth locations**.

For each type, we selected three candidates from the nearest 20,000, separated by more than 500 km. Candidates with at least 50% bare/sparse WorldCover cover were then displayed first within that shortlist. The catalogue preserves the original terrain ranks as well as display order.

Each explanation shows the physical measurements and the three squared-distance contributions. Those contributions sum to squared distance; they are a direct decomposition, not the live search's SHAP explanation.

## Included files

| Path | Contents |
|---|---|
| `models/mars_terrain_types.joblib` | Trained scaler, clustering model and type-ID remap; scikit-learn 1.6.1. |
| `artifacts/mars/mars_types_grid.bin` | Global 720 × 360 type map, one byte per 0.5° cell. |
| `artifacts/mars/mars_types.json` | Ten prototypes, thirty Earth candidates and supporting context. |
| `artifacts/mars/manifest.json` | Inference-asset SHA-256 checksums. |
| `runtime/` | Python and TypeScript lookup implementations. |
| `training/` | Data preparation, training and explanation scripts. |

The type grid runs north to south and west to east from −180°. The 0.5° map approximates the original sampling grid. No trained Moon clustering model is included.

## Run and export

From `ml/`, with `web/` and `api/` next to it:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/export.py --web ../web --api ../api
```

This synchronises the saved experimental release and environmental artifacts into the sibling repositories. Both applications can use the bundled files without a model server or scikit-learn at runtime.

The experimental API endpoint is `POST /terrain/analogs`:

```json
{"planet": "mars", "lat": 18.44, "lon": 77.45}
```

For retraining, install `training/requirements-terrain.txt` in an isolated environment. Run the preparation scripts from `training/`; `mars_types.py` expects `data/global_terrain/mars_cells.npz` and `earth_cells.npz`. Large DEMs, caches and processed inputs are excluded from Git and must be prepared first. Cover and climate enrichment need their documented source inputs before exporting a replacement release.

## Sources and interpretation

Terrain: [MOLA](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html) and [ETOPO](https://www.ncei.noaa.gov/products/etopo-global-relief-model). Cover: [ESA WorldCover 2021](https://esa-worldcover.org/en), screened at approximately 100 m. Climate: [NASA POWER](https://power.larc.nasa.gov/) and [Mars Climate Database 6.2](https://www-mars.lmd.jussieu.fr/). [Source details](DATA_SOURCES.md).

Climate is separate from matching. Some candidates are forested or icy, and some polar points lack cover data. The model groups regional terrain; it has not learned soil composition or rover passability.
