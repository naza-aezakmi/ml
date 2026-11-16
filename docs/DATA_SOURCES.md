# Data sources and coverage

These counts describe the **delivered API atlas**, rather than complete mission archives. [Method](MODEL_CARD.md) · [Machine-readable audit](production-data-audit.json).

## Terrain used in the ranking

| Body | Product | Native grid | Stored windows | Height samples |
|---|---|---|---:|---:|
| Moon | [NASA LRO LOLA, LDEM_64](https://ode.rsl.wustl.edu/mars/pagehelp/Content/Missions_Instruments/LRO/LOLA/GDR/GDRDEM.htm) | 64 pixels/degree; about 474 m at the equator | 380,956 | 9,523,900 |
| Mars | [NASA MGS MOLA, MEGDR 32](https://pds-geosciences.wustl.edu/missions/mgs/megdr.html) | 32 pixels/degree; about 1,853 m at the equator | 1,446,856 | 36,171,400 |
| Earth | [NOAA ETOPO 2022, 60 arc-seconds](https://www.ncei.noaa.gov/products/etopo-global-relief-model) | About 1,853 m at the equator | 1,400,048 before the water screen | 35,001,200 |

The Earth address grid has 5,106,644 centres. Profiles are stored for land windows; the water screen leaves **1,356,072 eligible centres**, including 85,381 in the US group.

There are **80,696,500 stored height samples** in total. These are derived records. Source DEM pixels can be reused, so this is not a count of independent measurements or training examples.

The atlas contains 127 compressed terrain tiles: Moon 20 files / 10,107,629 bytes; Mars 39 / 27,918,420 bytes; Earth 68 / 24,349,673 bytes. Sizes exclude the index and masks.

## Common comparison scale

Each window has 25 samples at 2.5 km spacing, covering about 10 × 10 km. Angular pixel size varies in physical distance with latitude. Resampling does not create finer detail than the source.

The EDA uses **3,183,884 eligible windows** across the three bodies. Statistics describe windows, not exact fractions of planetary surface area. Detailed HiRISE, LROC and Copernicus DEMs are not inputs to the live ranking.

## Water screening and location labels

| Source | Use |
|---|---|
| [Natural Earth physical vectors, 1:10m](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) | Regional coastlines and lake screening, encoded in `field-screen-v1.gz`. |
| [Natural Earth countries, 1:50m](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_admin_0_countries.geojson) | Country or territory of each returned window centre. Bundled with provenance under `artifacts/environment/`. |
| [Natural Earth admin-1 polygons, 1:10m](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_admin_1_states_provinces.geojson) | Regional labels in the experimental Mars catalogue. |
| [USGS/IAU Gazetteer](https://planetarynames.wr.usgs.gov/GIS_Downloads) | Named lunar and Martian features for map labels and search. |

The water screen handles major mapped water, not every stream or seasonal flood. Country labels describe the centre; a window can cross a boundary. Unknown labels remain unknown.

## Environmental passports: separate from ranking

| Source | What we show | Period and scale |
|---|---|---|
| [NASA POWER Climatology / MERRA-2](https://power.larc.nasa.gov/docs/services/api/temporal/climatology/) | Earth air and skin temperature, monthly temperature range, humidity, precipitation, 10 m wind and pressure. Parameters: T2M, TS, RH2M, PRECTOTCORR, WS10M, PS. | Gridded climate averages at the candidate centre. The period comes from the response header; the checked response covers 2001–2025. Monthly means are not daily extremes. |
| [Mars Climate Database 6.2](https://www-mars.lmd.jussieu.fr/) | Mars surface and atmospheric temperature, pressure and wind. | Bundled 48 × 64 grid, nearest node. One scenario: Ls = 0°, local noon, average solar activity; atmospheric fields at 10 m. Model output, not current weather or an annual average. |
| [NASA LRO Diviner Global Cumulative Product](https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gcp/) | Minimum, maximum and span of available mean bolometric surface temperatures, with coverage. | Observations from 2009-07-05 to 2015-04-01. Spatial bins of 0.5° and 96 quarter-hour bins of lunar local time. Incomplete coverage limits the range. |

These sources measure different quantities, heights and periods. They are shown for planning but **never mixed into the terrain score**. Missing values stay null. Missing Mars humidity or precipitation is not replaced by zero.

POWER is queried by point. Diviner uses a validated HTTP range request for the relevant 10,848-byte block. Passports load after terrain results, so an unavailable climate source does not block search.

## Images and earlier experiments

| Source | Use |
|---|---|
| [NASA Trek](https://trek.nasa.gov/) | Moon and Mars globe imagery. [Specific products and credits](https://github.com/naza-aezakmi/web/blob/main/public/textures/SOURCES.md). |
| [Esri World Imagery](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer) | Earth satellite view, with imagery-provider attribution in the app. |
| [ESA WorldCover 2021 v200](https://esa-worldcover.org/en) | Land-cover context and bare/sparse-cover screening in the separate KMeans catalogue. |

Images are visual context, not score inputs. WorldCover screening in the experimental catalogue is not applied to the live candidate pool. [Experimental model](EXPERIMENTAL_MODEL.md).

## Reproduce the counts

From `ml/`, with `api/` beside it:

```bash
python3 scripts/audit_production.py --api ../api
```

To save an updated snapshot, add `--output docs/production-data-audit.json`. The audit checks tile presence, counts records, decodes water-screen flags and records SHA-256 hashes for the index, mask and worker. It does not independently validate the original scientific DEMs.
