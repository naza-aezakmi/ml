# Implementation checks

Recorded **15 November 2026** using local production builds, Python 3.12, Node 22.23.5 and Chrome with software WebGL. [Saved results](validation-snapshot.json).

## What we checked

| Check | Result |
|---|---|
| API tests | 33 passed: known-plane slope, known residual roughness, SHAP, priorities, context, inputs and caching. |
| Experimental ML release | 3 passed: global type map and artifact integrity. |
| TypeScript experimental inference | 3 passed: catalogue lookup, coordinate conventions and repeated loading. |
| Search regression | Eight full responses matched before and after optimisation, including candidate order, measurements, countries and explanations. |
| Browser flow | Both planets returned ten candidates; explanations, deferred matrices and returning to the selected point worked. |
| First-visit guide | Cookie persistence, repeat visits, reopening and Escape were checked on desktop and mobile. |

The saved browser run used 1440 × 960 and 390 × 844 viewports, with no horizontal overflow in the checked mobile view.

## Repeat the checks

```bash
# api/; Python 3.12 and Node 22 available
python -m pytest -q

# ml/
python3 -m unittest discover -s tests -v
python3 scripts/audit_production.py --api ../api

# web/; Node 22 and pnpm 9.15.9
pnpm test:model
pnpm build
```

## Loading measurements

In the local Chrome comparison, first-visit JavaScript transferred over the network fell from **1,250,687 to 325,124 bytes**. This includes the guide after the change. Precompressed Brotli delivery and deferred results/matrix modules account for the reduction. It measures JavaScript bytes, not total page size or guaranteed load time.

Median time for four repeated, identical localhost searches fell from about **0.90 s to 0.003 s** with the 64-entry result cache. This applies to a warm worker and cached cell/weights; a new cell or initial atlas load still needs computation.

## What remains unvalidated

These checks establish software behaviour and mathematical consistency. They do not establish field-site suitability or rover passability.

We do not yet have an independent expert-labelled benchmark, measured precision@10 or field-test outcomes. [The method card](MODEL_CARD.md) describes the next steps.
