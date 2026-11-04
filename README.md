# Temporal asset correlation with Neo4j

Which assets move together, as of a given date? Two demos, the second one built on the first.

| | [`pearson-demo/`](pearson-demo) | [`fastpath-demo/`](fastpath-demo) |
|---|---|---|
| idea | daily prices chained in time, KNN on trailing Pearson correlation of log returns | prices turned into events and shared **Moments**, embedded with **FastPath** (Aura Graph Analytics) |
| as-of date | sliding window of `AssetDay` nodes | one projection per month-end, 180-day look-back |
| output | `CORRELATES_WITH` relationships | a time-aware embedding per asset and date: neighbours, clusters, vector index |
| evaluation | none | out of sample: correlation realised over the **next 60 trading days** |
| status | kept for the record | current |

## FastPath demo: headline result

The same FastPath improves by **+24% (99 assets) / +31% (S&P 500)** when it reads the Moment graph instead of
per-event attributes. It then beats the original 30-day Pearson demo and the sector map without computing a
correlation. Trailing 120-day Pearson stays ahead.

| forward 60-day correlation of the top-10 picks | 99 assets | S&P 500 |
|---|---|---|
| Pearson 120d | 0.512 | 0.470 |
| **FastPath on the Moment graph** | **0.488** | **0.430** |
| Pearson 30d (the original demo) | 0.471 | 0.415 |
| same sector | 0.458 | 0.399 |
| FastPath on event attributes only | 0.395 | 0.329 |

- Write-up (approach, key code, every step, 13F ownership enrichment): [`fastpath-demo/docs/FastPath_connected_data.md`](fastpath-demo/docs/FastPath_connected_data.md)
- Reproducible notebook: [`fastpath-demo/notebooks/FastPath_connected_data_experiments.ipynb`](fastpath-demo/notebooks/FastPath_connected_data_experiments.ipynb)
- Setup and scripts: [`fastpath-demo/README.md`](fastpath-demo/README.md)
