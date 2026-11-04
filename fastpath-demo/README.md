# FastPath on markets: who will move together next?

> **Write-up with approach, code and results:** [`docs/FastPath_connected_data.md`](docs/FastPath_connected_data.md)
> **Reproducible experiments:** [`notebooks/FastPath_connected_data_experiments.ipynb`](notebooks/FastPath_connected_data_experiments.ipynb)
> **Stage demo:** [`notebooks/FastPath_finance_demo.ipynb`](notebooks/FastPath_finance_demo.ipynb)
> **The original Pearson / KNN demo** (the baseline here): [`../pearson-demo`](../pearson-demo)

Prices are turned into a **connected event graph**, and FastPath (Aura Graph Analytics) embeds
each asset's recent path, as of a date. The demo then checks, **out of sample**, whether the
embedding's nearest neighbours co-move over the next 60 trading days better than the obvious
methods. The old KNN-on-Pearson demo becomes the baseline.

## The graph

```
(:Asset)-[:HAS_SNAPSHOT]->(:Snapshot {date, observationDay})      FastPath base node, one per month-end
(:Asset)-[:FIRST_EVENT]->(:MarketEvent)-[:NEXT]->(:MarketEvent)…  FastPath event chain, one per trading day
(:MarketEvent)-[:IN_MOMENT]->(:Moment {date, direction})          FastPath context node
```

| FastPath input | What it carries |
|---|---|
| categorical | `moveBucket` (return z: ≤-2σ … ≥+2σ), `excessBucket` (vs equal-weighted market), `volBucket`, `dayType` (normal / FOMC / market shock / both), `earnings` |
| feature vector | `[ret_z, excess_z, vol_z, range_z, gap_z]`, clipped and scaled |
| context | **Moment** = "day d, the assets that beat (or lagged) the market by ≥ 1σ". Assets that keep landing in the same Moments get close. That puts co-movement into the graph's connectivity, not into a vector-to-vector correlation. Projected under the label `Context` (FastPath takes one context label) |
| time | `day` on events and `observationDay` on snapshots (integer days since 2020-01-01), 180-day look-back |

Everything is point-in-time: every z-score at date t uses only data before t. The notebook projects
**one graph per as-of date** that contains only the events inside the look-back window, so FastPath
cannot see the future whatever its internal semantics. Sector is never an input. It is only used
afterwards to name the clusters.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env                                     # AuraDB + Aura API credentials

python scripts/01_download_prices.py --earnings          # S&P 500 from yfinance (run on your Mac)
#   or: python scripts/01_download_prices.py --from-legacy path/to/real_asset_prices.csv
python scripts/02_build_events.py                        # events, Moments, snapshots -> data/built
python scripts/03_score_baselines.py                     # the bar to beat -> results/baselines.csv
python scripts/04_load_neo4j.py --reset                  # load AuraDB
jupyter lab notebooks/FastPath_finance_demo.ipynb
```

`data/` currently holds the 99-ticker set of the old demo (Jul 2023 to Jul 2025) so you can
try the whole pipeline before the S&P 500 download. Running `01` replaces it.

## Notebook flow (about 12 minutes)

1. Connect and start a session
2. The graph: NVDA through the April 2025 tariff week, and the biggest Moments
3. The bar to beat (baselines)
4. FastPath **as of 2025-03-31**, live, with a stopwatch
5. NVDA's neighbours, FastPath vs Pearson 30d, next to what happened next
6. All month-ends, **with and without Moments** (the ablation tells you what the connected space adds)
7. **The reveal**: scores table plus a per-date chart
8. k-means clusters, named after the fact (sub-industries and signature Moments)
9. Who changed neighbourhood across the shock
10. Why two assets are neighbours: their shared Moments, in Cypher
11. Hedge or substitute finder through a vector index on `Snapshot.fastPathEmbedding`

## Results: what the graph adds to FastPath

Out of sample, 60-day forward window, top-10 neighbours. Same FastPath (dim 512, no decay, 180-day
look-back, 26 time anchors), two inputs:
- **connected**: the event chain plus the Moments each event belongs to; per-event attributes off
- **events only**: the event chain plus each event's attributes (buckets, flags, 5 z-scores); no context

| method | 99 assets (13 dates) | S&P 500 (36 dates) | what it is |
|---|---|---|---|
| oracle | 0.601 | 0.621 | the forward matrix itself (ceiling) |
| pearson_120d | 0.512 | 0.470 | trailing 120-day correlation |
| co_moment | 0.495 | 0.440 | Moment overlap (Jaccard), no FastPath |
| **fastpath (connected)** | **0.488** | **0.430** | |
| pearson_30d | 0.471 | 0.415 | the original demo |
| same_sector | 0.458 | 0.399 | GICS sub-industry, then sector |
| fastpath_events_only | 0.395 | 0.329 | |
| random | 0.253 | 0.200 | floor |

The story: connectivity is worth **+24% (99) / +31% (S&P 500)** to FastPath, which then beats the original
Pearson demo and the sector map without ever computing a correlation. Trailing 120-day Pearson stays ahead.
Turning the per-event attributes back on *with* Moments gives 0.399 on the 99 assets: they put a component
shared by every asset into each embedding and drown the context.

Tried and dropped (scripts kept): 13F common ownership (`05_download_13f.py`, `06_ownership_graph.py`,
`S.co_owned`, `S.co_holder`) scores 0.37 alone on the 99 assets and lowers FastPath to 0.405 when its
Holder nodes are added as context (about 8x more links than Moments, weaker signal); same-day peer
context and denser Moments do not beat the Moments as built.

## The S&P 500 next to the 99 assets

`FASTPATH_UNIVERSE=sp500` points every script at `data/sp500/` and `results/sp500/` instead:

```bash
FASTPATH_UNIVERSE=sp500 python scripts/01_download_prices.py --earnings
FASTPATH_UNIVERSE=sp500 python scripts/02_build_events.py
```

AuraDB keeps the 99-asset graph. For the S&P 500, `fp.embed_dates(..., construct=True)` builds each as-of
graph from `data/sp500/built` in pandas (same shape as the Cypher projection) and sends it to the session
with `gds.graph.construct`.

## Open questions for the FastPath team

- Can several base nodes share one event chain with different `observationDay`s? If yes, all
  snapshots fit in one projection instead of one projection per date, which is much faster.
- Does `lookback_horizon` drop events *after* the observation time, or only events that are too old?
  The per-date windows make the demo safe either way.
- How are context nodes used: one hop from events, with any relationship type? Context-only runs
  (no categorical, no feature vector) work well here, so `IN_MOMENT` to `Context` is picked up.
- Can context and event attributes be weighted? With both on, the attributes drown the context.
- `decay_rate=1.0` zeroes most embeddings with day-valued times: what unit is the rate in?
- Jobs are sometimes cancelled with "Remote job was aborted by the runtime" on a healthy session;
  `fp.embed_dates` retries the date.
- Are embeddings from two runs with the same seed in the same space? The drift step avoids
  relying on it by comparing neighbour sets.

## Caveats

- Today's S&P 500 constituents → survivorship bias.
- FOMC dates are a hand-maintained file (`data/reference/fomc_dates.csv`). Check 2026.
- Earnings dates from yfinance are partial for older quarters.
