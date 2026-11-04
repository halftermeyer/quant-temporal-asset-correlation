# FastPath on connected market data

**Question.** Given what we know at a month-end, which assets will move together over the next 60 trading days?
**Claim tested.** FastPath (Aura Graph Analytics) can use the *connections* between assets, not only each asset's own history.

**Result.** The same FastPath gets **+24% (99 assets) / +31% (S&P 500)** better when it reads the Moment graph
instead of per-event attributes. It then beats the original Pearson-30d demo and the sector map without
computing a single correlation. Trailing 120-day Pearson stays ahead by 0.02 to 0.04.

**Reproduce it:** [`notebooks/FastPath_connected_data_experiments.ipynb`](../notebooks/FastPath_connected_data_experiments.ipynb)
(every table below; outputs saved in the notebook). The stage version is
[`notebooks/FastPath_finance_demo.ipynb`](../notebooks/FastPath_finance_demo.ipynb).

---

## 1. Data and evaluation

| | 99 assets | S&P 500 |
|---|---|---|
| prices | old demo set, Jul 2023 to Jul 2025 | yfinance, Jan 2023 to Nov 2026, 498 tickers |
| graph | loaded in AuraDB | built in pandas, sent with `gds.graph.construct` |
| month-ends embedded / scored | 17 / 13 | 40 / 36 |

Everything is point-in-time. Every z-score at date *t* uses only data before *t*. FastPath sees only the 180 days
before each as-of date. 13F filings count only once they are public.

**Metric.** As of each month-end, every method ranks each asset's 10 nearest neighbours. Ground truth is the
Pearson correlation of daily log returns over the **next 60 trading days**, which no method sees.

- `fwd_corr@10`: mean forward correlation of the top-10 picks (the headline number)
- `hit@10`: share of the picks that are in the true forward top-10
- `hedge_err`: shortfall of the #1 pick against the best possible partner

`oracle` (the forward matrix itself) is the ceiling. `random` is the floor.

```python
# scoring.py: any method is a function (date, tickers, ctx) -> similarity matrix
F = ctx.future(t, C.FORWARD_DAYS)[tickers].corr().to_numpy()           # what happened next
S = method(t, tickers, ctx)                                             # what the method predicted at t
picks = np.argsort(-S, axis=1)[:, :k]
fwd_corr_at_k = np.take_along_axis(F, picks, axis=1).mean()
```

## 2. The graph

```
(:Snapshot {date, observationDay})-[:FIRST_EVENT]->(:MarketEvent)-[:NEXT]->(:MarketEvent) ...   one chain per asset
(:MarketEvent {day, moveBucket, excessBucket, volBucket, dayType, earnings, features[5]})
(:MarketEvent)-[:IN_MOMENT]->(:Moment {date, direction})        co-movement
(:MarketEvent)-[:HELD_BY]->(:Holder {cik, period})              13F ownership (enrichment)
```

**Events** (`scripts/02_build_events.py`) come from daily log returns, normalised by each asset's own trailing volatility:

```python
p["ret"] = np.log(p["adj_close"] / prev_close)
p["excess"] = p["ret"] - p.groupby("date")["ret"].transform("mean")     # vs the equal-weighted market
p["excess_z"] = trailing_z(p["excess"], t, C.VOL_WINDOW, center=False)  # 60 past days only
```

**Moments** are the connected part. A Moment is "day *d*, the assets that beat (or lagged) the market by ≥ 1σ".
Assets that keep landing in the same Moments are moving together. That co-movement now lives in the graph's
**connectivity**, not in a correlation matrix. The 99 assets have 915 Moments and 11,208 `IN_MOMENT` links.

**Holders** are the enrichment (`scripts/05_download_13f.py`, `scripts/06_ownership_graph.py`):
- Source: SEC Form 13F data sets from 2023 Q2 to 2026 Q3, with CUSIPs mapped to tickers through OpenFIGI plus issuer-name matching. 505 of 506 tickers are covered.
- One Holder node per concentrated book (≤ 200 positions) per quarter.
- The book links to every asset that is at least 2% of it *and* overweight against the all-managers book.
- The link starts on the first trading day after the 45-day filing deadline.

The rationale is that funds that bet on the same stocks trade them together (Anton & Polk 2014). On the 99 assets this gives 18,107 Holders and 98,331 links.

## 3. Key code

**Projection as of a date** (`fp.py`). Each as-of graph contains only the look-back window, so FastPath cannot see the future.
Moments and Holders are both projected under one label, `Context`, because FastPath takes a single context label:

```cypher
MATCH (s:Snapshot {date: $date})
CALL (s) {
    MATCH (e:MarketEvent {ticker: s.ticker})
    WHERE s.observationDay - $lookback < e.day <= s.observationDay
    WITH e ORDER BY e.day ASC LIMIT 1
    RETURN s AS src, e AS tgt, 'FIRST_EVENT' AS type
  UNION
    MATCH (e:MarketEvent {ticker: s.ticker})-[:NEXT]->(e2:MarketEvent)
    WHERE s.observationDay - $lookback < e.day AND e2.day <= s.observationDay
    RETURN e AS src, e2 AS tgt, 'NEXT' AS type
  UNION
    MATCH (e:MarketEvent {ticker: s.ticker})-[:IN_MOMENT]->(m:Moment)
    WHERE $withMoments AND s.observationDay - $lookback < e.day <= s.observationDay
    RETURN e AS src, m AS tgt, 'IN_MOMENT' AS type
  UNION
    MATCH (e:MarketEvent {ticker: s.ticker})-[:HELD_BY]->(h:Holder)
    WHERE $withHolders AND s.observationDay - $lookback < e.day <= s.observationDay
    RETURN e AS src, h AS tgt, 'HELD_BY' AS type
}
RETURN gds.graph.project.remote(src, tgt, {
  sourceNodeLabels: labels(src),
  targetNodeLabels: CASE WHEN tgt:Moment OR tgt:Holder THEN ['Context'] ELSE labels(tgt) END,
  ...   // day, observationDay, categorical buckets, features
  relationshipType: type })
```

**FastPath** (`fp.run_fastpath`). One switch separates "events" from "connected":

```python
gds.fast_path.mutate(
    G,
    base_node_label="Snapshot", event_node_label="MarketEvent", mutate_property="fastPathEmbedding",
    embedding_dimension=512, lookback_horizon=180, num_time_anchors=26,      # ~1 anchor per week
    first_relationship_type="FIRST_EVENT", next_relationship_type="NEXT",
    event_node_time_property="day", base_node_observation_time_property="observationDay",
    context_node_label="Context",                    # Moments (and Holders) one hop from each event
    event_node_categorical_properties=[],            # connected: per-event attributes OFF
    event_node_feature_vector_property=None,         #   (events only: the 5 buckets and the z-score vector)
    decay_rate=0.0, random_seed=42,
)
```

**Same graph without a database** (`fp.construct_as_of`, used for the S&P 500). The pandas build gives exactly
the counts of the Cypher projection on the 99 assets: 12,419 nodes and 15,117 relationships on 2025-03-31.

```python
G = gds.graph.construct(f"fpc_{date}",
    [snapshots[["nodeId", "labels", "observationDay", "sidx"]],
     events[["nodeId", "labels", "day", "features", *CATEGORICAL]],
     moments[["nodeId", "labels"]]],                                     # labels = ["Context"]
    [first_event, next_rels, in_moment])
```

**Every month-end, cached and resumable**:

```python
emb = fp.embed_dates(gds, dates, with_moments=True, cache=fp.cache_name(P), **P)   # results/<setting>.csv
```

## 4. Steps and results

### Step 1: baselines, no FastPath

| method | 99 assets | S&P 500 |
|---|---|---|
| oracle (ceiling) | 0.601 | 0.621 |
| pearson_120d | **0.512** | **0.470** |
| co_moment (Moment overlap, Jaccard) | 0.495 | 0.440 |
| pearson_30d (the original demo) | 0.471 | 0.415 |
| same_sector | 0.458 | 0.399 |
| co_owned (13F holder profiles) | 0.374 | 0.324 |
| co_holder (shared Holder nodes) | 0.369 | 0.321 |
| random (floor) | 0.253 | 0.200 |

On its own, the Moment graph (`co_moment`) already beats the original demo and the sector map. Ownership is a
real but weak signal, below sector. Blended with Pearson-120d at 25% weight it adds nothing (0.511 vs 0.512).

### Step 2: FastPath with everything on

The first version fed FastPath both the per-event attributes and the Moments. It scored **0.399** on the 99
assets, last of every real method. NVDA's neighbours were oil names: their *event profiles* look alike, but they
don't move on the same days.

### Step 3: ablation, what does FastPath actually use? (99 assets, `fwd_corr@10`)

| FastPath input | dim 256 | dim 512 |
|---|---|---|
| events only (attributes, no context) | 0.387 | 0.395 |
| events + Moments | 0.399 | |
| **Moments only (context, attributes off)** | 0.481 | **0.488** |

The per-event attributes ("normal day", volume bucket, …) are shared by nearly every event of every asset. They put a
large common component into each embedding and drown the context. With them off, FastPath works from the graph
alone: which assets moved together, and when.

Settings tried on the connected variant:
- dim 512: 0.488 (kept)
- smoothing rate 0.5, window 5: 0.479
- `decay_rate=1.0`: 0.321, because most embeddings come out zero with times in days

### Step 4: enrichment with 13F ownership (99 assets)

| FastPath input | `fwd_corr@10` |
|---|---|
| Moments (context only) | 0.481 |
| Moments + Holders (context only) | 0.405 |
| events + Holders | 0.372 |
| events + Moments + Holders | 0.382 |

Holders lower the score. Per window there are about 8 times more Holder links than Moment links, and ownership alone
predicts co-movement worse than sector. FastPath cannot weight one context type against another, so the weaker source
dominates. Two other enrichments were tested as cheap proxies before spending GDS time, and neither beat the Moments
as built:
- same-day peer assets as context (pattern comprehension in the projection): 0.479 at best
- denser Moments (0.5σ bands): 0.502 vs 0.495

### Step 5: the S&P 500

| method (498 assets, 36 month-ends) | `fwd_corr@10` | `hit@10` |
|---|---|---|
| pearson_120d | 0.470 | 0.299 |
| co_moment | 0.440 | 0.265 |
| **FastPath connected (dim 512)** | **0.430** | 0.250 |
| pearson_30d | 0.415 | 0.225 |
| same_sector | 0.399 | 0.218 |
| FastPath events only (dim 512) | 0.329 | 0.138 |

### Summary

| | events only → connected | vs pearson_30d | vs pearson_120d |
|---|---|---|---|
| 99 assets | 0.395 → 0.488 (**+24%**) | +0.017 | −0.024 |
| S&P 500 | 0.329 → 0.430 (**+31%**) | +0.015 | −0.040 |

## 5. Takeaways

1. **FastPath uses connected data.** The connected embedding gains about +0.1 forward correlation over events-only on both
   universes, at five times the scale on the S&P 500. It never computes a correlation.
2. **Context needs room.** Event attributes shared by every asset, or a large, weak context source (13F), hide the
   useful connections. Fewer, sharper context nodes work best.
3. **The bar is trailing 120-day Pearson.** It is still ahead. The honest pitch is "close to the best correlation baseline,
   ahead of the original demo, plus what a correlation matrix cannot do": time-aware embeddings in a vector index, clusters,
   and explanations by Cypher (*which Moments did these two assets share?*, section 10 of the demo notebook).
4. **Enrichment has to be tested before it goes into the graph.** Scoring each new source alone, as `co_moment`,
   `co_owned` and `co_holder` are scored, took minutes. Every FastPath variant took about 6 minutes on 99 assets and
   15 minutes on the S&P 500.

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env                                   # AuraDB + Aura API credentials

# 99 assets, loaded in AuraDB
python scripts/02_build_events.py && python scripts/04_load_neo4j.py --reset
python scripts/05_download_13f.py                      # SEC 13F, ~1.2 GB, shared by both universes
python scripts/06_ownership_graph.py --load            # Holder nodes + HELD_BY

# S&P 500, side by side in data/sp500 and results/sp500
FASTPATH_UNIVERSE=sp500 python scripts/01_download_prices.py --earnings
FASTPATH_UNIVERSE=sp500 python scripts/02_build_events.py
FASTPATH_UNIVERSE=sp500 python scripts/06_ownership_graph.py   # CSVs only, for co_holder

jupyter lab notebooks/FastPath_connected_data_experiments.ipynb
```

With empty `results/`, the notebook computes about 8 × 17 + 2 × 40 FastPath runs, roughly 1 h 30 on a 16 GB session.
With the cached CSVs it runs in about a minute.

## 7. Caveats

- **Survivorship bias:** the S&P 500 universe is today's constituents. The 99-asset set comes from the old demo.
- **Small samples:** 13 scored dates on 99 assets, which is why tuning stopped early. The S&P 500 (36 dates) is the more reliable read.
- **13F timing:** filings are about 45 days stale when they become public. Amendments and late filers are left out.
- **Transient aborts:** FastPath jobs are occasionally cancelled ("Remote job was aborted by the runtime") on a healthy
  session. `fp.embed_dates` retries the date.
