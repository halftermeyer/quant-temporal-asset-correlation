"""FastPath helpers used by the notebook: projection, run, stream, and the analysis steps.

The GDS parts need an attached Aura Graph Analytics session (`gds`). The analysis parts
(neighbours, drift, cluster naming) are plain pandas so they can be tested offline.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

import config as C

# the GDS client looks up db_node_properties with id(), which floods the output with
# deprecation notices from the server: harmless, so keep them quiet
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

FEATURE_DIM = 5
CATEGORICAL = ["moveBucket", "excessBucket", "volBucket", "dayType", "earnings"]

# One projection per observation date. Each Snapshot of that date is linked to the first event
# of ITS look-back window, and only events inside (obs - lookback, obs] are projected, so FastPath
# cannot see the future whatever its internal semantics are.
# FastPath takes one context label, so Moments (co-movement) and Holders (13F books, see
# scripts/06_ownership_graph.py) are both projected as `Context`, each with its own relationship type.
PROJECTION = """
CYPHER 25
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
WITH src, tgt, type, [i IN range(1, $featureDim) | 0.0] AS zeros
RETURN gds.graph.project.remote(src, tgt, {
  sourceNodeLabels: labels(src),
  targetNodeLabels: CASE WHEN tgt:Moment OR tgt:Holder THEN ['Context'] ELSE labels(tgt) END,
  sourceNodeProperties: {
    moveBucket: coalesce(src.moveBucket, -1), excessBucket: coalesce(src.excessBucket, -1),
    volBucket: coalesce(src.volBucket, -1), dayType: coalesce(src.dayType, -1),
    earnings: coalesce(src.earnings, -1),
    day: toFloat(coalesce(src.day, src.observationDay, 0)),
    observationDay: toFloat(coalesce(src.observationDay, 0)),
    sidx: coalesce(src.sidx, -1),
    features: coalesce(src.features, zeros)
  },
  targetNodeProperties: {
    moveBucket: coalesce(tgt.moveBucket, -1), excessBucket: coalesce(tgt.excessBucket, -1),
    volBucket: coalesce(tgt.volBucket, -1), dayType: coalesce(tgt.dayType, -1),
    earnings: coalesce(tgt.earnings, -1),
    day: toFloat(coalesce(tgt.day, tgt.observationDay, 0)),
    observationDay: toFloat(coalesce(tgt.observationDay, 0)),
    sidx: coalesce(tgt.sidx, -1),
    features: coalesce(tgt.features, zeros)
  },
  relationshipType: type
})
"""


def to_day(date: str) -> int:
    return (pd.Timestamp(date) - pd.Timestamp(C.EPOCH)).days


# ------------------------------------------------------------------ GDS (needs a session)
def project_as_of(gds, date: str, with_moments: bool = True, name: str | None = None,
                  with_holders: bool = False):
    name = name or f"fp_{date}{'' if with_moments else '_nomoments'}{'_holders' if with_holders else ''}"
    G, res = gds.graph.project.cypher(
        name, PROJECTION, overwrite=True, logging=False,
        query_parameters={"date": date, "lookback": C.LOOKBACK_DAYS,
                          "withMoments": with_moments, "withHolders": with_holders,
                          "featureDim": FEATURE_DIM})
    return G


_BUILT = {}


def _built() -> dict:
    """data/built CSVs (of the active universe) as frames, read once."""
    if not _BUILT:
        ev = pd.read_csv(C.BUILT / "events.csv")
        ev["features"] = ev["features"].map(lambda x: [float(v) for v in x.split(";")])
        snap = pd.read_csv(C.BUILT / "snapshots.csv")
        snap["sidx"] = range(len(snap))  # same rule as 04_load_neo4j.py
        _BUILT.update(events=ev, snapshots=snap, moments=pd.read_csv(C.BUILT / "moments.csv"),
                      next=pd.read_csv(C.BUILT / "rels_next.csv"), in_moment=pd.read_csv(C.BUILT / "rels_moment.csv"))
    return _BUILT


def construct_as_of(gds, date: str, with_moments: bool = True, name: str | None = None):
    """Same graph as project_as_of, built from data/built in pandas and sent with gds.graph.construct:
    for a universe that is not loaded in Neo4j (FASTPATH_UNIVERSE=sp500)."""
    b = _built()
    snap = b["snapshots"][b["snapshots"]["date"] == date]
    obs = snap.set_index("ticker")["observationDay"]
    ev = b["events"][b["events"]["ticker"].isin(obs.index)]
    o = ev["ticker"].map(obs)
    ev = ev[(ev["day"] > o - C.LOOKBACK_DAYS) & (ev["day"] <= o)].reset_index(drop=True)
    ev["nodeId"] = np.arange(len(ev)) + len(snap)
    eid = ev.set_index("eventId")["nodeId"]
    snap = snap.assign(nodeId=np.arange(len(snap)))
    first = ev.loc[ev.groupby("ticker")["day"].idxmin(), ["ticker", "nodeId"]]
    rels = [pd.DataFrame({"sourceNodeId": snap.set_index("ticker").loc[first["ticker"], "nodeId"].to_numpy(),
                          "targetNodeId": first["nodeId"].to_numpy(), "relationshipType": "FIRST_EVENT"})]
    nx = b["next"][b["next"]["src"].isin(eid.index) & b["next"]["dst"].isin(eid.index)]
    rels.append(pd.DataFrame({"sourceNodeId": eid[nx["src"]].to_numpy(), "targetNodeId": eid[nx["dst"]].to_numpy(),
                              "relationshipType": "NEXT"}))
    nodes = [snap.assign(labels=[["Snapshot"]] * len(snap))[["nodeId", "labels", "observationDay", "sidx"]]
             .astype({"observationDay": float}),
             ev.assign(labels=[["MarketEvent"]] * len(ev), day=ev["day"].astype(float))
             [["nodeId", "labels", "day", "features", *CATEGORICAL]]]
    if with_moments:
        im = b["in_moment"][b["in_moment"]["src"].isin(eid.index)]
        mom = pd.DataFrame({"momentId": im["dst"].unique()})
        mom["nodeId"] = np.arange(len(mom)) + len(snap) + len(ev)
        mid = mom.set_index("momentId")["nodeId"]
        nodes.append(mom.assign(labels=[["Context"]] * len(mom))[["nodeId", "labels"]])
        rels.append(pd.DataFrame({"sourceNodeId": eid[im["src"]].to_numpy(), "targetNodeId": mid[im["dst"]].to_numpy(),
                                  "relationshipType": "IN_MOMENT"}))
    return gds.graph.construct(name or f"fpc_{date}", nodes, rels, overwrite=True)


def run_fastpath(gds, G, with_moments: bool = True, dim: int = 64, seed: int = 42,
                 decay_rate: float = 0.0, prop: str = "fastPathEmbedding", with_holders: bool = False,
                 categorical: list | None = None, features: bool = True, **extra):
    """categorical / features: the per-event inputs (default: all of CATEGORICAL, and the feature
    vector). categorical=[] and features=False leave only the graph context, to see what it adds.
    extra: any other gds.fast_path setting (e.g. smoothing_rate, smoothing_window)."""
    if any(prop in p for p in G.node_properties().values()):
        gds.graph.node_properties.drop(G, [prop], fail_if_missing=False)
    return gds.fast_path.mutate(
        G,
        base_node_label="Snapshot",
        event_node_label="MarketEvent",
        mutate_property=prop,
        embedding_dimension=dim,
        lookback_horizon=C.LOOKBACK_DAYS,
        num_time_anchors=C.NUM_TIME_ANCHORS,
        first_relationship_type="FIRST_EVENT",
        next_relationship_type="NEXT",
        event_node_time_property="day",
        base_node_observation_time_property="observationDay",
        event_node_categorical_properties=CATEGORICAL if categorical is None else categorical,
        event_node_ignored_category=-1,
        context_node_label="Context" if with_moments or with_holders else None,
        event_node_feature_vector_property="features" if features else None,
        decay_rate=decay_rate,
        random_seed=seed,
        **extra,
    )


_SNAP = None


def _snapshots() -> pd.DataFrame:
    global _SNAP
    if _SNAP is None:
        _SNAP = pd.read_csv(C.BUILT / "snapshots.csv")[["ticker", "date"]]
        _SNAP.index.name = "sidx"  # 04_load_neo4j.py sets sidx = row number of this file
    return _SNAP


def stream_embeddings(gds, G, prop: str = "fastPathEmbedding") -> pd.DataFrame:
    df = gds.graph.node_properties.stream(G, ["sidx", prop], node_labels=["Snapshot"])
    if "nodeProperty" in df.columns:  # long format -> wide
        df = df.pivot(index="nodeId", columns="nodeProperty", values="propertyValue").reset_index()
    df = df.astype({"sidx": int}).join(_snapshots(), on="sidx")
    return df[["ticker", "date", prop]].rename(columns={prop: "embedding"})


def embed_dates(gds, dates, with_moments: bool = True, write_back: bool = False,
                cache: str | None = None, with_holders: bool = False, retries: int = 3,
                construct: bool = False, **kw) -> pd.DataFrame:
    """Project + FastPath for every date; returns ticker, date, embedding. Drops each projection.
    With `cache` (a file name in results/), dates already computed are loaded instead of re-run."""
    import json
    path = C.RESULTS / cache if cache else None
    done = pd.DataFrame(columns=["ticker", "date", "embedding"])
    if path is not None and path.exists():
        done = pd.read_csv(path)
        done["embedding"] = done["embedding"].map(json.loads)
    todo = [d for d in dates if d not in set(done["date"])]
    print(f"{len(dates) - len(todo)} dates cached, {len(todo)} to compute")
    out = [done[done["date"].isin(dates)]]
    for i, d in enumerate(todo, 1):
        for attempt in range(1, retries + 1):
            t0 = time.perf_counter()
            G = (construct_as_of(gds, d, with_moments) if construct
                 else project_as_of(gds, d, with_moments, with_holders=with_holders))
            t1 = time.perf_counter()
            n_nodes = G.node_count()
            try:
                r = run_fastpath(gds, G, with_moments, with_holders=with_holders, **kw)
                break
            except Exception as exc:  # the session sometimes aborts a job ("aborted by the runtime"): redo the date
                if "aborted" not in str(exc) or attempt == retries:
                    raise
                print(f"[{i:>2}/{len(todo)}] {d}  job aborted by the session, retry {attempt}/{retries - 1}", flush=True)
                time.sleep(10)
        t2 = time.perf_counter()
        e = stream_embeddings(gds, G)
        if write_back:
            gds.graph.node_properties.write(G, ["fastPathEmbedding"], node_labels=["Snapshot"])
        gds.graph.drop(G)
        t3 = time.perf_counter()
        out.append(e)
        if path is not None:  # save as we go, so an interrupted run resumes
            C.RESULTS.mkdir(exist_ok=True)
            e.assign(embedding=e["embedding"].map(lambda v: json.dumps([round(float(x), 6) for x in v]))) \
             .to_csv(path, mode="a", header=not path.exists(), index=False)
        print(f"[{i:>2}/{len(todo)}] {d}  {n_nodes:>7,} nodes  {r.node_properties_written:>4} emb  "
              f"project {t1 - t0:5.1f}s  fastpath {t2 - t1:5.1f}s  stream+drop {t3 - t2:5.1f}s", flush=True)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ analysis (pandas only)
def similarity(emb: pd.DataFrame, date: str) -> pd.DataFrame:
    e = emb[emb["date"] == date].set_index("ticker")["embedding"]
    X = np.vstack(e.map(np.asarray).to_numpy())
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return pd.DataFrame(X @ X.T, index=e.index, columns=e.index)


def neighbours(emb: pd.DataFrame, date: str, ticker: str, k: int = 10) -> pd.Series:
    s = similarity(emb, date)[ticker].drop(ticker)
    return s.sort_values(ascending=False).head(k)


def drift(emb: pd.DataFrame, d0: str, d1: str, k: int = 10) -> pd.DataFrame:
    """Which assets changed neighbourhood between two dates? (Jaccard of top-k sets, so it does
    not need the two embedding spaces to be aligned.)"""
    S0, S1 = similarity(emb, d0), similarity(emb, d1)
    common = S0.index.intersection(S1.index)
    rows = []
    for t in common:
        n0 = set(S0.loc[t, common].drop(t).nlargest(k).index)
        n1 = set(S1.loc[t, common].drop(t).nlargest(k).index)
        rows.append({"ticker": t, "overlap": len(n0 & n1) / k,
                     "left": ", ".join(sorted(n0 - n1)[:5]), "joined": ", ".join(sorted(n1 - n0)[:5])})
    return pd.DataFrame(rows).sort_values("overlap").reset_index(drop=True)


def name_clusters(clusters: pd.DataFrame, assets: pd.DataFrame, member_moments: pd.DataFrame,
                  min_share: float = 0.5) -> pd.DataFrame:
    """Name each cluster AFTER the fact. Sector was never an input, so it is fair to use it here.

    clusters        ticker, cluster
    assets          ticker, sector, subIndustry
    member_moments  ticker, momentId (look-back window of the date)
    """
    df = clusters.merge(assets, on="ticker")
    n_all = len(df)
    rows = []
    for cl, g in df.groupby("cluster"):
        sub = g["subIndustry"].value_counts()
        lift = (sub / len(g)) / (df["subIndustry"].value_counts()[sub.index] / n_all)
        top_sub = [f"{s} ({sub[s]})" for s in lift[sub >= 2].sort_values(ascending=False).index[:3]]
        mm = member_moments[member_moments["ticker"].isin(g["ticker"])]
        share = mm.groupby("momentId")["ticker"].nunique() / len(g)
        base = member_moments.groupby("momentId")["ticker"].nunique() / n_all
        sig = (share[share >= min_share] / base.reindex(share.index)).sort_values(ascending=False)
        rows.append({"cluster": cl, "size": len(g),
                     "sectors": ", ".join(f"{k} {v}" for k, v in g["sector"].value_counts().head(3).items()),
                     "distinctive sub-industries": "; ".join(top_sub),
                     "signature moments": ", ".join(sig.index[:4]),
                     "examples": ", ".join(g["ticker"].head(6))})
    return pd.DataFrame(rows).set_index("cluster")


# ------------------------------------------------------------------ variants (one cache file per setting)
def cache_name(params: dict, with_moments: bool = True, with_holders: bool = False) -> str:
    p = {"dim": 64, "decay_rate": 0.0, "seed": 42, **params}
    inputs = ("" if p.get("categorical") is None else "_cat" + "-".join(p["categorical"])) + \
             ("" if p.get("features", True) else "_nofeat") + \
             "".join(f"_{k}{v}" for k, v in sorted(p.items())
                     if k not in ("dim", "decay_rate", "seed", "categorical", "features"))
    return (f"fastpath_dim{p['dim']}_decay{p['decay_rate']}_seed{p['seed']}{inputs}"
            f"{'' if with_moments else '_nomoments'}{'_holders' if with_holders else ''}.csv")


def load_variants() -> dict:
    """Every cached FastPath run in results/, keyed by its setting (e.g. 'dim256_decay0.0_seed42')."""
    import json
    out = {}
    for f in sorted(C.RESULTS.glob("fastpath_dim*.csv")):
        e = pd.read_csv(f)
        e["embedding"] = e["embedding"].map(json.loads)
        out[f.stem.removeprefix("fastpath_")] = e
    return out
