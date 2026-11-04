"""The reveal: score any neighbour-finding method against what happened *next*.

For each snapshot date t and asset a, a method ranks the other assets using only data up to t.
Ground truth is the Pearson correlation of daily returns over the FORWARD_DAYS trading days
after t, which no method has seen.

Metrics, averaged over every (t, a):
  fwd_corr@k   mean forward correlation of a's top-k picks          (higher = better neighbours)
  hit@k        share of a's top-k picks that are in a's true forward top-k  (precision@k)
  hedge_err    |forward corr of the #1 pick| shortfall vs the best possible partner (lower = better)

Reference rows: `random` (floor) and `oracle` (ceiling = the forward matrix itself).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# A method gets (t, tickers, context) and returns a similarity matrix (len(tickers) x len(tickers)).
Method = Callable[[str, list, "Context"], np.ndarray]


class Context:
    """Point-in-time data access shared by all methods."""

    def __init__(self, prices: pd.DataFrame | None = None, assets: pd.DataFrame | None = None,
                 events: pd.DataFrame | None = None):
        p = prices if prices is not None else pd.read_csv(C.RAW / "prices.csv")
        p = p.sort_values(["ticker", "date"])
        p["ret"] = np.log(p["adj_close"] / p.groupby("ticker")["adj_close"].shift(1))
        self.R = p.pivot(index="date", columns="ticker", values="ret").sort_index()
        self.dates = list(self.R.index)
        self.assets = (assets if assets is not None else pd.read_csv(C.BUILT / "assets.csv")).set_index("ticker")
        self._events = events
        self._moments = None
        self._holdings = None
        self._holder_links = None

    def past(self, t: str, n: int) -> pd.DataFrame:
        i = self.dates.index(t)
        return self.R.iloc[max(0, i - n + 1): i + 1]

    def future(self, t: str, n: int) -> pd.DataFrame:
        i = self.dates.index(t)
        return self.R.iloc[i + 1: i + 1 + n]

    def moments(self) -> pd.DataFrame:
        if self._moments is None:
            self._moments = pd.read_csv(C.BUILT / "rels_moment.csv")
            self._moments[["ticker", "date"]] = self._moments["src"].str.split("|", expand=True)
        return self._moments

    def holdings(self) -> pd.DataFrame:
        """13F positions from scripts/05_download_13f.py (one row per filing x ticker)."""
        if self._holdings is None:
            self._holdings = pd.read_parquet(C.HOLDINGS)
        return self._holdings

    def holdings_as_of(self, t: str, max_age_days: int = 200) -> pd.DataFrame:
        """Each manager's latest 13F that was public before t (and not older than one quarter)."""
        h = self.holdings()
        tt = pd.Timestamp(t)
        w = h[(h["filing_date"] < tt) & (h["period"] >= tt - pd.Timedelta(days=max_age_days))]
        w = w[w["period"] == w.groupby("cik")["period"].transform("max")]
        return w[w["filing_date"] == w.groupby("cik")["filing_date"].transform("max")]


# ---------------------------------------------------------------- baseline methods
def pearson(window: int) -> Method:
    def f(t, tickers, ctx):
        return ctx.past(t, window)[tickers].corr(min_periods=window // 2).to_numpy()
    f.__name__ = f"pearson_{window}d"
    return f


def same_sector(t, tickers, ctx):
    a = ctx.assets.reindex(tickers)
    sub = a["subIndustry"].to_numpy()
    sec = a["sector"].to_numpy()
    rng = np.random.default_rng(0)  # random tie-break inside the group
    return (2.0 * (sub[:, None] == sub[None, :]) + 1.0 * (sec[:, None] == sec[None, :])
            + 0.01 * rng.random((len(tickers), len(tickers))))


def co_moment(t, tickers, ctx, lookback_days=C.LOOKBACK_DAYS):
    """Graph-only proxy: Jaccard overlap of the Moments two assets shared in the look-back.
    No return values are used, only 'which days did you clearly beat/lag the market together'."""
    m = ctx.moments()
    lo = (pd.Timestamp(t) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    w = m[(m["date"] > lo) & (m["date"] <= t) & m["ticker"].isin(tickers)]
    X = pd.crosstab(w["ticker"], w["dst"]).reindex(tickers, fill_value=0).to_numpy().astype(float)
    inter = X @ X.T
    n = X.sum(1)
    union = n[:, None] + n[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def co_holder(t, tickers, ctx, lookback_days=C.LOOKBACK_DAYS):
    """Graph-only proxy for the Holder context (scripts/06_ownership_graph.py): Jaccard overlap of
    the 13F books that bet on both assets, among filings public in the look-back."""
    if ctx._holder_links is None:
        r = pd.read_csv(C.BUILT / "rels_holder.csv")
        ctx._holder_links = r.assign(ticker=r["src"].str.split("|").str[0], date=r["src"].str.split("|").str[1])
    m = ctx._holder_links
    lo = (pd.Timestamp(t) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    w = m[(m["date"] > lo) & (m["date"] <= t) & m["ticker"].isin(tickers)]
    X = pd.crosstab(w["ticker"], w["dst"]).reindex(tickers, fill_value=0).to_numpy().astype(float)
    inter = X @ X.T
    n = X.sum(1)
    union = n[:, None] + n[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def co_owned(max_positions: int | None = None) -> Method:
    """Common 13F ownership, no prices at all (scripts/05_download_13f.py).

    Each asset is a vector over managers of the weight it has in their book; two assets are close
    when the same managers over- or under-weight them together (correlation of the two vectors,
    so being held by everyone, as mega-caps are, does not make assets similar).
    max_positions  keep only managers with at most that many positions (drops the index-like books)"""
    def f(t, tickers, ctx):
        w = ctx.holdings_as_of(t)
        if max_positions:
            w = w[w["n_positions"] <= max_positions]
        w = w[w["ticker"].isin(tickers)].assign(weight=lambda d: d["value"] / d["total_value"])
        X = (w.pivot_table(index="cik", columns="ticker", values="weight", aggfunc="sum")
             .reindex(columns=tickers).fillna(0.0).to_numpy())
        X = X - X.mean(axis=0)
        n = np.linalg.norm(X, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (X.T @ X) / np.outer(n, n)
    f.__name__ = "co_owned" + (f"_max{max_positions}" if max_positions else "")
    return f


def embeddings(emb: pd.DataFrame, name: str = "fastpath") -> Method:
    """Cosine similarity of per-snapshot embeddings. `emb` columns: ticker, date, embedding (list)."""
    by_date = {d: g.set_index("ticker")["embedding"] for d, g in emb.groupby("date")}

    def f(t, tickers, ctx):
        s = by_date.get(t)
        if s is None:
            return None
        X = np.vstack([np.asarray(s[tk]) if tk in s.index else np.full(len(s.iloc[0]), np.nan)
                       for tk in tickers])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X @ X.T
    f.__name__ = name
    return f


def random_method(t, tickers, ctx):
    return np.random.default_rng(int(t.replace("-", ""))).random((len(tickers), len(tickers)))


random_method.__name__ = "random"


# ---------------------------------------------------------------- evaluation
def evaluate(methods: list[Method], ctx: Context, snapshot_dates: list[str] | None = None,
             k: int = C.TOP_K, forward: int = C.FORWARD_DAYS, per_date: bool = False):
    if snapshot_dates is None:
        snapshot_dates = sorted(pd.read_csv(C.BUILT / "snapshots.csv")["date"].unique())
    snapshot_dates = [d for d in snapshot_dates if d in ctx.dates
                      and ctx.dates.index(d) + forward < len(ctx.dates)]
    rows = []
    for t in snapshot_dates:
        fut = ctx.future(t, forward)
        tickers = [c for c in fut.columns if fut[c].notna().sum() >= forward * 0.9
                   and ctx.past(t, 120)[c].notna().sum() >= 100]
        F = fut[tickers].corr().to_numpy().copy()
        np.fill_diagonal(F, np.nan)
        true_top = np.argsort(-np.nan_to_num(F, nan=-9), axis=1)[:, :k]
        best = np.nanmax(F, axis=1)
        for m in [*methods, random_method, "oracle"]:
            name = m if isinstance(m, str) else m.__name__
            S = F.copy() if m == "oracle" else m(t, tickers, ctx)
            if S is None:
                continue
            S = np.array(S, dtype=float)
            np.fill_diagonal(S, -np.inf)
            S = np.nan_to_num(S, nan=-9)
            picks = np.argsort(-S, axis=1)[:, :k]
            fc = np.take_along_axis(np.nan_to_num(F), picks, axis=1)
            hit = np.array([len(set(p) & set(q)) / k for p, q in zip(picks, true_top)])
            rows.append({"date": t, "method": name, "n_assets": len(tickers),
                         f"fwd_corr@{k}": fc.mean(), "fwd_corr@1": fc[:, 0].mean(),
                         f"hit@{k}": hit.mean(), "hedge_err": np.mean(best - fc[:, 0])})
    df = pd.DataFrame(rows)
    order = df.groupby("method")[f"fwd_corr@{k}"].mean().sort_values(ascending=False).index
    summary = df.groupby("method").mean(numeric_only=True).loc[order].drop(columns="n_assets").round(3)
    summary.insert(0, "dates", df.groupby("method")["date"].nunique().loc[order])
    return (summary, df) if per_date else summary
