"""Turn price series into a connected event graph for FastPath.

Graph written as CSVs in data/built/ :

  assets.csv     (:Asset {ticker, name, sector, subIndustry})          naming / explanation only
  snapshots.csv  (:Snapshot {snapshotId, ticker, date, observationDay}) FastPath base nodes
  events.csv     (:MarketEvent {eventId, ticker, date, day, moveBucket, excessBucket, volBucket,
                                dayType, earnings, features[5], ret})   FastPath event nodes
  moments.csv    (:Moment {momentId, date, day, direction})             FastPath context nodes
  rels_first.csv  (Snapshot|Asset)-[:FIRST_EVENT]->(MarketEvent)
  rels_next.csv   (MarketEvent)-[:NEXT]->(MarketEvent)
  rels_moment.csv (MarketEvent)-[:IN_MOMENT]->(Moment)

Why Moments: a Moment is "day d, assets that beat (or lagged) the market by >= 1 sigma".
Two assets that keep landing in the same Moments become close in FastPath's embedding.
That is co-movement as graph connectivity instead of a correlation of two vectors, and it
composes with the other inputs (volume, FOMC days, earnings, timing).

Everything is point-in-time: every z-score at date t uses only data strictly before t.
Sector is NOT an input to the graph FastPath sees, so it cannot leak into the clusters.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

DAY_TYPES = {"normal": 0, "fomc": 1, "shock": 2, "fomc_shock": 3}


def trailing_z(x: pd.Series, by: pd.Series, window: int, center: bool = True) -> pd.Series:
    """z-score of x against the previous `window` observations of the same group (excludes t)."""
    g = x.groupby(by)
    mu = g.transform(lambda s: s.shift(1).rolling(window, min_periods=C.MIN_HISTORY).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(window, min_periods=C.MIN_HISTORY).std())
    return ((x - mu) if center else x) / sd


def bucket(z: pd.Series, edges) -> pd.Series:
    return pd.Series(np.digitize(z.to_numpy(), edges), index=z.index).astype(int)


def to_day(dates) -> np.ndarray:
    return ((pd.to_datetime(dates) - pd.Timestamp(C.EPOCH)).dt.days).to_numpy()


def main() -> None:
    p = pd.read_csv(C.RAW / "prices.csv").sort_values(["ticker", "date"]).reset_index(drop=True)
    assets = pd.read_csv(C.RAW / "assets.csv")
    t = p["ticker"]

    prev_close = p.groupby("ticker")["adj_close"].shift(1)
    p["ret"] = np.log(p["adj_close"] / prev_close)
    # gap uses unadjusted open vs unadjusted previous close
    p["gap"] = np.log(p["open"] / p.groupby("ticker")["close"].shift(1))
    p["log_range"] = np.log(p["high"] / p["low"]).replace([np.inf, -np.inf], np.nan)
    p["log_vol"] = np.log1p(p["volume"].astype(float))

    # market = equal-weighted universe return that day; excess = idiosyncratic part
    mkt = p.groupby("date")["ret"].mean().rename("mkt")
    p = p.join(mkt, on="date")
    p["excess"] = p["ret"] - p["mkt"]

    p["ret_z"] = trailing_z(p["ret"], t, C.VOL_WINDOW, center=False)
    p["excess_z"] = trailing_z(p["excess"], t, C.VOL_WINDOW, center=False)
    p["gap_z"] = p["gap"] / p.groupby("ticker")["ret"].transform(
        lambda s: s.shift(1).rolling(C.VOL_WINDOW, min_periods=C.MIN_HISTORY).std())
    p["vol_z"] = trailing_z(p["log_vol"], t, C.VOL_WINDOW)
    p["range_z"] = trailing_z(p["log_range"], t, C.VOL_WINDOW)

    # market shock days, z-scored on the market's own trailing history
    m = mkt.to_frame()
    m["sd"] = m["mkt"].shift(1).rolling(C.VOL_WINDOW, min_periods=C.MIN_HISTORY).std()
    shock = (m["mkt"].abs() / m["sd"] >= C.MARKET_SHOCK_Z)
    fomc = set(pd.read_csv(C.FOMC_FILE)["date"])
    p["is_shock"] = p["date"].map(shock).fillna(False).astype(bool)
    p["is_fomc"] = p["date"].isin(fomc)
    p["dayType"] = np.select([p.is_fomc & p.is_shock, p.is_shock, p.is_fomc],
                             [DAY_TYPES["fomc_shock"], DAY_TYPES["shock"], DAY_TYPES["fomc"]],
                             DAY_TYPES["normal"])

    earn_file = C.RAW / "earnings.csv"
    if earn_file.exists():
        e = pd.read_csv(earn_file)
        keys = set(zip(e.ticker, e.date))
        # a release after the close moves the *next* session: flag the release day and the day after
        prev_date = p.groupby("ticker")["date"].shift(1)
        p["earnings"] = [int((tk, d) in keys or (tk, pd_) in keys)
                         for tk, d, pd_ in zip(p.ticker, p.date, prev_date)]
    else:
        p["earnings"] = 0

    ev = p.dropna(subset=["ret_z", "excess_z", "vol_z", "range_z", "gap_z"]).copy()
    ev["moveBucket"] = bucket(ev["ret_z"], C.MOVE_EDGES)
    ev["excessBucket"] = bucket(ev["excess_z"], C.MOVE_EDGES)
    ev["volBucket"] = bucket(ev["vol_z"], C.VOL_EDGES)
    feats = ev[["ret_z", "excess_z", "vol_z", "range_z", "gap_z"]].clip(-5, 5) / 5.0
    ev["features"] = [";".join(f"{v:.4f}" for v in row) for row in feats.to_numpy()]
    ev["day"] = to_day(ev["date"])
    ev["eventId"] = ev["ticker"] + "|" + ev["date"]

    # Moments: (date, direction) for clear out/under-performance vs the market
    ev["direction"] = np.where(ev["excess_z"] >= C.MOMENT_MIN_ABS_Z, "up",
                               np.where(ev["excess_z"] <= -C.MOMENT_MIN_ABS_Z, "down", ""))
    linked = ev[ev["direction"] != ""]
    linked = linked.assign(momentId=linked["date"] + "|" + linked["direction"])
    moments = (linked.groupby("momentId").agg(date=("date", "first"), direction=("direction", "first"),
                                               size=("ticker", "size")).reset_index())
    moments["day"] = to_day(moments["date"])

    # NEXT chain per asset
    ev = ev.sort_values(["ticker", "date"])
    ev["next"] = ev.groupby("ticker")["eventId"].shift(-1)
    first = ev.groupby("ticker")["eventId"].first()

    # month-end snapshots with a full look-back behind them
    dates = pd.to_datetime(pd.Series(sorted(ev["date"].unique())))
    month_ends = dates.groupby(dates.dt.to_period(C.SNAPSHOT_FREQ)).max()
    min_obs = dates.min() + pd.Timedelta(days=C.LOOKBACK_DAYS)
    month_ends = month_ends[month_ends >= min_obs]
    span = ev.groupby("ticker")["date"].agg(["min", "max"])
    snaps = []
    for d in month_ends.dt.strftime("%Y-%m-%d"):
        ok = span[(span["min"] <= (pd.Timestamp(d) - pd.Timedelta(days=C.LOOKBACK_DAYS)).strftime("%Y-%m-%d"))
                  & (span["max"] >= d)].index
        snaps += [{"snapshotId": f"{tk}@{d}", "ticker": tk, "date": d} for tk in ok]
    snaps = pd.DataFrame(snaps)
    snaps["observationDay"] = to_day(snaps["date"])
    snaps["firstEvent"] = snaps["ticker"].map(first)

    out = C.BUILT
    out.mkdir(parents=True, exist_ok=True)
    assets.rename(columns={"sub_industry": "subIndustry"}).to_csv(out / "assets.csv", index=False)
    snaps[["snapshotId", "ticker", "date", "observationDay"]].to_csv(out / "snapshots.csv", index=False)
    cols = ["eventId", "ticker", "date", "day", "moveBucket", "excessBucket", "volBucket",
            "dayType", "earnings", "features", "ret"]
    ev[cols].to_csv(out / "events.csv", index=False)
    moments[["momentId", "date", "day", "direction", "size"]].to_csv(out / "moments.csv", index=False)
    pd.concat([
        pd.DataFrame({"src": first.index, "srcLabel": "Asset", "dst": first.values}),
        pd.DataFrame({"src": snaps["snapshotId"], "srcLabel": "Snapshot", "dst": snaps["firstEvent"]}),
    ]).to_csv(out / "rels_first.csv", index=False)
    ev.dropna(subset=["next"])[["eventId", "next"]].rename(columns={"eventId": "src", "next": "dst"}) \
        .to_csv(out / "rels_next.csv", index=False)
    linked[["eventId", "momentId"]].rename(columns={"eventId": "src", "momentId": "dst"}) \
        .to_csv(out / "rels_moment.csv", index=False)

    print(f"assets     {len(assets):>9,}")
    print(f"events     {len(ev):>9,}  ({ev['date'].min()} .. {ev['date'].max()})")
    print(f"moments    {len(moments):>9,}  linked events {len(linked):,} ({len(linked) / len(ev):.0%})")
    print(f"snapshots  {len(snaps):>9,}  over {snaps['date'].nunique()} month-ends")
    print(f"shock days {int(shock.sum()):>9}  FOMC days in range {int(p.drop_duplicates('date').is_fomc.sum())}")
    print(f"earnings-flagged events {int(ev['earnings'].sum()):,}")
    print("bucket mix (move):", ev["moveBucket"].value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == "__main__":
    main()
