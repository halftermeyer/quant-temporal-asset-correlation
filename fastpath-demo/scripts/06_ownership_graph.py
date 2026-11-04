"""Put 13F ownership into the event graph as a second kind of context node, next to Moments.

    python scripts/06_ownership_graph.py           # build data/built/holders.csv + rels_holder.csv
    python scripts/06_ownership_graph.py --load    # ... and load them into Neo4j (additive, MERGE)

Model
  (:Holder {holderId, cik, manager, period, date, day})    one 13F book (manager x quarter)
  (:MarketEvent)-[:HELD_BY {activeWeight}]->(:Holder)

Each quarter, a concentrated book (at most MAX_POSITIONS names, so index-like books do not connect
everything) links to every asset that is at least MIN_WEIGHT of the book AND overweight (above its
weight in the all-managers book). Books linking fewer than 2 assets are dropped: they connect nothing. The link starts from the asset's event on
the first trading day after the 45-day filing deadline: from that day the whole quarter is public,
so the per-date projection (180-day window) can never see a filing before it was published.
Two assets the same managers bet on share Holder nodes, the way assets that beat the market on
the same day share a Moment.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

MAX_POSITIONS = 200
MIN_WEIGHT = 0.02
DEADLINE_DAYS = 45
BATCH = 10_000


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_parquet(C.HOLDINGS)
    ev = pd.read_csv(C.BUILT / "events.csv", usecols=["eventId", "ticker", "date", "day"])
    tickers = set(ev["ticker"])
    h["public"] = h["period"] + pd.Timedelta(days=DEADLINE_DAYS)
    h = h[h["filing_date"] <= h["public"]]               # late filers are left out, not back-dated
    h = h[h["filing_date"] == h.groupby(["cik", "period"])["filing_date"].transform("max")]
    h["weight"] = h["value"] / h["total_value"]
    books = h.drop_duplicates(["cik", "period"]).groupby("period")["total_value"].sum()
    bench = h.groupby(["period", "ticker"])["value"].transform("sum") / h["period"].map(books)
    h["activeWeight"] = h["weight"] - bench
    h = h[(h["n_positions"] <= MAX_POSITIONS) & (h["weight"] >= MIN_WEIGHT) & (h["activeWeight"] > 0)
          & h["ticker"].isin(tickers)]
    h = h[h.groupby(["cik", "period"])["ticker"].transform("size") >= 2]

    # anchor: the asset's first event on or after the day the quarter is fully public
    ev = ev.assign(date=pd.to_datetime(ev["date"])).sort_values("date")
    h = h.sort_values("public")
    h = pd.merge_asof(h, ev.rename(columns={"date": "eventDate"}), left_on="public", right_on="eventDate",
                      by="ticker", direction="forward").dropna(subset=["eventId"])
    h["holderId"] = h["cik"].astype(str) + "|" + h["period"].dt.strftime("%Y-%m-%d")
    holders = (h.groupby("holderId").agg(cik=("cik", "first"), manager=("manager", "first"),
                                         period=("period", "first"), date=("eventDate", "min"))
               .reset_index())
    holders["day"] = (holders["date"] - pd.Timestamp(C.EPOCH)).dt.days
    for c in ["period", "date"]:
        holders[c] = holders[c].dt.strftime("%Y-%m-%d")
    rels = h[["eventId", "holderId", "activeWeight"]].rename(columns={"eventId": "src", "holderId": "dst"})
    holders.to_csv(C.BUILT / "holders.csv", index=False)
    rels.to_csv(C.BUILT / "rels_holder.csv", index=False)
    per = h.groupby("period").agg(assets=("ticker", "nunique"), holders=("holderId", "nunique"), links=("eventId", "size"))
    print(per.to_string())
    print(f"{len(holders):,} Holder nodes, {len(rels):,} HELD_BY links, "
          f"median {h.groupby('holderId').size().median():.0f} assets per holder")
    return holders, rels


def load(holders: pd.DataFrame, rels: pd.DataFrame) -> None:
    load_dotenv(find_dotenv(usecwd=True))
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver.execute_query("CREATE CONSTRAINT holder_id IF NOT EXISTS FOR (n:Holder) REQUIRE n.holderId IS UNIQUE",
                         database_=db)
    for df, q in [(holders, """
            UNWIND $rows AS r
            MERGE (h:Holder {holderId: r.holderId})
            SET h.cik = toInteger(r.cik), h.manager = r.manager, h.period = r.period, h.date = r.date,
                h.day = toInteger(r.day)"""),
                  (rels, """
            UNWIND $rows AS r
            MATCH (e:MarketEvent {eventId: r.src}), (h:Holder {holderId: r.dst})
            MERGE (e)-[x:HELD_BY]->(h) SET x.activeWeight = toFloat(r.activeWeight)""")]:
        rows = df.astype(str).to_dict("records")
        t0 = time.time()
        for i in range(0, len(rows), BATCH):
            driver.execute_query(q, rows=rows[i:i + BATCH], database_=db)
        print(f"loaded {len(rows):,} rows in {time.time() - t0:.1f}s", flush=True)
    print(driver.execute_query("RETURN COUNT { (:Holder) } AS holders, COUNT { ()-[:HELD_BY]->() } AS heldBy",
                               database_=db).records[0].data())
    driver.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="also load into Neo4j")
    a = ap.parse_args()
    holders, rels = build()
    if a.load:
        load(holders, rels)
