"""Load data/built/*.csv into Neo4j (AuraDB) with the Python driver.

    python scripts/04_load_neo4j.py            # load (idempotent MERGE on ids)
    python scripts/04_load_neo4j.py --reset    # wipe this demo's labels first

Reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE from .env

Model
  (:Asset {ticker, name, sector, subIndustry})
  (:Asset)-[:HAS_SNAPSHOT]->(:Snapshot {snapshotId, ticker, date, observationDay, sidx})
  (:Asset)-[:FIRST_EVENT]->(:MarketEvent)-[:NEXT]->(:MarketEvent) ...
  (:MarketEvent {eventId, ticker, date, day, moveBucket, excessBucket, volBucket, dayType,
                 earnings, features, ret})-[:IN_MOMENT]->(:Moment {momentId, date, day, direction, size})

Snapshots carry no events of their own: the notebook links each one to its look-back window at
projection time, so FastPath can never see events after the observation day.
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

BATCH = 10_000

SCHEMA = [
    "CREATE CONSTRAINT asset_ticker IF NOT EXISTS FOR (n:Asset) REQUIRE n.ticker IS UNIQUE",
    "CREATE CONSTRAINT snapshot_id IF NOT EXISTS FOR (n:Snapshot) REQUIRE n.snapshotId IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:MarketEvent) REQUIRE n.eventId IS UNIQUE",
    "CREATE CONSTRAINT moment_id IF NOT EXISTS FOR (n:Moment) REQUIRE n.momentId IS UNIQUE",
    "CREATE INDEX event_ticker_day IF NOT EXISTS FOR (n:MarketEvent) ON (n.ticker, n.day)",
    "CREATE INDEX event_day IF NOT EXISTS FOR (n:MarketEvent) ON (n.day)",  # same-day peers in fp.PROJECTION
    "CREATE INDEX snapshot_date IF NOT EXISTS FOR (n:Snapshot) ON (n.date)",
    "CREATE INDEX moment_date IF NOT EXISTS FOR (n:Moment) ON (n.date)",
]

LOADS = [
    ("assets.csv", """
        UNWIND $rows AS r
        MERGE (a:Asset {ticker: r.ticker})
        SET a.name = r.name, a.sector = r.sector, a.subIndustry = r.subIndustry"""),
    ("moments.csv", """
        UNWIND $rows AS r
        MERGE (m:Moment {momentId: r.momentId})
        SET m.date = r.date, m.day = toInteger(r.day), m.direction = r.direction, m.size = toInteger(r.size)"""),
    ("snapshots.csv", """
        UNWIND $rows AS r
        MATCH (a:Asset {ticker: r.ticker})
        MERGE (s:Snapshot {snapshotId: r.snapshotId})
        SET s.ticker = r.ticker, s.date = r.date, s.observationDay = toInteger(r.observationDay),
            s.sidx = toInteger(r.sidx)
        MERGE (a)-[:HAS_SNAPSHOT]->(s)"""),
    ("events.csv", """
        UNWIND $rows AS r
        MERGE (e:MarketEvent {eventId: r.eventId})
        SET e.ticker = r.ticker, e.date = r.date, e.day = toInteger(r.day),
            e.moveBucket = toInteger(r.moveBucket), e.excessBucket = toInteger(r.excessBucket),
            e.volBucket = toInteger(r.volBucket), e.dayType = toInteger(r.dayType),
            e.earnings = toInteger(r.earnings), e.ret = toFloat(r.ret),
            e.features = [x IN split(r.features, ';') | toFloat(x)]"""),
    ("rels_next.csv", """
        UNWIND $rows AS r
        MATCH (a:MarketEvent {eventId: r.src}), (b:MarketEvent {eventId: r.dst})
        MERGE (a)-[:NEXT]->(b)"""),
    ("rels_moment.csv", """
        UNWIND $rows AS r
        MATCH (e:MarketEvent {eventId: r.src}), (m:Moment {momentId: r.dst})
        MERGE (e)-[:IN_MOMENT]->(m)"""),
    ("rels_first.csv", """
        UNWIND $rows AS r
        WITH r WHERE r.srcLabel = 'Asset'
        MATCH (a:Asset {ticker: r.src}), (e:MarketEvent {eventId: r.dst})
        MERGE (a)-[:FIRST_EVENT]->(e)"""),
]


def main(reset: bool) -> None:
    load_dotenv(find_dotenv(usecwd=True))
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    driver.verify_connectivity()

    if reset:
        for label in ["MarketEvent", "Moment", "Holder", "Snapshot", "Asset"]:
            driver.execute_query(f"MATCH (n:{label}) CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF 10000 ROWS",
                                 database_=db)
        print("reset done")
    for q in SCHEMA:
        driver.execute_query(q, database_=db)
    driver.execute_query("CALL db.awaitIndexes(300)", database_=db)

    for fname, q in LOADS:
        df = pd.read_csv(C.BUILT / fname, dtype=str, keep_default_na=False)
        if fname == "snapshots.csv":
            df["sidx"] = range(len(df))
        rows = df.to_dict("records")
        t0 = time.time()
        for i in range(0, len(rows), BATCH):
            driver.execute_query(q, rows=rows[i:i + BATCH], database_=db)
        print(f"{fname:<16} {len(rows):>9,} rows  {time.time() - t0:6.1f}s", flush=True)

    c = driver.execute_query("""
        RETURN COUNT { (:Asset) } AS assets, COUNT { (:Snapshot) } AS snapshots,
               COUNT { (:MarketEvent) } AS events, COUNT { (:Moment) } AS moments,
               COUNT { ()-[:NEXT]->() } AS next, COUNT { ()-[:IN_MOMENT]->() } AS inMoment""",
                            database_=db).records[0].data()
    print(c)
    driver.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    main(ap.parse_args().reset)
