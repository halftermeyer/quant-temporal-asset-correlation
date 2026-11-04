"""Set the bar before touching FastPath: how good are the obvious methods at picking
next-60-day co-movers?  Writes results/baselines.csv and results/baselines_by_date.csv.

    python scripts/03_score_baselines.py
    python scripts/03_score_baselines.py --embeddings results/fastpath_embeddings.csv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import scoring as S  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", help="CSV with ticker,date,embedding(JSON list) to score too")
    a = ap.parse_args()

    ctx = S.Context()
    methods = [S.pearson(w) for w in C.TRAILING_WINDOWS] + [S.same_sector, S.co_moment]
    if a.embeddings:
        e = pd.read_csv(a.embeddings)
        e["embedding"] = e["embedding"].map(json.loads)
        methods.append(S.embeddings(e))
    summary, by_date = S.evaluate(methods, ctx, per_date=True)
    C.RESULTS.mkdir(exist_ok=True)
    summary.to_csv(C.RESULTS / "baselines.csv")
    by_date.to_csv(C.RESULTS / "baselines_by_date.csv", index=False)
    pd.set_option("display.width", 140)
    print(f"forward window {C.FORWARD_DAYS} trading days, top-{C.TOP_K} neighbours\n")
    print(summary.to_string())
