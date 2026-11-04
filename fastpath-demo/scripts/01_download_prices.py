"""Download daily prices for the S&P 500 universe with yfinance.

Run this on your own machine (Yahoo is blocked from some sandboxes):

    python scripts/01_download_prices.py                 # S&P 500, START..today
    python scripts/01_download_prices.py --earnings      # + earnings dates (slow: one call per ticker)
    python scripts/01_download_prices.py --from-legacy path/to/real_asset_prices.csv
                                                         # reuse the 99-ticker CSV of the old demo

Output: data/raw/prices.csv  (ticker,date,open,high,low,close,adj_close,volume)
        data/raw/assets.csv  (ticker,name,sector,sub_industry)
        data/raw/earnings.csv (ticker,date,surprise_pct)   only with --earnings

Caveat: the universe is *today's* constituents, so there is survivorship bias. Fine for a
demo, say it out loud if a quant asks.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(C.UNIVERSE_FILE)
    return u.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector",
                             "GICS Sub-Industry": "sub_industry"})[["ticker", "name", "sector", "sub_industry"]]


def from_legacy(path: str) -> None:
    d = pd.read_csv(path)
    prices = d[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    uni = load_universe().set_index("ticker")
    assets = d[["ticker", "name", "category"]].drop_duplicates("ticker").set_index("ticker")
    assets["sector"] = uni["sector"].reindex(assets.index).fillna(assets["category"])
    assets["sub_industry"] = uni["sub_industry"].reindex(assets.index).fillna(assets["category"])
    write(prices, assets.reset_index()[["ticker", "name", "sector", "sub_industry"]])


def download(start: str, end: str | None, chunk: int, earnings: bool) -> None:
    import yfinance as yf

    uni = load_universe()
    yahoo = {t: t.replace(".", "-") for t in uni["ticker"]}          # BRK.B -> BRK-B
    frames = []
    tickers = list(yahoo.values())
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        print(f"prices {i + 1}-{i + len(batch)} / {len(tickers)}", flush=True)
        raw = yf.download(batch, start=start, end=end, auto_adjust=False, group_by="ticker",
                          threads=True, progress=False)
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(subset=["Adj Close"]).reset_index()
            if df.empty:
                continue
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df["ticker"] = t
            frames.append(df)
        time.sleep(1)
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    prices = prices[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    uni["ticker"] = uni["ticker"].map(yahoo)
    write(prices, uni[uni["ticker"].isin(prices["ticker"].unique())])

    if earnings:
        rows = []
        for n, t in enumerate(prices["ticker"].unique(), 1):
            try:
                e = yf.Ticker(t).get_earnings_dates(limit=20)
            except Exception as exc:  # yfinance raises on missing data, keep going
                print(f"  earnings {t}: {exc}")
                continue
            if e is None or e.empty:
                continue
            e = e.reset_index()
            for _, r in e.iterrows():
                rows.append({"ticker": t, "date": pd.Timestamp(r.iloc[0]).strftime("%Y-%m-%d"),
                             "surprise_pct": r.get("Surprise(%)")})
            if n % 50 == 0:
                print(f"earnings {n}", flush=True)
        pd.DataFrame(rows).to_csv(C.RAW / "earnings.csv", index=False)
        print(f"wrote {len(rows)} earnings dates")


def write(prices: pd.DataFrame, assets: pd.DataFrame) -> None:
    C.RAW.mkdir(parents=True, exist_ok=True)
    prices.sort_values(["ticker", "date"]).to_csv(C.RAW / "prices.csv", index=False)
    assets.to_csv(C.RAW / "assets.csv", index=False)
    print(f"wrote {len(prices):,} price rows for {prices['ticker'].nunique()} assets "
          f"({prices['date'].min()} .. {prices['date'].max()}) to {C.RAW}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-legacy", help="convert the old demo's real_asset_prices.csv instead")
    ap.add_argument("--start", default=C.START)
    ap.add_argument("--end", default=C.END)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--earnings", action="store_true")
    a = ap.parse_args()
    if a.from_legacy:
        from_legacy(a.from_legacy)
    else:
        download(a.start, a.end, a.chunk, a.earnings)
