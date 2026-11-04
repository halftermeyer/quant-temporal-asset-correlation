"""Download SEC Form 13F holdings and keep the positions in our universe (the 99-ticker set and
the S&P 500 both, so the same file serves either price universe).

    python scripts/05_download_13f.py

Output: data/built/holdings.parquet  one row per (13F-HR filing, ticker)
          cik, manager, filing_date, period, ticker, value, n_positions, total_value
        data/reference/cusip_ticker.csv  CUSIP -> ticker cache (OpenFIGI), reused on re-runs

Point in time: a filing is public on its `filing_date` (about 45 days after `period`, the quarter
end), so scoring must only use filings with filing_date < as-of date.

Why: funds that hold the same stocks buy and sell them together when their own flows come in,
so common ownership predicts *future* co-movement beyond sector and past correlation
(Anton & Polk 2014, "Connected Stocks"). It is not in the price series.

Set SEC_USER_AGENT in .env ("name contact") as SEC asks; OPENFIGI_API_KEY is optional (faster).
"""
import csv
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

load_dotenv(C.ROOT / ".env")
SEC = "https://www.sec.gov"
INDEX = SEC + "/data-research/sec-markets-data/form-13f-data-sets"
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "fast-path-finance research demo contact@example.com")}
FIRST_FILE = pd.Timestamp("2023-04-01")   # filings from here on: covers START = 2023-01-01 snapshots
RAW13F = C.DATA / "raw" / "13f"
CUSIP_MAP = C.REFERENCE / "cusip_ticker.csv"
TOP_CUSIPS = 3000                         # most-held CUSIPs sent to OpenFIGI (S&P 500 is well inside)
ALIASES = {"BK": "BNY"}                   # old ticker in the 99-ticker set -> ticker OpenFIGI returns today
MANUAL = {"JCI": "JOHNSON CTLS INTL PLC"}  # 13F issuer abbreviations name matching cannot guess


def universe() -> set:
    tickers = set(pd.read_csv(C.UNIVERSE_FILE)["Symbol"])
    for u in [C.DATA, C.DATA / "sp500"]:
        if (u / "raw" / "assets.csv").exists():
            tickers |= set(pd.read_csv(u / "raw" / "assets.csv")["ticker"])
    return tickers


def file_start(name: str) -> pd.Timestamp:
    """'2023q3_form13f.zip' or '01mar2024-31may2024_form13f.zip' -> first filing day covered."""
    m = re.match(r"(\d{4})q(\d)", name)
    if m:
        return pd.Timestamp(year=int(m[1]), month=3 * int(m[2]) - 2, day=1)
    return pd.to_datetime(name[:9], format="%d%b%Y")


def download() -> list[Path]:
    html = requests.get(INDEX, headers=UA, timeout=60).text
    links = sorted(set(re.findall(r'href="([^"]*form-13f-data-sets/[^"]*\.zip)"', html)))
    RAW13F.mkdir(parents=True, exist_ok=True)
    out = []
    for href in links:
        name = href.rsplit("/", 1)[1]
        if file_start(name) < FIRST_FILE:
            continue
        path = RAW13F / name
        if not path.exists():
            print(f"download {name}", flush=True)
            r = requests.get(SEC + href, headers=UA, timeout=600)
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(0.5)
        out.append(path)
    return sorted(out, key=lambda p: file_start(p.name))


def read_tsv(z: zipfile.ZipFile, name: str, cols: list) -> pd.DataFrame:
    name = next(n for n in z.namelist() if n.rsplit("/", 1)[-1] == name)  # some archives use a subfolder
    return pd.read_csv(io.TextIOWrapper(z.open(name), encoding="latin-1"), sep="\t", usecols=cols,
                       dtype=str, quoting=csv.QUOTE_NONE)


def parse(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Original 13F-HR filings only (amendments are a small share and often partial)."""
    with zipfile.ZipFile(path) as z:
        sub = read_tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK",
                                             "PERIODOFREPORT"])
        cov = read_tsv(z, "COVERPAGE.tsv", ["ACCESSION_NUMBER", "FILINGMANAGER_NAME"])
        info = read_tsv(z, "INFOTABLE.tsv", ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMTTYPE",
                                             "PUTCALL"])
    sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"].merge(cov, on="ACCESSION_NUMBER", how="left")
    info = info[info["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])
                & (info["SSHPRNAMTTYPE"] == "SH") & info["PUTCALL"].isna()]
    info = info.assign(CUSIP=info["CUSIP"].str.upper().str.strip(),
                       VALUE=pd.to_numeric(info["VALUE"], errors="coerce").fillna(0.0))  # dollars since 2023
    hold = info.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False)["VALUE"].sum()
    names = info.groupby("CUSIP")["NAMEOFISSUER"].first()
    stats = hold.groupby("ACCESSION_NUMBER").agg(n_positions=("CUSIP", "size"), total_value=("VALUE", "sum"))
    filings = sub.set_index("ACCESSION_NUMBER").join(stats, how="inner").reset_index()
    filings = filings.assign(filing_date=pd.to_datetime(filings["FILING_DATE"], format="%d-%b-%Y"),
                             period=pd.to_datetime(filings["PERIODOFREPORT"], format="%d-%b-%Y"),
                             cik=filings["CIK"].astype(int))
    return hold, filings[["ACCESSION_NUMBER", "cik", "FILINGMANAGER_NAME", "filing_date", "period",
                          "n_positions", "total_value"]].rename(columns={"FILINGMANAGER_NAME": "manager"}), names


SUFFIXES = r"\b(THE|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|COS|PLC|LTD|LIMITED|NV|N V|SA|AG|AND|HLDGS|" \
           r"HOLDINGS?|GROUP|GRP|CL [A-C]|CLASS [A-C]|COM|NEW|SHS|ORD)\b"


def norm(name: str) -> str:
    """'Exxon Mobil Corp' and 'ExxonMobil' -> 'EXXONMOBIL'."""
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " AND "))
    return re.sub(r"[^A-Z0-9]", "", re.sub(SUFFIXES, " ", s))


def match_by_name(missing: set, names: pd.Series, value: pd.Series, taken: set) -> pd.DataFrame:
    """Fallback for CUSIPs OpenFIGI will not resolve (e.g. XOM, ACN) and renamed tickers (BK -> BNY):
    match the 13F issuer name to the universe's company names, keep the most-held CUSIP."""
    uni = pd.read_csv(C.UNIVERSE_FILE).rename(columns={"Symbol": "ticker", "Security": "name"})[["ticker", "name"]]
    for u in [C.DATA, C.DATA / "sp500"]:
        if (u / "raw" / "assets.csv").exists():
            uni = pd.concat([uni, pd.read_csv(u / "raw" / "assets.csv")[["ticker", "name"]]])
    cand = pd.DataFrame({"cusip": names.index, "key": names.map(norm).to_numpy()})
    cand = cand[~cand["cusip"].isin(taken) & (cand["key"].str.len() >= 3)]
    cand["value"] = cand["cusip"].map(value)
    rows = []
    for t, g in uni[uni["ticker"].isin(missing)].groupby("ticker"):
        keys = {k for k in g["name"].map(norm) if len(k) >= 3}
        hit = cand[cand["key"].isin(keys)]
        if hit.empty:  # one name is a prefix of the other ("LINDE" vs "LINDEPLC" is handled above; this catches the rest)
            hit = cand[cand["key"].map(lambda k: any(k.startswith(x) or x.startswith(k) for x in keys if len(x) >= 5)
                                       and len(k) >= 5)]
        if not hit.empty:
            rows.append({"cusip": hit.nlargest(1, "value")["cusip"].iat[0], "ticker": t})
    return pd.DataFrame(rows, columns=["cusip", "ticker"])


def map_cusips(cusips: list) -> pd.DataFrame:
    """CUSIP -> US ticker through OpenFIGI, cached. Unkeyed: 10 per request, 25 requests/min."""
    cache = pd.read_csv(CUSIP_MAP, dtype=str) if CUSIP_MAP.exists() else pd.DataFrame(columns=["cusip", "ticker"])
    todo = [c for c in cusips if c not in set(cache["cusip"])]
    key = os.environ.get("OPENFIGI_API_KEY")
    batch, pause = (100, 0.25) if key else (10, 2.5)
    headers = {"Content-Type": "application/json", **({"X-OPENFIGI-APIKEY": key} if key else {})}
    rows = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
        while True:
            r = requests.post("https://api.openfigi.com/v3/mapping", json=jobs, headers=headers, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(10)
        r.raise_for_status()
        for c, res in zip(chunk, r.json()):
            data = [d for d in res.get("data", []) if d.get("marketSector") == "Equity"]
            rows.append({"cusip": c, "ticker": data[0]["ticker"].replace("/", ".") if data else ""})
        if (i // batch) % 20 == 0:
            print(f"openfigi {i + len(chunk)} / {len(todo)}", flush=True)
        time.sleep(pause)
    cache = pd.concat([cache, pd.DataFrame(rows, columns=["cusip", "ticker"])], ignore_index=True)
    C.REFERENCE.mkdir(exist_ok=True)
    cache.to_csv(CUSIP_MAP, index=False)
    return cache


if __name__ == "__main__":
    holds, filings, names = [], [], []
    for path in download():
        t0 = time.time()
        h, f, n = parse(path)
        holds.append(h); filings.append(f); names.append(n)
        print(f"{path.name}: {len(f):,} filings, {len(h):,} positions ({time.time() - t0:.0f}s)", flush=True)
    hold = pd.concat(holds, ignore_index=True)
    filings = pd.concat(filings, ignore_index=True).drop_duplicates("ACCESSION_NUMBER")

    value = hold.groupby("CUSIP")["VALUE"].sum()
    top = value.nlargest(TOP_CUSIPS).index
    cmap = map_cusips([c for c in top if len(c) == 9])
    cmap = cmap[cmap["ticker"].isin(universe())]
    names = pd.concat(names).groupby(level=0).last().reindex(top).dropna()
    cmap = pd.concat([cmap] + [cmap[cmap["ticker"] == new].assign(ticker=old) for old, new in ALIASES.items()])
    manual = [{"cusip": names[names == n].index[0], "ticker": t} for t, n in MANUAL.items() if (names == n).any()]
    cmap = pd.concat([cmap, pd.DataFrame(manual, columns=["cusip", "ticker"])], ignore_index=True)
    extra = match_by_name(universe() - set(cmap["ticker"]), names, value, set(cmap["cusip"]))
    print(f"matched by name: {', '.join(f'{r.ticker}={names[r.cusip]}' for r in extra.itertuples())}")
    cmap = pd.concat([cmap, extra], ignore_index=True)

    out = (hold.merge(cmap, left_on="CUSIP", right_on="cusip")
               .groupby(["ACCESSION_NUMBER", "ticker"], as_index=False)["VALUE"].sum()
               .rename(columns={"VALUE": "value"})
               .merge(filings, on="ACCESSION_NUMBER"))
    out = out[["cik", "manager", "filing_date", "period", "ticker", "value", "n_positions", "total_value"]]
    C.HOLDINGS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.HOLDINGS, index=False)

    u = universe()
    missing = sorted(u - set(out["ticker"]))
    print(f"\n{len(out):,} rows, {out['cik'].nunique():,} managers, {out['ticker'].nunique()} / {len(u)} tickers, "
          f"filed {out['filing_date'].min():%Y-%m-%d} .. {out['filing_date'].max():%Y-%m-%d}")
    if missing:
        print(f"no 13F match for {len(missing)} tickers: {', '.join(missing[:40])}")
