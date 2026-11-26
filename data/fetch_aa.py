"""Fetch the desk's latest Indian series from its asset-allocation repository (wm-AA) for review, plus a second source
from the research Postgres, read-only. Writes raw files into data/; they are NOT readable by any study until the data
owner approves them on APPROVED_DATA.md after the review (data/data_review_aa.py → data/DATA_REVIEW_AA.md).

    WM_AA_REPO=<owner>/wm-AA [WM_AA_COMMIT=<sha>] WM_DB_HOST=... WM_DB_NAME=... WM_DB_USER=... WM_DB_PASSWORD=... python data/fetch_aa.py

The repository is read through the GitHub CLI (`gh`, already signed in) at a pinned commit, so the copies are
reproducible; its owner is given in the environment (WM_AA_REPO), not written here. Each file is stored byte for byte
under a new name, and its git blob hash is checked against the commit's tree (the copy is the commit's file exactly).
Database credentials come only from the environment; nothing secret is written anywhere. Prod VPN required.

From wm-AA (its own approval IDs in brackets):
    aa_index_{n100,mc100,sc100}_pre2019.csv     daily OHLCV of Nifty 100, Midcap 100, Smallcap 100 to 2019-01-04 [D25]
    aa_index_{n100,mc100,sc100}_2019_2026.csv   the same, 2019-01-01 to 2026-11-20 [D25d]
    aa_mauc1_daily.csv                          MCX gold front-month continuation (MAUc1), cleaned, 2003-11-10 to 2026-11-20 [D24c]
    aa_liquid_lbes.csv                          the desk's liquid-ETF cash level ("LBES"), 2007-06-29 to 2026-11-25 [D11]
    aa_liquid_case.csv                          the desk's LIQUIDCASE cash level, 2007-06-29 to 2026-11-25 [D11]
From the research Postgres (second source, read only by the review; not for approval):
    aa_db_check.csv   MAUc1 (ohlcv_eikon_futures); LIQUIDCASE, LIQUIDBEES, NIFTY 100, NIFTY MIDCAP 100, NIFTY SMLCAP 100
                      (merged_historical_data_with_rics): source, ticker, date, open, high, low, close, volume
"""
import hashlib
import json
import os
import subprocess
import sys
import urllib.parse
from datetime import date

import pandas as pd
import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT = os.environ.get("WM_AA_COMMIT", "e9cf7b25b45b9b66b53388decb5552031da87a14")   # wm-AA main on 2026-11-26
FILES = {
    "AA/Historical Data/extended_2026-11-12/index_daily_pre2019/Nifty100_daily_ohlcv.csv": "aa_index_n100_pre2019.csv",
    "AA/Historical Data/extended_2026-11-12/index_daily_pre2019/MC100_daily_ohlcv.csv": "aa_index_mc100_pre2019.csv",
    "AA/Historical Data/extended_2026-11-12/index_daily_pre2019/SC100_daily_ohlcv.csv": "aa_index_sc100_pre2019.csv",
    "AA/Historical Data/extended_2026-11-20/index_daily_2019_2026/Nifty100_daily_ohlcv.csv": "aa_index_n100_2019_2026.csv",
    "AA/Historical Data/extended_2026-11-20/index_daily_2019_2026/MC100_daily_ohlcv.csv": "aa_index_mc100_2019_2026.csv",
    "AA/Historical Data/extended_2026-11-20/index_daily_2019_2026/SC100_daily_ohlcv.csv": "aa_index_sc100_2019_2026.csv",
    "AA/Historical Data/extended_2026-11-20/MAUc1_from_2003/mauc1_daily_clean_extended.csv": "aa_mauc1_daily.csv",
    "AA/Historical Data/LBES_Adjusted_Price_data_extended_2026-11-25.csv": "aa_liquid_lbes.csv",
    "AA/Historical Data/liquidEtfFinal_extended_2026-11-25.csv": "aa_liquid_case.csv",
}
DB_END = "2026-11-25"


def gh(args, raw=False):
    out = subprocess.run(["gh", "api"] + (["-H", "Accept: application/vnd.github.raw"] if raw else []) + args, capture_output=True, timeout=300)
    if out.returncode != 0:
        sys.exit(f"gh api failed: {out.stderr.decode(errors='replace')[:300]}")
    return out.stdout


def blob_sha(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def from_repo(repo: str) -> list:
    tree = {e["path"]: e["sha"] for e in json.loads(gh([f"repos/{repo}/git/trees/{COMMIT}?recursive=1"]))["tree"] if e["type"] == "blob"}
    rows = []
    for src, name in FILES.items():
        b = gh([f"repos/{repo}/contents/{urllib.parse.quote(src)}?ref={COMMIT}"], raw=True)
        if src not in tree or blob_sha(b) != tree[src]:
            sys.exit(f"{src}: the download is not the commit's file (git blob hash differs)")
        open(os.path.join(HERE, name), "wb").write(b)
        rows.append((name, src, len(b), hashlib.sha256(b).hexdigest()))
    return rows


def from_db() -> pd.DataFrame:
    need = ["WM_DB_HOST", "WM_DB_NAME", "WM_DB_USER", "WM_DB_PASSWORD"]
    if any(not os.environ.get(k) for k in need):
        sys.exit(f"set {', '.join(need)} (and WM_DB_PORT) in the environment")
    c = psycopg2.connect(host=os.environ["WM_DB_HOST"], port=int(os.environ.get("WM_DB_PORT", 5432)), dbname=os.environ["WM_DB_NAME"], user=os.environ["WM_DB_USER"],
                         password=os.environ["WM_DB_PASSWORD"], sslmode=os.environ.get("WM_DB_SSLMODE", "require"), connect_timeout=20)
    c.set_session(readonly=True, autocommit=True); cur = c.cursor(); cur.execute("SET statement_timeout = '300s'")
    a = pd.read_sql_query("""select 'ohlcv_eikon_futures' as source, ric as ticker, date, open, high, low, close, volume from ohlcv_eikon_futures
                             where ric = 'MAUc1' and date <= %(end)s order by date""", c, params={"end": DB_END})
    b = pd.read_sql_query("""select 'merged_historical_data_with_rics' as source, ticker, date, open, high, low, close, volume from merged_historical_data_with_rics
                             where ticker = any(%(t)s) and date <= %(end)s order by ticker, date""", c,
                          params={"t": ["LIQUIDCASE", "LIQUIDBEES", "NIFTY 100", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100"], "end": DB_END})
    c.close()
    d = pd.concat([a, b]); d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.drop_duplicates(["source", "ticker", "date"]).sort_values(["source", "ticker", "date"])


def main():
    repo = os.environ.get("WM_AA_REPO")
    if not repo:
        sys.exit("set WM_AA_REPO (owner/name of the desk's asset-allocation repository) in the environment")
    rows = from_repo(repo)
    ck = from_db(); ck.to_csv(os.path.join(HERE, "aa_db_check.csv"), index=False)
    print(f"fetched {date.today()} from wm-AA commit {COMMIT[:12]} (git blob hashes checked):")
    for name, src, n, h in rows:
        print(f"  {name:28s} {n:>9,} bytes  sha256 {h[:16]}…  ← {src.split('/', 2)[-1]}")
    print("second source: " + "; ".join(f"{t} {len(g):,} rows ({g['date'].min()} → {g['date'].max()})" for t, g in ck.groupby("ticker")))


if __name__ == "__main__":
    main()
