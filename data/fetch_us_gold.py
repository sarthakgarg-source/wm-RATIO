"""Fetch the US gold ETF raw data for review (GLD, SPDR Gold Shares) from the research Postgres, read-only — for SPY
against gold. Writes raw files into data/; they are NOT readable by any study until the data owner approves them on
APPROVED_DATA.md after the review (data/data_review_us_gold.py → data/DATA_REVIEW_US_GOLD.md).

    WM_DB_HOST=... WM_DB_PORT=5432 WM_DB_NAME=... WM_DB_USER=... WM_DB_PASSWORD=... python data/fetch_us_gold.py

Credentials come only from the environment; nothing secret is written anywhere. Prod VPN required.
The approved US ETF file (us_etf_ohlcv.csv, which holds SPY) ends on 2026-11-18, so this fetch ends there too.

Files written:
    us_gold_ohlcv.csv        ohlcv_eikon_us: date, ticker, ric, open, high, low, close, volume — prices as traded
    us_gold_eodhd_check.csv  us_ohlcv_eodhd: date, ticker, close, adjusted_close — a second source, read only by the review
The distribution table (us_corp_action_div_master_b_ed) is queried too: GLD is a grantor trust that pays none, so the
fetch stops if any event is found (it would need a dividend file and rule H4).
"""
import os
import sys
from datetime import date

import pandas as pd
import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["GLD"]
END = "2026-11-18"   # the last date of the approved us_etf_ohlcv.csv (SPY)


def main():
    need = ["WM_DB_HOST", "WM_DB_NAME", "WM_DB_USER", "WM_DB_PASSWORD"]
    if any(not os.environ.get(k) for k in need):
        sys.exit(f"set {', '.join(need)} (and WM_DB_PORT) in the environment")
    c = psycopg2.connect(host=os.environ["WM_DB_HOST"], port=int(os.environ.get("WM_DB_PORT", 5432)), dbname=os.environ["WM_DB_NAME"], user=os.environ["WM_DB_USER"],
                         password=os.environ["WM_DB_PASSWORD"], sslmode=os.environ.get("WM_DB_SSLMODE", "require"), connect_timeout=20)
    c.set_session(readonly=True, autocommit=True)
    cur = c.cursor(); cur.execute("SET statement_timeout = '300s'")
    q = lambda sql: pd.read_sql_query(sql, c, params={"t": TICKERS, "end": END})  # noqa: E731
    px = q("select date, ticker, ric, open, high, low, close, volume from ohlcv_eikon_us where ticker = any(%(t)s) and date <= %(end)s order by ticker, date")
    dv = q("select count(*) as n from us_corp_action_div_master_b_ed where ticker = any(%(t)s)")
    ck = q("select date, ticker, close, adjusted_close from us_ohlcv_eodhd where ticker = any(%(t)s) and date <= %(end)s order by ticker, date")
    c.close()
    if int(dv["n"].iloc[0]):
        sys.exit(f"{int(dv['n'].iloc[0])} distribution rows found for {TICKERS}: a dividend file and rule H4 would be needed")
    px["date"] = pd.to_datetime(px["date"]).dt.date
    ck["date"] = pd.to_datetime(ck["date"]).dt.date
    px.to_csv(os.path.join(HERE, "us_gold_ohlcv.csv"), index=False)
    ck.to_csv(os.path.join(HERE, "us_gold_eodhd_check.csv"), index=False)
    print(f"fetched {date.today()}: prices {len(px):,} rows ({px['date'].min()} → {px['date'].max()}); distribution events 0; "
          f"second source {len(ck):,} rows ({ck['date'].min()} → {ck['date'].max()})")


if __name__ == "__main__":
    main()
