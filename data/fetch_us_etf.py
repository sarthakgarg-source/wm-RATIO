"""Fetch the US ETF raw data for review (SPY and 13 broad commodity ETFs) from the research Postgres, read-only.
Writes raw files into data/; they are NOT readable by any study until the data owner approves them on APPROVED_DATA.md
after the review (data/data_review_us.py → data/DATA_REVIEW_US_ETF.md).

    WM_DB_HOST=... WM_DB_PORT=5432 WM_DB_NAME=... WM_DB_USER=... WM_DB_PASSWORD=... python data/fetch_us_etf.py

Credentials come only from the environment; nothing secret is written anywhere. Prod VPN required.

Files written:
    us_etf_ohlcv.csv      ohlcv_eikon_us: date, ticker, ric, open, high, low, close, volume — prices as traded (unadjusted)
    us_etf_dividends.csv  us_corp_action_div_master_b_ed: one row per distinct dividend event (ticker, ex_date, div_type, gross)
    us_etf_eodhd_check.csv us_ohlcv_eodhd: date, ticker, close, adjusted_close — a second source, read only by the review
"""
import os
import sys
from datetime import date

import pandas as pd
import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["SPY", "BCI", "BCD", "CMDY", "CERY", "PDBC", "COMT", "SDCI", "HGER", "COM", "FTGC", "NBCM", "PIT", "GCC"]   # SPY + the 13 ETFs of the desk's commodity-ETF comparison


def main():
    need = ["WM_DB_HOST", "WM_DB_NAME", "WM_DB_USER", "WM_DB_PASSWORD"]
    if any(not os.environ.get(k) for k in need):
        sys.exit(f"set {', '.join(need)} (and WM_DB_PORT) in the environment")
    c = psycopg2.connect(host=os.environ["WM_DB_HOST"], port=int(os.environ.get("WM_DB_PORT", 5432)), dbname=os.environ["WM_DB_NAME"], user=os.environ["WM_DB_USER"],
                         password=os.environ["WM_DB_PASSWORD"], sslmode=os.environ.get("WM_DB_SSLMODE", "require"), connect_timeout=20)
    c.set_session(readonly=True, autocommit=True)
    cur = c.cursor(); cur.execute("SET statement_timeout = '300s'")
    q = lambda sql, params: pd.read_sql_query(sql, c, params=params)  # noqa: E731
    px = q("select date, ticker, ric, open, high, low, close, volume from ohlcv_eikon_us where ticker = any(%(t)s) order by ticker, date", {"t": TICKERS})
    dv = q("""select distinct ticker, ex_date::date as ex_date, div_type, unadjusted_gross::float as gross, div_currency, div_event_status, coalesce(is_rescinded, '') as is_rescinded
              from us_corp_action_div_master_b_ed where ticker = any(%(t)s) order by ticker, ex_date""", {"t": TICKERS})
    ck = q("select date, ticker, close, adjusted_close from us_ohlcv_eodhd where ticker = any(%(t)s) order by ticker, date", {"t": TICKERS})
    c.close()
    px["date"] = pd.to_datetime(px["date"]).dt.date
    # one row per distinct event (the table holds the same event once per pull and once per window it was pulled in)
    dv = dv.drop_duplicates(["ticker", "ex_date", "div_type", "gross", "div_currency", "div_event_status", "is_rescinded"])
    ck["date"] = pd.to_datetime(ck["date"]).dt.date
    px.to_csv(os.path.join(HERE, "us_etf_ohlcv.csv"), index=False)
    dv.to_csv(os.path.join(HERE, "us_etf_dividends.csv"), index=False)
    ck.to_csv(os.path.join(HERE, "us_etf_eodhd_check.csv"), index=False)
    print(f"fetched {date.today()}: prices {len(px):,} rows ({px['ticker'].nunique()} tickers, {px['date'].min()} → {px['date'].max()}); "
          f"dividend events {len(dv):,}; second source {len(ck):,} rows ({ck['ticker'].nunique()} tickers)")


if __name__ == "__main__":
    main()
