"""The data boundary in code. The only way a test may open a data file: it must be on data/APPROVED_DATA.md and its
sha256 must match at the moment of opening. Everything else raises.

    from analysis.approved_data import load_daily
    px = load_daily()          # dict: asset -> DataFrame(date index; open, high, low, close), NSE/MCX daily bars
"""
import hashlib
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
INDEX_KEYS = {"nifty100": "N100", "midcap100": "MC100", "smallcap100": "SC100", "microcap250": "Micro250"}
GOLD = "GoldMCX"
ASSETS = ["N100", "MC100", "SC100", "Micro250", GOLD]


class NotApproved(RuntimeError):
    pass


def _approved() -> dict:
    txt = open(os.path.join(DATA, "APPROVED_DATA.md")).read()
    return dict(re.findall(r"^\| (\S+\.csv) \| `([0-9a-f]{64})` \|", txt, flags=re.M))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_approved(fname: str) -> str:
    """Return the path of an approved file after checking its fingerprint; raise otherwise."""
    ap = _approved()
    if fname not in ap:
        raise NotApproved(f"{fname} is not on data/APPROVED_DATA.md")
    path = os.path.join(DATA, fname)
    if _sha256(path) != ap[fname]:
        raise NotApproved(f"{fname}: sha256 does not match the approved fingerprint")
    return path


def _clean(s: pd.DataFrame) -> pd.DataFrame:
    """The two handling rules of the data review (DATA_REVIEW.md, "how the loader handles them"), applied the same way to every series:
    H1. a row whose open, high and low are all equal while the close differs (a stale OHL print — SC100, 2011-04-01 … 04-07)
        gets `fill` = close instead of OHLC4 — tested on the raw row, before H2;
    H2. high := max(high, open, close) and low := min(low, open, close) — repairs the 1-point rounding rows and the 2009-05-18
        circuit-breaker session where the close printed above the high. Every other row gets fill = (open + high + low + close) / 4;
    H3. a row with a close but a blank open, high or low (gold, 3 rows) gets fill = close and is counted with the stale rows.
    Rows with a blank close are dropped (the day is simply absent from that leg's calendar); nothing is carried forward."""
    s = s.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").set_index("date").astype(float)
    stale = (s["open"] == s["high"]) & (s["high"] == s["low"]) & (s["low"] != s["close"])
    s["high"] = s[["high", "open", "close"]].max(axis=1); s["low"] = s[["low", "open", "close"]].min(axis=1)
    s["fill"] = (s["open"] + s["high"] + s["low"] + s["close"]) / 4.0
    s.loc[stale, "fill"] = s.loc[stale, "close"]
    blank = s["fill"].isna()                                   # H3: a close with a blank open / high / low (gold, 3 rows): fill = close
    s.loc[blank, "fill"] = s.loc[blank, "close"]
    s["stale_ohl"] = stale | blank
    return s


def load_daily() -> dict:
    """Every asset's daily bars as its own frame (date index, sorted, no duplicates; columns open, high, low, close, fill,
    stale_ohl). `fill` is the assumed trade price (OHLC4, or the close on a stale-OHL row). Nothing is filled or carried forward."""
    out = {}
    raw = pd.read_csv(open_approved("index_ohlcv_all.csv"), parse_dates=["date"])
    for key, name in INDEX_KEYS.items():
        out[name] = _clean(raw[raw["index_key"] == key][["date", "open", "high", "low", "close"]])
    g = pd.read_csv(open_approved("MAUc1_daily.csv"), parse_dates=["Date"]).rename(columns={"Date": "date", "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close"})
    out[GOLD] = _clean(g[["date", "open", "high", "low", "close"]])
    return out


# ------------------------------------------------------------------------------------------------------- US ETFs (added 2026-11-19)
US_PRICES, US_DIVIDENDS = "us_etf_ohlcv.csv", "us_etf_dividends.csv"
US_START = "2010-01-01"   # the source's dividend history starts in 2010: earlier prices cannot be put on a total-return basis


def total_return(s: pd.DataFrame, dv: pd.DataFrame) -> pd.DataFrame:
    """Rule H4 (US ETFs): dividends reinvested. Every open/high/low/close before an ex-date is multiplied by
    (1 − D ÷ the last close before the ex-date), compounded backwards, so the latest prices are the traded ones and a
    holder's return includes every distribution. D = the sum of the gross amounts on that ex-date; zero amounts and
    ex-dates outside the price history are ignored. `s`: date, open, high, low, close (raw, sorted); `dv`: ex_date, gross."""
    s = s.sort_values("date").reset_index(drop=True).copy(); dates = s["date"].to_numpy()
    d = dv[dv["gross"] > 0].groupby("ex_date")["gross"].sum()
    f = pd.Series(1.0, index=s.index)
    for ex, amt in d.items():
        k = int(np.searchsorted(dates, np.datetime64(ex), side="left"))   # the first trading day on or after the ex-date
        if k <= 0 or k >= len(s):
            continue
        f.iloc[:k] *= 1.0 - amt / float(s["close"].iloc[k - 1])
    for col in ("open", "high", "low", "close"):
        s[col] = s[col] * f.to_numpy()
    return s


def us_start(listing: pd.Timestamp) -> pd.Timestamp:
    """Rule H5 (US ETFs): the launch period is dropped. A series starts on the first 1 January at least 12 months after
    its first day in the file, and not before 2010-01-01. (The review found the thin early trading there: missing
    sessions, and closes that disagree with a second source, e.g. PDBC in 2015.)"""
    t = pd.Timestamp(listing) + pd.DateOffset(years=1)
    jan1 = pd.Timestamp(year=t.year if (t.month == 1 and t.day == 1) else t.year + 1, month=1, day=1)
    return max(pd.Timestamp(US_START), jan1)


def load_us_etf() -> dict:
    """Each US ETF's daily bars from its start (rule H5) on a total-return basis (rule H4), then the same handling rules
    H1–H3 as load_daily(): dict ticker -> DataFrame(date index; open, high, low, close, fill, stale_ohl). US dollars.
    The source has closes only on most rows (blank open/high/low), so rule H3 makes the fill price the close there."""
    raw = pd.read_csv(open_approved(US_PRICES), parse_dates=["date"]); dv = pd.read_csv(open_approved(US_DIVIDENDS), parse_dates=["ex_date"])
    out = {}
    for t, g in raw.groupby("ticker"):
        g = g.dropna(subset=["close"]).drop_duplicates("date")
        s = g[g["date"] >= us_start(g["date"].min())][["date", "open", "high", "low", "close"]]
        out[t] = _clean(total_return(s, dv[dv["ticker"] == t]))
    return out


def fingerprints() -> dict:
    return {f: _sha256(os.path.join(DATA, f)) for f in _approved()}
