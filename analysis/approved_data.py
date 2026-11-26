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


# ------------------------------------------------------------------------------------------------------- US gold ETF (added 2026-11-20)
US_GOLD = "us_gold_ohlcv.csv"


def load_us_gold() -> dict:
    """GLD (SPDR Gold Shares), for SPY against gold: the same rules as load_us_etf() — start under rule H5, then H1–H3.
    GLD pays no distributions (none in the source; review DATA_REVIEW_US_GOLD.md), so its price is already a total
    return and rule H4 has nothing to apply. dict 'GLD' -> DataFrame(date index; open, high, low, close, fill, stale_ohl)."""
    raw = pd.read_csv(open_approved(US_GOLD), parse_dates=["date"])
    out = {}
    for t, g in raw.groupby("ticker"):
        g = g.dropna(subset=["close"]).drop_duplicates("date")
        out[t] = _clean(g[g["date"] >= us_start(g["date"].min())][["date", "open", "high", "low", "close"]].sort_values("date"))
    return out


# ------------------------------------------------------------------------------------------------------- US small-cap ETF (added 2026-11-23)
US_SMALL_PRICES, US_SMALL_DIVIDENDS = "us_small_ohlcv.csv", "us_small_dividends.csv"
SPLITS = {"IJR": (("2017-01-19", 2.0),)}   # rule H6: measured in DATA_REVIEW_US_SMALL.md from the second source's raw closes


def split_adjust_dividends(dv: pd.DataFrame, ticker: str, splits: dict = None) -> pd.DataFrame:
    """Rule H6 (IJR): the price file is split-adjusted (its closes are the second source's raw closes divided by the split
    ratio before each split) but the distribution table holds the amounts as paid, so a distribution whose ex-date is
    before a split is divided by that split's ratio. IJR: 2-for-1 on 2017-01-19. `splits` (ticker -> ((date, ratio), ...))
    replaces the IJR table SPLITS when given (the factor ETFs: factor_splits())."""
    dv = dv.copy()
    for d, k in (SPLITS if splits is None else splits).get(ticker, ()):
        dv.loc[dv["ex_date"] < pd.Timestamp(d), "gross"] = dv.loc[dv["ex_date"] < pd.Timestamp(d), "gross"] / k
    return dv


def load_us_small() -> dict:
    """IJR (iShares Core S&P Small-Cap ETF): start under rule H5, distributions put on the price file's split basis (H6),
    total return (H4), then H1–H3 as load_us_etf(). dict 'IJR' -> DataFrame(date index; open, high, low, close, fill, stale_ohl)."""
    raw = pd.read_csv(open_approved(US_SMALL_PRICES), parse_dates=["date"]); dv = pd.read_csv(open_approved(US_SMALL_DIVIDENDS), parse_dates=["ex_date"])
    out = {}
    for t, g in raw.groupby("ticker"):
        g = g.dropna(subset=["close"]).drop_duplicates("date")
        s = g[g["date"] >= us_start(g["date"].min())][["date", "open", "high", "low", "close"]].sort_values("date")
        out[t] = _clean(total_return(s, split_adjust_dividends(dv[dv["ticker"] == t], t)))
    return out


# ------------------------------------------------------------------------------------------------------- US factor and style ETFs (added 2026-11-26)
US_FACTOR_BATCHES = (("us_factor_ohlcv.csv", "us_factor_dividends.csv", "us_factor_splits.csv"),           # DATA_REVIEW_US_FACTOR.md
                     ("us_sp_factor_ohlcv.csv", "us_sp_factor_dividends.csv", "us_sp_factor_splits.csv"))  # DATA_REVIEW_US_SP_FACTOR.md
US_FACTOR_START = {"SPMO": "2018-01-01"}   # F-RA-D16: SPMO traded thinly until 2018 — a per-fund override of rule H5


def factor_splits() -> dict:
    """Rule H6's splits for the factor ETFs, from the approved splits files (F-RA-D15): ticker -> ((ex-date, new shares ÷ old
    shares), ...). MTUM, QUAL and IWF 4-for-1, NOBL 2-for-1, all in 2026; none for SPMO, SPHQ and RPV."""
    out = {}
    for _, _, f in US_FACTOR_BATCHES:
        sp = pd.read_csv(open_approved(f), parse_dates=["ex_date"])
        sp = sp[(sp["event_type"] == "Share Split") & (sp["event_status"] == "Approved") & (sp["is_rescinded"].astype(str) != "True")]
        for t, s in sp.groupby("ticker"):
            out[t] = out.get(t, ()) + tuple((r.ex_date.strftime("%Y-%m-%d"), r.new_shares / r.old_shares) for r in s.itertuples())
    return out


def load_us_factor() -> dict:
    """The US factor and style ETFs, from two reviews:
      DATA_REVIEW_US_FACTOR.md — MTUM (MSCI momentum), QUAL (MSCI quality), VLUE (MSCI enhanced value), USMV (MSCI min vol),
        SPLV (S&P 500 low volatility), NOBL (S&P 500 dividend aristocrats), RSP (S&P 500 equal weight), IWF (Russell 1000 growth);
      DATA_REVIEW_US_SP_FACTOR.md — SPMO (S&P 500 momentum), SPHQ (S&P 500 quality), RPV (S&P 500 pure value, standing in for
        SPVU, S&P 500 enhanced value, which the research database does not hold).
    Each fund starts under rule H5 (SPMO on 2018-01-01: F-RA-D16), distributions are put on the price file's split basis (rule
    H6, splits from factor_splits()), total return (H4), then H1–H3 as load_us_etf(). MTUM and QUAL have open, high and low on
    every row, so their fill is OHLC4; the others have closes only (fill = close, rule H3). A study comparing them should fill
    every fund at the close. SPHQ changed index on 2016-03-21 (DATA_REVIEW_US_SP_FACTOR.md).
    dict ticker -> DataFrame(date index; open, high, low, close, fill, stale_ohl). US dollars."""
    spl = factor_splits(); out = {}
    for fp, fd, _ in US_FACTOR_BATCHES:
        raw = pd.read_csv(open_approved(fp), parse_dates=["date"]); dv = pd.read_csv(open_approved(fd), parse_dates=["ex_date"])
        for t, g in raw.groupby("ticker"):
            g = g.dropna(subset=["close"]).drop_duplicates("date")
            st = max(us_start(g["date"].min()), pd.Timestamp(US_FACTOR_START.get(t, US_START)))
            s = g[g["date"] >= st][["date", "open", "high", "low", "close"]].sort_values("date")
            out[t] = _clean(total_return(s, split_adjust_dividends(dv[dv["ticker"] == t], t, spl)))
    return out


# ------------------------------------------------------------------------------------------------------- the latest Indian series (added 2026-11-26)
AA_INDEX = {"N100": "n100", "MC100": "mc100", "SC100": "sc100"}
AA_GOLD, AA_LBES, AA_CASE = "aa_mauc1_daily.csv", "aa_liquid_lbes.csv", "aa_liquid_case.csv"
AA_GOLD_START = "2005-01-01"   # rule H8 (F-RA-D18): MCX gold from the first day the research Postgres (the review's second source) covers


def _level(dates, values) -> pd.DataFrame:
    """A cash level as daily bars: open = high = low = close = the level, so the fill is the level."""
    s = pd.DataFrame({"date": pd.DatetimeIndex(dates), "close": pd.Series(values).to_numpy(float)}); s["open"] = s["high"] = s["low"] = s["close"]
    return _clean(s[["date", "open", "high", "low", "close"]])


def load_india_latest() -> dict:
    """The desk's latest Indian series, copied from its asset-allocation repository (wm-AA; DATA_REVIEW_AA.md):
      N100, MC100, SC100 — daily bars to 2026-11-20, equal to load_daily()'s on every common day; rules H1–H3;
      GOLD — MCX gold (MAUc1) from 2005-01-01 to 2026-11-20: rules H7 (a row with a non-positive close is dropped) and H8
        (from 2005-01-01), F-RA-D18; then H1–H3;
      'Liquid' — wm-AA's liquid-ETF cash leg (LBES) and 'LiquidCase' — the LIQUIDCASE level (the fund's traded close from
        2024-01-24), both to 2026-11-25, as single daily prices (open = high = low = close). Before 2024-01-24 the two are one
        series with no recorded source (F-RA-D17): every study that uses them must say so.
    No Micro250 (wm-AA has no newer file). load_daily() is unchanged, so the locked tests keep their inputs.
    dict name -> DataFrame(date index; open, high, low, close, fill, stale_ohl)."""
    out = {}
    for name, k in AA_INDEX.items():
        a = pd.read_csv(open_approved(f"aa_index_{k}_pre2019.csv"), parse_dates=["date"]); b = pd.read_csv(open_approved(f"aa_index_{k}_2019_2026.csv"), parse_dates=["date"])
        out[name] = _clean(pd.concat([a[a["date"] < "2019-01-01"], b])[["date", "open", "high", "low", "close"]])
    g = pd.read_csv(open_approved(AA_GOLD), parse_dates=["date"])
    out[GOLD] = _clean(g[(g["date"] >= AA_GOLD_START) & (g["close"] > 0)][["date", "open", "high", "low", "close"]])   # rules H8 and H7
    lb = pd.read_csv(open_approved(AA_LBES)); out["Liquid"] = _level(pd.to_datetime(lb["date"], format="%d/%m/%y %H:%M"), lb["price"])
    lc = pd.read_csv(open_approved(AA_CASE)); out["LiquidCase"] = _level(pd.to_datetime(lc["date"], format="%d-%m-%Y"), lc["liquid_case"])
    return out


def fingerprints() -> dict:
    return {f: _sha256(os.path.join(DATA, f)) for f in _approved()}
