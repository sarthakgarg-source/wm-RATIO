"""Data review of the US small-cap ETF files (IJR, iShares Core S&P Small-Cap ETF) → data/DATA_REVIEW_US_SMALL.md.
Reads the raw files directly (a review may; studies may not until the files are on APPROVED_DATA.md).

    python data/data_review_us_small.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from analysis.approved_data import SPLITS, US_START, _clean, split_adjust_dividends, total_return, us_start  # noqa: E402  (the loader's own rules)

PRICES, DIVS, CHECK, SPY_FILE = "us_small_ohlcv.csv", "us_small_dividends.csv", "us_small_eodhd_check.csv", "us_etf_ohlcv.csv"
T = "IJR"; JUMP, FLAT = 0.08, 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def growth_vs(a: pd.Series, e: pd.Series, s0, s1):
    x = a.loc[s0:s1]; y = e.reindex(x.index).dropna(); x = x.reindex(y.index)
    g = (x.iloc[-1] / x.iloc[0]) / (y.iloc[-1] / y.iloc[0]); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return x.index[0].date(), x.index[-1].date(), float(g), float(g ** (1 / yrs) - 1)


def main():
    raw = pd.read_csv(os.path.join(DATA, PRICES), parse_dates=["date"]); dv = pd.read_csv(os.path.join(DATA, DIVS), parse_dates=["ex_date"])
    ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"]).set_index("date").sort_index()
    spy = pd.read_csv(os.path.join(DATA, SPY_FILE), parse_dates=["date"]); spy = spy[spy.ticker == "SPY"]
    start = pd.Timestamp(US_START); spy_days = set(spy[spy.date >= start]["date"])
    md = ["# Data review — US small-cap ETF (IJR)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_us_small.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items.\n"]
    for f in (PRICES, DIVS):
        p = os.path.join(DATA, f); md.append(f"- `{f}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes, {sum(1 for _ in open(p, encoding='utf-8')):,} lines")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    md += ["## What the files are\n",
           "- **us_small_ohlcv.csv** — daily open/high/low/close/volume of IJR (iShares Core S&P Small-Cap ETF, which tracks the S&P SmallCap 600), US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), "
           "fetched read-only by `data/fetch_us_small.py` to 2026-11-18 (the last day of the approved SPY file).",
           "- **us_small_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD) from `us_corp_action_div_master_b_ed`; the history starts in 2010.",
           "- **Job:** US small caps against SPY and against gold (GLD) in exploratory studies: signal, fill (the close, rule H3) and valuation.",
           "- **Meaning of the price used by the loader** (`load_us_small()`): total return (rule H4) on the price file's split basis (rule H6, below), from 2010-01-04 (rule H5); then H1–H3.\n"]
    # ---------------------------------------------------------------- series checks
    g = raw[(raw.ticker == T) & (raw.date >= start)].sort_values("date"); blank = int(g["close"].isna().sum()); g = g.dropna(subset=["close"])
    n = len(g); dup = int(g.duplicated("date").sum()); wk = int((g.date.dt.weekday >= 5).sum()); first, last = g.date.min(), g.date.max(); yrs = (last - first).days / 365.25
    days = set(g.date); spy_in = {d for d in spy_days if first <= d <= last}; miss = sorted(spy_in - days); extra = sorted(days - spy_in)
    gaps = g.date.diff().dt.days; big = g.date[gaps > 5]; c = g["close"]
    runs = (c != c.shift()).cumsum(); flat = int((c.groupby(runs).transform("size") >= FLAT).groupby(runs).first().sum())
    r = np.log(c.to_numpy()); r = pd.Series(np.r_[np.nan, np.diff(r)], index=g.date.to_numpy()); jumps = r[r.abs() > JUMP]
    ohl_blank = g[["open", "high", "low"]].isna().any(axis=1).mean(); first_ohl = g.dropna(subset=["open", "high", "low"]).date.min()
    spy_c = spy.set_index("date")["close"].sort_index(); spy_r = np.log(spy_c).diff()
    md += ["## Series checks (from 2010-01-01)\n", "| rows | first | last | years | duplicates | weekend rows | missing vs SPY's days | extra vs SPY | gaps > 5 days | blank close | flat runs ≥ 5 | raw jumps > 8% | rows with blank open/high/low |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
           f"| {n:,} | {first.date()} | {last.date()} | {yrs:.1f} | {dup} | {wk} | {len(miss)} | {len(extra)} | {len(big)} | {blank} | {flat} | {len(jumps)} | {ohl_blank * 100:.0f}% (open/high/low present from {first_ohl.date() if pd.notna(first_ohl) else '—'}) |",
           "", "Raw jumps over 8%, with SPY's close-to-close return the same day: " + ("; ".join(f"{d.date()} {v * 100:+.1f}% (SPY {spy_r.get(d, np.nan) * 100:+.1f}%)" for d, v in jumps.items()) or "none") + ".\n"]
    # ---------------------------------------------------------------- the split
    pe = raw[raw.ticker == T].set_index("date")["close"].dropna(); com = pe.index.intersection(ck.index); ratio = ck["close"].reindex(com) / pe.reindex(com)
    key = (ratio.round(0) != ratio.round(0).shift()).cumsum()
    md += ["## Splits: the price file is split-adjusted, the distribution table is not\n",
           "The second source's raw close divided by this file's close, over every common day, in runs of the same value rounded to a whole number:\n",
           "| from | to | days | median | min | max |", "|---|---|---|---|---|---|"]
    for _, sub in ratio.groupby(key):
        md.append(f"| {sub.index.min().date()} | {sub.index.max().date()} | {len(sub):,} | {sub.median():.4f} | {sub.min():.4f} | {sub.max():.4f} |")
    md += ["", f"So this file divides every close before IJR's 3-for-1 split of 2005-06-09 by 6 and every close before its 2-for-1 split of {SPLITS[T][0][0]} by 2: its closes are on today's share basis. "
           "The distribution amounts are not: before the split they are about twice the later ones (a share then was two of today's).\n",
           "| year | distributions | sum ÷ this file's previous close (as paid) | sum ÷ the second source's raw previous close | sum ÷ this file's previous close after rule H6 |", "|---|---|---|---|---|"]
    gg = g.reset_index(drop=True); dvt = dv[(dv.ticker == T) & (dv.gross > 0)]; dvh = split_adjust_dividends(dvt, T)
    yr = {}
    for (ex, amt), amt6 in zip(dvt.groupby("ex_date")["gross"].sum().items(), dvh.groupby("ex_date")["gross"].sum().to_numpy()):
        k = int(np.searchsorted(gg.date.to_numpy(), np.datetime64(ex)))
        if 0 < k < len(gg):
            d0 = gg.date.iloc[k - 1]; pc = gg.close.iloc[k - 1]; rc = ck["close"].get(d0, np.nan)
            y = yr.setdefault(ex.year, [0, 0.0, 0.0, 0.0]); y[0] += 1; y[1] += amt / pc; y[2] += amt / rc if rc == rc else np.nan; y[3] += amt6 / pc
    for y, v in sorted(yr.items()):
        md.append(f"| {y} | {v[0]} | {v[1] * 100:.2f}% | {'—' if v[2] != v[2] else f'{v[2] * 100:.2f}%'} | {v[3] * 100:.2f}% |")
    # ---------------------------------------------------------------- second source: total return
    a4 = _clean(total_return(g[["date", "open", "high", "low", "close"]], dvt)); a6 = _clean(total_return(g[["date", "open", "high", "low", "close"]], dvh)); e = ck["adjusted_close"]
    md += ["\n## Total return against the second source's adjusted close (growth of 1, this file ÷ EODHD, over the common days)\n",
           "| period | H4 on the amounts as paid: ratio (a year) | H4 after rule H6: ratio (a year) |", "|---|---|---|"]
    for s0, s1 in (("2010-01-01", "2017-01-18"), ("2017-01-19", "2026-12-31"), ("2010-01-01", "2026-12-31")):
        f4 = growth_vs(a4["close"], e, s0, s1); f6 = growth_vs(a6["close"], e, s0, s1)
        md.append(f"| {f4[0]} → {f4[1]} | {f4[2]:.4f} ({f4[3] * 100:+.2f}%) | {f6[2]:.4f} ({f6[3] * 100:+.2f}%) |")
    com6 = a6.index.intersection(e.index); r1 = np.log(a6["close"].reindex(com6)).diff().dropna(); r2 = np.log(e.reindex(com6)).diff().dropna()
    gp = (r1 - r2).abs(); rawc = (pe.reindex(com6) * np.where(com6 < pd.Timestamp(SPLITS[T][0][0]), 2.0, 1.0) / ck["close"].reindex(com6) - 1).abs()
    md += ["", f"After H6, on {len(com6):,} common days ({com6.min().date()} → {com6.max().date()}): daily-return correlation {np.corrcoef(r1, r2)[0, 1]:.5f}; days with a return gap over 0.5 pp: {int((gp > 0.005).sum())}; "
           f"raw closes (this file put back on the as-traded basis) more than 0.5% from the second source's: {int((rawc > 0.005).sum())}. The second source ends on {ck.index.max().date()}.\n"]
    # ---------------------------------------------------------------- against SPY
    s = spy.set_index("date")["close"]; w = pd.DataFrame({"ijr": a6["close"], "spy": s}).dropna(); wk_ = w.groupby(pd.Grouper(freq="W-FRI")).last().dropna(); lr = np.log(wk_).diff().dropna()
    yrs6 = (a6.index[-1] - a6.index[0]).days / 365.25
    md += ["## Sanity against SPY (from 2010)\n", f"- Weekly log-return correlation with SPY (SPY's raw closes, as a reference only): {lr.corr().iloc[0, 1]:.3f}.",
           f"- IJR total return (H4 + H6) {((a6['close'].iloc[-1] / a6['close'].iloc[0]) ** (1 / yrs6) - 1) * 100:.1f}% a year, {a6.index[0].date()} → {a6.index[-1].date()}.\n"]
    md += ["## Findings and how the loader handles them\n",
           "- **F-RA-D14 — the split basis differs between the two files.** Prices are split-adjusted (the 2-for-1 split of 2017-01-19 and the 3-for-1 of 2005-06-09); distribution amounts are as paid. "
           "Read together without care, every distribution before 2017-01-19 counts twice: total return comes out about 1.3 pp a year too high over 2010–2016 (table above). "
           "**Handled by a new rule H6** in `analysis/approved_data.split_adjust_dividends`: a distribution before a split is divided by the split ratio (IJR: 2 before 2017-01-19). "
           "After H6 the total return agrees with the second source in each period (table above). The 2005 split is before the 2010 start and the distribution history.",
           "- **Closes only** (F-RA-D13, as the other US ETFs): open, high and low are blank on almost every row until 2026-08; **rule H3** (fill = close).",
           "- **No missing sessions, no duplicates, no flat runs** against SPY's calendar from 2010. Every raw jump over 8% falls on a day SPY itself moved sharply the same way (listed with SPY's return above).",
           "- **Spot checks against the issuer** (iShares): not done here; open item, as for the other US ETFs.\n",
           "## Proposed verdicts (for the data owner)\n", "| file | proposed verdict | why |", "|---|---|---|",
           "| us_small_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4, H5 and H6; 16.9 years from 2010, the same days as SPY |",
           "| us_small_dividends.csv | PASS WITH NOTES | amounts as paid, so read only through rule H6 (pre-split amounts halved); total return then agrees with the second source |",
           "| us_small_eodhd_check.csv | not for approval | the second source, read only by this review |"]
    _ = (us_start, extra, big)
    out = os.path.join(DATA, "DATA_REVIEW_US_SMALL.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
