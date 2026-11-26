"""Data review of the US S&P 500 factor ETF files (SPMO momentum, SPHQ quality, RPV pure value) →
data/DATA_REVIEW_US_SP_FACTOR.md. Reads the raw files directly (a review may; studies may not until the files are on
APPROVED_DATA.md); SPY's calendar and total return come from the approved file through the loader. The checks are the
factor review's (data_review_us_factor.py), imported from it.

    python data/data_review_us_sp_factor.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT); sys.path.insert(0, DATA)
from analysis.approved_data import US_START, _clean, load_us_etf, total_return, us_start  # noqa: E402  (the loader's own rules)
from data_review_us_factor import JUMP, FLAT, MIN_YEARS, exdate_gaps, growth, h6, lst, sha256, yields  # noqa: E402  (the same checks)

PRICES, DIVS, SPLITF, CHECK = "us_sp_factor_ohlcv.csv", "us_sp_factor_dividends.csv", "us_sp_factor_splits.csv", "us_sp_factor_eodhd_check.csv"
ORDER = ["SPMO", "SPHQ", "RPV"]
SPMO_START = "2018-01-01"   # the proposed handling of F-RA-D16: SPMO's first calendar year with every session
NAME = {"SPMO": "Invesco S&P 500 Momentum (S&P 500 Momentum Index)", "SPHQ": "Invesco S&P 500 Quality (S&P 500 Quality Index)",
        "RPV": "Invesco S&P 500 Pure Value (S&P 500 Pure Value Index) — standing in for SPVU (S&P 500 Enhanced Value), which the research database does not hold"}


def main():
    raw = pd.read_csv(os.path.join(DATA, PRICES), parse_dates=["date"]); dv = pd.read_csv(os.path.join(DATA, DIVS), parse_dates=["ex_date"])
    sp = pd.read_csv(os.path.join(DATA, SPLITF), parse_dates=["ex_date", "effective_date"]); ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"])
    spl = sp[(sp.event_type == "Share Split") & (sp.event_status == "Approved") & (sp.is_rescinded.astype(str) != "True")] if len(sp) else sp
    SPL = {t: tuple((r.ex_date, r.new_shares / r.old_shares) for r in s.itertuples()) for t, s in spl.groupby("ticker")} if len(spl) else {}
    spy = load_us_etf()["SPY"]; spy_days = set(spy.index); spy_c = spy["close"]; spy_r = np.log(spy_c).diff(); start = pd.Timestamp(US_START)
    G = {t: raw[raw.ticker == t].dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True) for t in ORDER}
    DV = {t: dv[(dv.ticker == t) & (dv.gross > 0)] for t in ORDER}; DV6 = {t: h6(DV[t], SPL, t) for t in ORDER}
    E = {t: ck[ck.ticker == t].set_index("date").sort_index() for t in ORDER}
    md = ["# Data review — US S&P 500 factor ETFs (SPMO, SPHQ, RPV)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_us_sp_factor.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items. SPY (S&P 500) is already approved in `us_etf_ohlcv.csv` and is the reference calendar and return here; "
          "the rest of the desk's S&P factor list (SPLV low volatility, NOBL dividend aristocrats, RSP equal weight, and IWF Russell 1000 growth) is approved in `us_factor_ohlcv.csv`.\n"]
    for f in (PRICES, DIVS, SPLITF):
        p = os.path.join(DATA, f); md.append(f"- `{f}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes, {sum(1 for _ in open(p, encoding='utf-8')):,} lines")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    md += ["## What the files are\n",
           "- **us_sp_factor_ohlcv.csv** — daily open/high/low/close/volume, US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_sp_factor.py` to 2026-11-18 "
           "(the last day of the approved SPY file); split-adjusted, not adjusted for distributions:"] + [f"    - **{t}** — {NAME[t]}" for t in ORDER] + [
           "- **us_sp_factor_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD, as paid) from `us_corp_action_div_master_b_ed`; the history starts in 2010.",
           f"- **us_sp_factor_splits.csv** — share splits from `us_corp_action_cap_struc_master_b_ed`: {len(spl)} for these funds (the table is not a full history; the checks below look for splits from 2010 independently).",
           "- **Job:** S&P 500 factor ETFs against SPY and each other in exploratory studies: signal, fill (the close, rule H3) and valuation.",
           "- **Meaning of the price a loader would use:** total return (rule H4) from each fund's start under rule H5, then H1–H3, as `load_us_factor()`.\n"]
    # ---------------------------------------------------------------- per series, from 2010
    md += ["## Series checks (from 2010-01-01)\n", "| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | blank open/high/low | distributions (n, sum as paid, largest) |",
           "|---|" + "---|" * 16]
    notes, ohl_share, miss_all = [], {}, {}
    for t in ORDER:
        g0 = raw[(raw.ticker == t) & (raw.date >= start)].sort_values("date"); blank = int(g0["close"].isna().sum()); g = g0.dropna(subset=["close"])
        n = len(g); dup = int(g.duplicated("date").sum()); wk = int((g.date.dt.weekday >= 5).sum()); first, last = g.date.min(), g.date.max(); yrs = (last - first).days / 365.25
        days = set(g.date); spy_in = {d for d in spy_days if first <= d <= last}; miss = sorted(spy_in - days); extra = sorted(days - spy_in); miss_all[t] = miss
        gaps = g.date.diff().dt.days; big = g.date[gaps > 5]
        o, h, l_, c = g["open"], g["high"], g["low"], g["close"]
        ordr = int(((h < np.maximum(o, c) - 1e-9) | (l_ > np.minimum(o, c) + 1e-9)).sum()); eq = int(((o == h) & (h == l_) & (l_ != c)).sum())
        runs = (c != c.shift()).cumsum(); flat = int((c.groupby(runs).transform("size") >= FLAT).groupby(runs).first().sum())
        r = pd.Series(np.log(c.to_numpy()), index=g.date.to_numpy()).diff(); jumps = r[r.abs() > JUMP]
        d_in = DV[t][(DV[t].ex_date >= first) & (DV[t].ex_date <= last)]; ohl_share[t] = g[["open", "high", "low"]].isna().any(axis=1).mean()
        md.append(f"| {t} | {n:,} | {first.date()} | {last.date()} | {yrs:.1f} | {dup} | {wk} | {len(miss)} | {len(extra)} | {len(big)} | {blank} | {ordr} | {eq} | {flat} | {len(jumps)} | {ohl_share[t] * 100:.0f}% | "
                  f"{len(d_in)} (${d_in.gross.sum():.2f}, ${d_in.gross.max() if len(d_in) else 0:.2f}) |")
        if miss:
            yc = pd.Series([d.year for d in miss]).value_counts().sort_index()
            notes.append(f"- **{t}** misses {len(miss)} of SPY's trading days (by year: " + ", ".join(f"{y} {v}" for y, v in yc.items()) + "): " + lst(miss))
        if extra:
            notes.append(f"- **{t}** has {len(extra)} days SPY does not: " + lst(extra))
        if len(big):
            notes.append(f"- **{t}** gaps over 5 calendar days end on: " + lst(big))
        if len(jumps):
            notes.append(f"- **{t}** raw jumps > 8% (SPY's return that day): " + "; ".join(f"{d.date()} {v * 100:+.1f}% (SPY {spy_r.get(d, np.nan) * 100:+.1f}%)" for d, v in jumps.items()))
        if eq:
            notes.append(f"- **{t}** {eq} rows with open = high = low ≠ close (stale OHL; rule H1 → fill = close): " + lst(g.date[(o == h) & (h == l_) & (l_ != c)], 8))
        if flat:
            notes.append(f"- **{t}** {flat} runs of 5 or more equal closes")
    md += ["", "### Notes per series\n"] + (notes or ["- none"])
    # ---------------------------------------------------------------- splits
    md += ["\n## Splits: is the price file on one share basis, and are the distributions on the same basis?\n",
           f"Share splits in the corporate-action table: {'; '.join(f'{r.ticker} {r.ex_date.date()} {r.old_shares:g} → {r.new_shares:g}' for r in spl.itertuples()) if len(spl) else 'none for these funds'}.\n",
           "The second source's raw close ÷ this file's close on every common day, in runs of the same value rounded to a whole number:\n",
           "| ticker | runs of the ratio (from → to: rounded value, days) | common days off a whole number by > 0.5% |", "|---|---|---|"]
    for t in ORDER:
        pe = G[t].set_index("date")["close"]; e = E[t]; com = pe.index.intersection(e.index)
        if len(com) < 30:
            md.append(f"| {t} | — (not in the second source) | |"); continue
        ratio = e["close"].reindex(com) / pe.reindex(com); rr = ratio.round(0); key = (rr != rr.shift()).cumsum()
        segs = [(s.index.min(), s.index.max(), int(rr.loc[s.index[0]]), len(s)) for _, s in ratio.groupby(key) if len(s) >= 5]
        md.append(f"| {t} | " + "; ".join(f"{a.date()} → {b.date()}: {v} ({k:,})" for a, b, v, k in segs) + f" | {int(((ratio - rr).abs() / rr > 0.005).sum())} |")
    md += ["", "Distribution yield by year, as paid (sum of the year's amounts ÷ this file's close the day before each ex-date), %; a split adjusted in the prices but not in the amounts would multiply the yield before it:\n",
           "| ticker | " + " | ".join(str(y) for y in range(2010, 2027)) + " |", "|---|" + "---|" * 17]
    for t in ORDER:
        yy = yields(G[t], DV6[t].groupby("ex_date")["gross"].sum()); md.append(f"| {t} | " + " | ".join(f"{yy[y] * 100:.2f}" if y in yy else "—" for y in range(2010, 2027)) + " |")
    # ---------------------------------------------------------------- distributions on their ex-dates
    md += ["\n## Distributions: does the price fall by about the amount on the ex-date?\n",
           "Over every distribution from 2010: the close-to-close fall on the ex-date minus SPY's fall that day, against the amount, both as % of the previous close; the median of (fall − amount) is near zero when the amounts are on the price file's basis.\n",
           "| ticker | ex-dates | median amount | median (market-adjusted fall − amount) | mean (market-adjusted fall − amount) |", "|---|---|---|---|---|"]
    gaps_ = {}
    for t in ORDER:
        a = exdate_gaps(G[t][G[t].date >= start].reset_index(drop=True), DV6[t].groupby("ex_date")["gross"].sum(), spy_c); gaps_[t] = np.median(a[:, 1] - a[:, 0])
        md.append(f"| {t} | {len(a)} | {np.median(a[:, 0]) * 100:.2f}% | {gaps_[t] * 100:+.2f} pp | {np.mean(a[:, 1] - a[:, 0]) * 100:+.2f} pp |")
    # ---------------------------------------------------------------- second source
    md += ["\n## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available\n",
           "Total return from the loader's rules (H4 on Eikon prices and Refinitiv distributions) against EODHD's adjusted close on the dates both have, from 2010.\n",
           "| ticker | common days | raw closes differing > 0.5% | growth of 1, H4 ÷ EODHD (a year) | daily-return correlation | days with return gap > 0.5 pp |", "|---|---|---|---|---|---|"]
    gr = {}
    for t in ORDER:
        e = E[t][E[t].index >= start]
        if not len(e):
            md.append(f"| {t} | — (not in the second source) | | | | |"); continue
        g = G[t][G[t].date >= start][["date", "open", "high", "low", "close"]]; a6 = _clean(total_return(g, DV6[t]))["close"]
        g6, yy6, com = growth(a6, e["adjusted_close"]); gr[t] = yy6
        rawc = G[t].set_index("date")["close"]; cr = (rawc.reindex(com) / e["close"].reindex(com) - 1).abs()
        r1 = np.log(a6.reindex(com)).diff().dropna(); r2 = np.log(e["adjusted_close"].reindex(com)).diff().dropna()
        md.append(f"| {t} | {len(com):,} ({com.min().date()} → {com.max().date()}) | {int((cr > 0.005).sum())} | {g6:.4f} ({yy6 * 100:+.2f}%) | {np.corrcoef(r1, r2)[0, 1]:.5f} | {int(((r1 - r2).abs() > 0.005).sum())} |")
    # ---------------------------------------------------------------- after the launch period (rule H5)
    md += ["\n## After the launch period (rule H5: the first 1 January at least 12 months after the first day; not before 2010)\n",
           "| ticker | first day in the file | start under H5 | years | missing vs SPY's days | jumps > 8% (SPY's return that day) | total return a year (SPY, same days) | weekly correlation with SPY | usable for a 52-week rule (≥ 5 years)? |",
           "|---|" + "---|" * 8]
    post_miss, yrs_h5 = {}, {}
    for t in ORDER:
        gfull = G[t]; st = us_start(gfull.date.min()); g = gfull[gfull.date >= st]
        last = g.date.max(); yrs = (last - st).days / 365.25; yrs_h5[t] = yrs; spy_in = {d for d in spy_days if st <= d <= last}; miss = sorted(spy_in - set(g.date)); post_miss[t] = miss
        a = _clean(total_return(g[["date", "open", "high", "low", "close"]], DV6[t])); ra = np.log(a["close"]).diff(); ja = ra[ra.abs() > JUMP]
        w = pd.DataFrame({"x": a["close"], "s": spy_c}).dropna(); yy = (w.index[-1] - w.index[0]).days / 365.25
        cg = lambda s_: (s_.iloc[-1] / s_.iloc[0]) ** (1 / yy) - 1  # noqa: E731
        wk_ = np.log(w.groupby(pd.Grouper(freq="W-FRI")).last().dropna()).diff().dropna()
        md.append(f"| {t} | {gfull.date.min().date()} | {st.date()} | {yrs:.1f} | {len(miss)}{(' (' + lst(miss, 8) + ')') if miss else ''} | "
                  + ("; ".join(f"{d.date()} {v * 100:+.1f}% ({spy_r.get(d, np.nan) * 100:+.1f}%)" for d, v in ja.items()) or "0")
                  + f" | {cg(w['x']) * 100:.1f}% ({cg(w['s']) * 100:.1f}%) | {wk_.corr().iloc[0, 1]:.3f} | {'yes' if yrs >= MIN_YEARS else 'short — report only'} |")
    # ---------------------------------------------------------------- SPMO from 2018 (the proposed handling of F-RA-D16)
    s18 = pd.Timestamp(SPMO_START); g = G["SPMO"]; g18 = g[g.date >= s18]; last = g18.date.max(); yrs18 = (last - s18).days / 365.25
    miss18 = sorted({d for d in spy_days if s18 <= d <= last} - set(g18.date)); c18 = g18["close"]; runs18 = (c18 != c18.shift()).cumsum()
    flat18 = int((c18.groupby(runs18).transform("size") >= FLAT).groupby(runs18).first().sum()); cfull = g["close"]; runsf = (cfull != cfull.shift()).cumsum()
    flat_before = int((cfull.groupby(runsf).transform("size") >= FLAT).groupby(runsf).first().sum()) - flat18
    a18 = exdate_gaps(g18.reset_index(drop=True), DV6["SPMO"][DV6["SPMO"].ex_date >= s18].groupby("ex_date")["gross"].sum(), spy_c)
    aa18 = _clean(total_return(g18[["date", "open", "high", "low", "close"]], DV6["SPMO"])); e18 = E["SPMO"]["adjusted_close"]; gg18, gy18, com18 = growth(aa18["close"], e18)
    w = pd.DataFrame({"x": aa18["close"], "s": spy_c}).dropna(); yy = (w.index[-1] - w.index[0]).days / 365.25
    tr18, sp18 = (w["x"].iloc[-1] / w["x"].iloc[0]) ** (1 / yy) - 1, (w["s"].iloc[-1] / w["s"].iloc[0]) ** (1 / yy) - 1
    miss_y = pd.Series([d.year for d in miss_all["SPMO"]]).value_counts().sort_index() if miss_all["SPMO"] else pd.Series(dtype=int)
    md += [f"\n## SPMO from {SPMO_START} (the proposed handling of F-RA-D16)\n",
           f"| start | years | missing vs SPY's days | flat runs ≥ 5 | ex-dates | median (market-adjusted fall − amount) | growth of 1 vs EODHD (a year), {com18.min().date()} → {com18.max().date()} | total return a year (SPY, same days) |",
           "|---|---|---|---|---|---|---|---|",
           f"| {SPMO_START} | {yrs18:.1f} | {len(miss18)} | {flat18} | {len(a18)} | {np.median(a18[:, 1] - a18[:, 0]) * 100:+.2f} pp | {gg18:.4f} ({gy18 * 100:+.2f}%) | {tr18 * 100:.1f}% ({sp18 * 100:.1f}%) |"]
    # ---------------------------------------------------------------- findings, handling, verdicts
    c_only = [t for t in ORDER if ohl_share[t] > 0.95]; nos = [t for t in ORDER if not len(E[t])]; rpv_r = np.log(G["RPV"].set_index("date")["close"]).diff()
    md += ["\n## Findings and how a loader would handle them\n",
           f"- **F-RA-D16 — SPMO traded thinly for its first two years.** The file has no row on {len(miss_all['SPMO'])} of SPY's trading days from its first day to 2017 (by year: "
           + ", ".join(f"{y} {v}" for y, v in miss_y.items()) + f"), and {flat_before} runs of five or more equal closes; the second source misses the same days, so they are sessions without trades. "
           f"Rule H5 (start {us_start(G['SPMO'].date.min()).date()}) still leaves {len(post_miss['SPMO'])} missing sessions in 2017, which a study on a common calendar would drop for every fund. "
           f"**Handled by starting SPMO on {SPMO_START}**, its first calendar year with every session (a per-fund override of H5): from then {len(miss18)} missing sessions, {flat18} flat runs, "
           f"ex-date gap {np.median(a18[:, 1] - a18[:, 0]) * 100:+.2f} pp, total return {gy18 * 100:+.2f}% a year against the second source; {yrs18:.1f} years.",
           f"- **Distributions are on the price file's basis for all three:** no splits in the corporate-action table, stable yields, ex-date falls near the amounts (SPHQ {gaps_['SPHQ'] * 100:+.2f} pp, RPV {gaps_['RPV'] * 100:+.2f} pp; "
           f"SPMO {gaps_['SPMO'] * 100:+.2f} pp over all years, pulled by the thin 2015–2017 sessions), and SPMO's total return agrees with the second source ({gr['SPMO'] * 100:+.2f}% a year). No rule H6 needed.",
           f"- **Closes only** ({', '.join(c_only)}): open, high and low are blank on almost every row (F-RA-D13); **rule H3** (fill = close).",
           f"- **No second source for {', '.join(nos)}**; the second source ends on {ck.date.max().date()}. These two rest on the ex-date falls, the yields and the weekly correlation with SPY above.",
           "- **SPHQ changed index on 2016-03-21** (TradingView's fund description: before that it tracked an S&P quality index with different metrics on the same S&P 500 universe). "
           "Not a data error, but a study over 2010–2026 measures two definitions of quality; a study should report results before and after the change. Open item: confirm the date and the earlier index against the issuer.",
           "- **RPV stands in for SPVU** (S&P 500 Enhanced Value), which the research database does not hold (neither price table): S&P 500 Pure Value is a concentrated, value-score-weighted cut of the S&P 500, "
           f"the data owner's choice of 2026-11-26. Its jumps over 8% are larger than SPY's on the same days (2020-03-09 {rpv_r.get(pd.Timestamp('2020-03-09'), np.nan) * 100:+.1f}% against "
           f"{spy_r.get(pd.Timestamp('2020-03-09'), np.nan) * 100:+.1f}%), as expected of a deep-value basket.",
           "- **Spot checks against the issuer** (Invesco): not done here; open item, as for the other US ETFs.\n",
           "## Proposed verdicts (for the data owner)\n", "| file | proposed verdict | why |", "|---|---|---|",
           f"| us_sp_factor_ohlcv.csv | PASS WITH NOTES | usable with rules H3, H4 and H5, SPMO from {SPMO_START} (F-RA-D16); SPHQ and RPV {yrs_h5['SPHQ']:.1f} years, SPMO {yrs18:.1f} |",
           "| us_sp_factor_dividends.csv | PASS WITH NOTES | amounts on the price file's basis (no splits from 2010); SPMO's total return agrees with the second source |",
           "| us_sp_factor_splits.csv | PASS WITH NOTES | no splits recorded for these funds; kept so the loader reads the same three files for every factor batch |",
           "| us_sp_factor_eodhd_check.csv | not for approval | the second source, read only by this review |"]
    out = os.path.join(DATA, "DATA_REVIEW_US_SP_FACTOR.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
