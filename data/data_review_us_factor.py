"""Data review of the US factor and style ETF files (MTUM, QUAL, VLUE, USMV, SPLV, NOBL, RSP, IWF) →
data/DATA_REVIEW_US_FACTOR.md. Reads the raw files directly (a review may; studies may not until the files are on
APPROVED_DATA.md); SPY's calendar and total return come from the approved file through the loader.

    python data/data_review_us_factor.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from analysis.approved_data import US_START, _clean, load_us_etf, total_return, us_start  # noqa: E402  (the loader's own rules)

PRICES, DIVS, SPLITF, CHECK = "us_factor_ohlcv.csv", "us_factor_dividends.csv", "us_factor_splits.csv", "us_factor_eodhd_check.csv"
ORDER = ["MTUM", "QUAL", "VLUE", "USMV", "SPLV", "NOBL", "RSP", "IWF"]
NAME = {"MTUM": "iShares MSCI USA Momentum Factor", "QUAL": "iShares MSCI USA Quality Factor", "VLUE": "iShares MSCI USA Value Factor (MSCI USA Enhanced Value)",
        "USMV": "iShares MSCI USA Min Vol Factor", "SPLV": "Invesco S&P 500 Low Volatility", "NOBL": "ProShares S&P 500 Dividend Aristocrats",
        "RSP": "Invesco S&P 500 Equal Weight", "IWF": "iShares Russell 1000 Growth"}
JUMP, FLAT, MIN_YEARS = 0.08, 5, 5.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lst(xs, k=12):
    xs = list(xs)
    return ", ".join(str(pd.Timestamp(x).date()) for x in xs[:k]) + (" …" if len(xs) > k else "")


def h6(dv: pd.DataFrame, splits: dict, t: str) -> pd.DataFrame:
    """Rule H6 as in analysis/approved_data.split_adjust_dividends: a distribution whose ex-date is before a split is divided by the split ratio."""
    dv = dv.copy()
    for d, k in splits.get(t, ()):
        dv.loc[dv["ex_date"] < pd.Timestamp(d), "gross"] = dv.loc[dv["ex_date"] < pd.Timestamp(d), "gross"] / k
    return dv


def exdate_gaps(g: pd.DataFrame, d_: pd.Series, spy_c: pd.Series):
    """(amount ÷ previous close, market-adjusted fall) for each ex-date: the fall minus SPY's fall the same day."""
    rows, dates = [], g.date.to_numpy()
    for ex, amt in d_.items():
        k = int(np.searchsorted(dates, np.datetime64(ex)))
        if 0 < k < len(g):
            d0, d1 = g.date.iloc[k - 1], g.date.iloc[k]; pc = g.close.iloc[k - 1]
            if d0 in spy_c.index and d1 in spy_c.index:
                rows.append((amt / pc, (pc - g.close.iloc[k]) / pc - (spy_c[d0] - spy_c[d1]) / spy_c[d0]))
    return np.array(rows)


def yields(g: pd.DataFrame, d_: pd.Series) -> dict:
    yy, dates = {}, g.date.to_numpy()
    for ex, amt in d_.items():
        k = int(np.searchsorted(dates, np.datetime64(ex)))
        if 0 < k < len(g):
            yy[ex.year] = yy.get(ex.year, 0.0) + amt / g.close.iloc[k - 1]
    return yy


def growth(a: pd.Series, e: pd.Series):
    com = a.index.intersection(e.index); x, y = a.reindex(com), e.reindex(com)
    gr = (x.iloc[-1] / x.iloc[0]) / (y.iloc[-1] / y.iloc[0]); yrs = (com.max() - com.min()).days / 365.25
    return gr, gr ** (1 / yrs) - 1, com


def main():
    raw = pd.read_csv(os.path.join(DATA, PRICES), parse_dates=["date"]); dv = pd.read_csv(os.path.join(DATA, DIVS), parse_dates=["ex_date"])
    sp = pd.read_csv(os.path.join(DATA, SPLITF), parse_dates=["ex_date", "effective_date"]); ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"])
    spl = sp[(sp.event_type == "Share Split") & (sp.event_status == "Approved") & (sp.is_rescinded.astype(str) != "True")]
    SPL = {t: tuple((r.ex_date, r.new_shares / r.old_shares) for r in s.itertuples()) for t, s in spl.groupby("ticker")}
    spy = load_us_etf()["SPY"]; spy_days = set(spy.index); spy_c = spy["close"]; spy_r = np.log(spy_c).diff(); start = pd.Timestamp(US_START)
    G = {t: raw[raw.ticker == t].dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True) for t in ORDER}
    DV = {t: dv[(dv.ticker == t) & (dv.gross > 0)] for t in ORDER}; DV6 = {t: h6(DV[t], SPL, t) for t in ORDER}
    E = {t: ck[ck.ticker == t].set_index("date").sort_index() for t in ORDER}
    md = ["# Data review — US factor and style ETFs (MTUM, QUAL, VLUE, USMV, SPLV, NOBL, RSP, IWF)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_us_factor.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items. SPY (S&P 500) is already approved in `us_etf_ohlcv.csv` and is the reference calendar and return here.\n"]
    for f in (PRICES, DIVS, SPLITF):
        p = os.path.join(DATA, f); md.append(f"- `{f}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes, {sum(1 for _ in open(p, encoding='utf-8')):,} lines")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    md += ["## What the files are\n",
           "- **us_factor_ohlcv.csv** — daily open/high/low/close/volume, US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_factor.py` to 2026-11-18 "
           "(the last day of the approved SPY file). Prices are split-adjusted (on today's share basis) and not adjusted for distributions:"] + [f"    - **{t}** — {NAME[t]}" for t in ORDER] + [
           "- **us_factor_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD, **as paid**) from `us_corp_action_div_master_b_ed`; the history starts in 2010.",
           "- **us_factor_splits.csv** — one row per distinct share split from `us_corp_action_cap_struc_master_b_ed` (ex-date, effective date, old and new shares).",
           "- **Job:** factor and style ETFs against SPY (and each other) in exploratory studies: signal, fill (the close, rule H3) and valuation.",
           "- **Meaning of the price a loader would use:** total return (rule H4) on the price file's split basis (rule H6) from each fund's start under rule H5, then H1–H3, as `load_us_small()`.\n"]
    # ---------------------------------------------------------------- per series, from 2010
    md += ["## Series checks (from 2010-01-01)\n", "| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | blank open/high/low | distributions (n, sum as paid, largest) |",
           "|---|" + "---|" * 16]
    notes, ohl_share = [], {}
    for t in ORDER:
        g0 = raw[(raw.ticker == t) & (raw.date >= start)].sort_values("date"); blank = int(g0["close"].isna().sum()); g = g0.dropna(subset=["close"])
        n = len(g); dup = int(g.duplicated("date").sum()); wk = int((g.date.dt.weekday >= 5).sum()); first, last = g.date.min(), g.date.max(); yrs = (last - first).days / 365.25
        days = set(g.date); spy_in = {d for d in spy_days if first <= d <= last}; miss = sorted(spy_in - days); extra = sorted(days - spy_in)
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
    md += ["\n## Splits: the price file is split-adjusted, the distribution amounts are not\n",
           "Share splits in the corporate-action table (`us_corp_action_cap_struc_master_b_ed`):\n", "| ticker | ex-date | effective date | old → new shares | distributions within 10 days of the ex-date |", "|---|---|---|---|---|"]
    for r in spl.itertuples():
        near = DV[r.ticker][(DV[r.ticker].ex_date - r.ex_date).abs() <= pd.Timedelta(days=10)]
        md.append(f"| {r.ticker} | {r.ex_date.date()} | {r.effective_date.date()} | {r.old_shares:g} → {r.new_shares:g} | {len(near)} |")
    md += ["", "The second source's raw close ÷ this file's close on every common day, in runs of the same value rounded to a whole number (a run at 4 = closes this file has divided by 4 for a later split):\n",
           "| ticker | runs of the ratio (from → to: rounded value, days) | common days off a whole number by > 0.5% |", "|---|---|---|"]
    runs_of = {}
    for t in ORDER:
        pe = G[t].set_index("date")["close"]; e = E[t]; com = pe.index.intersection(e.index)
        if len(com) < 30:
            md.append(f"| {t} | — (not in the second source) | |"); continue
        ratio = e["close"].reindex(com) / pe.reindex(com); rr = ratio.round(0); key = (rr != rr.shift()).cumsum()
        segs = [(s.index.min(), s.index.max(), int(rr.loc[s.index[0]]), len(s)) for _, s in ratio.groupby(key) if len(s) >= 5]; runs_of[t] = segs
        md.append(f"| {t} | " + "; ".join(f"{a.date()} → {b.date()}: {v} ({k:,})" for a, b, v, k in segs) + f" | {int(((ratio - rr).abs() / rr > 0.005).sum())} |")
    md += ["", "Distribution yield by year (sum of the year's amounts ÷ this file's close the day before each ex-date), %: as paid for the four split funds, and after rule H6 for all:\n",
           "| ticker | " + " | ".join(str(y) for y in range(2010, 2027)) + " |", "|---|" + "---|" * 17]
    for t in [x for x in ORDER if x in SPL]:
        yy = yields(G[t], DV[t].groupby("ex_date")["gross"].sum()); md.append(f"| {t} as paid | " + " | ".join(f"{yy[y] * 100:.2f}" if y in yy else "—" for y in range(2010, 2027)) + " |")
    y6 = {}
    for t in ORDER:
        yy = yields(G[t], DV6[t].groupby("ex_date")["gross"].sum()); y6[t] = yy
        md.append(f"| **{t}**{' after H6' if t in SPL else ''} | " + " | ".join(f"{yy[y] * 100:.2f}" if y in yy else "—" for y in range(2010, 2027)) + " |")
    # ---------------------------------------------------------------- distributions on their ex-dates
    md += ["\n## Distributions: does the price fall by about the amount on the ex-date?\n",
           "Over every distribution from 2010: the close-to-close fall on the ex-date minus SPY's fall that day (to take out the market), against the amount, both as % of the previous close. "
           "If the amounts are on the price file's basis, the median of (fall − amount) is near zero; amounts four times too large put it near minus three quarters of the amount.\n",
           "| ticker | ex-dates | median amount, as paid | median (fall − amount), as paid | median amount after H6 | median (fall − amount) after H6 |", "|---|---|---|---|---|---|"]
    gap6 = {}
    for t in ORDER:
        g = G[t][G[t].date >= start].reset_index(drop=True)
        a = exdate_gaps(g, DV[t].groupby("ex_date")["gross"].sum(), spy_c); b = exdate_gaps(g, DV6[t].groupby("ex_date")["gross"].sum(), spy_c)
        gap6[t] = (np.median(a[:, 1] - a[:, 0]), np.median(b[:, 1] - b[:, 0]))
        md.append(f"| {t} | {len(a)} | {np.median(a[:, 0]) * 100:.2f}% | {gap6[t][0] * 100:+.2f} pp | {np.median(b[:, 0]) * 100:.2f}% | {gap6[t][1] * 100:+.2f} pp |")
    # ---------------------------------------------------------------- second source
    md += ["\n## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available\n",
           "Total return from the loader's rules (H4 on Eikon prices and Refinitiv distributions) against EODHD's adjusted close on the dates both have, from 2010: growth of 1, this file ÷ EODHD, with the distributions as paid and after rule H6; "
           "daily-return correlation and days with a return gap over 0.5 pp after H6.\n",
           "| ticker | common days | growth ratio, as paid (a year) | growth ratio after H6 (a year) | daily-return correlation | days with return gap > 0.5 pp |", "|---|---|---|---|---|---|"]
    gr6 = {}
    for t in ORDER:
        e = E[t][E[t].index >= start]
        if not len(e):
            md.append(f"| {t} | — (not in the second source) | | | | |"); continue
        g = G[t][G[t].date >= start][["date", "open", "high", "low", "close"]]
        a4 = _clean(total_return(g, DV[t]))["close"]; a6 = _clean(total_return(g, DV6[t]))["close"]
        g4, y4, com = growth(a4, e["adjusted_close"]); g6, yy6, _ = growth(a6, e["adjusted_close"]); gr6[t] = (y4, yy6)
        r1 = np.log(a6.reindex(com)).diff().dropna(); r2 = np.log(e["adjusted_close"].reindex(com)).diff().dropna()
        md.append(f"| {t} | {len(com):,} ({com.min().date()} → {com.max().date()}) | {g4:.4f} ({y4 * 100:+.2f}%) | {g6:.4f} ({yy6 * 100:+.2f}%) | {np.corrcoef(r1, r2)[0, 1]:.5f} | {int(((r1 - r2).abs() > 0.005).sum())} |")
    # ---------------------------------------------------------------- after the launch period (rule H5), with H6
    md += ["\n## After the launch period (rule H5: the first 1 January at least 12 months after the first day; not before 2010), total return with H6\n",
           "| ticker | first day in the file | start under H5 | years | missing vs SPY's days | jumps > 8% (SPY's return that day) | total return a year (SPY, same days) | weekly correlation with SPY | usable for a 52-week rule (≥ 5 years)? |",
           "|---|" + "---|" * 8]
    post_miss = {}
    for t in ORDER:
        gfull = G[t]; st = us_start(gfull.date.min()); g = gfull[gfull.date >= st]
        last = g.date.max(); yrs = (last - st).days / 365.25; spy_in = {d for d in spy_days if st <= d <= last}; miss = sorted(spy_in - set(g.date)); post_miss[t] = miss
        a = _clean(total_return(g[["date", "open", "high", "low", "close"]], DV6[t])); ra = np.log(a["close"]).diff(); ja = ra[ra.abs() > JUMP]
        w = pd.DataFrame({"x": a["close"], "s": spy_c}).dropna(); yy = (w.index[-1] - w.index[0]).days / 365.25
        cg = lambda s_: (s_.iloc[-1] / s_.iloc[0]) ** (1 / yy) - 1  # noqa: E731
        wk_ = np.log(w.groupby(pd.Grouper(freq="W-FRI")).last().dropna()).diff().dropna()
        md.append(f"| {t} | {gfull.date.min().date()} | {st.date()} | {yrs:.1f} | {len(miss)}{(' (' + lst(miss, 8) + ')') if miss else ''} | "
                  + ("; ".join(f"{d.date()} {v * 100:+.1f}% ({spy_r.get(d, np.nan) * 100:+.1f}%)" for d, v in ja.items()) or "0")
                  + f" | {cg(w['x']) * 100:.1f}% ({cg(w['s']) * 100:.1f}%) | {wk_.corr().iloc[0, 1]:.3f} | {'yes' if yrs >= MIN_YEARS else 'short — report only'} |")
    # ---------------------------------------------------------------- findings, handling, verdicts
    nos = [t for t in ORDER if not len(E[t])]; e_end = ck.date.max().date(); full_ohl = [t for t in ORDER if ohl_share[t] < 0.05]; c_only = [t for t in ORDER if ohl_share[t] > 0.95]
    rsp_run = [s for s in runs_of.get("RSP", []) if s[2] == 4]; mtum_j = np.log(G["MTUM"].set_index("date")["close"]).diff()
    md += ["\n## Findings and how a loader would handle them\n",
           "- **F-RA-D15 — four funds split, and the split basis differs between the price and distribution files (as F-RA-D14 for IJR).** "
           + "; ".join(f"{t} {k:g}-for-1 on {pd.Timestamp(d).date()}" for t in SPL for d, k in SPL[t])
           + ". The price file divides every earlier close by the split ratio; the distribution table holds the amounts as paid. Read together, every pre-split distribution counts two or four times: "
           f"the as-paid yields above are two to four times the funds' real ones, the ex-date falls are far smaller than the amounts (median gap as paid: "
           + ", ".join(f"{t} {gap6[t][0] * 100:+.2f} pp" for t in SPL) + "), and total return is overstated against the second source ("
           + ", ".join(f"{t} {gr6[t][0] * 100:+.2f}% a year" for t in SPL if t in gr6) + "). "
           "**Handled by rule H6** (`analysis/approved_data.split_adjust_dividends`, extended with these four splits): a distribution before a split is divided by its ratio. After H6 the ex-date gaps are "
           + ", ".join(f"{t} {gap6[t][1] * 100:+.2f} pp" for t in SPL) + ", and total return agrees with the second source ("
           + ", ".join(f"{t} {gr6[t][1] * 100:+.2f}% a year" for t in SPL if t in gr6) + "). No distribution falls within 10 days of a split.",
           f"- **The split table is not a full history:** it holds only these 2026 splits, but the second source shows RSP's earlier 4-for-1 ({rsp_run[0][0].date()} → {rsp_run[0][1].date()} at 4)" if rsp_run else "- **The split table is not a full history.**",
           "  (before 2010, so outside the series and the distribution history). The review therefore also checks every fund's yields and ex-date falls: after H6 they show no other split from 2010.",
           f"- **Closes only for six funds** ({', '.join(c_only)}): open, high and low are blank on almost every row (F-RA-D13); **rule H3** (fill = close). {', '.join(full_ohl)} have open, high and low on every row; "
           "a study comparing these funds should still fill all of them at the close, so that every fund trades at the same price point.",
           f"- **No second source for {', '.join(nos)}**; the second source ends on {e_end}. These three rest on the ex-date falls and yields above (which agree after H6) and on the weekly correlation with SPY.",
           f"- **Missing sessions** fall in the launch periods, which rule H5 drops, except VLUE's {len(post_miss['VLUE'])} ({lst(post_miss['VLUE'], 8)}); "
           "a study on the funds' common calendar drops those days for every fund.",
           f"- **Jumps over 8%** are on days SPY moved sharply the same way (March 2020, 2025-04-09), except MTUM on 2026-11-05 ({mtum_j.get(pd.Timestamp('2026-11-05'), np.nan) * 100:+.1f}%, SPY "
           f"{spy_r.get(pd.Timestamp('2026-11-05'), np.nan) * 100:+.1f}%): the day after its split took effect and after the second source ends, so it cannot be checked here; QUAL, split the same day, shows no jump. Open item.",
           "- **Spot checks against the issuers** (iShares, Invesco, ProShares): not done here; open item, as for the other US ETFs.\n",
           "## Proposed verdicts (for the data owner)\n", "| file | proposed verdict | why |", "|---|---|---|",
           "| us_factor_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4, H5 and H6; every fund has at least 11.9 years after H5 |",
           "| us_factor_dividends.csv | PASS WITH NOTES | amounts as paid, so read only through rule H6 (pre-split amounts divided by the split ratio); total return then agrees with the second source where it exists |",
           "| us_factor_splits.csv | PASS WITH NOTES | the four 2026 splits that H6 needs; not a full history (RSP's 2006 split is missing; before the 2010 start) |",
           "| us_factor_eodhd_check.csv | not for approval | the second source, read only by this review |"]
    out = os.path.join(DATA, "DATA_REVIEW_US_FACTOR.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
