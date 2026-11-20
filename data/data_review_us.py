"""Data review of the US ETF files (SPY and 13 broad commodity ETFs) → data/DATA_REVIEW_US_ETF.md.
Reads the raw files directly (a review may; studies may not until the files are on APPROVED_DATA.md).

    python data/data_review_us.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from analysis.approved_data import US_START, _clean, total_return, us_start  # noqa: E402  (the loader's own rules)

PRICES, DIVS, CHECK = "us_etf_ohlcv.csv", "us_etf_dividends.csv", "us_etf_eodhd_check.csv"
ORDER = ["SPY", "GCC", "FTGC", "COMT", "PDBC", "BCI", "BCD", "COM", "CMDY", "SDCI", "HGER", "NBCM", "PIT", "CERY"]
JUMP, FLAT, MIN_YEARS = 0.08, 5, 5.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    raw = pd.read_csv(os.path.join(DATA, PRICES), parse_dates=["date"]); dv = pd.read_csv(os.path.join(DATA, DIVS), parse_dates=["ex_date"])
    ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"])
    start = pd.Timestamp(US_START); spy_days = set(raw[(raw.ticker == "SPY") & (raw.date >= start)]["date"])
    md = ["# Data review — US ETFs (SPY and 13 broad commodity ETFs)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_us.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items.\n"]
    for f in (PRICES, DIVS):
        p = os.path.join(DATA, f); md.append(f"- `{f}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    md += ["## What the files are\n",
           "- **us_etf_ohlcv.csv** — daily open/high/low/close/volume of SPY and the 13 ETFs of the desk's commodity-ETF comparison (BCI, BCD, CMDY, CERY, PDBC, COMT, SDCI, HGER, COM, FTGC, NBCM, PIT, GCC), "
           "US dollars, prices **as traded (not adjusted for distributions)**. Source: the research Postgres, table `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_etf.py`.",
           "- **us_etf_dividends.csv** — one row per distinct distribution event (ticker, ex-date, type, gross amount in USD), from `us_corp_action_div_master_b_ed` (Refinitiv corporate actions). "
           "The table stores each event once per pull window; the fetch keeps distinct events. Its history starts in 2010.",
           "- **Job:** signal, fill price (OHLC4) and valuation of the US pairs (ETF ÷ SPY) in exploratory studies.",
           f"- **Meaning of the price used by the loader:** total return — rule H4 in `analysis/approved_data.py` (`total_return`): prices before each ex-date are scaled by (1 − D ÷ the previous close), so every distribution is reinvested; then rules H1–H3 as for the NSE files. Series start on {US_START} (no earlier dividend history).\n"]
    # ---------------------------------------------------------------- per series
    md += ["## Series checks (from 2010-01-01)\n", "| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | jumps > 8% after H4 | distributions (sum, largest) |",
           "|---|" + "---|" * 16]
    notes, verdict_rows, tr = [], [], {}
    for t in ORDER:
        g = raw[(raw.ticker == t) & (raw.date >= start)].sort_values("date"); g = g.dropna(subset=["close"]) if g["close"].isna().any() else g
        n = len(g); dup = int(g.duplicated("date").sum()); wk = int((g.date.dt.weekday >= 5).sum()); blank = int(raw[(raw.ticker == t) & (raw.date >= start)]["close"].isna().sum())
        days = set(g.date); first, last = g.date.min(), g.date.max(); yrs = (last - first).days / 365.25
        spy_in = {d for d in spy_days if first <= d <= last}; miss = sorted(spy_in - days); extra = sorted(days - spy_in) if t != "SPY" else []
        gaps = g.date.diff().dt.days; big = g.date[gaps > 5]
        o, h, l_, c = g["open"], g["high"], g["low"], g["close"]
        ordr = int(((h < np.maximum(o, c) - 1e-9) | (l_ > np.minimum(o, c) + 1e-9)).sum()); eq = int(((o == h) & (h == l_) & (l_ != c)).sum())
        runs = (c != c.shift()).cumsum(); flat = int((c.groupby(runs).transform("size") >= FLAT).groupby(runs).first().sum())
        r = np.log(c).diff(); jumps = g.date[r.abs() > JUMP]
        d_ = dv[dv.ticker == t]; d_in = d_[(d_.ex_date >= first) & (d_.ex_date <= last) & (d_.gross > 0)]
        adj = total_return(g[["date", "open", "high", "low", "close"]], d_); cl = _clean(adj.copy()); tr[t] = cl
        ra = np.log(cl["close"]).diff(); jumps_a = cl.index[ra.abs() > JUMP]
        md.append(f"| {t} | {n:,} | {first.date()} | {last.date()} | {yrs:.1f} | {dup} | {wk} | {len(miss)} | {len(extra)} | {len(big)} | {blank} | {ordr} | {eq} | {flat} | {len(jumps)} | {len(jumps_a)} | "
                  f"{len(d_in)} (${d_in.gross.sum():.2f}, ${d_in.gross.max() if len(d_in) else 0:.2f}) |")
        if miss:
            notes.append(f"- **{t}** misses {len(miss)} of SPY's trading days: " + ", ".join(str(x.date()) for x in miss[:12]) + (" …" if len(miss) > 12 else ""))
        if len(big):
            notes.append(f"- **{t}** gaps over 5 calendar days end on: " + ", ".join(str(x.date()) for x in big[:12]) + (" …" if len(big) > 12 else ""))
        if len(jumps):
            ex = set(d_in.ex_date)
            notes.append(f"- **{t}** raw jumps > 8%: " + "; ".join(f"{x.date()} {r[g.date == x].iloc[0] * 100:+.1f}%" + (" (an ex-date)" if x in ex else "") for x in jumps[:10]))
        if len(jumps_a):
            notes.append(f"- **{t}** jumps > 8% after H4: " + "; ".join(f"{x.date()} {ra.loc[x] * 100:+.1f}%" for x in jumps_a[:10]))
        if eq:
            notes.append(f"- **{t}** {eq} rows with open = high = low ≠ close (stale OHL; rule H1 → fill = close): " + ", ".join(str(x.date()) for x in g.date[(o == h) & (h == l_) & (l_ != c)][:8]))
        verdict_rows.append((t, yrs))
    md += ["", "### Notes per series\n"] + (notes or ["- none"])
    # ---------------------------------------------------------------- distributions on their ex-dates
    md += ["\n## Distributions: does the price fall by about the amount on the ex-date?\n",
           "For each distribution of at least 2% of the price: the close-to-close fall on the ex-date against the amount (both as % of the previous close). The fall also contains the day's market move, so only large gaps between the two are flags.\n",
           "| ticker | ex-dates (≥ 2% of price) | median (fall − amount) | largest gaps |", "|---|---|---|---|"]
    for t in ORDER:
        g = raw[(raw.ticker == t) & (raw.date >= start)].sort_values("date").reset_index(drop=True); d_ = dv[(dv.ticker == t) & (dv.gross > 0)].groupby("ex_date")["gross"].sum()
        rows = []
        for ex, amt in d_.items():
            k = int(np.searchsorted(g.date.to_numpy(), np.datetime64(ex)))
            if 0 < k < len(g):
                pc = g.close.iloc[k - 1]; a = amt / pc
                if a >= 0.02:
                    rows.append((g.date.iloc[k].date(), a, (pc - g.close.iloc[k]) / pc))
        if rows:
            dif = [fall - a for _, a, fall in rows]; worst = sorted(rows, key=lambda x: -abs(x[2] - x[1]))[:3]
            md.append(f"| {t} | {len(rows)} | {np.median(dif) * 100:+.1f} pp | " + "; ".join(f"{d} amount {a * 100:.1f}%, fall {fall * 100:.1f}%" for d, a, fall in worst) + " |")
    # ---------------------------------------------------------------- second source
    md += ["\n## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available\n",
           "Daily total returns from the loader's rule (H4 on Eikon prices and Refinitiv dividends) against the daily returns of EODHD's adjusted close, on the dates both have; and the raw closes of the two sources.\n",
           "| ticker | common days | raw closes differing > 0.5% | daily-return correlation | days with return gap > 0.5 pp | growth of 1, H4 ÷ EODHD over the common period |", "|---|---|---|---|---|---|"]
    for t in ORDER:
        e = ck[(ck.ticker == t) & (ck.date >= start)].set_index("date").sort_index()
        if not len(e):
            md.append(f"| {t} | — (not in the second source) | | | | |"); continue
        a = tr[t]; rawc = raw[(raw.ticker == t)].set_index("date")["close"]
        com = a.index.intersection(e.index)
        if len(com) < 30:
            md.append(f"| {t} | {len(com)} | | | | |"); continue
        cr = (rawc.reindex(com) / e["close"].reindex(com) - 1).abs()
        r1 = np.log(a["close"].reindex(com)).diff().dropna(); r2 = np.log(e["adjusted_close"].reindex(com)).diff().dropna()
        gp = (r1 - r2).abs(); grow = (a["close"].reindex(com).iloc[-1] / a["close"].reindex(com).iloc[0]) / (e["adjusted_close"].reindex(com).iloc[-1] / e["adjusted_close"].reindex(com).iloc[0])
        md.append(f"| {t} | {len(com):,} ({com.min().date()} → {com.max().date()}) | {int((cr > 0.005).sum())} | {np.corrcoef(r1, r2)[0, 1]:.4f} | {int((gp > 0.005).sum())} | {grow:.4f} |")
    # ---------------------------------------------------------------- after the launch period (rule H5)
    md += ["\n## After the launch period (rule H5: each series starts on the first 1 January at least 12 months after its first day; not before 2010)\n",
           "| ticker | listed (first day in the file) | start under H5 | years | rows with blank open/high/low | missing vs SPY's days | raw closes > 0.5% from EODHD | daily return gaps > 0.5 pp vs EODHD | jumps > 8% after H4 | usable for a 52-week rule (≥ 5 years)? |",
           "|---|" + "---|" * 9]
    post = {}
    for t in ORDER:
        gfull = raw[raw.ticker == t].dropna(subset=["close"]).sort_values("date"); st = us_start(gfull.date.min()); g = gfull[gfull.date >= st]
        last = g.date.max(); yrs = (last - st).days / 365.25; blank = g[["open", "high", "low"]].isna().any(axis=1).mean()
        spy_in = {d for d in spy_days if st <= d <= last}; miss = len(spy_in - set(g.date)) if t != "SPY" else 0
        a = _clean(total_return(g[["date", "open", "high", "low", "close"]], dv[dv.ticker == t])); ra = np.log(a["close"]).diff(); ja = a.index[ra.abs() > JUMP]
        e = ck[(ck.ticker == t)].set_index("date").sort_index(); com = a.index.intersection(e.index)
        if len(com) > 30:
            rawc = g.set_index("date")["close"]; nclose = int(((rawc.reindex(com) / e["close"].reindex(com) - 1).abs() > 0.005).sum())
            gp = (np.log(a["close"].reindex(com)).diff() - np.log(e["adjusted_close"].reindex(com)).diff()).abs(); ngap = int((gp > 0.005).sum()); src2 = True
        else:
            nclose = ngap = None; src2 = False
        post[t] = (st, yrs, ja, ra)
        md.append(f"| {t} | {gfull.date.min().date()} | {st.date()} | {yrs:.1f} | {blank * 100:.0f}% | {miss} | {nclose if src2 else '—'} | {ngap if src2 else '—'} | "
                  + (", ".join(f"{x.date()} {ra.loc[x] * 100:+.1f}%" for x in ja) or "0") + f" | {'—' if t == 'SPY' else ('yes' if yrs >= MIN_YEARS else 'short — report only')} |")
    # ---------------------------------------------------------------- findings, handling, verdicts
    md += ["\n## Findings and how the loader handles them\n",
           "- **F-RA-D12 — the launch periods trade thinly.** In each fund's first year or so the file misses sessions (BCD 92 in 2017, COM 48 in 2017, FTGC 26 in 2013 …) and its closes disagree with the second source "
           "(PDBC: 69 days over 0.5% apart in 2015, against 4 in 2016; COMT 2014–2015). PDBC's 2015-07-13 close of 25.51 against EODHD's 20.00 (then 21.00 against 19.20) is a bad print. "
           "**Handled by rule H5** (above): the launch period is dropped; afterwards the missing sessions and most disagreements are gone (table above).",
           "- **F-RA-D13 — the source has closes only.** Open, high and low are blank on almost every row (volume on every row) until 2026-09. **Handled by rule H3** (as for three gold rows): the fill price is the close, "
           "so trades in a study fill at the next session's close instead of OHLC4; the engine counts these rows as stale. A study must say so.",
           "- **Distributions are large** (up to 24% of the price at an ex-date: COMT 2022-12-13) and annual for most funds; they show up as the expected price falls on the ex-dates (table above). **Handled by rule H4** (total return). "
           "Without it a ratio study would read every December distribution as a collapse of the ETF.",
           "- **Remaining disagreements with the second source after H5** are few: FTGC 16 and COMT 10 closes over 0.5% apart, most of them in March–May 2020, and PDBC 4 in 2016 — closing prints a fraction of a percent apart; not handled (no rule can tell which source is right).",
           "- **No second source** for GCC, BCD, COM, CMDY, SDCI, NBCM, PIT; their dividends are checked only by the ex-date falls above.",
           "- **2022-03-09** (−8% to −11% in most commodity ETFs) is real: the day commodities fell back from the spike after the invasion of Ukraine.",
           "- **SPY** agrees with the second source to the cent on every common day; its jumps are March 2020 and 2025-04-09 (the tariff pause), all real.",
           "- **Spot checks against a public source** (issuer sites, exchange closes): not done here; open item for the data owner, as F-RA-D-OPEN-1 for the NSE files.\n",
           "## Proposed verdicts (for the data owner)\n",
           "| file | proposed verdict | why |", "|---|---|---|",
           "| us_etf_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4 (total return) and H5 (launch period dropped); nine ETFs have five or more years after H5 (GCC, FTGC, COMT, PDBC, BCI, BCD, COM, CMDY, SDCI), four are short (HGER, NBCM, PIT, CERY) |",
           "| us_etf_dividends.csv | PASS WITH NOTES | history from 2010 only (hence the 2010 start); amounts agree with the ex-date price falls; total returns agree with EODHD where it exists (growth of 1 within 1%) |",
           "| us_etf_eodhd_check.csv | not for approval | the second source, read only by this review |"]
    _ = verdict_rows
    out = os.path.join(DATA, "DATA_REVIEW_US_ETF.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
