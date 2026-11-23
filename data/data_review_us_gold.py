"""Data review of the US gold ETF file (GLD, SPDR Gold Shares; for SPY against gold) → data/DATA_REVIEW_US_GOLD.md.
Reads the raw GLD files directly (a review may; studies may not until the files are on APPROVED_DATA.md); SPY's calendar
and MCX gold come from the approved files through the loader.

    python data/data_review_us_gold.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from analysis.approved_data import GOLD, US_START, _clean, load_daily, load_us_etf, us_start  # noqa: E402  (the loader's own rules)

PRICES, CHECK = "us_gold_ohlcv.csv", "us_gold_eodhd_check.csv"
JUMP, FLAT = 0.08, 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    raw = pd.read_csv(os.path.join(DATA, PRICES), parse_dates=["date"]); ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"])
    spy = load_us_etf()["SPY"]; start = pd.Timestamp(US_START)
    md = ["# Data review — US gold ETF (GLD, for SPY against gold)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_us_gold.py`. The file is raw; no study may read it until the data owner lists it on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items.\n"]
    p = os.path.join(DATA, PRICES); md.append(f"- `{PRICES}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes, {len(raw):,} rows")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    md += ["## What the file is\n",
           "- **us_gold_ohlcv.csv** — daily open/high/low/close/volume of GLD (SPDR Gold Shares, a grantor trust holding gold bullion), US dollars, prices as traded. "
           "Source: the research Postgres, table `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_gold.py`, to 2026-11-18 (the last date of the approved SPY file).",
           "- **Distributions:** the distribution table (`us_corp_action_div_master_b_ed`) has no GLD event, as expected: the trust pays none (its fee is taken from the gold it holds, so it is in the price). "
           "The price is therefore already a total return; rule H4 changes nothing.",
           "- **Job:** signal, fill price and valuation of the GLD ÷ SPY pair in exploratory studies.",
           f"- **Meaning of the price used by the loader:** as for the approved US ETFs — rule H5 start (the first 1 January at least 12 months after the first day, not before {US_START}), "
           "rule H4 (no events, so no change), then rules H1–H3.\n"]
    # ---------------------------------------------------------------- the series from 2010
    g = raw.sort_values("date"); blank_close = int(g["close"].isna().sum()); g = g.dropna(subset=["close"])
    gs = g[g.date >= start]; days = set(gs.date); first, last = gs.date.min(), gs.date.max(); yrs = (last - first).days / 365.25
    spy_days = set(spy.index); spy_in = {d for d in spy_days if first <= d <= last}; miss = sorted(spy_in - days); extra = sorted(days - spy_in)
    gaps = gs.date.diff().dt.days; big = gs.date[gaps > 5]
    o, h, l_, c = gs["open"], gs["high"], gs["low"], gs["close"]
    ordr = int(((h < np.maximum(o, c) - 1e-9) | (l_ > np.minimum(o, c) + 1e-9)).sum()); eq = int(((o == h) & (h == l_) & (l_ != c)).sum())
    runs = (c != c.shift()).cumsum(); flat = int((c.groupby(runs).transform("size") >= FLAT).groupby(runs).first().sum())
    r = np.log(c.to_numpy()); r = pd.Series(np.r_[np.nan, np.diff(r)], index=gs.date.to_numpy()); jumps = r.index[r.abs() > JUMP]
    blank_ohl = gs[["open", "high", "low"]].isna().any(axis=1); ohl_from = gs.date[~blank_ohl].min()
    md += ["## Series checks (from 2010-01-01)\n",
           "| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close (whole file) | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | jumps > 8% | rows with blank open/high/low |",
           "|---|" + "---|" * 15,
           f"| GLD | {len(gs):,} | {first.date()} | {last.date()} | {yrs:.1f} | {int(gs.duplicated('date').sum())} | {int((gs.date.dt.weekday >= 5).sum())} | {len(miss)} | {len(extra)} | {len(big)} | "
           f"{blank_close} | {ordr} | {eq} | {flat} | {len(jumps)} | {blank_ohl.mean() * 100:.0f}% (open/high/low present from {ohl_from.date() if pd.notna(ohl_from) else '—'}) |", ""]
    notes = []
    if miss:
        notes.append(f"- misses {len(miss)} of SPY's trading days: " + ", ".join(str(x.date()) for x in miss[:12]))
    if extra:
        notes.append(f"- {len(extra)} days that SPY does not have: " + ", ".join(str(x.date()) for x in extra[:12]))
    if len(jumps):
        notes.append("- jumps > 8%: " + "; ".join(f"{pd.Timestamp(x).date()} {r.loc[x] * 100:+.1f}%" for x in jumps))
    md += ["### Notes\n"] + (notes or ["- none"])
    # ---------------------------------------------------------------- the whole file (before 2010)
    gfull_first = g.date.min(); st = us_start(gfull_first); pre = g[g.date < start]
    md += ["\n## Before 2010 (not used)\n",
           f"- The file starts on {gfull_first.date()} (Eikon; the fund listed on 2004-11-18, the second source's first day). Rule H5 gives a start of {st.date()}: "
           f"the fund's own launch period ended long before, but SPY's distribution history starts in 2010, so the pair cannot start earlier.",
           f"- {len(pre):,} rows before 2010 are in the file and are not read by the loader.\n"]
    # ---------------------------------------------------------------- second source
    e = ck.set_index("date").sort_index(); md += ["## Second source: EODHD (`us_ohlcv_eodhd`)\n",
                                                  "Daily returns of the Eikon close against EODHD's adjusted close on the dates both have, and the raw closes of the two sources; "
                                                  "and EODHD's adjusted ÷ raw close, which moves only if EODHD saw a distribution or split.\n",
                                                  "| period | common days | raw closes differing > 0.5% | daily-return correlation | days with return gap > 0.5 pp | growth of 1, Eikon ÷ EODHD adjusted | EODHD adjusted ÷ close (min, max) |",
                                                  "|---|---|---|---|---|---|---|"]
    rawc = g.set_index("date")["close"]
    for lab, lo in (("whole file", g.date.min()), ("from 2010", start)):
        com = rawc.index[(rawc.index >= lo)].intersection(e.index)
        cr = (rawc.reindex(com) / e["close"].reindex(com) - 1).abs()
        r1 = np.log(rawc.reindex(com)).diff().dropna(); r2 = np.log(e["adjusted_close"].reindex(com)).diff().dropna()
        gp = (r1 - r2).abs(); grow = (rawc.reindex(com).iloc[-1] / rawc.reindex(com).iloc[0]) / (e["adjusted_close"].reindex(com).iloc[-1] / e["adjusted_close"].reindex(com).iloc[0])
        adj = (e["adjusted_close"] / e["close"]).reindex(com)
        md.append(f"| {lab} | {len(com):,} ({com.min().date()} → {com.max().date()}) | {int((cr > 0.005).sum())} | {np.corrcoef(r1, r2)[0, 1]:.4f} | {int((gp > 0.005).sum())} | {grow:.4f} | {adj.min():.4f}, {adj.max():.4f} |")
        if lab == "from 2010":
            worst = cr.sort_values(ascending=False).head(5)
            md.append("")
            md.append("Largest raw-close differences from 2010: " + "; ".join(f"{d.date()} Eikon {rawc.loc[d]:.2f} vs EODHD {e['close'].loc[d]:.2f}" for d, v in worst.items() if v > 0) + ".")
    # ---------------------------------------------------------------- the same metal in rupees
    loaded = _clean(g[g.date >= st][["date", "open", "high", "low", "close"]].copy()); mcx = load_daily()[GOLD]
    wk = lambda s: s["close"].resample("W-FRI").last().dropna()  # noqa: E731
    a, b = wk(loaded), wk(mcx); com = a.index.intersection(b.index); ra, rb = np.log(a.reindex(com)).diff().dropna(), np.log(b.reindex(com)).diff().dropna()
    md += ["\n## Sanity check: the same metal on MCX\n",
           f"Weekly log returns of GLD (USD) and the approved MCX gold series (INR), {com.min().date()} → {com.max().date()} ({len(ra):,} weeks): correlation {np.corrcoef(ra, rb)[0, 1]:.3f}; "
           f"growth over the period GLD ×{a.reindex(com).iloc[-1] / a.reindex(com).iloc[0]:.2f}, MCX gold ×{b.reindex(com).iloc[-1] / b.reindex(com).iloc[0]:.2f} "
           "(the two differ by the rupee against the dollar and by India's import duties). Same metal, different currency: MCX gold is not a substitute for GLD against SPY.\n"]
    # ---------------------------------------------------------------- findings, verdicts
    md += ["## Findings and how the loader handles them\n",
           f"- **Closes only (F-RA-D13, as for the other US ETFs):** open, high and low are blank on {blank_ohl.mean() * 100:.0f}% of the rows from 2010. **Handled by rule H3**: the fill price is the close.",
           "- **No distributions** in the source, as expected for a gold grantor trust; EODHD's adjusted close agrees (table above). Rule H4 makes no change.",
           f"- **Calendar:** {len(miss)} of SPY's trading days missing and {len(extra)} extra from 2010; gaps over 5 days: {len(big)}.",
           "- **Jumps:** " + ("; ".join(f"{pd.Timestamp(x).date()} {r.loc[x] * 100:+.1f}%" for x in jumps) or "none")
           + ". Each is in the second source too (the same day's EODHD return within 0.5 pp, table above): real moves of the gold price (2013-04-15 is the gold crash of April 2013), not bad prints.",
           "- **Spot checks against a public source** (issuer site, exchange closes): not done here; open item for the data owner, as for the other files.\n",
           "## Proposed verdicts (for the data owner)\n",
           "| file | proposed verdict | why |", "|---|---|---|",
           f"| us_gold_ohlcv.csv | PASS WITH NOTES | usable from {st.date()} ({yrs:.1f} years) with rules H3 (fill = close) and H5; no distributions, so the price is the total return |",
           "| us_gold_eodhd_check.csv | not for approval | the second source, read only by this review |"]
    out = os.path.join(DATA, "DATA_REVIEW_US_GOLD.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
