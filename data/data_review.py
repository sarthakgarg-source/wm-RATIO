"""
Step 0 of the playbook: review every input file once, write data/DATA_REVIEW.md and data/APPROVED_DATA.md.

    python data/data_review.py          # from the project root

For each series: what it is, where it came from, its fingerprint (sha256, rows, first/last date), the calendar
(weekday gaps, weekend rows, duplicates), the values (blanks, non-positive prices, low > close, close > high, flat runs,
large jumps), a chart, and a verdict. Problems get IDs F-RA-D##. The approved list is what the tests may read.
"""
import hashlib
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FILES = {
    "index_ohlcv_all.csv": {"what": "daily OHLC of the NSE indices (price indices, points; volume blank for most)", "job": "signal, fill price (OHLC4), valuation", "source": "the desk's index file (NSE data via the research data store), copied from the wm-BREAK repo on 2026-11-12; fetched there on 2026-11-02", "adjust": "index levels: price return (no dividends); no corporate actions apply"},
    "MAUc1_daily.csv": {"what": "daily OHLCV of the MCX gold front-month futures continuation (MAUc1), rupees per 10 g", "job": "signal, fill price (OHLC4), valuation of the gold leg", "source": "Refinitiv Eikon get_timeseries on MAUc1, copied from the wm-BREAK repo on 2026-11-12; fetched there on 2026-11-02", "adjust": "front-month continuation: roll yield is inside the return; no dividends"},
}
USE_INDICES = {"nifty100": "N100", "midcap100": "MC100", "smallcap100": "SC100", "microcap250": "Micro250"}
JUMP = 0.08          # a one-day |log return| above this is listed and must be explained
FLAT = 5             # the same close this many days running is listed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def series_checks(name, s: pd.DataFrame, problems, pid):
    """s: columns date, open, high, low, close sorted by date. Returns a dict of check results."""
    out = {"rows": len(s), "first": s["date"].min().date(), "last": s["date"].max().date()}
    d = s["date"]
    out["duplicates"] = int(d.duplicated().sum())
    out["weekend_rows"] = int(d.dt.dayofweek.isin([5, 6]).sum())
    gaps = d.diff().dt.days
    out["gaps_over_5_days"] = int((gaps > 5).sum())
    big_gaps = s.loc[gaps > 5, "date"].dt.date.tolist()[:8]
    out["blank_close"] = int(s["close"].isna().sum())
    out["nonpositive"] = int((s[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    out["low_gt_close"] = int((s["low"] > s["close"] + 1e-9).sum()); out["close_gt_high"] = int((s["close"] > s["high"] + 1e-9).sum())
    out["low_gt_open"] = int((s["low"] > s["open"] + 1e-9).sum()); out["open_gt_high"] = int((s["open"] > s["high"] + 1e-9).sum())
    r = np.log(s["close"]).diff()
    jumps = s.loc[r.abs() > JUMP, ["date", "close"]].assign(ret=r[r.abs() > JUMP].values)
    out["jumps"] = [(x.date.date(), float(x.ret)) for x in jumps.itertuples()]
    flat = (s["close"].diff() == 0).astype(int)
    runs = flat.groupby((flat != flat.shift()).cumsum()).transform("sum") * flat
    out["flat_runs"] = s.loc[runs >= FLAT - 1, "date"].dt.date.tolist()[:6]
    out["big_gaps"] = big_gaps
    for key, label in (("duplicates", "duplicate dates"), ("weekend_rows", "weekend rows"), ("blank_close", "blank closes"), ("nonpositive", "non-positive prices"), ("low_gt_close", "low above close"), ("close_gt_high", "close above high"), ("low_gt_open", "low above open"), ("open_gt_high", "open above high")):
        if out[key]:
            problems.append((f"F-RA-D{pid[0]:02d}", name, f"{out[key]} {label}")); pid[0] += 1
    if out["flat_runs"]:
        problems.append((f"F-RA-D{pid[0]:02d}", name, f"close unchanged for {FLAT}+ days starting {out['flat_runs'][:3]}")); pid[0] += 1
    return out


def chart(series: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 2.2 * len(series)), sharex=False)
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        ax.plot(s["date"], s["close"], lw=0.7, color="#1b1a17"); ax.set_yscale("log"); ax.set_title(name, fontsize=9, loc="left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    problems, pid = [], [1]
    lines = ["# Data review (Step 0)", "", f"Reviewed {date.today().isoformat()}. Every file the tests read is listed here with its fingerprint; the approved list is `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##`.", ""]
    approved = ["# Approved data", "", "Tests may read ONLY the files below, through `analysis/approved_data.py`, which checks the fingerprint on every open. Refreshing or extending a file needs the data owner's sign-off and a new review.", "", "| file | sha256 | bytes | rows | first date | last date | verdict |", "|---|---|---|---|---|---|---|"]
    series_for_chart = {}
    for fname, meta in FILES.items():
        path = os.path.join(DATA, fname); fp = sha256(path); nbytes = os.path.getsize(path)
        lines += [f"## {fname}", "", f"- **What:** {meta['what']}", f"- **Job:** {meta['job']}", f"- **Source:** {meta['source']}", f"- **Adjustment / meaning of the price:** {meta['adjust']}", f"- **Fingerprint:** sha256 `{fp}`, {nbytes:,} bytes", ""]
        if fname == "index_ohlcv_all.csv":
            raw = pd.read_csv(path, parse_dates=["date"])
            lines += [f"- Rows: {len(raw):,}; index keys in the file: {', '.join(sorted(raw['index_key'].unique()))}.", f"- Used by the tests: {', '.join(f'{k} → {v}' for k, v in USE_INDICES.items())}. The other keys (nifty50, midcap150, smallcap250, nifty500_synth) are not read — nifty500_synth is a synthetic blend and is never read as a result.", ""]
            lines += ["| series | rows | first | last | dup | weekend | gaps > 5d | blank close | OHLC order errors | flat runs | jumps > 8% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for key, label in USE_INDICES.items():
                s = raw[raw["index_key"] == key][["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)
                c = series_checks(label, s, problems, pid); series_for_chart[label] = s
                ohlc_err = c["low_gt_close"] + c["close_gt_high"] + c["low_gt_open"] + c["open_gt_high"]
                lines.append(f"| {label} | {c['rows']:,} | {c['first']} | {c['last']} | {c['duplicates']} | {c['weekend_rows']} | {c['gaps_over_5_days']} | {c['blank_close']} | {ohlc_err} | {len(c['flat_runs'])} | {len(c['jumps'])} |")
                if c["jumps"]:
                    lines.append(f"|  | jumps: {'; '.join(f'{d} {r:+.1%}' for d, r in c['jumps'])} |||||||||| ")
                if c["big_gaps"]:
                    lines.append(f"|  | gaps over 5 days start after: {', '.join(str(x) for x in c['big_gaps'])} |||||||||| ")
            first_rows = raw.groupby("index_key")["date"].min()
            approved.append(f"| {fname} | `{fp}` | {nbytes:,} | {len(raw):,} | {raw['date'].min().date()} | {raw['date'].max().date()} | PASS WITH NOTES |")
        else:
            raw = pd.read_csv(path, parse_dates=["Date"]).rename(columns={"Date": "date", "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "VOLUME": "volume"}).sort_values("date").reset_index(drop=True)
            c = series_checks("GoldMCX", raw, problems, pid); series_for_chart["GoldMCX"] = raw
            ohlc_err = c["low_gt_close"] + c["close_gt_high"] + c["low_gt_open"] + c["open_gt_high"]
            lines += ["| series | rows | first | last | dup | weekend | gaps > 5d | blank close | OHLC order errors | flat runs | jumps > 8% |", "|---|---|---|---|---|---|---|---|---|---|---|",
                      f"| GoldMCX | {c['rows']:,} | {c['first']} | {c['last']} | {c['duplicates']} | {c['weekend_rows']} | {c['gaps_over_5_days']} | {c['blank_close']} | {ohlc_err} | {len(c['flat_runs'])} | {len(c['jumps'])} |"]
            if c["jumps"]:
                lines.append(f"|  | jumps: {'; '.join(f'{d} {r:+.1%}' for d, r in c['jumps'])} |||||||||| ")
            if c["big_gaps"]:
                lines.append(f"|  | gaps over 5 days start after: {', '.join(str(x) for x in c['big_gaps'])} |||||||||| ")
            approved.append(f"| {fname} | `{fp}` | {nbytes:,} | {len(raw):,} | {raw['date'].min().date()} | {raw['date'].max().date()} | PASS WITH NOTES |")
        lines.append("")
    chart(series_for_chart, os.path.join(DATA, "data_review_charts.png"))
    lines += ["## What the flagged rows are, and how the loader handles them", "",
              "- **Weekend rows (8 per index, 8 in gold):** real sessions — Muhurat trading on Diwali (2006-04-29*, 2009-10-17, 2013-11-03, 2016-10-30, 2019-10-27, 2020-11-14, 2023-11-12), Budget-day Saturdays (2015-02-28, 2020-02-01, 2025-02-01) and Sunday 2026-02-01, and the NSE special live sessions of 2024-01-20 and 2024-05-18. Kept as trading days. (*2006-04-29 was a special Saturday session.)",
              "- **OHLC order errors:** N100 2009-05-18 (close 4087.3 above high 4078.25 — the election-day session that was halted limit-up; the printed high stops at the halt), SC100 2015-08-20 and gold 2023-10-16 (a point or two of rounding). **Handled in the loader (rule H2 in `analysis/approved_data.py`):** high := max(high, open, close), low := min(low, open, close).",
              "- **SC100 2011-04-01 … 04-07:** open = high = low = 3584.17 on five consecutive days while the close moves 3650 → 3874 — the OHL fields are a stale print. **Handled in the loader (rule H1):** on a row whose open, high and low are all equal and differ from the close, the fill price is the close instead of OHLC4 (the engine counts such rows in audit.json).",
              "- **Gold blank closes (2015-10-02, 2015-11-11, 2020-04-14):** dropped; those days are absent from gold's calendar and therefore from every gold pair's calendar. Three further gold rows have a close but a blank open / high / low (2021-10-04, 2025-01-02, 2026-02-04): **handled in the loader (rule H3):** the fill price is the close; counted with the stale rows.",
              "- **Gold 2026-01-30 −12.4%:** the day after an intraday range of 157,808–180,779 (14%); volume falls from ~13,000 lots to 619, 528 and 103 over the next three sessions, which is the front-month contract going into expiry while the continuation still points at it. A real spike plus a roll at thin volume. **Open item F-RA-D-OPEN-2:** the data owner should confirm against the MCX bhavcopy and, if the continuation is mis-rolled, re-fetch with a rolled series. Affects the four gold pairs only, which are reported, not deciding.",
              "", "## Calendar and cross-checks", "",
              "- The index series share the NSE calendar (no weekend rows; the gaps over 5 days are exchange holidays bridging a weekend). Gold trades on the MCX calendar, which differs from the NSE on some holidays; a pair's calendar in the tests is the intersection of its two legs' trading days, so no price is ever carried forward.",
              "- Micro250 starts on 2022-01-10 (NSE back-computed history); every pair with Micro250 starts there, so its tests cover about four and a half years.",
              "- The index file ends 2026-08-28 and gold 2026-09-08; a pair's window ends at the earlier of its two legs.",
              "- Two sources of the same thing: only one source per series is available in this repo; no comparison made.",
              "- Spot checks against a public source (NSE site / factsheets, MCX bhavcopy): **not done in this environment** (no network). Listed as an open item F-RA-D-OPEN-1 below; three dates per series should be checked by the data owner before any number from this repo is acted on.",
              "", "## Verdicts", "", "| file | verdict | why |", "|---|---|---|",
              "| index_ohlcv_all.csv | PASS WITH NOTES | price indices (no dividends); Micro250 short history; spot checks pending |",
              "| MAUc1_daily.csv | PASS WITH NOTES | futures continuation (roll yield inside the return); MCX calendar differs from NSE; spot checks pending |",
              "", "## Problems found", "", "| id | series | what |", "|---|---|---|"]
    lines += [f"| {i} | {n} | {w} |" for i, n, w in problems] or ["| — | — | none beyond the notes above |"]
    lines += ["| F-RA-D-OPEN-1 | all | three dates per series not yet checked against a public source (no network here) |", "| F-RA-D-OPEN-2 | GoldMCX | the 2026-01-30 fall and the thin-volume sessions that follow: confirm the continuation's roll against the MCX bhavcopy |", "", "## Charts", "", "![every series, log scale](data_review_charts.png)", ""]
    open(os.path.join(DATA, "DATA_REVIEW.md"), "w").write("\n".join(lines))
    approved += ["", f"Approved {date.today().isoformat()} by the review above. Both files are read-only in the working tree.", ""]
    open(os.path.join(DATA, "APPROVED_DATA.md"), "w").write("\n".join(approved))
    print("wrote data/DATA_REVIEW.md, data/APPROVED_DATA.md, data/data_review_charts.png;", len(problems), "problems")


if __name__ == "__main__":
    main()
