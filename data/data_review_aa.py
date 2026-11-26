"""Data review of the desk's latest Indian series copied from its asset-allocation repository (wm-AA): Nifty 100,
Midcap 100 and Smallcap 100 daily OHLCV to 2026-11-20, MCX gold (MAUc1) from 2003-11-10 to 2026-11-20, and two
liquid-ETF cash levels to 2026-11-25 → data/DATA_REVIEW_AA.md. Reads the raw files directly (a review may; studies may
not until the files are on APPROVED_DATA.md); the approved index and gold files through open_approved; the research
Postgres's rows (aa_db_check.csv) as the second source.

    python data/data_review_aa.py
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from analysis.approved_data import open_approved  # noqa: E402

IDX = {"N100": ("n100", "nifty100", "NIFTY 100"), "MC100": ("mc100", "midcap100", "NIFTY MIDCAP 100"), "SC100": ("sc100", "smallcap100", "NIFTY SMLCAP 100")}
GOLD_F, LBES_F, CASE_F, CHECK = "aa_mauc1_daily.csv", "aa_liquid_lbes.csv", "aa_liquid_case.csv", "aa_db_check.csv"
FILES = [f"aa_index_{k}_{p}.csv" for k, _, _ in IDX.values() for p in ("pre2019", "2019_2026")] + [GOLD_F, LBES_F, CASE_F]
JUMP, FLAT = 0.08, 5
GOLD_START = "2005-01-01"   # the proposed rule H8: MCX gold from the first day the Postgres (second source) covers


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lst(xs, k=10):
    xs = list(xs)
    return ", ".join(str(pd.Timestamp(x).date()) for x in xs[:k]) + (" …" if len(xs) > k else "")


def series_checks(g: pd.DataFrame) -> dict:
    """g: date, open, high, low, close (sorted)."""
    o, h, l_, c = g["open"], g["high"], g["low"], g["close"]; r = np.log(c).diff()
    runs = (c != c.shift()).cumsum(); flat = int((c.groupby(runs).transform("size") >= FLAT).groupby(runs).first().sum())
    alleq = (o == h) & (h == l_) & (l_ == c)
    return {"rows": len(g), "first": g.date.min(), "last": g.date.max(), "dup": int(g.date.duplicated().sum()), "weekend": int((g.date.dt.weekday >= 5).sum()),
            "weekend_dates": list(g.date[g.date.dt.weekday >= 5]), "blank_close": int(c.isna().sum()), "blank_ohl": int(g[["open", "high", "low"]].isna().any(axis=1).sum()),
            "order": int(((h < np.maximum(o, c) - 1e-9) | (l_ > np.minimum(o, c) + 1e-9)).sum()), "stale": int(((o == h) & (h == l_) & (l_ != c)).sum()),
            "all_equal": int(alleq.sum()), "all_equal_last": g.date[alleq].max() if alleq.any() else None, "flat": flat,
            "jumps": [(d, float(v)) for d, v in zip(g.date[r.abs() > JUMP], r[r.abs() > JUMP])], "gaps": list(g.date[g.date.diff().dt.days > 5])}


def compare(a: pd.DataFrame, b: pd.DataFrame, cols=("open", "high", "low", "close")) -> dict:
    """a, b indexed by date: the common days, the largest absolute difference per column, the days that differ."""
    com = a.index.intersection(b.index); d = {c: float((a.loc[com, c] - b.loc[com, c]).abs().max()) if len(com) else np.nan for c in cols}
    diff = com[np.any([(a.loc[com, c] - b.loc[com, c]).abs().to_numpy() > 1e-9 for c in cols], axis=0)] if len(com) else com
    lo, hi = max(a.index.min(), b.index.min()), min(a.index.max(), b.index.max())
    return {"common": len(com), "max": d, "days_differ": list(diff), "only_a": list(a.index[(a.index >= lo) & (a.index <= hi)].difference(b.index)),
            "only_b": list(b.index[(b.index >= lo) & (b.index <= hi)].difference(a.index))}


def main():
    md = ["# Data review — the desk's latest Indian series from its asset-allocation repository (wm-AA)\n",
          f"Reviewed {pd.Timestamp.today().date()} by `data/data_review_aa.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. "
          "Problems carry IDs `F-RA-D##` (counter in README.md) or open items. The files were copied by `data/fetch_aa.py` from wm-AA's commit "
          "`e9cf7b25b45b9b66b53388decb5552031da87a14` (2026-11-26), byte for byte (each copy's git blob hash equals the commit's).\n"]
    for f in FILES:
        p = os.path.join(DATA, f); md.append(f"- `{f}`: sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes, {sum(1 for _ in open(p, encoding='utf-8')):,} lines")
    p = os.path.join(DATA, CHECK); md.append(f"- `{CHECK}` (second source from the research Postgres, read by this review only; not for approval): sha256 `{sha256(p)}`, {os.path.getsize(p):,} bytes\n")
    ck = pd.read_csv(os.path.join(DATA, CHECK), parse_dates=["date"])
    dbs = {t: g.set_index("date").sort_index() for t, g in ck.groupby("ticker")}
    # ---------------------------------------------------------------- the indices
    ours = pd.read_csv(open_approved("index_ohlcv_all.csv"), parse_dates=["date"]); ours_last = ours.date.max()
    md += ["## 1 · Nifty 100, Midcap 100, Smallcap 100\n",
           "Each index comes in two files: to 2019-01-04 (wm-AA's D25) and 2019-01-01 to 2026-11-20 (its D25d, copied from the desk's production index file). The combined series takes the first file to 2018-12-31 and the second from 2019-01-01.\n",
           "| index | rows | first | last | overlap 2019-01-01 → 01-04 equal? | dup | weekend | blank close | OHLC order errors | stale OHL (O=H=L≠C) | bars with O=H=L=C (last) | flat runs ≥ 5 | jumps > 8% | gaps > 5 days |",
           "|---|" + "---|" * 13]
    IX, notes = {}, []
    for name, (k, key, dbt) in IDX.items():
        a = pd.read_csv(os.path.join(DATA, f"aa_index_{k}_pre2019.csv"), parse_dates=["date"]); b = pd.read_csv(os.path.join(DATA, f"aa_index_{k}_2019_2026.csv"), parse_dates=["date"])
        ov = compare(a.set_index("date"), b.set_index("date"), ("open", "high", "low", "close", "volume")); same = ov["common"] > 0 and all(v == 0 for v in ov["max"].values()) and not ov["only_a"] and not ov["only_b"]
        g = pd.concat([a[a.date < "2019-01-01"], b]).sort_values("date").reset_index(drop=True); IX[name] = g; s = series_checks(g)
        md.append(f"| {name} | {s['rows']:,} | {s['first'].date()} | {s['last'].date()} | {'yes' if same else 'NO'} ({ov['common']} days) | {s['dup']} | {s['weekend']} | {s['blank_close']} | {s['order']} | {s['stale']} | "
                  f"{s['all_equal']:,} ({s['all_equal_last'].date() if s['all_equal_last'] is not None else '—'}) | {s['flat']} | {len(s['jumps'])} | {len(s['gaps'])} |")
        if s["jumps"]:
            notes.append(f"- **{name}** jumps over 8%: " + "; ".join(f"{d.date()} {v * 100:+.1f}%" for d, v in s["jumps"]))
        if s["weekend"]:
            notes.append(f"- **{name}** weekend sessions: " + lst(s["weekend_dates"]))
        if s["gaps"]:
            notes.append(f"- **{name}** gaps over 5 calendar days end on: " + lst(s["gaps"]))
    md += ["", "### Notes\n"] + (notes or ["- none"])
    md += ["\n### Against the approved index file and the research Postgres\n",
           f"The approved `index_ohlcv_all.csv` ends on {ours_last.date()}. The Postgres's index rows (`merged_historical_data_with_rics`) end on 2025-09-15, so they check the history, not the new rows.\n",
           "| index | common days with the approved file | largest difference O / H / L / C | days that differ | days only in one of them (within the shared span) | new rows after the approved file (first → last) | common days with the Postgres | closes more than 0.5% from the Postgres |",
           "|---|---|---|---|---|---|---|---|"]
    for name, (k, key, dbt) in IDX.items():
        g = IX[name].set_index("date"); o = ours[ours.index_key == key].set_index("date").sort_index()
        cp = compare(o, g); new = g.index[g.index > ours_last]
        db = dbs.get(dbt); cdb = g.index.intersection(db.index) if db is not None else []
        far = int(((g.loc[cdb, "close"] / db.loc[cdb, "close"] - 1).abs() > 0.005).sum()) if len(cdb) else 0
        mx = " / ".join(f"{cp['max'][c]:.2f}" for c in ("open", "high", "low", "close"))
        md.append(f"| {name} | {cp['common']:,} | {mx} | {len(cp['days_differ'])} | {len(cp['only_a'])} only approved, {len(cp['only_b'])} only new | "
                  f"{len(new)} ({new.min().date() if len(new) else '—'} → {new.max().date() if len(new) else '—'}) | {len(cdb):,} | {far} |")
    # ---------------------------------------------------------------- gold
    gd = pd.read_csv(os.path.join(DATA, GOLD_F), parse_dates=["date"]).sort_values("date").reset_index(drop=True); sg = series_checks(gd)
    og = pd.read_csv(open_approved("MAUc1_daily.csv"), parse_dates=["Date"]).rename(columns={"Date": "date", "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close"}).set_index("date").sort_index()
    G = gd.set_index("date"); cpo = compare(og, G); dbg = dbs["MAUc1"]; cpd = compare(dbg, G)
    pre = G.index[G.index < dbg.index.min()]; vol0 = int((gd["volume"].fillna(0) <= 0).sum())
    yv = gd.assign(y=gd.date.dt.year).groupby("y")["volume"].median()
    mxo = " / ".join(f"{cpo['max'][c]:.1f}" for c in ("open", "high", "low", "close")); mxd = " / ".join(f"{cpd['max'][c]:.1f}" for c in ("open", "high", "low", "close"))
    md += ["\n## 2 · MCX gold (MAUc1, front-month continuation, rupees per 10 g)\n",
           "wm-AA's cleaned daily series (its D24c): a desk export to 2026-11-11 and the Postgres's Eikon futures rows from 2026-11-12, with rows that have no close dropped and bars widened to span open, high, low and close.\n",
           "| rows | first | last | dup | weekend | blank close | OHLC order errors | stale OHL | bars with O=H=L=C | flat runs ≥ 5 | jumps > 8% | gaps > 5 days | rows with zero volume |", "|---|" + "---|" * 12,
           f"| {sg['rows']:,} | {sg['first'].date()} | {sg['last'].date()} | {sg['dup']} | {sg['weekend']} | {sg['blank_close']} | {sg['order']} | {sg['stale']} | {sg['all_equal']} | {sg['flat']} | {len(sg['jumps'])} | {len(sg['gaps'])} | {vol0} |",
           "", "Jumps over 8%: " + ("; ".join(f"{d.date()} {v * 100:+.1f}%" for d, v in sg["jumps"]) or "none") + ". Weekend sessions: " + (lst(sg["weekend_dates"]) or "none") + ".",
           "", "Median daily volume (contracts) by year: " + ", ".join(f"{y} {v:,.0f}" for y, v in yv.items()) + ".\n",
           "| compared with | span | common days | largest difference O / H / L / C | days that differ | days only in the other (within the shared span) | days only in this file |", "|---|---|---|---|---|---|---|",
           f"| approved `MAUc1_daily.csv` (Eikon, the 2026-11-02 pull) | {og.index.min().date()} → {og.index.max().date()} | {cpo['common']:,} | {mxo} | "
           f"{len(cpo['days_differ'])} ({lst(cpo['days_differ'], 6)}) | {len(cpo['only_a'])} ({lst(cpo['only_a'], 6)}) | {len(cpo['only_b'])} ({lst(cpo['only_b'], 6)}) |",
           f"| Postgres `ohlcv_eikon_futures` MAUc1 | {dbg.index.min().date()} → {dbg.index.max().date()} | {cpd['common']:,} | {mxd} | "
           f"{len(cpd['days_differ'])} ({lst(cpd['days_differ'], 6)}) | {len(cpd['only_a'])} ({lst(cpd['only_a'], 6)}) | {len(cpd['only_b'])} ({lst(cpd['only_b'], 6)}) |",
           "", f"Rows before the Postgres's first day ({dbg.index.min().date()}), which no second source covers: {len(pre)} ({pre.min().date()} → {pre.max().date()})." if len(pre) else ""]
    # ---------------------------------------------------------------- the liquid-ETF cash levels
    lb = pd.read_csv(os.path.join(DATA, LBES_F)); lb["date"] = pd.to_datetime(lb["date"], format="%d/%m/%y %H:%M"); lb = lb.set_index("date")["price"].sort_index()
    lc = pd.read_csv(os.path.join(DATA, CASE_F)); lc["date"] = pd.to_datetime(lc["date"], format="%d-%m-%Y"); lc = lc.set_index("date")["liquid_case"].sort_index()
    dlc, dlb = dbs["LIQUIDCASE"], dbs["LIQUIDBEES"]; case_first = dlc.index.min()
    md += ["\n## 3 · The liquid-ETF cash levels\n",
           "- **aa_liquid_lbes.csv** (wm-AA's `LBES_Adjusted_Price_data`): date (dd/mm/yy), price — a cash level that starts at 1,000. wm-AA uses it as the cash leg.",
           "- **aa_liquid_case.csv** (wm-AA's `liquidEtfFinal`): date (dd-mm-yyyy), liquid_case — the desk's LIQUIDCASE level (Zerodha Nifty 1D Rate Liquid ETF), for robustness.",
           f"- **Second source:** the Postgres's daily closes of LIQUIDCASE ({dlc.index.min().date()} → {dlc.index.max().date()}, its listing onward) and LIQUIDBEES ({dlb.index.min().date()} → {dlb.index.max().date()}).\n",
           "| series | rows | first | last | dup | non-positive | days with zero return | days with a negative return | largest daily return | smallest | growth a year over the whole file |", "|---|" + "---|" * 10]
    for nm, s in (("LBES", lb), ("LIQUIDCASE level", lc)):
        r = s.pct_change().dropna(); yrs = (s.index[-1] - s.index[0]).days / 365.25
        md.append(f"| {nm} | {len(s):,} | {s.index.min().date()} | {s.index.max().date()} | {int(s.index.duplicated().sum())} | {int((s <= 0).sum())} | {int((r == 0).sum())} ({(r == 0).mean() * 100:.0f}%) | "
                  f"{int((r < 0).sum())} | {r.max() * 100:+.3f}% ({r.idxmax().date()}) | {r.min() * 100:+.3f}% ({r.idxmin().date()}) | {((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100:.2f}% |")
    yy = pd.DataFrame({"LBES": lb.groupby(lb.index.year).apply(lambda x: x.iloc[-1]), "CASE": lc.groupby(lc.index.year).apply(lambda x: x.iloc[-1])}).pct_change() * 100
    md += ["", "Growth by calendar year (year-end to year-end), %:\n", "| year | " + " | ".join(str(y) for y in yy.index[1:]) + " |", "|---|" + "---|" * (len(yy) - 1),
           "| LBES | " + " | ".join(f"{v:.2f}" for v in yy["LBES"].iloc[1:]) + " |", "| LIQUIDCASE level | " + " | ".join(f"{v:.2f}" for v in yy["CASE"].iloc[1:]) + " |"]
    com = lb.index.intersection(lc.index); ratio = (lb.loc[com] / lc.loc[com]); before = ratio[ratio.index < case_first]; after = ratio[ratio.index >= case_first]
    rb_dev = float((before / before.iloc[0] - 1).abs().max()) if len(before) else np.nan; ya = (after.index[-1] - after.index[0]).days / 365.25 if len(after) > 1 else np.nan
    drift = float((after.iloc[-1] / after.iloc[0]) ** (1 / ya) - 1) if len(after) > 1 else np.nan
    cc = lc.index.intersection(dlc.index); rr = lc.loc[cc] / dlc.loc[cc, "close"]
    r_lc, r_db = lc.pct_change(), dlc["close"].pct_change(); cr = r_lc.index.intersection(r_db.dropna().index)
    ext_lb = lb[lb.index > pd.Timestamp("2026-08-09")]; ext_lc = lc[lc.index > pd.Timestamp("2026-09-07")]
    def ext_err(s, ext):
        rs = s.pct_change().reindex(ext.index); rd = r_db.reindex(ext.index); ok = rd.notna()
        return int(ok.sum()), float((rs[ok] - rd[ok]).abs().max()) if ok.any() else np.nan, int((~ok).sum())
    e1, e2 = ext_err(lb, ext_lb), ext_err(lc, ext_lc)
    md += ["", "### The two levels against each other and against the Postgres\n",
           f"- **The same series before LIQUIDCASE existed:** on the {len(before):,} common days before LIQUIDCASE's first day ({case_first.date()}), LBES ÷ LIQUIDCASE level stays within {rb_dev * 100:.4f}% of its first value: one series at two scales. "
           f"From {case_first.date()} the ratio drifts {drift * 100:+.2f}% a year (LBES grows that much more slowly).",
           f"- **The LIQUIDCASE level against the traded fund:** on {len(cc):,} common days from {cc.min().date()}, the level ÷ LIQUIDCASE's close is between {rr.min():.6f} and {rr.max():.6f} (the level is the fund's close{' exactly' if rr.max() - rr.min() < 1e-9 and abs(rr.iloc[0] - 1) < 1e-9 else ''}); "
           f"the daily returns of the two correlate {np.corrcoef(r_lc.loc[cr].to_numpy(float), r_db.loc[cr].to_numpy(float))[0, 1]:.4f} over {len(cr):,} days.",
           f"- **The extended rows:** wm-AA extended LBES after 2026-08-09 and the LIQUIDCASE level after 2026-09-07 by compounding LIQUIDCASE's daily returns. Recomputed from the Postgres's closes: LBES {e1[0]} rows, largest return difference {e1[1]:.2e} "
           f"({e1[2]} rows without a Postgres return); LIQUIDCASE level {e2[0]} rows, {e2[1]:.2e} ({e2[2]} without).",
           f"- **LIQUIDBEES as traded:** its close in the Postgres stays between {dlb['close'].min():,.2f} and {dlb['close'].max():,.2f} (median {dlb['close'].median():,.2f}) from {dlb.index.min().date()} to {dlb.index.max().date()}: it pays its income as daily dividends in new units, "
           "so its price is not a return series. A cash leg needs a level such as these files, built from the fund's NAV with the dividends reinvested."]
    # ---------------------------------------------------------------- gold: the bad rows, Saturdays, and the proposed rules
    bad0 = G.index[G["close"] <= 0]; nb = G["close"].shift(1); na_ = G["close"].shift(-1)
    spike = G.index[(G["close"] > 0) & (nb > 0) & (na_ > 0) & ((G["close"] / nb - 1).abs() > 0.3) & ((G["close"] / na_ - 1).abs() > 0.3)]
    sat = G.index[G.index.weekday == 5]; satc = pd.Series(sat.year).value_counts().sort_index()
    g8 = G[(G.index >= pd.Timestamp(GOLD_START)) & (G["close"] > 0)]; r8 = np.log(g8["close"]).diff(); j8 = r8[r8.abs() > JUMP]
    od = og.index.max(); od_note = (f"- **The approved file's last day ({od.date()})** differs from this file in low and close (approved: low {og.loc[od, 'low']:,.0f}, close {og.loc[od, 'close']:,.0f}; this file: low {G.loc[od, 'low']:,.0f}, "
                                    f"close {G.loc[od, 'close']:,.0f}; the Postgres: low {dbg.loc[od, 'low']:,.0f}, close {dbg.loc[od, 'close']:,.0f}): the approved file holds a provisional print for its last day. "
                                    "The other difference, 2023-10-16, is a widened high that the loader's rule H2 makes equal.") if od in G.index and od in dbg.index else ""
    md += ["\n### Gold: the rows that are not prices, and Saturdays\n", od_note,
           f"- **Non-positive closes:** {len(bad0)} ({lst(bad0)}) — " + "; ".join(f"{d.date()} ({d.day_name()}) open {G.loc[d, 'open']:,.0f}, close {G.loc[d, 'close']:,.0f}, low {G.loc[d, 'low']:,.0f} (the Postgres: low {dbg.loc[d, 'low']:,.0f}, close {dbg.loc[d, 'close']:,.0f})"
                                                                          if d in dbg.index else f"{d.date()}" for d in bad0) + ". The cleaning widened the bar to the zero close, so the low is zero too.",
           f"- **One-day spikes that revert** (a close more than 30% from both neighbours): {len(spike)} — " + "; ".join(f"{d.date()} ({d.day_name()}) {G.loc[d, 'close']:,.0f} between {nb.loc[d]:,.0f} and {na_.loc[d]:,.0f}, volume {G.loc[d, 'volume']:,.0f}" for d in spike)
           + f" (before the Postgres's first day, so no second source).",
           f"- **Saturday sessions:** {len(sat)}, by year: " + ", ".join(f"{y} {v}" for y, v in satc.items()) + ". MCX traded on Saturdays until 2013; later ones are special sessions. On the NSE calendar (the indices) they drop out; on gold's own weekly bars (W-FRI) a Saturday opens the next week.",
           f"- **After the proposed rules** (gold from {GOLD_START}, non-positive closes dropped): {len(g8):,} rows, {g8.index.min().date()} → {g8.index.max().date()}; jumps over 8%: " + "; ".join(f"{d.date()} {v * 100:+.1f}%" for d, v in j8.items())
           + f". The 2014-02-05 / 02-06 pair ({r8.get(pd.Timestamp('2014-02-05'), np.nan) * 100:+.1f}% on {G.loc['2014-02-05', 'volume']:,.0f} contracts, then {r8.get(pd.Timestamp('2014-02-06'), np.nan) * 100:+.1f}%) "
           "falls on the February contract's expiry and the roll to April, in both sources: the continuation's own behaviour, not an error; 2026-01-30 is the gold crash of that day."]
    sc_db = dbs["NIFTY SMLCAP 100"]; gs = IX["SC100"].set_index("date"); cs = gs.index.intersection(sc_db.index); far_d = cs[((gs.loc[cs, "close"] / sc_db.loc[cs, "close"] - 1).abs() > 0.005).to_numpy()]
    s11 = gs[gs.index < pd.Timestamp("2011-10-03")]; eq11 = int(((s11["open"] == s11["high"]) & (s11["high"] == s11["low"]) & (s11["low"] == s11["close"])).sum())
    y_pre = yy.loc[yy.index < case_first.year, "LBES"].dropna()
    md += ["\n## 4 · Findings and how a loader would handle them\n",
           f"- **The indices are the approved file, extended.** On every common day (from each index's first day to {ours_last.date()}) the three series equal the approved `index_ohlcv_all.csv` to the paisa, with no day missing on either side; "
           f"they add 56 sessions each, 2026-08-31 → 2026-11-20. The approved file's handling applies unchanged (H1 stale OHL, H2 out-of-range bars). SC100's open, high and low equal its close on {eq11:,} of its {len(s11):,} bars before 2011-10-03 "
           "(as in the approved file; wm-AA treats them as not data, so SC100's weekly bars before then are bars of closes); "
           f"its 33 closes more than 0.5% from the Postgres are old days ({lst(far_d, 4)}) where the approved file and the Postgres's feed disagree, not new rows. No independent second source covers the 56 new sessions (the Postgres's index rows end on 2025-09-15): open item, with spot checks against NSE.",
           f"- **F-RA-D18 — gold: two rows that are not prices, and a thin start.** A zero close on {lst(bad0)} (in both sources) and a one-day spike to {G.loc[spike[0], 'close']:,.0f} on {spike[0].date()} (export only, zero volume)"
           if len(bad0) and len(spike) else "- **Gold.**",
           f"  would read as a −100% day and a +58% day. The export's rows before {dbg.index.min().date()} have no second source and very thin trading (median {yv.loc[2003]:,.0f} contracts a day in 2003, {yv.loc[2004]:,.0f} in 2004). "
           f"**Handled by two new loader rules:** H7, a row with a non-positive close is dropped; H8, MCX gold starts on {GOLD_START}, the first day the Postgres covers. After them the series equals the Postgres on every common day except where the cleaning widened a bar (H2 would do the same).",
           f"- **F-RA-D17 — the liquid-ETF cash levels are one constructed series before {case_first.date()}.** LBES and the LIQUIDCASE level are the same series at two scales until LIQUIDCASE's first trading day; wm-AA records no source for it (its F-AA-D09). "
           f"Its yearly growth ({y_pre.min():.1f}% in {y_pre.idxmin()} to {y_pre.max():.1f}% in {y_pre.idxmax()}) is consistent with Indian overnight money rates less fees; that is a plausibility check, not a source. "
           f"From {case_first.date()} the LIQUIDCASE level is LIQUIDCASE's traded close exactly; LBES grows {abs(drift) * 100:.2f}% a year less (a dearer fund's NAV with dividends reinvested would do that; not verified) until 2026-08-09, and then compounds LIQUIDCASE's returns (checked against the Postgres). "
           "**Handled as a cash level:** a single price a day (open = high = low = close, so the fill is the close); every study that uses it must state the provenance gap. LBES is wm-AA's cash leg; the LIQUIDCASE level is the robustness series.",
           "- **Not taken:** Micro250 (wm-AA has no newer file) and the Postgres's LIQUIDBEES price (about 1,000 every day: not a return series).",
           "- **Spot checks against public sources** (NSE index closes, MCX settlement prices, the liquid ETFs' published NAVs): not done here; open item.\n",
           "## 5 · Proposed verdicts (for the data owner)\n", "| file | proposed verdict | why |", "|---|---|---|",
           "| aa_index_{n100,mc100,sc100}_pre2019.csv, aa_index_{n100,mc100,sc100}_2019_2026.csv | PASS | equal to the approved index file on every common day; 56 new sessions each to 2026-11-20; rules H1–H2 as before |",
           f"| aa_mauc1_daily.csv | PASS WITH NOTES | read only through rules H7 (non-positive close dropped) and H8 (from {GOLD_START}); then equal to the Postgres apart from widened bars; front-month continuation (rolls inside the return), Saturday sessions to 2013 |",
           "| aa_liquid_lbes.csv | PASS WITH NOTES | a cash level; the source before 2026-08-10 is not recorded (F-RA-D17); the extended rows equal LIQUIDCASE's returns |",
           "| aa_liquid_case.csv | PASS WITH NOTES | a cash level; equal to LIQUIDCASE's traded close from 2024-01-24; before that the same unrecorded series as LBES (F-RA-D17) |",
           "| aa_db_check.csv | not for approval | the second source, read only by this review |"]
    out = os.path.join(DATA, "DATA_REVIEW_AA.md"); open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
