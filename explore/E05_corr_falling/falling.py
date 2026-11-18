"""
E05 (EXPLORATORY): does the ratio switch do better when the legs' correlation is falling as z reaches its extreme?
Design: explore/E05_corr_falling/E05.md (written before the first run).

    python explore/E05_corr_falling/falling.py [--perms 2000]

Every number on the page and in E05_results.md is computed here; the page's script only picks the numbers of the chosen
controls from the embedded results and draws them. Not a finding and not a test of any rule.
"""
import argparse
import html
import json
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles")); sys.path.insert(0, os.path.join(ROOT, "explore", "E03_trade_conditions"))
from analysis.approved_data import GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (palette, page styling, percentile)
import zwin as ZW  # noqa: E402  (the z windows)
import variants as VR  # noqa: E402  (the rule through the engine's book)
import conditions as E3  # noqa: E402  (the correlation periods and their look-ahead check)

THRS = (("t10", 1.0, "±1"), ("t15", 1.5, "±1.5"), ("t20", 2.0, "±2")); TDEFAULT = "t10"
NBACKS = (4, 13, 26); NDEFAULT = 13
CWINS = E3.CWINS; CDEFAULT = E3.CDEFAULT
SEED = 107; MIN_GROUP = 5; COST = VR.COST
ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR
POOL = {p: ("gold" if GOLD in p else "index") for p in PAIRS}; INDEX_PAIRS = [p for p in PAIRS if POOL[p] == "index"]
LO, HI = E1.LO, E1.HI


def r4(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def falling_flags(cf: pd.DataFrame) -> dict:
    """Per (period, N): 1 = the correlation is lower than N decision weeks earlier, 0 = not lower, -1 = unknown."""
    out = {}
    for c, *_ in CWINS:
        cur = cf[f"corr_{c}"]
        for n in NBACKS:
            prev = cur.shift(n); known = cur.notna() & prev.notna()
            out[(c, n)] = np.where(known, np.where(cur < prev, 1, 0), -1).astype(np.int8)
    return out


def run_stats(r: dict, bh: pd.Series) -> tuple:
    """Closed trades, share won, mean extra, CAGR, 50/50's CAGR, excess (E04's definition), max drawdown; and strategy ÷ 50/50."""
    ex = r["extras"]; cl = ex[~ex["open"].astype(bool)] if len(ex) else ex
    nv = pd.DataFrame({"nav": r["nav"], "bh": bh}).dropna(); yrs = (nv.index[-1] - nv.index[0]).days / 365.25
    cg = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
    st = {"trades": int(len(cl)), "open": int(ex["open"].astype(bool).sum()) if len(ex) else 0,
          "win": float((cl["extra"] > 0).mean()) if len(cl) else float("nan"), "mean_extra": float(cl["extra"].mean()) if len(cl) else float("nan"),
          "cagr": float(cg(nv["nav"])), "cagr_5050": float(cg(nv["bh"])), "maxdd": float((nv["nav"] / nv["nav"].cummax() - 1).min()),
          "first": nv.index[0].date().isoformat(), "last": nv.index[-1].date().isoformat()}
    st["excess"] = st["cagr"] - st["cagr_5050"]
    return st, nv["nav"] / nv["bh"]


def month_ends(rel: pd.Series) -> tuple:
    m = rel.groupby([rel.index.year, rel.index.month]).last()
    y0, m0 = m.index[0]; y1, m1 = m.index[-1]
    assert len(m) == (y1 - y0) * 12 + (m1 - m0) + 1, "a calendar month without a trading day"
    return f"{y0}-{m0:02d}", [int(round(v * 1000)) for v in m.to_numpy()]


def group_stats(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {"n": 0, "win": np.nan, "mean": np.nan, "median": np.nan, "weeks": np.nan, "long": np.nan, "first": None, "last": None}
    return {"n": int(len(d)), "win": float((d["extra"] > 0).mean()), "mean": float(d["extra"].mean()), "median": float(d["extra"].median()),
            "weeks": float(d["weeks"].median()), "long": float((d["weeks"] > 52).mean()), "first": str(d["entry_trade"].min()), "last": str(d["exit_trade"].max())}


def split(df: pd.DataFrame, flag: str, rng, P: int) -> dict:
    """Falling vs not falling among closed trades; the real differences against P within-pair shuffles of the labels."""
    known = df[df[flag] >= 0]; f = known[flag].to_numpy() == 1
    g1, g0 = group_stats(known[f]), group_stats(known[~f])
    out = {"unknown": int((df[flag] < 0).sum()), **{f"{k}_f": v for k, v in g1.items()}, **{f"{k}_nf": v for k, v in g0.items()}}
    out["d_mean"] = g1["mean"] - g0["mean"] if g1["n"] and g0["n"] else np.nan; out["d_win"] = g1["win"] - g0["win"] if g1["n"] and g0["n"] else np.nan
    out["d_mean_pct"] = out["d_win_pct"] = np.nan
    if g1["n"] >= MIN_GROUP and g0["n"] >= MIN_GROUP:
        y = known["extra"].to_numpy(float); w = (y > 0).astype(float); pr = known["pair"].to_numpy(); F = np.tile(f, (P, 1))
        for p in np.unique(pr):
            idx = np.flatnonzero(pr == p); F[:, idx] = f[rng.permuted(np.tile(idx, (P, 1)), axis=1)]
        n1 = F.sum(axis=1); n0 = (~F).sum(axis=1); assert (n1 == g1["n"]).all() and (n0 == g0["n"]).all()
        dm = (F * y).sum(axis=1) / n1 - ((~F) * y).sum(axis=1) / n0; dw = (F * w).sum(axis=1) / n1 - ((~F) * w).sum(axis=1) / n0
        out["d_mean_pct"] = E1.pctile(out["d_mean"], dm); out["d_win_pct"] = E1.pctile(out["d_win"], dw)
    return out


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=2000); P = ap.parse_args().perms
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily()
    checks = {"t03_mismatch": [], "allow_all_mismatch": [], "e04_excess_max_abs_diff": None, "corr_look_ahead": [0, []]}
    dates, Z, C, FL, TRS, FS, REL, trade_rows = {}, {}, {}, {}, {}, {}, {}, []
    split_pair, filt_rows = [], []
    for pair in PAIRS:
        pf = pair_frame(px, pair, "plain"); cf = E3.corr_frame(pf); dec = cf.index
        c_, b_ = E3.corr_pit_check(pf); checks["corr_look_ahead"][0] += c_; checks["corr_look_ahead"][1] += [f"{pair}: {x}" for x in b_]
        flags = falling_flags(cf); dates[pair] = [d.date().isoformat() for d in dec]
        C[pair] = {c: [None if v != v else int(round(v * 1000)) for v in cf[f"corr_{c}"].to_numpy()] for c, *_ in CWINS}
        FL[pair] = {c: {str(n): "".join({1: "1", 0: "0", -1: "-"}[int(v)] for v in flags[(c, n)]) for n in NBACKS} for c, *_ in CWINS}
        Z[pair], TRS[pair], FS[pair], REL[pair] = {}, {}, {}, {}
        for wi, (w, wlab, f, n_) in enumerate(ZW.WINDOWS):
            z = VR.z_at_decisions(pf, w); assert z.index.equals(dec)
            Z[pair][w] = [None if v != v else int(round(v * 1000)) for v in z.to_numpy()]
            TRS[pair][w], FS[pair][w] = {}, {}; REL[pair][w] = {"u": {}, "f": {}}; bh = None
            for ti, (tk, thr, tlab) in enumerate(THRS):
                unf = VR.run_states(pf, z, COST, thr, comp=bh is None); bh = unf.get("bh", bh)
                if w == ZW.DEFAULT and tk == TDEFAULT:
                    checks["t03_mismatch"] += VR.check_against_t03({"pair": pair, **unf})
                al = VR.run_states(pf, z, COST, thr, allow=np.ones(len(z), bool), comp=False)
                if not (al["nav"].equals(unf["nav"]) and al["decisions"]["state_after"].equals(unf["decisions"]["state_after"])):
                    checks["allow_all_mismatch"].append(f"{pair} {w} {tk}")
                su, rel_u = run_stats(unf, bh); m0, REL[pair][w]["u"][tk] = month_ends(rel_u); REL[pair][w]["m0"] = m0
                # the unfiltered rule's trades, with the correlation flags at entry
                ex = unf["extras"]; trs = []
                if len(ex):
                    for _, t in ex.iterrows():
                        ed = pd.Timestamp(t["entry_decision"]); ei = int(dec.get_loc(ed)); op = bool(t["open"])
                        row = {"pair": pair, "pool": POOL[pair], "window": w, "extreme": tk, "entry_decision": ed.date().isoformat(), "entry_trade": pd.Timestamp(t["entry_trade"]).date().isoformat(),
                               "exit_trade": None if op else pd.Timestamp(t["exit_trade"]).date().isoformat(), "held": t["held"], "open": op, "z_entry": float(z.iloc[ei]),
                               "extra": float("nan") if op else float(t["extra"]), "weeks": None if op else int(((dec > ed) & (dec < pd.Timestamp(t["exit_trade"]))).sum()),
                               "days": None if op else int((pd.Timestamp(t["exit_trade"]) - pd.Timestamp(t["entry_trade"])).days)}
                        for c, *_ in CWINS:
                            row[f"corr_{c}"] = float(cf[f"corr_{c}"].iloc[ei])
                            for n in NBACKS:
                                row[f"fall_{c}_{n}"] = int(flags[(c, n)][ei])
                        trade_rows.append(row)
                        trs.append([ei, t["held"], None if op else int(round(row["extra"] * 10000)), row["weeks"], row["days"], row["entry_trade"], row["exit_trade"]])
                TRS[pair][w][tk] = trs
                # the filtered rule: an entry only when the correlation is falling (and, for contrast, only when it is not)
                FS[pair][w][tk] = {"u": su, "f": {}}; REL[pair][w]["f"][tk] = {}
                for ci, (c, *_) in enumerate(CWINS):
                    FS[pair][w][tk]["f"][c] = {}; REL[pair][w]["f"][tk][c] = {}
                    for n in NBACKS:
                        fl = flags[(c, n)]; rf = VR.run_states(pf, z, COST, thr, allow=fl == 1, comp=False); rn = VR.run_states(pf, z, COST, thr, allow=fl == 0, comp=False)
                        sf, rel_f = run_stats(rf, bh); sn, _ = run_stats(rn, bh); _, REL[pair][w]["f"][tk][c][str(n)] = month_ends(rel_f)
                        FS[pair][w][tk]["f"][c][str(n)] = [sf, sn]
                        filt_rows.append({"pair": pair, "pool": POOL[pair], "window": w, "extreme": tk, "corr_period": c, "falling_over": n, "first": su["first"], "last": su["last"], "cagr_5050": su["cagr_5050"],
                                          **{f"unfiltered_{k}": su[k] for k in ("trades", "win", "cagr", "excess", "maxdd")}, **{f"falling_{k}": sf[k] for k in ("trades", "win", "cagr", "excess", "maxdd")},
                                          **{f"not_falling_{k}": sn[k] for k in ("trades", "win", "cagr", "excess", "maxdd")}})
        print(f"  {pair:18s} {dates[pair][0]} → {dates[pair][-1]}  [{time.time() - t0:4.0f}s]", flush=True)
    if checks["t03_mismatch"] or checks["allow_all_mismatch"] or checks["corr_look_ahead"][1] or checks["corr_look_ahead"][0] == 0:
        sys.exit(f"checks FAILED: {checks}")
    TR = pd.DataFrame(trade_rows); TR.to_csv(os.path.join(OUT, "trades_flags.csv"), index=False)
    FR = pd.DataFrame(filt_rows); FR.to_csv(os.path.join(OUT, "filtered_summary.csv"), index=False)
    # the unfiltered rule's excess at ±1 must be E04's
    e4 = pd.read_csv(os.path.join(ROOT, "explore", "E04_signal_strength", "out", "outperformance_windows.csv"))
    mine = FR[(FR["extreme"] == TDEFAULT) & (FR["corr_period"] == CDEFAULT) & (FR["falling_over"] == NDEFAULT)].set_index(["window", "pair"])["unfiltered_excess"]
    theirs = e4.set_index(["window", "pair"])["excess_cagr"].reindex(mine.index)
    checks["e04_excess_max_abs_diff"] = float(np.nanmax(np.abs(mine.to_numpy() - theirs.to_numpy())))
    if not checks["e04_excess_max_abs_diff"] < 1e-12 or theirs.isna().any():
        sys.exit(f"the unfiltered rule's excess differs from E04's: {checks['e04_excess_max_abs_diff']}")
    print(f"  checks: T03 equal, allow-all equal, E04 excess equal (max {checks['e04_excess_max_abs_diff']:.1e}), correlation look-ahead {checks['corr_look_ahead'][0]} checks, 0 violations  [{time.time() - t0:4.0f}s]", flush=True)

    # ---------------------------------------------------------------- section 1: the split, pooled and per pair
    closed = TR[~TR["open"]]; PS, SP, srows = {}, {}, []
    for wi, (w, *_) in enumerate(ZW.WINDOWS):
        PS[w] = {}
        for ti, (tk, *_) in enumerate(THRS):
            PS[w][tk] = {}; sub = closed[(closed["window"] == w) & (closed["extreme"] == tk)]
            for ci, (c, *_) in enumerate(CWINS):
                PS[w][tk][c] = {}
                for ni, n in enumerate(NBACKS):
                    PS[w][tk][c][str(n)] = {}; flag = f"fall_{c}_{n}"
                    for pi, pool in enumerate(("index", "gold")):
                        s = split(sub[sub["pool"] == pool], flag, np.random.default_rng([SEED, wi, ti, ci, ni, pi]), P)
                        PS[w][tk][c][str(n)][pool] = s; srows.append({"window": w, "extreme": tk, "corr_period": c, "falling_over": n, "pool": pool, **s})
                    for p in PAIRS:
                        d = sub[sub["pair"] == p]; kn = d[d[flag] >= 0]; g1, g0 = group_stats(kn[kn[flag] == 1]), group_stats(kn[kn[flag] == 0])
                        SP.setdefault(p, {}).setdefault(w, {}).setdefault(tk, {}).setdefault(c, {})[str(n)] = [g1["n"], g0["n"], int((d[flag] < 0).sum()), r4(g1["win"]), r4(g0["win"]), r4(g1["mean"]), r4(g0["mean"])]
                        split_pair.append({"pair": p, "window": w, "extreme": tk, "corr_period": c, "falling_over": n, "n_f": g1["n"], "n_nf": g0["n"], "unknown": int((d[flag] < 0).sum()),
                                           "win_f": g1["win"], "win_nf": g0["win"], "mean_f": g1["mean"], "mean_nf": g0["mean"]})
        print(f"  split {w:5s} [{time.time() - t0:4.0f}s]", flush=True)
    SS = pd.DataFrame(srows); SS.to_csv(os.path.join(OUT, "split_summary.csv"), index=False); pd.DataFrame(split_pair).to_csv(os.path.join(OUT, "split_by_pair.csv"), index=False)
    allp = [v for v in SS["d_mean_pct"].tolist() + SS["d_win_pct"].tolist() if v == v]; n_not = sum(E1.notable(v) for v in allp)
    # which way the notable mean-extra differences point, per pool (the settings overlap, so these are not independent counts)
    dirc = {pool: {"worse": int((g["d_mean_pct"] <= LO).sum()), "better": int((g["d_mean_pct"] >= HI).sum()), "n": int(g["d_mean_pct"].notna().sum())} for pool, g in SS.groupby("pool")}
    # the filtered rule across settings, six index pairs: [falling-only beats every-entry, pairs, median change vs every-entry,
    # not-falling-only beats every-entry, falling-only beats not-falling-only, median (falling-only − not-falling-only)]; excess CAGR
    FH = {}
    for (w, tk, c, n), g in FR[FR["pool"] == "index"].groupby(["window", "extreme", "corr_period", "falling_over"]):
        FH.setdefault(w, {}).setdefault(tk, {}).setdefault(c, {})[str(n)] = [int((g["falling_excess"] > g["unfiltered_excess"]).sum()), int(len(g)),
                                                                             r4((g["falling_excess"] - g["unfiltered_excess"]).median()), int((g["not_falling_excess"] > g["unfiltered_excess"]).sum()),
                                                                             int((g["falling_excess"] > g["not_falling_excess"]).sum()), r4((g["falling_excess"] - g["not_falling_excess"]).median())]
    fh_all = [v[0] for w in FH for tk in FH[w] for c in FH[w][tk] for v in FH[w][tk][c].values()]
    fh_h2h = [v[4] for w in FH for tk in FH[w] for c in FH[w][tk] for v in FH[w][tk][c].values()]
    fh_nf = [v[3] for w in FH for tk in FH[w] for c in FH[w][tk] for v in FH[w][tk][c].values()]
    ix_ = FR[FR["pool"] == "index"]; beat = {k: int((ix_[f"{k}_excess"] > 0).sum()) for k in ("unfiltered", "falling", "not_falling")}; beat["cases"] = int(len(ix_))   # (setting, index pair) cases ahead of 50/50
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "perms": P, "seed": f"default_rng([{SEED}, z_window_id, extreme_id, corr_period_id, falling_over_id, pool_id])",
             "controls": {"z_windows": [w[0] for w in ZW.WINDOWS], "extremes": [t[0] for t in THRS], "corr_periods": [c[0] for c in CWINS], "falling_over_weeks": list(NBACKS)},
             "defaults": {"z_window": ZW.DEFAULT, "extreme": TDEFAULT, "corr_period": CDEFAULT, "falling_over": NDEFAULT, "pair": DEFAULT_PAIR}, "cost_bps": COST, "min_group": MIN_GROUP,
             "checks": checks, "notable": [n_not, len(allp)], "filtered_settings_falling_beats_unfiltered_in_4_of_6": int(sum(v >= 4 for v in fh_all)), "filtered_settings": len(fh_all),
             "filtered_settings_not_falling_beats_unfiltered_in_4_of_6": int(sum(v >= 4 for v in fh_nf)), "filtered_settings_falling_beats_not_falling_in_4_of_6": int(sum(v >= 4 for v in fh_h2h)),
             "filtered_settings_not_falling_beats_falling_in_4_of_6": int(sum(v <= 2 for v in fh_h2h)), "direction_mean_extra": dirc, "index_cases_ahead_of_5050": beat,
             "trades": int(len(TR)), "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return "—" if v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
    def pc(v): return "—" if v != v else f"{v * 100:.0f}%"
    def fp(v): return E1.fp(v)
    wl = {w: f"{lab}{' (daily)' if f == 'D' else ''}" for w, lab, f, _ in ZW.WINDOWS}; cl_ = {c: f"{lab}{' (daily)' if f == 'D' else ''}" for c, lab, f, _ in CWINS}; tl_ = {t[0]: t[2] for t in THRS}
    md = [f"# E05 results — the ratio switch when the legs' correlation is falling (EXPLORATORY)\n",
          f"_Generated by `explore/E05_corr_falling/falling.py` on {audit['run_date']} from the approved data of 2026-11-12. Rule R (T03's rule) with the chosen z window and extreme; falling = the legs' correlation for the chosen period is lower than N decision weeks earlier. {P:,} within-pair shuffles per setting. Description, not a finding. Percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Against chance:** {n_not} of {len(allp)} split percentiles are notable across all settings (about {0.1 * len(allp):.0f} by chance; the settings overlap heavily, so the count is not of independent results).\n",
          f"**Which way:** mean extra, falling − not falling — index pairs: notably worse when falling in {dirc['index']['worse']} of {dirc['index']['n']} settings, notably better in {dirc['index']['better']}; gold pairs: worse in {dirc['gold']['worse']}, better in {dirc['gold']['better']} of {dirc['gold']['n']}.\n",
          f"**The filtered rule (six index pairs, {len(fh_all)} settings):** entering only on falling correlation beats entering on every cross in at least 4 of 6 pairs in {audit['filtered_settings_falling_beats_unfiltered_in_4_of_6']} settings, but entering only when it is *not* falling does so in {audit['filtered_settings_not_falling_beats_unfiltered_in_4_of_6']}: skipping trades alone helps, because the rule with every entry trails 50/50. Head to head, falling-only beats not-falling-only in at least 4 of 6 pairs in {audit['filtered_settings_falling_beats_not_falling_in_4_of_6']} settings and loses in at least 4 of 6 in {audit['filtered_settings_not_falling_beats_falling_in_4_of_6']}. Against 50/50 bought once, over all {beat['cases']:,} (setting, index pair) cases: every entry is ahead in {beat['unfiltered']}, falling-only in {beat['falling']}, not-falling-only in {beat['not_falling']}.\n",
          f"**Checks:** the unfiltered rule at 52 weeks and ±1 equals T03 on every pair; an allow-everything filter equals the unfiltered rule everywhere; the unfiltered rule's excess at ±1 equals E04's (max difference {checks['e04_excess_max_abs_diff']:.1e}); correlation look-ahead check {checks['corr_look_ahead'][0]} cuts, 0 violations.\n"]
    def split_line(s):
        return (f"{s['n_f']} / {s['n_nf']} ({s['unknown']}) | {pc(s['win_f'])} / {pc(s['win_nf'])} ({fp(s['d_win'])}, {pq(s['d_win_pct'])}) | {fp(s['mean_f'])} / {fp(s['mean_nf'])} ({fp(s['d_mean'])}, {pq(s['d_mean_pct'])}) | "
                f"{E1.f2(s['weeks_f'], 0)} / {E1.f2(s['weeks_nf'], 0)} | {pc(s['long_f'])} / {pc(s['long_nf'])}")
    hdr = "| trades falling / not (unknown) | share won falling / not (difference, pct) | mean extra falling / not (difference, pct) | median weeks held | held over 52 weeks |"
    md += [f"## 1. The default setting (z window {wl[ZW.DEFAULT]}, extreme {tl_[TDEFAULT]}, correlation {cl_[CDEFAULT]}, falling over {NDEFAULT} weeks)\n", "| pool " + hdr, "|---|---|---|---|---|---|"]
    for pool in ("index", "gold"):
        s = PS[ZW.DEFAULT][TDEFAULT][CDEFAULT][str(NDEFAULT)][pool]; md.append(f"| {pool} pairs ({s['first_f'] or s['first_nf']} → {s['last_f'] or s['last_nf']}) | " + split_line(s) + " |")
    md += [f"\n## 2. Every z window and extreme (correlation {cl_[CDEFAULT]}, falling over {NDEFAULT} weeks), six index pairs\n", "| z window | extreme " + hdr, "|---|---|---|---|---|---|---|"]
    for w, *_ in ZW.WINDOWS:
        for tk, *_ in THRS:
            md.append(f"| {wl[w]} | {tl_[tk]} | " + split_line(PS[w][tk][CDEFAULT][str(NDEFAULT)]["index"]) + " |")
    md += [f"\n## 3. Every correlation period and N (z window {wl[ZW.DEFAULT]}, extreme {tl_[TDEFAULT]}), six index pairs\n", "| correlation | falling over " + hdr, "|---|---|---|---|---|---|---|"]
    for c, *_ in CWINS:
        for n in NBACKS:
            md.append(f"| {cl_[c]} | {n} wk | " + split_line(PS[ZW.DEFAULT][TDEFAULT][c][str(n)]["index"]) + " |")
    md += [f"\n## 4. The filtered rule at the default setting, per pair (excess CAGR over 50/50 bought once; closed trades; max drawdown)\n",
           "| pair | window | 50/50 CAGR | unfiltered | falling only | not falling only |", "|---|---|---|---|---|---|"]
    for p in ORDER:
        r = FR[(FR["pair"] == p) & (FR["window"] == ZW.DEFAULT) & (FR["extreme"] == TDEFAULT) & (FR["corr_period"] == CDEFAULT) & (FR["falling_over"] == NDEFAULT)].iloc[0]
        cell = lambda k: f"{r[k + '_excess'] * 100:+.1f} pp ({int(r[k + '_trades'])} trades, dd {r[k + '_maxdd'] * 100:.0f}%)"  # noqa: E731
        md.append(f"| {nice(p)} | {r['first']} → {r['last']} | {r['cagr_5050'] * 100:.1f}% | {cell('unfiltered')} | {cell('falling')} | {cell('not_falling')} |")
    md += [f"\n## 5. The filtered rule across settings, six index pairs, z window {wl[ZW.DEFAULT]}, extreme {tl_[TDEFAULT]}: pairs where falling-only beats not-falling-only (median difference in excess) · pairs where falling-only beats every entry · pairs where not-falling-only beats every entry\n",
           "| correlation | " + " | ".join(f"falling over {n} wk" for n in NBACKS) + " |", "|---|" + "---|" * len(NBACKS)]
    for c, *_ in CWINS:
        md.append(f"| {cl_[c]} | " + " | ".join((lambda r: f"{r[4]} of {r[1]} ({r[5] * 100:+.1f} pp) · {r[0]} · {r[3]}")(FH[ZW.DEFAULT][TDEFAULT][c][str(n)]) for n in NBACKS) + " |")
    open(os.path.join(OUT, "E05_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- the page: the results embedded, drawn by the page's script
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return r4(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    def slim_stats(s):   # [trades, open, win, mean extra, cagr, cagr 50/50, excess, maxdd]
        return [s["trades"], s["open"], r4(s["win"]), r4(s["mean_extra"]), r4(s["cagr"]), r4(s["cagr_5050"]), r4(s["excess"]), r4(s["maxdd"])]
    FSJ = {p: {w: {tk: {"u": slim_stats(FS[p][w][tk]["u"]), "win": [FS[p][w][tk]["u"]["first"], FS[p][w][tk]["u"]["last"]],
                        "f": {c: {n: [slim_stats(v[0]), slim_stats(v[1])] for n, v in FS[p][w][tk]["f"][c].items()} for c in FS[p][w][tk]["f"]}} for tk in FS[p][w]} for w in FS[p]} for p in FS}
    PSJ = {w: {tk: {c: {n: {pool: [s["n_f"], s["n_nf"], s["unknown"], r4(s["win_f"]), r4(s["win_nf"]), r4(s["mean_f"]), r4(s["mean_nf"]), r4(s["median_f"]), r4(s["median_nf"]), r4(s["weeks_f"], 1), r4(s["weeks_nf"], 1),
                                            r4(s["long_f"]), r4(s["long_nf"]), r4(s["d_mean"]), r4(s["d_mean_pct"], 1), r4(s["d_win"]), r4(s["d_win_pct"], 1), s["first_f"] or s["first_nf"], s["last_f"] or s["last_nf"]]
                                     for pool, s in PS[w][tk][c][n].items()} for n in PS[w][tk][c]} for c in PS[w][tk]} for tk in PS[w]} for w in PS}
    DATA = clean({"pairs": ORDER, "nice": {p: nice(p) for p in ORDER}, "pool": POOL, "windows": [[w, lab, f] for w, lab, f, _ in ZW.WINDOWS], "thrs": [[t[0], t[1], t[2]] for t in THRS],
                  "cwins": [[c, lab, f] for c, lab, f, _ in CWINS], "nbacks": list(NBACKS), "defaults": audit["defaults"], "lo": LO, "hi": HI, "minGroup": MIN_GROUP,
                  "dates": dates, "z": Z, "corr": C, "flags": FL, "trades": TRS, "fs": FSJ, "rel": REL, "ps": PSJ, "sp": SP, "fh": FH,
                  "pal": {"blue": E1.BLUE, "orange": E1.ORANGE, "aqua": E1.AQUA, "muted": E1.MUTED, "divLight": E1.DIV_LIGHT, "divDark": E1.DIV_DARK}})
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    def sel(i, lab, opts, default):
        return f"<label>{lab} <select id='{i}'>" + "".join(f"<option value='{html.escape(str(v), quote=True)}'{' selected' if str(v) == str(default) else ''}>{html.escape(t)}</option>" for v, t in opts) + "</select></label>"
    controls = (sel("pair", "Pair", [(p, nice(p)) for p in ORDER], DEFAULT_PAIR) + sel("zwin", "z window", [(w, wl[w]) for w, *_ in ZW.WINDOWS], ZW.DEFAULT)
                + sel("zthr", "z extreme", [(t[0], t[2]) for t in THRS], TDEFAULT) + sel("cwin", "Correlation period", [(c, cl_[c]) for c, *_ in CWINS], CDEFAULT)
                + sel("nback", "Falling over", [(n, f"{n} weeks") for n in NBACKS], NDEFAULT))
    from plotly.offline import get_plotlyjs
    plotly_js = "<script>" + get_plotlyjs() + "</script>"   # the plotly library, inlined once
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Falling correlation</title>"),
         "<header class='bar'><div class='bar-inner'><b>E05 · falling correlation</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Does the ratio switch do better when the legs' correlation is falling?</h1>",
         f"<p>The rule (T03's, with the extreme chosen at the top): every Friday close, z = (ratio − its trailing mean) ÷ its trailing sd over the chosen z window. From 50/50, the first week z crosses below −extreme → hold only the first index; above +extreme → hold only the second. "
         f"Back to 50/50 when z returns to 0. Trade the next session at the day's average price, 10.4 bps each way, no time limit. <b>Correlation falling</b> = the two indices' correlation over the chosen period is lower on the signal week than it was N weeks earlier (chosen at the top). "
         f"Generated {audit['run_date']} from the approved data of 2026-11-12. <b>Description, not a finding</b>: a filter that looks good here is a candidate for a pre-registered test (T04), not a rule.</p>",
         f"<div class='card'><p style='margin:4px 0'><b>Against chance:</b> {n_not} of {len(allp)} split percentiles across all {len(fh_all)} settings are notable (about {0.1 * len(allp):.0f} would be by chance; neighbouring settings share most of their trades, so these are not independent). "
         f"<b>Which way:</b> in the index pairs, trades entered while the correlation was falling did notably <i>worse</i> (mean extra) in {dirc['index']['worse']} of {dirc['index']['n']} settings and notably better in {dirc['index']['better']}; in the gold pairs, worse in {dirc['gold']['worse']} and better in {dirc['gold']['better']} of {dirc['gold']['n']}. "
         f"<b>The filtered rule:</b> entering only on falling correlation beats entering on every cross in at least 4 of the 6 index pairs in {audit['filtered_settings_falling_beats_unfiltered_in_4_of_6']} settings, but entering only when it is <i>not</i> falling does so in {audit['filtered_settings_not_falling_beats_unfiltered_in_4_of_6']} — skipping trades alone helps, because the rule with every entry trails 50/50. "
         f"Head to head, falling-only beats not-falling-only in at least 4 of 6 index pairs in {audit['filtered_settings_falling_beats_not_falling_in_4_of_6']} settings and loses in at least 4 of 6 in {audit['filtered_settings_not_falling_beats_falling_in_4_of_6']}. Against 50/50 bought once, over all {beat['cases']:,} (setting, index pair) cases: every entry is ahead in {beat['unfiltered']}, falling-only in {beat['falling']}, not-falling-only in {beat['not_falling']}. "
         f"<b>Checks:</b> 52 weeks and ±1 = T03 on every pair; the unfiltered rule's excess at ±1 = E04's; correlations from data up to the signal week only ({checks['corr_look_ahead'][0]} cut checks, 0 violations).</p></div>",
         "<div class='card'><p id='summary' style='margin:4px 0'></p></div>",
         "<h2>1 · The rule's trades, split by the correlation at entry</h2><p class='note'>Every closed trade of the unfiltered rule at the chosen z window and extreme. Falling = blue, not falling = grey. Share won = the held index beat the other (fill to fill, before costs); the percentile compares the real difference with 2,000 shuffles of the labels within each pair (≤ 5 or ≥ 95 = notable).</p>",
         "<div class='card'><div id='g_pool' class='plot'></div><div id='t_pool'></div></div>",
         "<div class='card'><div id='g_heatc' class='plot'></div></div>",
         "<div class='card'><div id='g_heatz' class='plot'></div></div>",
         "<h3>The chosen pair</h3><div class='card'><div id='g_zc' class='plot'></div></div>",
         "<div class='card'><div id='g_tr' class='plot'></div><details><summary>Table view</summary><div id='t_tr'></div></details></div>",
         "<h2>2 · The rule that enters only when the correlation is falling</h2><p class='note'>The same rule, but a first cross is taken only if the correlation is falling that week (an unknown counts as not falling); for contrast, also the rule that enters only when it is not falling. Skipping an entry can change later trades, so each version runs through the engine's own book. Excess = CAGR − 50/50 bought once's CAGR over the same days. <b>Read the falling-only line against the not-falling-only one</b>: because the rule with every entry trails 50/50, any version that trades less tends to look better than it.</p>",
         "<div class='card'><div id='g_rel' class='plot'></div><div id='t_rel'></div></div>",
         "<div class='card'><div id='g_heatf' class='plot'></div></div>",
         "<div class='card'><p class='note' style='margin:4px 0'>Every pair at the chosen setting:</p><div id='t_all'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Rule and book.</b> explore/E01_ratio_cycles/variants.py runs the state machine on z and passes the targets to analysis/ratio_engine.book (fills next session at OHLC4, 10.4 bps per side, units held, daily NAV). Decisions are weekly; daily z windows are read at each week's last trading day.</li>"
         "<li><b>Correlation.</b> E03's series: trailing Pearson correlation of weekly log returns on decision weeks (13/26/52 weeks) or of daily log returns (20/60/120/250 days, read at the decision day); the full window is required. Falling compares the value on the signal week with the value N decision weeks earlier; missing either = unknown.</li>"
         f"<li><b>Chance check.</b> Within each pair the falling/not-falling labels are shuffled among that pair's closed trades {P:,} times, seed default_rng([{SEED}, z_window_id, extreme_id, corr_period_id, falling_over_id, pool_id]); a group with fewer than {MIN_GROUP} trades gets no percentile. It asks only whether the label goes with better trades. The filtered rule has no null here; whether it beats random timing is a new pre-registered test.</li>"
         f"<li><b>Limits.</b> {len(fh_all)} settings per pair and two pools: about one percentile in ten is notable by chance, and neighbouring settings share most of their trades. Few trades per pair (fewer still at ±2). Gold pairs from 2015 with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); Micro250 pairs from 2022. Design written before the first run: explore/E05_corr_falling/E05.md.</li></ul></div>",
         "</main>", f"<script id='e05-data' type='application/json'>{blob}</script>", plotly_js, "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E05_falling.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable {n_not} of {len(allp)}; page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.legend-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}
td.n{font-weight:600}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e05-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif";
const FALL = D.pal.blue, NOTF = '#a8a69e';
const CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function v(id){return document.getElementById(id).value;}
function lab(list, key){const r=list.find(x=>String(x[0])===String(key));return r?(r[2]==='D'?r[1]+' (daily)':r[1]):key;}
function tlab(key){return D.thrs.find(t=>t[0]===key)[2];}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function fpp(x){if(x===null||x===undefined)return '—';let s=(x*100).toFixed(1);if(Number(s)===0)s='0.0';return (Number(s)>=0?'+':'')+s+' pp';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title, height, extra){
  const c=th();
  const base={template:'none', height:height, autosize:true, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:FONT,size:12,color:c.ink}, title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},
    margin:{l:60,r:24,t:112,b:56}, hovermode:'closest', legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',font:{color:c.ink2,size:11}}};
  return Object.assign(base, extra||{});
}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head, rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function months(m0, n){let [y,m]=m0.split('-').map(Number);const out=[];for(let i=0;i<n;i++){out.push(y+'-'+String(m).padStart(2,'0')+'-15');m++;if(m>12){m=1;y++;}}return out;}
function S(){return {p:v('pair'), w:v('zwin'), t:v('zthr'), c:v('cwin'), n:v('nback')};}
// pooled split record: [n_f,n_nf,unk,win_f,win_nf,mean_f,mean_nf,med_f,med_nf,wk_f,wk_nf,long_f,long_nf,d_mean,d_mean_pct,d_win,d_win_pct,first,last]
function ps(w,t,c,n,pool){return D.ps[w][t][c][String(n)][pool];}
function heatColors(){return isDark()?D.pal.divDark:D.pal.divLight;}

function drawSummary(s){
  const a=ps(s.w,s.t,s.c,s.n,'index'), g=ps(s.w,s.t,s.c,s.n,'gold'), fh=D.fh[s.w][s.t][s.c][String(s.n)];
  const set='z window '+lab(D.windows,s.w)+', extreme '+tlab(s.t)+', correlation '+lab(D.cwins,s.c)+', falling over '+s.n+' weeks';
  const line=(r,name)=>name+': with correlation falling, '+r[0]+' trades won '+pc(r[3])+' (mean extra '+fp(r[5])+'); not falling, '+r[1]+' trades won '+pc(r[4])+' (mean extra '+fp(r[6])+'); difference in mean extra '+fp(r[13])+', percentile '+pq(r[14])+' ('+(r[17]||'—')+' → '+(r[18]||'—')+')';
  document.getElementById('summary').innerHTML='<b>'+set+'.</b> '+line(a,'<b>Six index pairs</b>')+'. '+line(g,'<b>Four gold pairs</b>')+'. <b>The rule entering only on falling correlation</b> beats the rule entering only when it is not falling in '+fh[4]+' of '+fh[1]+' index pairs (median difference in excess '+fpp(fh[5])+'). Against the rule with every entry: falling-only better in '+fh[0]+' of '+fh[1]+', not-falling-only better in '+fh[3]+' of '+fh[1]+' (trading less helps a rule that trails 50/50).';
}
function drawPool(s){
  const c=th(), pools=['index','gold'], names=['six index pairs','four gold pairs'];
  const rec=pools.map(p=>ps(s.w,s.t,s.c,s.n,p));
  const tr=[];
  [['falling',FALL,3,5],['not falling',NOTF,4,6]].forEach(([nm,col,iw,im],k)=>{
    tr.push({type:'bar',x:names,y:rec.map(r=>r[iw]===null?null:r[iw]*100),name:'correlation '+nm,marker:{color:col},legendgroup:nm,xaxis:'x',yaxis:'y',
      text:rec.map(r=>r[iw]===null?'':(r[iw]*100).toFixed(0)+'% of '+r[k]),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: '+nm+' — %{text} won<extra></extra>'});
    tr.push({type:'bar',x:names,y:rec.map(r=>r[im]===null?null:r[im]*100),name:'correlation '+nm,marker:{color:col},legendgroup:nm,showlegend:false,xaxis:'x2',yaxis:'y2',
      text:rec.map(r=>r[im]===null?'':fp(r[im])),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: '+nm+' — mean extra %{text}<extra></extra>'});
  });
  const t=two('The rule\'s trades split by the correlation at entry · z window '+lab(D.windows,s.w)+', extreme '+tlab(s.t)+', correlation '+lab(D.cwins,s.c)+' falling over '+s.n+' weeks',
    'index pairs '+(rec[0][17]||'—')+' → '+(rec[0][18]||'—')+': mean-extra difference percentile '+pq(rec[0][14]).replace(/<\/?b>/g,'')+' · gold pairs '+(rec[1][17]||'—')+' → '+(rec[1][18]||'—')+': '+pq(rec[1][14]).replace(/<\/?b>/g,''));
  const L=layout(t,450,{barmode:'group',
    xaxis:ax({domain:[0,0.46],anchor:'y'}),yaxis:ax({anchor:'x',title:{text:'share of trades won',font:{color:c.ink2}},ticksuffix:'%',rangemode:'tozero'}),
    xaxis2:ax({domain:[0.54,1],anchor:'y2'}),yaxis2:ax({anchor:'x2',title:{text:'mean extra per trade',font:{color:c.ink2}},ticksuffix:'%'}),
    margin:{l:60,r:24,t:140,b:56},legend:{orientation:'h',x:0,y:1.16,yanchor:'bottom',font:{color:c.ink2,size:11}},
    annotations:[{text:'share won',x:0.23,xref:'paper',y:1.01,yref:'paper',yanchor:'bottom',showarrow:false,font:{color:c.ink2}},{text:'mean extra (held − other, before costs)',x:0.77,xref:'paper',y:1.01,yref:'paper',yanchor:'bottom',showarrow:false,font:{color:c.ink2}}]});
  Plotly.react('g_pool',tr,L,CFG);
  const rows=pools.map((p,i)=>{const r=rec[i];return [names[i]+' ('+(r[17]||'—')+' → '+(r[18]||'—')+')', r[0]+' / '+r[1]+' ('+r[2]+')', pc(r[3])+' / '+pc(r[4]), fp(r[15])+' ('+pq(r[16])+')', fp(r[5])+' / '+fp(r[6]), fp(r[13])+' ('+pq(r[14])+')', fp(r[7])+' / '+fp(r[8]), (r[9]===null?'—':r[9].toFixed(0))+' / '+(r[10]===null?'—':r[10].toFixed(0)), pc(r[11])+' / '+pc(r[12])];});
  document.getElementById('t_pool').innerHTML=table(['pool','trades falling / not (unknown)','share won falling / not','difference (percentile)','mean extra falling / not','difference (percentile)','median extra falling / not','median weeks held','held over 52 weeks'],rows);
}
function heat(div, rowsKeys, rowLabels, colDefs, cell, title, chosen, height, zrange, colorbar){
  const c=th(); const z=[], tx=[], hv=[];
  rowsKeys.forEach(rk=>{const zz=[],tt=[],hh=[];colDefs.forEach(cd=>{const o=cell(rk,cd);zz.push(o.z);tt.push(o.t);hh.push(o.h);});z.push(zz);tx.push(tt);hv.push(hh);});
  const shapes=chosen.map(ch=>({type:'rect',xref:'x',yref:'y',x0:ch[1]-0.5,x1:ch[1]+0.5,y0:ch[0]-0.5,y1:ch[0]+0.5,line:{color:c.ink,width:2}}));
  const half=colDefs.length/2; if(Number.isInteger(half))shapes.push({type:'line',xref:'x',yref:'paper',x0:half-0.5,x1:half-0.5,y0:0,y1:1,line:{color:c.axis,width:1}});
  Plotly.react(div,[{type:'heatmap',z:z,x:colDefs.map((cd,i)=>i),y:rowLabels.map((r,i)=>i),zmin:zrange[0],zmax:zrange[1],colorscale:heatColors(),text:tx,texttemplate:'%{text}',customdata:hv,hovertemplate:'%{customdata}<extra></extra>',xgap:2,ygap:2,
    colorbar:Object.assign({thickness:12,tickfont:{color:c.muted},title:{side:'right',font:{color:c.ink2}}},colorbar)}],
    layout(title,height,{margin:{l:190,r:24,t:112,b:30},shapes:shapes,xaxis:ax({tickvals:colDefs.map((cd,i)=>i),ticktext:colDefs.map(cd=>cd.label),side:'top',showgrid:false,showline:false}),
      yaxis:ax({tickvals:rowLabels.map((r,i)=>i),ticktext:rowLabels,autorange:'reversed',showgrid:false,showline:false,ticksuffix:'  '})}),CFG);
}
function cellSplit(r, what){
  if(!r)return {z:null,t:'',h:''};
  const pct=r[14], d=r[13];
  return {z:pct, t:(pct!==null&&(pct<=D.lo||pct>=D.hi))?fp(d):'', h:what+'<br>falling '+r[0]+' trades, mean extra '+fp(r[5])+'; not falling '+r[1]+', '+fp(r[6])+'<br>difference '+fp(d)+', percentile '+(pct===null?'— (a group under '+D.minGroup+' trades)':pct.toFixed(1))};
}
function drawHeats(s){
  const cols=[];['index','gold'].forEach(pool=>D.nbacks.forEach(n=>cols.push({pool:pool,n:n,label:(pool==='index'?'index':'gold')+' · '+n+' wk'})));
  const ci=D.cwins.findIndex(x=>x[0]===s.c), ni=D.nbacks.indexOf(Number(s.n));
  heat('g_heatc',D.cwins.map(x=>x[0]),D.cwins.map(x=>'correlation '+lab(D.cwins,x[0])),cols,(ck,cd)=>cellSplit(ps(s.w,s.t,ck,cd.n,cd.pool),lab(D.cwins,ck)+' correlation, falling over '+cd.n+' wk, '+cd.pool+' pairs'),
    two('Every correlation period and lookback · z window '+lab(D.windows,s.w)+', extreme '+tlab(s.t),'colour = percentile of (mean extra falling − not falling) among the shuffles; red = falling did better, blue = worse; labels on notable cells; outlined = the chosen setting'),
    [[ci,ni],[ci,ni+D.nbacks.length]],430,[0,100],{title:{text:'percentile',side:'right'},tickvals:[0,5,50,95,100]});
  const colsZ=[];['index','gold'].forEach(pool=>D.thrs.forEach(t=>colsZ.push({pool:pool,t:t[0],label:(pool==='index'?'index':'gold')+' · '+t[2]})));
  const wi=D.windows.findIndex(x=>x[0]===s.w), ti=D.thrs.findIndex(x=>x[0]===s.t);
  heat('g_heatz',D.windows.map(x=>x[0]),D.windows.map(x=>'z window '+lab(D.windows,x[0])),colsZ,(wk,cd)=>cellSplit(ps(wk,cd.t,s.c,s.n,cd.pool),'z window '+lab(D.windows,wk)+', extreme '+tlab(cd.t)+', '+cd.pool+' pairs'),
    two('Every z window and extreme · correlation '+lab(D.cwins,s.c)+' falling over '+s.n+' weeks','colour = percentile of (mean extra falling − not falling) among the shuffles; red = falling did better, blue = worse; labels on notable cells; outlined = the chosen setting'),
    [[wi,ti],[wi,ti+D.thrs.length]],430,[0,100],{title:{text:'percentile',side:'right'},tickvals:[0,5,50,95,100]});
  const colsF=D.nbacks.map(n=>({n:n,label:'falling over '+n+' wk'}));
  heat('g_heatf',D.cwins.map(x=>x[0]),D.cwins.map(x=>'correlation '+lab(D.cwins,x[0])),colsF,(ck,cd)=>{const r=D.fh[s.w][s.t][ck][String(cd.n)];return {z:r[4],t:r[4]+' of '+r[1],h:lab(D.cwins,ck)+' correlation, falling over '+cd.n+' wk<br>falling-only beats not-falling-only in '+r[4]+' of '+r[1]+' index pairs (median difference in excess '+fpp(r[5])+')<br>against every entry: falling-only better in '+r[0]+', not-falling-only better in '+r[3]};},
    two('Entering only on falling correlation against entering only when it is not falling: index pairs (of 6) where falling-only does better · z window '+lab(D.windows,s.w)+', extreme '+tlab(s.t),'excess CAGR over 50/50 compared pair by pair; red = falling-only better in most pairs, blue = worse; 3 of 6 is a coin toss; outlined = the chosen setting'),
    [[ci,ni]],430,[0,6],{title:{text:'pairs',side:'right'},tickvals:[0,3,6]});
}
function drawPair(s){
  const c=th(), dates=D.dates[s.p], z=D.z[s.p][s.w], cr=D.corr[s.p][s.c], fl=D.flags[s.p][s.c][String(s.n)], trs=D.trades[s.p][s.w][s.t], thr=D.thrs.find(t=>t[0]===s.t)[1], N=Number(s.n);
  const zv=z.map(x=>x===null?null:x/1000), cv=cr.map(x=>x===null?null:x/1000), cprev=cv.map((x,i)=>i>=N?cv[i-N]:null);
  const groups={'1':{name:'entry, correlation falling',col:FALL,sym:'circle'},'0':{name:'entry, not falling',col:NOTF,sym:'circle'},'-':{name:'entry, correlation unknown',col:c.muted,sym:'circle-open'}};
  const tr=[{x:dates,y:zv,type:'scatter',mode:'lines',line:{color:D.pal.blue,width:1.2},name:'z',xaxis:'x',yaxis:'y',hovertemplate:'%{x}<br>z %{y:+.2f}<extra></extra>',opacity:0.85},
    {x:dates,y:cv,type:'scatter',mode:'lines',line:{color:c.ink2,width:1.4},name:'correlation now',xaxis:'x',yaxis:'y2',hovertemplate:'%{x}<br>correlation %{y:.2f}<extra></extra>'},
    {x:dates,y:cprev,type:'scatter',mode:'lines',line:{color:c.muted,width:1,dash:'dot'},name:'correlation '+N+' weeks earlier',xaxis:'x',yaxis:'y2',hovertemplate:'%{x}<br>'+N+' weeks earlier %{y:.2f}<extra></extra>'}];
  Object.keys(groups).forEach(k=>{const g=groups[k];const sub=trs.filter(t=>fl[t[0]]===k);
    const hov=sub.map(t=>'entered '+t[5]+' (signal '+dates[t[0]]+'): hold '+(t[1]==='A'?s.p.split('|')[0]:s.p.split('|')[1])+'<br>z '+(zv[t[0]]===null?'—':zv[t[0]].toFixed(2))+' · correlation '+(cv[t[0]]===null?'—':cv[t[0]].toFixed(2))+' vs '+(cprev[t[0]]===null?'—':cprev[t[0]].toFixed(2))+' '+N+' wk earlier<br>'+(t[2]===null?'still open':'extra '+fp(t[2]/10000)+', '+t[3]+' weeks'));
    tr.push({x:sub.map(t=>dates[t[0]]),y:sub.map(t=>zv[t[0]]),type:'scatter',mode:'markers',marker:{color:g.col,size:8,symbol:g.sym,line:{width:1,color:g.col}},name:g.name,legendgroup:k,xaxis:'x',yaxis:'y',customdata:hov,hovertemplate:'%{customdata}<extra></extra>'});
    tr.push({x:sub.map(t=>dates[t[0]]),y:sub.map(t=>cv[t[0]]),type:'scatter',mode:'markers',marker:{color:g.col,size:8,symbol:g.sym,line:{width:1,color:g.col}},name:g.name,legendgroup:k,showlegend:false,xaxis:'x',yaxis:'y2',customdata:hov,hovertemplate:'%{customdata}<extra></extra>'});});
  const nf=trs.filter(t=>fl[t[0]]==='1').length, nn=trs.filter(t=>fl[t[0]]==='0').length, nu=trs.filter(t=>fl[t[0]]==='-').length;
  const first=dates.find((d,i)=>zv[i]!==null)||dates[0];
  const L=layout(two(D.nice[s.p]+' · z ('+lab(D.windows,s.w)+') and the correlation ('+lab(D.cwins,s.c)+') · '+first+' → '+dates[dates.length-1]+' · '+trs.length+' entries at '+tlab(s.t)+': '+nf+' with correlation falling, '+nn+' not, '+nu+' unknown',
      'top: z with the extreme (dashed); bottom: the correlation now (solid) and '+N+' weeks earlier (dotted) — falling where the solid line is under the dotted one; dots = the rule\'s entries'),640,
    {xaxis:ax({domain:[0,1],anchor:'y2',type:'date'}),yaxis:ax({domain:[0.56,1],anchor:'x',title:{text:'z',font:{color:c.ink2}}}),yaxis2:ax({domain:[0,0.44],anchor:'x',title:{text:'correlation',font:{color:c.ink2}}}),
     shapes:[-thr,thr].map(y=>({type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:y,y1:y,line:{color:c.axis,width:1,dash:'dash'}})).concat([{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}])});
  Plotly.react('g_zc',tr,L,CFG);
  // trade results as stems
  const DAY=86400000, st=[];
  Object.keys(groups).forEach(k=>{const g=groups[k];const sub=trs.filter(t=>fl[t[0]]===k&&t[2]!==null);
    st.push({type:'bar',x:sub.map(t=>t[5]),y:sub.map(t=>t[2]/100),width:12*DAY,marker:{color:g.col,line:{width:0}},name:g.name.replace('entry, ',''),
      customdata:sub.map(t=>t[5]+' → '+t[6]+': '+t[3]+' weeks, '+t[4]+' days'),hovertemplate:'%{customdata}<br>extra %{y:+.1f}%<extra></extra>'});});
  const sp=D.sp[s.p][s.w][s.t][s.c][String(s.n)];
  Plotly.react('g_tr',st,layout(two(D.nice[s.p]+' · each closed trade\'s extra at its entry date · falling: '+sp[0]+' trades, '+pc(sp[3])+' won, mean '+fp(sp[5])+' · not falling: '+sp[1]+', '+pc(sp[4])+', mean '+fp(sp[6])+(sp[2]?' · unknown: '+sp[2]:''),
      'extra = the held index\'s return minus the other\'s, entry fill to exit fill, before costs; blue = correlation falling at entry, grey = not'),420,
    {barmode:'overlay',bargap:0,xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%',title:{text:'extra',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}]}),CFG);
  const legs=s.p.split('|');
  document.getElementById('t_tr').innerHTML=table(['signal','entered','exited','held','z at signal','correlation',''+N+' wk earlier','falling?','weeks','extra'],
    trs.map(t=>[dates[t[0]],t[5],t[6]||'open',t[1]==='A'?legs[0]:legs[1],zv[t[0]]===null?'—':zv[t[0]].toFixed(2),cv[t[0]]===null?'—':cv[t[0]].toFixed(2),cprev[t[0]]===null?'—':cprev[t[0]].toFixed(2),{'1':'yes','0':'no','-':'unknown'}[fl[t[0]]],t[3]===null?'—':t[3],t[2]===null?'—':fp(t[2]/10000)]));
}
function stRow(name, r){return [name, r[0]+(r[1]?' (+'+r[1]+' open)':''), pc(r[2]), fp(r[3]), (r[4]*100).toFixed(1)+'%', fpp(r[6]), (r[7]*100).toFixed(0)+'%'];}
function drawRel(s){
  const c=th(), R=D.rel[s.p][s.w], f=D.fs[s.p][s.w][s.t], fr=f.f[s.c][String(s.n)];
  const ru=R.u[s.t], rf=R.f[s.t][s.c][String(s.n)], x=months(R.m0, ru.length);
  const tr=[{x:x,y:ru.map(v=>v/1000),type:'scatter',mode:'lines',line:{color:NOTF,width:1.6},name:'the rule, every entry',hovertemplate:'%{x|%Y-%m}: strategy ÷ 50/50 %{y:.3f}<extra></extra>'},
            {x:x,y:rf.map(v=>v/1000),type:'scatter',mode:'lines',line:{color:FALL,width:2},name:'entries only on falling correlation',hovertemplate:'%{x|%Y-%m}: strategy ÷ 50/50 %{y:.3f}<extra></extra>'}];
  Plotly.react('g_rel',tr,layout(two(D.nice[s.p]+' · the rule ÷ 50/50 bought once · '+f.win[0]+' → '+f.win[1]+' · excess a year: every entry '+fpp(f.u[6])+', falling only '+fpp(fr[0][6])+', not falling only '+fpp(fr[1][6]),
      'month-end values; above 1 = ahead of 50/50; z window '+lab(D.windows,s.w)+', extreme '+tlab(s.t)+', correlation '+lab(D.cwins,s.c)+' falling over '+s.n+' weeks'),440,
    {xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'strategy ÷ 50/50',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:1,y1:1,line:{color:c.axis,width:1}}]}),CFG);
  const head=['version','closed trades','share won','mean extra','CAGR','excess over 50/50','max drawdown'];
  document.getElementById('t_rel').innerHTML=table(head,[stRow('the rule, every entry',f.u),stRow('entries only on falling correlation',fr[0]),stRow('entries only when not falling',fr[1]),['50/50 bought once','—','—','—',(f.u[5]*100).toFixed(1)+'%','—','—']])+"<p class='note'>"+D.nice[s.p]+', '+f.win[0]+' → '+f.win[1]+'.</p>';
  const rows=D.pairs.map(p=>{const g=D.fs[p][s.w][s.t], q=g.f[s.c][String(s.n)];return [D.nice[p]+(D.pool[p]==='gold'?' (gold)':''), g.win[0]+' → '+g.win[1], (g.u[5]*100).toFixed(1)+'%', fpp(g.u[6])+' ('+g.u[0]+')', fpp(q[0][6])+' ('+q[0][0]+')', fpp(q[1][6])+' ('+q[1][0]+')'];});
  document.getElementById('t_all').innerHTML=table(['pair','window','50/50 CAGR','excess: every entry (trades)','falling only (trades)','not falling only (trades)'],rows);
}
function render(){
  const s=S();
  try{localStorage.setItem('e05-state',JSON.stringify(s));}catch(e){}
  drawSummary(s); drawPool(s); drawHeats(s); drawPair(s); drawRel(s);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  try{const st=JSON.parse(localStorage.getItem('e05-state')||'null');if(st){[['pair','p'],['zwin','w'],['zthr','t'],['cwin','c'],['nback','n']].forEach(([id,k])=>{const el=document.getElementById(id);if(st[k]!==undefined&&Array.from(el.options).some(o=>o.value===String(st[k])))el.value=String(st[k]);});}}catch(e){}
  ['pair','zwin','zthr','cwin','nback'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    main()
