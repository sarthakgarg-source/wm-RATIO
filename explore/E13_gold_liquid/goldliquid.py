"""
E13 (EXPLORATORY): gold on SELL goes to the liquid ETF — every Indian gold-holding strategy of E03–E11 (Rule R, E10's
supertrend schemes and momentum on the gold pairs, E11's India book), on the latest data from 2012, as before and with
the switch. Design: explore/E13_gold_liquid/E13.md (written before the first run).

    python explore/E13_gold_liquid/goldliquid.py [--max-shifts N]

Every number on the page and in E13_results.md is computed here; the page's script only picks the numbers of the chosen
strategy from the embedded results and draws them. Not a finding and not a test of any rule.
"""
import argparse
import html
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
for p_ in (ROOT, os.path.join(ROOT, "explore", "E01_ratio_cycles"), os.path.join(ROOT, "explore", "E10_trend_rotation"), os.path.join(ROOT, "explore", "E11_multi_asset")):
    sys.path.insert(0, p_)
from analysis.approved_data import GOLD, fingerprints, load_india_latest  # noqa: E402
from analysis.ratio_engine import PIT, book, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (page styling, percentile, notable)
import variants as VR  # noqa: E402  (Rule R's z and states)
import rotation as E10  # noqa: E402  (the desk's supertrend, the schemes, momentum, the two-leg book)
import multi as M  # noqa: E402  (the N-asset book and E11's strategies)

COST = VR.COST; C = COST / 2 / 1e4; LO, HI = E1.LO, E1.HI; MIN_SHIFT = 26
DATA_FROM, FIRST_DEC = pd.Timestamp("2011-01-01"), pd.Timestamp("2012-01-01")
IDX = ["N100", "MC100", "SC100"]; LIQ, LIQ2 = "Liquid", "LiquidCase"; BOOK_A = IDX + [GOLD]
CLS_B = [("equity", np.array([0, 1, 2]), 0.5), ("gold", np.array([3]), 0.5)]; NEUTRAL_B = np.array([1 / 6, 1 / 6, 1 / 6, 0.5])
TW3 = {"A": (1.0, 0.0, 0.0), "B": (0.0, 1.0, 0.0), "N": (0.5, 0.5, 0.0)}
SRC = {"R": "E03–E07, E09 · T03's Rule R", "S": "E10 · supertrend schemes", "M": "E10 · momentum", "B": "E11 · India book"}
FAM = {"R": "Rule R (±1 sd band)", "S": "supertrend schemes S1–S4", "M": "momentum (12-1, 12, 6 months)", "B": "the India book"}
LAB = {"N100": "N100", "MC100": "MC100", "SC100": "SC100", GOLD: "gold", LIQ: "liquid ETF", "5050": "50/50"}
r6, jclean = E10.r6, E10.jclean


def pair_frame_from(px: dict, a: str, b: str, start: pd.Timestamp) -> pd.DataFrame:
    """The engine's pair frame (analysis.ratio_engine.pair_frame, the same construction), from `start` instead of the engine's
    fixed gold-pair start (2015-01-12, the first day of the earlier gold file). Checked equal to the engine's on that start."""
    fa, fb = px[a], px[b]; idx = fa.index.intersection(fb.index); idx = idx[idx >= start]
    df = pd.DataFrame({"ca": fa.loc[idx, "close"], "cb": fb.loc[idx, "close"], "fa": fa.loc[idx, "fill"], "fb": fb.loc[idx, "fill"],
                       "stale": fa.loc[idx, "stale_ohl"] | fb.loc[idx, "stale_ohl"]}, index=idx)
    iso = df.index.isocalendar(); key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    df["decision"] = (~pd.Series(key.values, index=df.index).duplicated(keep="last")).values; df["ratio"] = df["ca"] / df["cb"]
    return df


def liq_on(idx: pd.DatetimeIndex, level: pd.Series):
    """The cash level as of each day of idx (the latest level on or before it) and the number of days that used an earlier day's level."""
    return level.reindex(idx, method="ffill").to_numpy(float), int((~idx.isin(level.index)).sum())


def switch(W: np.ndarray, gcol: int, lcol: int, sell: np.ndarray) -> np.ndarray:
    """With the switch: on a gold SELL week the weight in gold moves to the liquid ETF; nothing else changes."""
    W = np.array(W, float, copy=True); W[sell, lcol] += W[sell, gcol]; W[sell, gcol] = 0.0
    return W


def bench(cl, fi, kd, w):
    kt = kd[0] + 1; u = np.asarray(w, float) / (fi[kt] * (1 + C)); return np.r_[1.0, cl[kt:] @ u]


def gold_state(px: dict, dec: pd.DatetimeIndex, cal: pd.Series) -> np.ndarray:
    b = E10.desk_bars(px[GOLD], False); s, _ = E10.desk_supertrend(b); return E10.leg_state_at(dec, b, s, cal)


# ------------------------------------------------------------------------------------------------------- one strategy, as before and with the switch
def run_one(ck, cl, fi, cl2, fi2, kd, days, W0, w0, gcol, lcol, sell, shifts, base_nav=None):
    """W0, w0 as before (lcol = 0 there, or cash for E10's S1/S4 when base_nav is given); the switch moves gold to lcol on SELL weeks."""
    W1 = switch(W0, gcol, lcol, sell); w1 = switch(np.asarray(w0, float)[None, :], gcol, lcol, sell[:1])[0]
    ck["switch_gold_on_sell_max"] = max(ck["switch_gold_on_sell_max"], float(np.abs(W1[sell, gcol]).max()) if sell.any() else 0.0)
    ck["switch_moved_on_buy_max"] = max(ck["switch_moved_on_buy_max"], float(np.abs(W1[~sell] - W0[~sell]).max()) if (~sell).any() else 0.0)
    ck["weights_row_sum_max_dev"] = max(ck["weights_row_sum_max_dev"], float(np.abs(W0.sum(1) - 1).max()), float(np.abs(W1.sum(1) - 1).max()))
    nav0, sw0 = (base_nav if base_nav is not None else M.nbook(cl, fi, kd, W0, w0, rec=True))
    nav1, sw1 = M.nbook(cl, fi, kd, W1, w1, rec=True); nav1b, _ = M.nbook(cl2, fi2, kd, W1, w1)
    dd = (days[-1] - days[0]).days; NC, NDD = [], []
    for s_ in shifts:
        ss = np.roll(sell, s_); ck["shift_checks"] += 1; ck["shift_mismatch"] += int(ss.sum() != sell.sum())
        c_, m_ = E10.quick(M.nbook(cl, fi, kd, switch(W0, gcol, lcol, ss), switch(np.asarray(w0, float)[None, :], gcol, lcol, ss[:1])[0])[0], dd); NC.append(c_); NDD.append(m_)
    s0, s1, s1b = E10.nstats(nav0, days), E10.nstats(nav1, days), E10.nstats(nav1b, days); yrs = dd / 365.25
    return {"W0": W0, "W1": W1, "nav0": nav0, "nav1": nav1, "nc": NC,
            "row": {"cagr_before": s0["cagr"], "maxdd_before": s0["maxdd"], "vol_before": s0["vol"], "cagr_switch": s1["cagr"], "maxdd_switch": s1["maxdd"], "vol_switch": s1["vol"],
                    "d_cagr": s1["cagr"] - s0["cagr"], "d_maxdd": s1["maxdd"] - s0["maxdd"], "cagr_switch_liquidcase": s1b["cagr"],
                    "liquid_weeks": float((W1[:, lcol] > 1e-12).mean()), "gold_on_sell_before": float(((W0[:, gcol] > 1e-12) & sell).mean()),
                    "switches_yr_before": len(sw0) / yrs, "switches_yr_switch": len(sw1) / yrs, "pct_cagr": E1.pctile(s1["cagr"], NC), "pct_dd": E1.pctile(s1["maxdd"], NDD),
                    "null_median_cagr": float(np.median(NC)) if NC else np.nan, "shifts": len(shifts)}}


def shift_list(n: int, ms: int) -> list:
    s = list(range(MIN_SHIFT, n - MIN_SHIFT + 1))
    return s[:: max(1, len(s) // ms)][:ms] if ms else s


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-shifts", type=int, default=0); MS = ap.parse_args().max_shifts
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); L = load_india_latest()
    px = {k: L[k][L[k].index >= DATA_FROM] for k in BOOK_A}; gcal = E10.cal_end_of(px[GOLD].index)
    ck = {"close_max_abs_diff": 0.0, "engine_book_max_abs_diff": 0.0, "book5_vs_book4_max_abs_diff": 0.0, "gbook_vs_nbook_max_abs_diff": 0.0, "book_gold_state_mismatch": 0,
          "switch_gold_on_sell_max": 0.0, "switch_moved_on_buy_max": 0.0, "weights_row_sum_max_dev": 0.0, "cut_checks": 0, "cut_violations": [], "shift_checks": 0, "shift_mismatch": 0,
          "st_missing_after_start": 0, "liquid_carried_days": {}, "pair_frame_vs_engine_mismatch": 0}
    ROWS, CURV, WTS, NULLS, BEN = [], {}, {}, {}, {}
    # ---------------------------------------------------------------- the three gold pairs: R, S, M
    for x in IDX:
        pf = pair_frame_from(px, x, GOLD, DATA_FROM); key = f"{x}|{GOLD}"
        pe, pm = pair_frame(px, key, "plain"), pair_frame_from(px, x, GOLD, pd.Timestamp("2015-01-12"))
        ck["pair_frame_vs_engine_mismatch"] += int(not (pe.index.equals(pm.index) and np.allclose(pe[["ca", "cb", "fa", "fb", "ratio"]].to_numpy(), pm[["ca", "cb", "fa", "fb", "ratio"]].to_numpy(), rtol=0, atol=0)
                                                        and (pe["decision"] == pm["decision"]).all() and (pe["stale"] == pm["stale"]).all()))
        ck["close_max_abs_diff"] = max(ck["close_max_abs_diff"], float((pf["ca"] - px[x]["close"].reindex(pf.index)).abs().max()), float((pf["cb"] - px[GOLD]["close"].reindex(pf.index)).abs().max()))
        dec = pf.index[pf["decision"]]; k0 = int(np.searchsorted(dec.values, np.datetime64(FIRST_DEC))); kd = pf.index.get_indexer(dec[k0:]); days = pf.index[kd[0]:]
        lv, carried = liq_on(pf.index[kd[0]:], L[LIQ]["close"]); lv = np.r_[np.full(kd[0], np.nan), lv]; lv2, _ = liq_on(pf.index, L[LIQ2]["close"]); ck["liquid_carried_days"][key] = carried
        cl = np.column_stack([pf["ca"], pf["cb"], lv]); fi = np.column_stack([pf["fa"], pf["fb"], lv]); cl2 = np.column_stack([pf["ca"], pf["cb"], lv2]); fi2 = np.column_stack([pf["fa"], pf["fb"], lv2])
        gs = gold_state(px, dec, gcal); sell = gs[k0:] == -1; ck["st_missing_after_start"] += int(np.isnan(gs[k0:]).sum())
        n = len(dec) - k0; shifts = shift_list(n, MS); pos = pf.index.get_indexer(dec[k0:]) - kd[0]; wk = pf.loc[dec]
        BEN[key] = {"5050": bench(cl, fi, kd, (0.5, 0.5, 0)), x: bench(cl, fi, kd, (1, 0, 0)), GOLD: bench(cl, fi, kd, (0, 1, 0)), LIQ: bench(cl, fi, kd, (0, 0, 1)), "pos": pos, "t": dec[k0:], "days": days}
        # point in time: gold's state and Rule R's states on the history cut at 8 decision weeks
        z, sR = VR.z_at_decisions(pf, "w52"), None; sR = np.array(VR.rule_states(z.to_numpy())[0], dtype=object)
        for T in np.unique(np.linspace(k0 + 10, len(dec) - 1, 8).astype(int)):
            pc_ = {k: v.loc[: dec[T]] for k, v in px.items()}; cut = pf.loc[: dec[T]]; ck["cut_checks"] += 1
            zc = VR.z_at_decisions(cut, "w52"); okk = np.array_equal(np.array(VR.rule_states(zc.to_numpy())[0], dtype=object), sR[: len(zc)]) and np.array_equal(gold_state(pc_, dec[: T + 1], gcal), gs[: T + 1], equal_nan=True)
            if not okk:
                ck["cut_violations"].append(f"{key} {dec[T].date()}")
        # R — Rule R
        W0 = np.array([TW3[s] for s in sR[k0:]], float); res = run_one(ck, cl, fi, cl2, fi2, kd, days, W0, (0.5, 0.5, 0.0), 1, 2, sell, shifts)
        ne, _ = book(pf, dec[k0:], list(sR[k0:]), COST, PIT()); ck["engine_book_max_abs_diff"] = max(ck["engine_book_max_abs_diff"], float(np.nanmax(np.abs(ne.to_numpy() - res["nav0"]))))
        ROWS.append({"family": "R", "code": f"R · {x} ÷ gold", "set": key, "label": f"Rule R on {x} ÷ gold", **res["row"]}); CURV[ROWS[-1]["code"]] = (res, key, (x, GOLD, LIQ))
        # S — E10's supertrend schemes (as before: cash at zero return through E10's own book)
        b_ = E10.desk_bars(px[x], False); sx, _ = E10.desk_supertrend(b_); sa = E10.leg_state_at(dec, b_, sx, E10.cal_end_of(px[x].index)); ck["st_missing_after_start"] += int(np.isnan(sa[k0:]).sum())
        cmb = E10.combo_of(sa == 1, gs == 1)[k0:]; A = {k: pf[k].to_numpy() for k in ("ca", "cb", "fa", "fb")}
        for s_ in ("S1", "S2", "S3", "S4"):
            W0 = E10.SCHEMES[s_][cmb]; base = E10.gbook(A, kd, W0, rec=True)
            if not W0[:, 2].any():   # no cash rows: the two books must agree
                ck["gbook_vs_nbook_max_abs_diff"] = max(ck["gbook_vs_nbook_max_abs_diff"], float(np.abs(base[0] - M.nbook(cl, fi, kd, W0, (0.5, 0.5, 0.0))[0]).max()))
            res = run_one(ck, cl, fi, cl2, fi2, kd, days, W0, (0.5, 0.5, 0.0), 1, 2, sell, shifts, base_nav=base)
            ROWS.append({"family": "S", "code": f"{s_} · {x} ÷ gold", "set": key, "label": f"{E10.SCH_LAB[s_]} on {x} and gold", **res["row"]}); CURV[ROWS[-1]["code"]] = (res, key, (x, GOLD, LIQ))
        # M — E10's momentum
        for mk, mlab, look, skip, _ in E10.MOM:
            st = E10.mom_states(wk["ca"].to_numpy(), wk["cb"].to_numpy(), look, skip)[k0:]; W0 = np.array([TW3[s] for s in st], float)
            res = run_one(ck, cl, fi, cl2, fi2, kd, days, W0, (0.5, 0.5, 0.0), 1, 2, sell, shifts)
            ROWS.append({"family": "M", "code": f"{mlab.replace('momentum ', 'mom ')} · {x} ÷ gold", "set": key, "label": f"{mlab} on {x} ÷ gold", **res["row"]}); CURV[ROWS[-1]["code"]] = (res, key, (x, GOLD, LIQ))
        print(f"  {x} ÷ gold  [{time.time() - t0:4.0f}s]", flush=True)
    # ---------------------------------------------------------------- B — E11's India book
    idx, cl4, fi4, decB = M.multi_frame(px, BOOK_A, DATA_FROM); cal_end = {a: E10.cal_end_of(px[a].index) for a in BOOK_A}
    ck["close_max_abs_diff"] = max(ck["close_max_abs_diff"], max(float(np.abs(cl4[:, j] - px[a]["close"].reindex(idx).to_numpy()).max()) for j, a in enumerate(BOOK_A)))
    S, Mo = M.signals(px, BOOK_A, idx, cl4, decB, cal_end, False); k0 = int(np.searchsorted(decB.values, np.datetime64(FIRST_DEC))); kd = idx.get_indexer(decB[k0:]); days = idx[kd[0]:]
    Sb, Mb = S[k0:], {k: v[k0:] for k, v in Mo.items()}; ck["st_missing_after_start"] += int(np.isnan(Sb).sum())
    sell = Sb[:, 3] == -1; ck["book_gold_state_mismatch"] = int((gold_state(px, decB, gcal)[k0:] != Sb[:, 3]).sum())
    for T in np.unique(np.linspace(k0 + 10, len(decB) - 1, 8).astype(int)):
        pc_ = {a: px[a].loc[: decB[T]] for a in BOOK_A}; ic = idx[idx <= decB[T]]; Sc, Mc = M.signals(pc_, BOOK_A, ic, cl4[: len(ic)], decB[: T + 1], cal_end, False); ck["cut_checks"] += 1
        if not (np.array_equal(Sc, S[: T + 1], equal_nan=True) and all(np.allclose(Mc[k], Mo[k][: T + 1], rtol=0, atol=1e-12, equal_nan=True) for k in Mo)):
            ck["cut_violations"].append(f"book {decB[T].date()}")
    lv, carried = liq_on(idx[kd[0]:], L[LIQ]["close"]); lv = np.r_[np.full(kd[0], np.nan), lv]; lv2, _ = liq_on(idx, L[LIQ2]["close"]); ck["liquid_carried_days"]["book"] = carried
    cl5, fi5 = np.column_stack([cl4, lv]), np.column_stack([fi4, lv]); cl5b, fi5b = np.column_stack([cl4, lv2]), np.column_stack([fi4, lv2])
    n = len(decB) - k0; shifts = shift_list(n, MS); pos = idx.get_indexer(decB[k0:]) - kd[0]
    BEN["book"] = {"neutral": bench(cl5, fi5, kd, np.r_[NEUTRAL_B, 0]), "equal": bench(cl5, fi5, kd, np.r_[np.full(4, 0.25), 0]), **{a: bench(cl5, fi5, kd, np.eye(5)[j]) for j, a in enumerate(BOOK_A + [LIQ])},
                   "pos": pos, "t": decB[k0:], "days": days}
    for code, fam, lab, look in M.STRATS + [(M.REF_CODE, "REF", "SC100 + gold only (E10's S2)", None)]:
        if fam == "REF":
            W4 = M.ref_targets(Sb, 2, 3); w0 = np.array([0, 0, 0.5, 0.5])
        else:
            W4 = M.targets(code, Sb, Mb[look] if look else None, CLS_B, NEUTRAL_B); w0 = NEUTRAL_B
        W5 = np.column_stack([W4, np.zeros(len(W4))]); w05 = np.r_[w0, 0.0]
        ck["book5_vs_book4_max_abs_diff"] = max(ck["book5_vs_book4_max_abs_diff"], float(np.abs(M.nbook(cl5, fi5, kd, W5, w05)[0] - M.nbook(cl4, fi4, kd, W4, w0)[0]).max()))
        res = run_one(ck, cl5, fi5, cl5b, fi5b, kd, days, W5, w05, 3, 4, sell, shifts)
        ROWS.append({"family": "B", "code": f"{code} · book", "set": "book", "label": f"{lab} (E11's India book)", **res["row"]}); CURV[ROWS[-1]["code"]] = (res, "book", tuple(BOOK_A + [LIQ]))
    print(f"  book  [{time.time() - t0:4.0f}s]", flush=True)
    bad = (ck["close_max_abs_diff"] > 0 or not ck["engine_book_max_abs_diff"] < 1e-9 or not ck["book5_vs_book4_max_abs_diff"] < 1e-12 or not ck["gbook_vs_nbook_max_abs_diff"] < 1e-12
           or ck["book_gold_state_mismatch"] or ck["switch_gold_on_sell_max"] > 0 or ck["switch_moved_on_buy_max"] > 0 or ck["weights_row_sum_max_dev"] > 1e-12 or ck["cut_violations"]
           or ck["shift_mismatch"] or ck["st_missing_after_start"] or ck["pair_frame_vs_engine_mismatch"])
    if bad:
        sys.exit(f"checks FAILED: {ck}")
    # ---------------------------------------------------------------- outputs
    for r in ROWS:
        res, key, _ = CURV[r["code"]]; b = BEN[key]; r.update({"source": SRC[r["family"]], "first": b["days"][0].date().isoformat(), "last": b["days"][-1].date().isoformat()})
    SR = pd.DataFrame(ROWS); SR.to_csv(os.path.join(OUT, "strategies.csv"), index=False)
    pd.DataFrame([{"code": c, "shift_index": i, "cagr": v} for c, (res, _, _) in CURV.items() for i, v in enumerate(res["nc"])]).to_csv(os.path.join(OUT, "nulls.csv"), index=False)
    BST = {k: {kk: E10.nstats(v, BEN[k]["days"]) for kk, v in BEN[k].items() if kk not in ("pos", "t", "days")} for k in BEN}
    gst = BEN["book"]; sellB = sell
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "data_from": DATA_FROM.date().isoformat(), "first_decision_from": FIRST_DEC.date().isoformat(),
             "cost_bps_round_trip": COST, "checks": ck, "benchmarks": BST, "gold_sell_share_book": float(sellB.mean()),
             "summary": {f: {"n": int((SR.family == f).sum()), "cagr_up": int(((SR.family == f) & (SR.d_cagr > 0)).sum()), "dd_shallower": int(((SR.family == f) & (SR.d_maxdd > 0)).sum()),
                             "median_d_cagr": float(SR[SR.family == f].d_cagr.median()), "median_d_maxdd": float(SR[SR.family == f].d_maxdd.median()),
                             "pct_high": int(((SR.family == f) & (SR.pct_cagr >= HI)).sum()), "pct_low": int(((SR.family == f) & (SR.pct_cagr <= LO)).sum())} for f in ("R", "S", "M", "B")},
             "seconds": round(time.time() - t0)}
    json.dump(jclean(audit), open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)
    write_md(audit, SR)
    write_page(audit, SR, CURV, BEN, px, gcal, L)
    print(f"done in {time.time() - t0:.0f}s; checks {ck}")


# ------------------------------------------------------------------------------------------------------- results (markdown)
def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")


def write_md(audit, SR):
    c = audit["checks"]; B = audit["benchmarks"]
    md = ["# E13 results — gold on SELL goes to the liquid ETF: every Indian gold-holding strategy of E03–E11, from 2012, as before and with the switch (EXPLORATORY)\n",
          f"_Generated by `explore/E13_gold_liquid/goldliquid.py` on {audit['run_date']}. Data from {audit['data_from']} (approved `load_india_latest()`), first decision in 2012, weekly decisions, next-session fills, 10.4 bps per side. "
          "With the switch: on every week gold's weekly supertrend is SELL, the strategy's gold weight is held in the liquid ETF (and E10's cash too). Description, not a finding; percentiles in **bold** are notable (≤ 5 or ≥ 95)._\n",
          f"**Checks:** closes equal the loader's (max {c['close_max_abs_diff']:.1e}); Rule R as before equals the engine's book (max NAV difference {c['engine_book_max_abs_diff']:.1e}); the book with a liquid column at zero equals E11's four-asset book "
          f"(max {c['book5_vs_book4_max_abs_diff']:.1e}); E10's book and this study's agree where there is no cash (max {c['gbook_vs_nbook_max_abs_diff']:.1e}); the switch never leaves gold on a SELL week (max {c['switch_gold_on_sell_max']:.1e}) "
          f"and never moves weight on a BUY week (max {c['switch_moved_on_buy_max']:.1e}); weight rows sum to 1 (max deviation {c['weights_row_sum_max_dev']:.1e}); signals pass the cut-history check ({c['cut_checks']} cuts, "
          f"{len(c['cut_violations'])} violations); every shift kept gold's SELL weeks ({c['shift_checks']:,} shifts, {c['shift_mismatch']} mismatches). Days the liquid level was read from an earlier day: "
          + ", ".join(f"{k} {v}" for k, v in c["liquid_carried_days"].items()) + ".\n",
          f"Gold's weekly supertrend was SELL in {pc(audit['gold_sell_share_book'])} of the book's weeks. The liquid ETF alone grew {fp(B['book'][LIQ]['cagr'])} a year over the book's period (max drawdown {fp(B['book'][LIQ]['maxdd'], 1)}); "
          f"gold alone {fp(B['book'][GOLD]['cagr'])} ({fp(B['book'][GOLD]['maxdd'], 0)}); neutral bought once {fp(B['book']['neutral']['cagr'])} ({fp(B['book']['neutral']['maxdd'], 0)}).\n",
          "## Summary by family\n", "| family | source | strategies | CAGR up with the switch | drawdown shallower | median CAGR change | median drawdown change | N-G notable high / low |", "|---|---|---|---|---|---|---|---|"]
    for f, s in audit["summary"].items():
        md.append(f"| {FAM[f]} | {SRC[f]} | {s['n']} | {s['cagr_up']} | {s['dd_shallower']} | {pp(s['median_d_cagr'])} | {pp(s['median_d_maxdd'])} | {s['pct_high']} / {s['pct_low']} |")
    md += ["", "## Every strategy\n",
           "| strategy | period | CAGR before → with the switch | change | max drawdown before → with | change | volatility before → with | weeks in the liquid ETF | weeks the old strategy held gold on SELL | N-G pct: CAGR / drawdown | with LiquidCase |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in SR.itertuples():
        md.append(f"| {r.label} | {r.first} → {r.last} | {fp(r.cagr_before)} → {fp(r.cagr_switch)} | {pp(r.d_cagr)} | {fp(r.maxdd_before, 0)} → {fp(r.maxdd_switch, 0)} | {pp(r.d_maxdd)} | {pc(r.vol_before)} → {pc(r.vol_switch)} | "
                  f"{pc(r.liquid_weeks)} | {pc(r.gold_on_sell_before)} | {pq(r.pct_cagr)} / {pq(r.pct_dd)} | {fp(r.cagr_switch_liquidcase)} |")
    md += ["", "Benchmarks (bought once, over each set's period): " + "; ".join(f"{k.replace('|' + GOLD, ' ÷ gold')}: " + ", ".join(f"{LAB.get(kk, kk)} {fp(v['cagr'])} ({fp(v['maxdd'], 0)})" for kk, v in B[k].items()) for k in B) + "."]
    open(os.path.join(OUT, "E13_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


# ------------------------------------------------------------------------------------------------------- the page
def write_page(audit, SR, CURV, BEN, px, gcal, L):
    strat = {}
    for r in SR.to_dict("records"):
        res, key, assets = CURV[r["code"]]; b = BEN[key]; pos = b["pos"]
        strat[r["code"]] = {**{k: (r6(v) if isinstance(v, float) else v) for k, v in r.items()}, "assets": [LAB.get(a, a) for a in assets],
                            "before": [r6(v) for v in res["nav0"][pos]], "switch": [r6(v) for v in res["nav1"][pos]], "w1": np.round(res["W1"], 4), "w0": np.round(res["W0"], 4), "nc": res["nc"]}
    sets = {k: {"t": [d.date().isoformat() for d in b["t"]], "bench": {LAB.get(kk, kk): [r6(x) for x in v[b["pos"]]] for kk, v in b.items() if kk not in ("pos", "t", "days")}} for k, b in BEN.items()}
    gb = E10.desk_bars(px[GOLD], False); gs_, gstop = E10.desk_supertrend(gb); m_ = (gb["last_session"] >= pd.Timestamp(audit["first_decision_from"])).to_numpy()
    gold = {"t": [pd.Timestamp(v).date().isoformat() for v in gb["last_session"][m_]], "c": [r6(v) for v in gb["close"].to_numpy()[m_]], "stop": [r6(v) for v in gstop[m_]], "buy": [int(v == 1) for v in gs_[m_]]}
    lq = {k: L[k]["close"][L[k].index >= pd.Timestamp(audit["first_decision_from"])] for k in (LIQ, LIQ2)}
    liq = {k: {"t": [d.date().isoformat() for d in v.index[::5]], "v": [r6(x / v.iloc[0]) for x in v.to_numpy()[::5]]} for k, v in lq.items()}
    data = {"strat": strat, "order": list(SR["code"]), "fam": FAM, "src": SRC, "sets": sets, "gold": gold, "liq": liq, "summary": audit["summary"], "bst": audit["benchmarks"], "lo": LO, "hi": HI,
            "families": [[f, FAM[f]] for f in ("R", "S", "M", "B")], "sell_share": audit["gold_sell_share_book"]}
    blob = json.dumps(jclean(data), separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")
    ssel = "<label>Strategy <select id='strat'>" + "".join(f"<optgroup label='{html.escape(FAM[f], quote=True)}'>" + "".join(f"<option value='{html.escape(c, quote=True)}'>{html.escape(c)}</option>" for c in SR[SR.family == f]["code"])
                                                           + "</optgroup>" for f in ("R", "S", "M", "B")) + "</select></label>"
    sm = audit["summary"]; B = audit["benchmarks"]["book"]
    key = ("<p style='margin:4px 0'>Gold's weekly supertrend was SELL in " + pc(audit["gold_sell_share_book"]) + " of the weeks from 2012. With the switch: " + "; ".join(
        f"<b>{FAM[f]}</b> ({SRC[f]}): CAGR up in {s['cagr_up']} of {s['n']}, drawdown shallower in {s['dd_shallower']}, median change {pp(s['median_d_cagr'])} a year and {pp(s['median_d_maxdd'])} of drawdown"
        for f, s in sm.items()) + f". The liquid ETF alone grew {fp(B[LIQ]['cagr'])} a year; gold alone {fp(B[GOLD]['cagr'])}; the book's neutral {fp(B['neutral']['cagr'])}.</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + M.CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Gold to liquid</title>"),
         "<header class='bar'><div class='bar-inner'><b>E13 · gold on SELL → liquid ETF</b><span class='tag'>EXPLORATORY</span>", ssel, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Gold on SELL goes to the liquid ETF: every Indian gold-holding strategy of E03–E11, from 2012, as before and with the switch</h1>",
         f"<p>Whenever a strategy holds MCX gold while gold's weekly supertrend is SELL, the switch holds that money in the liquid ETF instead (and E10's cash too). Latest approved data (load_india_latest), from 2011 for the signals' first year; "
         f"decisions from the first week of 2012 to 2026-11-20; weekly decisions, next-session fills, 10.4 bps per side. Generated {audit['run_date']}. <b>Description, not a finding</b>.</p>",
         "<div class='card'>" + key + "<p class='note' style='margin:4px 0'>E01 and E02 hold nothing; E08 and E12 are US studies (the liquid ETF is Indian, and the markets are kept apart); Micro250 ÷ gold is left out (no newer Micro250). "
         "The liquid level before 2024-01-24 has no recorded source (F-RA-D17).</p></div>",
         "<h2>1 · Gold's supertrend and the liquid ETF</h2>", "<div class='card'><div id='g_gold' class='plot'></div><div id='g_liq' class='plot'></div></div>",
         "<h2>2 · Every strategy, as before and with the switch</h2><p class='note' id='n_all'></p>", "<div class='card'><div id='g_bars' class='plot'></div></div><div class='card'><div id='t_all'></div></div>",
         "<h2>3 · The chosen strategy</h2>", "<div class='card'><div id='g_eq' class='plot'></div><div id='g_dd' class='plot'></div></div><div class='card'><div id='g_w' class='plot'></div></div><div class='card'><div id='g_null' class='plot'></div><div id='t_one'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Data.</b> analysis/approved_data.load_india_latest(): N100, MC100, SC100 and MCX gold to 2026-11-20 (gold from 2011 here, as asked), the liquid-ETF cash level (LBES) to 2026-11-25; LiquidCase as a check. Everything from 2011-01-01; decisions from 2012.</li>"
         "<li><b>The switch.</b> Gold on SELL = the desk's Weekly LW supertrend on MCX gold (E10's functions, gold's own weekly bars) at the decision week. On those weeks the strategy's gold weight is held in the liquid ETF; nothing else changes. The liquid level is read as of each day and traded at 10.4 bps per side.</li>"
         "<li><b>The strategies.</b> T03's Rule R on the three gold pairs (the rule of E03–E07 and E09), E10's supertrend schemes S1–S4 and momentum rules on the same pairs, and E11's India book — each exactly as in its study, from 2012.</li>"
         "<li><b>Null N-G.</b> The same strategy and switch with gold's SELL weeks slid in time (every circular shift of 26 weeks to the length less 26): does gold's actual SELL timing matter, or any SELL periods of the same length?</li>"
         "<li><b>Limits.</b> About 14.9 years; 37 strategies on seen data; the liquid level's early history has no recorded source; a rule worth keeping needs its own locked test. Design written before the run: explore/E13_gold_liquid/E13.md.</li></ul></div>",
         "</main>", f"<script id='e13-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E13_gold_liquid.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e13-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7', sell:'rgba(227,73,72,0.10)'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835', sell:'rgba(230,103,103,0.16)'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function ACOL(){return isDark()?['#3987e5','#d95926','#199e70','#c98500','#9c6ade']:['#2a78d6','#eb6834','#1baf7a','#eda100','#8a57d2'];}
function FCOL(){return isDark()?{R:'#898781',S:'#3987e5',M:'#c98500',B:'#199e70'}:{R:'#898781',S:'#2a78d6',M:'#eda100',B:'#1baf7a'};}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function fpp(x){if(x===null||x===undefined)return '—';let s=(x*100).toFixed(2);if(Number(s)===0)s='0.00';return (Number(s)>=0?'+':'')+s+' pp';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function f1(x,d){return x===null||x===undefined?'—':(+x).toFixed(d===undefined?1:d);}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:100,b:50},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',traceorder:'normal',font:{color:c.ink2,size:11}}},extra||{});}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function ddOf(a){let pk=-Infinity;return a.map(x=>{if(x===null)return null;pk=Math.max(pk,x);return (x/pk-1)*100;});}
function span(t){return t[0]+' → '+t[t.length-1];}
function sellShapes(G){const c=th(),out=[];let s=null;for(let i=0;i<G.t.length;i++){if(!G.buy[i]&&s===null)s=G.t[i];if((G.buy[i]||i===G.t.length-1)&&s!==null){out.push({type:'rect',xref:'x',yref:'paper',x0:s,x1:G.t[i],y0:0,y1:1,fillcolor:c.sell,line:{width:0},layer:'below'});s=null;}}return out;}
function drawTop(){
  const c=th(), G=D.gold;
  Plotly.react('g_gold',[{x:G.t,y:G.c,type:'scatter',mode:'lines',name:'MCX gold, weekly close',line:{color:c.ink,width:1.6},hovertemplate:'gold %{y:,.0f}<extra></extra>'},
    {x:G.t,y:G.stop,type:'scatter',mode:'lines',name:'supertrend line',line:{color:c.muted,width:1,dash:'dot'},hovertemplate:'supertrend %{y:,.0f}<extra></extra>'}],
    layout(two('MCX gold and its weekly supertrend · '+span(G.t),'shaded = SELL weeks ('+pc(D.sell_share)+' of the book\'s weeks): with the switch, gold\'s weight is in the liquid ETF there'),420,{shapes:sellShapes(G),hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'rupees per 10 g',font:{color:c.ink2}}})}),CFG);
  const Lq=D.liq, AC=ACOL();
  Plotly.react('g_liq',Object.keys(Lq).map((k,j)=>({x:Lq[k].t,y:Lq[k].v,type:'scatter',mode:'lines',name:k==='Liquid'?'liquid ETF (LBES, used)':'LiquidCase level (check)',line:{color:j?c.muted:AC[4],width:j?1.2:2,dash:j?'dash':'solid'},hovertemplate:'%{y:.3f}<extra></extra>'})),
    layout(two('The liquid ETF\'s cash level, value of 1 from 2012','its growth is what the switch earns instead of gold\'s on SELL weeks · before 2024-01-24 no recorded source (F-RA-D17)'),320,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({})}),CFG);
}
function drawAll(code){
  const c=th(), F=FCOL(), O=D.order, S=D.strat;
  document.getElementById('n_all').innerHTML='Change = with the switch − as before (CAGR in pp a year; drawdown in pp, positive = shallower). N-G percentile: the switch\'s CAGR among the same switch with gold\'s SELL weeks slid in time. ≤ 5 or ≥ 95 = notable.';
  Plotly.react('g_bars',[{type:'bar',name:'CAGR change (pp a year)',x:O,y:O.map(k=>S[k].d_cagr),marker:{color:O.map(k=>F[S[k].family]),line:{color:O.map(k=>k===code?c.ink:'rgba(0,0,0,0)'),width:2}},
    customdata:O.map(k=>S[k].label+'<br>CAGR '+fp(S[k].cagr_before)+' → '+fp(S[k].cagr_switch)+'<br>max drawdown '+fp(S[k].maxdd_before,0)+' → '+fp(S[k].maxdd_switch,0)+'<br>N-G percentile '+f1(S[k].pct_cagr)),hovertemplate:'%{customdata}<extra></extra>'},
    {type:'scatter',mode:'markers',name:'drawdown change (pp, + = shallower)',x:O,y:O.map(k=>S[k].d_maxdd),marker:{color:c.ink,size:7,symbol:'diamond'},hovertemplate:'drawdown change %{y:.1%}<extra></extra>'}],
    layout(two('Every strategy · what the switch changes · from 2012','bars: CAGR change, colour = family (grey Rule R, blue supertrend schemes, yellow momentum, aqua the India book); diamonds: drawdown change; outlined = chosen'),460,
    {margin:{l:60,r:24,t:110,b:150},xaxis:ax({tickangle:-50,tickfont:{size:10,color:c.muted}}),yaxis:ax({tickformat:'.1%',zeroline:true,zerolinecolor:c.axis})}),CFG);
  document.getElementById('t_all').innerHTML=table(['strategy','source','CAGR before → with','change','max drawdown before → with','change','weeks in the liquid ETF','old: gold held on SELL','N-G pct: CAGR / drawdown','with LiquidCase'],
    O.map(k=>{const q=S[k];return [(k===code?'<b>':'')+q.label+(k===code?'</b>':''),q.source,fp(q.cagr_before)+' → '+fp(q.cagr_switch),fpp(q.d_cagr),fp(q.maxdd_before,0)+' → '+fp(q.maxdd_switch,0),fpp(q.d_maxdd),pc(q.liquid_weeks),pc(q.gold_on_sell_before),pq(q.pct_cagr)+' / '+pq(q.pct_dd),fp(q.cagr_switch_liquidcase)];}));
}
function drawOne(code){
  const c=th(), AC=ACOL(), q=D.strat[code], St=D.sets[q.set], t=St.t, Bn=St.bench;
  const lines=[[q.switch,'with the switch',c.ink,2.6,'solid'],[q.before,'as before',c.muted,2.0,'dash']].concat(Object.keys(Bn).map((k,j)=>[Bn[k],k+' bought once',k==='liquid ETF'?AC[4]:AC[j%4],1.0,'dot']));
  Plotly.react('g_eq',lines.map(([y,nm,col,wd,dash])=>({x:t,y:y,type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(q.label+' · value of 1 · '+span(t),'after costs; log scale · CAGR '+fp(q.cagr_before)+' as before, '+fp(q.cagr_switch)+' with the switch'),480,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_dd',lines.slice(0,2).map(([y,nm,col,wd,dash])=>({x:t,y:ddOf(y),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · '+span(t),'max '+fp(q.maxdd_before,0)+' as before, '+fp(q.maxdd_switch,0)+' with the switch'),300,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
  const W=q.w1;
  Plotly.react('g_w',q.assets.map((a,j)=>({x:t,y:W.map(r=>r[j]),type:'scatter',mode:'lines',name:a,stackgroup:'one',line:{width:0.5,color:a==='liquid ETF'?AC[4]:AC[j%4],shape:'hv'},fillcolor:a==='liquid ETF'?AC[4]:AC[j%4],hovertemplate:a+': %{y:.0%}<extra></extra>'})),
    layout(two(q.label+' · target weights with the switch','liquid ETF in '+pc(q.liquid_weeks)+' of weeks; the old strategy held gold on a SELL week in '+pc(q.gold_on_sell_before)),360,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({tickformat:'.0%',range:[0,1]})}),CFG);
  Plotly.react('g_null',[{type:'histogram',x:q.nc.map(v=>v*100),name:'N-G: gold\'s SELL weeks slid in time',marker:{color:isDark()?'rgba(195,194,183,0.55)':'rgba(82,81,78,0.45)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} shifts<extra></extra>'}],
    layout(two(q.label+' · the switch\'s CAGR among the slid SELL weeks · '+q.shifts+' shifts','solid line = with the switch '+fp(q.cagr_switch)+' (percentile '+f1(q.pct_cagr)+'); dashed = as before '+fp(q.cagr_before)),340,
    {showlegend:false,shapes:[{type:'line',xref:'x',yref:'paper',x0:q.cagr_switch*100,x1:q.cagr_switch*100,y0:0,y1:1,line:{color:c.ink,width:2}},{type:'line',xref:'x',yref:'paper',x0:q.cagr_before*100,x1:q.cagr_before*100,y0:0,y1:1,line:{color:c.muted,width:2,dash:'dash'}}],
     xaxis:ax({ticksuffix:'%',title:{text:'CAGR',font:{color:c.ink2}}}),yaxis:ax({title:{text:'shifts',font:{color:c.ink2}}})}),CFG);
  document.getElementById('t_one').innerHTML=table(['','as before','with the switch'],[['CAGR',fp(q.cagr_before),fp(q.cagr_switch)],['max drawdown',fp(q.maxdd_before,0),fp(q.maxdd_switch,0)],['volatility',pc(q.vol_before),pc(q.vol_switch)],
    ['switches a year',f1(q.switches_yr_before),f1(q.switches_yr_switch)],['with LiquidCase instead','—',fp(q.cagr_switch_liquidcase)],['period',q.first+' → '+q.last,'']]);
}
function render(){const k=document.getElementById('strat').value;drawTop();drawAll(k);drawOne(k);}
document.getElementById('strat').addEventListener('change',render);
const tb=document.getElementById('theme');
function setTb(){tb.textContent=isDark()?'Light':'Dark';}
tb.addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';setTb();render();});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(!document.documentElement.dataset.theme){setTb();render();}});
setTb();render();
"""


if __name__ == "__main__":
    main()
