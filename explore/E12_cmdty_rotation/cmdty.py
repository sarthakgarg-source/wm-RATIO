"""
E12 (EXPLORATORY): SPY with three commodity ETFs (PDBC, SDCI, COM) — the desk's weekly supertrend on each asset and on
each ETF ÷ SPY, a rotation book of the four, and each pair's ±1 sd band (Rule R) and 50/50.
Design: explore/E12_cmdty_rotation/E12.md (written before the first run).

    python explore/E12_cmdty_rotation/cmdty.py [--draws 500]

Every number on the page and in E12_results.md is computed here; the page's script only picks the numbers of the chosen
controls from the embedded results and draws them. Not a finding and not a test of any rule.
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
from analysis.approved_data import fingerprints, load_us_etf  # noqa: E402
from analysis.ratio_engine import PIT, book, comparators, counts, null_A, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (page styling, percentile, notable)
import zwin as ZW  # noqa: E402  (the band's rolling window)
import variants as VR  # noqa: E402  (Rule R's z and states)
import rotation as E10  # noqa: E402  (the desk's supertrend, the two-leg book, nulls; checked in E10)
import multi as M  # noqa: E402  (the N-asset book and the class rules; checked in E11)

COST = VR.COST; C = COST / 2 / 1e4; LO, HI = E1.LO, E1.HI; K0, MIN_SHIFT = 52, 26
CM = ["PDBC", "SDCI", "COM"]; A4 = ["SPY"] + CM; PAIR_ORDER = ["PDBC", "COM", "SDCI"]
E10_PID = {"PDBC": 3, "COM": 6, "SDCI": 8}   # E10's pair ids (its US_MAIN order) — Rule R's N-A seeds are E10's, so its numbers must be E10's
RS_RULE_ID = 5                              # Rule RS's N-S seeds: default_rng([0, E10 pair id, 5, draw])
NAME = {"SPY": "SPY (S&P 500)", "PDBC": "PDBC (Invesco diversified commodity)", "SDCI": "SDCI (USCF SummerHaven dynamic commodity)", "COM": "COM (Direxion Auspice broad commodity)"}
TW = E10.TW
STRATS = [("ST-EQ", "ST", "absolute supertrend, equal split"), ("ST-BR", "ST", "absolute supertrend, breadth"), ("ST-TL", "ST", "absolute supertrend, tilt"),
          ("ST-MO-12", "STMO", "absolute supertrend + 12-month momentum pick"), ("GR-12", "GR", "gate and rank (12-month)"), ("MO-12", "MO", "momentum only, top 1 of 4 (12-month)"),
          ("RS-TL", "RS", "relative supertrend (ETF ÷ SPY), tilt"), ("RS-ROT", "RS", "relative supertrend (ETF ÷ SPY), rotation"), ("RR-3", "RR", "Rule R sleeves (±1 sd band, a third per pair)")]
FAMILIES = {"ST": "absolute supertrend", "STMO": "supertrend + momentum pick", "GR": "gate and rank", "MO": "momentum only", "RS": "relative supertrend (ETF ÷ SPY)", "RR": "Rule R (±1 sd band)"}
CLS = [("equity", np.array([0]), 0.5), ("commodity", np.array([1, 2, 3]), 0.5)]; NEUTRAL = np.array([0.5, 1 / 6, 1 / 6, 1 / 6])
r6, jclean = E10.r6, E10.jclean


# ------------------------------------------------------------------------------------------------------- signals
def ratio_bars(pf: pd.DataFrame):
    """The desk's weekly bars of the daily ratio X ÷ SPY on the pair's calendar (closes only) and its supertrend."""
    b = E10.desk_bars(pd.DataFrame({"close": pf["ratio"].to_numpy()}, index=pf.index), True); st, stop = E10.desk_supertrend(b)
    return b, st, stop


def rule_r(pf: pd.DataFrame):
    z = VR.z_at_decisions(pf, "w52"); return z, np.array(VR.rule_states(z.to_numpy())[0], dtype=object)


def book_signals(px: dict, idx, cl, dec, PF: dict, cal_end: dict, cal_pair: dict):
    """The book's weekly signal panel at its decision days: absolute supertrend S (n × 4, ±1), 12-month momentum (n × 4),
    relative supertrend RS (n × 3, ±1, ETF ÷ SPY) and Rule R states RR (n × 3, 'A' ETF / 'B' SPY / 'N' 50/50)."""
    S, Mo = M.signals(px, A4, idx, cl, dec, cal_end, True)
    RS = np.column_stack([E10.leg_state_at(dec, *ratio_bars(PF[x])[:2], cal_pair[x]) for x in CM])
    RR = np.column_stack([pd.Series(rule_r(PF[x])[1], index=PF[x].index[PF[x]["decision"]]).reindex(dec, method="ffill").fillna("N").to_numpy(dtype=object) for x in CM])
    return S, Mo["12"], RS, RR


def targets_all(S, Mo, RS, RR) -> dict:
    """Every strategy's target weights (rows sum to 1); a row depends only on that week's signals (E12.md, the strategies)."""
    n = len(S); W = {c: M.targets(c, S, Mo, CLS, NEUTRAL) for c in ("ST-EQ", "ST-BR", "ST-TL", "ST-MO-12", "GR-12", "MO-12")}
    w = np.tile(NEUTRAL, (n, 1)); sell = RS != 1
    for j in range(3):
        w[sell[:, j], 1 + j] -= 1 / 6; w[sell[:, j], 0] += 1 / 6
    W["RS-TL"] = w
    buy = RS == 1; k = buy.sum(1); w = np.zeros((n, 4)); w[k == 0, 0] = 1.0; w[k > 0, 1:] = buy[k > 0] / k[k > 0][:, None]; W["RS-ROT"] = w
    w = np.zeros((n, 4))
    for j in range(3):
        w[:, 0] += np.where(RR[:, j] == "B", 1 / 3, np.where(RR[:, j] == "N", 1 / 6, 0.0)); w[:, 1 + j] += np.where(RR[:, j] == "A", 1 / 3, np.where(RR[:, j] == "N", 1 / 6, 0.0))
    W["RR-3"] = w
    return W


# ------------------------------------------------------------------------------------------------------- part 1: the pairs
def run_pair(px: dict, x: str, ND: int, e10: pd.DataFrame, ck: dict) -> dict:
    pair = f"{x}|SPY"; pf = pair_frame(px, pair, "plain"); pid = E10_PID[x]
    ck["close_max_abs_diff"] = max(ck["close_max_abs_diff"], float((pf["ca"] - px[x]["close"].reindex(pf.index)).abs().max()), float((pf["cb"] - px["SPY"]["close"].reindex(pf.index)).abs().max()))
    dec = pf.index[pf["decision"]]; kd = pf.index.get_indexer(dec[K0:]); days = pf.index[kd[0]:]; dd_ = (days[-1] - days[0]).days; yrs = dd_ / 365.25
    A = {k: pf[k].to_numpy() for k in ("ca", "cb", "fa", "fb")}; wk = pf.loc[dec]
    z, sR = rule_r(pf)
    r = wk["ratio"]; roll = r.rolling(52, min_periods=ZW.min_obs(52)); mu, sd = roll.mean(), roll.std(); zz = (r - mu) / sd; ok = z.notna()
    ck["band_vs_z_max_abs_diff"] = max(ck["band_vs_z_max_abs_diff"], float((zz[ok] - z[ok]).abs().max()))
    rb, rst, rstop = ratio_bars(pf); srel = E10.leg_state_at(dec, rb, rst, E10.cal_end_of(pf.index))
    LEG = {}
    for leg in (x, "SPY"):
        b_ = E10.desk_bars(px[leg], True); s_, _ = E10.desk_supertrend(b_); LEG[leg] = E10.leg_state_at(dec, b_, s_, E10.cal_end_of(px[leg].index))
    ck["st_missing_after_start"] += int(np.isnan(srel[K0:]).sum() + np.isnan(LEG[x][K0:]).sum() + np.isnan(LEG["SPY"][K0:]).sum())
    W = {"R52": np.array([TW[s] for s in sR[K0:]], float), "RS": np.array([TW["A" if v == 1 else "B"] for v in srel[K0:]], float)}
    real = {k: E10.gbook(A, kd, W[k], rec=True) for k in W}
    ne, _ = book(pf, dec[K0:], list(sR[K0:]), COST, PIT()); ck["engine_book_max_abs_diff"] = max(ck["engine_book_max_abs_diff"], float(np.nanmax(np.abs(ne.to_numpy() - real["R52"][0]))))
    kt = kd[0] + 1
    bh = np.r_[1.0, 0.5 / (A["fa"][kt] * (1 + C)) * A["ca"][kt:] + 0.5 / (A["fb"][kt] * (1 + C)) * A["cb"][kt:]]
    ha = np.r_[1.0, A["ca"][kt:] / (A["fa"][kt] * (1 + C))]; hb = np.r_[1.0, A["cb"][kt:] / (A["fb"][kt] * (1 + C))]
    cmp_ = comparators(pf, dec[K0], COST, dec[K0:], PIT())
    for mine, k in ((bh, "5050_buyhold"), (ha, "hold_a"), (hb, "hold_b")):
        ck["benchmarks_vs_engine_max_abs_diff"] = max(ck["benchmarks_vs_engine_max_abs_diff"], float(np.nanmax(np.abs(mine - cmp_[k].to_numpy()))))
    # nulls: Rule R against N-A (E10's seeds), Rule RS against N-S (its spells in a random order)
    NC = {"R52": [], "RS": []}; NDD = {"R52": [], "RS": []}; tR = list(sR[K0:]); baseR = counts(tR)
    for dr in range(ND):
        rng = np.random.default_rng([0, pid, 0, dr]); tg = null_A(tR, rng); ck["null_draws"] += 1; ck["null_count_mismatch"] += int(counts(tg) != baseR)
        c_, m_ = E10.quick(E10.gbook(A, kd, np.array([TW[s] for s in tg], float))[0], dd_); NC["R52"].append(c_); NDD["R52"].append(m_)
    xs = srel[K0:] == 1; base = E10.spell_counts(xs)
    for dr in range(ND):
        rng = np.random.default_rng([0, pid, RS_RULE_ID, dr]); y = E10.shuffle_spells(xs, rng); ck["null_draws"] += 1; ck["null_count_mismatch"] += int(E10.spell_counts(y) != base)
        c_, m_ = E10.quick(E10.gbook(A, kd, np.array([TW["A" if v else "B"] for v in y], float))[0], dd_); NC["RS"].append(c_); NDD["RS"].append(m_)
    sbh, sa, sb = E10.nstats(bh, days), E10.nstats(ha, days), E10.nstats(hb, days); rows = []
    for k, lab, null in (("R52", "Rule R (±1 sd band, 52 weeks)", "N-A"), ("RS", "Rule RS (relative supertrend, ETF ÷ SPY)", "N-S")):
        nav, sw = real[k]; st = E10.nstats(nav, days); w = W[k]
        rows.append({"pair": pair, "x": x, "rule": k, "label": lab, "first": days[0].date().isoformat(), "last": days[-1].date().isoformat(), "years": yrs, "cagr": st["cagr"], "vol": st["vol"], "maxdd": st["maxdd"],
                     "cagr_5050": sbh["cagr"], "maxdd_5050": sbh["maxdd"], "cagr_x": sa["cagr"], "maxdd_x": sa["maxdd"], "cagr_spy": sb["cagr"], "maxdd_spy": sb["maxdd"], "excess_5050": st["cagr"] - sbh["cagr"],
                     "switches": len(sw), "switches_yr": len(sw) / yrs, "in_x": float((w[:, 0] == 1).mean()), "in_spy": float((w[:, 1] == 1).mean()), "in_5050": float((w[:, 0] == 0.5).mean()),
                     "null": null, "draws": ND, "pct": E1.pctile(st["cagr"], NC[k]), "pct_dd": E1.pctile(st["maxdd"], NDD[k]), "null_median_cagr": float(np.median(NC[k])) if ND else np.nan})
    q = e10[(e10["pair"] == pair) & (e10["rule"] == "R52")]
    if len(q):
        r0 = rows[0]; ck["e10_cagr_max_abs_diff"] = max(ck["e10_cagr_max_abs_diff"], float(abs(r0["cagr"] - q["cagr"].iloc[0])))
        ck["e10_switch_mismatch"] += int(r0["switches"] != int(q["switches"].iloc[0])); ck["e10_pct_max_abs_diff"] = max(ck["e10_pct_max_abs_diff"], float(abs(r0["pct"] - q["pct"].iloc[0])))
    else:
        ck["e10_missing"].append(pair)
    # the page's series: the band (every decision week), the trades, the curves and the supertrend states (from the start)
    tr = []; swR = real["R52"][1]
    for j, s in enumerate(swR):
        tr.append({"t": dec[K0 + s["i"]].date().isoformat(), "fill": pf.index[s["k"]].date().isoformat(), "to": "x" if s["to"][0] == 1.0 else ("spy" if s["to"][1] == 1.0 else "5050"), "ratio": r6(wk["ratio"].iloc[K0 + s["i"]])})
    pos = pf.index.get_indexer(dec[K0:]) - kd[0]
    ser = {"band": {"t": [d.date().isoformat() for d in dec], "ratio": [r6(v) for v in r.to_numpy()], "mean": [r6(v) for v in mu.to_numpy()], "up": [r6(v) for v in (mu + sd).to_numpy()],
                    "lo": [r6(v) for v in (mu - sd).to_numpy()], "z": [None if v != v else round(float(v), 3) for v in z.to_numpy()], "start": dec[K0].date().isoformat(), "trades": tr,
                    "stop": [None if v != v else r6(v) for v in pd.Series(rstop, index=rb["last_session"]).reindex(dec, method="ffill").to_numpy()],
                    "rel_buy": [None if v != v else int(v == 1) for v in srel]},
           "curves": {"t": [d.date().isoformat() for d in dec[K0:]], "R52": [r6(v) for v in real["R52"][0][pos]], "RS": [r6(v) for v in real["RS"][0][pos]], "bh": [r6(v) for v in bh[pos]],
                      "x": [r6(v) for v in ha[pos]], "spy": [r6(v) for v in hb[pos]]},
           "states": {"t": [d.date().isoformat() for d in dec[K0:]], "x": LEG[x][K0:].astype(int), "spy": LEG["SPY"][K0:].astype(int), "rel": srel[K0:].astype(int)},
           "null": {"R52": NC["R52"], "RS": NC["RS"]}}
    sx, ss, sr = LEG[x][K0:] == 1, LEG["SPY"][K0:] == 1, srel[K0:] == 1
    stt = {"pair": pair, "first": dec[K0].date().isoformat(), "last": dec[-1].date().isoformat(), "weeks": int(len(sr)), "buy_x": float(sx.mean()), "buy_spy": float(ss.mean()), "buy_rel": float(sr.mean()),
           "rel_flips_yr": float((sr[1:] != sr[:-1]).sum() / yrs), "rel_vs_x_minus_spy": float((sr == (sx & ~ss)).mean()), "rel_agrees_rule_r": float(((sR[K0:] == "A") & sr | (sR[K0:] == "B") & ~sr)[sR[K0:] != "N"].mean()) if (sR[K0:] != "N").any() else np.nan,
           "in_band_weeks": float((zz.iloc[K0:].abs() <= 1).mean()), "below_band": float((zz.iloc[K0:] < -1).mean()), "above_band": float((zz.iloc[K0:] > 1).mean())}
    return {"pair": pair, "x": x, "pf": pf, "rows": rows, "ser": ser, "stats": stt}


# ------------------------------------------------------------------------------------------------------- part 2: the book
def run_book(px: dict, PR: dict, ck: dict) -> dict:
    start = max(px[a].index[0] for a in A4); idx, cl, fi, dec = M.multi_frame(px, A4, start)
    ck["close_max_abs_diff"] = max(ck["close_max_abs_diff"], max(float(np.abs(cl[:, j] - px[a]["close"].reindex(idx).to_numpy()).max()) for j, a in enumerate(A4)))
    PF = {x: PR[x]["pf"] for x in CM}; cal_end = {a: E10.cal_end_of(px[a].index) for a in A4}; cal_pair = {x: E10.cal_end_of(PF[x].index) for x in CM}
    S, Mo, RS, RR = book_signals(px, idx, cl, dec, PF, cal_end, cal_pair)
    Sb, Mb, RSb, RRb = S[K0:], Mo[K0:], RS[K0:], RR[K0:]; n_b = len(Sb); kd = idx.get_indexer(dec[K0:]); days = idx[kd[0]:]; d_days = (days[-1] - days[0]).days; yrs = d_days / 365.25
    ck["st_missing_after_start"] += int(np.isnan(Sb).sum() + np.isnan(RSb).sum()); ck["mom_missing_after_start"] = int(np.isnan(Mb).sum())
    # point in time: everything recomputed on the history cut at 8 decision weeks
    W = targets_all(Sb, Mb, RSb, RRb)
    for T in np.unique(np.linspace(K0 + 10, len(dec) - 1, 8).astype(int)):
        pc_ = {a: px[a].loc[: dec[T]] for a in A4}; ic = idx[idx <= dec[T]]; PFc = {x: PF[x].loc[: dec[T]] for x in CM}; ck["cut_checks"] += 1
        Sc, Mc, RSc, RRc = book_signals(pc_, ic, cl[: len(ic)], dec[: T + 1], PFc, cal_end, cal_pair)
        okk = (np.array_equal(Sc, S[: T + 1], equal_nan=True) and np.allclose(Mc, Mo[: T + 1], rtol=0, atol=1e-12, equal_nan=True) and np.array_equal(RSc, RS[: T + 1], equal_nan=True)
               and (RRc == RR[: T + 1]).all())
        Wc = targets_all(Sc[K0:], Mc[K0:], RSc[K0:], RRc[K0:]); okk = okk and all(np.array_equal(Wc[c], W[c][: T + 1 - K0]) for c in W)
        if not okk:
            ck["cut_violations"].append(dec[T].date().isoformat())
    ck["weights_row_sum_max_dev"] = max(float(np.abs(w.sum(1) - 1).max()) for w in W.values())
    shifts = list(range(MIN_SHIFT, n_b - MIN_SHIFT + 1)); s1 = shifts[0]
    Wr = targets_all(np.roll(Sb, s1, 0), np.roll(Mb, s1, 0), np.roll(RSb, s1, 0), np.roll(RRb, s1, 0)); ck["roll_equivalence_failures"] = sum(int(not np.array_equal(Wr[c], np.roll(W[c], s1, 0))) for c in W)
    base = ((Sb == 1).sum(0), (RSb == 1).sum(0), [pd.Series(RRb[:, j]).value_counts().to_dict() for j in range(3)])
    for s_ in shifts:
        Sr, Rr, RRr = np.roll(Sb, s_, 0), np.roll(RSb, s_, 0), np.roll(RRb, s_, 0); ck["shift_count_checks"] += 1
        ck["shift_count_mismatch"] += int(not (np.array_equal((Sr == 1).sum(0), base[0]) and np.array_equal((Rr == 1).sum(0), base[1]) and [pd.Series(RRr[:, j]).value_counts().to_dict() for j in range(3)] == base[2]))
    bench = {}
    for name, wb in [("neutral", NEUTRAL), ("equal", np.full(4, 0.25))] + [(a, np.eye(4)[j]) for j, a in enumerate(A4)]:
        u = wb / (fi[kd[0] + 1] * (1 + C)); bench[name] = np.r_[1.0, cl[kd[0] + 1:] @ u]
    BST = {k: E10.nstats(v, days) for k, v in bench.items()}; best_asset = max(BST[a]["cagr"] for a in A4)
    ROWS, CURV, NT, NULLS = [], {}, {}, []; pos_w = idx.get_indexer(dec[K0:]) - kd[0]
    for code, fam, lab in STRATS:
        nav, sw = M.nbook(cl, fi, kd, W[code], NEUTRAL, rec=True); st = E10.nstats(nav, days)
        nt = [(s_,) + tuple(E10.quick(M.nbook(cl, fi, kd, np.roll(W[code], s_, 0), NEUTRAL)[0], d_days)) for s_ in shifts]; NT[code] = [x[1] for x in nt]
        NULLS += [{"code": code, "shift": x[0], "cagr": x[1], "maxdd": x[2]} for x in nt]
        ROWS.append({"code": code, "family": fam, "label": lab, "first": days[0].date().isoformat(), "last": days[-1].date().isoformat(), "cagr": st["cagr"], "vol": st["vol"], "maxdd": st["maxdd"],
                     "final": float(nav[-1]), "excess_neutral": st["cagr"] - BST["neutral"]["cagr"], "excess_equal": st["cagr"] - BST["equal"]["cagr"], "excess_spy": st["cagr"] - BST["SPY"]["cagr"],
                     "excess_best_asset": st["cagr"] - best_asset, "switches": len(sw), "switches_yr": len(sw) / yrs, **{f"w_{a}": float(W[code][:, j].mean()) for j, a in enumerate(A4)},
                     "all_in_one": float((W[code].max(1) > 0.99).mean()), "pct_T": E1.pctile(st["cagr"], NT[code]), "pct_T_dd": E1.pctile(st["maxdd"], [x[2] for x in nt]),
                     "null_T_median": float(np.median(NT[code])), "shifts": len(shifts)})
        CURV[code] = [r6(v) for v in nav[pos_w]]
    on, rel = Sb == 1, RSb == 1; lr = np.diff(np.log(cl[idx.get_indexer(dec[K0:])]), axis=0)
    pat = pd.Series([" · ".join(f"{a} {'BUY' if on[i, j] else 'SELL'}" for j, a in enumerate(A4)) for i in range(n_b)]).value_counts(normalize=True)
    sync = {"weeks": n_b, "first": dec[K0].date().isoformat(), "last": dec[-1].date().isoformat(), "buy": {a: float(on[:, j].mean()) for j, a in enumerate(A4)},
            "rel_buy": {x: float(rel[:, j].mean()) for j, x in enumerate(CM)}, "cm_all_agree": float((on[:, 1:].all(1) | ~on[:, 1:].any(1)).mean()),
            "cm_buy_count": [float((on[:, 1:].sum(1) == k).mean()) for k in range(4)], "patterns": {k: float(v) for k, v in pat.head(8).items()},
            "corr": np.corrcoef(lr.T).tolist(), "rr_share": {x: {s: float((RRb[:, j] == s).mean()) for s in ("A", "B", "N")} for j, x in enumerate(CM)}}
    return {"idx": idx, "dec": dec, "kd": kd, "days": days, "W": W, "rows": ROWS, "curves": CURV, "bench": {k: [r6(x) for x in v[pos_w]] for k, v in bench.items()}, "bst": BST, "nt": NT,
            "nulls": NULLS, "sync": sync, "shifts": shifts, "Sb": Sb, "RSb": RSb, "RRb": RRb, "Mb": Mb, "t": [d.date().isoformat() for d in dec[K0:]]}


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=500); ND = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_us_etf()
    e10 = pd.read_csv(os.path.join(ROOT, "explore", "E10_trend_rotation", "out", "strategies.csv"))
    ck = {"close_max_abs_diff": 0.0, "band_vs_z_max_abs_diff": 0.0, "engine_book_max_abs_diff": 0.0, "benchmarks_vs_engine_max_abs_diff": 0.0, "e10_cagr_max_abs_diff": 0.0, "e10_switch_mismatch": 0,
          "e10_pct_max_abs_diff": 0.0, "e10_missing": [], "null_draws": 0, "null_count_mismatch": 0, "st_missing_after_start": 0, "mom_missing_after_start": 0, "cut_checks": 0, "cut_violations": [],
          "weights_row_sum_max_dev": 0.0, "roll_equivalence_failures": 0, "shift_count_checks": 0, "shift_count_mismatch": 0}
    PR = {}
    for x in PAIR_ORDER:
        PR[x] = run_pair(px, x, ND, e10, ck); print(f"  pair {x} ÷ SPY  [{time.time() - t0:4.0f}s]", flush=True)
    BK = run_book(px, PR, ck); print(f"  book  [{time.time() - t0:4.0f}s]", flush=True)
    bad = (ck["close_max_abs_diff"] > 0 or ck["band_vs_z_max_abs_diff"] > 1e-9 or not ck["engine_book_max_abs_diff"] < 1e-9 or not ck["benchmarks_vs_engine_max_abs_diff"] < 1e-12
           or not ck["e10_cagr_max_abs_diff"] < 1e-12 or ck["e10_switch_mismatch"] or (ND == 500 and not ck["e10_pct_max_abs_diff"] < 1e-9) or ck["e10_missing"] or ck["null_count_mismatch"]
           or ck["st_missing_after_start"] or ck["mom_missing_after_start"] or ck["cut_violations"] or ck["weights_row_sum_max_dev"] > 1e-12 or ck["roll_equivalence_failures"] or ck["shift_count_mismatch"])
    if bad:
        sys.exit(f"checks FAILED: {ck}")
    PRS = pd.DataFrame([r for x in PAIR_ORDER for r in PR[x]["rows"]]); PRS.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    pd.DataFrame([PR[x]["stats"] for x in PAIR_ORDER]).to_csv(os.path.join(OUT, "pair_supertrend.csv"), index=False)
    SR = pd.DataFrame(BK["rows"]); SR.to_csv(os.path.join(OUT, "book_strategies.csv"), index=False); pd.DataFrame(BK["nulls"]).to_csv(os.path.join(OUT, "book_nulls.csv"), index=False)
    wl = [pd.DataFrame({"date": BK["t"], "code": c, "asset": a, "weight": W_[:, j]}) for c, W_ in BK["W"].items() for j, a in enumerate(A4)]; pd.concat(wl).to_csv(os.path.join(OUT, "book_weights_weekly.csv"), index=False)
    sig = pd.DataFrame({"date": BK["t"], **{f"st_{a}": BK["Sb"][:, j].astype(int) for j, a in enumerate(A4)}, **{f"rel_st_{x}": BK["RSb"][:, j].astype(int) for j, x in enumerate(CM)},
                        **{f"rule_r_{x}": BK["RRb"][:, j] for j, x in enumerate(CM)}, **{f"mom12_{a}": BK["Mb"][:, j] for j, a in enumerate(A4)}}); sig.to_csv(os.path.join(OUT, "book_signals_weekly.csv"), index=False)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "draws_per_pair_null": ND, "cost_bps_round_trip": COST, "common_start_decision_week": K0 + 1,
             "seeds": "Rule R N-A: default_rng([0, E10 pair id, 0, draw]) (E10's); Rule RS N-S: default_rng([0, E10 pair id, 5, draw]); E10 pair ids PDBC 3, COM 6, SDCI 8",
             "ohlcv": "ohlcv_eikon_us holds no open/high/low for SPY, PDBC, SDCI, COM before 2026 (22% of 2026 rows); us_ohlcv_eodhd holds OHLCV for SPY and PDBC to 2026-07-31, none for SDCI or COM; so every supertrend is on weekly bars from closes",
             "checks": ck, "book": {"first": BK["rows"][0]["first"], "last": BK["rows"][0]["last"], "shifts": len(BK["shifts"]), "shift_range_weeks": [BK["shifts"][0], BK["shifts"][-1]], "benchmarks": BK["bst"],
                                    "ahead_of_neutral": int((SR["excess_neutral"] > 0).sum()), "ahead_of_spy": int((SR["excess_spy"] > 0).sum()), "ahead_of_equal": int((SR["excess_equal"] > 0).sum()),
                                    "pct_T_high": int((SR["pct_T"] >= HI).sum()), "pct_T_low": int((SR["pct_T"] <= LO).sum()), "sync": BK["sync"]},
             "pairs": {x: PR[x]["stats"] for x in PAIR_ORDER}, "seconds": round(time.time() - t0)}
    json.dump(jclean(audit), open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)
    write_md(audit, PRS, SR, BK)
    write_page(audit, PRS, SR, BK, PR)
    print(f"done in {time.time() - t0:.0f}s; checks {ck}")


# ------------------------------------------------------------------------------------------------------- results (markdown)
def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")


def write_md(audit, PRS, SR, BK):
    c = audit["checks"]; b = BK["bst"]; bk = audit["book"]; sy = bk["sync"]
    md = ["# E12 results — SPY with PDBC, SDCI and COM: the supertrend on each and against SPY, a rotation book of the four, and each pair's ±1 sd band and 50/50 (EXPLORATORY)\n",
          f"_Generated by `explore/E12_cmdty_rotation/cmdty.py` on {audit['run_date']}. Total return (loader rule H4), trades at the next session's close (H3), 10.4 bps per side, weekly decisions. "
          "Description, not a finding; percentiles in **bold** are notable (≤ 5 or ≥ 95)._\n",
          f"**OHLCV:** {audit['ohlcv']}.\n",
          f"**Checks:** closes equal the loader's (max {c['close_max_abs_diff']:.1e}); the band's (ratio − mean) ÷ sd equals Rule R's z (max {c['band_vs_z_max_abs_diff']:.1e}); Rule R through the general book equals the engine's book (max NAV difference {c['engine_book_max_abs_diff']:.1e}) "
          f"and reproduces E10's R52 rows (CAGR difference {c['e10_cagr_max_abs_diff']:.1e}, switches {c['e10_switch_mismatch']} mismatches, N-A percentile difference {c['e10_pct_max_abs_diff']:.1e}); "
          f"{c['null_draws']:,} null draws kept their counts ({c['null_count_mismatch']} mismatches); the book's signals and targets pass the cut-history check ({c['cut_checks']} cuts, {len(c['cut_violations'])} violations); "
          f"weight rows sum to 1 (max deviation {c['weights_row_sum_max_dev']:.1e}); every shift kept each asset's BUY weeks and Rule R's state counts ({c['shift_count_checks']} shifts, {c['shift_count_mismatch']} mismatches).\n",
          "## Part 1 — each pair: the ±1 sd band (Rule R), the relative supertrend (Rule RS) and the 50/50\n",
          "| pair | rule | period | CAGR | 50/50 bought once | over 50/50 | X alone | SPY alone | max drawdown (50/50) | switches a year | weeks in X / SPY / 50/50 | null pct: CAGR / drawdown |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in PRS.itertuples():
        md.append(f"| {r.x} ÷ SPY | {r.label} | {r.first} → {r.last} | {fp(r.cagr)} | {fp(r.cagr_5050)} | {pp(r.excess_5050)} | {fp(r.cagr_x)} | {fp(r.cagr_spy)} | {fp(r.maxdd, 0)} ({fp(r.maxdd_5050, 0)}) | {r.switches_yr:.1f} | "
                  f"{pc(r.in_x)} / {pc(r.in_spy)} / {pc(r.in_5050)} | {r.null} {pq(r.pct)} / {pq(r.pct_dd)} |")
    md += ["", "Where each ratio stood against its band, and the supertrend against SPY (from each pair's start):\n",
           "| pair | weeks | below −1 sd / inside / above +1 sd | ETF on BUY | SPY on BUY | ratio (ETF ÷ SPY) on BUY | ratio flips a year | ratio BUY = (ETF BUY and SPY SELL) | Rule RS agrees with Rule R's holding (weeks Rule R is in a leg) |", "|---|---|---|---|---|---|---|---|---|"]
    for x, s in audit["pairs"].items():
        md.append(f"| {x} ÷ SPY | {s['weeks']} | {pc(s['below_band'])} / {pc(s['in_band_weeks'])} / {pc(s['above_band'])} | {pc(s['buy_x'])} | {pc(s['buy_spy'])} | {pc(s['buy_rel'])} | {s['rel_flips_yr']:.1f} | {pc(s['rel_vs_x_minus_spy'])} | {pc(s['rel_agrees_rule_r'])} |")
    md += [f"\n## Part 2 — the four-asset book (SPY, PDBC, SDCI, COM), {bk['first']} → {bk['last']}\n",
           f"Neutral (SPY 50%, each commodity ETF 16.7%) bought once {fp(b['neutral']['cagr'])} a year (max drawdown {fp(b['neutral']['maxdd'], 0)}); 25% each {fp(b['equal']['cagr'])} ({fp(b['equal']['maxdd'], 0)}); "
           + "; ".join(f"{a} {fp(b[a]['cagr'])} ({fp(b[a]['maxdd'], 0)})" for a in A4) + ".\n",
           f"Supertrend (absolute) on BUY: " + ", ".join(f"{a} {pc(v)}" for a, v in sy["buy"].items()) + f"; the three commodity ETFs agree in {pc(sy['cm_all_agree'])} of weeks; relative supertrend (ETF ÷ SPY) on BUY: "
           + ", ".join(f"{x} {pc(v)}" for x, v in sy["rel_buy"].items()) + ".\n",
           "| strategy | family | CAGR | over neutral | over 25% each | over SPY | volatility | max drawdown | switches a year | average weight SPY / PDBC / SDCI / COM | weeks all in one asset | N-T pct: CAGR / drawdown |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in SR.itertuples():
        md.append(f"| {r.code} {r.label} | {FAMILIES[r.family]} | {fp(r.cagr)} | {pp(r.excess_neutral)} | {pp(r.excess_equal)} | {pp(r.excess_spy)} | {pc(r.vol)} | {fp(r.maxdd, 0)} | {r.switches_yr:.1f} | "
                  f"{pc(r.w_SPY)} / {pc(r.w_PDBC)} / {pc(r.w_SDCI)} / {pc(r.w_COM)} | {pc(r.all_in_one)} | {pq(r.pct_T)} / {pq(r.pct_T_dd)} |")
    md += ["", f"The {len(SR)} strategies: ahead of neutral {bk['ahead_of_neutral']}, of 25% each {bk['ahead_of_equal']}, of SPY {bk['ahead_of_spy']}; N-T notable high in {bk['pct_T_high']}, low in {bk['pct_T_low']} "
           f"({bk['shifts']} shifts of {bk['shift_range_weeks'][0]}–{bk['shift_range_weeks'][1]} weeks)."]
    open(os.path.join(OUT, "E12_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


# ------------------------------------------------------------------------------------------------------- the page
def write_page(audit, PRS, SR, BK, PR):
    pairs = {}
    for x in PAIR_ORDER:
        rows = {r["rule"]: {k: (r6(v) if isinstance(v, float) else v) for k, v in r.items()} for r in PRS[PRS["x"] == x].to_dict("records")}
        pairs[x] = {"x": x, "name": NAME[x], "rows": rows, "stats": audit["pairs"][x], **PR[x]["ser"]}
    STJ = {r["code"]: {k: (r6(v) if isinstance(v, float) else v) for k, v in r.items()} for r in SR.to_dict("records")}
    data = {"pairs": pairs, "order": PAIR_ORDER, "assets": A4, "cm": CM, "names": NAME, "strats": [[c, f, l] for c, f, l in STRATS], "families": FAMILIES, "st": STJ, "t": BK["t"], "curves": BK["curves"],
            "bench": BK["bench"], "bst": BK["bst"], "neutral": NEUTRAL, "w": {c: np.round(W, 4) for c, W in BK["W"].items()}, "nt": BK["nt"], "sync": BK["sync"], "book": audit["book"],
            "S": BK["Sb"].astype(int), "RS": BK["RSb"].astype(int), "RR": BK["RRb"], "lo": LO, "hi": HI}
    blob = json.dumps(jclean(data), separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")
    psel = "<label>Pair <select id='pair'>" + "".join(f"<option value='{x}'>{x} ÷ SPY</option>" for x in PAIR_ORDER) + "</select></label>"
    ssel = "<label>Strategy <select id='strat'>" + "".join(f"<optgroup label='{html.escape(FAMILIES[f], quote=True)}'>" + "".join(f"<option value='{c}'>{html.escape(c + ' · ' + l)}</option>" for c, ff, l in STRATS if ff == f)
                                                           + "</optgroup>" for f in FAMILIES) + "</select></label>"
    b = BK["bst"]; bk = audit["book"]; best = SR.loc[SR["cagr"].idxmax()]; low = SR.loc[SR["maxdd"].idxmax()]
    key = "<p style='margin:4px 0'><b>The pairs</b> (each from its own start): " + "; ".join(
        f"{r.x} ÷ SPY ({r.first} → {r.last}) Rule R {fp(r.cagr)} against 50/50 {fp(r.cagr_5050)} (N-A percentile {r.pct:.1f})" for r in PRS[PRS["rule"] == "R52"].itertuples()) + ". Rule RS (the relative supertrend): " + "; ".join(
        f"{r.x} {fp(r.cagr)} (N-S {r.pct:.1f})" for r in PRS[PRS["rule"] == "RS"].itertuples()) + ".</p>"
    key += (f"<p style='margin:4px 0'><b>The four-asset book</b> ({bk['first']} → {bk['last']}): neutral {fp(b['neutral']['cagr'])} a year (max drawdown {fp(b['neutral']['maxdd'], 0)}), SPY {fp(b['SPY']['cagr'])} ({fp(b['SPY']['maxdd'], 0)}). "
            f"Ahead of neutral {bk['ahead_of_neutral']} of {len(SR)} strategies, of SPY {bk['ahead_of_spy']}; notable against the timing null: {bk['pct_T_high']} high, {bk['pct_T_low']} low. "
            f"Highest CAGR {best['code']} {fp(best['cagr'])} (max drawdown {fp(best['maxdd'], 0)}, timing percentile {best['pct_T']:.1f}); shallowest drawdown {low['code']} {fp(low['maxdd'], 0)} ({fp(low['cagr'])} a year).</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + M.CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Commodity rotation</title>"),
         "<header class='bar'><div class='bar-inner'><b>E12 · SPY with PDBC, SDCI, COM</b><span class='tag'>EXPLORATORY</span>", psel, ssel, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>SPY with three commodity ETFs: the supertrend on each and against SPY, a rotation book of the four, and each pair's ±1 sd band and 50/50</h1>",
         f"<p>PDBC, SDCI and COM (broad commodity ETFs) against SPY, total return, in dollars. Weekly decisions, trades at the next session's close, 10.4 bps each way. Generated {audit['run_date']}. "
         "<b>Description, not a finding</b>: these series have been studied before (E08–E11).</p>",
         f"<div class='card'><p class='note' style='margin:4px 0'><b>OHLCV.</b> {html.escape(audit['ohlcv'])}.</p>" + key + "</div>",
         "<h2>1 · Each pair: the ±1 sd band and the 50/50</h2><p class='note' id='n_pair'></p>",
         "<div class='card'><div id='g_band' class='plot'></div></div><div class='card'><div id='g_peq' class='plot'></div><div id='g_pdd' class='plot'></div></div>",
         "<div class='card'><div id='t_pair'></div></div><div class='card'><div id='g_pst' class='plot'></div><div id='g_pnull' class='plot'></div></div>",
         "<h2>2 · The four-asset rotation book</h2><p class='note' id='n_book'></p>",
         "<div class='card'><div id='g_map' class='plot'></div><div id='t_sync'></div></div><div class='card'><div id='g_bars' class='plot'></div></div><div class='card'><div id='t_book'></div></div>",
         "<h2>3 · The chosen strategy</h2>",
         "<div class='card'><div id='g_beq' class='plot'></div><div id='g_bdd' class='plot'></div></div><div class='card'><div id='g_bw' class='plot'></div></div><div class='card'><div id='g_bnull' class='plot'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Data.</b> analysis/approved_data.load_us_etf(): SPY, PDBC, SDCI, COM, total return (H4), each from its start under rule H5 (SPY 2010-01-04, PDBC 2016-01-04, COM 2019-01-02, SDCI 2020-01-02), to 2026-11-18. "
         "The research database holds no open/high/low for these funds before 2026 in the approved source, so every supertrend uses weekly bars from closes.</li>"
         "<li><b>Rule R (±1 sd).</b> T03's rule on the plain ratio ETF ÷ SPY: the band is the trailing 52-week mean ± 1 sd; first week below the band → hold the ETF until the ratio is back at the mean; above → hold SPY; otherwise 50/50. "
         "Through the engine's book; reproduces E10's numbers. 50/50 = half in each, bought once and never rebalanced.</li>"
         "<li><b>Supertrend.</b> The desk's Weekly LW supertrend (5-bar linear-weighted ATR × 2; E10's functions) on each fund's own weekly bars (absolute) and on weekly bars of the daily ratio ETF ÷ SPY (relative). Rule RS holds the ETF while the ratio is BUY, SPY while SELL.</li>"
         "<li><b>Book.</b> E11's book over the four funds on their common calendar from 2020-01-02; every strategy starts on the 53rd decision week at neutral (SPY 50%, each ETF 16.7%), always invested. Classes: equity = SPY, commodity = the three ETFs.</li>"
         "<li><b>Nulls.</b> Pairs: Rule R against N-A (500 draws, E10's seeds), Rule RS against N-S (its spells in a random order). Book: N-T, every circular shift of the weekly signal panel (26 weeks to the length less 26), all signals together.</li>"
         "<li><b>Limits.</b> The book has about 5.9 years and misses 2020; nine strategies on seen data, some may look good by luck; closes only; no cash. Design written before the run: explore/E12_cmdty_rotation/E12.md.</li></ul></div>",
         "</main>", f"<script id='e12-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E12_cmdty.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e12-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7', band:'rgba(42,120,214,0.10)'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835', band:'rgba(57,135,229,0.16)'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function ACOL(){return isDark()?['#3987e5','#d95926','#199e70','#c98500']:['#2a78d6','#eb6834','#1baf7a','#eda100'];}
function SIG(){return isDark()?{buy:'#3987e5',sell:'#e66767'}:{buy:'#2a78d6',sell:'#e34948'};}
function FCOL(){return isDark()?{ST:'#3987e5',STMO:'#d95926',GR:'#199e70',MO:'#c98500',RS:'#9c6ade',RR:'#898781'}:{ST:'#2a78d6',STMO:'#eb6834',GR:'#1baf7a',MO:'#eda100',RS:'#8a57d2',RR:'#898781'};}
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
function lab(code){const s=D.strats.find(x=>x[0]===code);return s?s[0]+' · '+s[2]:code;}
function drawPair(x){
  const c=th(), K=SIG(), AC=ACOL(), P=D.pairs[x], B=P.band, R=P.rows, Rr=R.R52, Rs=R.RS, st=P.stats;
  document.getElementById('n_pair').innerHTML='The plain ratio '+x+' ÷ SPY at each decision week (total return), its trailing 52-week mean and mean ± 1 sd. Rule R (T03\'s rule) buys the leg that has lagged at the first week outside the band and goes back to 50/50 when the ratio is back at the mean. '+
    'The strategy period ('+Rr.first+' → '+Rr.last+') starts at the dashed line; before it the band is shown for context.';
  const t=B.t, sh=[{type:'line',xref:'x',yref:'paper',x0:B.start,x1:B.start,y0:0,y1:1,line:{color:c.muted,width:1,dash:'dash'}}];
  const tr=[{x:t,y:B.up,type:'scatter',mode:'lines',name:'mean + 1 sd',line:{color:c.muted,width:1},hovertemplate:'mean + 1 sd: %{y:.4f}<extra></extra>'},
    {x:t,y:B.lo,type:'scatter',mode:'lines',name:'mean − 1 sd',line:{color:c.muted,width:1},fill:'tonexty',fillcolor:c.band,hovertemplate:'mean − 1 sd: %{y:.4f}<extra></extra>'},
    {x:t,y:B.mean,type:'scatter',mode:'lines',name:'52-week mean',line:{color:c.muted,width:1.2,dash:'dot'},hovertemplate:'mean: %{y:.4f}<extra></extra>'},
    {x:t,y:B.ratio,type:'scatter',mode:'lines',name:x+' ÷ SPY',line:{color:c.ink,width:1.8},customdata:B.z,hovertemplate:'ratio %{y:.4f} (z %{customdata})<extra></extra>'}];
  const mk={x:{sym:'triangle-up',col:AC[1],nm:'enter: hold '+x},spy:{sym:'triangle-down',col:AC[0],nm:'enter: hold SPY'},'5050':{sym:'circle-open',col:c.ink2,nm:'exit: back to 50/50'}};
  Object.keys(mk).forEach(k=>{const T=B.trades.filter(q=>q.to===k);if(!T.length)return;tr.push({x:T.map(q=>q.t),y:T.map(q=>q.ratio),type:'scatter',mode:'markers',name:mk[k].nm,marker:{symbol:mk[k].sym,size:11,color:mk[k].col,line:{width:1.5,color:mk[k].col}},
    customdata:T.map(q=>q.fill),hovertemplate:mk[k].nm+' · decided %{x}, traded %{customdata}<extra></extra>'});});
  Plotly.react('g_band',tr,layout(two(x+' ÷ SPY · the ratio, its 52-week mean and the ±1 sd band · '+span(t),'Rule R\'s trades from '+B.start+': '+B.trades.length+' switches; the ratio was below the band '+pc(st.below_band)+', inside '+pc(st.in_band_weeks)+', above '+pc(st.above_band)+' of the strategy\'s weeks'),520,
    {shapes:sh,hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({title:{text:x+' ÷ SPY',font:{color:c.ink2}}})}),CFG);
  const C=P.curves, L=[[C.R52,'Rule R (±1 sd)',c.ink,2.4,'solid'],[C.RS,'Rule RS (relative supertrend)','#8a57d2',1.8,'solid'],[C.bh,'50/50 bought once',c.muted,1.8,'dash'],[C.x,x+' alone',AC[1],1.1,'solid'],[C.spy,'SPY alone',AC[0],1.1,'solid']];
  Plotly.react('g_peq',L.map(([y,nm,col,wd,dash])=>({x:C.t,y:y,type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(x+' ÷ SPY · value of 1 · '+span(C.t),'after costs; log scale · Rule R '+fp(Rr.cagr)+', Rule RS '+fp(Rs.cagr)+', 50/50 '+fp(Rr.cagr_5050)+', '+x+' '+fp(Rr.cagr_x)+', SPY '+fp(Rr.cagr_spy)+' a year'),460,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_pdd',L.slice(0,3).map(([y,nm,col,wd,dash])=>({x:C.t,y:ddOf(y),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · '+span(C.t),'Rule R, Rule RS and the 50/50'),280,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
  document.getElementById('t_pair').innerHTML=table(['rule','CAGR','over 50/50','volatility','max drawdown','switches a year','weeks in '+x+' / SPY / 50/50','null pct: CAGR / drawdown'],
    [Rr,Rs].map(q=>[q.label,fp(q.cagr),fpp(q.excess_5050),pc(q.vol),fp(q.maxdd,0),f1(q.switches_yr),pc(q.in_x)+' / '+pc(q.in_spy)+' / '+pc(q.in_5050),q.null+' '+pq(q.pct)+' / '+pq(q.pct_dd)]))+
    "<p class='note'>Over the same days: 50/50 bought once "+fp(Rr.cagr_5050)+' (max drawdown '+fp(Rr.maxdd_5050,0)+'), '+x+' '+fp(Rr.cagr_x)+' ('+fp(Rr.maxdd_x,0)+'), SPY '+fp(Rr.cagr_spy)+' ('+fp(Rr.maxdd_spy,0)+'). Nulls: N-A = Rule R\'s trades in a random order with the same counts (500 draws, E10\'s seeds); '+
    'N-S = Rule RS\'s BUY and SELL spells in a random order (500 draws). The relative supertrend was BUY '+pc(st.buy_rel)+' of weeks and flipped '+f1(st.rel_flips_yr)+' times a year; it agreed with Rule R\'s holding in '+pc(st.rel_agrees_rule_r)+' of the weeks Rule R held one leg.</p>';
  const S=P.states, rowsS=[[S.x,x+' (absolute)'],[S.spy,'SPY (absolute)'],[S.rel,x+' ÷ SPY (relative)']];
  Plotly.react('g_pst',[{type:'heatmap',x:S.t,y:rowsS.map(r=>r[1]),z:rowsS.map(r=>r[0]),zmin:-1,zmax:1,colorscale:[[0,K.sell],[0.5,K.sell],[0.5,K.buy],[1,K.buy]],showscale:false,ygap:3,
    customdata:rowsS.map(r=>r[0].map(v=>v===1?'BUY':'SELL')),hovertemplate:'%{y} · %{x}: %{customdata}<extra></extra>'}],
    layout(two(x+' ÷ SPY · the desk\'s weekly supertrend · '+span(S.t),'blue = BUY, red = SELL · weekly bars from closes · on BUY: '+x+' '+pc(st.buy_x)+', SPY '+pc(st.buy_spy)+', the ratio '+pc(st.buy_rel)),260,{margin:{l:150,r:24,t:90,b:40},xaxis:ax({type:'date'}),yaxis:ax({autorange:'reversed',showgrid:false})}),CFG);
  const nn=P.null;
  Plotly.react('g_pnull',[{type:'histogram',x:nn.R52.map(v=>v*100),name:'N-A: Rule R\'s trades in a random order',marker:{color:isDark()?'rgba(195,194,183,0.55)':'rgba(82,81,78,0.45)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} draws<extra></extra>'},
    {type:'histogram',x:nn.RS.map(v=>v*100),name:'N-S: Rule RS\'s spells in a random order',marker:{color:isDark()?'rgba(156,106,222,0.5)':'rgba(138,87,210,0.4)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} draws<extra></extra>'}],
    layout(two(x+' ÷ SPY · the real CAGR among its nulls','lines: Rule R '+fp(Rr.cagr)+' (percentile '+f1(Rr.pct)+') and Rule RS '+fp(Rs.cagr)+' (percentile '+f1(Rs.pct)+')'),340,
    {barmode:'overlay',shapes:[{type:'line',xref:'x',yref:'paper',x0:Rr.cagr*100,x1:Rr.cagr*100,y0:0,y1:1,line:{color:c.ink,width:2}},{type:'line',xref:'x',yref:'paper',x0:Rs.cagr*100,x1:Rs.cagr*100,y0:0,y1:1,line:{color:'#8a57d2',width:2,dash:'dash'}}],
     xaxis:ax({ticksuffix:'%',title:{text:'CAGR',font:{color:c.ink2}}}),yaxis:ax({title:{text:'draws',font:{color:c.ink2}}})}),CFG);
}
function drawBook(code){
  const c=th(), K=SIG(), F=FCOL(), AC=ACOL(), S=D.strats, Y=D.sync, bk=D.book, t=D.t;
  document.getElementById('n_book').innerHTML='SPY, PDBC, SDCI and COM on their common calendar; every strategy starts on '+bk.first+' at neutral (SPY 50%, each commodity ETF 16.7%) and is always invested. Excess = CAGR − the benchmark\'s CAGR over the same days ('+bk.first+' → '+bk.last+'). '+
    'Percentile: the real result among the signals slid in time by every shift of '+bk.shift_range_weeks[0]+'–'+bk.shift_range_weeks[1]+' weeks ('+bk.shifts+' shifts, N-T). ≤ 5 or ≥ 95 = notable.';
  const rows=D.assets.map((a,j)=>[D.S.map(r=>r[j]),a+' (absolute)']).concat(D.cm.map((x,j)=>[D.RS.map(r=>r[j]),x+' ÷ SPY (relative)']));
  Plotly.react('g_map',[{type:'heatmap',x:t,y:rows.map(r=>r[1]),z:rows.map(r=>r[0]),zmin:-1,zmax:1,colorscale:[[0,K.sell],[0.5,K.sell],[0.5,K.buy],[1,K.buy]],showscale:false,ygap:3,
    customdata:rows.map(r=>r[0].map(v=>v===1?'BUY':'SELL')),hovertemplate:'%{y} · %{x}: %{customdata}<extra></extra>'}],
    layout(two('The four assets\' weekly supertrend, absolute and relative to SPY · '+span(t),'blue = BUY, red = SELL · the three commodity ETFs agree in '+pc(Y.cm_all_agree)+' of weeks'),360,{margin:{l:170,r:24,t:90,b:40},xaxis:ax({type:'date'}),yaxis:ax({autorange:'reversed',showgrid:false})}),CFG);
  const P=Object.entries(Y.patterns);
  document.getElementById('t_sync').innerHTML="<p class='note'>On BUY (absolute): "+D.assets.map(a=>a+' '+pc(Y.buy[a])).join(', ')+'; relative to SPY: '+D.cm.map(x=>x+' '+pc(Y.rel_buy[x])).join(', ')+'. Rule R holding the ETF / SPY / 50/50: '+
    D.cm.map(x=>x+' '+pc(Y.rr_share[x].A)+' / '+pc(Y.rr_share[x].B)+' / '+pc(Y.rr_share[x].N)).join('; ')+'.</p>'+table(['most common absolute states','share of weeks'],P.map(([k,v])=>[k,pc(v)]))+
    table(['weekly return correlation'].concat(D.assets),D.assets.map((a,i)=>[a].concat(D.assets.map((b,j)=>(Y.corr[i][j]>=0?'+':'')+Y.corr[i][j].toFixed(2)))));
  Plotly.react('g_bars',[{type:'bar',x:S.map(s=>s[0]),y:S.map(s=>D.st[s[0]].excess_neutral),marker:{color:S.map(s=>F[s[1]]),line:{color:S.map(s=>s[0]===code?c.ink:'rgba(0,0,0,0)'),width:2}},
    text:S.map(s=>fpp(D.st[s[0]].excess_neutral)+((D.st[s[0]].pct_T<=D.lo||D.st[s[0]].pct_T>=D.hi)?'*':'')),textposition:'outside',cliponaxis:false,
    customdata:S.map(s=>lab(s[0])+' ('+D.families[s[1]]+')<br>CAGR '+fp(D.st[s[0]].cagr)+', max drawdown '+fp(D.st[s[0]].maxdd,0)+'<br>timing percentile '+f1(D.st[s[0]].pct_T)),hovertemplate:'%{customdata}<extra></extra>'}],
    layout(two('Every strategy · CAGR over neutral bought once (pp a year) · '+bk.first+' → '+bk.last,'colour = family (blue absolute supertrend, orange + momentum pick, aqua gate and rank, yellow momentum only, purple relative supertrend, grey Rule R); * = notable against the timing null; outlined = chosen'),420,
    {showlegend:false,margin:{l:60,r:24,t:110,b:80},xaxis:ax({tickangle:-30}),yaxis:ax({tickformat:'.1%',zeroline:true,zerolinecolor:c.axis,title:{text:'over neutral',font:{color:c.ink2}}})}),CFG);
  const Bs=D.bst;
  document.getElementById('t_book').innerHTML=table(['strategy','CAGR','over neutral','over 25% each','over SPY','volatility','max drawdown','switches a year','average weight '+D.assets.join(' / '),'weeks all in one asset','N-T pct: CAGR / drawdown'],
    S.map(s=>{const q=D.st[s[0]];return [(s[0]===code?'<b>':'')+lab(s[0])+(s[0]===code?'</b>':''),fp(q.cagr),fpp(q.excess_neutral),fpp(q.excess_equal),fpp(q.excess_spy),pc(q.vol),fp(q.maxdd,0),f1(q.switches_yr),
      D.assets.map(a=>pc(q['w_'+a])).join(' / '),pc(q.all_in_one),pq(q.pct_T)+' / '+pq(q.pct_T_dd)];}))+
    "<p class='note'>Benchmarks over the same days: neutral bought once "+fp(Bs.neutral.cagr)+' (max drawdown '+fp(Bs.neutral.maxdd,0)+'), 25% each '+fp(Bs.equal.cagr)+' ('+fp(Bs.equal.maxdd,0)+'), '+D.assets.map(a=>a+' '+fp(Bs[a].cagr)+' ('+fp(Bs[a].maxdd,0)+')').join(', ')+
    '. Ahead of neutral: '+bk.ahead_of_neutral+' of '+S.length+'; of SPY: '+bk.ahead_of_spy+'.</p>';
  const q=D.st[code], Bn=D.bench, lines=[[D.curves[code],lab(code),c.ink,2.6,'solid'],[Bn.neutral,'neutral bought once',c.muted,1.6,'dash'],[Bn.equal,'25% each bought once',c.muted,1.4,'dot']].concat(D.assets.map((a,j)=>[Bn[a],a+' bought once',AC[j],1.1,'solid']));
  Plotly.react('g_beq',lines.map(([y,nm,col,wd,dash])=>({x:t,y:y,type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(lab(code)+' · value of 1 · '+span(t),'after costs; weekly decision closes; log scale · CAGR '+fp(q.cagr)+' against neutral '+fp(Bs.neutral.cagr)+' and SPY '+fp(Bs.SPY.cagr)),480,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_bdd',lines.slice(0,3).map(([y,nm,col,wd,dash])=>({x:t,y:ddOf(y),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · '+span(t),'the strategy, neutral and 25% each'),300,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
  const W=D.w[code];
  Plotly.react('g_bw',D.assets.map((a,j)=>({x:t,y:W.map(r=>r[j]),type:'scatter',mode:'lines',name:a,stackgroup:'one',line:{width:0.5,color:AC[j],shape:'hv'},fillcolor:AC[j],hovertemplate:a+': %{y:.0%}<extra></extra>'})),
    layout(two(lab(code)+' · target weights each week · '+span(t),'average: '+D.assets.map(a=>a+' '+pc(q['w_'+a])).join(', ')),360,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({tickformat:'.0%',range:[0,1]})}),CFG);
  Plotly.react('g_bnull',[{type:'histogram',x:D.nt[code].map(v=>v*100),name:'N-T: signals slid in time',marker:{color:isDark()?'rgba(195,194,183,0.55)':'rgba(82,81,78,0.45)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} shifts<extra></extra>'}],
    layout(two(lab(code)+' · its CAGR among the slid signals · '+bk.shifts+' shifts','the line = the real CAGR '+fp(q.cagr)+' · percentile '+f1(q.pct_T)),340,
    {shapes:[{type:'line',xref:'x',yref:'paper',x0:q.cagr*100,x1:q.cagr*100,y0:0,y1:1,line:{color:c.ink,width:2}}],showlegend:false,xaxis:ax({ticksuffix:'%',title:{text:'CAGR',font:{color:c.ink2}}}),yaxis:ax({title:{text:'shifts',font:{color:c.ink2}}})}),CFG);
}
function render(){drawPair(document.getElementById('pair').value);drawBook(document.getElementById('strat').value);}
document.getElementById('pair').addEventListener('change',render);document.getElementById('strat').addEventListener('change',render);
const tb=document.getElementById('theme');
function setTb(){tb.textContent=isDark()?'Light':'Dark';}
tb.addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';setTb();render();});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(!document.documentElement.dataset.theme){setTb();render();}});
setTb();render();
"""


if __name__ == "__main__":
    main()
