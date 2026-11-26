"""
E11 (EXPLORATORY): multi-asset books, kept apart — India (N100, MC100, SC100, MCX gold) and the US (SPY, IJR, GCC,
GLD). The desk's weekly supertrend decides which asset classes are on; a class's money is spread equally, by breadth,
with a tilt, or by a momentum pick; beside them gate-and-rank and momentum-only rotations, and E10's two-asset rules as
references. Design: explore/E11_multi_asset/E11.md (each part written before its run).

    python explore/E11_multi_asset/multi.py [--max-shifts N]

Every number on the page and in E11_results.md is computed here; the page's script only picks the numbers of the chosen
controls from the embedded results and draws them. Not a finding and not a test of any rule.

Re-run on 2026-11-26 on the latest data (explore/E01_ratio_cycles/latest.py; E11.md, the block of that date): the Indian
book on N100, MC100, SC100 and MCX gold from load_india_latest(), 2011-01-01 to 2026-11-20; in it every strategy holds
the liquid ETF instead of gold in any week whose decision falls in gold's weekly supertrend SELL. The US book is
unchanged. The first run is kept in earlier_run_2026-11-25/.
"""
import argparse
import html
import itertools
import json
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
for p_ in (ROOT, os.path.join(ROOT, "explore", "E01_ratio_cycles"), os.path.join(ROOT, "explore", "E10_trend_rotation")):
    sys.path.insert(0, p_)
from analysis.approved_data import GOLD, fingerprints, load_us_etf, load_us_gold, load_us_small  # noqa: E402
import analysis.ratio_engine as RE  # noqa: E402  (its pair start, set per book)
from analysis.ratio_engine import PIT, book, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (page styling, percentile, notable)
import variants as VR  # noqa: E402  (Rule R's states, for the book check)
import rotation as E10  # noqa: E402  (the desk's supertrend, checked in E10 against the desk's own code)
import latest as LT  # noqa: E402  (the latest Indian data and the gold → liquid switch)
START0 = dict(RE.START); SWI = None   # the engine's own pair start (the US book keeps it); the switch, set in main()

BOOKS = {
    "india": {"label": "India", "unit": "rupees", "assets": ["N100", "MC100", "SC100", GOLD], "lab": {"N100": "N100", "MC100": "MC100", "SC100": "SC100", GOLD: "gold", "Liquid": "liquid"},
              "name": {"N100": "N100 (large caps)", "MC100": "MC100 (mid caps)", "SC100": "SC100 (small caps)", GOLD: "gold (MCX futures)", "Liquid": "the liquid ETF (gold's weight on SELL weeks)"},
              "classes": (("equity", ("N100", "MC100", "SC100"), 0.5), ("gold", (GOLD,), 0.5)), "start": "2011-01-01", "closes_only": False, "switch": True,   # the latest data (was 2015-01-12)
              "ref": ("SC100", GOLD), "ref_label": "SC100 + gold only (E10's S2)", "pair": f"SC100|{GOLD}"},
    "us": {"label": "US", "unit": "dollars", "assets": ["SPY", "IJR", "GCC", "GLD"], "lab": {"SPY": "SPY", "IJR": "IJR", "GCC": "GCC", "GLD": "GLD"},
           "name": {"SPY": "SPY (US large caps)", "IJR": "IJR (US small caps)", "GCC": "GCC (broad commodities)", "GLD": "GLD (gold)"},
           "classes": (("equity", ("SPY", "IJR"), 0.5), ("commodity", ("GCC",), 0.25), ("gold", ("GLD",), 0.25)), "start": "2010-01-04", "closes_only": True,
           "ref": ("IJR", "GLD"), "ref_label": "IJR + GLD only (E10's S2)", "pair": "IJR|GLD"},
}
CHECK_BOOK = {**BOOKS["us"], "label": "US with PDBC", "assets": ["SPY", "IJR", "PDBC", "GLD"], "lab": {"SPY": "SPY", "IJR": "IJR", "PDBC": "PDBC", "GLD": "GLD"},
              "name": {"SPY": "SPY", "IJR": "IJR", "PDBC": "PDBC (broad commodities)", "GLD": "GLD"}, "classes": (("equity", ("SPY", "IJR"), 0.5), ("commodity", ("PDBC",), 0.25), ("gold", ("GLD",), 0.25)),
              "start": "2016-01-04", "ref": None}
K0, MIN_SHIFT = 52, 26                   # common start = the 53rd decision week; nulls slide the signals by >= 26 weeks
COST = VR.COST; C = COST / 2 / 1e4; LO, HI = E1.LO, E1.HI
LOOKS = (("12_1", "12-1", 52, 4), ("12", "12-month", 52, 0), ("6", "6-month", 26, 0))
FAMILIES = {"ST": "supertrend only", "STMO": "supertrend + momentum pick", "GR": "gate and rank", "MO": "momentum only", "REF": "reference: two assets"}
STRATS = [("ST-EQ", "ST", "supertrend, equal split", None), ("ST-BR", "ST", "supertrend, breadth", None), ("ST-TL", "ST", "supertrend, tilt", None)]
STRATS += [(f"ST-MO-{k.replace('_', '-')}", "STMO", f"supertrend + momentum pick ({lab})", k) for k, lab, _, _ in LOOKS]
STRATS += [(f"GR-{k.replace('_', '-')}", "GR", f"gate and rank ({lab})", k) for k, lab, _, _ in LOOKS]
STRATS += [(f"MO-{k.replace('_', '-')}", "MO", f"momentum only, top 1 of 4 ({lab})", k) for k, lab, _, _ in LOOKS]
REF_CODE = "REF-S2"
DEFAULT = {"book": "india", "strat": "ST-MO-12"}


def r6(v):
    return None if v is None or v != v else round(float(v), 6)


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return r6(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------------- frame and signals
def multi_frame(px: dict, assets: list, start: pd.Timestamp):
    """The assets' common calendar from start: closes and fills (n × assets) and the decision days (the last day of each ISO week)."""
    idx = px[assets[0]].index
    for a in assets[1:]:
        idx = idx.intersection(px[a].index)
    idx = idx[idx >= start]
    cl = np.column_stack([px[a]["close"].reindex(idx).to_numpy(float) for a in assets]); fi = np.column_stack([px[a]["fill"].reindex(idx).to_numpy(float) for a in assets])
    iso = idx.isocalendar(); key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    return idx, cl, fi, idx[~pd.Series(key).duplicated(keep="last").to_numpy()]


def signals(px_src: dict, assets: list, idx, cl, dec, cal_end: dict, closes_only: bool):
    """Supertrend states (±1, n_dec × assets) from each asset's own weekly bars, and momentum (look → n_dec × assets) on the
    frame's decision-week closes."""
    S = np.full((len(dec), len(assets)), np.nan)
    for j, a in enumerate(assets):
        b = E10.desk_bars(px_src[a], closes_only); S[:, j] = E10.leg_state_at(dec, b, E10.desk_supertrend(b)[0], cal_end[a])
    Wc = cl[idx.get_indexer(dec)]; n = len(Wc); M = {}
    for key, _, look, skip in LOOKS:
        m = np.full(Wc.shape, np.nan)
        if n > look:
            m[look:] = Wc[look - skip:n - skip] / Wc[:n - look] - 1
        M[key] = m
    return S, M


def onehot(i: np.ndarray, n: int, width: int) -> np.ndarray:
    W = np.zeros((n, width)); W[np.arange(n), i] = 1.0
    return W


def targets(code: str, S: np.ndarray, M, cls: list, neutral: np.ndarray) -> np.ndarray:
    """Target weights (rows sum to 1) from the supertrend states S (n × assets) and the momentum M (n × assets) of the
    strategy's lookback; cls = [(class, member columns, base weight)]. A row depends only on that week's S and M
    (E11.md: the strategies, and the rules written for any number of classes)."""
    n, na = S.shape; on = S == 1
    if code.startswith("GR"):
        W = np.tile(neutral, (n, 1)); any_ = on.any(1); W[any_] = onehot(np.argmax(np.where(on, M, -np.inf), axis=1), n, na)[any_]
        return W
    if code.startswith("MO"):
        return onehot(np.argmax(M, axis=1), n, na)
    nc = len(cls); B = np.array([b for _, _, b in cls]); Sg = np.zeros((n, nc)); D = []
    for c, (_, m, _) in enumerate(cls):
        onm = on[:, m]; cnt = onm.sum(1)
        if code == "ST-BR":
            Sg[:, c] = cnt / len(m); D.append(np.full((n, len(m)), 1.0 / len(m)))
        else:
            Sg[:, c] = (cnt > 0).astype(float)
            if code.startswith("ST-MO") and len(m) > 1:
                D.append(onehot(np.argmax(np.where(onm, M[:, m], -np.inf), axis=1), n, len(m)))
            else:
                D.append(onm / np.maximum(cnt, 1)[:, None])
    off = Sg == 0
    if code == "ST-TL":
        kept = B * 0.5 * off; freed = B * 0.5 * off
    else:
        kept = np.zeros((n, nc)); freed = B * (1 - Sg)
    T = B * Sg + kept
    for c in range(nc):   # a class's freed money: to the other classes that are on, pro rata to their base weights
        rec = Sg > 0; rec[:, c] = False; wts = rec * B; tot = wts.sum(1); has = tot > 0
        T[has] += freed[has, c][:, None] * wts[has] / tot[has][:, None]
        stay = ~has & (Sg[:, c] > 0); T[stay, c] += freed[stay, c]   # no other class on: it stays in its own class
    W = np.zeros((n, na))
    for c, (_, m, _) in enumerate(cls):
        W[:, m] = (T[:, c] - kept[:, c])[:, None] * D[c] + (kept[:, c] / len(m))[:, None]
    W[(Sg == 0).all(1)] = neutral
    return W


def ref_targets(S: np.ndarray, ia: int, ib: int) -> np.ndarray:
    """E10's S2 on two of the book's assets alone."""
    w3 = E10.SCHEMES["S2"][E10.combo_of(S[:, ia] == 1, S[:, ib] == 1)]; W = np.zeros(S.shape); W[:, ia] = w3[:, 0]; W[:, ib] = w3[:, 1]
    return W


# ------------------------------------------------------------------------------------------------------- the book
def nbook(cl: np.ndarray, fi: np.ndarray, kd: np.ndarray, W: np.ndarray, w0: np.ndarray, rec: bool = False):
    """Daily NAV (1.0 on the first decision day) of a book over any number of assets: w0 bought on the next session, then a
    trade at the next session's fills whenever the weekly target changes (the first row is not traded on, as in the
    engine's book); selling before buying, 10.4 bps per side, so the post-trade value V' solves V' = V - c Σ|w_i V' - v_i|."""
    n = len(cl); k0 = int(kd[0]); kt = k0 + 1
    u = w0 / (fi[kt] * (1 + C)); cur = tuple(w0); starts = [kt]; U = [u.copy()]; sw = []
    for i in range(1, len(kd)):
        tw = tuple(W[i])
        if tw == cur:
            continue
        k = int(kd[i]) + 1
        if k >= n:
            break
        w = W[i]; p = fi[k]; v = u * p; V = float(v.sum()); Vn = V
        for _ in range(6):
            Vn = V - C * float(np.abs(w * Vn - v).sum())
        u = w * Vn / p
        if rec:
            sw.append({"i": i, "k": k, "from": cur, "to": tw, "cost": (V - Vn) / V})
        cur = tw; starts.append(k); U.append(u.copy())
    nav = np.empty(n - k0); nav[0] = 1.0; b = starts + [n]
    for j, uu in enumerate(U):
        nav[b[j] - k0:b[j + 1] - k0] = cl[b[j]:b[j + 1]] @ uu
    return nav, sw


# ------------------------------------------------------------------------------------------------------- §1 how in sync
def sync_stats(Sb: np.ndarray, closes_b: np.ndarray, dec_b, bk: dict) -> dict:
    A = bk["assets"]; L = bk["lab"]; eqm = [A.index(x) for x in bk["classes"][0][1]]; others = [j for j in range(len(A)) if j not in eqm]; ne = len(eqm)
    on = Sb == 1; k = on[:, eqm].sum(1); maj = 2 * k > ne; yrs = (dec_b[-1] - dec_b[0]).days / 365.25; eqs = Sb[:, eqm]
    first = {"to SELL": [], "to BUY": []}; prev = None; start = None
    for t in range(len(k)):
        b = int(k[t])
        if b in (0, ne):
            if prev is not None and prev != b:
                tgt = -1 if b == 0 else 1; seg = eqs[start:t + 1]; when = [int(np.argmax(seg[:, j] == tgt)) for j in range(ne)]; mn = min(when)
                first["to SELL" if b == 0 else "to BUY"].append([A[eqm[j]] for j in range(ne) if when[j] == mn])
            prev, start = b, t
    eqword = np.where(k == ne, "equity all BUY", np.where(k == 0, "equity all SELL", "equity split"))
    pat = pd.Series([w + "".join(f" · {L[A[j]]} {'BUY' if on[i, j] else 'SELL'}" for j in others) for i, w in enumerate(eqword)]).value_counts(normalize=True)
    lr = np.diff(np.log(closes_b), axis=0)
    return {"weeks": int(len(Sb)), "first": dec_b[0].date().isoformat(), "last": dec_b[-1].date().isoformat(), "equity_members": [L[A[j]] for j in eqm], "others": [L[A[j]] for j in others],
            "breadth": [float((k == j).mean()) for j in range(ne + 1)], "all_agree": float(np.isin(k, [0, ne]).mean()),
            "differs_from_majority": {L[A[j]]: float((on[:, j] != maj).mean()) for j in others}, "patterns": {str(kk): float(v) for kk, v in pat.items()},
            "agree": [[float((Sb[:, i] == Sb[:, j]).mean()) for j in range(len(A))] for i in range(len(A))],
            "buy_share": {L[a]: float(on[:, j].mean()) for j, a in enumerate(A)}, "flips_yr": {L[a]: float((np.diff(Sb[:, j]) != 0).sum() / yrs) for j, a in enumerate(A)},
            "turns": {kk: {"n": len(v), "first": {L[A[j]]: sum(A[j] in e for e in v) for j in eqm}, "alone_first": {L[A[j]]: sum(e == [A[j]] for e in v) for j in eqm}} for kk, v in first.items()},
            "corr": np.corrcoef(lr.T).tolist()}


# ------------------------------------------------------------------------------------------------------- one book
def prepare(bk: dict, px: dict) -> dict:
    A = bk["assets"]; idx, cl, fi, dec = multi_frame(px, A, pd.Timestamp(bk["start"])); cal_end = {a: E10.cal_end_of(px[a].index) for a in A}
    S, M = signals(px, A, idx, cl, dec, cal_end, bk["closes_only"])
    cls = [(name, np.array([A.index(x) for x in members]), b) for name, members, b in bk["classes"]]; neutral = np.zeros(len(A))
    for _, m, b in cls:
        neutral[m] = b / len(m)
    kd = idx.get_indexer(dec[K0:]); days = idx[kd[0]:]
    # the holdings: on the Indian book (the latest data) the liquid ETF is a fifth column, its level read as of each day of the calendar
    sw = bool(bk.get("switch")) and SWI is not None; AH = A + ["Liquid"] if sw else list(A)
    clh, fih = (np.column_stack([cl, SWI.liquid_on(idx)]), np.column_stack([fi, SWI.liquid_on(idx)])) if sw else (cl, fi)
    return {"bk": bk, "A": A, "idx": idx, "cl": cl, "fi": fi, "dec": dec, "cal_end": cal_end, "S": S, "M": M, "cls": cls, "neutral": neutral, "kd": kd, "days": days,
            "d_days": (days[-1] - days[0]).days, "Sb": S[K0:], "Mb": {k: v[K0:] for k, v in M.items()}, "dec_b": dec[K0:],
            "sw": sw, "AH": AH, "clh": clh, "fih": fih, "neutral_h": np.r_[neutral, 0.0] if sw else neutral, "gcol": A.index(GOLD) if sw else None,
            "sell_b": SWI.sell_at(dec[K0:]) if sw else None, "liquid_days_carried": SWI.carried_days(idx) if sw else 0}


def weights_of(P: dict, code: str, fam: str, look, Sb=None, Mb=None):
    Sb = P["Sb"] if Sb is None else Sb; Mk = (P["Mb"][look] if Mb is None else Mb) if look else None
    if fam == "REF":
        A = P["A"]; ia, ib = A.index(P["bk"]["ref"][0]), A.index(P["bk"]["ref"][1]); W = ref_targets(Sb, ia, ib); w0 = np.zeros(len(A)); w0[[ia, ib]] = 0.5
        return W, w0
    return targets(code, Sb, Mk, P["cls"], P["neutral"]), P["neutral"]


def hold(P: dict, W: np.ndarray, w0: np.ndarray):
    """The weights as held. On the Indian book (the latest data) a fifth column, the liquid ETF, takes gold's weight in any
    week whose decision falls in gold's weekly supertrend SELL (the switch), the start included; elsewhere W itself."""
    if not P["sw"]:
        return W, w0
    g, sb = P["gcol"], P["sell_b"]
    Wh = np.column_stack([W, np.zeros(len(W))]); Wh[sb, -1] = Wh[sb, g]; Wh[sb, g] = 0.0
    w0h = np.r_[w0, 0.0]
    if sb[0]:
        w0h[-1], w0h[g] = w0h[g], 0.0
    return Wh, w0h


def run_book(key: str, px: dict, MS: int, t0: float) -> dict:
    bk = BOOKS[key]; P = prepare(bk, px); A = P["A"]; L = bk["lab"]; idx, cl, fi, dec, kd, days, d_days = (P[x] for x in ("idx", "cl", "fi", "dec", "kd", "days", "d_days"))
    Sb, Mb, dec_b = P["Sb"], P["Mb"], P["dec_b"]; n_b = len(Sb); yrs = d_days / 365.25; AH, clh, fih = P["AH"], P["clh"], P["fih"]
    ck = {"close_max_abs_diff": max(float(np.abs(cl[:, j] - px[a]["close"].reindex(idx).to_numpy()).max()) for j, a in enumerate(A)), "calendar_equals_ref_pair": None,
          "engine_book_max_abs_diff": None, "ref_vs_e10_cagr_diff": None, "st_missing_after_start": int(np.isnan(Sb).sum()),
          "mom_missing_after_start": int(sum(np.isnan(v).sum() for v in Mb.values())), "weights_row_sum_max_dev": 0.0, "roll_equivalence_failures": 0,
          "cut_checks": 0, "cut_violations": [], "shift_count_checks": 0, "shift_count_mismatch": 0, "switch_book_vs_book3_max_abs_diff": None}
    # the book against the engine's (Rule R on the reference pair), and the frame against that pair's frame
    pf = pair_frame(px, bk["pair"], "plain"); same = bool(pf.index.equals(idx) and pf.index[pf["decision"]].equals(dec)); ck["calendar_equals_ref_pair"] = same
    ia, ib = A.index(bk["ref"][0]), A.index(bk["ref"][1])
    if same:
        stR = VR.rule_states(VR.z_at_decisions(pf, "w52").to_numpy())[0][K0:]; ne_, _ = book(pf, dec[K0:], stR, COST, PIT())
        Wr = np.array([{"A": (1.0, 0.0), "B": (0.0, 1.0), "N": (0.5, 0.5)}[s] for s in stR])
        ck["engine_book_max_abs_diff"] = float(np.max(np.abs(ne_.to_numpy() - nbook(cl[:, [ia, ib]], fi[:, [ia, ib]], kd, Wr, np.array([0.5, 0.5]))[0])))
        if P["sw"]:   # the switch: Rule R on the reference pair through this book with the liquid ETF, against variants.book3 (written separately)
            sb_ = P["sell_b"]; sel_ = [ia, ib, len(A)]; W3 = np.column_stack([Wr, np.zeros(len(Wr))]); W3[sb_, 2] = W3[sb_, 1]; W3[sb_, 1] = 0.0
            w03 = np.array([0.5, 0.0, 0.5]) if sb_[0] else np.array([0.5, 0.5, 0.0])
            n3 = VR.book3(pf, dec[K0:], list(stR), sb_, SWI.liquid_on(pf.index), COST)[0]
            ck["switch_book_vs_book3_max_abs_diff"] = float(np.max(np.abs(n3.to_numpy() - nbook(clh[:, sel_], fih[:, sel_], kd, W3, w03)[0])))
    # point in time: everything recomputed on the history cut at 8 decision weeks
    for T in np.unique(np.linspace(K0 + 10, len(dec) - 1, 8).astype(int)):
        pc = {a: px[a].loc[: dec[T]] for a in A}; ic = idx[idx <= dec[T]]; Sc, Mc = signals(pc, A, ic, cl[: len(ic)], dec[: T + 1], P["cal_end"], bk["closes_only"]); ck["cut_checks"] += 1
        ok = np.array_equal(Sc, P["S"][: T + 1], equal_nan=True) and all(np.allclose(Mc[k], P["M"][k][: T + 1], rtol=0, atol=1e-12, equal_nan=True) for k in P["M"])
        for code, fam, _, look in STRATS:
            ok = ok and np.array_equal(targets(code, Sc[K0:], Mc[look][K0:] if look else None, P["cls"], P["neutral"]),
                                       targets(code, P["S"][K0:T + 1], P["M"][look][K0:T + 1] if look else None, P["cls"], P["neutral"]))
        if not ok:
            ck["cut_violations"].append(dec[T].date().isoformat())
    # benchmarks: bought once on the session after the first decision day
    bench = {}
    eq_ = np.r_[np.full(len(A), 1 / len(A)), np.zeros(len(AH) - len(A))]   # 25% each of the book's assets (the liquid ETF is not one)
    for name, wb in [("neutral", P["neutral_h"]), ("equal", eq_)] + [(L[a], np.eye(len(AH))[j]) for j, a in enumerate(AH)]:
        u = wb / (fih[kd[0] + 1] * (1 + C)); bench[name] = np.r_[1.0, clh[kd[0] + 1:] @ u]
    BST = {k: E10.nstats(v, days) for k, v in bench.items()}; best_asset = max(BST[L[a]]["cagr"] for a in A)
    shifts = list(range(MIN_SHIFT, n_b - MIN_SHIFT + 1))
    if MS:
        shifts = shifts[:: max(1, len(shifts) // MS)][:MS]
    eqm = P["cls"][0][1]; buy0 = (Sb == 1).sum(0); agree0 = int(np.isin((Sb[:, eqm] == 1).sum(1), [0, len(eqm)]).sum())
    for s_ in shifts:   # the shifts keep every asset's BUY weeks and the equity members' agreement (measured)
        Sr = np.roll(Sb, s_, axis=0); ck["shift_count_checks"] += 1
        ck["shift_count_mismatch"] += int(not (np.array_equal((Sr == 1).sum(0), buy0) and int(np.isin((Sr[:, eqm] == 1).sum(1), [0, len(eqm)]).sum()) == agree0))
    pos_w = idx.get_indexer(dec_b) - kd[0]
    ROWS, CURV, WTS, NULLS, SWR, NT, NP = [], {}, {}, [], [], {}, {}
    for code, fam, lab, look in STRATS + [(REF_CODE, "REF", bk["ref_label"], None)]:
        W, w0 = weights_of(P, code, fam, look)
        ck["weights_row_sum_max_dev"] = max(ck["weights_row_sum_max_dev"], float(np.abs(W.sum(1) - 1).max()))
        s1 = shifts[0]; Wchk, _ = weights_of(P, code, fam, look, np.roll(Sb, s1, axis=0), np.roll(Mb[look], s1, axis=0) if look else None)
        ck["roll_equivalence_failures"] += int(not np.array_equal(Wchk, np.roll(W, s1, axis=0)))
        Wh, w0h = hold(P, W, w0); nav, sw = nbook(clh, fih, kd, Wh, w0h, rec=True); st = E10.nstats(nav, days)
        # the nulls: the strategy's rows slid in time, then held through the same switch (gold's SELL weeks are fixed by the calendar)
        nt = [(s_,) + tuple(E10.quick(nbook(clh, fih, kd, *hold(P, np.roll(W, s_, axis=0), w0))[0], d_days)) for s_ in shifts]   # N-T: rows follow their week's signals
        npk = [(s_,) + tuple(E10.quick(nbook(clh, fih, kd, *hold(P, weights_of(P, code, fam, look, None, np.roll(Mb[look], s_, axis=0))[0], w0))[0], d_days)) for s_ in shifts] if fam in ("STMO", "GR") else []
        NT[code] = [x[1] for x in nt]; NP[code] = [x[1] for x in npk]
        NULLS += [{"book": key, "code": code, "null": "N-T", "shift": x[0], "cagr": x[1], "maxdd": x[2]} for x in nt] + [{"book": key, "code": code, "null": "N-P", "shift": x[0], "cagr": x[1], "maxdd": x[2]} for x in npk]
        pick = Wh.argmax(1)
        ROWS.append({"book": key, "code": code, "family": fam, "label": lab, "look": look, "first": days[0].date().isoformat(), "last": days[-1].date().isoformat(),
                     "cagr": st["cagr"], "vol": st["vol"], "maxdd": st["maxdd"], "final": float(nav[-1]), "excess_neutral": st["cagr"] - BST["neutral"]["cagr"],
                     "excess_equal": st["cagr"] - BST["equal"]["cagr"], "excess_best_asset": st["cagr"] - best_asset, "switches": len(sw), "switches_yr": len(sw) / yrs,
                     **{f"w{j + 1}": float(Wh[:, j].mean()) for j in range(len(AH))}, "concentrated": float((Wh.max(1) > 0.99).mean()),
                     **{f"top{j + 1}": float(((Wh.max(1) > 0.99) & (pick == j)).mean()) for j in range(len(AH))},
                     "pct_T": E1.pctile(st["cagr"], NT[code]), "pct_T_dd": E1.pctile(st["maxdd"], [x[2] for x in nt]),
                     "pct_P": E1.pctile(st["cagr"], NP[code]) if npk else np.nan, "pct_P_dd": E1.pctile(st["maxdd"], [x[2] for x in npk]) if npk else np.nan,
                     "null_T_median": float(np.median(NT[code])), "null_P_median": float(np.median(NP[code])) if npk else np.nan, "shifts": len(shifts)})
        CURV[code] = [r6(v) for v in nav[pos_w]]; WTS[code] = np.round(Wh, 4)
        for s in sw:
            SWR.append({"book": key, "code": code, "decision_date": dec_b[s["i"]].date().isoformat(), "trade_date": idx[s["k"]].date().isoformat(),
                        "from": " / ".join(f"{v:.3f}" for v in s["from"]), "to": " / ".join(f"{v:.3f}" for v in s["to"]), "cost_frac": s["cost"]})
        print(f"  {key:5s} {code:10s} CAGR {st['cagr'] * 100:5.1f}%  maxdd {st['maxdd'] * 100:4.0f}%  [{time.time() - t0:4.0f}s]", flush=True)
    e10p = os.path.join(ROOT, "explore", "E10_trend_rotation", "out", "strategies.csv")
    if same and os.path.exists(e10p):
        e = pd.read_csv(e10p); q = e[(e["pair"] == bk["pair"]) & (e["rule"] == "S2")]
        if len(q):
            ck["ref_vs_e10_cagr_diff"] = float(abs([r for r in ROWS if r["code"] == REF_CODE][0]["cagr"] - q["cagr"].iloc[0]))
    SY = sync_stats(Sb, cl[idx.get_indexer(dec_b)], dec_b, bk)
    return {"key": key, "bk": bk, "P": P, "checks": ck, "rows": ROWS, "curves": CURV, "w": WTS, "nulls": NULLS, "sw": SWR, "nt": NT, "np": NP, "bench": bench, "bst": BST,
            "pos_w": pos_w, "sync": SY, "shifts": shifts}


def run_check_book(px: dict) -> list:
    """The US book with PDBC as the commodity: every strategy, no nulls (E11.md, the check)."""
    bk = CHECK_BOOK; P = prepare(bk, px); A = P["A"]; cl, fi, kd, days = P["cl"], P["fi"], P["kd"], P["days"]
    u = P["neutral"] / (fi[kd[0] + 1] * (1 + C)); nst = E10.nstats(np.r_[1.0, cl[kd[0] + 1:] @ u], days); rows = []
    for code, fam, lab, look in STRATS:
        W, w0 = weights_of(P, code, fam, look); st = E10.nstats(nbook(cl, fi, kd, W, w0)[0], days)
        rows.append({"code": code, "label": lab, "first": days[0].date().isoformat(), "last": days[-1].date().isoformat(), "cagr": st["cagr"], "maxdd": st["maxdd"],
                     "neutral_cagr": nst["cagr"], "excess_neutral": st["cagr"] - nst["cagr"]})
    return rows


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-shifts", type=int, default=0); MS = ap.parse_args().max_shifts
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    global SWI
    px = LT.load_india(); SWI = LT.activate(px); START_IN = dict(RE.START)   # the Indian book from 2011-01-01 with the switch
    px.update(load_us_etf()); px.update(load_us_gold()); px.update(load_us_small())
    R = {}
    for k in BOOKS:   # the US book keeps the engine's own pair start (its reference pair's frame starts in 2010)
        RE.START.update(START0 if k == "us" else START_IN); R[k] = run_book(k, px, MS, t0)
    RE.START.update(START0); CHK = run_check_book(px)
    bad = {}
    for k, r in R.items():
        c = r["checks"]
        if (c["close_max_abs_diff"] > 0 or not c["calendar_equals_ref_pair"] or c["engine_book_max_abs_diff"] is None or not c["engine_book_max_abs_diff"] < 1e-9
                or c["ref_vs_e10_cagr_diff"] is None or not c["ref_vs_e10_cagr_diff"] < 1e-12 or c["st_missing_after_start"] or c["mom_missing_after_start"]
                or c["weights_row_sum_max_dev"] > 1e-12 or c["roll_equivalence_failures"] or c["cut_violations"] or c["shift_count_mismatch"]
                or (r["P"]["sw"] and not (c["switch_book_vs_book3_max_abs_diff"] is not None and c["switch_book_vs_book3_max_abs_diff"] < 1e-9))):
            bad[k] = c
    if bad:
        sys.exit(f"checks FAILED: {bad}")
    SR = pd.DataFrame([row for r in R.values() for row in r["rows"]]); SR.to_csv(os.path.join(OUT, "strategies.csv"), index=False)
    pd.DataFrame([x for r in R.values() for x in r["nulls"]]).to_csv(os.path.join(OUT, "nulls.csv"), index=False)
    pd.DataFrame([x for r in R.values() for x in r["sw"]]).to_csv(os.path.join(OUT, "switches.csv"), index=False)
    pd.DataFrame(CHK).to_csv(os.path.join(OUT, "check_us_pdbc.csv"), index=False)
    wl, sl = [], []
    for k, r in R.items():
        A = r["P"]["A"]; AH = r["P"]["AH"]; L = r["bk"]["lab"]; dts = [d.date().isoformat() for d in r["P"]["dec_b"]]
        for c, W in r["w"].items():
            for j, a in enumerate(AH):
                wl.append(pd.DataFrame({"book": k, "code": c, "date": dts, "asset": L[a], "weight": W[:, j]}))
        for j, a in enumerate(A):
            sl.append(pd.DataFrame({"book": k, "date": dts, "asset": L[a], "supertrend": r["P"]["Sb"][:, j].astype(int), **{f"mom{kk}": v[:, j] for kk, v in r["P"]["Mb"].items()}}))
    pd.concat(wl).to_csv(os.path.join(OUT, "weights_weekly.csv"), index=False); pd.concat(sl).to_csv(os.path.join(OUT, "signals_weekly.csv"), index=False)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "data": LT.NOTE + " (the Indian book; the US book unchanged)", "us_book_pair_start": START0,
             "switch": "Indian book: in any week whose decision falls in gold's weekly supertrend SELL, gold's weight is held in the liquid ETF (every strategy, its start and every null shift; gold's SELL weeks fixed by the calendar); the benchmarks bought once are not switched",
             "inputs": fingerprints(), "cost_bps_round_trip": COST, "common_start_decision_week": K0 + 1, "books": {}}
    for k, r in R.items():
        m = SR[(SR["book"] == k) & (SR["family"] != "REF")]
        audit["books"][k] = {"label": r["bk"]["label"], "assets": r["P"]["A"], "holdings": r["P"]["AH"], "liquid_days_carried": r["P"]["liquid_days_carried"], "classes": [[nm, list(mm), b] for nm, mm, b in r["bk"]["classes"]], "first": r["P"]["days"][0].date().isoformat(),
                             "last": r["P"]["days"][-1].date().isoformat(), "shifts": len(r["shifts"]), "shift_range_weeks": [r["shifts"][0], r["shifts"][-1]], "checks": r["checks"],
                             "benchmarks": r["bst"], "n_strategies": int(len(m)), "ahead_of_neutral": int((m["excess_neutral"] > 0).sum()), "ahead_of_equal": int((m["excess_equal"] > 0).sum()),
                             "ahead_of_best_asset": int((m["excess_best_asset"] > 0).sum()), "pct_T_high": int((m["pct_T"] >= HI).sum()), "pct_T_low": int((m["pct_T"] <= LO).sum()),
                             "pct_P_high": int((m["pct_P"] >= HI).sum()), "pct_P_low": int((m["pct_P"] <= LO).sum()), "sync": r["sync"]}
    audit["check_us_pdbc"] = {"first": CHK[0]["first"], "last": CHK[0]["last"], "neutral_cagr": CHK[0]["neutral_cagr"], "ahead_of_neutral": int(sum(x["excess_neutral"] > 0 for x in CHK))}
    audit["seconds"] = round(time.time() - t0)
    json.dump(jclean(audit), open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)
    write_md(audit, SR, R, CHK)
    write_page(audit, SR, R, CHK)
    print(f"done in {time.time() - t0:.0f}s; checks " + "; ".join(f"{k}: {r['checks']}" for k, r in R.items()))


# ------------------------------------------------------------------------------------------------------- results (markdown)
def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")


def write_md(audit, SR, R, CHK):
    md = ["# E11 results — multi-asset books, India and the US kept apart: the desk's supertrend for the asset classes, momentum for the pick (EXPLORATORY)\n",
          f"_Generated by `explore/E11_multi_asset/multi.py` on {audit['run_date']}. Weekly decisions, next-session fills, 10.4 bps per side, always invested (no cash; in the Indian book the liquid ETF holds gold's weight in any week gold's supertrend is SELL); every strategy in a book starts at its "
          "neutral weights on the same day. Nulls: every shift of the signals by at least 26 weeks. Description, not a finding; percentiles in **bold** are notable "
          f"(≤ {LO:g} or ≥ {HI:g})._\n"]
    for k, r in R.items():
        bk = r["bk"]; A = r["P"]["A"]; AH = r["P"]["AH"]; L = bk["lab"]; ab = audit["books"][k]; ck = ab["checks"]; SY = r["sync"]; b = r["bst"]
        md += [f"## {bk['label']} book ({bk['unit']}): " + ", ".join(f"{nm} {' + '.join(L[x] for x in mm)} ({b_ * 100:.0f}%)" for nm, mm, b_ in bk["classes"]) + f", {ab['first']} → {ab['last']}\n",
               f"**Checks:** closes equal the loaders' (max {ck['close_max_abs_diff']:.1e}); the calendar equals the {bk['pair'].replace('|', ' ÷ ').replace(GOLD, 'gold')} pair frame's; "
               f"the book equals the engine's on Rule R (max NAV difference {ck['engine_book_max_abs_diff']:.1e}) and reproduces E10's S2 on the reference pair (CAGR difference {ck['ref_vs_e10_cagr_diff']:.1e}); "
               f"signals and every strategy's targets pass the cut-history check ({ck['cut_checks']} cuts, 0 violations); weight rows sum to 1 (max deviation {ck['weights_row_sum_max_dev']:.1e}); "
               f"every shift kept each asset's BUY weeks and the equity members' agreement ({ck['shift_count_checks']} shifts of {ab['shift_range_weeks'][0]}–{ab['shift_range_weeks'][1]} weeks)"
               + (f"; with the switch, Rule R on the reference pair through this book equals variants.book3 (max NAV difference {ck['switch_book_vs_book3_max_abs_diff']:.1e}). **Data:** {LT.NOTE}.\n" if r["P"]["sw"] else ".\n"),
               f"### How in sync ({SY['first']} → {SY['last']}, {SY['weeks']} weeks)\n",
               f"- Equity members ({', '.join(SY['equity_members'])}) on BUY: " + ", ".join(f"{j} {pc(v)}" for j, v in enumerate(SY["breadth"])) + f" of weeks; all agree in {pc(SY['all_agree'])}.",
               "- Differs from the equity majority: " + ", ".join(f"{a} {pc(v)}" for a, v in SY["differs_from_majority"].items()) + " of weeks.",
               "- Most common states: " + "; ".join(f"{p_} {pc(v)}" for p_, v in list(SY["patterns"].items())[:6]) + ".",
               "- Weeks on BUY: " + ", ".join(f"{a} {pc(v)}" for a, v in SY["buy_share"].items()) + "; flips a year: " + ", ".join(f"{a} {v:.1f}" for a, v in SY["flips_yr"].items()) + ".",
               "- When the equity members turned together: " + "; ".join(f"{kk}: {v['n']} times, first to flip (ties counted for each) " + ", ".join(f"{a} {c}" for a, c in v["first"].items())
                                                                         + " (alone: " + ", ".join(f"{a} {c}" for a, c in v["alone_first"].items()) + ")" for kk, v in SY["turns"].items()) + ".",
               "- Weekly return correlations: " + "; ".join(f"{L[A[i]]}–{L[A[j]]} {SY['corr'][i][j]:+.2f}" for i in range(len(A)) for j in range(i + 1, len(A))) + ".\n",
               f"### Every strategy ({ab['first']} → {ab['last']})\n",
               f"Benchmarks: neutral bought once ({', '.join(f'{L[a]} {w * 100:.1f}%' for a, w in zip(A, r['P']['neutral']))}, not switched) {fp(b['neutral']['cagr'])} a year (max drawdown {fp(b['neutral']['maxdd'], 0)}); "
               f"25% each {fp(b['equal']['cagr'])} ({fp(b['equal']['maxdd'], 0)}); " + "; ".join(f"{L[a]} {fp(b[L[a]]['cagr'])} ({fp(b[L[a]]['maxdd'], 0)})" for a in AH) + ".\n",
               "| strategy | family | CAGR | over neutral | over 25% each | over the best asset | volatility | max drawdown | switches a year | average weight " + " / ".join(L[a] for a in AH)
               + " | weeks all in one asset | N-T pct: CAGR / drawdown | N-P pct: CAGR / drawdown |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, x in SR[SR["book"] == k].iterrows():
            md.append(f"| {x['code']} {x['label']} | {FAMILIES[x['family']]} | {fp(x['cagr'])} | {pp(x['excess_neutral'])} | {pp(x['excess_equal'])} | {pp(x['excess_best_asset'])} | {pc(x['vol'])} | {fp(x['maxdd'], 0)} | "
                      f"{x['switches_yr']:.1f} | " + " / ".join(pc(x[f"w{j + 1}"]) for j in range(len(AH))) + f" | {pc(x['concentrated'])} | {pq(x['pct_T'])} / {pq(x['pct_T_dd'])} | {pq(x['pct_P'])} / {pq(x['pct_P_dd'])} |")
        md.append(f"\nThe {ab['n_strategies']} strategies (the reference aside): ahead of neutral {ab['ahead_of_neutral']}, of 25% each {ab['ahead_of_equal']}, of the best single asset {ab['ahead_of_best_asset']}; "
                  f"N-T notable high in {ab['pct_T_high']}, low in {ab['pct_T_low']}; N-P (six strategies) notable high in {ab['pct_P_high']}, low in {ab['pct_P_low']}.\n")
    c0 = CHK[0]
    md += [f"## Check: the US book with PDBC as the commodity ({c0['first']} → {c0['last']}; neutral {fp(c0['neutral_cagr'])} a year; no nulls)\n", "| strategy | CAGR | over neutral | max drawdown |", "|---|---|---|---|"]
    md += [f"| {x['code']} {x['label']} | {fp(x['cagr'])} | {pp(x['excess_neutral'])} | {fp(x['maxdd'], 0)} |" for x in CHK]
    open(os.path.join(OUT, "E11_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


# ------------------------------------------------------------------------------------------------------- the page
def rules_html(r: dict) -> str:
    """The weights of ST-EQ, ST-BR and ST-TL in every combination of the assets' signals, computed by targets() itself."""
    P = r["P"]; A = P["A"]; L = r["bk"]["lab"]; na = len(A)
    combos = sorted(itertools.product([1, -1], repeat=na), key=lambda t: (-sum(x == 1 for x in t), [-x for x in t]))
    Sx = np.array(combos, float); Mz = np.zeros(Sx.shape)
    W = {c: targets(c, Sx, Mz, P["cls"], P["neutral"]) for c in ("ST-EQ", "ST-BR", "ST-TL")}
    fmt = lambda w: " · ".join(f"{L[A[j]]} {w[j] * 100:.0f}" for j in range(na) if w[j] > 1e-9)  # noqa: E731
    rows = "".join("<tr><td class='nw'>" + " ".join(f"{L[A[j]]} {'BUY' if t[j] == 1 else 'SELL'}" for j in range(na)) + "</td>" + "".join(f"<td>{fmt(W[c][i])}</td>" for c in W) + "</tr>"
                   for i, t in enumerate(combos))
    cls = " · ".join(f"<b>{nm}</b> {' + '.join(L[x] for x in mm)} (base {b * 100:.0f}%)" for nm, mm, b in r["bk"]["classes"])
    return ("<div class='card'><p class='note' style='margin:4px 0'><b>Classes:</b> " + cls + ". <b>Neutral:</b> " + ", ".join(f"{L[a]} {w * 100:.1f}%" for a, w in zip(A, P["neutral"]))
            + " — at the start, and whenever nothing is on. Every version is always invested (no cash)."
            + (" <b>The switch (the latest data):</b> in this book, in any week whose decision falls in gold's weekly supertrend SELL, whatever weight a version gives gold is held in the liquid ETF instead (the start included); the table below shows the weights before the switch." if P.get("sw") else "") + "</p>"
            "<ul class='note' style='margin:4px 0 4px 18px'><li><b>ST-EQ · equal split:</b> a class is on when any member is BUY; its money is split equally among its BUY members. "
            "The money of a class that is off goes to the classes that are on, in proportion to their base weights.</li>"
            "<li><b>ST-BR · breadth:</b> a class is on in proportion to its BUY members, and its money is always spread over all its members; the off part goes to the other classes that are on (or stays, if none is).</li>"
            "<li><b>ST-TL · tilt:</b> as ST-EQ, but a class that is off keeps half its base weight (spread over its members); only the other half moves.</li>"
            "<li><b>ST-MO · momentum pick:</b> as ST-EQ, but a class's money goes to its BUY member with the highest momentum (12-1, 12-month or 6-month).</li>"
            "<li><b>GR · gate and rank:</b> among all assets on BUY, 100% in the one with the highest momentum; none on BUY → neutral. <b>MO · momentum only:</b> 100% in the asset with the highest momentum.</li>"
            f"<li><b>REF:</b> {r['bk']['ref_label']} — both BUY 50/50, one BUY all in it, both SELL 50/50.</li></ul>"
            "<details><summary>The weights (%) of ST-EQ, ST-BR and ST-TL in every combination of signals (computed by the strategy code)</summary><div class='scroll'><table class='txt'><thead><tr><th>signals</th>"
            "<th>ST-EQ</th><th>ST-BR</th><th>ST-TL</th></tr></thead><tbody>" + rows + "</tbody></table></div></details></div>")


def write_page(audit, SR, R, CHK):
    books = {}
    for k, r in R.items():
        A = r["P"]["A"]; L = r["bk"]["lab"]
        STJ = {x["code"]: {kk: (r6(v) if isinstance(v, float) else v) for kk, v in x.items()} for x in SR[SR["book"] == k].to_dict("records")}
        AH = r["P"]["AH"]
        books[k] = {"label": r["bk"]["label"], "unit": r["bk"]["unit"], "assets": [L[a] for a in AH], "names": [r["bk"]["name"][a] for a in AH], "ref_label": r["bk"]["ref_label"],
                    "sig": [L[a] for a in A], "signames": [r["bk"]["name"][a] for a in A], "sw": bool(r["P"]["sw"]),
                    "t": [d.date().isoformat() for d in r["P"]["dec_b"]], "curves": r["curves"], "bench": {kk: [r6(x) for x in v[r["pos_w"]]] for kk, v in r["bench"].items()},
                    "bst": r["bst"], "neutral": r["P"]["neutral_h"], "w": r["w"], "nt": r["nt"], "np": r["np"], "st": STJ, "states": r["P"]["Sb"].astype(int), "sync": r["sync"],
                    "audit": {kk: audit["books"][k][kk] for kk in ("ahead_of_neutral", "ahead_of_equal", "ahead_of_best_asset", "n_strategies", "pct_T_high", "pct_T_low", "pct_P_high", "pct_P_low",
                                                                    "shifts", "shift_range_weeks", "first", "last")},
                    "rules": rules_html(r)}
    books["us"]["check"] = CHK
    DATA = {"books": books, "order": list(R), "strats": [[c, f, l] for c, f, l, _ in STRATS] + [[REF_CODE, "REF", "the two-asset reference (E10's S2)"]], "families": FAMILIES, "lo": LO, "hi": HI, "default": DEFAULT}
    blob = json.dumps(jclean(DATA), separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")
    bsel = "<label>Book <select id='book'>" + "".join(f"<option value='{k}'{' selected' if k == DEFAULT['book'] else ''}>{html.escape(R[k]['bk']['label'])}</option>" for k in R) + "</select></label>"
    opts = "".join(f"<optgroup label='{html.escape(FAMILIES[f], quote=True)}'>" + "".join(f"<option value='{c}'{' selected' if c == DEFAULT['strat'] else ''}>{html.escape(c + ' · ' + l)}</option>"
                                                                                       for c, ff, l in DATA["strats"] if ff == f) + "</optgroup>" for f in FAMILIES)
    controls = bsel + f"<label>Strategy <select id='strat'>{opts}</select></label>"
    key = ""
    for k, r in R.items():
        ab = audit["books"][k]; m = SR[(SR["book"] == k) & (SR["family"] != "REF")]; best = m.loc[m["cagr"].idxmax()]; ref = SR[(SR["book"] == k) & (SR["family"] == "REF")].iloc[0]
        b = r["bst"]; SY = r["sync"]; L = r["bk"]["lab"]; A = r["P"]["A"]
        key += (f"<p style='margin:4px 0'><b>{r['bk']['label']}</b> ({ab['first']} → {ab['last']}; " + ", ".join(L[a] for a in A) + f"): the equity members give the same supertrend signal in {pc(SY['all_agree'])} of weeks; "
                + ", ".join(f"{a} differs from the equity majority in {pc(v)}" for a, v in SY["differs_from_majority"].items()) + f". Neutral bought once {fp(b['neutral']['cagr'])} a year (max drawdown {fp(b['neutral']['maxdd'], 0)}); "
                + ", ".join(f"{L[a]} {fp(b[L[a]]['cagr'])}" for a in A) + f". Strategies ahead of neutral {ab['ahead_of_neutral']} of {ab['n_strategies']}, of the best single asset {ab['ahead_of_best_asset']}; "
                f"notable against the timing null: {ab['pct_T_high']} high, {ab['pct_T_low']} low. Highest CAGR {best['code']} {fp(best['cagr'])} (max drawdown {fp(best['maxdd'], 0)}, timing percentile {best['pct_T']:.1f}); "
                f"the reference ({r['bk']['ref_label']}) {fp(ref['cagr'])}.</p>")
    c0 = audit["check_us_pdbc"]
    key += f"<p style='margin:4px 0'><b>Check:</b> the US book with PDBC instead of GCC ({c0['first']} → {c0['last']}): ahead of its neutral ({fp(c0['neutral_cagr'])} a year) in {c0['ahead_of_neutral']} of {len(CHK)} strategies.</p>"
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Multi-asset books</title>"),
         "<header class='bar'><div class='bar-inner'><b>E11 · multi-asset books</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Multi-asset books, India and the US kept apart: the desk's supertrend for the asset classes, momentum for the pick</h1>",
         f"<p class='note'><b>Data:</b> {html.escape(LT.NOTE)} (the Indian book; the US book is unchanged). In the Indian book every strategy holds the liquid ETF in place of gold in any week gold's supertrend is SELL.</p>",
         f"<p>Two separate books. <b>India:</b> N100, MC100, SC100 (equity) and MCX gold, in rupees, with the liquid ETF holding gold's weight on gold's SELL weeks. <b>US:</b> SPY and IJR (equity), GCC (broad commodities) and GLD (gold), in dollars. "
         "The indices of a market mostly agree, so the design decides which asset classes are on with the desk's weekly supertrend, and spreads a class's money equally, by breadth, with a tilt, or by momentum; "
         f"gate-and-rank and momentum-only rotations run beside them. Weekly decisions, trades at the next session's fill, 10.4 bps each way, always invested. Generated {audit['run_date']}. "
         "<b>Description, not a finding</b>: these series have been studied before.</p>",
         "<div class='card'>" + key + "</div>",
         "<h2>1 · How in sync are the assets?</h2><p class='note'>The desk's weekly supertrend of each asset, read at each decision week. Blue = BUY, red = SELL.</p>",
         "<div class='card'><div id='g_map' class='plot'></div></div><div class='card'><div id='g_breadth' class='plot'></div><div id='t_sync'></div></div>",
         "<h2>2 · The rules</h2><div id='rules'></div>",
         "<h2>3 · Every strategy</h2><p class='note' id='n_all'></p>",
         "<div class='card'><div id='g_bars' class='plot'></div></div><div class='card'><div id='t_all'></div></div>",
         "<h2>4 · The chosen strategy</h2>",
         "<div class='card'><div id='g_eq' class='plot'></div><div id='g_dd' class='plot'></div></div><div class='card'><div id='g_w' class='plot'></div></div><div class='card'><div id='g_null' class='plot'></div><div id='t_one'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Data.</b> India: analysis/approved_data.load_india_latest() (NSE price indices and MCX gold futures in rupees, from 2011-01-01 to 2026-11-20; the liquid ETF's level read as of each day, no recorded source before 2024-01-24, F-RA-D17). US: load_us_etf() (SPY, GCC; total return, closes only), load_us_small() (IJR) and load_us_gold() (GLD) from 2010-01-04. "
         "Each book on its assets' common calendar, which equals its reference pair's frame (SC100 ÷ gold; IJR ÷ GLD). The two books are never mixed.</li>"
         "<li><b>Signals.</b> The desk's Weekly LW supertrend on each asset's own weekly bars (E10's functions, checked there against the desk's code; US bars from closes); momentum on decision-week closes. Recomputed on histories cut at 8 dates per book and found equal.</li>"
         "<li><b>Book.</b> One book for any number of assets: weekly target weights traded at the next session's fills only when they change, selling before buying, 10.4 bps per side; it equals the engine's book on a two-asset run in each book and reproduces E10's S2 on the reference pair. In the Indian book the liquid ETF is a fifth holding: in any week whose decision falls in gold's weekly supertrend SELL it holds gold's weight, in every strategy, at the start and in every null shift (gold's SELL weeks fixed by the calendar); with the switch Rule R on the reference pair equals variants.book3. The benchmarks bought once are not switched.</li>"
         "<li><b>Nulls.</b> Every circular shift of the weekly signal panel (all assets together, so their agreement is kept) from 26 weeks to the period's length less 26: N-T slides supertrend and momentum together, N-P only the momentum.</li>"
         f"<li><b>Limits.</b> India about {(pd.Timestamp(audit['books']['india']['last']) - pd.Timestamp(audit['books']['india']['first'])).days / 365.25:.1f} years, the US about {(pd.Timestamp(audit['books']['us']['last']) - pd.Timestamp(audit['books']['us']['first'])).days / 365.25:.1f}; Indian price indices (no dividends) against gold futures; GCC is the only commodity ETF with the US book's history (PDBC from 2016 is the check); no cash in the US book, and in the Indian book the liquid ETF only through the switch; the gold leg has an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2). "
         "Twelve versions per book on seen data: some may look good by luck. Design written before each run: explore/E11_multi_asset/E11.md.</li></ul></div>",
         "</main>", f"<script id='e11-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E11_multi.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.nw{white-space:nowrap}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e11-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
// the reference palette's first four categorical slots, in order, for the four assets of a book; the strategy line is ink
function ACOL(){return isDark()?['#3987e5','#d95926','#199e70','#c98500','#9c6ade']:['#2a78d6','#eb6834','#1baf7a','#eda100','#8a57d2'];}   // the fifth: the liquid ETF (Indian book)
function SIG(){return isDark()?{buy:'#3987e5',sell:'#e66767'}:{buy:'#2a78d6',sell:'#e34948'};}
function FCOL(){return isDark()?{ST:'#3987e5',STMO:'#d95926',GR:'#199e70',MO:'#c98500',REF:'#898781'}:{ST:'#2a78d6',STMO:'#eb6834',GR:'#1baf7a',MO:'#eda100',REF:'#898781'};}
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
function lab(B,code){if(code==='REF-S2')return 'REF-S2 · '+B.ref_label;const s=D.strats.find(x=>x[0]===code);return s?s[0]+' · '+s[2]:code;}
function wts(B,q){return B.assets.map((a,j)=>pc(q['w'+(j+1)])).join(' / ');}
function drawSync(B){
  const c=th(), K=SIG(), Y=B.sync, t=B.t, A=B.sig;   // the signal assets (the liquid ETF has no supertrend)
  const z=A.map((a,j)=>B.states.map(r=>r[j]));
  Plotly.react('g_map',[{type:'heatmap',x:t,y:B.signames,z:z,zmin:-1,zmax:1,colorscale:[[0,K.sell],[0.5,K.sell],[0.5,K.buy],[1,K.buy]],showscale:false,ygap:3,
    customdata:z.map(r=>r.map(v=>v===1?'BUY':'SELL')),hovertemplate:'%{y} · %{x}: %{customdata}<extra></extra>'}],
    layout(two(B.label+' · the desk\'s weekly supertrend of each asset · '+span(t),'blue = BUY, red = SELL'),300,{margin:{l:170,r:24,t:90,b:40},xaxis:ax({type:'date'}),yaxis:ax({autorange:'reversed',showgrid:false})}),CFG);
  const ne=Y.breadth.length-1, xs=Y.breadth.map((v,j)=>j===0?'none':(j===ne?'all '+ne:j+' of '+ne));
  Plotly.react('g_breadth',[{type:'bar',x:xs,y:Y.breadth,marker:{color:c.ink2},text:Y.breadth.map(v=>pc(v)),textposition:'outside',cliponaxis:false,hovertemplate:'%{x} on BUY: %{text} of weeks<extra></extra>'}],
    layout(two(B.label+' · how many equity members ('+Y.equity_members.join(', ')+') are on BUY · share of weeks · '+Y.first+' → '+Y.last,'all agree in '+pc(Y.all_agree)+' of weeks'),320,{showlegend:false,margin:{l:60,r:24,t:90,b:40},xaxis:ax({type:'category'}),yaxis:ax({tickformat:'.0%',rangemode:'tozero'})}),CFG);
  const T=Y.turns, cr=Y.corr, ag=Y.agree, P=Object.entries(Y.patterns).slice(0,8);
  document.getElementById('t_sync').innerHTML="<p class='note'>Differs from the equity majority: "+Object.keys(Y.differs_from_majority).map(a=>a+' '+pc(Y.differs_from_majority[a])).join(', ')+' of weeks. Weeks on BUY: '+A.map(a=>a+' '+pc(Y.buy_share[a])).join(', ')+'; flips a year: '+A.map(a=>a+' '+f1(Y.flips_yr[a])).join(', ')+'.</p>'+
    table(['most common states','share of weeks'],P.map(([k,v])=>[k,pc(v)]))+
    table(['when the equity members turned together','times','first to flip (ties counted for each)','first alone'],Object.keys(T).map(k=>[k,T[k].n,Object.keys(T[k].first).map(a=>a+' '+T[k].first[a]).join(', '),Object.keys(T[k].alone_first).map(a=>a+' '+T[k].alone_first[a]).join(', ')]))+
    table(['same supertrend signal (share of weeks)'].concat(A),A.map((a,i)=>[a].concat(A.map((b,j)=>pc(ag[i][j])))))+
    table(['weekly return correlation'].concat(A),A.map((a,i)=>[a].concat(A.map((b,j)=>(cr[i][j]>=0?'+':'')+cr[i][j].toFixed(2)))));
}
function drawAll(B,code){
  const c=th(), F=FCOL(), S=D.strats;
  document.getElementById('rules').innerHTML=B.rules;
  document.getElementById('n_all').innerHTML='Excess = CAGR − the benchmark\'s CAGR over the same days ('+B.audit.first+' → '+B.audit.last+'). Percentiles: the real result among the signals slid in time by every shift of '+B.audit.shift_range_weeks[0]+'–'+B.audit.shift_range_weeks[1]+' weeks ('+B.audit.shifts+' shifts). N-T slides the supertrend states and the momentum together (the timing); N-P keeps the supertrend real and slides only the momentum (the pick). ≤ 5 or ≥ 95 = notable.';
  Plotly.react('g_bars',[{type:'bar',x:S.map(s=>s[0]),y:S.map(s=>B.st[s[0]].excess_neutral),marker:{color:S.map(s=>F[s[1]]),line:{color:S.map(s=>s[0]===code?c.ink:'rgba(0,0,0,0)'),width:2}},
    text:S.map(s=>fpp(B.st[s[0]].excess_neutral)+((B.st[s[0]].pct_T<=D.lo||B.st[s[0]].pct_T>=D.hi)?'*':'')),textposition:'outside',cliponaxis:false,
    customdata:S.map(s=>lab(B,s[0])+' ('+D.families[s[1]]+')<br>CAGR '+fp(B.st[s[0]].cagr)+', max drawdown '+fp(B.st[s[0]].maxdd,0)+'<br>timing percentile '+f1(B.st[s[0]].pct_T)),hovertemplate:'%{customdata}<extra></extra>'}],
    layout(two(B.label+' · every strategy · CAGR over neutral bought once (pp a year) · '+B.audit.first+' → '+B.audit.last,'colour = family (blue supertrend only, orange supertrend + momentum pick, aqua gate and rank, yellow momentum only, grey the two-asset reference); * = notable against the timing null; outlined = chosen'),440,
    {showlegend:false,margin:{l:60,r:24,t:110,b:80},xaxis:ax({tickangle:-30}),yaxis:ax({tickformat:'.1%',zeroline:true,zerolinecolor:c.axis,title:{text:'over neutral',font:{color:c.ink2}}})}),CFG);
  const Bs=B.bst;
  let html=table(['strategy','CAGR','over neutral','over 25% each','over the best asset','volatility','max drawdown','switches a year','average weight '+B.assets.join(' / '),'weeks all in one asset','N-T pct: CAGR / drawdown','N-P pct: CAGR / drawdown'],
    S.map(s=>{const q=B.st[s[0]];return [(s[0]===code?'<b>':'')+lab(B,s[0])+(s[0]===code?'</b>':''),fp(q.cagr),fpp(q.excess_neutral),fpp(q.excess_equal),fpp(q.excess_best_asset),pc(q.vol),fp(q.maxdd,0),f1(q.switches_yr),
      wts(B,q),pc(q.concentrated),pq(q.pct_T)+' / '+pq(q.pct_T_dd),pq(q.pct_P)+' / '+pq(q.pct_P_dd)];}))+
    "<p class='note'>Benchmarks over the same days: neutral bought once ("+B.assets.map((a,j)=>a+' '+(B.neutral[j]*100).toFixed(1)+'%').join(', ')+') '+fp(Bs.neutral.cagr)+' (max drawdown '+fp(Bs.neutral.maxdd,0)+'), 25% each '+fp(Bs.equal.cagr)+' ('+fp(Bs.equal.maxdd,0)+'), '+
    B.assets.map(a=>a+' '+fp(Bs[a].cagr)+' ('+fp(Bs[a].maxdd,0)+')').join(', ')+'. Ahead of neutral: '+B.audit.ahead_of_neutral+' of '+B.audit.n_strategies+'; of 25% each: '+B.audit.ahead_of_equal+'; of the best single asset: '+B.audit.ahead_of_best_asset+'.</p>';
  if(B.check)html+="<p class='note' style='margin-top:8px'><b>Check: the US book with PDBC instead of GCC</b> ("+B.check[0].first+' → '+B.check[0].last+', its neutral '+fp(B.check[0].neutral_cagr)+' a year; no nulls):</p>'+
    table(['strategy','CAGR','over its neutral','max drawdown'],B.check.map(x=>[x.code+' · '+x.label,fp(x.cagr),fpp(x.excess_neutral),fp(x.maxdd,0)]));
  document.getElementById('t_all').innerHTML=html;
}
function drawOne(B,code){
  const c=th(), AC=ACOL(), t=B.t, q=B.st[code], Bn=B.bench;
  const lines=[[B.curves[code],lab(B,code),c.ink,2.6,'solid'],[Bn.neutral,'neutral bought once',c.muted,1.6,'dash'],[Bn.equal,'25% each bought once',c.muted,1.4,'dot']].concat(B.assets.map((a,j)=>[Bn[a],a+' bought once',AC[j],1.1,'solid']));
  Plotly.react('g_eq',lines.map(([y,nm,col,wd,dash])=>({x:t,y:y,type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(B.label+' · '+lab(B,code)+' · value of 1 · '+span(t),'after costs; weekly decision closes; log scale · CAGR '+fp(q.cagr)+' against neutral '+fp(B.bst.neutral.cagr)),480,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_dd',lines.slice(0,3).map(([y,nm,col,wd,dash])=>({x:t,y:ddOf(y),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · weekly closes · '+span(t),'the strategy, neutral and 25% each'),300,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
  const W=B.w[code];
  Plotly.react('g_w',B.assets.map((a,j)=>({x:t,y:W.map(r=>r[j]),type:'scatter',mode:'lines',name:a,stackgroup:'one',line:{width:0.5,color:AC[j],shape:'hv'},fillcolor:AC[j],hovertemplate:a+': %{y:.0%}<extra></extra>'})),
    layout(two(B.label+' · '+lab(B,code)+' · target weights each week · '+span(t),'what the rule asks for at each decision (held weights drift between trades); average: '+B.assets.map((a,j)=>a+' '+pc(q['w'+(j+1)])).join(', ')),380,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({tickformat:'.0%',range:[0,1]})}),CFG);
  const tr=[{type:'histogram',x:B.nt[code].map(v=>v*100),name:'N-T: signals slid in time',marker:{color:isDark()?'rgba(195,194,183,0.55)':'rgba(82,81,78,0.45)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} shifts<extra></extra>'}];
  if(B.np[code]&&B.np[code].length)tr.push({type:'histogram',x:B.np[code].map(v=>v*100),name:'N-P: momentum slid, supertrend real',marker:{color:isDark()?'rgba(217,89,38,0.5)':'rgba(235,104,52,0.45)'},nbinsx:40,hovertemplate:'CAGR %{x:.1f}%: %{y} shifts<extra></extra>'});
  Plotly.react('g_null',tr,layout(two(B.label+' · '+lab(B,code)+' · its CAGR among the slid signals · '+B.audit.shifts+' shifts of '+B.audit.shift_range_weeks[0]+'–'+B.audit.shift_range_weeks[1]+' weeks','the line = the real CAGR '+fp(q.cagr)+' · percentile N-T '+f1(q.pct_T)+(q.pct_P!==null&&q.pct_P!==undefined?', N-P '+f1(q.pct_P):'')),360,
    {barmode:'overlay',xaxis:ax({ticksuffix:'%',title:{text:'CAGR',font:{color:c.ink2}}}),yaxis:ax({title:{text:'shifts',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'x',yref:'paper',x0:q.cagr*100,x1:q.cagr*100,y0:0,y1:1,line:{color:c.ink,width:2}}]}),CFG);
  document.getElementById('t_one').innerHTML=table(['','value'],[['dates',q.first+' → '+q.last],['CAGR / volatility / max drawdown',fp(q.cagr)+' / '+pc(q.vol)+' / '+fp(q.maxdd,0)],['over neutral / 25% each / the best asset',fpp(q.excess_neutral)+' / '+fpp(q.excess_equal)+' / '+fpp(q.excess_best_asset)],
    ['switches (a year)',q.switches+' ('+f1(q.switches_yr)+')'],['weeks with everything in one asset',pc(q.concentrated)+' ('+B.assets.map((a,j)=>a+' '+pc(q['top'+(j+1)])).join(', ')+')'],
    ['timing null N-T: percentile CAGR / drawdown; median CAGR of the shifts',pq(q.pct_T)+' / '+pq(q.pct_T_dd)+'; '+fp(q.null_T_median)],['pick null N-P: percentile CAGR / drawdown; median CAGR',q.pct_P===null||q.pct_P===undefined?'— (no momentum pick)':pq(q.pct_P)+' / '+pq(q.pct_P_dd)+'; '+fp(q.null_P_median)]]);
}
function render(){
  const bk=document.getElementById('book').value, code=document.getElementById('strat').value, B=D.books[bk];
  try{localStorage.setItem('e11-state',JSON.stringify({b:bk,s:code}));}catch(e){}
  drawSync(B); drawAll(B,code); drawOne(B,code);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  try{const st=JSON.parse(localStorage.getItem('e11-state')||'null');if(st){[['book','b'],['strat','s']].forEach(([id,k])=>{const el=document.getElementById(id);if(st[k]!==undefined&&Array.from(el.options).some(o=>o.value===String(st[k])))el.value=String(st[k]);});}}catch(e){}
  ['book','strat'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
