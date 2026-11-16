"""
E02 (EXPLORATORY, HINDSIGHT): peaks, troughs and trend segments of the ratio series with the desk's market-segregation method
(scipy topographic prominence -> local maxima / minima -> peak-to-peak regression -> merges), re-implemented here, on DAILY
ratios, with the trend measured in % a year. Design: explore/E02_ratio_segments/E02.md (written before the first run).

    python explore/E02_ratio_segments/segments.py   -> explore/E02_ratio_segments/out/

HINDSIGHT BY DESIGN: prominence looks to the right of each peak, the filter uses the average of all peaks, the local-maximum
step uses the next peak, and every regression spans its whole segment. Description of the past only; never an input to a rule.
"""
import html
import json
import math
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.stats import linregress

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles"))
from analysis.approved_data import fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (E01's point-in-time zig-zag for the cross-check, and the page's styling)

M_GRID = (1, 2, 3, 4, 5)                 # prominence multiplier; 5 = the code's default
T_GRID = (0.05, 0.10, 0.20)              # trend threshold, % a year; merge threshold = T / 2
REF = (3, 0.10)                          # reference setting for the charts: the middle of each grid, chosen by position
MIN_SEG = 30                             # minimum segment, rows (trading days), as the code
CODE = {"m": 5, "up": 30.0, "down": -30.0, "merge": 15.0}   # the code exactly: degrees on the rebased-to-100 series
MATCH_DAYS = 56                          # cross-check window: ±8 weeks
ORDER = E1.ORDER; DEFAULT_PAIR = E1.DEFAULT_PAIR; nice = E1.nice
UP_C, DOWN_C, SIDE_C = E1.AQUA, E1.ORANGE, E1.MUTED
STATE_COLOR = {"up": UP_C, "down": DOWN_C, "sideways": SIDE_C}


# ------------------------------------------------------------------------------------------------ peaks and troughs
def prominent(x, m):
    """Every local maximum of x with its topographic prominence; keep those with prominence >= m x the average."""
    idx, props = find_peaks(x, prominence=0); p = props["prominences"]
    return idx, p, (idx[p >= m * p.mean()] if len(p) else idx)


def local_extrema(x, keep):
    """Of the kept peaks, those higher than the kept peak before and after them (the first and last are dropped)."""
    return [int(keep[i]) for i in range(1, len(keep) - 1) if x[keep[i]] > x[keep[i - 1]] and x[keep[i]] > x[keep[i + 1]]]


def pivots(lv, m):
    _, pp, kp = prominent(lv, m); _, pt, kt = prominent(-lv, m)
    piv = sorted([(i, "peak") for i in local_extrema(lv, kp)] + [(i, "trough") for i in local_extrema(-lv, kt)])
    same = sum(1 for (_, u), (_, v) in zip(piv[:-1], piv[1:]) if u == v)
    return piv, {"local_peaks": len(pp), "local_troughs": len(pt), "kept_peaks": len(kp), "kept_troughs": len(kt), "pivots": len(piv), "not_alternating": same}


# ------------------------------------------------------------------------------------------------ segments
def fit_rescaled(lnr, ty, a, b):
    """ln(ratio) on time in years over rows a..b inclusive: trend in % a year (e^slope - 1), R^2, MSE, fitted line."""
    r = linregress(ty[a:b + 1], lnr[a:b + 1]); fitted = r.intercept + r.slope * ty[a:b + 1]
    return {"key": math.exp(r.slope) - 1, "r2": r.rvalue ** 2, "mse": float(np.mean((lnr[a:b + 1] - fitted) ** 2)), "fit": (r.intercept, r.slope)}


def fit_code(lv, a, b):
    """The code: rebased level on row number 1..n over rows a..b-1 (it drops the end row); angle in degrees."""
    n = max(b - a, 2); r = linregress(np.arange(1, n + 1), lv[a:a + n])
    return {"key": math.degrees(math.atan(r.slope)), "r2": r.rvalue ** 2, "mse": float("nan"), "fit": (r.intercept, r.slope)}


def classify(key, up, down): return "up" if key > up else ("down" if key < down else "sideways")


def segment_series(piv, fitter, up, down, merge_thr, code_lengths=False):
    segs = []
    for (a, _), (b, _) in zip(piv[:-1], piv[1:]):
        f = fitter(a, b); segs.append({"a": a, "b": b, "n": b - a + 1, **f})

    def merged(s1, s2):
        f = fitter(s1["a"], s2["b"]); return {"a": s1["a"], "b": s2["b"], "n": (s1["n"] + s2["n"]) if code_lengths else (s2["b"] - s1["a"] + 1), **f}
    changed = True                                   # 1. merge a segment shorter than MIN_SEG with the closer-trend neighbour
    while changed:
        changed = False
        for i in range(1, len(segs) - 1):
            if segs[i]["n"] < MIN_SEG:
                dn, dp = abs(segs[i]["key"] - segs[i + 1]["key"]), abs(segs[i]["key"] - segs[i - 1]["key"])
                if dn < dp:
                    segs[i:i + 2] = [merged(segs[i], segs[i + 1])]
                else:
                    segs[i - 1:i + 1] = [merged(segs[i - 1], segs[i])]
                changed = True; break
    changed = True                                   # 2. merge neighbours whose trends differ by no more than the merge threshold
    while changed:
        changed = False
        for i in range(1, len(segs) - 1):
            dn, dp = abs(segs[i]["key"] - segs[i + 1]["key"]), abs(segs[i]["key"] - segs[i - 1]["key"])
            if min(dn, dp) > merge_thr:
                continue
            if min(dn, dp) == dn:
                segs[i:i + 2] = [merged(segs[i], segs[i + 1])]
            else:
                segs[i - 1:i + 1] = [merged(segs[i - 1], segs[i])]
            changed = True; break
    for s in segs:
        s["state"] = classify(s["key"], up, down)
    return segs


def add_moves(segs, lv):
    """The total move over each segment, start turning point to end turning point (a short segment's % a year can be extreme)."""
    for s in segs:
        s["move"] = float(lv[s["b"]] / lv[s["a"]] - 1)
    return segs


def shares(segs):
    tot = sum(s["b"] - s["a"] for s in segs) or 1
    return {k: sum(s["b"] - s["a"] for s in segs if s["state"] == k) / tot for k in ("up", "down", "sideways")}


def agreement(piv_dates, e01, start):
    """Share of E02 pivots (inside E01's window) with an E01 turning point of the same type within ±MATCH_DAYS, and the reverse."""
    def near(d, t, other):
        return any(t == t2 and abs((d - d2).days) <= MATCH_DAYS for d2, t2 in other)
    mine = [(d, t) for d, t in piv_dates if d >= start]
    a = sum(near(d, t, e01) for d, t in mine) / len(mine) if mine else float("nan")
    b = sum(near(d, t, piv_dates) for d, t in e01) / len(e01) if e01 else float("nan")
    return a, b, len(mine), len(e01)


# ------------------------------------------------------------------------------------------------ the run
def main():
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily(); R = {}
    for pid, pair in enumerate(PAIRS):
        pf = pair_frame(px, pair, "plain"); s = pf["ratio"]; dates = s.index
        lv = (s / s.iloc[0] * 100).to_numpy(); lnr = np.log(s.to_numpy()); ty = ((dates - dates[0]).days / 365.25).to_numpy()
        # E01's point-in-time zig-zag turning points on the weekly ratio, for the cross-check
        wk = pf.loc[pf["decision"], "ratio"]; g = np.diff(np.log(wk.to_numpy())); _, _, _, z, hx = E1.point_in_time(g)
        e01 = [(wk.index[E1.BURN + i], "peak" if t > 0 else "trough") for i, t in E1.zigzag(np.concatenate([[0.0], np.cumsum(z)]).tolist(), hx.tolist())]
        e01_start = wk.index[E1.BURN]
        grid = {}
        for m in M_GRID:
            piv, pc = pivots(lv, m); pdates = [(dates[i], t) for i, t in piv]
            agr = agreement(pdates, e01, e01_start)
            for T in T_GRID:
                segs = add_moves(segment_series(piv, lambda a, b: fit_rescaled(lnr, ty, a, b), T, -T, T / 2), lv) if len(piv) >= 2 else []
                grid[(m, T)] = {"piv": piv, "pc": pc, "segs": segs, "shares": shares(segs), "agree": agr}
        piv_c, pc_c = pivots(lv, CODE["m"])
        segs_c = add_moves(segment_series(piv_c, lambda a, b: fit_code(lv, a, b), CODE["up"], CODE["down"], CODE["merge"], code_lengths=True), lv) if len(piv_c) >= 2 else []
        R[pair] = {"pid": pid, "s": s, "lnr": lnr, "ty": ty, "first": dates[0].date().isoformat(), "last": dates[-1].date().isoformat(), "days": len(s),
                   "grid": grid, "code": {"piv": piv_c, "pc": pc_c, "segs": segs_c, "shares": shares(segs_c), "agree": agreement([(dates[i], t) for i, t in piv_c], e01, e01_start)},
                   "e01": e01, "e01_start": e01_start.date().isoformat()}
        print(f"  {pair:18s} {R[pair]['first']} → {R[pair]['last']}  {len(s):5d} days  ref: {grid[REF]['pc']['pivots']} pivots, {len(grid[REF]['segs'])} segments  [{time.time() - t0:4.0f}s]", flush=True)

    # ------------------------------------------------------------ tables
    piv_rows, seg_rows, grid_rows = [], [], []
    for p in ORDER:
        r = R[p]; d = r["s"].index
        for (m, T), gset in list(r["grid"].items()) + [(("code", "code"), r["code"])]:
            sh = gset["shares"]; ag = gset["agree"]
            grid_rows.append({"pair": p, "setting": "as the code (m 5, ±30°, merge 15°)" if m == "code" else f"m {m}, T {T:.0%}", "m": m, "T": T, **{k: v for k, v in gset["pc"].items()},
                              "segments": len(gset["segs"]), "share_up": sh["up"], "share_down": sh["down"], "share_sideways": sh["sideways"],
                              "e02_pivots_matched_by_e01": ag[0], "e01_turns_matched_by_e02": ag[1], "e02_pivots_in_e01_window": ag[2], "e01_turns": ag[3]})
            if m != "code":
                for i, t in gset["piv"]:
                    piv_rows.append({"pair": p, "m": m, "T": T, "date": d[i].date().isoformat(), "type": t, "ratio": float(r["s"].iloc[i])})
            for sg in gset["segs"]:
                seg_rows.append({"pair": p, "setting": "code" if m == "code" else f"m{m}_T{T:g}", "start": d[sg["a"]].date().isoformat(), "end": d[sg["b"]].date().isoformat(),
                                 "trading_days": sg["n"], "move": sg["move"], "trend": sg["key"], "trend_unit": "degrees" if m == "code" else "% a year", "r2": sg["r2"], "state": sg["state"]})
    G = pd.DataFrame(grid_rows); G.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    pd.DataFrame(piv_rows).drop_duplicates(["pair", "m", "date", "type"]).drop(columns=["T"]).to_csv(os.path.join(OUT, "pivots.csv"), index=False)
    pd.DataFrame(seg_rows).to_csv(os.path.join(OUT, "segments.csv"), index=False)
    # turning points next to the unconfirmed gold roll of 2026-01-30 (F-RA-D-OPEN-2) may be data artefacts: flag them
    roll = pd.Timestamp("2026-01-30")
    near_roll = [f"{nice(p)} {R[p]['s'].index[i].date()} {t}" for p in ORDER if "GoldMCX" in p for i, t in R[p]["grid"][REF]["piv"] if abs((R[p]["s"].index[i] - roll).days) <= 5]
    code_all_side = sum(1 for p in ORDER for sg in R[p]["code"]["segs"] if sg["state"] == "sideways"); code_all = sum(len(R[p]["code"]["segs"]) for p in ORDER)
    code_max_angle = max((abs(sg["key"]) for p in ORDER for sg in R[p]["code"]["segs"]), default=float("nan"))
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY, HINDSIGHT", "inputs": fingerprints(), "pairs": PAIRS, "series": "plain ratio P_a / P_b, daily close on the pair's common calendar (pair_frame)",
             "method_source": "wm-strategy-analytics market_segregation/scipy_peak_finding/market_segments.py and parameters.py (read 2026-11-16), re-implemented",
             "grid": {"m": M_GRID, "T_per_year": T_GRID, "merge": "T / 2", "min_segment_rows": MIN_SEG}, "reference": {"m": REF[0], "T": REF[1]}, "code_exact": CODE,
             "code_exact_result": {"segments": code_all, "sideways": code_all_side, "max_abs_angle_deg": code_max_angle}, "cross_check_days": MATCH_DAYS, "turning_points_near_gold_roll": near_roll,
             "hindsight": ["prominence looks right of each peak", "filter uses the average prominence of all peaks", "local-maximum step uses the next peak", "regressions span whole segments"],
             "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    def pc(v): return "—" if v != v else f"{v:.0%}"
    def ref_row(p):
        gs = R[p]["grid"][REF]; sh = gs["shares"]; ag = gs["agree"]; last = gs["segs"][-1] if gs["segs"] else None
        return [nice(p), f"{R[p]['first']} → {R[p]['last']}", f"{gs['pc']['kept_peaks']} / {gs['pc']['kept_troughs']}", gs["pc"]["pivots"], len(gs["segs"]),
                f"{pc(sh['up'])} / {pc(sh['down'])} / {pc(sh['sideways'])}",
                (f"{last['state']}, {R[p]['s'].index[last['a']].date()} → {R[p]['s'].index[last['b']].date()} ({last['n']} days, move {last['move']:+.0%}, {last['key']:+.0%} a year)" if last else "—"),
                (f"{(R[p]['s'].index[-1] - R[p]['s'].index[last['b']]).days / 365.25:.1f} yrs (move {float(R[p]['s'].iloc[-1] / R[p]['s'].iloc[last['b']] - 1):+.0%})" if last else "—"), f"{pc(ag[0])} / {pc(ag[1])}"]
    HEAD = ["pair", "window", "prominent peaks / troughs kept", "turning points", "segments", "time up / down / sideways", "last segment (provisional)", "unclassified since the last turning point", "agreement with E01: E02→E01 / E01→E02"]
    md = ["# E02 results — hindsight segmentation of the ratio series (EXPLORATORY, HINDSIGHT)\n",
          f"_Generated by `explore/E02_ratio_segments/segments.py` on {audit['run_date']}. HINDSIGHT: every date below uses information from after it; description of the past only. Reference setting: prominence multiplier m = {REF[0]}, trend threshold T = {REF[1]:.0%} a year (merge T/2), minimum segment {MIN_SEG} trading days._\n",
          f"**The code exactly as written** (daily, m 5, ±30°, merge 15°): {code_all_side} of {code_all} segments across the ten pairs come out sideways; the steepest is {code_max_angle:.1f}°. Hence the trend in % a year below (E02.md, change A).\n",
          (f"**Data caution:** turning points within 5 days of the unconfirmed gold roll of 2026-01-30 (F-RA-D-OPEN-2) may be data artefacts: {'; '.join(near_roll)}.\n" if near_roll else ""),
          "## Per pair, reference setting\n", "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    md += ["| " + " | ".join(str(c) for c in ref_row(p)) + " |" for p in ORDER]
    md += ["\n## The grid: segments and time up / down / sideways, per pair\n"]
    for p in ORDER:
        md += [f"\n**{nice(p)}** ({R[p]['first']} → {R[p]['last']})\n", "| setting | prominent peaks / troughs | turning points | not alternating | segments | up / down / sideways | E02→E01 / E01→E02 |", "|---|---|---|---|---|---|---|"]
        for _, x in G[G["pair"] == p].iterrows():
            md.append(f"| {x['setting']} | {x['kept_peaks']} / {x['kept_troughs']} | {x['pivots']} | {x['not_alternating']} | {x['segments']} | {pc(x['share_up'])} / {pc(x['share_down'])} / {pc(x['share_sideways'])} | {pc(x['e02_pivots_matched_by_e01'])} / {pc(x['e01_turns_matched_by_e02'])} |")
    md += ["\n## Segments, reference setting\n", "_The trend a year is the fitted line's slope; on a short segment it can be extreme, so the total move from turning point to turning point is shown too. The first and last segments are never merged (as in the code), so they can be shorter than the minimum._\n",
           "| pair | start | end | trading days | move | trend a year | R² | state |", "|---|---|---|---|---|---|---|---|"]
    for p in ORDER:
        d = R[p]["s"].index
        md += [f"| {nice(p)} | {d[sg['a']].date()} | {d[sg['b']].date()} | {sg['n']} | {sg['move']:+.1%} | {sg['key']:+.1%} | {sg['r2']:.2f} | {sg['state']} |" for sg in R[p]["grid"][REF]["segs"]]
    open(os.path.join(OUT, "E02_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ------------------------------------------------------------ figures
    titles = {"seg": {}, "x": {}}
    fA, fB = go.Figure(), go.Figure()
    for p in ORDER:
        r = R[p]; s = r["s"]; d = s.index; gs = r["grid"][REF]; vis = p == DEFAULT_PAIR
        fA.add_trace(go.Scatter(x=d, y=s, mode="lines", line={"color": E1.BLUE, "width": 1.1}, name="ratio (daily close)", meta=p, visible=vis, hovertemplate="%{x|%Y-%m-%d}<br>ratio %{y:.4f}<extra></extra>"))
        shown = set()
        for sg in gs["segs"]:
            a, b = sg["a"], sg["b"]; ic, sl = sg["fit"]; yy = np.exp(ic + sl * r["ty"][a:b + 1])
            fA.add_trace(go.Scatter(x=d[a:b + 1], y=yy, mode="lines", line={"color": STATE_COLOR[sg["state"]], "width": 3.5}, name=f"segment: {sg['state']}", legendgroup=sg["state"],
                                    showlegend=sg["state"] not in shown, meta=p, visible=vis,
                                    hovertemplate=f"{sg['state']} · {d[a].date()} → {d[b].date()}<br>move {sg['move']:+.1%} over {sg['n']} days · trend {sg['key']:+.1%} a year · R² {sg['r2']:.2f}<extra></extra>"))
            shown.add(sg["state"])
        for t, sym in (("peak", "triangle-down"), ("trough", "triangle-up")):
            ii = [i for i, tt in gs["piv"] if tt == t]
            fA.add_trace(go.Scatter(x=d[ii], y=s.iloc[ii], mode="markers", marker={"symbol": sym, "size": 11, "color": E1.INK2, "line": {"color": "#fcfcfb", "width": 1.5}}, name=t,
                                    meta=p, visible=vis, hovertemplate=t + " %{x|%Y-%m-%d}<br>ratio %{y:.4f}<extra></extra>"))
        titles["seg"][p] = E1.two_line(f"{nice(p)} · {r['first']} → {r['last']} · HINDSIGHT", f"peaks, troughs and trend segments, m = {REF[0]}, T = {REF[1]:.0%} a year: {gs['pc']['pivots']} turning points, {len(gs['segs'])} segments")
        # cross-check timeline: E02 turning points (top row) vs E01's point-in-time zig-zag (bottom row)
        for row, lab, ev in ((1, "E02 (prominence, hindsight)", [(d[i], t) for i, t in gs["piv"]]), (0, "E01 zig-zag (point-in-time)", r["e01"])):
            for t, sym, col in (("peak", "triangle-down", E1.ORANGE), ("trough", "triangle-up", E1.AQUA)):
                xs = [dd for dd, tt in ev if tt == t]
                fB.add_trace(go.Scatter(x=xs, y=[row] * len(xs), mode="markers", marker={"symbol": sym, "size": 12, "color": col}, name=t, legendgroup=t, showlegend=row == 1,
                                        meta=p, visible=vis, hovertemplate=f"{lab}: {t} %{{x|%Y-%m-%d}}<extra></extra>"))
        ag = gs["agree"]
        titles["x"][p] = E1.two_line(f"{nice(p)} · the two methods' turning points", f"same type within ±8 weeks: {pc(ag[0])} of E02's (inside E01's window, from {r['e01_start']}) and {pc(ag[1])} of E01's")
    fA.update_yaxes(title={"text": "ratio", "font": {"color": E1.INK2}})
    figA = E1.style(fA, titles["seg"][DEFAULT_PAIR], 560, legend_top=True)
    fB.update_yaxes(tickvals=[0, 1], ticktext=["E01 zig-zag", "E02 prominence"], range=[-0.6, 1.6], showgrid=False)
    figB = E1.style(fB, titles["x"][DEFAULT_PAIR], 300, legend_top=True, left=120)
    # all pairs: state timeline, reference setting, month-end sampling
    months = pd.date_range("2006-01-31", "2026-08-31", freq="ME"); zz, hv = [], []
    for p in ORDER:
        r = R[p]; d = r["s"].index; st = pd.Series(np.nan, index=d)
        for sg in r["grid"][REF]["segs"]:
            st.iloc[sg["a"]:sg["b"] + 1] = {"up": 1.0, "down": -1.0, "sideways": 0.0}[sg["state"]]
        mm = st.reindex(months, method="ffill", tolerance=pd.Timedelta(days=7))
        zz.append(mm.to_list()); hv.append([({1.0: "up", -1.0: "down", 0.0: "sideways"}.get(v, "outside the segments") if v == v else "outside the segments") for v in mm])
    cs = [[0, DOWN_C], [0.333, DOWN_C], [0.334, SIDE_C], [0.666, SIDE_C], [0.667, UP_C], [1, UP_C]]
    fC = go.Figure(go.Heatmap(z=zz, x=months, y=[nice(p) for p in ORDER], zmin=-1, zmax=1, colorscale=cs, customdata=hv, meta="fixed", ygap=3,
                              hovertemplate="%{y} · %{x|%Y-%m}<br>%{customdata}<extra></extra>", colorbar={"tickvals": [-0.67, 0, 0.67], "ticktext": ["down", "sideways", "up"], "thickness": 12}))
    fC.update_yaxes(autorange="reversed", showgrid=False, automargin=True); fC.update_xaxes(showgrid=False)
    figC = E1.style(fC, E1.two_line("All ten pairs: trend state over time (HINDSIGHT)", f"m = {REF[0]}, T = {REF[1]:.0%} a year; up = the first leg ahead; blank = before the first or after the last turning point"), 470, legend=False, left=130)

    # ------------------------------------------------------------ page
    def per_pair(key, fig, tables, first=False):
        t = html.escape(json.dumps(titles[key]), quote=True)
        tb = "".join(f"<div data-pair='{html.escape(p, quote=True)}'{'' if p == DEFAULT_PAIR else ' hidden'}>{tables[p]}</div>" for p in ORDER)
        return f"<div class='card per-pair' data-titles='{t}'>{E1.html_fig(fig, first=first)}<details><summary>Table view</summary>{tb}</details></div>"
    tab_seg = {p: E1.tbl(["start", "end", "trading days", "move", "trend a year", "R²", "state"], [[str(R[p]["s"].index[sg["a"]].date()), str(R[p]["s"].index[sg["b"]].date()), sg["n"], f"{sg['move']:+.1%}", f"{sg['key']:+.1%}", f"{sg['r2']:.2f}", sg["state"]] for sg in R[p]["grid"][REF]["segs"]]) for p in ORDER}
    tab_x = {p: E1.tbl(["E02 turning points (date, type)", "E01 zig-zag turning points (date, type)"], [["<br>".join(f"{R[p]['s'].index[i].date()} {t}" for i, t in R[p]["grid"][REF]["piv"]), "<br>".join(f"{dd.date()} {t}" for dd, t in R[p]["e01"])]]) for p in ORDER}
    grid_tables = "".join(f"<div data-pair='{html.escape(p, quote=True)}'{'' if p == DEFAULT_PAIR else ' hidden'}>" + E1.tbl(
        ["setting", "prominent peaks / troughs", "turning points", "two in a row", "segments", "time up / down / sideways", "agreement E02→E01 / E01→E02"],
        [[("<b>" + x["setting"] + "</b>") if (x["m"] == REF[0] and x["T"] == REF[1]) else x["setting"], f"{x['kept_peaks']} / {x['kept_troughs']}", x["pivots"], x["not_alternating"], x["segments"],
          f"{pc(x['share_up'])} / {pc(x['share_down'])} / {pc(x['share_sideways'])}", f"{pc(x['e02_pivots_matched_by_e01'])} / {pc(x['e01_turns_matched_by_e02'])}"] for _, x in G[G["pair"] == p].iterrows()]) + "</div>" for p in ORDER)
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    JS = E1.JS.replace("if(t.type==='heatmap')", "if(t.type==='heatmap'&&t.meta!=='fixed')").replace("__DIV_LIGHT__", json.dumps(E1.DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(E1.DIV_DARK)).replace("e01-pair", "e02-pair")
    H = [E1.PAGE_HEAD.replace("<title>Ratio cycles</title>", "<title>Ratio segments</title>").replace("__CSS__", E1.CSS + ".warn{border-left:4px solid #e34948}"),
         "<header class='bar'><div class='bar-inner'><b>E02 · ratio segments</b><span class='tag'>HINDSIGHT</span><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label><button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Where are the peaks, troughs and trends of the ratio series?</h1>",
         f"<p>The desk's market-segregation method — scipy topographic prominence to find the significant peaks and troughs, then a regression between consecutive turning points to call each stretch up, down or sideways — applied to the ten daily ratios P_a ÷ P_b. Generated {audit['run_date']} from the approved data of 2026-11-12. Reference setting: prominence multiplier m = {REF[0]}, trend threshold T = {REF[1]:.0%} a year, chosen as the middle of the grid before the run.</p>",
         "<div class='card warn'><p style='margin:4px 0'><b>Hindsight, by design.</b> A peak's prominence depends on how far the ratio falls <i>after</i> it; the filter compares it with the average of all peaks; a peak is kept only if it beats the next one; each trend line spans its whole segment. So every turning point and segment here is drawn with knowledge of what came later — a description of the past, never a signal, and the latest segment can still change as data is added. E01's zig-zag is the point-in-time counterpart; the cross-check below compares them.</p></div>",
         f"<div class='card'><p style='margin:4px 0'><b>Why the trend is in % a year:</b> the code calls a stretch a trend when its regression line on the series rebased to 100 is steeper than ±30° — 145 points a year on daily data, a pace no ratio reaches. Run exactly as written it labels <b>{code_all_side} of {code_all}</b> segments across the ten pairs sideways (steepest {code_max_angle:.1f}°). Here the line is fitted to ln(ratio) against time, so its slope is the stretch's trend in % a year at any level; up beyond +T, down beyond −T, and neighbours within T ÷ 2 merge.</p></div>",
         "<h2>What each pair shows (reference setting)</h2><div class='card'>" + E1.tbl(HEAD, [ref_row(p) for p in ORDER], cls="txt")
         + (f"<p class='note'><b>Data caution:</b> turning points within 5 days of the unconfirmed gold roll of 2026-01-30 (open item F-RA-D-OPEN-2, a −12.4% gold day on thin volume) may be artefacts of the data, not market turns: {'; '.join(near_roll)}.</p>" if near_roll else "") + "</div>",
         "<h2>The chosen pair</h2><p class='note'>Pick the pair at the top. Thick lines are the segments' trend lines (aqua up, orange down, grey sideways); triangles are the turning points that bound them.</p>",
         per_pair("seg", figA, tab_seg, first=True),
         "<h3>Against E01's point-in-time zig-zag</h3><p class='note'>Top row: this method's turning points (hindsight). Bottom row: E01's zig-zag on the weekly, drift-removed ratio, each one confirmed only after the reversal. Where both rows have a triangle of the same kind within a few weeks, the two methods see the same swing.</p>",
         per_pair("x", figB, tab_x),
         "<h3>The grid: how much the answer depends on the settings</h3><p class='note'>Every setting for the chosen pair, plus the code exactly as written. A higher prominence multiplier keeps fewer, bigger turning points; a higher trend threshold calls more of the time sideways. 'Two in a row' counts consecutive peaks or consecutive troughs, which the method allows.</p>",
         "<div class='card'>" + grid_tables + "</div>",
         "<h2>All pairs at once</h2><div class='card'>" + E1.html_fig(figC) + "</div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Turning points.</b> scipy.signal.find_peaks on the daily ratio (rebased to 100) gives every local maximum and its topographic prominence: its height above the higher of the two lowest points reached on either side before higher ground or the end of the series. Keep those with prominence ≥ m × the average prominence; then keep each kept peak that is higher than the kept peaks before and after it. Troughs: the same on the negated series.</li>"
         f"<li><b>Segments.</b> Consecutive turning points bound a segment. Fit ln(ratio) on time in years; trend = e^slope − 1 a year; up if > +T, down if < −T, else sideways. Merge a segment shorter than {MIN_SEG} trading days into the neighbour with the closer trend; then merge neighbours whose trends differ by ≤ T ÷ 2; repeat until nothing changes (first and last segments only merge as neighbours, as in the code, so they can stay shorter than the minimum and their % a year can be extreme — the total move is shown beside it). Nothing after the last turning point belongs to a segment.</li>"
         f"<li><b>Grid.</b> m ∈ {{{', '.join(str(m) for m in M_GRID)}}}, T ∈ {{{', '.join(f'{t:.0%}' for t in T_GRID)}}} a year; reference m = {REF[0]}, T = {REF[1]:.0%}. There is no hand-labelled truth for these ratios, so the grid is shown, not optimised.</li>"
         "<li><b>Differences from the code.</b> Trend in % a year instead of degrees (above); merge threshold in the same units; a segment's regression includes its end day; a merged segment does not count the shared day twice. The code was read from wm-strategy-analytics (market_segregation/scipy_peak_finding) and re-implemented, not run.</li>"
         f"<li><b>Cross-check.</b> Same type within ±{MATCH_DAYS // 7} weeks, counted only inside E01's window (E01 starts after its 52-week warm-up).</li>"
         "<li><b>Unclassified tails.</b> The local-maximum step keeps a peak only if it is higher than the kept peaks on both sides, so in a long, steady trend every intermediate peak is dropped, and the first and last kept peaks and troughs never qualify. A trend still running at the end of the window can therefore be left out of every segment — the summary's 'unclassified since the last turning point' column shows how much, per pair.</li>"
         "<li><b>Limits.</b> Hindsight throughout; the prominence filter is relative to each pair's own peaks, so the count depends on the window; the Micro250 pairs cover 2022–2026; the gold leg has an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2). Design written before the run: explore/E02_ratio_segments/E02.md.</li></ul></div>",
         "</main>", "<script>" + JS + "</script></body></html>"]
    out = os.path.join(OUT, "E02_segments.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; code exactly as written: {code_all_side} of {code_all} segments sideways (max {code_max_angle:.1f}°) → {out}")


if __name__ == "__main__":
    main()
