"""
E01 (EXPLORATORY): cyclicality in the ratio series. Design: explore/E01_ratio_cycles/E01.md (written before the first run).
Reads the approved data only through analysis/approved_data, on each pair's common calendar and decision weeks exactly as
T02/T03 build them (analysis.ratio_engine.pair_frame). Writes explore/E01_ratio_cycles/out/:
E01_cycles.html (the page), E01_results.md (every table), summary.csv, spectrum_bands.csv, variance_ratios.csv,
seasonality.csv, swings.csv, drift_check.csv (expanding drift removal vs none), audit.json.

Point-in-time throughout (changed after the first run; E01.md "Added after the run"): the drift removed at week t is the average
of the weekly changes BEFORE t and the zig-zag threshold at t comes from their sd (expanding windows, after a 52-week
warm-up); nothing from later in the history enters any series, threshold or random walk.

    python explore/E01_ratio_cycles/cycles.py [--draws 2000]

Not a finding and not a test of any rule: every number is description against a random-walk comparison.
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT)
from analysis.approved_data import ASSETS, GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import INDEX_PAIRS, MIN_OBS_FRAC, PAIRS, pair_frame  # noqa: E402
from analysis.ratio_engine import PIT as EnginePIT, decide  # noqa: E402  (z check only: the band's mean and sd columns are read)
sys.path.insert(0, HERE)
import zwin as ZW  # noqa: E402  (the z windows: weekly 13 / 26 / 52, daily 20 / 60 / 120 / 250)

BANDS = ((2, 4), (4, 8), (8, 13), (13, 26), (26, 52), (52, 104), (104, 208), (208, 520))   # cycle-length bands, weeks
VR_K = (2, 4, 8, 13, 26, 52, 104)
MIN_CYCLES, MIN_K_FRAC, MIN_YEARS_MONTH = 3, 4, 8     # a period must fit 3 times; k <= weeks / 4; 8 years in every month
H_WEEKS = 26                                          # zig-zag threshold = sd of earlier weeks * sqrt(26): one typical half-year move
BURN = 52                                             # warm-up: the expanding drift and sd start from a year of earlier weeks
SEED_SIGNS, SEED_MONTHS = 101, 104
LO, HI = 5.0, 95.0                                    # notable: outside the random walks' middle 90%
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ORDER = INDEX_PAIRS + [p for p in PAIRS if p not in INDEX_PAIRS]
DEFAULT_PAIR = "N100|SC100"
Z_WIN = 52                                            # z-score: T02's band window (weekly closes, this week included)
Z_MIN = int(round(MIN_OBS_FRAC * Z_WIN))              # first z once 42 weeks exist, as the band
Z_TESTED = (("beyond1", "weeks beyond ±1"), ("beyond2", "weeks beyond ±2"), ("med_str", "median stretch beyond ±1"),
            ("z_peak", "median z at peaks"), ("z_trough", "median z at troughs"))   # the z statistics given a percentile

# palette: the dataviz reference instance (slots 1-3 validated all-pairs in both modes), chrome tokens light / dark
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CAT = {"N100": "#2a78d6", "MC100": "#eb6834", "SC100": "#1baf7a", "Micro250": "#eda100", GOLD: "#e87ba4"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NULL_FILL, BAND_FILL = "rgba(137,135,129,0.24)", "rgba(137,135,129,0.18)"
# diverging blue <-> red with a grey midpoint; the middle (20th-80th percentile) stays near grey so that only the tails
# (what a random walk rarely gives) carry strong colour
DIV_LIGHT = [[0, "#104281"], [0.05, "#2a78d6"], [0.2, "#cde2fb"], [0.5, "#f0efec"], [0.8, "#f9d3d2"], [0.95, "#e34948"], [1, "#a32b2a"]]
DIV_DARK = [[0, "#86b6ef"], [0.05, "#3987e5"], [0.2, "#34465e"], [0.5, "#383835"], [0.8, "#5e3a3a"], [0.95, "#e66767"], [1, "#f29a9a"]]
# the z timeline: the same hues on a -3 .. +3 scale with stops at z = ±1 (the band's edges) and ±2
Z_LIGHT = [[0, "#104281"], [1 / 6, "#2a78d6"], [1 / 3, "#cde2fb"], [0.5, "#f0efec"], [2 / 3, "#f9d3d2"], [5 / 6, "#e34948"], [1, "#a32b2a"]]
Z_DARK = [[0, "#86b6ef"], [1 / 6, "#3987e5"], [1 / 3, "#34465e"], [0.5, "#383835"], [2 / 3, "#5e3a3a"], [5 / 6, "#e66767"], [1, "#f29a9a"]]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def nice(p): return p.replace("|", " ÷ ")


def pctile(x, null) -> float:
    null = np.asarray(null, float); null = null[~np.isnan(null)]
    if x is None or np.isnan(x) or len(null) == 0:
        return float("nan")
    return float(((null < x).sum() + 0.5 * (null == x).sum()) / len(null) * 100)


def notable(p) -> bool: return (p == p) and (p <= LO or p >= HI)


# ------------------------------------------------------------------------------------------------------------ the checks
def periodogram(x):
    n = x.shape[-1]
    return (np.abs(np.fft.rfft(x, axis=-1)) ** 2 / n)[..., 1:]             # Fourier bins j = 1 .. n//2


def spectrum(z, Z) -> dict:
    n = len(z); j = np.arange(1, n // 2 + 1); period = n / j; ok = j >= MIN_CYCLES
    P, PN = periodogram(z), periodogram(Z); tot, totN = P.sum(), PN.sum(axis=1)
    bands, lr_real, lr_null = [], [], []
    for lo, hi in BANDS:
        m = ok & (period >= lo) & (period < hi)
        row = {"band": f"{lo}–{hi}", "lo": lo, "hi": hi, "bins": int(m.sum())}
        if m.sum():
            s = P[m].sum() / tot; sN = PN[:, m].sum(axis=1) / totN; med = float(np.median(sN))
            row.update({"share": float(s), "null_median": med, "ratio": float(s / med), "pct": pctile(s, sN)})
            lr_real.append(np.log(s / med)); lr_null.append(np.log(sN / med))
        bands.append(row)
    stat = float(np.max(np.abs(lr_real))); statN = np.max(np.abs(np.stack(lr_null, axis=1)), axis=1)
    p_global = float((1 + (statN >= stat).sum()) / (len(statN) + 1))
    rep = [b for b in bands if b["bins"]]; top = max(rep, key=lambda b: abs(np.log(b["ratio"])))
    # display curve: power averaged over a ±12% window of cycle length (constant width on the log axis), real and every path alike
    # (widened where needed so that every point averages at least 3 frequencies; long cycle lengths have few)
    per = period[ok]; grid = np.exp(np.linspace(np.log(2), np.log(per.max()), 90))
    dist = np.abs(np.log(per)[None, :] - np.log(grid)[:, None]); half = np.maximum(0.12, np.sort(dist, axis=1)[:, min(2, dist.shape[1] - 1)])
    W = (dist <= half[:, None]).astype(float); keep = W.sum(axis=1) > 0
    W = W[keep] / W[keep].sum(axis=1, keepdims=True); Ps, PNs = P[ok] @ W.T, PN[:, ok] @ W.T; medc = np.median(PNs, axis=0)
    curve = {"period": grid[keep], "ratio": Ps / medc, "lo": np.percentile(PNs, 5, axis=0) / medc, "hi": np.percentile(PNs, 95, axis=0) / medc}
    return {"bands": bands, "p_global": p_global, "stat": stat, "top": top, "curve": curve, "max_period": float(n / MIN_CYCLES)}


def variance_ratios(z, Z) -> list:
    def vr(x):
        C = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1); v1 = (x ** 2).mean(axis=-1)
        return np.stack([((C[..., k:] - C[..., :-k]) ** 2).mean(axis=-1) / (k * v1) for k in VR_K], axis=-1)
    real, null = vr(z), vr(Z); n = len(z); out = []
    for i, k in enumerate(VR_K):
        rep = k <= n / MIN_K_FRAC
        out.append({"k": k, "reportable": rep, "vr": float(real[i]), "pct": pctile(real[i], null[:, i]) if rep else float("nan"),
                    "null_lo": float(np.percentile(null[:, i], 5)), "null_hi": float(np.percentile(null[:, i], 95)), "null_median": float(np.median(null[:, i]))})
    return out


def monthly(pf: pd.DataFrame) -> pd.Series:
    me = pf["ratio"].groupby(pf.index.to_period("M")).last()
    return np.log(me).diff().dropna()


def seasonality(m: pd.Series, rng, D: int) -> dict:
    vals = m.to_numpy(); mon = m.index.month.to_numpy(); yr = m.index.year.to_numpy()
    real = np.array([vals[mon == k].mean() for k in range(1, 13)]); cnt = np.array([(mon == k).sum() for k in range(1, 13)])
    up = np.array([(vals[mon == k] > 0).mean() for k in range(1, 13)])
    M = np.empty((D, len(vals)))
    for y in np.unique(yr):
        idx = np.where(yr == y)[0]; M[:, idx] = rng.permuted(np.tile(vals[idx], (D, 1)), axis=1)
    null = np.stack([M[:, mon == k].mean(axis=1) for k in range(1, 13)], axis=1)
    disp, dispN = float(real.std()), null.std(axis=1)
    return {"reportable": bool(cnt.min() >= MIN_YEARS_MONTH), "months": [{"month": MONTHS[k], "mean": float(real[k]), "share_up": float(up[k]), "years": int(cnt[k]),
            "pct": pctile(real[k], null[:, k]), "null_lo": float(np.percentile(null[:, k], 5)), "null_hi": float(np.percentile(null[:, k], 95))} for k in range(12)],
            "dispersion": disp, "p": float((1 + (dispN >= disp).sum()) / (D + 1)), "first": str(m.index[0]), "last": str(m.index[-1]), "n_months": int(len(m))}


def zigzag(x: list, h: list) -> list:
    """Turning points (index, +1 peak / -1 trough), each confirmed at point i by a reversal of at least h[i] from the extreme
    since the last one; h[i] is known at point i (it comes from the weeks before it)."""
    tps = []; trend = 0; hi_i = lo_i = 0; ext = 0
    for i in range(1, len(x)):
        v = x[i]; hh = h[i]
        if trend == 0:
            if v > x[hi_i]:
                hi_i = i
            if v < x[lo_i]:
                lo_i = i
            if x[hi_i] - x[lo_i] >= hh:
                if hi_i > lo_i:
                    tps.append((lo_i, -1)); trend = 1; ext = hi_i
                else:
                    tps.append((hi_i, 1)); trend = -1; ext = lo_i
        elif trend == 1:
            if v > x[ext]:
                ext = i
            elif x[ext] - v >= hh:
                tps.append((ext, 1)); trend = -1; ext = i
        else:
            if v < x[ext]:
                ext = i
            elif v - x[ext] >= hh:
                tps.append((ext, -1)); trend = 1; ext = i
    return tps


def swing_stats(x: list, h: list):
    tps = zigzag(x, h); d = np.diff([i for i, _ in tps])
    if len(d) < 3:
        return len(d), float("nan"), float("nan"), tps, d
    return len(d), float(np.median(d)), float(d.std() / d.mean()), tps, d


def swings(z, Z, dates, hx) -> dict:
    """z: the weekly residuals; hx[i]: the threshold at point i of the path x = cumsum(z) (x[0] = 0), from earlier weeks only.
    The same threshold path is used for every random walk."""
    x = np.concatenate([[0.0], np.cumsum(z)]); hl = [float(v) for v in hx]
    n_sw, med, cv, tps, d = swing_stats(x.tolist(), hl)
    X = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1); nN, medN, cvN, pooled, tpsN = [], [], [], [], []
    for row in X:
        a, b, c, tp, dd = swing_stats(row.tolist(), hl); nN.append(a); medN.append(b); cvN.append(c); pooled.append(dd); tpsN.append(tp)
    amp = np.abs(np.diff(x[[i for i, _ in tps]])) if len(tps) > 1 else np.array([])
    hp = np.exp(np.asarray(hl)) - 1
    return {"h_pct_median": float(np.median(hp)), "h_pct_min": float(hp.min()), "h_pct_max": float(hp.max()), "h_pct_last": float(hp[-1]), "x": x, "tps": tps, "durations": d, "n": n_sw, "median": med, "cv": cv, "median_amp_pct": float(np.exp(np.median(amp)) - 1) if len(amp) else float("nan"),
            "pct_n": pctile(n_sw, nN), "pct_median": pctile(med, medN), "pct_cv": pctile(cv, cvN), "null_n_median": float(np.median(nN)), "null_median_median": float(np.nanmedian(medN)),
            "null_cv_median": float(np.nanmedian(cvN)), "pooled": np.concatenate(pooled) if pooled else np.array([]), "dates": dates, "null_tps": tpsN}


def point_in_time(g):
    """The expanding estimates at each week t >= BURN, from g[0 .. t-1] only: drift m, sd, residual z = g[t] - m, and the
    zig-zag threshold at each point of the path x[0 .. len(z)] (x[0] = 0 at the warm-up's last week)."""
    cs, cs2 = np.concatenate([[0.0], np.cumsum(g)]), np.concatenate([[0.0], np.cumsum(g * g)])
    t = np.arange(BURN, len(g)); k = t.astype(float)
    m = cs[t] / k                                              # expanding drift: mean of g[0 .. t-1]
    sd = np.sqrt((cs2[t] - k * m * m) / (k - 1))               # expanding sd (n - 1) of g[0 .. t-1]
    z = g[t] - m
    hx = np.concatenate([[sd[0]], sd]) * np.sqrt(H_WEEKS)
    return t, m, sd, z, hx


def pit_check(g, n_cuts=12):
    """Look-ahead check: everything recomputed from the history cut at T must equal the full run's values up to T, and the
    turning points found on the cut history must be the first turning points of the full run. Returns (checks, violations)."""
    _, m, sd, z, hx = point_in_time(g); full = zigzag(np.concatenate([[0.0], np.cumsum(z)]).tolist(), hx.tolist())
    checks, bad = 0, []
    for T in np.unique(np.linspace(BURN + 10, len(g) - 1, n_cuts).astype(int)):
        _, mc, sdc, zc, hxc = point_in_time(g[:T]); L = len(zc); checks += 2
        if not (np.array_equal(mc, m[:L]) and np.array_equal(sdc, sd[:L]) and np.array_equal(zc, z[:L]) and np.array_equal(hxc, hx[:L + 1])):
            bad.append(f"estimates differ when the history is cut at week {T}")
        tc = zigzag(np.concatenate([[0.0], np.cumsum(zc)]).tolist(), hxc.tolist())
        if tc != full[:len(tc)]:
            bad.append(f"turning points differ when the history is cut at week {T}")
    return checks, bad


# ------------------------------------------------------------------------------------------------------------ the z-score
def band_z(wk: pd.Series) -> pd.Series:
    """z_t = (r_t - mean of the last Z_WIN weekly closes including t) / their sd (n - 1); NaN until Z_MIN weeks exist.
    The same mean and sd as T02's band, so z = ±1 is the band's edge. Only weeks up to t enter z_t."""
    roll = wk.rolling(Z_WIN, min_periods=Z_MIN)
    return (wk - roll.mean()) / roll.std()


def trailing_z(paths: np.ndarray) -> np.ndarray:
    """band_z for many paths at once (rows = paths, columns = weeks), from running sums. Each path is divided by its first
    value (z does not change) so the sums stay near 1; checked against band_z on the real ratio every run."""
    r = np.atleast_2d(paths); r = r / r[:, :1]; L = r.shape[1]
    cs = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r * r, axis=1)], axis=1)
    t = np.arange(L); lo = np.maximum(0, t - Z_WIN + 1); k = (t - lo + 1).astype(float)
    m = (cs[:, t + 1] - cs[:, lo]) / k; var = (cs2[:, t + 1] - cs2[:, lo] - k * m * m) / np.maximum(k - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):         # the first weeks (k < Z_MIN) divide 0 by 0 and are blanked
        z = (r - m) / np.sqrt(np.maximum(var, 0))
    z[:, k < Z_MIN] = np.nan
    return z


def z_pit_check(wk: pd.Series, n_cuts=12):
    """Look-ahead check for the z-score: recomputed from the history cut at T, it must equal the full run's up to T."""
    full = band_z(wk).to_numpy(); checks, bad = 0, []
    for T in np.unique(np.linspace(Z_MIN + 10, len(wk) - 1, n_cuts).astype(int)):
        checks += 1
        if not np.allclose(band_z(wk.iloc[:T]).to_numpy(), full[:T], rtol=0, atol=1e-12, equal_nan=True):
            bad.append(f"z differs when the history is cut at week {T}")
    return checks, bad


def stretches(z: np.ndarray) -> np.ndarray:
    """Lengths in weeks of the runs of consecutive weeks with z > +1 and of those with z < -1 (cut at the window's ends)."""
    out = []
    for b in (z > 1, z < -1):
        d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
        out.append(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return np.concatenate(out)


def z_stats(z: np.ndarray, tps) -> dict:
    """Description of one z path over the window; tps = zig-zag turning points [(index into the window, +1 / -1)]."""
    s = stretches(z); pk = [z[i] for i, t in tps if t > 0]; tr = [z[i] for i, t in tps if t < 0]
    return {"beyond1": float(np.mean(np.abs(z) > 1)), "above1": float(np.mean(z > 1)), "below1": float(np.mean(z < -1)), "beyond2": float(np.mean(np.abs(z) > 2)),
            "n_str": int(len(s)), "med_str": float(np.median(s)) if len(s) else float("nan"), "max_str": int(s.max()) if len(s) else 0,
            "z_peak": float(np.median(pk)) if pk else float("nan"), "z_trough": float(np.median(tr)) if tr else float("nan")}


def z_pack(zs, zw, wdates, zreal, znull, hist_real, hist_null, tested) -> dict:
    """One window's z description: the series, the real statistics over the window, the random walks' median / 5th / 95th
    and the percentile of each tested statistic, the lowest / highest / latest z with dates, the histograms."""
    zcol = {k: np.array([x[k] for x in znull], float) for k in zreal}; iw = {"min": int(np.nanargmin(zw)), "max": int(np.nanargmax(zw))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"series": zs, "real": zreal, "first_z": zs.first_valid_index().date().isoformat(),
                "null_median": {k: float(np.nanmedian(v)) for k, v in zcol.items()}, "null_lo": {k: float(np.nanpercentile(v, 5)) for k, v in zcol.items()},
                "null_hi": {k: float(np.nanpercentile(v, 95)) for k, v in zcol.items()}, "pct": {k: pctile(zreal[k], zcol[k]) for k in tested},
                "min": (float(zw[iw["min"]]), wdates[iw["min"]].date().isoformat()), "max": (float(zw[iw["max"]]), wdates[iw["max"]].date().isoformat()),
                "last": (float(zw[-1]), wdates[-1].date().isoformat()), "hist_real": hist_real, "hist_null": hist_null}


def zwin_pit(s_: pd.Series, n: int, n_cuts=12):
    full = ZW.z_of(s_, n).to_numpy(); checks, bad = 0, []
    for T in np.unique(np.linspace(ZW.min_obs(n) + 10, len(s_) - 1, n_cuts).astype(int)):
        checks += 1
        if not np.allclose(ZW.z_of(s_.iloc[:T], n).to_numpy(), full[:T], rtol=0, atol=1e-12, equal_nan=True):
            bad.append(f"z ({n}) differs when the history is cut at {T}")
    return checks, bad


def verdict(r) -> dict:
    vr = [v for v in r["vr"] if v["reportable"]]
    tr = [v["k"] for v in vr if v["pct"] >= HI]; rv = [v["k"] for v in vr if v["pct"] <= LO]
    t_vr = "; ".join(x for x in (("continues at " + ", ".join(map(str, tr)) + " wk") if tr else "", ("reverses at " + ", ".join(map(str, rv)) + " wk") if rv else "") if x) or "like a random walk at every horizon"
    sp = r["spec"]; top = sp["top"]
    t_sp = (f"differs from a random walk (p {sp['p_global']:.3f}); most {'excess' if top['ratio'] > 1 else 'deficit'} at {top['band']} wk" if sp["p_global"] <= 0.05
            else f"no cycle length stands out (p {sp['p_global']:.2f})")
    se = r["season"]
    t_se = ("too short to test (" + str(min(m["years"] for m in se["months"])) + " years)") if not se["reportable"] else (f"month pattern (p {se['p']:.3f})" if se["p"] <= 0.05 else f"no month pattern (p {se['p']:.2f})")
    sw = r["swing"]; extra = []
    if sw["pct_cv"] == sw["pct_cv"] and sw["pct_cv"] <= LO:
        extra.append("more regular than random")
    if sw["pct_n"] >= HI:
        extra.append("more swings than random")
    elif sw["pct_n"] <= LO:
        extra.append("fewer swings than random")
    t_sw = f"{sw['n']} swings, median {f2(sw['median'], 0)} wk" + (" — " + ", ".join(extra) if extra else "")
    zz = r["z"]; zr, zn, zp = zz["real"], zz["null_median"], zz["pct"]
    words = {"beyond1": ("beyond ±1 less often than random", "beyond ±1 more often than random"), "beyond2": ("beyond ±2 less often than random", "beyond ±2 more often than random"),
             "med_str": ("shorter stretches than random", "longer stretches than random"), "z_peak": ("peaks at a lower z than random", "peaks at a higher z than random"),
             "z_trough": ("troughs at a lower z than random", "troughs at a higher z than random")}
    zx = [words[k][0] if zp[k] <= LO else words[k][1] for k, _ in Z_TESTED if notable(zp[k])]
    t_z = (f"beyond ±1 in {zr['beyond1'] * 100:.0f}% of weeks (random walks {zn['beyond1'] * 100:.0f}%), beyond ±2 in {zr['beyond2'] * 100:.0f}% ({zn['beyond2'] * 100:.0f}%)"
           + (" — " + ", ".join(zx) if zx else ""))
    return {"vr": t_vr, "spectrum": t_sp, "season": t_se, "swings": t_sw, "z": t_z}


# ------------------------------------------------------------------------------------------------------------ charts
def two_line(head, sub) -> str:
    return f"{head}<br><span style='font-size:12px'>{sub}</span>"


def style(fig, title, height, legend=True, legend_top=False, legend_y=-0.14, left=60):
    """legend_top: the legend sits between the (two-line) title and the plot, clear of the x-axis title."""
    leg = {"orientation": "h", "x": 0, "font": {"color": INK2, "size": 11}, **({"y": 1.02, "yanchor": "bottom"} if legend_top else {"y": legend_y, "yanchor": "top"})}
    fig.update_layout(template="none", height=height, autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": FONT, "size": 12, "color": INK}, title={"text": title, "x": 0, "xanchor": "left", "y": 1, "yref": "container", "yanchor": "top", "pad": {"t": 10}, "automargin": True, "font": {"size": 14, "color": INK}},
                      margin={"l": left, "r": 24, "t": 112 if legend_top else 80, "b": 56}, showlegend=legend, hovermode="closest", legend=leg)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, showline=True, zeroline=False, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, showline=True, zeroline=False, tickfont={"color": MUTED})
    fig.update_annotations(font={"color": INK2, "size": 12})
    return fig


def html_fig(fig, first=False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=True if first else False, config={"displaylogo": False, "responsive": True})


def wtick(k):
    return {52: "1 yr", 104: "2 yr", 208: "4 yr", 416: "8 yr", 26: "½ yr"}.get(k, f"{k} wk")


def tbl(head, rows, cls="") -> str:
    h = "".join(f"<th>{c}</th>" for c in head); b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def f2(v, d=2): return "—" if v != v else f"{v:.{d}f}"
def fp(v, d=1): return "—" if v != v else f"{v * 100:+.{d}f}%"
def fpct(v): return "—" if v != v else (f"<b>{v:.1f}</b>" if notable(v) else f"{v:.1f}")


def ab_html(a, b):
    """Percentile with the expanding drift removed (A, the page) → with no drift removal (B); bold when the notable call differs."""
    s = f"{f2(a, 0)}&nbsp;→&nbsp;{f2(b, 0)}"
    return f"<b>{s}</b>" if notable(a) != notable(b) else s


# ------------------------------------------------------------------------------------------------------------ the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); D = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily(); R = {}; PIT = [0, []]
    ZC = {"engine_max_abs_diff": 0.0, "running_sums_max_abs_diff": 0.0, "rebuild_max_rel_diff": 0.0, "missing_mismatch": []}   # the z-score's checks
    ZWC = {"weekly_running_sums_max_abs_diff": 0.0, "daily_running_sums_max_abs_diff": 0.0, "pit_checks": 0, "pit_violations": []}   # the other windows' checks
    DAILY = tuple(w for w, _, f, _ in ZW.WINDOWS if f == "D"); Z_TESTED_D = (("beyond1", "days beyond ±1"), ("beyond2", "days beyond ±2"), ("med_str", "median stretch beyond ±1"))
    edges = np.arange(-4, 4.001, 0.25)                             # z histogram bins; values beyond ±4 go in the end bins
    for pid, pair in enumerate(PAIRS):
        pf = pair_frame(px, pair, "plain"); wk = pf.loc[pf["decision"], "ratio"]
        g = np.diff(np.log(wk.to_numpy()))                         # g[i] = the change over the week ending wk.index[i + 1]
        # point-in-time: the drift and sd at week t come only from the weekly changes before it (expanding windows)
        t, m, sd, z, hx = point_in_time(g); n = len(z)
        c_, b_ = pit_check(g); PIT[0] += c_; PIT[1] += [f"{pair}: {x}" for x in b_]
        pw = wk.index[BURN:]                                       # the path's dates: x[0] at wk.index[BURN]
        S = np.random.default_rng([SEED_SIGNS, pid]).choice(np.array([-1.0, 1.0]), size=(D, n)); Z = S * z   # random signs; no demeaning
        R[pair] = {"pid": pid, "wk": wk, "n": n, "pw": pw, "ratio_first": wk.index[0].date().isoformat(), "first": pw[0].date().isoformat(), "last": wk.index[-1].date().isoformat(),
                   "drift_wk": float(g.mean()), "vol_wk": float(g.std(ddof=1)),                       # descriptive summaries only; used by no calculation
                   "spec": spectrum(z, Z), "vr": variance_ratios(z, Z), "season": seasonality(monthly(pf), np.random.default_rng([SEED_MONTHS, pid]), D), "swing": swings(z, Z, pw, hx)}
        # the same checks with NO drift removal: the raw weekly changes, against random walks with the expanding drift added back
        R[pair]["vr_raw"] = variance_ratios(g[t], Z + m); R[pair]["swing_raw"] = swings(g[t], Z + m, pw, hx); R[pair]["spec_raw"] = spectrum(g[t], Z + m)
        R[pair]["band_mean"] = wk.rolling(52, min_periods=42).mean(); R[pair]["band_sd"] = wk.rolling(52, min_periods=42).std()
        # the z-score of the ratio: T02's band in units of its own sd; description only (nothing about what follows a touch)
        zs = band_z(wk); zv = zs.to_numpy(); wv = wk.to_numpy()
        c_, b_ = z_pit_check(wk); PIT[0] += c_; PIT[1] += [f"{pair}: {x}" for x in b_]
        dec = decide(pf, Z_WIN, "E1", EnginePIT(), ratio="plain")   # read: the band's mean and sd only
        ez = ((dec["ratio"] - dec["mean"]) / dec["sd"]).reindex(wk.index).to_numpy()
        if not np.array_equal(np.isnan(ez), np.isnan(zv)):
            ZC["missing_mismatch"].append(pair)
        ZC["engine_max_abs_diff"] = max(ZC["engine_max_abs_diff"], float(np.nanmax(np.abs(ez - zv))))
        ZC["running_sums_max_abs_diff"] = max(ZC["running_sums_max_abs_diff"], float(np.nanmax(np.abs(trailing_z(wv)[0] - zv))))
        ZC["rebuild_max_rel_diff"] = max(ZC["rebuild_max_rel_diff"], float(np.max(np.abs(wv[BURN] * np.exp(np.cumsum(z + m)) / wv[BURN + 1:] - 1))))
        # random paths of the ratio: the real warm-up weeks, then each residual with its random sign plus the expanding drift
        rp = np.concatenate([np.tile(wv[: BURN + 1], (D, 1)), wv[BURN] * np.exp(np.cumsum(Z + m, axis=1))], axis=1)
        zR = trailing_z(rp)[:, BURN:]; zw = zv[BURN:]; sw = R[pair]["swing"]
        zreal = z_stats(zw, sw["tps"]); znull = [z_stats(zR[j], sw["null_tps"][j]) for j in range(D)]
        zcol = {k: np.array([s[k] for s in znull], float) for k in zreal}
        iw = {"min": int(np.argmin(zw)), "max": int(np.argmax(zw))}
        R[pair]["z"] = {"series": zs, "real": zreal, "first_z": zs.first_valid_index().date().isoformat(),
                        "null_median": {k: float(np.nanmedian(v)) for k, v in zcol.items()}, "null_lo": {k: float(np.nanpercentile(v, 5)) for k, v in zcol.items()},
                        "null_hi": {k: float(np.nanpercentile(v, 95)) for k, v in zcol.items()}, "pct": {k: pctile(zreal[k], zcol[k]) for k, _ in Z_TESTED},
                        "min": (float(zw[iw["min"]]), pw[iw["min"]].date().isoformat()), "max": (float(zw[iw["max"]]), pw[iw["max"]].date().isoformat()),
                        "last": (float(zw[-1]), pw[-1].date().isoformat()),
                        "hist_real": np.histogram(np.clip(zw, -3.999, 3.999), edges)[0], "hist_null": np.histogram(np.clip(zR, -3.999, 3.999), edges)[0]}
        # ---- the other z windows: weekly on E01's random paths (and turning points), daily on the daily random paths
        R[pair]["zw"] = {"w52": R[pair]["z"]}
        for w in ("w26", "w13"):
            n_ = ZW.WIN[w][3]; zs_w = ZW.z_of(wk, n_); zw_ = zs_w.to_numpy()[BURN:]; zRw = ZW.trailing_z(rp, n_)[:, BURN:]
            ZWC["weekly_running_sums_max_abs_diff"] = max(ZWC["weekly_running_sums_max_abs_diff"], float(np.nanmax(np.abs(ZW.trailing_z(wv, n_)[0][BURN:] - zw_))))
            c_, b_ = zwin_pit(wk, n_); ZWC["pit_checks"] += c_; ZWC["pit_violations"] += [f"{pair}: {x}" for x in b_]
            R[pair]["zw"][w] = z_pack(zs_w, zw_, pw, z_stats(zw_, sw["tps"]), [z_stats(zRw[j], sw["null_tps"][j]) for j in range(D)],
                                      np.histogram(np.clip(zw_, -3.999, 3.999), edges)[0], np.histogram(np.clip(zRw, -3.999, 3.999), edges)[0], [k for k, _ in Z_TESTED])
        dser = ZW.series(pf, "D"); lnd = np.log(dser["ratio"].to_numpy()); bd = ZW.BURN["D"]; zd = {}
        accN = {w: [] for w in DAILY}; accH = {w: np.zeros(len(edges) - 1) for w in DAILY}
        for w in DAILY:
            n_ = ZW.WIN[w][3]; zd[w] = ZW.z_of(dser["ratio"], n_)
            ZWC["daily_running_sums_max_abs_diff"] = max(ZWC["daily_running_sums_max_abs_diff"], float(np.nanmax(np.abs(ZW.trailing_z(np.exp(lnd[None, :]), n_)[0] - zd[w].to_numpy()))))
            c_, b_ = zwin_pit(dser["ratio"], n_); ZWC["pit_checks"] += c_; ZWC["pit_violations"] += [f"{pair}: {x}" for x in b_]
        for _, Pc in ZW.random_log_paths(lnd, pid, "D", D):
            Ec = np.exp(Pc)
            for w in DAILY:
                zc = ZW.trailing_z(Ec, ZW.WIN[w][3])[:, bd:]; accN[w] += [z_stats(zc[j], []) for j in range(len(zc))]; accH[w] += np.histogram(np.clip(zc, -3.999, 3.999), edges)[0]
        for w in DAILY:
            zwd = zd[w].to_numpy()[bd:]
            R[pair]["zw"][w] = z_pack(zd[w], zwd, dser.index[bd:], z_stats(zwd, []), accN[w], np.histogram(np.clip(zwd, -3.999, 3.999), edges)[0], accH[w], [k for k, _ in Z_TESTED_D])
        R[pair]["zw_first"] = {"W": pw[0].date().isoformat(), "D": dser.index[bd].date().isoformat()}
        sw.pop("null_tps"); R[pair]["swing_raw"].pop("null_tps")
        R[pair]["verdict"] = verdict(R[pair])
        print(f"  {pair:18s} {R[pair]['first']} → {R[pair]['last']}  {n:4d} weekly changes  [{time.time() - t0:5.0f}s]", flush=True)

    if PIT[1] or PIT[0] == 0:                                      # a look-ahead violation is a failed gate, not a warning
        sys.exit("look-ahead check FAILED: " + "; ".join(PIT[1]) if PIT[1] else "look-ahead check ran nothing")
    print(f"  look-ahead check: {PIT[0]} checks, 0 violations")
    z_ok = (not ZC["missing_mismatch"] and ZC["engine_max_abs_diff"] <= 1e-9 and ZC["running_sums_max_abs_diff"] <= 1e-8 and ZC["rebuild_max_rel_diff"] <= 1e-9)
    if not z_ok:                                                   # the z-score must be the engine's band, and the random paths its method
        sys.exit(f"z-score check FAILED: {ZC}")
    if ZWC["pit_violations"] or ZWC["weekly_running_sums_max_abs_diff"] > 1e-8 or ZWC["daily_running_sums_max_abs_diff"] > 1e-6:
        sys.exit(f"z-window check FAILED: {ZWC}")
    print(f"  z windows: {ZWC['pit_checks']} look-ahead checks, 0 violations; running sums weekly {ZWC['weekly_running_sums_max_abs_diff']:.1e}, daily {ZWC['daily_running_sums_max_abs_diff']:.1e}")
    print(f"  z-score checks: engine band max |diff| {ZC['engine_max_abs_diff']:.1e}, running sums {ZC['running_sums_max_abs_diff']:.1e}, rebuild {ZC['rebuild_max_rel_diff']:.1e}")
    # ---------------------------------------------------------------- tables (CSV + markdown)
    summ, bands, vrs, seas, sws, zrows, zweek = [], [], [], [], [], [], []
    for p in ORDER:
        r = R[p]; sp = r["spec"]; se = r["season"]; sw = r["swing"]
        summ.append({"pair": p, "ratio_first": r["ratio_first"], "first": r["first"], "last": r["last"], "weekly_changes": r["n"], "drift_pct_per_yr": (np.exp(r["drift_wk"] * 52.18) - 1), "vol_pct_per_yr": r["vol_wk"] * np.sqrt(52.18),
                     "spectrum_p": sp["p_global"], "spectrum_top_band": sp["top"]["band"], "spectrum_top_ratio": sp["top"]["ratio"], **{f"vr_{v['k']}": v["vr"] for v in r["vr"]}, **{f"vr_{v['k']}_pct": v["pct"] for v in r["vr"]},
                     "season_reportable": se["reportable"], "season_p": se["p"], "swing_h_pct_median": sw["h_pct_median"], "swing_h_pct_min": sw["h_pct_min"], "swing_h_pct_max": sw["h_pct_max"], "swings": sw["n"], "swing_median_wk": sw["median"], "swing_cv": sw["cv"], "swing_pct_n": sw["pct_n"], "swing_pct_median": sw["pct_median"], "swing_pct_cv": sw["pct_cv"],
                     **{f"verdict_{k}": v for k, v in r["verdict"].items()}})
        bands += [{"pair": p, **{k: v for k, v in b.items()}} for b in sp["bands"]]
        vrs += [{"pair": p, **v} for v in r["vr"]]
        seas += [{"pair": p, "reportable": se["reportable"], "p_dispersion": se["p"], **m} for m in se["months"]]
        sws += [{"pair": p, "date": sw["dates"][i].date().isoformat(), "turn": "peak" if t > 0 else "trough", "ratio": float(r["wk"].iloc[BURN + i]), "drift_removed": float(r["wk"].iloc[BURN] * np.exp(sw["x"][i])),
                 "z": float(r["z"]["series"].iloc[BURN + i])} for i, t in sw["tps"]]
        zz = r["z"]
        zrows.append({"pair": p, "z_first": zz["first_z"], "first": r["first"], "last": r["last"], "weeks": r["n"] + 1, **{k: v for k, v in zz["real"].items()},
                      **{f"{k}_rw_median": v for k, v in zz["null_median"].items()}, **{f"{k}_rw_p5": v for k, v in zz["null_lo"].items()}, **{f"{k}_rw_p95": v for k, v in zz["null_hi"].items()},
                      **{f"{k}_pct": v for k, v in zz["pct"].items()}, "z_min": zz["min"][0], "z_min_date": zz["min"][1], "z_max": zz["max"][0], "z_max_date": zz["max"][1], "z_last": zz["last"][0], "z_last_date": zz["last"][1]})
        zweek += [{"pair": p, "date": d.date().isoformat(), "ratio": float(v), "mean_52": float(mu), "sd_52": float(s), "z": float(q)}
                  for d, v, mu, s, q in zip(r["wk"].index, r["wk"], r["band_mean"], r["band_sd"], zz["series"]) if q == q]
    zwr = []
    for p in ORDER:
        for w, lab, f, n_ in ZW.WINDOWS:
            q = R[p]["zw"][w]
            zwr.append({"pair": p, "window": w, "window_label": lab, "freq": f, "first": R[p]["zw_first"][f], "last": R[p]["last"], **{k: v for k, v in q["real"].items()},
                        **{f"{k}_rw_median": v for k, v in q["null_median"].items()}, **{f"{k}_pct": v for k, v in q["pct"].items()}, "z_last": q["last"][0], "z_last_date": q["last"][1]})
    pd.DataFrame(zwr).to_csv(os.path.join(OUT, "zscores_windows.csv"), index=False)
    S_, B_, V_, M_, W_, Z_, ZW_ = (pd.DataFrame(x) for x in (summ, bands, vrs, seas, sws, zrows, zweek))
    for df, name in ((S_, "summary"), (B_, "spectrum_bands"), (V_, "variance_ratios"), (M_, "seasonality"), (W_, "swings"), (Z_, "zscores"), (ZW_, "z_weekly")):
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    # notable count against chance
    allp = [b["pct"] for p in ORDER for b in R[p]["spec"]["bands"] if b["bins"]] + [v["pct"] for p in ORDER for v in R[p]["vr"] if v["reportable"]]
    allp += [m["pct"] for p in ORDER if R[p]["season"]["reportable"] for m in R[p]["season"]["months"]] + [R[p]["swing"][k] for p in ORDER for k in ("pct_n", "pct_median", "pct_cv")]
    allp += [R[p]["z"]["pct"][k] for p in ORDER for k, _ in Z_TESTED]
    allp = [x for x in allp if x == x]; n_not = sum(notable(x) for x in allp); n_all = len(allp)
    # the drift check: every statistic that the drift removal can touch, (A) expanding drift removed (the page) vs (B) none
    SW_LAB = (("pct_n", "swings: count"), ("pct_median", "swings: median length"), ("pct_cv", "swings: regularity (CV)"))
    dc = []
    for p in ORDER:
        r = R[p]
        dc += [{"pair": p, "statistic": f"VR {a['k']} wk", "pct_A_expanding_drift_removed": a["pct"], "pct_B_no_drift_removal": b["pct"], "flips": bool(notable(a["pct"]) != notable(b["pct"]))}
               for a, b in zip(r["vr"], r["vr_raw"]) if a["reportable"]]
        dc += [{"pair": p, "statistic": lab, "pct_A_expanding_drift_removed": r["swing"][k], "pct_B_no_drift_removal": r["swing_raw"][k], "flips": bool(notable(r["swing"][k]) != notable(r["swing_raw"][k]))} for k, lab in SW_LAB]
        dc += [{"pair": p, "statistic": f"spectrum {a['band']} wk", "pct_A_expanding_drift_removed": a["pct"], "pct_B_no_drift_removal": b["pct"], "flips": bool(notable(a["pct"]) != notable(b["pct"]))}
               for a, b in zip(r["spec"]["bands"], r["spec_raw"]["bands"]) if a["bins"]]
    DC = pd.DataFrame(dc); DC.to_csv(os.path.join(OUT, "drift_check.csv"), index=False); n_flip, n_dc = int(DC["flips"].sum()), len(DC)
    flipped = [f"{nice(x['pair'])} {x['statistic']} ({x['pct_A_expanding_drift_removed']:.1f} → {x['pct_B_no_drift_removal']:.1f})" for _, x in DC[DC["flips"]].iterrows()]

    def nflip(prefix):
        sub = DC[DC["statistic"].str.startswith(prefix)]; return f"{int(sub['flips'].sum())} of {len(sub)}"

    def spec_changed(p):
        sub = DC[(DC["pair"] == p) & DC["statistic"].str.startswith("spectrum")]; return f"{int(sub['flips'].sum())} of {len(sub)}"
    by_check = f"variance ratios {nflip('VR')}, swings {nflip('swings')}, spectrum bands {nflip('spectrum')}"
    # pairs where A and B really disagree: more than one call changes AND the percentile moves by 10 points or more
    # (a percentile drifting from 95.4 to 94.9 changes the call but is not a disagreement)
    big = DC["flips"] & ((DC["pct_A_expanding_drift_removed"] - DC["pct_B_no_drift_removal"]).abs() >= 10)
    flip_pairs = [q for q in ORDER if int((big & (DC["pair"] == q)).sum()) >= 2]
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "draws": D, "seeds": {"sign_flips": f"default_rng([{SEED_SIGNS}, pair_id])", "month_shuffles": f"default_rng([{SEED_MONTHS}, pair_id])"},
             "pairs": PAIRS, "series": "plain ratio P_a / P_b on the last trading day of each ISO week (pair_frame); weekly change g_t = ln(r_t / r_t-1)",
             "point_in_time": {"drift": "expanding mean of the weekly changes before week t", "zigzag_h": f"expanding sd (n - 1) of the weekly changes before week t * sqrt({H_WEEKS})",
                               "burn_in_weeks": BURN, "random_walks": "the point-in-time residuals with random signs; no demeaning", "whole_history_estimates_used": "none",
                               "look_ahead_check": {"what": "history cut at 12 dates per pair: estimates up to the cut equal the full run's, turning points are a prefix of the full run's, and the z-score equals the full run's (to 1e-12)", "checks": PIT[0], "violations": PIT[1]}},
             "zscore": {"definition": f"z_t = (r_t - mean of the last {Z_WIN} weekly closes including t) / their sd (n - 1), from week {Z_MIN}: T02's band, z = ±1 is its edge",
                        "scope": "description only: nothing about what follows a band touch (no return, no time back to the mean, no hit rate)",
                        "random_paths": "the real warm-up weeks, then each point-in-time residual with the page's random sign plus the expanding drift (the drift check's version B); z on the path's ratio; turning points from the zig-zag on its drift-removed path",
                        "window": "from the end of the warm-up to the last week, as the rest of the page; stretches running at either end are cut there", "tested": [k for k, _ in Z_TESTED],
                        "checks": {**ZC, "engine": "(ratio - mean) / sd from analysis.ratio_engine.decide(plain, window 52); only those columns read"}},
             "z_windows": {"windows": [w for w, _, _, _ in ZW.WINDOWS], "daily_seed": f"default_rng([{ZW.SEED['D']}, pair_id])", "daily_warm_up_days": ZW.BURN["D"], "checks": ZWC},
             "bands_weeks": BANDS, "vr_k": VR_K, "min_cycles": MIN_CYCLES, "min_k_frac": MIN_K_FRAC, "min_years_month": MIN_YEARS_MONTH, "notable": [LO, HI], "notable_count": [n_not, n_all],
             "drift_check": {"what": "A = expanding drift removed (the page); B = no drift removal, random walks with the expanding drift added back", "flips": n_flip, "statistics": n_dc, "by_check": by_check, "flipped": flipped},
             "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    md = [f"# E01 results — cyclicality in the ratio series (EXPLORATORY)\n", f"_Generated by `explore/E01_ratio_cycles/cycles.py` on {audit['run_date']}; {D:,} random-walk paths per pair (sign flips, seed `default_rng([{SEED_SIGNS}, pair_id])`). Description, not a finding. Percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"_Point-in-time: the drift removed at each week is the average of the weekly changes before it, and the zig-zag threshold comes from their sd (expanding windows after a {BURN}-week warm-up); no whole-history estimate enters any series, threshold or random walk. Windows below start after the warm-up (the month-of-year check needs no estimate and uses every month)._\n",
          f"**Against chance:** {n_not} of {n_all} percentiles are notable; about {0.1 * n_all:.0f} would be if every ratio were a random walk with its own volatility (the percentiles are not independent: the pairs share legs and the checks overlap).\n",
          "## Per pair\n", "| pair | window | weekly changes | variance ratios | spectrum | month of year | swings | z-score (T02's band) |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {nice(p)} | {R[p]['first']} → {R[p]['last']} | {R[p]['n']} | {R[p]['verdict']['vr']} | {R[p]['verdict']['spectrum']} | {R[p]['verdict']['season']} | {R[p]['verdict']['swings']} | {R[p]['verdict']['z']} |" for p in ORDER]
    md += ["\n## Variance ratios (VR; below 1 = reverses, above 1 = continues) and percentile among the random walks\n", "| pair | " + " | ".join(f"{k} wk" for k in VR_K) + " |", "|---|" + "---|" * len(VR_K)]
    md += [f"| {nice(p)} | " + " | ".join((f"{v['vr']:.2f} ({'**' if notable(v['pct']) else ''}{v['pct']:.1f}{'**' if notable(v['pct']) else ''})" if v["reportable"] else "—") for v in R[p]["vr"]) + " |" for p in ORDER]
    md += ["\n## Spectrum: share of variance by cycle length ÷ the random walks' median (percentile); one overall p per pair\n", "| pair | overall p | " + " | ".join(f"{lo}–{hi} wk" for lo, hi in BANDS) + " |", "|---|---|" + "---|" * len(BANDS)]
    md += [f"| {nice(p)} | {R[p]['spec']['p_global']:.3f} | " + " | ".join((f"{b['ratio']:.2f} ({'**' if notable(b['pct']) else ''}{b['pct']:.1f}{'**' if notable(b['pct']) else ''})" if b["bins"] else "—") for b in R[p]["spec"]["bands"]) + " |" for p in ORDER]
    md += ["\n## Month of year: average change of the ratio in each calendar month (percentile among within-year shuffles)\n", "| pair | p | " + " | ".join(MONTHS) + " |", "|---|---|" + "---|" * 12]
    md += [f"| {nice(p)} | {R[p]['season']['p']:.3f}{'' if R[p]['season']['reportable'] else ' (too short)'} | " + " | ".join(f"{m['mean'] * 100:+.1f}% ({'**' if notable(m['pct']) and R[p]['season']['reportable'] else ''}{m['pct']:.0f}{'**' if notable(m['pct']) and R[p]['season']['reportable'] else ''})" for m in R[p]["season"]["months"]) + " |" for p in ORDER]
    md += ["\n## Swings (zig-zag on the drift-removed ratio, reversal ≥ one typical half-year move from earlier weeks)\n", "| pair | threshold: median (range) | swings (pct) | median length, wk (pct) | regularity CV (pct) | random walks: swings / median / CV | median swing size |", "|---|---|---|---|---|---|---|"]
    md += [f"| {nice(p)} | {R[p]['swing']['h_pct_median'] * 100:.1f}% ({R[p]['swing']['h_pct_min'] * 100:.1f}–{R[p]['swing']['h_pct_max'] * 100:.1f}%) | {R[p]['swing']['n']} ({R[p]['swing']['pct_n']:.1f}) | {f2(R[p]['swing']['median'], 0)} ({f2(R[p]['swing']['pct_median'], 1)}) | {f2(R[p]['swing']['cv'])} ({f2(R[p]['swing']['pct_cv'], 1)}) | {R[p]['swing']['null_n_median']:.0f} / {R[p]['swing']['null_median_median']:.0f} / {R[p]['swing']['null_cv_median']:.2f} | {fp(R[p]['swing']['median_amp_pct'])} |" for p in ORDER]
    def mdp(v): return f"**{v:.1f}**" if notable(v) else f2(v, 1)
    md += [f"\n## z-score of the ratio: (ratio − trailing {Z_WIN}-week mean) ÷ trailing sd — T02's band, whose edges are z = ±1\n",
           "Description only: nothing about what follows a band touch is computed. Over each pair's window (after the warm-up); a stretch = consecutive weeks above +1 or below −1. "
           "In brackets: the median of the random walks (the same sign flips, with the expanding drift added back, z on the path's ratio) and, for the five tested statistics, the percentile among them."
           + (f" These random walks are the drift check's version B, so where A and B disagree ({', '.join(nice(q) for q in flip_pairs)}) the z comparison is not reliable either: the drift added back comes from the warm-up year." if flip_pairs else "") + "\n",
           "| pair | window | beyond ±1 (median, pct) | above +1 / below −1 | beyond ±2 (median, pct) | stretches beyond ±1 | median stretch, wk (median, pct) | longest, wk | median z at peaks (median, pct) | median z at troughs (median, pct) | lowest z | highest z | latest z |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in ORDER:
        zz = R[p]["z"]; a, nm, pc = zz["real"], zz["null_median"], zz["pct"]
        md.append(f"| {nice(p)} | {R[p]['first']} → {R[p]['last']} | {a['beyond1'] * 100:.0f}% ({nm['beyond1'] * 100:.0f}%, {mdp(pc['beyond1'])}) | {a['above1'] * 100:.0f}% / {a['below1'] * 100:.0f}% "
                  f"| {a['beyond2'] * 100:.1f}% ({nm['beyond2'] * 100:.1f}%, {mdp(pc['beyond2'])}) | {a['n_str']} ({nm['n_str']:.0f}) | {f2(a['med_str'], 0)} ({f2(nm['med_str'], 0)}, {mdp(pc['med_str'])}) | {a['max_str']} ({nm['max_str']:.0f}) "
                  f"| {a['z_peak']:+.2f} ({nm['z_peak']:+.2f}, {mdp(pc['z_peak'])}) | {a['z_trough']:+.2f} ({nm['z_trough']:+.2f}, {mdp(pc['z_trough'])}) | {zz['min'][0]:+.2f} ({zz['min'][1]}) | {zz['max'][0]:+.2f} ({zz['max'][1]}) | {zz['last'][0]:+.2f} ({zz['last'][1]}) |")
    md += ["\n## The z-score for every window (weekly windows on decision-week closes with E01's random walks; daily windows on daily closes with daily random walks, seed `default_rng([111, pair_id])`, after a 250-day warm-up)\n",
           "Per window: the share of periods beyond ±1 (random walks' median, percentile), beyond ±2, the median stretch beyond ±1 in the window's units (weeks or trading days), and the latest z. Z at the zig-zag turning points is for the weekly windows only (the turning points are weekly).\n",
           "| pair | " + " | ".join(lab + (" (daily)" if f == "D" else "") for _, lab, f, _ in ZW.WINDOWS) + " |", "|---|" + "---|" * len(ZW.WINDOWS)]
    for p in ORDER:
        cells = []
        for w, lab, f, n_ in ZW.WINDOWS:
            q = R[p]["zw"][w]; a_, nm_, pc_ = q["real"], q["null_median"], q["pct"]; bo = lambda v: f"**{v:.0f}**" if notable(v) else f"{v:.0f}"   # noqa: E731
            cells.append(f"±1 {a_['beyond1'] * 100:.0f}% ({nm_['beyond1'] * 100:.0f}%, {bo(pc_['beyond1'])}); ±2 {a_['beyond2'] * 100:.1f}% ({nm_['beyond2'] * 100:.1f}%, {bo(pc_['beyond2'])}); stretch {f2(a_['med_str'], 0)} ({f2(nm_['med_str'], 0)}, {bo(pc_['med_str'])}); now {q['last'][0]:+.2f}")
        md.append(f"| {nice(p)} | " + " | ".join(cells) + " |")
    def ab(a, b): return f"{'**' if notable(a) != notable(b) else ''}{f2(a, 0)} → {f2(b, 0)}{'**' if notable(a) != notable(b) else ''}"
    md += ["\n## Does the drift removal matter?\n",
           "Both versions use only past data. **A** = the page: the drift removed at each week is the average of the weekly changes before it (expanding). "
           "**B** = no drift removal: the real ratio's raw weekly changes, against the same random walks with the expanding drift added back. "
           "Percentiles A → B; **bold** = the notable call changes; the spectrum column counts the cycle-length bands whose call changes (all bands are in drift_check.csv), with the overall p A → B. The month-of-year check never removes a drift.\n",
           f"**{n_flip} of {n_dc}** statistics change their notable call ({by_check})" + (": " + "; ".join(flipped) + "." if flipped else ".")
           + (f" Where A and B disagree on more than one call ({', '.join(nice(q) for q in flip_pairs)}) neither reading is reliable: the point-in-time drift starts from the warm-up year, and where that year differs sharply from what followed (the Micro250 pairs' warm-up is 2022) removing it leaves a slow trend in the residuals (A leans to 'continues') while adding it to the random walks leans B the other way." if flip_pairs else ""), "",
           "| pair | drift a year (description) | " + " | ".join(f"VR {k} wk" for k in VR_K) + " | swings | median length | regularity (CV) | spectrum bands changed | spectrum overall p |", "|---|---|" + "---|" * (len(VR_K) + 5)]
    md += [f"| {nice(p)} | {(np.exp(R[p]['drift_wk'] * 52.18) - 1) * 100:+.1f}% | " + " | ".join((ab(a["pct"], b["pct"]) if a["reportable"] else "—") for a, b in zip(R[p]["vr"], R[p]["vr_raw"]))
           + " | " + " | ".join(ab(R[p]["swing"][k], R[p]["swing_raw"][k]) for k, _ in SW_LAB) + f" | {spec_changed(p)} | {R[p]['spec']['p_global']:.2f} → {R[p]['spec_raw']['p_global']:.2f} |" for p in ORDER]
    open(os.path.join(OUT, "E01_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- figures
    figs = {}
    # 0a. the ten ratios
    f = make_subplots(rows=5, cols=2, subplot_titles=[f"{nice(p)}<br><span style='font-size:11px'>{R[p]['ratio_first']} → {R[p]['last']}</span>" for p in ORDER], vertical_spacing=0.06, horizontal_spacing=0.08)
    for i, p in enumerate(ORDER):
        r = R[p]; row, col = i // 2 + 1, i % 2 + 1; lead = i == 0; wk = r["wk"]
        f.add_trace(go.Scatter(x=wk.index, y=r["band_mean"] - r["band_sd"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, legendgroup="band"), row=row, col=col)
        f.add_trace(go.Scatter(x=wk.index, y=r["band_mean"] + r["band_sd"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=BAND_FILL, name="trailing 52-week mean ± 1 sd", legendgroup="band", showlegend=lead, hoverinfo="skip"), row=row, col=col)
        f.add_trace(go.Scatter(x=wk.index, y=r["band_mean"], mode="lines", line={"color": INK2, "width": 1}, name="trailing 52-week mean", legendgroup="mean", showlegend=lead, hovertemplate="%{x|%Y-%m-%d}<br>mean %{y:.4f}<extra></extra>"), row=row, col=col)
        f.add_trace(go.Scatter(x=wk.index, y=wk, mode="lines", line={"color": BLUE, "width": 1.5}, name="ratio (weekly close)", legendgroup="ratio", showlegend=lead, hovertemplate="%{x|%Y-%m-%d}<br>ratio %{y:.4f}<extra>" + nice(p) + "</extra>"), row=row, col=col)
    figs["ratios"] = style(f, two_line("The ten plain ratios P_a ÷ P_b, weekly", "with T02's band: trailing 52-week mean ± 1 sd"), 1500, legend_y=-0.03); f.update_layout(margin={"t": 110})
    # 0b. the five indices
    f = go.Figure(); idx_rows = []
    for a in ASSETS:
        c = px[a]["close"]; key = c.index.isocalendar(); wkc = c[~pd.Series((key["year"] * 100 + key["week"]).values, index=c.index).duplicated(keep="last").values]
        reb = wkc / wkc.iloc[0] * 100
        f.add_trace(go.Scatter(x=reb.index, y=reb, mode="lines", line={"color": CAT[a], "width": 1.6}, name=a, hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>" + a + "</extra>"))
        yrs = (wkc.index[-1] - wkc.index[0]).days / 365.25
        idx_rows.append([a, wkc.index[0].date().isoformat(), wkc.index[-1].date().isoformat(), f"{wkc.iloc[0]:,.2f}", f"{wkc.iloc[-1]:,.2f}", f"{(wkc.iloc[-1] / wkc.iloc[0]) ** (1 / yrs) * 100 - 100:.1f}%"])
    f.update_yaxes(type="log", tickvals=[50, 100, 200, 500, 1000, 2000], title={"text": "rebased to 100 at each series' first week (log scale)", "font": {"color": INK2}})
    figs["indices"] = style(f, f"The five legs, weekly closes rebased to 100 · {min(r[1] for r in idx_rows)} → {max(r[2] for r in idx_rows)}", 460)

    # per-pair figures: every pair's traces, one visible at a time (the selector at the top)
    titles = {k: {} for k in ("ratio", "spec", "vr", "season", "dur", "z")}
    fA = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.42, 0.33, 0.25], vertical_spacing=0.07,
                       subplot_titles=("the ratio with T02's band (trailing 52-week mean ± 1 sd)", "the same ratio with its drift removed point-in-time (the average of earlier weeks only), and the zig-zag turning points",
                                       "z-score for the z window chosen at the top: (ratio − trailing mean) ÷ trailing sd; grey strip = z between −1 and +1 (for 52 weeks, T02/T03's band), dotted lines ±2; daily windows shown at each week's last day"))
    fB, fC, fD, fE, fZ = go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
    zmid = (edges[:-1] + edges[1:]) / 2
    for p in ORDER:
        r = R[p]; wk = r["wk"]; sw = r["swing"]; sp = r["spec"]; vis = p == DEFAULT_PAIR; pw = r["pw"]; dr = wk.iloc[BURN] * np.exp(sw["x"]); zz = r["z"]; zs = zz["series"]
        fA.add_trace(go.Scatter(x=wk.index, y=r["band_mean"] - r["band_sd"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=p, visible=vis), row=1, col=1)
        fA.add_trace(go.Scatter(x=wk.index, y=r["band_mean"] + r["band_sd"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=BAND_FILL, name="mean ± 1 sd", hoverinfo="skip", meta=p, visible=vis), row=1, col=1)
        fA.add_trace(go.Scatter(x=wk.index, y=wk, mode="lines", line={"color": BLUE, "width": 1.5}, name="ratio", meta=p, visible=vis, hovertemplate="%{x|%Y-%m-%d}<br>ratio %{y:.4f}<extra></extra>"), row=1, col=1)
        fA.add_trace(go.Scatter(x=pw, y=dr, mode="lines", line={"color": INK2, "width": 1.2}, name="drift removed (point-in-time)", meta=p, visible=vis, hovertemplate="%{x|%Y-%m-%d}<br>drift removed %{y:.4f}<extra></extra>"), row=2, col=1)
        pk = [i for i, t in sw["tps"] if t > 0]; tr = [i for i, t in sw["tps"] if t < 0]
        fA.add_trace(go.Scatter(x=pw[pk], y=dr[pk], mode="markers", marker={"symbol": "triangle-down", "size": 10, "color": ORANGE}, name="peak", legendgroup="pk", meta=p, visible=vis, hovertemplate="peak %{x|%Y-%m-%d}<extra></extra>"), row=2, col=1)
        fA.add_trace(go.Scatter(x=pw[tr], y=dr[tr], mode="markers", marker={"symbol": "triangle-up", "size": 10, "color": AQUA}, name="trough", legendgroup="tr", meta=p, visible=vis, hovertemplate="trough %{x|%Y-%m-%d}<extra></extra>"), row=2, col=1)
        for w, lab, f, n_ in ZW.WINDOWS:
            q = r["zw"][w]; zser = q["series"] if f == "W" else q["series"].reindex(wk.index); zx = zser.dropna(); ends = [zx.index[0], zx.index[-1]]
            mw = p + ZW.SEP + w; vw = vis and w == ZW.DEFAULT; unit = "trading days" if f == "D" else "weeks"
            fA.add_trace(go.Scatter(x=ends, y=[-1, -1], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=mw, visible=vw), row=3, col=1)
            fA.add_trace(go.Scatter(x=ends, y=[1, 1], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=BAND_FILL, hoverinfo="skip", showlegend=False, meta=mw, visible=vw), row=3, col=1)
            fA.add_trace(go.Scatter(x=zx.index.strftime("%Y-%m-%d").tolist(), y=np.round(zx.to_numpy(), 3), mode="lines", line={"color": BLUE, "width": 1.3}, name="z-score (bottom)", legendgroup="zline", meta=mw, visible=vw,
                                    hovertemplate="%{x|%Y-%m-%d}<br>z (" + lab + ") %{y:+.2f}<extra></extra>"), row=3, col=1)
            ztp = np.round(zser.reindex(pw).to_numpy(), 3)
            fA.add_trace(go.Scatter(x=pw[pk], y=ztp[pk], mode="markers", marker={"symbol": "triangle-down", "size": 9, "color": ORANGE}, showlegend=False, legendgroup="pk", meta=mw, visible=vw, hovertemplate="peak %{x|%Y-%m-%d}<br>z %{y:+.2f}<extra></extra>"), row=3, col=1)
            fA.add_trace(go.Scatter(x=pw[tr], y=ztp[tr], mode="markers", marker={"symbol": "triangle-up", "size": 9, "color": AQUA}, showlegend=False, legendgroup="tr", meta=mw, visible=vw, hovertemplate="trough %{x|%Y-%m-%d}<br>z %{y:+.2f}<extra></extra>"), row=3, col=1)
            hrw, hnw = q["hist_real"], q["hist_null"]; aw, nmw = q["real"], q["null_median"]
            fZ.add_trace(go.Bar(x=zmid, y=hrw / max(hrw.sum(), 1) * 100, width=0.22, marker={"color": BLUE}, name=f"the ratio's {unit}", meta=mw, visible=vw, hovertemplate="z %{x:+.2f} ± 0.125: %{y:.1f}% of " + unit + "<extra></extra>"))
            fZ.add_trace(go.Scatter(x=zmid, y=hnw / max(hnw.sum(), 1) * 100, mode="lines", line={"color": MUTED, "width": 2, "shape": "hvh"}, name=f"random walks' {unit} (pooled)", meta=mw, visible=vw,
                                    hovertemplate="z %{x:+.2f} ± 0.125: %{y:.1f}% of random-walk " + unit + "<extra></extra>"))
            titles["z"][mw] = two_line(f"{nice(p)} · z window {lab}{' (daily closes)' if f == 'D' else ''} · {r['zw_first'][f]} → {r['last']} · latest z {q['last'][0]:+.2f} ({q['last'][1]})",
                                       f"beyond ±1 in {aw['beyond1'] * 100:.0f}% of {unit} (random walks {nmw['beyond1'] * 100:.0f}%), beyond ±2 in {aw['beyond2'] * 100:.1f}% ({nmw['beyond2'] * 100:.1f}%); grey strip = inside ±1")
        titles["ratio"][p] = two_line(f"{nice(p)} · {r['ratio_first']} → {r['last']} · lower panel from {r['first']} (after a {BURN}-week warm-up)",
                                      f"turning point = a reversal of one typical half-year move, from earlier weeks only: {sw['h_pct_min'] * 100:.1f}–{sw['h_pct_max'] * 100:.1f}% over the window, {sw['h_pct_last'] * 100:.1f}% now")
        cv = sp["curve"]
        fB.add_trace(go.Scatter(x=cv["period"], y=cv["lo"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=p, visible=vis))
        fB.add_trace(go.Scatter(x=cv["period"], y=cv["hi"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=p, visible=vis))
        fB.add_trace(go.Scatter(x=cv["period"], y=cv["ratio"], mode="lines", line={"color": BLUE, "width": 2}, name="the ratio", meta=p, visible=vis, hovertemplate="cycle length %{x:.0f} wk<br>power ÷ random-walk median %{y:.2f}<extra></extra>"))
        titles["spec"][p] = two_line(f"{nice(p)} · {r['first']} → {r['last']} · overall p = {sp['p_global']:.3f}", "power at each cycle length ÷ what a random walk gives (1 = random); cycle lengths that fit at least 3 times in the window")
        ks = [v for v in r["vr"] if v["reportable"]]
        fC.add_trace(go.Scatter(x=[v["k"] for v in ks], y=[v["null_lo"] for v in ks], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=p, visible=vis))
        fC.add_trace(go.Scatter(x=[v["k"] for v in ks], y=[v["null_hi"] for v in ks], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=p, visible=vis))
        fC.add_trace(go.Scatter(x=[v["k"] for v in ks], y=[v["vr"] for v in ks], mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 8, "color": BLUE}, name="the ratio", meta=p, visible=vis,
                                customdata=[[v["pct"]] for v in ks], hovertemplate="%{x} weeks<br>VR %{y:.2f} · percentile %{customdata[0]:.1f}<extra></extra>"))
        titles["vr"][p] = two_line(f"{nice(p)} · {r['first']} → {r['last']}", "variance ratio by horizon: below 1 = moves reverse, above 1 = moves continue")
        se = r["season"]; ms = se["months"]
        fD.add_trace(go.Bar(x=list(MONTHS), y=[(m["null_hi"] - m["null_lo"]) * 100 for m in ms], base=[m["null_lo"] * 100 for m in ms], marker={"color": NULL_FILL}, width=0.55, name="within-year shuffles, middle 90%", meta=p, visible=vis, hoverinfo="skip"))
        fD.add_trace(go.Scatter(x=list(MONTHS), y=[m["mean"] * 100 for m in ms], mode="markers", marker={"size": 11, "color": BLUE}, name="the ratio: average change in the month", meta=p, visible=vis,
                                customdata=[[m["years"], m["share_up"] * 100, m["pct"]] for m in ms], hovertemplate="%{x}: %{y:+.2f}% on average<br>%{customdata[0]} years · up in %{customdata[1]:.0f}% · percentile %{customdata[2]:.0f}<extra></extra>"))
        titles["season"][p] = two_line(f"{nice(p)} · {se['first']} → {se['last']} · " + (f"p = {se['p']:.3f}" if se["reportable"] else "too short to test"), "average change of the ratio in each calendar month, month-end to month-end")
        bins = np.arange(0, max(int(np.nanmax(sw["durations"])) if len(sw["durations"]) else 8, int(np.percentile(sw["pooled"], 99)) if len(sw["pooled"]) else 8) + 8, 4)
        hr = np.histogram(sw["durations"], bins)[0]; hn = np.histogram(sw["pooled"], bins)[0]
        fE.add_trace(go.Bar(x=bins[:-1] + 2, y=hr / max(hr.sum(), 1) * 100, width=3.6, marker={"color": BLUE}, name="the ratio's swings", meta=p, visible=vis, hovertemplate="%{x:.0f} ± 2 weeks: %{y:.0f}% of swings<extra></extra>"))
        fE.add_trace(go.Scatter(x=bins[:-1] + 2, y=hn / max(hn.sum(), 1) * 100, mode="lines", line={"color": MUTED, "width": 2, "shape": "hvh"}, name="random walks' swings (pooled)", meta=p, visible=vis, hovertemplate="%{x:.0f} ± 2 weeks: %{y:.1f}% of random-walk swings<extra></extra>"))
        titles["dur"][p] = two_line(f"{nice(p)} · {r['first']} → {r['last']}", f"how long the swings last: {sw['n']} swings, median {f2(sw['median'], 0)} weeks (random walks: {sw['null_median_median']:.0f})")
    fA.add_hline(y=0, line={"color": AXIS, "width": 1}, row=3, col=1)
    for y in (-2, 2):
        fA.add_hline(y=y, line={"color": AXIS, "width": 1, "dash": "dot"}, row=3, col=1)
    fA.update_yaxes(title={"text": "z", "font": {"color": INK2}}, row=3, col=1)
    figs["ratio"] = style(fA, titles["ratio"][DEFAULT_PAIR], 900); fA.update_layout(margin={"t": 100})
    fZ.add_vrect(x0=-1, x1=1, fillcolor=BAND_FILL, line={"width": 0}, layer="below")
    for x in (-2, 2):
        fZ.add_vline(x=x, line={"color": AXIS, "width": 1, "dash": "dot"})
    fZ.update_xaxes(range=[-4, 4], dtick=1, title={"text": "z-score: (ratio − trailing mean) ÷ trailing sd; the end bins hold everything beyond ±4", "font": {"color": INK2}})
    fZ.update_yaxes(ticksuffix="%", title={"text": "share of periods", "font": {"color": INK2}})
    fZ.update_layout(bargap=0.05); figs["z"] = style(fZ, titles["z"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 440, legend_top=True)
    tv = [2, 4, 8, 13, 26, 52, 104, 208, 416]
    fB.update_xaxes(type="log", tickvals=tv, ticktext=[wtick(k) for k in tv], title={"text": "cycle length", "font": {"color": INK2}})
    fB.update_yaxes(type="log", tickvals=[0.125, 0.25, 0.5, 1, 2, 4, 8], ticktext=["⅛", "¼", "½", "1", "2", "4", "8"], title={"text": "power ÷ random-walk median", "font": {"color": INK2}}); fB.add_hline(y=1, line={"color": AXIS, "width": 1})
    figs["spec"] = style(fB, titles["spec"][DEFAULT_PAIR], 470, legend_top=True)
    fC.update_xaxes(type="log", tickvals=list(VR_K), ticktext=[wtick(k) for k in VR_K], title={"text": "horizon", "font": {"color": INK2}})
    fC.update_yaxes(title={"text": "variance ratio", "font": {"color": INK2}}); fC.add_hline(y=1, line={"color": AXIS, "width": 1})
    figs["vr"] = style(fC, titles["vr"][DEFAULT_PAIR], 450, legend_top=True)
    fD.update_yaxes(ticksuffix="%", title={"text": "average monthly change of the ratio", "font": {"color": INK2}}); fD.add_hline(y=0, line={"color": AXIS, "width": 1})
    figs["season"] = style(fD, titles["season"][DEFAULT_PAIR], 450, legend_top=True)
    fE.update_xaxes(title={"text": "swing length, weeks (peak to trough or trough to peak)", "font": {"color": INK2}}); fE.update_yaxes(ticksuffix="%", title={"text": "share of swings", "font": {"color": INK2}})
    fE.update_layout(bargap=0.05); figs["dur"] = style(fE, titles["dur"][DEFAULT_PAIR], 430, legend_top=True)

    # heat maps across pairs: percentile among the random walks, 50 = what chance gives
    def heat(zv, xl, txt, hov, title, h):
        f = go.Figure(go.Heatmap(z=zv, x=xl, y=[nice(p) for p in ORDER], zmin=0, zmax=100, colorscale=DIV_LIGHT, text=txt, texttemplate="%{text}", customdata=hov,
                                 hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2, colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="top")
        return style(f, title, h, legend=False, left=130)
    nan = float("nan")
    z1 = [[v["pct"] if v["reportable"] else nan for v in R[p]["vr"]] for p in ORDER]
    t1 = [[(f"{v['vr']:.2f}" if notable(v["pct"]) else "") if v["reportable"] else "" for v in R[p]["vr"]] for p in ORDER]
    h1 = [[(f"VR {v['vr']:.2f}, percentile {v['pct']:.1f}" if v["reportable"] else "not reportable (window too short for this horizon)") for v in R[p]["vr"]] for p in ORDER]
    figs["h_vr"] = heat(z1, [wtick(k) for k in VR_K], t1, h1, "Variance ratios: blue = moves reverse more than a random walk's, red = they continue more (labels on notable cells)", 470)
    z2 = [[b["pct"] if b["bins"] else nan for b in R[p]["spec"]["bands"]] for p in ORDER]
    t2 = [[(f"{b['ratio']:.2f}" if b["bins"] and notable(b["pct"]) else "") for b in R[p]["spec"]["bands"]] for p in ORDER]
    h2 = [[(f"power ÷ random-walk median {b['ratio']:.2f}, percentile {b['pct']:.1f}" if b["bins"] else "cycle length does not fit 3 times in the window") for b in R[p]["spec"]["bands"]] for p in ORDER]
    figs["h_spec"] = heat(z2, [f"{lo}–{hi} wk" for lo, hi in BANDS], t2, h2, "Spectrum by cycle length: red = more power than a random walk, blue = less (labels: power ÷ random-walk median, notable cells)", 470)
    z3 = [[m["pct"] if R[p]["season"]["reportable"] else nan for m in R[p]["season"]["months"]] for p in ORDER]
    t3 = [[(f"{m['mean'] * 100:+.1f}%" if R[p]["season"]["reportable"] and notable(m["pct"]) else "") for m in R[p]["season"]["months"]] for p in ORDER]
    h3 = [[(f"average {m['mean'] * 100:+.2f}% over {m['years']} years, up in {m['share_up'] * 100:.0f}%, percentile {m['pct']:.0f}" if R[p]["season"]["reportable"] else "too short to test") for m in R[p]["season"]["months"]] for p in ORDER]
    figs["h_season"] = heat(z3, list(MONTHS), t3, h3, "Month of year: red = the ratio tends to rise in that month (first leg ahead), blue = tends to fall (labels: average change, notable cells)", 470)
    ZFMT = {"beyond1": lambda v: f"{v * 100:.0f}%", "beyond2": lambda v: f"{v * 100:.1f}%", "med_str": lambda v: f"{f2(v, 0)} wk", "z_peak": lambda v: f"{v:+.2f}", "z_trough": lambda v: f"{v:+.2f}"}
    ZFMT_D = {**ZFMT, "med_str": lambda v: f"{f2(v, 0)} d"}
    def iso_friday(d): y, w, _ = d.isocalendar(); return pd.Timestamp(date.fromisocalendar(y, w, 5))
    zcols = sorted({iso_friday(d) for p in ORDER for d in R[p]["wk"].index}); ci = {d: j for j, d in enumerate(zcols)}
    fHz, fT = go.Figure(), go.Figure(); titles["h_z"], titles["z_time"] = {}, {}; zwin_notable = {}
    for w, lab, f, n_ in ZW.WINDOWS:
        tested = Z_TESTED if f == "W" else Z_TESTED_D; fm = ZFMT if f == "W" else ZFMT_D; wv_ = w == ZW.DEFAULT
        zv_ = [[R[p]["zw"][w]["pct"][k] for k, _ in tested] for p in ORDER]
        zwin_notable[w] = (int(sum(notable(v) for row in zv_ for v in row)), int(sum(v == v for row in zv_ for v in row)))
        fHz.add_trace(go.Heatmap(z=zv_, x=[lab_ for _, lab_ in tested], y=[nice(p) for p in ORDER], zmin=0, zmax=100, colorscale=DIV_LIGHT, xgap=2, ygap=2, meta=ZW.SEP + w, visible=wv_,
                                 text=[[(fm[k](R[p]["zw"][w]["real"][k]) if notable(R[p]["zw"][w]["pct"][k]) else "") for k, _ in tested] for p in ORDER], texttemplate="%{text}",
                                 customdata=[[f"{fm[k](R[p]['zw'][w]['real'][k])} (random walks' median {fm[k](R[p]['zw'][w]['null_median'][k])}), percentile {R[p]['zw'][w]['pct'][k]:.1f}" for k, _ in tested] for p in ORDER],
                                 hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        titles["h_z"][w] = f"z-score, window {lab}{' (daily closes)' if f == 'D' else ''}: red = higher than the random walks give, blue = lower (labels: the ratio's value, notable cells; {zwin_notable[w][0]} of {zwin_notable[w][1]} notable)"
        zm = np.full((len(ORDER), len(zcols)), np.nan); zd_ = np.full((len(ORDER), len(zcols)), "", dtype=object)
        for i, p in enumerate(ORDER):
            q = R[p]["zw"][w]["series"]; q = (q if f == "W" else q.reindex(R[p]["wk"].index)).dropna()
            for d, v in q.items():
                zm[i, ci[iso_friday(d)]] = v; zd_[i, ci[iso_friday(d)]] = d.date().isoformat()
        fT.add_trace(go.Heatmap(z=np.round(np.clip(zm, -3, 3), 2), x=[d.strftime("%Y-%m-%d") for d in zcols], y=[nice(p) for p in ORDER], zmin=-3, zmax=3, colorscale=Z_LIGHT, meta="z" + ZW.SEP + w, visible=wv_, customdata=zd_, hoverongaps=False,
                                hovertemplate="%{y} · %{customdata}<br>z %{z:+.2f} (shown within ±3)<extra></extra>", ygap=2,
                                colorbar={"title": {"text": "z", "side": "right"}, "tickvals": [-3, -2, -1, 0, 1, 2, 3], "ticktext": ["≤ −3", "−2", "−1", "0", "+1", "+2", "≥ +3"], "thickness": 12}))
        zfirst_w = min(R[p]["zw"][w]["first_z"] for p in ORDER)
        titles["z_time"][w] = two_line(f"All ten pairs' z-score, window {lab}{' (daily closes, shown at each week’s last day)' if f == 'D' else ''} · {zfirst_w} → {max(R[p]['last'] for p in ORDER)}",
                                       "red = the ratio above its trailing mean (first leg ahead), blue = below; beyond ±1 = outside the ±1 sd band; blank = no z yet")
    fHz.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fHz.update_xaxes(showgrid=False, side="top"); figs["h_z"] = style(fHz, titles["h_z"][ZW.DEFAULT], 470, legend=False, left=130)
    fT.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fT.update_xaxes(showgrid=False); figs["z_time"] = style(fT, titles["z_time"][ZW.DEFAULT], 470, legend=False, left=130)

    # ---------------------------------------------------------------- the page
    def per_pair(key, fig, tables) -> str:
        t = html.escape(json.dumps(titles[key]), quote=True)
        def div(k, tb_):
            p, w = (k.split(ZW.SEP) + [None])[:2]; shown = p == DEFAULT_PAIR and (w is None or w == ZW.DEFAULT)
            return f"<div data-pair='{html.escape(p, quote=True)}'" + (f" data-win='{w}'" if w else "") + f"{'' if shown else ' hidden'}>{tb_}</div>"
        tb = "".join(div(k, v) for k, v in tables.items())
        return f"<div class='card per-pair' data-titles='{t}'>{html_fig(fig)}<details><summary>Table view</summary>{tb}</details></div>"
    def per_win(key, fig, tables=None) -> str:
        tb = "" if not tables else "<details><summary>Table view</summary>" + "".join(f"<div data-winonly='{w}'{'' if w == ZW.DEFAULT else ' hidden'}>{v}</div>" for w, v in tables.items()) + "</details>"
        return f"<div class='card per-win' data-titles='{html.escape(json.dumps(titles[key]), quote=True)}'>{html_fig(fig)}{tb}</div>"
    tab_ratio = {p: tbl(["date", "turn", "ratio", "drift removed", "z"], [[w["date"], w["turn"], f"{w['ratio']:.4f}", f"{w['drift_removed']:.4f}", f"{w['z']:+.2f}"] for w in sws if w["pair"] == p]) for p in ORDER}

    def zrow(p, k, lab, fmt, tested, w=ZW.DEFAULT):
        zz = R[p]["zw"][w]; pc_ = zz["pct"].get(k, float("nan")); return [lab, fmt(zz["real"][k]), fmt(zz["null_median"][k]), f"{fmt(zz['null_lo'][k])} … {fmt(zz['null_hi'][k])}", fpct(pc_) if tested else ""]
    pc0, pc1, n0, wk0 = (lambda v: f"{v * 100:.0f}%"), (lambda v: f"{v * 100:.1f}%"), (lambda v: f"{v:.0f}"), (lambda v: f2(v, 0))
    zz2 = (lambda v: f"{v:+.2f}")
    tab_z = {p: tbl(["", "the ratio", "random walks (median)", "random walks 5th–95th", "percentile"],
                    [zrow(p, "beyond1", "weeks beyond ±1 (outside the band)", pc0, True), zrow(p, "above1", "… above +1", pc0, False), zrow(p, "below1", "… below −1", pc0, False),
                     zrow(p, "beyond2", "weeks beyond ±2", pc1, True), zrow(p, "n_str", "stretches beyond ±1", n0, False), zrow(p, "med_str", "median stretch, weeks", wk0, True),
                     zrow(p, "max_str", "longest stretch, weeks", n0, False), zrow(p, "z_peak", "median z at the zig-zag peaks", zz2, True), zrow(p, "z_trough", "median z at the zig-zag troughs", zz2, True),
                     ["lowest z", f"{R[p]['z']['min'][0]:+.2f} ({R[p]['z']['min'][1]})", "", "", ""], ["highest z", f"{R[p]['z']['max'][0]:+.2f} ({R[p]['z']['max'][1]})", "", "", ""],
                     ["latest z", f"{R[p]['z']['last'][0]:+.2f} ({R[p]['z']['last'][1]})", "", "", ""]]) for p in ORDER}
    tab_z = {p + ZW.SEP + ZW.DEFAULT: v for p, v in tab_z.items()}
    for w, lab, f, n_ in ZW.WINDOWS:
        if w == ZW.DEFAULT:
            continue
        u = "weeks" if f == "W" else "trading days"
        for p in ORDER:
            q = R[p]["zw"][w]
            rows_ = [zrow(p, "beyond1", f"{u} beyond ±1", pc0, True, w), zrow(p, "above1", "… above +1", pc0, False, w), zrow(p, "below1", "… below −1", pc0, False, w), zrow(p, "beyond2", f"{u} beyond ±2", pc1, True, w),
                     zrow(p, "n_str", "stretches beyond ±1", n0, False, w), zrow(p, "med_str", f"median stretch, {u}", wk0, True, w), zrow(p, "max_str", f"longest stretch, {u}", n0, False, w)]
            if f == "W":
                rows_ += [zrow(p, "z_peak", "median z at the zig-zag peaks", zz2, True, w), zrow(p, "z_trough", "median z at the zig-zag troughs", zz2, True, w)]
            rows_ += [["lowest z", f"{q['min'][0]:+.2f} ({q['min'][1]})", "", "", ""], ["highest z", f"{q['max'][0]:+.2f} ({q['max'][1]})", "", "", ""], ["latest z", f"{q['last'][0]:+.2f} ({q['last'][1]})", "", "", ""]]
            tab_z[p + ZW.SEP + w] = tbl(["", "the ratio", "random walks (median)", "random walks 5th–95th", "percentile"], rows_)
    tab_spec = {p: tbl(["cycle length", "bins", "share of variance", "random-walk median", "ratio", "percentile"], [[b["band"] + " wk", b["bins"], fp(b.get("share", nan)).lstrip("+") if b["bins"] else "—", fp(b.get("null_median", nan)).lstrip("+") if b["bins"] else "—", f2(b.get("ratio", nan)), fpct(b.get("pct", nan))] for b in R[p]["spec"]["bands"]]) for p in ORDER}
    tab_vr = {p: tbl(["horizon", "VR", "random walks 5th–95th", "percentile"], [[f"{v['k']} wk", f2(v["vr"]) if v["reportable"] else "—", f"{v['null_lo']:.2f}–{v['null_hi']:.2f}" if v["reportable"] else "—", fpct(v["pct"])] for v in R[p]["vr"]]) for p in ORDER}
    tab_season = {p: tbl(["month", "years", "average change", "up in", "shuffles 5th–95th", "percentile"], [[m["month"], m["years"], fp(m["mean"], 2), f"{m['share_up'] * 100:.0f}%", f"{fp(m['null_lo'], 2)} … {fp(m['null_hi'], 2)}", fpct(m["pct"]) if R[p]["season"]["reportable"] else f"{m['pct']:.0f} (not tested)"] for m in R[p]["season"]["months"]]) for p in ORDER}
    tab_dur = {p: tbl(["", "the ratio", "random walks (median)", "percentile"], [["completed swings", R[p]["swing"]["n"], f"{R[p]['swing']['null_n_median']:.0f}", fpct(R[p]["swing"]["pct_n"])], ["median length, weeks", f2(R[p]["swing"]["median"], 0), f"{R[p]['swing']['null_median_median']:.0f}", fpct(R[p]["swing"]["pct_median"])],
                                                                                    ["regularity (CV of lengths; lower = more regular)", f2(R[p]["swing"]["cv"]), f"{R[p]['swing']['null_cv_median']:.2f}", fpct(R[p]["swing"]["pct_cv"])], ["median swing size", fp(R[p]["swing"]["median_amp_pct"]), "", ""], ["reversal threshold (from earlier weeks; median, range)", f"{R[p]['swing']['h_pct_median'] * 100:.1f}% ({R[p]['swing']['h_pct_min'] * 100:.1f}–{R[p]['swing']['h_pct_max'] * 100:.1f}%)", "same path", ""]]) for p in ORDER}
    summary_rows = [[nice(p), f"{R[p]['first']} → {R[p]['last']}", R[p]["n"], R[p]["verdict"]["vr"], R[p]["verdict"]["spectrum"], R[p]["verdict"]["season"], R[p]["verdict"]["swings"], R[p]["verdict"]["z"]] for p in ORDER]
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    H = [PAGE_HEAD.replace("__CSS__", CSS), "<header class='bar'><div class='bar-inner'><b>E01 · ratio cycles</b><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label>{ZW.select_html()}<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Is there cyclicality in the ratio series?</h1>",
         f"<p>The ten pair ratios P_a ÷ P_b (plain, not logged), weekly, on each pair's common calendar as T02/T03 use it. Every statistic is compared with {D:,} random walks that have the ratio's own week-by-week volatility (each week's change kept, its sign flipped at random), so a pattern any random series would also show is not mistaken for a cycle. Generated {audit['run_date']} from the approved data of 2026-11-12. <b>Description, not a finding</b>: nothing here tests a trading rule; anything that looks usable needs its own pre-registered test.</p>"
         f"<p><b>Point-in-time throughout.</b> At every week, the drift that is removed is the average of the weekly changes <i>before</i> it, and the swing threshold comes from their sd (expanding windows, after a {BURN}-week warm-up). No number from later in the history enters any series, threshold or random walk; the band and the z-score are the trailing 52 weeks. "
         f"Checked, not assumed: each pair's history is cut at 12 dates and everything recomputed from the cut history (the z-score included) must equal the full run up to the cut, with the turning points found on the cut history the first ones of the full run — {PIT[0]} checks, {len(PIT[1])} violations. "
         "The statistics themselves summarise each pair's whole window, as any description of a history does.</p>",
         f"<div class='card'><p style='margin:4px 0'><b>Against chance:</b> {n_not} of {n_all} percentiles on this page are notable (at or below the 5th or at or above the 95th); about {0.1 * n_all:.0f} would be if every ratio were a random walk. The pairs share legs and the checks overlap, so these are not independent.</p></div>",
         "<h2>What each pair shows</h2><div class='card'>" + tbl(["pair", f"window (after the {BURN}-week warm-up)", "weekly changes", "variance ratios (does a move continue or reverse?)", "spectrum (any cycle length?)", "month of year", "swings (zig-zag)", "z-score (how far from the trailing mean, in sds)"], summary_rows, cls="txt") + "</div>",
         "<h2>The data</h2><div class='card'>" + html_fig(figs["ratios"], first=True) + "</div>",
         "<div class='card'>" + html_fig(figs["indices"]) + "<details><summary>Table view</summary>" + tbl(["index", "first week", "last week", "first close", "last close", "growth a year"], idx_rows) + "</details></div>",
         "<h2>The chosen pair</h2><p class='note'>Pick the pair at the top; every chart below follows it. Grey ranges are the middle 90% of the random walks.</p>",
         per_pair("ratio", figs["ratio"], tab_ratio),
         "<h3>1 · Cycle lengths (spectrum)</h3><p class='note'>A regular cycle would show as the blue line rising above the grey range near its length. Below the range at long lengths = moves reverse; above = they persist. Only lengths that fit at least three times in the window are shown.</p>",
         per_pair("spec", figs["spec"], tab_spec),
         "<h3>2 · Does a move continue or reverse? (variance ratios)</h3><p class='note'>1 = random walk. Below the grey range at a horizon: moves over that many weeks tend to undo themselves, the swing-back the ±1 sd band bets on. Above: they tend to continue.</p>",
         per_pair("vr", figs["vr"], tab_vr),
         "<h3>3 · Month of year</h3><p class='note'>Blue dots: the ratio's average change in each calendar month. Grey bars: where the averages fall when each year's months are shuffled. A dot outside its bar is notable, but with 12 months about one in ten is expected to be by chance.</p>",
         per_pair("season", figs["season"], tab_season),
         "<h3>4 · How long the swings last</h3><p class='note'>Swings between zig-zag turning points on the drift-removed ratio. A real cycle would bunch the blue bars more tightly than the grey line (more regular than random).</p>",
         per_pair("dur", figs["dur"], tab_dur),
         f"<h3>5 · How far from its trailing mean? (z-score)</h3><p class='note'>z = (ratio − trailing {Z_WIN}-week mean) ÷ trailing sd, the same mean and sd as T02's band, so the grey strip (z between −1 and +1) is inside the band. Blue bars: the share of the ratio's weeks at each z. Grey line: the random walks' weeks, with the expanding drift added back so they trend like the ratio (a trailing z sits on a trend's side). "
         "Description only. The z at each turning point is in the table under the first chart. <b>The z window dropdown at the top</b> switches this section, the z panel under the ratio and the all-pairs z views between 52 weeks (the band) and shorter weekly and daily windows; daily windows use daily closes and daily random walks (z at turning points: weekly windows only)."
         + (f" Where the drift check below finds A and B disagreeing ({', '.join(nice(q) for q in flip_pairs)}), the z comparison is not reliable either: its random walks are B's, with the drift of the warm-up year added back." if flip_pairs else "") + "</p>",
         per_pair("z", figs["z"], tab_z),
         "<h2>All pairs at once</h2>",
         per_win("z_time", figs["z_time"], {w: tbl(["pair", "z from", "latest z", "lowest z", "highest z", "beyond ±1", "beyond ±2"],
                                                  [[nice(p), R[p]["zw"][w]["first_z"], f"{R[p]['zw'][w]['last'][0]:+.2f} ({R[p]['zw'][w]['last'][1]})", f"{R[p]['zw'][w]['min'][0]:+.2f} ({R[p]['zw'][w]['min'][1]})",
                                                    f"{R[p]['zw'][w]['max'][0]:+.2f} ({R[p]['zw'][w]['max'][1]})", f"{R[p]['zw'][w]['real']['beyond1'] * 100:.0f}%", f"{R[p]['zw'][w]['real']['beyond2'] * 100:.1f}%"] for p in ORDER])
                                              + "<p class='note'>Lowest, highest and the shares are over each pair's window after the warm-up; z_weekly.csv has every week of the 52-week z, zscores_windows.csv every window's summary.</p>" for w, _, _, _ in ZW.WINDOWS}),
         "<p class='note'>Below: colour is the percentile among the random walks: grey = what chance gives, blue or red = below or above it. Blank = not reportable for that pair's window.</p>",
         "<div class='card'>" + html_fig(figs["h_vr"]) + "<details><summary>Table view</summary>" + tbl(["pair"] + [wtick(k) for k in VR_K], [[nice(p)] + [(f"{v['vr']:.2f} ({fpct(v['pct'])})" if v["reportable"] else "—") for v in R[p]["vr"]] for p in ORDER]) + "</details></div>",
         "<div class='card'>" + html_fig(figs["h_spec"]) + "<details><summary>Table view</summary>" + tbl(["pair", "overall p"] + [f"{lo}–{hi} wk" for lo, hi in BANDS], [[nice(p), f"{R[p]['spec']['p_global']:.3f}"] + [(f"{b['ratio']:.2f} ({fpct(b['pct'])})" if b["bins"] else "—") for b in R[p]["spec"]["bands"]] for p in ORDER]) + "</details></div>",
         "<div class='card'>" + html_fig(figs["h_season"]) + "<details><summary>Table view</summary>" + tbl(["pair", "p"] + list(MONTHS), [[nice(p), f"{R[p]['season']['p']:.3f}" if R[p]["season"]["reportable"] else "too short"] + [(f"{m['mean'] * 100:+.1f}% ({fpct(m['pct'])})" if R[p]["season"]["reportable"] else f"{m['mean'] * 100:+.1f}%") for m in R[p]["season"]["months"]] for p in ORDER]) + "</details></div>",
         per_win("h_z", figs["h_z"], {w: tbl(["pair"] + [lab_ for _, lab_ in (Z_TESTED if f == "W" else Z_TESTED_D)],
                                             [[nice(p)] + [f"{(ZFMT if f == 'W' else ZFMT_D)[k](R[p]['zw'][w]['real'][k])} ({(ZFMT if f == 'W' else ZFMT_D)[k](R[p]['zw'][w]['null_median'][k])}, {fpct(R[p]['zw'][w]['pct'][k])})"
                                                           for k, _ in (Z_TESTED if f == "W" else Z_TESTED_D)] for p in ORDER]) + "<p class='note'>The ratio's value (random walks' median, percentile).</p>" for w, _, f, _ in ZW.WINDOWS}),
         "<h2>Does the drift removal matter?</h2><p class='note'>Both versions below use only past data.</p>",
         "<div class='card'><p style='margin:4px 0'><b>A</b> = this page: at each week the drift removed is the average of the weekly changes before it (expanding window). "
         "<b>B</b> = no drift removal at all: the real ratio's raw weekly changes, against the same random walks with that expanding drift added back. "
         f"Percentiles A → B; <b>bold</b> = the notable call changes. <b>{n_flip} of {n_dc}</b> statistics change their notable call ({by_check})" + (": " + "; ".join(flipped) if flipped else "") + ". "
         "The month-of-year check never removes a drift. "
         + (f"Where A and B disagree ({', '.join(nice(q) for q in flip_pairs)}), neither reading is reliable: a point-in-time drift starts from the warm-up year, and where that year was very different from what followed (the Micro250 pairs' warm-up is 2022) removing it leaves a slow trend in the residuals that pushes A towards 'continues', while adding it to the random walks pushes B the other way. " if flip_pairs else "")
         + "If anything here became a trading rule it would be tested on its own, pre-registered, with counted look-ahead checks; the turning points are confirmed only after the ratio has already reversed, so they cannot be traded as drawn.</p>"
         + tbl(["pair", "drift a year (description)"] + [f"VR {wtick(k)}" for k in VR_K] + ["swings", "median length", "regularity", "spectrum bands changed", "spectrum overall p"],
               [[nice(p), f"{(np.exp(R[p]['drift_wk'] * 52.18) - 1) * 100:+.1f}%"] + [(ab_html(a["pct"], b["pct"]) if a["reportable"] else "—") for a, b in zip(R[p]["vr"], R[p]["vr_raw"])]
                + [ab_html(R[p]["swing"][k], R[p]["swing_raw"][k]) for k, _ in SW_LAB] + [spec_changed(p), f"{R[p]['spec']['p_global']:.2f}&nbsp;→&nbsp;{R[p]['spec_raw']['p_global']:.2f}"] for p in ORDER]) + "</div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         f"<li><b>Series.</b> r_t = P_a ÷ P_b on the last trading day of each ISO week; weekly change g_t = ln(r_t ÷ r_t−1), which is the plain ratio's week-on-week change, the same whether the ratio is drawn plain or logged. The residual at week t is g_t minus the average of g over all earlier weeks (expanding); the first {BURN} weeks are a warm-up for that average and are not themselves analysed (the ratio charts still show them).</li>"
         f"<li><b>Random walks.</b> {D:,} per pair: each week's residual keeps its size and gets a random sign (seed default_rng([{SEED_SIGNS}, pair_id])); no demeaning. Keeps volatility and its clustering; removes momentum, reversion and cycles. Percentile = (paths below + ½ ties) ÷ paths.</li>"
         f"<li><b>Spectrum.</b> Periodogram of the weekly changes; power summed in cycle-length bands as a share of variance; lengths over weeks ÷ {MIN_CYCLES} are left out. The overall p uses the largest |log(power ÷ random-walk median)| across bands, so eight bands are one test. The curve averages the power over a ±12% window of cycle length (the same on every random path), for display only. No moving-average detrending anywhere: smoothing a random walk manufactures cycles about as long as the window (Slutsky–Yule).</li>"
         f"<li><b>Variance ratio.</b> VR(k) = variance of overlapping k-week changes ÷ (k × variance of weekly changes), k ≤ weeks ÷ {MIN_K_FRAC}.</li>"
         f"<li><b>Month of year.</b> Month-end to month-end changes; null = the months shuffled within each year (seed default_rng([{SEED_MONTHS}, pair_id])); tested only with ≥ {MIN_YEARS_MONTH} years in every month. The last month of every window ends 2026-08-28.</li>"
         f"<li><b>Swings.</b> Zig-zag on the drift-removed ratio: at week t a turning point is confirmed by a reversal of h_t = sd(weekly changes before t) × √{H_WEEKS} (one typical half-year move, known at t) from the last extreme; the same threshold path on every random walk. In plain per cent a trough needs a rise of e^h − 1 and a peak a fall of 1 − e^−h (the same distance in log terms).</li>"
         f"<li><b>z-score.</b> z_t = (r_t − mean of the last {Z_WIN} weekly closes including week t) ÷ their sd (n − 1), from the {Z_MIN}nd week: T02's band in units of its own sd, so z = ±1 is its edge. Every run checks that it equals (ratio − mean) ÷ sd from the engine's own band (analysis.ratio_engine.decide, plain form, window 52; only those columns are read; largest difference {ZC['engine_max_abs_diff']:.0e}) and that it is unchanged when the history is cut (in the look-ahead count above). "
         "Random walks for it: each path starts from the real ratio's warm-up weeks, then takes each week's point-in-time residual with the page's random sign plus the expanding drift (as in the drift check's B), and z is taken on that path's ratio; turning points on a path come from the zig-zag on its drift-removed path, as for the real ratio. "
         "A stretch = consecutive weeks above +1 or below −1, cut at the window's ends on the real ratio and the paths alike. Five z statistics get a percentile (weeks beyond ±1, beyond ±2, median stretch, median z at peaks and at troughs) and join the count against chance; they overlap with the variance ratios (a ratio that reverses more than a random walk also spends less time far from its mean). Description only: nothing about what follows a band touch is computed.</li>"
         f"<li><b>Limits.</b> 20 years hold only a few repetitions of a multi-year cycle; the {BURN}-week warm-up leaves the Micro250 pairs about 3½ years (2023–2026) and the gold pairs about 10½; the gold leg is a futures continuation with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); price indices without dividends. The design was written before the first run (explore/E01_ratio_cycles/E01.md); the switch to expanding estimates came after it and is recorded there.</li></ul></div>",
         "</main>", "<script>" + JS.replace("__DIV_LIGHT__", json.dumps(DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(DIV_DARK)) + "</script></body></html>"]
    out = os.path.join(OUT, "E01_cycles.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable {n_not} of {n_all} (≈{0.1 * n_all:.0f} by chance) → {out}")


PAGE_HEAD = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Ratio cycles</title><style>__CSS__</style></head><body>"
CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);color-scheme:dark}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:0 16px 48px}
header.bar{position:sticky;top:0;z-index:10;background:var(--page);border-bottom:1px solid var(--border);padding:8px 16px}
.bar-inner{max-width:1180px;margin:0 auto;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
select,button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:3px 8px}
button{cursor:pointer;margin-left:auto}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:8px 12px;margin:12px 0}
h1{font-size:22px;margin:20px 0 6px}h2{font-size:17px;margin:30px 0 4px}h3{font-size:15px;margin:24px 0 2px}
p,li{color:var(--ink2)}.note{font-size:13px;color:var(--ink2);margin:2px 0}
.tag{font-size:11px;letter-spacing:.04em;border:1px solid var(--border);border-radius:999px;padding:1px 8px;color:var(--ink2)}
table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}
th,td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}
.txt{min-width:860px}.txt td,.txt th{text-align:left}.txt td:nth-child(2){white-space:nowrap}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin:6px 0}
"""
JS = """
const LIGHT={ink:'#0b0b0b',ink2:'#52514e',muted:'#898781',grid:'#e1e0d9',axis:'#c3c2b7'};
const DARK={ink:'#f0efec',ink2:'#c3c2b7',muted:'#898781',grid:'#2c2c2a',axis:'#383835'};
const DIV_LIGHT=__DIV_LIGHT__, DIV_DARK=__DIV_DARK__;
const Z_LIGHT=__Z_LIGHT__, Z_DARK=__Z_DARK__;
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function plots(){return Array.from(document.querySelectorAll('.js-plotly-plot'));}
function applyTheme(){
  const d=isDark(),c=d?DARK:LIGHT;
  plots().forEach(div=>{
    const u={'font.color':c.ink,'title.font.color':c.ink,'legend.font.color':c.ink2};
    Object.keys(div.layout).forEach(k=>{if(/^[xy]axis\\d*$/.test(k)){u[k+'.gridcolor']=c.grid;u[k+'.linecolor']=c.axis;u[k+'.tickfont.color']=c.muted;u[k+'.title.font.color']=c.ink2;}});
    (div.layout.annotations||[]).forEach((a,i)=>{u['annotations['+i+'].font.color']=c.ink2;});
    (div.layout.shapes||[]).forEach((s,i)=>{u['shapes['+i+'].line.color']=c.axis;});
    Plotly.relayout(div,u);
    const hm=[];div.data.forEach((t,i)=>{if(t.type==='heatmap')hm.push(i);});
    if(hm.length)Plotly.restyle(div,{colorscale:[d?DIV_DARK:DIV_LIGHT],'colorbar.tickfont.color':c.muted,'colorbar.title.font.color':c.ink2},hm);
    const hz=[];div.data.forEach((t,i)=>{if(t.type==='heatmap'&&(t.meta==='z'||String(t.meta).startsWith('z§')))hz.push(i);});
    if(hz.length)Plotly.restyle(div,{colorscale:[d?Z_DARK:Z_LIGHT]},hz);
  });
  document.getElementById('theme').textContent=d?'Light':'Dark';
}
function curWin(){const s=document.getElementById('zwin');return s?s.value:null;}
function showPair(p){
  const w=curWin();
  document.querySelectorAll('.per-pair').forEach(card=>{
    const div=card.querySelector('.js-plotly-plot');if(!div)return;
    Plotly.restyle(div,{visible:div.data.map(t=>t.meta===p||(w!==null&&t.meta===p+'§'+w))});
    const T=JSON.parse(card.dataset.titles);
    Plotly.relayout(div,{'title.text':(w!==null&&T[p+'§'+w]!==undefined)?T[p+'§'+w]:T[p]});
  });
  document.querySelectorAll('.per-win').forEach(card=>{
    const div=card.querySelector('.js-plotly-plot');if(!div||w===null)return;
    Plotly.restyle(div,{visible:div.data.map(t=>String(t.meta).endsWith('§'+w))});
    const T=JSON.parse(card.dataset.titles||'{}');if(T[w]!==undefined)Plotly.relayout(div,{'title.text':T[w]});
  });
  document.querySelectorAll('[data-pair]').forEach(el=>{el.hidden=el.dataset.pair!==p||(el.dataset.win!==undefined&&el.dataset.win!==w);});
  document.querySelectorAll('[data-winonly]').forEach(el=>{el.hidden=el.dataset.winonly!==w;});
  try{localStorage.setItem('e01-pair',p);if(w!==null)localStorage.setItem('e01-pair-win',w);}catch(e){}
}
window.addEventListener('load',()=>{
  const sel=document.getElementById('pair');let p=sel.value;
  try{const s=localStorage.getItem('e01-pair');if(s&&Array.from(sel.options).some(o=>o.value===s)){p=s;sel.value=s;}}catch(e){}
  const ws=document.getElementById('zwin');
  if(ws){try{const s=localStorage.getItem('e01-pair-win');if(s&&Array.from(ws.options).some(o=>o.value===s))ws.value=s;}catch(e){}ws.addEventListener('change',()=>showPair(sel.value));}
  showPair(p);applyTheme();
  sel.addEventListener('change',e=>showPair(e.target.value));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';applyTheme();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',applyTheme);
});
"""
JS = JS.replace("__Z_LIGHT__", json.dumps(Z_LIGHT)).replace("__Z_DARK__", json.dumps(Z_DARK))   # filled here so pages reusing JS (E02) get valid code

if __name__ == "__main__":
    main()
