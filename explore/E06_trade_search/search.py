"""
E06 (EXPLORATORY, a search): trade by trade, when did the ratio switch work? Design: explore/E06_trade_search/E06.md
(written before the first run).

    python explore/E06_trade_search/search.py [--perms 2000] [--snoop 1000]

Every number on the page and in E06_results.md is computed here; the page's script only picks and draws them.
Not a finding and not a test of any rule.
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles")); sys.path.insert(0, os.path.join(ROOT, "explore", "E03_trade_conditions"))
from analysis.approved_data import GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402
import zwin as ZW  # noqa: E402
import variants as VR  # noqa: E402
import conditions as E3  # noqa: E402

CONDS = (  # key, label, format, short label
    ("z_depth", "|z| at entry", "num", "|z|"), ("zslope4", "how fast |z| rose over the last 4 weeks", "num", "|z| rise 4 wk"),
    ("since_zero", "weeks since z changed sign", "wk", "weeks since sign change"), ("band_width", "band width (2 sd ÷ mean)", "pct", "band width"),
    ("breadth", "share of the ten ratios beyond ±1", "pct", "breadth"),
    ("lag4", "held − peer, last 4 weeks", "pct", "lag 4 wk"), ("lag13", "held − peer, last 13 weeks", "pct", "lag 13 wk"),
    ("held13", "held leg's 13-week return", "pct", "held 13 wk"), ("peer13", "peer's 13-week return", "pct", "peer 13 wk"),
    ("held52", "held leg's 52-week return", "pct", "held 52 wk"), ("peer52", "peer's 52-week return", "pct", "peer 52 wk"),
    ("trend52", "ratio's 52-week change in the held leg's favour", "pct", "trend 52 wk"), ("trend104", "ratio's 104-week change in the held leg's favour", "pct", "trend 104 wk"),
    ("long_cheap", "held leg's cheapness on a 156-week z", "num", "156-wk cheapness"),
    ("corr26", "legs' correlation, 26 weeks", "num", "corr 26 wk"), ("corr52", "legs' correlation, 52 weeks", "num", "corr 52 wk"),
    ("corr52_chg13", "change in the 52-week correlation over 13 weeks", "num", "corr change"),
    ("vol_held", "held leg's volatility, 26 weeks", "pct", "vol held"), ("vol_peer", "peer's volatility, 26 weeks", "pct", "vol peer"), ("vol_ratio", "ratio's volatility, 26 weeks", "pct", "vol ratio"),
    ("mkt52", "N100's 52-week return", "pct", "N100 52 wk"), ("mkt_dd", "N100's drawdown from its high", "pct", "N100 drawdown"), ("mkt_ma200", "N100 against its 200-day average", "pct", "N100 vs 200-day"),
    ("held_first", "held leg is the first leg (larger index; the index in a gold pair)", "bin", "held first leg"),
)
CKEYS = [c[0] for c in CONDS]; CLAB = {c[0]: c[1] for c in CONDS}; CFMT = {c[0]: c[2] for c in CONDS}; CSHORT = {c[0]: c[3] for c in CONDS}
E03_SHARED = ("corr26", "corr52", "mkt52", "mkt_dd", "vol_held", "vol_peer", "vol_ratio", "band_width", "z_depth", "since_zero", "lag4", "lag13", "held13", "peer13", "breadth")
SPLIT = pd.Timestamp("2016-01-01"); SEED_UNI, SEED_SNOOP, SEED_VAL, SEED_RND = 108, 109, 110, 111; RND = 200
MIN_SHARE, MIN_N = 0.15, 40; COST = VR.COST
ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR; LO, HI = E1.LO, E1.HI
POOL = {p: ("gold" if GOLD in p else "index") for p in PAIRS}
WLAB = {w: f"{lab}{' (daily)' if f == 'D' else ''}" for w, lab, f, _ in ZW.WINDOWS}


def r4(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def fmtv(key, v):
    if v is None or v != v:
        return "—"
    k = CFMT[key]
    return {"pct": f"{v * 100:+.1f}%", "num": f"{v:+.2f}", "wk": f"{v:.0f}", "bin": "yes" if v >= 0.5 else "no"}[k]


# ------------------------------------------------------------------------------------------------------- the condition frame
def base_frame(pf, mkt):
    """Pair-level series at every decision week (not yet oriented to a held leg)."""
    wk, c, ra, rb, lr = E3.condition_series(pf); dec = wk.index; lnr = np.log(wk["ratio"]); b = pd.DataFrame(index=dec)
    for k in ("corr26", "corr52", "vol_a", "vol_b", "vol_ratio", "ret_a4", "ret_b4", "ret_a13", "ret_b13"):
        b[k] = c[k]
    b["ret_a52"] = wk["ca"] / wk["ca"].shift(52) - 1; b["ret_b52"] = wk["cb"] / wk["cb"].shift(52) - 1
    b["dlnr52"] = lnr - lnr.shift(52); b["dlnr104"] = lnr - lnr.shift(104); b["z156"] = ZW.z_of(wk["ratio"], 156)
    b["corr52_chg13"] = c["corr52"] - c["corr52"].shift(13)
    m_at = mkt.asof(dec).to_numpy(); b["mkt52"] = m_at / mkt.asof(dec - pd.Timedelta(days=364)).to_numpy() - 1
    b["mkt_dd"] = m_at / mkt.cummax().asof(dec).to_numpy() - 1; b["mkt_ma200"] = m_at / mkt.rolling(200).mean().asof(dec).to_numpy() - 1
    return b


def breadth(dec, zall):
    """Share of the ten ratios whose z (same window) is beyond ±1 at each date (E03's rule: the last value at or before the
    date, if no more than 10 days old)."""
    vals = []
    for s in zall.values():
        si = s.index.values; idx = np.searchsorted(si, dec.values, side="right") - 1; ok = idx >= 0
        age = np.full(len(dec), np.inf); age[ok] = (dec.values[ok] - si[idx[ok]]) / np.timedelta64(1, "D")
        v = np.full(len(dec), np.nan); good = ok & (age <= 10); v[good] = s.to_numpy()[idx[good]]; vals.append(v)
    V = np.vstack(vals); known = ~np.isnan(V)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.where(known.sum(axis=0) > 0, (np.abs(np.nan_to_num(V)) > 1).sum(axis=0) / known.sum(axis=0), np.nan)


def window_frame(b, zc, br):
    """Every condition at every decision week, oriented to the leg the rule would hold that week (a when z < 0, b when z > 0)."""
    z = zc["z"]; A = (z < 0).to_numpy(); ok = z.notna().to_numpy(); sg = np.where(A, 1.0, -1.0); f = pd.DataFrame(index=zc.index)
    f["z"] = z; f["z_depth"] = z.abs(); f["zslope4"] = z.abs() - z.abs().shift(4); f["since_zero"] = zc["since_zero"]; f["band_width"] = zc["band_width"]; f["breadth"] = br
    pick = lambda xa, xb: np.where(A, b[xa].to_numpy(), b[xb].to_numpy())  # noqa: E731
    f["lag4"] = np.where(A, b["ret_a4"] - b["ret_b4"], b["ret_b4"] - b["ret_a4"]); f["lag13"] = np.where(A, b["ret_a13"] - b["ret_b13"], b["ret_b13"] - b["ret_a13"])
    f["held13"] = pick("ret_a13", "ret_b13"); f["peer13"] = pick("ret_b13", "ret_a13"); f["held52"] = pick("ret_a52", "ret_b52"); f["peer52"] = pick("ret_b52", "ret_a52")
    f["trend52"] = sg * b["dlnr52"].to_numpy(); f["trend104"] = sg * b["dlnr104"].to_numpy(); f["long_cheap"] = -sg * b["z156"].to_numpy()
    for k in ("corr26", "corr52", "corr52_chg13", "vol_ratio", "mkt52", "mkt_dd", "mkt_ma200"):
        f[k] = b[k].to_numpy()
    f["vol_held"] = pick("vol_a", "vol_b"); f["vol_peer"] = pick("vol_b", "vol_a"); f["held_first"] = A.astype(float)
    oriented = ["lag4", "lag13", "held13", "peer13", "held52", "peer52", "trend52", "trend104", "long_cheap", "vol_held", "vol_peer", "held_first"]
    f.loc[~ok, oriented] = np.nan
    return f


def frame_for(pf, mkt, w, zall):
    zc = E3.zcond(pf, w); return window_frame(base_frame(pf, mkt), zc, breadth(zc.index, zall)), zc


# ------------------------------------------------------------------------------------------------------- statistics
def uni(df, key, rng, P):
    """Rank correlation of the condition with the extra (both ranked within each pair) against P within-pair shuffles; thirds."""
    sub = df[[key, "extra", "pair", "half"]].dropna(); out = {"key": key, "n": int(len(sub))}
    if len(sub) < 20:
        return {**out, "rho": np.nan, "pct": np.nan, "rho_D": np.nan, "rho_V": np.nan, **{f"{s}_{t}": np.nan for t in ("lo", "mid", "hi") for s in ("n", "mean", "win")}}
    def rho_of(d):
        if len(d) < 10:
            return np.nan
        xr = d.groupby("pair")[key].rank(pct=True).to_numpy(); yr = d.groupby("pair")["extra"].rank(pct=True).to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1]) if xr.std() > 0 and yr.std() > 0 else np.nan
    xr = sub.groupby("pair")[key].rank(pct=True).to_numpy(); yr = sub.groupby("pair")["extra"].rank(pct=True).to_numpy(); pr = sub["pair"].to_numpy()
    out["rho"] = float(np.corrcoef(xr, yr)[0, 1]) if xr.std() > 0 else np.nan
    YR = np.tile(yr, (P, 1))
    for p in np.unique(pr):
        idx = np.flatnonzero(pr == p); YR[:, idx] = yr[rng.permuted(np.tile(idx, (P, 1)), axis=1)]
    xs = (xr - xr.mean()) / xr.std() if xr.std() > 0 else xr * 0; YS = (YR - YR.mean(axis=1, keepdims=True)) / YR.std(axis=1, keepdims=True)
    out["pct"] = E1.pctile(out["rho"], YS @ xs / len(xs)) if xr.std() > 0 else np.nan
    out["rho_D"] = rho_of(sub[sub["half"] == "D"]); out["rho_V"] = rho_of(sub[sub["half"] == "V"])
    y = sub["extra"].to_numpy()
    if CFMT[key] == "bin":
        groups = {"lo": sub[key].to_numpy() < 0.5, "mid": np.zeros(len(sub), bool), "hi": sub[key].to_numpy() >= 0.5}
    else:
        t3 = np.minimum(np.ceil(xr * 3), 3); groups = {"lo": t3 == 1, "mid": t3 == 2, "hi": t3 == 3}
    for t, g in groups.items():
        out[f"n_{t}"] = int(g.sum()); out[f"mean_{t}"] = float(y[g].mean()) if g.any() else np.nan; out[f"win_{t}"] = float((y[g] > 0).mean()) if g.any() else np.nan
    return out


def shuffles(groups, y, P, rng):
    """P columns of y shuffled within each group (rows = trades)."""
    Y = np.tile(y[:, None], (1, P)).astype(np.float32)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) > 1:
            Y[idx, :] = y[idx][rng.permuted(np.tile(np.arange(len(idx)), (P, 1)), axis=1).T]
    return Y


def run_stats(nav, bh, start=None):
    nv = pd.DataFrame({"nav": nav, "bh": bh}).dropna()
    if start is not None:
        nv = nv[nv.index >= start]
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25; cg = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
    return {"cagr": float(cg(nv["nav"])), "cagr_5050": float(cg(nv["bh"])), "excess": float(cg(nv["nav"]) - cg(nv["bh"])), "first": nv.index[0].date().isoformat(), "last": nv.index[-1].date().isoformat()}


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--snoop", type=int, default=1000)
    a_ = ap.parse_args(); P, PS = a_.perms, a_.snoop
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily(); mkt = px["N100"]["close"].astype(float)
    checks = {"t03_mismatch": [], "e03_max_abs_diff": None, "e04_piece_max_abs_diff": None, "cut_checks": 0, "cut_violations": []}
    pfs = {p: pair_frame(px, p, "plain") for p in PAIRS}
    zall = {w: {p: E3.zcond(pfs[p], w)["z"].dropna() for p in PAIRS} for w, *_ in ZW.WINDOWS}
    frames, rows, runs, ratio_s, zser = {}, [], {}, {}, {}
    for pair in PAIRS:
        pf = pfs[pair]; b = base_frame(pf, mkt); wk = pf.loc[pf["decision"]]; ratio_s[pair] = wk["ratio"]; zser[pair] = {}
        for w, *_ in ZW.WINDOWS:
            zc = E3.zcond(pf, w); f = window_frame(b, zc, breadth(zc.index, zall[w])); frames[(pair, w)] = f; zser[pair][w] = zc["z"]
            r = VR.run_states(pf, zc["z"], COST, 1.0); runs[(pair, w)] = r
            if w == ZW.DEFAULT:
                checks["t03_mismatch"] += VR.check_against_t03({"pair": pair, **r})
            nv = pd.DataFrame({"nav": r["nav"], "bh": r["bh"]}).dropna(); lr = np.log(nv["nav"] / nv["bh"]); days = lr.index; dec = f.index
            for _, t in r["extras"].iterrows():
                ed = pd.Timestamp(t["entry_decision"]); et = pd.Timestamp(t["entry_trade"]); op = bool(t["open"]); xt = None if op else pd.Timestamp(t["exit_trade"])
                piece = np.nan if op else float(lr.iloc[max(days.searchsorted(xt) - 1, 0)] - lr.iloc[max(days.searchsorted(et) - 1, 0)])
                la, lb = pair.split("|"); row = {"window": w, "pair": pair, "pool": POOL[pair], "half": "D" if ed < SPLIT else "V", "entry_decision": ed.date().isoformat(), "entry_trade": et.date().isoformat(),
                                                 "exit_trade": None if op else xt.date().isoformat(), "held": t["held"], "held_leg": la if t["held"] == "A" else lb, "open": op,
                                                 "extra": np.nan if op else float(t["extra"]), "held_ret": np.nan if op else float(t["held_ret"]), "peer_ret": np.nan if op else float(t["peer_ret"]),
                                                 "weeks": np.nan if op else int(((dec > ed) & (dec < xt)).sum()), "days": np.nan if op else int((xt - et).days), "piece": piece, "z_entry": float(f.loc[ed, "z"])}
                row.update({k: float(f.loc[ed, k]) for k in CKEYS}); rows.append(row)
        print(f"  {pair:18s} [{time.time() - t0:4.0f}s]", flush=True)
    TR = pd.DataFrame(rows); TR["good"] = TR["extra"] > 0
    if checks["t03_mismatch"]:
        sys.exit(f"the 52-week run is not T03: {checks['t03_mismatch']}")
    # the shared conditions must equal E03's; the pieces must equal E04's
    e3 = pd.read_csv(os.path.join(ROOT, "explore", "E03_trade_conditions", "out", "trades_conditions_windows.csv"))
    m3 = TR[~TR["open"]].merge(e3, on=["window", "pair", "entry_decision"], suffixes=("", "_e3"), how="outer", indicator=True)
    if (m3["_merge"] != "both").any():
        sys.exit(f"trades differ from E03's: {(m3['_merge'] != 'both').sum()} unmatched")
    def maxdiff(a, b):   # a value missing on one side only counts as a failure
        a, b = np.asarray(a, float), np.asarray(b, float); d = np.abs(a - b); d[np.isnan(a) != np.isnan(b)] = np.inf
        return float(np.nanmax(d)) if (~np.isnan(d)).any() else 0.0
    checks["e03_max_abs_diff"] = float(max(maxdiff(m3[k], m3[f"{k}_e3"]) for k in E03_SHARED + ("extra",)))
    e4 = pd.read_csv(os.path.join(ROOT, "explore", "E04_signal_strength", "out", "shortfall_by_trade_windows.csv")); e4 = e4[(e4["kind"] == "trade") & (~e4["open"])]
    m4 = TR[~TR["open"]].merge(e4, left_on=["window", "pair", "entry_trade"], right_on=["window", "pair", "start"], how="outer", indicator=True)
    if (m4["_merge"] != "both").any():
        sys.exit(f"trades differ from E04's: {(m4['_merge'] != 'both').sum()} unmatched")
    checks["e04_piece_max_abs_diff"] = maxdiff(m4["piece"], m4["log_contrib"])
    if not (checks["e03_max_abs_diff"] < 1e-9 and checks["e04_piece_max_abs_diff"] < 1e-9):
        sys.exit(f"cross-checks FAILED: {checks}")
    # look-ahead: the frame recomputed from the history cut at 12 dates (breadth excepted)
    for pair in PAIRS:
        pf = pfs[pair]; dec = pf.index[pf["decision"]]
        for T in np.unique(np.linspace(160, len(dec) - 1, 12).astype(int)):
            cut = pf.loc[: dec[T]]; b_ = base_frame(cut, mkt[: dec[T]])
            for w, *_ in ZW.WINDOWS:
                zc_ = E3.zcond(cut, w); f_ = window_frame(b_, zc_, np.full(len(zc_), np.nan)).drop(columns="breadth"); full = frames[(pair, w)].drop(columns="breadth").iloc[: len(f_)]
                checks["cut_checks"] += 1
                if not np.allclose(f_.to_numpy(float), full.to_numpy(float), rtol=0, atol=1e-10, equal_nan=True):
                    checks["cut_violations"].append(f"{pair} {w} cut {dec[T].date()}")
    if checks["cut_violations"]:
        sys.exit(f"look-ahead check FAILED: {checks['cut_violations'][:5]}")
    TR.to_csv(os.path.join(OUT, "trades_all.csv"), index=False)
    print(f"  {len(TR)} trades; E03 equal (max {checks['e03_max_abs_diff']:.1e}), E04 pieces equal (max {checks['e04_piece_max_abs_diff']:.1e}), look-ahead {checks['cut_checks']} cuts, 0 violations  [{time.time() - t0:4.0f}s]", flush=True)

    # ---------------------------------------------------------------- 2. condition by condition
    CL = TR[~TR["open"]]; IX = CL[CL["pool"] == "index"]; GD = CL[CL["pool"] == "gold"]; urows = []
    wins = [w for w, *_ in ZW.WINDOWS] + ["all"]
    for wi, w in enumerate(wins):
        sub = IX if w == "all" else IX[IX["window"] == w]
        for ci, k in enumerate(CKEYS):
            urows.append({"window": w, "pool": "index", **uni(sub.assign(pair=sub["pair"] + ("|" + sub["window"] if w == "all" else "")), k, np.random.default_rng([SEED_UNI, wi, ci]), P)})
    for ci, k in enumerate(CKEYS):
        urows.append({"window": "all", "pool": "gold", **uni(GD.assign(pair=GD["pair"] + "|" + GD["window"]), k, np.random.default_rng([SEED_UNI, 99, ci]), P)})
    UN = pd.DataFrame(urows); UN.to_csv(os.path.join(OUT, "univariate.csv"), index=False)
    print(f"  univariate [{time.time() - t0:4.0f}s]", flush=True)

    # ---------------------------------------------------------------- 3. the search
    D = IX[IX["half"] == "D"].reset_index(drop=True); V = IX[IX["half"] == "V"].reset_index(drop=True); G = GD.reset_index(drop=True)
    singles = []
    for k in CKEYS:
        if CFMT[k] == "bin":
            singles += [(k, "==", 1.0), (k, "==", 0.0)]
        else:
            for q in (1 / 3, 0.5, 2 / 3):
                t = float(D[k].quantile(q)); singles += [(k, "<=", t), (k, ">", t)]
    def smask(df, s):
        x = df[s[0]].to_numpy(float); ok = ~np.isnan(x)
        return ok & ((x == s[2]) if s[1] == "==" else ((x <= s[2]) if s[1] == "<=" else (x > s[2])))
    SD = {s: smask(D, s) for s in singles}; SV = {s: smask(V, s) for s in singles}; SG = {s: smask(G, s) for s in singles}
    rules = [(s,) for s in singles] + [(s1, s2) for s1, s2 in itertools.combinations(singles, 2) if s1[0] != s2[0]]
    need = max(MIN_SHARE * len(D), MIN_N)
    MD = np.array([np.logical_and.reduce([SD[s] for s in r]) for r in rules]); elig = MD.sum(axis=1) >= need
    rules = [r for r, e in zip(rules, elig) if e]; MD = MD[elig].astype(np.float32)
    MV = np.array([np.logical_and.reduce([SV[s] for s in r]) for r in rules]).astype(np.float32); MG = np.array([np.logical_and.reduce([SG[s] for s in r]) for r in rules]).astype(np.float32)
    yD, yV, yG = D["extra"].to_numpy(np.float32), V["extra"].to_numpy(np.float32), G["extra"].to_numpy(np.float32)
    nD, nV, nG = MD.sum(axis=1), MV.sum(axis=1), MG.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sD = MD @ yD / nD; wD = MD @ (yD > 0).astype(np.float32) / nD
        sV = np.where(nV > 0, MV @ yV / np.maximum(nV, 1), np.nan); wV = np.where(nV > 0, MV @ (yV > 0).astype(np.float32) / np.maximum(nV, 1), np.nan)
        sG = np.where(nG > 0, MG @ yG / np.maximum(nG, 1), np.nan); wG = np.where(nG > 0, MG @ (yG > 0).astype(np.float32) / np.maximum(nG, 1), np.nan)
    print(f"  search: {len(singles)} single conditions, {len(rules)} eligible rules (≥ {need:.0f} of {len(D)} discovery trades)  [{time.time() - t0:4.0f}s]", flush=True)
    gD = (D["pair"] + "|" + D["window"]).to_numpy(); gV = (V["pair"] + "|" + V["window"]).to_numpy()
    YD = shuffles(gD, yD, PS, np.random.default_rng([SEED_SNOOP, 0])); null_max = (MD @ YD / nD[:, None]).max(axis=0)
    real_max = float(sD.max()); snoop_pct = E1.pctile(real_max, null_max)
    adj_p = np.array([(null_max >= s).mean() for s in sD])
    YV = shuffles(gV, yV, PS, np.random.default_rng([SEED_VAL, 0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SVn = MV @ YV / np.maximum(nV, 1)[:, None]
    pV = np.array([E1.pctile(sV[i], SVn[i]) if nV[i] >= 10 else np.nan for i in range(len(rules))])
    print(f"  snooping null and validation shuffles  [{time.time() - t0:4.0f}s]", flush=True)
    def rtxt(r):
        out = []
        for k, op, t in r:
            if CFMT[k] == "bin":
                out.append(f"{CSHORT[k]}: {'yes' if t >= 0.5 else 'no'}")
            else:
                out.append(f"{CSHORT[k]} {'≤' if op == '<=' else '>'} {fmtv(k, t).lstrip('+') if CFMT[k] != 'pct' else fmtv(k, t)}")
        return " and ".join(out)
    RU = pd.DataFrame({"rule": [rtxt(r) for r in rules], "conditions": [len(r) for r in rules], "n_D": nD.astype(int), "mean_D": sD, "win_D": wD, "adj_p": adj_p,
                       "n_V": nV.astype(int), "mean_V": sV, "win_V": wV, "pct_V": pV, "n_G": nG.astype(int), "mean_G": sG, "win_G": wG})
    RU["spec"] = [json.dumps([[k, op, t] for k, op, t in r]) for r in rules]
    RU = RU.sort_values("mean_D", ascending=False).reset_index(drop=True); RU.to_csv(os.path.join(OUT, "rules.csv"), index=False)
    rho_dv = float(pd.Series(RU["mean_D"]).corr(pd.Series(RU["mean_V"]), method="spearman"))
    top = RU.head(20); base = {"D": (len(D), float(yD.mean()), float((yD > 0).mean())), "V": (len(V), float(yV.mean()), float((yV > 0).mean())), "G": (len(G), float(yG.mean()), float((yG > 0).mean()))}
    n_val_pos = int(((RU["mean_V"] > base["V"][1]) & (RU["pct_V"] >= HI)).sum()); n_val_neg = int(((RU["pct_V"] <= LO)).sum())

    # ---------------------------------------------------------------- 4. the top three discovery rules as filters
    frows = []; top3 = [json.loads(s) for s in RU["spec"].head(3)]
    def rule_mask_frame(f, spec):
        m = np.ones(len(f), bool); known = np.ones(len(f), bool)
        for k, op, t in spec:
            x = f[k].to_numpy(float); known &= ~np.isnan(x)
            m &= (x == t) if op == "==" else ((x <= t) if op == "<=" else (x > t))
        return m & known, known
    for ri, spec in enumerate(top3):
        for pair in [p for p in PAIRS if POOL[p] == "index"]:
            pf = pfs[pair]
            for w, *_ in ZW.WINDOWS:
                f = frames[(pair, w)]; m, known = rule_mask_frame(f, spec); z = zser[pair][w]; base_r = runs[(pair, w)]
                for ver, allow in (("every entry", None), ("rule holds", m), ("rule fails", known & ~m)):
                    r = base_r if allow is None else VR.run_states(pf, z, COST, 1.0, allow=allow, comp=False)
                    ex = r["extras"]; cl = ex[~ex["open"].astype(bool)] if len(ex) else ex
                    sf = run_stats(r["nav"], base_r["bh"]); sv = run_stats(r["nav"], base_r["bh"], SPLIT)
                    frows.append({"rule_rank": ri + 1, "rule": RU["rule"].iloc[ri], "pair": pair, "window": w, "version": ver, "trades": int(len(cl)), "win": float((cl["extra"] > 0).mean()) if len(cl) else np.nan,
                                  "excess_full": sf["excess"], "cagr_5050_full": sf["cagr_5050"], "first": sf["first"], "last": sf["last"], "excess_val": sv["excess"], "cagr_5050_val": sv["cagr_5050"], "first_val": sv["first"]})
    FI = pd.DataFrame(frows); FI.to_csv(os.path.join(OUT, "filters.csv"), index=False)
    fsum = []
    for (ri, w), g in FI.groupby(["rule_rank", "window"]):
        h = g[g["version"] == "rule holds"].set_index("pair"); fl = g[g["version"] == "rule fails"].set_index("pair"); e = g[g["version"] == "every entry"].set_index("pair")
        fsum.append({"rule_rank": int(ri), "window": w, "holds_beats_fails_val": int((h["excess_val"] > fl["excess_val"]).sum()), "holds_beats_fails_full": int((h["excess_full"] > fl["excess_full"]).sum()),
                     "holds_beats_5050_val": int((h["excess_val"] > 0).sum()), "every_beats_5050_val": int((e["excess_val"] > 0).sum()), "median_diff_val": float((h["excess_val"] - fl["excess_val"]).median()), "pairs": int(len(h))})
    FS = pd.DataFrame(fsum)
    print(f"  filters  [{time.time() - t0:4.0f}s]", flush=True)
    # 4b (added after the first run): does the rule beat skipping the same share of entries at random? Per index pair and
    # window: the weeks where the rule can be evaluated, an entry allowed at random with the rule's own rate, RND draws.
    rrows = []; IXP = [p for p in PAIRS if POOL[p] == "index"]
    for ri, spec in enumerate(top3):
        for pi, pair in enumerate(IXP):
            pf = pfs[pair]
            for wi, (w, *_) in enumerate(ZW.WINDOWS):
                f = frames[(pair, w)]; m, known = rule_mask_frame(f, spec); z = zser[pair][w]; base_r = runs[(pair, w)]
                known = known & z.notna().to_numpy(); prob = float((m & known).sum() / max(known.sum(), 1)); rng = np.random.default_rng([SEED_RND, ri, pi, wi]); ex_r = []
                for _ in range(RND):
                    r = VR.run_states(pf, z, COST, 1.0, allow=known & (rng.random(len(z)) < prob), comp=False); ex_r.append(run_stats(r["nav"], base_r["bh"], SPLIT)["excess"])
                real = float(FI[(FI["rule_rank"] == ri + 1) & (FI["pair"] == pair) & (FI["window"] == w) & (FI["version"] == "rule holds")]["excess_val"].iloc[0])
                rrows.append({"rule_rank": ri + 1, "pair": pair, "window": w, "rate": prob, "excess_val": real, "random_median": float(np.median(ex_r)), "pct": E1.pctile(real, ex_r)})
    RF = pd.DataFrame(rrows); RF.to_csv(os.path.join(OUT, "filters_vs_random.csv"), index=False)
    RFS = RF.groupby(["rule_rank", "window"]).agg(beats_random_median=("pct", lambda s: int((s > 50).sum())), notable_high=("pct", lambda s: int((s >= HI).sum())), median_pct=("pct", "median")).reset_index()
    FS = FS.merge(RFS, on=["rule_rank", "window"])
    print(f"  random filters ({RND} per pair and window)  [{time.time() - t0:4.0f}s]", flush=True)
    # 3b (added after the first run): the top three rules year by year (index pairs, all windows): does the rule pick the
    # better trades within a year, or mainly the better years?
    def trade_mask(df, spec):
        m = np.ones(len(df), bool)
        for k, op, t in spec:
            x = df[k].to_numpy(float); m &= ~np.isnan(x) & ((x == t) if op == "==" else ((x <= t) if op == "<=" else (x > t)))
        return m
    yrows, ysum = [], []; IXY = IX.assign(year=IX["entry_trade"].str[:4].astype(int))
    for ri, spec in enumerate(top3):
        sel = trade_mask(IXY, spec); g = IXY.assign(sel=sel).groupby(["year", "sel"])["extra"].agg(["size", "mean"]).unstack()
        for y in sorted(IXY["year"].unique()):
            n1 = int(g.loc[y, ("size", True)]) if (y in g.index and ("size", True) in g.columns and g.loc[y, ("size", True)] == g.loc[y, ("size", True)]) else 0
            n0 = int(g.loc[y, ("size", False)]) if (y in g.index and ("size", False) in g.columns and g.loc[y, ("size", False)] == g.loc[y, ("size", False)]) else 0
            yrows.append({"rule_rank": ri + 1, "year": int(y), "n_sel": n1, "mean_sel": float(g.loc[y, ("mean", True)]) if n1 else np.nan, "n_rest": n0, "mean_rest": float(g.loc[y, ("mean", False)]) if n0 else np.nan})
        yy = pd.DataFrame([r for r in yrows if r["rule_rank"] == ri + 1]); both = yy[(yy["n_sel"] >= 5) & (yy["n_rest"] >= 5)]
        top_years = yy.sort_values("n_sel", ascending=False).head(3)
        ysum.append({"rule_rank": ri + 1, "years_with_selected": int((yy["n_sel"] > 0).sum()), "years_compared": int(len(both)), "selected_better": int((both["mean_sel"] > both["mean_rest"]).sum()),
                     "share_in_3_busiest_years": float(top_years["n_sel"].sum() / max(yy["n_sel"].sum(), 1)), "busiest_years": ", ".join(str(int(y)) for y in top_years["year"])})
    YR = pd.DataFrame(yrows); YS = pd.DataFrame(ysum); YR.to_csv(os.path.join(OUT, "rules_by_year.csv"), index=False)

    # ---------------------------------------------------------------- per-window overview
    ov = []
    for w, *_ in ZW.WINDOWS:
        for pool in ("index", "gold"):
            s = CL[(CL["window"] == w) & (CL["pool"] == pool)]
            ov.append({"window": w, "pool": pool, "n": int(len(s)), "win": float(s["good"].mean()), "mean": float(s["extra"].mean()), "median": float(s["extra"].median()),
                       "best": float(s["extra"].max()), "worst": float(s["extra"].min()), "first": s["entry_trade"].min(), "last": s["exit_trade"].max(),
                       "win_short": float(s[s["weeks"] <= 13]["good"].mean()), "win_long": float(s[s["weeks"] > 52]["good"].mean()) if (s["weeks"] > 52).any() else np.nan,
                       "n_short": int((s["weeks"] <= 13).sum()), "n_long": int((s["weeks"] > 52).sum())})
    OV = pd.DataFrame(ov)
    # consistency of each condition across windows (index pairs)
    cons = []
    for k in CKEYS:
        u = UN[(UN["pool"] == "index") & (UN["key"] == k) & (UN["window"] != "all")]; a = UN[(UN["pool"] == "index") & (UN["key"] == k) & (UN["window"] == "all")].iloc[0]
        g = UN[(UN["pool"] == "gold") & (UN["key"] == k)].iloc[0]
        cons.append({"key": k, "label": CLAB[k], "pos": int((u["pct"] >= HI).sum()), "neg": int((u["pct"] <= LO).sum()), "rho_all": a["rho"], "pct_all": a["pct"], "rho_D": a["rho_D"], "rho_V": a["rho_V"],
                     "same_sign_halves": bool(a["rho_D"] * a["rho_V"] > 0) if a["rho_D"] == a["rho_D"] and a["rho_V"] == a["rho_V"] else False, "rho_gold": g["rho"], "pct_gold": g["pct"]})
    CO = pd.DataFrame(cons)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY (search)", "inputs": fingerprints(), "perms": P, "snoop_draws": PS, "split": SPLIT.date().isoformat(),
             "seeds": {"univariate": f"default_rng([{SEED_UNI}, window_id, condition_id]) (gold: window_id 99)", "snooping": f"default_rng([{SEED_SNOOP}, 0])", "validation": f"default_rng([{SEED_VAL}, 0])"},
             "conditions": CKEYS, "min_support": {"share": MIN_SHARE, "n": MIN_N, "applied": need}, "checks": checks, "trades": int(len(TR)), "closed": int(len(CL)),
             "search": {"single": len(singles), "eligible_rules": len(rules), "discovery_trades": len(D), "validation_trades": len(V), "gold_trades": len(G), "best_discovery_mean": real_max,
                        "best_discovery_pct_among_noise_searches": snoop_pct, "spearman_discovery_vs_validation": rho_dv, "rules_validating_high": n_val_pos, "rules_validating_low": n_val_neg},
             "added_after_first_run": {"random_filters": {"draws": RND, "seed": f"default_rng([{SEED_RND}, rule, pair_id, window_id])"}, "year_by_year": YS.to_dict("records")},
             "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return "—" if v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
    def pc(v): return "—" if v != v else f"{v * 100:.0f}%"
    fp = E1.fp
    md = [f"# E06 results — trade by trade: when did the ratio switch work? (EXPLORATORY, a search)\n",
          f"_Generated by `explore/E06_trade_search/search.py` on {audit['run_date']} from the approved data of 2026-11-12. T03's rule (extreme ±1) on every z window, all ten pairs: {len(TR)} trades ({len(CL)} closed). Extra = held leg − peer, fill to fill, before costs; worked = extra > 0. Search on index-pair trades entered before {SPLIT.date()}, checked on those from {SPLIT.date()} and on the gold pairs. Description, not a finding; percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Checks:** the 52-week run is T03 on every pair; the conditions shared with E03 equal E03's for every trade (max difference {checks['e03_max_abs_diff']:.1e}); every trade's piece equals E04's (max {checks['e04_piece_max_abs_diff']:.1e}); the condition frame passes the cut-history check ({checks['cut_checks']} cuts, 0 violations).\n",
          "## 1. Every trade, per window\n", "| z window | pool | closed trades (first → last) | share worked | mean / median extra | best / worst | worked, held ≤ 13 weeks | worked, held > 52 weeks |", "|---|---|---|---|---|---|---|---|"]
    for _, r in OV.iterrows():
        md.append(f"| {WLAB[r['window']]} | {r['pool']} | {r['n']} ({r['first']} → {r['last']}) | {pc(r['win'])} | {fp(r['mean'])} / {fp(r['median'])} | {fp(r['best'])} / {fp(r['worst'])} | {pc(r['win_short'])} of {r['n_short']} | {pc(r['win_long'])} of {r['n_long']} |")
    md += ["\n## 2. Condition by condition (index pairs): rank correlation with the extra, within each pair (percentile among 2,000 shuffles)\n",
           "| condition | " + " | ".join(WLAB[w] for w, *_ in ZW.WINDOWS) + " | all windows | before 2016 / from 2016 | gold pairs |", "|---|" + "---|" * (len(ZW.WINDOWS) + 3)]
    for k in CKEYS:
        cells = []
        for w, *_ in ZW.WINDOWS:
            u = UN[(UN["pool"] == "index") & (UN["key"] == k) & (UN["window"] == w)].iloc[0]; cells.append(f"{E1.f2(u['rho'])} ({pq(u['pct'])})")
        c_ = CO[CO["key"] == k].iloc[0]
        md.append(f"| {CLAB[k]} | " + " | ".join(cells) + f" | {E1.f2(c_['rho_all'])} ({pq(c_['pct_all'])}) | {E1.f2(c_['rho_D'])} / {E1.f2(c_['rho_V'])} | {E1.f2(c_['rho_gold'])} ({pq(c_['pct_gold'])}) |")
    md += [f"\n## 3. The search\n",
           f"{len(singles)} single conditions and their pairs: {len(rules)} rules select at least {need:.0f} of the {len(D)} discovery trades. All discovery trades: mean extra {fp(base['D'][1])}, {pc(base['D'][2])} worked; validation ({len(V)} trades): {fp(base['V'][1])}, {pc(base['V'][2])}; gold ({len(G)}): {fp(base['G'][1])}, {pc(base['G'][2])}.\n",
           f"**How good the best rule looks by luck:** the best discovery rule selects trades with a mean extra of {fp(real_max)}; searching {PS:,} shuffled versions of the same trades finds a best of {fp(float(np.median(null_max)))} at the median (5th–95th: {fp(float(np.percentile(null_max, 5)))} to {fp(float(np.percentile(null_max, 95)))}); the real best sits at percentile {snoop_pct:.1f}.\n",
           f"**Does discovery carry over?** Across the {len(rules)} rules, the rank correlation between the discovery score and the validation score is {rho_dv:+.2f}. {n_val_pos} rules select validation trades that are better than all validation trades and notable (percentile ≥ {HI:g}); {n_val_neg} are notably worse.\n",
           "| rank | rule | discovery: trades, mean extra, worked (luck-adjusted p) | validation: trades, mean extra, worked (pct) | gold: trades, mean extra, worked |", "|---|---|---|---|---|"]
    for i, r in top.iterrows():
        md.append(f"| {i + 1} | {r['rule']} | {r['n_D']}, {fp(r['mean_D'])}, {pc(r['win_D'])} ({r['adj_p']:.3f}) | {r['n_V']}, {fp(r['mean_V'])}, {pc(r['win_V'])} ({pq(r['pct_V'])}) | {r['n_G']}, {fp(r['mean_G'])}, {pc(r['win_G'])} |")
    md += ["\n### 3b. The top three rules year by year (index pairs, all windows; added after the first run)\n",
           "Does the rule pick the better trades within a year, or mainly the better years? Years with at least 5 trades in both groups are compared.\n",
           "| rule | years with selected trades | years compared | selected trades better | share of selected trades in the 3 busiest years |", "|---|---|---|---|---|"]
    for _, r in YS.iterrows():
        md.append(f"| {r['rule_rank']}. {RU['rule'].iloc[r['rule_rank'] - 1]} | {r['years_with_selected']} | {r['years_compared']} | {r['selected_better']} of {r['years_compared']} | {r['share_in_3_busiest_years'] * 100:.0f}% ({r['busiest_years']}) |")
    md += ["", "| rule | year | selected: trades, mean extra | the rest: trades, mean extra |", "|---|---|---|---|"]
    md += [f"| {r['rule_rank']} | {r['year']} | {r['n_sel']}, {fp(r['mean_sel'])} | {r['n_rest']}, {fp(r['mean_rest'])} |" for _, r in YR.iterrows() if r["n_sel"] > 0]
    md += ["\n## 4. The top three discovery rules as entry filters (six index pairs; excess CAGR over 50/50 bought once)\n",
           f"'Against random skipping' (added after the first run): the rule's 2016–2026 excess against {RND} filters that allow an entry at random at the rule's own rate (seed default_rng([{SEED_RND}, rule, pair_id, window_id])) — trading less helps a rule that trails 50/50, so beating 'rule fails' alone is not enough.\n",
           "| rule | z window | pairs where 'rule holds' beats 'rule fails', 2016–2026 (whole window) | median difference 2016–2026 | pairs ahead of 50/50 2016–2026: rule holds / every entry | against random skipping: pairs above the median (notable), median percentile |", "|---|---|---|---|---|---|"]
    for _, r in FS.iterrows():
        md.append(f"| {r['rule_rank']}. {RU['rule'].iloc[r['rule_rank'] - 1]} | {WLAB[r['window']]} | {r['holds_beats_fails_val']} of {r['pairs']} ({r['holds_beats_fails_full']} of {r['pairs']}) | {r['median_diff_val'] * 100:+.1f} pp | {r['holds_beats_5050_val']} / {r['every_beats_5050_val']} of {r['pairs']} | {r['beats_random_median']} of {r['pairs']} ({r['notable_high']}), {r['median_pct']:.0f} |")
    open(os.path.join(OUT, "E06_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- the page
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return r4(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    TRJ = [[r["window"], r["pair"], r["half"], r["entry_decision"], r["entry_trade"], r["exit_trade"], r["held_leg"], None if r["open"] else r4(r["extra"]), None if r["open"] else (None if r["weeks"] != r["weeks"] else int(r["weeks"])),
            None if r["open"] else (None if r["days"] != r["days"] else int(r["days"])), r4(r["piece"]), r4(r["z_entry"], 3)] + [r4(r[k]) for k in CKEYS] for _, r in TR.iterrows()]
    UNJ = {f"{r['pool']}|{r['window']}|{r['key']}": [r["n"], r4(r["rho"]), r4(r["pct"], 1), r4(r["rho_D"]), r4(r["rho_V"]), r["n_lo"], r4(r["mean_lo"]), r4(r["win_lo"]), r["n_mid"], r4(r["mean_mid"]), r4(r["win_mid"]), r["n_hi"], r4(r["mean_hi"]), r4(r["win_hi"])] for _, r in UN.iterrows()}
    DATA = clean({"pairs": ORDER, "nice": {p: nice(p) for p in ORDER}, "pool": POOL, "windows": [[w, lab, f] for w, lab, f, _ in ZW.WINDOWS], "wlab": WLAB, "lo": LO, "hi": HI, "split": SPLIT.date().isoformat(),
                  "conds": [[k, CLAB[k], CFMT[k], CSHORT[k]] for k in CKEYS], "trades": TRJ, "uni": UNJ, "ov": OV.to_dict("records"), "cons": CO.to_dict("records"),
                  "rules_top": top.drop(columns=["spec"]).to_dict("records"), "rules_xy": [[r4(a), r4(b_)] for a, b_ in zip(RU["mean_D"], RU["mean_V"])],
                  "null_max": [r4(v) for v in null_max], "real_max": real_max, "snoop_pct": snoop_pct, "rho_dv": rho_dv, "base": base, "n_rules": len(rules), "n_single": len(singles), "need": need,
                  "n_val_pos": n_val_pos, "n_val_neg": n_val_neg, "filters": FI.to_dict("records"), "fsum": FS.to_dict("records"), "top3": [RU["rule"].iloc[i] for i in range(3)],
                  "years": YR.to_dict("records"), "ysum": YS.to_dict("records"), "rnd": RND, "rf": RF.to_dict("records"),
                  "dates": {p: [d.date().isoformat() for d in ratio_s[p].index] for p in PAIRS}, "ratio": {p: [r4(v, 5) for v in ratio_s[p].to_numpy()] for p in PAIRS},
                  "z": {p: {w: [None if v != v else int(round(v * 1000)) for v in zser[p][w].to_numpy()] for w, *_ in ZW.WINDOWS} for p in PAIRS},
                  "pal": {"blue": E1.BLUE, "orange": E1.ORANGE, "aqua": E1.AQUA, "muted": E1.MUTED, "divLight": E1.DIV_LIGHT, "divDark": E1.DIV_DARK, "cat": E1.CAT}})
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    from plotly.offline import get_plotlyjs
    best = top.iloc[0]
    summary = (f"<b>{len(CL)} closed trades</b> of T03's rule over the seven z windows and ten pairs. Index pairs: {pc(float(IX['good'].mean()))} worked, mean extra {fp(float(IX['extra'].mean()))}; "
               f"trades held 13 weeks or less worked {pc(float(IX[IX['weeks'] <= 13]['good'].mean()))} of the time, trades held over 52 weeks {pc(float(IX[IX['weeks'] > 52]['good'].mean()))}. "
               f"<b>The search</b> tried {len(rules):,} rules on the {len(D)} index-pair trades entered before {SPLIT.date()}. The best — <i>{html.escape(best['rule'])}</i> — picked {best['n_D']} trades with a mean extra of {fp(best['mean_D'])} ({pc(best['win_D'])} worked), "
               f"but the same search on shuffled outcomes finds a best of {fp(float(np.median(null_max)))} at the median, so the real best sits at percentile {snoop_pct:.1f} of what luck alone finds. "
               f"From {SPLIT.date()} on, that rule's {best['n_V']} trades had a mean extra of {fp(best['mean_V'])} ({pc(best['win_V'])} worked) against {fp(base['V'][1])} ({pc(base['V'][2])}) for all trades. "
               f"Across all rules, discovery and validation scores correlate {rho_dv:+.2f}. <b>The top three, checked:</b> "
               + "; ".join(f"<i>{html.escape(RU['rule'].iloc[i])}</i> — after {SPLIT.year}: {fp(RU['mean_V'].iloc[i])} ({pc(RU['win_V'].iloc[i])} worked); its trades sit {YS['share_in_3_busiest_years'].iloc[i] * 100:.0f}% in {YS['busiest_years'].iloc[i]}, "
                           f"and beat the other trades of the same year in {YS['selected_better'].iloc[i]} of {YS['years_compared'].iloc[i]} years" for i in range(3)) + ".")
    sel_cond = "<label>Condition <select id='cond'>" + "".join(f"<option value='{k}'{' selected' if k == 'trend104' else ''}>{html.escape(CLAB[k])}</option>" for k in CKEYS) + "</select></label>"
    sel_rule = "<label>Rule <select id='rule'>" + "".join(f"<option value='{i + 1}'>{i + 1}. {html.escape(RU['rule'].iloc[i])}</option>" for i in range(3)) + "</select></label>"
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Trade search</title>"),
         "<header class='bar'><div class='bar-inner'><b>E06 · trade by trade</b><span class='tag'>EXPLORATORY · SEARCH</span>" + sel_cond + "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Trade by trade: when did the ratio switch work?</h1>",
         f"<p>Every trade of T03's rule (z crosses ±1 from 50/50 → hold the lagging index; back to 50/50 when z returns to 0; next-session fills, 10.4 bps each way) on all seven z windows and all ten pairs, each with {len(CKEYS)} conditions known at its entry. "
         f"<b>Worked</b> = the held index beat the other between the fills (extra &gt; 0, before costs). Searching conditions on known outcomes always finds something, so every search result here is checked on trades the search did not see: index-pair trades from {SPLIT.date()} on, and the gold pairs. "
         f"Generated {audit['run_date']} from the approved data of 2026-11-12. <b>Description, not a finding</b>.</p>",
         f"<div class='card'><p style='margin:4px 0'>{summary}</p><p class='note' style='margin:4px 0'><b>Checks:</b> the 52-week run is T03; conditions equal E03's for every trade; each trade's piece equals E04's; the condition frame uses only data up to each entry ({checks['cut_checks']} cut checks, 0 violations).</p></div>",
         "<h2>1 · Every trade, per window</h2><div class='card'><div id='g_box' class='plot'></div><div id='t_ov'></div></div>",
         "<h2>2 · Condition by condition</h2><p class='note'>For each condition and z window (index pairs): the rank correlation of the condition with the trade's extra, both ranked within each pair, against 2,000 shuffles of the extras within each pair. Red = higher values went with better trades, blue = with worse; labels on notable cells. The last columns: all windows pooled, the two halves of the history, and the gold pairs.</p>",
         "<div class='card'><div id='g_heat' class='plot'></div></div>",
         "<div class='card'><div id='g_cond' class='plot'></div><p class='note'>Pick the condition at the top. Thirds are within each pair, so 'high' means high for that pair.</p></div>",
         f"<h2>3 · The search, and whether it survives</h2><p class='note'>Every single condition (cut at its discovery thirds and median) and every pair of them on different conditions: {len(rules):,} rules that pick at least {need:.0f} of the {len(D)} index-pair trades entered before {SPLIT.date()}. A rule's score is the mean extra of the trades it picks. Left: the best score the same search finds when the outcomes are shuffled (luck), with the real best. Right: every rule's score before {SPLIT.date()} against its score after; if the search found something real, the cloud slopes up.</p>",
         "<div class='card'><div id='g_snoop' class='plot'></div></div><div class='card'><div id='g_dv' class='plot'></div></div>",
         "<div class='card'><p class='note' style='margin:4px 0'>The twenty best rules before " + SPLIT.date().isoformat() + " and what they did after (luck-adjusted p = the share of shuffled searches whose best beat this rule's score; validation percentile = against 1,000 shuffles of the validation outcomes within each pair and window).</p><div id='t_rules'></div></div>",
         "<div class='card'><p class='note' style='margin:4px 0'><b>Year by year</b> (added after the first run): a market-wide condition can pick good <i>years</i> rather than good <i>trades</i>, and the six pairs share their years. For each of the top three rules: the selected trades against the rest within each calendar year (years with at least 5 trades in both groups).</p><div id='t_ysum'></div><details><summary>Every year</summary><div id='t_years'></div></details></div>",
         "<h2>4 · The top three rules as entry filters</h2><p class='note'>The three best rules before " + SPLIT.date().isoformat() + " (chosen before their later numbers were looked at), used as an entry filter on each index pair and window through the engine's book: enter only on a first cross where the rule holds; for contrast, only where it fails; and every entry. Excess = CAGR − 50/50 bought once's CAGR, for 2016–2026 and for the whole window.</p>",
         "<div class='card'><div id='t_fsum'></div></div>", "<div class='card'>" + sel_rule + "<div id='t_fdet'></div></div>",
         "<h2>5 · Every trade, one by one</h2><div class='card'><div class='filters'><label>z window <select id='f_w'><option value=''>all</option>" + "".join(f"<option value='{w}'>{html.escape(WLAB[w])}</option>" for w, *_ in ZW.WINDOWS) + "</select></label>"
         "<label>Pair <select id='f_p'><option value=''>all</option>" + "".join(f"<option value='{html.escape(p, quote=True)}'>{nice(p)}</option>" for p in ORDER) + "</select></label>"
         "<label>Outcome <select id='f_o'><option value=''>all</option><option value='w'>worked</option><option value='l'>did not work</option><option value='o'>still open</option></select></label>"
         f"<label>Period <select id='f_h'><option value=''>all</option><option value='D'>entered before {SPLIT.date()}</option><option value='V'>entered from {SPLIT.date()}</option></select></label><span id='f_n' class='note'></span></div>"
         "<p class='note'>Click a column title to sort; click a row to draw that trade.</p><div id='t_trades'></div><div id='pager'></div></div>",
         "<div class='card'><div id='g_trade' class='plot'></div><div id='t_trade'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Trades.</b> explore/E01_ratio_cycles/variants.py (T03's state machine on each window's z) through analysis/ratio_engine.book. Weeks held = decision weeks between the entry decision and the exit trade. Piece = change of ln(strategy ÷ 50/50) over the hold, costs included (E04's).</li>"
         "<li><b>Conditions.</b> All at the entry decision's close from data up to that day; oriented to the leg the rule holds. Defined in explore/E06_trade_search/E06.md.</li>"
         f"<li><b>Shuffles.</b> Condition by condition: {P:,} shuffles of the extras within each pair (pooled windows: within each pair and window), seed default_rng([{SEED_UNI}, window_id, condition_id]). Search: {PS:,} shuffles within each (pair, window) for the luck yardstick (seed [{SEED_SNOOP}, 0]) and for validation (seed [{SEED_VAL}, 0]).</li>"
         "<li><b>Limits.</b> The seven windows' trades overlap, so pooled counts overstate independent trades and the shuffles are too narrow; the validation half and the gold pairs are the stronger checks. The design and the rule were settled after the whole history had been seen; only data after 2026-09-08 is a true hold-out. Anything here that looks usable is a candidate for a pre-registered test (T04), not a rule.</li></ul></div>",
         "</main>", f"<script id='e06-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E06_trades.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; best discovery rule at percentile {snoop_pct:.1f} of noise searches; discovery vs validation {rho_dv:+.2f}; page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.filters{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:6px 0}
#t_trades th{cursor:pointer;white-space:nowrap}#t_trades tr.sel td{background:rgba(42,120,214,.12)}#t_trades tbody tr{cursor:pointer}
.pos{color:#1b8f62}.neg{color:#c4462a}
#pager button{margin:4px 4px 0 0}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e06-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
const NCOL = 12, CK = D.conds.map(c=>c[0]), CI = {}; CK.forEach((k,i)=>CI[k]=NCOL+i);
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function f2(x){return x===null||x===undefined?'—':x.toFixed(2);}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function cv(k,x){if(x===null||x===undefined)return '—';const f=D.conds.find(c=>c[0]===k)[2];return f==='pct'?fp(x):(f==='wk'?x.toFixed(0):(f==='bin'?(x>=0.5?'yes':'no'):(x>=0?'+':'')+x.toFixed(2)));}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:112,b:56},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',font:{color:c.ink2,size:11}}},extra||{});}
function ax(e){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},e||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function wl(w){return w==='all'?'all windows':D.wlab[w];}
function heatColors(){return isDark()?D.pal.divDark:D.pal.divLight;}

function drawOverview(){
  const c=th(), tr=[];
  D.windows.forEach(([w])=>{const ys=D.trades.filter(t=>t[0]===w&&D.pool[t[1]]==='index'&&t[7]!==null).map(t=>t[7]*100);
    tr.push({type:'box',y:ys,name:D.wlab[w],boxpoints:'outliers',marker:{color:D.pal.blue,size:3},line:{color:D.pal.blue,width:1.2},fillcolor:'rgba(42,120,214,0.15)',hovertemplate:'%{y:+.1f}%<extra>'+esc(D.wlab[w])+'</extra>'});});
  const all=D.trades.filter(t=>D.pool[t[1]]==='index'&&t[7]!==null);
  Plotly.react('g_box',tr,layout(two('Every closed trade\'s extra, per z window · six index pairs · '+all.length+' trades',
    'extra = the held index\'s return minus the other\'s, fill to fill, before costs; box = middle half, line = median, whiskers to 1.5 × the box, dots beyond'),460,
    {showlegend:false,yaxis:ax({ticksuffix:'%',title:{text:'extra',font:{color:c.ink2}}}),xaxis:ax({}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}]}),CFG);
  document.getElementById('t_ov').innerHTML=table(['z window','pool','closed trades (first → last)','share worked','mean / median extra','best / worst','worked, held ≤ 13 weeks','worked, held > 52 weeks'],
    D.ov.map(r=>[D.wlab[r.window],r.pool,r.n+' ('+r.first+' → '+r.last+')',pc(r.win),fp(r.mean)+' / '+fp(r.median),fp(r.best)+' / '+fp(r.worst),pc(r.win_short)+' of '+r.n_short,(r.n_long?pc(r.win_long):'—')+' of '+r.n_long]));
}
function drawHeat(){
  const c=th(), cols=D.windows.map(x=>x[0]).concat(['all','D','V','gold']);
  const colLab=D.windows.map(x=>x[1]+(x[2]==='D'?' d':'')).concat(['all windows','before '+D.split.slice(0,4),'from '+D.split.slice(0,4),'gold pairs']);
  const z=[],tx=[],hv=[];
  CK.forEach(k=>{const zz=[],tt=[],hh=[];cols.forEach(col=>{
    let u,pct=null,rho=null,txt='';
    if(col==='D'||col==='V'){u=D.uni['index|all|'+k];rho=col==='D'?u[3]:u[4];zz.push(rho===null?null:50+rho*250);tt.push(rho===null?'':f2(rho));hh.push(D.conds.find(c=>c[0]===k)[1]+' · index pairs, all windows, entries '+(col==='D'?'before ':'from ')+D.split+'<br>rank correlation '+f2(rho));return;}
    u=col==='gold'?D.uni['gold|all|'+k]:D.uni['index|'+col+'|'+k];rho=u[1];pct=u[2];
    zz.push(pct);tt.push(pct!==null&&(pct<=D.lo||pct>=D.hi)?f2(rho):'');
    hh.push(D.conds.find(c=>c[0]===k)[1]+' · '+(col==='gold'?'gold pairs, all windows':wl(col)+', index pairs')+'<br>rank correlation '+f2(rho)+', percentile '+(pct===null?'—':pct.toFixed(1))+' ('+u[0]+' trades)<br>mean extra low / mid / high third: '+fp(u[6])+' / '+fp(u[9])+' / '+fp(u[12])+'<br>worked: '+pc(u[7])+' / '+pc(u[10])+' / '+pc(u[13]));});
    z.push(zz);tx.push(tt);hv.push(hh);});
  Plotly.react('g_heat',[{type:'heatmap',z:z,x:colLab.map((l,i)=>i),y:CK.map((k,i)=>i),zmin:0,zmax:100,colorscale:heatColors(),text:tx,texttemplate:'%{text}',customdata:hv,hovertemplate:'%{customdata}<extra></extra>',xgap:2,ygap:2,
    colorbar:{thickness:12,tickvals:[0,5,50,95,100],tickfont:{color:c.muted},title:{text:'percentile',side:'right',font:{color:c.ink2}}}}],
    layout(two('Which conditions at entry went with better trades? · index pairs, per z window','percentile of the within-pair rank correlation among 2,000 shuffles (red = higher value, better trade); the two half columns show the correlation itself (colour: 0.2 = full red)'),860,
    {margin:{l:300,r:24,t:112,b:30},xaxis:ax({tickvals:colLab.map((l,i)=>i),ticktext:colLab,side:'top',showgrid:false,showline:false,tickangle:-30}),yaxis:ax({tickvals:CK.map((k,i)=>i),ticktext:D.conds.map(c=>c[1]),autorange:'reversed',showgrid:false,showline:false,ticksuffix:'  '}),
     shapes:[{type:'line',xref:'x',yref:'paper',x0:6.5,x1:6.5,y0:0,y1:1,line:{color:c.axis,width:1}},{type:'line',xref:'x',yref:'paper',x0:7.5,x1:7.5,y0:0,y1:1,line:{color:c.axis,width:1}},{type:'line',xref:'x',yref:'paper',x0:9.5,x1:9.5,y0:0,y1:1,line:{color:c.axis,width:1}}]}),CFG);
}
function drawCond(){
  const c=th(), k=document.getElementById('cond').value, lab=D.conds.find(x=>x[0]===k)[1], bin=D.conds.find(x=>x[0]===k)[2]==='bin';
  const groups=D.windows.map(x=>x[0]).concat(['all']), names=groups.map(wl), tr=[];
  [['low third',5,6,7,D.pal.blue],['middle third',8,9,10,'#a8a69e'],['high third',11,12,13,D.pal.orange]].forEach(([nm,ni,mi,wi,col])=>{
    if(bin&&nm==='middle third')return;
    const us=groups.map(g=>D.uni['index|'+g+'|'+k]);
    tr.push({type:'bar',x:names,y:us.map(u=>u[mi]===null?null:u[mi]*100),name:bin?(nm==='low third'?'no':'yes'):nm,marker:{color:col},text:us.map(u=>u[wi]===null?'':pc(u[wi])),textposition:'outside',cliponaxis:false,
      customdata:us.map(u=>u[ni]+' trades, '+pc(u[wi])+' worked'),hovertemplate:'%{x}: mean extra %{y:+.1f}%<br>%{customdata}<extra>'+nm+'</extra>'});});
  const ua=D.uni['index|all|'+k];
  Plotly.react('g_cond',tr,layout(two(esc(lab)+' · mean extra of the trades in its '+(bin?'two groups':'low, middle and high third (within each pair)')+', per z window · index pairs',
      'labels = share worked; all windows: rank correlation '+f2(ua[1])+' (percentile '+(ua[2]===null?'—':ua[2].toFixed(1))+'); before '+D.split+' '+f2(ua[3])+', from '+D.split+' '+f2(ua[4])),460,
    {barmode:'group',xaxis:ax({}),yaxis:ax({ticksuffix:'%',title:{text:'mean extra',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}]}),CFG);
}
function drawSearch(){
  const c=th();
  Plotly.react('g_snoop',[{type:'histogram',x:D.null_max.map(v=>v*100),marker:{color:'#a8a69e'},name:'best rule in a shuffled search',nbinsx:40,hovertemplate:'%{x:.1f}%: %{y} shuffled searches<extra></extra>'}],
    layout(two('How good the best rule looks by luck · '+D.null_max.length.toLocaleString()+' searches on shuffled outcomes, index pairs before '+D.split,
      'grey = the best mean extra each shuffled search found; line = the real search\'s best ('+fp(D.real_max)+'), at percentile '+D.snoop_pct.toFixed(1)),400,
      {xaxis:ax({ticksuffix:'%',title:{text:'best mean extra found',font:{color:c.ink2}}}),yaxis:ax({title:{text:'searches',font:{color:c.ink2}}}),showlegend:false,
       shapes:[{type:'line',xref:'x',x0:D.real_max*100,x1:D.real_max*100,yref:'paper',y0:0,y1:1,line:{color:D.pal.orange,width:2}}]}),CFG);
  const xy=D.rules_xy, top=D.rules_top;
  Plotly.react('g_dv',[{type:'scattergl',mode:'markers',x:xy.map(p=>p[0]*100),y:xy.map(p=>p[1]===null?null:p[1]*100),marker:{color:'rgba(137,135,129,0.35)',size:4},name:'every rule',hovertemplate:'before: %{x:+.1f}%<br>after: %{y:+.1f}%<extra></extra>'},
    {type:'scatter',mode:'markers',x:top.map(r=>r.mean_D*100),y:top.map(r=>r.mean_V*100),marker:{color:D.pal.blue,size:8},name:'the 20 best before '+D.split,customdata:top.map(r=>esc(r.rule)),hovertemplate:'%{customdata}<br>before %{x:+.1f}%, after %{y:+.1f}%<extra></extra>'}],
    layout(two('Every rule: mean extra of the trades it picks, before '+D.split+' (x) and after (y) · '+D.n_rules.toLocaleString()+' rules, rank correlation '+(D.rho_dv>=0?'+':'')+D.rho_dv.toFixed(2),
      'dashed = all trades in each period ('+fp(D.base.D[1])+' before, '+fp(D.base.V[1])+' after); a real pattern would make the cloud slope up'),520,
      {xaxis:ax({ticksuffix:'%',title:{text:'before '+D.split,font:{color:c.ink2}}}),yaxis:ax({ticksuffix:'%',title:{text:'from '+D.split,font:{color:c.ink2}}}),
       shapes:[{type:'line',xref:'x',x0:D.base.D[1]*100,x1:D.base.D[1]*100,yref:'paper',y0:0,y1:1,line:{color:c.axis,width:1,dash:'dash'}},{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:D.base.V[1]*100,y1:D.base.V[1]*100,line:{color:c.axis,width:1,dash:'dash'}}]}),CFG);
  document.getElementById('t_rules').innerHTML=table(['rank','rule','before: trades, mean extra, worked','luck-adjusted p','after: trades, mean extra, worked','percentile','gold: trades, mean extra, worked'],
    [['—','all trades',D.base.D[0]+', '+fp(D.base.D[1])+', '+pc(D.base.D[2]),'—',D.base.V[0]+', '+fp(D.base.V[1])+', '+pc(D.base.V[2]),'—',D.base.G[0]+', '+fp(D.base.G[1])+', '+pc(D.base.G[2])]].concat(
    top.map((r,i)=>[i+1,esc(r.rule),r.n_D+', '+fp(r.mean_D)+', '+pc(r.win_D),r.adj_p.toFixed(3),r.n_V+', '+fp(r.mean_V)+', '+pc(r.win_V),pq(r.pct_V),r.n_G+', '+fp(r.mean_G)+', '+pc(r.win_G)])));
}
function drawFilters(){
  document.getElementById('t_fsum').innerHTML=table(['rule','z window',"index pairs where 'rule holds' beats 'rule fails', 2016–2026",'whole window','median difference 2016–2026','pairs ahead of 50/50 2016–2026: rule holds / every entry','against random skipping: pairs above the median (notable), median percentile'],
    D.fsum.map(r=>[r.rule_rank+'. '+esc(D.top3[r.rule_rank-1]),D.wlab[r.window],r.holds_beats_fails_val+' of '+r.pairs,r.holds_beats_fails_full+' of '+r.pairs,(r.median_diff_val>=0?'+':'')+(r.median_diff_val*100).toFixed(1)+' pp',r.holds_beats_5050_val+' / '+r.every_beats_5050_val+' of '+r.pairs,r.beats_random_median+' of '+r.pairs+' ('+r.notable_high+'), '+r.median_pct.toFixed(0)]));
  const ri=Number(document.getElementById('rule').value), rows=D.filters.filter(r=>r.rule_rank===ri);
  const rf={}; D.rf.filter(r=>r.rule_rank===ri).forEach(r=>{rf[r.pair+'|'+r.window]=r;});
  document.getElementById('t_fdet').innerHTML=table(['pair','z window','version','closed trades','worked','excess 2016–2026','percentile among '+D.rnd+' random filters at the same rate','excess, whole window (first → last)'],
    rows.map(r=>{const q=rf[r.pair+'|'+r.window];return [D.nice[r.pair],D.wlab[r.window],r.version,r.trades,pc(r.win),(r.excess_val>=0?'+':'')+(r.excess_val*100).toFixed(1)+' pp',r.version==='rule holds'&&q?pq(q.pct)+' (random median '+(q.random_median>=0?'+':'')+(q.random_median*100).toFixed(1)+' pp, rate '+pc(q.rate)+')':'',(r.excess_full>=0?'+':'')+(r.excess_full*100).toFixed(1)+' pp ('+r.first+' → '+r.last+')'];}));
  document.getElementById('t_ysum').innerHTML=table(['rule','years with selected trades','years compared','selected trades better','share of selected trades in the 3 busiest years'],
    D.ysum.map(r=>[r.rule_rank+'. '+esc(D.top3[r.rule_rank-1]),r.years_with_selected,r.years_compared,r.selected_better+' of '+r.years_compared,pc(r.share_in_3_busiest_years)+' ('+r.busiest_years+')']));
  document.getElementById('t_years').innerHTML=table(['rule','year','selected: trades, mean extra','the rest: trades, mean extra'],
    D.years.filter(r=>r.n_sel>0).map(r=>[r.rule_rank,r.year,r.n_sel+', '+fp(r.mean_sel),r.n_rest+', '+fp(r.mean_rest)]));
}
// ---- the trade explorer
let SORT={col:7,dir:-1}, PAGE=0, SEL=null; const PER=50;
const TCOLS=[['z window',0],['pair',1],['entered',4],['exited',5],['held',6],['extra',7],['weeks',8],['piece',10],['z at entry',11]].concat(['lag13','trend104','long_cheap','corr52','vol_ratio','mkt52','breadth'].map(k=>[D.conds.find(c=>c[0]===k)[3],CI[k],k]));
function filtered(){const w=document.getElementById('f_w').value,p=document.getElementById('f_p').value,o=document.getElementById('f_o').value,h=document.getElementById('f_h').value;
  return D.trades.map((t,i)=>i).filter(i=>{const t=D.trades[i];return (!w||t[0]===w)&&(!p||t[1]===p)&&(!h||t[2]===h)&&(!o||(o==='o'?t[7]===null:(t[7]!==null&&(o==='w'?t[7]>0:t[7]<=0))));});}
function drawTable(){
  const idx=filtered(), col=SORT.col, dir=SORT.dir;
  idx.sort((a,b)=>{const x=D.trades[a][col],y=D.trades[b][col];if(x===null&&y===null)return 0;if(x===null)return 1;if(y===null)return -1;return (x<y?-1:x>y?1:0)*dir;});
  const pages=Math.max(1,Math.ceil(idx.length/PER)); PAGE=Math.min(PAGE,pages-1); const show=idx.slice(PAGE*PER,(PAGE+1)*PER);
  const head=TCOLS.map((c,j)=>'<th data-j="'+j+'">'+c[0]+(TCOLS[j][1]===col?(dir>0?' ▲':' ▼'):'')+'</th>').join('');
  const body=show.map(i=>{const t=D.trades[i];const cells=TCOLS.map(c=>{const v=t[c[1]];if(c[2])return cv(c[2],v);if(c[1]===7||c[1]===10)return v===null?(c[1]===7?'open':'—'):'<span class="'+(v>0?'pos':'neg')+'">'+fp(v)+'</span>';if(c[1]===0)return D.wlab[v];if(c[1]===1)return D.nice[v];if(c[1]===11)return v===null?'—':(v>=0?'+':'')+v.toFixed(2);return v===null?'—':v;});
    return '<tr data-i="'+i+'"'+(i===SEL?' class="sel"':'')+'>'+cells.map(x=>'<td>'+x+'</td>').join('')+'</tr>';}).join('');
  document.getElementById('t_trades').innerHTML="<div class='scroll'><table><thead><tr>"+head+"</tr></thead><tbody>"+body+"</tbody></table></div>";
  const nw=idx.filter(i=>D.trades[i][7]!==null&&D.trades[i][7]>0).length, nc=idx.filter(i=>D.trades[i][7]!==null).length;
  document.getElementById('f_n').textContent=idx.length+' trades, '+nc+' closed, '+(nc?Math.round(nw/nc*100):0)+'% worked';
  document.getElementById('pager').innerHTML=(PAGE>0?'<button data-pg="-1">← previous</button>':'')+'<span class="note">page '+(PAGE+1)+' of '+pages+'</span>'+(PAGE<pages-1?' <button data-pg="1">next →</button>':'');
  document.querySelectorAll('#t_trades th').forEach(el=>el.addEventListener('click',()=>{const j=Number(el.dataset.j),c=TCOLS[j][1];SORT=SORT.col===c?{col:c,dir:-SORT.dir}:{col:c,dir:-1};PAGE=0;drawTable();}));
  document.querySelectorAll('#t_trades tbody tr').forEach(el=>el.addEventListener('click',()=>{SEL=Number(el.dataset.i);drawTable();drawTrade();}));
  document.querySelectorAll('#pager button').forEach(el=>el.addEventListener('click',()=>{PAGE+=Number(el.dataset.pg);drawTable();}));
  if(SEL===null&&show.length){SEL=show[0];drawTable();drawTrade();}
}
function drawTrade(){
  if(SEL===null)return; const c=th(), t=D.trades[SEL], p=t[1], w=t[0], dates=D.dates[p], ratio=D.ratio[p], z=D.z[p][w];
  const ie=dates.indexOf(t[3]); let ix=t[5]?dates.findIndex(d=>d>=t[5]):dates.length-1; if(ix<0)ix=dates.length-1;
  const a=Math.max(0,ie-52), b=Math.min(dates.length-1,ix+26), xs=dates.slice(a,b+1);
  const legs=p.split('|'), col=t[7]===null?c.muted:(t[7]>0?D.pal.aqua:D.pal.orange);
  const tr=[{x:xs,y:ratio.slice(a,b+1),type:'scatter',mode:'lines',line:{color:D.pal.blue,width:1.6},name:'ratio '+D.nice[p],yaxis:'y',hovertemplate:'%{x}<br>ratio %{y:.4f}<extra></extra>'},
    {x:xs,y:z.slice(a,b+1).map(v=>v===null?null:v/1000),type:'scatter',mode:'lines',line:{color:c.ink2,width:1.3},name:'z ('+D.wlab[w]+')',yaxis:'y2',hovertemplate:'%{x}<br>z %{y:+.2f}<extra></extra>'}];
  const shapes=[{type:'rect',xref:'x',yref:'paper',x0:t[4],x1:t[5]||dates[dates.length-1],y0:0,y1:1,fillcolor:col,opacity:0.14,line:{width:0}},
    {type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:-1,y1:-1,line:{color:c.axis,width:1,dash:'dash'}},{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:1,y1:1,line:{color:c.axis,width:1,dash:'dash'}},{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:0,y1:0,line:{color:c.axis,width:1}}];
  Plotly.react('g_trade',tr,layout(two(D.nice[p]+' · z window '+D.wlab[w]+' · held '+t[6]+' '+t[4]+' → '+(t[5]||'still open')+(t[7]===null?'':' · extra '+fp(t[7])+' over '+t[8]+' weeks ('+t[9]+' days) · piece '+fp(t[10])),
      'top: the ratio '+legs[0]+' ÷ '+legs[1]+' (shaded = the hold; aqua = worked, orange = did not); bottom: z with ±1 dashed; from a year before the entry to half a year after the exit'),520,
    {xaxis:ax({type:'date',anchor:'y2'}),yaxis:ax({domain:[0.48,1],anchor:'x',title:{text:'ratio',font:{color:c.ink2}}}),yaxis2:ax({domain:[0,0.4],anchor:'x',title:{text:'z',font:{color:c.ink2}}}),shapes:shapes}),CFG);
  document.getElementById('t_trade').innerHTML=table(['condition at entry','value'],D.conds.map(cd=>[esc(cd[1]),cv(cd[0],t[CI[cd[0]]])]));
}
function renderAll(){drawOverview();drawHeat();drawCond();drawSearch();drawFilters();drawTable();if(SEL!==null)drawTrade();document.getElementById('theme').textContent=isDark()?'Light':'Dark';}
window.addEventListener('load',()=>{
  try{const s=localStorage.getItem('e06-cond');if(s&&CK.includes(s))document.getElementById('cond').value=s;}catch(e){}
  document.getElementById('cond').addEventListener('change',()=>{try{localStorage.setItem('e06-cond',document.getElementById('cond').value);}catch(e){}drawCond();});
  document.getElementById('rule').addEventListener('change',drawFilters);
  ['f_w','f_p','f_o','f_h'].forEach(id=>document.getElementById(id).addEventListener('change',()=>{PAGE=0;SEL=null;drawTable();}));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';renderAll();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',renderAll);
  renderAll();
});
"""

if __name__ == "__main__":
    main()
