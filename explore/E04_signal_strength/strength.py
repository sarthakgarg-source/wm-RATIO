"""
E04 (EXPLORATORY): does the ratio band have any predictive power? Design: explore/E04_signal_strength/E04.md (written before
the first run). 1. T03's outperformance against 50/50 over time. 2. What follows a band touch (event study). 3. The z-score's
information coefficient against forward relative returns. 4. The T03 rule run on E01's random walks: are its trade
statistics signal or mechanics?

    python explore/E04_signal_strength/strength.py [--draws 2000]

Not a finding and not a test of any rule.
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
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles"))
from analysis.approved_data import GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import INDEX_PAIRS, PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (E01's random paths, z-score, percentiles and the page's styling)
import zwin as ZW  # noqa: E402  (the z windows: weekly 13 / 26 / 52, daily 20 / 60 / 120 / 250, and the daily random paths)
import variants as V  # noqa: E402  (the rule run with each window's band through the engine's book)

TEST, RUN, WIN, EXIT, COST = "T03", "headline", 52, "E1", 20.8
EV = os.path.join(ROOT, "tests", TEST, "evidence", RUN)
K_MAX = 52; IC_H = (1, 4, 13, 26, 52); EVENT_SHOW = (4, 13, 26, 52)
K_D = (1,) + tuple(range(5, 251, 5)); IC_D = (1, 5, 20, 65, 130, 250)      # daily windows: horizons in trading days (5 = a week)
DAILY = tuple(w for w, _, f, _ in ZW.WINDOWS if f == "D")
ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR; LO, HI = E1.LO, E1.HI
BLUE, ORANGE, AQUA, MUTED, INK2, AXIS, NULL_FILL = E1.BLUE, E1.ORANGE, E1.AQUA, E1.MUTED, E1.INK2, E1.AXIS, E1.NULL_FILL
STATS = (("n", "trades", "num"), ("hit", "hit rate", "pct"), ("mean", "mean relative return", "pct"), ("median", "median relative return", "pct"),
         ("weeks", "mean weeks held", "num"), ("long", "share held over 52 weeks", "pct"))


def fmt(kind, v, d=1):
    if v != v:
        return "—"
    return f"{v * 100:+.{d}f}%" if kind == "pct" and kind != "share" else f"{v:.{d}f}"


# ------------------------------------------------------------------------------------------------------- the paths
def paths_for(pair, pid, px, D):
    """Row 0 = the real log ratio on decision weeks; rows 1..D = E01's random paths (real warm-up, then the point-in-time
    residuals with E01's random signs plus the expanding drift)."""
    pf = pair_frame(px, pair, "plain"); wk = pf.loc[pf["decision"]]; dates = wk.index
    r = (wk["ca"] / wk["cb"]).to_numpy(); lnr = np.log(r); g = np.diff(lnr)
    _, m, _, z, _ = E1.point_in_time(g); n = len(z)
    S = np.random.default_rng([E1.SEED_SIGNS, pid]).choice(np.array([-1.0, 1.0]), size=(D, n))
    rnd = np.concatenate([np.tile(lnr[: E1.BURN + 1], (D, 1)), lnr[E1.BURN] + np.cumsum(S * z + m, axis=1)], axis=1)
    P = np.concatenate([lnr[None, :], rnd], axis=0)
    Z = E1.trailing_z(np.exp(P)); chk = float(np.nanmax(np.abs(Z[0] - E1.band_z(pd.Series(r, index=dates)).to_numpy())))
    return dates, P, Z, chk


def positions(Z):
    pos = np.where(Z < -1, -1.0, np.where(Z > 1, 1.0, 0.0)); pos[np.isnan(Z)] = np.nan
    return pos


def first_touches(pos):
    """+1 = first week below the band (hold a), -1 = first week above (hold b); 0 otherwise. As the rule: the week before had
    a band and was not beyond the same side."""
    prev = np.concatenate([np.full((pos.shape[0], 1), np.nan), pos[:, :-1]], axis=1)
    below = (pos == -1) & ((prev == 0) | (prev == 1)); above = (pos == 1) & ((prev == 0) | (prev == -1))
    return below.astype(float) - above.astype(float)


def run_rule(Z):
    """The T03 rule on every row at once: entry on a first touch from neutral, exit when z is back through 0 (the ratio at
    its trailing mean), no maximum hold, touches ignored in a trade. Returns closed trades (row, t_entry, t_exit, leg)
    and the open ones (row, t_entry, leg); leg +1 = long a, -1 = long b."""
    Pn, L = Z.shape; state = np.zeros(Pn, dtype=np.int8); prev = np.full(Pn, np.nan); t_in = np.full(Pn, -1); closed = []
    for t in range(L):
        zt = Z[:, t]
        if np.isnan(zt).all():
            prev[:] = np.nan; continue
        pos = np.where(zt < -1, -1.0, np.where(zt > 1, 1.0, 0.0))
        ex = ((state == 1) & (zt >= 0)) | ((state == -1) & (zt <= 0))
        for p in np.flatnonzero(ex):
            closed.append((p, t_in[p], t, state[p]))
        state[ex] = 0; t_in[ex] = -1
        neu = state == 0
        en_a = neu & (pos == -1) & ((prev == 0) | (prev == 1)); en_b = neu & (pos == 1) & ((prev == 0) | (prev == -1))
        state[en_a] = 1; state[en_b] = -1; t_in[en_a | en_b] = t
        prev = pos
    opened = [(p, t_in[p], state[p]) for p in np.flatnonzero(state != 0)]
    return np.array(closed, dtype=int).reshape(-1, 4), opened


def trade_stats(ret, weeks):
    if len(ret) == 0:
        return {k: np.nan for k, _, _ in STATS} | {"n": 0}
    return {"n": len(ret), "hit": float(np.mean(ret > 0)), "mean": float(ret.mean()), "median": float(np.median(ret)), "weeks": float(weeks.mean()), "long": float(np.mean(weeks > 52))}


def spearman_rows(x, y):
    """Spearman correlation per row (rows = paths), ignoring nothing (the windows are complete)."""
    rx = rankdata(x, axis=1); ry = rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True); ry -= ry.mean(axis=1, keepdims=True)
    return (rx * ry).sum(axis=1) / np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))


def ev_ic(P, Z, burn, kgrid, hgrid):
    """Sections 2 and 3 on a block of paths: per row the summed relative return after each first touch at every k, the
    number of touches, and the Spearman IC of -z with the forward relative return at every h (from the warm-up's end)."""
    L = P.shape[1]; kmax = max(kgrid)
    sig = first_touches(positions(Z)); okc = np.zeros(L, bool); okc[burn: L - kmax] = True
    S_ = np.where(okc[None, :], sig, 0.0); cnt = np.abs(S_).sum(axis=1)
    ev = np.stack([(S_[:, :L - k] * (P[:, k:] - P[:, :-k])).sum(axis=1) for k in kgrid], axis=1)
    ic = np.stack([spearman_rows(-Z[:, burn:L - h], P[:, burn + h:] - P[:, burn:L - h]) for h in hgrid], axis=1)
    return ev, cnt, ic


def ic_halves(p, z, burn, hgrid):
    out = []
    for h in hgrid:
        x = -z[None, burn:len(p) - h]; y = (p[burn + h:] - p[burn:len(p) - h])[None, :]; half = x.shape[1] // 2
        out.append((float(spearman_rows(x[:, :half], y[:, :half])[0]), float(spearman_rows(x[:, half:], y[:, half:])[0])))
    return out


def rule_block(Zs, Ps, burn):
    """Section 4 on a block of paths at decision weeks: the rule's closed trades entered after the warm-up, their relative
    return close to close, and the trade statistics per row."""
    closed, opened = run_rule(Zs); keep = closed[closed[:, 1] >= burn]
    ret = np.where(keep[:, 3] == 1, 1.0, -1.0) * (Ps[keep[:, 0], keep[:, 2]] - Ps[keep[:, 0], keep[:, 1]]); wks = keep[:, 2] - keep[:, 1]
    order = np.argsort(keep[:, 0], kind="stable"); cut_ = np.searchsorted(keep[order, 0], np.arange(len(Zs) + 1))
    stats = [trade_stats(ret[order[cut_[r]:cut_[r + 1]]], wks[order[cut_[r]:cut_[r + 1]]]) for r in range(len(Zs))]
    return stats, keep, ret, wks, closed, opened


def outperf(nav: pd.Series, bh: pd.Series):
    nv = pd.DataFrame({"nav": nav, "5050_buyhold": bh}).dropna()
    rel = nv["nav"] / nv["5050_buyhold"]; e = np.log(nv["nav"]).diff().dropna() - np.log(nv["5050_buyhold"]).diff().dropna()
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25; cg = lambda s_: (s_.iloc[-1] / s_.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
    ye = nv.groupby(nv.index.year).last(); y0 = nv.iloc[[0]].rename(index=lambda d: d.year - 1); yy = pd.concat([y0, ye]); yr = (yy / yy.shift(1) - 1).dropna()
    yearly = [{"year": int(y_), "strategy": float(row["nav"]), "5050": float(row["5050_buyhold"]), "excess": float(row["nav"] - row["5050_buyhold"])} for y_, row in yr.iterrows()]
    o = {"first": nv.index[0].date().isoformat(), "last": nv.index[-1].date().isoformat(), "cagr": cg(nv["nav"]), "cagr_5050": cg(nv["5050_buyhold"]), "excess_cagr": cg(nv["nav"]) - cg(nv["5050_buyhold"]),
         "ir": float(e.mean() * 252 / (e.std(ddof=1) * np.sqrt(252))), "years_ahead": float((yr["nav"] > yr["5050_buyhold"]).mean()), "n_years": int(len(yr)),
         "rel_maxdd": float((rel / rel.cummax() - 1).min()), "rel_end": float(rel.iloc[-1])}
    return o, yearly, rel


def shortfall(lr: pd.Series, tp: pd.DataFrame, dd: np.ndarray):
    """ln(strategy / 50/50) cut at the close before each trade day: one piece per holding period (trades and neutral stretches)."""
    days = lr.index; tp = tp.sort_values("trade_date").reset_index(drop=True); segs = []
    bnd = [float(lr.iloc[max(days.searchsorted(d) - 1, 0)]) for d in tp["trade_date"]] + [float(lr.iloc[-1])]
    for i, r in tp.iterrows():
        last = i + 1 == len(tp); end_dec = tp["decision_date"].iloc[i + 1] if not last else None
        wk = int(((dd >= np.datetime64(r["decision_date"])) & ((dd < np.datetime64(end_dec)) if end_dec is not None else True)).sum())
        segs.append({"state": r["to"], "kind": "trade" if r["to"] in ("A", "B") else "neutral", "start": r["trade_date"].date().isoformat(),
                     "end": (tp["trade_date"].iloc[i + 1] if not last else days[-1]).date().isoformat(), "open": bool(last and r["to"] in ("A", "B")), "weeks": wk, "log_contrib": bnd[i + 1] - bnd[i]})
    sg = pd.DataFrame(segs); tt = sg[sg["kind"] == "trade"].sort_values("log_contrib")
    assert abs(sg["log_contrib"].sum() - float(lr.iloc[-1])) < 1e-9
    lng, sht = tt[tt["weeks"] > 52], tt[tt["weeks"] <= 52]
    sc = {"first": days[0].date().isoformat(), "last": days[-1].date().isoformat(), "total": float(lr.iloc[-1]), "trades": len(tt), "long_n": len(lng), "long_sum": float(lng["log_contrib"].sum()),
          "short_n": len(sht), "short_sum": float(sht["log_contrib"].sum()), "neutral_sum": float(sg[sg["kind"] == "neutral"]["log_contrib"].sum()), "worst3_sum": float(tt["log_contrib"].head(3).sum()),
          "worst3_weeks": ", ".join(str(int(w)) for w in tt["weeks"].head(3)), "corr_weeks": float(np.corrcoef(tt["weeks"], tt["log_contrib"])[0, 1]) if len(tt) > 2 else float("nan")}
    return segs, sc


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); D = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily()
    audit_t = json.load(open(os.path.join(EV, "audit.json"), encoding="utf-8"))
    decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]); decs = decs[(decs["window"] == WIN) & (decs["exit"] == EXIT)]
    extras = pd.read_csv(os.path.join(EV, "trade_extras.csv"), parse_dates=["entry_decision"]); extras = extras[(extras["window"] == WIN) & (extras["exit"] == EXIT)]
    summ = pd.read_csv(os.path.join(EV, "summary.csv"))
    trs = pd.read_csv(os.path.join(EV, "trades.csv"), parse_dates=["decision_date", "trade_date"]); trs = trs[(trs["window"] == WIN) & (trs["exit"] == EXIT)]
    R = {}; checks = {"z_max_abs_diff": 0.0, "rule_mismatch": [], "rule_trades_matched": 0, "zwin_daily_max_abs_diff": 0.0, "variant_rule_mismatch": []}
    # the rule run with every window's band through the engine's book; the 52-week one must be T03 exactly
    VAR = {(p, w): V.run_variant(px, p, w) for p in PAIRS for w, _, _, _ in ZW.WINDOWS}
    var_bad = sum((V.check_against_t03(VAR[(p, ZW.DEFAULT)]) for p in PAIRS), [])
    if var_bad:
        sys.exit(f"E04: the 52-week variant is not T03: {var_bad}")
    print(f"  variants: {len(VAR)} pair-window runs; the 52-week one equals T03's decisions and NAV on every pair  [{time.time() - t0:4.0f}s]", flush=True)
    for pid, pair in enumerate(PAIRS):
        dates, P, Z, chk = paths_for(pair, pid, px, D); checks["z_max_abs_diff"] = max(checks["z_max_abs_diff"], chk); L = P.shape[1]
        # ---- 4. the rule on every row; row 0 must give T03's trades exactly
        closed, opened = run_rule(Z)
        mine = sorted((dates[t_in].date().isoformat(), "A" if leg == 1 else "B") for p_, t_in, t_out, leg in closed if p_ == 0)
        mine += sorted((dates[t_in].date().isoformat(), "A" if leg == 1 else "B") for p_, t_in, leg in opened if p_ == 0)
        dd = decs[decs["pair"] == pair]; ent = dd[dd["reason"].fillna("").str.contains("enter")]
        theirs = sorted((d.date().isoformat(), s) for d, s in zip(ent["date"], ent["state_after"]))
        if sorted(mine) != theirs:
            checks["rule_mismatch"].append(pair)
        checks["rule_trades_matched"] += len(theirs)
        keep = closed[closed[:, 1] >= E1.BURN]
        ret = np.where(keep[:, 3] == 1, 1.0, -1.0) * (P[keep[:, 0], keep[:, 2]] - P[keep[:, 0], keep[:, 1]]); wks = keep[:, 2] - keep[:, 1]
        by_row = {}
        for row in (0,) + tuple(range(1, D + 1)):
            pass
        order = np.argsort(keep[:, 0], kind="stable"); rows_sorted = keep[order, 0]; cut = np.searchsorted(rows_sorted, np.arange(D + 2))
        stats_rows = []
        for row in range(D + 1):
            sl = order[cut[row]:cut[row + 1]]; stats_rows.append(trade_stats(ret[sl], wks[sl]))
        # real trades recomputed close to close vs T03's fill-based extras (description)
        r0 = order[cut[0]:cut[1]]; real_tr = pd.DataFrame({"entry_decision": [dates[t] for t in keep[r0, 1]], "ret": ret[r0], "weeks": wks[r0]})
        ex = extras[(extras["pair"] == pair) & (~extras["open"])].merge(real_tr, on="entry_decision", how="inner")
        corr_fill = float(np.corrcoef(ex["ret"], ex["extra"])[0, 1]) if len(ex) > 2 else float("nan")
        # ---- 2. event study on every first touch
        sig = first_touches(positions(Z)); okc = np.zeros(L, bool); okc[E1.BURN: L - K_MAX] = True
        S_ = np.where(okc[None, :], sig, 0.0); cnt = np.abs(S_).sum(axis=1)
        ev_sum = np.stack([(S_[:, :L - k] * (P[:, k:] - P[:, :-k])).sum(axis=1) for k in range(1, K_MAX + 1)], axis=1)   # (rows, K)
        # ---- 3. information coefficient
        ic = {}
        for k in IC_H:
            sl = slice(E1.BURN, L - k); x = -Z[:, sl]; y = P[:, E1.BURN + k: L] - P[:, sl]
            ic[k] = spearman_rows(x, y)
            half = (L - k - E1.BURN) // 2
            ic[f"{k}_h1"] = float(spearman_rows(x[:1, :half], y[:1, :half])[0]); ic[f"{k}_h2"] = float(spearman_rows(x[:1, half:], y[:1, half:])[0])
        # ---- the other z windows (sections 2 and 3): weekly on the same paths, daily on the daily paths in chunks
        WR = {"w52": {"freq": "W", "ev": ev_sum, "cnt": cnt, "ic": np.stack([ic[k] for k in IC_H], axis=1), "halves": [(ic[f"{k}_h1"], ic[f"{k}_h2"]) for k in IC_H],
                      "first": dates[E1.BURN].date().isoformat(), "last": dates[-1].date().isoformat()}}
        RW = {"w52": {"stats": stats_rows, "rows": keep[:, 0], "ret": ret, "wks": wks}}
        def entries(closed_, opened_, dts):
            return sorted([(dts[t_in].date().isoformat(), "A" if leg == 1 else "B") for p_, t_in, t_out, leg in closed_ if p_ == 0] + [(dts[t_in].date().isoformat(), "A" if leg == 1 else "B") for p_, t_in, leg in opened_ if p_ == 0])
        def var_entries(w_):
            d_ = VAR[(pair, w_)]["decisions"]; e_ = d_[d_["reason"].str.contains("enter")]
            return sorted((d.date().isoformat(), s_) for d, s_ in zip(e_.index, e_["state_after"]))
        for w in ("w26", "w13"):
            Zw = ZW.trailing_z(np.exp(P), ZW.WIN[w][3]); e_, c_, i_ = ev_ic(P, Zw, E1.BURN, range(1, K_MAX + 1), IC_H)
            WR[w] = {"freq": "W", "ev": e_, "cnt": c_, "ic": i_, "halves": ic_halves(P[0], Zw[0], E1.BURN, IC_H), "first": dates[E1.BURN].date().isoformat(), "last": dates[-1].date().isoformat()}
            Zr = Zw.copy(); Zr[0] = VAR[(pair, w)]["decisions"]["z"].to_numpy()               # the real row: the variant's own z
            st_, kp_, rt_, wk_, cl_, op_ = rule_block(Zr, P, E1.BURN); RW[w] = {"stats": st_, "rows": kp_[:, 0], "ret": rt_, "wks": wk_}
            if entries(cl_, op_, dates) != var_entries(w):
                checks["variant_rule_mismatch"].append(f"{pair} {w}")
        pf_ = pair_frame(px, pair, "plain"); dser = ZW.series(pf_, "D"); lnd = np.log(dser["ratio"].to_numpy()); bd = ZW.BURN["D"]; acc = {}
        dpos = dser.index.get_indexer(pf_.index[pf_["decision"]]); bw = int(np.searchsorted(dpos, bd)); racc = {}
        for w in DAILY:
            n_ = ZW.WIN[w][3]; zr = ZW.trailing_z(np.exp(lnd[None, :]), n_)
            checks["zwin_daily_max_abs_diff"] = max(checks["zwin_daily_max_abs_diff"], float(np.nanmax(np.abs(zr[0] - ZW.z_of(dser["ratio"], n_).to_numpy()))))
            e_, c_, i_ = ev_ic(lnd[None, :], zr, bd, K_D, IC_D); acc[w] = ([e_], [c_], [i_], ic_halves(lnd, zr[0], bd, IC_D))
            zreal = VAR[(pair, w)]["decisions"]["z"].to_numpy()[None, :]; st_, kp_, rt_, wk_, cl_, op_ = rule_block(zreal, lnd[dpos][None, :], bw)
            if entries(cl_, op_, dates) != var_entries(w):
                checks["variant_rule_mismatch"].append(f"{pair} {w}")
            racc[w] = {"stats": list(st_), "rows": [kp_[:, 0]], "ret": [rt_], "wks": [wk_]}
        for a0, Pc in ZW.random_log_paths(lnd, pid, "D", D):
            Ec = np.exp(Pc)
            for w in DAILY:
                zc = ZW.trailing_z(Ec, ZW.WIN[w][3]); e_, c_, i_ = ev_ic(Pc, zc, bd, K_D, IC_D); acc[w][0].append(e_); acc[w][1].append(c_); acc[w][2].append(i_)
                st_, kp_, rt_, wk_, _, _ = rule_block(zc[:, dpos], Pc[:, dpos], bw)
                racc[w]["stats"] += st_; racc[w]["rows"].append(kp_[:, 0] + 1 + a0); racc[w]["ret"].append(rt_); racc[w]["wks"].append(wk_)
        for w in DAILY:
            WR[w] = {"freq": "D", "ev": np.concatenate(acc[w][0]), "cnt": np.concatenate(acc[w][1]), "ic": np.concatenate(acc[w][2]), "halves": acc[w][3],
                     "first": dser.index[bd].date().isoformat(), "last": dser.index[-1].date().isoformat()}
            RW[w] = {"stats": racc[w]["stats"], "rows": np.concatenate(racc[w]["rows"]), "ret": np.concatenate(racc[w]["ret"]), "wks": np.concatenate(racc[w]["wks"])}
        R[pair] = {"pid": pid, "dates": dates, "first": dates[E1.BURN].date().isoformat(), "last": dates[-1].date().isoformat(), "rule": stats_rows, "trade_rows": keep, "trade_ret": ret, "trade_wks": wks,
                   "corr_fill": corr_fill, "n_fill_matched": len(ex), "ev_sum": ev_sum, "ev_cnt": cnt, "ic": ic, "win": WR, "rwin": RW}
        print(f"  {pair:18s} touches {int(cnt[0]):3d}  trades {stats_rows[0]['n']:3d}  [{time.time() - t0:4.0f}s]", flush=True)
    if checks["z_max_abs_diff"] > 1e-9 or checks["rule_mismatch"] or checks["zwin_daily_max_abs_diff"] > 1e-6 or checks["variant_rule_mismatch"]:   # daily running sums over ~5,000 days: float error ~1e-8
        sys.exit(f"E04 check FAILED (the paths' z or the rule copy does not match): {checks}")
    print(f"  checks: z {checks['z_max_abs_diff']:.1e}; the rule copy reproduces all {checks['rule_trades_matched']} T03 entries")

    # ---------------------------------------------------------------- 1. outperformance
    outp, yearly = [], []
    for p in ORDER:
        nv = pd.read_csv(os.path.join(EV, f"nav_{p.replace('|', '_')}_{WIN}_{EXIT}_{COST:g}.csv"), index_col=0, parse_dates=True)[["nav", "5050_buyhold", "hold_a", "hold_b"]].dropna()
        rel = nv["nav"] / nv["5050_buyhold"]; e = np.log(nv["nav"]).diff().dropna() - np.log(nv["5050_buyhold"]).diff().dropna()
        yrs = (nv.index[-1] - nv.index[0]).days / 365.25; cg = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
        ye = nv.groupby(nv.index.year).last(); y0 = nv.iloc[[0]].rename(index=lambda d: d.year - 1); yy = pd.concat([y0, ye]); yr = (yy / yy.shift(1) - 1).dropna()
        for y_, row in yr.iterrows():
            yearly.append({"pair": p, "year": int(y_), "strategy": float(row["nav"]), "5050": float(row["5050_buyhold"]), "excess": float(row["nav"] - row["5050_buyhold"])})
        R[p]["rel"] = rel; R[p]["lrel"] = np.log(rel)
        outp.append({"pair": p, "first": nv.index[0].date().isoformat(), "last": nv.index[-1].date().isoformat(), "cagr": cg(nv["nav"]), "cagr_5050": cg(nv["5050_buyhold"]), "excess_cagr": cg(nv["nav"]) - cg(nv["5050_buyhold"]),
                     "ir": float(e.mean() * 252 / (e.std(ddof=1) * np.sqrt(252))), "years_ahead": float((yr["nav"] > yr["5050_buyhold"]).mean()), "n_years": int(len(yr)),
                     "rel_maxdd": float((rel / rel.cummax() - 1).min()), "rel_end": float(rel.iloc[-1])})
    OPF = pd.DataFrame(outp); YE = pd.DataFrame(yearly); OPF.to_csv(os.path.join(OUT, "outperformance.csv"), index=False); YE.to_csv(os.path.join(OUT, "yearly_excess.csv"), index=False)

    # ---------------------------------------------------------------- 5. where the shortfall comes from (trade by trade)
    # ln(strategy / 50/50) changes only through what the sleeve holds; cut at the close before each trade day it telescopes
    # exactly into one piece per holding period: each trade, and each neutral stretch between trades.
    seg_rows, sc_rows = [], []
    for p in ORDER:
        lr = R[p]["lrel"]; days = lr.index; tp = trs[trs["pair"] == p].sort_values("trade_date").reset_index(drop=True)
        dd = decs[decs["pair"] == p]["date"].sort_values().to_numpy()
        bnd = [float(lr.iloc[max(days.searchsorted(d) - 1, 0)]) for d in tp["trade_date"]] + [float(lr.iloc[-1])]
        for i, r in tp.iterrows():
            last = i + 1 == len(tp); end_dec = tp["decision_date"].iloc[i + 1] if not last else None
            wk = int(((dd >= np.datetime64(r["decision_date"])) & ((dd < np.datetime64(end_dec)) if end_dec is not None else True)).sum())
            seg_rows.append({"pair": p, "state": r["to"], "kind": "trade" if r["to"] in ("A", "B") else "neutral", "start": r["trade_date"].date().isoformat(),
                             "end": (tp["trade_date"].iloc[i + 1] if not last else days[-1]).date().isoformat(), "open": bool(last and r["to"] in ("A", "B")),
                             "weeks": wk, "log_contrib": bnd[i + 1] - bnd[i]})
        sg = pd.DataFrame([x for x in seg_rows if x["pair"] == p]); tt = sg[sg["kind"] == "trade"].sort_values("log_contrib")
        assert abs(sg["log_contrib"].sum() - float(lr.iloc[-1])) < 1e-9, p                     # the pieces add up to the whole
        lng, sht = tt[tt["weeks"] > 52], tt[tt["weeks"] <= 52]
        sc_rows.append({"pair": p, "first": days[0].date().isoformat(), "last": days[-1].date().isoformat(), "total": float(lr.iloc[-1]), "trades": len(tt),
                        "long_n": len(lng), "long_sum": float(lng["log_contrib"].sum()), "short_n": len(sht), "short_sum": float(sht["log_contrib"].sum()),
                        "neutral_sum": float(sg[sg["kind"] == "neutral"]["log_contrib"].sum()), "worst3_sum": float(tt["log_contrib"].head(3).sum()),
                        "worst3_weeks": ", ".join(str(int(w)) for w in tt["weeks"].head(3)), "corr_weeks": float(np.corrcoef(tt["weeks"], tt["log_contrib"])[0, 1]) if len(tt) > 2 else float("nan")})
    SEG = pd.DataFrame(seg_rows); SC = pd.DataFrame(sc_rows); SEG.to_csv(os.path.join(OUT, "shortfall_by_trade.csv"), index=False); SC.to_csv(os.path.join(OUT, "shortfall_summary.csv"), index=False)
    pct_x = lambda v: (np.exp(v) - 1) * 100                                                    # noqa: E731  (log piece -> % change of strategy / 50/50)
    # ---------------------------------------------------------------- 1 and 5 for every z window (the variants; 52 weeks = the rows above)
    opw, yew, segw, scw, RELW = [], [], [], [], {}
    for w, lab, f, n_ in ZW.WINDOWS:
        for p in ORDER:
            v = VAR[(p, w)]; o, yrl, rel = outperf(v["nav"], v["bh"]); RELW[(p, w)] = rel
            opw.append({"window": w, "pair": p, **o}); yew += [{"window": w, "pair": p, **y_} for y_ in yrl]
            sg_, sc_ = shortfall(np.log(rel), v["trades"], v["decisions"].index.to_numpy()); segw += [{"window": w, "pair": p, **x_} for x_ in sg_]; scw.append({"window": w, "pair": p, **sc_})
    OPW, YEW, SEGW, SCW = pd.DataFrame(opw), pd.DataFrame(yew), pd.DataFrame(segw), pd.DataFrame(scw)
    a_ = OPW[OPW["window"] == ZW.DEFAULT].set_index("pair")[["cagr", "ir", "rel_end"]]; b_ = OPF.set_index("pair")[["cagr", "ir", "rel_end"]]
    assert np.allclose(a_.reindex(b_.index).to_numpy(), b_.to_numpy(), atol=1e-9), "the 52-week variant's outperformance is not T03's"
    a_ = SCW[SCW["window"] == ZW.DEFAULT].set_index("pair")["total"]; assert np.allclose(a_.reindex(SC.set_index("pair").index).to_numpy(), SC["total"].to_numpy(), atol=1e-9)
    OPW.to_csv(os.path.join(OUT, "outperformance_windows.csv"), index=False); YEW.to_csv(os.path.join(OUT, "yearly_excess_windows.csv"), index=False)
    SEGW.to_csv(os.path.join(OUT, "shortfall_by_trade_windows.csv"), index=False); SCW.to_csv(os.path.join(OUT, "shortfall_summary_windows.csv"), index=False)

    # ---------------------------------------------------------------- 2. event study (per pair and pooled index pairs)
    ev_rows = []; EV_ = {}
    def ev_summary(sums, cnts):
        means = sums / np.maximum(cnts, 1)[:, None]; real = means[0]; null = means[1:]
        return real, np.percentile(null, 5, axis=0), np.percentile(null, 50, axis=0), np.percentile(null, 95, axis=0), np.array([E1.pctile(real[k], null[:, k]) for k in range(K_MAX)])
    for p in ORDER:
        EV_[p] = ev_summary(R[p]["ev_sum"], R[p]["ev_cnt"])
    EV_["index"] = ev_summary(sum(R[p]["ev_sum"] for p in INDEX_PAIRS), sum(R[p]["ev_cnt"] for p in INDEX_PAIRS))
    for key, (real, lo, md, hi, pc) in EV_.items():
        cnt = int(R[key]["ev_cnt"][0]) if key in R else int(sum(R[p]["ev_cnt"][0] for p in INDEX_PAIRS))
        ev_rows += [{"pair": key, "k": k + 1, "touches": cnt, "real": real[k], "rw_p5": lo[k], "rw_median": md[k], "rw_p95": hi[k], "pct": pc[k]} for k in range(K_MAX)]
    EVS = pd.DataFrame(ev_rows); EVS.to_csv(os.path.join(OUT, "event_study.csv"), index=False)

    # ---------------------------------------------------------------- 3. IC
    ic_rows = []
    for p in ORDER:
        for k in IC_H:
            v = R[p]["ic"][k]; ic_rows.append({"pair": p, "k": k, "ic": float(v[0]), "rw_p5": float(np.percentile(v[1:], 5)), "rw_median": float(np.median(v[1:])), "rw_p95": float(np.percentile(v[1:], 95)),
                                              "pct": E1.pctile(v[0], v[1:]), "ic_first_half": R[p]["ic"][f"{k}_h1"], "ic_second_half": R[p]["ic"][f"{k}_h2"]})
    for k in IC_H:
        v = np.mean([R[p]["ic"][k] for p in INDEX_PAIRS], axis=0)
        ic_rows.append({"pair": "index", "k": k, "ic": float(v[0]), "rw_p5": float(np.percentile(v[1:], 5)), "rw_median": float(np.median(v[1:])), "rw_p95": float(np.percentile(v[1:], 95)), "pct": E1.pctile(v[0], v[1:]),
                        "ic_first_half": float(np.mean([R[p]["ic"][f"{k}_h1"] for p in INDEX_PAIRS])), "ic_second_half": float(np.mean([R[p]["ic"][f"{k}_h2"] for p in INDEX_PAIRS]))})
    ICS = pd.DataFrame(ic_rows); ICS.to_csv(os.path.join(OUT, "ic.csv"), index=False)

    # ---------------------------------------------------------------- 2 and 3 for every z window
    kgrid = lambda f: tuple(range(1, K_MAX + 1)) if f == "W" else K_D                  # noqa: E731
    hgrid = lambda f: IC_H if f == "W" else IC_D                                      # noqa: E731
    xwk = lambda f, k: float(k) if f == "W" else k / 5.0                              # noqa: E731  (in weeks: 5 trading days a week)
    evw_rows, icw_rows = [], []
    for w, lab, f, n_ in ZW.WINDOWS:
        for key in ORDER + ["index"]:
            if key == "index":
                sums = sum(R[p]["win"][w]["ev"] for p in INDEX_PAIRS); cnts = sum(R[p]["win"][w]["cnt"] for p in INDEX_PAIRS); icm = np.mean([R[p]["win"][w]["ic"] for p in INDEX_PAIRS], axis=0)
                halves = [tuple(np.mean([R[p]["win"][w]["halves"][j] for p in INDEX_PAIRS], axis=0)) for j in range(len(hgrid(f)))]
                first, last = min(R[p]["win"][w]["first"] for p in INDEX_PAIRS), max(R[p]["win"][w]["last"] for p in INDEX_PAIRS)
            else:
                ww = R[key]["win"][w]; sums, cnts, icm, halves, first, last = ww["ev"], ww["cnt"], ww["ic"], ww["halves"], ww["first"], ww["last"]
            means = sums / np.maximum(cnts, 1)[:, None]; real = means[0]; null = means[1:]
            for j, k in enumerate(kgrid(f)):
                evw_rows.append({"window": w, "pair": key, "k": k, "x_weeks": xwk(f, k), "touches": int(cnts[0]), "first": first, "last": last, "real": float(real[j]), "rw_p5": float(np.percentile(null[:, j], 5)),
                                 "rw_median": float(np.median(null[:, j])), "rw_p95": float(np.percentile(null[:, j], 95)), "pct": E1.pctile(real[j], null[:, j])})
            for j, h in enumerate(hgrid(f)):
                icw_rows.append({"window": w, "pair": key, "h": h, "x_weeks": xwk(f, h), "ic": float(icm[0, j]), "rw_p5": float(np.percentile(icm[1:, j], 5)), "rw_median": float(np.median(icm[1:, j])),
                                 "rw_p95": float(np.percentile(icm[1:, j], 95)), "pct": E1.pctile(icm[0, j], icm[1:, j]), "ic_first_half": halves[j][0], "ic_second_half": halves[j][1], "first": first, "last": last})
    EVW = pd.DataFrame(evw_rows); ICW = pd.DataFrame(icw_rows); EVW.to_csv(os.path.join(OUT, "event_study_windows.csv"), index=False); ICW.to_csv(os.path.join(OUT, "ic_windows.csv"), index=False)
    for w in ("w52",):                                                                  # the 52-week rows must be the existing ones
        a_ = EVW[(EVW["window"] == w)].set_index(["pair", "k"])["real"]; b_ = EVS.set_index(["pair", "k"])["real"]; assert np.allclose(a_.reindex(b_.index), b_, atol=1e-12)
    win_notable = {w: (int(sum(E1.notable(v) for v in ICW[ICW["window"] == w]["pct"])), int(ICW[ICW["window"] == w]["pct"].notna().sum())) for w, _, _, _ in ZW.WINDOWS}

    # ---------------------------------------------------------------- 4. the rule on random walks
    rr_rows = []
    def rr(key, rows):
        real = rows[0]; null = rows[1:]
        for s, lab, kind in STATS:
            col = np.array([x[s] for x in null], float)
            rr_rows.append({"pair": key, "stat": s, "label": lab, "real": real[s], "rw_p5": float(np.nanpercentile(col, 5)), "rw_median": float(np.nanmedian(col)), "rw_p95": float(np.nanpercentile(col, 95)), "pct": E1.pctile(real[s], col)})
    for p in ORDER:
        rr(p, R[p]["rule"])
    pooled = []
    for row in range(D + 1):
        rets = np.concatenate([R[p]["trade_ret"][R[p]["trade_rows"][:, 0] == row] for p in INDEX_PAIRS]); wks = np.concatenate([R[p]["trade_wks"][R[p]["trade_rows"][:, 0] == row] for p in INDEX_PAIRS])
        pooled.append(trade_stats(rets, wks))
    rr("index", pooled)
    RRS = pd.DataFrame(rr_rows); RRS.to_csv(os.path.join(OUT, "rule_on_random.csv"), index=False)
    rrw = []
    for w, lab, f, n_ in ZW.WINDOWS:
        def rrw_add(key, rows):
            real = rows[0]; null = rows[1:]
            for s_, lab_, kind in STATS:
                col = np.array([x[s_] for x in null], float)
                rrw.append({"window": w, "pair": key, "stat": s_, "label": lab_, "real": real[s_], "rw_p5": float(np.nanpercentile(col, 5)), "rw_median": float(np.nanmedian(col)), "rw_p95": float(np.nanpercentile(col, 95)), "pct": E1.pctile(real[s_], col)})
        for p in ORDER:
            rrw_add(p, R[p]["rwin"][w]["stats"])
        pooled_w = []
        for row in range(D + 1):
            rets = np.concatenate([R[p]["rwin"][w]["ret"][R[p]["rwin"][w]["rows"] == row] for p in INDEX_PAIRS]); wks_ = np.concatenate([R[p]["rwin"][w]["wks"][R[p]["rwin"][w]["rows"] == row] for p in INDEX_PAIRS])
            pooled_w.append(trade_stats(rets, wks_))
        rrw_add("index", pooled_w)
    RRW = pd.DataFrame(rrw); RRW.to_csv(os.path.join(OUT, "rule_on_random_windows.csv"), index=False)
    a_ = RRW[RRW["window"] == ZW.DEFAULT].set_index(["pair", "stat"])["real"]; assert np.allclose(a_.reindex(RRS.set_index(["pair", "stat"]).index).to_numpy(), RRS["real"].to_numpy(), equal_nan=True)
    counted = ICS["pct"].tolist() + RRS["pct"].tolist(); counted = [v for v in counted if v == v]; n_not = sum(E1.notable(v) for v in counted)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "test": TEST, "test_run_date": audit_t.get("run_date"), "setting": {"window": WIN, "exit": EXIT, "cost_bps": COST},
             "draws": D, "seed": f"default_rng([{E1.SEED_SIGNS}, pair_id]) (E01's signs)", "checks": checks, "notable": [n_not, len(counted)],
             "fill_vs_close": {p: {"corr": R[p]["corr_fill"], "trades": R[p]["n_fill_matched"]} for p in ORDER},
             "z_windows": {"windows": [w for w, _, _, _ in ZW.WINDOWS], "daily_seed": f"default_rng([{ZW.SEED['D']}, pair_id])", "daily_warm_up_days": ZW.BURN["D"], "ic_notable_by_window": win_notable,
                           "daily_event_k_days": list(K_D), "daily_ic_h_days": list(IC_D)}, "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return f"**{v:.1f}**" if E1.notable(v) else E1.f2(v, 1)
    def g(df, p, **kw):
        sub = df[df["pair"] == p]
        for k_, v_ in kw.items():
            sub = sub[sub[k_] == v_]
        return sub.iloc[0]
    md = [f"# E04 results — does the ratio band have any predictive power? (EXPLORATORY)\n",
          f"_Generated by `explore/E04_signal_strength/strength.py` on {audit['run_date']} from T03 (run {audit['test_run_date']}; window {WIN}, exit {EXIT}, {COST} bps, no maximum hold) and {D:,} random paths per pair (E01's). Description, not a finding. Percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Checks:** the paths' z equals E01's (max |diff| {checks['z_max_abs_diff']:.1e}); the vectorised rule reproduces all {checks['rule_trades_matched']} of T03's entries exactly. **Against chance:** {n_not} of {len(counted)} counted percentiles are notable (about {0.1 * len(counted):.0f} by chance).\n",
          "## 1. Outperformance against 50/50 bought once (T03's book)\n", "| pair | window | CAGR | 50/50 | excess a year | information ratio | years ahead | worst fall of strategy ÷ 50/50 | strategy ÷ 50/50 at the end |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {nice(r['pair'])} | {r['first']} → {r['last']} | {r['cagr'] * 100:.1f}% | {r['cagr_5050'] * 100:.1f}% | {r['excess_cagr'] * 100:+.1f} pp | {r['ir']:+.2f} | {r['years_ahead'] * 100:.0f}% of {r['n_years']} | {r['rel_maxdd'] * 100:.0f}% | {r['rel_end']:.2f} |" for _, r in OPF.iterrows()]
    md += ["\n## 2. What follows a band touch: the signalled leg against the other, k weeks later (mean over touches; random walks' median, percentile)\n", "| pair | touches | " + " | ".join(f"{k} wk" for k in EVENT_SHOW) + " |", "|---|---|" + "---|" * len(EVENT_SHOW)]
    for p in ["index"] + ORDER:
        rws = [g(EVS, p, k=k) for k in EVENT_SHOW]
        md.append(f"| {'six index pairs' if p == 'index' else nice(p)} | {int(rws[0]['touches'])} | " + " | ".join(f"{r_['real'] * 100:+.1f}% ({r_['rw_median'] * 100:+.1f}%, {pq(r_['pct'])})" for r_ in rws) + " |")
    md += ["\n## 3. Information coefficient: Spearman of −z with the forward relative return (random walks' 5th–95th, percentile; first half / second half)\n", "| pair | " + " | ".join(f"{k} wk" for k in IC_H) + " |", "|---|" + "---|" * len(IC_H)]
    for p in ["index"] + ORDER:
        md.append(f"| {'six index pairs (mean)' if p == 'index' else nice(p)} | " + " | ".join(f"{r_['ic']:+.3f} ({r_['rw_p5']:+.2f}…{r_['rw_p95']:+.2f}, {pq(r_['pct'])}; {r_['ic_first_half']:+.2f} / {r_['ic_second_half']:+.2f})" for r_ in (g(ICS, p, k=k) for k in IC_H)) + " |")
    md += ["\n## 2–3 for every z window (pooled six index pairs; event study at about 13 and 26 weeks, IC at about 4 and 26 weeks; random walks' median, percentile)\n",
           "Weekly windows on decision-week closes with E01's random walks; daily windows on daily closes with daily random walks (seed `default_rng([111, pair_id])`, the same construction after a 250-day warm-up). Horizons in trading days for the daily windows (65 ≈ 13 weeks, 130 ≈ 26).\n",
           "| z window | touches | after ≈13 weeks | after ≈26 weeks | IC ≈4 weeks | IC ≈26 weeks | IC percentiles notable (all pairs and horizons) |", "|---|---|---|---|---|---|---|"]
    for w, lab, f, n_ in ZW.WINDOWS:
        k13, k26, h4, h26 = (13, 26, 4, 26) if f == "W" else (65, 130, 20, 130)
        e13 = EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == k13)].iloc[0]; e26 = EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == k26)].iloc[0]
        i4 = ICW[(ICW["window"] == w) & (ICW["pair"] == "index") & (ICW["h"] == h4)].iloc[0]; i26 = ICW[(ICW["window"] == w) & (ICW["pair"] == "index") & (ICW["h"] == h26)].iloc[0]
        md.append(f"| {lab}{' (daily)' if f == 'D' else ''} | {int(e13['touches'])} | {e13['real'] * 100:+.2f}% ({e13['rw_median'] * 100:+.2f}%, {pq(e13['pct'])}) | {e26['real'] * 100:+.2f}% ({e26['rw_median'] * 100:+.2f}%, {pq(e26['pct'])}) | "
                  f"{i4['ic']:+.3f} ({i4['rw_median']:+.3f}, {pq(i4['pct'])}) | {i26['ic']:+.3f} ({i26['rw_median']:+.3f}, {pq(i26['pct'])}) | {win_notable[w][0]} of {win_notable[w][1]} |")
    md += ["\n## 4. The T03 rule on the real ratio and on the random walks (trades after the warm-up; relative return close to close)\n", "| pair | " + " | ".join(lab for _, lab, _ in STATS) + " |", "|---|" + "---|" * len(STATS)]
    for p in ["index"] + ORDER:
        cells = []
        for s, lab, kind in STATS:
            r_ = g(RRS, p, stat=s); f_ = (lambda v: fmt("pct", v)) if s in ("mean", "median") else ((lambda v: f"{v * 100:.0f}%") if s in ("hit", "long") else (lambda v: f"{v:.0f}" if s == "n" else f"{v:.1f}"))
            cells.append(f"{f_(r_['real'])} ({f_(r_['rw_median'])}, {pq(r_['pct'])})")
        md.append(f"| {'six index pairs' if p == 'index' else nice(p)} | " + " | ".join(cells) + " |")
    md += ["\n_The real trades recomputed close to close against T03's fill-based extras: correlation " + ", ".join(f"{nice(p)} {R[p]['corr_fill']:.3f} ({R[p]['n_fill_matched']} trades)" for p in ORDER) + "._"]
    md += ["\n## 1, 4 and 5 for every z window (the rule run with that window's band; decisions weekly, daily windows read at each week's last trading day; 52 weeks = T03)\n",
           "| z window | excess a year, index pairs (range) | information ratio (range) | hit rate, pooled index pairs (random walks, pct) | mean trade (random walks, pct) | trades held over a year (random walks, pct) | index pairs: long holds' piece / the rest's |", "|---|---|---|---|---|---|---|"]
    for w, lab, f, n_ in ZW.WINDOWS:
        ox = OPW[(OPW["window"] == w) & OPW["pair"].isin(INDEX_PAIRS)]; rw_ = RRW[(RRW["window"] == w) & (RRW["pair"] == "index")].set_index("stat"); sx = SCW[(SCW["window"] == w) & SCW["pair"].isin(INDEX_PAIRS)]
        md.append(f"| {lab}{' (daily)' if f == 'D' else ''} | {ox['excess_cagr'].min() * 100:+.1f} to {ox['excess_cagr'].max() * 100:+.1f} pp | {ox['ir'].min():+.2f} to {ox['ir'].max():+.2f} | "
                  f"{rw_.loc['hit', 'real'] * 100:.0f}% ({rw_.loc['hit', 'rw_median'] * 100:.0f}%, {pq(rw_.loc['hit', 'pct'])}) | {rw_.loc['mean', 'real'] * 100:+.2f}% ({rw_.loc['mean', 'rw_median'] * 100:+.2f}%, {pq(rw_.loc['mean', 'pct'])}) | "
                  f"{rw_.loc['long', 'real'] * 100:.0f}% ({rw_.loc['long', 'rw_median'] * 100:.0f}%, {pq(rw_.loc['long', 'pct'])}) | {pct_x(sx['long_sum'].sum()):+.1f}% / {pct_x(sx['short_sum'].sum()):+.1f}% |")
    md += ["\n## 5. Where the shortfall against 50/50 comes from (change of strategy ÷ 50/50 during each holding period)\n",
           "The pieces multiply exactly to the whole: trades held over 52 weeks, trades held 52 weeks or less, and the neutral stretches between trades.\n",
           "| pair | window | strategy ÷ 50/50, whole window | trades held over 52 weeks | trades held 52 weeks or less | neutral stretches | the 3 worst trades (weeks held) | correlation of weeks held with the trade's piece |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {nice(r['pair'])} | {r['first']} → {r['last']} | {pct_x(r['total']):+.1f}% | {(str(r['long_n']) + ' trade' + ('s' if r['long_n'] != 1 else '') + ': ' + format(pct_x(r['long_sum']), '+.1f') + '%') if r['long_n'] else 'none'} | {r['short_n']} trade{'s' if r['short_n'] != 1 else ''}: {pct_x(r['short_sum']):+.1f}% | {pct_x(r['neutral_sum']):+.1f}% | {pct_x(r['worst3_sum']):+.1f}% ({r['worst3_weeks']}) | {r['corr_weeks']:+.2f} |" for _, r in SC.iterrows()]
    open(os.path.join(OUT, "E04_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- figures
    figs = {}; titles = {"op": {}, "ev": {}, "ic": {}, "sf": {}}
    fO = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.1, subplot_titles=("strategy ÷ 50/50 bought once (above 1 = ahead)", "calendar-year return: strategy − 50/50"))
    for w, wlab, f, n_ in ZW.WINDOWS:
        for p in ORDER:
            vis = p == DEFAULT_PAIR and w == ZW.DEFAULT; rel = RELW[(p, w)]; ye = YEW[(YEW["window"] == w) & (YEW["pair"] == p)]; o = OPW[(OPW["window"] == w) & (OPW["pair"] == p)].iloc[0]; mw = p + ZW.SEP + w
            fO.add_trace(go.Scatter(x=rel.index.strftime("%Y-%m-%d").tolist(), y=np.round(rel.to_numpy(), 4), mode="lines", line={"color": BLUE, "width": 1.5}, name="strategy ÷ 50/50", meta=mw, visible=vis, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
            fO.add_trace(go.Bar(x=ye["year"], y=ye["excess"] * 100, marker={"color": [AQUA if v_ > 0 else ORANGE for v_ in ye["excess"]]}, name="excess in the year", meta=mw, visible=vis,
                                customdata=np.c_[ye["strategy"] * 100, ye["5050"] * 100], hovertemplate="%{x}: strategy %{customdata[0]:+.1f}%, 50/50 %{customdata[1]:+.1f}% → %{y:+.1f} pp<extra></extra>"), row=2, col=1)
            titles["op"][mw] = E1.two_line(f"{nice(p)} · z window {wlab}{' (daily closes)' if f == 'D' else ''} · {o['first']} → {o['last']} · CAGR {o['cagr'] * 100:.1f}% vs 50/50 {o['cagr_5050'] * 100:.1f}% ({o['excess_cagr'] * 100:+.1f} pp a year) · information ratio {o['ir']:+.2f}",
                                           f"ahead in {o['years_ahead'] * 100:.0f}% of {o['n_years']} calendar years; the strategy ÷ 50/50 fell at worst {o['rel_maxdd'] * 100:.0f}% and ends at {o['rel_end']:.2f}")
    # sections 2 and 3: one set of traces per pair and z window (meta pair§window), and per window for the pooled and all-pairs views
    fE, fI, fEp, fH = go.Figure(), go.Figure(), go.Figure(), go.Figure(); tE, tI, tEp, tH = {}, {}, {}, {}
    rows_lab = ["six index pairs"] + [nice(p) for p in ORDER]; keys = ["index"] + ORDER
    for w, lab, f, n_ in ZW.WINDOWS:
        unit = "weeks" if f == "W" else "trading days"; wtag = f"z window {lab}{' (daily closes)' if f == 'D' else ''}"
        for p in ORDER + ["index"]:
            ev = EVW[(EVW["window"] == w) & (EVW["pair"] == p)]; icp = ICW[(ICW["window"] == w) & (ICW["pair"] == p)]
            tgt, meta = (fEp, ZW.SEP + w) if p == "index" else (fE, p + ZW.SEP + w); vis = (w == ZW.DEFAULT) and (p in ("index", DEFAULT_PAIR))
            x = ev["x_weeks"].to_numpy()
            tgt.add_trace(go.Scatter(x=x, y=ev["rw_p5"] * 100, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=meta, visible=vis))
            tgt.add_trace(go.Scatter(x=x, y=ev["rw_p95"] * 100, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=meta, visible=vis))
            tgt.add_trace(go.Scatter(x=x, y=ev["real"] * 100, mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 4}, name="the ratio's touches" if p != "index" else "the six index pairs' touches", meta=meta, visible=vis,
                                     customdata=np.c_[ev["k"], ev["rw_median"] * 100, ev["pct"]], hovertemplate="%{customdata[0]} " + unit + " after: %{y:+.2f}% (random walks %{customdata[1]:+.2f}%, percentile %{customdata[2]:.1f})<extra></extra>"))
            if p == "index":
                tEp[w] = E1.two_line(f"The six index pairs together · {wtag} · {int(ev['touches'].iloc[0])} first touches of the band · {ev['first'].iloc[0]} → {ev['last'].iloc[0]}",
                                     "mean relative return of the signalled leg, k weeks after the touch, against the same on random walks" + (" (daily: trading days ÷ 5)" if f == "D" else ""))
                continue
            tE[p + ZW.SEP + w] = E1.two_line(f"{nice(p)} · {wtag} · {ev['first'].iloc[0]} → {ev['last'].iloc[0]} · {int(ev['touches'].iloc[0])} first touches of the band",
                                             "mean relative return of the signalled leg against the other, k weeks after the touch (above 0 = the swing back the rule bets on)" + (" — daily: trading days ÷ 5" if f == "D" else ""))
            fI.add_trace(go.Scatter(x=icp["x_weeks"], y=icp["rw_p5"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=meta, visible=vis))
            fI.add_trace(go.Scatter(x=icp["x_weeks"], y=icp["rw_p95"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=meta, visible=vis))
            fI.add_trace(go.Scatter(x=icp["x_weeks"], y=icp["ic"], mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 8}, name="the ratio", meta=meta, visible=vis,
                                    customdata=np.c_[icp["h"], icp["pct"], icp["ic_first_half"], icp["ic_second_half"]],
                                    hovertemplate="%{customdata[0]} " + unit + ": IC %{y:+.3f}, percentile %{customdata[1]:.1f}<br>first half %{customdata[2]:+.3f}, second half %{customdata[3]:+.3f}<extra></extra>"))
            tI[p + ZW.SEP + w] = E1.two_line(f"{nice(p)} · {wtag} · {icp['first'].iloc[0]} → {icp['last'].iloc[0]} · rank correlation of −z with the next k {unit}' relative return",
                                             "above 0 = a stretched ratio tends to come back (a below-band ratio is followed by the first leg doing better); grey = what random walks give")
        hg = IC_H if f == "W" else IC_D; ic_w = ICW[ICW["window"] == w]
        cell = lambda p, h: ic_w[(ic_w["pair"] == p) & (ic_w["h"] == h)].iloc[0]        # noqa: E731
        fH.add_trace(go.Heatmap(z=[[cell(p, h)["pct"] for h in hg] for p in keys], x=[f"{h} {'wk' if f == 'W' else 'd'}" for h in hg], y=rows_lab, zmin=0, zmax=100, colorscale=E1.DIV_LIGHT,
                                text=[[(f"{cell(p, h)['ic']:+.2f}" if E1.notable(cell(p, h)["pct"]) else "") for h in hg] for p in keys], texttemplate="%{text}",
                                customdata=[[f"IC {cell(p, h)['ic']:+.3f} (random walks {cell(p, h)['rw_median']:+.3f}), percentile {cell(p, h)['pct']:.1f}" for h in hg] for p in keys],
                                hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2, meta=ZW.SEP + w, visible=(w == ZW.DEFAULT),
                                colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        tH[w] = f"Information coefficient against random walks · {wtag}: red = the z-score ranks future relative returns better than chance (labels on notable cells; {win_notable[w][0]} of {win_notable[w][1]} notable)"
    titles["ev"] = tE; titles["ic"] = tI; titles["ev_pool"] = tEp; titles["h_ic"] = tH
    for fg in (fE, fEp):
        fg.add_hline(y=0, line={"color": AXIS, "width": 1}); fg.update_xaxes(title={"text": "weeks after the first touch (daily windows: trading days ÷ 5)", "font": {"color": INK2}}); fg.update_yaxes(ticksuffix="%", title={"text": "signalled leg − the other", "font": {"color": INK2}})
    figs["ev"] = E1.style(fE, tE[DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 460, legend_top=True); figs["ev_pool"] = E1.style(fEp, tEp[ZW.DEFAULT], 440, legend_top=True)
    fI.add_hline(y=0, line={"color": AXIS, "width": 1}); fI.update_xaxes(type="log", tickvals=[0.2, 1, 4, 13, 26, 52], ticktext=["1 d", "1 wk", "4 wk", "13 wk", "26 wk", "52 wk"], title={"text": "horizon", "font": {"color": INK2}})
    fI.update_yaxes(title={"text": "information coefficient", "font": {"color": INK2}}); figs["ic"] = E1.style(fI, tI[DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 440, legend_top=True)
    fH.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fH.update_xaxes(showgrid=False, side="top"); figs["h_ic"] = E1.style(fH, tH[ZW.DEFAULT], 520, legend=False, left=150)
    fO.add_hline(y=1, line={"color": AXIS, "width": 1}, row=1, col=1); fO.add_hline(y=0, line={"color": AXIS, "width": 1}, row=2, col=1)
    fO.update_yaxes(title={"text": "relative NAV", "font": {"color": INK2}}, row=1, col=1); fO.update_yaxes(ticksuffix=" pp", row=2, col=1)
    figs["op"] = E1.style(fO, titles["op"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 640, legend_top=True)
    # heat maps
    def heat(zv, xl, yl, txt, hov, title, h):
        f = go.Figure(go.Heatmap(z=zv, x=xl, y=yl, zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=txt, texttemplate="%{text}", customdata=hov, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
                                 colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="top")
        return E1.style(f, title, h, legend=False, left=150)
    def rr_txt(p, s, w=ZW.DEFAULT):
        r_ = RRW[(RRW["window"] == w) & (RRW["pair"] == p) & (RRW["stat"] == s)].iloc[0]; v = r_["real"]
        return (f"{v * 100:+.1f}%" if s in ("mean", "median") else (f"{v * 100:.0f}%" if s in ("hit", "long") else f"{v:.0f}" if s == "n" else f"{v:.1f}"))
    def rr_cell(p, s, w):
        return RRW[(RRW["window"] == w) & (RRW["pair"] == p) & (RRW["stat"] == s)].iloc[0]
    # 4. the rule on random walks: one heat map per window
    fRR = go.Figure(); titles["h_rr"] = {}
    for w, wlab, f, n_ in ZW.WINDOWS:
        fRR.add_trace(go.Heatmap(z=[[rr_cell(p, s_, w)["pct"] for s_, _, _ in STATS] for p in keys], x=[lab_ for _, lab_, _ in STATS], y=rows_lab, zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, xgap=2, ygap=2, meta=ZW.SEP + w, visible=(w == ZW.DEFAULT),
                                 text=[[rr_txt(p, s_, w) for s_, _, _ in STATS] for p in keys], texttemplate="%{text}",
                                 customdata=[[f"real {rr_txt(p, s_, w)}, random walks' median {rr_cell(p, s_, w)['rw_median'] * (100 if s_ in ('hit', 'long', 'mean', 'median') else 1):.1f}{'%' if s_ in ('hit', 'long', 'mean', 'median') else ''}, percentile {rr_cell(p, s_, w)['pct']:.1f}" for s_, _, _ in STATS] for p in keys],
                                 hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        titles["h_rr"][w] = f"The rule's trades against the same rule on random walks · z window {wlab}{' (daily closes, decisions weekly)' if f == 'D' else ''}: grey = what a series with no swing-back gives (labels: the real value)"
    fRR.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fRR.update_xaxes(showgrid=False, side="top"); figs["h_rr"] = E1.style(fRR, titles["h_rr"][ZW.DEFAULT], 520, legend=False, left=150)
    # 5. per pair and window: each trade's piece of the shortfall; all index-pair trades per window
    fS = go.Figure(); fSa = go.Figure(); titles["sf_all"] = {}; BUCKET = (("up to 13 weeks", 0, 13, AQUA), ("14–52 weeks", 14, 52, BLUE), ("over 52 weeks", 53, 10 ** 6, ORANGE))
    for w, wlab, f, n_ in ZW.WINDOWS:
        wtag = f"z window {wlab}{' (daily closes)' if f == 'D' else ''}"
        for p in ORDER:
            vis = p == DEFAULT_PAIR and w == ZW.DEFAULT; tt = SEGW[(SEGW["window"] == w) & (SEGW["pair"] == p) & (SEGW["kind"] == "trade")]; sc_ = SCW[(SCW["window"] == w) & (SCW["pair"] == p)].iloc[0]; mw = p + ZW.SEP + w
            for lab, lo_, hi_, colr in BUCKET:
                b_ = tt[(tt["weeks"] >= lo_) & (tt["weeks"] <= hi_)]
                fS.add_trace(go.Bar(x=b_["start"].tolist(), y=np.round(pct_x(b_["log_contrib"]), 2), width=40 * 86400000, marker={"color": colr}, name=f"held {lab}", legendgroup=lab, meta=mw, visible=vis,
                                    text=[f"{wk_}w" for wk_ in b_["weeks"]], textposition="outside", textfont={"size": 9}, constraintext="none", cliponaxis=False,
                                    customdata=np.c_[b_["start"], b_["end"], b_["weeks"], ["still open" if o_ else "" for o_ in b_["open"]]],
                                    hovertemplate="%{customdata[0]} → %{customdata[1]} %{customdata[3]}: %{customdata[2]} weeks held<br>strategy ÷ 50/50 changed %{y:+.1f}%<extra></extra>"))
            def n_tr(n, other=False):
                n = int(n); return "no trade" if n == 0 else (("the other" if other else "the") + (" trade" if n == 1 else f" {n} trades"))
            long_txt = (f"{n_tr(sc_['long_n'])} held over 52 weeks: {pct_x(sc_['long_sum']):+.1f}%" if sc_["long_n"] else "no trade held over 52 weeks")
            short_txt = f"{n_tr(sc_['short_n'], other=bool(sc_['long_n']))} {'together ' if sc_['short_n'] > 1 else ''}{pct_x(sc_['short_sum']):+.1f}%"
            titles["sf"][mw] = E1.two_line(f"{nice(p)} · {wtag} · {sc_['first']} → {sc_['last']} · strategy ÷ 50/50 {pct_x(sc_['total']):+.1f}% · {long_txt}",
                                           f"{short_txt}, the neutral stretches {pct_x(sc_['neutral_sum']):+.1f}%; each bar = one trade at its entry date, its height the change of strategy ÷ 50/50 while it was held, its label the weeks held")
        ti = SEGW[(SEGW["window"] == w) & (SEGW["kind"] == "trade") & SEGW["pair"].isin(INDEX_PAIRS)]; ix = SCW[(SCW["window"] == w) & SCW["pair"].isin(INDEX_PAIRS)]
        fSa.add_trace(go.Scatter(x=ti["weeks"], y=np.round(pct_x(ti["log_contrib"]), 2), mode="markers", marker={"size": 8, "color": [ORANGE if wk_ > 52 else (BLUE if wk_ > 13 else AQUA) for wk_ in ti["weeks"]], "opacity": 0.8}, meta=ZW.SEP + w, visible=(w == ZW.DEFAULT),
                                 customdata=np.c_[[nice(q) for q in ti["pair"]], ti["start"], ti["end"]], hovertemplate="%{customdata[0]} · %{customdata[1]} → %{customdata[2]}<br>%{x} weeks held · strategy ÷ 50/50 %{y:+.1f}%<extra></extra>", showlegend=False))
        titles["sf_all"][w] = E1.two_line(f"Every trade on the six index pairs · {wtag}: how long it was held against what it did to strategy ÷ 50/50 · {len(ti)} trades",
                                          (f"the {int(ix['long_n'].sum())} trade{'s' if int(ix['long_n'].sum()) != 1 else ''} held over a year (right of the dotted line) add{'' if int(ix['long_n'].sum()) != 1 else 's'} up, in log terms across the six pairs, to {pct_x(ix['long_sum'].sum()):+.1f}%; the other {int(ix['short_n'].sum())} to {pct_x(ix['short_sum'].sum()):+.1f}%"
                                           if int(ix["long_n"].sum()) else f"no trade was held over a year; the {int(ix['short_n'].sum())} trades add up, in log terms across the six pairs, to {pct_x(ix['short_sum'].sum()):+.1f}%"))
    fS.add_hline(y=0, line={"color": AXIS, "width": 1}); fS.update_yaxes(ticksuffix="%", title={"text": "change of strategy ÷ 50/50 during the trade", "font": {"color": INK2}})
    fS.update_layout(barmode="overlay"); figs["sf"] = E1.style(fS, titles["sf"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 480, legend_top=True)
    fSa.add_hline(y=0, line={"color": AXIS, "width": 1}); fSa.add_vline(x=52, line={"color": AXIS, "width": 1, "dash": "dot"})
    fSa.update_xaxes(title={"text": "weeks held", "font": {"color": INK2}}); fSa.update_yaxes(ticksuffix="%", title={"text": "change of strategy ÷ 50/50 during the trade", "font": {"color": INK2}})
    figs["sf_all"] = E1.style(fSa, titles["sf_all"][ZW.DEFAULT], 460, legend=False)

    # ---------------------------------------------------------------- the page
    def per_pair(key, fig, tables, first=False) -> str:
        t = html.escape(json.dumps(titles[key]), quote=True)
        def div(k, tb_):
            p, w = (k.split(ZW.SEP) + [None])[:2]; shown = p == DEFAULT_PAIR and (w is None or w == ZW.DEFAULT)
            return f"<div data-pair='{html.escape(p, quote=True)}'" + (f" data-win='{w}'" if w else "") + f"{'' if shown else ' hidden'}>{tb_}</div>"
        tb = "".join(div(k, v) for k, v in tables.items())
        return f"<div class='card per-pair' data-titles='{t}'>{E1.html_fig(fig, first=first)}<details><summary>Table view</summary>{tb}</details></div>"
    def per_win(key, fig) -> str:
        return f"<div class='card per-win' data-titles='{html.escape(json.dumps(titles[key]), quote=True)}'>{E1.html_fig(fig)}</div>"
    tab_sf = {p + ZW.SEP + w: E1.tbl(["entered", "exited", "leg", "weeks held", "strategy ÷ 50/50 during the trade"], [[r_["start"], r_["end"] + (" (open)" if r_["open"] else ""), r_["state"], int(r_["weeks"]), f"{pct_x(r_['log_contrib']):+.1f}%"]
                                                                                                                  for _, r_ in SEGW[(SEGW["window"] == w) & (SEGW["pair"] == p) & (SEGW["kind"] == "trade")].sort_values("log_contrib").iterrows()]) for w, _, _, _ in ZW.WINDOWS for p in ORDER}
    tab_op = {p + ZW.SEP + w: E1.tbl(["year", "strategy", "50/50", "excess"], [[int(r_["year"]), f"{r_['strategy'] * 100:+.1f}%", f"{r_['5050'] * 100:+.1f}%", f"{r_['excess'] * 100:+.1f} pp"]
                                                                               for _, r_ in YEW[(YEW["window"] == w) & (YEW["pair"] == p)].iterrows()]) for w, _, _, _ in ZW.WINDOWS for p in ORDER}
    def win_divs(d) -> str:
        return "".join(f"<div data-winonly='{w}'{'' if w == ZW.DEFAULT else ' hidden'}>{v}</div>" for w, v in d.items())
    def summary_par(w) -> str:
        _, wlab, f, _ = ZW.WIN[w]; ox = OPW[(OPW["window"] == w) & OPW["pair"].isin(INDEX_PAIRS)]
        e13 = EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == (13 if f == "W" else 65))].iloc[0]; e26 = EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == (26 if f == "W" else 130))].iloc[0]
        i26 = ICW[(ICW["window"] == w) & (ICW["pair"] == "index") & (ICW["h"] == (26 if f == "W" else 130))].iloc[0]; rw_ = RRW[(RRW["window"] == w) & (RRW["pair"] == "index")].set_index("stat")
        return (f"<p style='margin:4px 0'><b>In one paragraph — z window {wlab}{' (daily closes, decisions weekly)' if f == 'D' else ''}{' (T03)' if w == ZW.DEFAULT else ''}.</b> On the six index pairs the strategy's CAGR minus 50/50's is {ox['excess_cagr'].min() * 100:+.1f} to {ox['excess_cagr'].max() * 100:+.1f} pp a year, with information ratios of {ox['ir'].min():+.2f} to {ox['ir'].max():+.2f}. "
                f"After a first touch of the band the signalled leg does {e13['real'] * 100:+.2f}% against the other over about 13 weeks and {e26['real'] * 100:+.2f}% over about 26 (random walks: {e13['rw_median'] * 100:+.2f}% and {e26['rw_median'] * 100:+.2f}%; percentiles {e13['pct']:.0f} and {e26['pct']:.0f}). "
                f"The z-score's rank correlation with the next ≈26 weeks' relative return averages {i26['ic']:+.3f} over the index pairs (random walks {i26['rw_median']:+.3f}, percentile {i26['pct']:.0f}). "
                f"The rule's trades win {rw_.loc['hit', 'real'] * 100:.0f}% of the time — random walks give {rw_.loc['hit', 'rw_median'] * 100:.0f}% with the same rule (percentile {rw_.loc['hit', 'pct']:.0f}) — and average {rw_.loc['mean', 'real'] * 100:+.2f}% (random walks {rw_.loc['mean', 'rw_median'] * 100:+.2f}%, percentile {rw_.loc['mean', 'pct']:.0f}); {rw_.loc['long', 'real'] * 100:.0f}% of trades are held over a year (random walks {rw_.loc['long', 'rw_median'] * 100:.0f}%)."
                + (f" <b>Against chance:</b> {n_not} of {len(counted)} counted percentiles are notable (about {0.1 * len(counted):.0f} by chance)." if w == ZW.DEFAULT else "") + "</p>")
    SHOW_K = {"W": (1, 2, 4, 8, 13, 26, 39, 52), "D": (1, 5, 20, 40, 65, 130, 195, 250)}
    tab_ev = {p + ZW.SEP + w: E1.tbl([f"{'weeks' if f == 'W' else 'trading days'} after", "the ratio", "random walks' median", "5th–95th", "percentile"],
                                     [[int(r_["k"]), f"{r_['real'] * 100:+.2f}%", f"{r_['rw_median'] * 100:+.2f}%", f"{r_['rw_p5'] * 100:+.2f}% … {r_['rw_p95'] * 100:+.2f}%", E1.fpct(r_["pct"])]
                                      for _, r_ in EVW[(EVW["window"] == w) & (EVW["pair"] == p)].iterrows() if int(r_["k"]) in SHOW_K[f]]) for w, _, f, _ in ZW.WINDOWS for p in ORDER}
    tab_ic = {p + ZW.SEP + w: E1.tbl(["horizon", "IC", "random walks 5th–95th", "percentile", "first half", "second half"],
                                     [[f"{int(r_['h'])} {'wk' if f == 'W' else 'd'}", f"{r_['ic']:+.3f}", f"{r_['rw_p5']:+.3f} … {r_['rw_p95']:+.3f}", E1.fpct(r_["pct"]), f"{r_['ic_first_half']:+.3f}", f"{r_['ic_second_half']:+.3f}"]
                                      for _, r_ in ICW[(ICW["window"] == w) & (ICW["pair"] == p)].iterrows()]) for w, _, f, _ in ZW.WINDOWS for p in ORDER}
    o_rows_w = {w: [[nice(r_["pair"]), f"{r_['first']} → {r_['last']}", f"{r_['cagr'] * 100:.1f}%", f"{r_['cagr_5050'] * 100:.1f}%", f"{r_['excess_cagr'] * 100:+.1f} pp", f"{r_['ir']:+.2f}", f"{r_['years_ahead'] * 100:.0f}% of {r_['n_years']}", f"{r_['rel_maxdd'] * 100:.0f}%"]
                    for _, r_ in OPW[OPW["window"] == w].iterrows()] for w, _, _, _ in ZW.WINDOWS}
    ixp = OPF[OPF["pair"].isin(INDEX_PAIRS)]; ev26 = g(EVS, "index", k=26); ev13 = g(EVS, "index", k=13); ic26 = g(ICS, "index", k=26); hit = g(RRS, "index", stat="hit"); mean_ = g(RRS, "index", stat="mean"); lng = g(RRS, "index", stat="long")
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS).replace("<title>Ratio cycles</title>", "<title>Signal strength</title>"),
         "<header class='bar'><div class='bar-inner'><b>E04 · signal strength</b><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label>{ZW.select_html()}<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Does the ratio band have any predictive power?</h1>",
         f"<p>Four looks at T03 — the ±1 sd first-touch switch on the plain ratio, exit at the trailing mean, no maximum hold (window {WIN}, exit {EXIT}, {COST} bps), refuted against random timing. "
         f"<b>1.</b> Its outperformance against 50/50 over time. <b>2.</b> What follows a touch of the band. <b>3.</b> Whether the z-score ranks future relative returns at all. <b>4.</b> The same rule run on {D:,} random walks per pair (E01's: each ratio's own week-by-week volatility and point-in-time drift, no swing-back), to see whether its trade statistics are signal or mechanics. "
         f"Generated {audit['run_date']} from the approved data of 2026-11-12 and T03's evidence (run {audit['test_run_date']}). <b>Description, not a finding.</b></p>",
         "<div class='card'>" + win_divs({w: summary_par(w) for w, _, _, _ in ZW.WINDOWS}) + "</div>",
         "<p class='note'><b>The z window dropdown at the top changes every section:</b> sections 1, 4 and 5 run the same rule with that window's band (first touch of ±1, exit at the trailing mean, no maximum hold, 20.8 bps, the engine's own book; decisions weekly, daily windows read at each week's last trading day); 52 weeks is T03. Sections 2 and 3 describe the z-score itself for that window.</p>",
         "<h2>1 · Outperformance against 50/50</h2><div class='card'>" + win_divs({w: E1.tbl(["pair", "window", "CAGR", "50/50 bought once", "excess a year", "information ratio", "years ahead", "worst fall of strategy ÷ 50/50"], rows_) for w, rows_ in o_rows_w.items()}) + "</div>",
         per_pair("op", figs["op"], tab_op, first=True),
         "<h2>2 · What follows a touch of the band (signal strength)</h2><p class='note'>Every first touch, traded or not. Blue: the signalled leg's return against the other, on average, k weeks later. Grey: the same on random walks. The rule makes money only if the blue line runs above the grey band. <b>The z window dropdown at the top</b> switches sections 2 and 3 between the 52-week band (T03's) and shorter weekly and daily windows; daily windows use daily closes and daily random walks.</p>",
         per_win("ev_pool", figs["ev_pool"]),
         per_pair("ev", figs["ev"], tab_ev),
         "<h2>3 · Does the z-score rank future relative returns? (information coefficient)</h2><p class='note'>Every week after the warm-up: the rank correlation of −z with the next k weeks' relative return. The weeks overlap, so the random walks (with the same overlap) are the yardstick, not a textbook t-statistic.</p>",
         per_pair("ic", figs["ic"], tab_ic),
         per_win("h_ic", figs["h_ic"]),
         "<div class='card'><p class='note' style='margin:4px 0'>Every window at a glance (pooled six index pairs; random walks' median, percentile):</p>" + E1.tbl(["z window", "touches", "after ≈13 weeks", "after ≈26 weeks", "IC ≈4 weeks", "IC ≈26 weeks", "IC percentiles notable"],
             [[f"{lab}{' (daily)' if f == 'D' else ''}"] + (lambda rr_: [int(rr_[0]["touches"])] + [f"{x['real'] * 100:+.2f}% ({x['rw_median'] * 100:+.2f}%, {E1.fpct(x['pct'])})" for x in rr_[:2]] + [f"{x['ic']:+.3f} ({x['rw_median']:+.3f}, {E1.fpct(x['pct'])})" for x in rr_[2:]])(
                 [EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == (13 if f == "W" else 65))].iloc[0], EVW[(EVW["window"] == w) & (EVW["pair"] == "index") & (EVW["k"] == (26 if f == "W" else 130))].iloc[0],
                  ICW[(ICW["window"] == w) & (ICW["pair"] == "index") & (ICW["h"] == (4 if f == "W" else 20))].iloc[0], ICW[(ICW["window"] == w) & (ICW["pair"] == "index") & (ICW["h"] == (26 if f == "W" else 130))].iloc[0]]) + [f"{win_notable[w][0]} of {win_notable[w][1]}"]
              for w, lab, f, _ in ZW.WINDOWS]) + "</div>",
         "<h2>4 · The rule on random walks: signal or mechanics?</h2><p class='note'>An exit at a <i>trailing</i> mean wins often on any series, because the mean also moves towards the ratio. So a high hit rate means something only if it beats what the same rule gives on random walks. Trades after the warm-up, relative return decision close to decision close (no cost).</p>",
         f"<div class='card per-win' data-titles='{html.escape(json.dumps(titles['h_rr']), quote=True)}'>" + E1.html_fig(figs["h_rr"]) + "<details><summary>Table view</summary>"
         + win_divs({w: E1.tbl(["pair"] + [lab_ for _, lab_, _ in STATS], [[("six index pairs" if p == "index" else nice(p))] + [f"{rr_txt(p, s_, w)} ({E1.fpct(rr_cell(p, s_, w)['pct'])})" for s_, _, _ in STATS] for p in keys]) for w, _, _, _ in ZW.WINDOWS})
         + "<p class='note'>The real value (percentile among the random walks).</p></details></div>",
         "<h2>5 · Where the shortfall comes from</h2><p class='note'>The strategy's value against 50/50 changes only through what the sleeve holds, so it splits exactly into one piece per holding period: each trade, and each neutral stretch between trades. Orange = held over 52 weeks.</p>",
         "<div class='card'>" + win_divs({w: E1.tbl(["pair", "window", "strategy ÷ 50/50, whole window", "trades held over 52 weeks", "trades held 52 weeks or less", "neutral stretches", "the 3 worst trades (weeks held)", "correlation of weeks held with the trade's piece"],
                                                    [[nice(r_["pair"]), f"{r_['first']} → {r_['last']}", f"{pct_x(r_['total']):+.1f}%", (f"{r_['long_n']} trade{'s' if r_['long_n'] != 1 else ''}: {pct_x(r_['long_sum']):+.1f}%" if r_['long_n'] else "none"), f"{r_['short_n']} trade{'s' if r_['short_n'] != 1 else ''}: {pct_x(r_['short_sum']):+.1f}%", f"{pct_x(r_['neutral_sum']):+.1f}%", f"{pct_x(r_['worst3_sum']):+.1f}% ({r_['worst3_weeks']})", f"{r_['corr_weeks']:+.2f}"]
                                                     for _, r_ in SCW[SCW["window"] == w].iterrows()]) for w, _, _, _ in ZW.WINDOWS}) + "<p class='note'>The pieces multiply exactly to the whole-window figure. Hindsight split: which trades will run long is not known at entry.</p></div>",
         per_pair("sf", figs["sf"], tab_sf),
         per_win("sf_all", figs["sf_all"]),
         "<h2>Method and limits</h2><div class='card'><ul>"
         f"<li><b>Paths.</b> E01's random walks: each pair's real first {E1.BURN + 1} weeks, then the point-in-time weekly residuals with E01's random signs (seed default_rng([{E1.SEED_SIGNS}, pair_id])) plus the expanding drift; z = (ratio − trailing 52-week mean) ÷ trailing sd on every path (row 0 equals E01's z to {checks['z_max_abs_diff']:.0e}).</li>"
         f"<li><b>The rule copy.</b> Vectorised over the paths; on the real ratio it reproduces all {checks['rule_trades_matched']} of T03's entries (date and leg) exactly, so the random walks run the identical rule. Relative return of a trade = ln(r_exit ÷ r_entry) for long a, the negative for long b, decision close to decision close; against T03's fill-based extras the correlation is " + ", ".join(f"{R[p]['corr_fill']:.2f}" for p in INDEX_PAIRS) + " on the index pairs.</li>"
         "<li><b>The windows' rules.</b> analysis/ratio_engine.py's book, comparators and trade_extras, run on the rule's weekly decisions for each window (explore/E01_ratio_cycles/variants.py); the 52-week run equals T03's decisions and NAV on every pair (checked each run), and on the real ratio the vectorised rule reproduces every window's entries.</li>"
         "<li><b>Outperformance.</b> the daily NAV against 50/50 bought once; information ratio = annualised mean ÷ annualised sd of the daily excess log return; calendar years with the first year from the start date.</li>"
         "<li><b>Limits.</b> Everything here is after the fact; the paths keep each ratio's volatility and drift and remove any swing-back; gold pairs from 2015 with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); Micro250 pairs from 2022 (few touches). The design was written before the first run (explore/E04_signal_strength/E04.md).</li></ul></div>",
         "</main>", "<script>" + E1.JS.replace("__DIV_LIGHT__", json.dumps(E1.DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(E1.DIV_DARK)).replace("e01-pair", "e04-pair") + "</script></body></html>"]
    out = os.path.join(OUT, "E04_signal.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable {n_not} of {len(counted)} → {out}")


if __name__ == "__main__":
    main()
