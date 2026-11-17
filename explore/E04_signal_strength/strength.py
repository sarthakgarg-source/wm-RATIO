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

TEST, RUN, WIN, EXIT, COST = "T03", "headline", 52, "E1", 20.8
EV = os.path.join(ROOT, "tests", TEST, "evidence", RUN)
K_MAX = 52; IC_H = (1, 4, 13, 26, 52); EVENT_SHOW = (4, 13, 26, 52)
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


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); D = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily()
    audit_t = json.load(open(os.path.join(EV, "audit.json"), encoding="utf-8"))
    decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]); decs = decs[(decs["window"] == WIN) & (decs["exit"] == EXIT)]
    extras = pd.read_csv(os.path.join(EV, "trade_extras.csv"), parse_dates=["entry_decision"]); extras = extras[(extras["window"] == WIN) & (extras["exit"] == EXIT)]
    summ = pd.read_csv(os.path.join(EV, "summary.csv"))
    trs = pd.read_csv(os.path.join(EV, "trades.csv"), parse_dates=["decision_date", "trade_date"]); trs = trs[(trs["window"] == WIN) & (trs["exit"] == EXIT)]
    R = {}; checks = {"z_max_abs_diff": 0.0, "rule_mismatch": [], "rule_trades_matched": 0}
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
        R[pair] = {"pid": pid, "dates": dates, "first": dates[E1.BURN].date().isoformat(), "last": dates[-1].date().isoformat(), "rule": stats_rows, "trade_rows": keep, "trade_ret": ret, "trade_wks": wks,
                   "corr_fill": corr_fill, "n_fill_matched": len(ex), "ev_sum": ev_sum, "ev_cnt": cnt, "ic": ic}
        print(f"  {pair:18s} touches {int(cnt[0]):3d}  trades {stats_rows[0]['n']:3d}  [{time.time() - t0:4.0f}s]", flush=True)
    if checks["z_max_abs_diff"] > 1e-9 or checks["rule_mismatch"]:
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
    counted = ICS["pct"].tolist() + RRS["pct"].tolist(); counted = [v for v in counted if v == v]; n_not = sum(E1.notable(v) for v in counted)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "test": TEST, "test_run_date": audit_t.get("run_date"), "setting": {"window": WIN, "exit": EXIT, "cost_bps": COST},
             "draws": D, "seed": f"default_rng([{E1.SEED_SIGNS}, pair_id]) (E01's signs)", "checks": checks, "notable": [n_not, len(counted)],
             "fill_vs_close": {p: {"corr": R[p]["corr_fill"], "trades": R[p]["n_fill_matched"]} for p in ORDER}, "seconds": round(time.time() - t0)}
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
    md += ["\n## 4. The T03 rule on the real ratio and on the random walks (trades after the warm-up; relative return close to close)\n", "| pair | " + " | ".join(lab for _, lab, _ in STATS) + " |", "|---|" + "---|" * len(STATS)]
    for p in ["index"] + ORDER:
        cells = []
        for s, lab, kind in STATS:
            r_ = g(RRS, p, stat=s); f_ = (lambda v: fmt("pct", v)) if s in ("mean", "median") else ((lambda v: f"{v * 100:.0f}%") if s in ("hit", "long") else (lambda v: f"{v:.0f}" if s == "n" else f"{v:.1f}"))
            cells.append(f"{f_(r_['real'])} ({f_(r_['rw_median'])}, {pq(r_['pct'])})")
        md.append(f"| {'six index pairs' if p == 'index' else nice(p)} | " + " | ".join(cells) + " |")
    md += ["\n_The real trades recomputed close to close against T03's fill-based extras: correlation " + ", ".join(f"{nice(p)} {R[p]['corr_fill']:.3f} ({R[p]['n_fill_matched']} trades)" for p in ORDER) + "._"]
    md += ["\n## 5. Where the shortfall against 50/50 comes from (change of strategy ÷ 50/50 during each holding period)\n",
           "The pieces multiply exactly to the whole: trades held over 52 weeks, trades held 52 weeks or less, and the neutral stretches between trades.\n",
           "| pair | window | strategy ÷ 50/50, whole window | trades held over 52 weeks | trades held 52 weeks or less | neutral stretches | the 3 worst trades (weeks held) | correlation of weeks held with the trade's piece |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {nice(r['pair'])} | {r['first']} → {r['last']} | {pct_x(r['total']):+.1f}% | {r['long_n']} trade{'s' if r['long_n'] != 1 else ''}: {pct_x(r['long_sum']):+.1f}% | {r['short_n']} trade{'s' if r['short_n'] != 1 else ''}: {pct_x(r['short_sum']):+.1f}% | {pct_x(r['neutral_sum']):+.1f}% | {pct_x(r['worst3_sum']):+.1f}% ({r['worst3_weeks']}) | {r['corr_weeks']:+.2f} |" for _, r in SC.iterrows()]
    open(os.path.join(OUT, "E04_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- figures
    figs = {}; titles = {"op": {}, "ev": {}, "ic": {}, "sf": {}}
    fO = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], vertical_spacing=0.1, subplot_titles=("strategy ÷ 50/50 bought once (above 1 = ahead)", "calendar-year return: strategy − 50/50"))
    fE, fI = go.Figure(), go.Figure()
    for p in ORDER:
        vis = p == DEFAULT_PAIR; rel = R[p]["rel"]; ye = YE[YE["pair"] == p]; o = OPF[OPF["pair"] == p].iloc[0]
        fO.add_trace(go.Scatter(x=rel.index, y=rel, mode="lines", line={"color": BLUE, "width": 1.5}, name="strategy ÷ 50/50", meta=p, visible=vis, hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"), row=1, col=1)
        fO.add_trace(go.Bar(x=ye["year"], y=ye["excess"] * 100, marker={"color": [AQUA if v > 0 else ORANGE for v in ye["excess"]]}, name="excess in the year", meta=p, visible=vis,
                            customdata=np.c_[ye["strategy"] * 100, ye["5050"] * 100], hovertemplate="%{x}: strategy %{customdata[0]:+.1f}%, 50/50 %{customdata[1]:+.1f}% → %{y:+.1f} pp<extra></extra>"), row=2, col=1)
        titles["op"][p] = E1.two_line(f"{nice(p)} · {o['first']} → {o['last']} · CAGR {o['cagr'] * 100:.1f}% vs 50/50 {o['cagr_5050'] * 100:.1f}% ({o['excess_cagr'] * 100:+.1f} pp a year) · information ratio {o['ir']:+.2f}",
                                      f"ahead in {o['years_ahead'] * 100:.0f}% of {o['n_years']} calendar years; the strategy ÷ 50/50 fell at worst {o['rel_maxdd'] * 100:.0f}% and ends at {o['rel_end']:.2f}")
        real, lo, md_, hi, pc = EV_[p]; ks = np.arange(1, K_MAX + 1)
        fE.add_trace(go.Scatter(x=ks, y=lo * 100, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=p, visible=vis))
        fE.add_trace(go.Scatter(x=ks, y=hi * 100, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=p, visible=vis))
        fE.add_trace(go.Scatter(x=ks, y=real * 100, mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 4}, name="the ratio's touches", meta=p, visible=vis,
                                customdata=np.c_[md_ * 100, pc], hovertemplate="%{x} weeks after: %{y:+.2f}% (random walks %{customdata[0]:+.2f}%, percentile %{customdata[1]:.1f})<extra></extra>"))
        titles["ev"][p] = E1.two_line(f"{nice(p)} · {R[p]['first']} → {R[p]['last']} · {int(R[p]['ev_cnt'][0])} first touches of the band",
                                      "mean relative return of the signalled leg against the other, k weeks after the touch (above 0 = the swing back the rule bets on)")
        icp = ICS[ICS["pair"] == p]
        fI.add_trace(go.Scatter(x=[str(k) for k in IC_H], y=icp["rw_p5"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False, meta=p, visible=vis))
        fI.add_trace(go.Scatter(x=[str(k) for k in IC_H], y=icp["rw_p95"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip", meta=p, visible=vis))
        fI.add_trace(go.Scatter(x=[str(k) for k in IC_H], y=icp["ic"], mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 8}, name="the ratio", meta=p, visible=vis,
                                customdata=np.c_[icp["pct"], icp["ic_first_half"], icp["ic_second_half"]], hovertemplate="%{x} weeks: IC %{y:+.3f}, percentile %{customdata[0]:.1f}<br>first half %{customdata[1]:+.3f}, second half %{customdata[2]:+.3f}<extra></extra>"))
        titles["ic"][p] = E1.two_line(f"{nice(p)} · {R[p]['first']} → {R[p]['last']} · rank correlation of −z with the next k weeks' relative return",
                                      "above 0 = a stretched ratio tends to come back (a below-band ratio is followed by the first leg doing better); grey = what random walks give")
    fO.add_hline(y=1, line={"color": AXIS, "width": 1}, row=1, col=1); fO.add_hline(y=0, line={"color": AXIS, "width": 1}, row=2, col=1)
    fO.update_yaxes(title={"text": "relative NAV", "font": {"color": INK2}}, row=1, col=1); fO.update_yaxes(ticksuffix=" pp", row=2, col=1)
    figs["op"] = E1.style(fO, titles["op"][DEFAULT_PAIR], 640, legend_top=True)
    fE.add_hline(y=0, line={"color": AXIS, "width": 1}); fE.update_xaxes(title={"text": "weeks after the first touch", "font": {"color": INK2}}); fE.update_yaxes(ticksuffix="%", title={"text": "signalled leg − the other", "font": {"color": INK2}})
    figs["ev"] = E1.style(fE, titles["ev"][DEFAULT_PAIR], 460, legend_top=True)
    fI.add_hline(y=0, line={"color": AXIS, "width": 1}); fI.update_xaxes(type="category", title={"text": "horizon, weeks", "font": {"color": INK2}}); fI.update_yaxes(title={"text": "information coefficient", "font": {"color": INK2}})
    figs["ic"] = E1.style(fI, titles["ic"][DEFAULT_PAIR], 440, legend_top=True)
    # pooled event study (static)
    real, lo, md_, hi, pc = EV_["index"]; ks = np.arange(1, K_MAX + 1); nt = int(sum(R[p]["ev_cnt"][0] for p in INDEX_PAIRS))
    f = go.Figure([go.Scatter(x=ks, y=lo * 100, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False),
                   go.Scatter(x=ks, y=hi * 100, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=NULL_FILL, name="random walks, middle 90%", hoverinfo="skip"),
                   go.Scatter(x=ks, y=real * 100, mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 4}, name="the six index pairs' touches", customdata=np.c_[md_ * 100, pc],
                              hovertemplate="%{x} weeks after: %{y:+.2f}% (random walks %{customdata[0]:+.2f}%, percentile %{customdata[1]:.1f})<extra></extra>")])
    f.add_hline(y=0, line={"color": AXIS, "width": 1}); f.update_xaxes(title={"text": "weeks after the first touch", "font": {"color": INK2}}); f.update_yaxes(ticksuffix="%", title={"text": "signalled leg − the other", "font": {"color": INK2}})
    figs["ev_pool"] = E1.style(f, E1.two_line(f"The six index pairs together · {nt} first touches of the band · 2007–2026 (Micro250 pairs from 2023)", "mean relative return of the signalled leg, k weeks after the touch, against the same on random walks"), 440, legend_top=True)
    # heat maps
    def heat(zv, xl, yl, txt, hov, title, h):
        f = go.Figure(go.Heatmap(z=zv, x=xl, y=yl, zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=txt, texttemplate="%{text}", customdata=hov, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
                                 colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="top")
        return E1.style(f, title, h, legend=False, left=150)
    rows_lab = ["six index pairs"] + [nice(p) for p in ORDER]; keys = ["index"] + ORDER
    figs["h_ic"] = heat([[g(ICS, p, k=k)["pct"] for k in IC_H] for p in keys], [f"{k} wk" for k in IC_H], rows_lab,
                        [[(f"{g(ICS, p, k=k)['ic']:+.2f}" if E1.notable(g(ICS, p, k=k)["pct"]) else "") for k in IC_H] for p in keys],
                        [[f"IC {g(ICS, p, k=k)['ic']:+.3f} (random walks {g(ICS, p, k=k)['rw_median']:+.3f}), percentile {g(ICS, p, k=k)['pct']:.1f}" for k in IC_H] for p in keys],
                        "Information coefficient against random walks: red = the z-score ranks future relative returns better than chance (labels on notable cells)", 520)
    def rr_txt(p, s):
        r_ = g(RRS, p, stat=s); v = r_["real"]
        return (f"{v * 100:+.1f}%" if s in ("mean", "median") else (f"{v * 100:.0f}%" if s in ("hit", "long") else f"{v:.0f}" if s == "n" else f"{v:.1f}"))
    # 5. per pair: each trade's piece of the shortfall at its entry date, coloured by how long it was held
    fS = go.Figure(); BUCKET = (("up to 13 weeks", 0, 13, AQUA), ("14–52 weeks", 14, 52, BLUE), ("over 52 weeks", 53, 10 ** 6, ORANGE))
    for p in ORDER:
        vis = p == DEFAULT_PAIR; tt = SEG[(SEG["pair"] == p) & (SEG["kind"] == "trade")]; sc_ = SC[SC["pair"] == p].iloc[0]
        for lab, lo_, hi_, colr in BUCKET:
            b_ = tt[(tt["weeks"] >= lo_) & (tt["weeks"] <= hi_)]
            fS.add_trace(go.Bar(x=pd.to_datetime(b_["start"]), y=pct_x(b_["log_contrib"]), width=40 * 86400000, marker={"color": colr}, name=f"held {lab}", legendgroup=lab, meta=p, visible=vis,
                                text=[f"{w}w" for w in b_["weeks"]], textposition="outside", textfont={"size": 9}, constraintext="none", cliponaxis=False,
                                customdata=np.c_[b_["start"], b_["end"], b_["weeks"], ["still open" if o else "" for o in b_["open"]]],
                                hovertemplate="%{customdata[0]} → %{customdata[1]} %{customdata[3]}: %{customdata[2]} weeks held<br>strategy ÷ 50/50 changed %{y:+.1f}%<extra></extra>"))
        titles["sf"][p] = E1.two_line(f"{nice(p)} · {sc_['first']} → {sc_['last']} · strategy ÷ 50/50 {pct_x(sc_['total']):+.1f}% over the window · the {sc_['long_n']} trades held over 52 weeks: {pct_x(sc_['long_sum']):+.1f}%",
                                      f"the other {sc_['short_n']} trades together {pct_x(sc_['short_sum']):+.1f}%, the neutral stretches {pct_x(sc_['neutral_sum']):+.1f}%; each bar = one trade at its entry date, its height the change of strategy ÷ 50/50 while it was held, its label the weeks held")
    fS.add_hline(y=0, line={"color": AXIS, "width": 1}); fS.update_yaxes(ticksuffix="%", title={"text": "change of strategy ÷ 50/50 during the trade", "font": {"color": INK2}})
    fS.update_layout(barmode="overlay"); figs["sf"] = E1.style(fS, titles["sf"][DEFAULT_PAIR], 480, legend_top=True)
    ti = SEG[(SEG["kind"] == "trade") & SEG["pair"].isin(INDEX_PAIRS)]
    f = go.Figure([go.Scatter(x=ti["weeks"], y=pct_x(ti["log_contrib"]), mode="markers", marker={"size": 8, "color": [ORANGE if w > 52 else (BLUE if w > 13 else AQUA) for w in ti["weeks"]], "opacity": 0.8},
                              customdata=np.c_[[nice(q) for q in ti["pair"]], ti["start"], ti["end"]], hovertemplate="%{customdata[0]} · %{customdata[1]} → %{customdata[2]}<br>%{x} weeks held · strategy ÷ 50/50 %{y:+.1f}%<extra></extra>", showlegend=False)])
    f.add_hline(y=0, line={"color": AXIS, "width": 1}); f.add_vline(x=52, line={"color": AXIS, "width": 1, "dash": "dot"})
    f.update_xaxes(title={"text": "weeks held", "font": {"color": INK2}}); f.update_yaxes(ticksuffix="%", title={"text": "change of strategy ÷ 50/50 during the trade", "font": {"color": INK2}})
    ix = SC[SC["pair"].isin(INDEX_PAIRS)]
    figs["sf_all"] = E1.style(f, E1.two_line(f"Every trade on the six index pairs: how long it was held against what it did to strategy ÷ 50/50 · {len(ti)} trades",
                                             f"the {int(ix['long_n'].sum())} trades held over a year (right of the dotted line) add up, in log terms across the six pairs, to {pct_x(ix['long_sum'].sum()):+.1f}%; the other {int(ix['short_n'].sum())} to {pct_x(ix['short_sum'].sum()):+.1f}%"), 460, legend=False)
    figs["h_rr"] = heat([[g(RRS, p, stat=s)["pct"] for s, _, _ in STATS] for p in keys], [lab for _, lab, _ in STATS], rows_lab,
                        [[rr_txt(p, s) for s, _, _ in STATS] for p in keys],
                        [[f"real {rr_txt(p, s)}, random walks' median {g(RRS, p, stat=s)['rw_median'] * (100 if s in ('hit', 'long', 'mean', 'median') else 1):.1f}{'%' if s in ('hit', 'long', 'mean', 'median') else ''}, percentile {g(RRS, p, stat=s)['pct']:.1f}" for s, _, _ in STATS] for p in keys],
                        "The T03 rule's trades against the same rule on random walks: grey = what a series with no swing-back gives (labels: the real value)", 520)

    # ---------------------------------------------------------------- the page
    def per_pair(key, fig, tables, first=False) -> str:
        t = html.escape(json.dumps(titles[key]), quote=True)
        tb = "".join(f"<div data-pair='{html.escape(p, quote=True)}'{'' if p == DEFAULT_PAIR else ' hidden'}>{tables[p]}</div>" for p in ORDER)
        return f"<div class='card per-pair' data-titles='{t}'>{E1.html_fig(fig, first=first)}<details><summary>Table view</summary>{tb}</details></div>"
    tab_sf = {p: E1.tbl(["entered", "exited", "leg", "weeks held", "strategy ÷ 50/50 during the trade"], [[r_["start"], r_["end"] + (" (open)" if r_["open"] else ""), r_["state"], int(r_["weeks"]), f"{pct_x(r_['log_contrib']):+.1f}%"]
                                                                                                      for _, r_ in SEG[(SEG["pair"] == p) & (SEG["kind"] == "trade")].sort_values("log_contrib").iterrows()]) for p in ORDER}
    tab_op = {p: E1.tbl(["year", "strategy", "50/50", "excess"], [[int(r_["year"]), f"{r_['strategy'] * 100:+.1f}%", f"{r_['5050'] * 100:+.1f}%", f"{r_['excess'] * 100:+.1f} pp"] for _, r_ in YE[YE["pair"] == p].iterrows()]) for p in ORDER}
    tab_ev = {p: E1.tbl(["weeks after", "the ratio", "random walks' median", "5th–95th", "percentile"], [[int(r_["k"]), f"{r_['real'] * 100:+.2f}%", f"{r_['rw_median'] * 100:+.2f}%", f"{r_['rw_p5'] * 100:+.2f}% … {r_['rw_p95'] * 100:+.2f}%", E1.fpct(r_["pct"])] for _, r_ in EVS[EVS["pair"] == p].iterrows() if int(r_["k"]) in (1, 2, 4, 8, 13, 26, 39, 52)]) for p in ORDER}
    tab_ic = {p: E1.tbl(["horizon", "IC", "random walks 5th–95th", "percentile", "first half", "second half"], [[f"{int(r_['k'])} wk", f"{r_['ic']:+.3f}", f"{r_['rw_p5']:+.3f} … {r_['rw_p95']:+.3f}", E1.fpct(r_["pct"]), f"{r_['ic_first_half']:+.3f}", f"{r_['ic_second_half']:+.3f}"] for _, r_ in ICS[ICS["pair"] == p].iterrows()]) for p in ORDER}
    o_rows = [[nice(r_["pair"]), f"{r_['first']} → {r_['last']}", f"{r_['cagr'] * 100:.1f}%", f"{r_['cagr_5050'] * 100:.1f}%", f"{r_['excess_cagr'] * 100:+.1f} pp", f"{r_['ir']:+.2f}", f"{r_['years_ahead'] * 100:.0f}% of {r_['n_years']}", f"{r_['rel_maxdd'] * 100:.0f}%"] for _, r_ in OPF.iterrows()]
    ixp = OPF[OPF["pair"].isin(INDEX_PAIRS)]; ev26 = g(EVS, "index", k=26); ev13 = g(EVS, "index", k=13); ic26 = g(ICS, "index", k=26); hit = g(RRS, "index", stat="hit"); mean_ = g(RRS, "index", stat="mean"); lng = g(RRS, "index", stat="long")
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS).replace("<title>Ratio cycles</title>", "<title>Signal strength</title>"),
         "<header class='bar'><div class='bar-inner'><b>E04 · signal strength</b><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label><button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Does the ratio band have any predictive power?</h1>",
         f"<p>Four looks at T03 — the ±1 sd first-touch switch on the plain ratio, exit at the trailing mean, no maximum hold (window {WIN}, exit {EXIT}, {COST} bps), refuted against random timing. "
         f"<b>1.</b> Its outperformance against 50/50 over time. <b>2.</b> What follows a touch of the band. <b>3.</b> Whether the z-score ranks future relative returns at all. <b>4.</b> The same rule run on {D:,} random walks per pair (E01's: each ratio's own week-by-week volatility and point-in-time drift, no swing-back), to see whether its trade statistics are signal or mechanics. "
         f"Generated {audit['run_date']} from the approved data of 2026-11-12 and T03's evidence (run {audit['test_run_date']}). <b>Description, not a finding.</b></p>",
         f"<div class='card'><p style='margin:4px 0'><b>In one paragraph.</b> On the six index pairs the strategy's CAGR minus 50/50's is {ixp['excess_cagr'].min() * 100:+.1f} to {ixp['excess_cagr'].max() * 100:+.1f} pp a year, with information ratios of {ixp['ir'].min():+.2f} to {ixp['ir'].max():+.2f}. "
         f"After a first touch of the band the signalled leg does {ev13['real'] * 100:+.2f}% against the other over 13 weeks and {ev26['real'] * 100:+.2f}% over 26 (random walks: {ev13['rw_median'] * 100:+.2f}% and {ev26['rw_median'] * 100:+.2f}%; percentiles {ev13['pct']:.0f} and {ev26['pct']:.0f}). "
         f"The z-score's rank correlation with the next 26 weeks' relative return averages {ic26['ic']:+.3f} over the index pairs (random walks {ic26['rw_median']:+.3f}, percentile {ic26['pct']:.0f}). "
         f"The rule's trades win {hit['real'] * 100:.0f}% of the time — random walks give {hit['rw_median'] * 100:.0f}% with the same rule (percentile {hit['pct']:.0f}) — and average {mean_['real'] * 100:+.2f}% (random walks {mean_['rw_median'] * 100:+.2f}%, percentile {mean_['pct']:.0f}); {lng['real'] * 100:.0f}% of trades are held over a year (random walks {lng['rw_median'] * 100:.0f}%). "
         f"<b>Against chance:</b> {n_not} of {len(counted)} counted percentiles are notable (about {0.1 * len(counted):.0f} by chance).</p></div>",
         "<h2>1 · Outperformance against 50/50</h2><div class='card'>" + E1.tbl(["pair", "window", "CAGR", "50/50 bought once", "excess a year", "information ratio", "years ahead", "worst fall of strategy ÷ 50/50"], o_rows) + "</div>",
         per_pair("op", figs["op"], tab_op, first=True),
         "<h2>2 · What follows a touch of the band (signal strength)</h2><p class='note'>Every first touch, traded or not. Blue: the signalled leg's return against the other, on average, k weeks later. Grey: the same on random walks. The rule makes money only if the blue line runs above the grey band.</p>",
         "<div class='card'>" + E1.html_fig(figs["ev_pool"]) + "</div>",
         per_pair("ev", figs["ev"], tab_ev),
         "<h2>3 · Does the z-score rank future relative returns? (information coefficient)</h2><p class='note'>Every week after the warm-up: the rank correlation of −z with the next k weeks' relative return. The weeks overlap, so the random walks (with the same overlap) are the yardstick, not a textbook t-statistic.</p>",
         per_pair("ic", figs["ic"], tab_ic),
         "<div class='card'>" + E1.html_fig(figs["h_ic"]) + "</div>",
         "<h2>4 · The rule on random walks: signal or mechanics?</h2><p class='note'>An exit at a <i>trailing</i> mean wins often on any series, because the mean also moves towards the ratio. So a high hit rate means something only if it beats what the same rule gives on random walks. Trades after the warm-up, relative return decision close to decision close (no cost).</p>",
         "<div class='card'>" + E1.html_fig(figs["h_rr"]) + "<details><summary>Table view</summary>" + E1.tbl(["pair"] + [lab for _, lab, _ in STATS], [[("six index pairs" if p == "index" else nice(p))] + [f"{rr_txt(p, s)} ({E1.fpct(g(RRS, p, stat=s)['pct'])})" for s, _, _ in STATS] for p in keys]) + "<p class='note'>The real value (percentile among the random walks).</p></details></div>",
         "<h2>5 · Where the shortfall comes from</h2><p class='note'>The strategy's value against 50/50 changes only through what the sleeve holds, so it splits exactly into one piece per holding period: each trade, and each neutral stretch between trades. Orange = held over 52 weeks.</p>",
         "<div class='card'>" + E1.tbl(["pair", "window", "strategy ÷ 50/50, whole window", "trades held over 52 weeks", "trades held 52 weeks or less", "neutral stretches", "the 3 worst trades (weeks held)", "correlation of weeks held with the trade's piece"],
                                     [[nice(r_["pair"]), f"{r_['first']} → {r_['last']}", f"{pct_x(r_['total']):+.1f}%", f"{r_['long_n']} trade{'s' if r_['long_n'] != 1 else ''}: {pct_x(r_['long_sum']):+.1f}%", f"{r_['short_n']} trade{'s' if r_['short_n'] != 1 else ''}: {pct_x(r_['short_sum']):+.1f}%", f"{pct_x(r_['neutral_sum']):+.1f}%", f"{pct_x(r_['worst3_sum']):+.1f}% ({r_['worst3_weeks']})", f"{r_['corr_weeks']:+.2f}"] for _, r_ in SC.iterrows()]) + "<p class='note'>The pieces multiply exactly to the whole-window figure. Hindsight split: which trades will run long is not known at entry.</p></div>",
         per_pair("sf", figs["sf"], tab_sf),
         "<div class='card'>" + E1.html_fig(figs["sf_all"]) + "</div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         f"<li><b>Paths.</b> E01's random walks: each pair's real first {E1.BURN + 1} weeks, then the point-in-time weekly residuals with E01's random signs (seed default_rng([{E1.SEED_SIGNS}, pair_id])) plus the expanding drift; z = (ratio − trailing 52-week mean) ÷ trailing sd on every path (row 0 equals E01's z to {checks['z_max_abs_diff']:.0e}).</li>"
         f"<li><b>The rule copy.</b> Vectorised over the paths; on the real ratio it reproduces all {checks['rule_trades_matched']} of T03's entries (date and leg) exactly, so the random walks run the identical rule. Relative return of a trade = ln(r_exit ÷ r_entry) for long a, the negative for long b, decision close to decision close; against T03's fill-based extras the correlation is " + ", ".join(f"{R[p]['corr_fill']:.2f}" for p in INDEX_PAIRS) + " on the index pairs.</li>"
         "<li><b>Outperformance.</b> T03's daily NAV and 50/50 bought once, from its evidence; information ratio = annualised mean ÷ annualised sd of the daily excess log return; calendar years with the first year from the start date.</li>"
         "<li><b>Limits.</b> Everything here is after the fact; the paths keep each ratio's volatility and drift and remove any swing-back; gold pairs from 2015 with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); Micro250 pairs from 2022 (few touches). The design was written before the first run (explore/E04_signal_strength/E04.md).</li></ul></div>",
         "</main>", "<script>" + E1.JS.replace("__DIV_LIGHT__", json.dumps(E1.DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(E1.DIV_DARK)).replace("e01-pair", "e04-pair") + "</script></body></html>"]
    out = os.path.join(OUT, "E04_signal.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable {n_not} of {len(counted)} → {out}")


if __name__ == "__main__":
    main()
