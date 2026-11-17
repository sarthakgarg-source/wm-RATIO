"""
E03 (EXPLORATORY): when does the ratio switch work? Design: explore/E03_trade_conditions/E03.md (written before the first run).
A. Forward returns of the cheaper and the richer leg by the ratio's z-score bucket, every week, against E01's random walks.
B. The market's condition at the entry of every T01 / T02 trade (window 52, exit E1, 20.8 bps) and during it, related to
   the trade's outcome (the per-trade extra), with permutation nulls.

    python explore/E03_trade_conditions/conditions.py [--draws 2000] [--perms 2000]

Run only after T02's run finished (both tests locked and run). Not a finding and not a test of any rule.
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles"))
from analysis.approved_data import GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (E01's z-score, random paths, percentiles and the page's styling)

HORIZONS = (4, 13, 26)
BUCKETS = (("< −2", -np.inf, -2.0), ("−2 to −1", -2.0, -1.0), ("−1 to 0", -1.0, 0.0), ("0 to 1", 0.0, 1.0), ("1 to 2", 1.0, 2.0), ("> 2", 2.0, np.inf), ("all weeks", -np.inf, np.inf))
CHART_BUCKETS = [b[0] for b in BUCKETS[:-1]]
TESTS = ("T02", "T01"); RUN = "headline"; WIN, EXIT = 52, "E1"
MARKET = "N100"; SEED_PERM = 105; MIN_DURING = 4; MIN_TRADES = 10
ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR; LO, HI = E1.LO, E1.HI
POOLS = (("T02", "index"), ("T01", "index"), ("T02", "gold"), ("T01", "gold"))
CONDS = (  # key, label, when, format
    ("corr26", "legs' correlation, 26 wk", "entry", "num"), ("corr52", "legs' correlation, 52 wk", "entry", "num"),
    ("mkt52", "N100 52-week return", "entry", "pct"), ("mkt_dd", "N100 drawdown from its high", "entry", "pct"),
    ("vol_held", "held leg's volatility, 26 wk", "entry", "pct"), ("vol_peer", "peer's volatility, 26 wk", "entry", "pct"),
    ("vol_ratio", "ratio's volatility, 26 wk", "entry", "pct"), ("band_width", "band width (2 sd ÷ mean)", "entry", "pct"),
    ("z_depth", "|z| at entry", "entry", "num"), ("since_zero", "weeks since z changed sign", "entry", "wk"),
    ("lag4", "held − peer, 4 wk before entry", "entry", "pct"), ("lag13", "held − peer, 13 wk before entry", "entry", "pct"),
    ("held13", "held leg's 13-week return before entry", "entry", "pct"), ("peer13", "peer's 13-week return before entry", "entry", "pct"),
    ("breadth", "share of the 10 ratios outside their band", "entry", "pct"),
    ("mkt_during", "N100 return during the trade", "during", "pct"), ("corr_during", "legs' correlation during the trade", "during", "num"),
    ("vol_ratio_during", "ratio's volatility during the trade", "during", "pct"),
)
KEY_CONDS = ("z_depth", "corr52", "mkt52", "mkt_dd", "band_width", "breadth")
BLUE, ORANGE, AQUA, MUTED, INK2, AXIS, NULL_FILL = E1.BLUE, E1.ORANGE, E1.AQUA, E1.MUTED, E1.INK2, E1.AXIS, E1.NULL_FILL


def fmtc(kind, v):
    """Signed version for differences (good − bad)."""
    if v != v:
        return "—"
    return {"pct": f"{v * 100:+.1f}%", "num": f"{v:+.2f}", "wk": f"{v:+.0f} wk"}[kind]


def fmtv(kind, v):
    """Unsigned version for levels (a correlation of 0.62, a volatility of 18.0%)."""
    if v != v:
        return "—"
    return {"pct": f"{v * 100:.1f}%", "num": f"{v:.2f}", "wk": f"{v:.0f}"}[kind]


# ------------------------------------------------------------------------------------------------------- Part A
def profiles(pair, pid, px, D):
    """Forward returns of the cheaper / richer leg by z bucket, on the real ratio (row 0) and E01's D random paths of it."""
    pf = pair_frame(px, pair, "plain"); wk = pf.loc[pf["decision"]]; dates = wk.index
    a, b = wk["ca"].to_numpy(), wk["cb"].to_numpy(); r = a / b; L = len(r); lnr = np.log(r)
    g = np.diff(lnr); t, m, sd, z, hx = E1.point_in_time(g); n = len(z)
    S = np.random.default_rng([E1.SEED_SIGNS, pid]).choice(np.array([-1.0, 1.0]), size=(D, n)); Z = S * z       # E01's signs exactly
    rnd = np.concatenate([np.tile(lnr[: E1.BURN + 1], (D, 1)), lnr[E1.BURN] + np.cumsum(Z + m, axis=1)], axis=1)
    paths = np.concatenate([lnr[None, :], rnd], axis=0); assert paths.shape == (D + 1, L)
    zp = E1.trailing_z(np.exp(paths)); zc = E1.band_z(pd.Series(r, index=dates)).to_numpy()
    chk = float(np.nanmax(np.abs(zp[0] - zc)))                                                                  # row 0 must be E01's z
    M = 0.5 * (np.log(a) + np.log(b)); lna = M[None, :] + 0.5 * paths; lnb = M[None, :] - 0.5 * paths            # the common movement kept real
    assert np.allclose(np.exp(lna[0]), a) and np.allclose(np.exp(lnb[0]), b)
    rows = []
    for k in HORIZONS:
        Ra = np.exp(lna[:, k:] - lna[:, :-k]) - 1; Rb = np.exp(lnb[:, k:] - lnb[:, :-k]) - 1; zk = zp[:, :-k]
        cheap = np.where(zk < 0, Ra, Rb); rich = np.where(zk < 0, Rb, Ra); spread = cheap - rich; both = 0.5 * (Ra + Rb)
        ok = np.zeros(L - k, bool); ok[E1.BURN:] = True
        for lab, lo, hi in BUCKETS:
            mask = ok[None, :] & (zk >= lo) & (zk < hi)
            row = {"pair": pair, "horizon": k, "bucket": lab, "weeks": int(mask[0].sum()),
                   "episodes": int((np.diff(np.concatenate([[0], mask[0].astype(np.int8), [0]])) == 1).sum())}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name, X in (("cheap", cheap), ("rich", rich), ("spread", spread), ("both", both)):
                    Y = np.where(mask, X, np.nan); mu = np.nanmean(Y, axis=1); md = np.nanmedian(Y, axis=1)
                    row[f"{name}_mean"] = float(mu[0]); row[f"{name}_median"] = float(md[0])
                    row[f"{name}_mean_rw"] = float(np.nanmedian(mu[1:])); row[f"{name}_median_rw"] = float(np.nanmedian(md[1:]))
                    if name == "spread":
                        row["spread_mean_lo"] = float(np.nanpercentile(mu[1:], 5)); row["spread_mean_hi"] = float(np.nanpercentile(mu[1:], 95))
                        row["spread_mean_pct"] = E1.pctile(mu[0], mu[1:]); row["spread_median_pct"] = E1.pctile(md[0], md[1:])
            rows.append(row)
    return rows, chk, dates[E1.BURN].date().isoformat(), dates[-1].date().isoformat()


# ------------------------------------------------------------------------------------------------------- Part B
def condition_series(pf):
    """Every condition that is a series, on the pair's decision weeks, from data up to each week only."""
    wk = pf.loc[pf["decision"], ["ca", "cb", "ratio"]].copy()
    ra, rb = np.log(wk["ca"]).diff(), np.log(wk["cb"]).diff(); lr = np.log(wk["ratio"]).diff()
    c = pd.DataFrame(index=wk.index)
    c["corr26"] = ra.rolling(26).corr(rb); c["corr52"] = ra.rolling(52).corr(rb)
    c["vol_a"] = ra.rolling(26).std() * np.sqrt(52); c["vol_b"] = rb.rolling(26).std() * np.sqrt(52); c["vol_ratio"] = lr.rolling(26).std() * np.sqrt(52)
    c["band_width"] = 2 * wk["ratio"].rolling(52).std() / wk["ratio"].rolling(52).mean()
    z = E1.band_z(wk["ratio"]); c["z"] = z
    sgn = np.sign(z.to_numpy()); since = np.full(len(sgn), np.nan); run = 0
    for i in range(len(sgn)):
        if np.isnan(sgn[i]):
            run = 0; continue
        run = run + 1 if (i > 0 and sgn[i] == sgn[i - 1]) else 1; since[i] = run
    c["since_zero"] = since
    for k in (4, 13):
        c[f"ret_a{k}"] = wk["ca"] / wk["ca"].shift(k) - 1; c[f"ret_b{k}"] = wk["cb"] / wk["cb"].shift(k) - 1
    return wk, c, ra, rb, lr


def cond_pit_check(pf, n_cuts=12):
    """The condition series recomputed from the history cut at 12 dates must equal the full series up to each cut."""
    wk, full, *_ = condition_series(pf); dec = pf.index[pf["decision"]]; checks, bad = 0, []
    for T in np.unique(np.linspace(80, len(dec) - 1, n_cuts).astype(int)):
        cut = pf.loc[: dec[T]]; _, c, *_ = condition_series(cut); checks += 1
        if not all(np.allclose(c[k].to_numpy(), full[k].to_numpy()[: len(c)], rtol=0, atol=1e-12, equal_nan=True) for k in c.columns):
            bad.append(f"conditions differ when the history is cut at {dec[T].date()}")
    return checks, bad


def load_trades(test):
    ev = os.path.join(ROOT, "tests", test, "evidence", RUN)
    x = pd.read_csv(os.path.join(ev, "trade_extras.csv"), parse_dates=["entry_decision", "entry_trade", "exit_trade"])
    x = x[(x["window"] == WIN) & (x["exit"] == EXIT)].copy()
    d = pd.read_csv(os.path.join(ev, "decisions.csv"), parse_dates=["date"]); d = d[(d["window"] == WIN) & (d["exit"] == EXIT)]
    audit = json.load(open(os.path.join(ev, "audit.json"), encoding="utf-8"))
    return x, d, audit


def trade_rows(test, px, series, zall, mkt):
    x, dec, _ = load_trades(test); rows = []; n_open = int(x["open"].sum())
    for _, tr in x[~x["open"]].iterrows():
        pair = tr["pair"]; wk, c, ra, rb, lr = series[pair]; ed, et, xt = tr["entry_decision"], tr["entry_trade"], tr["exit_trade"]
        A = tr["held"] == "A"; la, lb = pair.split("|"); held_leg, peer_leg = (la, lb) if A else (lb, la)
        dd = dec[(dec["pair"] == pair) & (dec["date"] > ed) & (dec["date"] < xt)]; ex = dd[dd["reason"].astype(str).str.startswith("exit:")]
        reason = str(ex["reason"].iloc[-1]) if len(ex) else ""
        cc = c.loc[ed]; z = float(cc["z"])
        br = [s.asof(ed) for s in zall.values() if s.index[0] <= ed and (ed - s.index[s.index <= ed][-1]).days <= 10]
        br = [v for v in br if v == v]
        idx = wk.index[(wk.index > ed) & (wk.index < xt)]
        rows.append({"test": test, "pair": pair, "pool": "gold" if GOLD in pair else "index", "entry_decision": ed.date().isoformat(), "entry_trade": et.date().isoformat(), "exit_trade": xt.date().isoformat(),
                     "held": tr["held"], "held_leg": held_leg, "peer_leg": peer_leg, "extra": float(tr["extra"]), "held_ret": float(tr["held_ret"]), "peer_ret": float(tr["peer_ret"]), "good": bool(tr["extra"] > 0),
                     "weeks_held": int(len(dd)), "exit_reason": "mean" if "back at the mean" in reason else ("cap" if "max hold" in reason else "other"), "month": int(ed.month),
                     "corr26": float(cc["corr26"]), "corr52": float(cc["corr52"]), "mkt52": float(mkt.asof(ed) / mkt.asof(ed - pd.Timedelta(days=364)) - 1), "mkt_dd": float(mkt.asof(ed) / mkt[:ed].max() - 1),
                     "vol_held": float(cc["vol_a"] if A else cc["vol_b"]), "vol_peer": float(cc["vol_b"] if A else cc["vol_a"]), "vol_ratio": float(cc["vol_ratio"]), "band_width": float(cc["band_width"]),
                     "z_entry": z, "z_depth": abs(z), "since_zero": float(cc["since_zero"]),
                     "lag4": float((cc["ret_a4"] - cc["ret_b4"]) if A else (cc["ret_b4"] - cc["ret_a4"])), "lag13": float((cc["ret_a13"] - cc["ret_b13"]) if A else (cc["ret_b13"] - cc["ret_a13"])),
                     "held13": float(cc["ret_a13"] if A else cc["ret_b13"]), "peer13": float(cc["ret_b13"] if A else cc["ret_a13"]), "breadth": float(np.mean(np.abs(br) > 1)) if br else float("nan"),
                     "mkt_during": float(mkt.asof(xt) / mkt.asof(et) - 1),
                     "corr_during": float(ra[idx].corr(rb[idx])) if len(idx) >= MIN_DURING else float("nan"),
                     "vol_ratio_during": float(lr[idx].std() * np.sqrt(52)) if len(idx) >= MIN_DURING else float("nan")})
    return rows, n_open


def assoc(df, rng, P):
    """Per condition: the median among good minus among bad trades, and the rank correlation with the extra; both against
    P permutations of the outcomes within each pair."""
    out = []
    for key, lab, when, kind in CONDS:
        sub = df[[key, "extra", "pair"]].dropna()
        base = {"key": key, "label": lab, "when": when, "kind": kind, "n": int(len(sub)), "n_good": int((sub["extra"] > 0).sum())}
        if len(sub) < MIN_TRADES or sub["extra"].gt(0).sum() < 3 or sub["extra"].le(0).sum() < 3:
            out.append({**base, "med_good": np.nan, "med_bad": np.nan, "diff": np.nan, "diff_pct": np.nan, "rho": np.nan, "rho_pct": np.nan}); continue
        x = sub[key].to_numpy(float); y = sub["extra"].to_numpy(float); good = y > 0; pr = sub["pair"].to_numpy()
        xr = sub.groupby("pair")[key].rank(pct=True).to_numpy(); yr = sub.groupby("pair")["extra"].rank(pct=True).to_numpy()
        YR = np.tile(yr, (P, 1)); G = np.tile(good, (P, 1))
        for p in np.unique(pr):
            idx = np.flatnonzero(pr == p); perm = rng.permuted(np.tile(idx, (P, 1)), axis=1); YR[:, idx] = yr[perm]; G[:, idx] = good[perm]
        xs = (xr - xr.mean()) / xr.std(); ys = (yr - yr.mean()) / yr.std(); rho = float((xs * ys).mean())
        YS = (YR - YR.mean(axis=1, keepdims=True)) / YR.std(axis=1, keepdims=True); rho_null = YS @ xs / len(xs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg, mb = float(np.median(x[good])), float(np.median(x[~good]))
            dn = np.nanmedian(np.where(G, x, np.nan), axis=1) - np.nanmedian(np.where(~G, x, np.nan), axis=1)
        out.append({**base, "med_good": mg, "med_bad": mb, "diff": mg - mb, "diff_pct": E1.pctile(mg - mb, dn), "rho": rho, "rho_pct": E1.pctile(rho, rho_null)})
    return out


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args(); D, P = args.draws, args.perms
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily()
    audits = {}
    for test in TESTS:                                                   # both tests must be locked and run
        ev = os.path.join(ROOT, "tests", test, "evidence", RUN)
        if not os.path.exists(os.path.join(ev, "trade_extras.csv")):
            sys.exit(f"{test}: no run at {ev}; E03 runs only after both tests have run")
        a = json.load(open(os.path.join(ev, "audit.json"), encoding="utf-8")); audits[test] = {"run_date": a.get("run_date"), "ratio": a.get("ratio", "log"), "draws": a.get("draws")}
    # ---------------------------------------------------------------- Part A
    prof, A_chk, win = [], {}, {}
    for pid, pair in enumerate(PAIRS):
        rows, chk, first, last = profiles(pair, pid, px, D); prof += rows; A_chk[pair] = chk; win[pair] = (first, last)
        print(f"  A {pair:18s} {first} → {last}  z check {chk:.1e}  [{time.time() - t0:4.0f}s]", flush=True)
    if max(A_chk.values()) > 1e-9:
        sys.exit(f"Part A: the real path's z is not E01's z: {A_chk}")
    PR = pd.DataFrame(prof); PR.to_csv(os.path.join(OUT, "z_profiles.csv"), index=False)
    # ---------------------------------------------------------------- Part B
    series, zall, PIT = {}, {}, [0, []]
    for pair in PAIRS:
        pf = pair_frame(px, pair, "plain"); series[pair] = condition_series(pf); zall[pair] = series[pair][1]["z"].dropna()
        c_, b_ = cond_pit_check(pf); PIT[0] += c_; PIT[1] += [f"{pair}: {x}" for x in b_]
    if PIT[1] or PIT[0] == 0:
        sys.exit("look-ahead check FAILED: " + "; ".join(PIT[1]) if PIT[1] else "look-ahead check ran nothing")
    print(f"  B look-ahead check: {PIT[0]} checks, 0 violations")
    mkt = px[MARKET]["close"].astype(float)
    trades, n_open = [], {}
    for test in TESTS:
        r, n_open[test] = trade_rows(test, px, series, zall, mkt); trades += r
    TR = pd.DataFrame(trades); TR.to_csv(os.path.join(OUT, "trades_conditions.csv"), index=False)
    key = lambda df: set(zip(df["pair"], df["entry_decision"], df["held"]))                                       # noqa: E731
    k1, k2 = key(TR[TR["test"] == "T01"]), key(TR[TR["test"] == "T02"])
    overlap = {"T02_trades": len(k2), "T01_trades": len(k1), "shared": len(k1 & k2), "share_of_T02_in_T01": len(k1 & k2) / max(len(k2), 1)}
    stats = {}
    for ti, (test, pool) in enumerate(POOLS):
        sub = TR[(TR["test"] == test) & (TR["pool"] == pool)]
        stats[(test, pool)] = assoc(sub, np.random.default_rng([SEED_PERM, TESTS.index(test), 0 if pool == "index" else 1]), P)
        print(f"  B {test} {pool:5s} {len(sub):3d} trades  [{time.time() - t0:4.0f}s]", flush=True)
    CS = pd.DataFrame([{"test": t, "pool": p, **s} for (t, p), rows in stats.items() for s in rows]); CS.to_csv(os.path.join(OUT, "conditions_summary.csv"), index=False)
    PP = TR.groupby(["test", "pair"]).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    by_exit = TR.groupby(["test", "pool", "exit_reason"]).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    # notable counts
    A_p = [v for (v, b) in zip(PR["spread_mean_pct"].tolist() + PR["spread_median_pct"].tolist(), PR["bucket"].tolist() * 2) if v == v and b != "all weeks"]
    B_p = [v for v in CS["diff_pct"].tolist() + CS["rho_pct"].tolist() if v == v]
    nA, nB = sum(E1.notable(v) for v in A_p), sum(E1.notable(v) for v in B_p)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "tests": audits, "setting": {"window": WIN, "exit": EXIT, "run": RUN},
             "part_A": {"draws": D, "seed": f"default_rng([{E1.SEED_SIGNS}, pair_id]) (E01's signs)", "random_legs": "M = (ln a + ln b) / 2 kept real; a* = exp(M + ln r* / 2), b* = exp(M - ln r* / 2)",
                        "z_equals_E01_max_abs_diff": max(A_chk.values()), "horizons": HORIZONS, "buckets": [b[0] for b in BUCKETS], "notable": [nA, len(A_p)]},
             "part_B": {"perms": P, "seed": f"default_rng([{SEED_PERM}, test_id, pool_id])", "conditions": [c[0] for c in CONDS], "look_ahead_check": {"checks": PIT[0], "violations": PIT[1]},
                        "open_trades_left_out": n_open, "overlap_T01_T02": overlap, "notable": [nB, len(B_p)]}, "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return f"**{v:.1f}**" if E1.notable(v) else E1.f2(v, 1)
    md = [f"# E03 results — when does the ratio switch work? (EXPLORATORY)\n",
          f"_Generated by `explore/E03_trade_conditions/conditions.py` on {audit['run_date']} from T01 (run {audits['T01']['run_date']}) and T02 (run {audits['T02']['run_date']}) at window {WIN}, exit {EXIT}, 20.8 bps; Part A against {D:,} random paths per pair, Part B against {P:,} permutations. Description, not a finding. Percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Against chance:** Part A {nA} of {len(A_p)} percentiles notable (about {0.1 * len(A_p):.0f} by chance); Part B {nB} of {len(B_p)} (about {0.1 * len(B_p):.0f} by chance). T01's and T02's trades overlap: {overlap['shared']} of T02's {overlap['T02_trades']} closed trades are also T01's ({overlap['share_of_T02_in_T01'] * 100:.0f}%), so the two tests are not two replications.\n",
          "## Part A — forward returns by z bucket (cheaper leg = the one that has lagged)\n",
          "Per pair, bucket and horizon: weeks (episodes), cheaper leg mean / median, richer leg mean / median, spread = cheaper − richer mean (random walks' median, percentile) / median (percentile), both legs' average. Window after the 52-week warm-up.\n"]
    for p in ORDER:
        md += [f"### {nice(p)} · {win[p][0]} → {win[p][1]}\n", "| horizon | z bucket | weeks (episodes) | cheaper mean / median | richer mean / median | spread mean (rw, pct) | spread median (pct) | both legs mean |", "|---|---|---|---|---|---|---|---|"]
        for _, r in PR[PR["pair"] == p].iterrows():
            md.append(f"| {r['horizon']} wk | {r['bucket']} | {r['weeks']} ({r['episodes']}) | {E1.fp(r['cheap_mean'])} / {E1.fp(r['cheap_median'])} | {E1.fp(r['rich_mean'])} / {E1.fp(r['rich_median'])} | {E1.fp(r['spread_mean'])} ({E1.fp(r['spread_mean_rw'])}, {pq(r['spread_mean_pct'])}) | {E1.fp(r['spread_median'])} ({pq(r['spread_median_pct'])}) | {E1.fp(r['both_mean'])} |")
        md.append("")
    md += ["## Part B — the market's condition at entry, good trades (extra > 0) vs bad\n"]
    for test, pool in POOLS:
        rows = stats[(test, pool)]; sub = TR[(TR["test"] == test) & (TR["pool"] == pool)]
        md += [f"### {test}, {pool} pairs · {len(sub)} closed trades, {int(sub['good'].sum())} good\n",
               "| condition | when | n | median, good trades | median, bad trades | good − bad (pct) | rank correlation with the extra (pct) |", "|---|---|---|---|---|---|---|"]
        md += [f"| {s['label']} | {s['when']} | {s['n']} | {fmtv(s['kind'], s['med_good'])} | {fmtv(s['kind'], s['med_bad'])} | {fmtc(s['kind'], s['diff'])} ({pq(s['diff_pct'])}) | {E1.f2(s['rho'])} ({pq(s['rho_pct'])}) |" for s in rows]
        md.append("")
    md += ["## Trades by exit reason\n", "| test | pool | exit | trades | share good | median extra | mean extra | median weeks held |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {r['pool']} | {r['exit_reason']} | {r['n']} | {r['good'] * 100:.0f}% | {E1.fp(r['median_extra'])} | {E1.fp(r['mean_extra'])} | {r['weeks']:.0f} |" for _, r in by_exit.iterrows()]
    md += ["\n## Trades per pair\n", "| test | pair | closed trades | share good | median extra | mean extra | median weeks held |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {nice(r['pair'])} | {r['n']} | {r['good'] * 100:.0f}% | {E1.fp(r['median_extra'])} | {E1.fp(r['mean_extra'])} | {r['weeks']:.0f} |" for _, r in PP.iterrows()]
    open(os.path.join(OUT, "E03_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- figures
    figs = {}; titles = {"prof": {}, "tl": {}}
    # A1. per pair: mean forward return by bucket, three horizons
    fA = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.09, subplot_titles=[f"{k} weeks ahead" for k in HORIZONS])
    for p in ORDER:
        vis = p == DEFAULT_PAIR; sub = PR[(PR["pair"] == p) & (PR["bucket"] != "all weeks")]
        for i, k in enumerate(HORIZONS):
            s = sub[sub["horizon"] == k].set_index("bucket").reindex(CHART_BUCKETS); lead = i == 0
            hv = [f"{b}: {w} weeks, {e} episodes" for b, w, e in zip(CHART_BUCKETS, s["weeks"], s["episodes"])]
            fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s["cheap_mean"] * 100, marker={"color": AQUA}, name="cheaper leg (has lagged)", meta=p, visible=vis, showlegend=lead, legendgroup="c", customdata=hv, hovertemplate="%{customdata}<br>cheaper leg %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
            fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s["rich_mean"] * 100, marker={"color": ORANGE}, name="richer leg", meta=p, visible=vis, showlegend=lead, legendgroup="r", customdata=hv, hovertemplate="%{customdata}<br>richer leg %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
            fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s["spread_mean"] * 100, marker={"color": BLUE}, name="spread (cheaper − richer)", meta=p, visible=vis, showlegend=lead, legendgroup="s", customdata=[[v] for v in s["spread_mean_pct"]], hovertemplate="spread %{y:+.1f}% · percentile %{customdata[0]:.1f}<extra></extra>"), row=i + 1, col=1)
            fA.add_trace(go.Scatter(x=CHART_BUCKETS, y=s["spread_mean_rw"] * 100, mode="markers", marker={"color": MUTED, "size": 7, "symbol": "line-ew-open", "line": {"width": 2}}, name="random walks' spread: median and 5th–95th", meta=p, visible=vis, showlegend=lead, legendgroup="n",
                                    error_y={"type": "data", "symmetric": False, "array": (s["spread_mean_hi"] - s["spread_mean_rw"]) * 100, "arrayminus": (s["spread_mean_rw"] - s["spread_mean_lo"]) * 100, "color": MUTED, "thickness": 1.2, "width": 6},
                                    hovertemplate="random walks: median %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
        titles["prof"][p] = E1.two_line(f"{nice(p)} · {win[p][0]} → {win[p][1]} · mean forward return of each leg by the ratio's z-score that week",
                                        "cheaper leg = the one that has lagged (a when z < 0, b when z > 0); the grey mark is what the random walks give the spread (their median, 5th–95th)")
    fA.update_layout(barmode="group", bargap=0.25)
    for i in range(3):
        fA.update_yaxes(ticksuffix="%", title={"text": "forward return", "font": {"color": INK2}}, row=i + 1, col=1); fA.add_hline(y=0, line={"color": AXIS, "width": 1}, row=i + 1, col=1)
    fA.update_xaxes(title={"text": "z-score bucket at the start (z < −1 or > 1 = outside T02's band)", "font": {"color": INK2}}, row=3, col=1)
    figs["prof"] = E1.style(fA, titles["prof"][DEFAULT_PAIR], 900, legend_top=True)
    # A2. all pairs: percentile of the spread's mean among the random walks, bucket x horizon
    cols = [f"{b} · {k} wk" for k in HORIZONS for b in CHART_BUCKETS]
    zv, tx, hv = [], [], []
    for p in ORDER:
        sub = PR[PR["pair"] == p].set_index(["horizon", "bucket"]); zz, tt, hh = [], [], []
        for k in HORIZONS:
            for b in CHART_BUCKETS:
                r = sub.loc[(k, b)]; zz.append(r["spread_mean_pct"] if r["weeks"] >= 8 else float("nan"))
                tt.append(f"{r['spread_mean'] * 100:+.1f}%" if (r["weeks"] >= 8 and E1.notable(r["spread_mean_pct"])) else "")
                hh.append(f"{r['weeks']} weeks ({r['episodes']} episodes): spread {r['spread_mean'] * 100:+.1f}% (random walks {r['spread_mean_rw'] * 100:+.1f}%), percentile {r['spread_mean_pct']:.1f}" if r["weeks"] >= 8 else f"{r['weeks']} weeks: too few")
        zv.append(zz); tx.append(tt); hv.append(hh)
    f = go.Figure(go.Heatmap(z=zv, x=cols, y=[nice(p) for p in ORDER], zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=tx, texttemplate="%{text}", customdata=hv, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
                             colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
    f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="bottom", tickangle=-45, automargin=True)
    for i in (1, 2):
        f.add_vline(x=i * 6 - 0.5, line={"color": AXIS, "width": 1})
    figs["h_prof"] = E1.style(f, E1.two_line("The spread's mean forward return against the random walks, by z bucket and horizon (all pairs)", "red = the cheaper leg beat the richer leg by more than a random walk gives; blue = by less; labels on notable cells; blank = fewer than 8 weeks"), 620, legend=False, left=130)
    # B1. per pair: the trades on the two legs
    fT = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[f"{t} trades" for t in TESTS])
    for p in ORDER:
        vis = p == DEFAULT_PAIR; wk = series[p][0]; la, lb = p.split("|"); reb_a = wk["ca"] / wk["ca"].iloc[0] * 100; reb_b = wk["cb"] / wk["cb"].iloc[0] * 100
        ylo, yhi = float(min(reb_a.min(), reb_b.min())) * 0.9, float(max(reb_a.max(), reb_b.max())) * 1.1
        for i, test in enumerate(TESTS):
            lead = i == 0
            for name, s in ((la, reb_a), (lb, reb_b)):
                fT.add_trace(go.Scatter(x=s.index, y=s, mode="lines", line={"color": E1.CAT[name], "width": 1.4}, name=name, legendgroup=name, showlegend=lead, meta=p, visible=vis, hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:.0f}<extra></extra>"), row=i + 1, col=1)
            sub = TR[(TR["test"] == test) & (TR["pair"] == p)]
            for good, colr, lab in ((True, AQUA, "good trade (extra > 0)"), (False, ORANGE, "bad trade")):
                tt = sub[sub["good"] == good]; xs, ys, hx, hy, hc = [], [], [], [], []
                for _, t in tt.iterrows():
                    e, x_ = pd.Timestamp(t["entry_trade"]), pd.Timestamp(t["exit_trade"])
                    xs += [e, x_, x_, e, e, None]; ys += [ylo, ylo, yhi, yhi, ylo, None]
                    hx.append(e + (x_ - e) / 2); hy.append(yhi); hc.append(f"held {t['held_leg']} {t['entry_trade']} → {t['exit_trade']} ({t['weeks_held']} wk, exit: {t['exit_reason']})<br>extra {t['extra'] * 100:+.1f}% · held leg {t['held_ret'] * 100:+.1f}%, peer {t['peer_ret'] * 100:+.1f}%<br>at entry: z {t['z_entry']:+.2f}, corr52 {t['corr52']:.2f}, N100 52-wk {t['mkt52'] * 100:+.0f}%, drawdown {t['mkt_dd'] * 100:.0f}%")
                fT.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=colr, opacity=0.22, line={"width": 0}, name=lab, legendgroup=lab, showlegend=lead, meta=p, visible=vis, hoverinfo="skip"), row=i + 1, col=1)
                fT.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker={"color": colr, "size": 7, "symbol": "triangle-down"}, showlegend=False, meta=p, visible=vis, customdata=hc, hovertemplate="%{customdata}<extra></extra>"), row=i + 1, col=1)
        n2, n1 = int((TR["test"] == "T02")[TR["pair"] == p].sum()), int((TR["test"] == "T01")[TR["pair"] == p].sum())
        g2 = TR[(TR["test"] == "T02") & (TR["pair"] == p)]["good"].mean() if n2 else float("nan")
        titles["tl"][p] = E1.two_line(f"{nice(p)} · the trades over the two legs (rebased to 100 at {wk.index[0].date()}) · T02 {n2} closed trades, {g2 * 100:.0f}% good · T01 {n1}",
                                      "shaded = a trade's hold, aqua if the held leg beat the peer (extra > 0), orange if not; hover the marker for the trade and the market at its entry")
    for i in range(2):
        fT.update_yaxes(type="log", title={"text": "rebased", "font": {"color": INK2}}, row=i + 1, col=1)
    figs["tl"] = E1.style(fT, titles["tl"][DEFAULT_PAIR], 720, legend_top=True)
    # B2. conditions heat map: rows conditions, columns (test, pool) x statistic
    ccols = [f"{t} {pl}<br>{st}" for t, pl in POOLS for st in ("good − bad", "rank corr")]
    zv, tx, hv = [], [], []
    for key, lab, when, kind in CONDS:
        zz, tt, hh = [], [], []
        for t, pl in POOLS:
            s = next(x for x in stats[(t, pl)] if x["key"] == key)
            for st, val, pc in (("good − bad", s["diff"], s["diff_pct"]), ("rank corr", s["rho"], s["rho_pct"])):
                zz.append(pc); vtxt = fmtc(kind, val) if st == "good − bad" else E1.f2(val)
                tt.append(vtxt if E1.notable(pc) else ""); hh.append(f"{s['n']} trades: {st} {vtxt}, percentile {pc:.1f}" if pc == pc else f"{s['n']} trades: too few")
        zv.append(zz); tx.append(tt); hv.append(hh)
    f = go.Figure(go.Heatmap(z=zv, x=ccols, y=[f"{lab}{' (during)' if when == 'during' else ''}" for _, lab, when, _ in CONDS], zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=tx, texttemplate="%{text}", customdata=hv, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
                             colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
    f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="top")
    for i in (1, 2, 3):
        f.add_vline(x=i * 2 - 0.5, line={"color": AXIS, "width": 1})
    figs["h_cond"] = E1.style(f, E1.two_line("Is the condition associated with the trade's outcome? Percentile among permutations of the outcomes within each pair", "red = good trades had the higher value (or the correlation with the extra is positive), blue = the lower; labels on notable cells"), 760, legend=False, left=250)
    # B3. scatter: extra vs the key conditions, index pairs, T02 (blue) and T01 (grey)
    labs = {c[0]: (c[1], c[3]) for c in CONDS}
    fS = make_subplots(rows=2, cols=3, subplot_titles=[labs[k][0] for k in KEY_CONDS], vertical_spacing=0.16, horizontal_spacing=0.07)
    for j, k in enumerate(KEY_CONDS):
        row, col = j // 3 + 1, j % 3 + 1; pct = labs[k][1] == "pct"
        for test, colr, sz in (("T01", MUTED, 5), ("T02", BLUE, 7)):
            sub = TR[(TR["test"] == test) & (TR["pool"] == "index")].dropna(subset=[k])
            fS.add_trace(go.Scatter(x=sub[k] * (100 if pct else 1), y=sub["extra"] * 100, mode="markers", marker={"color": colr, "size": sz, "opacity": 0.75}, name=f"{test} trades, index pairs", showlegend=j == 0, legendgroup=test,
                                    customdata=[[nice(a), b, c] for a, b, c in zip(sub["pair"], sub["entry_decision"], sub["exit_reason"])], hovertemplate="%{customdata[0]} · entered %{customdata[1]} (exit: %{customdata[2]})<br>" + labs[k][0] + " %{x:.2f}" + ("%" if pct else "") + " · extra %{y:+.1f}%<extra></extra>"), row=row, col=col)
        fS.add_hline(y=0, line={"color": AXIS, "width": 1}, row=row, col=col)
        fS.update_xaxes(ticksuffix="%" if pct else "", row=row, col=col)
    fS.update_yaxes(ticksuffix="%", title={"text": "extra", "font": {"color": INK2}}, col=1)
    figs["scatter"] = E1.style(fS, E1.two_line("Each trade's extra (held leg − peer) against six conditions at its entry · index pairs, both tests", "one dot per closed trade; T02 in blue, T01 in grey (the two overlap heavily)"), 620, legend_top=True)

    # ---------------------------------------------------------------- the page
    def per_pair(key, fig, tables, first=False) -> str:
        t = html.escape(json.dumps(titles[key]), quote=True)
        tb = "".join(f"<div data-pair='{html.escape(p, quote=True)}'{'' if p == DEFAULT_PAIR else ' hidden'}>{tables[p]}</div>" for p in ORDER)
        return f"<div class='card per-pair' data-titles='{t}'>{E1.html_fig(fig, first=first)}<details><summary>Table view</summary>{tb}</details></div>"
    tab_prof = {p: E1.tbl(["horizon", "z bucket", "weeks (episodes)", "cheaper: mean / median", "richer: mean / median", "spread: mean (random walks, pct)", "spread: median (pct)", "both legs: mean"],
                          [[f"{r['horizon']} wk", r["bucket"], f"{r['weeks']} ({r['episodes']})", f"{E1.fp(r['cheap_mean'])} / {E1.fp(r['cheap_median'])}", f"{E1.fp(r['rich_mean'])} / {E1.fp(r['rich_median'])}",
                            f"{E1.fp(r['spread_mean'])} ({E1.fp(r['spread_mean_rw'])}, {E1.fpct(r['spread_mean_pct'])})", f"{E1.fp(r['spread_median'])} ({E1.fpct(r['spread_median_pct'])})", E1.fp(r["both_mean"])] for _, r in PR[PR["pair"] == p].iterrows()]) for p in ORDER}
    tab_tl = {p: E1.tbl(["test", "entered", "exited", "held", "weeks", "exit", "extra", "held leg", "peer", "z at entry", "corr 52 wk", "N100 52-wk", "N100 drawdown", "band width", "breadth"],
                        [[t["test"], t["entry_trade"], t["exit_trade"], t["held_leg"], t["weeks_held"], t["exit_reason"], E1.fp(t["extra"]), E1.fp(t["held_ret"]), E1.fp(t["peer_ret"]), f"{t['z_entry']:+.2f}", E1.f2(t["corr52"]), E1.fp(t["mkt52"]), E1.fp(t["mkt_dd"]), fmtv("pct", t["band_width"]), fmtv("pct", t["breadth"])]
                         for _, t in TR[TR["pair"] == p].sort_values(["test", "entry_trade"]).iterrows()]) for p in ORDER}
    def cond_table(test, pool):
        rows = stats[(test, pool)]; n = len(TR[(TR["test"] == test) & (TR["pool"] == pool)])
        return (f"<p class='note'><b>{test}, {pool} pairs</b> · {n} closed trades</p>" +
                E1.tbl(["condition", "when", "n", "median: good trades", "median: bad trades", "good − bad", "percentile", "rank correlation with the extra", "percentile"],
                       [[s["label"], s["when"], s["n"], fmtv(s["kind"], s["med_good"]), fmtv(s["kind"], s["med_bad"]), fmtc(s["kind"], s["diff"]), E1.fpct(s["diff_pct"]), E1.f2(s["rho"]), E1.fpct(s["rho_pct"])] for s in rows]))
    # the sentence per pool: the notable conditions, in words
    def words(test, pool):
        out = []
        for s in stats[(test, pool)]:
            if E1.notable(s["diff_pct"]):
                out.append(f"{s['label']}{' (during)' if s['when'] == 'during' else ''}: good trades {fmtv(s['kind'], s['med_good'])} vs bad {fmtv(s['kind'], s['med_bad'])} (percentile {s['diff_pct']:.0f})")
        return "; ".join(out) if out else "no condition's good-vs-bad difference is notable"
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS).replace("<title>Ratio cycles</title>", "<title>Trade conditions</title>"),
         "<header class='bar'><div class='bar-inner'><b>E03 · trade conditions</b><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label><button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>When does the ratio switch work?</h1>",
         f"<p>Two descriptions, both made after T01 (REFUTED) and T02 were locked and run. <b>A.</b> Every week, the ratio's z-score (T02's band in units of its sd) is bucketed and the next 4, 13 and 26 weeks' returns of the <b>cheaper</b> leg (the one that has lagged) and the <b>richer</b> leg are shown, against {D:,} random paths of the ratio that keep the two legs' common movement real. "
         f"<b>B.</b> Every closed trade of T01 and T02 at the reference setting (window {WIN}, exit {EXIT}, 20.8 bps) carries the market's condition at its entry and during it; conditions are compared between good trades (the held leg beat the peer, fill to fill) and bad ones, against {P:,} permutations of the outcomes within each pair. "
         f"Generated {audit['run_date']} from the approved data of 2026-11-12 and the two tests' evidence (T01 run {audits['T01']['run_date']}, T02 run {audits['T02']['run_date']}). <b>Description, not a finding</b>: a condition that goes with good trades here is a candidate for a pre-registered test, not a rule.</p>",
         f"<div class='card'><p style='margin:4px 0'><b>Against chance:</b> Part A {nA} of {len(A_p)} percentiles are notable (about {0.1 * len(A_p):.0f} would be by chance); Part B {nB} of {len(B_p)} (about {0.1 * len(B_p):.0f}). "
         f"T01's and T02's trades overlap: {overlap['shared']} of T02's {overlap['T02_trades']} closed trades are also T01's ({overlap['share_of_T02_in_T01'] * 100:.0f}%), so the two tests are one sample seen twice, not two replications. Conditions at entry use data up to the entry decision only ({PIT[0]} cut checks, {len(PIT[1])} violations); conditions during the trade are hindsight by definition.</p></div>",
         "<h2>A · What each leg does next, by how stretched the ratio is</h2><p class='note'>Pick the pair at the top. Bars: the mean forward return of the cheaper leg, the richer leg and their difference (the spread: what holding the cheaper leg earns against the richer one, before costs). The grey mark is the spread on the random walks (median, 5th–95th); a blue bar outside the grey range is notable. Weeks overlap, so the episode count in the table is the honest sample size.</p>",
         per_pair("prof", figs["prof"], tab_prof, first=True),
         "<div class='card'>" + E1.html_fig(figs["h_prof"]) + "</div>",
         "<h2>B · The market's condition at each trade</h2><p class='note'>The trades of both tests on the chosen pair, over the two legs. Hover a marker for the trade's outcome and the market at its entry.</p>",
         per_pair("tl", figs["tl"], tab_tl),
         "<h3>Which conditions go with good trades?</h3>"
         f"<div class='card'><p style='margin:4px 0'><b>T02, index pairs:</b> {words('T02', 'index')}. <b>T01, index pairs:</b> {words('T01', 'index')}. <b>Gold pairs</b> (T02): {words('T02', 'gold')}.</p></div>",
         "<div class='card'>" + E1.html_fig(figs["h_cond"]) + "<details><summary>Table view</summary>" + "".join(cond_table(t, pl) for t, pl in POOLS) + "</details></div>",
         "<div class='card'>" + E1.html_fig(figs["scatter"]) + "</div>",
         "<h3>By exit reason, and per pair</h3><div class='card'>" + E1.tbl(["test", "pool", "exit", "trades", "share good", "median extra", "mean extra", "median weeks held"], [[r["test"], r["pool"], r["exit_reason"], r["n"], f"{r['good'] * 100:.0f}%", E1.fp(r["median_extra"]), E1.fp(r["mean_extra"]), f"{r['weeks']:.0f}"] for _, r in by_exit.iterrows()])
         + "<details><summary>Per pair</summary>" + E1.tbl(["test", "pair", "closed trades", "share good", "median extra", "mean extra", "median weeks held"], [[r["test"], nice(r["pair"]), r["n"], f"{r['good'] * 100:.0f}%", E1.fp(r["median_extra"]), E1.fp(r["mean_extra"]), f"{r['weeks']:.0f}"] for _, r in PP.iterrows()]) + "</details></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Part A.</b> z = (plain ratio − trailing 52-week mean) ÷ trailing sd (E01's). Cheaper leg = a when z < 0, b when z > 0. Forward returns are close to close on decision weeks, simple, no costs or fills. Random paths: E01's (the point-in-time residuals with the same random signs plus the expanding drift, after the real 52-week warm-up); with M = ½(ln a + ln b) kept real, the random legs are exp(M ± ½ ln r*), so only the legs' relative movement is random. The window starts after the warm-up. Percentile = (paths below + ½ ties) ÷ paths.</li>"
         f"<li><b>Part B.</b> Trades from tests/T0x/evidence/{RUN}/trade_extras.csv at window {WIN}, exit {EXIT}; open trades left out (T02 {n_open['T02']}, T01 {n_open['T01']}). Extra = held leg's fill-to-fill return − the peer's (T01 §4.3), no cost. Conditions at the entry decision's close from the pair's decision-week series and N100's daily closes (N100 is the market for every pair); |z| is the depth; breadth = the share of the ten ratios whose own z is beyond ±1 that week. During: entry fill to exit fill (N100), and the decision weeks held (correlation and volatility need {MIN_DURING} weeks or more).</li>"
         f"<li><b>Null.</b> Within each pair the trades' extras (for the rank correlation) or good/bad labels (for the medians) are shuffled {P:,} times, seed default_rng([{SEED_PERM}, test_id, pool_id]); the trades are ranked within their pair before pooling for the correlation. The null asks only whether the condition is associated with the outcome. Whether a filtered rule beats random timing is a new pre-registered test.</li>"
         "<li><b>Limits.</b> 6–41 closed trades per index pair; many correlated conditions; the two tests overlap; gold pairs from 2015 with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); Micro250 pairs from 2022. The design was written before the first run (explore/E03_trade_conditions/E03.md).</li></ul></div>",
         "</main>", "<script>" + E1.JS.replace("__DIV_LIGHT__", json.dumps(E1.DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(E1.DIV_DARK)).replace("e01-pair", "e03-pair") + "</script></body></html>"]
    out = os.path.join(OUT, "E03_conditions.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable A {nA} of {len(A_p)}, B {nB} of {len(B_p)} → {out}")


if __name__ == "__main__":
    main()
