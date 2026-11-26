"""
E03 (EXPLORATORY): when does the ratio switch work? Design: explore/E03_trade_conditions/E03.md (written before the first run).
A. Forward returns of the cheaper and the richer leg by the ratio's z-score bucket, every week, against E01's random walks.
B. The market's condition at the entry of every T03 trade (window 52, exit E1, 20.8 bps, no maximum hold) and during it, related to
   the trade's outcome (the per-trade extra), with permutation nulls.

    python explore/E03_trade_conditions/conditions.py [--draws 2000] [--perms 2000]

Run only after T03's run finished (locked and run). Part B uses T03's trades: the rule with no maximum hold.
Not a finding and not a test of any rule.

Re-run on 2026-11-26 on the latest data (explore/E01_ratio_cycles/latest.py; E03.md, the block of that date): N100, MC100,
SC100 and MCX gold from load_india_latest() to 2026-11-20, Micro250 on its earlier data, all from 2011-01-01; whenever the
rule holds gold while gold's weekly supertrend is SELL it holds the liquid ETF instead. T03's evidence is on the earlier data,
so Part B's 52-week trades are T03's rule computed here (variants.py through the engine's book, the switch on the gold pairs),
checked against the engine's decide() and, with the switch off, the engine's book.
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
from analysis.approved_data import GOLD, fingerprints  # noqa: E402
from analysis.ratio_engine import PAIRS, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (E01's z-score, random paths, percentiles and the page's styling)
import zwin as ZW  # noqa: E402  (the z windows: weekly 13 / 26 / 52, daily 20 / 60 / 120 / 250)
import variants as VR  # noqa: E402  (the T03 rule on each z window's band, through the engine's book)
import latest as LT  # noqa: E402  (the latest data and the gold → liquid switch)

HORIZONS = (4, 13, 26)
HORIZONS_D = (20, 65, 130)                      # daily windows: the same horizons in trading days (5 a week)
BUCKETS = (("< −2", -np.inf, -2.0), ("−2 to −1", -2.0, -1.0), ("−1 to 0", -1.0, 0.0), ("0 to 1", 0.0, 1.0), ("1 to 2", 1.0, 2.0), ("> 2", 2.0, np.inf), ("all weeks", -np.inf, np.inf))
CHART_BUCKETS = [b[0] for b in BUCKETS[:-1]]
TESTS = ("T03",); MAIN = TESTS[0]; RUN = "headline"; WIN, EXIT = 52, "E1"   # T03 = the rule with no maximum hold
MARKET = "N100"; SEED_PERM = 105; MIN_DURING = 4; MIN_TRADES = 10
ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR; LO, HI = E1.LO, E1.HI
POOLS = ((MAIN, "index"), (MAIN, "gold"))
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
KEY_CONDS = ("z_depth", "corr", "mkt52", "mkt_dd", "band_width", "breadth")   # "corr" = the legs' correlation over the period chosen at the top
# the correlation period (the second dropdown): the legs' trailing Pearson correlation of weekly log returns on decision weeks
# (13 / 26 / 52 weeks) or of daily log returns on the pair's common trading days (20 / 60 / 120 / 250 days), read at the decision
CWINS = (("cw13", "13 weeks", "W", 13), ("cw26", "26 weeks", "W", 26), ("cw52", "52 weeks", "W", 52),
         ("cd20", "20 days", "D", 20), ("cd60", "60 days", "D", 60), ("cd120", "120 days", "D", 120), ("cd250", "250 days", "D", 250))
CWIN = {c[0]: c for c in CWINS}; CDEFAULT = "cw52"; CSEP = "¤"; SEED_CORR = 106
CP_CONDS = tuple((f"corr_{c}", f"legs' correlation, {lab}{' (daily returns)' if f == 'D' else ''}", "entry", "num") for c, lab, f, _ in CWINS)
CP_SAME = {"corr_cw26": "corr26", "corr_cw52": "corr52"}   # the same series as two of CONDS: their statistics are reused, not redrawn
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


# ------------------------------------------------------------------------------------------------------- Part A, any window
def profiles_win(pair, pid, px, D, w):
    """Part A for z window w: the real ratio and D random paths of the window's frequency (weekly: E01's, identical to
    profiles(); daily: the daily paths), in chunks. Weekly windows: every statistic as profiles(). Daily windows: the
    random walks' per-path means for every series and per-path medians for the spread only (the chart and the heat map use
    means and the spread); the real ratio's medians are exact."""
    _, lab, f, n = ZW.WIN[w]; burn = ZW.BURN[f]; hz = HORIZONS if f == "W" else HORIZONS_D
    ser = ZW.series(pair_frame(px, pair, "plain"), f); dates = ser.index; a, b = ser["ca"].to_numpy(), ser["cb"].to_numpy(); lnr = np.log(a / b); L = len(lnr)
    M = 0.5 * (np.log(a) + np.log(b))

    def block(Pb, full):
        Zb = ZW.trailing_z(np.exp(Pb), n); lna = M[None, :] + 0.5 * Pb; lnb = M[None, :] - 0.5 * Pb; out = {}
        for k in hz:
            Ra = np.exp(lna[:, k:] - lna[:, :-k]) - 1; Rb = np.exp(lnb[:, k:] - lnb[:, :-k]) - 1; zk = Zb[:, :-k]
            X = {"cheap": np.where(zk < 0, Ra, Rb), "rich": np.where(zk < 0, Rb, Ra)}; X["spread"] = X["cheap"] - X["rich"]; X["both"] = 0.5 * (Ra + Rb)
            ok = np.zeros(L - k, bool); ok[burn:] = True
            for blab, lo, hi in BUCKETS:
                mask = ok[None, :] & (zk >= lo) & (zk < hi); cntm = mask.sum(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for name, Xv in X.items():
                        mu = np.where(mask, Xv, 0.0).sum(axis=1) / np.where(cntm > 0, cntm, np.nan)
                        md = np.nanmedian(np.where(mask, Xv, np.nan), axis=1) if (full or name == "spread") else np.full(len(Pb), np.nan)
                        out[(k, blab, name)] = (mu, md)
                out[(k, blab, "mask0")] = mask[0]
        return out
    real = block(lnr[None, :], True); nulls = []
    for _, Pc in ZW.random_log_paths(lnr, pid, f, D):
        nulls.append(block(Pc, f == "W"))
    rows = []
    for k in hz:
        for blab, _, _ in BUCKETS:
            m0 = real[(k, blab, "mask0")]
            row = {"window": w, "pair": pair, "horizon": k, "bucket": blab, "weeks": int(m0.sum()), "episodes": int((np.diff(np.concatenate([[0], m0.astype(np.int8), [0]])) == 1).sum())}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name in ("cheap", "rich", "spread", "both"):
                    mu = np.concatenate([real[(k, blab, name)][0]] + [nb[(k, blab, name)][0] for nb in nulls]); md = np.concatenate([real[(k, blab, name)][1]] + [nb[(k, blab, name)][1] for nb in nulls])
                    row[f"{name}_mean"] = float(mu[0]); row[f"{name}_median"] = float(md[0]); row[f"{name}_mean_rw"] = float(np.nanmedian(mu[1:]))
                    row[f"{name}_median_rw"] = float(np.nanmedian(md[1:])) if np.isfinite(md[1:]).any() else float("nan")
                    if name == "spread":
                        row["spread_mean_lo"] = float(np.nanpercentile(mu[1:], 5)); row["spread_mean_hi"] = float(np.nanpercentile(mu[1:], 95))
                        row["spread_mean_pct"] = E1.pctile(mu[0], mu[1:]); row["spread_median_pct"] = E1.pctile(md[0], md[1:])
            rows.append(row)
    return rows, dates[burn].date().isoformat(), dates[-1].date().isoformat()


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
    c["since_zero"] = since_zero(z.to_numpy())
    for k in (4, 13):
        c[f"ret_a{k}"] = wk["ca"] / wk["ca"].shift(k) - 1; c[f"ret_b{k}"] = wk["cb"] / wk["cb"].shift(k) - 1
    return wk, c, ra, rb, lr


def since_zero(z: np.ndarray) -> np.ndarray:
    """Decision weeks since z last had the other sign (1 = it changed sign this week)."""
    sgn = np.sign(z); since = np.full(len(sgn), np.nan); run = 0
    for i in range(len(sgn)):
        if np.isnan(sgn[i]):
            run = 0; continue
        run = run + 1 if (i > 0 and sgn[i] == sgn[i - 1]) else 1; since[i] = run
    return since


def zcond(pf, w):
    """The conditions that depend on the z window, at the pair's decision days: z, weeks since z changed sign, and the band's
    relative width (2 sd ÷ mean of the window's last n closes: decision-week closes for weekly windows, daily closes for daily
    ones). With w52 these are condition_series()'s (checked in main)."""
    _, _, f, n = ZW.WIN[w]; dec = pf.index[pf["decision"]]; s = ZW.series(pf, f)["ratio"]
    z = ZW.z_of(s, n); bw = 2 * s.rolling(n).std() / s.rolling(n).mean()
    if f == "D":
        z, bw = z.reindex(dec), bw.reindex(dec)
    return pd.DataFrame({"z": z.to_numpy(), "since_zero": since_zero(z.to_numpy()), "band_width": bw.to_numpy()}, index=dec)


def corr_frame(pf):
    """The legs' trailing correlation for every period of CWINS, at the pair's decision days, from data up to that day only."""
    dec = pf.index[pf["decision"]]; out = pd.DataFrame(index=dec)
    for c, _, f, n in CWINS:
        s = ZW.series(pf, f); ra, rb = np.log(s["ca"]).diff(), np.log(s["cb"]).diff()
        out[f"corr_{c}"] = ra.rolling(n).corr(rb).reindex(dec)
    return out


def corr_pit_check(pf, n_cuts=12):
    """corr_frame() recomputed from the history cut at 12 decision days must equal the full one up to each cut."""
    full = corr_frame(pf); dec = pf.index[pf["decision"]]; checks, bad = 0, []
    for T in np.unique(np.linspace(80, len(dec) - 1, n_cuts).astype(int)):
        c = corr_frame(pf.loc[: dec[T]]); checks += 1
        if not np.allclose(c.to_numpy(), full.to_numpy()[: len(c)], rtol=0, atol=1e-12, equal_nan=True):
            bad.append(f"correlations differ when the history is cut at {dec[T].date()}")
    return checks, bad


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


_VT = {}


def variant_trades(px, w):
    """The T03 rule on z window w's band for every pair (explore/E01_ratio_cycles/variants.py, the engine's book): the trade
    extras and the decisions in load_trades()'s shape, and, for the 52-week window, the check against T03's evidence (on the
    latest data: against the engine's decide() and book). Computed once per window."""
    if w in _VT:
        return _VT[w]
    xs, ds, bad = [], [], []
    for pair in PAIRS:
        v = VR.run_variant(px, pair, w)
        if w == ZW.DEFAULT:
            bad += VR.check_against_t03(v)
        if len(v["extras"]):
            xs.append(v["extras"].assign(pair=pair))
        ds.append(v["decisions"].reset_index().assign(pair=pair))
    x = pd.concat(xs, ignore_index=True); x["open"] = x["open"].astype(bool)
    for col in ("entry_decision", "entry_trade", "exit_trade"):
        x[col] = pd.to_datetime(x[col])
    _VT[w] = (x, pd.concat(ds, ignore_index=True), bad)
    return _VT[w]


def trade_rows(test, x, dec, series, zc, zall, mkt, corr=None):
    """One row per closed trade: the outcome and the conditions. x = the trade extras, dec = the decisions (pair, date,
    reason); zc[pair] = the z window's z, weeks since z changed sign and band width at the decision days; zall = {pair: z} of
    that window (breadth); corr[pair] = corr_frame() (adds one column per correlation period) or None."""
    rows = []; n_open = int(x["open"].sum())
    opened = [{"test": test, "pair": t["pair"], "entry_trade": t["entry_trade"].date().isoformat(), "held_leg": t["pair"].split("|")[0 if t["held"] == "A" else 1],
               "days_open": int((series[t["pair"]][0].index[-1] - t["entry_trade"]).days), "to": series[t["pair"]][0].index[-1].date().isoformat()} for _, t in x[x["open"]].iterrows()]
    for _, tr in x[~x["open"]].iterrows():
        pair = tr["pair"]; wk, c, ra, rb, lr = series[pair]; ed, et, xt = tr["entry_decision"], tr["entry_trade"], tr["exit_trade"]
        A = tr["held"] == "A"; la, lb = pair.split("|"); held_leg, peer_leg = (la, lb) if A else (lb, la)
        dd = dec[(dec["pair"] == pair) & (dec["date"] > ed) & (dec["date"] < xt)]; ex = dd[dd["reason"].astype(str).str.startswith("exit:")]
        reason = str(ex["reason"].iloc[-1]) if len(ex) else ""
        cc = c.loc[ed]; zz = zc[pair].loc[ed]; z = float(zz["z"])
        br = [s.asof(ed) for s in zall.values() if s.index[0] <= ed and (ed - s.index[s.index <= ed][-1]).days <= 10]
        br = [v for v in br if v == v]
        idx = wk.index[(wk.index > ed) & (wk.index < xt)]
        rows.append({"test": test, "pair": pair, "pool": "gold" if GOLD in pair else "index", "entry_decision": ed.date().isoformat(), "entry_trade": et.date().isoformat(), "exit_trade": xt.date().isoformat(),
                     "held": tr["held"], "held_leg": held_leg, "peer_leg": peer_leg, "extra": float(tr["extra"]), "held_ret": float(tr["held_ret"]), "peer_ret": float(tr["peer_ret"]), "good": bool(tr["extra"] > 0),
                     "weeks_held": int(len(dd)), "days_held": int((xt - et).days), "exit_reason": "mean" if "back at the mean" in reason else "other",
                     "hold": "up to 13 weeks" if len(dd) <= 13 else ("14–52 weeks" if len(dd) <= 52 else "over 52 weeks"), "month": int(ed.month),
                     "corr26": float(cc["corr26"]), "corr52": float(cc["corr52"]), "mkt52": float(mkt.asof(ed) / mkt.asof(ed - pd.Timedelta(days=364)) - 1), "mkt_dd": float(mkt.asof(ed) / mkt[:ed].max() - 1),
                     "vol_held": float(cc["vol_a"] if A else cc["vol_b"]), "vol_peer": float(cc["vol_b"] if A else cc["vol_a"]), "vol_ratio": float(cc["vol_ratio"]), "band_width": float(zz["band_width"]),
                     "z_entry": z, "z_depth": abs(z), "since_zero": float(zz["since_zero"]),
                     "lag4": float((cc["ret_a4"] - cc["ret_b4"]) if A else (cc["ret_b4"] - cc["ret_a4"])), "lag13": float((cc["ret_a13"] - cc["ret_b13"]) if A else (cc["ret_b13"] - cc["ret_a13"])),
                     "held13": float(cc["ret_a13"] if A else cc["ret_b13"]), "peer13": float(cc["ret_b13"] if A else cc["ret_a13"]), "breadth": float(np.mean(np.abs(br) > 1)) if br else float("nan"),
                     "mkt_during": float(mkt.asof(xt) / mkt.asof(et) - 1),
                     "corr_during": float(ra[idx].corr(rb[idx])) if len(idx) >= MIN_DURING else float("nan"),
                     "vol_ratio_during": float(lr[idx].std() * np.sqrt(52)) if len(idx) >= MIN_DURING else float("nan")})
        if corr is not None:
            rows[-1].update({k: float(v) for k, v in corr[pair].loc[ed].items()})
    return rows, n_open, opened


def assoc(df, rng, P, conds=CONDS):
    """Per condition: the median among good minus among bad trades, and the rank correlation with the extra; both against
    P permutations of the outcomes within each pair."""
    out = []
    for key, lab, when, kind in conds:
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
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = LT.load_india(); LT.activate(px)
    audits = {}
    for test in TESTS:                                                   # the test must be locked and run; its evidence (earlier data) is not read
        ev = os.path.join(ROOT, "tests", test, "evidence", RUN)
        if not os.path.exists(os.path.join(ev, "trade_extras.csv")):
            sys.exit(f"{test}: no run at {ev}; E03 runs only after the test has run")
        a = json.load(open(os.path.join(ev, "audit.json"), encoding="utf-8"))
        audits[test] = {"run_date": a.get("run_date"), "ratio": a.get("ratio", "log"), "evidence_read": False,
                        "why": "the evidence is on the earlier data; the 52-week trades are the rule computed here on the latest data, with the switch on the gold pairs"}
    # ---------------------------------------------------------------- Part A
    prof, A_chk, win = [], {}, {}
    for pid, pair in enumerate(PAIRS):
        rows, chk, first, last = profiles(pair, pid, px, D); prof += rows; A_chk[pair] = chk; win[pair] = (first, last)
        print(f"  A {pair:18s} {first} → {last}  z check {chk:.1e}  [{time.time() - t0:4.0f}s]", flush=True)
    if max(A_chk.values()) > 1e-9:
        sys.exit(f"Part A: the real path's z is not E01's z: {A_chk}")
    PR = pd.DataFrame(prof); PR.to_csv(os.path.join(OUT, "z_profiles.csv"), index=False)
    # Part A for the other z windows (the 52-week rows are the ones above; the general code is checked against them)
    profw, winw = [], {}
    for pid, pair in enumerate(PAIRS):
        for w, lab, f, n_ in ZW.WINDOWS:
            if w == ZW.DEFAULT and pair != PAIRS[0]:
                continue
            rows_w, f_, l_ = profiles_win(pair, pid, px, D, w); winw[(pair, w)] = (f_, l_)
            if w == ZW.DEFAULT:                                                 # the general code must reproduce profiles() exactly
                a_ = pd.DataFrame(rows_w).set_index(["horizon", "bucket"])["spread_mean"]; b_ = PR[PR["pair"] == pair].set_index(["horizon", "bucket"])["spread_mean"]
                if not np.allclose(a_.reindex(b_.index).to_numpy(), b_.to_numpy(), atol=1e-12, equal_nan=True):
                    sys.exit("Part A: the window code does not reproduce the 52-week profiles")
                continue
            profw += rows_w
        print(f"  A windows {pair:18s} [{time.time() - t0:4.0f}s]", flush=True)
    PRW = pd.concat([PR.assign(window=ZW.DEFAULT), pd.DataFrame(profw)], ignore_index=True); PRW.to_csv(os.path.join(OUT, "z_profiles_windows.csv"), index=False)
    for pair in PAIRS:
        winw[(pair, ZW.DEFAULT)] = win[pair]
    # ---------------------------------------------------------------- Part B
    series, zall, PIT = {}, {}, [0, []]
    for pair in PAIRS:
        pf = pair_frame(px, pair, "plain"); series[pair] = condition_series(pf); zall[pair] = series[pair][1]["z"].dropna()
        c_, b_ = cond_pit_check(pf); PIT[0] += c_; PIT[1] += [f"{pair}: {x}" for x in b_]
    if PIT[1] or PIT[0] == 0:
        sys.exit("look-ahead check FAILED: " + "; ".join(PIT[1]) if PIT[1] else "look-ahead check ran nothing")
    print(f"  B look-ahead check: {PIT[0]} checks, 0 violations")
    mkt = px[MARKET]["close"].astype(float)
    trades, n_open, opened = [], {}, []
    zc52 = {p: series[p][1][["z", "since_zero", "band_width"]] for p in PAIRS}
    for test in TESTS:
        x_, d_, bad_ = variant_trades(px, ZW.DEFAULT)                   # the latest data: T03's rule computed here (not T03's evidence)
        if bad_:
            sys.exit("the 52-week run fails its check against the engine: " + "; ".join(bad_))
        r, n_open[test], o = trade_rows(test, x_, d_, series, zc52, zall, mkt); trades += r; opened += o
    TR = pd.DataFrame(trades); TR.to_csv(os.path.join(OUT, "trades_conditions.csv"), index=False)
    OP = pd.DataFrame(opened, columns=["test", "pair", "entry_trade", "held_leg", "days_open", "to"])
    # ---------------------------------------------------------------- Part B for every z window, and the correlation periods
    # the T03 rule on each window's band (variants.py); with the 52-week window the trades and the conditions must be the above
    zcw, corr, CPIT = {}, {}, [0, []]
    for pair in PAIRS:
        pf = pair_frame(px, pair, "plain"); corr[pair] = corr_frame(pf)
        c_, b_ = corr_pit_check(pf); CPIT[0] += c_; CPIT[1] += [f"{pair}: {x}" for x in b_]
        for w, *_ in ZW.WINDOWS:
            zcw[(w, pair)] = zcond(pf, w)
        g_ = zcw[(ZW.DEFAULT, pair)]; h_ = zc52[pair].reindex(g_.index)
        if not np.allclose(g_.to_numpy(float), h_.to_numpy(float), rtol=0, atol=1e-12, equal_nan=True):
            sys.exit(f"{pair}: the window code's 52-week z, weeks since a sign change or band width differ from condition_series()")
        for k_, s_ in CP_SAME.items():
            if not np.allclose(corr[pair][k_].to_numpy(), series[pair][1][s_].reindex(corr[pair].index).to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
                sys.exit(f"{pair}: {k_} differs from {s_}")
    if CPIT[1] or CPIT[0] == 0:
        sys.exit("correlation look-ahead check FAILED: " + "; ".join(CPIT[1]) if CPIT[1] else "correlation look-ahead check ran nothing")
    trw, opw, n_open_w = [], [], {}
    for w, *_ in ZW.WINDOWS:
        x_, d_, bad_ = variant_trades(px, w)
        if bad_:
            sys.exit("the 52-week run fails its check against the engine: " + "; ".join(bad_))
        r, n_open_w[w], o = trade_rows(MAIN, x_, d_, series, {p: zcw[(w, p)] for p in PAIRS}, {p: zcw[(w, p)]["z"].dropna() for p in PAIRS}, mkt, corr)
        trw += [dict(rr, window=w) for rr in r]; opw += [dict(oo, window=w) for oo in o]
        print(f"  B window {w:5s} {len(r):4d} closed trades, {n_open_w[w]} open  [{time.time() - t0:4.0f}s]", flush=True)
    TRW = pd.DataFrame(trw); OPW = pd.DataFrame(opw, columns=["test", "pair", "entry_trade", "held_leg", "days_open", "to", "window"])
    a_ = TRW[TRW["window"] == ZW.DEFAULT][TR.columns].sort_values(["pair", "entry_trade"]).reset_index(drop=True); b_ = TR.sort_values(["pair", "entry_trade"]).reset_index(drop=True)
    num_ = [c for c in TR.columns if pd.api.types.is_float_dtype(TR[c])]
    if len(a_) != len(b_) or not a_.drop(columns=num_).equals(b_.drop(columns=num_)) or not np.allclose(a_[num_].to_numpy(float), b_[num_].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True):
        sys.exit("Part B: the window code's 52-week trade rows (conditions included) differ from condition_series()'s")
    print(f"  B the window code's 52-week {len(a_)} trade rows equal condition_series()'s, conditions included; correlation look-ahead check: {CPIT[0]} checks, 0 violations")
    TRW.to_csv(os.path.join(OUT, "trades_conditions_windows.csv"), index=False)
    # trade lengths per test and pair: calendar days from the entry fill to the exit fill (closed trades); open ones separately
    def durations(T_, O_, label):
        out = []
        for p in ORDER:
            s_ = T_[T_["pair"] == p]; o_ = O_[O_["pair"] == p]; wk = series[p][0]
            out.append({"test": MAIN, "window": label, "pair": p, "first": wk.index[0].date().isoformat(), "last": wk.index[-1].date().isoformat(), "closed": len(s_), "open": len(o_),
                        "median_days": float(s_["days_held"].median()) if len(s_) else np.nan, "mean_days": float(s_["days_held"].mean()) if len(s_) else np.nan,
                        "longest_days": int(s_["days_held"].max()) if len(s_) else 0, "over_52": int((s_["weeks_held"] > 52).sum()),
                        "per_year": len(s_) / ((wk.index[-1] - wk.index[0]).days / 365.25), "open_days": int(o_["days_open"].max()) if len(o_) else 0})
        return out
    DUR = pd.DataFrame(durations(TR[TR["test"] == MAIN], OP[OP["test"] == MAIN], ZW.DEFAULT)).drop(columns="window"); DUR.to_csv(os.path.join(OUT, "trade_durations.csv"), index=False)
    DURW = pd.DataFrame([r for w, *_ in ZW.WINDOWS for r in durations(TRW[TRW["window"] == w], OPW[OPW["window"] == w], w)]); DURW.to_csv(os.path.join(OUT, "trade_durations_windows.csv"), index=False)
    stats = {}
    for ti, (test, pool) in enumerate(POOLS):
        sub = TR[(TR["test"] == test) & (TR["pool"] == pool)]
        stats[(test, pool)] = assoc(sub, np.random.default_rng([SEED_PERM, TESTS.index(test), 0 if pool == "index" else 1]), P)
        print(f"  B {test} {pool:5s} {len(sub):3d} trades  [{time.time() - t0:4.0f}s]", flush=True)
    CS = pd.DataFrame([{"test": t, "pool": p, **s} for (t, p), rows in stats.items() for s in rows]); CS.to_csv(os.path.join(OUT, "conditions_summary.csv"), index=False)
    # per window: the conditions (seed [105, window_id, pool_id]; the 52-week window is window_id 0, so its statistics are the above),
    # and the correlation at entry for every period (seed [106, window_id, pool_id]; the 26- and 52-week periods are CONDS' corr26 / corr52)
    statsW, statsC = {}, {}; new_c = tuple(c for c in CP_CONDS if c[0] not in CP_SAME)
    for wi, (w, *_) in enumerate(ZW.WINDOWS):
        for pi, pool in enumerate(("index", "gold")):
            sub = TRW[(TRW["window"] == w) & (TRW["pool"] == pool)]
            statsW[(w, pool)] = stats[(MAIN, pool)] if w == ZW.DEFAULT else assoc(sub, np.random.default_rng([SEED_PERM, wi, pi]), P)
            got = {s["key"]: s for s in assoc(sub, np.random.default_rng([SEED_CORR, wi, pi]), P, new_c)}; base = {s["key"]: s for s in statsW[(w, pool)]}
            for k_, s_ in CP_SAME.items():
                got[k_] = {**base[s_], "key": k_, "label": next(c[1] for c in CP_CONDS if c[0] == k_)}
            statsC[(w, pool)] = [got[c[0]] for c in CP_CONDS]
        print(f"  B window {w:5s} statistics  [{time.time() - t0:4.0f}s]", flush=True)
    CSW = pd.DataFrame([{"window": w, "pool": p, **s} for (w, p), rows in statsW.items() for s in rows]); CSW.to_csv(os.path.join(OUT, "conditions_summary_windows.csv"), index=False)
    CSC = pd.DataFrame([{"window": w, "pool": p, **s} for (w, p), rows in statsC.items() for s in rows]); CSC.to_csv(os.path.join(OUT, "corr_periods_summary.csv"), index=False)
    PP = TR.groupby(["test", "pair"]).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    HOLDS = ["up to 13 weeks", "14–52 weeks", "over 52 weeks"]
    by_exit = TR.assign(hold=pd.Categorical(TR["hold"], HOLDS, ordered=True)).groupby(["test", "pool", "hold"], observed=True).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    WORD = pd.Categorical(TRW["window"], [w[0] for w in ZW.WINDOWS], ordered=True)   # the dropdown's order, not alphabetical
    PPW = TRW.assign(window=WORD).groupby(["window", "pair"], observed=True).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    by_exit_w = TRW.assign(window=WORD, hold=pd.Categorical(TRW["hold"], HOLDS, ordered=True)).groupby(["window", "pool", "hold"], observed=True).agg(n=("extra", "size"), good=("good", "mean"), median_extra=("extra", "median"), mean_extra=("extra", "mean"), weeks=("weeks_held", "median")).reset_index()
    # notable counts
    A_p = [v for (v, b) in zip(PR["spread_mean_pct"].tolist() + PR["spread_median_pct"].tolist(), PR["bucket"].tolist() * 2) if v == v and b != "all weeks"]
    B_p = [v for v in CS["diff_pct"].tolist() + CS["rho_pct"].tolist() if v == v]
    nA, nB = sum(E1.notable(v) for v in A_p), sum(E1.notable(v) for v in B_p)
    B_pw = {w: [v for v in CSW[CSW["window"] == w]["diff_pct"].tolist() + CSW[CSW["window"] == w]["rho_pct"].tolist() if v == v] for w, *_ in ZW.WINDOWS}
    nBw = {w: sum(E1.notable(v) for v in B_pw[w]) for w in B_pw}
    C_pw = {w: [v for v in CSC[CSC["window"] == w]["diff_pct"].tolist() + CSC[CSC["window"] == w]["rho_pct"].tolist() if v == v] for w, *_ in ZW.WINDOWS}
    nCw = {w: sum(E1.notable(v) for v in C_pw[w]) for w in C_pw}
    assert nBw[ZW.DEFAULT] == nB and len(B_pw[ZW.DEFAULT]) == len(B_p)
    sw_n = int(sum(len(VR.run_variant(px, p, ZW.DEFAULT).get("switch_trades", [])) for p in PAIRS if GOLD in p))
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "data": LT.NOTE, "inputs": fingerprints(), "tests": audits, "setting": {"window": WIN, "exit": EXIT, "run": RUN},
             "switch": {"gold_pairs": [p for p in PAIRS if GOLD in p], "gold_liquid_switches_52wk": sw_n,
                        "rule": "on a gold-held week whose decision falls in gold's weekly supertrend SELL, the gold leg is the liquid ETF; extra = (gold or liquid, compounded) − the peer"},
             "part_A": {"draws": D, "seed": f"default_rng([{E1.SEED_SIGNS}, pair_id]) (E01's signs)", "random_legs": "M = (ln a + ln b) / 2 kept real; a* = exp(M + ln r* / 2), b* = exp(M - ln r* / 2)",
                        "z_equals_E01_max_abs_diff": max(A_chk.values()), "horizons": HORIZONS, "buckets": [b[0] for b in BUCKETS], "notable": [nA, len(A_p)]},
             "part_B": {"perms": P, "seed": f"default_rng([{SEED_PERM}, test_id, pool_id])", "conditions": [c[0] for c in CONDS], "look_ahead_check": {"checks": PIT[0], "violations": PIT[1]},
                        "open_trades_left_out": n_open, "max_hold": "none (T03)", "notable": [nB, len(B_p)]},
             "part_B_windows": {"rule": "T03's rule on each z window's band (explore/E01_ratio_cycles/variants.py), decisions weekly, through the engine's book, 20.8 bps",
                                "check_52_week": "on the latest data: the 52-week run's decisions equal the engine's decide() on every pair, its book with the switch off equals the engine's book (gold pairs), and the window code's trade rows (conditions included) equal condition_series()'s",
                                "seed": f"default_rng([{SEED_PERM}, window_id, pool_id]), window_id = the index in {[w[0] for w in ZW.WINDOWS]}",
                                "windows": {w: {"closed_trades": int((TRW["window"] == w).sum()), "open_trades": n_open_w[w], "notable": [nBw[w], len(B_pw[w])]} for w, *_ in ZW.WINDOWS}},
             "corr_periods": {"periods": [c[0] for c in CWINS], "default": CDEFAULT, "seed": f"default_rng([{SEED_CORR}, window_id, pool_id]); cw26 and cw52 reuse corr26 / corr52",
                              "look_ahead_check": {"checks": CPIT[0], "violations": CPIT[1]}, "notable": {w: [nCw[w], len(C_pw[w])] for w in C_pw}},
             "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return f"**{v:.1f}**" if E1.notable(v) else E1.f2(v, 1)
    md = [f"# E03 results — when does the ratio switch work? (EXPLORATORY)\n",
          f"_Generated by `explore/E03_trade_conditions/conditions.py` on {audit['run_date']} with {MAIN}'s rule computed here at window {WIN}, exit {EXIT}, 20.8 bps, no maximum hold; Part A against {D:,} random paths per pair, Part B against {P:,} permutations. Description, not a finding. Percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Data:** {LT.NOTE}. On the gold pairs the held gold leg was the liquid ETF on {sw_n} switches at the 52-week window; a trade's extra uses what was held. {MAIN}'s own evidence is on the earlier data and is not read.\n",
          f"**Against chance:** Part A {nA} of {len(A_p)} percentiles notable (about {0.1 * len(A_p):.0f} by chance); Part B {nB} of {len(B_p)} (about {0.1 * len(B_p):.0f} by chance).\n",
          "## Part A — forward returns by z bucket (cheaper leg = the one that has lagged)\n",
          "Per pair, bucket and horizon: weeks (episodes), cheaper leg mean / median, richer leg mean / median, spread = cheaper − richer mean (random walks' median, percentile) / median (percentile), both legs' average. Window after the 52-week warm-up.\n"]
    for p in ORDER:
        md += [f"### {nice(p)} · {win[p][0]} → {win[p][1]}\n", "| horizon | z bucket | weeks (episodes) | cheaper mean / median | richer mean / median | spread mean (rw, pct) | spread median (pct) | both legs mean |", "|---|---|---|---|---|---|---|---|"]
        for _, r in PR[PR["pair"] == p].iterrows():
            md.append(f"| {r['horizon']} wk | {r['bucket']} | {r['weeks']} ({r['episodes']}) | {E1.fp(r['cheap_mean'])} / {E1.fp(r['cheap_median'])} | {E1.fp(r['rich_mean'])} / {E1.fp(r['rich_median'])} | {E1.fp(r['spread_mean'])} ({E1.fp(r['spread_mean_rw'])}, {pq(r['spread_mean_pct'])}) | {E1.fp(r['spread_median'])} ({pq(r['spread_median_pct'])}) | {E1.fp(r['both_mean'])} |")
        md.append("")
    md += ["## Part A for every z window — the spread's mean at the middle horizon (≈13 weeks; daily: 65 trading days), stretched buckets (random walks' median, percentile)\n",
           "| pair | " + " | ".join(f"{lab}{' (daily)' if f == 'D' else ''}" for _, lab, f, _ in ZW.WINDOWS) + " |", "|---|" + "---|" * len(ZW.WINDOWS)]
    for p in ORDER:
        cells = []
        for w, lab, f, n_ in ZW.WINDOWS:
            kk = 13 if f == "W" else 65; s_ = PRW[(PRW["window"] == w) & (PRW["pair"] == p) & (PRW["horizon"] == kk)].set_index("bucket")
            cells.append("; ".join(f"{b_} {s_.loc[b_, 'spread_mean'] * 100:+.1f}% ({pq(s_.loc[b_, 'spread_mean_pct'])})" for b_ in ("< −2", "−2 to −1", "1 to 2", "> 2")))
        md.append(f"| {nice(p)} | " + " | ".join(cells) + " |")
    md += ["", "## Part B — the market's condition at entry, good trades (extra > 0) vs bad\n"]
    for test, pool in POOLS:
        rows = stats[(test, pool)]; sub = TR[(TR["test"] == test) & (TR["pool"] == pool)]
        md += [f"### {test}, {pool} pairs · {len(sub)} closed trades, {int(sub['good'].sum())} good\n",
               "| condition | when | n | median, good trades | median, bad trades | good − bad (pct) | rank correlation with the extra (pct) |", "|---|---|---|---|---|---|---|"]
        md += [f"| {s['label']} | {s['when']} | {s['n']} | {fmtv(s['kind'], s['med_good'])} | {fmtv(s['kind'], s['med_bad'])} | {fmtc(s['kind'], s['diff'])} ({pq(s['diff_pct'])}) | {E1.f2(s['rho'])} ({pq(s['rho_pct'])}) |" for s in rows]
        md.append("")
    md += ["## Trades by the length of the hold\n", "| test | pool | held | trades | share good | median extra | mean extra | median weeks held |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {r['pool']} | {r['hold']} | {r['n']} | {r['good'] * 100:.0f}% | {E1.fp(r['median_extra'])} | {E1.fp(r['mean_extra'])} | {r['weeks']:.0f} |" for _, r in by_exit.iterrows()]
    md += ["\n## Trades per pair\n", "| test | pair | closed trades | share good | median extra | mean extra | median weeks held |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['test']} | {nice(r['pair'])} | {r['n']} | {r['good'] * 100:.0f}% | {E1.fp(r['median_extra'])} | {E1.fp(r['mean_extra'])} | {r['weeks']:.0f} |" for _, r in PP.iterrows()]
    md += ["\n## Trade lengths (calendar days from the entry fill to the exit fill)\n", "| test | pair | data window | closed trades | trades a year | median days | mean days | longest | held over 52 weeks | open at the end (days so far) |", "|---|---|---|---|---|---|---|---|---|---|"]
    def open_txt(r): return f"{r['open']} ({r['open_days']})" if r["open"] else "0"
    md += [f"| {r['test']} | {nice(r['pair'])} | {r['first']} → {r['last']} | {r['closed']} | {r['per_year']:.1f} | {E1.f2(r['median_days'], 0)} | {E1.f2(r['mean_days'], 0)} | {r['longest_days']} | {r['over_52']} | {open_txt(r)} |" for _, r in DUR.iterrows()]
    md += ["\n## Part B for every z window — T03's rule on each window's band (the 52-week rows are T03 itself)\n",
           f"Decisions weekly, fills next session, 20.8 bps round trip, no maximum hold; daily windows read the daily z at each decision day. Data windows per pair as in the trade-length table above. Conditions as above, with z, |z|, weeks since z changed sign, band width and breadth taken from the chosen window; {P:,} permutations, seed default_rng([{SEED_PERM}, window_id, pool_id]).\n",
           "| z window | pool | closed trades | share good | median extra | mean extra | median weeks held | notable percentiles | notable good − bad differences |", "|---|---|---|---|---|---|---|---|---|"]
    def words_w(w, pool, md_=False):
        out = []
        for s in statsW[(w, pool)]:
            if E1.notable(s["diff_pct"]):
                out.append(f"{s['label']}{' (during)' if s['when'] == 'during' else ''}: good {fmtv(s['kind'], s['med_good'])} vs bad {fmtv(s['kind'], s['med_bad'])} (percentile {s['diff_pct']:.0f})")
        return "; ".join(out) if out else ("none" if md_ else "no condition's good-vs-bad difference is notable")
    for w, wlab, f, _ in ZW.WINDOWS:
        for pool in ("index", "gold"):
            s_ = TRW[(TRW["window"] == w) & (TRW["pool"] == pool)]; nn = [v for s in statsW[(w, pool)] for v in (s["diff_pct"], s["rho_pct"]) if v == v]
            md.append(f"| {wlab}{' (daily)' if f == 'D' else ''} | {pool} | {len(s_)} | {s_['good'].mean() * 100:.0f}% | {E1.fp(s_['extra'].median())} | {E1.fp(s_['extra'].mean())} | {s_['weeks_held'].median():.0f} | {sum(E1.notable(v) for v in nn)} of {len(nn)} | {words_w(w, pool, True)} |")
    md += ["\n### Trades by the length of the hold, every z window\n", "| z window | pool | held | trades | share good | median extra | mean extra | median weeks held |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {ZW.WIN[r['window']][1]}{' (daily)' if ZW.WIN[r['window']][2] == 'D' else ''} | {r['pool']} | {r['hold']} | {r['n']} | {r['good'] * 100:.0f}% | {E1.fp(r['median_extra'])} | {E1.fp(r['mean_extra'])} | {r['weeks']:.0f} |" for _, r in by_exit_w.iterrows()]
    md += ["\n### Closed trades (median calendar days held) per pair, every z window\n", "| pair | data window | " + " | ".join(f"{lab}{' (daily)' if f == 'D' else ''}" for _, lab, f, _ in ZW.WINDOWS) + " |", "|---|---|" + "---|" * len(ZW.WINDOWS)]
    for p in ORDER:
        d0 = DURW[DURW["pair"] == p].set_index("window")
        md.append(f"| {nice(p)} | {d0.iloc[0]['first']} → {d0.iloc[0]['last']} | " + " | ".join(f"{d0.loc[w, 'closed']} ({E1.f2(d0.loc[w, 'median_days'], 0)})" for w, *_ in ZW.WINDOWS) + " |")
    md += ["\n## The legs' correlation at entry, by correlation period and z window — good − bad median (percentile) · rank correlation with the extra (percentile)\n",
           f"Weekly periods: weekly log returns on decision weeks; daily periods: daily log returns on the pair's common trading days, read at the entry decision. Seed default_rng([{SEED_CORR}, window_id, pool_id]); the 26- and 52-week rows are the conditions above.\n",
           "| z window | pool | " + " | ".join(f"{lab}{' (daily)' if f == 'D' else ''}" for _, lab, f, _ in CWINS) + " |", "|---|---|" + "---|" * len(CWINS)]
    for w, wlab, f, _ in ZW.WINDOWS:
        for pool in ("index", "gold"):
            md.append(f"| {wlab}{' (daily)' if f == 'D' else ''} | {pool} | " + " | ".join(f"{fmtc('num', s['diff'])} ({pq(s['diff_pct'])}) · {E1.f2(s['rho'])} ({pq(s['rho_pct'])})" for s in statsC[(w, pool)]) + " |")
    open(os.path.join(OUT, "E03_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- figures
    figs = {}; titles = {"prof": {}, "tl": {}, "days": {}}
    # A1. per pair: mean forward return by bucket, three horizons
    fA = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.09, subplot_titles=["≈4 weeks ahead (daily windows: 20 trading days)", "≈13 weeks ahead (65 trading days)", "≈26 weeks ahead (130 trading days)"])
    for w, wlab, f, n_ in ZW.WINDOWS:
        hz = HORIZONS if f == "W" else HORIZONS_D; per = "weeks" if f == "W" else "days"
        for p in ORDER:
            vis = (p == DEFAULT_PAIR) and (w == ZW.DEFAULT); sub = PRW[(PRW["window"] == w) & (PRW["pair"] == p) & (PRW["bucket"] != "all weeks")]; mw = p + ZW.SEP + w
            for i, k in enumerate(hz):
                s_ = sub[sub["horizon"] == k].set_index("bucket").reindex(CHART_BUCKETS); lead = i == 0
                hv = [f"{b_}: {wk_} {per}, {e_} episodes" for b_, wk_, e_ in zip(CHART_BUCKETS, s_["weeks"], s_["episodes"])]
                fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s_["cheap_mean"] * 100, marker={"color": AQUA}, name="cheaper leg (has lagged)", meta=mw, visible=vis, showlegend=lead, legendgroup="c", customdata=hv, hovertemplate="%{customdata}<br>cheaper leg %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
                fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s_["rich_mean"] * 100, marker={"color": ORANGE}, name="richer leg", meta=mw, visible=vis, showlegend=lead, legendgroup="r", customdata=hv, hovertemplate="%{customdata}<br>richer leg %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
                fA.add_trace(go.Bar(x=CHART_BUCKETS, y=s_["spread_mean"] * 100, marker={"color": BLUE}, name="spread (cheaper − richer)", meta=mw, visible=vis, showlegend=lead, legendgroup="s", customdata=[[v] for v in s_["spread_mean_pct"]], hovertemplate="spread %{y:+.1f}% · percentile %{customdata[0]:.1f}<extra></extra>"), row=i + 1, col=1)
                fA.add_trace(go.Scatter(x=CHART_BUCKETS, y=s_["spread_mean_rw"] * 100, mode="markers", marker={"color": MUTED, "size": 7, "symbol": "line-ew-open", "line": {"width": 2}}, name="random walks' spread: median and 5th–95th", meta=mw, visible=vis, showlegend=lead, legendgroup="n",
                                        error_y={"type": "data", "symmetric": False, "array": (s_["spread_mean_hi"] - s_["spread_mean_rw"]) * 100, "arrayminus": (s_["spread_mean_rw"] - s_["spread_mean_lo"]) * 100, "color": MUTED, "thickness": 1.2, "width": 6},
                                        hovertemplate="random walks: median %{y:+.1f}%<extra></extra>"), row=i + 1, col=1)
            f0, l0 = winw[(p, w)]
            titles["prof"][mw] = E1.two_line(f"{nice(p)} · z window {wlab}{' (daily closes)' if f == 'D' else ''} · {f0} → {l0} · mean forward return of each leg by the ratio's z-score",
                                             "cheaper leg = the one that has lagged (a when z < 0, b when z > 0); the grey mark is what the random walks give the spread (their median, 5th–95th)")
    fA.update_layout(barmode="group", bargap=0.25)
    for i in range(3):
        fA.update_yaxes(ticksuffix="%", title={"text": "forward return", "font": {"color": INK2}}, row=i + 1, col=1); fA.add_hline(y=0, line={"color": AXIS, "width": 1}, row=i + 1, col=1)
    fA.update_xaxes(title={"text": "z-score bucket at the start (z < −1 or > 1 = outside the ±1 sd band)", "font": {"color": INK2}}, row=3, col=1)
    figs["prof"] = E1.style(fA, titles["prof"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 900, legend_top=True)
    # A2. all pairs: percentile of the spread's mean among the random walks, bucket x horizon, one heat map per window
    fHp = go.Figure(); titles["h_prof"] = {}; a_notable = {}
    for w, wlab, f, n_ in ZW.WINDOWS:
        hz = HORIZONS if f == "W" else HORIZONS_D; per = "weeks" if f == "W" else "days"; unit = "wk" if f == "W" else "d"
        cols = [f"{b_} · {k} {unit}" for k in hz for b_ in CHART_BUCKETS]; zv, tx, hv = [], [], []; cnt = [0, 0]
        for p in ORDER:
            sub = PRW[(PRW["window"] == w) & (PRW["pair"] == p)].set_index(["horizon", "bucket"]); zz, tt, hh = [], [], []
            for k in hz:
                for b_ in CHART_BUCKETS:
                    r = sub.loc[(k, b_)]; okr = r["weeks"] >= 8; zz.append(r["spread_mean_pct"] if okr else float("nan"))
                    if okr:
                        cnt[1] += 1; cnt[0] += int(E1.notable(r["spread_mean_pct"]))
                    tt.append(f"{r['spread_mean'] * 100:+.1f}%" if (okr and E1.notable(r["spread_mean_pct"])) else "")
                    hh.append(f"{r['weeks']} {per} ({r['episodes']} episodes): spread {r['spread_mean'] * 100:+.1f}% (random walks {r['spread_mean_rw'] * 100:+.1f}%), percentile {r['spread_mean_pct']:.1f}" if okr else f"{r['weeks']} {per}: too few")
            zv.append(zz); tx.append(tt); hv.append(hh)
        a_notable[w] = tuple(cnt)
        fHp.add_trace(go.Heatmap(z=zv, x=cols, y=[nice(p) for p in ORDER], zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=tx, texttemplate="%{text}", customdata=hv, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
                                 meta=ZW.SEP + w, visible=(w == ZW.DEFAULT), colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12}))
        titles["h_prof"][w] = E1.two_line(f"The spread's mean forward return against the random walks, by z bucket and horizon (all pairs) · z window {wlab}{' (daily closes)' if f == 'D' else ''}",
                                          f"red = the cheaper leg beat the richer leg by more than a random walk gives; blue = by less; labels on notable cells ({cnt[0]} of {cnt[1]}); blank = fewer than 8 periods")
    fHp.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fHp.update_xaxes(showgrid=False, side="bottom", tickangle=-45, automargin=True)
    for i in (1, 2):
        fHp.add_vline(x=i * 6 - 0.5, line={"color": AXIS, "width": 1})
    figs["h_prof"] = E1.style(fHp, titles["h_prof"][ZW.DEFAULT], 620, legend=False, left=130)
    # B1. per pair: the trades on the two legs, for every z window (the legs are the same for every window; the trades are the window's)
    def ds(d):
        """Dates as YYYY-MM-DD strings (a quarter of the size of plotly's timestamps on the page)."""
        return pd.Timestamp(d).strftime("%Y-%m-%d") if isinstance(d, (pd.Timestamp, str)) else [x.strftime("%Y-%m-%d") for x in pd.DatetimeIndex(d)]
    def wname(w):
        _, lab, f, _ = ZW.WIN[w]; return f"{lab}{' (daily z)' if f == 'D' else ''}" + (f" = {MAIN} itself" if w == ZW.DEFAULT else "")
    def hold_shapes(sub, ylo, yhi, hover):
        """Shaded holds (one filled trace per outcome) and a hover marker per trade."""
        out = []
        for good, colr, lab in ((True, AQUA, "good trade (extra > 0)"), (False, ORANGE, "bad trade")):
            tt = sub[sub["good"] == good]; xs, ys, hx, hy, hc = [], [], [], [], []
            for _, t in tt.iterrows():
                e, x_ = pd.Timestamp(t["entry_trade"]), pd.Timestamp(t["exit_trade"])
                xs += [ds(e), ds(x_), ds(x_), ds(e), ds(e), None]; ys += [ylo, ylo, yhi, yhi, ylo, None]; hx.append(ds(e + (x_ - e) / 2)); hy.append(yhi); hc.append(hover(t))
            out.append((go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=colr, opacity=0.22, line={"width": 0}, name=lab, legendgroup=lab, hoverinfo="skip"),
                        go.Scatter(x=hx, y=hy, mode="markers", marker={"color": colr, "size": 7, "symbol": "triangle-down"}, showlegend=False, customdata=hc, hovertemplate="%{customdata}<extra></extra>")))
        return out
    def tl_hover(t):
        return (f"held {t['held_leg']} {t['entry_trade']} → {t['exit_trade']} ({t['weeks_held']} wk, exit: {t['exit_reason']})<br>extra {t['extra'] * 100:+.1f}% · held leg {t['held_ret'] * 100:+.1f}%, peer {t['peer_ret'] * 100:+.1f}%"
                f"<br>at entry: z {t['z_entry']:+.2f}, corr 52 wk {t['corr52']:.2f}, N100 52-wk {t['mkt52'] * 100:+.0f}%, drawdown {t['mkt_dd'] * 100:.0f}%")
    fT = go.Figure()
    for p in ORDER:
        vis_p = p == DEFAULT_PAIR; wk = series[p][0]; la, lb = p.split("|"); reb_a = wk["ca"] / wk["ca"].iloc[0] * 100; reb_b = wk["cb"] / wk["cb"].iloc[0] * 100
        ylo, yhi = float(min(reb_a.min(), reb_b.min())) * 0.9, float(max(reb_a.max(), reb_b.max())) * 1.1
        for name, s in ((la, reb_a), (lb, reb_b)):
            fT.add_trace(go.Scatter(x=ds(s.index), y=np.round(s.to_numpy(), 2), mode="lines", line={"color": E1.CAT[name], "width": 1.4}, name=name, legendgroup=name, meta=p, visible=vis_p, hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:.0f}<extra></extra>"))
        for w, *_ in ZW.WINDOWS:
            mw = p + ZW.SEP + w; sub = TRW[(TRW["window"] == w) & (TRW["pair"] == p)]
            for shade, mark in hold_shapes(sub, ylo, yhi, tl_hover):
                fT.add_trace(shade.update(meta=mw, visible=vis_p and w == ZW.DEFAULT)); fT.add_trace(mark.update(meta=mw, visible=vis_p and w == ZW.DEFAULT))
            n2 = len(sub); g2 = sub["good"].mean() if n2 else float("nan")
            titles["tl"][mw] = E1.two_line(f"{nice(p)} · z window {wname(w)} · the trades over the two legs (rebased to 100 at {wk.index[0].date()}, data to {wk.index[-1].date()}) · {n2} closed trades, {E1.f2(g2 * 100, 0)}% good",
                                           "shaded = a trade's hold, aqua if the held leg beat the peer (extra > 0), orange if not; hover the marker for the trade and the market at its entry")
    fT.update_yaxes(type="log", title={"text": "rebased", "font": {"color": INK2}})
    figs["tl"] = E1.style(fT, titles["tl"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 600, legend_top=True)
    # B1b. per pair: each trade's length in days, as a stem at its entry date, and the trades entered each year
    DAY_MS = 86400000; titles["days"] = {}
    fD = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08,
                       subplot_titles=("calendar days held, at the trade's entry date (label = days)", "trades entered each year (an open trade counts in its entry year)"))
    for p in ORDER:
        wk = series[p][0]
        for w, *_ in ZW.WINDOWS:
            vis = p == DEFAULT_PAIR and w == ZW.DEFAULT; mw = p + ZW.SEP + w
            sub = TRW[(TRW["window"] == w) & (TRW["pair"] == p)].sort_values("entry_trade"); op = OPW[(OPW["window"] == w) & (OPW["pair"] == p)]
            for good, colr, lab in ((True, AQUA, "good trade (extra > 0)"), (False, ORANGE, "bad trade")):
                tt = sub[sub["good"] == good]
                fD.add_trace(go.Bar(x=tt["entry_trade"].tolist(), y=tt["days_held"], width=10 * DAY_MS, marker={"color": colr, "line": {"width": 0}}, text=tt["days_held"], textposition="outside", textangle=0, textfont={"size": 10}, constraintext="none",
                                    cliponaxis=False, name=lab, legendgroup=lab, meta=mw, visible=vis,
                                    customdata=[[a, b, wh, e, x * 100] for a, b, wh, e, x in zip(tt["entry_trade"], tt["exit_trade"], tt["weeks_held"], tt["exit_reason"], tt["extra"])],
                                    hovertemplate="%{customdata[0]} → %{customdata[1]}: %{y} days (%{customdata[2]} decision weeks)<br>exit: %{customdata[3]} · extra %{customdata[4]:+.1f}%<extra></extra>"), row=1, col=1)
            if len(op):
                fD.add_trace(go.Bar(x=op["entry_trade"].tolist(), y=op["days_open"], width=10 * DAY_MS, marker={"color": MUTED}, text=[f"{d} (open)" for d in op["days_open"]], textposition="outside", textangle=0, textfont={"size": 10}, constraintext="none",
                                    cliponaxis=False, name="still open at the end", legendgroup="open", meta=mw, visible=vis, customdata=[[a, t_] for a, t_ in zip(op["entry_trade"], op["to"])],
                                    hovertemplate="%{customdata[0]} → still open on %{customdata[1]}: %{y} days so far<extra></extra>"), row=1, col=1)
            years = range(wk.index[0].year, wk.index[-1].year + 1); cnt = pd.to_datetime(pd.concat([sub["entry_trade"], op["entry_trade"]])).dt.year.value_counts().reindex(years, fill_value=0)   # open trades included
            fD.add_trace(go.Bar(x=[f"{y}-07-01" for y in years], y=cnt.to_numpy(), width=300 * DAY_MS, marker={"color": BLUE}, text=cnt.to_numpy(), textposition="outside", textfont={"size": 10}, cliponaxis=False,
                                name="trades entered that year", legendgroup="yr", meta=mw, visible=vis, hovertemplate="%{x|%Y}: %{y} trades entered<extra></extra>"), row=2, col=1)
            d_ = DURW[(DURW["window"] == w) & (DURW["pair"] == p)].iloc[0]; open_s = f" + {d_['open']} open" if d_["open"] else ""
            titles["days"][mw] = E1.two_line(f"{nice(p)} · z window {wname(w)} · {d_['closed']} closed trades{open_s} · data {d_['first']} → {d_['last']} · {d_['per_year']:.1f} trades a year · median {E1.f2(d_['median_days'], 0)} days, longest {d_['longest_days']}",
                                             f"each stem is one trade at its entry date, its height and label the calendar days from entry to exit; aqua = the held leg beat the peer, orange = it did not; {d_['over_52']} held over 52 weeks")
    fD.update_yaxes(title={"text": "days held", "font": {"color": INK2}}, rangemode="tozero", row=1, col=1); fD.update_yaxes(title={"text": "trades", "font": {"color": INK2}}, rangemode="tozero", row=2, col=1)
    fD.update_layout(bargap=0, barmode="overlay"); figs["days"] = E1.style(fD, titles["days"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT], 640, legend_top=True)
    # B2. conditions heat map per z window: rows conditions, columns pool x statistic
    def assoc_heat(rows_of, conds, ylabels):
        """One heat map's z, text and hover: rows = conds, columns = (index, gold) x (good − bad, rank corr)."""
        zv, tx, hv = [], [], []
        for key, lab, when, kind in conds:
            zz, tt, hh = [], [], []
            for pl in ("index", "gold"):
                s = next(x for x in rows_of[pl] if x["key"] == key)
                for st, val, pc in (("good − bad", s["diff"], s["diff_pct"]), ("rank corr", s["rho"], s["rho_pct"])):
                    zz.append(pc); vtxt = fmtc(kind, val) if st == "good − bad" else E1.f2(val)
                    tt.append(vtxt if E1.notable(pc) else ""); hh.append(f"{s['n']} trades: {st} {vtxt}, percentile {pc:.1f}" if pc == pc else f"{s['n']} trades: too few")
            zv.append(zz); tx.append(tt); hv.append(hh)
        return go.Heatmap(z=zv, x=[f"{pl} pairs<br>{st}" for pl in ("index", "gold") for st in ("good − bad", "rank corr")], y=ylabels, zmin=0, zmax=100, colorscale=E1.DIV_LIGHT, text=tx, texttemplate="%{text}", customdata=hv,
                          hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>", xgap=2, ygap=2, colorbar={"title": {"text": "percentile", "side": "right"}, "tickvals": [0, 5, 50, 95, 100], "thickness": 12})
    f = go.Figure(); titles["h_cond"] = {}
    for w, *_ in ZW.WINDOWS:
        f.add_trace(assoc_heat({pl: statsW[(w, pl)] for pl in ("index", "gold")}, CONDS, [f"{lab}{' (during)' if when == 'during' else ''}" for _, lab, when, _ in CONDS]).update(meta=ZW.SEP + w, visible=w == ZW.DEFAULT))
        ni, ng = int(((TRW["window"] == w) & (TRW["pool"] == "index")).sum()), int(((TRW["window"] == w) & (TRW["pool"] == "gold")).sum())
        titles["h_cond"][w] = E1.two_line(f"Is the condition associated with the trade's outcome? · z window {wname(w)} · {ni} index-pair and {ng} gold-pair trades · {nBw[w]} of {len(B_pw[w])} percentiles notable (about {0.1 * len(B_pw[w]):.0f} by chance)",
                                          "percentile among permutations of the outcomes within each pair; red = good trades had the higher value (or the correlation with the extra is positive), blue = the lower; labels on notable cells")
    f.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); f.update_xaxes(showgrid=False, side="top")
    f.add_vline(x=1.5, line={"color": AXIS, "width": 1})
    figs["h_cond"] = E1.style(f, titles["h_cond"][ZW.DEFAULT], 760, legend=False, left=250)
    # B2b. the legs' correlation at entry for every period: one heat map per z window, rows = periods
    fHc = go.Figure(); titles["h_corr"] = {}
    for w, *_ in ZW.WINDOWS:
        fHc.add_trace(assoc_heat({pl: statsC[(w, pl)] for pl in ("index", "gold")}, CP_CONDS, [c[1] for c in CP_CONDS]).update(meta=ZW.SEP + w, visible=w == ZW.DEFAULT))
        titles["h_corr"][w] = E1.two_line(f"The legs' correlation at entry, every period, against the outcome · z window {wname(w)} · {nCw[w]} of {len(C_pw[w])} percentiles notable (about {0.1 * len(C_pw[w]):.0f} by chance)",
                                          "rows = the correlation period (weekly or daily log returns); percentile among permutations of the outcomes within each pair; red = good trades had the higher correlation, blue = the lower")
    fHc.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticksuffix="  "); fHc.update_xaxes(showgrid=False, side="top"); fHc.add_vline(x=1.5, line={"color": AXIS, "width": 1})
    figs["h_corr"] = E1.style(fHc, titles["h_corr"][ZW.DEFAULT], 430, legend=False, left=250)
    # B2c. per pair: the legs' correlation over time for the chosen period, with the chosen window's trades shaded
    fC = go.Figure(); titles["corr_tl"] = {}
    def c_hover(t):
        return (f"held {t['held_leg']} {t['entry_trade']} → {t['exit_trade']} ({t['weeks_held']} wk) · extra {t['extra'] * 100:+.1f}%<br>correlation at entry: "
                + ", ".join(f"{lab} {t['corr_' + c]:.2f}" for c, lab, _, _ in CWINS[:3]) + "<br>daily: " + ", ".join(f"{lab} {t['corr_' + c]:.2f}" for c, lab, _, _ in CWINS[3:]))
    for p in ORDER:
        vis_p = p == DEFAULT_PAIR; cf = corr[p]; ylo = max(-1.0, float(np.nanmin(cf.to_numpy())) - 0.05); yhi = min(1.0, float(np.nanmax(cf.to_numpy())) + 0.05)
        for w, *_ in ZW.WINDOWS:
            for shade, mark in hold_shapes(TRW[(TRW["window"] == w) & (TRW["pair"] == p)], ylo, yhi, c_hover):
                fC.add_trace(shade.update(meta=p + ZW.SEP + w, visible=vis_p and w == ZW.DEFAULT)); fC.add_trace(mark.update(meta=p + ZW.SEP + w, visible=vis_p and w == ZW.DEFAULT))
        for c, clab, cf_, _ in CWINS:
            s = cf["corr_" + c].dropna()
            fC.add_trace(go.Scatter(x=ds(s.index), y=np.round(s.to_numpy(), 3), mode="lines", line={"color": BLUE, "width": 1.5}, name=f"correlation, trailing {clab}", meta=p + CSEP + c, visible=vis_p and c == CDEFAULT,
                                    hovertemplate="%{x|%Y-%m-%d}<br>correlation %{y:.2f}<extra></extra>"))
            for w, *_ in ZW.WINDOWS:
                sub = TRW[(TRW["window"] == w) & (TRW["pair"] == p)]; g_, b_ = sub[sub["good"]]["corr_" + c].median(), sub[~sub["good"]]["corr_" + c].median()
                titles["corr_tl"][p + ZW.SEP + w + CSEP + c] = E1.two_line(
                    f"{nice(p)} · the legs' correlation, trailing {clab} ({'weekly log returns on decision weeks' if cf_ == 'W' else 'daily log returns, shown at decision days'}) · {s.index[0].date()} → {s.index[-1].date()}",
                    f"shaded = the trades of z window {wname(w)} ({len(sub)} closed; aqua good, orange bad) · median correlation at entry: good trades {E1.f2(g_)}, bad {E1.f2(b_)}")
    fC.update_yaxes(title={"text": "correlation", "font": {"color": INK2}})
    figs["corr_tl"] = E1.style(fC, titles["corr_tl"][DEFAULT_PAIR + ZW.SEP + ZW.DEFAULT + CSEP + CDEFAULT], 480, legend_top=True)
    # B3. scatter: extra vs the key conditions, index pairs, per z window; the correlation panel follows the correlation period
    labs = {c[0]: (c[1], c[3]) for c in CONDS + CP_CONDS}; labs["corr"] = ("legs' correlation at entry (period chosen at the top)", "num")
    fS = make_subplots(rows=2, cols=3, subplot_titles=[labs[k][0] for k in KEY_CONDS], vertical_spacing=0.16, horizontal_spacing=0.07); titles["scatter"] = {}
    for w, *_ in ZW.WINDOWS:
        sub0 = TRW[(TRW["window"] == w) & (TRW["pool"] == "index")]
        for j, k in enumerate(KEY_CONDS):
            row, col = j // 3 + 1, j % 3 + 1
            for ck in ([("corr_" + c, ZW.SEP + w + CSEP + c, c) for c, *_ in CWINS] if k == "corr" else [(k, ZW.SEP + w, None)]):
                key, meta, c = ck; pct = labs[key][1] == "pct"; sub = sub0.dropna(subset=[key])
                fS.add_trace(go.Scatter(x=np.round(sub[key] * (100 if pct else 1), 4), y=np.round(sub["extra"] * 100, 3), mode="markers", marker={"color": BLUE, "size": 7, "opacity": 0.75}, name="closed trades, index pairs", showlegend=j == 0, legendgroup="tr",
                                        meta=meta, visible=(w == ZW.DEFAULT and c in (None, CDEFAULT)),
                                        customdata=[[nice(a), b, e] for a, b, e in zip(sub["pair"], sub["entry_decision"], sub["exit_reason"])], hovertemplate="%{customdata[0]} · entered %{customdata[1]} (exit: %{customdata[2]})<br>" + labs[key][0] + " %{x:.2f}" + ("%" if pct else "") + " · extra %{y:+.1f}%<extra></extra>"), row=row, col=col)
        titles["scatter"][w] = E1.two_line(f"Each trade's extra (held leg − peer) against six conditions at its entry · index pairs · z window {wname(w)} · {len(sub0)} closed trades",
                                           "one dot per closed trade; |z|, band width and breadth use the chosen z window; the correlation panel uses the correlation period chosen at the top")
    for j, k in enumerate(KEY_CONDS):
        row, col = j // 3 + 1, j % 3 + 1; fS.add_hline(y=0, line={"color": AXIS, "width": 1}, row=row, col=col); fS.update_xaxes(ticksuffix="%" if labs[k][1] == "pct" else "", row=row, col=col)
    fS.update_yaxes(ticksuffix="%", title={"text": "extra", "font": {"color": INK2}}, col=1)
    figs["scatter"] = E1.style(fS, titles["scatter"][ZW.DEFAULT], 620, legend_top=True)

    # ---------------------------------------------------------------- the page
    def per_pair(key, fig, tables, first=False) -> str:
        t = html.escape(json.dumps(titles[key]), quote=True)
        def div(k, tb_):
            p, w = (k.split(ZW.SEP) + [None])[:2]; shown = p == DEFAULT_PAIR and (w is None or w == ZW.DEFAULT)
            return f"<div data-pair='{html.escape(p, quote=True)}'" + (f" data-win='{w}'" if w else "") + f"{'' if shown else ' hidden'}>{tb_}</div>"
        tb = "".join(div(k, v) for k, v in tables.items())
        return f"<div class='card per-pair' data-titles='{t}'>{E1.html_fig(fig, first=first)}<details><summary>Table view</summary>{tb}</details></div>"
    def per_win(key, fig, tables=None) -> str:
        tb = "" if not tables else "<details><summary>Table view</summary>" + win_divs(tables) + "</details>"
        return f"<div class='card per-win' data-titles='{html.escape(json.dumps(titles[key]), quote=True)}'>{E1.html_fig(fig)}{tb}</div>"
    def win_divs(d) -> str:
        """Blocks that follow the z window only (any pair)."""
        return "".join(f"<div data-winonly='{w}'{'' if w == ZW.DEFAULT else ' hidden'}>{v}</div>" for w, v in d.items())
    tab_prof = {p + ZW.SEP + w: E1.tbl(["horizon", "z bucket", "periods (episodes)", "cheaper: mean / median", "richer: mean / median", "spread: mean (random walks, pct)", "spread: median (pct)", "both legs: mean"],
                          [[f"{r['horizon']} {'wk' if ZW.WIN[w][2] == 'W' else 'd'}", r["bucket"], f"{r['weeks']} ({r['episodes']})", f"{E1.fp(r['cheap_mean'])} / {E1.fp(r['cheap_median'])}", f"{E1.fp(r['rich_mean'])} / {E1.fp(r['rich_median'])}",
                            f"{E1.fp(r['spread_mean'])} ({E1.fp(r['spread_mean_rw'])}, {E1.fpct(r['spread_mean_pct'])})", f"{E1.fp(r['spread_median'])} ({E1.fpct(r['spread_median_pct'])})", E1.fp(r["both_mean"])]
                           for _, r in PRW[(PRW["window"] == w) & (PRW["pair"] == p)].iterrows()]) for w, _, _, _ in ZW.WINDOWS for p in ORDER}
    def dur_tbl(w):
        return E1.tbl(["pair", "data window", "closed trades", "trades a year", "median days", "mean days", "longest", "held over 52 weeks", "open at the end"],
                      [[nice(r["pair"]), f"{r['first']} → {r['last']}", r["closed"], f"{r['per_year']:.1f}", E1.f2(r["median_days"], 0), E1.f2(r["mean_days"], 0), r["longest_days"], r["over_52"], (f"{r['open']} ({r['open_days']} days so far)" if r["open"] else "0")]
                       for _, r in DURW[DURW["window"] == w].iterrows()])
    dur_t = {w: dur_tbl(w) for w, *_ in ZW.WINDOWS}
    tab_days = {p + ZW.SEP + w: E1.tbl(["entered", "exited", "days", "decision weeks", "exit", "extra"], [[t["entry_trade"], t["exit_trade"], t["days_held"], t["weeks_held"], t["exit_reason"], E1.fp(t["extra"])]
                                                                                                      for _, t in TRW[(TRW["window"] == w) & (TRW["pair"] == p)].sort_values("entry_trade").iterrows()]) + f"<p class='note'>All pairs, z window {wname(w)}:</p>" + dur_t[w]
                for w, *_ in ZW.WINDOWS for p in ORDER}
    tab_tl = {p + ZW.SEP + w: E1.tbl(["entered", "exited", "held", "weeks", "exit", "extra", "held leg", "peer", "z at entry", "corr 52 wk", "N100 52-wk", "N100 drawdown", "band width", "breadth"],
                                     [[t["entry_trade"], t["exit_trade"], t["held_leg"], t["weeks_held"], t["exit_reason"], E1.fp(t["extra"]), E1.fp(t["held_ret"]), E1.fp(t["peer_ret"]), f"{t['z_entry']:+.2f}", E1.f2(t["corr52"]), E1.fp(t["mkt52"]), E1.fp(t["mkt_dd"]), fmtv("pct", t["band_width"]), fmtv("pct", t["breadth"])]
                                      for _, t in TRW[(TRW["window"] == w) & (TRW["pair"] == p)].sort_values("entry_trade").iterrows()]) for w, *_ in ZW.WINDOWS for p in ORDER}
    tab_corr = {p + ZW.SEP + w: E1.tbl(["entered", "exited", "held", "extra"] + [f"corr {lab}{' (daily)' if f_ == 'D' else ''}" for _, lab, f_, _ in CWINS],
                                       [[t["entry_trade"], t["exit_trade"], t["held_leg"], E1.fp(t["extra"])] + [E1.f2(t["corr_" + c]) for c, *_ in CWINS]
                                        for _, t in TRW[(TRW["window"] == w) & (TRW["pair"] == p)].sort_values("entry_trade").iterrows()]) for w, *_ in ZW.WINDOWS for p in ORDER}
    def cond_table(rows, n, head):
        return (f"<p class='note'><b>{head}</b> · {n} closed trades</p>" +
                E1.tbl(["condition", "when", "n", "median: good trades", "median: bad trades", "good − bad", "percentile", "rank correlation with the extra", "percentile"],
                       [[s["label"], s["when"], s["n"], fmtv(s["kind"], s["med_good"]), fmtv(s["kind"], s["med_bad"]), fmtc(s["kind"], s["diff"]), E1.fpct(s["diff_pct"]), E1.f2(s["rho"]), E1.fpct(s["rho_pct"])] for s in rows]))
    def n_trw(w, pool): return int(((TRW["window"] == w) & (TRW["pool"] == pool)).sum())
    tab_cond = {w: "".join(cond_table(statsW[(w, pl)], n_trw(w, pl), f"z window {wname(w)}, {pl} pairs") for pl in ("index", "gold")) for w, *_ in ZW.WINDOWS}
    tab_hcorr = {w: "".join(cond_table(statsC[(w, pl)], n_trw(w, pl), f"z window {wname(w)}, {pl} pairs: the legs' correlation at entry") for pl in ("index", "gold")) for w, *_ in ZW.WINDOWS}
    n_idx = TR[(TR["test"] == MAIN) & (TR["pool"] == "index")].groupby("pair").size()
    hold_t = {w: E1.tbl(["pool", "held", "trades", "share good", "median extra", "mean extra", "median weeks held"], [[r["pool"], r["hold"], r["n"], f"{r['good'] * 100:.0f}%", E1.fp(r["median_extra"]), E1.fp(r["mean_extra"]), f"{r['weeks']:.0f}"] for _, r in by_exit_w[by_exit_w["window"] == w].iterrows()])
              + "<details><summary>Per pair</summary>" + E1.tbl(["pair", "closed trades", "share good", "median extra", "mean extra", "median weeks held"], [[nice(r["pair"]), r["n"], f"{r['good'] * 100:.0f}%", E1.fp(r["median_extra"]), E1.fp(r["mean_extra"]), f"{r['weeks']:.0f}"] for _, r in PPW[PPW["window"] == w].iterrows()]) + "</details>"
              for w, *_ in ZW.WINDOWS}
    words_t = {w: f"<p style='margin:4px 0'><b>z window {wname(w)} · index pairs ({n_trw(w, 'index')} trades):</b> {words_w(w, 'index')}. <b>Gold pairs ({n_trw(w, 'gold')} trades):</b> {words_w(w, 'gold')}.</p>" for w, *_ in ZW.WINDOWS}
    copts = "".join(f"<option value='{c}'{' selected' if c == CDEFAULT else ''}>{lab}{' (daily)' if f_ == 'D' else ''}</option>" for c, lab, f_, _ in CWINS)
    opts = "".join(f"<option value='{html.escape(p, quote=True)}'{' selected' if p == DEFAULT_PAIR else ''}>{nice(p)}</option>" for p in ORDER)
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS).replace("<title>Ratio cycles</title>", "<title>Trade conditions</title>"),
         "<header class='bar'><div class='bar-inner'><b>E03 · trade conditions</b><span class='tag'>EXPLORATORY</span>",
         f"<label>Pair <select id='pair'>{opts}</select></label>{ZW.select_html()}<label>Correlation period <select id='cwin'>{copts}</select></label><button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>When does the ratio switch work?</h1>",
         f"<p class='note'><b>Data:</b> {html.escape(LT.NOTE)}. On the gold pairs the held gold leg was the liquid ETF on {sw_n} switches at the 52-week window; a trade's extra uses what was held.</p>",
         f"<p>Two descriptions, made after the ±1 sd ratio-band switch was locked, run and refuted (T02, T03). <b>A.</b> Every week, the ratio's z-score (the ±1 sd band in units of its sd) is bucketed and the next 4, 13 and 26 weeks' returns of the <b>cheaper</b> leg (the one that has lagged) and the <b>richer</b> leg are shown, against {D:,} random paths of the ratio that keep the two legs' common movement real. "
         f"<b>B.</b> Every closed trade of {MAIN}'s rule — <b>no maximum hold</b>: a trade lasts until the ratio is back at its trailing mean — (exit {EXIT}, 20.8 bps) carries the market's condition at its entry and during it; conditions are compared between good trades (the held leg beat the peer, fill to fill) and bad ones, against {P:,} permutations of the outcomes within each pair. "
         f"With the 52-week window the trades are {MAIN}'s rule (computed here on the latest data); with the other windows the same rule runs on that window's band (decisions still weekly). "
         f"Generated {audit['run_date']} from the latest data (the note above). <b>Description, not a finding</b>: a condition that goes with good trades here is a candidate for a pre-registered test, not a rule.</p>",
         f"<div class='card'><p style='margin:4px 0'><b>The two dropdowns at the top.</b> <b>z window</b> switches every chart and table on the page: Part A's band and random walks, and Part B's trades (the rule on that window's band) with every condition that depends on z. "
         f"<b>Correlation period</b> sets the lookback of the legs' correlation (13, 26 or 52 weeks of weekly returns; 20, 60, 120 or 250 days of daily returns) on the correlation chart and the scatter's correlation panel; the correlation heat map shows every period at once.</p></div>",
         f"<div class='card'><p style='margin:4px 0'><b>Against chance:</b> Part A {nA} of {len(A_p)} percentiles are notable at the 52-week window (about {0.1 * len(A_p):.0f} would be by chance); Part B "
         + "".join(f"<span data-winonly='{w}'{'' if w == ZW.DEFAULT else ' hidden'}>{nBw[w]} of {len(B_pw[w])} at z window {wname(w)} (about {0.1 * len(B_pw[w]):.0f})</span>" for w, *_ in ZW.WINDOWS) + ". "
         f"Conditions at entry use data up to the entry decision only ({PIT[0]} + {CPIT[0]} cut checks, {len(PIT[1]) + len(CPIT[1])} violations); conditions during the trade are hindsight by definition.</p></div>",
         "<h2>A · What each leg does next, by how stretched the ratio is</h2><p class='note'>Pick the pair at the top. Bars: the mean forward return of the cheaper leg, the richer leg and their difference (the spread: what holding the cheaper leg earns against the richer one, before costs). The grey mark is the spread on the random walks (median, 5th–95th); a blue bar outside the grey range is notable. Weeks overlap, so the episode count in the table is the honest sample size. <b>The z window dropdown at the top</b> switches Part A between the 52-week band and shorter weekly and daily windows (daily: daily closes, horizons of 20, 65 and 130 trading days, daily random walks).</p>",
         per_pair("prof", figs["prof"], tab_prof, first=True),
         per_win("h_prof", figs["h_prof"]),
         "<h2>B · The market's condition at each trade</h2><p class='note'>The trades on the chosen pair and z window, over the two legs (no maximum hold). With the 52-week window these are T03's rule's trades on the latest data. Hover a marker for the trade's outcome and the market at its entry.</p>",
         per_pair("tl", figs["tl"], tab_tl),
         "<h3>How many trades, and how long each one lasted</h3><p class='note'>The trades on the chosen pair and z window, with no maximum hold: each one lasts until the ratio is back at its trailing mean (or is still open at the end). The table under the chart has every pair.</p>",
         per_pair("days", figs["days"], tab_days),
         "<h3>Which conditions go with good trades?</h3>",
         "<div class='card'>" + win_divs(words_t) + "</div>",
         per_win("h_cond", figs["h_cond"], tab_cond),
         "<div class='card per-win' data-titles='" + html.escape(json.dumps(titles["scatter"]), quote=True) + "'>" + E1.html_fig(figs["scatter"]) + "</div>",
         "<h3>The legs' correlation, by period</h3><p class='note'>The <b>correlation period</b> dropdown at the top picks the lookback; the shading is the chosen z window's trades on the chosen pair. Weekly periods use weekly log returns on decision weeks; daily periods use daily log returns on the pair's common trading days and are drawn at decision days.</p>",
         per_pair("corr_tl", figs["corr_tl"], tab_corr),
         per_win("h_corr", figs["h_corr"], tab_hcorr),
         "<h3>By the length of the hold, and per pair</h3><div class='card'>" + win_divs(hold_t) + "</div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Part A.</b> z = (plain ratio − trailing 52-week mean) ÷ trailing sd (E01's). Cheaper leg = a when z < 0, b when z > 0. Forward returns are close to close on decision weeks, simple, no costs or fills. Random paths: E01's (the point-in-time residuals with the same random signs plus the expanding drift, after the real 52-week warm-up); with M = ½(ln a + ln b) kept real, the random legs are exp(M ± ½ ln r*), so only the legs' relative movement is random. The window starts after the warm-up. Percentile = (paths below + ½ ties) ÷ paths.</li>"
         f"<li><b>Part B.</b> Every z window, the 52-week one included: {MAIN}'s rule (window {WIN}, exit {EXIT}) on that window's band (explore/E01_ratio_cycles/variants.py: enter on the first decision week with z beyond ±1, exit when z is back through 0; decisions weekly, daily windows read the daily z at the decision day; fills, costs and valuation through the engine's own book). On the latest data the 52-week run is checked against the engine's decide() (decisions) and, on the gold pairs with the switch off, against the engine's book; {MAIN}'s evidence is on the earlier data and is not read. "
         f"Gold pairs: whenever the rule holds gold in a week whose decision falls in gold's weekly supertrend SELL, it holds the liquid ETF instead (bought at the next session's fill, 10.4 bps per side); the held leg's return compounds gold and liquid over the trade. "
         f"No maximum hold; open trades left out of the outcome statistics ({n_open[MAIN]} at 52 weeks; their days so far are on the length chart). Extra = held leg's fill-to-fill return − the peer's (T02.md §4.3), no cost. Conditions at the entry decision's close from the pair's decision-week series and N100's daily closes (N100 is the market for every pair); z, |z|, weeks since z changed sign, band width (2 sd ÷ mean of the window's closes) and breadth (the share of the ten ratios whose own z is beyond ±1) use the chosen z window. During: entry fill to exit fill (N100), and the decision weeks held (correlation and volatility need {MIN_DURING} weeks or more).</li>"
         f"<li><b>Correlation periods.</b> The legs' trailing Pearson correlation of weekly log returns on decision weeks (13, 26, 52 weeks) or of daily log returns on the pair's common trading days (20, 60, 120, 250 days), at the entry decision's close; the full window is required. The 26- and 52-week series are the conditions' corr 26 / corr 52 (checked equal), so their heat-map rows reuse those statistics; the other periods are drawn with seed default_rng([{SEED_CORR}, window_id, pool_id]). Recomputed from the history cut at 12 dates per pair ({CPIT[0]} checks, {len(CPIT[1])} violations).</li>"
         f"<li><b>Null.</b> Within each pair the trades' extras (for the rank correlation) or good/bad labels (for the medians) are shuffled {P:,} times, seed default_rng([{SEED_PERM}, window_id, pool_id]) (window_id 0 = 52 weeks, as before); the trades are ranked within their pair before pooling for the correlation. The null asks only whether the condition is associated with the outcome. Whether a filtered rule beats random timing is a new pre-registered test.</li>"
         f"<li><b>Limits.</b> {n_idx.min()}–{n_idx.max()} closed trades per index pair; many correlated conditions; all series from {LT.DATA_FROM.date()} (the first band a year later); the gold leg has an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); the liquid level has no recorded source before 2024-01-24 (F-RA-D17); Micro250 pairs from 2022 and to 2026-08-28. The design was written before the first run (explore/E03_trade_conditions/E03.md).</li></ul></div>",
         "</main>", "<script>" + E1.JS.replace("__DIV_LIGHT__", json.dumps(E1.DIV_LIGHT)).replace("__DIV_DARK__", json.dumps(E1.DIV_DARK)).replace("e01-pair", "e03-pair") + "</script></body></html>"]
    out = os.path.join(OUT, "E03_conditions.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; notable A {nA} of {len(A_p)}, B {nB} of {len(B_p)} → {out}")


if __name__ == "__main__":
    main()
