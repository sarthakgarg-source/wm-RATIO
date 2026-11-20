"""
E07 (EXPLORATORY): good and bad trades — what tells them apart, and can the rule be managed better?
Design: explore/E07_good_bad/E07.md (written before the first run).

    python explore/E07_good_bad/goodbad.py [--perms 2000] [--null-lr 200] [--null-gb 100] [--rnd 200]

Every number on the page and in E07_results.md is computed here; the page's script only picks the numbers of the chosen
controls from the embedded results and draws them. Not a finding and not a test of any rule.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
for p_ in (ROOT, os.path.join(ROOT, "explore", "E01_ratio_cycles"), os.path.join(ROOT, "explore", "E03_trade_conditions"), os.path.join(ROOT, "explore", "E06_trade_search")):
    sys.path.insert(0, p_)
from analysis.approved_data import GOLD, fingerprints, load_daily  # noqa: E402
from analysis.ratio_engine import PAIRS, PIT, book, pair_frame, trade_extras  # noqa: E402
import cycles as E1  # noqa: E402  (palette, page styling, percentile)
import zwin as ZW  # noqa: E402  (the z windows)
import variants as VR  # noqa: E402  (the rule through the engine's book)
import conditions as E3  # noqa: E402  (the z-window conditions)
import search as E6  # noqa: E402  (E06's condition frame and statistics)

MEAS = (("eff26", "path efficiency, 26 weeks (1 = a straight line)", "efficiency 26 wk"), ("eff52", "path efficiency, 52 weeks", "efficiency 52 wk"),
        ("cross104", "times z changed sign, last 104 weeks", "z sign changes 104 wk"), ("ac104", "autocorrelation of weekly changes, 104 weeks", "autocorrelation 104 wk"),
        ("vr104", "variance ratio, 4-week ÷ weekly, 104 weeks", "variance ratio 104 wk"), ("r2_52", "R² of a straight line, 52 weeks", "trend R² 52 wk"),
        ("hl104", "mean-reversion half-life in weeks, 104 weeks", "half-life 104 wk"))
MKEYS = [m[0] for m in MEAS]
WLEN = {"w52": 52, "w26": 26, "w13": 13, "d250": 50, "d120": 24, "d60": 12, "d20": 4}   # the z window's length in weeks
VARS = (("base", "Rule R (first touch, exit at 0)", {}), ("zs20", "z-stop at ±2", {"zstop": 2.0}), ("zs25", "z-stop at ±2.5", {"zstop": 2.5}), ("zs30", "z-stop at ±3", {"zstop": 3.0}),
        ("ts13", "time stop, 13 weeks", {"tstop": 13}), ("ts26", "time stop, 26 weeks", {"tstop": 26}), ("ts52", "time stop, 52 weeks", {"tstop": 52}),
        ("recross", "enter on the re-cross into the band", {"entry": "recross"}), ("rebound", "enter after a 0.5 rebound from the extreme", {"entry": "rebound"}))
KS = (4, 8, 13, 26); HORIZON = 52; DEEPER = 0.5
SPLIT = E6.SPLIT; WF_START = pd.Timestamp("2010-01-01"); YEARS = list(range(2010, 2027))
SEED_UNI, SEED_NULL, SEED_RND = 120, 121, 122
COST = VR.COST; ORDER = E1.ORDER; nice = E1.nice; DEFAULT_PAIR = E1.DEFAULT_PAIR; LO, HI = E1.LO, E1.HI
POOL = {p: ("gold" if GOLD in p else "index") for p in PAIRS}; INDEX_PAIRS = [p for p in PAIRS if POOL[p] == "index"]
WLAB = E6.WLAB; FEATS = E6.CKEYS + MKEYS + ["wlen"]
E6.CFMT.update({k: ("wk" if k in ("cross104", "hl104") else "num") for k in MKEYS})   # E06's uni() looks the format up


def r4(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def r6(v):   # the page's numbers: enough digits that rounding for display matches the CSVs
    return r4(v, 6)


# ------------------------------------------------------------------------------------------------------- §2 the measures
def character(pf: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    """The ratio's character at every decision week, from decision-week closes up to that week only."""
    wk = pf.loc[pf["decision"]]; lnr = np.log(wk["ratio"]); d = lnr.diff(); m = pd.DataFrame(index=wk.index)
    for n in (26, 52):
        m[f"eff{n}"] = (lnr - lnr.shift(n)).abs() / d.abs().rolling(n).sum()
    s = np.sign(z.reindex(wk.index)); both = s.notna() & s.shift(1).notna()
    ch = (s != s.shift(1)).astype(float); ch[~both] = np.nan
    m["cross104"] = ch.rolling(104).sum()
    m["ac104"] = d.rolling(104).corr(d.shift(1))
    m["vr104"] = (lnr - lnr.shift(4)).rolling(101).var() / (4 * d.rolling(104).var())
    t = pd.Series(np.arange(len(lnr), dtype=float), index=lnr.index); m["r2_52"] = lnr.rolling(52).corr(t) ** 2
    lag = lnr.shift(1); b = d.rolling(104).cov(lag) / lag.rolling(104).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        hl = np.where(b < 0, -np.log(2) / np.log1p(b.clip(lower=-0.999999)), 520.0)
    m["hl104"] = np.where(b.isna(), np.nan, np.minimum(hl, 520.0))
    return m


# ------------------------------------------------------------------------------------------------------- §3–§4 variants
def states_v(z, thr=1.0, zstop=None, tstop=None, entry="first", reb=0.5):
    """Rule R's state machine (variants.rule_states) with an optional z-stop, time stop, or a waiting entry. After a stop
    the sleeve cannot re-enter in the same week; a new entry needs a fresh cross."""
    state, prev, ent_i, side, ext, out = "N", None, None, None, None, []
    for i, x in enumerate(z):
        if np.isnan(x):
            out.append(state); prev = None; side = None; continue
        pos = -1 if x < -thr else (1 if x > thr else 0); stopped = False
        if (state == "A" and x >= 0) or (state == "B" and x <= 0):
            state = "N"
        elif state in ("A", "B"):
            if zstop is not None and ((state == "A" and x <= -zstop) or (state == "B" and x >= zstop)):
                state, stopped = "N", True
            elif tstop is not None and i - ent_i >= tstop:
                state, stopped = "N", True
        if state == "N" and not stopped:
            new = None
            if entry == "first":
                if pos == -1 and prev in (0, 1):
                    new = "A"
                elif pos == 1 and prev in (0, -1):
                    new = "B"
            elif entry == "recross":
                if prev == -1 and -thr <= x < 0:
                    new = "A"
                elif prev == 1 and 0 < x <= thr:
                    new = "B"
            elif entry == "rebound":
                if (side == -1 and x >= 0) or (side == 1 and x <= 0):
                    side = None
                if pos == -1:
                    side, ext = (-1, x) if side != -1 else (-1, min(ext, x))
                elif pos == 1:
                    side, ext = (1, x) if side != 1 else (1, max(ext, x))
                if side == -1 and x - ext >= reb and x < 0:
                    new = "A"
                elif side == 1 and ext - x >= reb and x > 0:
                    new = "B"
            if new:
                state, ent_i = new, i
        if state != "N":
            side = None
        out.append(state); prev = pos
    return out


def run_targets(pf, z, states):
    dec = pd.Series(states, index=z.index)[z.notna().to_numpy()]
    nav, trades = book(pf, dec.index, dec.tolist(), COST, PIT(), record_trades=True)
    return nav, pd.DataFrame(trade_extras(trades))


def vstats(nav, ex, bh):
    cl = ex[~ex["open"].astype(bool)] if len(ex) else ex
    full = E6.run_stats(nav, bh); v = E6.run_stats(nav, bh, SPLIT)
    return {"trades": int(len(cl)), "open": int(ex["open"].astype(bool).sum()) if len(ex) else 0, "win": float((cl["extra"] > 0).mean()) if len(cl) else np.nan,
            "mean_extra": float(cl["extra"].mean()) if len(cl) else np.nan, "cagr": full["cagr"], "cagr_5050": full["cagr_5050"], "excess": full["excess"], "excess_1626": v["excess"],
            "maxdd": float((nav / nav.cummax() - 1).min()), "first": full["first"], "last": full["last"], "first_1626": v["first"]}


# ------------------------------------------------------------------------------------------------------- §5 the models
def fit_model(kind, Xtr, ytr):
    """Fit on the training trades; returns (predict function, training median of the predictions)."""
    med = np.nanmedian(np.where(np.isnan(Xtr).all(axis=0), 0.0, Xtr), axis=0) if len(Xtr) else np.zeros(Xtr.shape[1])
    med = np.where(np.isnan(med), 0.0, med)
    def imp(X):
        return np.where(np.isnan(X), med, X)
    A = imp(Xtr)
    if kind == "lr":
        mu = A.mean(axis=0); sd = A.std(axis=0); sd[sd == 0] = 1.0
        m = LogisticRegression(C=1.0, max_iter=5000).fit((A - mu) / sd, ytr)
        def pr(X):
            return m.predict_proba((imp(X) - mu) / sd)[:, 1]
    else:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0).fit(A, ytr)
        def pr(X):
            return m.predict_proba(imp(X))[:, 1]
    return pr, float(np.median(pr(Xtr)))


def walk(kind, X, y, closed, exit_dt, entry_year, is_ix, keep=False):
    """Year by year: train on index trades closed before 1 January Y, predict every trade entered in Y."""
    pred = np.full(len(y), np.nan); above = np.zeros(len(y), bool); models = {}; n_train = {}
    for Y in YEARS:
        y0 = np.datetime64(f"{Y}-01-01", "ns"); tr = is_ix & closed & (exit_dt < y0); te = entry_year == Y
        assert (exit_dt[tr] < y0).all()
        n_train[Y] = int(tr.sum())
        if tr.sum() < 50 or len(np.unique(y[tr])) < 2 or not te.any():
            continue
        pr, thr = fit_model(kind, X[tr], y[tr]); pred[te] = pr(X[te]); above[te] = pred[te] >= thr
        if keep:
            models[Y] = (pr, thr)
    return pred, above, models, n_train


def auc(y, p):
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok])) if ok.sum() > 10 and len(np.unique(y[ok])) == 2 else np.nan


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--null-lr", type=int, default=200)
    ap.add_argument("--null-gb", type=int, default=100); ap.add_argument("--rnd", type=int, default=200); a_ = ap.parse_args()
    P, NLR, NGB, RND = a_.perms, a_.null_lr, a_.null_gb, a_.rnd
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_daily(); mkt = px["N100"]["close"].astype(float)
    checks = {"t03_mismatch": [], "e06_trades_unmatched": None, "e06_max_abs_diff": None, "states_base_mismatch": [], "base_nav_mismatch": [], "cut_checks": 0, "cut_violations": [],
              "train_trades_closed_before_year": True, "random_rate_max_abs_dev": 0.0}
    pfs = {p: pair_frame(px, p, "plain") for p in PAIRS}
    zall = {w: {p: E3.zcond(pfs[p], w)["z"].dropna() for p in PAIRS} for w, *_ in ZW.WINDOWS}
    frames, chars, zser, runs, rows, paths, VROWS = {}, {}, {}, {}, [], [], []
    for pair in PAIRS:
        pf = pfs[pair]; b = E6.base_frame(pf, mkt); wk = pf.loc[pf["decision"]]; zser[pair] = {}; la, lb = pair.split("|")
        for wi, (w, *_) in enumerate(ZW.WINDOWS):
            zc = E3.zcond(pf, w); z = zc["z"]; f = E6.window_frame(b, zc, E6.breadth(zc.index, zall[w])); m = character(pf, z)
            frames[(pair, w)] = f; chars[(pair, w)] = m; zser[pair][w] = z
            r = VR.run_states(pf, z, COST, 1.0); runs[(pair, w)] = r; bh = r["bh"]
            if w == ZW.DEFAULT:
                checks["t03_mismatch"] += VR.check_against_t03({"pair": pair, **r})
            if states_v(z.to_numpy()) != VR.rule_states(z.to_numpy())[0]:
                checks["states_base_mismatch"].append(f"{pair} {w}")
            dec = f.index; zz = z.to_numpy(); ca, cb = wk["ca"].to_numpy(), wk["cb"].to_numpy(); tr_ = r["trades"]
            for _, t in r["extras"].iterrows():
                ed = pd.Timestamp(t["entry_decision"]); et = pd.Timestamp(t["entry_trade"]); op = bool(t["open"]); xt = None if op else pd.Timestamp(t["exit_trade"])
                i0 = int(dec.get_loc(ed)); A = t["held"] == "A"; sg = -1.0 if A else 1.0
                fr = tr_[pd.to_datetime(tr_["trade_date"]) == et].iloc[0]; fh, fp = (fr["fill_a"], fr["fill_b"]) if A else (fr["fill_b"], fr["fill_a"]); ch_, cp_ = (ca, cb) if A else (cb, ca)
                weeks = np.nan if op else int(dec.searchsorted(xt) - 1 - i0)
                s = [sg * zz[i0 + k] if i0 + k < len(zz) else np.nan for k in range(HORIZON + 1)]
                x = [np.nan] + [(ch_[i0 + k] / fh - 1) - (cp_[i0 + k] / fp - 1) if i0 + k < len(zz) else np.nan for k in range(1, HORIZON + 1)]
                kmax = (len(zz) - i0) if op else max(int(weeks), 1)   # the weeks the trade was open (to the data's end if still open)
                row = {"window": w, "pair": pair, "pool": POOL[pair], "half": "D" if ed < SPLIT else "V", "entry_decision": ed.date().isoformat(), "entry_trade": et.date().isoformat(),
                       "exit_trade": None if op else xt.date().isoformat(), "held": t["held"], "held_leg": la if A else lb, "open": op, "extra": np.nan if op else float(t["extra"]),
                       "weeks": weeks, "z_entry": float(zz[i0]), "deepest": float(np.nanmax(sg * zz[i0:i0 + kmax])), "wlen": float(np.log(WLEN[w]))}
                row.update({k: float(f.loc[ed, k]) for k in E6.CKEYS}); row.update({k: float(m.loc[ed, k]) for k in MKEYS}); rows.append(row)
                paths.append(s + x[1:])
            # §3–§4: every variant through the engine's book
            for vk, vlab, kw in VARS:
                st = states_v(z.to_numpy(), **kw); nav, ex = run_targets(pf, z, st)
                if vk == "base" and not np.allclose(nav.to_numpy(), r["nav"].to_numpy(), rtol=0, atol=1e-12):
                    checks["base_nav_mismatch"].append(f"{pair} {w}")
                VROWS.append({"pair": pair, "pool": POOL[pair], "window": w, "variant": vk, **vstats(nav, ex, bh)})
        print(f"  {pair:18s} [{time.time() - t0:4.0f}s]", flush=True)
    TR = pd.DataFrame(rows); TR["good"] = TR["extra"] > 0
    PATH = pd.DataFrame(paths, columns=[f"s{k}" for k in range(HORIZON + 1)] + [f"x{k}" for k in range(1, HORIZON + 1)])
    # checks: the trades and conditions are E06's; the base state machine and book are the rule's; the measures are point in time
    e6 = pd.read_csv(os.path.join(ROOT, "explore", "E06_trade_search", "out", "trades_all.csv"))
    mm = TR.merge(e6, on=["window", "pair", "entry_decision"], suffixes=("", "_e6"), how="outer", indicator=True)
    checks["e06_trades_unmatched"] = int((mm["_merge"] != "both").sum())
    def maxdiff(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); d = np.abs(a - b); d[np.isnan(a) != np.isnan(b)] = np.inf
        return float(np.nanmax(d)) if (~np.isnan(d)).any() else 0.0
    checks["e06_max_abs_diff"] = max(maxdiff(mm[k], mm[f"{k}_e6"]) for k in E6.CKEYS + ["extra", "weeks", "z_entry"])
    exit_ok = (mm["exit_trade"].fillna("") == mm["exit_trade_e6"].fillna("")).all()
    for pair in PAIRS:
        pf = pfs[pair]; dec = pf.index[pf["decision"]]
        for T in np.unique(np.linspace(160, len(dec) - 1, 12).astype(int)):
            cut = pf.loc[: dec[T]]
            for w, *_ in ZW.WINDOWS:
                m_ = character(cut, E3.zcond(cut, w)["z"]); full = chars[(pair, w)].iloc[: len(m_)]; checks["cut_checks"] += 1
                if not np.allclose(m_.to_numpy(float), full.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True):
                    checks["cut_violations"].append(f"{pair} {w} cut {dec[T].date()}")
    if checks["t03_mismatch"] or checks["e06_trades_unmatched"] or not checks["e06_max_abs_diff"] < 1e-9 or not exit_ok or checks["states_base_mismatch"] or checks["base_nav_mismatch"] or checks["cut_violations"]:
        sys.exit(f"checks FAILED: {checks}")
    pd.concat([TR, PATH.round(5)], axis=1).to_csv(os.path.join(OUT, "trades_e07.csv"), index=False)
    VA = pd.DataFrame(VROWS); VA.to_csv(os.path.join(OUT, "variants.csv"), index=False)
    print(f"  {len(TR)} trades = E06's (max difference {checks['e06_max_abs_diff']:.1e}); base state machine and book = the rule's; measures: {checks['cut_checks']} cuts, 0 violations  [{time.time() - t0:4.0f}s]", flush=True)

    CL = TR[~TR["open"]].copy(); CLP = PATH[~TR["open"].to_numpy()].reset_index(drop=True); CL = CL.reset_index(drop=True)
    # ---------------------------------------------------------------- §1 good and bad side by side
    FAN, HOLD = {}, {}
    for w, *_ in ZW.WINDOWS:
        FAN[w], HOLD[w] = {}, {}
        for pool in ("index", "gold"):
            sel = (CL["window"] == w) & (CL["pool"] == pool); FAN[w][pool] = {}
            for gname, gm in (("good", CL["good"]), ("bad", ~CL["good"])):
                P_ = CLP[(sel & gm).to_numpy()]
                q = lambda cols: [[r4(v, 3) for v in P_[cols].quantile(qq).to_numpy()] for qq in (0.5, 0.25, 0.75)]  # noqa: E731
                FAN[w][pool][gname] = {"n": int(len(P_)), "s": q([f"s{k}" for k in range(HORIZON + 1)]), "x": q([f"x{k}" for k in range(1, HORIZON + 1)])}
            g, bd = CL[sel & CL["good"]], CL[sel & ~CL["good"]]
            HOLD[w][pool] = [int(len(g)), int(len(bd)), r4(g["weeks"].median(), 1), r4(bd["weeks"].median(), 1), r4(g["z_entry"].abs().median(), 2), r4(bd["z_entry"].abs().median(), 2),
                             r4(g["deepest"].median(), 2), r4(bd["deepest"].median(), 2), r6(g["extra"].median()), r6(bd["extra"].median()),
                             CL[sel]["entry_trade"].min(), CL[sel]["exit_trade"].max()]
    # ---------------------------------------------------------------- §2 the measures, condition by condition
    IX = CL[CL["pool"] == "index"]; GD = CL[CL["pool"] == "gold"]; urows = []
    wins = [w for w, *_ in ZW.WINDOWS] + ["all"]
    for wi, w in enumerate(wins):
        sub = IX if w == "all" else IX[IX["window"] == w]
        for ci, k in enumerate(MKEYS):
            urows.append({"window": w, "pool": "index", **E6.uni(sub.assign(pair=sub["pair"] + ("|" + sub["window"] if w == "all" else "")), k, np.random.default_rng([SEED_UNI, wi, ci]), P)})
    for ci, k in enumerate(MKEYS):
        urows.append({"window": "all", "pool": "gold", **E6.uni(GD.assign(pair=GD["pair"] + "|" + GD["window"]), k, np.random.default_rng([SEED_UNI, 99, ci]), P)})
    UN = pd.DataFrame(urows); UN.to_csv(os.path.join(OUT, "character_univariate.csv"), index=False)
    print(f"  §2 measures [{time.time() - t0:4.0f}s]", flush=True)
    # ---------------------------------------------------------------- §3 early warning
    EW, ewrows = {}, []
    for w, *_ in ZW.WINDOWS:
        EW[w] = {}
        for pool in ("index", "gold"):
            EW[w][pool] = {}
            for k in KS:
                sel = ((CL["window"] == w) & (CL["pool"] == pool) & (CL["weeks"] > k)).to_numpy(); d = CL[sel]; s0 = CLP.loc[sel, "s0"].to_numpy(); sk = CLP.loc[sel, f"s{k}"].to_numpy()
                deep = sk >= s0 + DEEPER; rec = sk < s0; mid = ~deep & ~rec; y = d["extra"].to_numpy(); gd = y > 0
                cell = lambda mk: [int(mk.sum()), r6(gd[mk].mean()) if mk.any() else None, r6(y[mk].mean()) if mk.any() else None]  # noqa: E731
                rec_ = [int(sel.sum())] + cell(deep) + cell(mid) + cell(rec) + [r6(deep[~gd].mean()) if (~gd).any() else None, r6(deep[gd].mean()) if gd.any() else None]
                EW[w][pool][str(k)] = rec_
                ewrows.append({"window": w, "pool": pool, "week": k, "open": rec_[0], "deeper_n": rec_[1], "deeper_good": rec_[2], "deeper_mean": rec_[3], "between_n": rec_[4], "between_good": rec_[5],
                               "between_mean": rec_[6], "recovering_n": rec_[7], "recovering_good": rec_[8], "recovering_mean": rec_[9], "bad_that_were_deeper": rec_[10], "good_that_were_deeper": rec_[11]})
    pd.DataFrame(ewrows).to_csv(os.path.join(OUT, "early_warning.csv"), index=False)
    # variants summary over the six index pairs, per window
    VSUM = {}; vix = VA[VA["pool"] == "index"]
    for w, *_ in ZW.WINDOWS:
        VSUM[w] = {}; base_ = vix[(vix["window"] == w) & (vix["variant"] == "base")].set_index("pair")
        for vk, *_ in VARS:
            v_ = vix[(vix["window"] == w) & (vix["variant"] == vk)].set_index("pair").reindex(base_.index)
            VSUM[w][vk] = [int((v_["excess"] > base_["excess"]).sum()), int((v_["excess_1626"] > base_["excess_1626"]).sum()), int((v_["excess"] > 0).sum()), int((v_["excess_1626"] > 0).sum()),
                           r6(v_["excess"].mean()), r6(v_["excess_1626"].mean()), r6(v_["win"].mean()), int(v_["trades"].sum())]
    VALL = {vk: [sum(VSUM[w][vk][i] for w in VSUM) for i in range(4)] + [r6(vix[vix["variant"] == vk]["excess"].median()), r6(vix[vix["variant"] == vk]["excess_1626"].median())] for vk, *_ in VARS}
    print(f"  §3 early warning and variants [{time.time() - t0:4.0f}s]", flush=True)

    # ---------------------------------------------------------------- §5 walk-forward
    X = TR[FEATS].to_numpy(float); yall = TR["good"].to_numpy(bool).astype(int); closed = ~TR["open"].to_numpy(bool)
    exit_dt = pd.to_datetime(TR["exit_trade"]).to_numpy(); entry_year = pd.to_datetime(TR["entry_trade"]).dt.year.to_numpy(); is_ix = (TR["pool"] == "index").to_numpy()
    exit_dt = np.where(closed, exit_dt, np.datetime64("2100-01-01")).astype("datetime64[ns]")
    scored_ix = is_ix & closed & (entry_year >= YEARS[0]); scored_g = ~is_ix & closed & (entry_year >= YEARS[0])
    grp = (TR["pair"] + "|" + TR["window"]).to_numpy()
    WF, WFROWS, WFT, MODELS = {}, [], TR[["window", "pair", "pool", "entry_trade", "exit_trade", "extra", "good", "open"]].copy(), {}
    WFT["year"] = entry_year
    for mi, (kind, mlab) in enumerate((("lr", "logistic regression"), ("gb", "gradient-boosted trees"))):
        pred, above, models, ntr = walk(kind, X, yall, closed, exit_dt, entry_year, is_ix, keep=True); MODELS[kind] = models
        WFT[f"p_{kind}"] = pred; WFT[f"above_{kind}"] = np.where(np.isnan(pred), np.nan, above.astype(float))
        a_ix = auc(yall[scored_ix], pred[scored_ix]); a_g = auc(yall[scored_g], pred[scored_g])
        nulls = []; NN = NLR if kind == "lr" else NGB
        for dr in range(NN):
            rng = np.random.default_rng([SEED_NULL, mi, dr]); yp = yall.copy()
            for g in np.unique(grp[is_ix & closed]):
                idx = np.flatnonzero((grp == g) & is_ix & closed); yp[idx] = yp[idx][rng.permutation(len(idx))]
            pn, *_ = walk(kind, X, yp, closed, exit_dt, entry_year, is_ix); nulls.append(auc(yp[scored_ix], pn[scored_ix]))
        nulls = np.array(nulls); pct = E1.pctile(a_ix, nulls)
        def halves(sc):
            e = TR["extra"].to_numpy(); ab = above & sc & ~np.isnan(pred); be = ~above & sc & ~np.isnan(pred)
            return [int(ab.sum()), r6(np.nanmean(e[ab])), r6((e[ab] > 0).mean()), int(be.sum()), r6(np.nanmean(e[be])), r6((e[be] > 0).mean())]
        yrs = [[Y, r4(auc(yall[scored_ix & (entry_year == Y)], pred[scored_ix & (entry_year == Y)]), 3), int((scored_ix & (entry_year == Y) & ~np.isnan(pred)).sum()), ntr[Y]] for Y in YEARS]
        WF[kind] = {"label": mlab, "auc": r4(a_ix, 3), "pct": r4(pct, 1), "null": [r4(v, 3) for v in nulls], "null_med": r4(np.nanmedian(nulls), 3), "null_hi": r4(np.nanquantile(nulls, 0.95), 3),
                    "years": yrs, "ix": halves(scored_ix), "gold": [r4(a_g, 3)] + halves(scored_g), "n_ix": int((scored_ix & ~np.isnan(pred)).sum()), "n_gold": int((scored_g & ~np.isnan(pred)).sum()),
                    "first": str(TR.loc[scored_ix, "entry_trade"].min()), "last": str(TR.loc[scored_ix, "entry_trade"].max())}
        WFROWS.append({"model": kind, "pool": "index", "year": "all", "auc": a_ix, "null_pct": pct, "null_median": np.nanmedian(nulls), "n": WF[kind]["n_ix"], **dict(zip(["n_above", "mean_above", "win_above", "n_below", "mean_below", "win_below"], WF[kind]["ix"]))})
        WFROWS.append({"model": kind, "pool": "gold", "year": "all", "auc": a_g, "n": WF[kind]["n_gold"], **dict(zip(["n_above", "mean_above", "win_above", "n_below", "mean_below", "win_below"], WF[kind]["gold"][1:]))})
        WFROWS += [{"model": kind, "pool": "index", "year": Y, "auc": a, "n": n, "train_trades": nt} for Y, a, n, nt in yrs]
        print(f"  §5 {mlab}: AUC {a_ix:.3f} (percentile {pct:.1f} among {NN} label shuffles), gold {a_g:.3f}  [{time.time() - t0:4.0f}s]", flush=True)
    WFT.to_csv(os.path.join(OUT, "walkforward_trades.csv"), index=False); pd.DataFrame(WFROWS).to_csv(os.path.join(OUT, "walkforward_summary.csv"), index=False)
    # the models as entry filters, index pairs, from 2010-01-01
    FROWS, FILT = [], {}
    for mi, kind in enumerate(("lr", "gb")):
        FILT[kind] = {}
        for pid, pair in enumerate(INDEX_PAIRS):
            pf = pfs[pair]
            for wi, (w, *_) in enumerate(ZW.WINDOWS):
                f, m, z = frames[(pair, w)], chars[(pair, w)], zser[pair][w]; Xw = np.column_stack([f[E6.CKEYS].to_numpy(float), m[MKEYS].to_numpy(float), np.full(len(f), np.log(WLEN[w]))])
                yrs_ = f.index.year.to_numpy(); allow = np.ones(len(f), bool)
                for Y, (pr, thr) in MODELS[kind].items():
                    sel = yrs_ == Y
                    if sel.any():
                        allow[sel] = pr(Xw[sel]) >= thr
                bh = runs[(pair, w)]["bh"]; post = (f.index >= WF_START) & z.notna().to_numpy(); rate = float(allow[post].mean())
                em = E6.run_stats(VR.run_states(pf, z, COST, 1.0, allow=allow, comp=False)["nav"], bh, WF_START); ee = E6.run_stats(runs[(pair, w)]["nav"], bh, WF_START)
                rnd = []; rates = []
                for dr in range(RND):
                    rng = np.random.default_rng([SEED_RND, mi, pid, wi, dr]); ar = np.ones(len(f), bool); ar[post] = rng.random(int(post.sum())) < rate; rates.append(ar[post].mean())
                    rnd.append(E6.run_stats(VR.run_states(pf, z, COST, 1.0, allow=ar, comp=False)["nav"], bh, WF_START)["excess"])
                checks["random_rate_max_abs_dev"] = max(checks["random_rate_max_abs_dev"], abs(float(np.mean(rates)) - rate))
                rnd = np.array(rnd); pct = E1.pctile(em["excess"], rnd)
                FILT[kind].setdefault(w, {})[pair] = [r4(rate, 3), r6(em["excess"]), r6(ee["excess"]), r6(float(np.median(rnd))), r4(pct, 3), em["first"], em["last"]]
                FROWS.append({"model": kind, "pair": pair, "window": w, "rate": rate, "excess_model": em["excess"], "excess_every": ee["excess"], "random_median": float(np.median(rnd)), "pct": pct, "first": em["first"], "last": em["last"]})
        print(f"  §5 filter {kind} [{time.time() - t0:4.0f}s]", flush=True)
    FD = pd.DataFrame(FROWS); FD.to_csv(os.path.join(OUT, "walkforward_filters.csv"), index=False)
    FSUM = {kind: {"beats_every": int((g["excess_model"] > g["excess_every"]).sum()), "ahead_5050": int((g["excess_model"] > 0).sum()), "every_ahead_5050": int((g["excess_every"] > 0).sum()),
                   "above_random_median": int((g["pct"] > 50).sum()), "notable": int((g["pct"] >= HI).sum()), "median_pct": r4(g["pct"].median(), 1), "n": int(len(g))} for kind, g in FD.groupby("model")}

    # ---------------------------------------------------------------- audit
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "perms": P, "null_draws": {"lr": NLR, "gb": NGB}, "random_filters": RND,
             "seeds": {"measures": f"default_rng([{SEED_UNI}, window_id, measure_id])", "walkforward_null": f"default_rng([{SEED_NULL}, model_id, draw])", "random_filters": f"default_rng([{SEED_RND}, model_id, pair_id, window_id, draw])"},
             "checks": checks, "trades": int(len(TR)), "closed": int(len(CL)), "walkforward": {k: {kk: v[kk] for kk in ("auc", "pct", "null_med", "null_hi", "n_ix", "n_gold")} for k, v in WF.items()},
             "filters": FSUM, "variants_all_windows": VALL, "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def pq(v): return "—" if v != v or v is None else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
    def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
    def fp(v): return "—" if v is None or v != v else f"{v * 100:+.1f}%"
    def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
    vlab = {v[0]: v[1] for v in VARS}; mlab_ = {m[0]: m[2] for m in MEAS}
    md = [f"# E07 results — good and bad trades (EXPLORATORY)\n",
          f"_Generated by `explore/E07_good_bad/goodbad.py` on {audit['run_date']} from the approved data of 2026-11-12. Rule R (T03's rule, ±1, exit at 0) on every z window, all ten pairs: {len(TR)} trades ({len(CL)} closed), E06's exactly. Good = extra > 0 (held − peer, fill to fill, before costs). Description, not a finding; percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Checks:** the trades and E06's 24 conditions equal E06's (max difference {checks['e06_max_abs_diff']:.1e}); the 52-week run is T03; the variants' state machine with no added rule is Rule R's and its book gives Rule R's NAV on every pair and window; the six measures pass the cut-history check ({checks['cut_checks']} cuts, 0 violations); every walk-forward training trade closed before its year; random filters keep their rate (largest deviation {checks['random_rate_max_abs_dev']:.4f}).\n"]
    md += ["## 1. Good and bad trades side by side (index pairs)\n", "| z window | good / bad trades | median weeks held | median \\|z\\| at entry | median deepest \\|z\\| while open | median extra |", "|---|---|---|---|---|---|"]
    for w, *_ in ZW.WINDOWS:
        h = HOLD[w]["index"]; md.append(f"| {WLAB[w]} ({h[10]} → {h[11]}) | {h[0]} / {h[1]} | {h[2]} / {h[3]} | {h[4]} / {h[5]} | {h[6]} / {h[7]} | {fp(h[8])} / {fp(h[9])} |")
    md += [f"\n## 2. The ratio's character at entry: rank correlation with the extra, within each pair (percentile among {P:,} shuffles)\n",
           "| measure | " + " | ".join(WLAB[w] for w, *_ in ZW.WINDOWS) + " | all windows | before 2016 / from 2016 | gold pairs | all windows: mean extra low / middle / high third |", "|---|" + "---|" * (len(ZW.WINDOWS) + 4)]
    for k in MKEYS:
        cells = []
        for w, *_ in ZW.WINDOWS:
            r = UN[(UN["window"] == w) & (UN["pool"] == "index") & (UN["key"] == k)].iloc[0]; cells.append(f"{r['rho']:+.2f} ({pq(r['pct'])})")
        ra = UN[(UN["window"] == "all") & (UN["pool"] == "index") & (UN["key"] == k)].iloc[0]; rg = UN[(UN["window"] == "all") & (UN["pool"] == "gold") & (UN["key"] == k)].iloc[0]
        md.append(f"| {mlab_[k]} | " + " | ".join(cells) + f" | {ra['rho']:+.2f} ({pq(ra['pct'])}) | {ra['rho_D']:+.2f} / {ra['rho_V']:+.2f} | {rg['rho']:+.2f} ({pq(rg['pct'])}) | {fp(ra['mean_lo'])} / {fp(ra['mean_mid'])} / {fp(ra['mean_hi'])} |")
    md += [f"\n## 3a. Early warning (index pairs, all windows pooled by week): trades still open k weeks after entry\n",
           "| z window | week | open | deeper by ≥ 0.5: trades, ended good, mean extra | in between | recovering | ended bad that were deeper / ended good that were deeper |", "|---|---|---|---|---|---|---|"]
    for w, *_ in ZW.WINDOWS:
        for k in KS:
            e = EW[w]["index"][str(k)]
            if e[0] == 0:
                continue
            md.append(f"| {WLAB[w]} | {k} | {e[0]} | {e[1]}, {pc(e[2])}, {fp(e[3])} | {e[4]}, {pc(e[5])}, {fp(e[6])} | {e[7]}, {pc(e[8])}, {fp(e[9])} | {pc(e[10])} / {pc(e[11])} |")
    md += ["\n## 3b–4. Exits and entries: the six index pairs, all seven windows (42 pair-windows)\n",
           "| variant | pair-windows ahead of Rule R: whole window / 2016–2026 | ahead of 50/50: whole window / 2016–2026 | median excess over 50/50: whole window / 2016–2026 |", "|---|---|---|---|"]
    for vk, *_ in VARS:
        v = VALL[vk]; md.append(f"| {vlab[vk]} | {'—' if vk == 'base' else str(v[0]) + ' / ' + str(v[1])} | {v[2]} / {v[3]} | {pp(v[4])} / {pp(v[5])} |")
    md += [f"\n_Index pairs from 2006 (the rule's first day depends on the window) or 2022 (Micro250) to 2026-08-28; 2016–2026 = from 2016-01-01. Every pair, window and variant: `out/variants.csv`._\n"]
    md += [f"\n## 5. Walk-forward: trained on trades closed before each year, predicting the trades entered that year ({WF['lr']['first']} → {WF['lr']['last']})\n",
           "| model | AUC, index pairs (percentile among label shuffles; shuffles' median / 95th) | predicted above the training median: trades, mean extra, worked | below | gold pairs: AUC; above / below mean extra |", "|---|---|---|---|---|"]
    for kind in ("lr", "gb"):
        v = WF[kind]; md.append(f"| {v['label']} | {v['auc']:.3f} ({pq(v['pct'])}; {v['null_med']:.3f} / {v['null_hi']:.3f}) | {v['ix'][0]}, {fp(v['ix'][1])}, {pc(v['ix'][2])} | {v['ix'][3]}, {fp(v['ix'][4])}, {pc(v['ix'][5])} | {v['gold'][0]:.3f}; {fp(v['gold'][2])} / {fp(v['gold'][5])} |")
    md += ["\n**As entry filters** (six index pairs × seven windows, excess CAGR over 50/50 from 2010-01-01; random = 200 filters at the model's own entry rate):\n",
           "| model | pair-windows where the filter beats every entry | ahead of 50/50 (every entry) | above the random median (notable ≥ 95) | median percentile |", "|---|---|---|---|---|"]
    for kind in ("lr", "gb"):
        s = FSUM[kind]; md.append(f"| {WF[kind]['label']} | {s['beats_every']} of {s['n']} | {s['ahead_5050']} ({s['every_ahead_5050']}) | {s['above_random_median']} ({s['notable']}) | {s['median_pct']} |")
    open(os.path.join(OUT, "E07_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- the page
    UNJ = {}
    for _, r in UN.iterrows():
        UNJ.setdefault(r["window"], {}).setdefault(r["pool"], {})[r["key"]] = [r6(r["rho"]), r4(r["pct"], 3), r6(r["rho_D"]), r6(r["rho_V"]), int(r["n"]), r6(r["mean_lo"]), r6(r["mean_mid"]), r6(r["mean_hi"]), r6(r["win_lo"]), r6(r["win_mid"]), r6(r["win_hi"])]
    VJ, VW = {}, {}
    for _, r in VA.iterrows():
        VJ.setdefault(r["pair"], {}).setdefault(r["window"], {})[r["variant"]] = [r["trades"], r6(r["win"]), r6(r["mean_extra"]), r6(r["cagr"]), r6(r["excess"]), r6(r["excess_1626"]), r6(r["maxdd"]), r6(r["cagr_5050"]), r["open"]]
        VW.setdefault(r["pair"], {})[r["window"]] = [r["first"], r["last"], r["first_1626"]]
    DATA = {"windows": [[w, lab, f] for w, lab, f, _ in ZW.WINDOWS], "wlab": WLAB, "pairs": ORDER, "nice": {p: nice(p) for p in ORDER}, "pool": POOL, "defaults": {"w": ZW.DEFAULT, "pair": DEFAULT_PAIR, "pool": "index"},
            "lo": LO, "hi": HI, "perms": P, "ks": list(KS), "horizon": HORIZON, "deeper": DEEPER, "meas": [list(m) for m in MEAS], "fan": FAN, "hold": HOLD, "uni": UNJ, "ew": EW,
            "vars": [[v[0], v[1]] for v in VARS], "var": VJ, "vwin": VW, "vsum": VSUM, "vall": VALL, "wf": WF, "filt": FILT, "fsum": FSUM, "split": SPLIT.date().isoformat(), "wfstart": WF_START.date().isoformat(),
            "pal": {"blue": E1.BLUE, "orange": E1.ORANGE, "divLight": E1.DIV_LIGHT, "divDark": E1.DIV_DARK}}
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    def sel(i, lab, opts, default):
        return f"<label>{lab} <select id='{i}'>" + "".join(f"<option value='{html.escape(str(v), quote=True)}'{' selected' if str(v) == str(default) else ''}>{html.escape(t)}</option>" for v, t in opts) + "</select></label>"
    controls = (sel("zwin", "z window", [(w, WLAB[w]) for w, *_ in ZW.WINDOWS], ZW.DEFAULT) + sel("pool", "Pairs", [("index", "six index pairs"), ("gold", "four gold pairs")], "index")
                + sel("pair", "Pair (section 3)", [(p, nice(p)) for p in ORDER], DEFAULT_PAIR))
    lrw, gbw = WF["lr"], WF["gb"]; fl, fg = FSUM["lr"], FSUM["gb"]; best_v = max((vk for vk, *_ in VARS if vk != "base"), key=lambda vk: VALL[vk][3])
    key_card = (f"<p style='margin:4px 0'><b>Can good trades be picked before entry?</b> Trained only on trades already closed and predicting each next year ({lrw['first']} → {lrw['last']}, {lrw['n_ix']} index-pair trades): "
                f"logistic regression AUC {lrw['auc']:.3f} (percentile {lrw['pct']:.1f} among {NLR} label shuffles, whose median is {lrw['null_med']:.3f}); gradient-boosted trees {gbw['auc']:.3f} (percentile {gbw['pct']:.1f} among {NGB}). "
                f"0.5 = no better than a coin. Trades the logistic model rated above its training median: mean extra {fp(lrw['ix'][1])}, {pc(lrw['ix'][2])} worked; below: {fp(lrw['ix'][4])}, {pc(lrw['ix'][5])}. "
                f"As an entry filter (from 2010) it beats the rule with every entry in {fl['beats_every']} of {fl['n']} index pair-windows and is ahead of 50/50 in {fl['ahead_5050']} (every entry: {fl['every_ahead_5050']}); "
                f"against random skipping at the same rate it is above the median in {fl['above_random_median']} and notable in {fl['notable']} (trees: {fg['above_random_median']} and {fg['notable']}).</p>"
                f"<p style='margin:4px 0'><b>Managing the trade:</b> of the exits and entries tried, the one ahead of 50/50 in the most index pair-windows from 2016 is <i>{html.escape(vlab[best_v])}</i>: {VALL[best_v][3]} of 42 (Rule R: {VALL['base'][3]}); it beats Rule R in {VALL[best_v][1]} of 42 from 2016.</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Good and bad trades</title>"),
         "<header class='bar'><div class='bar-inner'><b>E07 · good and bad trades</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Good and bad trades: what tells them apart, and can the rule be managed better?</h1>",
         f"<p>The rule (T03's): every Friday close, z = (ratio − its trailing mean) ÷ its trailing sd over the chosen z window. From 50/50, the first week z crosses below −1 → hold only the first index; above +1 → hold only the second. Back to 50/50 when z returns to 0; trade the next session at the day's average price, 10.4 bps each way. "
         f"<b>Good trade</b> = the held index beat the other between entry and exit (before costs). {len(TR):,} trades over seven z windows and ten pairs ({len(CL):,} closed), the same as E06's. Generated {audit['run_date']} from the approved data of 2026-11-12. <b>Description, not a finding.</b></p>",
         "<div class='card'>" + key_card + "</div>",
         "<h2>1 · Good and bad trades, side by side</h2><p class='note'>After the signal week (week 0), how far z stayed stretched on the entry side (1 = the band, 0 = the mean, where the trade exits) and how the held index did against the other from the entry fill. Lines = medians, bands = middle half of the trades. The paths continue after a trade has closed, to show what the market did; the colour is the trade's actual outcome.</p>",
         "<div class='card'><div id='g_fan_s' class='plot'></div></div><div class='card'><div id='g_fan_x' class='plot'></div><div id='t_hold'></div></div>",
         f"<h2>2 · The ratio's character before entry</h2><p class='note'>Six measures of how straight or how back-and-forth the ratio had been moving, at the entry decision. Rank correlation with the trade's extra within each pair, against {P:,} shuffles within each pair (percentile ≤ 5 or ≥ 95 = notable). Positive = higher values went with better trades.</p>",
         "<div class='card'><div id='g_uni' class='plot'></div><div id='t_uni'></div></div>",
         "<h2>3 · The path after entry: can a bad trade be seen early, and can the rule be managed better?</h2><p class='note'><b>Early warning:</b> for trades still open k weeks after the signal: <i>deeper</i> = z at least 0.5 further from 0 than at entry; <i>recovering</i> = closer to 0 than at entry. "
         "<b>Variants</b> (each through the engine's book): a z-stop (back to 50/50 when z reaches ±2, ±2.5 or ±3 on the held side), a time stop (after 13, 26 or 52 weeks), or waiting for the turn before entering (the re-cross back into the band, or a 0.5 rebound from the extreme). The desk removed T02/T03's time limit on 2026-11-17; the time stops here are description only.</p>",
         "<div class='card'><div id='g_ew' class='plot'></div><div id='t_ew'></div></div>",
         "<div class='card'><div id='g_vall' class='plot'></div></div>",
         "<div class='card'><div id='g_var' class='plot'></div><div id='t_var'></div></div>",
         "<h2>4 · Can good trades be picked before entry? A walk-forward test</h2><p class='note'>Each year from 2010, two models (a logistic regression and small gradient-boosted trees) are trained on the index-pair trades already closed, using E06's 24 conditions, the six measures of section 2 and the window's length, and they rate the trades entered that year. "
         "AUC = the chance a random good trade is rated above a random bad one (0.5 = a coin). The same procedure on labels shuffled within each pair and window shows what luck gives. As a filter, the rule enters only where the rating is at or above the training median; it is compared with the rule with every entry, with 50/50, and with filters that skip entries at random at the same rate.</p>",
         "<div class='card'><div id='g_auc' class='plot'></div><div id='t_wf'></div></div>",
         "<div class='card'><div id='g_years' class='plot'></div></div>",
         "<div class='card'><div id='t_filt'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Trades and book.</b> explore/E01_ratio_cycles/variants.py and analysis/ratio_engine.book (fills next session at OHLC4, 10.4 bps per side, units held, daily NAV); the variants change only the state machine. With no added rule it equals Rule R on every pair and window (checked), and the trades equal E06's.</li>"
         "<li><b>Measures.</b> From the ratio's decision-week closes up to the entry decision; recomputed from the history cut at 12 dates per pair (checked equal).</li>"
         f"<li><b>Walk-forward.</b> Training uses only trades whose exit was before 1 January of the year predicted (asserted). The windows' trades overlap, so the pooled counts overstate independent evidence. Shuffles: {NLR} (logistic) and {NGB} (trees), within each pair and window.</li>"
         "<li><b>Limits.</b> The same history that refuted the rule, searched again from new angles; neighbouring variants share most of their trades; 2016–2026 is regime-heavy; gold is a futures continuation with an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); Micro250 pairs have four years. Design written before the first run: explore/E07_good_bad/E07.md.</li></ul></div>",
         "</main>", f"<script id='e07-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E07_goodbad.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.legend-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 0;vertical-align:middle}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e07-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function GOOD(){return isDark()?'#3987e5':'#2a78d6';} function BAD(){return isDark()?'#d95926':'#eb6834';}
function v(id){return document.getElementById(id).value;}
function wl(w){return D.wlab[w]||w;}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function fpp(x){if(x===null||x===undefined)return '—';let s=(x*100).toFixed(2);if(Number(s)===0)s='0.00';return (Number(s)>=0?'+':'')+s+' pp';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function f2(x,d){return x===null||x===undefined?'—':(+x).toFixed(d===undefined?2:d);}
function rgba(hex,a){const n=parseInt(hex.slice(1),16);return 'rgba('+(n>>16)+','+((n>>8)&255)+','+(n&255)+','+a+')';}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:100,b:50},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',font:{color:c.ink2,size:11}}},extra||{});}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function S(){return {w:v('zwin'), pool:v('pool'), p:v('pair')};}
function poolName(p){return p==='index'?'six index pairs':'four gold pairs';}

function fan(div, key, s, title, sub, ylab, ysuf, scale){
  const c=th(), F=D.fan[s.w][s.pool], tr=[];
  [['good',GOOD()],['bad',BAD()]].forEach(([g,col])=>{
    const q=F[g][key], n=F[g].n, x0=key==='s'?0:1, xs=q[0].map((_,i)=>i+x0), sc=y=>y===null?null:y*scale;
    tr.push({x:xs,y:q[2].map(sc),type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
    tr.push({x:xs,y:q[1].map(sc),type:'scatter',mode:'lines',line:{width:0},fill:'tonexty',fillcolor:rgba(col,0.18),showlegend:false,hoverinfo:'skip'});
    tr.push({x:xs,y:q[0].map(sc),type:'scatter',mode:'lines',line:{color:col,width:2.4},name:(g==='good'?'good trades':'bad trades')+' ('+n+')',hovertemplate:'week %{x}: median %{y:.2f}'+ysuf+'<extra>'+g+'</extra>'});
  });
  const shapes=key==='s'?[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:1,y1:1,line:{color:c.axis,width:1,dash:'dash'}},{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.ink2,width:1}}]
    :[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.ink2,width:1}}];
  Plotly.react(div,tr,layout(two(title,sub),420,{xaxis:ax({title:{text:'decision weeks after the signal',font:{color:c.ink2}}}),yaxis:ax({title:{text:ylab,font:{color:c.ink2}},ticksuffix:ysuf}),shapes:shapes,hovermode:'x unified',legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',traceorder:'normal',font:{color:c.ink2,size:11}}}),CFG);
}
function drawFans(s){
  const h=D.hold[s.w][s.pool], dates=(h[10]||'—')+' → '+(h[11]||'—');
  fan('g_fan_s','s',s,'How far z stayed stretched after the signal · z window '+wl(s.w)+' · '+poolName(s.pool)+' · '+dates,'z on the entry side: 1 = the band (dashed), 0 = the mean (the exit); median and middle half of the trades','stretch (z on the entry side)','',1);
  fan('g_fan_x','x',s,'The held index against the other, from the entry fill · z window '+wl(s.w)+' · '+poolName(s.pool)+' · '+dates,'held − other, cumulative, before costs; median and middle half of the trades','held − other','%',100);
  document.getElementById('t_hold').innerHTML=table(['','trades','median weeks held','median |z| at entry','median deepest |z| while open','median extra'],
    [['good',h[0],f2(h[2],0),f2(h[4]),f2(h[6]),fp(h[8])],['bad',h[1],f2(h[3],0),f2(h[5]),f2(h[7]),fp(h[9])]])+"<p class='note'>"+poolName(s.pool)+', z window '+wl(s.w)+', closed trades '+dates+'.</p>';
}
function drawUni(s){
  const c=th(), U=D.uni[s.w]&&D.uni[s.w][s.pool]?D.uni[s.w][s.pool]:null, UA=D.uni['all'][s.pool];
  const src=U||UA, lab=U?('z window '+wl(s.w)):'all windows pooled (no per-window result for the gold pairs)';
  const ms=D.meas, y=ms.map(m=>src[m[0]]?src[m[0]][0]:null), pct=ms.map(m=>src[m[0]]?src[m[0]][1]:null);
  const cols=pct.map(p=>p!==null&&(p<=D.lo||p>=D.hi)?GOOD():(isDark()?'#5f5e5a':'#c3c2b7'));
  Plotly.react('g_uni',[{type:'bar',orientation:'h',y:ms.map(m=>m[2]),x:y,marker:{color:cols},text:y.map((r,i)=>r===null?'':(r>=0?'+':'')+r.toFixed(2)+' ('+(pct[i]===null?'—':pct[i].toFixed(1))+')'),textposition:'outside',cliponaxis:false,
    hovertemplate:'%{y}: %{text}<extra></extra>'}],layout(two('The ratio\'s character at entry and the trade\'s extra · '+lab+' · '+poolName(s.pool),'rank correlation within each pair (percentile among '+D.perms.toLocaleString('en')+' shuffles); blue = notable; positive = higher values went with better trades'),400,
    {margin:{l:170,r:60,t:100,b:40},xaxis:ax({range:(function(){const m=Math.max(0.05,...y.filter(q=>q!==null).map(Math.abs))*1.7;return [-m,m];})(),title:{text:'rank correlation with the extra',font:{color:c.ink2}},zeroline:true,zerolinecolor:c.axis}),yaxis:ax({autorange:'reversed'})}),CFG);
  const rows=ms.map(m=>{const r=src[m[0]], a=UA[m[0]];return [m[1], r?(r[0]>=0?'+':'')+r[0].toFixed(2)+' ('+pq(r[1])+')':'—', a?(a[0]>=0?'+':'')+a[0].toFixed(2)+' ('+pq(a[1])+')':'—', a?f2(a[2])+' / '+f2(a[3]):'—', a?fp(a[5])+' / '+fp(a[6])+' / '+fp(a[7]):'—', a?pc(a[8])+' / '+pc(a[9])+' / '+pc(a[10]):'—'];});
  document.getElementById('t_uni').innerHTML=table(['measure',lab,'all windows pooled','before 2016 / from 2016','all windows: mean extra, low / middle / high third','worked, low / middle / high third'],rows);
}
function drawEW(s){
  const c=th(), E=D.ew[s.w][s.pool], ks=D.ks.filter(k=>E[String(k)][0]>0), tr=[];
  [['deeper by ≥ 0.5',1,BAD()],['in between',4,isDark()?'#898781':'#a8a69e'],['recovering',7,GOOD()]].forEach(([nm,i,col])=>{
    tr.push({type:'bar',x:ks.map(k=>'week '+k),y:ks.map(k=>E[String(k)][i+1]===null?null:E[String(k)][i+1]*100),name:nm,marker:{color:col},text:ks.map(k=>E[String(k)][i]?pc(E[String(k)][i+1])+' of '+E[String(k)][i]:''),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}, '+nm+': %{text} ended good<extra></extra>'});
  });
  Plotly.react('g_ew',tr,layout(two('Early warning · share of trades still open at week k that ended good, by where z was · z window '+wl(s.w)+' · '+poolName(s.pool),'deeper = z at least '+D.deeper+' further from 0 than at entry; recovering = closer to 0 than at entry'),420,{barmode:'group',xaxis:ax(),yaxis:ax({ticksuffix:'%',range:[0,110],title:{text:'ended good',font:{color:c.ink2}}})}),CFG);
  document.getElementById('t_ew').innerHTML=table(['week','still open','deeper: trades, ended good, mean extra','in between','recovering','of trades that ended bad: deeper','of trades that ended good: deeper'],
    D.ks.map(k=>{const e=E[String(k)];return e[0]?[k,e[0],e[1]+', '+pc(e[2])+', '+fp(e[3]),e[4]+', '+pc(e[5])+', '+fp(e[6]),e[7]+', '+pc(e[8])+', '+fp(e[9]),pc(e[10]),pc(e[11])]:[k,0,'—','—','—','—','—'];}));
}
function heatColors(){return isDark()?[[0,'#86b6ef'],[0.2,'#3987e5'],[0.4,'#34465e'],[0.5,'#383835'],[0.6,'#5e3a3a'],[0.8,'#e66767'],[1,'#f29a9a']]:[[0,'#104281'],[0.2,'#2a78d6'],[0.4,'#cde2fb'],[0.5,'#f0efec'],[0.6,'#f9d3d2'],[0.8,'#e34948'],[1,'#a32b2a']];}
function drawVar(s){
  const c=th(), V=D.vars;
  // across all windows: mean excess over 50/50 from 2016, six index pairs
  const ws=D.windows.map(x=>x[0]);
  const z1=V.map(vr=>ws.map(w=>D.vsum[w][vr[0]][5])), t1=V.map(vr=>ws.map(w=>D.vsum[w][vr[0]][3]+'/6')), h1=V.map(vr=>ws.map(w=>{const q=D.vsum[w][vr[0]];return vr[1]+', z window '+wl(w)+'<br>six index pairs, from 2016: mean excess over 50/50 '+fpp(q[5])+'; ahead of 50/50 in '+q[3]+' of 6; ahead of Rule R in '+q[1]+' of 6<br>whole window: mean excess '+fpp(q[4])+', ahead of 50/50 in '+q[2]+'; mean share worked '+pc(q[6])+', '+q[7]+' closed trades';}));
  const zr=Math.max(0.01,...z1.flat().filter(x=>x!==null).map(Math.abs));
  Plotly.react('g_vall',[{type:'heatmap',z:z1,x:ws.map(wl),y:V.map(x=>x[1]),text:t1,texttemplate:'%{text}',customdata:h1,hovertemplate:'%{customdata}<extra></extra>',zmin:-zr,zmax:zr,colorscale:heatColors(),xgap:2,ygap:2,
    colorbar:{thickness:12,tickformat:'.1%',tickfont:{color:c.muted},title:{text:'mean excess',side:'right',font:{color:c.ink2}}}}],
    layout(two('Exits and entries, every z window · six index pairs, from 2016-01-01','colour = mean excess CAGR over 50/50 bought once; label = index pairs (of 6) ahead of 50/50; red = ahead'),440,{margin:{l:250,r:24,t:100,b:40},xaxis:ax({side:'top',showgrid:false,showline:false}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
  // the chosen window: every pair
  const ps=D.pairs, z2=V.map(vr=>ps.map(p=>D.var[p][s.w][vr[0]][4])), h2=V.map(vr=>ps.map(p=>{const q=D.var[p][s.w][vr[0]], wn=D.vwin[p][s.w];return D.nice[p]+' · '+vr[1]+'<br>'+wn[0]+' → '+wn[1]+': excess over 50/50 '+fpp(q[4])+' (from 2016: '+fpp(q[5])+')<br>'+q[0]+' closed trades, '+pc(q[1])+' worked, mean extra '+fp(q[2])+'; CAGR '+fp(q[3])+', max drawdown '+fp(q[6],0);}));
  const zr2=Math.max(0.01,...z2.flat().filter(x=>x!==null).map(Math.abs));
  Plotly.react('g_var',[{type:'heatmap',z:z2,x:ps.map(p=>D.nice[p]),y:V.map(x=>x[1]),text:z2.map(r=>r.map(x=>x===null?'':(x*100).toFixed(1))),texttemplate:'%{text}',customdata:h2,hovertemplate:'%{customdata}<extra></extra>',zmin:-zr2,zmax:zr2,colorscale:heatColors(),xgap:2,ygap:2,
    colorbar:{thickness:12,tickformat:'.1%',tickfont:{color:c.muted},title:{text:'excess',side:'right',font:{color:c.ink2}}}}],
    layout(two('Every pair at z window '+wl(s.w)+' · excess CAGR over 50/50 bought once, whole window (pp a year)','index pairs from the rule\'s first day in 2006 (Micro250: 2022), gold pairs from 2015, to 2026-08-28; red = ahead of 50/50'),460,{margin:{l:250,r:24,t:100,b:90},xaxis:ax({tickangle:-30,showgrid:false,showline:false}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
  const wn=D.vwin[s.p][s.w];
  document.getElementById('t_var').innerHTML="<p class='note' style='margin:8px 0 4px'><b>"+D.nice[s.p]+'</b>, z window '+wl(s.w)+', '+wn[0]+' → '+wn[1]+' (from 2016: '+wn[2]+' →):</p>'+
    table(['variant','closed trades','worked','mean extra','CAGR','excess over 50/50','from 2016','max drawdown'],V.map(vr=>{const q=D.var[s.p][s.w][vr[0]];return [vr[1],q[0]+(q[8]?' (+'+q[8]+' open)':''),pc(q[1]),fp(q[2]),fp(q[3]),fpp(q[4]),fpp(q[5]),fp(q[6],0)];}))+
    "<p class='note'>50/50 bought once: CAGR "+fp(D.var[s.p][s.w].base[7])+'.</p>';
}
function drawWF(s){
  const c=th(), M=['lr','gb'], cols={lr:GOOD(),gb:BAD()}, tr=[], shapes=[], ann=[];
  M.forEach((m,i)=>{const w=D.wf[m];tr.push({type:'histogram',x:w.null,name:w.label+': shuffled labels ('+w.null.length+')',marker:{color:rgba(cols[m],0.45)},xbins:{size:0.005},hovertemplate:'AUC %{x}: %{y} shuffles<extra></extra>'});
    shapes.push({type:'line',xref:'x',yref:'paper',x0:w.auc,x1:w.auc,y0:0,y1:1,line:{color:cols[m],width:2.5}});
    ann.push({x:w.auc,xref:'x',y:1-i*0.1,yref:'paper',text:w.label+' '+w.auc.toFixed(3)+' (pct '+w.pct.toFixed(1)+')',showarrow:false,xanchor:'left',xshift:4,font:{color:c.ink2,size:11}});});
  shapes.push({type:'line',xref:'x',yref:'paper',x0:0.5,x1:0.5,y0:0,y1:1,line:{color:c.axis,width:1,dash:'dash'}});
  Plotly.react('g_auc',tr,layout(two('Walk-forward AUC, six index pairs · '+D.wf.lr.first+' → '+D.wf.lr.last+' ('+D.wf.lr.n_ix+' trades, all windows)','lines = the real models; bars = the same procedure on labels shuffled within each pair and window; 0.5 (dashed) = a coin'),400,{barmode:'overlay',shapes:shapes,annotations:ann,xaxis:ax({title:{text:'AUC',font:{color:c.ink2}}}),yaxis:ax({title:{text:'shuffles',font:{color:c.ink2}}})}),CFG);
  const yt=[];M.forEach(m=>{const Y=D.wf[m].years.filter(r=>r[1]!==null);yt.push({x:Y.map(r=>r[0]),y:Y.map(r=>r[1]),type:'scatter',mode:'lines+markers',name:D.wf[m].label,line:{color:cols[m],width:2},marker:{size:7},customdata:Y.map(r=>r[2]+' trades; trained on '+r[3]),hovertemplate:'%{x}: AUC %{y:.3f} (%{customdata})<extra>'+D.wf[m].label+'</extra>'});});
  Plotly.react('g_years',yt,layout(two('Walk-forward AUC year by year · six index pairs','each year rated by models trained on the trades closed before it'),360,{xaxis:ax({dtick:2}),yaxis:ax({title:{text:'AUC',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0.5,y1:0.5,line:{color:c.axis,width:1,dash:'dash'}}]}),CFG);
  document.getElementById('t_wf').innerHTML=table(['model','AUC (percentile among shuffles)','rated above the training median: trades, mean extra, worked','rated below','gold pairs: AUC; above / below mean extra'],M.map(m=>{const w=D.wf[m];return [w.label,w.auc.toFixed(3)+' ('+pq(w.pct)+'; shuffles\' median '+w.null_med.toFixed(3)+')',w.ix[0]+', '+fp(w.ix[1])+', '+pc(w.ix[2]),w.ix[3]+', '+fp(w.ix[4])+', '+pc(w.ix[5]),(w.gold[0]===null?'—':w.gold[0].toFixed(3))+'; '+fp(w.gold[2])+' / '+fp(w.gold[5])];}));
  const rows=[];D.pairs.filter(p=>D.pool[p]==='index').forEach(p=>{const a=D.filt.lr[s.w][p], b=D.filt.gb[s.w][p];rows.push([D.nice[p]+' ('+a[5]+' → '+a[6]+')',fpp(a[2]),fpp(a[1])+' (allowed in '+pc(a[0])+' of weeks)',fpp(a[3])+', '+pq(a[4]),fpp(b[1])+' ('+pc(b[0])+')',fpp(b[3])+', '+pq(b[4])]);});
  const fl=D.fsum.lr, fg=D.fsum.gb;
  document.getElementById('t_filt').innerHTML="<p class='note' style='margin:4px 0'><b>The models as entry filters</b> · z window "+wl(s.w)+' · excess CAGR over 50/50 bought once from '+D.wfstart+'. Over all 42 index pair-windows: logistic beats every entry in '+fl.beats_every+', is ahead of 50/50 in '+fl.ahead_5050+' (every entry: '+fl.every_ahead_5050+'), above the random median in '+fl.above_random_median+' (notable '+fl.notable+'); trees: '+fg.beats_every+', '+fg.ahead_5050+', '+fg.above_random_median+' ('+fg.notable+').</p>'+
    table(['pair','every entry','logistic filter (weeks allowed)','random at that rate: median, percentile','trees filter (weeks allowed)','random: median, percentile'],rows);
}
function render(){
  const s=S(); try{localStorage.setItem('e07-state',JSON.stringify(s));}catch(e){}
  drawFans(s); drawUni(s); drawEW(s); drawVar(s); drawWF(s);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  try{const st=JSON.parse(localStorage.getItem('e07-state')||'null');if(st){[['zwin','w'],['pool','pool'],['pair','p']].forEach(([id,k])=>{const el=document.getElementById(id);if(st[k]!==undefined&&Array.from(el.options).some(o=>o.value===String(st[k])))el.value=String(st[k]);});}}catch(e){}
  ['zwin','pool','pair'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
