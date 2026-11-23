"""
E10 (EXPLORATORY): small caps against gold beside SPY against gold; band touches against z signals; weekly supertrend
(four combinations, four weight schemes) and momentum rotation (12-1, 12 and 6 months) on the US pairs (the commodity
ETFs and GLD against SPY) and the Indian pairs (large, mid and small caps and gold), beside Rule R.
Design: explore/E10_trend_rotation/E10.md (written before the first run).

    python explore/E10_trend_rotation/rotation.py [--draws 500]

Every number on the page and in E10_results.md is computed here; the page's script only picks the numbers of the chosen
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

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "explore", "E01_ratio_cycles"))
from analysis.approved_data import GOLD, fingerprints, load_daily, load_us_etf, load_us_gold, load_us_small  # noqa: E402
from analysis.ratio_engine import PIT, book, comparators, counts, decide, null_A, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (page styling, percentile, notable)
import zwin as ZW  # noqa: E402  (z of a series)
import variants as VR  # noqa: E402  (Rule R's states, the T03 check)

COST = VR.COST; C = COST / 2 / 1e4; LO, HI = E1.LO, E1.HI
US_MAIN = ["GCC", "FTGC", "COMT", "PDBC", "BCI", "BCD", "COM", "CMDY", "SDCI"]
MARKETS = (("us", 0, [f"{e}|SPY" for e in US_MAIN] + ["GLD|SPY", "IJR|SPY", "IJR|GLD"]),   # IJR added 2026-11-23 at the end (seeds kept)
           ("india", 1, ["N100|MC100", "N100|SC100", "MC100|SC100", f"N100|{GOLD}", f"MC100|{GOLD}", f"SC100|{GOLD}"]))
MKT_LAB = {"us": "US", "india": "India"}
DEFAULT_PAIR = f"SC100|{GOLD}"
K0 = 52                                  # the common start: decision week 53 of the pair (the 12-month lookback needs 52)
ST_N, ST_M = 10, 3.0                     # the first run's supertrend (Wilder ATR 10, multiplier 3), kept as a check since 2026-11-23
DESK_N, DESK_M = 5, 2.0                  # the desk's Weekly LW supertrend (5-bar linear-weighted ATR, multiplier 2), from 2026-11-23
MOM = (("m12_1", "momentum 12-1", 52, 4, 1), ("m12", "momentum 12 months", 52, 0, 2), ("m6", "momentum 6 months", 26, 0, 3))
# weights (first leg, second leg, cash) for the combinations 0 both buy, 1 only the first leg buy, 2 only the second, 3 both sell
SCHEMES = {"S1": [[.5, .5, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], "S2": [[.5, .5, 0], [1, 0, 0], [0, 1, 0], [.5, .5, 0]],
           "S3": [[.5, .5, 0], [.75, .25, 0], [.25, .75, 0], [.5, .5, 0]], "S4": [[.5, .5, 0], [.5, 0, .5], [0, .5, .5], [0, 0, 1]]}
SCHEMES = {k: np.array(v, float) for k, v in SCHEMES.items()}
SCH_LAB = {"S1": "S1 rotate, cash if both sell", "S2": "S2 rotate, always invested", "S3": "S3 tilt 75/25", "S4": "S4 each leg on its own"}
RULES = ["R52", "m12_1", "m12", "m6", "S1", "S2", "S3", "S4"]
RLAB = {"R52": "Rule R (±1 sd, 52 weeks)", **{m[0]: m[1] for m in MOM}, **{k: "supertrend " + v for k, v in SCH_LAB.items()}}
RSHORT = {"R52": "Rule R", "m12_1": "Mom 12-1", "m12": "Mom 12", "m6": "Mom 6", "S1": "ST S1", "S2": "ST S2", "S3": "ST S3", "S4": "ST S4"}
RULE_ID = {"R52": 0, "m12_1": 1, "m12": 2, "m6": 3, "ST": 4}
TW = {"A": (1.0, 0.0, 0.0), "B": (0.0, 1.0, 0.0), "N": (0.5, 0.5, 0.0)}
SIM_START, SIM_END = pd.Timestamp("2015-01-12"), pd.Timestamp("2026-08-28")


def leg(x): return "gold" if x == GOLD else x


def nice(p):
    a, b = p.split("|"); return f"{leg(a)} ÷ {leg(b)}"


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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------------- the book
def gbook(A: dict, k_dec: np.ndarray, W: np.ndarray, rec: bool = False):
    """Daily NAV from the first decision day (1.0) following weekly target weights W (rows: first leg, second leg, cash).
    The engine's convention: 50/50 by value on the next session, then a trade at the next session's fills whenever the
    target changes (the first row is not traded on, as in analysis.ratio_engine.book); selling before buying, 10.4 bps per
    side on value traded, so the post-trade value V' solves V' = V - c (|w_a V' - v_a| + |w_b V' - v_b|). Cash earns 0."""
    ca, cb, fa, fb = A["ca"], A["cb"], A["fa"], A["fb"]; n = len(ca); k0 = int(k_dec[0]); kt = k0 + 1
    ua, ub, cash = 0.5 / (fa[kt] * (1 + C)), 0.5 / (fb[kt] * (1 + C)), 0.0
    cur = (0.5, 0.5, 0.0); starts = [kt]; U = [(ua, ub, cash)]; sw = []
    for i in range(1, len(k_dec)):
        w = (W[i, 0], W[i, 1], W[i, 2])
        if w == cur:
            continue
        k = int(k_dec[i]) + 1
        if k >= n:
            break                                         # a decision on the last day cannot be executed
        pa, pb = fa[k], fb[k]; va, vb = ua * pa, ub * pb; V = va + vb + cash; Vn = V
        for _ in range(6):
            Vn = V - C * (abs(w[0] * Vn - va) + abs(w[1] * Vn - vb))
        ua, ub = w[0] * Vn / pa, w[1] * Vn / pb; cash = Vn - w[0] * Vn - w[1] * Vn
        if rec:
            sw.append({"i": i, "k": k, "from": cur, "to": w, "fill_a": pa, "fill_b": pb, "cost": (V - Vn) / V})
        cur = w; starts.append(k); U.append((ua, ub, cash))
    nav = np.empty(n - k0); nav[0] = 1.0; bounds = starts + [n]
    for j, u in enumerate(U):
        s, e = bounds[j], bounds[j + 1]; nav[s - k0:e - k0] = u[0] * ca[s:e] + u[1] * cb[s:e] + u[2]
    return nav, sw


def nstats(nav: np.ndarray, days: pd.DatetimeIndex) -> dict:
    d = (days[-1] - days[0]).days; r = np.diff(np.log(nav))
    return {"cagr": float((nav[-1] / nav[0]) ** (365.25 / d) - 1), "maxdd": float((nav / np.maximum.accumulate(nav) - 1).min()), "vol": float(r.std(ddof=1) * np.sqrt(252))}


def quick(nav, d_days):
    return (nav[-1] / nav[0]) ** (365.25 / d_days) - 1, float((nav / np.maximum.accumulate(nav) - 1).min())


# ------------------------------------------------------------------------------------------------------- the signals
def weekly_bars(pf: pd.DataFrame, ha, la, hb, lb) -> pd.DataFrame:
    """Each leg's weekly bar on the pair's common calendar: the highest high and lowest low of the week's days, the close
    of its last day (the decision day)."""
    iso = pf.index.isocalendar(); key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    df = pd.DataFrame({"k": key, "ha": ha, "la": la, "ca": pf["ca"].to_numpy(), "hb": hb, "lb": lb, "cb": pf["cb"].to_numpy()}, index=pf.index)
    g = df.groupby("k", sort=False)
    out = pd.DataFrame({"ha": g["ha"].max(), "la": g["la"].min(), "ca": g["ca"].last(), "hb": g["hb"].max(), "lb": g["lb"].min(), "cb": g["cb"].last()})
    out.index = df.index[~pd.Series(key).duplicated(keep="last").to_numpy()]
    return out


def desk_bars(d: pd.DataFrame, closes_only: bool) -> pd.DataFrame:
    """The desk's weekly bars of one leg on its own sessions: labelled by Friday (W-FRI); high = the week's highest high
    (or close, for a closes-only series), low = the lowest low (or close), close = the last close; last_session."""
    h = d["close"] if closes_only else d["high"]; lo = d["close"] if closes_only else d["low"]
    r = pd.DataFrame({"high": h, "low": lo, "close": d["close"]}).resample("W-FRI")
    out = pd.DataFrame({"high": r["high"].max(), "low": r["low"].min(), "close": r["close"].last()})
    out["last_session"] = pd.Series(d.index, index=d.index).resample("W-FRI").last()
    return out.dropna(subset=["close"])


def desk_supertrend(bars: pd.DataFrame, n: int = DESK_N, m: float = DESK_M):
    """The desk's Weekly LW supertrend, as in wm-STstrat / wm-IDXrot / wm-IDXgate (reproduced, not re-derived): true range
    max(H-L, |H-prevC|, |L-prevC|); ATR = the last n true ranges weighted 1..n / (n(n+1)/2), newest heaviest; bands
    (H+L)/2 ± m ATR, ratcheted unless the previous close broke through; +1 when a close crosses above the final upper
    band from -1, -1 when a close crosses below the final lower band from +1; the first computable bar is +1 only if its
    close is above its upper band. Returns (state: +1 / -1, NaN before the first computable bar; stop: the band in force)."""
    h, lo, c = bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float); L = len(c)
    tr = np.empty(L); tr[0] = h[0] - lo[0]
    tr[1:] = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    w = np.arange(1, n + 1, dtype=float); w /= w.sum(); atr = np.full(L, np.nan)
    if L >= n:
        atr[n - 1:] = np.convolve(tr, w[::-1], mode="valid")
    hl2 = (h + lo) / 2; ub, db = hl2 + m * atr, hl2 - m * atr
    up, dn, st = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan)
    first = n - 1
    for i in range(first, L):
        if i == first:
            up[i], dn[i] = ub[i], db[i]; st[i] = 1.0 if c[i] > up[i] else -1.0; continue
        up[i] = ub[i] if (ub[i] < up[i - 1] or c[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = db[i] if (db[i] > dn[i - 1] or c[i - 1] < dn[i - 1]) else dn[i - 1]
        st[i] = 1.0 if (st[i - 1] == -1 and c[i] > up[i]) else (-1.0 if (st[i - 1] == 1 and c[i] < dn[i]) else st[i - 1])
    return st, np.where(st == 1, dn, up)


def cal_end_of(idx: pd.DatetimeIndex) -> pd.Series:
    """The leg's trading calendar: the last session of each W-FRI week (the day that week's bar is complete)."""
    return pd.Series(idx, index=idx).resample("W-FRI").last().dropna()


def leg_state_at(dec: pd.DatetimeIndex, bars: pd.DataFrame, state: np.ndarray, cal_end: pd.Series) -> np.ndarray:
    """A leg's state at each decision day: its latest bar whose week has ended by that day on the leg's own calendar
    (a bar is usable once its last session has closed, as in the desk's repos)."""
    usable = cal_end.reindex(bars.index).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(usable, dec.values, side="right") - 1
    return np.where(pos >= 0, state[np.clip(pos, 0, None)], np.nan)


def supertrend_wilder(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = ST_N, m: float = ST_M):
    """The first run's supertrend (TradingView's ta.supertrend, Wilder ATR): direction True = buy, False = sell; the line.
    Kept as a check since 2026-11-23."""
    L = len(c); tr = np.empty(L); tr[0] = h[0] - lo[0]
    tr[1:] = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    atr = np.full(L, np.nan)
    if L >= n:
        atr[n - 1] = tr[:n].mean()
        for i in range(n, L):
            atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    hl2 = (h + lo) / 2; up = hl2 + m * atr; dn = hl2 - m * atr
    fu = np.full(L, np.nan); fl = np.full(L, np.nan); buy = np.zeros(L, bool); line = np.full(L, np.nan)
    for i in range(L):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(atr[i - 1]):
            fu[i], fl[i], b = up[i], dn[i], False
        else:
            fl[i] = dn[i] if (dn[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
            fu[i] = up[i] if (up[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
            b = (c[i] > fu[i]) if not buy[i - 1] else not (c[i] < fl[i])
        buy[i] = b; line[i] = fl[i] if b else fu[i]
    return buy, line


def mom_states(ca: np.ndarray, cb: np.ndarray, look: int, skip: int) -> np.ndarray:
    """Hold the leg with the higher momentum: the return from `look` weeks back to `skip` weeks back."""
    out = np.array(["N"] * len(ca), dtype=object)
    for t in range(look, len(ca)):
        ma, mb = ca[t - skip] / ca[t - look], cb[t - skip] / cb[t - look]
        out[t] = "A" if ma > mb else ("B" if ma < mb else (out[t - 1] if out[t - 1] != "N" else "B"))
    return out


def combo_of(ba: np.ndarray, bb: np.ndarray) -> np.ndarray:
    return np.where(ba & bb, 0, np.where(ba, 1, np.where(bb, 2, 3)))


def shuffle_spells(x: np.ndarray, rng) -> np.ndarray:
    """N-S: the spells of each state permuted among that state's positions (two alternating states, so every count is kept)."""
    x = np.asarray(x); ch = np.flatnonzero(x[1:] != x[:-1]) + 1; st = np.r_[0, ch]; ln = np.diff(np.r_[st, len(x)]); vals = x[st]; out = ln.copy()
    for v in np.unique(vals):
        idx = np.flatnonzero(vals == v); out[idx] = ln[idx][rng.permutation(len(idx))]
    return np.repeat(vals, out)


def spell_counts(x) -> tuple:
    x = np.asarray(x); ch = int((x[1:] != x[:-1]).sum()); return (ch, int(x.sum()) if x.dtype == bool else tuple(sorted(pd.Series(x).value_counts().items())))


# ------------------------------------------------------------------------------------------------------- band touches
def touches(pf: pd.DataFrame) -> dict:
    wk = pf.loc[pf["decision"]]; r = wk["ratio"]; roll = r.rolling(52, min_periods=ZW.min_obs(52)); m, s = roll.mean(), roll.std()
    z = ZW.z_of(r, 52); ok = z.notna().to_numpy()
    pb = np.where(r < m - s, -1, np.where(r > m + s, 1, 0)); pzv = np.where(z < -1, -1, np.where(z > 1, 1, 0))
    band_mism = int((pb[ok] != pzv[ok]).sum())
    d = decide(pf, 52, "E1", PIT()); bd = d[d["mean"].notna()]
    zs, _ = VR.rule_states(z.to_numpy()); stz = pd.Series(zs, index=wk.index).reindex(bd.index)
    state_mism = int((stz.astype(str).to_numpy() != bd["state_after"].astype(str).to_numpy()).sum())
    pz = pd.Series(pzv.astype(float), index=wk.index).where(ok); prev = pz.shift(1)
    sig = (pz.abs() == 1) & prev.notna() & (prev != pz)
    # daily closes against the band known that day: the previous decision week's mean ± sd
    lo_prev, hi_prev = (m - s).shift(1).to_numpy(), (m + s).shift(1).to_numpy()
    dec = wk.index; widx = np.searchsorted(dec.values, pf.index.values, side="left")
    rd = pf["ratio"].to_numpy(); side = np.where(rd < lo_prev[widx], -1, np.where(rd > hi_prev[widx], 1, 0))
    seen, early, days_early = 0, 0, []
    sig_j = np.flatnonzero(sig.to_numpy())
    for j in sig_j:
        X = int(pz.iloc[j]); days_ = np.flatnonzero(widx == j); hit = days_[side[days_] == X]
        if len(hit):
            seen += 1
            if hit[0] < days_[-1]:
                early += 1; days_early.append(int(days_[-1] - hit[0]))
    prev_side = np.r_[0, side[:-1]]; ft = np.flatnonzero((side != 0) & (side != prev_side) & ~np.isnan(lo_prev[widx]))
    closed = sum(1 for k in ft if pzv[widx[k]] == side[k] and ok[widx[k]])
    became = sum(1 for k in ft if sig.iloc[widx[k]] and pzv[widx[k]] == side[k])
    band_days = pf.index[~np.isnan(lo_prev[widx])]; yrs = (band_days[-1] - band_days[0]).days / 365.25
    # weekly 52-week z against daily 250-day z
    z52 = VR.z_at_decisions(pf, "w52"); zd = VR.z_at_decisions(pf, "d250")
    s52 = np.array(VR.rule_states(z52.to_numpy())[0], dtype=object); sd = np.array(VR.rule_states(zd.to_numpy())[0], dtype=object)
    both = (z52.notna() & zd.notna()).to_numpy()

    def entries(s):
        return [(i, s[i]) for i in range(1, len(s)) if s[i] in ("A", "B") and s[i - 1] != s[i] and both[i]]
    e52, ed = entries(s52), entries(sd)
    matched = sum(1 for i, x in e52 if any(abs(j - i) <= 2 and y == x for j, y in ed))
    return {"weeks": int(ok.sum()), "first": dec[ok][0].date().isoformat(), "last": dec[-1].date().isoformat(), "band_vs_z_mismatch": band_mism, "engine_vs_z_state_mismatch": state_mism,
            "weekly_signals": int(len(sig_j)), "seen_daily_same_week": seen / len(sig_j) if len(sig_j) else np.nan, "seen_earlier": early / len(sig_j) if len(sig_j) else np.nan,
            "median_days_early": float(np.median(days_early)) if days_early else np.nan, "daily_first_touches": int(len(ft)), "daily_touches_a_year": len(ft) / yrs,
            "touch_closed_beyond": closed / len(ft) if len(ft) else np.nan, "touch_became_signal": became / len(ft) if len(ft) else np.nan,
            "entries_w52": len(e52), "entries_d250": len(ed), "w52_matched_d250": matched / len(e52) if e52 else np.nan, "same_state_w52_d250": float((s52[both] == sd[both]).mean())}


# ------------------------------------------------------------------------------------------------------- similarity
def iso_key(idx):
    iso = idx.isocalendar(); return (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()


def similarity(pf_in: pd.DataFrame, pf_us: pd.DataFrame, series: bool = False, us_eq: str = "cb") -> dict:
    """Equity ÷ gold in India (X ÷ MCX gold) against a US equity ÷ gold (SPY ÷ GLD from the GLD|SPY frame, us_eq = 'cb';
    IJR ÷ GLD from the IJR|GLD frame, us_eq = 'ca'), on their common ISO weeks between SIM_START and SIM_END."""
    us_gd = "ca" if us_eq == "cb" else "cb"
    wi, wu = pf_in.loc[pf_in["decision"]], pf_us.loc[pf_us["decision"]]
    zi_all = ZW.z_of(wi["ca"] / wi["cb"], 52); zu_all = ZW.z_of(wu[us_eq] / wu[us_gd], 52)
    si = pd.Series(VR.rule_states(VR.z_at_decisions(pf_in, "w52").to_numpy())[0], index=wi.index).map({"A": "equity", "B": "gold", "N": "50/50"})
    su = pd.Series(VR.rule_states(VR.z_at_decisions(pf_us, "w52").to_numpy())[0], index=wu.index).map(
        {"A": "gold", "B": "equity", "N": "50/50"} if us_eq == "cb" else {"A": "equity", "B": "gold", "N": "50/50"})
    zi_ok = VR.z_at_decisions(pf_in, "w52").notna(); zu_ok = VR.z_at_decisions(pf_us, "w52").notna()
    ki, ku = pd.Series(wi.index, index=iso_key(wi.index)), pd.Series(wu.index, index=iso_key(wu.index))
    kus = set(ku.index)
    common = [k for k in ki.index if k in kus and SIM_START <= ki[k] <= SIM_END and SIM_START <= ku[k] <= SIM_END]
    di, du = pd.DatetimeIndex(ki[common].to_numpy()), pd.DatetimeIndex(ku[common].to_numpy())
    eq_i, gd_i, gd_u, eq_u = wi["ca"].reindex(di).to_numpy(), wi["cb"].reindex(di).to_numpy(), wu[us_gd].reindex(du).to_numpy(), wu[us_eq].reindex(du).to_numpy()
    ri, ru = eq_i / gd_i, eq_u / gd_u; li, lu = np.log(ri), np.log(ru)
    d1i, d1u = np.diff(li), np.diff(lu); d4i, d4u = li[4:] - li[:-4], lu[4:] - lu[:-4]
    zi, zu = zi_all.reindex(di).to_numpy(), zu_all.reindex(du).to_numpy(); zok = ~np.isnan(zi) & ~np.isnan(zu)
    either = zok & ((np.abs(zi) > 1) | (np.abs(zu) > 1)); same_way = zok & (((zi > 1) & (zu > 1)) | ((zi < -1) & (zu < -1)))
    rk = (zi_ok.reindex(di).to_numpy() & zu_ok.reindex(du).to_numpy()); kind_same = (si.reindex(di).to_numpy() == su.reindex(du).to_numpy())[rk]
    yrs = (di[-1] - di[0]).days / 365.25; cg = lambda x: float((x[-1] / x[0]) ** (1 / yrs) - 1)  # noqa: E731
    out = {"weeks": len(common), "first": di[0].date().isoformat(), "last": di[-1].date().isoformat(), "first_us": du[0].date().isoformat(), "last_us": du[-1].date().isoformat(),
           "corr_weekly": float(np.corrcoef(d1i, d1u)[0, 1]), "corr_4week": float(np.corrcoef(d4i, d4u)[0, 1]), "corr_z": float(np.corrcoef(zi[zok], zu[zok])[0, 1]),
           "z_same_side": float((np.sign(zi[zok]) == np.sign(zu[zok])).mean()), "z_weeks": int(zok.sum()), "weeks_either_beyond": int(either.sum()), "both_beyond_same_way": int(same_way.sum()),
           "rule_same_kind": float(kind_same.mean()), "rule_weeks": int(rk.sum()),
           "corr_equity_legs": float(np.corrcoef(np.diff(np.log(eq_i)), np.diff(np.log(eq_u)))[0, 1]), "corr_gold_legs": float(np.corrcoef(np.diff(np.log(gd_i)), np.diff(np.log(gd_u)))[0, 1]),
           "cagr_eq_in": cg(eq_i), "cagr_gold_in": cg(gd_i), "cagr_eq_us": cg(eq_u), "cagr_gold_us": cg(gd_u), "cagr_ratio_in": cg(ri), "cagr_ratio_us": cg(ru)}
    if series:
        rc = pd.Series(d1i).rolling(52).corr(pd.Series(d1u)).to_numpy()
        out["series"] = {"t": [d.date().isoformat() for d in di], "eq_in": eq_i / eq_i[0], "gold_in": gd_i / gd_i[0], "eq_us": eq_u / eq_u[0], "gold_us": gd_u / gd_u[0],
                         "r_in": ri / ri[0], "r_us": ru / ru[0], "z_in": zi, "z_us": zu, "rc": np.r_[np.nan, rc]}
    return out


# ------------------------------------------------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=500); ND = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    pin = load_daily(); pus = load_us_etf(); pus.update(load_us_gold()); pus.update(load_us_small())
    checks = {"close_max_abs_diff": 0.0, "engine_book_max_abs_diff": 0.0, "engine_book_runs": 0, "benchmarks_vs_engine_max_abs_diff": 0.0, "weekly_bar_index_mismatch": [],
              "t03_mismatch": [], "cut_checks": 0, "cut_violations": [], "null_draws": 0, "null_count_mismatch": 0, "st_null_combo_weeks_changed": 0, "band_vs_z_mismatch": 0, "engine_vs_z_state_mismatch": 0, "st_missing_after_start": 0}
    LEGACY = []
    PAIRS, STRAT, CURV, RR, STL, STS, TOUCH, SW = [], [], {}, {}, {}, {}, [], []
    PF = {}
    for mk, mid, pairs in MARKETS:
        px = pus if mk == "us" else pin
        for pid, pair in enumerate(pairs):
            a, b = pair.split("|"); pf = pair_frame(px, pair, "plain"); PF[pair] = pf
            checks["close_max_abs_diff"] = max(checks["close_max_abs_diff"], float((pf["ca"] - px[a]["close"].reindex(pf.index)).abs().max()), float((pf["cb"] - px[b]["close"].reindex(pf.index)).abs().max()))
            dec = pf.index[pf["decision"]]; kd = pf.index.get_indexer(dec[K0:]); days = pf.index[kd[0]:]; dd_ = (days[-1] - days[0]).days
            A = {k: pf[k].to_numpy() for k in ("ca", "cb", "fa", "fb")}; wk = pf.loc[dec]
            # the signals on every decision week of the pair
            z = VR.z_at_decisions(pf, "w52"); sR = np.array(VR.rule_states(z.to_numpy())[0], dtype=object)
            sM = {m[0]: mom_states(wk["ca"].to_numpy(), wk["cb"].to_numpy(), m[2], m[3]) for m in MOM}
            # the desk's Weekly LW supertrend on each leg's own weekly bars, read at the pair's decision days
            closes_only = mk == "us"; LEG = {}
            for x in (a, b):
                bx = desk_bars(px[x], closes_only); sx, stopx = desk_supertrend(bx); LEG[x] = (bx, sx, stopx, cal_end_of(px[x].index))
            sa, sb = (leg_state_at(dec, LEG[x][0], LEG[x][1], LEG[x][3]) for x in (a, b))
            checks["st_missing_after_start"] += int(np.isnan(sa[K0:]).sum() + np.isnan(sb[K0:]).sum())
            ba, bb = sa == 1, sb == 1
            cmb = combo_of(ba, bb)
            agree = None
            if mk == "india":   # check: the same legs with closes-only bars
                cc = {}
                for x in (a, b):
                    bx_ = desk_bars(px[x], True); cc[x] = leg_state_at(dec, bx_, desk_supertrend(bx_)[0], LEG[x][3]) == 1
                ca_, cb_ = cc[a], cc[b]
                agree = {"a": float((ca_[K0:] == ba[K0:]).mean()), "b": float((cb_[K0:] == bb[K0:]).mean())}
                Wc = SCHEMES["S1"][combo_of(ca_, cb_)[K0:]]; nvc, _ = gbook(A, kd, Wc); agree["S1_cagr_closes"] = nstats(nvc, days)["cagr"]
            # check: the first run's (10, 3) Wilder supertrend on the pair's common-calendar weekly bars
            hl = (lambda x, col: px[x][col].reindex(pf.index).to_numpy()) if mk == "india" else (lambda x, col: pf["ca" if x == a else "cb"].to_numpy())
            bars = weekly_bars(pf, hl(a, "high"), hl(a, "low"), hl(b, "high"), hl(b, "low"))
            if not bars.index.equals(dec):
                checks["weekly_bar_index_mismatch"].append(pair)
            wa_, _ = supertrend_wilder(bars["ha"].to_numpy(), bars["la"].to_numpy(), bars["ca"].to_numpy()); wb_, _ = supertrend_wilder(bars["hb"].to_numpy(), bars["lb"].to_numpy(), bars["cb"].to_numpy())
            cmb_w = combo_of(wa_, wb_)
            # point in time: everything recomputed on the history cut at 8 decision weeks
            for T in np.unique(np.linspace(K0 + 10, len(dec) - 1, 8).astype(int)):
                cut = pf.loc[: dec[T]]; wc = cut.loc[cut["decision"]]; checks["cut_checks"] += 1
                zc = VR.z_at_decisions(cut, "w52"); okz = np.allclose(zc.to_numpy(float), z.iloc[: len(zc)].to_numpy(float), rtol=0, atol=1e-10, equal_nan=True)
                okm = all((mom_states(wc["ca"].to_numpy(), wc["cb"].to_numpy(), m[2], m[3]) == sM[m[0]][: len(wc)]).all() for m in MOM)
                oks = True   # each leg's daily history cut at the decision day; bars usable by the leg's calendar, as in the full run
                for x, sfull in ((a, sa), (b, sb)):
                    bxc = desk_bars(px[x].loc[: dec[T]], closes_only)
                    oks = oks and np.array_equal(leg_state_at(dec[: T + 1], bxc, desk_supertrend(bxc)[0], LEG[x][3]), sfull[: T + 1], equal_nan=True)
                if not (okz and okm and oks):
                    checks["cut_violations"].append(f"{pair} cut {dec[T].date()} z {okz} mom {okm} st {oks}")
            if mk == "india":
                bad = VR.check_against_t03(VR.run_variant(pin, pair, "w52"))
                checks["t03_mismatch"] += bad
            # the targets from the common start
            T = {"R52": [TW[s] for s in sR[K0:]], **{k: [TW[s] for s in v[K0:]] for k, v in sM.items()}}
            Wr = {k: np.array(v, float) for k, v in T.items()}
            for s_ in SCHEMES:
                Wr[s_] = SCHEMES[s_][cmb[K0:]]
            # benchmarks (the engine's comparators' formulas) and their check
            kt = kd[0] + 1; n_ = len(pf)
            bh = np.r_[1.0, 0.5 / (A["fa"][kt] * (1 + C)) * A["ca"][kt:] + 0.5 / (A["fb"][kt] * (1 + C)) * A["cb"][kt:]]
            ha_ = np.r_[1.0, A["ca"][kt:] / (A["fa"][kt] * (1 + C))]; hb_ = np.r_[1.0, A["cb"][kt:] / (A["fb"][kt] * (1 + C))]
            cmp_ = comparators(pf, dec[K0], COST, dec[K0:], PIT())
            for mine, k in ((bh, "5050_buyhold"), (ha_, "hold_a"), (hb_, "hold_b")):
                checks["benchmarks_vs_engine_max_abs_diff"] = max(checks["benchmarks_vs_engine_max_abs_diff"], float(np.nanmax(np.abs(mine - cmp_[k].to_numpy()))))
            st_bh, st_a, st_b = nstats(bh, days), nstats(ha_, days), nstats(hb_, days)
            # the real runs
            real = {}
            for r_ in RULES:
                nav, sw = gbook(A, kd, Wr[r_], rec=True); real[r_] = (nav, sw)
                if r_ in ("R52", "m12_1", "m12", "m6"):
                    tg = [{(1.0, 0.0, 0.0): "A", (0.0, 1.0, 0.0): "B", (0.5, 0.5, 0.0): "N"}[tuple(x)] for x in Wr[r_]]
                    ne, _ = book(pf, dec[K0:], tg, COST, PIT()); checks["engine_book_runs"] += 1
                    checks["engine_book_max_abs_diff"] = max(checks["engine_book_max_abs_diff"], float(np.nanmax(np.abs(ne.to_numpy() - nav))))
            # the nulls through the same book
            NC = {r_: [] for r_ in RULES}; ND_ = {r_: [] for r_ in RULES}
            tR = list(sR[K0:]); baseR = counts(tR)
            for dr in range(ND):
                rng = np.random.default_rng([mid, pid, RULE_ID["R52"], dr]); tg = null_A(tR, rng); cc = counts(tg); checks["null_draws"] += 1; checks["null_count_mismatch"] += (cc != baseR)
                c_, m_ = quick(gbook(A, kd, np.array([TW[s] for s in tg], float))[0], dd_); NC["R52"].append(c_); ND_["R52"].append(m_)
            for m in MOM:
                x = (sM[m[0]][K0:] == "A"); base = spell_counts(x)
                for dr in range(ND):
                    rng = np.random.default_rng([mid, pid, m[4], dr]); y = shuffle_spells(x, rng); checks["null_draws"] += 1; checks["null_count_mismatch"] += (spell_counts(y) != base)
                    c_, m_ = quick(gbook(A, kd, np.array([TW["A" if v else "B"] for v in y], float))[0], dd_); NC[m[0]].append(c_); ND_[m[0]].append(m_)
            xa, xb = ba[K0:], bb[K0:]; base_a, base_b = spell_counts(xa), spell_counts(xb); cmb_real = np.bincount(cmb[K0:], minlength=4)
            for dr in range(ND):
                rng = np.random.default_rng([mid, pid, RULE_ID["ST"], dr]); ya, yb = shuffle_spells(xa, rng), shuffle_spells(xb, rng); checks["null_draws"] += 1
                checks["null_count_mismatch"] += (spell_counts(ya) != base_a) or (spell_counts(yb) != base_b)
                cy = combo_of(ya, yb); checks["st_null_combo_weeks_changed"] += int(not np.array_equal(np.bincount(cy, minlength=4), cmb_real))
                for s_ in SCHEMES:
                    c_, m_ = quick(gbook(A, kd, SCHEMES[s_][cy])[0], dd_); NC[s_].append(c_); ND_[s_].append(m_)
            # results
            yrs = dd_ / 365.25; first, last = days[0].date().isoformat(), days[-1].date().isoformat()
            PAIRS.append({"pair": pair, "market": mk, "label": nice(pair), "a": leg(a), "b": leg(b), "first": first, "last": last, "years": yrs, "pair_first": pf.index[0].date().isoformat()})
            for r_ in RULES:
                nav, sw = real[r_]; st = nstats(nav, days); W = Wr[r_]
                STRAT.append({"pair": pair, "market": mk, "rule": r_, "first": first, "last": last, "cagr": st["cagr"], "cagr_5050": st_bh["cagr"], "cagr_a": st_a["cagr"], "cagr_b": st_b["cagr"],
                              "excess": st["cagr"] - st_bh["cagr"], "ex_a": st["cagr"] - st_a["cagr"], "ex_b": st["cagr"] - st_b["cagr"], "vol": st["vol"], "maxdd": st["maxdd"],
                              "maxdd_5050": st_bh["maxdd"], "maxdd_a": st_a["maxdd"], "maxdd_b": st_b["maxdd"], "vol_5050": st_bh["vol"], "switches": len(sw), "switches_yr": len(sw) / yrs,
                              "w_a": float(W[:, 0].mean()), "w_b": float(W[:, 1].mean()), "w_cash": float(W[:, 2].mean()), "pct": E1.pctile(st["cagr"], NC[r_]), "pct_dd": E1.pctile(st["maxdd"], ND_[r_]),
                              "null_median_cagr": float(np.median(NC[r_])) if ND else np.nan, "null": "N-A" if r_ == "R52" else "N-S"})
                for s in sw:
                    SW.append({"pair": pair, "rule": r_, "decision_date": dec[K0 + s["i"]].date().isoformat(), "trade_date": pf.index[s["k"]].date().isoformat(),
                               "from": "/".join(f"{v:g}" for v in s["from"]), "to": "/".join(f"{v:g}" for v in s["to"]), "fill_a": s["fill_a"], "fill_b": s["fill_b"], "cost_frac": s["cost"]})
            wsel = dec[K0:]; pos_w = pf.index.get_indexer(wsel) - kd[0]
            CURV[pair] = {"t": [d.date().isoformat() for d in wsel], **{r_: [r6(v) for v in real[r_][0][pos_w]] for r_ in RULES},
                          "bh": [r6(v) for v in bh[pos_w]], "a": [r6(v) for v in ha_[pos_w]], "b": [r6(v) for v in hb_[pos_w]]}
            # Rule R's ratio, z and trades (from the pair's first week)
            ent = []; swR = real["R52"][1]
            for j, s in enumerate(swR):
                if s["to"] in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
                    nx = swR[j + 1] if j + 1 < len(swR) else None; held_a = s["to"][0] == 1.0
                    ex = None if nx is None else ((nx["fill_a"] / s["fill_a"] - nx["fill_b"] / s["fill_b"]) if held_a else (nx["fill_b"] / s["fill_b"] - nx["fill_a"] / s["fill_a"]))
                    ent.append([K0 + s["i"], leg(a) if held_a else leg(b), pf.index[s["k"]].date().isoformat(), None if nx is None else pf.index[nx["k"]].date().isoformat(), None if ex is None else round(ex * 10000)])
            RR[pair] = {"dates": [d.date().isoformat() for d in dec], "ratio": [r6(v) for v in wk["ratio"].to_numpy()], "z": [None if v != v else round(float(v), 3) for v in z.to_numpy()], "entries": ent, "k0": K0}
            # supertrend lines, states and next-week returns, from the common start
            STL[pair] = {}   # each leg's own weekly bars (desk), from the common start: last session, close, stop, buy
            for key_, x in (("a", a), ("b", b)):
                bx, sx, stopx, _ = LEG[x]; m_ = (bx["last_session"] >= dec[K0]).to_numpy()
                STL[pair][key_] = {"t": [pd.Timestamp(v).date().isoformat() for v in bx["last_session"][m_]], "c": [r6(v) for v in bx["close"].to_numpy()[m_]],
                                   "stop": [r6(v) for v in stopx[m_]], "buy": [int(v == 1) for v in sx[m_]]}
            ra_n, rb_n = np.diff(np.log(wk["ca"].to_numpy()))[K0:], np.diff(np.log(wk["cb"].to_numpy()))[K0:]   # the week after each decision week
            for s_ in SCHEMES:   # check: the first run's (10, 3) supertrend
                cw = nstats(gbook(A, kd, SCHEMES[s_][cmb_w[K0:]])[0], days)["cagr"]
                LEGACY.append({"pair": pair, "market": mk, "scheme": s_, "first": first, "last": last, "cagr": cw, "excess": cw - st_bh["cagr"]})
            xa1, xb1 = ba[K0:-1], bb[K0:-1]
            fw = lambda r, x, v: float(r[x == v].mean() * 52) if (x == v).any() else np.nan  # noqa: E731
            cm = np.bincount(cmb[K0:], minlength=4) / len(cmb[K0:])
            STS[pair] = {"pair": pair, "market": mk, "first": first, "last": wsel[-1].date().isoformat(), "weeks": int(len(cmb[K0:])), "both_buy": cm[0], "a_only": cm[1], "b_only": cm[2], "both_sell": cm[3],
                         "buy_a": float(ba[K0:].mean()), "buy_b": float(bb[K0:].mean()), "a_buy_next": fw(ra_n, xa1, True), "a_sell_next": fw(ra_n, xa1, False), "b_buy_next": fw(rb_n, xb1, True), "b_sell_next": fw(rb_n, xb1, False),
                         "a_flips": int((ba[K0 + 1:] != ba[K0:-1]).sum()), "b_flips": int((bb[K0 + 1:] != bb[K0:-1]).sum()),
                         **({"closes_agree_a": agree["a"], "closes_agree_b": agree["b"], "S1_cagr_closes": agree["S1_cagr_closes"]} if agree else {})}
            tq = touches(pf); TOUCH.append({"pair": pair, "market": mk, **tq})
            checks["band_vs_z_mismatch"] += tq["band_vs_z_mismatch"]; checks["engine_vs_z_state_mismatch"] += tq["engine_vs_z_state_mismatch"]
            print(f"  {nice(pair):16s} {first} → {last}  [{time.time() - t0:4.0f}s]", flush=True)
    # §1 similarity: Indian equity ÷ gold against SPY ÷ GLD
    SIM = {}
    for x in ("N100", "MC100", "SC100"):
        SIM[x] = {"india": f"{x} ÷ gold", "us": "SPY ÷ GLD", **similarity(PF[f"{x}|{GOLD}"], PF["GLD|SPY"], series=(x == "SC100"))}
    SIM["SC100_IJR"] = {"india": "SC100 ÷ gold", "us": "IJR ÷ GLD", **similarity(PF[f"SC100|{GOLD}"], PF["IJR|GLD"], series=True, us_eq="ca")}   # added 2026-11-23
    s1_, s2_ = SIM["SC100"]["series"], SIM["SC100_IJR"]["series"]
    if s1_["t"] != s2_["t"] or not np.allclose(s1_["eq_in"], s2_["eq_in"]) or not np.allclose(s1_["gold_us"], s2_["gold_us"]):
        sys.exit("similarity: the SPY and IJR comparisons are not on the same weeks")
    SER = {**s1_, "eq_us2": s2_["eq_us"], "r_us2": s2_["r_us"], "z_us2": s2_["z_us"], "rc2": s2_["rc"]}
    if (checks["weekly_bar_index_mismatch"] or checks["t03_mismatch"] or checks["cut_violations"] or checks["null_count_mismatch"] or checks["close_max_abs_diff"] > 0
            or not checks["engine_book_max_abs_diff"] < 1e-9 or not checks["benchmarks_vs_engine_max_abs_diff"] < 1e-12 or checks["band_vs_z_mismatch"] or checks["engine_vs_z_state_mismatch"]
            or checks["st_missing_after_start"]):
        sys.exit(f"checks FAILED: {checks}")
    SR = pd.DataFrame(STRAT); SR.to_csv(os.path.join(OUT, "strategies.csv"), index=False)
    pd.DataFrame(TOUCH).to_csv(os.path.join(OUT, "touches.csv"), index=False); pd.DataFrame(list(STS.values())).to_csv(os.path.join(OUT, "supertrend_states.csv"), index=False)
    pd.DataFrame(SW).to_csv(os.path.join(OUT, "switches.csv"), index=False); LG = pd.DataFrame(LEGACY); LG.to_csv(os.path.join(OUT, "supertrend_check_10_3.csv"), index=False)
    pd.DataFrame([{k: v for k, v in SIM[x].items() if k != "series"} for x in SIM]).to_csv(os.path.join(OUT, "similarity.csv"), index=False)
    summ = {}
    for mk, _, pairs in MARKETS:
        for r_ in RULES:
            g = SR[(SR["market"] == mk) & (SR["rule"] == r_)]
            summ[f"{mk}|{r_}"] = {"n": int(len(g)), "ahead_5050": int((g["excess"] > 0).sum()), "ahead_both": int(((g["ex_a"] > 0) & (g["ex_b"] > 0)).sum()), "median_excess": float(g["excess"].median()),
                                  "shallower_dd_5050": int((g["maxdd"] > g["maxdd_5050"]).sum()), "pct_high": int((g["pct"] >= HI).sum()), "pct_low": int((g["pct"] <= LO).sum()),
                                  "dd_pct_high": int((g["pct_dd"] >= HI).sum()), "dd_pct_low": int((g["pct_dd"] <= LO).sum())}
    TQ = pd.DataFrame(TOUCH)
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "draws_per_null": ND, "seed": "default_rng([market_id (0 US, 1 India), pair_id, rule_id, draw]); rule_id 0 Rule R (N-A), 1-3 momentum 12-1/12/6, 4 supertrend (N-S)",
             "cost_bps_round_trip": COST, "common_start_decision_week": K0 + 1, "checks": checks, "summary": summ,
             "supertrend": {"rule": "the desk's Weekly LW: W-FRI bars on each leg's own sessions, true range, ATR = linear-weighted mean of the last 5 (1/15..5/15), bands (H+L)/2 ± 2 ATR ratcheted, flips on a close through the band; "
                                    "a bar is usable once its week has ended on the leg's calendar; US legs closes only", "atr": DESK_N, "mult": DESK_M, "check": f"the first run's Wilder ({ST_N}, {ST_M:g}) on the pair's common-calendar bars"},
             "touch_seen_daily_median": float(TQ["seen_daily_same_week"].median()), "touch_seen_earlier_median": float(TQ["seen_earlier"].median()), "touch_closed_beyond_median": float(TQ["touch_closed_beyond"].median()),
             "touch_w52_d250_matched_median": float(TQ["w52_matched_d250"].median()), "touch_same_state_median": float(TQ["same_state_w52_d250"].median()), "seconds": round(time.time() - t0)}
    json.dump(jclean(audit), open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)
    write_md(audit, SR, TQ, STS, SIM, PAIRS, LG)
    write_page(audit, SR, TQ, STS, SIM, PAIRS, CURV, RR, STL, SER)
    print(f"done in {time.time() - t0:.0f}s; checks {checks}")


# ------------------------------------------------------------------------------------------------------- results (markdown)
def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
def f2(v, d=2): return "—" if v is None or v != v else f"{v:+.{d}f}"


def write_md(audit, SR, TQ, STS, SIM, PAIRS, LG):
    ck = audit["checks"]; S = audit["summary"]; lab = {p["pair"]: p["label"] for p in PAIRS}
    md = ["# E10 results — small caps against gold beside SPY and IJR against gold; band touches; the desk's weekly supertrend and momentum rotation (EXPLORATORY)\n",
          f"_Generated by `explore/E10_trend_rotation/rotation.py` on {audit['run_date']} from the approved data (NSE indices and MCX gold to 2026-08-28; US ETFs, GLD and IJR to 2026-11-18). Weekly decisions, next-session fills, 10.4 bps per side; "
          f"every rule on a pair starts on the same day (the pair's 53rd decision week). {audit['draws_per_null']} draws per null. Description, not a finding; percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Checks:** pair closes equal the loaders' (max difference {ck['close_max_abs_diff']:.1e}); the book equals the engine's book on {ck['engine_book_runs']} Rule R and momentum runs (max NAV difference {ck['engine_book_max_abs_diff']:.1e}); "
          f"benchmarks equal the engine's comparators (max {ck['benchmarks_vs_engine_max_abs_diff']:.1e}); Rule R (52 weeks) equals T03 on the Indian pairs; z, momentum and supertrend pass the cut-history check ({ck['cut_checks']} cuts, 0 violations); "
          f"{ck['null_draws']:,} null draws, every one kept its counts (supertrend: each leg's; the combination's weeks changed in {ck['st_null_combo_weeks_changed']:,} draws, as expected).\n",
          "## 1. Indian equity ÷ gold against US equity ÷ gold (SPY ÷ GLD, IJR ÷ GLD), on the same weeks\n",
          "| India against US | weeks (India dates) | weekly change correlation | 4-week change correlation | z correlation | z same side | weeks either beyond ±1 / both beyond the same way | Rule R holds the same kind | legs: equity / gold correlation | CAGR: India equity / MCX gold / US equity / GLD | ratio CAGR: India / US |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x, s in SIM.items():
        md.append(f"| {s['india']} vs {s['us']} | {s['first']} → {s['last']} ({s['weeks']}) | {s['corr_weekly']:+.2f} | {s['corr_4week']:+.2f} | {s['corr_z']:+.2f} | {pc(s['z_same_side'])} | {s['weeks_either_beyond']} / {s['both_beyond_same_way']} | "
                  f"{pc(s['rule_same_kind'])} | {s['corr_equity_legs']:+.2f} / {s['corr_gold_legs']:+.2f} | {fp(s['cagr_eq_in'])} / {fp(s['cagr_gold_in'])} / {fp(s['cagr_eq_us'])} / {fp(s['cagr_gold_us'])} | {fp(s['cagr_ratio_in'])} / {fp(s['cagr_ratio_us'])} |")
    md += ["\n## 2–5. Every pair and rule: excess CAGR over 50/50 bought once (percentile against the null: N-A for Rule R, N-S for the others)\n",
           "| pair | dates | 50/50 CAGR | " + " | ".join(RSHORT[r] for r in RULES) + " |", "|---|---|---|" + "---|" * len(RULES)]
    for p in PAIRS:
        g = SR[SR["pair"] == p["pair"]].set_index("rule")
        md.append(f"| {p['label']} ({MKT_LAB[p['market']]}) | {p['first']} → {p['last']} | {fp(g.loc['R52', 'cagr_5050'])} | " + " | ".join(f"{pp(g.loc[r, 'excess'])} ({pq(g.loc[r, 'pct'])})" for r in RULES) + " |")
    md += ["\n### Summary by market and rule\n", "| market | rule | pairs | ahead of 50/50 | ahead of both legs | median excess | max drawdown shallower than 50/50 | CAGR percentile ≥ 95 / ≤ 5 | drawdown percentile ≥ 95 / ≤ 5 |", "|---|---|---|---|---|---|---|---|---|"]
    for mk, _, _ in MARKETS:
        for r in RULES:
            s = S[f"{mk}|{r}"]; md.append(f"| {MKT_LAB[mk]} | {RLAB[r]} | {s['n']} | {s['ahead_5050']} | {s['ahead_both']} | {pp(s['median_excess'])} | {s['shallower_dd_5050']} | {s['pct_high']} / {s['pct_low']} | {s['dd_pct_high']} / {s['dd_pct_low']} |")
    md += ["\n### Every pair and rule in full\n", "| pair | rule | CAGR | excess over 50/50 / first leg / second leg | volatility | max drawdown (50/50) | switches a year | average weight first / second / cash | CAGR pct | drawdown pct |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in SR.iterrows():
        md.append(f"| {lab[r['pair']]} | {RLAB[r['rule']]} | {fp(r['cagr'])} | {pp(r['excess'])} / {pp(r['ex_a'])} / {pp(r['ex_b'])} | {pc(r['vol'])} | {fp(r['maxdd'], 0)} ({fp(r['maxdd_5050'], 0)}) | {r['switches_yr']:.1f} | "
                  f"{pc(r['w_a'])} / {pc(r['w_b'])} / {pc(r['w_cash'])} | {pq(r['pct'])} | {pq(r['pct_dd'])} |")
    md += ["\n## 3. Band touches against z signals (52-week window)\n",
           f"On the same weekly closes the band test and the z test disagree in {audit['checks']['band_vs_z_mismatch']} weeks, and the T02/T03 engine's states and the z-based states in {audit['checks']['engine_vs_z_state_mismatch']} weeks, over every pair.\n",
           "| pair | banded weeks | weekly signals | touched by a daily close that week | touched on an earlier day (median days early) | daily first touches (a year) | of which the week closed beyond the band / became a weekly signal | Rule R entries 52 wk / 250 d | 52-wk entries with a 250-d entry within 2 weeks | weeks in the same state |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, t in TQ.iterrows():
        mde = "—" if t["median_days_early"] != t["median_days_early"] else f"{t['median_days_early']:.0f}"
        md.append(f"| {lab[t['pair']]} | {t['weeks']} ({t['first']} → {t['last']}) | {t['weekly_signals']} | {pc(t['seen_daily_same_week'])} | {pc(t['seen_earlier'])} ({mde}) | {t['daily_first_touches']} ({t['daily_touches_a_year']:.1f}) | "
                  f"{pc(t['touch_closed_beyond'])} / {pc(t['touch_became_signal'])} | {t['entries_w52']} / {t['entries_d250']} | {pc(t['w52_matched_d250'])} | {pc(t['same_state_w52_d250'])} |")
    md += ["\n## 4. The desk's Weekly LW supertrend (5-bar linear-weighted ATR, × 2) on each leg: the four combinations and each leg's next week\n",
           "| pair | weeks | both buy / first only / second only / both sell | first leg: buy weeks, next-week return a year in buy / in sell | second leg: the same | flips a year first / second | Indian legs: closes-only bars agree first / second, S1 CAGR with them |", "|---|---|---|---|---|---|---|"]
    for p in PAIRS:
        s = STS[p["pair"]]; ex = f"{pc(s['closes_agree_a'])} / {pc(s['closes_agree_b'])}, {fp(s['S1_cagr_closes'])}" if "closes_agree_a" in s else "—"
        md.append(f"| {p['label']} | {s['weeks']} ({s['first']} → {s['last']}) | {pc(s['both_buy'])} / {pc(s['a_only'])} / {pc(s['b_only'])} / {pc(s['both_sell'])} | {p['a']}: {pc(s['buy_a'])}, {fp(s['a_buy_next'])} / {fp(s['a_sell_next'])} | "
                  f"{p['b']}: {pc(s['buy_b'])}, {fp(s['b_buy_next'])} / {fp(s['b_sell_next'])} | {s['a_flips'] / p['years']:.1f} / {s['b_flips'] / p['years']:.1f} | {ex} |")
    md += ["\n### Check: S1–S4 with the first run's (10, 3) Wilder supertrend (excess over 50/50; the desk's supertrend in brackets)\n",
           "| pair | dates | S1 | S2 | S3 | S4 |", "|---|---|---|---|---|---|"]
    for p in PAIRS:
        q = LG[LG["pair"] == p["pair"]].set_index("scheme"); g = SR[SR["pair"] == p["pair"]].set_index("rule")
        md.append(f"| {p['label']} | {p['first']} → {p['last']} | " + " | ".join(f"{pp(q.loc[s_, 'excess'])} ({pp(g.loc[s_, 'excess'])})" for s_ in SCHEMES) + " |")
    open(os.path.join(OUT, "E10_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


# ------------------------------------------------------------------------------------------------------- the page
def write_page(audit, SR, TQ, STS, SIM, PAIRS, CURV, RR, STL, SER):
    S = audit["summary"]; lab = {p["pair"]: p["label"] for p in PAIRS}
    STJ = {}
    for _, r in SR.iterrows():
        STJ.setdefault(r["pair"], {})[r["rule"]] = [r["cagr"], r["excess"], r["ex_a"], r["ex_b"], r["vol"], r["maxdd"], r["switches_yr"], r["pct"], r["pct_dd"], r["w_a"], r["w_b"], r["w_cash"], r["cagr_5050"], r["cagr_a"], r["cagr_b"], r["maxdd_5050"], r["maxdd_a"], r["maxdd_b"]]
    sim_tab = [{"x": x, **{k: v for k, v in s.items() if k != "series"}} for x, s in SIM.items()]
    DATA = {"pairs": PAIRS, "rules": RULES, "rlab": RLAB, "rshort": RSHORT, "schemes": {k: v.tolist() for k, v in SCHEMES.items()}, "schlab": SCH_LAB, "lo": LO, "hi": HI, "draws": audit["draws_per_null"],
            "strat": STJ, "curves": CURV, "rr": RR, "stl": STL, "sts": STS, "touch": TQ.to_dict("records"), "sim": SER, "simtab": sim_tab, "summary": S,
            "defaults": {"pair": DEFAULT_PAIR, "mom": "m12_1", "sch": "S1"}}
    blob = json.dumps(jclean(DATA), separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")

    def sel(i, lab_, groups, default):
        h = f"<label>{lab_} <select id='{i}'>"
        for gname, opts in groups:
            h += (f"<optgroup label='{html.escape(gname, quote=True)}'>" if gname else "") + "".join(f"<option value='{html.escape(str(v), quote=True)}'{' selected' if str(v) == str(default) else ''}>{html.escape(t)}</option>" for v, t in opts) + ("</optgroup>" if gname else "")
        return h + "</select></label>"
    controls = (sel("pair", "Pair", [(MKT_LAB[mk], [(p["pair"], p["label"]) for p in PAIRS if p["market"] == mk]) for mk, _, _ in MARKETS], DEFAULT_PAIR)
                + sel("mom", "Momentum", [("", [(m[0], m[1].replace("momentum ", "")) for m in MOM])], "m12_1") + sel("sch", "Supertrend weights", [("", [(k, v) for k, v in SCH_LAB.items()])], "S1"))
    s1 = SIM["SC100"]; s2 = SIM["SC100_IJR"]; g = SR[SR["pair"] == DEFAULT_PAIR].set_index("rule"); ps = [p for p in PAIRS if p["pair"] == DEFAULT_PAIR][0]

    def line(mk, r):
        s = S[f"{mk}|{r}"]; return f"{RSHORT[r]}: ahead in {s['ahead_5050']} of {s['n']} (median {pp(s['median_excess'])}; notable against the null {s['pct_high']} high, {s['pct_low']} low)"
    key = (f"<p style='margin:4px 0'><b>1 · SC100 ÷ gold beside SPY ÷ GLD</b> ({s1['first']} → {s1['last']}, {s1['weeks']} common weeks): weekly changes correlate {s1['corr_weekly']:+.2f}, 4-week changes {s1['corr_4week']:+.2f}, their 52-week z {s1['corr_z']:+.2f}; "
           f"z on the same side of 0 in {pc(s1['z_same_side'])} of weeks; Rule R holds the same kind of asset (equity, gold or 50/50) in both in {pc(s1['rule_same_kind'])} of weeks. "
           f"Over the period SC100 made {fp(s1['cagr_eq_in'])} a year, MCX gold {fp(s1['cagr_gold_in'])}, SPY {fp(s1['cagr_eq_us'])}, GLD {fp(s1['cagr_gold_us'])}. "
           f"<b>Against IJR ÷ GLD</b> (US small caps against gold, the same weeks): weekly changes {s2['corr_weekly']:+.2f}, 4-week {s2['corr_4week']:+.2f}, z {s2['corr_z']:+.2f}; Rule R the same kind in {pc(s2['rule_same_kind'])} of weeks; "
           f"SC100 and IJR correlate {s2['corr_equity_legs']:+.2f} week to week; IJR made {fp(s2['cagr_eq_us'])} a year (IJR ÷ GLD {fp(s2['cagr_ratio_us'])}, SC100 ÷ gold {fp(s2['cagr_ratio_in'])}).</p>"
           f"<p style='margin:4px 0'><b>2 · Rule R on SC100 ÷ gold</b> ({ps['first']} → {ps['last']}): {fp(g.loc['R52', 'cagr'])} a year against 50/50 bought once {fp(g.loc['R52', 'cagr_5050'])} (excess {pp(g.loc['R52', 'excess'])}), "
           f"SC100 {fp(g.loc['R52', 'cagr_a'])}, gold {fp(g.loc['R52', 'cagr_b'])}; max drawdown {fp(g.loc['R52', 'maxdd'], 0)} (50/50 {fp(g.loc['R52', 'maxdd_5050'], 0)}).</p>"
           f"<p style='margin:4px 0'><b>3 · Band touches and z signals</b> are the same test on the same closes: {audit['checks']['band_vs_z_mismatch']} weeks differ over every pair. Across pairs, a daily close had touched the band in the week of a weekly signal "
           f"in a median {pc(audit['touch_seen_daily_median'])} of signals (on an earlier day in {pc(audit['touch_seen_earlier_median'])}); a median {pc(audit['touch_closed_beyond_median'])} of daily first touches ended their week beyond the band.</p>"
           f"<p style='margin:4px 0'><b>4 · The desk's weekly supertrend on SC100 and gold</b> ({ps['first']} → {ps['last']}): excess over 50/50 " + ", ".join(f"{k} {pp(g.loc[k, 'excess'])}" for k in SCHEMES) + ".</p>"
           f"<p style='margin:4px 0'><b>5 · US</b> (nine commodity ETFs and GLD against SPY, IJR against SPY and GLD; twelve pairs), ahead of 50/50 bought once: " + "; ".join(line("us", r) for r in RULES) + ".</p>"
           f"<p style='margin:4px 0'><b>India</b> (N100, MC100, SC100 and gold, six pairs): " + "; ".join(line("india", r) for r in RULES) + ".</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Trend and rotation</title>"),
         "<header class='bar'><div class='bar-inner'><b>E10 · trend and rotation</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Small caps against gold beside SPY and IJR against gold; band touches; the desk's weekly supertrend and momentum rotation, US and India</h1>",
         f"<p>The desk's notes of 2026-11-23, point by point. Weekly decisions on the last trading day of each ISO week; trades at the next session's fill (OHLC4 for NSE and MCX, the close for US ETFs), 10.4 bps each way; "
         f"every rule on a pair starts on the same day, the pair's 53rd decision week, at 50/50. US: the nine main commodity ETFs and GLD against SPY, and IJR (US small caps) against SPY and against GLD (total return, to 2026-11-18). India: N100, MC100, SC100 and MCX gold (price indices, to 2026-08-28). "
         f"Generated {audit['run_date']}. <b>Description, not a finding</b>: every pair here has been looked at before, so none of this is an unseen test.</p>",
         "<div class='card'>" + key + "</div>",
         "<h2>1 · SC100 ÷ gold beside SPY ÷ GLD and IJR ÷ GLD, on the same weeks</h2><p class='note'>All written as equity ÷ gold. India in rupees (SC100 price index, MCX gold futures), the US in dollars (SPY and IJR total return, GLD). Weekly closes on the weeks both markets trade; value of 1 from the first common week.</p>",
         "<div class='card'><div id='g_legs' class='plot'></div><div id='g_ratios' class='plot'></div></div><div class='card'><div id='g_simz' class='plot'></div><div id='g_rc' class='plot'></div></div><div class='card'><div id='t_sim'></div></div>",
         "<h2>2 · The ratio switch (Rule R, ±1 sd, 52 weeks) on the chosen pair</h2><p class='note'>T03's rule: from 50/50, z of the ratio first below −1 → hold only the first leg, first above +1 → only the second; back to 50/50 when z returns through 0. The equity curve starts on the common start day of the pair.</p>",
         "<div class='card'><div id='g_rr' class='plot'></div></div><div class='card'><div id='g_req' class='plot'></div><div id='t_req'></div></div>",
         "<h2>3 · Band touches and z-score signals</h2><p class='note'>The band is the trailing 52-week mean ± 1 sd of the weekly ratio; z = (ratio − mean) ÷ sd. On the same closes \"beyond the band\" and \"|z| above 1\" are one test. "
         "They part only when the views differ: a daily close against the band known that day (last week's band), or the daily 250-day z against the weekly 52-week z.</p>",
         "<div class='card'><div id='t_touch'></div></div>",
         "<h2>4 · The desk's weekly supertrend on the chosen pair's two legs</h2><p class='note'>The desk's Weekly LW supertrend (as in its earlier repositories): weekly bars labelled by Friday on each leg's own sessions, a 5-bar linear-weighted ATR (weights 1/15 … 5/15), bands at (H + L) ÷ 2 ± 2 × ATR, "
         "BUY on a close above the upper band, SELL on a close below the lower band; a bar counts once its week has ended. Indian legs use daily highs and lows, US legs their daily closes (the US data has closes only). "
         "The pair is in one of four combinations each week, and each weight scheme maps them to weights:</p>",
         "<div class='card'><div id='t_w'></div></div><div class='card'><div id='g_sta' class='plot'></div><div id='g_stb' class='plot'></div><div id='t_sts'></div></div><div class='card'><div id='g_steq' class='plot'></div><div id='t_steq'></div></div>",
         "<h2>5 · All the rules on every pair: momentum, supertrend and Rule R</h2><p class='note'>Momentum: hold the leg with the higher return over the lookback (12-1 = from 52 to 4 weeks back). Excess = CAGR − 50/50 bought once's CAGR, from the pair's common start. "
         f"Percentiles: the real CAGR among {audit['draws_per_null']} draws of the same spells in a random order (N-S; Rule R: T03's N-A); ≤ 5 or ≥ 95 = notable.</p>",
         "<div class='card'><div id='g_heat' class='plot'></div><div id='t_heat'></div></div><div class='card'><div id='g_cmp' class='plot'></div><div id='g_cdd' class='plot'></div><div id='t_cmp'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Data.</b> analysis/approved_data: load_daily() (NSE price indices, MCX gold futures in rupees), load_us_etf() (total return, launch periods dropped, closes only), load_us_gold() (GLD) and load_us_small() (IJR, approved 2026-11-23: total return with its "
         "pre-2017 distributions put on the split-adjusted price basis, rule H6, F-RA-D14). Each pair on its common calendar (analysis/ratio_engine.pair_frame).</li>"
         "<li><b>Book.</b> One book for every rule: weekly target weights (first leg, second leg, cash), traded at the next session's fills only when the target changes, selling before buying, 10.4 bps per side; cash earns 0 (there is no approved cash series). "
         f"It equals the engine's book on every Rule R and momentum run (max NAV difference {audit['checks']['engine_book_max_abs_diff']:.1e}).</li>"
         "<li><b>Signals.</b> Rule R from explore/E01_ratio_cycles/variants.rule_states (equals T03 on the Indian pairs); momentum on decision-week closes; the desk's Weekly LW supertrend on each leg's own weekly bars (the first run's (10, 3) Wilder version is kept as a check in out/supertrend_check_10_3.csv). All recomputed on histories cut at 8 dates per pair and found equal "
         f"({audit['checks']['cut_checks']} cuts, 0 violations).</li>"
         f"<li><b>Nulls.</b> N-S: each two-state series cut into spells, the spells of each state permuted among that state's positions (switches, spells and weeks kept); supertrend shuffles each leg on its own. Rule R: T03's N-A. {audit['draws_per_null']} draws per pair and rule, "
         "seed default_rng([market_id, pair_id, rule_id, draw]).</li>"
         "<li><b>Limits.</b> NSE series are price indices (dividends not included), US series total return; cash at 0 understates S1 and S4; one supertrend setting, three momentum lookbacks, nothing tuned; curves on this page are at weekly decision closes (the tables' drawdowns are daily). "
         "Design written before the first run: explore/E10_trend_rotation/E10.md.</li></ul></div>",
         "</main>", f"<script id='e10-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E10_rotation.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.nw{white-space:nowrap}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e10-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
const P = {}; D.pairs.forEach(p=>{P[p.pair]=p;});
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
// categorical slots of the reference palette, in their validated order: blue, orange, aqua, yellow, magenta
function COL(){return isDark()?{s1:'#3987e5',s2:'#d95926',s3:'#199e70',s4:'#c98500',s5:'#d55181',bh:'#898781',buy:'#3987e5',sell:'#e66767'}:{s1:'#2a78d6',s2:'#eb6834',s3:'#1baf7a',s4:'#eda100',s5:'#e87ba4',bh:'#898781',buy:'#2a78d6',sell:'#e34948'};}
function v(id){return document.getElementById(id).value;}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function fpp(x){if(x===null||x===undefined)return '—';let s=(x*100).toFixed(2);if(Number(s)===0)s='0.00';return (Number(s)>=0?'+':'')+s+' pp';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function sg(x,d){return x===null||x===undefined?'—':((x>=0?'+':'')+(+x).toFixed(d===undefined?2:d));}
function f1(x,d){return x===null||x===undefined?'—':(+x).toFixed(d===undefined?1:d);}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:100,b:50},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',traceorder:'normal',font:{color:c.ink2,size:11}}},extra||{});}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function heatColors(){return isDark()?[[0,'#86b6ef'],[0.2,'#3987e5'],[0.4,'#34465e'],[0.5,'#383835'],[0.6,'#5e3a3a'],[0.8,'#e66767'],[1,'#f29a9a']]:[[0,'#104281'],[0.2,'#2a78d6'],[0.4,'#cde2fb'],[0.5,'#f0efec'],[0.6,'#f9d3d2'],[0.8,'#e34948'],[1,'#a32b2a']];}
function ddOf(a){let pk=-Infinity;return a.map(x=>{if(x===null)return null;pk=Math.max(pk,x);return (x/pk-1)*100;});}
function S(){return {p:v('pair'), m:v('mom'), s:v('sch')};}
function span(t){return t[0]+' → '+t[t.length-1];}

function drawSim(){
  const c=th(), K=COL(), M=D.sim, t=M.t;
  const L=[['eq_in','SC100 (India)',K.s1,'solid'],['gold_in','MCX gold (India)',K.s1,'dot'],['eq_us','SPY (US)',K.s2,'solid'],['gold_us','GLD (US)',K.s2,'dot'],['eq_us2','IJR (US small caps)',K.s3,'solid']];
  Plotly.react('g_legs',L.map(([k,nm,col,dash])=>({x:t,y:M[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:1.8,dash:dash},hovertemplate:nm+': %{y:.2f}<extra></extra>'})),
    layout(two('The five legs · value of 1 · '+span(t),'blue = India, orange = SPY and GLD, aqua = IJR; solid = equity, dotted = gold; weekly closes; log scale'),420,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  const R3=[['r_in','SC100 ÷ MCX gold',K.s1],['r_us','SPY ÷ GLD',K.s2],['r_us2','IJR ÷ GLD',K.s3]], Z3=[['z_in','SC100 ÷ MCX gold',K.s1],['z_us','SPY ÷ GLD',K.s2],['z_us2','IJR ÷ GLD',K.s3]];
  Plotly.react('g_ratios',R3.map(([k,nm,col])=>({x:t,y:M[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:2},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two('Equity ÷ gold · value of 1 · '+span(t),'up = equity beat gold; weekly closes; log scale'),380,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'ratio, value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_simz',Z3.map(([k,nm,col])=>({x:t,y:M[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:1.5},hovertemplate:nm+': z %{y:+.2f}<extra></extra>'})),
    layout(two('The 52-week z of each ratio · '+span(t),'the band at ±1 (dashed); each z on its own ratio\'s history'),360,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({title:{text:'z',font:{color:c.ink2}}}),
      shapes:[-1,1].map(y=>({type:'line',xref:'paper',x0:0,x1:1,y0:y,y1:y,line:{color:c.axis,width:1,dash:'dash'}})).concat([{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{color:c.axis,width:1}}])}),CFG);
  Plotly.react('g_rc',[['rc','SC100 ÷ gold against SPY ÷ GLD',K.s2],['rc2','SC100 ÷ gold against IJR ÷ GLD',K.s3]].map(([k,nm,col])=>({x:t,y:M[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:1.6},hovertemplate:nm+': %{y:+.2f}<extra></extra>'})),
    layout(two('Rolling 52-week correlation of the ratios\' weekly changes · '+span(t),'SC100 ÷ gold against each US ratio'),320,{hovermode:'x unified',margin:{l:60,r:24,t:90,b:40},xaxis:ax({type:'date'}),yaxis:ax({range:[-1,1],zeroline:true,zerolinecolor:c.axis})}),CFG);
  document.getElementById('t_sim').innerHTML=table(['India ÷ gold against US ÷ gold','weeks','weekly changes corr.','4-week changes corr.','z corr.','z same side','weeks either beyond ±1 / both the same way','Rule R same kind','legs corr.: equity / gold','CAGR: India equity / MCX gold / US equity / GLD','ratio CAGR India / US'],
    D.simtab.map(s=>[(s.x.indexOf('SC100')===0?'<b>':'')+s.india+' against '+s.us+(s.x.indexOf('SC100')===0?'</b>':''),'<span class="nw">'+s.first+' → '+s.last+'</span> ('+s.weeks+')',sg(s.corr_weekly),sg(s.corr_4week),sg(s.corr_z),pc(s.z_same_side),s.weeks_either_beyond+' / '+s.both_beyond_same_way,pc(s.rule_same_kind),
      sg(s.corr_equity_legs)+' / '+sg(s.corr_gold_legs),fp(s.cagr_eq_in)+' / '+fp(s.cagr_gold_in)+' / '+fp(s.cagr_eq_us)+' / '+fp(s.cagr_gold_us),fp(s.cagr_ratio_in)+' / '+fp(s.cagr_ratio_us)]))+
    "<p class='note'>India's dates; the US week is the same ISO week. Rule R same kind: both hold equity, both gold, or both 50/50, over the weeks both have a band.</p>";
}
function drawRR(s){
  const c=th(), K=COL(), p=P[s.p], R=D.rr[s.p], tr=[];
  tr.push({x:R.dates,y:R.ratio,type:'scatter',mode:'lines',line:{color:c.ink2,width:1.3},name:p.label,xaxis:'x',yaxis:'y',hovertemplate:'%{x}: %{y:.4f}<extra></extra>'});
  tr.push({x:R.dates,y:R.z,type:'scatter',mode:'lines',line:{color:K.s1,width:1.2},name:'z (52 weeks)',xaxis:'x',yaxis:'y2',hovertemplate:'%{x}: z %{y:+.2f}<extra></extra>'});
  [[p.a,K.s4],[p.b,K.s5]].forEach(([h,col])=>{const sub=R.entries.filter(e=>e[1]===h);
    tr.push({x:sub.map(e=>R.dates[e[0]]),y:sub.map(e=>R.z[e[0]]),type:'scatter',mode:'markers',marker:{color:col,size:9,line:{width:1,color:c.ink}},name:'entry: hold '+h,xaxis:'x',yaxis:'y2',
      customdata:sub.map(e=>e[2]+' → '+(e[3]||'open')+(e[4]===null?'':': held minus other '+fp(e[4]/10000))),hovertemplate:'%{customdata}<extra></extra>'});});
  Plotly.react('g_rr',tr,layout(two(p.label+' · the ratio and its 52-week z · '+span(R.dates)+' · '+R.entries.length+' entries from the common start '+p.first,'top: the ratio (log scale); bottom: z with the band at ±1 (dashed); dots = entries'),600,
    {xaxis:ax({anchor:'y2',type:'date'}),yaxis:ax({domain:[0.56,1],type:'log',title:{text:'ratio',font:{color:c.ink2}}}),yaxis2:ax({domain:[0,0.46],title:{text:'z',font:{color:c.ink2}}}),
     shapes:[-1,1].map(y=>({type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:y,y1:y,line:{color:c.axis,width:1,dash:'dash'}})).concat([{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:0,y1:0,line:{color:c.axis,width:1}}])}),CFG);
  const C=D.curves[s.p], q=D.strat[s.p].R52;
  const lines=[['R52','Rule R',K.s1,2.4,'solid'],['bh','50/50 bought once',K.bh,1.6,'dash'],['a',p.a+' bought once',K.s4,1.4,'solid'],['b',p.b+' bought once',K.s5,1.4,'solid']];
  Plotly.react('g_req',lines.map(([k,nm,col,wd,dash])=>({x:C.t,y:C[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(p.label+' · Rule R · value of 1 · '+span(C.t),'after costs; weekly decision closes; log scale · Rule R '+fp(q[0])+' a year, 50/50 '+fp(q[12])+', '+p.a+' '+fp(q[13])+', '+p.b+' '+fp(q[14])),440,
    {hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  document.getElementById('t_req').innerHTML=table(['line','dates','CAGR','max drawdown (daily)'],[['Rule R (±1 sd, 52 weeks)',p.first+' → '+p.last,fp(q[0]),fp(q[5],0)],['50/50 bought once',p.first+' → '+p.last,fp(q[12]),fp(q[15],0)],[p.a+' bought once',p.first+' → '+p.last,fp(q[13]),fp(q[16],0)],[p.b+' bought once',p.first+' → '+p.last,fp(q[14]),fp(q[17],0)]])+
    "<p class='note'>Rule R: excess over 50/50 "+fpp(q[1])+'; '+f1(q[6])+' switches a year; average weight '+pc(q[9])+' '+p.a+', '+pc(q[10])+' '+p.b+'; CAGR percentile against N-A '+pq(q[7])+'.</p>';
}
function drawTouch(s){
  document.getElementById('t_touch').innerHTML=table(['pair','banded weeks','band vs z: weeks differing','weekly signals','touched by a daily close that week','touched on an earlier day (median days early)','daily first touches (a year)','… week closed beyond the band','… became a weekly signal','Rule R entries 52 wk / 250 d','52-wk entries matched by 250-d within 2 wk','weeks in the same state'],
    D.touch.map(t=>['<span class="nw">'+(t.pair===s.p?'<b>'+P[t.pair].label+'</b>':P[t.pair].label)+'</span>',t.weeks+'<br><span class="nw">'+t.first+' → '+t.last+'</span>',t.band_vs_z_mismatch+' (engine '+t.engine_vs_z_state_mismatch+')',t.weekly_signals,pc(t.seen_daily_same_week),pc(t.seen_earlier)+' ('+f1(t.median_days_early,0)+')',t.daily_first_touches+' ('+f1(t.daily_touches_a_year)+')',
      pc(t.touch_closed_beyond),pc(t.touch_became_signal),t.entries_w52+' / '+t.entries_d250,pc(t.w52_matched_d250),pc(t.same_state_w52_d250)]))+
    "<p class='note'>Weekly signal = the first week the 52-week z is beyond ±1 (the band's first touch at the week's close). Daily touch = a daily close beyond the band known that day (the previous week's mean ± sd); a first touch is a day beyond it after a day that was not.</p>";
}
function drawST(s){
  const c=th(), K=COL(), p=P[s.p], L=D.stl[s.p], Z=D.sts[s.p];
  const combos=[p.a+' buy, '+p.b+' buy',p.a+' buy, '+p.b+' sell',p.a+' sell, '+p.b+' buy',p.a+' sell, '+p.b+' sell'], share=[Z.both_buy,Z.a_only,Z.b_only,Z.both_sell];
  const w=x=>Math.round(x[0]*100)+' / '+Math.round(x[1]*100)+' / '+Math.round(x[2]*100);
  document.getElementById('t_w').innerHTML=table(['combination','weeks ('+Z.first+' → '+Z.last+')'].concat(Object.keys(D.schemes).map(k=>(k===s.s?'<b>'+D.schlab[k]+'</b>':D.schlab[k]))),
    combos.map((cb,i)=>[cb,pc(share[i])].concat(Object.keys(D.schemes).map(k=>w(D.schemes[k][i])))))+"<p class='note'>Weights: "+p.a+' / '+p.b+' / cash, in %. Cash earns 0.</p>';
  [['g_sta','a',p.a],['g_stb','b',p.b]].forEach(([id,k,nm])=>{const Q=L[k];
    const buyL=Q.stop.map((x,i)=>Q.buy[i]===1?x:null), sellL=Q.stop.map((x,i)=>Q.buy[i]===0?x:null);
    Plotly.react(id,[{x:Q.t,y:Q.c,type:'scatter',mode:'lines',name:nm+' weekly close',line:{color:c.ink2,width:1.3},hovertemplate:'%{x}: %{y:.2f}<extra></extra>'},
      {x:Q.t,y:buyL,type:'scatter',mode:'lines',name:'BUY: the stop (lower band)',line:{color:K.buy,width:1.8},connectgaps:false,hovertemplate:'stop %{y:.2f}<extra></extra>'},
      {x:Q.t,y:sellL,type:'scatter',mode:'lines',name:'SELL: the re-entry level (upper band)',line:{color:K.sell,width:1.8},connectgaps:false,hovertemplate:'re-entry %{y:.2f}<extra></extra>'}],
      layout(two(nm+' · the desk\'s weekly supertrend (5-bar LW ATR × 2) · '+span(Q.t),'the leg\'s own weekly bars, dated by their last session; blue = BUY, red = SELL; log scale'),340,{hovermode:'x unified',margin:{l:60,r:24,t:90,b:40},xaxis:ax({type:'date'}),yaxis:ax({type:'log'})}),CFG);});
  document.getElementById('t_sts').innerHTML=table(['leg','buy weeks','next-week return in buy weeks (a year)','in sell weeks (a year)','flips a year'].concat(Z.closes_agree_a!==undefined?['states with closes-only bars agree']:[]),
    [[p.a,pc(Z.buy_a),fp(Z.a_buy_next),fp(Z.a_sell_next),f1(Z.a_flips/p.years)].concat(Z.closes_agree_a!==undefined?[pc(Z.closes_agree_a)]:[]),[p.b,pc(Z.buy_b),fp(Z.b_buy_next),fp(Z.b_sell_next),f1(Z.b_flips/p.years)].concat(Z.closes_agree_a!==undefined?[pc(Z.closes_agree_b)]:[])])+
    "<p class='note'>Next-week return: the weekly close-to-close log return of the week after each decision week, averaged by the leg's supertrend state, times 52.</p>";
  const C=D.curves[s.p], cols={S1:K.s1,S2:K.s2,S3:K.s3,S4:K.s4};
  const lines=Object.keys(D.schemes).map(k=>[k,D.schlab[k],cols[k],k===s.s?2.6:1.5,'solid']).concat([['bh','50/50 bought once',K.bh,1.6,'dash']]);
  Plotly.react('g_steq',lines.map(([k,nm,col,wd,dash])=>({x:C.t,y:C[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(p.label+' · weekly supertrend, the four weight schemes · value of 1 · '+span(C.t),'after costs; cash at 0; weekly decision closes; log scale; the chosen scheme thicker'),460,{hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  const R=D.strat[s.p];
  document.getElementById('t_steq').innerHTML=table(['scheme','dates','CAGR','excess over 50/50','over '+p.a,'over '+p.b,'volatility','max drawdown','switches a year','average weight '+p.a+' / '+p.b+' / cash','CAGR pct (N-S)','drawdown pct'],
    Object.keys(D.schemes).map(k=>{const q=R[k];return [(k===s.s?'<b>'+D.schlab[k]+'</b>':D.schlab[k]),p.first+' → '+p.last,fp(q[0]),fpp(q[1]),fpp(q[2]),fpp(q[3]),pc(q[4]),fp(q[5],0),f1(q[6]),pc(q[9])+' / '+pc(q[10])+' / '+pc(q[11]),pq(q[7]),pq(q[8])];})
    .concat([['50/50 bought once',p.first+' → '+p.last,fp(R.R52[12]),'','','','',fp(R.R52[15],0),'','50 / 50 / 0 at the start','','']]));
}
function drawHeat(s){
  const c=th(), ps=D.pairs, rs=D.rules;
  const z=ps.map(p=>rs.map(r=>D.strat[p.pair][r][1])), txt=ps.map(p=>rs.map(r=>{const q=D.strat[p.pair][r];return (q[1]*100).toFixed(1)+((q[7]<=D.lo||q[7]>=D.hi)?'*':'');}));
  const hv=ps.map(p=>rs.map(r=>{const q=D.strat[p.pair][r];return p.label+' · '+D.rlab[r]+' ('+p.first+' → '+p.last+')<br>CAGR '+fp(q[0])+', excess over 50/50 '+fpp(q[1])+'<br>max drawdown '+fp(q[5],0)+' (50/50 '+fp(q[15],0)+'); '+f1(q[6])+' switches a year<br>percentile against the null: CAGR '+q[7].toFixed(1)+', drawdown '+q[8].toFixed(1);}));
  const zr=Math.max(0.01,...z.flat().map(Math.abs)), A=D.summary;
  Plotly.react('g_heat',[{type:'heatmap',z:z,x:rs.map(r=>D.rshort[r]),y:ps.map(p=>p.label+(p.market==='us'?' · US':' · India')),text:txt,texttemplate:'%{text}',customdata:hv,hovertemplate:'%{customdata}<extra></extra>',zmin:-zr,zmax:zr,colorscale:heatColors(),xgap:2,ygap:2,
    colorbar:{thickness:12,tickformat:'.1%',tickfont:{color:c.muted},title:{text:'excess',side:'right',font:{color:c.ink2}}}}],
    layout(two('Every pair and rule · excess CAGR over 50/50 bought once (pp a year), from each pair\'s common start','red = ahead of 50/50; * = CAGR notable against the null (either way); hover for dates, drawdowns and percentiles'),660,
    {margin:{l:150,r:24,t:110,b:40},xaxis:ax({side:'top',showgrid:false,showline:false}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
  const cell=(m,r)=>{const q=A[m+'|'+r];return [q.ahead_5050+' of '+q.n,q.ahead_both+' of '+q.n,fpp(q.median_excess),q.shallower_dd_5050+' of '+q.n,q.pct_high+' / '+q.pct_low,q.dd_pct_high+' / '+q.dd_pct_low];};
  document.getElementById('t_heat').innerHTML=table(['rule','US: ahead of 50/50','ahead of both legs','median excess','max drawdown shallower than 50/50','CAGR pct ≥ 95 / ≤ 5','drawdown pct ≥ 95 / ≤ 5','India: ahead of 50/50','ahead of both legs','median excess','max drawdown shallower','CAGR pct ≥ 95 / ≤ 5','drawdown pct ≥ 95 / ≤ 5'],
    rs.map(r=>[D.rlab[r]].concat(cell('us',r)).concat(cell('india',r))))+"<p class='note'>US: ten pairs, India: six; each from its own common start to its last day (dates in the hover and the tables).</p>";
}
function drawCmp(s){
  const c=th(), K=COL(), p=P[s.p], C=D.curves[s.p], R=D.strat[s.p];
  const lines=[['R52','Rule R (±1 sd, 52 weeks)',K.s1,2.2,'solid'],[s.m,D.rlab[s.m],K.s2,2.2,'solid'],[s.s,D.rlab[s.s],K.s3,2.2,'solid'],['bh','50/50 bought once',K.bh,1.6,'dash'],['a',p.a+' bought once',K.s4,1.3,'solid'],['b',p.b+' bought once',K.s5,1.3,'solid']];
  Plotly.react('g_cmp',lines.map(([k,nm,col,wd,dash])=>({x:C.t,y:C[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(p.label+' · Rule R, '+D.rlab[s.m]+' and '+D.rlab[s.s]+' · value of 1 · '+span(C.t),'after costs; weekly decision closes; log scale · pick the momentum lookback and the supertrend weights at the top'),480,
    {hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_cdd',lines.map(([k,nm,col,wd,dash])=>({x:C.t,y:ddOf(C[k]),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:dash},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · weekly closes · '+span(C.t),'the same lines'),300,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
  document.getElementById('t_cmp').innerHTML="<p class='note' style='margin:4px 0'><b>"+p.label+'</b>, every rule, '+p.first+' → '+p.last+' (50/50 bought once '+fp(R.R52[12])+' a year, '+p.a+' '+fp(R.R52[13])+', '+p.b+' '+fp(R.R52[14])+'):</p>'+
    table(['rule','CAGR','excess over 50/50','over '+p.a,'over '+p.b,'volatility','max drawdown','switches a year','average weight '+p.a+' / '+p.b+' / cash','CAGR pct','drawdown pct'],
    D.rules.map(r=>{const q=R[r],b=(r==='R52'||r===s.m||r===s.s);return [(b?'<b>':'')+D.rlab[r]+(b?'</b>':''),fp(q[0]),fpp(q[1]),fpp(q[2]),fpp(q[3]),pc(q[4]),fp(q[5],0),f1(q[6]),pc(q[9])+' / '+pc(q[10])+' / '+pc(q[11]),pq(q[7]),pq(q[8])];}));
}
function render(){
  const s=S(); try{localStorage.setItem('e10-state',JSON.stringify(s));}catch(e){}
  drawSim(); drawRR(s); drawTouch(s); drawST(s); drawHeat(s); drawCmp(s);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  try{const st=JSON.parse(localStorage.getItem('e10-state')||'null');if(st){[['pair','p'],['mom','m'],['sch','s']].forEach(([id,k])=>{const el=document.getElementById(id);if(st[k]!==undefined&&Array.from(el.options).some(o=>o.value===String(st[k])))el.value=String(st[k]);});}}catch(e){}
  ['pair','mom','sch'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
