"""
The T03 rule run with each z window's band, for E03 and E04 (EXPLORATORY variants; T03 itself is the 52-week window).
The rule: from neutral, enter on the first week z < -1 (hold a) or z > +1 (hold b); exit when z is back through 0 (the
ratio back at its trailing mean); no maximum hold; touches inside a trade are ignored. Decisions stay weekly (the last
trading day of each ISO week): weekly windows use z on decision-week closes, daily windows read the daily z (computed on
every trading day) at each decision day. Trades, costs, fills and daily valuation go through the engine's own book
(analysis.ratio_engine.book / comparators / trade_extras), 20.8 bps round trip, exactly as T03.

With the 52-week window the decisions, trades and NAV must equal T03's evidence; check_against_t03() asserts it.

On the latest data (latest.activate(), the desk's request of 2026-11-26) two things change:
- the switch: on a gold pair (gold is the second leg), whenever the rule would hold gold while gold's weekly supertrend
  is SELL, book3() holds the liquid ETF instead (the engine's book with a third holding; the switch off, it equals the
  engine's book, which run_states measures on every run); the trade diagnostics (extras) give a gold trade's held return
  over gold and the liquid ETF as held;
- T03's evidence is on the earlier data, so check_against_t03() checks the 52-week run against the engine's decide() and
  book on the same data instead.
"""
import os

import numpy as np
import pandas as pd

import zwin as ZW
from analysis.ratio_engine import PIT, book, comparators, decide, pair_frame, trade_extras

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST = 20.8
SWITCH = None    # latest.activate(): gold on SELL → the liquid ETF, on the gold pairs
LATEST = False   # latest.activate(): the studies run on the latest data, so T03's evidence does not apply
TW3 = {"A": (1.0, 0.0, 0.0), "B": (0.0, 1.0, 0.0), "N": (0.5, 0.5, 0.0)}


def z_at_decisions(pf: pd.DataFrame, w: str) -> pd.Series:
    _, _, f, n = ZW.WIN[w]; dec = pf.index[pf["decision"]]
    z = ZW.z_of(ZW.series(pf, f)["ratio"], n)
    return z if f == "W" else z.reindex(dec)


def rule_states(z: np.ndarray, thr: float = 1.0, allow=None):
    """The engine's state machine (T03.md §3 step 5) written on z: z < -1 is r < mean - sd, z >= 0 is r >= mean.
    thr = the extreme (1 = T03's band); allow = optional booleans per row: an entry is taken only where allow is True
    (E05's filter; a first cross that is not allowed is not taken, and the next week is no longer a first cross)."""
    state = "N"; prev = None; states, reasons = [], []
    for i, x in enumerate(z):
        if np.isnan(x):
            states.append(state); reasons.append("no band"); prev = None; continue
        pos = -1 if x < -thr else (1 if x > thr else 0); reason = ""
        if (state == "A" and x >= 0) or (state == "B" and x <= 0):
            state = "N"; reason = "exit: back at the mean"
        if state == "N" and (allow is None or allow[i]):
            if pos == -1 and prev in (0, 1):
                state = "A"; reason = (reason + "; " if reason else "") + f"enter A: first week below mean - {thr:g} sd"
            elif pos == 1 and prev in (0, -1):
                state = "B"; reason = (reason + "; " if reason else "") + f"enter B: first week above mean + {thr:g} sd"
        states.append(state); reasons.append(reason); prev = pos
    return states, reasons


def wbook(cl: np.ndarray, fi: np.ndarray, kd: np.ndarray, W: np.ndarray, w0, rec: bool = False):
    """A book over any number of holdings (E11's book, which equals the engine's on Rule R): w0 bought on the session after
    the first decision day; then a trade at the next session's fills whenever the weekly target row changes (the first row
    is not traded on); 10.4 bps per side, so the post-trade value V' solves V' = V - c Σ|w_i V' - v_i|. NAV from 1.0 on the
    first decision day."""
    c = COST / 2 / 1e4; n = len(cl); k0 = int(kd[0]); kt = k0 + 1; w0 = np.asarray(w0, float)
    u = w0 / (fi[kt] * (1 + c)); cur = tuple(w0); starts = [kt]; U = [u.copy()]; sw = []
    for i in range(1, len(kd)):
        tw = tuple(W[i])
        if tw == cur:
            continue
        k = int(kd[i]) + 1
        if k >= n:
            break
        w = np.asarray(W[i], float); p = fi[k]; v = u * p; V = float(v.sum()); Vn = V
        for _ in range(6):
            Vn = V - c * float(np.abs(w * Vn - v).sum())
        u = w * Vn / p
        if rec:
            sw.append({"i": i, "k": k, "from": cur, "to": tw, "cost": (V - Vn) / V})
        cur = tw; starts.append(k); U.append(u.copy())
    nav = np.empty(n - k0); nav[0] = 1.0; b = starts + [n]
    for j, uu in enumerate(U):
        nav[b[j] - k0:b[j + 1] - k0] = cl[b[j]:b[j + 1]] @ uu
    return nav, sw


def book3(pf: pd.DataFrame, dec_index: pd.DatetimeIndex, targets: list, sell: np.ndarray, liq: np.ndarray, cost: float = COST):
    """The engine's book with the liquid ETF as a third holding: on a week gold (the pair's second leg) is SELL, the weight the
    rule gives gold is held in the liquid ETF. Returns the NAV (from the first banded decision day, as the engine's), the
    rule's own trades (the state changes, in the engine's record shape plus fill_liquid and liquid_after), the trade
    diagnostics (a gold trade's held return over gold and the liquid ETF as held, fill to fill) and the switch's trades."""
    days = pf.index; pos = {d: k for k, d in enumerate(days)}; c = cost / 2 / 1e4
    C3 = np.column_stack([pf["ca"].to_numpy(float), pf["cb"].to_numpy(float), liq]); F3 = np.column_stack([pf["fa"].to_numpy(float), pf["fb"].to_numpy(float), liq])
    k0 = pos[dec_index[0]]
    if k0 + 1 >= len(days):
        return None, [], [], []

    def w_of(tgt, s):
        wa, wb, wl = TW3[tgt]; return (wa, 0.0, wl + wb) if s else (wa, wb, wl)
    kt = k0 + 1; w = w_of("N", bool(sell[0])); u = np.array(w) / (F3[kt] * (1 + c)); cur, state = w, "N"
    trades = [{"decision_date": days[k0], "trade_date": days[kt], "from": "start", "to": "N", "fill_a": F3[kt, 0], "fill_b": F3[kt, 1], "fill_liquid": F3[kt, 2],
               "value_traded": 1.0, "cost": 1.0 - float(u @ F3[kt]), "liquid_after": w[2]}]
    sw_trades, alloc, starts, U = [], [(kt, w)], [kt], [u.copy()]
    for i in range(1, len(dec_index)):
        t, tgt = dec_index[i], targets[i]; wn = w_of(tgt, bool(sell[i]))
        if wn == cur:
            continue
        k = pos[t] + 1
        if k >= len(days):
            break
        p = F3[k]; v = u * p; V = float(v.sum()); Vn = V; wv = np.array(wn)
        for _ in range(6):
            Vn = V - c * float(np.abs(wv * Vn - v).sum())
        rec = {"decision_date": t, "trade_date": days[k], "from": state, "to": tgt, "fill_a": p[0], "fill_b": p[1], "fill_liquid": p[2],
               "value_traded": float(np.abs(wv * Vn - v).sum()), "cost": V - Vn, "liquid_after": wn[2]}
        (trades if tgt != state else sw_trades).append(rec)
        u = wv * Vn / p; cur, state = wn, tgt; alloc.append((k, wn)); starts.append(k); U.append(u.copy())
    nav = np.full(len(days), np.nan); nav[k0] = 1.0; b = starts + [len(days)]
    for j, uu in enumerate(U):
        nav[b[j]:b[j + 1]] = C3[b[j]:b[j + 1]] @ uu
    ak = np.array([a[0] for a in alloc]); adict = dict(alloc); extras = []
    for j in range(1, len(trades)):
        tr = trades[j]
        if tr["to"] not in ("A", "B"):
            continue
        ke = pos[tr["trade_date"]]; nx = trades[j + 1] if j + 1 < len(trades) else None
        if nx is None:
            extras.append({"entry_decision": tr["decision_date"], "entry_trade": tr["trade_date"], "exit_trade": None, "held": tr["to"], "held_ret": np.nan, "peer_ret": np.nan, "extra": np.nan, "open": True}); continue
        kx = pos[nx["trade_date"]]; ra = F3[kx, 0] / F3[ke, 0] - 1; rb = F3[kx, 1] / F3[ke, 1] - 1
        if tr["to"] == "A":
            held, peer = ra, rb
        else:   # gold or the liquid ETF, as held between the switch's trades
            g = 1.0; seg = [k for k in ak if ke <= k < kx] + [kx]
            for s0, s1 in zip(seg[:-1], seg[1:]):
                col = 2 if adict[s0][2] > 0 else 1; g *= F3[s1, col] / F3[s0, col]
            held, peer = g - 1, ra
        extras.append({"entry_decision": tr["decision_date"], "entry_trade": tr["trade_date"], "exit_trade": nx["trade_date"], "held": tr["to"], "held_ret": held, "peer_ret": peer, "extra": held - peer, "open": False})
    return pd.Series(nav, index=days).iloc[k0:], trades, extras, sw_trades


def run_states(pf: pd.DataFrame, z: pd.Series, cost: float = COST, thr: float = 1.0, allow=None, comp: bool = True) -> dict:
    """The rule on a pair frame and its z at the decision days, through the engine's book (on the latest data, a gold pair
    goes through book3: the switch)."""
    states, reasons = rule_states(z.to_numpy(), thr, allow)
    dec = pd.DataFrame({"z": z.to_numpy(), "state_after": states, "reason": reasons}, index=z.index); dec.index.name = "date"
    banded = dec[dec["z"].notna()]; dec_index = banded.index; targets = banded["state_after"].tolist()
    if SWITCH is not None and SWITCH.is_gold_pair(pf):
        sell = SWITCH.sell_at(dec_index); liq = SWITCH.liquid_on(pf.index)
        nav, trades, extras, swt = book3(pf, dec_index, targets, sell, liq, cost)
        off = book3(pf, dec_index, targets, np.zeros(len(dec_index), bool), liq, cost)[0]; eng, _ = book(pf, dec_index, targets, cost, PIT())
        out = {"pf": pf, "decisions": dec, "nav": nav, "trades": pd.DataFrame(trades), "extras": pd.DataFrame(extras), "switch_trades": pd.DataFrame(swt), "switched": True,
               "sell_share": float(sell.mean()), "book3_vs_engine": float(np.nanmax(np.abs(off.to_numpy() - eng.to_numpy())))}
    else:
        nav, trades = book(pf, dec_index, targets, cost, PIT(), record_trades=True)
        out = {"pf": pf, "decisions": dec, "nav": nav, "trades": pd.DataFrame(trades), "extras": pd.DataFrame(trade_extras(trades)), "switched": False}
    if comp:
        out["bh"] = comparators(pf, dec_index[0], cost, dec_index, PIT())["5050_buyhold"]
    return out


def run_variant(px: dict, pair: str, w: str, cost: float = COST, thr: float = 1.0) -> dict:
    pf = pair_frame(px, pair, "plain"); z = z_at_decisions(pf, w)
    return {"pair": pair, "window": w, **run_states(pf, z, cost, thr)}


def check_against_engine(v: dict) -> list:
    """On the latest data: the 52-week run's decisions must equal the engine's decide() on the same pair frame, and its book
    with the switch off must equal the engine's book (a run without the switch uses the engine's book itself)."""
    p = v.get("pair", "?"); bad = []
    d = decide(v["pf"], 52, "E1", PIT()); d = d[d["mean"].notna()]
    mine = v["decisions"]["state_after"].reindex(d.index)
    if not (mine.astype(str).to_numpy() == d["state_after"].astype(str).to_numpy()).all():
        bad.append(f"{p}: decisions differ from the engine's on {(mine.astype(str).to_numpy() != d['state_after'].astype(str).to_numpy()).sum()} weeks")
    if v.get("switched") and not v["book3_vs_engine"] < 1e-9:
        bad.append(f"{p}: the book with the switch off differs from the engine's (max {v['book3_vs_engine']:.2e})")
    return bad


def check_against_t03(v: dict) -> list:
    """For the 52-week window: the decisions, the trades and the NAV must be T03's (its evidence of run headline). On the
    latest data T03's evidence does not apply: check_against_engine() instead."""
    if LATEST:
        return check_against_engine(v)
    ev = os.path.join(ROOT, "tests", "T03", "evidence", "headline"); p = v["pair"]; bad = []
    d = pd.read_csv(os.path.join(ev, "decisions.csv"), parse_dates=["date"]); d = d[(d["pair"] == p) & (d["window"] == 52) & (d["exit"] == "E1") & d["mean"].notna()].set_index("date")
    mine = v["decisions"]["state_after"].reindex(d.index)
    if not (mine.astype(str).to_numpy() == d["state_after"].astype(str).to_numpy()).all():
        bad.append(f"{p}: decisions differ from T03 on {(mine.astype(str).to_numpy() != d['state_after'].astype(str).to_numpy()).sum()} weeks")
    nv = os.path.join(ev, f"nav_{p.replace('|', '_')}_52_E1_{COST:g}.csv")
    if os.path.exists(nv):
        n_ = pd.read_csv(nv, index_col=0, parse_dates=True)["nav"].dropna(); m_ = v["nav"].reindex(n_.index)
        if not np.allclose(m_.to_numpy(), n_.to_numpy(), rtol=0, atol=1e-9):
            bad.append(f"{p}: NAV differs from T03 (max {float(np.nanmax(np.abs(m_ - n_))):.2e})")
    return bad
