"""
The T03 rule run with each z window's band, for E03 and E04 (EXPLORATORY variants; T03 itself is the 52-week window).
The rule: from neutral, enter on the first week z < -1 (hold a) or z > +1 (hold b); exit when z is back through 0 (the
ratio back at its trailing mean); no maximum hold; touches inside a trade are ignored. Decisions stay weekly (the last
trading day of each ISO week): weekly windows use z on decision-week closes, daily windows read the daily z (computed on
every trading day) at each decision day. Trades, costs, fills and daily valuation go through the engine's own book
(analysis.ratio_engine.book / comparators / trade_extras), 20.8 bps round trip, exactly as T03.

With the 52-week window the decisions, trades and NAV must equal T03's evidence; check_against_t03() asserts it.
"""
import os

import numpy as np
import pandas as pd

import zwin as ZW
from analysis.ratio_engine import PIT, book, comparators, pair_frame, trade_extras

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST = 20.8


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


def run_states(pf: pd.DataFrame, z: pd.Series, cost: float = COST, thr: float = 1.0, allow=None, comp: bool = True) -> dict:
    """The rule on a pair frame and its z at the decision days, through the engine's book."""
    states, reasons = rule_states(z.to_numpy(), thr, allow)
    dec = pd.DataFrame({"z": z.to_numpy(), "state_after": states, "reason": reasons}, index=z.index); dec.index.name = "date"
    banded = dec[dec["z"].notna()]; dec_index = banded.index; targets = banded["state_after"].tolist()
    nav, trades = book(pf, dec_index, targets, cost, PIT(), record_trades=True)
    out = {"pf": pf, "decisions": dec, "nav": nav, "trades": pd.DataFrame(trades), "extras": pd.DataFrame(trade_extras(trades))}
    if comp:
        out["bh"] = comparators(pf, dec_index[0], cost, dec_index, PIT())["5050_buyhold"]
    return out


def run_variant(px: dict, pair: str, w: str, cost: float = COST, thr: float = 1.0) -> dict:
    pf = pair_frame(px, pair, "plain"); z = z_at_decisions(pf, w)
    return {"pair": pair, "window": w, **run_states(pf, z, cost, thr)}


def check_against_t03(v: dict) -> list:
    """For the 52-week window: the decisions, the trades and the NAV must be T03's (its evidence of run headline)."""
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
