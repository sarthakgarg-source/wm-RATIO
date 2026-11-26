"""
The exploratory studies on the latest data — the desk's request of 2026-11-26: "try to take all the data from 2012 or from
the date it is available", "also take mcx gold data from 2011", and "whenever we have taken gold and it was in sell we
need to move to liquid etf" (the desk chose to replace the original rule; Micro250 stays on its earlier data).

    import latest as LT
    px = LT.load_india()   # N100, MC100, SC100, MCX gold (to 2026-11-20) and Micro250 (to 2026-08-28), all from 2011-01-01; the liquid ETF
    LT.activate(px)        # the engine's pair start → 2011-01-01 (this process only); the switch on in variants.py; T03 checks → engine checks

- Data: analysis/approved_data.load_india_latest() (DATA_REVIEW_AA.md) for N100, MC100, SC100, MCX gold and the liquid
  ETF ('Liquid', the LBES cash level to 2026-11-25; F-RA-D17: no recorded source before 2024-01-24); Micro250 from
  load_daily() (no newer file). Every series is cut at 2011-01-01, so the 52-week windows have their year before 2012.
- The engine (analysis/ratio_engine.pair_frame) starts index pairs on 2006-01-02 and gold pairs on 2015-01-12 (the first day
  of the earlier gold file). activate() sets both to 2011-01-01 in the running process only; the locked tests never import
  this module and keep their inputs.
- The switch: whenever a strategy would hold MCX gold while gold's weekly supertrend is SELL at the decision week, it holds
  the liquid ETF instead. "SELL" is the desk's Weekly LW supertrend (E10's functions, checked there against the desk's own
  code) on gold's own weekly bars with real highs and lows, a bar usable once its week has ended on gold's calendar. The
  liquid level is read as of each day (the latest level on or before it) and traded at 10.4 bps per side like any holding.
  Bought-and-held benchmarks are not switched. variants.run_states applies it to Rule R on the gold pairs; E10 and E11 apply
  it in their own books.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p_ in (ROOT, HERE, os.path.join(ROOT, "explore", "E10_trend_rotation")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import analysis.ratio_engine as RE  # noqa: E402
from analysis.approved_data import GOLD, load_daily, load_india_latest  # noqa: E402
import variants as VR  # noqa: E402

DATA_FROM = pd.Timestamp("2011-01-01"); LIQ = "Liquid"
NOTE = ("latest data (approved load_india_latest, 2026-11-26): N100, MC100, SC100 and MCX gold to 2026-11-20, Micro250 to 2026-08-28, all from 2011-01-01; "
        "whenever a strategy would hold gold while gold's weekly supertrend is SELL it holds the liquid ETF instead (the switch; the liquid level has no recorded source before 2024-01-24, F-RA-D17)")
SW = None


def load_india() -> dict:
    L = load_india_latest(); old = load_daily()
    out = {k: L[k][L[k].index >= DATA_FROM] for k in ("N100", "MC100", "SC100", GOLD)}
    out["Micro250"] = old["Micro250"][old["Micro250"].index >= DATA_FROM]
    out[LIQ] = L[LIQ]
    return out


class Switch:
    """Gold's weekly supertrend (SELL weeks) and the liquid level, for the switch."""

    def __init__(self, px: dict):
        import rotation as E10   # the desk's supertrend (E10), imported here so that E10 itself can import this module
        g = px[GOLD]; self.bars = E10.desk_bars(g, False); self.state, self.stop = E10.desk_supertrend(self.bars); self.cal = E10.cal_end_of(g.index)
        self._leg_state_at = E10.leg_state_at; self.gold_close = g["close"]; self.liq = px[LIQ]["close"]

    def state_at(self, dec) -> np.ndarray:
        return self._leg_state_at(pd.DatetimeIndex(dec), self.bars, self.state, self.cal)

    def sell_at(self, dec) -> np.ndarray:
        return self.state_at(dec) == -1

    def liquid_on(self, idx) -> np.ndarray:
        return self.liq.reindex(pd.DatetimeIndex(idx), method="ffill").to_numpy(float)

    def carried_days(self, idx) -> int:
        return int((~pd.DatetimeIndex(idx).isin(self.liq.index)).sum())

    def is_gold_pair(self, pf: pd.DataFrame) -> bool:
        """A pair frame whose second leg is MCX gold (gold is always the second leg of a pair)."""
        g = self.gold_close.reindex(pf.index)
        return bool(g.notna().all() and np.array_equal(pf["cb"].to_numpy(float), g.to_numpy(float)))


def activate(px: dict) -> Switch:
    global SW
    RE.START.update({"index": DATA_FROM.strftime("%Y-%m-%d"), "gold": DATA_FROM.strftime("%Y-%m-%d")})   # in place: studies that imported START see it
    SW = Switch(px); VR.SWITCH = SW; VR.LATEST = True
    return SW
