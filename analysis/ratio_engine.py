"""
T01 / T02 / T03 engine: the ±1 sd ratio band, long-only switch, daily-valued book, the two nulls. Writes tests/<test>/evidence/<run>/.

    python analysis/ratio_engine.py --test T01 --run headline [--draws 2000] [--out tests/T01/evidence/<run>]
    python analysis/ratio_engine.py --test T02 --run headline [--draws 2000]
    python analysis/ratio_engine.py --test T03 --run headline [--draws 2000]

Everything here is the pre-registered design of tests/T01/T01.md §3; the parameters below are its constants, not knobs.
The one difference between the tests is the series the band is drawn on (T02.md §3 step 4), fixed by the test number
in TEST_RATIO, never by a command-line switch: T01 = ln(P_a ÷ P_b) (column `lr`), T02 = P_a ÷ P_b (column `ratio`).
T03 is T02 with no maximum hold (T03.md §3 step 5), fixed the same way in TEST_MAX_HOLD.
"""
import argparse
import itertools
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.approved_data import ASSETS, GOLD, fingerprints, load_daily  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = [f"{a}|{b}" for a, b in itertools.combinations(ASSETS, 2)]
INDEX_PAIRS = [p for p in PAIRS if GOLD not in p]
WINDOWS = (26, 52, 104)              # band windows in decision weeks; 52 is the reference (the middle of the grid)
EXITS = ("E1", "E2")                 # E1 back at the mean (deciding), E2 the opposite band
COSTS_BPS = (0.0, 20.8, 50.0)        # round trip; 20.8 is the reference
REF = {"window": 52, "exit": "E1", "cost_bps": 20.8}
MAX_HOLD = 26                        # decision weeks: T01 / T02's locked cap only (TEST_MAX_HOLD); from T03 on there is no cap
SD_MULT = 1.0
MIN_OBS_FRAC = 0.8
START = {"index": "2006-01-02", "gold": "2015-01-12"}
N_DRAWS_DEFAULT = 2000
MIN_ROUND_TRIPS, MIN_WEEKS = 4, 200  # the reportability gate
RATIO_FORMS = {"log": ("lr", "ln(P_a / P_b)"), "plain": ("ratio", "P_a / P_b")}   # form -> (column in decisions.csv, formula)
TEST_RATIO = {"T01": "log", "T02": "plain", "T03": "plain"}                        # the locked form of each test
TEST_MAX_HOLD = {"T01": MAX_HOLD, "T02": MAX_HOLD, "T03": None}                   # the locked cap of each test; None = no cap


class PIT:
    """Point-in-time bookkeeping: every assertion is counted; a violation is recorded, never silently passed."""
    def __init__(self):
        self.checks = 0; self.violations = []

    def check(self, ok: bool, what: str):
        self.checks += 1
        if not ok:
            self.violations.append(what)


# ------------------------------------------------------------------------------------------------------- the pair frame
def pair_frame(px: dict, pair: str, ratio: str = "log") -> pd.DataFrame:
    """The pair's common calendar, closes, fills, decision days and the band's series: `lr` = ln(ca ÷ cb) for the log form,
    `ratio` = ca ÷ cb for the plain form. The orientation is the pair's own (a = the first asset in ASSETS) in both."""
    a, b = pair.split("|")
    start = pd.Timestamp(START["gold"] if GOLD in pair else START["index"])
    fa, fb = px[a], px[b]
    idx = fa.index.intersection(fb.index); idx = idx[idx >= start]
    df = pd.DataFrame({"ca": fa.loc[idx, "close"], "cb": fb.loc[idx, "close"], "fa": fa.loc[idx, "fill"], "fb": fb.loc[idx, "fill"],
                       "stale": fa.loc[idx, "stale_ohl"] | fb.loc[idx, "stale_ohl"]}, index=idx)
    iso = df.index.isocalendar()
    key = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    last_of_week = ~pd.Series(key.values, index=df.index).duplicated(keep="last")
    df["decision"] = last_of_week.values
    if ratio == "log":
        df["lr"] = np.log(df["ca"]) - np.log(df["cb"])
    elif ratio == "plain":
        df["ratio"] = df["ca"] / df["cb"]
    else:
        raise ValueError(f"unknown ratio form {ratio!r}")
    return df


# ------------------------------------------------------------------------------------------------------- the decisions
def decide(pf: pd.DataFrame, window: int, exit_rule: str, pit: PIT, max_hold: int | None = None, ratio: str = "log") -> pd.DataFrame:
    """One row per decision week: the band, the position vs the band, the state before and after, the reason. The band at
    week t is computed from the weekly series truncated at t (asserted), so nothing after t is used. The series is the
    column of `ratio` in RATIO_FORMS (lr or ratio); the variable `lr` below holds whichever it is. max_hold None (the default) = no cap; run() always passes the test's own."""
    col = RATIO_FORMS[ratio][0]
    dec = pf[pf["decision"]]
    lr = dec[col]; dates = dec.index; min_obs = int(round(MIN_OBS_FRAC * window))
    rows = []; state = "N"; entered = None; prev_pos = None; ignored = 0
    for i, t in enumerate(dates):
        hist = lr.iloc[: i + 1]
        pit.check(hist.index[-1] == t and len(hist) == i + 1, f"band at {t.date()} not from the truncated series")
        if len(hist) < min_obs:
            rows.append({"date": t, col: lr.iloc[i], "mean": np.nan, "sd": np.nan, "pos": np.nan, "state_before": state, "state_after": state, "reason": "no band", "held": np.nan})
            prev_pos = None; continue
        w = hist.iloc[-window:]; m, s = float(w.mean()), float(w.std(ddof=1))
        x = float(lr.iloc[i]); pos = -1 if x < m - SD_MULT * s else (1 if x > m + SD_MULT * s else 0)
        before = state; reason = ""
        if state in ("A", "B"):
            held = i - entered
            if state == "A":
                back = (x >= m) if exit_rule == "E1" else (pos == 1)
            else:
                back = (x <= m) if exit_rule == "E1" else (pos == -1)
            if back:
                reason = "exit: back at the mean" if exit_rule == "E1" else "exit: opposite band"; state = "N"; entered = None
            elif max_hold is not None and held >= max_hold:
                reason = f"exit: max hold {max_hold} weeks"; state = "N"; entered = None
        if state == "N":
            first_below = pos == -1 and prev_pos in (0, 1)
            first_above = pos == 1 and prev_pos in (0, -1)
            if first_below:
                state = "A"; entered = i; reason = (reason + "; " if reason else "") + "enter A: first week below mean - 1 sd"
            elif first_above:
                state = "B"; entered = i; reason = (reason + "; " if reason else "") + "enter B: first week above mean + 1 sd"
        elif pos != 0 and not reason:
            ignored += 1
        rows.append({"date": t, col: x, "mean": m, "sd": s, "pos": pos, "state_before": before, "state_after": state, "reason": reason, "held": (i - entered) if entered is not None else np.nan})
        prev_pos = pos
    out = pd.DataFrame(rows).set_index("date"); out.attrs["touches_ignored"] = ignored
    return out


# ------------------------------------------------------------------------------------------------------- spells and nulls
def spells(targets: list[str]) -> list[tuple[str, int]]:
    seg = []
    for s in targets:
        if seg and seg[-1][0] == s:
            seg[-1] = (s, seg[-1][1] + 1)
        else:
            seg.append((s, 1))
    return seg


def to_blocks(seg: list[tuple[str, int]]) -> tuple[list[int], list[list[tuple[str, int]]]]:
    """g0, B1, g1, B2, ..., BK, gK: a block is a maximal run of abutting trade spells (e.g. [A 5, B 3] when the sleeve passed
    straight from A to B); interior gaps are > 0 by construction, g0 and gK may be 0."""
    gaps, blocks = [], []; cur = []; g = 0; seen_block = False
    for s, n in seg:
        if s == "N":
            if cur:
                blocks.append(cur); cur = []
            g += n
        else:
            if not cur:
                gaps.append(g); g = 0
            cur.append((s, n))
    if cur:
        blocks.append(cur)
    gaps.append(g)
    assert len(gaps) == len(blocks) + 1
    return gaps, blocks


def rebuild(gaps: list[int], blocks: list[list[tuple[str, int]]]) -> list[str]:
    out = ["N"] * gaps[0]
    for k, blk in enumerate(blocks):
        for s, n in blk:
            out += [s] * n
        out += ["N"] * gaps[k + 1]
    return out


def counts(targets: list[str]) -> dict:
    seg = spells(targets)
    return {"spells": sum(1 for s, _ in seg if s != "N"), "switches": max(len(seg) - 1, 0), "weeks_A": targets.count("A"), "weeks_B": targets.count("B"), "weeks_N": targets.count("N")}


def null_A(targets: list[str], rng: np.random.Generator) -> list[str]:
    """Timing null: the blocks of trade spells are permuted among the block positions and the non-zero gaps among the non-zero
    gap positions (a zero gap can only sit at the two ends and stays there). Every count of the strategy is preserved exactly."""
    gaps, blocks = to_blocks(spells(targets))
    nz = [i for i, g in enumerate(gaps) if g > 0]
    g2 = list(gaps); vals = [gaps[i] for i in nz]
    for i, j in zip(nz, rng.permutation(len(vals))):
        g2[i] = vals[j]
    b2 = [blocks[j] for j in rng.permutation(len(blocks))]
    return rebuild(g2, b2)


def null_B(targets: list[str], rng: np.random.Generator) -> list[str]:
    """Leg null: the real timing kept; every block's legs are flipped together by one fair coin (A<->B), which keeps the
    alternation inside a block and therefore every count."""
    gaps, blocks = to_blocks(spells(targets))
    flip = {"A": "B", "B": "A"}
    b2 = [([(flip[s], n) for s, n in blk] if rng.integers(2) == 1 else list(blk)) for blk in blocks]
    return rebuild(gaps, b2)


# ------------------------------------------------------------------------------------------------------- the book
def book(pf: pd.DataFrame, dec_index: pd.DatetimeIndex, targets: list[str], cost_bps: float, pit: PIT, record_trades: bool = False):
    """Daily NAV of the sleeve following `targets` (one state per decision week, from the first banded week). Trades on the
    calendar day after each decision at the `fill` prices; 10.4 bps per side on value traded (cost_bps round trip)."""
    c = cost_bps / 2 / 1e4
    days = pf.index; pos_of = {d: k for k, d in enumerate(days)}
    ca, cb, fa, fb = pf["ca"].to_numpy(), pf["cb"].to_numpy(), pf["fa"].to_numpy(), pf["fb"].to_numpy()
    k0 = pos_of[dec_index[0]]                      # the first banded decision day: valuation starts here at 1.0
    if k0 + 1 >= len(days):
        return None, []
    ua = np.zeros(len(days)); ub = np.zeros(len(days)); nav = np.full(len(days), np.nan)
    # first trade: 50/50 by value on the next day
    kt = k0 + 1; pit.check(days[kt] > days[k0], "first trade not after the first decision")
    u_a = 0.5 / (fa[kt] * (1 + c)); u_b = 0.5 / (fb[kt] * (1 + c)); held = "N"
    trades = [{"decision_date": days[k0], "trade_date": days[kt], "from": "start", "to": "N", "fill_a": fa[kt], "fill_b": fb[kt], "value_traded": 1.0, "cost": 1.0 - (u_a * fa[kt] + u_b * fb[kt])}]
    seg_start = kt; nav[k0] = 1.0
    cur_a, cur_b = u_a, u_b
    for i, (t, tgt) in enumerate(zip(dec_index, targets)):
        if i == 0 or tgt == held:
            continue
        kd = pos_of[t]; kt2 = kd + 1
        if kt2 >= len(days):
            break                                              # a decision on the last day cannot be executed
        pit.check(days[kt2] > t, f"trade at {days[kt2].date()} not after the decision at {t.date()}")
        ua[seg_start:kt2] = cur_a; ub[seg_start:kt2] = cur_b
        pa, pb = fa[kt2], fb[kt2]; va, vb = cur_a * pa, cur_b * pb
        if tgt == "A":
            proceeds = vb * (1 - c); n_a = cur_a + proceeds / (pa * (1 + c)); n_b = 0.0; traded = vb + proceeds; cost = vb * c + proceeds * c / (1 + c)
        elif tgt == "B":
            proceeds = va * (1 - c); n_b = cur_b + proceeds / (pb * (1 + c)); n_a = 0.0; traded = va + proceeds; cost = va * c + proceeds * c / (1 + c)
        else:                                                   # back to 50/50 by value: sell x of the heavier leg
            if va >= vb:
                x = (va - vb) / (1 + (1 - c) / (1 + c)); proceeds = x * (1 - c)
                n_a = cur_a - x / pa; n_b = cur_b + proceeds / (pb * (1 + c)); traded = x + proceeds; cost = x * c + proceeds * c / (1 + c)
            else:
                x = (vb - va) / (1 + (1 - c) / (1 + c)); proceeds = x * (1 - c)
                n_b = cur_b - x / pb; n_a = cur_a + proceeds / (pa * (1 + c)); traded = x + proceeds; cost = x * c + proceeds * c / (1 + c)
        if record_trades:
            trades.append({"decision_date": t, "trade_date": days[kt2], "from": held, "to": tgt, "fill_a": pa, "fill_b": pb, "value_traded": traded, "cost": cost, "units_a_after": n_a, "units_b_after": n_b})
        cur_a, cur_b, held, seg_start = n_a, n_b, tgt, kt2
    ua[seg_start:] = cur_a; ub[seg_start:] = cur_b
    nav[kt:] = ua[kt:] * ca[kt:] + ub[kt:] * cb[kt:]
    s = pd.Series(nav, index=days).iloc[k0:]
    return s, trades


def stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    days = (nav.index[-1] - nav.index[0]).days
    r = np.log(nav).diff().dropna()
    return {"cagr": float(nav.iloc[-1] ** (365.25 / days) - 1) if days > 0 else np.nan, "maxdd": float((nav / nav.cummax() - 1).min()), "vol": float(r.std(ddof=1) * np.sqrt(252)),
            "final_nav": float(nav.iloc[-1]), "first_date": nav.index[0].date().isoformat(), "last_date": nav.index[-1].date().isoformat(), "days": int(days)}


def comparators(pf: pd.DataFrame, k0_date: pd.Timestamp, cost_bps: float, dec_index: pd.DatetimeIndex, pit: PIT) -> dict:
    c = cost_bps / 2 / 1e4
    sub = pf.loc[k0_date:]; days = sub.index
    out = {}
    for name, col, fcol in (("hold_a", "ca", "fa"), ("hold_b", "cb", "fb")):
        u = 1.0 / (sub[fcol].iloc[1] * (1 + c)); s = pd.Series(np.nan, index=days); s.iloc[0] = 1.0; s.iloc[1:] = u * sub[col].iloc[1:].to_numpy(); out[name] = s
    s, _ = book(pf, dec_index, ["N"] * len(dec_index), cost_bps, pit); out["always_neutral"] = s
    ua = 0.5 / (sub["fa"].iloc[1] * (1 + c)); ub = 0.5 / (sub["fb"].iloc[1] * (1 + c))
    s = pd.Series(np.nan, index=days); s.iloc[0] = 1.0; s.iloc[1:] = (ua * sub["ca"].iloc[1:] + ub * sub["cb"].iloc[1:]).to_numpy(); out["5050_buyhold"] = s
    # constant mix: rebalanced to 50/50 at every decision week's next-day fill, with cost on the value moved
    cur_a, cur_b = ua, ub; nav = np.full(len(days), np.nan); nav[0] = 1.0; pos_of = {d: k for k, d in enumerate(days)}
    seg = 1; UA = np.zeros(len(days)); UB = np.zeros(len(days))
    for t in dec_index[1:]:
        kd = pos_of.get(t); kt = (kd + 1) if kd is not None else None
        if kt is None or kt >= len(days):
            continue
        UA[seg:kt] = cur_a; UB[seg:kt] = cur_b
        pa, pb = sub["fa"].iloc[kt], sub["fb"].iloc[kt]; va, vb = cur_a * pa, cur_b * pb
        if va >= vb:
            x = (va - vb) / (1 + (1 - c) / (1 + c)); cur_a -= x / pa; cur_b += x * (1 - c) / (pb * (1 + c))
        else:
            x = (vb - va) / (1 + (1 - c) / (1 + c)); cur_b -= x / pb; cur_a += x * (1 - c) / (pa * (1 + c))
        seg = kt
    UA[seg:] = cur_a; UB[seg:] = cur_b
    nav[1:] = UA[1:] * sub["ca"].iloc[1:].to_numpy() + UB[1:] * sub["cb"].iloc[1:].to_numpy(); out["5050_rebal"] = pd.Series(nav, index=days)
    return out


def trade_extras(trades: list[dict]) -> list[dict]:
    """The diagnostic: for every trade spell, the held leg's fill-to-fill return minus the peer's over the hold."""
    out = []
    for k in range(1, len(trades)):
        tr = trades[k]
        if tr["to"] in ("A", "B"):
            nxt = trades[k + 1] if k + 1 < len(trades) else None
            if nxt is None:
                out.append({"entry_decision": tr["decision_date"], "entry_trade": tr["trade_date"], "exit_trade": None, "held": tr["to"], "held_ret": np.nan, "peer_ret": np.nan, "extra": np.nan, "open": True}); continue
            ra = nxt["fill_a"] / tr["fill_a"] - 1; rb = nxt["fill_b"] / tr["fill_b"] - 1
            held, peer = (ra, rb) if tr["to"] == "A" else (rb, ra)
            out.append({"entry_decision": tr["decision_date"], "entry_trade": tr["trade_date"], "exit_trade": nxt["trade_date"], "held": tr["to"], "held_ret": held, "peer_ret": peer, "extra": held - peer, "open": False})
    return out


# ------------------------------------------------------------------------------------------------------- the run
def run(out_dir: str, n_draws: int, test: str = "T01"):
    ratio = TEST_RATIO[test]; max_hold = TEST_MAX_HOLD[test]
    t_start = time.time(); os.makedirs(out_dir, exist_ok=True)
    px = load_daily(); pit = PIT()
    summary, decisions_all, trades_all, extras_all, draw_rows = [], [], [], [], {}
    null_counts_ok = {"draws_checked": 0, "draws_mismatched": 0}
    stale_rows = 0
    for pid, pair in enumerate(PAIRS):
        pf = pair_frame(px, pair, ratio); stale_rows += int(pf["stale"].sum())
        for window in WINDOWS:
            for eid, exit_rule in enumerate(EXITS):
                dec = decide(pf, window, exit_rule, pit, max_hold=max_hold, ratio=ratio)
                banded = dec[dec["mean"].notna()]
                if len(banded) < 2:
                    continue
                dec_index = banded.index; targets = banded["state_after"].tolist()
                dec_out = dec.assign(pair=pair, window=window, exit=exit_rule); decisions_all.append(dec_out.reset_index())
                base = counts(targets); n_rt = base["spells"]
                reportable = (n_rt >= MIN_ROUND_TRIPS) and (len(dec_index) >= MIN_WEEKS)
                is_ref = (window == REF["window"])
                for cost in COSTS_BPS:
                    if cost != REF["cost_bps"] and not is_ref:
                        continue                                   # R1 (costs) on the reference window only
                    nav, trades = book(pf, dec_index, targets, cost, pit, record_trades=True)
                    st = stats(nav)
                    comp = comparators(pf, dec_index[0], cost, dec_index, pit)
                    row = {"pair": pair, "window": window, "exit": exit_rule, "cost_bps": cost, "series": "strategy", **st, "round_trips": n_rt, "switches": base["switches"], "weeks_A": base["weeks_A"], "weeks_B": base["weeks_B"], "weeks_N": base["weeks_N"],
                           "n_decisions": len(dec_index), "touches_ignored": dec.attrs["touches_ignored"], "reportable": reportable, "deciding": pair in INDEX_PAIRS}
                    nav.rename("nav").to_frame().assign(**{k: v.reindex(nav.index) for k, v in comp.items()}).to_csv(os.path.join(out_dir, f"nav_{pair.replace('|', '_')}_{window}_{exit_rule}_{cost:g}.csv"))
                    if cost == REF["cost_bps"]:
                        tr = pd.DataFrame(trades).assign(pair=pair, window=window, exit=exit_rule); trades_all.append(tr)
                        ex = pd.DataFrame(trade_extras(trades)).assign(pair=pair, window=window, exit=exit_rule); extras_all.append(ex)
                    for name, s in comp.items():
                        summary.append({"pair": pair, "window": window, "exit": exit_rule, "cost_bps": cost, "series": name, **stats(s), "reportable": reportable, "deciding": pair in INDEX_PAIRS})
                    # the nulls: at every window/exit at the reference cost; at the other costs only on the reference window
                    if n_draws > 0 and (cost == REF["cost_bps"] or is_ref):
                        rows_d = []
                        for null_name, fn, seed_off in (("N-A", null_A, 0), ("N-B", null_B, 100)):
                            for d in range(n_draws):
                                rng = np.random.default_rng([pid + seed_off, eid, d, window])
                                tg = fn(targets, rng); cc = counts(tg)
                                null_counts_ok["draws_checked"] += 1
                                ok = (cc == base) if null_name == "N-A" else (cc["spells"] == base["spells"] and cc["switches"] == base["switches"] and cc["weeks_N"] == base["weeks_N"] and cc["weeks_A"] + cc["weeks_B"] == base["weeks_A"] + base["weeks_B"])
                                null_counts_ok["draws_mismatched"] += (not ok)
                                nv, _ = book(pf, dec_index, tg, cost, pit); sd_ = stats(nv)
                                rows_d.append({"null": null_name, "draw": d, "cagr": sd_["cagr"], "maxdd": sd_["maxdd"], "vol": sd_["vol"], **cc, "counts_ok": ok})
                        dd = pd.DataFrame(rows_d); draw_rows[(pair, window, exit_rule, cost)] = dd
                        for null_name in ("N-A", "N-B"):
                            g = dd[dd["null"] == null_name]
                            worse = (g["cagr"] < st["cagr"]).sum(); tie = (g["cagr"] == st["cagr"]).sum(); atleast = (g["cagr"] >= st["cagr"]).sum()
                            k = null_name.replace("-", "")
                            row[f"{k}_pct"] = float((worse + 0.5 * tie) / len(g) * 100); row[f"{k}_p"] = float((1 + atleast) / (len(g) + 1))
                            row[f"{k}_median_cagr"] = float(g["cagr"].median()); row[f"{k}_sd_cagr"] = float(g["cagr"].std(ddof=1)); row[f"{k}_edge"] = float(st["cagr"] - g["cagr"].median())
                            row[f"{k}_median_maxdd"] = float(g["maxdd"].median()); row[f"{k}_pct_dd"] = float(((g["maxdd"] < st["maxdd"]).sum() + 0.5 * (g["maxdd"] == st["maxdd"]).sum()) / len(g) * 100)
                            row[f"{k}_mde"] = float(2.8 * g["cagr"].std(ddof=1)); row[f"{k}_draws"] = int(len(g))
                        dd.to_csv(os.path.join(out_dir, f"draws_{pair.replace('|', '_')}_{window}_{exit_rule}_{cost:g}.csv"), index=False)
                    summary.append(row)
                print(f"  {pair:16s} W{window:<3d} {exit_rule}  decisions {len(dec_index):4d}  round trips {n_rt:3d}  {'' if reportable else 'NOT REPORTABLE'}  [{time.time() - t_start:5.0f}s]", flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.concat(decisions_all).to_csv(os.path.join(out_dir, "decisions.csv"), index=False)
    pd.concat(trades_all).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    pd.concat(extras_all).to_csv(os.path.join(out_dir, "trade_extras.csv"), index=False)
    audit = {"run_date": date.today().isoformat(), "test": test, "ratio": ratio, "ratio_formula": RATIO_FORMS[ratio][1], "inputs": fingerprints(), "pairs": PAIRS, "deciding_pairs": INDEX_PAIRS, "windows": WINDOWS, "exits": EXITS, "costs_bps": COSTS_BPS, "reference": REF, "max_hold_weeks": max_hold, "sd_mult": SD_MULT, "min_obs_frac": MIN_OBS_FRAC, "start": START,
             "draws_per_null": n_draws, "seed_formula": "default_rng([pair_id (+100 for N-B), exit_id, draw_id, window])", "pit_checks": pit.checks, "pit_violations": pit.violations, "null_counts": null_counts_ok, "stale_ohl_rows_in_pair_frames": stale_rows,
             "reportability": {"min_round_trips": MIN_ROUND_TRIPS, "min_weeks": MIN_WEEKS}, "seconds": round(time.time() - t_start)}
    json.dump(audit, open(os.path.join(out_dir, "audit.json"), "w"), indent=1, default=str)
    print(f"done in {time.time() - t_start:.0f}s → {out_dir}; ratio {RATIO_FORMS[ratio][1]}; max hold {max_hold}; PIT checks {pit.checks}, violations {len(pit.violations)}; null draws {null_counts_ok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", default="T01", choices=sorted(TEST_RATIO)); ap.add_argument("--run", default="headline"); ap.add_argument("--draws", type=int, default=N_DRAWS_DEFAULT); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "tests", a.test, "evidence", a.run)
    run(out, a.draws, a.test)
