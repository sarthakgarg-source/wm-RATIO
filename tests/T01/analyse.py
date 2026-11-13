"""
Step 8–9: the gates, then every table of T01.md §4, in the playbook's order, read ONLY from evidence/<run>/.
Writes evidence/<run>/analysis.log (markdown) and evidence/<run>/tables.json (every number, for fill_writeup / report / check_page).

    python tests/T01/analyse.py [run-name]          # default: headline
Exit 1 if any validity gate fails (then nothing after the gates may be read).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "headline"
EV = os.path.join(HERE, "evidence", RUN)
PASS_PCT, DD_RULE, BEST_PCT = 95.0, "median", 98.0
NEED = 4
PERIODS = (("2006–2015", "2006-01-01", "2015-12-31"), ("2016–2019", "2016-01-01", "2019-12-31"), ("2020–2023", "2020-01-01", "2023-12-31"), ("2024–2026", "2024-01-01", "2026-12-31"))
L = []; T = {}


def say(x=""):
    L.append(x)


def md(df: pd.DataFrame, floatfmt=None) -> str:
    cols = list(df.columns); rows = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join((floatfmt(c, v) if floatfmt else str(v)) for c, v in r.items()) + " |")
    return "\n".join(rows)


def pct(v): return "—" if pd.isna(v) else f"{v:.1%}"
def p1(v): return "—" if pd.isna(v) else f"{v:.1f}"
def p3(v): return "—" if pd.isna(v) else f"{v:.3f}"


def main():
    audit = json.load(open(os.path.join(EV, "audit.json"))); summ = pd.read_csv(os.path.join(EV, "summary.csv")); ref = audit["reference"]
    decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]); trades = pd.read_csv(os.path.join(EV, "trades.csv"), parse_dates=["decision_date", "trade_date"]); extras = pd.read_csv(os.path.join(EV, "trade_extras.csv"))
    deciding = audit["deciding_pairs"]; gold_pairs = [p for p in audit["pairs"] if p not in deciding]
    strat = summ[summ["series"] == "strategy"].copy()
    R = strat[(strat["window"] == ref["window"]) & (strat["exit"] == ref["exit"]) & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair")
    T["run"] = RUN; T["run_date"] = audit["run_date"]; T["draws"] = audit["draws_per_null"]; T["reference"] = ref

    # ---------------------------------------------------------------- 4.0 worked example, gates, null check
    say("### 4.0 Worked example · gates · null check"); say()
    pair0 = "N100|SC100"; d0 = decs[(decs["pair"] == pair0) & (decs["window"] == ref["window"]) & (decs["exit"] == ref["exit"])]
    first_entry = d0[d0["reason"].str.startswith("enter", na=False)].iloc[0]; tr0 = trades[(trades["pair"] == pair0) & (trades["window"] == ref["window"]) & (trades["exit"] == ref["exit"])]
    t_entry = tr0[tr0["decision_date"] == first_entry["date"]].iloc[0]; t_exit = tr0[tr0["decision_date"] > first_entry["date"]].iloc[0]
    ex0 = extras[(extras["pair"] == pair0) & (extras["window"] == ref["window"]) & (extras["exit"] == ref["exit"])].iloc[0]
    r0 = R.loc[pair0]; dr0 = pd.read_csv(os.path.join(EV, f"draws_{pair0.replace('|', '_')}_{ref['window']}_{ref['exit']}_{ref['cost_bps']:g}.csv")); dA = dr0[dr0["null"] == "N-A"]
    T["worked"] = {"pair": pair0, "entry_decision": first_entry["date"].date().isoformat(), "lr": float(first_entry["lr"]), "mean": float(first_entry["mean"]), "sd": float(first_entry["sd"]), "state": first_entry["state_after"],
                   "entry_trade": t_entry["trade_date"].date().isoformat(), "fill_a": float(t_entry["fill_a"]), "fill_b": float(t_entry["fill_b"]), "cost": float(t_entry["cost"]), "exit_decision": t_exit["decision_date"].date().isoformat(), "exit_trade": t_exit["trade_date"].date().isoformat(), "exit_reason": str(d0[d0["date"] == t_exit["decision_date"]].iloc[0]["reason"]),
                   "held_ret": float(ex0["held_ret"]), "peer_ret": float(ex0["peer_ret"]), "extra": float(ex0["extra"]), "first_date": r0["first_date"], "last_date": r0["last_date"], "final_nav": float(r0["final_nav"]), "cagr": float(r0["cagr"]), "maxdd": float(r0["maxdd"]),
                   "draw0_cagr": float(dA.iloc[0]["cagr"]), "draw0_spells": int(dA.iloc[0]["spells"]), "NA_pct": float(r0["NA_pct"]), "NA_p": float(r0["NA_p"]), "NA_median": float(r0["NA_median_cagr"]), "NA_edge": float(r0["NA_edge"])}
    w = T["worked"]
    say(f"**One decision ({pair0}, window {ref['window']}, exit {ref['exit']}).** On {w['entry_decision']} the log ratio ln(N100 ÷ SC100) closed at {w['lr']:+.4f} against a trailing-52-week mean of {w['mean']:+.4f} and sd {w['sd']:.4f}: {'below mean − 1 sd, so N100 is the lagging leg' if w['state'] == 'A' else 'above mean + 1 sd, so SC100 is the lagging leg'} → target {w['state']}. The trade happened on {w['entry_trade']} at fills N100 {w['fill_a']:.2f} / SC100 {w['fill_b']:.2f}; cost {w['cost']:.6f} of NAV. The position closed on the decision of {w['exit_decision']} ({w['exit_reason']}), traded {w['exit_trade']}: the held leg returned {w['held_ret']:+.2%} fill to fill against the peer's {w['peer_ret']:+.2%} — extra {w['extra']:+.2%}.")
    say(); say(f"**One path.** NAV 1.000 on {w['first_date']} → {w['final_nav']:.3f} on {w['last_date']}: CAGR {w['cagr']:.2%}, worst fall {w['maxdd']:.1%}.")
    say(); say(f"**One draw (N-A, draw 0).** The same {w['draw0_spells']} trade spells scattered at random: CAGR {w['draw0_cagr']:.2%}. **One ranking.** The strategy's {w['cagr']:.2%} sits at the {w['NA_pct']:.1f}th percentile of {T['draws']:,} draws (p = {w['NA_p']:.3f}); the null's median is {w['NA_median']:.2%}, edge {w['NA_edge']:+.2%}."); say()
    # gates
    vlog = open(os.path.join(EV, "verify.log")).read() if os.path.exists(os.path.join(EV, "verify.log")) else ""
    m = re.search(r"verify: (\d+) nav files, (\d+) comparisons, (\d+) failures", vlog); caught = ("caught" in vlog) and ("NOT CAUGHT" not in vlog)
    v1 = bool(m and int(m.group(3)) == 0 and int(m.group(2)) > 0 and caught)
    v2 = audit["pit_checks"] > 0 and len(audit["pit_violations"]) == 0
    draws_files = [f for f in os.listdir(EV) if f.startswith("draws_")]; all_ok = all(pd.read_csv(os.path.join(EV, f))["counts_ok"].all() for f in draws_files)
    v3 = audit["null_counts"]["draws_mismatched"] == 0 and audit["null_counts"]["draws_checked"] > 0 and all_ok
    an = summ[summ["series"] == "always_neutral"].set_index(["pair", "window", "exit", "cost_bps"])["final_nav"]; bh = summ[summ["series"] == "5050_buyhold"].set_index(["pair", "window", "exit", "cost_bps"])["final_nav"]
    v4a = bool(np.allclose(an.reindex(bh.index).to_numpy(), bh.to_numpy(), atol=1e-9))
    prev = os.path.join(HERE, "evidence", "previous_summary.csv"); v4b = None
    if os.path.exists(prev):
        ps = pd.read_csv(prev); key = ["pair", "window", "exit", "cost_bps", "series"]; mg = ps.merge(summ, on=key, suffixes=("_prev", ""))
        v4b = bool(np.allclose(mg["final_nav_prev"], mg["final_nav"], atol=1e-9)) if len(mg) else None
    gates = {"V1": v1, "V2": v2, "V3": v3, "V4": v4a and (v4b is not False)}; T["gates"] = {**gates, "verify": m.groups() if m else None, "pit_checks": audit["pit_checks"], "pit_violations": len(audit["pit_violations"]), "null_draws_checked": audit["null_counts"]["draws_checked"], "null_draws_mismatched": audit["null_counts"]["draws_mismatched"], "v4_regression": v4b}
    say("| gate | result | evidence |"); say("|---|---|---|")
    say(f"| V1 independent verifier | {'PASS' if v1 else 'FAIL'} | {m.group(1) if m else '?'} nav files, {m.group(2) if m else '?'} comparisons, {m.group(3) if m else '?'} failures; corrupted copies {'caught' if caught else 'NOT caught'} |")
    say(f"| V2 no look-ahead | {'PASS' if v2 else 'FAIL'} | {audit['pit_checks']:,} point-in-time checks, {len(audit['pit_violations'])} violations |")
    say(f"| V3 null held its promise | {'PASS' if v3 else 'FAIL'} | {audit['null_counts']['draws_checked']:,} draws checked, {audit['null_counts']['draws_mismatched']} with different counts |")
    say(f"| V4 regression | {'PASS' if gates['V4'] else 'FAIL'} | always-neutral sleeve = 50/50 buy-and-hold closed form: {'yes' if v4a else 'NO'}; previous run reproduced: {'n/a (first run)' if v4b is None else ('yes' if v4b else 'NO')} |"); say()
    if not all(gates.values()):
        open(os.path.join(EV, "analysis.log"), "w").write("\n".join(L)); json.dump(T, open(os.path.join(EV, "tables.json"), "w"), indent=1, default=str)
        sys.exit("a validity gate failed — stop; read nothing below")
    nd = pd.concat([pd.read_csv(os.path.join(EV, f)).assign(file=f) for f in draws_files]); say(f"**Null check.** Every one of the {len(nd):,} draws kept the strategy's number of spells, switches and weeks in each state (measured, not assumed): {'yes' if nd['counts_ok'].all() else 'NO'}."); say()

    # ---------------------------------------------------------------- 4.1 headline
    say("### 4.1 Headline — reference setting (exit E1, window 52, 20.8 bps), the six index pairs"); say()
    rows = []
    for p in deciding:
        r = R.loc[p]; rep = bool(r["reportable"])
        c1 = rep and r["NA_pct"] >= PASS_PCT; c2 = rep and r["maxdd"] > r["NA_median_maxdd"]; c3 = rep and r["NB_pct"] >= PASS_PCT
        rows.append({"pair": p.replace("|", " ÷ "), "round trips": int(r["round_trips"]), "weeks": int(r["n_decisions"]), "reportable": "yes" if rep else "NO", "CAGR": r["cagr"], "N-A median": r["NA_median_cagr"], "edge": r["NA_edge"], "pct N-A": r["NA_pct"], "p": r["NA_p"], "C1": "PASS" if c1 else "FAIL",
                     "max dd": r["maxdd"], "N-A median dd": r["NA_median_maxdd"], "C2": "PASS" if c2 else "FAIL", "pct N-B": r["NB_pct"], "p N-B": r["NB_p"], "C3": "PASS" if c3 else "FAIL", "MDE (pp/yr)": r["NA_mde"] * 100})
    H = pd.DataFrame(rows); T["headline"] = H.to_dict("records")
    say(md(H, lambda c, v: pct(v) if c in ("CAGR", "N-A median", "edge", "max dd", "N-A median dd") else (p1(v) if c in ("pct N-A", "pct N-B", "MDE (pp/yr)") else (p3(v) if c in ("p", "p N-B") else str(v))))); say()
    nC1, nC2, nC3 = (H["C1"] == "PASS").sum(), (H["C2"] == "PASS").sum(), (H["C3"] == "PASS").sum()
    T["criteria"] = {"C1": int(nC1), "C2": int(nC2), "C3": int(nC3), "need": NEED, "H1": bool(nC1 >= NEED and nC2 >= NEED), "H2": bool(nC3 >= NEED)}
    say(f"C1 (CAGR ≥ 95th pct of N-A): {nC1} of 6 · C2 (drawdown shallower than N-A median): {nC2} of 6 · C3 (CAGR ≥ 95th pct of N-B): {nC3} of 6 · needed {NEED}. **H1 {'PASS' if T['criteria']['H1'] else 'FAIL'} · H2 {'PASS' if T['criteria']['H2'] else 'FAIL'}.**"); say()
    say("Gold pairs, same setting (reported, not counted):"); say()
    rows = [{"pair": p.replace("|", " ÷ "), "round trips": int(R.loc[p, "round_trips"]), "reportable": "yes" if R.loc[p, "reportable"] else "NO", "CAGR": R.loc[p, "cagr"], "pct N-A": R.loc[p, "NA_pct"], "max dd": R.loc[p, "maxdd"], "N-A median dd": R.loc[p, "NA_median_maxdd"], "pct N-B": R.loc[p, "NB_pct"]} for p in gold_pairs]
    G = pd.DataFrame(rows); T["gold"] = G.to_dict("records"); say(md(G, lambda c, v: pct(v) if c in ("CAGR", "max dd", "N-A median dd") else (p1(v) if c.startswith("pct") else str(v)))); say()

    # ---------------------------------------------------------------- 4.2 comparators
    say("### 4.2 Comparators (context, not pass marks) — reference setting"); say()
    rows = []
    for p in deciding + gold_pairs:
        a, b = p.split("|"); sub = summ[(summ["pair"] == p) & (summ["window"] == ref["window"]) & (summ["exit"] == ref["exit"]) & (summ["cost_bps"] == ref["cost_bps"])].set_index("series")
        rows.append({"pair": p.replace("|", " ÷ "), "window": f"{sub.loc['strategy', 'first_date']} → {sub.loc['strategy', 'last_date']}", "strategy": sub.loc["strategy", "cagr"], "hold " + a: sub.loc["hold_a", "cagr"], "hold " + b: sub.loc["hold_b", "cagr"], "50/50 buy-hold": sub.loc["5050_buyhold", "cagr"], "50/50 rebalanced": sub.loc["5050_rebal", "cagr"],
                     "dd strategy": sub.loc["strategy", "maxdd"], "dd 50/50 b-h": sub.loc["5050_buyhold", "maxdd"], "dd " + a: sub.loc["hold_a", "maxdd"], "dd " + b: sub.loc["hold_b", "maxdd"]})
    Cm = pd.DataFrame(rows); T["comparators"] = Cm.to_dict("records")
    for _, r in Cm.iterrows():
        say(f"- **{r['pair']}** ({r['window']}): strategy {pct(r['strategy'])} / {pct(r['dd strategy'])} · 50/50 buy-and-hold {pct(r['50/50 buy-hold'])} / {pct(r['dd 50/50 b-h'])} · 50/50 rebalanced {pct(r['50/50 rebalanced'])} · " + " · ".join(f"{k} {pct(v)} / {pct(r['dd ' + k[5:]])}" for k, v in r.items() if k.startswith("hold ")) + " (CAGR / max drawdown)")
    say()

    # ---------------------------------------------------------------- 4.3 evidence and MDE
    say("### 4.3 How much evidence, and the MDE"); say()
    say(f"Independent observations are round trips, not weeks: " + ", ".join(f"{p.replace('|', ' ÷ ')} {int(R.loc[p, 'round_trips'])} round trips over {int(R.loc[p, 'n_decisions'])} weeks" for p in deciding) + ".")
    say(); say("Next to every FAIL above, the smallest edge the test could have seen (2.8 × the null's sd of CAGR): " + ", ".join(f"{p.replace('|', ' ÷ ')} undetectable below {R.loc[p, 'NA_mde'] * 100:.1f} pp a year" for p in deciding) + ". A FAIL means 'too small to see', not 'nothing there'.")
    say(); say("Per-trade diagnostic (descriptive only): the held leg's fill-to-fill return minus the peer's over each hold, reference setting:"); say()
    rows = []
    for p in deciding + gold_pairs:
        e = extras[(extras["pair"] == p) & (extras["window"] == ref["window"]) & (extras["exit"] == ref["exit"]) & (~extras["open"])]
        rows.append({"pair": p.replace("|", " ÷ "), "trades": len(e), "held beat peer": float((e["extra"] > 0).mean()) if len(e) else np.nan, "mean extra": e["extra"].mean() if len(e) else np.nan, "median extra": e["extra"].median() if len(e) else np.nan, "sum": e["extra"].sum() if len(e) else np.nan, "worst": e["extra"].min() if len(e) else np.nan, "best": e["extra"].max() if len(e) else np.nan, "touches ignored": int(R.loc[p, "touches_ignored"])})
    X = pd.DataFrame(rows); T["extras"] = X.to_dict("records"); say(md(X, lambda c, v: pct(v) if c in ("held beat peer", "mean extra", "median extra", "sum", "worst", "best") else str(v))); say()

    # ---------------------------------------------------------------- 4.4 robustness
    say("### 4.4 Robustness"); say()
    say("**R1 — costs** (reference window and exit; the decisions are the same, the numbers and the null change). Percentile of the strategy's CAGR on N-A:"); say()
    rows = []
    for p in deciding:
        row = {"pair": p.replace("|", " ÷ ")}
        for cst in audit["costs_bps"]:
            s_ = strat[(strat["pair"] == p) & (strat["window"] == ref["window"]) & (strat["exit"] == ref["exit"]) & (strat["cost_bps"] == cst)]
            row[f"CAGR @ {cst:g} bps"] = float(s_["cagr"].iloc[0]) if len(s_) else np.nan; row[f"pct @ {cst:g}"] = float(s_["NA_pct"].iloc[0]) if len(s_) and "NA_pct" in s_ else np.nan
        rows.append(row)
    R1 = pd.DataFrame(rows); T["R1"] = R1.to_dict("records"); say(md(R1, lambda c, v: pct(v) if c.startswith("CAGR") else (p1(v) if c.startswith("pct") else str(v)))); say()
    say("**R2 / R3 — windows and exits** (20.8 bps). Percentile of CAGR on N-A, with the number of round trips; a cell with fewer than 4 round trips is not reportable (—):"); say()
    rows = []
    for p in deciding:
        row = {"pair": p.replace("|", " ÷ ")}
        for ex in audit["exits"]:
            for wdw in audit["windows"]:
                s_ = strat[(strat["pair"] == p) & (strat["window"] == wdw) & (strat["exit"] == ex) & (strat["cost_bps"] == ref["cost_bps"])]
                if len(s_) and bool(s_["reportable"].iloc[0]):
                    row[f"{ex} W{wdw}"] = f"{s_['NA_pct'].iloc[0]:.1f} ({int(s_['round_trips'].iloc[0])} rt, dd {'shallower' if s_['maxdd'].iloc[0] > s_['NA_median_maxdd'].iloc[0] else 'deeper'})"
                else:
                    row[f"{ex} W{wdw}"] = "—"
        rows.append(row)
    R2 = pd.DataFrame(rows); T["R2R3"] = R2.to_dict("records"); say(md(R2)); say()
    e2 = strat[(strat["window"] == ref["window"]) & (strat["exit"] == "E2") & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair")
    nC1e2 = sum(1 for p in deciding if e2.loc[p, "reportable"] and e2.loc[p, "NA_pct"] >= PASS_PCT); nC2e2 = sum(1 for p in deciding if e2.loc[p, "reportable"] and e2.loc[p, "maxdd"] > e2.loc[p, "NA_median_maxdd"]); nC3e2 = sum(1 for p in deciding if e2.loc[p, "reportable"] and e2.loc[p, "NB_pct"] >= PASS_PCT)
    T["E2"] = {"C1": int(nC1e2), "C2": int(nC2e2), "C3": int(nC3e2)}; say(f"Exit E2 on the pre-named criteria: C1 {nC1e2} of 6, C2 {nC2e2} of 6, C3 {nC3e2} of 6 (E1 is the deciding exit)."); say()
    say("**R4 — sub-periods** (descriptive; reference setting; CAGR of the strategy vs 50/50 buy-and-hold, and max drawdown):"); say()
    rows = []
    for p in deciding:
        nav = pd.read_csv(os.path.join(EV, f"nav_{p.replace('|', '_')}_{ref['window']}_{ref['exit']}_{ref['cost_bps']:g}.csv"), index_col=0, parse_dates=True)
        row = {"pair": p.replace("|", " ÷ ")}
        for name, lo, hi in PERIODS:
            s_ = nav.loc[lo:hi].dropna(subset=["nav"])
            if len(s_) < 120:
                row[name] = "—"; continue
            days = (s_.index[-1] - s_.index[0]).days
            cg = (s_["nav"].iloc[-1] / s_["nav"].iloc[0]) ** (365.25 / days) - 1; cb = (s_["5050_buyhold"].iloc[-1] / s_["5050_buyhold"].iloc[0]) ** (365.25 / days) - 1
            dd = (s_["nav"] / s_["nav"].cummax() - 1).min(); ddb = (s_["5050_buyhold"] / s_["5050_buyhold"].cummax() - 1).min()
            row[name] = f"{cg:.1%} vs {cb:.1%} · dd {dd:.0%} vs {ddb:.0%}"
        rows.append(row)
    R4 = pd.DataFrame(rows); T["R4"] = R4.to_dict("records"); say(md(R4)); say()
    say("**R5 — gold pairs**: in 4.1 above."); say()

    # ---------------------------------------------------------------- 4.5 multiplicity
    say("### 4.5 Multiplicity"); say()
    n_settings = len(audit["windows"]) * len(audit["exits"]); rows = []
    for p in deciding:
        s_ = strat[(strat["pair"] == p) & (strat["cost_bps"] == ref["cost_bps"]) & strat["reportable"]]
        if not len(s_):
            continue
        b = s_.loc[s_["NA_pct"].idxmax()]
        rows.append({"pair": p.replace("|", " ÷ "), "settings tried": n_settings, "best setting": f"{b['exit']} W{int(b['window'])}", "its pct N-A": b["NA_pct"], "passes 98th": "yes" if b["NA_pct"] >= BEST_PCT else "no"})
    M = pd.DataFrame(rows); T["multiplicity"] = M.to_dict("records"); say(f"{n_settings} decision settings per pair (2 exits × 3 windows), at 3 cost levels; the reference was fixed by position before the run. The best-looking setting per pair, judged at the 98th percentile of its own null:"); say(); say(md(M, lambda c, v: p1(v) if c == "its pct N-A" else str(v))); say()

    # ---------------------------------------------------------------- 4.6 verdict
    say("### 4.6 Verdict table"); say()
    say("| criterion | needed | got | result |"); say("|---|---|---|---|")
    say(f"| C1 CAGR ≥ 95th pct of N-A (E1, W52, 20.8 bps) | {NEED} of 6 | {nC1} | {'PASS' if nC1 >= NEED else 'FAIL'} |"); say(f"| C2 drawdown shallower than N-A median | {NEED} of 6 | {nC2} | {'PASS' if nC2 >= NEED else 'FAIL'} |"); say(f"| C3 CAGR ≥ 95th pct of N-B | {NEED} of 6 | {nC3} | {'PASS' if nC3 >= NEED else 'FAIL'} |")
    say(f"| **H1 timing** (C1 and C2) | | | **{'PASS' if T['criteria']['H1'] else 'FAIL'}** |"); say(f"| **H2 leg choice** (C3) | | | **{'PASS' if T['criteria']['H2'] else 'FAIL'}** |"); say()
    open(os.path.join(EV, "analysis.log"), "w").write("\n".join(L)); json.dump(T, open(os.path.join(EV, "tables.json"), "w"), indent=1, default=str)
    print("\n".join(L)); print(f"\nwrote {EV}/analysis.log and tables.json")


if __name__ == "__main__":
    main()
