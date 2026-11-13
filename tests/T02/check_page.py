"""Check J1: every number embedded in the results page (<script id="numbers">) is recomputed here from the evidence files
directly — summary.csv, the draws files, decisions.csv, trades.csv, trade_extras.csv, t01_regression/summary.csv, and
T01's evidence/headline/ for the side-by-side — without analyse.py or tables.json.
    python tests/T02/check_page.py [run-name]        # exit 1 on any mismatch"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = sys.argv[1] if len(sys.argv) > 1 else "headline"; EV = os.path.join(HERE, "evidence", RUN)
T01_EV = os.path.join(HERE, "..", "T01", "evidence", "headline")
NEED, PASS_PCT, BEST_PCT, PASS_PCT_2F = 4, 95.0, 98.0, 97.5


def main():
    page = open(os.path.join(EV, "T02_results.html"), encoding="utf-8").read()
    NUM = json.loads(re.search(r"<script id='numbers' type='application/json'>(.*?)</script>", page, flags=re.S).group(1))
    audit = json.load(open(os.path.join(EV, "audit.json"))); ref = audit["reference"]; deciding = audit["deciding_pairs"]; gold = [p for p in audit["pairs"] if p not in deciding]
    summ = pd.read_csv(os.path.join(EV, "summary.csv")); strat = summ[summ["series"] == "strategy"]
    R = strat[(strat["window"] == ref["window"]) & (strat["exit"] == ref["exit"]) & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair")
    decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]); trades = pd.read_csv(os.path.join(EV, "trades.csv"), parse_dates=["decision_date", "trade_date"]); extras = pd.read_csv(os.path.join(EV, "trade_extras.csv"))
    exp = {}
    def draws(p, window=ref["window"], ex=ref["exit"], cost=ref["cost_bps"]):
        return pd.read_csv(os.path.join(EV, f"draws_{p.replace('|', '_')}_{window}_{ex}_{cost:g}.csv"))
    def pct_of(x, g):  # percentile and p, recomputed
        worse = (g < x).sum(); tie = (g == x).sum(); return (worse + 0.5 * tie) / len(g) * 100, (1 + (g >= x).sum()) / (len(g) + 1)
    exp["run_date"] = audit["run_date"]; exp["draws"] = audit["draws_per_null"]
    c1 = c2 = c3 = c1b = 0
    for p in deciding:
        k = p.replace("|", "_"); r = R.loc[p]; d = draws(p); gA = d[d["null"] == "N-A"]; gB = d[d["null"] == "N-B"]
        pa, ppa = pct_of(r["cagr"], gA["cagr"]); pb, _ = pct_of(r["cagr"], gB["cagr"]); medA = gA["cagr"].median(); meddd = gA["maxdd"].median()
        rep = r["round_trips"] >= 4 and r["n_decisions"] >= 200
        exp[f"rt_{k}"] = r["round_trips"]; exp[f"weeks_{k}"] = r["n_decisions"]; exp[f"cagr_{k}"] = r["cagr"]; exp[f"named_{k}"] = medA; exp[f"edge_{k}"] = r["cagr"] - medA; exp[f"napct_{k}"] = pa; exp[f"nap_{k}"] = ppa
        exp[f"maxdd_{k}"] = r["maxdd"]; exp[f"nadd_{k}"] = meddd; exp[f"nbpct_{k}"] = pb; exp[f"mde_{k}"] = 2.8 * gA["cagr"].std(ddof=1) * 100
        c1 += rep and pa >= PASS_PCT; c2 += rep and r["maxdd"] > meddd; c3 += rep and pb >= PASS_PCT; c1b += rep and pa >= PASS_PCT_2F
        sub = summ[(summ["pair"] == p) & (summ["window"] == ref["window"]) & (summ["exit"] == ref["exit"]) & (summ["cost_bps"] == ref["cost_bps"])].set_index("series")
        exp[f"cmp_strat_{k}"] = sub.loc["strategy", "cagr"]; exp[f"cmp_bh_{k}"] = sub.loc["5050_buyhold", "cagr"]
        s_ = strat[(strat["pair"] == p) & (strat["cost_bps"] == ref["cost_bps"]) & strat["reportable"]]
        if len(s_):
            exp[f"best_{k}"] = max(pct_of(row["cagr"], draws(p, int(row["window"]), row["exit"]).query("null == 'N-A'")["cagr"])[0] for _, row in s_.iterrows())
    exp["C1"], exp["C2"], exp["C3"] = c1, c2, c3; exp["H1"] = "PASS" if (c1 >= NEED and c2 >= NEED) else "FAIL"; exp["H2"] = "PASS" if c3 >= NEED else "FAIL"; exp["C1_two_forms"] = c1b
    for p in gold:
        k = p.replace("|", "_"); r = R.loc[p]; d = draws(p); gA = d[d["null"] == "N-A"]; exp[f"gcagr_{k}"] = r["cagr"]; exp[f"gnapct_{k}"] = pct_of(r["cagr"], gA["cagr"])[0]
        sub = summ[(summ["pair"] == p) & (summ["window"] == ref["window"]) & (summ["exit"] == ref["exit"]) & (summ["cost_bps"] == ref["cost_bps"])].set_index("series")
        exp[f"cmp_strat_{k}"] = sub.loc["strategy", "cagr"]; exp[f"cmp_bh_{k}"] = sub.loc["5050_buyhold", "cagr"]
    e2 = strat[(strat["window"] == ref["window"]) & (strat["exit"] == "E2") & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair"); a1 = a2 = a3 = 0
    for p in deciding:
        d = draws(p, ref["window"], "E2"); gA = d[d["null"] == "N-A"]; gB = d[d["null"] == "N-B"]; r = e2.loc[p]; rep = r["round_trips"] >= 4 and r["n_decisions"] >= 200
        a1 += rep and pct_of(r["cagr"], gA["cagr"])[0] >= PASS_PCT; a2 += rep and r["maxdd"] > gA["maxdd"].median(); a3 += rep and pct_of(r["cagr"], gB["cagr"])[0] >= PASS_PCT
    exp["E2_C1"], exp["E2_C2"], exp["E2_C3"] = a1, a2, a3
    # the worked example, from decisions / trades / extras
    p0 = "N100|SC100"; d0 = decs[(decs["pair"] == p0) & (decs["window"] == ref["window"]) & (decs["exit"] == ref["exit"])]; fe = d0[d0["reason"].str.startswith("enter", na=False)].iloc[0]
    tr0 = trades[(trades["pair"] == p0) & (trades["window"] == ref["window"]) & (trades["exit"] == ref["exit"])]; te = tr0[tr0["decision_date"] == fe["date"]].iloc[0]; tx = tr0[tr0["decision_date"] > fe["date"]].iloc[0]
    ex0 = extras[(extras["pair"] == p0) & (extras["window"] == ref["window"]) & (extras["exit"] == ref["exit"])].iloc[0]; r0 = R.loc[p0]; dA0 = draws(p0).query("null == 'N-A'")
    exp.update({"w_entry_decision": fe["date"].date().isoformat(), "w_ratio": fe["ratio"], "w_mean": fe["mean"], "w_sd": fe["sd"], "w_state": fe["state_after"], "w_entry_trade": te["trade_date"].date().isoformat(), "w_fill_a": te["fill_a"], "w_fill_b": te["fill_b"], "w_cost": te["cost"],
                "w_exit_decision": tx["decision_date"].date().isoformat(), "w_exit_trade": tx["trade_date"].date().isoformat(), "w_held_ret": ex0["held_ret"], "w_peer_ret": ex0["peer_ret"], "w_extra": ex0["extra"], "w_first_date": r0["first_date"], "w_last_date": r0["last_date"], "w_final_nav": r0["final_nav"], "w_cagr": r0["cagr"], "w_maxdd": r0["maxdd"],
                "w_draw0_spells": dA0.iloc[0]["spells"], "w_draw0_cagr": dA0.iloc[0]["cagr"], "w_NA_pct": pct_of(r0["cagr"], dA0["cagr"])[0], "w_NA_p": pct_of(r0["cagr"], dA0["cagr"])[1], "w_NA_median": dA0["cagr"].median(), "w_NA_edge": r0["cagr"] - dA0["cagr"].median()})
    # gates, from the audit and the verify log
    vlog = open(os.path.join(EV, "verify.log"), encoding="utf-8").read(); m = re.search(r"verify: (\d+) nav files, (\d+) comparisons, (\d+) failures", vlog)
    exp["V1"] = "PASS" if (m and int(m.group(3)) == 0 and "NOT CAUGHT" not in vlog and "caught" in vlog) else "FAIL"; exp["V2"] = "PASS" if (audit["pit_checks"] > 0 and not audit["pit_violations"]) else "FAIL"
    if audit.get("ratio") != "plain":
        exp["V1"] = "FAIL"
    exp["V3"] = "PASS" if audit["null_counts"]["draws_mismatched"] == 0 else "FAIL"; an = summ[summ["series"] == "always_neutral"]["final_nav"].to_numpy(); bh = summ[summ["series"] == "5050_buyhold"]["final_nav"].to_numpy()
    # V4b recomputed: the changed engine under T01's form against T01's headline summary
    rs = pd.read_csv(os.path.join(EV, "t01_regression", "summary.csv")); bs = pd.read_csv(os.path.join(T01_EV, "summary.csv")); mg = bs.merge(rs, on=["pair", "window", "exit", "cost_bps", "series"], suffixes=("_t01", ""))
    ok_rows = np.ones(len(mg), dtype=bool)
    for c in ("final_nav", "cagr", "maxdd"):
        ok_rows &= np.isclose(mg[c + "_t01"].to_numpy(), mg[c].to_numpy(), rtol=0, atol=1e-9)
    exp["v4_rows_ok"] = int(ok_rows.sum()); exp["v4_rows_all"] = len(bs)
    exp["V4"] = "PASS" if (np.allclose(an, bh, atol=1e-9) and len(mg) == len(bs) > 0 and ok_rows.all()) else "FAIL"
    exp["pit_checks"] = audit["pit_checks"]; exp["pit_violations"] = len(audit["pit_violations"]); exp["null_checked"] = audit["null_counts"]["draws_checked"]; exp["null_mismatched"] = audit["null_counts"]["draws_mismatched"]
    # T02 against T01, from the two tests' evidence files directly
    t1s = pd.read_csv(os.path.join(T01_EV, "summary.csv")); t1d = pd.read_csv(os.path.join(T01_EV, "decisions.csv"), parse_dates=["date"])
    t1R = t1s[(t1s["series"] == "strategy") & (t1s["window"] == ref["window"]) & (t1s["exit"] == ref["exit"]) & (t1s["cost_bps"] == ref["cost_bps"])].set_index("pair")
    t1draws = lambda p: pd.read_csv(os.path.join(T01_EV, f"draws_{p.replace('|', '_')}_{ref['window']}_{ref['exit']}_{ref['cost_bps']:g}.csv")).query("null == 'N-A'")["cagr"]
    for p in deciding + gold:
        k = p.replace("|", "_")
        a2 = decs[(decs["pair"] == p) & (decs["window"] == ref["window"]) & (decs["exit"] == ref["exit"])].set_index("date")
        a1 = t1d[(t1d["pair"] == p) & (t1d["window"] == ref["window"]) & (t1d["exit"] == ref["exit"])].set_index("date"); idx = a2.index.intersection(a1.index)
        e2 = set(zip(a2.index[a2["reason"].fillna("").str.contains("enter")], a2["state_after"][a2["reason"].fillna("").str.contains("enter")]))
        e1 = set(zip(a1.index[a1["reason"].fillna("").str.contains("enter")], a1["state_after"][a1["reason"].fillna("").str.contains("enter")]))
        exp[f"vs_weeks_{k}"] = len(idx); exp[f"vs_differ_{k}"] = int((a2.loc[idx, "state_after"] != a1.loc[idx, "state_after"]).sum()); exp[f"vs_same_{k}"] = len(e1 & e2); exp[f"vs_only1_{k}"] = len(e1 - e2); exp[f"vs_only2_{k}"] = len(e2 - e1)
        exp[f"vs_rt1_{k}"] = t1R.loc[p, "round_trips"]; exp[f"vs_rt2_{k}"] = R.loc[p, "round_trips"]; exp[f"vs_cagr1_{k}"] = t1R.loc[p, "cagr"]; exp[f"vs_cagr2_{k}"] = R.loc[p, "cagr"]
        exp[f"vs_pct1_{k}"] = pct_of(t1R.loc[p, "cagr"], t1draws(p))[0]; exp[f"vs_pct2_{k}"] = pct_of(R.loc[p, "cagr"], draws(p).query("null == 'N-A'")["cagr"])[0]
    # compare
    fails = []; checked = 0
    for key, val in NUM.items():
        if key not in exp:
            fails.append(f"{key}: on the page but not recomputed"); continue
        e = exp[key]; checked += 1
        if isinstance(val, str) or isinstance(e, str):
            ok = str(val) == str(e)
        else:
            ok = abs(float(val) - float(e)) <= 1e-6 * max(1.0, abs(float(e)))
        if not ok:
            fails.append(f"{key}: page {val} vs evidence {e}")
    for x in fails:
        print("FAIL", x)
    print(f"check_page: {checked} numbers checked, {len(fails)} failures")
    sys.exit(1 if fails or checked == 0 else 0)


if __name__ == "__main__":
    main()
