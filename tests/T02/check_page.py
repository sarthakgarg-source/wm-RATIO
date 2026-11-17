"""Check J1: every number embedded in the results page (<script id="numbers">) is recomputed here from the evidence files
directly — summary.csv, the draws files, decisions.csv, trades.csv, trade_extras.csv, and T03's evidence/headline/ for
gate V4b (the same rule under the other test number) — without analyse.py or tables.json.
    python tests/T02/check_page.py [run-name]        # exit 1 on any mismatch"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = sys.argv[1] if len(sys.argv) > 1 else "headline"; EV = os.path.join(HERE, "evidence", RUN)
OTHER = "T03"; OTHER_EV = os.path.join(HERE, "..", OTHER, "evidence", "headline")
NEED, PASS_PCT, BEST_PCT = 4, 95.0, 98.0


def main():
    page = open(os.path.join(EV, "T02_results.html"), encoding="utf-8").read()
    NUM = json.loads(re.search(r"<script id='numbers' type='application/json'>(.*?)</script>", page, flags=re.S).group(1))
    audit = json.load(open(os.path.join(EV, "audit.json"), encoding="utf-8")); ref = audit["reference"]; deciding = audit["deciding_pairs"]; gold = [p for p in audit["pairs"] if p not in deciding]
    summ = pd.read_csv(os.path.join(EV, "summary.csv")); strat = summ[summ["series"] == "strategy"]
    R = strat[(strat["window"] == ref["window"]) & (strat["exit"] == ref["exit"]) & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair")
    decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]); trades = pd.read_csv(os.path.join(EV, "trades.csv"), parse_dates=["decision_date", "trade_date"]); extras = pd.read_csv(os.path.join(EV, "trade_extras.csv"))
    exp = {}
    def draws(p, window=ref["window"], ex=ref["exit"], cost=ref["cost_bps"]):
        return pd.read_csv(os.path.join(EV, f"draws_{p.replace('|', '_')}_{window}_{ex}_{cost:g}.csv"))
    def pct_of(x, g):  # percentile and p, recomputed
        worse = (g < x).sum(); tie = (g == x).sum(); return (worse + 0.5 * tie) / len(g) * 100, (1 + (g >= x).sum()) / (len(g) + 1)
    exp["run_date"] = audit["run_date"]; exp["draws"] = audit["draws_per_null"]
    c1 = c2 = c3 = 0
    for p in deciding:
        k = p.replace("|", "_"); r = R.loc[p]; d = draws(p); gA = d[d["null"] == "N-A"]; gB = d[d["null"] == "N-B"]
        pa, ppa = pct_of(r["cagr"], gA["cagr"]); pb, _ = pct_of(r["cagr"], gB["cagr"]); medA = gA["cagr"].median(); meddd = gA["maxdd"].median()
        rep = r["round_trips"] >= 4 and r["n_decisions"] >= 200
        exp[f"rt_{k}"] = r["round_trips"]; exp[f"weeks_{k}"] = r["n_decisions"]; exp[f"cagr_{k}"] = r["cagr"]; exp[f"named_{k}"] = medA; exp[f"edge_{k}"] = r["cagr"] - medA; exp[f"napct_{k}"] = pa; exp[f"nap_{k}"] = ppa
        exp[f"maxdd_{k}"] = r["maxdd"]; exp[f"nadd_{k}"] = meddd; exp[f"nbpct_{k}"] = pb; exp[f"mde_{k}"] = 2.8 * gA["cagr"].std(ddof=1) * 100
        c1 += rep and pa >= PASS_PCT; c2 += rep and r["maxdd"] > meddd; c3 += rep and pb >= PASS_PCT
        sub = summ[(summ["pair"] == p) & (summ["window"] == ref["window"]) & (summ["exit"] == ref["exit"]) & (summ["cost_bps"] == ref["cost_bps"])].set_index("series")
        exp[f"cmp_strat_{k}"] = sub.loc["strategy", "cagr"]; exp[f"cmp_bh_{k}"] = sub.loc["5050_buyhold", "cagr"]
        s_ = strat[(strat["pair"] == p) & (strat["cost_bps"] == ref["cost_bps"]) & strat["reportable"]]
        if len(s_):
            exp[f"best_{k}"] = max(pct_of(row["cagr"], draws(p, int(row["window"]), row["exit"]).query("null == 'N-A'")["cagr"])[0] for _, row in s_.iterrows())
    exp["C1"], exp["C2"], exp["C3"] = c1, c2, c3; exp["H1"] = "PASS" if (c1 >= NEED and c2 >= NEED) else "FAIL"; exp["H2"] = "PASS" if c3 >= NEED else "FAIL"
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
    if audit.get("ratio") != "plain" or audit.get("max_hold_weeks") is not None:
        exp["V1"] = "FAIL"
    exp["V3"] = "PASS" if audit["null_counts"]["draws_mismatched"] == 0 else "FAIL"; an = summ[summ["series"] == "always_neutral"]["final_nav"].to_numpy(); bh = summ[summ["series"] == "5050_buyhold"]["final_nav"].to_numpy()
    # V4b recomputed: the other test number's evidence, merged on the decision week and on the pair-setting-cost
    oa = pd.read_csv(os.path.join(OTHER_EV, "summary.csv")); oa = oa[oa["series"] == "strategy"].set_index(["pair", "window", "exit", "cost_bps"])
    me = strat.set_index(["pair", "window", "exit", "cost_bps"]); both = me.join(oa, rsuffix="_o", how="inner")
    ok_rows = (np.abs(both["final_nav"] - both["final_nav_o"]) <= 1e-9) & (np.abs(both["cagr"] - both["cagr_o"]) <= 1e-9) & (np.abs(both["maxdd"] - both["maxdd_o"]) <= 1e-9)
    od = pd.read_csv(os.path.join(OTHER_EV, "decisions.csv"), parse_dates=["date"]).set_index(["pair", "window", "exit", "date"])["state_after"]
    md_ = decs.set_index(["pair", "window", "exit", "date"])["state_after"]; dec_same = len(od) == len(md_) and bool((od.reindex(md_.index).astype(str) == md_.astype(str)).all())
    exp["v4_rows_ok"] = int(ok_rows.sum()); exp["v4_rows_all"] = len(me); exp["v4_dec"] = "identical" if dec_same else "DIFFERENT"
    exp["V4"] = "PASS" if (np.allclose(an, bh, atol=1e-9) and len(both) == len(me) == len(oa) > 0 and bool(ok_rows.all()) and dec_same) else "FAIL"
    exp["pit_checks"] = audit["pit_checks"]; exp["pit_violations"] = len(audit["pit_violations"]); exp["null_checked"] = audit["null_counts"]["draws_checked"]; exp["null_mismatched"] = audit["null_counts"]["draws_mismatched"]
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
