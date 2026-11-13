"""Step 11: the results page, built from evidence/<run>/ only. Every number on the page is also written into an embedded JSON
(<script id="numbers">) with a key that names its source, so check_page.py can recompute each one from the evidence.
    python tests/T02/report.py [run-name]      → evidence/<run>/T02_results.html and tests/T02/T02_results.html"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = sys.argv[1] if len(sys.argv) > 1 else "headline"; EV = os.path.join(HERE, "evidence", RUN)
INK, INK2, MUTED, GRID, SURF, BLUE, RED, GOLD_C = "#1b1a17", "#6b6962", "#8a8880", "#e7e5df", "#f7f6f2", "#2a78d6", "#e34948", "#b8860b"
FONT = "Inter, Helvetica, Arial, sans-serif"
NUM = {}


def n(key, val):
    NUM[key] = float(val) if isinstance(val, (int, float, np.floating, np.integer)) and not isinstance(val, bool) else val
    return val


def fmt_pct(v, d=1): return f"{v:.{d}%}"


def layout(fig, title, h, sub=None):
    fig.update_layout(template="plotly_white", font={"family": FONT, "size": 12, "color": INK}, height=h, margin={"l": 60, "r": 30, "t": 90, "b": 60}, paper_bgcolor="white", plot_bgcolor="white",
                      title={"text": f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{MUTED}'>{sub}</span>" if sub else ""), "x": 0, "xanchor": "left", "font": {"size": 15}}, legend={"orientation": "h", "y": -0.12, "x": 0, "font": {"size": 11}})
    fig.update_xaxes(gridcolor=GRID, zeroline=False); fig.update_yaxes(gridcolor=GRID, zeroline=False); return fig


def main():
    T = json.load(open(os.path.join(EV, "tables.json"))); audit = json.load(open(os.path.join(EV, "audit.json"))); ref = audit["reference"]
    summ = pd.read_csv(os.path.join(EV, "summary.csv")); decs = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"])
    deciding = audit["deciding_pairs"]; gold_pairs = [p for p in audit["pairs"] if p not in deciding]; strat = summ[summ["series"] == "strategy"]
    R = strat[(strat["window"] == ref["window"]) & (strat["exit"] == ref["exit"]) & (strat["cost_bps"] == ref["cost_bps"])].set_index("pair")
    cr = T["criteria"]; w = T["worked"]
    # ------------------------------------------------------------------ charts
    # 1. the band chart for the worked example
    d0 = decs[(decs["pair"] == w["pair"]) & (decs["window"] == ref["window"]) & (decs["exit"] == ref["exit"])].set_index("date")
    f1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05, subplot_titles=(f"{w['pair'].replace('|', ' ÷ ')} (the plain ratio, not logged) with its trailing-{ref['window']}-week mean ± 1 sd — entries and exits of exit {ref['exit']}", "the sleeve's state (1 = all in the first leg, 0 = all in the second, 0.5 = 50/50)"))
    f1.add_trace(go.Scatter(x=d0.index, y=d0["ratio"], mode="lines", line={"color": INK, "width": 1.2}, name="ratio"), row=1, col=1)
    f1.add_trace(go.Scatter(x=d0.index, y=d0["mean"], mode="lines", line={"color": INK2, "width": 1, "dash": "dash"}, name="trailing mean"), row=1, col=1)
    f1.add_trace(go.Scatter(x=d0.index, y=d0["mean"] + d0["sd"], mode="lines", line={"color": RED, "width": 0.9, "dash": "dot"}, name="mean ± 1 sd"), row=1, col=1)
    f1.add_trace(go.Scatter(x=d0.index, y=d0["mean"] - d0["sd"], mode="lines", line={"color": RED, "width": 0.9, "dash": "dot"}, showlegend=False), row=1, col=1)
    ent = d0[d0["reason"].str.contains("enter", na=False)]; exi = d0[d0["reason"].str.startswith("exit", na=False)]
    f1.add_trace(go.Scatter(x=ent.index, y=ent["ratio"], mode="markers", marker={"symbol": "triangle-up", "size": 9, "color": BLUE, "line": {"color": INK, "width": 0.5}}, name="entry (first touch)"), row=1, col=1)
    f1.add_trace(go.Scatter(x=exi.index, y=exi["ratio"], mode="markers", marker={"symbol": "x", "size": 8, "color": RED}, name="exit"), row=1, col=1)
    st_num = d0["state_after"].map({"A": 1.0, "B": 0.0, "N": 0.5})
    f1.add_trace(go.Scatter(x=d0.index, y=st_num, mode="lines", line={"color": BLUE, "width": 1, "shape": "hv"}, name="state"), row=2, col=1)
    f1.add_vline(x=pd.Timestamp(w["entry_decision"]), line={"color": GOLD_C, "width": 2}, row=1, col=1)
    layout(f1, f"The worked example: {w['pair'].replace('|', ' ÷ ')}, {w['first_date']} → {w['last_date']} — the gold line is the decision of {w['entry_decision']} walked through below", 620, "Decisions on the last trading day of each ISO week; the band is the trailing 52 decision weeks ending that day")
    # 2. NAV charts, deciding pairs, menu
    f2 = go.Figure(); per = 0
    for p in deciding:
        nav = pd.read_csv(os.path.join(EV, f"nav_{p.replace('|', '_')}_{ref['window']}_{ref['exit']}_{ref['cost_bps']:g}.csv"), index_col=0, parse_dates=True); a, b = p.split("|"); k0 = len(f2.data)
        for col, nm, c_, wd in (("nav", "the switch (net of costs)", INK, 1.8), ("5050_buyhold", "50/50 bought once", MUTED, 1.3), ("5050_rebal", "50/50 rebalanced weekly", "#9a9890", 1.0), ("hold_a", a, BLUE, 1.0), ("hold_b", b, RED, 1.0)):
            f2.add_trace(go.Scatter(x=nav.index, y=nav[col], mode="lines", line={"color": c_, "width": wd, "dash": "dot" if col in ("hold_a", "hold_b") else "solid"}, name=nm, visible=(p == deciding[0])))
        per = len(f2.data) - k0
    f2.update_layout(yaxis={"type": "log", "title": "growth of 1"}, updatemenus=[{"type": "dropdown", "x": 1, "xanchor": "right", "y": 1.12, "yanchor": "top", "bgcolor": SURF, "bordercolor": GRID, "font": {"size": 11}, "buttons": [{"label": p.replace("|", " ÷ "), "method": "update", "args": [{"visible": [(i // per) == k for i in range(per * len(deciding))]}]} for k, p in enumerate(deciding)]}])
    layout(f2, f"The sleeve against its comparators — exit {ref['exit']}, window {ref['window']}, {ref['cost_bps']} bps; dates on the axis", 520, "NAV 1 on the first banded decision day; the first trade (50/50) on the next session is inside the path")
    # 3. null histograms, menu
    f3 = go.Figure(); per3 = 0
    for p in deciding:
        d = pd.read_csv(os.path.join(EV, f"draws_{p.replace('|', '_')}_{ref['window']}_{ref['exit']}_{ref['cost_bps']:g}.csv")); k0 = len(f3.data)
        for nm, c_ in (("N-A", MUTED), ("N-B", "#e39a6b")):
            g = d[d["null"] == nm]; f3.add_trace(go.Histogram(x=g["cagr"], nbinsx=60, marker_color=c_, opacity=0.55, name=f"{nm}: CAGR of {len(g):,} random draws", visible=(p == deciding[0])))
        cg = float(R.loc[p, "cagr"]); f3.add_trace(go.Scatter(x=[cg, cg], y=[0, 1], mode="lines", yaxis="y2", line={"color": INK, "width": 2.5}, name=f"the strategy: {cg:.1%} (pct N-A {R.loc[p, 'NA_pct']:.1f}, N-B {R.loc[p, 'NB_pct']:.1f})", visible=(p == deciding[0])))
        per3 = len(f3.data) - k0
    f3.update_layout(barmode="overlay", xaxis={"tickformat": ".1%", "title": "after-cost CAGR"}, yaxis2={"overlaying": "y", "range": [0, 1], "visible": False}, updatemenus=[{"type": "dropdown", "x": 1, "xanchor": "right", "y": 1.12, "yanchor": "top", "bgcolor": SURF, "bordercolor": GRID, "font": {"size": 11}, "buttons": [{"label": p.replace("|", " ÷ "), "method": "update", "args": [{"visible": [(i // per3) == k for i in range(per3 * len(deciding))]}]} for k, p in enumerate(deciding)]}])
    layout(f3, f"Where the strategy sits among {T['draws']:,} random timings (N-A) and {T['draws']:,} random leg choices (N-B) — reference setting", 460, "N-A keeps the strategy's trade spells (blocks where they abut) and gaps and shuffles their order; N-B keeps the timing and flips the legs by coin")
    # ------------------------------------------------------------------ page
    H = []
    H.append("<!doctype html><html><head><meta charset='utf-8'><title>T02 results — the ±1 sd band on the plain ratio</title>")
    H.append(f"<style>body{{font-family:{FONT};color:{INK};max-width:1300px;margin:24px auto;padding:0 16px;background:#fff}} h1{{font-size:22px}} h2{{font-size:17px;margin-top:34px;border-left:4px solid {INK};padding-left:10px}} p,li{{font-size:14px;line-height:1.55}} .box{{background:{SURF};border-left:3px solid {INK};padding:10px 14px;margin:12px 0}} .verdict{{font-size:16px;font-weight:600}} table{{border-collapse:collapse;font-size:12.5px;margin:10px 0}} th,td{{border:1px solid {GRID};padding:4px 8px;text-align:right;vertical-align:top}} th:first-child,td:first-child{{text-align:left}} td.pass{{background:#e3efe0}} td.fail{{background:#fde3e2}} .sm{{font-size:12px;color:{INK2}}} code{{font-size:12px}}</style></head><body>")
    H.append("<h1>T02 · With the band on the plain price ratio P_a ÷ P_b (not its logarithm), does holding the lagging leg from the first touch of ±1 sd until the ratio is back at its mean beat random timing?</h1>")
    H.append(f"<p class='sm'>Run <code>{RUN}</code> on {n('run_date', T['run_date'])} · {n('draws', T['draws']):,} draws per null · band on {T['gates']['ratio_formula']} · reference setting exit {ref['exit']}, window {ref['window']} weeks, {ref['cost_bps']} bps round trip, max hold {audit['max_hold_weeks']} weeks · the design (sections 1–3 of T02.md) was locked before the run · T01 is the same test on ln(P_a ÷ P_b).</p>")
    # the answer
    h1, h2 = cr["H1"], cr["H2"]
    H.append(f"<div class='box'><div class='verdict'>H1 (timing): {n('H1', 'PASS' if h1 else 'FAIL')} — C1 {n('C1', cr['C1'])} of 6, C2 {n('C2', cr['C2'])} of 6 (needed {cr['need']}). &nbsp; H2 (leg choice): {n('H2', 'PASS' if h2 else 'FAIL')} — C3 {n('C3', cr['C3'])} of 6.</div>"
             + ("<p>With the band on the plain ratio, the rule's timing is no better than random timing of the same trades, and its choice of leg is no better than a coin — on the six index pairs, at the pre-registered setting. The exact percentiles, the MDE beside every fail, the other exits and windows, the gold pairs and the side-by-side with T01 are below.</p>" if not (h1 or h2) else f"<p>See the headline table for which criteria passed and by how much. This is the second ratio form tried on the same window (T01 used the log), so any pass is PROVISIONAL; beside C1, the 97.5th percentile was reached in {n('C1_two_forms', cr['C1_two_forms'])} of 6 (a claim needs {cr['need']}).</p>") + "</div>")
    # the data
    H.append("<h2>The data</h2><table><tr><th>file</th><th>sha256</th><th>role</th></tr>" + "".join(f"<tr><td>{f}</td><td><code>{fp[:16]}…</code></td><td>{'daily OHLC of the NSE indices: signal, fill (OHLC4), valuation' if 'index' in f else 'daily OHLC of MCX gold front-month: the gold leg'}</td></tr>" for f, fp in audit["inputs"].items()) + "</table>"
             + f"<p class='sm'>Index pairs from {audit['start']['index']}, gold pairs from {audit['start']['gold']}, each to the earlier of its legs' last date. Reviewed in data/DATA_REVIEW.md; read only through analysis/approved_data.py (fingerprint checked on open). Open items for the data owner: three dates per series against a public source; the gold 2026-01-30 roll.</p>")
    # one backtest end to end
    H.append("<h2>One backtest, end to end</h2>")
    H.append(f1.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False}))
    H.append(f"<p><b>Inputs → signal.</b> On {n('w_entry_decision', w['entry_decision'])} the ratio N100 ÷ SC100 closed at {n('w_ratio', w['ratio']):.4f} against a trailing mean of {n('w_mean', w['mean']):.4f} and sd {n('w_sd', w['sd']):.4f}: {'more than one sd below, so N100 had lagged SC100' if w['state'] == 'A' else 'more than one sd above, so SC100 had lagged N100'} → target <b>{n('w_state', w['state'])}</b> ({'all in N100' if w['state'] == 'A' else 'all in SC100'}).</p>")
    H.append(f"<p><b>Trade → units.</b> On the next session, {n('w_entry_trade', w['entry_trade'])}, the sleeve traded at the fills N100 {n('w_fill_a', w['fill_a']):.2f} / SC100 {n('w_fill_b', w['fill_b']):.2f} (OHLC4); the cost of that switch was {n('w_cost', w['cost']):.5f} of NAV (10.4 bps on each side of the value traded).</p>")
    H.append(f"<p><b>Exit.</b> Decision of {n('w_exit_decision', w['exit_decision'])} — {w['exit_reason']} — traded {n('w_exit_trade', w['exit_trade'])}. Over the hold the held leg returned {n('w_held_ret', w['held_ret']):+.2%} fill to fill, the peer {n('w_peer_ret', w['peer_ret']):+.2%}: extra {n('w_extra', w['extra']):+.2%}.</p>")
    H.append(f"<p><b>Daily value → curve → statistics.</b> NAV 1.000 on {n('w_first_date', w['first_date'])} → {n('w_final_nav', w['final_nav']):.3f} on {n('w_last_date', w['last_date'])}: CAGR {n('w_cagr', w['cagr']):.2%}, worst fall {n('w_maxdd', w['maxdd']):.1%}.</p>")
    H.append(f"<p><b>One random draw → ranking.</b> N-A draw 0 scatters the same {n('w_draw0_spells', w['draw0_spells'])} trade spells at random: CAGR {n('w_draw0_cagr', w['draw0_cagr']):.2%}. Against all {T['draws']:,} draws the strategy's {w['cagr']:.2%} sits at the {n('w_NA_pct', w['NA_pct']):.1f}th percentile (p = {n('w_NA_p', w['NA_p']):.3f}); null median {n('w_NA_median', w['NA_median']):.2%}, edge {n('w_NA_edge', w['NA_edge']):+.2%}.</p>")
    H.append(f2.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))
    H.append(f3.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False}))
    # gates
    g = T["gates"]
    H.append("<h2>Gates</h2><table><tr><th>gate</th><th>result</th><th>evidence</th></tr>"
             + f"<tr><td>V1 independent verifier</td><td class='{'pass' if g['V1'] else 'fail'}'>{n('V1', 'PASS' if g['V1'] else 'FAIL')}</td><td>{g['verify'][0] if g['verify'] else '?'} nav files, {g['verify'][1] if g['verify'] else '?'} comparisons, {g['verify'][2] if g['verify'] else '?'} failures; corrupted copies caught</td></tr>"
             + f"<tr><td>V2 no look-ahead</td><td class='{'pass' if g['V2'] else 'fail'}'>{n('V2', 'PASS' if g['V2'] else 'FAIL')}</td><td>{n('pit_checks', g['pit_checks']):,} point-in-time checks, {n('pit_violations', g['pit_violations'])} violations</td></tr>"
             + f"<tr><td>V3 the null kept its counts</td><td class='{'pass' if g['V3'] else 'fail'}'>{n('V3', 'PASS' if g['V3'] else 'FAIL')}</td><td>{n('null_checked', g['null_draws_checked']):,} draws, {n('null_mismatched', g['null_draws_mismatched'])} with different counts</td></tr>"
             + f"<tr><td>V4 regression</td><td class='{'pass' if g['V4'] else 'fail'}'>{n('V4', 'PASS' if g['V4'] else 'FAIL')}</td><td>always-neutral = 50/50 buy-and-hold closed form; T01 reproduced by the changed engine: {n('v4_rows_ok', g['v4_rows'][0])} of {n('v4_rows_all', g['v4_rows'][1])} rows of T01's headline summary</td></tr></table>")
    # headline
    H.append(f"<h2>Headline — the six index pairs, reference setting</h2><p class='sm'>CAGR after cost; pct = percentile of the strategy among the null's draws; p = (1 + draws at least as good) ÷ (draws + 1); edge = strategy − null median; MDE = 2.8 × the null's sd, in pp a year.</p>")
    H.append("<table><tr><th>pair</th><th>round trips</th><th>weeks</th><th>CAGR</th><th>N-A median</th><th>edge</th><th>pct N-A</th><th>p</th><th>C1</th><th>max dd</th><th>N-A median dd</th><th>C2</th><th>pct N-B</th><th>C3</th><th>MDE pp/yr</th></tr>")
    for r in T["headline"]:
        k = r["pair"].replace(" ÷ ", "_")
        H.append(f"<tr><td>{r['pair']}</td><td>{n(f'rt_{k}', r['round trips'])}</td><td>{n(f'weeks_{k}', r['weeks'])}</td><td>{n(f'cagr_{k}', r['CAGR']):.2%}</td><td>{n(f'named_{k}', r['N-A median']):.2%}</td><td>{n(f'edge_{k}', r['edge']):+.2%}</td><td>{n(f'napct_{k}', r['pct N-A']):.1f}</td><td>{n(f'nap_{k}', r['p']):.3f}</td><td class='{r['C1'].lower()}'>{r['C1']}</td><td>{n(f'maxdd_{k}', r['max dd']):.1%}</td><td>{n(f'nadd_{k}', r['N-A median dd']):.1%}</td><td class='{r['C2'].lower()}'>{r['C2']}</td><td>{n(f'nbpct_{k}', r['pct N-B']):.1f}</td><td class='{r['C3'].lower()}'>{r['C3']}</td><td>{n(f'mde_{k}', r['MDE (pp/yr)']):.1f}</td></tr>")
    H.append("</table>")
    H.append("<p class='sm'>Gold pairs, same setting (reported, not counted):</p><table><tr><th>pair</th><th>round trips</th><th>reportable</th><th>CAGR</th><th>pct N-A</th><th>max dd</th><th>N-A median dd</th><th>pct N-B</th></tr>" + "".join(f"<tr><td>{r['pair']}</td><td>{r['round trips']}</td><td>{r['reportable']}</td><td>{n('gcagr_' + r['pair'].replace(' ÷ ', '_'), r['CAGR']):.2%}</td><td>{n('gnapct_' + r['pair'].replace(' ÷ ', '_'), r['pct N-A']):.1f}</td><td>{r['max dd']:.1%}</td><td>{r['N-A median dd']:.1%}</td><td>{r['pct N-B']:.1f}</td></tr>" for r in T["gold"]) + "</table>")
    # verdict
    H.append("<h2>Verdict</h2><table><tr><th>criterion</th><th>needed</th><th>got</th><th>result</th></tr>"
             + f"<tr><td>C1 CAGR ≥ 95th percentile of N-A</td><td>{cr['need']} of 6</td><td>{cr['C1']}</td><td class='{'pass' if cr['C1'] >= cr['need'] else 'fail'}'>{'PASS' if cr['C1'] >= cr['need'] else 'FAIL'}</td></tr>"
             + f"<tr><td>C2 drawdown shallower than the N-A median</td><td>{cr['need']} of 6</td><td>{cr['C2']}</td><td class='{'pass' if cr['C2'] >= cr['need'] else 'fail'}'>{'PASS' if cr['C2'] >= cr['need'] else 'FAIL'}</td></tr>"
             + f"<tr><td>C3 CAGR ≥ 95th percentile of N-B</td><td>{cr['need']} of 6</td><td>{cr['C3']}</td><td class='{'pass' if cr['C3'] >= cr['need'] else 'fail'}'>{'PASS' if cr['C3'] >= cr['need'] else 'FAIL'}</td></tr>"
             + f"<tr><td><b>H1 timing</b></td><td></td><td></td><td class='{'pass' if h1 else 'fail'}'><b>{'PASS' if h1 else 'FAIL'}</b></td></tr><tr><td><b>H2 leg choice</b></td><td></td><td></td><td class='{'pass' if h2 else 'fail'}'><b>{'PASS' if h2 else 'FAIL'}</b></td></tr>"
             + f"<tr><td class='sm'>beside C1: CAGR ≥ 97.5th percentile of N-A (two ratio forms tried; not part of H1)</td><td>{cr['need']} of 6</td><td>{n('C1_two_forms', cr['C1_two_forms'])}</td><td>{'reached' if cr['C1_two_forms'] >= cr['need'] else 'not reached'}</td></tr></table>")
    # T02 against T01 (descriptive)
    H.append("<h2>T02 against T01 — the same test on ln(P_a ÷ P_b) (descriptive, no claims)</h2><p class='sm'>Reference setting; the same decision weeks, nulls and seeds; only the series under the band differs. Weeks in a different state, round trips and entries are counted from the two decisions files; the T01 numbers come from tests/T01/evidence/headline/.</p>"
             + "<table><tr><th>pair</th><th>decision weeks</th><th>weeks in a different state</th><th>round trips T01 / T02</th><th>entries same week and leg</th><th>T01 only / T02 only</th><th>CAGR T01</th><th>CAGR T02</th><th>pct N-A T01</th><th>pct N-A T02</th></tr>"
             + "".join(f"<tr><td>{r['pair']}</td><td>{n('vs_weeks_' + r['pair'].replace(' ÷ ', '_'), r['decision weeks'])}</td><td>{n('vs_differ_' + r['pair'].replace(' ÷ ', '_'), r['weeks in a different state'])}</td><td>{n('vs_rt1_' + r['pair'].replace(' ÷ ', '_'), r['round trips T01'])} / {n('vs_rt2_' + r['pair'].replace(' ÷ ', '_'), r['round trips T02'])}</td><td>{n('vs_same_' + r['pair'].replace(' ÷ ', '_'), r['entries same week and leg'])}</td><td>{n('vs_only1_' + r['pair'].replace(' ÷ ', '_'), r['entries T01 only'])} / {n('vs_only2_' + r['pair'].replace(' ÷ ', '_'), r['entries T02 only'])}</td><td>{n('vs_cagr1_' + r['pair'].replace(' ÷ ', '_'), r['CAGR T01']):.2%}</td><td>{n('vs_cagr2_' + r['pair'].replace(' ÷ ', '_'), r['CAGR T02']):.2%}</td><td>{n('vs_pct1_' + r['pair'].replace(' ÷ ', '_'), r['pct N-A T01']):.1f}</td><td>{n('vs_pct2_' + r['pair'].replace(' ÷ ', '_'), r['pct N-A T02']):.1f}</td></tr>" for r in T["vs_T01"]) + "</table>")
    # comparators, robustness, multiplicity (from the log tables)
    H.append("<h2>Comparators, robustness, multiplicity</h2>")
    H.append("<table><tr><th>pair</th><th>window</th><th>strategy</th><th>50/50 bought once</th><th>50/50 rebalanced</th><th>legs alone</th><th>dd strategy / 50/50</th></tr>" + "".join(f"<tr><td>{r['pair']}</td><td class='sm'>{r['window']}</td><td>{n('cmp_strat_' + r['pair'].replace(' ÷ ', '_'), r['strategy']):.2%}</td><td>{n('cmp_bh_' + r['pair'].replace(' ÷ ', '_'), r['50/50 buy-hold']):.2%}</td><td>{r['50/50 rebalanced']:.2%}</td><td>" + " / ".join(f"{k[5:]} {v:.1%}" for k, v in r.items() if k.startswith("hold ") and v == v) + f"</td><td>{r['dd strategy']:.0%} / {r['dd 50/50 b-h']:.0%}</td></tr>" for r in T["comparators"]) + "</table>")
    H.append("<p class='sm'>R1 costs — percentile of CAGR on N-A at 0 / 20.8 / 50 bps:</p><table><tr><th>pair</th>" + "".join(f"<th>{c}</th>" for c in T["R1"][0] if c != "pair") + "</tr>" + "".join("<tr><td>" + r["pair"] + "</td>" + "".join(f"<td>{(f'{v:.2%}' if c.startswith('CAGR') else f'{v:.1f}') if isinstance(v, float) else v}</td>" for c, v in r.items() if c != "pair") + "</tr>" for r in T["R1"]) + "</table>")
    H.append("<p class='sm'>R2 / R3 windows and exits — percentile on N-A (round trips; drawdown vs the null median); — = not reportable:</p><table><tr><th>pair</th>" + "".join(f"<th>{c}</th>" for c in T["R2R3"][0] if c != "pair") + "</tr>" + "".join("<tr><td>" + r["pair"] + "</td>" + "".join(f"<td>{v}</td>" for c, v in r.items() if c != "pair") + "</tr>" for r in T["R2R3"]) + "</table>")
    H.append(f"<p class='sm'>Exit E2 on the pre-named criteria: C1 {n('E2_C1', T['E2']['C1'])} of 6, C2 {n('E2_C2', T['E2']['C2'])} of 6, C3 {n('E2_C3', T['E2']['C3'])} of 6.</p>")
    H.append("<p class='sm'>R4 sub-periods — CAGR strategy vs 50/50 bought once · max drawdown (descriptive):</p><table><tr><th>pair</th>" + "".join(f"<th>{c}</th>" for c in T["R4"][0] if c != "pair") + "</tr>" + "".join("<tr><td>" + r["pair"] + "</td>" + "".join(f"<td>{v}</td>" for c, v in r.items() if c != "pair") + "</tr>" for r in T["R4"]) + "</table>")
    H.append("<p class='sm'>Multiplicity — the best-looking of the 6 settings per pair, judged at the 98th percentile of its own null:</p><table><tr><th>pair</th><th>settings tried</th><th>best</th><th>its pct N-A</th><th>passes 98th</th></tr>" + "".join(f"<tr><td>{r['pair']}</td><td>{r['settings tried']}</td><td>{r['best setting']}</td><td>{n('best_' + r['pair'].replace(' ÷ ', '_'), r['its pct N-A']):.1f}</td><td>{r['passes 98th']}</td></tr>" for r in T["multiplicity"]) + "</table>")
    # formulas
    H.append("<h2>Formulas</h2><ul>"
             "<li>ratio: r_t = P_a,t ÷ P_b,t, not logged (T01 used ln of it), on the close of the last trading day of each ISO week on the pair's common calendar; a = the pair's first asset in the order N100, MC100, SC100, Micro250, GoldMCX (T01's orientation)</li>"
             f"<li>band: mean and sd (n − 1) of r over the trailing {ref['window']} decision weeks ending at t (at least {int(round(audit['min_obs_frac'] * ref['window']))}); entry on the first week r &lt; mean − sd (→ all in a) or r &gt; mean + sd (→ all in b) while neutral; exit E1 when r is back through the mean, E2 at the opposite band, or after {audit['max_hold_weeks']} weeks</li>"
             "<li>execution: decide on the decision day's close, trade the next common day at fill = (open + high + low + close) ÷ 4 (the close on a stale-OHL row); cost 10.4 bps of the value traded on each side; units held; NAV = Σ units × close every day</li>"
             "<li>CAGR = NAV_end^(365.25 ÷ days) − 1 from the first banded decision day; max drawdown = min(NAV ÷ running max − 1); percentile = (draws worse + ½ ties) ÷ draws; p = (1 + draws at least as good) ÷ (draws + 1); MDE = 2.8 × sd of the null's CAGR</li>"
             "<li>N-A: the strategy's trade spells (grouped where they abut) permuted among their positions and the non-zero gaps among theirs — same spells, lengths, weeks in each state, switches; N-B: the same timing, each group's legs flipped by a fair coin</li></ul>")
    H.append(f"<script id='numbers' type='application/json'>{json.dumps(NUM)}</script></body></html>")
    out = os.path.join(EV, "T02_results.html"); open(out, "w", encoding="utf-8").write("\n".join(H)); open(os.path.join(HERE, "T02_results.html"), "w", encoding="utf-8").write("\n".join(H))
    print(f"wrote {out} ({len(NUM)} numbers embedded) and tests/T02/T02_results.html")


if __name__ == "__main__":
    main()
