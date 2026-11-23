"""
E08 (EXPLORATORY): commodity ETFs against SPY — which relationship is best, and does the ratio switch work there?
Design: explore/E08_commodity_spy/E08.md (written before the first run).

    python explore/E08_commodity_spy/commodity.py [--draws 500]

Every number on the page and in E08_results.md is computed here; the page's script only picks the numbers of the chosen
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
from analysis.approved_data import fingerprints, load_us_etf, load_us_gold  # noqa: E402
from analysis.ratio_engine import PIT, book, counts, null_A, null_B, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (palette, page styling, percentile)
import zwin as ZW  # noqa: E402  (the z windows)
import variants as VR  # noqa: E402  (the rule through the engine's book)

MAIN = ["GCC", "FTGC", "COMT", "PDBC", "BCI", "BCD", "COM", "CMDY", "SDCI"]; SHORT = ["HGER", "NBCM", "PIT", "CERY"]
GOLD_ETF = ["GLD"]   # added 2026-11-20 (E08.md): SPY against gold, reported on its own; pair_id 13 keeps the ETFs' seeds
ETFS = MAIN + SHORT + GOLD_ETF
GROUP = {**{e: "main" for e in MAIN}, **{e: "short" for e in SHORT}, **{e: "gold" for e in GOLD_ETF}}
SUFFIX = {"main": "", "short": " (short)", "gold": " (gold)"}
NAMES = {"GCC": "WisdomTree Enhanced Commodity", "FTGC": "First Trust Global Tactical Commodity", "COMT": "iShares GSCI Commodity Dynamic Roll", "PDBC": "Invesco Optimum Yield Diversified Commodity",
         "BCI": "abrdn Bloomberg All Commodity", "BCD": "abrdn Bloomberg All Commodity Longer Dated", "COM": "Direxion Auspice Broad Commodity", "CMDY": "iShares Bloomberg Roll Select Commodity",
         "SDCI": "USCF SummerHaven Dynamic Commodity", "HGER": "Harbor Commodity All-Weather", "NBCM": "Neuberger Berman Commodity Strategy", "PIT": "VanEck Commodity Strategy", "CERY": "SPDR Bloomberg Enhanced Roll Yield",
         "GLD": "SPDR Gold Shares (gold bullion)"}
COST = VR.COST; SEED_NB = 100; LO, HI = E1.LO, E1.HI; DEFAULT_ETF = "PDBC"


def r4(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def r6(v):
    return r4(v, 6)


def pstats(nav, bh, start=None, end=None):
    """CAGR, 50/50's CAGR, excess and max drawdown of the strategy between start and end (inclusive)."""
    nv = pd.DataFrame({"nav": nav, "bh": bh}).dropna()
    if start is not None:
        nv = nv[nv.index >= start]
    if end is not None:
        nv = nv[nv.index <= end]
    if len(nv) < 20:
        return None
    yrs = (nv.index[-1] - nv.index[0]).days / 365.25; cg = lambda s: (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1  # noqa: E731
    s = nv["nav"] / nv["nav"].iloc[0]
    return {"cagr": float(cg(nv["nav"])), "cagr_5050": float(cg(nv["bh"])), "excess": float(cg(nv["nav"]) - cg(nv["bh"])), "maxdd": float((s / s.cummax() - 1).min()),
            "first": nv.index[0].date().isoformat(), "last": nv.index[-1].date().isoformat()}


def measures(lnr: pd.Series, z52: pd.Series) -> dict:
    """The ratio's character over a period of decision weeks (E07's measures, on the whole period)."""
    d = lnr.diff().dropna(); lag = lnr.shift(1).reindex(d.index)
    b = float(np.cov(d.to_numpy(), lag.to_numpy())[0, 1] / np.var(lag.to_numpy(), ddof=1))
    hl = float(min(-np.log(2) / np.log1p(max(b, -0.999999)), 520.0)) if b < 0 else 520.0
    vr = float((lnr - lnr.shift(4)).dropna().var() / (4 * d.var()))
    ac = float(np.corrcoef(d.to_numpy()[1:], d.to_numpy()[:-1])[0, 1])
    eff = float(abs(lnr.iloc[-1] - lnr.iloc[0]) / d.abs().sum())
    z = z52.reindex(lnr.index).dropna(); s = np.sign(z.to_numpy()); ch = int((s[1:] != s[:-1]).sum())
    zyrs = (z.index[-1] - z.index[0]).days / 365.25 if len(z) > 1 else np.nan
    return {"half_life": hl, "zcross_yr": ch / zyrs if zyrs and zyrs > 0 else np.nan, "vr": vr, "ac": ac, "eff": eff}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=500); ND = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); px = load_us_etf(); px.update(load_us_gold())
    checks = {"close_max_abs_diff": 0.0, "calendar_mismatch": [], "cut_checks": 0, "cut_violations": [], "null_draws": 0, "null_count_mismatch": 0}
    REL, RULE, TRADES, CURVES, PAIRD, SEL = {}, [], [], {}, {}, {}
    wk_ret = {}
    for pid, etf in enumerate(ETFS):
        pair = f"{etf}|SPY"; pf = pair_frame(px, pair, "plain"); a, b = px[etf], px["SPY"]
        cal = a.index.intersection(b.index); cal = cal[cal >= pd.Timestamp("2006-01-02")]
        if not pf.index.equals(cal):
            checks["calendar_mismatch"].append(etf)
        checks["close_max_abs_diff"] = max(checks["close_max_abs_diff"], float((pf["ca"] - a["close"].reindex(pf.index)).abs().max()), float((pf["cb"] - b["close"].reindex(pf.index)).abs().max()))
        wk = pf.loc[pf["decision"]]; lnr = np.log(wk["ratio"]); ra, rb = np.log(wk["ca"]).diff(), np.log(wk["cb"]).diff(); wk_ret[etf] = ra
        first, last = pf.index[0], pf.index[-1]; mid = first + (last - first) / 2
        z52 = VR.z_at_decisions(pf, "w52")
        # §1 the relationship over the whole period
        both = pd.concat([ra, rb], axis=1).dropna(); corr = float(both.corr().iloc[0, 1]); beta = float(both.cov().iloc[0, 1] / both.iloc[:, 1].var())
        rc = ra.rolling(52).corr(rb).dropna()
        mc = pf["cb"].groupby([pf.index.year, pf.index.month]).last(); me = pf["ca"].groupby([pf.index.year, pf.index.month]).last()
        mr = pd.DataFrame({"spy": mc.pct_change(), "etf": me.pct_change()}).dropna(); worst = mr.nsmallest(5, "spy")
        m_all = measures(lnr, z52)
        # §2 the score's inputs: the first half of the ETF's own period
        h1 = lnr[lnr.index <= mid]; m_h1 = measures(h1, z52[z52.index <= mid]) if len(h1) > 60 else None
        REL[etf] = {"name": NAMES[etf], "group": GROUP[etf], "first": first.date().isoformat(), "last": last.date().isoformat(), "mid": mid.date().isoformat(),
                    "years": (last - first).days / 365.25, "corr": corr, "beta": beta, "corr52_min": float(rc.min()) if len(rc) else np.nan, "corr52_max": float(rc.max()) if len(rc) else np.nan,
                    "worst5_spy": float(worst["spy"].mean()), "worst5_etf": float(worst["etf"].mean()), "worst5_months": [f"{y}-{m:02d}" for y, m in worst.index],
                    "etf_cagr": float((pf["ca"].iloc[-1] / pf["ca"].iloc[0]) ** (365.25 / (last - first).days) - 1), "spy_cagr": float((pf["cb"].iloc[-1] / pf["cb"].iloc[0]) ** (365.25 / (last - first).days) - 1),
                    **{f"all_{k}": v for k, v in m_all.items()}, **({f"h1_{k}": v for k, v in m_h1.items()} if m_h1 else {})}
        # cut-history check of the 52-week z and the measures' inputs
        dec = wk.index
        for T in (np.unique(np.linspace(80, len(dec) - 1, 8).astype(int)) if len(dec) > 100 else []):
            cut = pf.loc[: dec[T]]; zc = VR.z_at_decisions(cut, "w52"); full = z52.iloc[: len(zc)]; checks["cut_checks"] += 1
            if not np.allclose(zc.to_numpy(float), full.to_numpy(float), rtol=0, atol=1e-10, equal_nan=True):
                checks["cut_violations"].append(f"{etf} cut {dec[T].date()}")
        # §3 the rule on every z window, with the nulls
        PAIRD[etf] = {"dates": [d.date().isoformat() for d in dec], "ratio": [r6(v) for v in wk["ratio"].to_numpy()], "z": {}, "trades": {}}
        CURVES[etf] = {}
        for wi, (w, *_) in enumerate(ZW.WINDOWS):
            z = VR.z_at_decisions(pf, w); PAIRD[etf]["z"][w] = [None if v != v else round(float(v), 3) for v in z.to_numpy()]
            banded = z.dropna()
            if len(banded) < 30:
                RULE.append({"etf": etf, "window": w, "status": "too short"}); continue
            r = VR.run_states(pf, z, COST, 1.0); nav, bh = r["nav"], r["bh"]
            ex = r["extras"]; cl = ex[~ex["open"].astype(bool)] if len(ex) else ex
            whole, s1, s2 = pstats(nav, bh), pstats(nav, bh, end=mid), pstats(nav, bh, start=mid)
            # nulls: the same targets, permuted (N-A) or leg-flipped (N-B), through the same book
            dd = r["decisions"]; bd = dd[dd["z"].notna()]; targets = bd["state_after"].tolist(); base = counts(targets); cg = {"N-A": [], "N-B": []}
            for nname, fn, off in (("N-A", null_A, 0), ("N-B", null_B, SEED_NB)):
                for dr in range(ND):
                    tg = fn(targets, np.random.default_rng([pid + off, 0, dr, wi])); cc = counts(tg); checks["null_draws"] += 1
                    ok = (cc == base) if nname == "N-A" else (cc["spells"] == base["spells"] and cc["switches"] == base["switches"] and cc["weeks_N"] == base["weeks_N"])
                    checks["null_count_mismatch"] += (not ok)
                    nv_, _ = book(pf, bd.index, tg, COST, PIT()); cg[nname].append(pstats(nv_, bh)["cagr"])
            pa, pb = E1.pctile(whole["cagr"], np.array(cg["N-A"])), E1.pctile(whole["cagr"], np.array(cg["N-B"]))
            row = {"etf": etf, "group": REL[etf]["group"], "window": w, "status": "ok", "trades": int(len(cl)), "open": int(ex["open"].astype(bool).sum()) if len(ex) else 0,
                   "win": float((cl["extra"] > 0).mean()) if len(cl) else np.nan, "mean_extra": float(cl["extra"].mean()) if len(cl) else np.nan,
                   "cagr": whole["cagr"], "cagr_5050": whole["cagr_5050"], "excess": whole["excess"], "maxdd": whole["maxdd"], "first": whole["first"], "last": whole["last"],
                   "excess_h1": s1["excess"] if s1 else np.nan, "excess_h2": s2["excess"] if s2 else np.nan, "na_pct": pa, "nb_pct": pb,
                   "na_median_cagr": float(np.median(cg["N-A"])), "nb_median_cagr": float(np.median(cg["N-B"])), "weeks_etf": base["weeks_A"], "weeks_spy": base["weeks_B"], "weeks_5050": base["weeks_N"]}
            RULE.append(row)
            trs = []
            for _, t in ex.iterrows():
                op = bool(t["open"]); ed = pd.Timestamp(t["entry_decision"]); ei = int(dec.get_loc(ed))
                tr_ = {"etf": etf, "window": w, "entry_decision": ed.date().isoformat(), "entry_trade": pd.Timestamp(t["entry_trade"]).date().isoformat(), "exit_trade": None if op else pd.Timestamp(t["exit_trade"]).date().isoformat(),
                       "held": etf if t["held"] == "A" else "SPY", "open": op, "extra": np.nan if op else float(t["extra"]), "z_entry": float(z.iloc[ei])}
                TRADES.append(tr_); trs.append([ei, tr_["held"], tr_["entry_trade"], tr_["exit_trade"], None if op else round(tr_["extra"] * 10000)])
            PAIRD[etf]["trades"][w] = trs
            # the curves at decision-week closes, from the rule's first day: the rule, 50/50, the ETF and SPY bought once
            k0 = nav.index[0]; wsel = dec[dec >= k0]
            CURVES[etf][w] = {"t": [d.date().isoformat() for d in wsel], "rule": [r6(v) for v in nav.reindex(wsel).to_numpy()], "bh": [r6(v) for v in bh.reindex(wsel).to_numpy()],
                              "etf": [r6(v) for v in (pf["ca"].reindex(wsel) / pf["ca"].loc[k0]).to_numpy()], "spy": [r6(v) for v in (pf["cb"].reindex(wsel) / pf["cb"].loc[k0]).to_numpy()]}
        print(f"  {etf:5s} {REL[etf]['first']} → {REL[etf]['last']}  [{time.time() - t0:4.0f}s]", flush=True)
    if checks["calendar_mismatch"] or checks["cut_violations"] or checks["null_count_mismatch"] or checks["close_max_abs_diff"] > 0:
        sys.exit(f"checks FAILED: {checks}")
    RU = pd.DataFrame(RULE); RU.to_csv(os.path.join(OUT, "rule_by_window.csv"), index=False); pd.DataFrame(TRADES).to_csv(os.path.join(OUT, "trades.csv"), index=False)
    RL = pd.DataFrame([{"etf": k, **{kk: vv for kk, vv in v.items() if kk != "worst5_months"}, "worst5_months": " ".join(v["worst5_months"])} for k, v in REL.items()]); RL.to_csv(os.path.join(OUT, "relationship.csv"), index=False)
    # §2 the score on the first halves (main ETFs), fixed in E08.md before the run
    M = RL[RL["group"] == "main"].set_index("etf")
    rk = pd.DataFrame({"hl": M["h1_half_life"].rank(method="min"), "zc": (-M["h1_zcross_yr"]).rank(method="min"), "vr": M["h1_vr"].rank(method="min")})
    rk["score"] = rk.mean(axis=1); order = rk.sort_values(["score"]).index.tolist(); chosen = order[:3]
    for e in MAIN:
        g = RU[(RU["etf"] == e) & (RU["status"] == "ok")]
        SEL[e] = {"rank_hl": int(rk.loc[e, "hl"]), "rank_zc": int(rk.loc[e, "zc"]), "rank_vr": int(rk.loc[e, "vr"]), "score": float(rk.loc[e, "score"]), "chosen": e in chosen,
                  "h2_w52": float(g[g["window"] == "w52"]["excess_h2"].iloc[0]) if (g["window"] == "w52").any() else np.nan, "h2_median": float(g["excess_h2"].median()), "h2_ahead": int((g["excess_h2"] > 0).sum()), "n": int(len(g)),
                  "whole_median": float(g["excess"].median()), "whole_ahead": int((g["excess"] > 0).sum())}
    ch = [SEL[e] for e in chosen]; ot = [SEL[e] for e in MAIN if e not in chosen]
    rum = RU[(RU["group"] == "main") & (RU["status"] == "ok")]
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "inputs": fingerprints(), "draws_per_null": ND, "seed": f"default_rng([pair_id (+{SEED_NB} for N-B), 0, draw, window_id])",
             "cost_bps_round_trip": COST, "fill": "next session's close (rule H3; the source has closes only)", "checks": checks, "chosen_by_first_half_score": chosen, "score_order": order,
             "main_pair_windows": int(len(rum)), "main_ahead_of_5050": int((rum["excess"] > 0).sum()), "main_na_notable_high": int((rum["na_pct"] >= HI).sum()), "main_na_notable_low": int((rum["na_pct"] <= LO).sum()),
             "main_nb_notable_high": int((rum["nb_pct"] >= HI).sum()), "main_nb_notable_low": int((rum["nb_pct"] <= LO).sum())}
    rug = RU[(RU["group"] == "gold") & (RU["status"] == "ok")]   # GLD ÷ SPY, reported on its own (added 2026-11-20)
    audit.update({"gold_windows": int(len(rug)), "gold_ahead_of_5050": int((rug["excess"] > 0).sum()), "gold_na_notable_high": int((rug["na_pct"] >= HI).sum()), "gold_na_notable_low": int((rug["na_pct"] <= LO).sum()),
                  "gold_nb_notable_high": int((rug["nb_pct"] >= HI).sum()), "gold_nb_notable_low": int((rug["nb_pct"] <= LO).sum()), "gold_median_excess": float(rug["excess"].median()) if len(rug) else None,
                  "seconds": round(time.time() - t0)})
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
    def pp(v): return "—" if v is None or v != v else f"{v * 100:+.2f} pp"
    def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
    def sg2(v): return "—" if v is None or v != v else f"{v:+.2f}"
    wl = {w: f"{lab}{' (daily)' if f == 'D' else ''}" for w, lab, f, _ in ZW.WINDOWS}
    md = [f"# E08 results — commodity ETFs against SPY (EXPLORATORY)\n",
          f"_Generated by `explore/E08_commodity_spy/commodity.py` on {audit['run_date']} from the US ETF data approved 2026-11-19 and GLD approved 2026-11-20 (total return; launch periods dropped; fills at the next close). Each ETF ÷ SPY, and GLD ÷ SPY (gold, added 2026-11-20); Rule R (T03's: ±1, exit at 0) on every z window, 10.4 bps per side; {ND} draws per null. Description, not a finding; percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Checks:** pair closes equal the loader's (max difference {checks['close_max_abs_diff']:.1e}); calendars are the legs' common days; the 52-week z passes the cut-history check ({checks['cut_checks']} cuts, 0 violations); {checks['null_draws']:,} null draws, every one kept its counts.\n",
          "## 1. The relationship (whole usable period)\n",
          "| ETF | dates | ETF / SPY CAGR | weekly correlation with SPY (52-wk range) | beta | in SPY's 5 worst months: SPY / ETF | half-life (weeks) | z sign changes a year | variance ratio | autocorrelation |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ETFS:
        v = REL[e]; md.append(f"| {e}{SUFFIX[v['group']]} | {v['first']} → {v['last']} | {fp(v['etf_cagr'])} / {fp(v['spy_cagr'])} | {v['corr']:+.2f} ({sg2(v['corr52_min'])} to {sg2(v['corr52_max'])}) | {v['beta']:.2f} | "
                              f"{fp(v['worst5_spy'])} / {fp(v['worst5_etf'])} | {v['all_half_life']:.0f} | {v['all_zcross_yr']:.1f} | {v['all_vr']:.2f} | {v['all_ac']:+.2f} |")
    md += ["\n## 2. The best relationship for the switch, chosen on the first half of each ETF's period (fixed in E08.md before the run)\n",
           "| ETF | first half | rank: half-life / z changes / variance ratio | score | chosen | second half, Rule R 52 weeks: excess over 50/50 | second half, all windows: median excess (windows ahead) |", "|---|---|---|---|---|---|---|"]
    for e in order:
        s = SEL[e]; md.append(f"| {e} | {REL[e]['first']} → {REL[e]['mid']} | {s['rank_hl']} / {s['rank_zc']} / {s['rank_vr']} | {s['score']:.2f} | {'**yes**' if s['chosen'] else ''} | {pp(s['h2_w52'])} | {pp(s['h2_median'])} ({s['h2_ahead']} of {s['n']}) |")
    md.append(f"\nChosen ({', '.join(chosen)}): second-half median excess {pp(np.median([c['h2_median'] for c in ch]))}, windows ahead {sum(c['h2_ahead'] for c in ch)} of {sum(c['n'] for c in ch)}; the other six: {pp(np.median([o['h2_median'] for o in ot]))}, {sum(o['h2_ahead'] for o in ot)} of {sum(o['n'] for o in ot)}.\n")
    md += ["## 3. Rule R on every ETF and z window: excess CAGR over 50/50 bought once, whole period (N-A percentile · N-B percentile)\n", "| ETF | " + " | ".join(wl[w] for w, *_ in ZW.WINDOWS) + " |", "|---|" + "---|" * len(ZW.WINDOWS)]
    for e in ETFS:
        cells = []
        for w, *_ in ZW.WINDOWS:
            q = RU[(RU["etf"] == e) & (RU["window"] == w)]
            cells.append("too short" if not len(q) or q["status"].iloc[0] != "ok" else f"{pp(q['excess'].iloc[0])} ({pq(q['na_pct'].iloc[0])} · {pq(q['nb_pct'].iloc[0])}; {int(q['trades'].iloc[0])} tr)")
        md.append(f"| {e}{SUFFIX[GROUP[e]]} | " + " | ".join(cells) + " |")
    md.append(f"\nMain ETFs: ahead of 50/50 in {audit['main_ahead_of_5050']} of {audit['main_pair_windows']} ETF-windows; N-A percentile notable high in {audit['main_na_notable_high']}, notable low in {audit['main_na_notable_low']}; N-B notable high in {audit['main_nb_notable_high']}, low in {audit['main_nb_notable_low']}.\n")
    md.append(f"Gold (GLD ÷ SPY, added 2026-11-20, reported on its own): ahead of 50/50 in {audit['gold_ahead_of_5050']} of {audit['gold_windows']} windows (median excess {pp(audit['gold_median_excess'])}); "
              f"N-A notable high in {audit['gold_na_notable_high']}, low in {audit['gold_na_notable_low']}; N-B notable high in {audit['gold_nb_notable_high']}, low in {audit['gold_nb_notable_low']}.\n")
    open(os.path.join(OUT, "E08_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- the page
    # weekly-return correlations among the main ETFs and SPY (each pair over its common weeks)
    names_c = MAIN + GOLD_ETF + ["SPY"]; spyw = np.log(px["SPY"]["close"]).groupby(pd.Grouper(freq="W-FRI")).last().diff()
    wr = {e: np.log(px[e]["close"]).groupby(pd.Grouper(freq="W-FRI")).last().diff() for e in MAIN + GOLD_ETF}; wr["SPY"] = spyw
    CM = [[r4(pd.concat([wr[a_], wr[b_]], axis=1).dropna().corr().iloc[0, 1], 3) for b_ in names_c] for a_ in names_c]
    RUJ = {}
    for _, r in RU.iterrows():
        RUJ.setdefault(r["etf"], {})[r["window"]] = None if r["status"] != "ok" else [int(r["trades"]), r6(r["win"]), r6(r["mean_extra"]), r6(r["cagr"]), r6(r["cagr_5050"]), r6(r["excess"]), r6(r["maxdd"]),
                                                                                    r6(r["excess_h1"]), r6(r["excess_h2"]), r4(r["na_pct"], 2), r4(r["nb_pct"], 2), r["first"], r["last"], int(r["open"])]
    RELJ = {e: {k: (r6(v) if isinstance(v, float) else v) for k, v in REL[e].items()} for e in ETFS}
    SELJ = {e: {k: (r6(v) if isinstance(v, float) else v) for k, v in SEL[e].items()} for e in MAIN}
    DATA = {"etfs": ETFS, "main": MAIN, "group": GROUP, "suffix": SUFFIX, "names": NAMES, "windows": [[w, lab, f] for w, lab, f, _ in ZW.WINDOWS], "wlab": wl, "lo": LO, "hi": HI, "draws": ND, "rel": RELJ, "sel": SELJ, "order": order, "chosen": chosen,
            "rule": RUJ, "pair": PAIRD, "curves": CURVES, "cm": {"names": names_c, "m": CM}, "audit": {k: audit[k] for k in ("main_pair_windows", "main_ahead_of_5050", "main_na_notable_high", "main_na_notable_low", "main_nb_notable_high", "main_nb_notable_low", "gold_windows", "gold_ahead_of_5050")},
            "defaults": {"etf": DEFAULT_ETF, "w": ZW.DEFAULT}}
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    def sel(i, lab, opts, default):
        return f"<label>{lab} <select id='{i}'>" + "".join(f"<option value='{html.escape(str(v), quote=True)}'{' selected' if str(v) == str(default) else ''}>{html.escape(t)}</option>" for v, t in opts) + "</select></label>"
    controls = sel("etf", "ETF", [(e, e + SUFFIX[GROUP[e]]) for e in ETFS], DEFAULT_ETF) + sel("zwin", "z window", [(w, wl[w]) for w, *_ in ZW.WINDOWS], ZW.DEFAULT)
    best_corr = min(MAIN, key=lambda e: REL[e]["corr"])
    key = (f"<p style='margin:4px 0'><b>The relationship.</b> The nine main ETFs are all broad commodity baskets: their weekly returns correlate with SPY at {min(REL[e]['corr'] for e in MAIN):+.2f} to {max(REL[e]['corr'] for e in MAIN):+.2f} over their periods "
           f"(lowest: {best_corr}, {REL[best_corr]['corr']:+.2f}), and with each other far more. Over 2010–2026 SPY made {fp(REL['GCC']['spy_cagr'])} a year; the longest-lived ETF, GCC, {fp(REL['GCC']['etf_cagr'])}.</p>"
           f"<p style='margin:4px 0'><b>The best relationship for the switch</b>, chosen on the first half of each ETF's period by the score fixed before the run (a quickly reverting ratio): {', '.join(chosen)}. "
           f"On the second halves the rule's median excess over 50/50 is {pp(np.median([c['h2_median'] for c in ch]))} for them (windows ahead of 50/50: {sum(c['h2_ahead'] for c in ch)} of {sum(c['n'] for c in ch)}), "
           f"{pp(np.median([o['h2_median'] for o in ot]))} for the other six ({sum(o['h2_ahead'] for o in ot)} of {sum(o['n'] for o in ot)}).</p>"
           f"<p style='margin:4px 0'><b>The ratio switch</b> (Rule R) on the nine main ETFs over all seven windows: ahead of 50/50 bought once in {audit['main_ahead_of_5050']} of {audit['main_pair_windows']} ETF-windows; "
           f"against random timing (N-A) notably better in {audit['main_na_notable_high']} and notably worse in {audit['main_na_notable_low']}; against flipped legs (N-B) notably better in {audit['main_nb_notable_high']}, worse in {audit['main_nb_notable_low']}.</p>")
    gv = REL["GLD"]
    key += (f"<p style='margin:4px 0'><b>SPY against gold</b> (GLD ÷ SPY, added 2026-11-20, reported on its own; {gv['first']} → {gv['last']}): weekly correlation {gv['corr']:+.2f} (52-week range {sg2(gv['corr52_min'])} to {sg2(gv['corr52_max'])}), beta {gv['beta']:.2f}; "
            f"in SPY's five worst months ({fp(gv['worst5_spy'])} on average) GLD made {fp(gv['worst5_etf'])}. GLD {fp(gv['etf_cagr'])} a year against SPY's {fp(gv['spy_cagr'])}; the ratio's half-life {gv['all_half_life']:.0f} weeks, variance ratio {gv['all_vr']:.2f}. "
            f"Rule R: ahead of 50/50 in {audit['gold_ahead_of_5050']} of {audit['gold_windows']} windows (median excess {pp(audit['gold_median_excess'])}); against random timing (N-A) notably better in {audit['gold_na_notable_high']}, worse in {audit['gold_na_notable_low']}; "
            f"against flipped legs (N-B) better in {audit['gold_nb_notable_high']}, worse in {audit['gold_nb_notable_low']}.</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>Commodities vs SPY</title>"),
         "<header class='bar'><div class='bar-inner'><b>E08 · commodity ETFs ÷ SPY</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>Commodity ETFs against SPY: which relationship is best, and does the ratio switch work there?</h1>",
         f"<p>Thirteen broad commodity ETFs (the desk's comparison) against SPY, each as the ratio ETF ÷ SPY. All prices are total return (distributions reinvested); each fund's launch period is dropped; trades fill at the next session's close, 10.4 bps each way. "
         f"Nine ETFs have five or more years (GCC from 2010; FTGC 2015; COMT, PDBC 2016; BCI, BCD, COM 2019; CMDY, SDCI 2020); HGER, NBCM, PIT (2024) and CERY (2026) are short and only reported. Added on 2026-11-20: <b>SPY against gold</b>, GLD ÷ SPY from 2010, reported on its own. To 2026-11-18. Generated {audit['run_date']}. <b>Description, not a finding</b>; these pairs were explored, so they can no longer serve as an unseen test.</p>",
         "<div class='card'>" + key + "</div>",
         "<h2>1 · The relationship, ETF by ETF</h2><p class='note'>Weekly log returns over each ETF's usable period. The ratio's character: half-life of mean reversion (shorter = the ratio comes back faster), how often the 52-week z changes sign, the variance ratio (below 1 = moves tend to reverse), and the autocorrelation of weekly changes.</p>",
         "<div class='card'><div id='g_corr' class='plot'></div><div id='t_rel'></div></div><div class='card'><div id='g_cm' class='plot'></div></div>",
         "<h2>2 · The best relationship for the switch</h2><p class='note'>Fixed before the run (E08.md): each main ETF is ranked among the nine on the <b>first half</b> of its own period by half-life (shorter better), z sign changes a year (more better) and variance ratio (lower better); the three with the best average rank are chosen, and the rule is read on the <b>second half</b>.</p>",
         "<div class='card'><div id='g_sel' class='plot'></div><div id='t_sel'></div></div>",
         "<h2>3 · The ratio switch on every ETF and z window</h2><p class='note'>Rule R: from 50/50, z below −1 → hold only the ETF, above +1 → hold only SPY; back to 50/50 when z returns to 0. Excess = CAGR − 50/50 bought once's CAGR. Percentiles: the real CAGR among " + f"{ND}" + " draws of T03's nulls (N-A: the same trades at random times; N-B: the same timing with the legs flipped); ≤ 5 or ≥ 95 = notable.</p>",
         "<div class='card'><div id='g_heat' class='plot'></div></div><div class='card'><div id='t_rule'></div></div>",
         "<h2>4 · The chosen ETF: the ratio, the trades and the equity curves</h2>",
         "<div class='card'><div id='g_pair' class='plot'></div></div><div class='card'><div id='g_eq' class='plot'></div><div id='g_dd' class='plot'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Data.</b> analysis/approved_data.load_us_etf(): total return (H4), launch periods dropped (H5), fill = close (H3: the source has closes only). Review: data/DATA_REVIEW_US_ETF.md (F-RA-D12, F-RA-D13); spot checks against issuer data are an open item. "
         "GLD: analysis/approved_data.load_us_gold() (approved 2026-11-20; no distributions; review data/DATA_REVIEW_US_GOLD.md).</li>"
         "<li><b>Rule and book.</b> explore/E01_ratio_cycles/variants.py and analysis/ratio_engine.book, unchanged; decisions on the last trading day of each ISO week; daily windows read at the decision day.</li>"
         f"<li><b>Nulls.</b> analysis/ratio_engine.null_A and null_B, {ND} draws each, seed default_rng([pair_id (+100 for N-B), 0, draw, window_id]); every draw's counts measured ({checks['null_draws']:,} draws, all kept).</li>"
         "<li><b>Limits.</b> 6.9–16.9 years per main ETF, one regime for SPY; the nine ETFs move together, so they are not nine independent tests; curves on this page are at weekly decision closes (the tables' drawdowns are daily). Design written before the first run: explore/E08_commodity_spy/E08.md.</li></ul></div>",
         "</main>", f"<script id='e08-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E08_commodity.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; chosen {chosen}; page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
.legend-dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 0;vertical-align:middle}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e08-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function COL(){return isDark()?{rule:'#3987e5',etf:'#c98500',spy:'#199e70',bh:'#898781',bad:'#d95926'}:{rule:'#2a78d6',etf:'#eda100',spy:'#1baf7a',bh:'#898781',bad:'#eb6834'};}
function v(id){return document.getElementById(id).value;}
function wl(w){return D.wlab[w]||w;}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function fpp(x){if(x===null||x===undefined)return '—';let s=(x*100).toFixed(2);if(Number(s)===0)s='0.00';return (Number(s)>=0?'+':'')+s+' pp';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function f2(x,d){return x===null||x===undefined?'—':(+x).toFixed(d===undefined?2:d);}
function sg(x,d){return x===null||x===undefined?'—':((x>=0?'+':'')+(+x).toFixed(d===undefined?2:d));}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:100,b:50},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',traceorder:'normal',font:{color:c.ink2,size:11}}},extra||{});}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function heatColors(){return isDark()?[[0,'#86b6ef'],[0.2,'#3987e5'],[0.4,'#34465e'],[0.5,'#383835'],[0.6,'#5e3a3a'],[0.8,'#e66767'],[1,'#f29a9a']]:[[0,'#104281'],[0.2,'#2a78d6'],[0.4,'#cde2fb'],[0.5,'#f0efec'],[0.6,'#f9d3d2'],[0.8,'#e34948'],[1,'#a32b2a']];}
function S(){return {e:v('etf'), w:v('zwin')};}

function drawRel(s){
  const c=th(), K=COL(), es=D.etfs, R=D.rel;
  Plotly.react('g_corr',[{type:'bar',x:es,y:es.map(e=>R[e].corr),marker:{color:es.map(e=>e===s.e?K.rule:(D.group[e]!=='short'?(isDark()?'#5f5e5a':'#c3c2b7'):(isDark()?'#383835':'#e1e0d9')))},
    text:es.map(e=>sg(R[e].corr)),textposition:'outside',cliponaxis:false,customdata:es.map(e=>R[e].first+' → '+R[e].last+' · beta '+f2(R[e].beta)+' · 52-week range '+sg(R[e].corr52_min)+' to '+sg(R[e].corr52_max)),hovertemplate:'%{x}: %{text}<br>%{customdata}<extra></extra>'}],
    layout(two('Weekly correlation of each ETF with SPY, over its usable period','blue = the ETF picked at the top; pale = short history (reported only)'),380,{xaxis:ax(),yaxis:ax({title:{text:'correlation',font:{color:c.ink2}},zeroline:true,zerolinecolor:c.axis})}),CFG);
  document.getElementById('t_rel').innerHTML=table(['ETF','dates','ETF / SPY CAGR','correlation (52-wk range)','beta','SPY\'s 5 worst months: SPY / ETF','half-life (wk)','z changes a year','variance ratio','autocorr.'],
    es.map(e=>{const r=R[e];return [(e===s.e?'<b>'+e+'</b>':e)+D.suffix[r.group]+"<br><span class='note'>"+r.name+'</span>',r.first+' → '+r.last,fp(r.etf_cagr)+' / '+fp(r.spy_cagr),sg(r.corr)+' ('+sg(r.corr52_min)+' to '+sg(r.corr52_max)+')',f2(r.beta),fp(r.worst5_spy)+' / '+fp(r.worst5_etf),f2(r.all_half_life,0),f2(r.all_zcross_yr,1),f2(r.all_vr),sg(r.all_ac)];}));
  const n=D.cm.names, m=D.cm.m;
  Plotly.react('g_cm',[{type:'heatmap',z:m,x:n,y:n,zmin:-1,zmax:1,colorscale:heatColors(),text:m.map(r=>r.map(x=>x===null?'':x.toFixed(2))),texttemplate:'%{text}',hovertemplate:'%{y} with %{x}: %{z:.2f}<extra></extra>',xgap:2,ygap:2,colorbar:{thickness:12,tickfont:{color:c.muted}}}],
    layout(two('Weekly-return correlations among the nine main ETFs, gold (GLD) and SPY','each pair over the weeks both trade (from 2010 to 2020 depending on the ETFs)'),560,{margin:{l:70,r:24,t:100,b:60},xaxis:ax({showgrid:false,showline:false,side:'bottom'}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
}
function drawSel(s){
  const c=th(), K=COL(), o=D.order, S_=D.sel;
  Plotly.react('g_sel',[{type:'bar',x:o,y:o.map(e=>S_[e].h2_median),marker:{color:o.map(e=>S_[e].chosen?K.rule:(isDark()?'#5f5e5a':'#c3c2b7'))},text:o.map(e=>fpp(S_[e].h2_median)),textposition:'outside',cliponaxis:false,
    customdata:o.map(e=>'score '+S_[e].score.toFixed(2)+' (ranks '+S_[e].rank_hl+' / '+S_[e].rank_zc+' / '+S_[e].rank_vr+')<br>second half from '+D.rel[e].mid+': '+S_[e].h2_ahead+' of '+S_[e].n+' windows ahead of 50/50'),hovertemplate:'%{x}: median excess %{text}<br>%{customdata}<extra></extra>'}],
    layout(two('Rule R on the second half of each ETF\'s period: median excess over 50/50 across the seven windows','ETFs ordered by the first-half score (best first); blue = chosen before the run'),400,{xaxis:ax(),yaxis:ax({tickformat:'.1%',title:{text:'median excess (second half)',font:{color:c.ink2}},zeroline:true,zerolinecolor:c.axis})}),CFG);
  document.getElementById('t_sel').innerHTML=table(['ETF','first half','half-life (wk)','z changes a year','variance ratio','ranks','score','chosen','second half: 52 weeks','second half: median of 7 windows (ahead)'],
    o.map(e=>{const q=S_[e],r=D.rel[e];return [e,r.first+' → '+r.mid,f2(r.h1_half_life,0),f2(r.h1_zcross_yr,1),f2(r.h1_vr),q.rank_hl+' / '+q.rank_zc+' / '+q.rank_vr,q.score.toFixed(2),q.chosen?'<b>yes</b>':'',fpp(q.h2_w52),fpp(q.h2_median)+' ('+q.h2_ahead+' of '+q.n+')'];}));
}
function drawRule(s){
  const c=th(), es=D.etfs, ws=D.windows.map(x=>x[0]);
  const z=es.map(e=>ws.map(w=>D.rule[e]&&D.rule[e][w]?D.rule[e][w][5]:null));
  const txt=es.map(e=>ws.map(w=>{const q=D.rule[e]&&D.rule[e][w];return q?(q[5]*100).toFixed(1)+((q[9]<=D.lo||q[9]>=D.hi)?'*':''):'';}));
  const hv=es.map(e=>ws.map(w=>{const q=D.rule[e]&&D.rule[e][w];return q?e+' ÷ SPY, z window '+wl(w)+' ('+q[11]+' → '+q[12]+')<br>excess over 50/50 '+fpp(q[5])+' (first half '+fpp(q[7])+', second '+fpp(q[8])+')<br>'+q[0]+' trades, '+pc(q[1])+' worked; N-A percentile '+q[9].toFixed(1)+', N-B '+q[10].toFixed(1):e+', '+wl(w)+': too short';}));
  const zr=Math.max(0.01,...z.flat().filter(x=>x!==null).map(Math.abs));
  Plotly.react('g_heat',[{type:'heatmap',z:z,x:ws.map(wl),y:es.map(e=>e+D.suffix[D.group[e]]),text:txt,texttemplate:'%{text}',customdata:hv,hovertemplate:'%{customdata}<extra></extra>',zmin:-zr,zmax:zr,colorscale:heatColors(),xgap:2,ygap:2,
    colorbar:{thickness:12,tickformat:'.1%',tickfont:{color:c.muted},title:{text:'excess',side:'right',font:{color:c.ink2}}}}],
    layout(two('Rule R on every ETF ÷ SPY and z window · excess CAGR over 50/50 bought once (pp a year)','red = ahead of 50/50; * = notable against random timing (N-A); main ETFs: ahead in '+D.audit.main_ahead_of_5050+' of '+D.audit.main_pair_windows+' ETF-windows; gold (GLD): '+D.audit.gold_ahead_of_5050+' of '+D.audit.gold_windows),590,
    {margin:{l:110,r:24,t:100,b:40},xaxis:ax({side:'top',showgrid:false,showline:false}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
  const R=D.rule[s.e]||{};
  document.getElementById('t_rule').innerHTML="<p class='note' style='margin:4px 0'><b>"+s.e+' ÷ SPY</b> ('+D.names[s.e]+'), every z window:</p>'+table(['z window','dates','closed trades','worked','mean extra','CAGR','50/50 CAGR','excess','first half','second half','max drawdown','N-A pct','N-B pct'],
    ws.map(w=>{const q=R[w];return q?[w===s.w?'<b>'+wl(w)+'</b>':wl(w),q[11]+' → '+q[12],q[0]+(q[13]?' (+'+q[13]+' open)':''),pc(q[1]),fp(q[2]),fp(q[3]),fp(q[4]),fpp(q[5]),fpp(q[7]),fpp(q[8]),fp(q[6],0),pq(q[9]),pq(q[10])]:[wl(w),'too short','','','','','','','','','','',''];}));
}
function drawPair(s){
  const c=th(), K=COL(), P=D.pair[s.e], z=P.z[s.w], trs=P.trades[s.w]||[], tr=[];
  tr.push({x:P.dates,y:P.ratio,type:'scatter',mode:'lines',line:{color:c.ink2,width:1.3},name:s.e+' ÷ SPY (total return)',xaxis:'x',yaxis:'y',hovertemplate:'%{x}: %{y:.4f}<extra></extra>'});
  tr.push({x:P.dates,y:z,type:'scatter',mode:'lines',line:{color:K.rule,width:1.2},name:'z ('+wl(s.w)+')',xaxis:'x',yaxis:'y2',hovertemplate:'%{x}: z %{y:+.2f}<extra></extra>'});
  [['ETF',K.etf,'hold '+s.e],['SPY',K.spy,'hold SPY']].forEach(([h,col,nm])=>{const sub=trs.filter(t=>t[1]===(h==='ETF'?s.e:'SPY'));
    tr.push({x:sub.map(t=>P.dates[t[0]]),y:sub.map(t=>z[t[0]]),type:'scatter',mode:'markers',marker:{color:col,size:9,line:{width:1,color:c.ink}},name:'entry: '+nm,xaxis:'x',yaxis:'y2',
      customdata:sub.map(t=>t[2]+' → '+(t[3]||'open')+(t[4]===null?'':': extra '+fp(t[4]/10000))),hovertemplate:'%{customdata}<extra></extra>'});});
  const L=layout(two(s.e+' ÷ SPY · the ratio and its z ('+wl(s.w)+') · '+P.dates[0]+' → '+P.dates[P.dates.length-1]+' · '+trs.length+' trades','top: the ratio of total-return prices (log scale); bottom: z with the band at ±1 (dashed); dots = entries'),620,
    {xaxis:ax({anchor:'y2',type:'date'}),yaxis:ax({domain:[0.56,1],type:'log',title:{text:'ratio',font:{color:c.ink2}}}),yaxis2:ax({domain:[0,0.46],title:{text:'z',font:{color:c.ink2}}}),
     shapes:[-1,1].map(y=>({type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:y,y1:y,line:{color:c.axis,width:1,dash:'dash'}})).concat([{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:0,y1:0,line:{color:c.axis,width:1}}])});
  Plotly.react('g_pair',tr,L,CFG);
  const C=D.curves[s.e][s.w];
  if(!C){Plotly.purge('g_eq');Plotly.purge('g_dd');document.getElementById('g_eq').innerHTML="<p class='note'>Too short for this window.</p>";return;}
  const lines=[['rule','Rule R',K.rule,2.4],['bh','50/50 bought once',K.bh,1.6],['etf',s.e+' bought once',K.etf,1.6],['spy','SPY bought once',K.spy,1.6]];
  const dd=a=>{let pk=-Infinity;return a.map(x=>{if(x===null)return null;pk=Math.max(pk,x);return (x/pk-1)*100;});};
  const q=D.rule[s.e][s.w];
  Plotly.react('g_eq',lines.map(([k,nm,col,wd])=>({x:C.t,y:C[k],type:'scatter',mode:'lines',name:nm,line:{color:col,width:wd,dash:k==='bh'?'dash':'solid'},hovertemplate:nm+': %{y:.3f}<extra></extra>'})),
    layout(two(s.e+' ÷ SPY · value of 1 from the rule\'s first day · '+C.t[0]+' → '+C.t[C.t.length-1]+' · z window '+wl(s.w),'Rule R (after costs) against 50/50, the ETF and SPY bought once; weekly decision closes; log scale · rule CAGR '+fp(q[3])+', 50/50 '+fp(q[4])),460,
    {hovermode:'x unified',xaxis:ax({type:'date'}),yaxis:ax({type:'log',title:{text:'value of 1',font:{color:c.ink2}}})}),CFG);
  Plotly.react('g_dd',lines.map(([k,nm,col,wd])=>({x:C.t,y:dd(C[k]),type:'scatter',mode:'lines',name:nm,showlegend:false,line:{color:col,width:wd,dash:k==='bh'?'dash':'solid'},hovertemplate:nm+': %{y:.1f}%<extra></extra>'})),
    layout(two('Drawdown from the running peak · weekly closes','the same lines'),300,{hovermode:'x unified',margin:{l:60,r:24,t:70,b:40},xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%'})}),CFG);
}
function render(){
  const s=S(); try{localStorage.setItem('e08-state',JSON.stringify(s));}catch(e){}
  drawRel(s); drawSel(s); drawRule(s); drawPair(s);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  try{const st=JSON.parse(localStorage.getItem('e08-state')||'null');if(st){[['etf','e'],['zwin','w']].forEach(([id,k])=>{const el=document.getElementById(id);if(st[k]!==undefined&&Array.from(el.options).some(o=>o.value===String(st[k])))el.value=String(st[k]);});}}catch(e){}
  ['etf','zwin'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
