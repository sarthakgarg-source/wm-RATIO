"""
E09 (EXPLORATORY): while in a trade, how much did the held index beat its peer?
Design: explore/E09_intrade_extra/E09.md (written before the first run).

    python explore/E09_intrade_extra/intrade.py [--draws 1000]

Every number on the page and in E09_results.md is computed here; the page's script only picks the numbers of the chosen
controls from the embedded results and draws them. Not a finding and not a test of any rule.

Re-run on 2026-11-26 on the latest data (explore/E01_ratio_cycles/latest.py; E09.md, the block of that date): the Indian
and gold pairs on N100, MC100, SC100 and MCX gold from load_india_latest() to 2026-11-20, Micro250 on its earlier data,
all from 2011-01-01; on the gold pairs the rule holds the liquid ETF instead of gold while gold's weekly supertrend is
SELL, in the real runs and in every null draw. The US pairs are unchanged (the engine's own start, as in E08).
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
from analysis.approved_data import GOLD, fingerprints, load_us_etf, load_us_gold  # noqa: E402
import analysis.ratio_engine as RE  # noqa: E402  (its pair start, set per universe)
from analysis.ratio_engine import PAIRS, counts, null_A, null_B, pair_frame  # noqa: E402
import cycles as E1  # noqa: E402  (page styling, percentile)
import zwin as ZW  # noqa: E402  (the z windows)
import variants as VR  # noqa: E402  (the rule through the engine's book)
import latest as LT  # noqa: E402  (the latest Indian data and the gold → liquid switch)

US_ETFS = ["GCC", "FTGC", "COMT", "PDBC", "BCI", "BCD", "COM", "CMDY", "SDCI", "HGER", "NBCM", "PIT", "CERY"]
UNIVERSES = (("index", "Indian index pairs", [p for p in PAIRS if GOLD not in p]), ("gold", "gold pairs", [p for p in PAIRS if GOLD in p]),
             ("us", "commodity ETFs ÷ SPY", [f"{e}|SPY" for e in US_ETFS]),
             ("usgold", "SPY against gold", ["GLD|SPY"]))   # added 2026-11-20 (E09.md); universe_id 3 keeps the others' seeds
US_UNIVERSES = ("us", "usgold"); NSE_UNIVERSES = ("index", "gold")
BUCKETS = (("≤ 13 weeks", 0, 13), ("14–52 weeks", 14, 52), ("> 52 weeks", 53, 10 ** 6))
COST = VR.COST; LO, HI = E1.LO, E1.HI; SEED_NB = 100


def r4(v, d=4):
    return None if v is None or v != v else round(float(v), d)


def r6(v):
    return r4(v, 6)


def nice(p):
    return p.replace("|", " ÷ ")


def fast_extras(targets, kd, fa, fb):
    """The trades of a target sequence exactly as analysis.ratio_engine.book makes them (the first banded week is skipped;
    a switch trades the session after its decision; a decision on the last day cannot trade), and each closed trade's
    held and peer fill-to-fill returns: list of (entry_k, exit_k or None, leg, held_ret, peer_ret)."""
    n = len(fa); held = "N"; tr = []
    for i in range(1, len(targets)):
        t = targets[i]
        if t == held:
            continue
        kt = kd[i] + 1
        if kt >= n:
            break
        tr.append((kt, t)); held = t
    out = []
    for j, (kt, to) in enumerate(tr):
        if to == "N":
            continue
        if j + 1 < len(tr):
            kx = tr[j + 1][0]; ra = fa[kx] / fa[kt] - 1; rb = fb[kx] / fb[kt] - 1
            out.append((kt, kx, to, ra, rb) if to == "A" else (kt, kx, to, rb, ra))
        else:
            out.append((kt, None, to, np.nan, np.nan))
    return out


def fast_extras3(targets, sell, kd, fa, fb, fl, cb):
    """fast_extras with the switch (the latest data; variants.book3): in a week whose decision falls in gold's weekly
    supertrend SELL, gold's weight (the second leg's) is held in the liquid ETF (fl: its level on the pair's days, which is
    also its fill). The rule's trades are book3's; a gold trade's held return compounds gold and the liquid ETF as held
    (its peer is the first leg); an A trade's peer is gold itself. Returns (trades as fast_extras, gold_leg) where
    gold_leg(ke, None) is a gold trade's held return from the entry fill to the last close (for an open trade)."""
    n = len(fa); held = "N"; cur = ("N", bool(sell[0])); tr = []; alloc = [(int(kd[0]) + 1, bool(sell[0]))]
    for i in range(1, len(targets)):
        t = targets[i]; key = (t, bool(sell[i]) and t != "A")
        if key == cur:
            continue
        kt = int(kd[i]) + 1
        if kt >= n:
            break
        alloc.append((kt, key[1])); cur = key
        if t != held:
            tr.append((kt, t)); held = t
    ak = np.array([a[0] for a in alloc]); al = np.array([a[1] for a in alloc])

    def gold_leg(ke, kx):
        end = kx if kx is not None else n - 1; idx = np.flatnonzero((ak >= ke) & (ak < (kx if kx is not None else n)))
        stops = [int(ak[j]) for j in idx] + [end]; g = 1.0
        for j, jj in enumerate(idx):
            s0, s1 = stops[j], stops[j + 1]; lq = bool(al[jj])
            p1 = (fl[s1] if lq else fb[s1]) if kx is not None or j < len(idx) - 1 else (fl[s1] if lq else cb[s1])
            g *= p1 / (fl[s0] if lq else fb[s0])
        return g - 1
    out = []
    for j, (kt, to) in enumerate(tr):
        if to == "N":
            continue
        if j + 1 < len(tr):
            kx = tr[j + 1][0]; ra = fa[kx] / fa[kt] - 1; rb = fb[kx] / fb[kt] - 1
            out.append((kt, kx, to, ra, rb) if to == "A" else (kt, kx, to, gold_leg(kt, kx), ra))
        else:
            out.append((kt, None, to, np.nan, np.nan))
    return out, gold_leg


def summ(closed):
    """Mean extra per trade and compounded in-trade extra of a list of (held_ret, peer_ret)."""
    if not closed:
        return np.nan, np.nan
    h = np.array([c[0] for c in closed]); p = np.array([c[1] for c in closed])
    return float((h - p).mean()), float(np.prod((1 + h) / (1 + p)) - 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=1000); ND = ap.parse_args().draws
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    px_in, px_us = LT.load_india(), load_us_etf(); px_us.update(load_us_gold())
    START0 = dict(RE.START); SW = LT.activate(px_in); START_IN = dict(RE.START)   # the US pairs keep the engine's own start (as in E08)
    checks = {"t03_mismatch": [], "replay_mismatch": [], "e06_unmatched": None, "e06_max_abs_diff": None, "e08_unmatched": None, "e08_max_abs_diff": None, "null_draws": 0, "null_count_mismatch": 0}
    SROWS, TROWS, S, T, POOLED = [], [], {}, {}, {}
    for ui, (u, ulab, pairs) in enumerate(UNIVERSES):
        px = px_us if u in US_UNIVERSES else px_in; RE.START.update(START0 if u in US_UNIVERSES else START_IN)
        for pid, pair in enumerate(pairs):
            pf = pair_frame(px, pair, "plain"); dec = pf.index[pf["decision"]]; days = pf.index; pos = {d: k for k, d in enumerate(days)}
            fa, fb, ca, cb = pf["fa"].to_numpy(), pf["fb"].to_numpy(), pf["ca"].to_numpy(), pf["cb"].to_numpy(); la, lb = pair.split("|")
            S[pair], T[pair] = {}, {}
            swp = VR.SWITCH is not None and VR.SWITCH.is_gold_pair(pf)   # the switch on a gold pair
            if swp:
                fl = SW.liquid_on(days); sell_all = pd.Series(SW.sell_at(dec), index=dec)
            for wi, (w, *_) in enumerate(ZW.WINDOWS):
                z = VR.z_at_decisions(pf, w); bd = z.dropna()
                if len(bd) < 30:
                    S[pair][w] = None; SROWS.append({"universe": u, "pair": pair, "window": w, "status": "too short"}); continue
                r = VR.run_states(pf, z, COST, 1.0, comp=False)
                if u in NSE_UNIVERSES and w == ZW.DEFAULT:
                    checks["t03_mismatch"] += VR.check_against_t03({"pair": pair, **r})
                dd = r["decisions"]; dd = dd[dd["z"].notna()]; targets = dd["state_after"].tolist(); kd = np.array([pos[d] for d in dd.index])
                if swp:
                    sw_ = sell_all.reindex(dd.index).to_numpy(bool)
                    def FX(tg):
                        return fast_extras3(tg, sw_, kd, fa, fb, fl, cb)
                else:
                    def FX(tg):
                        return fast_extras(tg, kd, fa, fb), None
                fx, gleg = FX(targets); ex = r["extras"]
                # the fast replay must be the book's trades
                same = len(fx) == len(ex)
                if same and len(ex):
                    for (ke, kx, leg, h, p), (_, t) in zip(fx, ex.iterrows()):
                        if days[ke] != pd.Timestamp(t["entry_trade"]) or leg != t["held"] or (kx is None) != bool(t["open"]) or (kx is not None and (abs(h - t["held_ret"]) > 1e-12 or abs(p - t["peer_ret"]) > 1e-12)):
                            same = False; break
                if not same:
                    checks["replay_mismatch"].append(f"{pair} {w}")
                closed = [(h, p) for ke, kx, leg, h, p in fx if kx is not None]; mean_x, comp = summ(closed)
                rows, cum = [], 1.0
                for ke, kx, leg, h, p in fx:
                    ed = dec[dec < days[ke]][-1]; held_name, peer_name = (la, lb) if leg == "A" else (lb, la)
                    if kx is None:   # open: to date, from the entry fill to the last close
                        hc = (ca[-1] if leg == "A" else cb[-1]) / (fa[ke] if leg == "A" else fb[ke]) - 1; pc_ = (cb[-1] if leg == "A" else ca[-1]) / (fb[ke] if leg == "A" else fa[ke]) - 1
                        if gleg is not None and leg == "B":
                            hc = gleg(ke, None)   # gold or the liquid ETF as held, to the last close
                        rows.append({"entry_decision": ed, "entry": days[ke], "exit": None, "held": held_name, "peer": peer_name, "held_ret": hc, "peer_ret": pc_, "extra": hc - pc_,
                                     "weeks": int(((dec > ed)).sum()), "days": int((days[-1] - days[ke]).days), "open": True, "cum": np.nan})
                    else:
                        cum *= (1 + h) / (1 + p)
                        rows.append({"entry_decision": ed, "entry": days[ke], "exit": days[kx], "held": held_name, "peer": peer_name, "held_ret": h, "peer_ret": p, "extra": h - p,
                                     "weeks": int(((dec > ed) & (dec < days[kx])).sum()), "days": int((days[kx] - days[ke]).days), "open": False, "cum": cum - 1})
                cl = [x for x in rows if not x["open"]]; op = [x for x in rows if x["open"]]
                yrs_in = sum(x["days"] for x in cl) / 365.25; k0 = pos[dd.index[0]]; period = (days[-1] - days[k0]).days
                annual = (1 + comp) ** (1 / yrs_in) - 1 if cl and yrs_in > 0 else np.nan
                bk = []
                for _, lo_, hi_ in BUCKETS:
                    sel = [x for x in cl if lo_ <= x["weeks"] <= hi_]
                    bk.append([len(sel), float(np.mean([x["extra"] > 0 for x in sel])) if sel else np.nan, float(np.mean([x["extra"] for x in sel])) if sel else np.nan])
                # the nulls: the same trades at random times (N-A) and with random legs (N-B), replayed through the fills
                base = counts(targets); nm = {"N-A": [], "N-B": []}; nc = {"N-A": [], "N-B": []}
                for nname, fn, off in (("N-A", null_A, 0), ("N-B", null_B, SEED_NB)):
                    for drw in range(ND):
                        tg = fn(targets, np.random.default_rng([ui, pid + off, drw, wi])); cc = counts(tg); checks["null_draws"] += 1
                        ok = (cc == base) if nname == "N-A" else (cc["spells"] == base["spells"] and cc["switches"] == base["switches"] and cc["weeks_N"] == base["weeks_N"])
                        checks["null_count_mismatch"] += (not ok)
                        m_, c_ = summ([(h, p) for ke, kx, leg, h, p in FX(tg)[0] if kx is not None]); nm[nname].append(m_); nc[nname].append(c_)
                def pct(x, arr):
                    a = np.array(arr, float); a = a[~np.isnan(a)]
                    return E1.pctile(x, a) if x == x and len(a) else np.nan
                pa_m, pb_m, pa_c, pb_c = pct(mean_x, nm["N-A"]), pct(mean_x, nm["N-B"]), pct(comp, nc["N-A"]), pct(comp, nc["N-B"])
                rec = {"universe": u, "pair": pair, "window": w, "status": "ok", "first": days[k0].date().isoformat(), "last": days[-1].date().isoformat(), "trades": len(cl),
                       "won": float(np.mean([x["extra"] > 0 for x in cl])) if cl else np.nan, "mean_extra": mean_x, "median_extra": float(np.median([x["extra"] for x in cl])) if cl else np.nan,
                       "compounded": comp, "years_in_trade": yrs_in, "time_share": sum(x["days"] for x in cl) / period if period else np.nan, "per_year_in_trade": annual,
                       "na_pct_mean": pa_m, "nb_pct_mean": pb_m, "na_pct_comp": pa_c, "nb_pct_comp": pb_c, "na_median_mean": float(np.nanmedian(nm["N-A"])), "nb_median_mean": float(np.nanmedian(nm["N-B"])),
                       "open_held": op[0]["held"] if op else None, "open_entry": op[0]["entry"].date().isoformat() if op else None, "open_extra_to_date": op[0]["extra"] if op else np.nan}
                for (bl, _, _), b in zip(BUCKETS, bk):
                    rec[f"n {bl}"], rec[f"won {bl}"], rec[f"mean {bl}"] = b
                SROWS.append(rec)
                S[pair][w] = [rec["trades"], r6(rec["won"]), r6(mean_x), r6(rec["median_extra"]), r6(comp), r4(yrs_in, 2), r6(rec["time_share"]), r6(annual), r4(pa_m, 2), r4(pb_m, 2), r4(pa_c, 2), r4(pb_c, 2),
                              rec["first"], rec["last"], [rec["open_held"], rec["open_entry"], r6(rec["open_extra_to_date"])] if op else None, [[b[0], r6(b[1]), r6(b[2])] for b in bk]]
                T[pair][w] = [[x["entry"].date().isoformat(), x["exit"].date().isoformat() if x["exit"] is not None else None, x["held"], r6(x["held_ret"]), r6(x["peer_ret"]), r6(x["extra"]), x["weeks"], x["open"], r6(x["cum"])] for x in rows]
                for x in rows:
                    TROWS.append({"universe": u, "pair": pair, "window": w, "entry_decision": x["entry_decision"].date().isoformat(), "entry_trade": x["entry"].date().isoformat(),
                                  "exit_trade": x["exit"].date().isoformat() if x["exit"] is not None else None, "held": x["held"], "peer": x["peer"], "held_ret": x["held_ret"], "peer_ret": x["peer_ret"],
                                  "extra": x["extra"], "weeks": x["weeks"], "days": x["days"], "open": x["open"], "compounded_after": x["cum"]})
            print(f"  {u:5s} {pair:16s} [{time.time() - t0:4.0f}s]", flush=True)
    TR = pd.DataFrame(TROWS); SU = pd.DataFrame(SROWS)
    # cross-checks with E06 (Indian and gold pairs) and E08 (US pairs): the same trades and extras
    def maxdiff(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); d = np.abs(a - b); d[np.isnan(a) != np.isnan(b)] = np.inf
        return float(np.nanmax(d)) if (~np.isnan(d)).any() else 0.0
    cl_ = TR[~TR["open"]]
    e6 = pd.read_csv(os.path.join(ROOT, "explore", "E06_trade_search", "out", "trades_all.csv")); e6 = e6[~e6["open"].astype(bool)]
    m6 = cl_[cl_["universe"].isin(NSE_UNIVERSES)].merge(e6, on=["window", "pair", "entry_decision"], how="outer", suffixes=("", "_e6"), indicator=True)
    checks["e06_unmatched"] = int((m6["_merge"] != "both").sum()); checks["e06_max_abs_diff"] = maxdiff(m6["extra"], m6["extra_e6"])
    e8 = pd.read_csv(os.path.join(ROOT, "explore", "E08_commodity_spy", "out", "trades.csv")); e8 = e8[~e8["open"].astype(bool)].assign(pair=lambda d: d["etf"] + "|SPY")
    m8 = cl_[cl_["universe"].isin(US_UNIVERSES)].merge(e8, on=["window", "pair", "entry_decision"], how="outer", suffixes=("", "_e8"), indicator=True)
    checks["e08_unmatched"] = int((m8["_merge"] != "both").sum()); checks["e08_max_abs_diff"] = maxdiff(m8["extra"], m8["extra_e8"])
    if checks["t03_mismatch"] or checks["replay_mismatch"] or checks["null_count_mismatch"] or checks["e06_unmatched"] or checks["e08_unmatched"] or not (checks["e06_max_abs_diff"] < 1e-9 and checks["e08_max_abs_diff"] < 1e-9):
        sys.exit(f"checks FAILED: {checks}")
    TR.to_csv(os.path.join(OUT, "trades.csv"), index=False); SU.to_csv(os.path.join(OUT, "intrade_summary.csv"), index=False)
    # pooled, per universe and window (the US pooling uses the nine main ETFs)
    MAIN_US = [f"{e}|SPY" for e in US_ETFS[:9]]
    for u, ulab, pairs in UNIVERSES:
        POOLED[u] = {}
        ps = MAIN_US if u == "us" else pairs
        for w, *_ in ZW.WINDOWS:
            g = SU[(SU["universe"] == u) & (SU["window"] == w) & (SU["status"] == "ok") & SU["pair"].isin(ps)]; tt = cl_[(cl_["universe"] == u) & (cl_["window"] == w) & cl_["pair"].isin(ps)]
            bk = []
            for _, lo_, hi_ in BUCKETS:
                s_ = tt[(tt["weeks"] >= lo_) & (tt["weeks"] <= hi_)]; bk.append([int(len(s_)), r6((s_["extra"] > 0).mean()) if len(s_) else None, r6(s_["extra"].mean()) if len(s_) else None])
            POOLED[u][w] = {"pairs": int(len(g)), "pos_comp": int((g["compounded"] > 0).sum()), "median_per_year": r6(g["per_year_in_trade"].median()), "trades": int(len(tt)),
                            "won": r6((tt["extra"] > 0).mean()) if len(tt) else None, "mean": r6(tt["extra"].mean()) if len(tt) else None, "median": r6(tt["extra"].median()) if len(tt) else None,
                            "na_hi": int((g["na_pct_comp"] >= HI).sum()), "na_lo": int((g["na_pct_comp"] <= LO).sum()), "nb_hi": int((g["nb_pct_comp"] >= HI).sum()), "nb_lo": int((g["nb_pct_comp"] <= LO).sum()), "buckets": bk}
    audit = {"run_date": date.today().isoformat(), "status": "EXPLORATORY", "data": LT.NOTE, "us_pairs_start": START0, "inputs": fingerprints(), "draws_per_null": ND, "seed": f"default_rng([universe_id, pair_id (+{SEED_NB} for N-B), draw, window_id])",
             "checks": checks, "trades": int(len(TR)), "closed": int(len(cl_)), "pooled_w52": {u: {k: v for k, v in POOLED[u][ZW.DEFAULT].items() if k != "buckets"} for u in POOLED}, "seconds": round(time.time() - t0)}
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- results (markdown)
    def fp(v, d=1): return "—" if v is None or v != v else f"{v * 100:+.{d}f}%"
    def pc(v): return "—" if v is None or v != v else f"{v * 100:.0f}%"
    def pq(v): return "—" if v is None or v != v else (f"**{v:.1f}**" if E1.notable(v) else f"{v:.1f}")
    wl = {w: f"{lab}{' (daily)' if f == 'D' else ''}" for w, lab, f, _ in ZW.WINDOWS}
    md = [f"# E09 results — while in a trade, how much did the held index beat its peer? (EXPLORATORY)\n",
          f"_Generated by `explore/E09_intrade_extra/intrade.py` on {audit['run_date']}. Rule R (T03's: ±1, exit at 0) on every z window; extra = the held index's return minus its peer's, entry fill to exit fill, before costs. {ND:,} draws per null. Description, not a finding; percentiles in **bold** are notable (≤ {LO:g} or ≥ {HI:g})._\n",
          f"**Data:** {LT.NOTE} (the Indian and gold pairs; the US pairs unchanged). On the gold pairs the held leg is gold or the liquid ETF as held, in the real runs and in every null draw.\n",
          f"**Checks:** the 52-week Indian and gold runs have the engine's decisions (and, on the gold pairs with the switch off, the engine's book); the trades equal E06's ({checks['e06_max_abs_diff']:.1e}) and E08's ({checks['e08_max_abs_diff']:.1e}); the fast replay equals the book's trades on every pair and window; {checks['null_draws']:,} null draws, every one kept its counts.\n"]
    for u, ulab, pairs in UNIVERSES:
        md += [f"## {ulab} — z window {wl[ZW.DEFAULT]} (T03's)\n",
               "| pair | period | closed trades | held beat peer | mean / median extra per trade | compounded in-trade extra | years in trade (share of time) | extra a year in trade | percentile vs random times (N-A): mean · compounded | vs random legs (N-B): mean · compounded | open trade (to date) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in pairs:
            q = SU[(SU["pair"] == p) & (SU["window"] == ZW.DEFAULT)].iloc[0]
            if q["status"] != "ok":
                md.append(f"| {nice(p)} | too short |||||||||| "); continue
            opn = f"{q['open_held']} since {q['open_entry']}: {fp(q['open_extra_to_date'])}" if isinstance(q["open_held"], str) else "—"
            md.append(f"| {nice(p)} | {q['first']} → {q['last']} | {int(q['trades'])} | {pc(q['won'])} | {fp(q['mean_extra'])} / {fp(q['median_extra'])} | {fp(q['compounded'])} | {q['years_in_trade']:.1f} ({pc(q['time_share'])}) | {fp(q['per_year_in_trade'])} | "
                      f"{pq(q['na_pct_mean'])} · {pq(q['na_pct_comp'])} | {pq(q['nb_pct_mean'])} · {pq(q['nb_pct_comp'])} | {opn} |")
        md += [f"\n**{ulab}, every z window** (US: the nine main ETFs): pairs with a positive compounded in-trade extra; median extra a year in trade; all closed trades: held beat peer, mean extra; notable against random times (compounded) high / low\n",
               "| z window | pairs positive | median extra a year in trade | trades | held beat peer | mean extra | N-A notable high / low | N-B notable high / low | ≤ 13 wk: trades, won, mean | 14–52 wk | > 52 wk |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for w, *_ in ZW.WINDOWS:
            q = POOLED[u][w]; b = q["buckets"]
            md.append(f"| {wl[w]} | {q['pos_comp']} of {q['pairs']} | {fp(q['median_per_year'])} | {q['trades']} | {pc(q['won'])} | {fp(q['mean'])} | {q['na_hi']} / {q['na_lo']} | {q['nb_hi']} / {q['nb_lo']} | "
                      + " | ".join(f"{x[0]}, {pc(x[1])}, {fp(x[2])}" for x in b) + " |")
        md.append("")
    open(os.path.join(OUT, "E09_results.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    # ---------------------------------------------------------------- the page
    DATA = {"univ": [[u, lab, ps] for u, lab, ps in UNIVERSES], "nice": {p: nice(p) for _, _, ps in UNIVERSES for p in ps}, "windows": [[w, lab, f] for w, lab, f, _ in ZW.WINDOWS], "wlab": wl,
            "buckets": [b[0] for b in BUCKETS], "lo": LO, "hi": HI, "draws": ND, "S": S, "T": T, "P": POOLED, "mainUS": MAIN_US, "defaults": {"u": "index", "p": "N100|SC100", "w": ZW.DEFAULT}}
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    def sel(i, lab, opts, default):
        return f"<label>{lab} <select id='{i}'>" + "".join(f"<option value='{html.escape(str(v), quote=True)}'{' selected' if str(v) == str(default) else ''}>{html.escape(t)}</option>" for v, t in opts) + "</select></label>"
    controls = (sel("univ", "Universe", [(u, lab) for u, lab, _ in UNIVERSES], "index") + "<label>Pair <select id='pair'></select></label>"
                + sel("zwin", "z window", [(w, wl[w]) for w, *_ in ZW.WINDOWS], ZW.DEFAULT))
    pw = {u: POOLED[u][ZW.DEFAULT] for u in POOLED}
    key = ("<p style='margin:4px 0'><b>At T03's window (52 weeks), all closed trades:</b> " + "; ".join(
        f"<b>{lab}</b>: the held index beat its peer in {pc(pw[u]['won'])} of {pw[u]['trades']} trades, mean extra {fp(pw[u]['mean'])} (median {fp(pw[u]['median'])}); compounded in-trade extra positive in {pw[u]['pos_comp']} of {pw[u]['pairs']} pairs"
        for u, lab, _ in UNIVERSES) + ".</p>"
        "<p style='margin:4px 0'><b>Where it comes from:</b> " + "; ".join(
        f"{lab}: trades held 13 weeks or less {pc(pw[u]['buckets'][0][1])} won (mean {fp(pw[u]['buckets'][0][2])}), over 52 weeks {pc(pw[u]['buckets'][2][1])} (mean {fp(pw[u]['buckets'][2][2])})" for u, lab, _ in UNIVERSES) + ".</p>")
    from plotly.offline import get_plotlyjs
    H = [E1.PAGE_HEAD.replace("__CSS__", E1.CSS + CSS_EXTRA).replace("<title>Ratio cycles</title>", "<title>In-trade extra</title>"),
         "<header class='bar'><div class='bar-inner'><b>E09 · in-trade extra</b><span class='tag'>EXPLORATORY</span>", controls, "<button id='theme' type='button'>Dark</button></div></header><main>",
         "<h1>While in a trade, how much did the held index beat its peer?</h1>",
         f"<p class='note'><b>Data:</b> {html.escape(LT.NOTE)} (the Indian and gold pairs; the US pairs are unchanged). On the gold pairs the held leg is gold or the liquid ETF as held, in the real runs and in every null draw.</p>",
         f"<p>The rule (T03's): every Friday close, z = (ratio − its trailing mean) ÷ its trailing sd over the chosen z window; from 50/50, the first week z crosses below −1 → hold only the first index, above +1 → only the second; back to 50/50 when z returns to 0. "
         f"<b>Extra</b> = the held index's return minus the other's, from the entry fill to the exit fill, before costs — what the signal's choice of index was worth while the trade was open (against 50/50 the sleeve earns half of it). "
         f"Four universes: the six Indian index pairs (from {LT.DATA_FROM.year}; Micro250 from 2022), the four gold pairs (Indian indices against MCX gold, from {LT.DATA_FROM.year}; the liquid ETF in place of gold on SELL weeks), the thirteen commodity ETFs ÷ SPY (total return, from 2010–2026) and, added on 2026-11-20, "
         f"<b>SPY against gold</b> (GLD ÷ SPY, from 2010). Generated {audit['run_date']}. <b>Description, not a finding.</b></p>",
         "<div class='card'>" + key + "</div>",
         "<h2>1 · Every pair of the chosen universe</h2><p class='note'>Compounded in-trade extra = what holding the chosen index instead of its peer through every closed trade was worth; a year in trade = that, per year spent in trades. Percentiles: among " + f"{ND:,}" + " draws of the same trades at random times (N-A) and with the index chosen by a coin (N-B); ≤ 5 or ≥ 95 = notable.</p>",
         "<div class='card'><div id='g_univ' class='plot'></div><div id='t_univ'></div></div>",
         "<h2>2 · Every pair and z window</h2>", "<div class='card'><div id='g_heat' class='plot'></div></div>",
         "<h2>3 · The chosen pair, trade by trade</h2>", "<div class='card'><div id='g_tr' class='plot'></div><div id='g_cum' class='plot'></div><details><summary>Table view</summary><div id='t_tr'></div></details></div>",
         "<h2>4 · By how long the trade lasted</h2>", "<div class='card'><div id='g_bk' class='plot'></div><div id='t_bk'></div></div>",
         "<h2>Method and limits</h2><div class='card'><ul>"
         "<li><b>Trades.</b> explore/E01_ratio_cycles/variants.py and analysis/ratio_engine.book: fills the session after the decision (OHLC4 for the NSE and MCX legs, the close for the US ETFs and GLD), 10.4 bps per side; the 52-week Indian and gold runs have the engine's decisions (and, with the switch off, the engine's book); on the gold pairs a week whose decision falls in gold's weekly supertrend SELL holds the liquid ETF in place of gold (variants.book3), so a gold trade's held return compounds gold and the liquid ETF as held; the trades equal E06's and E08's. "
         "GLD through analysis/approved_data.load_us_gold() (approved 2026-11-20; no distributions).</li>"
         f"<li><b>Nulls.</b> analysis/ratio_engine.null_A and null_B on the same targets, {ND:,} draws each, seed default_rng([universe_id, pair_id (+100 for N-B), draw, window_id]); each draw's extras replayed from the fill prices exactly as the book trades (on the gold pairs with the switch, as variants.book3 trades; checked equal to the book on every real run); every draw's counts measured.</li>"
         "<li><b>Limits.</b> Extras before costs (a round trip costs the sleeve about 0.2%); the windows' trades overlap; different periods per universe; the US pairs were explored in E08 and the Indian pairs refuted the rule in T02/T03; the Indian series start in 2011; the gold leg has an unconfirmed roll on 2026-01-30 (F-RA-D-OPEN-2); the liquid level has no recorded source before 2024-01-24 (F-RA-D17). Design written before the first run: explore/E09_intrade_extra/E09.md.</li></ul></div>",
         "</main>", f"<script id='e09-data' type='application/json'>{blob}</script>", "<script>" + get_plotlyjs() + "</script>", "<script>" + PAGE_JS + "</script></body></html>"]
    out = os.path.join(OUT, "E09_intrade.html"); open(out, "w", encoding="utf-8").write("\n".join(H))
    print(f"done in {time.time() - t0:.0f}s; page {os.path.getsize(out) / 1e6:.1f} MB → {out}")


CSS_EXTRA = """
.plot{width:100%;min-height:120px}
"""

PAGE_JS = r"""
const D = JSON.parse(document.getElementById('e09-data').textContent);
const LIGHT = {ink:'#0b0b0b', ink2:'#52514e', muted:'#898781', grid:'#e1e0d9', axis:'#c3c2b7'};
const DARK = {ink:'#f0efec', ink2:'#c3c2b7', muted:'#898781', grid:'#2c2c2a', axis:'#383835'};
const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif", CFG = {displaylogo:false, responsive:true};
function isDark(){const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
function th(){return isDark()?DARK:LIGHT;}
function C1(){return isDark()?'#3987e5':'#2a78d6';} function C2(){return isDark()?'#d95926':'#eb6834';} function NEU(){return isDark()?'#5f5e5a':'#c3c2b7';}
function v(id){return document.getElementById(id).value;}
function wl(w){return D.wlab[w]||w;}
function pc(x){return x===null||x===undefined?'—':(x*100).toFixed(0)+'%';}
function fp(x,d){if(x===null||x===undefined)return '—';d=d===undefined?1:d;let s=(x*100).toFixed(d);if(Number(s)===0)s=(0).toFixed(d);return (Number(s)>=0?'+':'')+s+'%';}
function pq(x){if(x===null||x===undefined)return '—';const s=x.toFixed(1);return (x<=D.lo||x>=D.hi)?'<b>'+s+'</b>':s;}
function two(h,s){return h+"<br><span style='font-size:12px'>"+s+"</span>";}
function layout(title,height,extra){const c=th();return Object.assign({template:'none',height:height,autosize:true,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:FONT,size:12,color:c.ink},
  title:{text:title,x:0,xanchor:'left',y:1,yref:'container',yanchor:'top',pad:{t:10},automargin:true,font:{size:14,color:c.ink}},margin:{l:60,r:24,t:100,b:50},hovermode:'closest',
  legend:{orientation:'h',x:0,y:1.02,yanchor:'bottom',traceorder:'normal',font:{color:c.ink2,size:11}}},extra||{});}
function ax(extra){const c=th();return Object.assign({gridcolor:c.grid,linecolor:c.axis,showline:true,zeroline:false,tickfont:{color:c.muted},title:{font:{color:c.ink2}}},extra||{});}
function table(head,rows){return "<div class='scroll'><table><thead><tr>"+head.map(h=>'<th>'+h+'</th>').join('')+"</tr></thead><tbody>"+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+"</tbody></table></div>";}
function heatColors(){return isDark()?[[0,'#86b6ef'],[0.2,'#3987e5'],[0.4,'#34465e'],[0.5,'#383835'],[0.6,'#5e3a3a'],[0.8,'#e66767'],[1,'#f29a9a']]:[[0,'#104281'],[0.2,'#2a78d6'],[0.4,'#cde2fb'],[0.5,'#f0efec'],[0.6,'#f9d3d2'],[0.8,'#e34948'],[1,'#a32b2a']];}
function U(u){return D.univ.find(x=>x[0]===u);}
function S_(){return {u:v('univ'), p:v('pair'), w:v('zwin')};}
function fillPairs(keep){const sel=document.getElementById('pair'), ps=U(v('univ'))[2];sel.innerHTML=ps.map(p=>"<option value='"+p+"'>"+D.nice[p]+'</option>').join('');sel.value=ps.includes(keep)?keep:ps[0];}
// S record: [trades, won, mean, median, compounded, years_in, time_share, per_year, na_mean, nb_mean, na_comp, nb_comp, first, last, open, buckets]
function drawUniv(s){
  const c=th(), u=U(s.u), ps=u[2], rec=ps.map(p=>D.S[p][s.w]);
  const y=rec.map(r=>r?r[7]:null), ymax=Math.max(0,...y.filter(x=>x!==null).map(Math.abs));
  Plotly.react('g_univ',[{type:'bar',x:ps.map(p=>D.nice[p]),y:y,width:ps.length<3?0.3:undefined,marker:{color:ps.map((p,i)=>p===s.p?C1():(y[i]!==null&&y[i]>=0?(isDark()?'#5e3a3a':'#f9d3d2'):NEU()))},
    text:rec.map(r=>r?fp(r[7])+((r[10]<=D.lo||r[10]>=D.hi)?' *':''):'too short'),textposition:'outside',cliponaxis:false,
    customdata:rec.map((r,i)=>r?D.nice[ps[i]]+' · '+r[12]+' → '+r[13]+'<br>'+r[0]+' trades, held beat peer in '+pc(r[1])+'; mean extra '+fp(r[2])+'<br>compounded '+fp(r[4])+' over '+r[5].toFixed(1)+' years in trade; N-A percentile '+r[10].toFixed(1):''),hovertemplate:'%{customdata}<extra></extra>'}],
    layout(two(u[1]+' · extra a year while in a trade (held index − its peer) · z window '+wl(s.w),'compounded over the closed trades, per year spent in trades, before costs; blue = the pair picked at the top; * = notable against random times (N-A)'),420,
    {xaxis:ax({tickangle:ps.length>8?-30:0}),yaxis:ax({tickformat:ymax<0.02?'.2%':'.1%',title:{text:'extra a year in trade',font:{color:c.ink2}},zeroline:true,zerolinecolor:c.axis})}),CFG);
  document.getElementById('t_univ').innerHTML=table(['pair','period','closed trades','held beat peer','mean / median extra','compounded','years in trade (share of time)','a year in trade','vs random times: mean · compounded','vs random legs: mean · compounded','open trade (to date)'],
    ps.map(p=>{const r=D.S[p][s.w];return r?[p===s.p?'<b>'+D.nice[p]+'</b>':D.nice[p],r[12]+' → '+r[13],r[0],pc(r[1]),fp(r[2])+' / '+fp(r[3]),fp(r[4]),r[5].toFixed(1)+' ('+pc(r[6])+')',fp(r[7]),pq(r[8])+' · '+pq(r[10]),pq(r[9])+' · '+pq(r[11]),r[14]?r[14][0]+' since '+r[14][1]+': '+fp(r[14][2]):'—']:[D.nice[p],'too short','','','','','','','','',''];}))+
    "<p class='note'>"+(s.u==='us'?'The four short ETFs (HGER, NBCM, PIT, CERY) have 1–3 years.':(s.u==='usgold'?'One pair: GLD (SPDR Gold Shares) against SPY; the held leg is gold or SPY.':''))+'</p>';
}
function drawHeat(s){
  const c=th(), ws=D.windows.map(x=>x[0]), rows=[]; D.univ.forEach(u=>u[2].forEach(p=>rows.push([u[0],p])));
  const z=rows.map(([u,p])=>ws.map(w=>D.S[p][w]?D.S[p][w][7]:null));
  const tx=rows.map(([u,p])=>ws.map(w=>{const r=D.S[p][w];return r?(r[7]*100).toFixed(1)+((r[10]<=D.lo||r[10]>=D.hi)?'*':''):'';}));
  const hv=rows.map(([u,p])=>ws.map(w=>{const r=D.S[p][w];return r?D.nice[p]+', '+wl(w)+' ('+r[12]+' → '+r[13]+')<br>extra a year in trade '+fp(r[7])+'; compounded '+fp(r[4])+'<br>'+r[0]+' trades, '+pc(r[1])+' won, mean '+fp(r[2])+'<br>N-A percentile '+r[10].toFixed(1)+', N-B '+r[11].toFixed(1):D.nice[p]+', '+wl(w)+': too short';}));
  const zr=Math.max(0.01,...z.flat().filter(x=>x!==null).map(Math.abs));
  Plotly.react('g_heat',[{type:'heatmap',z:z,x:ws.map(wl),y:rows.map(([u,p])=>D.nice[p]),text:tx,texttemplate:'%{text}',customdata:hv,hovertemplate:'%{customdata}<extra></extra>',zmin:-zr,zmax:zr,colorscale:heatColors(),xgap:2,ygap:2,
    colorbar:{thickness:12,tickformat:'.0%',tickfont:{color:c.muted},title:{text:'a year in trade',side:'right',font:{color:c.ink2}}}}],
    layout(two('Extra a year while in a trade, every pair and z window (%)','red = the held index beat its peer; * = notable against random times (N-A, compounded); rows: Indian index pairs, gold pairs, commodity ETFs ÷ SPY, SPY against gold (GLD)'),790,
    {margin:{l:150,r:24,t:100,b:30},xaxis:ax({side:'top',showgrid:false,showline:false}),yaxis:ax({autorange:'reversed',showgrid:false,showline:false})}),CFG);
}
function drawPair(s){
  const c=th(), r=D.S[s.p][s.w], T=D.T[s.p][s.w]||[], legs=s.p.split('|');
  if(!r){['g_tr','g_cum'].forEach(id=>Plotly.purge(id));document.getElementById('t_tr').innerHTML="<p class='note'>Too short for this window.</p>";return;}
  const DAY=86400000, tr=[];
  [[legs[0],C1()],[legs[1],C2()]].forEach(([lg,col])=>{const sub=T.filter(t=>t[2]===lg);
    tr.push({type:'bar',x:sub.map(t=>t[0]),y:sub.map(t=>t[5]*100),width:sub.map(()=>14*DAY),marker:{color:col,opacity:sub.map(t=>t[7]?0.45:1)},name:'held '+lg,
      customdata:sub.map(t=>t[0]+' → '+(t[1]||'open (to date)')+', '+t[6]+' weeks<br>'+lg+' '+fp(t[3])+' vs '+(lg===legs[0]?legs[1]:legs[0])+' '+fp(t[4])),hovertemplate:'%{customdata}<br>extra %{y:+.1f}%<extra></extra>'});});
  Plotly.react('g_tr',tr,layout(two(D.nice[s.p]+' · each trade\'s extra (held − peer) at its entry date · z window '+wl(s.w)+' · '+r[12]+' → '+r[13],
    r[0]+' closed trades: held beat peer in '+pc(r[1])+', mean '+fp(r[2])+', median '+fp(r[3])+'; colour = the index held; faded = still open (to date)'),420,
    {barmode:'overlay',xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%',title:{text:'extra',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}]}),CFG);
  const cl=T.filter(t=>!t[7]);
  Plotly.react('g_cum',[{x:[r[12]].concat(cl.map(t=>t[1])),y:[0].concat(cl.map(t=>t[8]*100)),type:'scatter',mode:'lines+markers',line:{color:C1(),width:2,shape:'hv'},marker:{size:6},name:'compounded in-trade extra',
    customdata:[''].concat(cl.map(t=>'after the trade '+t[0]+' → '+t[1]+' (held '+t[2]+')')),hovertemplate:'%{x}: %{y:+.1f}%<br>%{customdata}<extra></extra>'}],
    layout(two(D.nice[s.p]+' · compounded in-trade extra after each closed trade · '+fp(r[4])+' over '+r[5].toFixed(1)+' years in trade ('+fp(r[7])+' a year)',
      'the value of holding the chosen index instead of its peer through the trades only (flat between trades); N-A percentile '+r[10].toFixed(1)+', N-B '+r[11].toFixed(1)),360,
    {xaxis:ax({type:'date'}),yaxis:ax({ticksuffix:'%',title:{text:'compounded extra',font:{color:c.ink2}}}),shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:0,y1:0,line:{color:c.axis,width:1}}]}),CFG);
  document.getElementById('t_tr').innerHTML=table(['entered','exited','weeks','held','held return','peer return','extra','compounded after'],T.map(t=>[t[0],t[1]||'open',t[6],t[2],fp(t[3]),fp(t[4]),fp(t[5]),t[7]?'(open)':fp(t[8])]));
}
function drawBk(s){
  const c=th(), P=D.P[s.u][s.w], b=P.buckets, lab=D.buckets;
  Plotly.react('g_bk',[{type:'bar',x:lab,y:b.map(x=>x[1]===null?null:x[1]*100),marker:{color:C1()},name:'held beat peer',xaxis:'x',yaxis:'y',text:b.map(x=>x[1]===null?'':pc(x[1])+' of '+x[0]),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: %{text}<extra></extra>'},
    {type:'bar',x:lab,y:b.map(x=>x[2]===null?null:x[2]*100),marker:{color:C2()},name:'mean extra',xaxis:'x2',yaxis:'y2',text:b.map(x=>fp(x[2])),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: mean extra %{text}<extra></extra>'}],
    layout(two(U(s.u)[1]+(s.u==='us'?' (nine main ETFs)':'')+' · closed trades by how long they lasted · z window '+wl(s.w),'left: share where the held index beat its peer; right: mean extra per trade'),420,
    {xaxis:ax({domain:[0,0.46],anchor:'y'}),yaxis:ax({anchor:'x',ticksuffix:'%',rangemode:'tozero',title:{text:'held beat peer',font:{color:c.ink2}}}),xaxis2:ax({domain:[0.54,1],anchor:'y2'}),yaxis2:ax({anchor:'x2',ticksuffix:'%',title:{text:'mean extra',font:{color:c.ink2}}}),showlegend:false}),CFG);
  document.getElementById('t_bk').innerHTML=table(['held for','trades','held beat peer','mean extra'],lab.map((l,i)=>[l,b[i][0],pc(b[i][1]),fp(b[i][2])]))+
    "<p class='note'>All closed trades: "+P.trades+', held beat peer in '+pc(P.won)+', mean extra '+fp(P.mean)+', median '+fp(P.median)+'; compounded in-trade extra positive in '+P.pos_comp+' of '+P.pairs+' pairs; notable against random times '+P.na_hi+' high / '+P.na_lo+' low.</p>';
}
function render(){
  const s=S_(); try{localStorage.setItem('e09-state',JSON.stringify(s));}catch(e){}
  drawUniv(s); drawHeat(s); drawPair(s); drawBk(s);
  document.getElementById('theme').textContent=isDark()?'Light':'Dark';
}
window.addEventListener('load',()=>{
  let st=null; try{st=JSON.parse(localStorage.getItem('e09-state')||'null');}catch(e){}
  if(st&&st.u&&U(st.u))document.getElementById('univ').value=st.u;
  fillPairs(st&&st.p?st.p:D.defaults.p);
  if(st&&st.w)document.getElementById('zwin').value=st.w;
  document.getElementById('univ').addEventListener('change',()=>{fillPairs(v('pair'));render();});
  ['pair','zwin'].forEach(id=>document.getElementById(id).addEventListener('change',render));
  document.getElementById('theme').addEventListener('click',()=>{document.documentElement.dataset.theme=isDark()?'light':'dark';render();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
  render();
});
"""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
