"""
Gate V1: an independent recomputation of every NAV in an evidence folder. Imports NOTHING from ratio_engine.py.

    python analysis/verify_ratio.py tests/T02/evidence/<run>              # verify; exit 1 on any mismatch or if nothing compared
    python analysis/verify_ratio.py tests/T02/evidence/<run> --selftest   # also prove it fails on corrupted copies (T03 the same)

The band is drawn on the plain ratio P_a / P_b and a trade ends only at its exit rule (no maximum hold), as the locked
designs say; the decisions are re-derived that way, and an audit.json that names another form is a failure.

Route: the engine holds UNITS and values them; this program holds VALUE WEIGHTS and compounds daily returns from the closes,
with the fill-to-close legs on trade days. The two agree only if the fills, the costs, the calendar and the decisions are
all handled identically. Also checked: the closed forms (hold_a = ca_t / (fa_1 (1 + c)); 50/50 buy-and-hold; always_neutral
equals 50/50 buy-and-hold), and the decisions themselves (the band and the state machine, re-derived from the data).
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.approved_data import GOLD, load_daily  # noqa: E402  (the data boundary, not the engine)

TOL = 1e-9


def pair_days(px, pair, start):
    a, b = pair.split("|"); idx = px[a].index.intersection(px[b].index); idx = idx[idx >= pd.Timestamp(start)]
    return pd.DataFrame({"ca": px[a].loc[idx, "close"], "cb": px[b].loc[idx, "close"], "fa": px[a].loc[idx, "fill"], "fb": px[b].loc[idx, "fill"]}, index=idx)


def redo_decisions(days: pd.DataFrame, window: int, exit_rule: str, sd_mult: float, min_obs_frac: float) -> pd.DataFrame:
    """The state machine, written again from the design text (T02.md / T03.md §3 steps 4–5), on the weekly closes: the band
    on ca / cb; a trade ends only at its exit rule."""
    iso = days.index.isocalendar(); key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).values
    last = ~pd.Series(key, index=days.index).duplicated(keep="last")
    wk = days[last.values]; dates = wk.index
    lr = (wk["ca"] / wk["cb"]).to_numpy()
    min_obs = int(round(min_obs_frac * window)); state = "N"; since = None; prev = None; out = []
    for i in range(len(lr)):
        if i + 1 < min_obs:
            out.append((dates[i], np.nan, np.nan, state)); prev = None; continue
        w = lr[max(0, i + 1 - window): i + 1]; m = w.mean(); s = w.std(ddof=1); x = lr[i]
        pos = -1 if x < m - sd_mult * s else (1 if x > m + sd_mult * s else 0)
        if state != "N":
            if state == "A":
                back = x >= m if exit_rule == "E1" else pos == 1
            else:
                back = x <= m if exit_rule == "E1" else pos == -1
            if back:
                state = "N"; since = None
        if state == "N":
            if pos == -1 and prev in (0, 1):
                state = "A"; since = i
            elif pos == 1 and prev in (0, -1):
                state = "B"; since = i
        out.append((dates[i], m, s, state)); prev = pos
    return pd.DataFrame(out, columns=["date", "mean", "sd", "state_after"]).set_index("date")


def value_path(days: pd.DataFrame, dec: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Weights-and-returns route. dec: banded decision weeks with state_after; the first row is the 50/50 purchase."""
    c = cost_bps / 2 / 1e4
    ca, cb, fa, fb = days["ca"].to_numpy(), days["cb"].to_numpy(), days["fa"].to_numpy(), days["fb"].to_numpy()
    k = {d: i for i, d in enumerate(days.index)}
    k0 = k[dec.index[0]]; nav = np.full(len(days), np.nan); nav[k0] = 1.0
    # buy 50/50 at the fill of day k0+1, then mark at the close: V = 0.5/(1+c) * ca/fa + 0.5/(1+c) * cb/fb
    wa = 0.5 / (1 + c); wb = 0.5 / (1 + c)                      # value in each leg per unit of starting money, at the fill
    V = wa * ca[k0 + 1] / fa[k0 + 1] + wb * cb[k0 + 1] / fb[k0 + 1]; nav[k0 + 1] = V
    va = wa * ca[k0 + 1] / fa[k0 + 1]; vb = wb * cb[k0 + 1] / fb[k0 + 1]      # value in each leg at the close
    held = "N"; pending = {}
    for d, row in dec.iloc[1:].iterrows():
        kd = k[d]; tgt = row["state_after"]
        if tgt != held and kd + 1 < len(days):
            pending[kd + 1] = tgt; held = tgt
    for i in range(k0 + 2, len(days)):
        if i in pending:
            tgt = pending[i]
            va_f = va * fa[i] / ca[i - 1]; vb_f = vb * fb[i] / cb[i - 1]              # values at the fill
            if tgt == "A":
                va_f, vb_f = va_f + vb_f * (1 - c) / (1 + c), 0.0
            elif tgt == "B":
                va_f, vb_f = 0.0, vb_f + va_f * (1 - c) / (1 + c)
            else:
                if va_f >= vb_f:
                    x = (va_f - vb_f) / (1 + (1 - c) / (1 + c)); va_f, vb_f = va_f - x, vb_f + x * (1 - c) / (1 + c)
                else:
                    x = (vb_f - va_f) / (1 + (1 - c) / (1 + c)); vb_f, va_f = vb_f - x, va_f + x * (1 - c) / (1 + c)
            va = va_f * ca[i] / fa[i]; vb = vb_f * cb[i] / fb[i]
        else:
            va = va * ca[i] / ca[i - 1]; vb = vb * cb[i] / cb[i - 1]
        nav[i] = va + vb
    return pd.Series(nav, index=days.index).iloc[k0:]


def rebal_path(days: pd.DataFrame, dec_dates, cost_bps: float, k0_date) -> pd.Series:
    c = cost_bps / 2 / 1e4; sub = days.loc[k0_date:]
    ca, cb, fa, fb = sub["ca"].to_numpy(), sub["cb"].to_numpy(), sub["fa"].to_numpy(), sub["fb"].to_numpy(); k = {d: i for i, d in enumerate(sub.index)}
    nav = np.full(len(sub), np.nan); nav[0] = 1.0
    va = 0.5 / (1 + c) * ca[1] / fa[1]; vb = 0.5 / (1 + c) * cb[1] / fb[1]; nav[1] = va + vb
    pend = set(k[d] + 1 for d in dec_dates[1:] if d in k and k[d] + 1 < len(sub))
    for i in range(2, len(sub)):
        if i in pend:
            va_f = va * fa[i] / ca[i - 1]; vb_f = vb * fb[i] / cb[i - 1]
            if va_f >= vb_f:
                x = (va_f - vb_f) / (1 + (1 - c) / (1 + c)); va_f, vb_f = va_f - x, vb_f + x * (1 - c) / (1 + c)
            else:
                x = (vb_f - va_f) / (1 + (1 - c) / (1 + c)); vb_f, va_f = vb_f - x, va_f + x * (1 - c) / (1 + c)
            va = va_f * ca[i] / fa[i]; vb = vb_f * cb[i] / fb[i]
        else:
            va = va * ca[i] / ca[i - 1]; vb = vb * cb[i] / cb[i - 1]
        nav[i] = va + vb
    return pd.Series(nav, index=sub.index)


def stats(nav):
    nav = nav.dropna(); days = (nav.index[-1] - nav.index[0]).days
    return nav.iloc[-1] ** (365.25 / days) - 1, float((nav / nav.cummax() - 1).min()), float(nav.iloc[-1])


def verify(ev: str, quiet=False) -> tuple[int, int]:
    audit = json.load(open(os.path.join(ev, "audit.json")))
    px = load_daily(); summ = pd.read_csv(os.path.join(ev, "summary.csv")); decs = pd.read_csv(os.path.join(ev, "decisions.csv"), parse_dates=["date"])
    compared = 0; failures = []
    if audit.get("ratio") != "plain":
        failures.append(f"audit.json says the band was drawn on the {audit.get('ratio')} ratio, the design says plain")
    if audit.get("max_hold_weeks") is not None:
        failures.append(f"audit.json says there was a maximum hold of {audit.get('max_hold_weeks')} weeks, the design has none")
    navs = sorted(f for f in os.listdir(ev) if f.startswith("nav_") and f.endswith(".csv"))
    for f in navs:
        stem = f[4:-4]; parts = stem.split("_"); cost = float(parts[-1]); ex = parts[-2]; window = int(parts[-3]); pair = "|".join(parts[:-3]).replace("_", "|")
        pair = parts[0] + "|" + parts[1]
        days = pair_days(px, pair, audit["start"]["gold"] if GOLD in pair else audit["start"]["index"])
        eng = pd.read_csv(os.path.join(ev, f), index_col=0, parse_dates=True)
        d_eng = decs[(decs["pair"] == pair) & (decs["window"] == window) & (decs["exit"] == ex)].set_index("date")
        d_me = redo_decisions(days, window, ex, audit["sd_mult"], audit["min_obs_frac"])
        banded = d_me[d_me["mean"].notna()]
        # 1. the decisions themselves
        same = d_eng["state_after"].reindex(d_me.index).fillna("?").to_numpy()
        if not (same == d_me["state_after"].to_numpy()).all():
            failures.append(f"{stem}: decisions differ on {(same != d_me['state_after'].to_numpy()).sum()} weeks")
        compared += len(d_me)
        # 2. the strategy path by the value route
        mine = value_path(days, banded, cost); e = eng["nav"].reindex(mine.index)
        diff = float((mine - e).abs().max())
        if not diff < 1e-8:
            failures.append(f"{stem}: strategy NAV path differs, max |diff| {diff:.2e}")
        for stat, (a, b) in zip(("cagr", "maxdd", "final"), zip(stats(mine), stats(e))):
            if not abs(a - b) < TOL:
                failures.append(f"{stem}: strategy {stat} {b:.10f} vs recomputed {a:.10f}")
        compared += 4
        # 3. closed forms for the comparators
        c = cost / 2 / 1e4; sub = days.loc[banded.index[0]:]
        ha = pd.Series(np.nan, index=sub.index); ha.iloc[0] = 1.0; ha.iloc[1:] = (sub["ca"].iloc[1:] / (sub["fa"].iloc[1] * (1 + c))).to_numpy()
        hb = pd.Series(np.nan, index=sub.index); hb.iloc[0] = 1.0; hb.iloc[1:] = (sub["cb"].iloc[1:] / (sub["fb"].iloc[1] * (1 + c))).to_numpy()
        bh = pd.Series(np.nan, index=sub.index); bh.iloc[0] = 1.0; bh.iloc[1:] = (0.5 * sub["ca"].iloc[1:] / (sub["fa"].iloc[1] * (1 + c)) + 0.5 * sub["cb"].iloc[1:] / (sub["fb"].iloc[1] * (1 + c))).to_numpy()
        for name, mine_s in (("hold_a", ha), ("hold_b", hb), ("5050_buyhold", bh), ("always_neutral", bh), ("5050_rebal", rebal_path(days, banded.index, cost, banded.index[0]))):
            e_s = eng[name].reindex(mine_s.index); diff = float((mine_s - e_s).abs().max())
            if not diff < 1e-8:
                failures.append(f"{stem}: {name} differs, max |diff| {diff:.2e}")
            compared += 1
        # 4. the summary rows carry the same numbers as the nav file
        srow = summ[(summ["pair"] == pair) & (summ["window"] == window) & (summ["exit"] == ex) & (summ["cost_bps"] == cost) & (summ["series"] == "strategy")]
        if len(srow) != 1 or not abs(float(srow["final_nav"].iloc[0]) - stats(e)[2]) < TOL:
            failures.append(f"{stem}: summary row missing or final_nav differs")
        compared += 1
    if not quiet:
        for x in failures:
            print("FAIL", x)
        print(f"verify: {len(navs)} nav files, {compared} comparisons, {len(failures)} failures")
    return compared, len(failures)


def selftest(ev: str) -> bool:
    """The verifier must FAIL on a corrupted copy: (a) one decision flipped; (b) one NAV file scaled by 1.001."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "a"); shutil.copytree(ev, cp)
        d = pd.read_csv(os.path.join(cp, "decisions.csv")); i = d.index[d["state_after"] == "A"][0]; d.loc[i, "state_after"] = "B"; d.to_csv(os.path.join(cp, "decisions.csv"), index=False)
        _, nf = verify(cp, quiet=True); print(f"selftest (a) flipped decision: {nf} failure(s) — {'caught' if nf else 'NOT CAUGHT'}"); ok &= nf > 0
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "b"); shutil.copytree(ev, cp)
        f = sorted(x for x in os.listdir(cp) if x.startswith("nav_"))[0]; n = pd.read_csv(os.path.join(cp, f), index_col=0); n["nav"] *= 1.001; n.to_csv(os.path.join(cp, f))
        _, nf = verify(cp, quiet=True); print(f"selftest (b) scaled NAV: {nf} failure(s) — {'caught' if nf else 'NOT CAUGHT'}"); ok &= nf > 0
    return ok


if __name__ == "__main__":
    ev = sys.argv[1]
    compared, nf = verify(ev)
    if "--selftest" in sys.argv:
        if not selftest(ev):
            sys.exit("verifier selftest FAILED: a corrupted copy passed")
    if compared == 0:
        sys.exit("verify: nothing compared")
    sys.exit(1 if nf else 0)
