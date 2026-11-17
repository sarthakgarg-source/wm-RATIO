"""
Shared z-score windows for E01, E03 and E04: weekly (13, 26, 52 weeks, on decision-week closes) and daily (20, 60, 120,
250 trading days, on the pair's daily closes), with random walks of the ratio for each frequency built the same way as
E01's (point-in-time residuals with random signs plus the expanding drift, after a warm-up of real data).

The 52-week weekly window is E01's and T02/T03's band; every other window is an addition of 2026-11-17 (E01.md, E03.md,
E04.md dated notes). z_t = (r_t - mean of the last n closes including t) / their sd (n - 1), from the first point with
at least 80% of n closes (the band's rule). Only data up to t enters z_t.
"""
import numpy as np
import pandas as pd

WINDOWS = (("w52", "52 weeks", "W", 52), ("w26", "26 weeks", "W", 26), ("w13", "13 weeks", "W", 13),
           ("d250", "250 days", "D", 250), ("d120", "120 days", "D", 120), ("d60", "60 days", "D", 60), ("d20", "20 days", "D", 20))
WIN = {w[0]: w for w in WINDOWS}
DEFAULT = "w52"
SEP = "§"                                   # trace meta = pair + SEP + window (window-specific traces)
BURN = {"W": 52, "D": 250}                  # warm-up of real data before the random paths start (one year)
SEED = {"W": 101, "D": 111}                 # W = E01's sign seed (identical paths); D = the daily paths' own
UNIT = {"W": "weeks", "D": "trading days"}


def min_obs(n: int) -> int:
    return int(round(0.8 * n))


def series(pf: pd.DataFrame, freq: str) -> pd.DataFrame:
    """The pair's closes and ratio on decision weeks (W) or on every common trading day (D)."""
    d = pf if freq == "D" else pf.loc[pf["decision"]]
    out = d[["ca", "cb"]].copy(); out["ratio"] = out["ca"] / out["cb"]
    return out


def z_of(s: pd.Series, n: int) -> pd.Series:
    roll = s.rolling(n, min_periods=min_obs(n))
    return (s - roll.mean()) / roll.std()


def trailing_z(paths: np.ndarray, n: int) -> np.ndarray:
    """z_of for many paths at once (rows = paths) from running sums; each row divided by its first value first (z does not
    change) so the sums stay near 1."""
    r = np.atleast_2d(paths); r = r / r[:, :1]; L = r.shape[1]
    cs = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r * r, axis=1)], axis=1)
    t = np.arange(L); lo = np.maximum(0, t - n + 1); k = (t - lo + 1).astype(float)
    m = (cs[:, t + 1] - cs[:, lo]) / k; var = (cs2[:, t + 1] - cs2[:, lo] - k * m * m) / np.maximum(k - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r - m) / np.sqrt(np.maximum(var, 0))
    z[:, k < min_obs(n)] = np.nan
    return z


def pit_residuals(g: np.ndarray, burn: int):
    """Expanding drift m_t = mean(g[0 .. t-1]) and residual g_t - m_t for t >= burn (E01's point_in_time, any warm-up)."""
    cs = np.concatenate([[0.0], np.cumsum(g)]); t = np.arange(burn, len(g)); m = cs[t] / t.astype(float)
    return m, g[t] - m


def signs(pid: int, freq: str, D: int, n: int) -> np.ndarray:
    """E01's draw for W (default_rng([101, pair_id]).choice(±1, (D, n))); the daily paths' own for D."""
    return np.random.default_rng([SEED[freq], pid]).choice(np.array([-1.0, 1.0]), size=(D, n))


def random_log_paths(lnr: np.ndarray, pid: int, freq: str, D: int, chunk: int = 250):
    """Yield (row offset, log-ratio paths) for D random paths in chunks: the real first BURN+1 points, then the residuals
    with random signs plus the expanding drift. Row j of the whole set = random path j (0-based)."""
    burn = BURN[freq]; m, z = pit_residuals(np.diff(lnr), burn); S = signs(pid, freq, D, len(z))
    for a in range(0, D, chunk):
        Sc = S[a:a + chunk]
        P = np.concatenate([np.tile(lnr[: burn + 1], (len(Sc), 1)), lnr[burn] + np.cumsum(Sc * z + m, axis=1)], axis=1)
        yield a, P


def options_html(default: str = DEFAULT) -> str:
    return "".join(f"<option value='{w}'{' selected' if w == default else ''}>{lab}{' (daily)' if f == 'D' else ''}</option>" for w, lab, f, _ in WINDOWS)


def select_html(default: str = DEFAULT) -> str:
    return f"<label>z window <select id='zwin'>{options_html(default)}</select></label>"
