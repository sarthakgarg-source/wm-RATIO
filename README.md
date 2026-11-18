# wm-RATIO — ratio analysis of the NSE index pairs, under the research playbook

**The question (T02, T03):** on an index pair, does holding only the leg that has lagged — from the first week the plain
ratio a ÷ b is more than one standard deviation below (or above) its trailing-52-week mean, until the ratio is back at
the mean, however long that takes — earn more and fall less than the same trades scattered at random? Long-only switch;
six index pairs decide, four gold pairs reported. A band on the ratio, entry at the first touch, exit at the mean (and,
as a second exit, at the opposite band); **no maximum hold**.

**Hard rules** (RESEARCH_PLAYBOOK.md, kept here in short):
1. Tests read data only through `analysis/approved_data.py`; the file must be on `data/APPROVED_DATA.md` and its
   sha256 must match at the moment of opening. Raw files are read-only. A refresh needs the data owner and a new review.
2. Sections 1–3 of a test's write-up are locked with a hash (`check_prereg.py`) before the first run and never change
   afterwards; anything learned later goes in a dated "Added after the run" block.
3. Decide on the decision day's close; trade the next session at `fill` (OHLC4, or the close on a stale-OHL row);
   10.4 bps per side on value traded; the book is valued every day; the first trade's cost is inside the measured path.
4. Every number in a write-up or on a page is printed by a script from `evidence/<run>/`; nothing is typed.
5. The random draws go through the same book code as the strategy; only the tested ingredient changes; seeds come
   from a stated formula; the null's promises (same switches, same weeks in each leg) are measured, not assumed.
6. An independent verifier that imports nothing from the engine must agree, and must fail on a corrupted copy.
7. If any gate fails, stop: read nothing, fix it, rerun into a new evidence folder.
8. No person's name anywhere in the repo — code, comments, docs, commits (write "the review of <date>", "the desk").
9. Finding IDs: `F-RA-S##` (strategy), `F-RA-M##` (method), `F-RA-D##` (data). **Counter:** S06, M02, D11 (+ 2 open).

**Current tests:** T02 and T03 are the same rule under two numbers (the desk's choice of 2026-11-17): T02 judged at the
95th percentile, T03 at the 98th. On 2026-11-17 the earlier log-ratio test and the time limit on each trade were removed
from the repository at the desk's request; both are in git under the tag `archive/2026-11-17-earlier-versions`. T02 is
rerun under its rewritten design; T03's evidence of 2026-11-17 is re-analysed under its rewritten text. Each test's gate
V4b checks that the two make the same decisions and the same numbers. The next free test number is T04.

**Layout**
```
data/        raw files (read-only), data_review.py → DATA_REVIEW.md, APPROVED_DATA.md, check_approved_data.py
analysis/    approved_data.py (the boundary), ratio_engine.py (engine + nulls), verify_ratio.py (independent checker)
tests/T02/   PLAN.md, T02.md (§1–3 locked), TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py, report.py,
             check_page.py, T02_results.html, evidence/<run>/
tests/T03/   the same files for T03 (the same rule at the 98th percentile)
tests/       check_names.py (hygiene: no person's name; names from WM_REDACT_NAMES in the git-ignored .env)
explore/     EXPLORATORY studies — description, never findings, never quoted as evidence for a rule:
             E01_ratio_cycles/ (E01.md design; cycles.py → out/E01_cycles.html, E01_results.md, CSVs) — cyclicality of
             the ten ratio series against sign-flipped random walks, point-in-time (expanding drift and threshold, counted
             look-ahead check) (2026-11-13), plus the z-score of the ratio (the band in units of its trailing-52-week sd;
             description only) (2026-11-17). peaks_troughs_xlsx.py → out/E01_peaks_troughs_explained.xlsx: the peak/trough
             rule as live Excel formulas (toy example + N100 ÷ SC100), matched against cycles.py
             E02_ratio_segments/ (E02.md design; segments.py → out/E02_segments.html, E02_results.md, CSVs) — HINDSIGHT
             segmentation of the daily ratios with the desk's market-segregation method (scipy prominence → peak-to-peak
             regression), trend in % a year, grid of settings, cross-checked against E01 (2026-11-16)
             E03_trade_conditions/ (E03.md design; conditions.py → out/E03_conditions.html, E03_results.md, CSVs) — forward
             returns of the cheaper / richer leg by z bucket against E01's random walks, and the market's condition at each
             T03 trade (correlation, market regime, volatility, entry depth and speed) related to the trade's outcome with
             within-pair permutation nulls, plus trade counts and lengths over time (2026-11-17)
             E04_signal_strength/ (E04.md design; strength.py → out/E04_signal.html, E04_results.md, CSVs) — T03's
             outperformance against 50/50, what follows a band touch, the z-score's information coefficient, and the T03
             rule run on E01's random walks: signal or mechanics? (2026-11-17)
             E01, E03 and E04 have a z window dropdown: 13 / 26 / 52 weeks (weekly closes) and 20 / 60 / 120 / 250 trading days
             (daily closes, daily random walks); shared code explore/E01_ratio_cycles/zwin.py (2026-11-17). On E03 and E04 it
             switches every section: the trade sections run T03's rule on the chosen window's band (variants.py; the 52-week
             run is checked to be T03 exactly). E03 also has a correlation-period dropdown (the legs' correlation over 13 / 26 /
             52 weeks or 20 / 60 / 120 / 250 days) (2026-11-17)
             E05_corr_falling/ (E05.md design; falling.py → out/E05_falling.html, E05_results.md, CSVs) — does the rule do
             better when the legs' correlation is falling as z reaches its extreme? T03's rule with a chosen extreme (±1,
             ±1.5, ±2), its trades split by falling / not falling correlation (lower than 4, 13 or 26 weeks earlier) with
             within-pair shuffles, and the rule entering only on falling (or only on not-falling) correlation against 50/50;
             controls for pair, z window, extreme, correlation period and lookback (2026-11-18)
```

**Results (run `headline`, 2026-11-17):** the rule is **REFUTED** at the MDE (2.8–12.1 pp/yr). T02 at the 95th percentile:
C1 0, C2 2, C3 0 of 6 (F-RA-S06); T03 at the 98th: C1 0, C2 2, C3 0 of 6 (F-RA-S04); 4 needed. The N-A percentiles on
the six index pairs are 11.5–57.5. The two runs make identical decisions (gate V4b in each). Pages tests/T02/T02_results.html
and tests/T03/T03_results.html.

`evidence/headline/` keeps every draw (`draws_*.csv`) in git; the daily NAV files (`nav_*.csv`) are regenerated by
`run.sh` and git-ignored.
