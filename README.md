# wm-RATIO — ratio analysis of the NSE index pairs, under the research playbook

**The question (T01):** on an index pair, does holding only the leg that has lagged — from the first week the ratio
ln(a ÷ b) is more than one standard deviation below (or above) its trailing-52-week mean, until the ratio is back at
the mean — earn more and fall less than the same trades scattered at random? Long-only switch; six index pairs decide,
four gold pairs reported. Not a z-score rule: a band on the ratio, entry at the first touch, exit at the mean (and, as
a second exit, at the opposite band).

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
9. Finding IDs: `F-RA-S##` (strategy), `F-RA-M##` (method), `F-RA-D##` (data). **Counter:** S03, M02, D11 (+ 2 open).

**Current test:** T03 (tests/T03/), **LOCKED, RUNNING** — T02 with no maximum hold: the 26-week cap removed, so a trade ends only at
its exit rule (the request of 2026-11-17). Pass mark raised to the 98th percentile because the change was suggested by
results seen on the same data (E03). T01 and T02 stay COMPLETE and reproducible; the shared engine takes `--test T01 | T02 |
T03`, which fixes the ratio form and the cap.

**Layout**
```
data/        raw files (read-only), data_review.py → DATA_REVIEW.md, APPROVED_DATA.md, check_approved_data.py
analysis/    approved_data.py (the boundary), ratio_engine.py (engine + nulls), verify_ratio.py (independent checker)
tests/T01/   PLAN.md, T01.md (§1–3 locked), TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py,
             report.py, check_page.py, null_blocks.py (blocks vs spells in the null), T01_results.html, evidence/<run>/
tests/T02/   PLAN.md, T02.md (§1–3 locked), TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py, report.py,
             check_page.py, T02_results.html, evidence/<run>/ (with t01_regression/ for gate V4b)
tests/T03/   PLAN.md, T03.md (§1–3 locked), TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py, report.py,
             check_page.py; evidence/<run>/ (with t02_regression/ and t01_regression/ for gates V4b, V4c) after the run
tests/       check_names.py (hygiene: no person's name; names from WM_REDACT_NAMES in the git-ignored .env)
explore/     EXPLORATORY studies — description, never findings, never quoted as evidence for a rule:
             E01_ratio_cycles/ (E01.md design; cycles.py → out/E01_cycles.html, E01_results.md, CSVs) — cyclicality of
             the ten ratio series against sign-flipped random walks, point-in-time (expanding drift and threshold, counted
             look-ahead check) (2026-11-13), plus the z-score of the ratio (T02's band in units of its trailing-52-week sd;
             description only, nothing about what follows a band touch) (2026-11-17); disclosed in T02.md §2.
             peaks_troughs_xlsx.py → out/E01_peaks_troughs_explained.xlsx:
             the peak/trough rule as live Excel formulas (toy example + N100 ÷ SC100), matched against cycles.py
             E02_ratio_segments/ (E02.md design; segments.py → out/E02_segments.html, E02_results.md, CSVs) — HINDSIGHT
             segmentation of the daily ratios with the desk's market-segregation method (scipy prominence → peak-to-peak
             regression), trend in % a year, grid of settings, cross-checked against E01 (2026-11-16); disclosed in T02.md §2
             E03_trade_conditions/ (E03.md design; conditions.py → out/E03_conditions.html, E03_results.md, CSVs) — after T01
             and T02 ran: forward returns of the cheaper / richer leg by z bucket against E01's random walks, and the market's
             condition at each T01 / T02 trade (correlation, market regime, volatility, entry depth and speed) related to the
             trade's outcome with within-pair permutation nulls (2026-11-17)
```

**Result of T02 (run `headline`, 2026-11-17):** H1 FAIL, H2 FAIL — C1 0 of 6, C2 2 of 6, C3 0 of 6 (4 needed). The plain-ratio
band gives the same verdict as the log band: **REFUTED at the MDE** (2.5–11.4 pp/yr), F-RA-S03; the two forms make the same
decision in at least 95.9% of weeks on every pair (F-RA-M02). Page tests/T02/T02_results.html.

**Result of T01 (run `headline`, 2026-11-12):** H1 FAIL, H2 FAIL — C1 0 of 6, C2 2 of 6, C3 0 of 6 (4 needed). The ±1 sd first-touch
long-only switch with exit at the mean is **REFUTED at the MDE** (2.6–11.4 pp/yr) on the six index pairs; E2 fails the same
criteria. Findings F-RA-S01, F-RA-S02, F-RA-M01 in tests/T01/T01.md §5; page tests/T01/T01_results.html.

`evidence/headline/` keeps every draw (`draws_*.csv`, 35 MB) in git; the daily NAV files (`nav_*.csv`, 31 MB) are regenerated by
`run.sh` and git-ignored.
