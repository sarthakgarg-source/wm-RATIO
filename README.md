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
9. Finding IDs: `F-RA-S##` (strategy), `F-RA-M##` (method), `F-RA-D##` (data). **Counter:** S06, M02, D18 (+ 2 open).

**Current tests:** T02 and T03 are the same rule under two numbers (the desk's choice of 2026-11-17): T02 judged at the
95th percentile, T03 at the 98th. On 2026-11-17 the earlier log-ratio test and the time limit on each trade were removed
from the repository at the desk's request; both are in git under the tag `archive/2026-11-17-earlier-versions`. T02 is
rerun under its rewritten design; T03's evidence of 2026-11-17 is re-analysed under its rewritten text. Each test's gate
V4b checks that the two make the same decisions and the same numbers. The next free test number is T04.

**Layout**
```
data/        raw files (read-only), data_review.py → DATA_REVIEW.md, APPROVED_DATA.md, check_approved_data.py;
             US ETFs (SPY + 13 broad commodity ETFs, approved 2026-11-19): fetch_us_etf.py (read-only pull from the research
             Postgres; credentials from the environment only), data_review_us.py → DATA_REVIEW_US_ETF.md (F-RA-D12, D13),
             read through approved_data.load_us_etf() (rules H3–H5); US gold ETF (GLD, for SPY against gold, approved
             2026-11-20): fetch_us_gold.py, data_review_us_gold.py → DATA_REVIEW_US_GOLD.md, read through
             approved_data.load_us_gold() (rules H3, H5; no distributions); US small-cap ETF (IJR, approved 2026-11-23):
             fetch_us_small.py, data_review_us_small.py → DATA_REVIEW_US_SMALL.md (F-RA-D14), read through
             approved_data.load_us_small() (rules H3–H5 and H6: pre-split distributions put on the price file's split basis);
             US factor and style ETFs (MTUM, QUAL, VLUE, USMV, SPLV, NOBL, RSP, IWF; approved 2026-11-26): fetch_us_factor.py,
             data_review_us_factor.py → DATA_REVIEW_US_FACTOR.md (F-RA-D15: four 2026 splits), read through
             approved_data.load_us_factor() (rules H3–H6; the splits from us_factor_splits.csv); US S&P 500 factor ETFs
             (SPMO, SPHQ, RPV standing in for SPVU; approved 2026-11-26): fetch_us_sp_factor.py, data_review_us_sp_factor.py →
             DATA_REVIEW_US_SP_FACTOR.md (F-RA-D16: SPMO from 2018-01-01), read through the same load_us_factor();
             the latest Indian series (approved 2026-11-26), copied byte for byte from the desk's asset-allocation repository
             (wm-AA) by fetch_aa.py: N100, MC100, SC100 to 2026-11-20, MCX gold from 2005 (rules H7–H8, F-RA-D18) and two
             liquid-ETF cash levels to 2026-11-25 (F-RA-D17); data_review_aa.py → DATA_REVIEW_AA.md; read through
             approved_data.load_india_latest() (load_daily() unchanged, so the locked tests keep their inputs)
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
             controls for pair, z window, extreme, correlation period and lookback (2026-11-18); section 3: daily equity
             curves and drawdowns of the rule's three versions against 50/50 and each asset class bought and held (N100,
             MC100, SC100, Micro250, gold), from the rule's first day or from gold's / Micro250's first day (2026-11-19)
             E07_good_bad/ (E07.md design; goodbad.py → out/E07_goodbad.html, E07_results.md, CSVs) — good against bad trades:
             their paths after entry, six measures of the ratio's character before entry, early warning, exit variants
             (z-stop, time stop) and waiting entries through the engine's book, and a walk-forward test (models trained on
             closed trades predicting each next year, against label shuffles and random skipping) (2026-11-19)
             E08_commodity_spy/ (E08.md design; commodity.py → out/E08_commodity.html, E08_results.md, CSVs) — 13 broad
             commodity ETFs ÷ SPY (US ETF data, total return): the relationship (correlation, beta, the ratio's mean
             reversion), the 'best relationship' chosen in advance on first halves, and Rule R on every ETF and z window
             against T03's nulls (2026-11-19); added: SPY against gold, GLD ÷ SPY, reported on its own (2026-11-20)
             E09_intrade_extra/ (E09.md design; intrade.py → out/E09_intrade.html, E09_results.md, CSVs) — while in a
             trade, how much did the held index beat its peer: per trade, compounded over the trades and per year in trade,
             by holding length, against T03's nulls; Indian index pairs, gold pairs, commodity ETFs ÷ SPY and SPY against
             gold (GLD ÷ SPY), every z window (2026-11-20)
             E10_trend_rotation/ (E10.md design; rotation.py → out/E10_rotation.html, E10_results.md, CSVs) — SC100 ÷ gold
             beside SPY ÷ GLD and IJR ÷ GLD on the same weeks; band touches against z signals; the desk's Weekly LW
             supertrend (5-bar linear-weighted ATR × 2, as in the desk's earlier repos; the first run's (10, 3) kept as a
             check) on each leg with four weight schemes for the four buy/sell combinations; relative momentum (12-1, 12,
             6 months); all beside Rule R on the US pairs (nine commodity ETFs and GLD against SPY, IJR against SPY and
             GLD) and the Indian pairs (N100, MC100, SC100, gold), against the same spells in a random order (2026-11-23)
             E11_multi_asset/ (E11.md design; multi.py → out/E11_multi.html, E11_results.md, CSVs) — multi-asset books, India
             (N100, MC100, SC100, gold) and the US (SPY, IJR, GCC, GLD) kept apart: how in sync their supertrends are; the
             desk's supertrend decides which asset classes are on, a class's money spread equally, by breadth, with a tilt
             or by a momentum pick (12-1, 12, 6 months); gate-and-rank and momentum-only rotations; E10's SC100 + gold and
             IJR + GLD rules as references; PDBC as a check on the US commodity; all always invested, against every shift
             of the signals in time (2026-11-25)
             E12_cmdty_rotation/ (E12.md design; cmdty.py → out/E12_cmdty.html, E12_results.md, CSVs) — SPY with PDBC, SDCI and
             COM (total return; supertrends on weekly bars from closes, since the research database holds no open/high/low for
             these funds before 2026): each pair's ±1 sd band (Rule R) and 50/50, the desk's supertrend on each fund and on each
             ETF ÷ SPY (relative, Rule RS), and a four-asset rotation book (absolute and relative supertrend, momentum, gate and
             rank, Rule R sleeves) against every shift of the signals in time (2026-11-26)
             E13_gold_liquid/ (E13.md design; goldliquid.py → out/E13_gold_liquid.html, E13_results.md, CSVs) — gold on SELL goes
             to the liquid ETF: every Indian gold-holding strategy of E03–E11 (T03's Rule R, E10's supertrend schemes and
             momentum on the gold pairs, E11's India book) on the latest data from 2012 (load_india_latest; gold from 2011), as
             before and with the switch (while gold's weekly supertrend is SELL, its weight is held in the liquid ETF), against
             gold's SELL weeks slid in time (2026-11-26)
             E06_trade_search/ (E06.md design; search.py → out/E06_trades.html, E06_results.md, CSVs) — SEARCH: every
             trade of the rule on all z windows with 24 conditions at entry; condition by condition; rules found on the
             first half of the history and checked on the second half and the gold pairs, against searches on shuffled
             outcomes; year by year and against random skipping; a sortable explorer of every trade (2026-11-18)
             RE-RUN 2026-11-26 (E01–E07, E09–E11; each design .md has a block of that date): the Indian parts on the latest
             data (load_india_latest: N100, MC100, SC100 and MCX gold to 2026-11-20, all from 2011-01-01; Micro250 on its
             earlier data, to 2026-08-28), through the shared module explore/E01_ratio_cycles/latest.py; whenever a strategy
             would hold gold while gold's weekly supertrend is SELL it holds the liquid ETF instead (the desk chose to
             replace the original rule; bought-and-held benchmarks are not switched); checks against T03's evidence (earlier
             data) became checks against the engine on the same data. The US parts and E08 are unchanged. E06's split moved
             to 2019-01-01 (6,805 rules found on index trades entered 2011–2018, checked on 2019–2026) and E07's walk-forward
             to 2015. E11's first run is kept in explore/E11_multi_asset/earlier_run_2026-11-25/
```

**Results (run `headline`, 2026-11-17):** the rule is **REFUTED** at the MDE (2.8–12.1 pp/yr). T02 at the 95th percentile:
C1 0, C2 2, C3 0 of 6 (F-RA-S06); T03 at the 98th: C1 0, C2 2, C3 0 of 6 (F-RA-S04); 4 needed. The N-A percentiles on
the six index pairs are 11.5–57.5. The two runs make identical decisions (gate V4b in each). Pages tests/T02/T02_results.html
and tests/T03/T03_results.html.

`evidence/headline/` keeps every draw (`draws_*.csv`) in git; the daily NAV files (`nav_*.csv`) are regenerated by
`run.sh` and git-ignored.
