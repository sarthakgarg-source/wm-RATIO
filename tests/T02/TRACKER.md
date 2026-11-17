# T02 tracker

| step | status | notes |
|---|---|---|
| 0 data review, approved list, boundary | reused 2026-11-13 | T01's approved data of 2026-11-12; fingerprints checked OK on 2026-11-13 |
| 1–2 question, kind of experiment | drafted 2026-11-13 | timing rule; unit = round trip; T01 with the band on P_a ÷ P_b instead of ln(P_a ÷ P_b) |
| 3 test folder | done 2026-11-13 | tests/T02/: PLAN.md, T02.md, TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py, report.py, check_page.py |
| 4 no look-ahead | unchanged from T01 | the same PIT assertions, counted into audit.json |
| 5 design locked | done 2026-11-17 | §1–§3 frozen (`check_prereg.py --freeze`) and committed before the first run; §2 discloses E01 (2026-11-13, z-score added 2026-11-17) and E02 (2026-11-16), all before the lock. 2026-11-13: exploratory study E01 (ratio cycles) looked at the plain ratios before the lock; disclosed in §2 "What I already saw" |
| 6 engine | done 2026-11-13 | analysis/ratio_engine.py: `--test T01 / T02` fixes the form; T01 rerun (0 draws) reproduces all 600 rows of T01's headline summary.csv exactly, and decisions.csv, trades.csv, trade_extras.csv identically |
| 7 verifier | done 2026-11-13 | analysis/verify_ratio.py `--ratio log / plain`; on the T01 rerun: 100 nav files, 61,230 comparisons, 0 failures, both corruptions caught; refuses a run whose audit names the other form. Plain path smoke-tested with 0 draws, output discarded (exit codes only; see T02.md §2) |
| 8 run + gates | started 2026-11-17 | `bash tests/T02/run.sh` (run `headline`, 2,000 draws) |
| 9–11 results, findings, page | not started | |
