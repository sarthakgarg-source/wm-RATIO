# T03 tracker

| step | status | notes |
|---|---|---|
| 0 data review, approved list, boundary | reused 2026-11-17 | T01's approved data of 2026-11-12; fingerprints checked by run.sh |
| 1–2 question, kind of experiment | drafted 2026-11-17 | timing rule; unit = round trip; T02 with no maximum hold (the request of 2026-11-17) |
| 3 test folder | done 2026-11-17 | tests/T03/: PLAN.md, T03.md, TRACKER.md, run.sh, check_prereg.py, analyse.py, fill_writeup.py, report.py, check_page.py |
| 4 no look-ahead | unchanged from T01/T02 | the same PIT assertions, counted into audit.json |
| 5 design locked | done 2026-11-17 | §1–§3 frozen and committed before the first run. §2 discloses T01, T02, E01–E03, the engine regression check and the one uncapped number printed by the verifier's deliberate-failure check |
| 6 engine | done 2026-11-17 | analysis/ratio_engine.py: TEST_MAX_HOLD (T01 = T02 = 26, T03 = none); T01 and T02 rerun with 0 draws: all 600 summary rows of each and decisions, trades, extras identical |
| 7 verifier | done 2026-11-17 | analysis/verify_ratio.py `--max-hold N | none` from the design; fails if the audit names another cap; T01 and T02 reruns verify with 0 failures |
| 8 run + gates | done 2026-11-17 | V1 (0 failures in 61,230 comparisons, corruptions caught), V2, V3, V4a, V4b (600 of 600 T02 rows) and V4c (600 of 600 T01 rows) PASS; check_page 0 failures. Run started 2026-11-17 | `bash tests/T03/run.sh` (run `headline`, 2,000 draws); a smoke run with 3 draws before the lock exited 0 on every step, output deleted unread |
| 9–11 results, findings, page | done 2026-11-17 | §4 generated; §5 F-RA-S04 (REFUTED) and F-RA-S05 (OBSERVED), §6, §7 written from tables.json; T03_results.html |
