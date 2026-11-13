# T01 tracker

| step | status | notes |
|---|---|---|
| 0 data review, approved list, boundary | done 2026-11-12 | 11 flagged rows explained; two loader rules H1/H2; two open items for the data owner (spot checks; gold 2026-01-30 roll) |
| 1–2 question, kind of experiment | done 2026-11-12 | timing rule; unit = round trip |
| 3 test folder | done 2026-11-12 | |
| 4 no look-ahead | done 2026-11-12 | band from the truncated series; trade date > decision date; 13,775,587 PIT checks, 0 violations (audit.json) |
| 5 design locked | done 2026-11-12 | sha256 fe9dc094de50… in the header; commit 5a32afd; check_prereg passes before and after the run |
| 6 engine | done 2026-11-12 | analysis/ratio_engine.py; headline run 1,267 s, 2,000 draws per null, 400,000 draws |
| 7 verifier | done 2026-11-12 | analysis/verify_ratio.py: 100 nav files, 61,230 comparisons, 0 failures; both corruptions caught |
| 8 run + gates | done 2026-11-12 | V1–V4 PASS (analysis.log); V4 regression baseline saved as evidence/previous_summary.csv for the next run |
| 9–11 results, findings, page | done 2026-11-12 | §4 generated; page 143 numbers, 0 failures; **H1 FAIL, H2 FAIL — REFUTED at the MDE** (F-RA-S01); F-RA-S02 provisional, F-RA-M01 fixed; §6–§7 written; after-run block lists 6 differences from the locked text |
