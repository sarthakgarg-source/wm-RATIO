# T01 — build plan (copied word for word from the review of 2026-11-12; never overwritten)

Build a ratio-analysis repo, in detail, under RESEARCH_PLAYBOOK.md. The rule is not a z-score: the ratio of a pair
against its trailing mean ± 1 sd band — enter the leg that has lagged on the first touch of the band, exit when the
ratio is back at the mean (deciding) and, as a second exit, when it reaches the opposite band. Long-only switch between
the two legs. Trailing 52-week band (26 and 104 as robustness). The six index pairs decide; the gold pairs are reported.

Steps, in the playbook's order:
0. data review + approved list + fingerprint check + the boundary module       → data/, analysis/approved_data.py
1–2. the question and the kind of experiment                                     → T01.md §1
3. the test folder                                                               → tests/T01/
4. no look-ahead: decide Friday close, trade next session at OHLC4, band from the truncated series, assertions counted
5. the full design, locked with a hash                                           → T01.md §2–§3, check_prereg.py
6. the engine                                                                    → analysis/ratio_engine.py
7. an independent verifier (no engine import; closed form; corrupted-copy test) → analysis/verify_ratio.py
8. run everything through the gates                                              → run.sh, analyse.py
9–11. results in a fixed order, findings, a results page re-checked number by number → fill_writeup.py, report.py, check_page.py
