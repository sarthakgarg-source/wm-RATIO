# T02 — build plan (the request of 2026-11-13, copied word for word; never overwritten)

> we do not want to take log ratio change it to normal ratio

## What that means under the playbook (derived from the request; not part of the quote)

The log ratio is part of T01's locked design (T01.md §3 step 4), so it cannot be changed inside T01: an old question under
changed rules gets a new test number (RESEARCH_PLAYBOOK.md Step 3; CLAUDE.md). T02 is T01 with exactly one ingredient
changed — the band is drawn on r_t = P_a,t ÷ P_b,t instead of ln(P_a,t ÷ P_b,t) — and everything else (pairs, orientation,
calendar, band width, entry, exits, cap, book, cost, nulls, seeds, criteria, gates) as T01 ran.

Steps, in the playbook's order:
0. data: unchanged — the approved data of 2026-11-12, same fingerprints
1–2. the question and the kind of experiment                                     → T02.md §1
3. the test folder                                                               → tests/T02/
4. no look-ahead: as T01 (band from the truncated series, trade next session, assertions counted)
5. the full design, locked with a hash                                           → T02.md §2–§3, check_prereg.py --freeze, then a commit
6. the engine: one shared engine, the form fixed by the test number (T01 = log, T02 = plain); T01 reproduced exactly → analysis/ratio_engine.py
7. the verifier: re-derives the decisions on the form named by run.sh, fails if the engine's audit names another → analysis/verify_ratio.py
8. run everything through the gates, including V4b (T01 reproduced by the changed engine) → run.sh, analyse.py
9–11. results, findings, a results page re-checked number by number, and the side-by-side with T01 → fill_writeup.py, report.py, check_page.py
