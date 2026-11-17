# T03 — build plan (the request of 2026-11-17, copied word for word; never overwritten)

> can we remove the 26 week rule then?

> yes please do and remove the 26 week rule

## What that means under the playbook (derived from the request; not part of the quote)

The 26-week maximum hold is part of T02's locked design (T02.md §3 step 5; T01's too), so it cannot be dropped inside T02:
changed rules get a new test number (RESEARCH_PLAYBOOK.md Step 3; CLAUDE.md). T03 is T02 with exactly one ingredient
changed — no maximum hold, so a trade ends only at its exit rule (E1: the ratio back at its trailing mean; E2: the opposite
band) — and everything else (plain ratio, pairs, orientation, calendar, band width, entry, book, cost, nulls, seeds, gates)
as T02 ran. The idea came from results already seen on the same data (E03: the trades that ran to the cap lost, those that
exited at the mean won), so the pass mark is raised from the 95th to the 98th percentile.

Steps, in the playbook's order:
0. data: unchanged — the approved data of 2026-11-12, same fingerprints
1–2. the question and the kind of experiment                                     → T03.md §1
3. the test folder                                                               → tests/T03/
4. no look-ahead: as T01/T02 (band from the truncated series, trade next session, assertions counted)
5. the full design, locked with a hash                                           → T03.md §2–§3, check_prereg.py --freeze, then a commit
6. the engine: the cap fixed by the test number (TEST_MAX_HOLD: T01 = T02 = 26, T03 = none); T01 and T02 reproduced exactly → analysis/ratio_engine.py
7. the verifier: re-derives the decisions with the cap named by run.sh (--max-hold none), fails if the audit names another → analysis/verify_ratio.py
8. run everything through the gates, including V4b (T02 reproduced) and V4c (T01 reproduced) → run.sh, analyse.py
9–11. results, findings, a results page re-checked number by number, the side-by-side with T02 and what became of T02's capped trades → fill_writeup.py, report.py, check_page.py
