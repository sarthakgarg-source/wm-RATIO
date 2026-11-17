# T02 — build plan (the requests, copied word for word; never overwritten)

> we do not want to take log ratio change it to normal ratio

(2026-11-13)

> first remove t01 from every where we will not procced with it as it is very similar to t02. Alos even if it is hard coded i wnat you to make changes and remove 26 week rule

(2026-11-17; the desk then chose to keep both T02 and T03 as the same rule under two names)

## What that means (derived from the requests; not part of the quotes)

T02 is the ±1 sd first-touch long-only switch on an index pair with the band drawn on the plain ratio r_t = P_a,t ÷ P_b,t,
exit at the trailing mean (E1) or at the opposite band (E2), and no maximum hold: a trade lasts until its exit rule fires.
T03 is the same rule judged at the 98th percentile instead of the 95th. On 2026-11-17 the design was rewritten and re-locked
without the log-ratio version and without the time limit the first T02 design had; both earlier versions are in git under
the tag `archive/2026-11-17-earlier-versions`.

Steps, in the playbook's order:
0. data: the approved data of 2026-11-12, same fingerprints
1–3. the question, the kind of experiment, the test folder                       → T02.md §1, tests/T02/
4. no look-ahead: band from the truncated series, trade next session, assertions counted
5. the full design, locked with a hash                                           → T02.md §2–§3, check_prereg.py --freeze, then a commit
6–7. the engine (plain ratio, no maximum hold) and the independent verifier      → analysis/ratio_engine.py, analysis/verify_ratio.py
8. run everything through the gates, including V4b (the same decisions and numbers as T03) → run.sh, analyse.py
9–11. results, findings, a results page re-checked number by number               → fill_writeup.py, report.py, check_page.py
