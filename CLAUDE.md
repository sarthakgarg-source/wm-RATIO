# wm-RATIO — conventions for agents
Read README.md, RESEARCH_PLAYBOOK.md (the team's playbook; the short form governs here) and tests/T02/T02.md.
- Data only through `analysis/approved_data.load_daily()`. Never open a CSV directly. Never edit `data/*.csv`.
- `python tests/T02/check_prereg.py` and `python tests/T03/check_prereg.py` must pass before and after every run.
  Sections 1–3 of a test's write-up are frozen; a new question or changed rules = a new test number (T04 next), never an
  edit to locked sections. (Exception on record: on 2026-11-17 the desk had T02's and T03's §1–3 rewritten and re-locked
  to remove the log-ratio version and the time limit; the earlier versions are in git tag archive/2026-11-17-earlier-versions.)
- The rule: the ±1 sd band on the plain ratio P_a ÷ P_b, first touch, exit at the trailing mean (E1) or the opposite band
  (E2); no maximum hold — a trade ends only at its exit rule.
- Engine convention: weekly decisions on the pair's common calendar (last day of the ISO week), fills next session at
  `fill`, 10.4 bps per side, units held, daily NAV, neutral = 50/50 by value at entry then unrebalanced. Point-in-time
  assertions are counted into audit.json; a violation is a failed gate, not a warning.
- Nulls: N-A permutes blocks of abutting trade spells and the non-zero gaps (timing), N-B flips each block's legs; both
  through the same book code; seeds `default_rng([pair_id (+100 for N-B), exit_id, draw_id, window])`. V3 measures that each
  draw kept the counts.
- The verifier imports nothing from the engine. Keep it that way.
- Every number in a test's §4 and on its results page comes from analyse.py / report.py over evidence/; check_page.py
  must report 0 failures before anything is shared. Dates (first and last) go with every number and chart.
- Python 3.11: no same-quote nesting inside f-strings. Weekly decisions only.
- No person's name anywhere. Finding IDs from the counter in README.md.
- Commit the locked design before the first run commit; never rewrite history that contains a locked design.
