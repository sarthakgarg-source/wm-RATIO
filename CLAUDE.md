# wm-RATIO — conventions for agents
Read README.md, RESEARCH_PLAYBOOK.md (the team's playbook; the short form governs here) and tests/T01/T01.md.
- Data only through `analysis/approved_data.load_daily()`. Never open a CSV directly. Never edit `data/*.csv`.
- `python tests/T01/check_prereg.py` must pass before and after every run. Sections 1–3 of T01.md are frozen; a new
  question or changed rules = a new test number (T02), never an edit to T01's locked sections.
- Engine convention: weekly decisions on the pair's common calendar (last day of the ISO week), fills next session at
  `fill`, 10.4 bps per side, units held, daily NAV, neutral = 50/50 by value at entry then unrebalanced. Point-in-time
  assertions are counted into audit.json; a violation is a failed gate, not a warning.
- Nulls: N-A permutes blocks of abutting trade spells and the non-zero gaps (timing), N-B flips each block's legs; both
  through the same book code; seeds `default_rng([pair_id (+100 for N-B), exit_id, draw_id, window])`. V3 measures that each
  draw kept the counts. The locked text says "spells"; the difference is recorded in T01.md "Added after the run" (F-RA-M01).
- The verifier imports nothing from the engine. Keep it that way.
- Every number in T01.md §4 and on the results page comes from analyse.py / report.py over evidence/; check_page.py
  must report 0 failures before anything is shared. Dates (first and last) go with every number and chart.
- Python 3.11: no same-quote nesting inside f-strings. Weekly decisions only.
- No person's name anywhere. Finding IDs from the counter in README.md.
- Commit the locked design before the first run commit; never rewrite history that contains a locked design.
