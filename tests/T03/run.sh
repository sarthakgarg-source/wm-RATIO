#!/usr/bin/env bash
# Regenerate every T03 artefact in the playbook's order. Run from the project root:  bash tests/T03/run.sh [run-name] [draws]
# Only after T03.md §1–§3 are frozen (check_prereg.py --freeze) and that freeze is committed.
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8               # Windows defaults to cp1252 for files and redirected output; the logs hold "→", "−"
PY=${PYTHON:-python3}; "$PY" -c "" 2>/dev/null || PY=python   # on Windows python3 can be the Store stub
RUN=${1:-headline}; DRAWS=${2:-2000}; EV=tests/T03/evidence/$RUN
"$PY" data/check_approved_data.py                        # 1. is the data exactly the approved data?
"$PY" tests/T01/check_prereg.py                          # 2a. T01's locked design unchanged (CLAUDE.md: before and after every run)
"$PY" tests/T02/check_prereg.py                          # 2b. T02's locked design unchanged
"$PY" tests/T03/check_prereg.py                          # 2c. T03's locked design unchanged (fails until it is frozen)
"$PY" analysis/ratio_engine.py --test T03 --run "$RUN" --draws "$DRAWS"                    # 3. strategy + every draw, no maximum hold
"$PY" analysis/ratio_engine.py --test T02 --run "$RUN" --draws 0 --out "$EV/t02_regression"  # 3b. gate V4b: the changed engine under T02's settings
"$PY" analysis/ratio_engine.py --test T01 --run "$RUN" --draws 0 --out "$EV/t01_regression"  # 3c. gate V4c: and under T01's
"$PY" analysis/verify_ratio.py "$EV" --ratio plain --max-hold none --selftest | tee "$EV/verify.log"   # 4. independent recomputation (+ corrupted-copy test)
"$PY" tests/T03/analyse.py "$RUN"                        # 5. gates + every table → analysis.log, tables.json (stops on a failed gate)
"$PY" tests/T03/fill_writeup.py "$RUN"                   # 6. tables into section 4 of T03.md
"$PY" tests/T03/report.py "$RUN"                         # 7. the results page
"$PY" tests/T03/check_page.py "$RUN"                     # 8. every number on the page re-checked
"$PY" tests/T03/check_prereg.py                          # 9. T03's design still unchanged
"$PY" tests/T02/check_prereg.py                          #    and T02's
"$PY" tests/T01/check_prereg.py                          #    and T01's
"$PY" tests/check_names.py                               # 10. no person's name in the repo (names from the git-ignored .env)
echo "T03 run '$RUN' complete"
