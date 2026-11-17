#!/usr/bin/env bash
# Regenerate every T02 artefact in the playbook's order. Run from the project root:  bash tests/T02/run.sh [run-name] [draws]
# Only after T02.md §1–§3 are frozen (check_prereg.py --freeze) and that freeze is committed. Gate V4b reads tests/T03/evidence/headline/.
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8               # Windows defaults to cp1252 for files and redirected output; the logs hold "→", "−"
PY=${PYTHON:-python3}; "$PY" -c "" 2>/dev/null || PY=python   # on Windows python3 can be the Store stub
RUN=${1:-headline}; DRAWS=${2:-2000}; EV=tests/T02/evidence/$RUN
"$PY" data/check_approved_data.py                        # 1. is the data exactly the approved data?
"$PY" tests/T02/check_prereg.py                          # 2. the locked designs unchanged (CLAUDE.md: before and after every run)
"$PY" tests/T03/check_prereg.py
"$PY" analysis/ratio_engine.py --test T02 --run "$RUN" --draws "$DRAWS"                    # 3. strategy + every draw (plain ratio, no maximum hold)
"$PY" analysis/verify_ratio.py "$EV" --selftest | tee "$EV/verify.log"                     # 4. independent recomputation (+ corrupted-copy test)
"$PY" tests/T02/analyse.py "$RUN"                        # 5. gates + every table → analysis.log, tables.json (stops on a failed gate)
"$PY" tests/T02/fill_writeup.py "$RUN"                   # 6. tables into section 4 of T02.md
"$PY" tests/T02/report.py "$RUN"                         # 7. the results page
"$PY" tests/T02/check_page.py "$RUN"                     # 8. every number on the page re-checked
"$PY" tests/T02/check_prereg.py                          # 9. the locked designs still unchanged
"$PY" tests/T03/check_prereg.py
"$PY" tests/check_names.py                               # 10. no person's name in the repo (names from the git-ignored .env)
echo "T02 run '$RUN' complete"
