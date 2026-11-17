#!/usr/bin/env bash
# Regenerate every T01 artefact in the playbook's order. Run from the project root:  bash tests/T01/run.sh [run-name] [draws]
set -euo pipefail
RUN=${1:-headline}; DRAWS=${2:-2000}; EV=tests/T01/evidence/$RUN
python3 data/check_approved_data.py                      # 1. is the data exactly the approved data?
python3 tests/T01/check_prereg.py                        # 2. is the locked design unchanged?
python3 analysis/ratio_engine.py --run "$RUN" --draws "$DRAWS"   # 3. strategy + every random draw → evidence/
python3 analysis/verify_ratio.py "$EV" --max-hold 26 --selftest | tee "$EV/verify.log"   # 4. independent recomputation (+ corrupted-copy test)
python3 tests/T01/analyse.py "$RUN"                      # 5. gates + every table → analysis.log, tables.json (stops on a failed gate)
python3 tests/T01/fill_writeup.py "$RUN"                 # 6. tables into section 4 of T01.md
python3 tests/T01/report.py "$RUN"                       # 7. the results page
python3 tests/T01/check_page.py "$RUN"                   # 8. every number on the page re-checked
python3 tests/T01/check_prereg.py                        # 9. design still unchanged
python3 tests/check_names.py                             # 10. no person's name in the repo (names from the git-ignored .env)
echo "T01 run '$RUN' complete"
