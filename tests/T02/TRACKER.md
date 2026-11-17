# T02 tracker

| step | status | notes |
|---|---|---|
| 0 data review, approved list, boundary | reused | the approved data of 2026-11-12; fingerprints checked by run.sh |
| 1–3 question, kind of experiment, folder | done 2026-11-17 | the plain-ratio ±1 sd first-touch switch with no maximum hold; same rule as T03 |
| 4 no look-ahead | done | the PIT assertions, counted into audit.json |
| 5 design locked | done 2026-11-17 | §1–§3 rewritten without the log-ratio version and the time limit (the desk's request), re-locked and committed before the run |
| 6–7 engine, verifier | done 2026-11-17 | plain ratio only, no maximum hold; the verifier re-derives that and fails on a corrupted copy |
| 8 run + gates | done 2026-11-17 | rerun 2026-11-17 under the re-locked design: V1 (0 failures, corruptions caught), V2, V3, V4a and V4b (identical to T03) PASS; check_page 0 failures |
| 9–11 results, findings, page | done 2026-11-17 | §4 generated; §5 (F-RA-S06, REFUTED), §6, §7 written from tables.json; T02_results.html |
