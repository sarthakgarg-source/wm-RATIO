# Approved data

Tests may read ONLY the files below, through `analysis/approved_data.py`, which checks the fingerprint on every open. Refreshing or extending a file needs the data owner's sign-off and a new review.

| file | sha256 | bytes | rows | first date | last date | verdict |
|---|---|---|---|---|---|---|
| index_ohlcv_all.csv | `3499a3b9c942c95ecbd711540d7fb0d92246992dff27ae30864295592b6a2efa` | 3,619,661 | 33,616 | 2000-01-03 | 2026-08-28 | PASS WITH NOTES |
| MAUc1_daily.csv | `29dd9d3756ad48d61b3c00cc5e40e76ddadae7516f625c535083e1f28c8e3227` | 121,658 | 3,000 | 2015-01-12 | 2026-09-08 | PASS WITH NOTES |

Approved 2026-11-12 by the review above. Both files are read-only in the working tree.
