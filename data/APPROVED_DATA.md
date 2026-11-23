# Approved data

Tests may read ONLY the files below, through `analysis/approved_data.py`, which checks the fingerprint on every open. Refreshing or extending a file needs the data owner's sign-off and a new review.

| file | sha256 | bytes | rows | first date | last date | verdict |
|---|---|---|---|---|---|---|
| index_ohlcv_all.csv | `3499a3b9c942c95ecbd711540d7fb0d92246992dff27ae30864295592b6a2efa` | 3,619,661 | 33,616 | 2000-01-03 | 2026-08-28 | PASS WITH NOTES |
| MAUc1_daily.csv | `29dd9d3756ad48d61b3c00cc5e40e76ddadae7516f625c535083e1f28c8e3227` | 121,658 | 3,000 | 2015-01-12 | 2026-09-08 | PASS WITH NOTES |
| us_etf_ohlcv.csv | `84729a36edc18c343c4d1768cec41c577d0ff5850265e2f3aabdb1e841904e85` | 1,227,105 | 36,229 | 1996-03-20 | 2026-11-18 | PASS WITH NOTES |
| us_etf_dividends.csv | `04392d6ff812359e1f2acbf89efd038dc81565315f340bc471cdc9615d0251e1` | 12,646 | 278 | 2010-03-19 | 2026-10-27 | PASS WITH NOTES |
| us_gold_ohlcv.csv | `75fa5f1bc47c0df7bf0f95b21c2d744d85c72fb7f039990d0dca1517fa4d3b92` | 179,468 | 5,466 | 2005-03-01 | 2026-11-18 | PASS WITH NOTES |
| us_small_ohlcv.csv | `726cb635d29845fe1645db33ae1c2ef3b85bd47332d2e9f55e1bdba0fb02a09d` | 200,991 | 6,054 | 2002-08-16 | 2026-11-18 | PASS WITH NOTES |
| us_small_dividends.csv | `332d0ae744bc2c815757ad742e1018efa4067015fe96c580ec64ef39ecbc3db3` | 3,187 | 67 | 2010-03-25 | 2026-09-15 | PASS WITH NOTES |

The first two approved 2026-11-12 by the review in `DATA_REVIEW.md`. The two US ETF files (SPY and 13 broad commodity ETFs) approved
2026-11-19 by the data owner on the review in `DATA_REVIEW_US_ETF.md`; they are read only through `load_us_etf()` with rules H3
(fill = close: the source has closes only), H4 (total return) and H5 (each fund's launch period dropped). `us_etf_eodhd_check.csv`
is the review's second source and is not approved for any study. The US gold ETF file (GLD, for SPY against gold) approved
2026-11-20 by the data owner on the review in `DATA_REVIEW_US_GOLD.md`; read only through `load_us_gold()` with rules H3 and
H5 (GLD pays no distributions, so rule H4 changes nothing); `us_gold_eodhd_check.csv` is that review's second source and is not
approved for any study. The two US small-cap ETF files (IJR, for US small caps against SPY and against gold) approved
2026-11-23 by the data owner on the review in `DATA_REVIEW_US_SMALL.md`; read only through `load_us_small()` with rules H3, H4,
H5 and H6 (F-RA-D14: the price file is split-adjusted and the distribution amounts are as paid, so distributions before the
2-for-1 split of 2017-01-19 are halved); `us_small_eodhd_check.csv` is that review's second source and is not approved for any
study. All approved files are read-only in the working tree.
