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
| us_factor_ohlcv.csv | `9156e69c99dacc81b32df98bdebd8b1211edf681d4d4b512110971dc04b27a5e` | 1,335,086 | 32,647 | 2002-08-19 | 2026-11-18 | PASS WITH NOTES |
| us_factor_dividends.csv | `d19722ff9d822e21429cdccf3ea3ffbb150b852d24382f81dcec804c96f6f684` | 28,139 | 599 | 2010-03-19 | 2026-10-19 | PASS WITH NOTES |
| us_factor_splits.csv | `8f2efbe210d45f1217593a4c0d3ee4725c86bfdfc24f106935c772b2de7cb8c4` | 378 | 4 | 2026-04-27 | 2026-11-03 | PASS WITH NOTES |
| us_sp_factor_ohlcv.csv | `ae51bf2de5aeac12947bac8d1570d248673732ba0345cba0789b17e5be39a792` | 427,719 | 12,744 | 2005-12-13 | 2026-11-18 | PASS WITH NOTES |
| us_sp_factor_dividends.csv | `5581388c1d149a5bc53a8e4dc1fb27a2b7e2cca66734b050584c0ef88635c8ce` | 8,264 | 178 | 2010-03-19 | 2026-09-21 | PASS WITH NOTES |
| us_sp_factor_splits.csv | `9353e39fe02edb88b4ccb4334e951fa25c0908a7a3e88dd6f941ff55d71edae8` | 108 | 0 | — | — | PASS WITH NOTES |

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
study. The three US factor and style ETF files (MTUM, QUAL, VLUE, USMV, SPLV, NOBL, RSP, IWF) approved 2026-11-26 by the data
owner on the review in `DATA_REVIEW_US_FACTOR.md`; read only through `load_us_factor()` with rules H3, H4, H5 and H6 (F-RA-D15:
MTUM, QUAL and IWF split 4-for-1 and NOBL 2-for-1 in 2026, so their earlier distributions are divided by the split ratio; the
splits come from `us_factor_splits.csv`); MTUM and QUAL have open, high and low on every row (fill = OHLC4), the other six closes
only (fill = close). `us_factor_eodhd_check.csv` is that review's second source and is not approved for any study. The three US
S&P 500 factor ETF files (SPMO momentum, SPHQ quality, RPV pure value standing in for SPVU, which the research database does not
hold) approved 2026-11-26 by the data owner on the review in `DATA_REVIEW_US_SP_FACTOR.md`; read through the same
`load_us_factor()` with rules H3, H4 and H5, SPMO from 2018-01-01 (F-RA-D16: it traded thinly until then; a per-fund override of
H5); no splits (the splits file is empty and kept for the loader's shape). `us_sp_factor_eodhd_check.csv` is that review's second
source and is not approved for any study. All approved files are read-only in the working tree.
