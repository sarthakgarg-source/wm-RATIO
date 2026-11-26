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
| aa_index_n100_pre2019.csv | `167c73492dd424cadf6e665546321ffd31d7da35ce39142f93c6f7282d742089` | 158,099 | 3,237 | 2005-11-30 | 2019-01-04 | PASS |
| aa_index_n100_2019_2026.csv | `4563e0b66f1aeeb0e7c0926d43e944b8b1bfe52f8527136e6f55042dd04c677b` | 107,276 | 1,951 | 2019-01-01 | 2026-11-20 | PASS |
| aa_index_mc100_pre2019.csv | `6586a0a392cbd73c3e6a8ee5d9644ba255c11f0171cb6bc3fd1cee4e6b26a65a` | 158,933 | 3,329 | 2005-07-18 | 2019-01-04 | PASS |
| aa_index_mc100_2019_2026.csv | `97387340ac4803b78ecf1097695624c0e1514ac608c0a09aa3ac07f66659d129` | 91,767 | 1,951 | 2019-01-01 | 2026-11-20 | PASS |
| aa_index_sc100_pre2019.csv | `6b6ebc3a3144ec74017cf24365efd0845ba6c639ca1a43603a646c2914fc9c21` | 163,764 | 3,715 | 2004-01-01 | 2019-01-04 | PASS |
| aa_index_sc100_2019_2026.csv | `e2421301b6ed87af64dbb67f06671bb1ec7591dc1b7d512c64385e7018cda04c` | 88,267 | 1,951 | 2019-01-01 | 2026-11-20 | PASS |
| aa_mauc1_daily.csv | `61cde69e5ce326072127095e4442a0f860555201a996424cc2d98a613511cb86` | 316,665 | 6,354 | 2003-11-10 | 2026-11-20 | PASS WITH NOTES |
| aa_liquid_lbes.csv | `26ae7f7c9e876ab41c31036926b418dbec69d5736774e5e91864cab1dd2c4ffd` | 167,127 | 5,767 | 2007-06-29 | 2026-11-25 | PASS WITH NOTES |
| aa_liquid_case.csv | `ac17a7fb1f2610010842a622231a9b8b6a71fcb3fd7b56bef95196eab555a749` | 125,370 | 5,630 | 2007-06-29 | 2026-11-25 | PASS WITH NOTES |

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
source and is not approved for any study. The nine files of the desk's latest Indian series (copied byte for byte from its
asset-allocation repository, wm-AA, commit e9cf7b25b45b9b66b53388decb5552031da87a14: N100, MC100 and SC100 to 2026-11-20, MCX
gold, and two liquid-ETF cash levels to 2026-11-25) approved 2026-11-26 by the data owner on the review in `DATA_REVIEW_AA.md`;
read only through `load_india_latest()`: the indices with rules H1–H3 (they equal `index_ohlcv_all.csv` on every common day),
gold with rules H7 (a non-positive close is dropped) and H8 (from 2005-01-01) (F-RA-D18), the cash levels as single daily prices
(F-RA-D17: before 2024-01-24 they are one series with no recorded source; disclose it). `load_daily()` and its two files are
unchanged, so the locked tests keep their inputs. `aa_db_check.csv` is that review's second source and is not approved for any
study. All approved files are read-only in the working tree.
