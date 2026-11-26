# Data review — US S&P 500 factor ETFs (SPMO, SPHQ, RPV)

Reviewed 2026-11-26 by `data/data_review_us_sp_factor.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items. SPY (S&P 500) is already approved in `us_etf_ohlcv.csv` and is the reference calendar and return here; the rest of the desk's S&P factor list (SPLV low volatility, NOBL dividend aristocrats, RSP equal weight, and IWF Russell 1000 growth) is approved in `us_factor_ohlcv.csv`.

- `us_sp_factor_ohlcv.csv`: sha256 `ae51bf2de5aeac12947bac8d1570d248673732ba0345cba0789b17e5be39a792`, 427,719 bytes, 12,745 lines
- `us_sp_factor_dividends.csv`: sha256 `5581388c1d149a5bc53a8e4dc1fb27a2b7e2cca66734b050584c0ef88635c8ce`, 8,264 bytes, 179 lines
- `us_sp_factor_splits.csv`: sha256 `9353e39fe02edb88b4ccb4334e951fa25c0908a7a3e88dd6f941ff55d71edae8`, 108 bytes, 1 lines
- `us_sp_factor_eodhd_check.csv` (second source, read by this review only; not for approval): sha256 `55acf24f8b5b30a6e71f66f652025414b887e0407975b1b9b09d496acdcee1e3`, 74,483 bytes

## What the files are

- **us_sp_factor_ohlcv.csv** — daily open/high/low/close/volume, US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_sp_factor.py` to 2026-11-18 (the last day of the approved SPY file); split-adjusted, not adjusted for distributions:
    - **SPMO** — Invesco S&P 500 Momentum (S&P 500 Momentum Index)
    - **SPHQ** — Invesco S&P 500 Quality (S&P 500 Quality Index)
    - **RPV** — Invesco S&P 500 Pure Value (S&P 500 Pure Value Index) — standing in for SPVU (S&P 500 Enhanced Value), which the research database does not hold
- **us_sp_factor_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD, as paid) from `us_corp_action_div_master_b_ed`; the history starts in 2010.
- **us_sp_factor_splits.csv** — share splits from `us_corp_action_cap_struc_master_b_ed`: 0 for these funds (the table is not a full history; the checks below look for splits from 2010 independently).
- **Job:** S&P 500 factor ETFs against SPY and each other in exploratory studies: signal, fill (the close, rule H3) and valuation.
- **Meaning of the price a loader would use:** total return (rule H4) from each fund's start under rule H5, then H1–H3, as `load_us_factor()`.

## Series checks (from 2010-01-01)

| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | blank open/high/low | distributions (n, sum as paid, largest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPMO | 2,493 | 2015-10-12 | 2026-11-18 | 11.1 | 0 | 0 | 300 | 0 | 47 | 0 | 0 | 0 | 3 | 5 | 98% | 43 ($6.98, $0.34) |
| SPHQ | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 99% | 62 ($8.98, $0.28) |
| RPV | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 99% | 67 ($22.15, $0.82) |

### Notes per series

- **SPMO** misses 300 of SPY's trading days (by year: 2015 40, 2016 124, 2017 136): 2015-10-13, 2015-10-14, 2015-10-15, 2015-10-19, 2015-10-20, 2015-10-21, 2015-10-23, 2015-10-27, 2015-11-02, 2015-11-03, 2015-11-04, 2015-11-09 …
- **SPMO** gaps over 5 calendar days end on: 2015-10-22, 2015-11-05, 2015-12-01, 2015-12-18, 2016-01-06, 2016-01-14, 2016-01-28, 2016-03-04, 2016-03-17, 2016-03-30, 2016-04-07, 2016-04-14 …
- **SPMO** raw jumps > 8% (SPY's return that day): 2020-03-12 -8.9% (SPY -10.1%); 2020-03-16 -16.7% (SPY -11.6%); 2020-03-17 +9.7% (SPY +5.3%); 2020-03-24 +8.9% (SPY +8.7%); 2025-04-09 +10.6% (SPY +10.0%)
- **SPMO** 3 runs of 5 or more equal closes
- **SPHQ** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.6% (SPY -10.1%); 2020-03-13 +9.4% (SPY +8.2%); 2020-03-16 -10.9% (SPY -11.6%); 2020-03-24 +9.8% (SPY +8.7%)
- **RPV** raw jumps > 8% (SPY's return that day): 2011-08-08 -8.6% (SPY -6.7%); 2020-03-09 -12.8% (SPY -8.1%); 2020-03-12 -13.9% (SPY -10.1%); 2020-03-13 +10.4% (SPY +8.2%); 2020-03-16 -13.9% (SPY -11.6%); 2020-03-18 -8.8% (SPY -5.2%); 2020-03-24 +12.1% (SPY +8.7%); 2020-06-11 -9.0% (SPY -5.9%)

## Splits: is the price file on one share basis, and are the distributions on the same basis?

Share splits in the corporate-action table: none for these funds.

The second source's raw close ÷ this file's close on every common day, in runs of the same value rounded to a whole number:

| ticker | runs of the ratio (from → to: rounded value, days) | common days off a whole number by > 0.5% |
|---|---|---|
| SPMO | 2015-10-12 → 2026-07-31: 1 (2,339) | 3 |
| SPHQ | — (not in the second source) | |
| RPV | — (not in the second source) | |

Distribution yield by year, as paid (sum of the year's amounts ÷ this file's close the day before each ex-date), %; a split adjusted in the prices but not in the amounts would multiply the yield before it:

| ticker | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPMO | — | — | — | — | — | 0.35 | 1.98 | 0.84 | 0.98 | 1.43 | 1.57 | 0.57 | 1.74 | 1.89 | 0.52 | 0.79 | 0.60 |
| SPHQ | 0.59 | 1.80 | 2.00 | 1.86 | 1.75 | 2.27 | 1.70 | 1.64 | 1.71 | 1.61 | 1.90 | 1.31 | 1.85 | 1.54 | 1.20 | 1.14 | 0.79 |
| RPV | 1.32 | 1.60 | 1.58 | 1.32 | 1.61 | 2.25 | 1.91 | 1.88 | 2.20 | 2.40 | 2.86 | 2.03 | 2.31 | 2.54 | 2.24 | 2.68 | 1.77 |

## Distributions: does the price fall by about the amount on the ex-date?

Over every distribution from 2010: the close-to-close fall on the ex-date minus SPY's fall that day, against the amount, both as % of the previous close; the median of (fall − amount) is near zero when the amounts are on the price file's basis.

| ticker | ex-dates | median amount | median (market-adjusted fall − amount) | mean (market-adjusted fall − amount) |
|---|---|---|---|---|
| SPMO | 43 | 0.26% | -0.27 pp | -0.30 pp |
| SPHQ | 62 | 0.41% | -0.03 pp | -0.02 pp |
| RPV | 67 | 0.52% | +0.05 pp | +0.18 pp |

## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available

Total return from the loader's rules (H4 on Eikon prices and Refinitiv distributions) against EODHD's adjusted close on the dates both have, from 2010.

| ticker | common days | raw closes differing > 0.5% | growth of 1, H4 ÷ EODHD (a year) | daily-return correlation | days with return gap > 0.5 pp |
|---|---|---|---|---|---|
| SPMO | 2,339 (2015-10-12 → 2026-07-31) | 3 | 1.0001 (+0.00%) | 0.99878 | 6 |
| SPHQ | — (not in the second source) | | | | |
| RPV | — (not in the second source) | | | | |

## After the launch period (rule H5: the first 1 January at least 12 months after the first day; not before 2010)

| ticker | first day in the file | start under H5 | years | missing vs SPY's days | jumps > 8% (SPY's return that day) | total return a year (SPY, same days) | weekly correlation with SPY | usable for a 52-week rule (≥ 5 years)? |
|---|---|---|---|---|---|---|---|---|
| SPMO | 2015-10-12 | 2017-01-01 | 9.9 | 136 (2017-01-05, 2017-01-06, 2017-01-11, 2017-01-12, 2017-01-13, 2017-01-17, 2017-01-19, 2017-01-23 …) | 2020-03-12 -8.9% (-10.1%); 2020-03-16 -16.7% (-11.6%); 2020-03-17 +9.7% (+5.3%); 2020-03-24 +8.9% (+8.7%); 2025-04-09 +10.6% (+10.0%) | 18.3% (14.4%) | 0.885 | yes |
| SPHQ | 2005-12-13 | 2010-01-01 | 16.9 | 0 | 2020-03-12 -9.6% (-10.1%); 2020-03-13 +9.4% (+8.2%); 2020-03-16 -10.9% (-11.6%); 2020-03-24 +9.8% (+8.7%) | 14.2% (13.6%) | 0.959 | yes |
| RPV | 2006-03-09 | 2010-01-01 | 16.9 | 0 | 2011-08-08 -8.6% (-6.7%); 2020-03-09 -12.8% (-8.1%); 2020-03-12 -13.9% (-10.1%); 2020-03-13 +10.4% (+8.2%); 2020-03-16 -13.9% (-11.6%); 2020-03-18 -8.8% (-5.2%); 2020-03-24 +12.1% (+8.7%); 2020-06-11 -9.0% (-5.9%) | 12.2% (13.6%) | 0.845 | yes |

## SPMO from 2018-01-01 (the proposed handling of F-RA-D16)

| start | years | missing vs SPY's days | flat runs ≥ 5 | ex-dates | median (market-adjusted fall − amount) | growth of 1 vs EODHD (a year), 2018-01-02 → 2026-07-31 | total return a year (SPY, same days) |
|---|---|---|---|---|---|---|---|
| 2018-01-01 | 8.9 | 0 | 0 | 35 | -0.13 pp | 1.0001 (+0.00%) | 17.1% (13.6%) |

## Findings and how a loader would handle them

- **F-RA-D16 — SPMO traded thinly for its first two years.** The file has no row on 300 of SPY's trading days from its first day to 2017 (by year: 2015 40, 2016 124, 2017 136), and 3 runs of five or more equal closes; the second source misses the same days, so they are sessions without trades. Rule H5 (start 2017-01-01) still leaves 136 missing sessions in 2017, which a study on a common calendar would drop for every fund. **Handled by starting SPMO on 2018-01-01**, its first calendar year with every session (a per-fund override of H5): from then 0 missing sessions, 0 flat runs, ex-date gap -0.13 pp, total return +0.00% a year against the second source; 8.9 years.
- **Distributions are on the price file's basis for all three:** no splits in the corporate-action table, stable yields, ex-date falls near the amounts (SPHQ -0.03 pp, RPV +0.05 pp; SPMO -0.27 pp over all years, pulled by the thin 2015–2017 sessions), and SPMO's total return agrees with the second source (+0.00% a year). No rule H6 needed.
- **Closes only** (SPMO, SPHQ, RPV): open, high and low are blank on almost every row (F-RA-D13); **rule H3** (fill = close).
- **No second source for SPHQ, RPV**; the second source ends on 2026-07-31. These two rest on the ex-date falls, the yields and the weekly correlation with SPY above.
- **SPHQ changed index on 2016-03-21** (TradingView's fund description: before that it tracked an S&P quality index with different metrics on the same S&P 500 universe). Not a data error, but a study over 2010–2026 measures two definitions of quality; a study should report results before and after the change. Open item: confirm the date and the earlier index against the issuer.
- **RPV stands in for SPVU** (S&P 500 Enhanced Value), which the research database does not hold (neither price table): S&P 500 Pure Value is a concentrated, value-score-weighted cut of the S&P 500, the data owner's choice of 2026-11-26. Its jumps over 8% are larger than SPY's on the same days (2020-03-09 -12.8% against -8.1%), as expected of a deep-value basket.
- **Spot checks against the issuer** (Invesco): not done here; open item, as for the other US ETFs.

## Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| us_sp_factor_ohlcv.csv | PASS WITH NOTES | usable with rules H3, H4 and H5, SPMO from 2018-01-01 (F-RA-D16); SPHQ and RPV 16.9 years, SPMO 8.9 |
| us_sp_factor_dividends.csv | PASS WITH NOTES | amounts on the price file's basis (no splits from 2010); SPMO's total return agrees with the second source |
| us_sp_factor_splits.csv | PASS WITH NOTES | no splits recorded for these funds; kept so the loader reads the same three files for every factor batch |
| us_sp_factor_eodhd_check.csv | not for approval | the second source, read only by this review |
