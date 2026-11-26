# Data review — US factor and style ETFs (MTUM, QUAL, VLUE, USMV, SPLV, NOBL, RSP, IWF)

Reviewed 2026-11-26 by `data/data_review_us_factor.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items. SPY (S&P 500) is already approved in `us_etf_ohlcv.csv` and is the reference calendar and return here.

- `us_factor_ohlcv.csv`: sha256 `9156e69c99dacc81b32df98bdebd8b1211edf681d4d4b512110971dc04b27a5e`, 1,335,086 bytes, 32,648 lines
- `us_factor_dividends.csv`: sha256 `d19722ff9d822e21429cdccf3ea3ffbb150b852d24382f81dcec804c96f6f684`, 28,139 bytes, 600 lines
- `us_factor_splits.csv`: sha256 `8f2efbe210d45f1217593a4c0d3ee4725c86bfdfc24f106935c772b2de7cb8c4`, 378 bytes, 5 lines
- `us_factor_eodhd_check.csv` (second source, read by this review only; not for approval): sha256 `b3260a1ae82c3c397f52853c23be6444ad075c6859e06e1f288d073f7a4da44b`, 715,984 bytes

## What the files are

- **us_factor_ohlcv.csv** — daily open/high/low/close/volume, US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_factor.py` to 2026-11-18 (the last day of the approved SPY file). Prices are split-adjusted (on today's share basis) and not adjusted for distributions:
    - **MTUM** — iShares MSCI USA Momentum Factor
    - **QUAL** — iShares MSCI USA Quality Factor
    - **VLUE** — iShares MSCI USA Value Factor (MSCI USA Enhanced Value)
    - **USMV** — iShares MSCI USA Min Vol Factor
    - **SPLV** — Invesco S&P 500 Low Volatility
    - **NOBL** — ProShares S&P 500 Dividend Aristocrats
    - **RSP** — Invesco S&P 500 Equal Weight
    - **IWF** — iShares Russell 1000 Growth
- **us_factor_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD, **as paid**) from `us_corp_action_div_master_b_ed`; the history starts in 2010.
- **us_factor_splits.csv** — one row per distinct share split from `us_corp_action_cap_struc_master_b_ed` (ex-date, effective date, old and new shares).
- **Job:** factor and style ETFs against SPY (and each other) in exploratory studies: signal, fill (the close, rule H3) and valuation.
- **Meaning of the price a loader would use:** total return (rule H4) on the price file's split basis (rule H6) from each fund's start under rule H5, then H1–H3, as `load_us_small()`.

## Series checks (from 2010-01-01)

| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | blank open/high/low | distributions (n, sum as paid, largest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MTUM | 3,414 | 2013-04-18 | 2026-11-18 | 13.6 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0% | 55 ($19.47, $1.03) |
| QUAL | 3,355 | 2013-07-18 | 2026-11-18 | 13.3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0% | 54 ($20.14, $0.61) |
| VLUE | 3,334 | 2013-04-18 | 2026-11-18 | 13.6 | 0 | 0 | 85 | 0 | 4 | 0 | 0 | 0 | 0 | 5 | 99% | 55 ($29.08, $0.95) |
| USMV | 3,785 | 2011-10-20 | 2026-11-18 | 15.1 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 99% | 62 ($16.05, $0.43) |
| SPLV | 3,909 | 2011-05-05 | 2026-11-18 | 15.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 99% | 185 ($17.27, $0.15) |
| NOBL | 3,297 | 2013-10-10 | 2026-11-18 | 13.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 99% | 52 ($19.73, $0.66) |
| RSP | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 99% | 67 ($29.47, $0.83) |
| IWF | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 99% | 69 ($23.96, $0.58) |

### Notes per series

- **MTUM** misses 5 of SPY's trading days (by year: 2013 5): 2013-04-19, 2013-04-22, 2013-09-05, 2013-09-20, 2013-10-11
- **MTUM** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.4% (SPY -10.1%); 2020-03-16 -13.2% (SPY -11.6%); 2020-03-24 +10.1% (SPY +8.7%); 2025-04-09 +10.0% (SPY +10.0%); 2026-11-05 -8.6% (SPY -4.5%)
- **QUAL** misses 1 of SPY's trading days (by year: 2013 1): 2013-08-01
- **QUAL** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.8% (SPY -10.1%); 2020-03-13 +8.7% (SPY +8.2%); 2020-03-16 -10.7% (SPY -11.6%); 2020-03-24 +9.9% (SPY +8.7%); 2025-04-09 +8.3% (SPY +10.0%)
- **VLUE** misses 85 of SPY's trading days (by year: 2013 60, 2014 17, 2015 7, 2016 1): 2013-04-19, 2013-04-22, 2013-04-23, 2013-04-29, 2013-05-01, 2013-05-03, 2013-05-06, 2013-05-07, 2013-05-20, 2013-06-03, 2013-06-06, 2013-06-07 …
- **VLUE** gaps over 5 calendar days end on: 2013-04-24, 2013-05-08, 2013-09-06, 2013-09-17
- **VLUE** raw jumps > 8% (SPY's return that day): 2020-03-09 -9.9% (SPY -8.1%); 2020-03-12 -12.0% (SPY -10.1%); 2020-03-13 +9.5% (SPY +8.2%); 2020-03-16 -13.6% (SPY -11.6%); 2020-03-24 +9.6% (SPY +8.7%)
- **USMV** misses 7 of SPY's trading days (by year: 2011 7): 2011-10-24, 2011-11-01, 2011-11-02, 2011-11-03, 2011-11-22, 2011-12-19, 2011-12-20
- **USMV** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.9% (SPY -10.1%); 2020-03-16 -10.6% (SPY -11.6%); 2020-03-24 +8.3% (SPY +8.7%)
- **SPLV** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.9% (SPY -10.1%); 2020-03-16 -13.2% (SPY -11.6%); 2020-03-17 +8.6% (SPY +5.3%); 2020-03-24 +9.4% (SPY +8.7%)
- **NOBL** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.6% (SPY -10.1%); 2020-03-16 -10.7% (SPY -11.6%); 2020-03-24 +9.5% (SPY +8.7%)
- **RSP** raw jumps > 8% (SPY's return that day): 2020-03-09 -9.1% (SPY -8.1%); 2020-03-12 -10.6% (SPY -10.1%); 2020-03-16 -12.8% (SPY -11.6%); 2020-03-24 +10.1% (SPY +8.7%)
- **IWF** raw jumps > 8% (SPY's return that day): 2020-03-12 -9.6% (SPY -10.1%); 2020-03-13 +8.6% (SPY +8.2%); 2020-03-16 -12.5% (SPY -11.6%); 2020-03-24 +8.7% (SPY +8.7%); 2025-04-09 +11.0% (SPY +10.0%)

## Splits: the price file is split-adjusted, the distribution amounts are not

Share splits in the corporate-action table (`us_corp_action_cap_struc_master_b_ed`):

| ticker | ex-date | effective date | old → new shares | distributions within 10 days of the ex-date |
|---|---|---|---|---|
| IWF | 2026-04-27 | 2026-04-28 | 1 → 4 | 0 |
| MTUM | 2026-11-03 | 2026-11-04 | 1 → 4 | 0 |
| NOBL | 2026-05-26 | 2026-05-27 | 1 → 2 | 0 |
| QUAL | 2026-11-03 | 2026-11-04 | 1 → 4 | 0 |

The second source's raw close ÷ this file's close on every common day, in runs of the same value rounded to a whole number (a run at 4 = closes this file has divided by 4 for a later split):

| ticker | runs of the ratio (from → to: rounded value, days) | common days off a whole number by > 0.5% |
|---|---|---|
| MTUM | 2013-04-18 → 2026-07-31: 4 (3,337) | 1 |
| QUAL | — (not in the second source) | |
| VLUE | — (not in the second source) | |
| USMV | 2011-10-20 → 2026-07-31: 1 (3,708) | 0 |
| SPLV | 2011-05-05 → 2026-07-31: 1 (3,832) | 1 |
| NOBL | — (not in the second source) | |
| RSP | 2003-07-07 → 2006-04-26: 4 (404); 2006-04-27 → 2026-07-31: 1 (5,097) | 36 |
| IWF | 2002-08-19 → 2026-04-28: 4 (5,833); 2026-04-29 → 2026-07-31: 1 (65) | 92 |

Distribution yield by year (sum of the year's amounts ÷ this file's close the day before each ex-date), %: as paid for the four split funds, and after rule H6 for all:

| ticker | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MTUM as paid | — | — | — | 4.43 | 4.36 | 4.54 | 5.71 | 4.53 | 4.65 | 6.26 | 4.35 | 2.24 | 7.25 | 6.03 | 3.01 | 3.88 | 1.61 |
| QUAL as paid | — | — | — | 2.65 | 5.66 | 6.57 | 8.07 | 7.55 | 7.27 | 6.96 | 6.80 | 5.32 | 6.12 | 5.42 | 4.13 | 3.98 | 2.42 |
| NOBL as paid | — | — | — | 0.61 | 3.40 | 4.05 | 4.26 | 3.70 | 4.60 | 4.00 | 4.99 | 4.09 | 3.93 | 4.29 | 4.05 | 4.34 | 2.03 |
| IWF as paid | 5.68 | 5.56 | 6.61 | 5.75 | 5.51 | 5.51 | 5.86 | 4.83 | 4.62 | 4.37 | 3.43 | 2.22 | 3.33 | 3.04 | 1.99 | 1.59 | 0.53 |
| **MTUM** after H6 | — | — | — | 1.11 | 1.09 | 1.14 | 1.43 | 1.13 | 1.16 | 1.56 | 1.09 | 0.56 | 1.81 | 1.51 | 0.75 | 0.97 | 0.40 |
| **QUAL** after H6 | — | — | — | 0.66 | 1.42 | 1.64 | 2.02 | 1.89 | 1.82 | 1.74 | 1.70 | 1.33 | 1.53 | 1.35 | 1.03 | 0.99 | 0.61 |
| **VLUE** | — | — | — | 1.43 | 1.70 | 2.32 | 2.24 | 2.33 | 2.37 | 2.82 | 2.95 | 2.33 | 3.03 | 2.89 | 2.68 | 2.43 | 1.26 |
| **USMV** | — | 0.30 | 1.95 | 2.29 | 2.00 | 2.04 | 2.22 | 1.85 | 2.04 | 1.97 | 2.06 | 1.39 | 1.61 | 1.92 | 1.70 | 1.50 | 1.05 |
| **SPLV** | — | 2.23 | 3.07 | 2.76 | 2.42 | 2.33 | 2.10 | 2.16 | 2.12 | 2.22 | 2.28 | 1.69 | 2.14 | 2.48 | 1.96 | 2.01 | 1.87 |
| **NOBL** after H6 | — | — | — | 0.31 | 1.70 | 2.02 | 2.13 | 1.85 | 2.30 | 2.00 | 2.49 | 2.05 | 1.96 | 2.15 | 2.03 | 2.17 | 1.54 |
| **RSP** | 1.40 | 1.53 | 1.68 | 1.41 | 1.51 | 1.64 | 1.28 | 1.62 | 1.85 | 1.79 | 2.19 | 1.40 | 1.82 | 1.75 | 1.56 | 1.71 | 1.19 |
| **IWF** after H6 | 1.42 | 1.39 | 1.65 | 1.44 | 1.38 | 1.38 | 1.46 | 1.21 | 1.15 | 1.09 | 0.86 | 0.56 | 0.83 | 0.76 | 0.50 | 0.40 | 0.27 |

## Distributions: does the price fall by about the amount on the ex-date?

Over every distribution from 2010: the close-to-close fall on the ex-date minus SPY's fall that day (to take out the market), against the amount, both as % of the previous close. If the amounts are on the price file's basis, the median of (fall − amount) is near zero; amounts four times too large put it near minus three quarters of the amount.

| ticker | ex-dates | median amount, as paid | median (fall − amount), as paid | median amount after H6 | median (fall − amount) after H6 |
|---|---|---|---|---|---|
| MTUM | 54 | 1.10% | -0.88 pp | 0.28% | +0.03 pp |
| QUAL | 54 | 1.48% | -1.14 pp | 0.37% | -0.04 pp |
| VLUE | 55 | 0.60% | +0.04 pp | 0.60% | +0.04 pp |
| USMV | 62 | 0.45% | +0.02 pp | 0.45% | +0.02 pp |
| SPLV | 185 | 0.19% | +0.07 pp | 0.19% | +0.07 pp |
| NOBL | 52 | 1.01% | -0.50 pp | 0.51% | -0.02 pp |
| RSP | 67 | 0.40% | -0.01 pp | 0.40% | -0.01 pp |
| IWF | 68 | 1.14% | -0.88 pp | 0.28% | +0.00 pp |

## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available

Total return from the loader's rules (H4 on Eikon prices and Refinitiv distributions) against EODHD's adjusted close on the dates both have, from 2010: growth of 1, this file ÷ EODHD, with the distributions as paid and after rule H6; daily-return correlation and days with a return gap over 0.5 pp after H6.

| ticker | common days | growth ratio, as paid (a year) | growth ratio after H6 (a year) | daily-return correlation | days with return gap > 0.5 pp |
|---|---|---|---|---|---|
| MTUM | 3,337 (2013-04-18 → 2026-07-31) | 1.5992 (+3.60%) | 1.0001 (+0.00%) | 0.99913 | 4 |
| QUAL | — (not in the second source) | | | | |
| VLUE | — (not in the second source) | | | | |
| USMV | 3,708 (2011-10-20 → 2026-07-31) | 1.0001 (+0.00%) | 1.0001 (+0.00%) | 0.99971 | 0 |
| SPLV | 3,832 (2011-05-05 → 2026-07-31) | 1.0000 (-0.00%) | 1.0000 (-0.00%) | 0.99966 | 2 |
| NOBL | — (not in the second source) | | | | |
| RSP | 4,169 (2010-01-04 → 2026-07-31) | 1.0000 (+0.00%) | 1.0000 (+0.00%) | 0.99992 | 0 |
| IWF | 4,169 (2010-01-04 → 2026-07-31) | 1.7015 (+3.26%) | 1.0006 (+0.00%) | 0.99993 | 0 |

## After the launch period (rule H5: the first 1 January at least 12 months after the first day; not before 2010), total return with H6

| ticker | first day in the file | start under H5 | years | missing vs SPY's days | jumps > 8% (SPY's return that day) | total return a year (SPY, same days) | weekly correlation with SPY | usable for a 52-week rule (≥ 5 years)? |
|---|---|---|---|---|---|---|---|---|
| MTUM | 2013-04-18 | 2015-01-01 | 11.9 | 0 | 2020-03-12 -9.4% (-10.1%); 2020-03-16 -13.2% (-11.6%); 2020-03-24 +10.1% (+8.7%); 2025-04-09 +10.0% (+10.0%); 2026-11-05 -8.6% (-4.5%) | 13.3% (13.1%) | 0.877 | yes |
| QUAL | 2013-07-18 | 2015-01-01 | 11.9 | 0 | 2020-03-12 -9.8% (-10.1%); 2020-03-13 +8.7% (+8.2%); 2020-03-16 -10.7% (-11.6%); 2020-03-24 +9.9% (+8.7%); 2025-04-09 +8.3% (+10.0%) | 12.8% (13.1%) | 0.983 | yes |
| VLUE | 2013-04-18 | 2015-01-01 | 11.9 | 8 (2015-01-05, 2015-03-31, 2015-06-17, 2015-06-23, 2015-07-17, 2015-08-07, 2015-12-03, 2016-01-26) | 2020-03-09 -9.9% (-8.1%); 2020-03-12 -12.0% (-10.1%); 2020-03-13 +9.5% (+8.2%); 2020-03-16 -13.6% (-11.6%); 2020-03-24 +9.6% (+8.7%) | 11.7% (13.1%) | 0.896 | yes |
| USMV | 2011-10-20 | 2013-01-01 | 13.9 | 0 | 2020-03-12 -9.9% (-10.1%); 2020-03-16 -10.6% (-11.6%); 2020-03-24 +8.3% (+8.7%) | 11.3% (14.2%) | 0.894 | yes |
| SPLV | 2011-05-05 | 2013-01-01 | 13.9 | 0 | 2020-03-12 -9.9% (-10.1%); 2020-03-16 -13.2% (-11.6%); 2020-03-17 +8.6% (+5.3%); 2020-03-24 +9.4% (+8.7%) | 9.9% (14.2%) | 0.792 | yes |
| NOBL | 2013-10-10 | 2015-01-01 | 11.9 | 0 | 2020-03-12 -9.6% (-10.1%); 2020-03-16 -10.7% (-11.6%); 2020-03-24 +9.5% (+8.7%) | 9.9% (13.1%) | 0.871 | yes |
| RSP | 2003-07-07 | 2010-01-01 | 16.9 | 0 | 2020-03-09 -9.1% (-8.1%); 2020-03-12 -10.6% (-10.1%); 2020-03-16 -12.8% (-11.6%); 2020-03-24 +10.1% (+8.7%) | 12.3% (13.6%) | 0.951 | yes |
| IWF | 2002-08-19 | 2010-01-01 | 16.9 | 0 | 2020-03-12 -9.6% (-10.1%); 2020-03-13 +8.6% (+8.2%); 2020-03-16 -12.5% (-11.6%); 2020-03-24 +8.7% (+8.7%); 2025-04-09 +11.0% (+10.0%) | 15.0% (13.6%) | 0.962 | yes |

## Findings and how a loader would handle them

- **F-RA-D15 — four funds split, and the split basis differs between the price and distribution files (as F-RA-D14 for IJR).** IWF 4-for-1 on 2026-04-27; MTUM 4-for-1 on 2026-11-03; NOBL 2-for-1 on 2026-05-26; QUAL 4-for-1 on 2026-11-03. The price file divides every earlier close by the split ratio; the distribution table holds the amounts as paid. Read together, every pre-split distribution counts two or four times: the as-paid yields above are two to four times the funds' real ones, the ex-date falls are far smaller than the amounts (median gap as paid: IWF -0.88 pp, MTUM -0.88 pp, NOBL -0.50 pp, QUAL -1.14 pp), and total return is overstated against the second source (IWF +3.26% a year, MTUM +3.60% a year). **Handled by rule H6** (`analysis/approved_data.split_adjust_dividends`, extended with these four splits): a distribution before a split is divided by its ratio. After H6 the ex-date gaps are IWF +0.00 pp, MTUM +0.03 pp, NOBL -0.02 pp, QUAL -0.04 pp, and total return agrees with the second source (IWF +0.00% a year, MTUM +0.00% a year). No distribution falls within 10 days of a split.
- **The split table is not a full history:** it holds only these 2026 splits, but the second source shows RSP's earlier 4-for-1 (2003-07-07 → 2006-04-26 at 4)
  (before 2010, so outside the series and the distribution history). The review therefore also checks every fund's yields and ex-date falls: after H6 they show no other split from 2010.
- **Closes only for six funds** (VLUE, USMV, SPLV, NOBL, RSP, IWF): open, high and low are blank on almost every row (F-RA-D13); **rule H3** (fill = close). MTUM, QUAL have open, high and low on every row; a study comparing these funds should still fill all of them at the close, so that every fund trades at the same price point.
- **No second source for QUAL, VLUE, NOBL**; the second source ends on 2026-07-31. These three rest on the ex-date falls and yields above (which agree after H6) and on the weekly correlation with SPY.
- **Missing sessions** fall in the launch periods, which rule H5 drops, except VLUE's 8 (2015-01-05, 2015-03-31, 2015-06-17, 2015-06-23, 2015-07-17, 2015-08-07, 2015-12-03, 2016-01-26); a study on the funds' common calendar drops those days for every fund.
- **Jumps over 8%** are on days SPY moved sharply the same way (March 2020, 2025-04-09), except MTUM on 2026-11-05 (-8.6%, SPY -4.5%): the day after its split took effect and after the second source ends, so it cannot be checked here; QUAL, split the same day, shows no jump. Open item.
- **Spot checks against the issuers** (iShares, Invesco, ProShares): not done here; open item, as for the other US ETFs.

## Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| us_factor_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4, H5 and H6; every fund has at least 11.9 years after H5 |
| us_factor_dividends.csv | PASS WITH NOTES | amounts as paid, so read only through rule H6 (pre-split amounts divided by the split ratio); total return then agrees with the second source where it exists |
| us_factor_splits.csv | PASS WITH NOTES | the four 2026 splits that H6 needs; not a full history (RSP's 2006 split is missing; before the 2010 start) |
| us_factor_eodhd_check.csv | not for approval | the second source, read only by this review |
