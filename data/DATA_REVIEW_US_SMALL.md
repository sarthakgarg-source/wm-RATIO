# Data review — US small-cap ETF (IJR)

Reviewed 2026-11-23 by `data/data_review_us_small.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items.

- `us_small_ohlcv.csv`: sha256 `726cb635d29845fe1645db33ae1c2ef3b85bd47332d2e9f55e1bdba0fb02a09d`, 200,991 bytes, 6,055 lines
- `us_small_dividends.csv`: sha256 `332d0ae744bc2c815757ad742e1018efa4067015fe96c580ec64ef39ecbc3db3`, 3,187 bytes, 68 lines
- `us_small_eodhd_check.csv` (second source, read by this review only; not for approval): sha256 `4ca6c7ddc9754a8345f5903253a5c324fe0480e24a7e34333558e9bccde3fe19`, 202,190 bytes

## What the files are

- **us_small_ohlcv.csv** — daily open/high/low/close/volume of IJR (iShares Core S&P Small-Cap ETF, which tracks the S&P SmallCap 600), US dollars, from `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_small.py` to 2026-11-18 (the last day of the approved SPY file).
- **us_small_dividends.csv** — one row per distinct distribution event (ex-date, type, gross amount in USD) from `us_corp_action_div_master_b_ed`; the history starts in 2010.
- **Job:** US small caps against SPY and against gold (GLD) in exploratory studies: signal, fill (the close, rule H3) and valuation.
- **Meaning of the price used by the loader** (`load_us_small()`): total return (rule H4) on the price file's split basis (rule H6, below), from 2010-01-04 (rule H5); then H1–H3.

## Series checks (from 2010-01-01)

| rows | first | last | years | duplicates | weekend rows | missing vs SPY's days | extra vs SPY | gaps > 5 days | blank close | flat runs ≥ 5 | raw jumps > 8% | rows with blank open/high/low |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 99% (open/high/low present from 2026-08-12) |

Raw jumps over 8%, with SPY's close-to-close return the same day: 2011-08-08 -8.7% (SPY -6.7%); 2020-03-09 -10.3% (SPY -8.1%); 2020-03-12 -12.0% (SPY -10.1%); 2020-03-16 -12.5% (SPY -11.6%); 2020-03-18 -9.7% (SPY -5.2%); 2020-03-24 +8.3% (SPY +8.7%); 2020-06-11 -8.5% (SPY -5.9%); 2025-04-09 +8.5% (SPY +10.0%).

## Splits: the price file is split-adjusted, the distribution table is not

The second source's raw close divided by this file's close, over every common day, in runs of the same value rounded to a whole number:

| from | to | days | median | min | max |
|---|---|---|---|---|---|
| 2002-08-16 | 2005-06-08 | 658 | 6.0000 | 5.8654 | 6.1066 |
| 2005-06-09 | 2017-01-18 | 2,923 | 2.0000 | 1.9847 | 2.0200 |
| 2017-01-19 | 2026-07-31 | 2,396 | 1.0000 | 1.0000 | 1.0000 |

So this file divides every close before IJR's 3-for-1 split of 2005-06-09 by 6 and every close before its 2-for-1 split of 2017-01-19 by 2: its closes are on today's share basis. The distribution amounts are not: before the split they are about twice the later ones (a share then was two of today's).

| year | distributions | sum ÷ this file's previous close (as paid) | sum ÷ the second source's raw previous close | sum ÷ this file's previous close after rule H6 |
|---|---|---|---|---|
| 2010 | 4 | 2.35% | 1.18% | 1.18% |
| 2011 | 4 | 2.10% | 1.05% | 1.05% |
| 2012 | 4 | 3.38% | 1.69% | 1.69% |
| 2013 | 4 | 2.27% | 1.13% | 1.13% |
| 2014 | 4 | 2.53% | 1.26% | 1.26% |
| 2015 | 4 | 2.83% | 1.42% | 1.42% |
| 2016 | 4 | 2.70% | 1.35% | 1.35% |
| 2017 | 4 | 1.28% | 1.28% | 1.28% |
| 2018 | 4 | 1.36% | 1.36% | 1.36% |
| 2019 | 4 | 1.52% | 1.52% | 1.52% |
| 2020 | 4 | 1.52% | 1.52% | 1.52% |
| 2021 | 4 | 1.57% | 1.57% | 1.57% |
| 2022 | 4 | 1.38% | 1.38% | 1.38% |
| 2023 | 4 | 1.46% | 1.46% | 1.46% |
| 2024 | 4 | 2.03% | 2.03% | 2.03% |
| 2025 | 4 | 1.51% | 1.51% | 1.51% |
| 2026 | 3 | 0.82% | — | 0.82% |

## Total return against the second source's adjusted close (growth of 1, this file ÷ EODHD, over the common days)

| period | H4 on the amounts as paid: ratio (a year) | H4 after rule H6: ratio (a year) |
|---|---|---|
| 2010-01-04 → 2017-01-18 | 1.0956 (+1.31%) | 1.0000 (+0.00%) |
| 2017-01-19 → 2026-07-31 | 1.0000 (+0.00%) | 1.0000 (+0.00%) |
| 2010-01-04 → 2026-07-31 | 1.0957 (+0.55%) | 1.0000 (+0.00%) |

After H6, on 4,169 common days (2010-01-04 → 2026-07-31): daily-return correlation 0.99993; days with a return gap over 0.5 pp: 0; raw closes (this file put back on the as-traded basis) more than 0.5% from the second source's: 0. The second source ends on 2026-07-31.

## Sanity against SPY (from 2010)

- Weekly log-return correlation with SPY (SPY's raw closes, as a reference only): 0.861.
- IJR total return (H4 + H6) 11.4% a year, 2010-01-04 → 2026-11-18.

## Findings and how the loader handles them

- **F-RA-D14 — the split basis differs between the two files.** Prices are split-adjusted (the 2-for-1 split of 2017-01-19 and the 3-for-1 of 2005-06-09); distribution amounts are as paid. Read together without care, every distribution before 2017-01-19 counts twice: total return comes out about 1.3 pp a year too high over 2010–2016 (table above). **Handled by a new rule H6** in `analysis/approved_data.split_adjust_dividends`: a distribution before a split is divided by the split ratio (IJR: 2 before 2017-01-19). After H6 the total return agrees with the second source in each period (table above). The 2005 split is before the 2010 start and the distribution history.
- **Closes only** (F-RA-D13, as the other US ETFs): open, high and low are blank on almost every row until 2026-08; **rule H3** (fill = close).
- **No missing sessions, no duplicates, no flat runs** against SPY's calendar from 2010. Every raw jump over 8% falls on a day SPY itself moved sharply the same way (listed with SPY's return above).
- **Spot checks against the issuer** (iShares): not done here; open item, as for the other US ETFs.

## Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| us_small_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4, H5 and H6; 16.9 years from 2010, the same days as SPY |
| us_small_dividends.csv | PASS WITH NOTES | amounts as paid, so read only through rule H6 (pre-split amounts halved); total return then agrees with the second source |
| us_small_eodhd_check.csv | not for approval | the second source, read only by this review |
