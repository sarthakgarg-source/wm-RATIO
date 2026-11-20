# Data review — US ETFs (SPY and 13 broad commodity ETFs)

Reviewed 2026-11-19 by `data/data_review_us.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items.

- `us_etf_ohlcv.csv`: sha256 `84729a36edc18c343c4d1768cec41c577d0ff5850265e2f3aabdb1e841904e85`, 1,227,105 bytes
- `us_etf_dividends.csv`: sha256 `04392d6ff812359e1f2acbf89efd038dc81565315f340bc471cdc9615d0251e1`, 12,646 bytes
- `us_etf_eodhd_check.csv` (second source, read by this review only; not for approval): sha256 `34ab1f72d8677bdf2548a5096f408828d8fc92fc75de51ab972f1d30fac8692b`, 662,468 bytes

## What the files are

- **us_etf_ohlcv.csv** — daily open/high/low/close/volume of SPY and the 13 ETFs of the desk's commodity-ETF comparison (BCI, BCD, CMDY, CERY, PDBC, COMT, SDCI, HGER, COM, FTGC, NBCM, PIT, GCC), US dollars, prices **as traded (not adjusted for distributions)**. Source: the research Postgres, table `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_etf.py`.
- **us_etf_dividends.csv** — one row per distinct distribution event (ticker, ex-date, type, gross amount in USD), from `us_corp_action_div_master_b_ed` (Refinitiv corporate actions). The table stores each event once per pull window; the fetch keeps distinct events. Its history starts in 2010.
- **Job:** signal, fill price (OHLC4) and valuation of the US pairs (ETF ÷ SPY) in exploratory studies.
- **Meaning of the price used by the loader:** total return — rule H4 in `analysis/approved_data.py` (`total_return`): prices before each ex-date are scaled by (1 − D ÷ the previous close), so every distribution is reinvested; then rules H1–H3 as for the NSE files. Series start on 2010-01-01 (no earlier dividend history).

## Series checks (from 2010-01-01)

| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | raw jumps > 8% | jumps > 8% after H4 | distributions (sum, largest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPY | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 67 ($83.70, $1.99) |
| GCC | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 11 ($15.27, $6.45) |
| FTGC | 3,251 | 2013-10-23 | 2026-11-18 | 13.1 | 0 | 0 | 37 | 0 | 5 | 0 | 0 | 0 | 0 | 4 | 2 | 22 ($11.06, $3.66) |
| COMT | 3,023 | 2014-10-16 | 2026-11-18 | 12.1 | 0 | 0 | 18 | 0 | 2 | 0 | 0 | 0 | 0 | 6 | 2 | 25 ($25.56, $8.40) |
| PDBC | 3,008 | 2014-11-07 | 2026-11-18 | 12.0 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 5 | 11 ($12.90, $5.39) |
| BCI | 2,397 | 2017-03-31 | 2026-11-18 | 9.6 | 0 | 0 | 26 | 0 | 3 | 0 | 0 | 0 | 0 | 4 | 1 | 10 ($15.46, $4.47) |
| BCD | 2,314 | 2017-03-31 | 2026-11-18 | 9.6 | 0 | 0 | 109 | 0 | 14 | 0 | 0 | 0 | 0 | 1 | 0 | 10 ($13.28, $5.33) |
| COM | 2,374 | 2017-03-30 | 2026-11-18 | 9.6 | 0 | 0 | 50 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 27 ($10.16, $3.01) |
| CMDY | 2,153 | 2018-04-05 | 2026-11-18 | 8.6 | 0 | 0 | 16 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 8 ($22.65, $7.96) |
| SDCI | 2,147 | 2018-05-03 | 2026-11-18 | 8.5 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 19 ($12.56, $3.15) |
| HGER | 1,198 | 2022-02-10 | 2026-11-18 | 4.8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 4 ($4.14, $1.76) |
| NBCM | 1,022 | 2022-10-24 | 2026-11-18 | 4.1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 ($4.09, $1.91) |
| PIT | 980 | 2022-12-22 | 2026-11-18 | 3.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 ($9.37, $4.71) |
| CERY | 554 | 2024-09-05 | 2026-11-18 | 2.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 ($1.56, $1.43) |

### Notes per series

- **SPY** raw jumps > 8%: 2020-03-09 -8.1%; 2020-03-12 -10.1%; 2020-03-13 +8.2%; 2020-03-16 -11.6%; 2020-03-24 +8.7%; 2025-04-09 +10.0%
- **SPY** jumps > 8% after H4: 2020-03-09 -8.1%; 2020-03-12 -10.1%; 2020-03-13 +8.2%; 2020-03-16 -11.6%; 2020-03-24 +8.7%; 2025-04-09 +10.0%
- **GCC** raw jumps > 8%: 2021-10-08 -9.2% (an ex-date); 2022-03-09 -10.1%; 2022-10-25 -19.5% (an ex-date); 2026-10-27 -28.3% (an ex-date)
- **GCC** jumps > 8% after H4: 2022-03-09 -10.1%
- **FTGC** misses 37 of SPY's trading days: 2013-10-30, 2013-11-05, 2013-11-06, 2013-11-08, 2013-11-11, 2013-11-12, 2013-11-13, 2013-11-15, 2013-11-18, 2013-11-19, 2013-11-20, 2013-11-22 …
- **FTGC** gaps over 5 calendar days end on: 2013-11-14, 2013-11-21, 2013-12-18, 2013-12-27, 2014-01-10
- **FTGC** raw jumps > 8%: 2021-03-26 +9.1%; 2022-03-09 -8.9%; 2022-12-15 -10.5% (an ex-date); 2025-12-16 -15.9% (an ex-date)
- **FTGC** jumps > 8% after H4: 2021-03-26 +9.1%; 2022-03-09 -8.9%
- **COMT** misses 18 of SPY's trading days: 2014-10-21, 2014-11-20, 2014-11-21, 2014-11-24, 2014-11-25, 2014-12-01, 2014-12-05, 2014-12-11, 2014-12-17, 2014-12-18, 2014-12-19, 2014-12-22 …
- **COMT** gaps over 5 calendar days end on: 2014-11-26, 2014-12-23
- **COMT** raw jumps > 8%: 2018-12-18 -10.7% (an ex-date); 2020-03-09 -10.8%; 2021-12-13 -17.1% (an ex-date); 2022-03-09 -11.1%; 2022-12-13 -24.9% (an ex-date); 2025-12-16 -8.9% (an ex-date)
- **COMT** jumps > 8% after H4: 2020-03-09 -10.8%; 2022-03-09 -11.1%
- **PDBC** misses 17 of SPY's trading days: 2014-11-12, 2014-11-14, 2014-11-20, 2014-11-26, 2014-12-05, 2014-12-08, 2014-12-15, 2015-02-02, 2015-04-09, 2015-06-18, 2015-07-02, 2015-07-10 …
- **PDBC** raw jumps > 8%: 2015-07-13 +24.2%; 2015-07-14 -19.5%; 2015-07-17 -10.4%; 2020-03-09 -8.2%; 2021-12-03 -30.9% (an ex-date); 2021-12-20 -13.0% (an ex-date); 2022-03-09 -8.2%; 2022-12-19 -12.7% (an ex-date)
- **PDBC** jumps > 8% after H4: 2015-07-13 +24.2%; 2015-07-14 -19.5%; 2015-07-17 -10.4%; 2020-03-09 -8.2%; 2022-03-09 -8.2%
- **BCI** misses 26 of SPY's trading days: 2017-04-03, 2017-04-07, 2017-04-10, 2017-04-12, 2017-04-13, 2017-04-17, 2017-04-21, 2017-04-26, 2017-05-02, 2017-05-05, 2017-05-09, 2017-05-10 …
- **BCI** gaps over 5 calendar days end on: 2017-04-18, 2017-05-22, 2017-06-15
- **BCI** raw jumps > 8%: 2021-12-21 -16.8% (an ex-date); 2022-03-09 -8.3%; 2022-12-16 -18.7% (an ex-date); 2025-12-30 -14.3% (an ex-date)
- **BCI** jumps > 8% after H4: 2022-03-09 -8.3%
- **BCD** misses 109 of SPY's trading days: 2017-04-04, 2017-04-05, 2017-04-21, 2017-05-08, 2017-05-11, 2017-05-18, 2017-05-19, 2017-05-24, 2017-05-26, 2017-06-05, 2017-06-07, 2017-06-08 …
- **BCD** gaps over 5 calendar days end on: 2017-06-13, 2017-06-20, 2017-06-30, 2017-08-17, 2017-09-07, 2017-09-14, 2017-09-26, 2017-10-25, 2017-11-06, 2017-11-15, 2017-11-27, 2017-12-04 …
- **BCD** raw jumps > 8%: 2025-12-30 -15.0% (an ex-date)
- **COM** misses 50 of SPY's trading days: 2017-04-03, 2017-04-04, 2017-04-07, 2017-04-11, 2017-04-28, 2017-05-02, 2017-05-03, 2017-05-04, 2017-05-05, 2017-05-10, 2017-05-15, 2017-05-25 …
- **COM** gaps over 5 calendar days end on: 2017-05-08, 2017-06-14, 2017-07-19, 2017-12-06
- **COM** raw jumps > 8%: 2021-12-21 -8.9% (an ex-date)
- **CMDY** misses 16 of SPY's trading days: 2018-04-06, 2018-04-10, 2018-04-11, 2018-04-12, 2018-04-17, 2018-04-20, 2018-04-26, 2018-05-07, 2018-05-16, 2018-05-17, 2018-05-18, 2018-05-22 …
- **CMDY** gaps over 5 calendar days end on: 2018-05-21, 2018-06-01
- **CMDY** raw jumps > 8%: 2021-12-13 -15.6% (an ex-date); 2025-12-16 -13.3% (an ex-date)
- **SDCI** misses 2 of SPY's trading days: 2018-05-16, 2018-06-01
- **SDCI** raw jumps > 8%: 2021-09-27 -8.7% (an ex-date); 2021-12-28 -9.5% (an ex-date); 2022-03-09 -9.1%; 2022-10-12 -14.8% (an ex-date); 2022-12-13 -13.4% (an ex-date)
- **SDCI** jumps > 8% after H4: 2022-03-09 -9.1%
- **HGER** raw jumps > 8%: 2022-03-09 -11.1%
- **HGER** jumps > 8% after H4: 2022-03-09 -11.1%

## Distributions: does the price fall by about the amount on the ex-date?

For each distribution of at least 2% of the price: the close-to-close fall on the ex-date against the amount (both as % of the previous close). The fall also contains the day's market move, so only large gaps between the two are flags.

| ticker | ex-dates (≥ 2% of price) | median (fall − amount) | largest gaps |
|---|---|---|---|
| GCC | 6 | +0.1 pp | 2024-10-28 amount 2.3%, fall 3.6%; 2022-10-25 amount 18.4%, fall 17.7%; 2025-10-28 amount 5.0%, fall 5.5% |
| FTGC | 3 | +0.5 pp | 2025-12-16 amount 13.8%, fall 14.7%; 2022-12-15 amount 9.5%, fall 9.9%; 2021-12-23 amount 6.8%, fall 6.6% |
| COMT | 8 | +0.3 pp | 2022-12-13 amount 23.6%, fall 22.1%; 2025-12-16 amount 7.3%, fall 8.5%; 2023-12-20 amount 4.8%, fall 6.0% |
| PDBC | 8 | -0.4 pp | 2025-12-22 amount 3.7%, fall 2.6%; 2021-12-20 amount 11.5%, fall 12.2%; 2017-12-18 amount 3.9%, fall 3.3% |
| BCI | 6 | -0.3 pp | 2021-12-21 amount 16.8%, fall 15.5%; 2025-12-30 amount 14.1%, fall 13.3%; 2022-12-16 amount 16.6%, fall 17.1% |
| BCD | 5 | -0.1 pp | 2021-12-21 amount 7.9%, fall 6.7%; 2025-12-30 amount 14.6%, fall 13.9%; 2022-12-16 amount 4.9%, fall 5.5% |
| COM | 3 | -0.7 pp | 2021-12-21 amount 9.6%, fall 8.5%; 2022-06-22 amount 3.5%, fall 4.2%; 2022-12-20 amount 2.9%, fall 2.2% |
| CMDY | 6 | +0.4 pp | 2022-12-13 amount 3.8%, fall 2.3%; 2023-12-20 amount 4.8%, fall 5.9%; 2019-12-16 amount 2.2%, fall 1.1% |
| SDCI | 6 | -0.0 pp | 2022-12-13 amount 14.4%, fall 12.5%; 2025-12-22 amount 3.5%, fall 2.3%; 2024-12-27 amount 5.6%, fall 6.3% |
| HGER | 3 | -0.3 pp | 2024-12-20 amount 3.2%, fall 2.3%; 2025-12-12 amount 6.6%, fall 6.4%; 2023-12-21 amount 6.7%, fall 6.6% |
| NBCM | 3 | -0.8 pp | 2024-12-18 amount 5.0%, fall 6.1%; 2025-12-19 amount 7.9%, fall 7.1%; 2023-12-19 amount 4.2%, fall 3.4% |
| PIT | 3 | -0.6 pp | 2025-12-22 amount 8.3%, fall 6.8%; 2023-12-18 amount 6.0%, fall 5.4%; 2024-12-23 amount 3.5%, fall 3.8% |
| CERY | 1 | +1.9 pp | 2025-12-29 amount 4.7%, fall 6.6% |

## Second source: EODHD adjusted closes (`us_ohlcv_eodhd`), where available

Daily total returns from the loader's rule (H4 on Eikon prices and Refinitiv dividends) against the daily returns of EODHD's adjusted close, on the dates both have; and the raw closes of the two sources.

| ticker | common days | raw closes differing > 0.5% | daily-return correlation | days with return gap > 0.5 pp | growth of 1, H4 ÷ EODHD over the common period |
|---|---|---|---|---|---|
| SPY | 4,169 (2010-01-04 → 2026-07-31) | 0 | 0.9997 | 0 | 1.0000 |
| GCC | — (not in the second source) | | | | |
| FTGC | 3,173 (2013-10-23 → 2026-07-31) | 21 | 0.9849 | 43 | 0.9993 |
| COMT | 2,946 (2014-10-16 → 2026-07-31) | 46 | 0.9824 | 76 | 1.0097 |
| PDBC | 2,904 (2014-11-07 → 2026-07-31) | 79 | 0.8729 | 132 | 1.0036 |
| BCI | 2,320 (2017-03-31 → 2026-07-31) | 0 | 0.9999 | 0 | 1.0000 |
| BCD | — (not in the second source) | | | | |
| COM | — (not in the second source) | | | | |
| CMDY | — (not in the second source) | | | | |
| SDCI | — (not in the second source) | | | | |
| HGER | 1,120 (2022-02-10 → 2026-07-31) | 0 | 0.9999 | 0 | 1.0000 |
| NBCM | — (not in the second source) | | | | |
| PIT | — (not in the second source) | | | | |
| CERY | 477 (2024-09-05 → 2026-07-31) | 0 | 1.0000 | 0 | 1.0000 |

## After the launch period (rule H5: each series starts on the first 1 January at least 12 months after its first day; not before 2010)

| ticker | listed (first day in the file) | start under H5 | years | rows with blank open/high/low | missing vs SPY's days | raw closes > 0.5% from EODHD | daily return gaps > 0.5 pp vs EODHD | jumps > 8% after H4 | usable for a 52-week rule (≥ 5 years)? |
|---|---|---|---|---|---|---|---|---|---|
| SPY | 1996-03-20 | 2010-01-01 | 16.9 | 99% | 0 | 0 | 0 | 2020-03-09 -8.1%, 2020-03-12 -10.1%, 2020-03-13 +8.2%, 2020-03-16 -11.6%, 2020-03-24 +8.7%, 2025-04-09 +10.0% | — |
| GCC | 2008-01-24 | 2010-01-01 | 16.9 | 99% | 0 | — | — | 2022-03-09 -10.1% | yes |
| FTGC | 2013-10-23 | 2015-01-01 | 11.9 | 99% | 0 | 16 | 31 | 2021-03-26 +9.1%, 2022-03-09 -8.9% | yes |
| COMT | 2014-10-16 | 2016-01-01 | 10.9 | 98% | 0 | 10 | 18 | 2020-03-09 -10.8%, 2022-03-09 -11.1% | yes |
| PDBC | 2014-11-07 | 2016-01-01 | 10.9 | 98% | 0 | 4 | 8 | 2020-03-09 -8.2%, 2022-03-09 -8.2% | yes |
| BCI | 2017-03-31 | 2019-01-01 | 7.9 | 98% | 0 | 0 | 0 | 2022-03-09 -8.3% | yes |
| BCD | 2017-03-31 | 2019-01-01 | 7.9 | 98% | 0 | — | — | 0 | yes |
| COM | 2017-03-30 | 2019-01-01 | 7.9 | 98% | 0 | — | — | 0 | yes |
| CMDY | 2018-04-05 | 2020-01-01 | 6.9 | 97% | 0 | — | — | 0 | yes |
| SDCI | 2018-05-03 | 2020-01-01 | 6.9 | 97% | 0 | — | — | 2022-03-09 -9.1% | yes |
| HGER | 2022-02-10 | 2024-01-01 | 2.9 | 94% | 0 | 0 | 0 | 0 | short — report only |
| NBCM | 2022-10-24 | 2024-01-01 | 2.9 | 94% | 0 | — | — | 0 | short — report only |
| PIT | 2022-12-22 | 2024-01-01 | 2.9 | 94% | 0 | — | — | 0 | short — report only |
| CERY | 2024-09-05 | 2026-01-01 | 0.9 | 80% | 0 | 0 | 0 | 0 | short — report only |

## Findings and how the loader handles them

- **F-RA-D12 — the launch periods trade thinly.** In each fund's first year or so the file misses sessions (BCD 92 in 2017, COM 48 in 2017, FTGC 26 in 2013 …) and its closes disagree with the second source (PDBC: 69 days over 0.5% apart in 2015, against 4 in 2016; COMT 2014–2015). PDBC's 2015-07-13 close of 25.51 against EODHD's 20.00 (then 21.00 against 19.20) is a bad print. **Handled by rule H5** (above): the launch period is dropped; afterwards the missing sessions and most disagreements are gone (table above).
- **F-RA-D13 — the source has closes only.** Open, high and low are blank on almost every row (volume on every row) until 2026-09. **Handled by rule H3** (as for three gold rows): the fill price is the close, so trades in a study fill at the next session's close instead of OHLC4; the engine counts these rows as stale. A study must say so.
- **Distributions are large** (up to 24% of the price at an ex-date: COMT 2022-12-13) and annual for most funds; they show up as the expected price falls on the ex-dates (table above). **Handled by rule H4** (total return). Without it a ratio study would read every December distribution as a collapse of the ETF.
- **Remaining disagreements with the second source after H5** are few: FTGC 16 and COMT 10 closes over 0.5% apart, most of them in March–May 2020, and PDBC 4 in 2016 — closing prints a fraction of a percent apart; not handled (no rule can tell which source is right).
- **No second source** for GCC, BCD, COM, CMDY, SDCI, NBCM, PIT; their dividends are checked only by the ex-date falls above.
- **2022-03-09** (−8% to −11% in most commodity ETFs) is real: the day commodities fell back from the spike after the invasion of Ukraine.
- **SPY** agrees with the second source to the cent on every common day; its jumps are March 2020 and 2025-04-09 (the tariff pause), all real.
- **Spot checks against a public source** (issuer sites, exchange closes): not done here; open item for the data owner, as F-RA-D-OPEN-1 for the NSE files.

## Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| us_etf_ohlcv.csv | PASS WITH NOTES | usable with rules H3 (fill = close), H4 (total return) and H5 (launch period dropped); nine ETFs have five or more years after H5 (GCC, FTGC, COMT, PDBC, BCI, BCD, COM, CMDY, SDCI), four are short (HGER, NBCM, PIT, CERY) |
| us_etf_dividends.csv | PASS WITH NOTES | history from 2010 only (hence the 2010 start); amounts agree with the ex-date price falls; total returns agree with EODHD where it exists (growth of 1 within 1%) |
| us_etf_eodhd_check.csv | not for approval | the second source, read only by this review |
