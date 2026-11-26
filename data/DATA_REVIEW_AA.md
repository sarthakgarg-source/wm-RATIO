# Data review — the desk's latest Indian series from its asset-allocation repository (wm-AA)

Reviewed 2026-11-26 by `data/data_review_aa.py`. The files are raw; no study may read them until the data owner lists them on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items. The files were copied by `data/fetch_aa.py` from wm-AA's commit `e9cf7b25b45b9b66b53388decb5552031da87a14` (2026-11-26), byte for byte (each copy's git blob hash equals the commit's).

- `aa_index_n100_pre2019.csv`: sha256 `167c73492dd424cadf6e665546321ffd31d7da35ce39142f93c6f7282d742089`, 158,099 bytes, 3,238 lines
- `aa_index_n100_2019_2026.csv`: sha256 `4563e0b66f1aeeb0e7c0926d43e944b8b1bfe52f8527136e6f55042dd04c677b`, 107,276 bytes, 1,952 lines
- `aa_index_mc100_pre2019.csv`: sha256 `6586a0a392cbd73c3e6a8ee5d9644ba255c11f0171cb6bc3fd1cee4e6b26a65a`, 158,933 bytes, 3,330 lines
- `aa_index_mc100_2019_2026.csv`: sha256 `97387340ac4803b78ecf1097695624c0e1514ac608c0a09aa3ac07f66659d129`, 91,767 bytes, 1,952 lines
- `aa_index_sc100_pre2019.csv`: sha256 `6b6ebc3a3144ec74017cf24365efd0845ba6c639ca1a43603a646c2914fc9c21`, 163,764 bytes, 3,716 lines
- `aa_index_sc100_2019_2026.csv`: sha256 `e2421301b6ed87af64dbb67f06671bb1ec7591dc1b7d512c64385e7018cda04c`, 88,267 bytes, 1,952 lines
- `aa_mauc1_daily.csv`: sha256 `61cde69e5ce326072127095e4442a0f860555201a996424cc2d98a613511cb86`, 316,665 bytes, 6,355 lines
- `aa_liquid_lbes.csv`: sha256 `26ae7f7c9e876ab41c31036926b418dbec69d5736774e5e91864cab1dd2c4ffd`, 167,127 bytes, 5,768 lines
- `aa_liquid_case.csv`: sha256 `ac17a7fb1f2610010842a622231a9b8b6a71fcb3fd7b56bef95196eab555a749`, 125,370 bytes, 5,631 lines
- `aa_db_check.csv` (second source from the research Postgres, read by this review only; not for approval): sha256 `9e81685c50a7ea307f4667f3fa2f7c2186787df3885206f9178dbd64dd10bf0b`, 2,434,977 bytes

## 1 · Nifty 100, Midcap 100, Smallcap 100

Each index comes in two files: to 2019-01-04 (wm-AA's D25) and 2019-01-01 to 2026-11-20 (its D25d, copied from the desk's production index file). The combined series takes the first file to 2018-12-31 and the second from 2019-01-01.

| index | rows | first | last | overlap 2019-01-01 → 01-04 equal? | dup | weekend | blank close | OHLC order errors | stale OHL (O=H=L≠C) | bars with O=H=L=C (last) | flat runs ≥ 5 | jumps > 8% | gaps > 5 days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N100 | 5,184 | 2005-11-30 | 2026-11-20 | yes (4 days) | 0 | 8 | 0 | 1 | 0 | 1 (2015-11-11) | 0 | 6 | 1 |
| MC100 | 5,276 | 2005-07-18 | 2026-11-20 | yes (4 days) | 0 | 8 | 0 | 1 | 0 | 1 (2015-11-11) | 0 | 8 | 1 |
| SC100 | 5,662 | 2004-01-01 | 2026-11-20 | yes (4 days) | 0 | 8 | 0 | 6 | 5 | 1,920 (2015-11-11) | 0 | 12 | 1 |

### Notes

- **N100** jumps over 8%: 2008-01-21 -9.7%; 2008-10-24 -12.6%; 2009-05-18 +15.9%; 2020-03-12 -8.7%; 2020-03-23 -13.7%; 2020-04-07 +8.1%
- **N100** weekend sessions: 2006-04-29, 2009-10-17, 2013-11-03, 2016-10-30, 2019-10-27, 2020-11-14, 2024-01-20, 2024-05-18
- **N100** gaps over 5 calendar days end on: 2014-10-07
- **MC100** jumps over 8%: 2006-06-08 -8.5%; 2008-01-21 -12.7%; 2008-10-24 -8.5%; 2009-05-18 +11.5%; 2015-08-24 -9.2%; 2020-03-12 -8.4%; 2020-03-23 -14.1%; 2024-06-04 -8.2%
- **MC100** weekend sessions: 2006-04-29, 2009-10-17, 2013-11-03, 2016-10-30, 2019-10-27, 2020-11-14, 2024-01-20, 2024-05-18
- **MC100** gaps over 5 calendar days end on: 2014-10-07
- **SC100** jumps over 8%: 2004-05-14 -8.2%; 2004-05-17 -11.8%; 2006-06-08 -9.3%; 2008-01-21 -12.9%; 2008-01-22 -9.3%; 2008-10-10 -9.4%; 2008-10-24 -9.9%; 2009-05-18 +8.9%; 2015-08-24 -11.2%; 2020-03-12 -10.0%; 2020-03-23 -14.1%; 2024-06-04 -8.6%
- **SC100** weekend sessions: 2006-04-29, 2009-10-17, 2013-11-03, 2016-10-30, 2019-10-27, 2020-11-14, 2024-01-20, 2024-05-18
- **SC100** gaps over 5 calendar days end on: 2014-10-07

### Against the approved index file and the research Postgres

The approved `index_ohlcv_all.csv` ends on 2026-08-28. The Postgres's index rows (`merged_historical_data_with_rics`) end on 2025-09-15, so they check the history, not the new rows.

| index | common days with the approved file | largest difference O / H / L / C | days that differ | days only in one of them (within the shared span) | new rows after the approved file (first → last) | common days with the Postgres | closes more than 0.5% from the Postgres |
|---|---|---|---|---|---|---|---|
| N100 | 5,128 | 0.00 / 0.00 / 0.00 / 0.00 | 0 | 0 only approved, 0 only new | 56 (2026-08-31 → 2026-11-20) | 4,893 | 2 |
| MC100 | 5,220 | 0.00 / 0.00 / 0.00 / 0.00 | 0 | 0 only approved, 0 only new | 56 (2026-08-31 → 2026-11-20) | 4,983 | 2 |
| SC100 | 5,606 | 0.00 / 0.00 / 0.00 / 0.00 | 0 | 0 only approved, 0 only new | 56 (2026-08-31 → 2026-11-20) | 5,371 | 33 |

## 2 · MCX gold (MAUc1, front-month continuation, rupees per 10 g)

wm-AA's cleaned daily series (its D24c): a desk export to 2026-11-11 and the Postgres's Eikon futures rows from 2026-11-12, with rows that have no close dropped and bars widened to span open, high, low and close.

| rows | first | last | dup | weekend | blank close | OHLC order errors | stale OHL | bars with O=H=L=C | flat runs ≥ 5 | jumps > 8% | gaps > 5 days | rows with zero volume |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6,354 | 2003-11-10 | 2026-11-20 | 0 | 470 | 0 | 0 | 0 | 15 | 0 | 8 | 2 | 4 |

Jumps over 8%: 2004-04-24 +45.9%; 2004-04-26 -45.6%; 2005-05-07 -inf%; 2005-05-09 +inf%; 2008-09-17 +8.1%; 2013-04-15 -8.6%; 2014-02-06 -9.1%; 2026-01-30 -12.4%. Weekend sessions: 2003-11-15, 2004-04-24, 2004-07-03, 2004-07-10, 2004-07-17, 2004-07-24, 2004-07-31, 2004-08-14, 2004-08-21, 2004-08-28 ….

Median daily volume (contracts) by year: 2003 46, 2004 1,159, 2005 4,428, 2006 32,552, 2007 22,059, 2008 45,476, 2009 32,922, 2010 37,965, 2011 31,732, 2012 31,714, 2013 24,290, 2014 13,929, 2015 13,552, 2016 13,910, 2017 8,173, 2018 7,470, 2019 11,162, 2020 11,663, 2021 6,664, 2022 5,288, 2023 4,935, 2024 5,519, 2025 9,238, 2026 5,973.

| compared with | span | common days | largest difference O / H / L / C | days that differ | days only in the other (within the shared span) | days only in this file |
|---|---|---|---|---|---|---|
| approved `MAUc1_daily.csv` (Eikon, the 2026-11-02 pull) | 2015-01-12 → 2026-09-08 | 2,999 | 0.0 / 128.0 / 399.0 / 471.0 | 2 (2023-10-16, 2026-09-08) | 1 (2015-11-11) | 0 () |
| Postgres `ohlcv_eikon_futures` MAUc1 | 2005-01-01 → 2026-11-25 | 6,056 | 0.0 / 373.0 / 6110.0 / 0.0 | 15 (2005-05-07, 2005-12-05, 2006-05-06, 2006-10-24, 2009-04-04, 2009-12-05 …) | 5 (2007-01-01, 2007-04-06, 2014-10-23, 2015-01-01, 2015-11-11) | 0 () |

Rows before the Postgres's first day (2005-01-01), which no second source covers: 298 (2003-11-10 → 2004-12-31).

## 3 · The liquid-ETF cash levels

- **aa_liquid_lbes.csv** (wm-AA's `LBES_Adjusted_Price_data`): date (dd/mm/yy), price — a cash level that starts at 1,000. wm-AA uses it as the cash leg.
- **aa_liquid_case.csv** (wm-AA's `liquidEtfFinal`): date (dd-mm-yyyy), liquid_case — the desk's LIQUIDCASE level (Zerodha Nifty 1D Rate Liquid ETF), for robustness.
- **Second source:** the Postgres's daily closes of LIQUIDCASE (2024-01-24 → 2026-11-25, its listing onward) and LIQUIDBEES (2015-01-01 → 2026-11-25).

| series | rows | first | last | dup | non-positive | days with zero return | days with a negative return | largest daily return | smallest | growth a year over the whole file |
|---|---|---|---|---|---|---|---|---|---|---|
| LBES | 5,767 | 2007-06-29 | 2026-11-25 | 0 | 0 | 9 (0%) | 0 | +0.092% (2008-03-23) | +0.000% (2026-08-13) | 4.67% |
| LIQUIDCASE level | 5,630 | 2007-06-29 | 2026-11-25 | 0 | 0 | 65 (1%) | 7 | +0.092% (2008-03-23) | -0.010% (2024-04-01) | 4.74% |

Growth by calendar year (year-end to year-end), %:

| year | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LBES | 6.59 | 2.83 | 4.07 | 6.12 | 6.05 | 5.97 | 5.94 | 4.96 | 4.27 | 3.88 | 4.08 | 3.66 | 2.49 | 2.44 | 3.95 | 5.85 | 5.90 | 4.94 | 4.20 |
| LIQUIDCASE level | 6.59 | 2.83 | 4.07 | 6.12 | 6.05 | 5.97 | 5.94 | 4.96 | 4.27 | 3.88 | 4.08 | 3.66 | 2.49 | 2.44 | 3.95 | 5.85 | 6.40 | 5.54 | 4.46 |

### The two levels against each other and against the Postgres

- **The same series before LIQUIDCASE existed:** on the 4,927 common days before LIQUIDCASE's first day (2024-01-24), LBES ÷ LIQUIDCASE level stays within 0.0000% of its first value: one series at two scales. From 2024-01-24 the ratio drifts -0.45% a year (LBES grows that much more slowly).
- **The LIQUIDCASE level against the traded fund:** on 703 common days from 2024-01-24, the level ÷ LIQUIDCASE's close is between 1.000000 and 1.000000 (the level is the fund's close exactly); the daily returns of the two correlate 1.0000 over 702 days.
- **The extended rows:** wm-AA extended LBES after 2026-08-09 and the LIQUIDCASE level after 2026-09-07 by compounding LIQUIDCASE's daily returns. Recomputed from the Postgres's closes: LBES 74 rows, largest return difference 3.20e-12 (0 rows without a Postgres return); LIQUIDCASE level 53 rows, 0.00e+00 (0 without).
- **LIQUIDBEES as traded:** its close in the Postgres stays between 999.84 and 1,000.15 (median 1,000.00) from 2015-01-01 to 2026-11-25: it pays its income as daily dividends in new units, so its price is not a return series. A cash leg needs a level such as these files, built from the fund's NAV with the dividends reinvested.

### Gold: the rows that are not prices, and Saturdays

- **The approved file's last day (2026-09-08)** differs from this file in low and close (approved: low 152,800, close 153,050; this file: low 152,401, close 152,579; the Postgres: low 152,401, close 152,579): the approved file holds a provisional print for its last day. The other difference, 2023-10-16, is a widened high that the loader's rule H2 makes equal.
- **Non-positive closes:** 1 (2005-05-07) — 2005-05-07 (Saturday) open 6,116, close 0, low 0 (the Postgres: low 6,110, close 0). The cleaning widened the bar to the zero close, so the low is zero too.
- **One-day spikes that revert** (a close more than 30% from both neighbours): 1 — 2004-04-24 (Saturday) 9,207 between 5,819 and 5,833, volume 0 (before the Postgres's first day, so no second source).
- **Saturday sessions:** 464, by year: 2003 1, 2004 22, 2005 52, 2006 52, 2007 48, 2008 50, 2009 48, 2010 48, 2011 52, 2012 52, 2013 35, 2015 1, 2020 2, 2025 1. MCX traded on Saturdays until 2013; later ones are special sessions. On the NSE calendar (the indices) they drop out; on gold's own weekly bars (W-FRI) a Saturday opens the next week.
- **After the proposed rules** (gold from 2005-01-01, non-positive closes dropped): 6,055 rows, 2005-01-01 → 2026-11-20; jumps over 8%: 2008-09-17 +8.1%; 2013-04-15 -8.6%; 2014-02-06 -9.1%; 2026-01-30 -12.4%. The 2014-02-05 / 02-06 pair (+5.5% on 267 contracts, then -9.1%) falls on the February contract's expiry and the roll to April, in both sources: the continuation's own behaviour, not an error; 2026-01-30 is the gold crash of that day.

## 4 · Findings and how a loader would handle them

- **The indices are the approved file, extended.** On every common day (from each index's first day to 2026-08-28) the three series equal the approved `index_ohlcv_all.csv` to the paisa, with no day missing on either side; they add 56 sessions each, 2026-08-31 → 2026-11-20. The approved file's handling applies unchanged (H1 stale OHL, H2 out-of-range bars). SC100's open, high and low equal its close on 1,919 of its 1,924 bars before 2011-10-03 (as in the approved file; wm-AA treats them as not data, so SC100's weekly bars before then are bars of closes); its 33 closes more than 0.5% from the Postgres are old days (2011-04-19, 2011-04-20, 2011-04-28, 2011-04-29 …) where the approved file and the Postgres's feed disagree, not new rows. No independent second source covers the 56 new sessions (the Postgres's index rows end on 2025-09-15): open item, with spot checks against NSE.
- **F-RA-D18 — gold: two rows that are not prices, and a thin start.** A zero close on 2005-05-07 (in both sources) and a one-day spike to 9,207 on 2004-04-24 (export only, zero volume)
  would read as a −100% day and a +58% day. The export's rows before 2005-01-01 have no second source and very thin trading (median 46 contracts a day in 2003, 1,159 in 2004). **Handled by two new loader rules:** H7, a row with a non-positive close is dropped; H8, MCX gold starts on 2005-01-01, the first day the Postgres covers. After them the series equals the Postgres on every common day except where the cleaning widened a bar (H2 would do the same).
- **F-RA-D17 — the liquid-ETF cash levels are one constructed series before 2024-01-24.** LBES and the LIQUIDCASE level are the same series at two scales until LIQUIDCASE's first trading day; wm-AA records no source for it (its F-AA-D09). Its yearly growth (2.4% in 2021 to 6.6% in 2008) is consistent with Indian overnight money rates less fees; that is a plausibility check, not a source. From 2024-01-24 the LIQUIDCASE level is LIQUIDCASE's traded close exactly; LBES grows 0.45% a year less (a dearer fund's NAV with dividends reinvested would do that; not verified) until 2026-08-09, and then compounds LIQUIDCASE's returns (checked against the Postgres). **Handled as a cash level:** a single price a day (open = high = low = close, so the fill is the close); every study that uses it must state the provenance gap. LBES is wm-AA's cash leg; the LIQUIDCASE level is the robustness series.
- **Not taken:** Micro250 (wm-AA has no newer file) and the Postgres's LIQUIDBEES price (about 1,000 every day: not a return series).
- **Spot checks against public sources** (NSE index closes, MCX settlement prices, the liquid ETFs' published NAVs): not done here; open item.

## 5 · Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| aa_index_{n100,mc100,sc100}_pre2019.csv, aa_index_{n100,mc100,sc100}_2019_2026.csv | PASS | equal to the approved index file on every common day; 56 new sessions each to 2026-11-20; rules H1–H2 as before |
| aa_mauc1_daily.csv | PASS WITH NOTES | read only through rules H7 (non-positive close dropped) and H8 (from 2005-01-01); then equal to the Postgres apart from widened bars; front-month continuation (rolls inside the return), Saturday sessions to 2013 |
| aa_liquid_lbes.csv | PASS WITH NOTES | a cash level; the source before 2026-08-10 is not recorded (F-RA-D17); the extended rows equal LIQUIDCASE's returns |
| aa_liquid_case.csv | PASS WITH NOTES | a cash level; equal to LIQUIDCASE's traded close from 2024-01-24; before that the same unrecorded series as LBES (F-RA-D17) |
| aa_db_check.csv | not for approval | the second source, read only by this review |
