# Data review — US gold ETF (GLD, for SPY against gold)

Reviewed 2026-11-20 by `data/data_review_us_gold.py`. The file is raw; no study may read it until the data owner lists it on `APPROVED_DATA.md`. Problems carry IDs `F-RA-D##` (counter in README.md) or open items.

- `us_gold_ohlcv.csv`: sha256 `75fa5f1bc47c0df7bf0f95b21c2d744d85c72fb7f039990d0dca1517fa4d3b92`, 179,468 bytes, 5,466 rows
- `us_gold_eodhd_check.csv` (second source, read by this review only; not for approval): sha256 `0c3cb0e02e7f3f621cebb4c590eb9162f486ae6effb333688e1715518cc03c9e`, 160,404 bytes

## What the file is

- **us_gold_ohlcv.csv** — daily open/high/low/close/volume of GLD (SPDR Gold Shares, a grantor trust holding gold bullion), US dollars, prices as traded. Source: the research Postgres, table `ohlcv_eikon_us` (Refinitiv Eikon), fetched read-only by `data/fetch_us_gold.py`, to 2026-11-18 (the last date of the approved SPY file).
- **Distributions:** the distribution table (`us_corp_action_div_master_b_ed`) has no GLD event, as expected: the trust pays none (its fee is taken from the gold it holds, so it is in the price). The price is therefore already a total return; rule H4 changes nothing.
- **Job:** signal, fill price and valuation of the GLD ÷ SPY pair in exploratory studies.
- **Meaning of the price used by the loader:** as for the approved US ETFs — rule H5 start (the first 1 January at least 12 months after the first day, not before 2010-01-01), rule H4 (no events, so no change), then rules H1–H3.

## Series checks (from 2010-01-01)

| ticker | rows | first | last | years | dup | weekend | missing vs SPY's days | extra vs SPY | gaps > 5d | blank close (whole file) | OHLC order errors | equal O=H=L ≠ C | flat runs ≥ 5 | jumps > 8% | rows with blank open/high/low |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLD | 4,246 | 2010-01-04 | 2026-11-18 | 16.9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 99% (open/high/low present from 2026-08-12) |

### Notes

- jumps > 8%: 2013-04-15 -9.2%; 2026-01-30 -10.8%

## Before 2010 (not used)

- The file starts on 2005-03-01 (Eikon; the fund listed on 2004-11-18, the second source's first day). Rule H5 gives a start of 2010-01-01: the fund's own launch period ended long before, but SPY's distribution history starts in 2010, so the pair cannot start earlier.
- 1,220 rows before 2010 are in the file and are not read by the loader.

## Second source: EODHD (`us_ohlcv_eodhd`)

Daily returns of the Eikon close against EODHD's adjusted close on the dates both have, and the raw closes of the two sources; and EODHD's adjusted ÷ raw close, which moves only if EODHD saw a distribution or split.

| period | common days | raw closes differing > 0.5% | daily-return correlation | days with return gap > 0.5 pp | growth of 1, Eikon ÷ EODHD adjusted | EODHD adjusted ÷ close (min, max) |
|---|---|---|---|---|---|---|
| whole file | 5,389 (2005-03-01 → 2026-07-31) | 0 | 0.9995 | 0 | 1.0002 | 1.0000, 1.0000 |
| from 2010 | 4,169 (2010-01-04 → 2026-07-31) | 0 | 0.9999 | 0 | 1.0002 | 1.0000, 1.0000 |

Largest raw-close differences from 2010: 2011-10-27 Eikon 169.55 vs EODHD 169.94; 2010-11-26 Eikon 132.82 vs EODHD 133.11; 2014-07-03 Eikon 127.00 vs EODHD 127.16; 2010-02-05 Eikon 104.55 vs EODHD 104.68; 2011-01-20 Eikon 131.04 vs EODHD 131.20.

## Sanity check: the same metal on MCX

Weekly log returns of GLD (USD) and the approved MCX gold series (INR), 2015-01-16 → 2026-09-11 (608 weeks): correlation 0.879; growth over the period GLD ×3.25, MCX gold ×5.53 (the two differ by the rupee against the dollar and by India's import duties). Same metal, different currency: MCX gold is not a substitute for GLD against SPY.

## Findings and how the loader handles them

- **Closes only (F-RA-D13, as for the other US ETFs):** open, high and low are blank on 99% of the rows from 2010. **Handled by rule H3**: the fill price is the close.
- **No distributions** in the source, as expected for a gold grantor trust; EODHD's adjusted close agrees (table above). Rule H4 makes no change.
- **Calendar:** 0 of SPY's trading days missing and 0 extra from 2010; gaps over 5 days: 0.
- **Jumps:** 2013-04-15 -9.2%; 2026-01-30 -10.8%. Each is in the second source too (the same day's EODHD return within 0.5 pp, table above): real moves of the gold price (2013-04-15 is the gold crash of April 2013), not bad prints.
- **Spot checks against a public source** (issuer site, exchange closes): not done here; open item for the data owner, as for the other files.

## Proposed verdicts (for the data owner)

| file | proposed verdict | why |
|---|---|---|
| us_gold_ohlcv.csv | PASS WITH NOTES | usable from 2010-01-01 (16.9 years) with rules H3 (fill = close) and H5; no distributions, so the price is the total return |
| us_gold_eodhd_check.csv | not for approval | the second source, read only by this review |
