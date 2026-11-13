# Research Playbook: how to test a trading idea

A short, plain-language version of [AGENT.md](AGENT.md), which is the full rulebook (1,328 lines).
Every example comes from the AA project in this repo (tests T01–T06). Where this file is brief,
the matching AGENT.md section is named so you can go deeper.

**Why any of this exists.** Most market "discoveries" turn out to be one of three things:
- **luck:** the idea worked on this data by chance;
- **look-ahead:** the backtest used information that wasn't available at the time;
- **a bug:** the code computes something other than what you think.

All three produce a number that looks exactly like a real edge. Every step below exists to catch
one of them before anyone acts on the number.

---

## Part 1: The flow

```
STEP 0   Review the data (once per project)         → data/APPROVED_DATA.md
STEP 1   Write the question in one sentence
STEP 2   Name the kind of experiment                → decides which "random" test you use
STEP 3   Set up the test folder                     → tests/T01/
STEP 4   Make sure the data is used right           → no peeking into the future
STEP 5   Write the full design, then LOCK it        → T01.md sections 1–3, plus a hash
STEP 6   Build the engine (the backtest)
STEP 7   Build an independent checker
STEP 8   Run everything and check the gates         → STOP if any gate fails
STEP 9   Read the results, in a fixed order
STEP 10  Log each claim as a finding
STEP 11  Publish a results page and keep history
```

### Step 0: Review the data (once per project, before the first test)

**For every input file, write down:**
- what it is and what job it does: signal, fill price, valuation, cash, benchmark or calendar;
- where it came from: the system, the query or script, and who fetched it;
- when it was fetched, and the last date it covers;
- its fingerprint: sha256, number of rows, first and last date;
- what the numbers mean: units, and what prices are adjusted for (splits, bonuses, dividends).
  In other words, is it price return or total return?

**Then check:**
- **The calendar against the exchange:** missing days, extra days (weekends, holidays) and duplicates.
- **The values:** blanks, zero or negative prices, low > close or close > high, the same value
  repeated for days, and big jumps. Explain every jump: a corporate action, a real event, or an error.
- **Two sources of the same thing:** compare their *returns* and choose one, with a reason.
- **At least 3 dates per series against a public source**, such as the NSE site or a factsheet.
- **A chart of every series.** You are looking for problems, not making it pretty.

**Give each file a verdict:** PASS, PASS WITH NOTES or FAIL. Every problem gets an ID, `F-<P>-D##`.

**Then make an approved-data list** (`data/APPROVED_DATA.md`):
- Tests may read **only** the files on this list.
- A script checks each file's fingerprint before every run, and the run stops on any mismatch.
- Nobody extends or refreshes data without the data owner's sign-off.

*In this repo:*
- [AA/data/DATA_REVIEW.md](AA/data/DATA_REVIEW.md) is the review.
- [AA/data/APPROVED_DATA.md](AA/data/APPROVED_DATA.md) is the approved list.
- [check_approved_data.py](AA/data/check_approved_data.py) checks the fingerprints.
- [approved_data.py](AA/analysis/approved_data.py) refuses to open any file not on the list.

**What skipping these checks has cost before (from AGENT.md):**

| Check skipped | What went wrong |
|---|---|
| What the price means | A liquid fund paying daily dividends had a flat price, so the cash leg earned 0% instead of about 5% a year |
| Adjustment basis | Prices not adjusted for demergers showed one-day falls of 22–44% that never happened to a holder |
| Extra dates | A cash file posted its accruals on Sundays (934 rows) |
| One convention per project | Two studies extended the same cash file in two different ways |

### Step 1: Frame the question

Write down:
- **The idea in one sentence:** what predicts what, over what horizon, in which universe.
- **Where it came from:** a paper, a colleague, an observation, or a hunch. A hunch is fine; call it one.
- **What decision the answer will change.** If the answer changes nothing, stop.
- **What is already known:** earlier tests, and how this one relates to them.
- **What this test is NOT asking:** the nearby question people will wrongly read it as answering.

*Example (T06):* "Holding the parent only when Nifty 100, Midcap 100 and Smallcap 100 are all BUY,
and 50/50 gold and cash otherwise, chooses *when* to hold better than random timings with the same
number of switches and the same weeks invested."

**Done when** a colleague can read the sentence and say what result would make it false.

### Step 2: Name the kind of experiment

The kind of experiment decides the unit you measure, the random comparison you use (the "null",
explained in Part 2) and the trap to watch for.

| Kind | The question | Random comparison (null) | Classic trap |
|---|---|---|---|
| **Stock-picking strategy** | Does the rule, run as a portfolio, beat chance after costs? | Same rules, ranking shuffled on each date (N4.1) | An unwritten trading calendar |
| **Timing rule** (when to be in or out) | Does it choose *when* better than chance? | Random timing with the same exposure (N4.9, N4.10) | Lower drawdown that comes only from holding less stock |
| **Cross-sectional signal** | Does a score order next period's returns? | Per-date shuffle, persistent random score, time shift | A null calmer than reality |
| **Event study** | Does a feature of an event move stocks? | Feature shuffled among same-day events (N4.8); placebo dates (N4.7) | An event dated before anyone could know it |
| **Filter or screen** | Does narrowing the universe help? | Exclude the same number of stocks at random (N4.6) | The filter removes the thing that worked |
| **Classifier, model or LLM** | Are its picks better than chance? | Labels shuffled within each date (N4.8) | The model already knows the outcome |
| **Parameter sweep** | Which setting is best? | Each setting's own null, plus a "best of sweep" null | Picking the best-looking setting, which is picking noise |

**Test at the unit where the decision is made.** In one study the average flagged stock earned
+0.84%, but run as a real 15-stock portfolio it earned −0.90%. An average over stocks is not a
portfolio.

### Step 3: Set up the test folder

```
<project>/
  CLAUDE.md or README.md   hard rules, the current test, the finding-ID counter
  data/                    the data review and APPROVED_DATA.md (shared by every test)
  analysis/                engines and verifiers (shared code)
  tests/
    T01/
      PLAN.md              the build plan, copied word for word, never overwritten (a revision is PLAN_v2.md)
      T01.md               the write-up; sections 1–3 are locked before any run
      TRACKER.md           which step is done, by whom, with notes
      run.sh               one command that regenerates everything
      check_prereg.py      proves sections 1–3 haven't changed
      analyse.py           reads only evidence/ and prints every table into analysis.log
      fill_writeup.py      pastes those tables into section 4
      report.py            builds the results page
      check_page.py        re-checks every number on the page
      evidence/<run>/      everything the run produced
```

**Rules:**
- Raw data is read-only. Never edit it; a derived series gets its own build script.
- A new question gets a new test number, and so does an old question under changed rules.
- Pick a finding prefix for the project (this repo uses `F-AA-`).

### Step 4: Make sure the data is used right (no look-ahead)

| Rule | What breaking it cost (from AGENT.md) |
|---|---|
| Every input has a date, and the **code asserts** it is on or before the decision date, counting the checks | Ranking on the entry day's own close turned +16.8% into +73.0% |
| Exits are point-in-time too | Selling at the average price of the very day the signal appeared added a fake +5.7 pp |
| Write the trading calendar down: which day the signal is read, which day and price the trade happens | An unwritten calendar was worth a fake +4.2 pp |
| Use adjusted prices and say what they include | An unadjusted 1:1 bonus looks like a −50% day |
| Use the universe **as it was on each date** | Today's top-1000 list applied to the past keeps only the survivors |
| A stock that stops trading is a loss, not a gap | Delistings cluster in cheap stocks and flatter value strategies |
| Date fundamentals by publication, not period end. In India, use period end + 45 days (quarters) or + 60 days (annual) | Earnings get used up to two months before anyone could read them |
| A model's knowledge has a date: record its training cutoff and only count decisions after it | In one study only two months of the window were clean |
| Report everything dropped: names, dates, missing bars | One day of missing bars would have cut 859 stocks to 40 |

### Step 5: Write the full design, then lock it

Write all of this **before running anything**:

1. **The hypothesis**, in a form that can fail, and the result you expect.
2. **What you already saw.** If you looked at the data, say exactly what you saw.
3. **The null:** what is randomised, what is held identical, and what isn't matched (see Part 2).
4. **The statistic** (CAGR, max drawdown…), its direction, and why it answers the question.
5. **The pass mark and the number of draws.** AGENT.md's default is the 95th percentile. The AA
   tests used ≥ 90th in at least 3 of 4 books.
6. **Validity gates:** checks that must pass before you may read any result (Step 8).
7. **Reportability gates:** when a number may not be quoted at all, for example fewer than 4 round trips.
8. **Pass criteria:** exact thresholds and counts.
9. **Robustness checks**, and how many settings or ideas were tried (multiplicity).
10. **The reference setting**, chosen by its position in the grid (the middle value), never by its result.
11. **The MDE:** the smallest edge the test could detect (see Part 2).
12. **What is descriptive only**, so it makes no claims.
13. **A plain-words walkthrough:** one series, one decision, one path, one random draw, one ranking,
    with blanks for the real numbers that the analysis script fills in later.

**Criteria rules:**
- Hold some data back unseen if you can.
- Make every criterion possible to meet, and state every count.
- Change one thing at a time.
- Optionally, write an **outcome map:** each pattern of results and the conclusion it leads to.

**Lock it.** Compute a sha256 of sections 1–3 and store it in the header. `check_prereg.py`
recomputes it before and after the run. After the run, sections 1–3 never change. Anything learned
later goes in a dated block marked "Added after the run".

*Example (T06):*
- **C1:** after-cost CAGR beats at least 90% of 2,000 random timings in at least 3 of 4 books.
- **C2:** the same test for worst fall (maximum drawdown).

### Step 6: Build the engine

The backtest convention (AGENT.md R1):
- **Decide** on session T's close.
- **Trade** on the next session (T+1) at that session's OHLC4 = (open + high + low + close) ÷ 4.
- **Hold units:** units = money × (1 − cost) ÷ price.
- **Value the book every day:** NAV = cash + Σ units × close.
- **Charge cost** on every trade, half per side. AA used 20.8 bps round trip, so 10.4 bps per side.
- **Measure** from the session before the first trade, so the first trade's cost is included.

**Build rules:**
- The random draws go through **the same code** as the strategy. Only the tested ingredient changes.
- Make it deterministic: fixed seeds from a stated formula, such as `default_rng([6, draw_id])`.
- Refuse, don't warn, when a setting combination would produce a misleading number.
- Write everything to `evidence/<run>/`:
  - a summary per setting, and **every** draw;
  - a decision log (one row per decision) and the daily NAV;
  - `audit.json`: the inputs and their fingerprints, the rules, the seeds, and the point-in-time
    check count and violations.

*Example (T06, one decision):*
- **Friday 2019-03-15:** all three signals were BUY, so the target is the parent.
- **Monday 2019-03-18:** trade at OHLC4. Value traded 2.0038 (both sides), cost 2.0038 × 0.104% = 0.0021.
- **The whole path:** NAV 1.000 on 2019-01-02 → 6.849 on 2026-10-22, CAGR 28.6%, worst fall −18.4%.

### Step 7: Build an independent checker

A second program reaches the same numbers by a different route, so a bug would have to be written
twice to survive.
- **It imports nothing from the engine.** Ideally a different person or agent writes it from a
  written spec. T06's verifier was written by a separate agent that never saw the engine.
- **It uses a closed-form check where one exists.** For a timing rule at zero cost: total return =
  the product of (exit price ÷ entry price) over the in-spells.
- **Test the checker both ways:** it must pass on good output **and** fail on a deliberately
  corrupted copy (flipped decisions, a scaled NAV, and so on).
- **Regression gate:** after any code change, rerun the previous test and require identical numbers.
- **It exits non-zero** on any mismatch, on a missing row, or if it compared nothing.

### Step 8: Run everything and check the gates

`run.sh` runs everything in this order (T06's [run.sh](AA/tests/T06/run.sh)):

```bash
python data/check_approved_data.py   # 1. is the data exactly the approved data?
python tests/T06/check_prereg.py     # 2. is the locked design unchanged?
python analysis/t06_engine.py        # 3. strategy + all random draws
python analysis/verify_t06.py        # 4. independent recomputation
python tests/T06/analyse.py          # 5. gates + every table → analysis.log
python tests/T06/fill_writeup.py     # 6. tables into section 4 of T06.md
python tests/T06/report.py           # 7. results page
python tests/T06/check_page.py       # 8. every number on the page re-checked
python tests/T06/check_prereg.py     # 9. design still unchanged
```

**The gates:**

| Gate | Plain meaning |
|---|---|
| V1 | The independent checker agrees, and it catches a corrupted copy |
| V2 | No look-ahead: 0 violations, out of a check count greater than 0 |
| V3 | The random draws really kept what they promised (same exposure, same number of switches…) |
| V4 | Regression: earlier tests are reproduced exactly, and "always invested" matches the published NAV |
| V5 | The random generator is fair (only needed if you wrote your own sampler) |

**If any gate fails, stop.** Read nothing, fix it, and rerun into a new folder. Settings that fail a
reportability gate are marked **NOT REPORTABLE**, never quoted.

### Step 9: Read the results, in this order

0. **The worked example:** one decision, one path, one draw, one ranking. Check it by hand first.
1. **The null check:** did the random draws hold what they promised? Show any mismatch.
2. **The headline:** each result with its percentile, p-value, number of draws and edge, before
   **and** after cost.
3. **How much evidence there really is:**
   - Count independent observations, not rows. T06 had 409 weeks but only **9 round trips**.
   - Next to every FAIL, write "undetectable below X" (the MDE). Never write "no effect".
4. **Robustness:** do the pre-planned checks agree?
5. **Multiplicity:** how many settings and ideas were tried, and the best one judged against the best of chance.
6. **The verdict table:** PASS or FAIL per criterion, then a verdict per hypothesis.

Every number in section 4 is printed by `analyse.py`. Nothing is typed by hand.

### Step 10: Log each claim as a finding

- **One finding per claim**, with an ID from the counter (`F-<P>-S##` strategy, `M##` method, `D##` data).
- **Statuses:**
  - **CONFIRMED:** survived adversarial checks.
  - **PROVISIONAL:** real in-sample, but rests on one window.
  - **REFUTED:** tested and killed.
  - **FIXED:** a defect found and corrected.
  - **OPEN:** a known unknown.
- **Every finding says what would prove it wrong.** A claim that nothing could disprove is an opinion.
- **Record killed ideas as carefully as survivors.**
- **List what the test does NOT establish,** in two separate lists: tested but inconclusive, and not tested at all.
- **List the next steps,** ordered by how much each would change a decision per unit of effort.

### Step 11: Publish a results page and keep history

- **The page is built by a script** from the evidence, never with typed numbers. It contains:
  - the answer;
  - the data;
  - **one backtest end to end:** inputs → signal → trade → units → daily value → curve →
    statistics → one random draw → ranking;
  - the headline and the verdict table;
  - formulas, with **start and end dates on every number and chart**.
- **`check_page.py` re-checks every number on the page.** Publish only when it reports 0 failures.
- **Never edit old results.** Put a banner on the old write-up saying what replaced it, and set its
  status to SUPERSEDED.

---

## Part 2: The checks to perform (master list)

| # | Check | What it catches | When | In this repo |
|---|---|---|---|---|
| **A. Data** |||||
| A1 | Data review: inventory, calendar, values, spot checks, charts, verdicts | Wrong or misunderstood data | Once per project, and whenever data changes | `data/DATA_REVIEW.md`, `data_review.py` |
| A2 | Approved-data fingerprint check | Data silently changing between runs | Start of every run | `check_approved_data.py` |
| A3 | Rebuild against an official series | A wrong reconstruction | Once per derived series | Parent rebuild within 0.25 pp a year of the published NAV (G0) |
| **B. No look-ahead** |||||
| B1 | Point-in-time assertion, counted, inside the engine | Using the future | Every run | Gate V2: 0 of 3,266 violations |
| B2 | Trade strictly after the decision | Same-day look-ahead | Every run | Gate V2 |
| **C. Correctness** |||||
| C1 | Independent verifier (separate code) | Engine bugs | Every run | `verify_t06.py`: 973,395 comparisons |
| C2 | Corrupted-copy test of the verifier | A checker that can't fail | Once per verifier | Flipped decisions, scaled NAVs |
| C3 | Closed-form check | Subtle accounting bugs | Once per engine | Product of exit ÷ entry prices |
| C4 | Regression to the previous test | New code changing old answers | After any code change | Gate V4 |
| C5 | Design lock (hash) | Moving the goalposts | Before and after each run | `check_prereg.py` |
| **D. Skill or luck (the null)** |||||
| D1 | The rule against ≥ 1,000 random draws | Luck | Every test | 2,000 draws per book, cost and null |
| D2 | The null held what it promised | An unfair comparison | Every run | Gate V3 |
| D3 | A second, stricter or different null | A result that depends on the null chosen | When the first null might be too easy | N4.9 and N4.10 in T06 |
| **E. Comparators (context, not pass marks)** |||||
| E1 | Buy-and-hold | "Did it help at all?" | Every test | Hold the parent |
| E2 | Constant mix at the same average exposure | Drawdown benefit that only comes from holding less stock | Every timing test | About 49% in the parent at all times |
| **F. Controls (checking the method itself)** |||||
| F1 | Oracle: a signal that knows the future must reach the top of the null | A pipeline that can't see any edge | Once per method | Not run in AA (listed as untested) |
| F2 | Negative control: ≥ 100 useless signals should pass about 5% of the time | A null that is too easy | Once per method | Not run in AA |
| F3 | Planted edge at the MDE should pass about 80% of the time | A wrong MDE | Once per method | Not run in AA |
| **G. Robustness** |||||
| G1 | Other costs (0 / 20.8 / 50 bps) | A cost-sensitive edge | Every test | R1 |
| G2 | Other data choices (another cash series…) | A data-choice-sensitive edge | Every test | R2 (LIQUIDCASE) |
| G3 | Sub-periods (live window, leave one period out, up and down markets) | An edge from one episode | Every test | R3 (live window only) |
| **H. Multiplicity** |||||
| H1 | Count every setting and idea tried; judge the best one against the best of chance | Picking noise | Whenever more than one setting is tried | Bonferroni 98th percentile in T04 and T05 |
| **I. Power** |||||
| I1 | MDE written beside every FAIL | Saying "no effect" when the test was too weak to see one | Every FAIL | T06: MDE 15–19 pp a year |
| **J. Presentation** |||||
| J1 | Page check: every number on the page recomputed | Typos and stale numbers | Before publishing | `check_page.py`: 766 checks |

### The null, in detail (the heart of it)

**What a null is.** It answers one question: *what would I have seen if my idea were worthless?*
Run the **same experiment** (same stocks, dates, rules and costs) with **one ingredient replaced
by randomness**, many times over. Each run is a "draw". Then see where the real result falls among
the draws.

**The one-ingredient rule.** Randomise exactly what the hypothesis says carries the information, and
hold everything else fixed.

| If the claim is… | Randomise… |
|---|---|
| "Ranking by X picks better stocks" | the ranking, on each date |
| "Being in only when the rule says so beats chance" | *when* you are in (the in and out spells) |
| "Excluding stocks with Z helps" | which stocks get excluded (same number) |
| "Stocks that beat estimates move" | which announcements count as beats (same day) |

**Designing a null in 7 steps:**
1. Name the ingredient.
2. List every other dimension: pool, number held, calendar, weights, costs, turnover, style.
3. Decide how each is held: constant, or matched on each date (the default).
4. Choose how to randomise.
5. Predict what the randomisation breaks (for example, shuffling destroys persistence, so the null trades more).
6. Choose the statistic, the pass mark and the number of draws.
7. Write the null table (Appendix B), and the measurements that prove it held.

**How to judge the result:**
- **Percentile** = (draws worse + 0.5 × draws tied) ÷ draws × 100.
- **p-value** = (1 + draws at least as good) ÷ (draws + 1). It is never 0.
- **Edge** = the strategy's result − the null's median.
- **Number of draws:** at least 200 for a headline and 1,000 for claims beyond the 99th percentile.
  With m tests at level α, you also need draws + 1 ≥ m ÷ α.
- **Before and after cost:** if the null trades more than the strategy, the after-cost edge is
  partly a turnover advantage. The before-cost edge measures selection.

*T06 example:* the rule's 28.6% CAGR beat 1,948 of 2,000 random timings, so it sits at the 97.4th
percentile (p = 0.026). The null's median was 14.7%, so the edge is +14.0 pp.

**MDE (the smallest edge you could detect).** For a percentile test it is about 2.5 × the null's
standard deviation at the 95th percentile. T06 used 2.8 × the null's SD (80% power, 5% two-sided),
which gave 15–19 pp a year. If your edge is below the MDE, a FAIL means "too small to see", not "nothing there".

**When the null misleads you (AGENT.md N7):**

| Symptom | Likely cause |
|---|---|
| Percentiles near 100 everywhere, but a weak t-statistic | The null is calmer than reality: no persistence or style tilt |
| Percentile exactly 50, or many ties | The null has no real choice; the pool is no bigger than the portfolio |
| It beats the null only after cost | A turnover advantage, not skill |
| The verdict flips between two nulls | Let the pre-named deciding null decide, and report both |

---

## Part 3: How much of this to do (my suggestion, not from AGENT.md)

AGENT.md expects everything for every test. In practice:

**Minimum, for a quick look at an idea:**
1. Data you've checked, with fingerprints (A1 light, A2).
2. The question, the null and the pass criteria written down **before** running (Steps 1, 2 and 5).
3. Point-in-time assertions in code (B1, B2).
4. The rule against at least 1,000 random draws through the same code (D1, D2).
5. Buy-and-hold and constant-mix comparators (E1, E2).
6. One independent closed-form check that also fails on a corrupted copy (C1 light, C2).
7. Before and after cost, with dates on every number.
8. Label the result **EXPLORATORY**, not a finding.

**Full, for anything that will change money or production:** every step and every check above.

This repo did exactly that. Studies 1–5 and the "final rule" were exploratory. Tests T01–T06 then
re-tested the surviving ideas with the full process, and several claims shrank or died along the way.

---

## Part 4: Checklists

**Before the first run**
- [ ] The question in one sentence, with what would make it false
- [ ] The kind of experiment and its unit named
- [ ] Every input approved, fingerprinted and dated
- [ ] Point-in-time checks in the code; universe as of each date; prices adjusted
- [ ] A null table written: randomised, held, not matched
- [ ] The statistic, pass mark, number of draws and deciding null fixed
- [ ] Validity and reportability gates written; criteria complete and possible to meet
- [ ] Robustness checks and multiplicity planned
- [ ] The plain-words walkthrough written
- [ ] Sections 1–3 locked with a hash
- [ ] The independent check passes on good output and fails on a corrupted copy

**Before marking a test COMPLETE**
- [ ] The worked example checked by hand
- [ ] Every gate passed, printed by a script
- [ ] The null check shown first
- [ ] Every percentile with its statistic, draws and p-value; before and after cost
- [ ] The MDE beside every FAIL
- [ ] Multiplicity reported
- [ ] Every claim is a finding with "falsified by"; killed ideas logged too
- [ ] "Does not establish" written as two separate lists; next steps listed
- [ ] Anything written after the run is in a dated block

**Before quoting any number**
- [ ] It comes from a test's evidence folder, not a scratch run
- [ ] Its window (first and last date), its null, percentile and draws, and before and after cost go with it
- [ ] Its finding ID and status go with it
- [ ] The biggest open caveat is said out loud

---

## Part 5: Glossary

| Term | Plain meaning |
|---|---|
| **Null / draw** | A random version of your strategy; one run of it is a draw |
| **Percentile** | The share of random draws your strategy beats |
| **p-value** | The chance of doing at least this well by luck |
| **Edge** | Your result minus the median random result |
| **MDE** | The smallest edge the test could reliably detect |
| **Pre-registration** | Writing the design and pass marks down before running |
| **Point in time** | Using only what was known on the decision date |
| **OHLC4** | (open + high + low + close) ÷ 4, the assumed fill price |
| **bps** | Basis points: 1 bp = 0.01%. 20.8 bps round trip = 10.4 bps when buying + 10.4 when selling |
| **CAGR** | Compound annual growth rate |
| **Max drawdown** | The worst fall from a previous peak |
| **Calmar** | CAGR ÷ |max drawdown| |
| **Ulcer index** | The average depth below the last high |
| **Round trip** | One exit plus one entry; the independent unit for timing rules |
| **Constant mix** | Holding the strategy's average exposure all the time; shows if a benefit is only lower exposure |
| **Regression gate** | New code must reproduce old results exactly |
| **Reportability gate** | A condition without which a number may not be quoted |
| **Multiplicity** | Trying many things makes one look good by chance; adjust for it |

---

## Appendix A: Write-up template (`tests/T<NN>/T<NN>.md`)

```markdown
# T<NN> — <the question, as a question>

**Status:** DRAFT | RUNNING | COMPLETE | SUPERSEDED
**§1–§3 sha256:** <filled in by check_prereg.py when frozen>
**Run date:**
**Kind of experiment and unit:**
**Scripts / check:** engine, verifier, analyse.py, report.py, check_page.py
**Evidence:** evidence/<run>/
**Data boundary:** APPROVED_DATA.md as of <date>; data review of <date>
**Window / universe / hold-out:** <first date> → <last date>; <universe>; <hold-out or why none>
**Cost / null / draws / seeds:**
**Extends / supersedes:**

## 1. Idea and references            (written BEFORE the run)
- The idea in one sentence:
- Where it came from:
- The decision this will change:
- What is already known (table: ref | what it says | relationship):
- What this test is NOT asking:

## 2. Hypothesis and pre-registered criteria   (BEFORE)
- What I already saw:
- Hypotheses H1, H2… (each falsified if …):
- Null table (Appendix B), and which null decides:
- Statistic, direction, why:
- Pass mark and number of draws:
- Validity gates V1…:
- Reportability gate:
- Pass criteria C1, C2… (exact thresholds and counts):
- Robustness checks R1…:
- Multiplicity:
- Descriptive only:
- Outcome map (pattern → conclusion):

## 3. Methodology                     (BEFORE)
- Data table (file | contributes | through-date | review verdict):
- Construction, step by step:
- How the test works, in plain words (series, one decision, one path, one draw, ranking, comparators):
- Point-in-time rule:
- Execution and cost:
- Effective sample size and MDE:
- Known contaminants (issue | handling):

## 4. Results                         (AFTER — generated by analyse.py, never typed)
4.0 Worked example · gates · null check
4.1 Headline · 4.2 Comparators · 4.3 Evidence and MDE · 4.4 Robustness · 4.5 Verdict table
### Added after the run (<date>)

## 5. Findings                        (AFTER — Appendix C format)
## 6. What this does not establish    (tested and inconclusive / not tested at all)
## 7. Next things to work on          (question | method | cost | what it changes)
## 8. Reproduction                    (every command, in order; what makes it deterministic)
```

## Appendix B: Null table template (goes in section 2)

| Dimension | Randomised / held / not matched | How (constant, per date…) | Measured in 4.0 as |
|---|---|---|---|
| *the ingredient under test* | randomised | *how; number of draws; seed formula* | — |
| pool / universe | held | per date | pool size, strategy vs null |
| number held (timing: weeks invested, number of switches) | held | per date | must be identical |
| calendar, rules, weights, cost | held | constant | — |
| turnover / persistence | not matched? | | the mismatch, shown |
| risk and style | not matched? | | volatility and drawdown, strategy vs null |

## Appendix C: Finding template (goes in section 5)

```markdown
### F-<P>-S<NN> — <the claim as a full sentence, with its headline number and dates>

**Status:** PROVISIONAL | CONFIRMED | REFUTED | FIXED | OPEN — <why not the next level up>
**Evidence:** evidence/<run>/<files>
**Claim:** the numbers, each with its denominator, window, null, percentile and draws
**What limits it:**
**Falsified by:** <a concrete, checkable result — written before the data exists>
**Cross-check:** `python tests/T<NN>/analyse.py`
```

## Appendix D: `run.sh` template

```bash
#!/usr/bin/env bash
# Regenerate every T<NN> artefact. Run from the project root: bash tests/T<NN>/run.sh
set -euo pipefail
python data/check_approved_data.py        # stop if the data isn't the approved data
python tests/T<NN>/check_prereg.py        # stop if the locked design changed
python analysis/t<nn>_engine.py --run headline
python analysis/verify_t<nn>.py           # independent check; non-zero exit = stop
python tests/T<NN>/analyse.py             # gates + tables → analysis.log
python tests/T<NN>/fill_writeup.py        # tables into section 4
python tests/T<NN>/report.py              # results page
python tests/T<NN>/check_page.py          # every number on the page re-checked
python tests/T<NN>/check_prereg.py        # design still unchanged
```

**Where to look in this repo for a working example of each piece:**
- [AA/tests/T06/T06.md](AA/tests/T06/T06.md): a complete write-up.
- [AA/tests/T06/TRACKER.md](AA/tests/T06/TRACKER.md): step tracking.
- [AA/tests/T06/run.sh](AA/tests/T06/run.sh): the run order.
- [AA/analysis/t06_engine.py](AA/analysis/t06_engine.py) and [AA/analysis/verify_t06.py](AA/analysis/verify_t06.py): an engine and its independent verifier.
- [AA/analysis/approved_data.py](AA/analysis/approved_data.py): the data boundary in code.
