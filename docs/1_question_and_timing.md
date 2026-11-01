# 1. The question, the target, and what the model may know

[Back to the README](../README.md) · Next: [2. Economy and market inputs](2_economy_and_market_inputs.md)

This chapter fixes three things before any model appears: what is forecast, how many times, and what information is
allowed into each forecast. Everything later depends on these choices.

**The example meeting used throughout the docs:** the FOMC meeting announced on **16 March 2022**, when the Fed began
its post-Covid hiking cycle (target range up 0.25 percentage point, from 0-0.25% to 0.25-0.50%). The docs follow this
one meeting from raw data to the forecast probability.

## 1.1 The research question

Before each FOMC meeting we ask: **will the Fed cut, hold or hike?** Then we ask whether adding the **tone** of what
the Fed itself said or wrote (hawkish or dovish) makes that forecast better than using economic statistics and market
prices alone.

The main result of this repository uses the **single tone Hd** (attempt 12): one averaged tone per meeting and its
change, tested with all four models side by side. A secondary result, attempt 14, asks a sharper version: does the
tone help **when it is kept split by the kind of document it came from** (policy statements and press conferences,
minutes, speeches), instead of being averaged into one number?

## 1.2 The target

- **The rate.** The Fed's target rate. Until 16 December 2008 the Fed announced a single number (FRED series
  `DFEDTAR`). Since then it announces a range, and we use its **midpoint** (`DFEDTARL` and `DFEDTARU`), for example
  0.375% for 0.25-0.50%.
- **The answer for each meeting** is one of three classes, from the change in the target rate:

  | Class | Rule | Test meetings 2001-2026 | From 2010 |
  |---|---|---|---|
  | 0 = cut | falls by more than 0.02 percentage point | 34 | 11 |
  | 1 = hold | changes by at most ±0.02 | 140 | 103 |
  | 2 = hike | rises by more than 0.02 percentage point | 38 | 21 |
  | total | | 212 | 135 |

  The Fed moves in steps of at least 0.25 percentage point, so the 0.02 band means "any move counts"
  (`config.CLASS_EPS = 0.02`).

**Why a class and not the rate itself?** The Fed's decision is a choice among three directions, and what an analyst
wants is "the probability of a hike", not a decimal rate.

**Note the imbalance.** From 2010, 103 of the 135 meetings (76%) were holds. A model that always says "hold" is right
76% of the time. The hard part is the 32 meetings where the Fed moved. This is why accuracy alone is a weak measure
and the docs lead with log loss (chapter 6).

## 1.3 The unit of data

- **One row = one FOMC meeting**, scheduled (8 a year) or unscheduled (emergency).
- **260 meetings** have every input, from 1995-03-28 to 2026-09-16.
- **Test periods** (the meetings that are scored):
  - full period: 212 meetings from 2001-01-03;
  - from 2010: 135 meetings from 2010-01-27 (the **primary** period for both Hd and Hs);
  - the networks are scored on scheduled meetings only: 206 (full) and 134 (from 2010).
- Meetings before 2001 are only ever used for training.

## 1.4 Look-ahead: the main danger

If a model sees anything published **after** the decision it is forecasting, it looks better than it could ever be in
real life. This is called look-ahead. A common example: using an inflation number that was revised two years later,
which nobody knew on the day. The project therefore fixes the timing of every input strictly.

## 1.5 The cutoff

- **Cutoff = one day before the announcement.** An FOMC meeting lasts two days, and the decision is announced on the
  second day.
- Every input must be **dated before the cutoff**.

**Example: the meeting of 16 March 2022**

```
25 Jan 2022       26 Jan 2022                                 14 Mar      15 Mar       16 Mar
previous cutoff   previous decision announced                 last market  cutoff      decision
   |------------- news window: Fed documents released in this span -------|
```

What the model knows at the cutoff of 15 March 2022:

| Input | Latest month the model sees | Why |
|---|---|---|
| CPI and core CPI | February 2022 | released 10 March, before the cutoff |
| payrolls, U-6, earnings | February 2022 | the jobs report comes out early in March |
| PCE and core PCE | January 2022 | February's PCE is not out yet |
| PPI | January 2022 | February's PPI came out on 15 March, which is not **before** the cutoff |
| M2 | January 2022 | |
| house prices | December 2021 | published about 2 months late |
| GDP | 2021 Q4 | |
| market prices (yields, VIX, oil) | 14 March 2022 | the last value before the cutoff |
| Fed text | documents released 25 January to 14 March 2022 | the news window (chapter 3) |

## 1.6 Real-time ("vintage") economic data

Economic statistics are revised often. The project reads them from **ALFRED**, the St. Louis Fed archive that keeps
**every vintage** (every published version) of a series, and uses the vintage that existed before the cutoff.

**Known gap.** Some series start their vintage record late: average earnings (2011), U-6 (2011), house prices (2012)
and PCE (2000). Before those years the first available vintage is used (which is already revised), and each month's
value is treated as published a fixed, conservative delay after the month ends (for example payrolls 9 days, CPI 14
days, house prices 62 days). So there can be a little leakage in test meetings before 2011-2012, including the first
two years of the from-2010 period. It affects every arm equally (with and without tone), so it cannot by itself create
a gain for the tone.

Code: `src/fed_news/download/alfred_vintages.py` (vintages), `src/fed_news/features/panels.py` (`macro_at`,
`market_at`, `build_t2`).
