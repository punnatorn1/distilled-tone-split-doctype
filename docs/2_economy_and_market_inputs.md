# 2. The economy and market inputs (S and M) and the z-score

[Back to the README](../README.md) · Previous: [1. Question and timing](1_question_and_timing.md) · Next:
[3. From Fed text to tone columns](3_from_text_to_tone.md)

Every forecast in this repository starts from the same 24 numbers per meeting: 16 economic statistics (block **S**)
and 8 market numbers (block **M**). The model without tone is called **S+M**. The tone columns of chapter 3 are added
on top of these.

## 2.1 Where S comes from

The S block is the variable set of Table 1 in the paper this project started from (Chan-Lau, Quach and Shi,
"Forecasting Federal Funds Rates with AI"), but read in real time (chapter 1.6) so nothing leaks.

**Why the Fed cares.** The Fed has two legal goals (the dual mandate): stable prices (inflation near 2% a year,
officially measured with PCE, with core PCE watched for the trend) and maximum employment. As a rule of thumb, **high
inflation or a hot economy pushes the Fed towards hikes**, and **low inflation or high unemployment pushes it towards
cuts**. S measures inflation, the labour market, growth, money and the yield curve.

## 2.2 Three transformations used again and again

| Suffix | Meaning | Formula | Example |
|---|---|---|---|
| `_yoy` | % change from **12 months earlier** | (this month ÷ 12 months earlier − 1) × 100 | CPI in Feb 2022 was 7.9% above Feb 2021 |
| `_mom` | change from **the previous month** | M2: in %; U-6: in percentage points | |
| `_chg` | difference from the previous month, in the original unit | this month − last month | payrolls +678 thousand |

**Why year-over-year and not the level?** An index level (CPI = 284) means nothing by itself and trends up for ever,
but "how many % higher than a year ago" *is* inflation, which is what the Fed watches. It also removes seasonality.

## 2.3 The 16 S variables

The last column is the raw value for the 16 March 2022 meeting (before the z-score).

| # | Column | What it is | FRED source | Unit | Why it matters for the Fed | 16 Mar 2022 |
|---|---|---|---|---|---|---|
| 1 | `s_rate_lag` | target rate **before** the meeting | DFEDTAR / range midpoint | % | the Fed moves in small steps from where it is, and cannot cut below 0 | 0.125 |
| 2 | `s_m2_mom` | M2 money stock, change from last month | M2SL | % | fast money growth can lead to inflation | 1.135 |
| 3 | `s_m2_yoy` | M2, change from 12 months earlier | M2SL | % | same, over one year | 12.648 |
| 4 | `s_pce_yoy` | PCE inflation (prices of what households buy) | PCEPI | % a year | **the Fed's 2% target is defined on this** | 6.061 |
| 5 | `s_pce_core_yoy` | core PCE = PCE without food and energy | PCEPILFE | % a year | strips volatile food and energy to show the underlying trend | 5.209 |
| 6 | `s_payems_chg` | change in non-farm payrolls in the latest month | PAYEMS | thousands | the most watched labour-market number | 678 |
| 7 | `s_earn_yoy` | average weekly earnings of production and non-supervisory workers, change from a year earlier | CES0500000030 | % a year | fast wage growth raises costs and prices | 6.650 |
| 8 | `s_gdp_yoy` | nominal GDP, change from 4 quarters earlier | GDP | % a year | how fast the economy grows | 11.784 |
| 9 | `s_u6_mom` | broad unemployment rate U-6, change from last month | U6RATE | pp | U-6 also counts involuntary part-timers and discouraged workers | 0.1 |
| 10 | `s_u6_yoy` | U-6, change from 12 months earlier | U6RATE | pp | same, over one year | −3.9 |
| 11 | `s_ppi_yoy` | producer prices, change from a year earlier | PPIACO | % a year | producer costs pass through to consumer prices | 19.266 |
| 12 | `s_cpi_yoy` | CPI inflation | CPIAUCSL | % a year | the best-known inflation number, out earlier than PCE | 7.912 |
| 13 | `s_cpi_core_yoy` | core CPI | CPILFESL | % a year | underlying inflation | 6.416 |
| 14 | `s_hpi_yoy` | FHFA purchase-only house price index, change from a year earlier | HPIPONM226S | % a year | house prices reflect credit conditions and feed into rents | 17.612 |
| 15 | `s_slope_10y2y` | 10-year minus 2-year Treasury yield | DGS10 − DGS2 | pp | an inverted curve (negative) often signals recession and expected cuts | 0.27 |
| 16 | `s_slope_1y6m` | 1-year minus 6-month yield | DGS1 − DGS6MO | pp | a steep short end means the market expects hikes within months | 0.42 |

(pp = percentage point.) **Reading March 2022:** every inflation measure was very high (PCE 6.1% against a 2% target,
core PCE 5.2%, CPI 7.9%, PPI 19.3%) and hiring was strong (+678,000 jobs). Everything pointed to a hike, and the Fed
hiked. Variables 15 and 16 are market prices but sit in S because the original paper put them there.

## 2.4 The 8 M variables: the bond market's bet on the Fed

If investors believe the Fed will hike within 3 months, anyone holding a 6-month bill today asks for a yield above the
current target rate, because the rate during the holding period will be higher. So

$$\text{spread}_n = \text{yield of an } n\text{-maturity Treasury} - \text{current target rate}$$

- positive spread: the market expects **hikes** within that maturity;
- spread near 0: the market expects **no change**;
- negative spread: the market expects **cuts**.

Several maturities show how **soon** and how **much**. These variables are strong because the market has already
priced in everything investors know, including what the Fed has said.

| # | Column | What it is | FRED source | Unit | What it tells | 16 Mar 2022 |
|---|---|---|---|---|---|---|
| 1 | `m_sp_tbill_3m` | 3-month bill yield − target | DTB3 | pp | expectations over about 1-2 meetings | 0.325 |
| 2 | `m_sp_y6m` | 6-month yield − target | DGS6MO | pp | about 3-4 meetings | 0.735 |
| 3 | `m_sp_y1` | 1-year yield − target | DGS1 | pp | one year | 1.155 |
| 4 | `m_sp_y2` | 2-year yield − target | DGS2 | pp | two years | 1.745 |
| 5 | `m_target_level` | target rate at the cutoff | DFEDTAR / range midpoint | % | is the rate high or low | 0.125 |
| 6 | `m_zero_bound` | 1 if the target ≤ 0.30%, else 0 | from #5 | 0 or 1 | at zero the Fed cannot cut, so it behaves differently | 1 |
| 7 | `m_vix` | VIX, the stock market's expected 30-day volatility | VIXCLS | points | market panic often leads to cuts (2008, March 2020) | 31.77 |
| 8 | `m_wti` | WTI crude oil price | DCOILWTICO | dollars a barrel | expensive oil pushes inflation up | 103.22 |

**Reading 16 March 2022:** the target was 0.125%, but the 6-month yield was 0.86% (spread +0.735) and the 2-year 1.87%
(spread +1.745). The market was clearly betting on several hikes.

### One duplicated pair

On the meeting panel `s_rate_lag` (S) and `m_target_level` (M) are **equal at every meeting** (largest difference 0):
both are the target rate before the decision. So the model really has 23 distinct inputs, not 24. This does not make
any result wrong (the ordered logit's ridge penalty splits the weight exactly in half between the two, chapter 5.1),
but keep it in mind when reading coefficients.

## 2.5 The z-score: putting every input on the same scale

**The problem.** Inputs have very different units: payrolls are in hundreds or thousands, spreads below 1. With raw
values a model learns badly (large-valued inputs dominate the gradient) and a penalty on coefficient size (chapter 5)
is unfair, because a coefficient's size depends on its input's unit.

**The formula**, applied to every input column $x$:

$$z = \frac{x - \text{mean of the training rows}}{\text{standard deviation of the training rows}}$$

$z$ says "how many typical deviations above or below normal". $z = 0$ is the average; $z = 2$ is far above normal.

**The rule that prevents look-ahead:** the mean and standard deviation come from the **training rows only**, and are
recomputed at every refit. Using all rows (including the future) would leak.

**Worked example: core CPI at the 16 March 2022 meeting**

- raw value = 6.416%;
- training rows = the 223 earlier meetings (1995 to January 2022): mean 2.164%, standard deviation 0.609;
- $z = (6.416 - 2.164) / 0.609 = 6.98$.

Underlying inflation that month was almost 7 typical deviations above its past average.

**Values never seen before.** If an input on the forecast day is far outside the range seen in training, a model
extrapolates and can be confidently wrong. Chapter 5.1 shows a real case (July 2020).

Code: `src/fed_news/features/panels.py` (`S_COLS`, `M_COLS`, `build_t2`), `src/fed_news/models/walk_forward.py`
(`_standardise`, used by the ordered logit), `src/fed_news/models/neural.py` (`standardise`, used by the networks).
