# 6. How "better" is measured: scores, tests and placebo controls

[Back to the README](../README.md) · Previous: [5. The models](5_models.md) · Next: [7. Results](7_results.md)

## 6.1 Log loss: the main score

$$\text{log loss of meeting } j = -\ln\big(\text{probability the model gave to the outcome that happened}\big)$$

| Probability given to the true outcome | log loss |
|---|---|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 1/3 (a blind three-way guess) | 1.099 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

**Why log loss and not accuracy?** Accuracy only asks whether the most likely class was right. Log loss rewards being
confident and right, and punishes being confident and wrong hard, which is what a user of probabilities needs. The
reported number is the mean over meetings, in nats; lower is better.

**Other scores reported:**

- **accuracy** = share of meetings where the most likely class was the outcome;
- **RMSE** (root mean squared error) of the forecast rate (chapter 5.1), in percentage points;
- **ranked probability score** (RPS) = the squared distance between the forecast's cumulative probabilities and the
  outcome's, over the ordered classes; used only in the small-sample checks.

## 6.2 The meeting-by-meeting difference: the raw material of every test

$$d_j = \text{loss}_j(\text{S+M}) - \text{loss}_j(\text{arm with tone})$$

$d_j > 0$ means the tone helped at meeting $j$. Every test asks the same question: **is the mean of $d$ really above
0, or is it luck?** The idea is a t-test:

$$t = \frac{\text{mean of } d}{\text{standard error of the mean}},\qquad p = P(\text{a } t \text{ this large if the tone did not help})$$

Every test here is **one-sided** (it only asks "is the tone better?"). The tests differ in which loss they use and how
they compute the standard error.

## 6.3 Newey-West: neighbouring meetings are not independent

**The problem:** errors come in runs (2008, 2020), when a model is wrong several meetings in a row. Treating meetings as
independent understates the variance and gives p-values that look better than they are.

**The fix:** add the covariances of $d$ with its own values 1 to 8 meetings apart, with weights falling with distance:

$$\widehat{\text{Var}}(\bar d) = \frac{1}{n}\Big[\gamma_0 + 2\sum_{k=1}^{8}\Big(1 - \frac{k}{9}\Big)\gamma_k\Big]$$

$\gamma_k$ is the covariance of $d$ with $d$ shifted by $k$ meetings. Eight lags because the networks are refit every 8
meetings: meetings in one block share one model and so tend to err together. **This is the test of the networks**
(attempt 14b), on log loss.

Code: `src/fed_news/evaluation/significance.py` (`nw_summary`), `src/fed_news/studies/sigtests/tests.py`
(`newey_west`).

## 6.4 Clark-West: the primary test of the ordered logit

**The problem it solves:** the arm with tone **nests** S+M (set the tone coefficients to 0 and you get S+M). Even if
the tone truly helps, estimating extra coefficients from little data adds noise to the forecast, which makes the
bigger model look worse than it is. A plain comparison is therefore **biased towards finding nothing**.

**Clark and West's fix (2007):** add that noise term back.

$$f_j = \underbrace{(r_j - \hat r_j^{S})^2}_{\text{S+M squared error}} - \Big[\underbrace{(r_j - \hat r_j^{N})^2}_{\text{tone arm squared error}} - \underbrace{(\hat r_j^{S} - \hat r_j^{N})^2}_{\text{gap between the two forecasts}}\Big]$$

$$t = \frac{\bar f}{\text{sd}(f)/\sqrt{n}},\qquad p = 1 - \Phi(t)$$

It uses the forecast rates of chapter 5.1. $\Phi$ is the standard normal distribution function.

**Worked example: ordered logit with Hs, meeting of 8 November 2018** (rate 2.125%, outcome = hold)

| | S+M | S+M+Hs |
|---|---|---|
| P(hold) | 0.113 | 0.301 |
| P(hike) | 0.887 | 0.699 |
| log loss = −ln P(hold) | 2.180 | 1.200 |
| forecast rate | 2.354 | 2.305 |
| squared error | (2.125 − 2.354)² = 0.0523 | (2.125 − 2.305)² = 0.0324 |

- $d$ (log loss) = 2.180 − 1.200 = **+0.98**: the tone helped at this meeting;
- $f$ (Clark-West) = 0.0523 − [0.0324 − (2.354 − 2.305)²] = 0.0523 − [0.0324 − 0.0024] = **+0.0223**;
- doing this for all 135 meetings from 2010 and computing $t$ gives **p = 0.0100**.

Why S+M got this meeting wrong: the 6-month spread stood at +0.395 because the market expected a hike in December 2018
(which happened). The model cannot tell "a hike soon" from "a hike now". The Hs arm lowered P(hike).

**Limit:** Clark-West is valid only when the models really nest. Separately tuned networks do not nest; when
Clark-West was tried on networks with **placebo** tone it reported "significant" in 11 of 12 cases. So **Clark-West is
not used for the networks**.

Code: `src/fed_news/evaluation/clark_west.py` (`clark_west`, `compare`).

## 6.5 Other comparisons reported

- **DM-HLN** (Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction): the t-test of 6.2 on the log
  loss or squared error, without the nesting correction. Reported for the logit, without a decision rule.
- **McNemar**: counts the meetings the tone arm "fixes" (S+M wrong, tone right) and "breaks" (the reverse).
- **Small-sample checks for the networks** (reported, no rule): **fixed-b** (Kiefer-Vogelsang critical values, which
  stay honest when the lag window is large compared with the sample), **fixed-m** (a t-test on block means), and the
  RPS difference with Newey-West and fixed-b.

Code: `src/fed_news/evaluation/significance.py` (`dm_hln`, `mcnemar_greater`), `src/fed_news/studies/sigtests/tests.py`
(`fixed_b`, `fixed_m`, `rps`).

## 6.6 Holm: testing several things at once

If you run 3 tests at the 5% level and the tone never helps, the chance that **at least one** gets p < 0.05 by luck is
1 − 0.95³ = 14%. Holm's correction sorts the p-values from smallest up, multiplies the first by the number of tests $m$,
the next by $m - 1$, and so on (never letting an adjusted value fall below the previous one).

| Test | Holm over |
|---|---|
| ordered logit, S+M+Hs vs S+M from 2010 | 1 (a single primary test, no adjustment) |
| networks, S+M+Hs vs S+M from 2010 | 3 (LSTM, GRU, TFT) |

**Holm cannot fix trying many attempts.** It corrects within one attempt. Attempt 14 is the 14th attempt of this
project, which Holm does not account for; that is why every attempt is reported in
[`results/ATTEMPTS.md`](../results/ATTEMPTS.md).

Code: `src/fed_news/evaluation/significance.py` (`holm`).

## 6.7 The gain sits in a few meetings

From 2010, the ordered logit with Hs beats S+M on log loss at only 72 of 135 meetings (53%). Most of the gain comes from
a few:

| Meeting | Outcome | Probability of the outcome: S+M → S+M+Hs | d (log loss) |
|---|---|---|---|
| 2020-07-29 | hold | 0.008 → 0.035 | +1.48 |
| 2018-11-08 | hold | 0.113 → 0.301 | +0.98 |
| 2020-06-10 | hold | 0.128 → 0.308 | +0.88 |
| 2023-05-03 | hike | 0.533 → 0.358 | −0.40 |

The largest (29 July 2020) is the confident mistake of chapter 5.1: the tone made S+M less wrong, but it was still very
wrong.

## 6.8 Placebo controls: why they matter and how they are built

If adding tone columns improves the forecast, the reason may not be the **content** of the tone:

- the model has more columns to fit with;
- a network gets a different tuned setting (some settings are just lucky);
- training randomness.

A **placebo** is fake tone that **looks exactly like the real one** (same distribution of values) **but is not lined up
in time**. If the fake tone helps about as much as the real one, the gain is not from the tone's content.

| Control | How it is built | Example: meeting of 16 March 2022 |
|---|---|---|
| **placebo-Hs, ordered logit** | at every fit, the tone columns are shuffled across the training rows, and the forecast row gets the tone of one random training row | gets the tone of some random meeting before 2022 |
| **placebo-Hs, networks** | the tone columns are shifted circularly by 130 meetings (half of 260), so each meeting gets the tone of a meeting about 16 years away, while the meeting-to-meeting pattern is kept (which matters to a model that reads sequences) | gets the tone of a meeting about 16 years away |
| **shuffled-Hs** (ordered logit) | the outcomes (cut/hold/hike) are shuffled in training, so the model cannot learn anything real | it ends up forecasting "hold" every time: full-period accuracy 66.0% (the hold share) and log loss 0.88 |

Code: `src/fed_news/models/walk_forward.py` (`placebo_matrix`, `run_spec`), `src/fed_news/studies/doctype/run_networks.py`
(`placebo`).

## 6.9 The decision rules (fixed before the run)

**Ordered logit (the void rule).** The result is **void** if

- the placebo beats S+M at Clark-West p < 0.05, or
- the shuffled-label control beats S+M at Clark-West p < 0.01.

If fake tone, or a model trained on scrambled answers, still "passes", the test cannot be trusted on these data.

**Networks (the placebo reading).** A network's gain is counted as coming from the tone's content only if **both**
hold:

1. the real tone beats its own placebo at p < 0.05 (Newey-West on log loss);
2. the placebo does not beat S+M at p < 0.05.

The attempt 14 design that fixed these rules before the runs is in
[`docs/studies/5_doctype/2026-10-21-doctype-tone-design.md`](studies/5_doctype/2026-10-21-doctype-tone-design.md).

Code: `src/fed_news/studies/coral/run_logit.py` (`void`), `src/fed_news/studies/doctype/run_networks.py` (`reading`,
`primary_holm`).
