# 8. Limits: what can and cannot be said

[Back to the README](../README.md) · Previous: [7. Results](7_results.md)

## 8.1 The caveats, in order of weight

1. **Many attempts.** The single tone Hd is attempt 12 and the split tone Hs attempt 14; each was designed after the
   results of the earlier attempts were known. A p-value from the 14th attempt (Hs: 0.0100) carries less weight than
   the same p-value from a first attempt, because the more one tries, the likelier a lucky result. Holm corrects within
   an attempt, not across attempts; every attempt is listed in [`results/ATTEMPTS.md`](../results/ATTEMPTS.md) so
   readers can judge this themselves. As a rough guide, a Bonferroni-style correction over all the primary tests run
   in the project would not leave the Hs result significant.
2. **The headline was switched after every result was known.** On 1 November 2026 this repository changed its main
   result from Hs (attempt 14) to Hd (attempt 12), at the author's choice, so that the main result is the single tone
   compared across all four models side by side. Hd's own tests were fixed before its run
   ([`docs/studies/3_sigtests_distill/`](studies/3_sigtests_distill/2026-10-21-tests-and-distill-design.md)); only the
   choice of headline came later.
3. **The "from 2010" period was added after seeing the full-period results** of an earlier study (amendment 3 of the
   CORAL study). It is the right period for the distilled reader (its tone is clean of Haiku's memory only from 2010),
   but it was not the project's first choice of test period.
4. **Few informative meetings.** From 2010 the Fed moved only 32 times. The tone's gain comes from a handful of
   meetings (chapter 6.7), mostly where S+M was confidently wrong.
5. **Haiku's memory.** The distilled tone is protected from the teacher's possible memory of outcomes only from 2010
   (chapter 3.7). Before 2010 the cross-fitted models learned from Haiku's labels of the same era, so the full-period
   number is less clean.
6. **Real-time data before 2011-2012.** Some economic series do not have true vintages before 2011-2012 (chapter 1.6).
   This affects every arm equally.
7. **Network p-values ignore seed and tuning randomness**, which is larger than the effect being measured (chapter
   7.4).
8. **One placebo per arm.** Each placebo is a single random draw and can be lucky (the TFT placebo on 2 May 2018, see
   the README section 3.2). Many placebos, fixed in advance, would give a cleaner comparison (8.4).
9. **The zero lower bound.** From December 2008 to December 2015 and from March 2020 to March 2022 the target sat at
   0-0.25%, where a cut is impossible; the models learn this only from `m_zero_bound` and the rate level. 64 of the 135
   meetings from 2010 fall in these spells.
10. **`s_rate_lag` equals `m_target_level`** at every meeting, so there are 23 distinct inputs, not 24 (chapter 2.4).
    This does not change any forecast but matters when reading coefficients.

## 8.2 What can be said

**The main result (Hd):**

> "From 2010 to 2026 (135 FOMC meetings), adding the Fed's tone, read by a ChronoBERT model distilled from Claude Haiku
> and frozen at the end of 2009, raised accuracy and lowered log loss in all four forecasting models (ordered logit,
> LSTM, GRU, TFT). The gain is not proven: no model both beat the no-news model at p < 0.05 and beat a placebo tone at
> p < 0.05."

**The secondary result (Hs):**

> "In an exploratory analysis (the 14th attempt), the Fed's tone split by document type, read by the same distilled
> ChronoBERT, improved ordered-logit forecasts of FOMC decisions from 2010 (one-sided Clark-West p = 0.01, 135
> meetings), and a placebo tone did not."

## 8.3 What cannot be said yet

- "The networks benefit from the tone's content." (No network passes both checks: the TFT with Hd beats S+M after
  Holm but not its placebo; the LSTM with Hd beats its placebo but not S+M.)
- "Six columns are better than two." (No model shows Hs beating Hd at p < 0.05 from 2010.)
- "Fed tone helps forecast FOMC decisions", without conditions. (The Hd tests do not show it, and the Hs result comes
  from the 14th attempt, in a period chosen later.)

Not significant is not the same as "does not help": the network results **do not show** a gain; they do not show that
there is none.

## 8.4 What would strengthen the result (not done here)

- **New meetings.** Every FOMC decision after 16 September 2026 is a genuinely out-of-sample test with the code frozen
  as it is in this repository.
- **Other central banks**, with the same pipeline and the same decision rule fixed in advance.
- **More network seeds**, so seed randomness can be measured instead of ignored.
- **Many placebos**, for example 20 shifts fixed in advance; if the real tone beats all 20, p = 1/21 = 0.048.
- **The ordered logit's real tone against its placebo** (not computed yet).
- **A Taylor-rule baseline** (attempt 13, designed in
  [`docs/studies/4_text_vs_economy/`](studies/4_text_vs_economy/2026-10-21-text-vs-economy-design.md), not built): does
  the tone add to basic economic data, without market prices?
