# 9. How the data are split: training, validation and test

This chapter explains every split in the project: which data each model learns from, which data choose its settings,
and which data judge it. All dates and counts below were printed from the real data and code, not estimated.

## 9.1 The simple version

Most machine-learning projects make one fixed split, for example 60% training, 20% validation and 20% test. This
project does not. It uses **three splits stacked inside each other**:

| Layer | Question it answers | How |
|---|---|---|
| 1. **Test** (outer) | How good are the forecasts? | walk-forward: each meeting is forecast by a model that only saw earlier meetings |
| 2. **Validation** (inner) | Which settings should the model use? | time-ordered folds cut from the training meetings only |
| 3. **Tone reader** (separate) | How is the Fed's tone measured without hindsight? | the reader learns from sentences before 2010 and is tested on sentences from 2010 on |

The three words mean:

- **Training data:** the examples a model learns its weights from.
- **Validation data:** examples held back from training to compare settings (penalty strength, network size, look-back
  and so on). The model never learns its weights from them.
- **Test data:** examples used only to judge the finished forecasts. Every reported score and p-value comes from them.

*Analogy: training is doing homework, validation is a practice exam used to choose how to study, and the test is the
real exam.*

## 9.2 Layer 1: the test is walk-forward

There are 260 Fed meetings, from 28 March 1995 to 16 September 2026, one row each.

**Walk-forward** means: to forecast meeting *i*, train only on meetings 1 to *i* − 1; forecast meeting *i*; then move
to meeting *i* + 1 and train again with one more meeting. It imitates a real forecaster, who on each meeting day knows
only the past.

```
forecast 2001-01-03:  train on 1995-03-28 ... 2000-12-19   (48 meetings)   first forecast of the full period
forecast 2010-01-27:  train on 1995-03-28 ... 2009-12-16   (125 meetings)  first forecast of the main test
forecast 2022-03-16:  train on 1995-03-28 ... 2022-01-26   (223 meetings)
forecast 2026-09-16:  train on 1995-03-28 ... 2026-07-29   (259 meetings)  last forecast
```

- **The test set** is every forecast meeting. The main result uses the 135 meetings from 27 January 2010 to 16
  September 2026 (11 cuts, 103 holds, 21 hikes). The networks are scored on the 134 scheduled ones; the emergency cut of
  3 March 2020 is left out. Forecasts for 2001-2009 are also made but only reported, because the tone for those years
  can contain hindsight (section 9.5).
- **The training set grows** by one meeting each time: from 125 meetings (January 2010) to 259 (September 2026).
- **Refitting:** the ordered logit is refitted before every meeting. The networks are refitted every 8 meetings
  (about once a year) and at every tuning point, because they are slow to train. Between refits the same network
  forecasts the next meetings; it still knows only the past.
- **Scaling:** each input becomes a z-score, (value − mean) ÷ standard deviation, where the mean and standard
  deviation come from the training meetings only. The test meeting's own value never enters them.

Code: `train_mask` and `run_spec` in `src/fed_news/models/walk_forward.py`; `run_tuned_coral` in
`src/fed_news/studies/coral/run_networks.py`.

## 9.3 Layer 2: validation, inside the training meetings only

Settings are chosen with **expanding time-ordered folds**. The training meetings are cut into equal blocks in date
order. Each fold trains on all blocks before its validation block and is scored on that block. Validation is therefore
always later than training, as it would be in real use. Random folds are never used, because they would let a model
learn from 2008 and be checked on 2005.

### The ordered logit: 5 folds, refitted every meeting

The only setting is the **penalty**: how strongly the weights are pulled towards zero to avoid overfitting. Six values
are tried: 0.01, 0.03, 0.1, 0.3, 1.0 and 3.0. The training meetings are cut into 6 blocks; folds 1 to 5 validate
blocks 2 to 6. The first block is only ever used for training.

**Worked example: the forecast for 27 January 2010** (125 training meetings)

| Fold | Train on | Validate on |
|---|---|---|
| 1 | 1995-03-28 to 1997-08-19 (20 meetings) | 1997-09-30 to 2000-02-02 (21) |
| 2 | 1995-03-28 to 2000-02-02 (41) | 2000-03-21 to 2002-05-07 (21) |
| 3 | 1995-03-28 to 2002-05-07 (62) | 2002-06-26 to 2004-12-14 (21) |
| 4 | 1995-03-28 to 2004-12-14 (83) | 2005-02-02 to 2007-08-07 (21) |
| 5 | 1995-03-28 to 2007-08-07 (104) | 2007-09-18 to 2009-12-16 (21) |

For each penalty, the logit is fitted on each fold's training meetings and scored by **log loss** on its validation
meetings. Log loss is −log(probability given to what really happened), averaged over meetings; lower is better. The
penalty with the lowest average log loss over the 5 folds wins. The logit is then fitted once more, on all 125
meetings, with that penalty, and forecasts 27 January 2010. The whole process repeats for the next meeting.

Code: `expanding_folds` and `choose_penalty` in `src/fed_news/models/linear.py`; grid `ORDLOGIT_GRID` and
`CV_FOLDS = 5` in `src/fed_news/config.py`.

### The networks (LSTM, GRU, TFT): 4 folds, tuned at 7 points

Networks have many settings: look-back (2, 4, 8 or 12 meetings), memory size, dropout, learning rate, weight decay,
batch size, and layers or attention heads. Searching them takes hours, so it happens only at 7 **tuning points**: the
first meeting of 2001, 2005, 2009, 2013, 2017, 2021 and 2025. The settings chosen at a point are used for every forecast
until the next point.

At each point the Optuna library tries 20 settings. Each try is scored on 4 expanding folds of the meetings before the
point. The folds use the same validation meetings for every look-back, so a look-back of 2 and a look-back of 12
compete on exactly the same meetings.

**Worked example: the tuning point 28 January 2009** (117 earlier meetings; its settings forecast 2009 to 2012,
including the start of the main test). With a look-back of 4:

| Fold | Train on (meetings that end a 4-meeting window) | Validate on |
|---|---|---|
| 1 | 1995-08-22 to 1998-12-22 (29) | 1999-02-02 to 2001-05-15 (21) |
| 2 | 1995-08-22 to 2001-05-15 (50) | 2001-06-27 to 2003-10-28 (21) |
| 3 | 1995-08-22 to 2003-10-28 (71) | 2003-12-09 to 2006-06-29 (21) |
| 4 | 1995-08-22 to 2006-06-29 (92) | 2006-08-08 to 2008-12-16 (22) |

Training starts at 1995-08-22, not 1995-03-28, because the first meeting with 3 meetings before it is the fourth one.

Inside each fold, training uses **early stopping**: after each pass over the data (an epoch), the validation log loss
is checked, and training stops when it has not improved for 20 epochs (at most 300). The try's score is the average
of the 4 folds' best validation log loss. The try with the lowest score wins.

**After tuning, there is no validation set.** Each refit trains on all meetings before the forecast for a fixed
number of epochs: the median of the 4 folds' best epochs. Each network is trained 8 times from different random
starting weights (seeds 0 to 7), and the 8 sets of probabilities are averaged into one forecast.

Code: `folds_for` and `suggest` in `src/fed_news/studies/networks/run_networks.py`; `cv_score_coral`,
`tune_point_coral` and `run_tuned_coral` in `src/fed_news/studies/coral/run_networks.py`. Every chosen setting is in
[HYPERPARAMETERS.md](HYPERPARAMETERS.md).

## 9.4 Layer 3: the tone reader's own split

The tone numbers come from a distilled reader: ChronoBERT (a language model pre-trained only on text up to 2009)
taught to copy Claude Haiku's sentence labels (chapter 3). It has its own split, by year:

| Role | Data |
|---|---|
| Training while choosing settings | a random half of the labelled sentences from 1994-2006 (55,600 sentences) |
| Validation for the settings | 6,000 random labelled sentences from 2007-2009 |
| Final training | all 144,211 labelled sentences dated before 2010; the reader is then frozen |
| Test (scored by the frozen reader) | all 199,495 sentences from 2010 on; no Haiku label from 2010 on is used by any forecast |

Six settings were tried; the winner was learning rate 0.00002, 2 epochs, 16 sentences per step, at most 256 word
pieces per sentence.

**Training-year tone is cross-fitted.** Sentences from 1994-2001 are scored by a copy trained on 2002-2009, and
sentences from 2002-2009 by a copy trained on 1994-2001. No sentence is scored by a model that learned from it. The
forecasting models need tone for their training meetings, and this gives it without letting a reader grade its own
homework.

Code: `tuning_split` and the constants `TUNE_LAST`, `VALID_YEARS`, `N_VALID` in
`src/fed_news/studies/distill/chronobert_distill.py`.

## 9.5 Why the main test starts in 2010

Haiku was trained on text up to 2026 and may remember what the Fed did after a speech. Its labels for sentences
before 2010 therefore may contain hindsight, and so does the tone of 1995-2009. From 2010 on, the frozen reader scores
sentences alone, so whatever Haiku remembers can only reach the training labels, all dated before the test. That is
why the headline result counts meetings from 2010 only.

## 9.6 What the split cannot do: the test period has been seen before

Inside each attempt the split is clean: no model ever trains on, tunes on, or scales with the meeting it forecasts.
But the same 2010-2026 meetings have been the test for all 14 attempts in [ATTEMPTS.md](../results/ATTEMPTS.md).
After each result we changed something and tested again on the same meetings.

*Analogy: a student sits the same exam 14 times and changes how they study after each sitting. Every sitting is
honest, but a good score on the 14th try may be partly luck from trying many times.*

What this means:

- **It is not cheating.** No forecast used a future meeting.
- **It weakens any single small p-value.** Attempt 14 (Clark-West p = 0.0100) is therefore labelled exploratory, not
  confirmed. Holm's correction and the fake-tone (placebo) checks reduce this problem but do not remove it.
- **The headline single tone (Hd) is not significant anyway**, so the repeated testing does not overturn a positive
  claim.

### Why not re-split into 60% / 20% / 20%?

We considered it. On these 260 meetings it gives:

| Part | Meetings | Dates | Cuts | Holds | Hikes |
|---|---|---|---|---|---|
| Training (60%) | 156 | 1995-03-28 to 2013-10-30 | 29 | 103 | 24 |
| Validation (20%) | 52 | 2013-12-18 to 2020-03-15 | 5 | 38 | 9 |
| Test (20%) | 52 | 2020-04-29 to 2026-09-16 | 6 | 34 | 12 |

It does not help, for three reasons:

1. **The test would not be untouched.** The 2020-2026 meetings are already inside the current test and have been seen
   in all 14 attempts.
2. **The test would shrink from 135 meetings to 52**, with only 18 rate moves. The current p-values are already
   borderline with 135 meetings, so even a real gain would very likely give p > 0.05, and one unusual meeting would
   matter even more.
3. **Choosing the best of the 8 seeds on validation adds noise, not skill.** A seed only changes the random starting
   weights. The seed that wins on 52 validation meetings got lucky on them, and that luck rarely carries over to the
   test. Averaging the 8 seeds, as now, gives a steadier forecast.

### How to get a truly untouched test

1. **Another central bank (recommended).** Freeze the method as it is, write the plan down (split, models, the one
   p-value test) and commit it, then run it once on the Bank of England or the European Central Bank. A 60/20/20 split
   inside that new data is fine: choose settings on validation, refit on training plus validation, forecast the test
   meetings walk-forward.
2. **Future Fed meetings.** The attempt 14 method was committed on 21-22 October 2026, before the meeting of 28
   October 2026, so that meeting and later ones are untouched. But a p-value needs years of meetings.

## 9.7 Independent review of the time-series setup

<!-- review:start -->
On 2026-11-10 a second Claude agent read the code alone (no data run) and looked for time-series problems. Every
claim below was then checked against the code. Nothing was rerun, so the size of each effect is **not measured**.

### What was checked and found clean

- The walk-forward loop: every forecast trains only on earlier meetings (`train_mask`: rows before *i*).
- Scaling, target statistics and average move sizes: computed from training meetings only.
- Validation folds: always after their training rows and before the tuning point.
- Test sequences: the look-back window ends at the forecast meeting and never reaches later meetings.
- Inputs: economic data use only versions published before the cutoff; market data are strictly before the cutoff;
  each tone window covers only documents released before the cutoff.
- The logit's fake tone (placebo) and shuffled-outcome control: shuffled within the training meetings only.

**In short: no future outcome reaches any test forecast.** The points below are about how strong the evidence is,
not about cheating.

### Points to know or fix

| # | Point | Where | Leaks a future outcome into the test? | Effect | Status |
|---|---|---|---|---|---|
| 1 | The **networks' fake tone** is the real tone shifted by half the panel (about 130 meetings), wrapping around. Training rows before about 2011 therefore get tone from about 2018-2026. It is one single shift, not many. | `placebo_panel` in `studies/networks/run_networks.py` | No: the shifted tone is from years away, so it says nothing about the meeting's outcome. | A single shift is one random draw: the "real versus fake tone" comparison for the networks rests on one draw. Direction unknown. | Open. Fix: 20 fixed shifts, decided in advance (proposed before, not run). |
| 2 | **Clark-West uses a plain standard error**, not a Newey-West one. | `clark_west` in `evaluation/clark_west.py` | No | For one-step-ahead forecasts this is the textbook form (Clark and West 2007). If the score differences are correlated over time, the p-value would be too small, which favours the tone. Attempt 14's p = 0.0100 comes from this test. | Open. Check: recompute with Newey-West (8 lags) as a robustness check. Not run yet. |
| 3 | **Tone for the training years 1994-2009 is cross-fitted**: 1994-2001 is scored by a copy trained on 2002-2009 labels, and the other way round. | `plan` in `studies/distill/chronobert_distill.py` | Not for the main test from 2010: everything used is before 2010. Yes for the forecasts of 2001-2009. | This is why the main test starts in 2010 (section 9.5). Training-year tone also comes from different reader copies than test-year tone. | Known and disclosed. |
| 4 | **Network training length.** Tuning finds the best number of epochs on folds with 1/5 to 4/5 of the training meetings. The final fit uses that same number on all training meetings. | `run_tuned_coral` in `studies/coral/run_networks.py` | No | More data with the same epochs means more training steps, so some overfitting is possible. It hurts the networks with and without tone alike. | Known; minor. |
| 5 | **The first tuning point (2001) has very little data**: 48 earlier meetings, so validation blocks of about 7 meetings, and a look-back of 12 is impossible there. | `folds_for` in `studies/networks/run_networks.py` | No | Settings for 2001-2004 are unreliable. Only the full-period results are affected, not the main test from 2010. | Known; minor. |
| 6 | **The split-versus-single comparison** (Hs against Hd) uses Clark-West. | `evaluate` in `studies/doctype/run_logit.py` | No | Clark-West needs one model to contain the other. The single tone is the average of the three group tones, so this holds only approximately. Treat that p-value (0.071) as rough. | Known; minor. |
| 7 | **Early economic data**: before 2011-2012 some series have no archived first version, so the earliest archived version is used. | `asof_series` in `download/alfred_vintages.py` | No | Affects the models with and without tone equally. | Known and disclosed. |
| 8 | **Look-back counts meetings, not days.** Emergency meetings and uneven gaps mean the same look-back covers different lengths of time. | `make_sequences` in `models/neural.py` | No | Design choice; direction unknown. | Known. |

The reviewer also noted that the older ChronoBERT tone readers (C and Co, attempts 9-11) score 1994-1999 documents
with a classifier trained on labels up to 1999. Those readers are not used by the single tone (Hd) or the split tone
(Hs) in this repository, so the headline results are not affected.

### What to do first

1. Recompute attempt 14's Clark-West p with a Newey-West standard error and report both (point 2).
2. Replace the single shifted fake tone for the networks with 20 shifts fixed in advance (point 1).
3. Before any "supported" wording, run both checks and the untouched test of section 9.6.
<!-- review:end -->
