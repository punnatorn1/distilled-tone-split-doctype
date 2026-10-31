# Glossary

[Back to the README](../README.md)

| Term | Meaning | Where explained |
|---|---|---|
| **FOMC** | Federal Open Market Committee, the Fed body that sets the target rate, 8 scheduled meetings a year | [1](1_question_and_timing.md) |
| **cut / hold / hike** | the three classes: target rate down by more than 0.02 pp / within ±0.02 / up by more than 0.02 | [1.2](1_question_and_timing.md#12-the-target) |
| **pp** | percentage point | |
| **cutoff** | one day before the announcement; every input must be dated before it | [1.5](1_question_and_timing.md#15-the-cutoff) |
| **vintage, ALFRED** | a published version of an economic series; ALFRED is the St. Louis Fed archive of every vintage | [1.6](1_question_and_timing.md#16-real-time-vintage-economic-data) |
| **S** | the 16 economic statistics (columns `s_*`) | [2.3](2_economy_and_market_inputs.md#23-the-16-s-variables) |
| **M** | the 8 market numbers (columns `m_*`) | [2.4](2_economy_and_market_inputs.md#24-the-8-m-variables-the-bond-markets-bet-on-the-fed) |
| **S+M** | the model without tone, 24 inputs | [2](2_economy_and_market_inputs.md) |
| **z-score** | (value − training mean) ÷ training standard deviation | [2.5](2_economy_and_market_inputs.md#25-the-z-score-putting-every-input-on-the-same-scale) |
| **tone** | P(hawkish) − P(dovish) of a sentence, from −1 to +1; averaged up to documents and meetings | [3](3_from_text_to_tone.md) |
| **hawkish / dovish / neutral** | language pointing to higher rates / lower rates / neither | [3.2](3_from_text_to_tone.md#32-words-hawkish-dovish-neutral) |
| **policy, minutes, speech** | the three document groups (policy = statements and press conferences) | [3.3](3_from_text_to_tone.md#33-the-text) |
| **ChronoBERT** | a family of BERT-style language models, each pre-trained only on text up to the end of one year; the 2009 vintage is used | [3.7](3_from_text_to_tone.md#37-the-distilled-reader-hd-reader-teacher-and-student) |
| **CORAL** | an output layer for ordered classes: one score, two thresholds, two nested yes/no probabilities $P_1 \ge P_2$ | [3.4](3_from_text_to_tone.md#34-the-coral-head-how-one-sentence-becomes-three-probabilities) |
| **teacher, student, distillation** | the student model (ChronoBERT 2009) learns to copy the teacher's (Claude Haiku 5.5) probabilities on pre-2010 sentences | [3.7](3_from_text_to_tone.md#37-the-distilled-reader-hd-reader-teacher-and-student) |
| **cross-fitting** | no model scores a sentence it was trained on | [3.7](3_from_text_to_tone.md#37-the-distilled-reader-hd-reader-teacher-and-student) |
| **news window** | the span from the previous meeting's cutoff to just before this meeting's cutoff | [3.8](3_from_text_to_tone.md#38-from-sentences-to-a-meetings-numbers-the-single-tone-hd-2-columns) |
| **Hd** | the single distilled tone: 2 columns `n_stance`, `n_stance_chg` (attempt 12) | [3.8](3_from_text_to_tone.md#38-from-sentences-to-a-meetings-numbers-the-single-tone-hd-2-columns) |
| **Hs** | the split distilled tone: 6 columns, one level and one change per document group (attempt 14) | [3.9](3_from_text_to_tone.md#39-hs-six-tone-columns-instead-of-two-attempt-14) |
| **arm** | one set of inputs given to a model: S+M, S+M+Hd, S+M+Hs, placebo-Hs, shuffled-Hs | [7](7_results.md) |
| **lookback L** | how many meetings a network sees per forecast | [4.1](4_what_each_forecast_sees.md#41-a-the-input-at-forecast-time) |
| **walk-forward** | forecast meetings one at a time in date order, training only on earlier meetings | [4.2](4_what_each_forecast_sees.md#42-b-the-training-data-walk-forward) |
| **ordered logit** | one score $\eta$ and two cut points $t_1 < t_2$ give P(cut), P(hold), P(hike) | [5.1](5_models.md#51-the-ordered-logit-step-by-step) |
| **ridge (L2) penalty, λ** | pulls coefficients towards 0; λ sets its strength, chosen by cross-validation | [5.1](5_models.md#51-the-ordered-logit-step-by-step) |
| **LSTM, GRU** | recurrent networks with gates that read a sequence of meetings | [5.3](5_models.md#53-lstm-memory-with-gates), [5.4](5_models.md#54-gru-a-slimmer-lstm) |
| **TFT, GLU, GRN** | Temporal Fusion Transformer and its gated building blocks | [5.5](5_models.md#55-tft-a-compact-temporal-fusion-transformer) |
| **Optuna, TPE** | the tuning library and its sampler (Tree-structured Parzen Estimator) | [5.6](5_models.md#56-training-the-networks) |
| **seed** | the random starting point of a training run; each network forecast averages 8 seeds | [5.6](5_models.md#56-training-the-networks) |
| **log loss** | −ln(probability given to the outcome that happened), in nats; lower is better | [6.1](6_tests_and_controls.md#61-log-loss-the-main-score) |
| **RMSE** | root mean squared error of the forecast rate, in pp | [6.1](6_tests_and_controls.md#61-log-loss-the-main-score) |
| **RPS** | ranked probability score, squared distance of cumulative probabilities | [6.1](6_tests_and_controls.md#61-log-loss-the-main-score) |
| **Clark-West (CW)** | test for nested models that adds back the noise of estimating extra coefficients; the primary test | [6.4](6_tests_and_controls.md#64-clark-west-the-primary-test-of-the-ordered-logit) |
| **Newey-West (NW, NW8)** | standard error that allows correlation between neighbouring meetings (8 lags) | [6.3](6_tests_and_controls.md#63-newey-west-neighbouring-meetings-are-not-independent) |
| **DM-HLN** | Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction | [6.5](6_tests_and_controls.md#65-other-comparisons-reported) |
| **fixed-b, fixed-m** | small-sample versions of the Newey-West test | [6.5](6_tests_and_controls.md#65-other-comparisons-reported) |
| **McNemar** | counts meetings fixed and broken by adding the tone | [6.5](6_tests_and_controls.md#65-other-comparisons-reported) |
| **Holm** | correction of p-values for several tests run together | [6.6](6_tests_and_controls.md#66-holm-testing-several-things-at-once) |
| **placebo** | fake tone with the real tone's values but the wrong timing | [6.8](6_tests_and_controls.md#68-placebo-controls-why-they-matter-and-how-they-are-built) |
| **shuffled-label** | a control trained on scrambled outcomes | [6.8](6_tests_and_controls.md#68-placebo-controls-why-they-matter-and-how-they-are-built) |
| **void rule, placebo reading** | the decision rules fixed before the run | [6.9](6_tests_and_controls.md#69-the-decision-rules-fixed-before-the-run) |
| **attempt** | one pre-registered test of the project; attempt 14 is the 14th | [`results/ATTEMPTS.md`](../results/ATTEMPTS.md) |
| **zero lower bound** | the target at 0-0.25%, where a cut is impossible (`m_zero_bound` = 1) | [8.1](8_limits.md#81-the-caveats-in-order-of-weight) |
