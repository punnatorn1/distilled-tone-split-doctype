# Code map: from each step of attempt 14 to the code that does it

[Back to the README](../README.md)

Every file in `src/fed_news/` starts with a header saying what it does, what it reads, what it writes, and the command
that runs it. This page puts those files in the order the data flows, so you can go from a sentence in the docs to the
function that does it in one step.

## How the package is organised

```
src/fed_news/
  config.py              every setting and path in one place (data folder, test start, penalty grid, alpha, ...)
  http.py                the only function that opens a network connection
  download/              step 1: everything fetched from the internet (FRED, ALFRED, Fed website)
  features/              step 2: the meeting panel (S, M) and the generic tone-to-columns helpers
  models/                the model code shared by every study: ordered logit, walk-forward, LSTM/GRU, TFT
  evaluation/            the scores and the registered tests (Clark-West, DM-HLN, Newey-West, McNemar, Holm)
  studies/
    coral/               the CORAL output layer, the Haiku teacher, and the CORAL network machinery (tuning, refits, seeds)
    distill/             attempt 12: the distilled reader (Hd) and its logit and network runs
    doctype/             attempt 14: the split tone (Hs) and its logit and network runs   <- start here
    sigtests/            the extra tests used for the small-sample checks (fixed-b, fixed-m, RPS)
    networks/, simple_model/   earlier studies (attempts 6-8); attempt 14 imports shared constants and helpers from them
```

**Why there is code from other attempts here.** Attempt 14 was built on top of attempts 6 to 12 and reuses their code
(for example the network tuning loop of the CORAL study and the fold layout of Study 4). The package is kept exactly
as it was at commit `c35052f` of the original repository, so that every saved result can be reproduced byte for
byte. The tables below list only what attempt 14 uses. Four modules are not reached from attempt 14 at all:
`download/__main__.py` (the download entry point, still useful), `studies/networks/check_no_lookahead.py`,
`studies/networks/pooled_news_test.py` and `studies/sigtests/run_battery.py`.

## Step 1. Download the raw data (network; free FRED API key)

| What | File | Key functions | Writes (in the data folder) | Docs |
|---|---|---|---|---|
| run every download in order | `download/__main__.py` | `main` | everything below | [DATA](DATA.md) |
| FOMC calendar and decisions | `download/fomc_calendar.py`, `download/fomc_meetings.py` | | `calendar.pkl`, `meetings.pkl`, `fedfunds.pkl` | [1.2](1_question_and_timing.md#12-the-target) |
| real-time economic statistics | `download/alfred_vintages.py` | | `alfred/<series>.pkl` | [1.6](1_question_and_timing.md#16-real-time-vintage-economic-data) |
| daily market prices | `download/fred_market.py` | | `market_daily.pkl` | [2.4](2_economy_and_market_inputs.md#24-the-8-m-variables-the-bond-markets-bet-on-the-fed) |
| Fed documents, split into sentences | `download/fed_documents.py` | `build_corpus`, `split_sentences` | `corpus.pkl` | [3.3](3_from_text_to_tone.md#33-the-text) |

## Step 2. The meeting panel: S and M (CPU)

| What | File | Key functions | Writes | Docs |
|---|---|---|---|---|
| the 16 S and 8 M columns, cutoff, classes | `features/panels.py` | `S_COLS`, `M_COLS`, `macro_at`, `market_at`, `build_t2` | `t2.pkl` (one row per meeting) | [2](2_economy_and_market_inputs.md) |

## Step 3. The reader: sentence tone (GPU; the teacher step is paid and already done)

| What | File | Key functions | Writes | Docs |
|---|---|---|---|---|
| CORAL output layer, loss, levels → classes → tone | `studies/coral/core.py` | `coral_loss`, `level_probs`, `class_probs`, `tone_from_levels` | | [3.4](3_from_text_to_tone.md#34-the-coral-head-how-one-sentence-becomes-three-probabilities) |
| ChronoBERT + CORAL head network | `studies/coral/chronobert_coral.py` | `StanceNetCoral` | | [3.5](3_from_text_to_tone.md#35-the-tensor-shapes-inside-the-reader) |
| the teacher: Claude Haiku probabilities (paid API) | `studies/coral/haiku_ordinal.py` | `request_params`, `parse_probs`, `submit`, `collect` | `coral/haiku_sentence_probs.pkl` (in the data release) | [3.6](3_from_text_to_tone.md#36-the-teacher-claude-haiku-55) |
| the student: distil into ChronoBERT 2009, cross-fit, score | `studies/distill/chronobert_distill.py` | `soft_levels`, `fine_tune_soft`, `tune`, `final`, `tone`, `fidelity` | `distill/scores_*.pkl`, `distill/doc_stance_distill.pkl`; `results/distill/tuning.csv`, `settings.json`, `fidelity.json` | [3.7](3_from_text_to_tone.md#37-the-distilled-reader-hd-reader-teacher-and-student) |

## Step 4. From document tone to meeting columns (CPU)

| What | File | Key functions | Docs |
|---|---|---|---|
| document group of each document type | `features/news_columns.py` | `GROUP` | [3.3](3_from_text_to_tone.md#33-the-text) |
| single tone Hd: `n_stance`, `n_stance_chg` | `features/news_columns.py`, `features/swap_tone.py` | `fed_stance`, `build_text_features`, `naive_panel` | [3.8](3_from_text_to_tone.md#38-from-sentences-to-a-meetings-numbers-the-single-tone-hd-2-columns) |
| **split tone Hs: 6 columns** | **`studies/doctype/group_tone.py`** | **`GROUPS`, `HS`, `group_tones`, `split_panel`** | [3.9](3_from_text_to_tone.md#39-hs-six-tone-columns-instead-of-two-attempt-14) |

## Step 5. The ordered logit, walk-forward (CPU, minutes)

| What | File | Key functions | Writes | Docs |
|---|---|---|---|---|
| the model and its penalty choice | `models/linear.py` | `OrderedLogit`, `expanding_folds`, `choose_penalty`, `move_sizes` | | [5.1](5_models.md#51-the-ordered-logit-step-by-step) |
| refit at every meeting, z-score on training rows, placebo and shuffled controls | `models/walk_forward.py` | `Spec`, `common_rows`, `train_mask`, `placebo_matrix`, `run_spec` | | [4.2](4_what_each_forecast_sees.md#42-b-the-training-data-walk-forward), [6.8](6_tests_and_controls.md#68-placebo-controls-why-they-matter-and-how-they-are-built) |
| the single-tone arm S+M+Hd (attempt 12) | `studies/distill/run_logit.py` | `specs`, `panels`, `run` | `results/distill/logit/` | [7](7_results.md) |
| **attempt 14: S+M, S+M+Hs, placebo-Hs, shuffled-Hs; verdict** | **`studies/doctype/run_logit.py`** | **`specs`, `panels`, `evaluate`, `verdict`, `run`** | **`results/doctype/logit/`** | [7](7_results.md) |

## Step 6. The networks: LSTM, GRU, TFT (GPU for tuning and training; CPU is enough to re-evaluate)

| What | File | Key functions | Writes | Docs |
|---|---|---|---|---|
| input windows `[L, columns]`, z-score | `models/neural.py` | `make_sequences`, `standardise`, `RNNRegressor` | | [4.1](4_what_each_forecast_sees.md#41-a-the-input-at-forecast-time) |
| TFT building blocks | `models/tft.py` | `GLU`, `GRN`, `VariableSelection`, `InterpretableMultiHeadAttention`, `CompactTFT` | | [5.5](5_models.md#55-tft-a-compact-temporal-fusion-transformer) |
| networks with a CORAL head, training loops | `studies/coral/networks.py` | `build_coral_model`, `fit_validate_coral`, `fit_fixed_coral`, `predict_levels` | | [5.3-5.6](5_models.md#53-lstm-memory-with-gates) |
| tuning, refits every 8 meetings, 8 seeds, settings guard | `studies/coral/run_networks.py` | `tune_point_coral`, `cv_score_coral`, `run_tuned_coral`, `seed_forecast`, `ensemble`, `panel_fingerprint`, `pair_stats_coral` | | [5.6](5_models.md#56-training-the-networks) |
| the S+M arm (shared baseline) | `studies/coral/run_networks.py` | `tune_all`, `forecast_all` | `results/coral/networks/*_S+M*` | |
| the single-tone arm S+M+Hd (attempt 12b) | `studies/distill/run_networks.py` | `tune`, `forecast` | `results/distill/networks/` | [7](7_results.md) |
| **attempt 14b: S+M+Hs and placebo-Hs; comparisons; Holm; placebo reading** | **`studies/doctype/run_networks.py`** | **`placebo`, `panels`, `tune`, `forecast`, `primary_holm`, `reading`, `small_sample`** | **`results/doctype/networks/`** | [7](7_results.md) |

## Step 7. Scores and tests (CPU)

| What | File | Key functions | Docs |
|---|---|---|---|
| log loss, accuracy, RMSE; the pass/fail rule | `evaluation/metrics.py` | `metrics`, `decide` | [6.1](6_tests_and_controls.md#61-log-loss-the-main-score) |
| Clark-West | `evaluation/clark_west.py` | `clark_west`, `compare` | [6.4](6_tests_and_controls.md#64-clark-west-the-primary-test-of-the-ordered-logit) |
| DM-HLN, McNemar, Holm, Newey-West | `evaluation/significance.py` | `dm_hln`, `mcnemar_greater`, `holm`, `nw_summary` | [6.3](6_tests_and_controls.md#63-newey-west-neighbouring-meetings-are-not-independent), [6.5](6_tests_and_controls.md#65-other-comparisons-reported), [6.6](6_tests_and_controls.md#66-holm-testing-several-things-at-once) |
| fixed-b, fixed-m, ranked probability score | `studies/sigtests/tests.py` | `fixed_b`, `fixed_m`, `rps`, `newey_west` | [6.5](6_tests_and_controls.md#65-other-comparisons-reported) |
| the void rule | `studies/coral/run_logit.py` | `void`, `compare_row` | [6.9](6_tests_and_controls.md#69-the-decision-rules-fixed-before-the-run) |

## Commands

Run from the repository root with the project environment (see the README's quick start). The data folder is found
automatically (`config.DATA`: the `FED_NEWS_DATA` environment variable, else `data/`).

| Step | Command | Needs | Time |
|---|---|---|---|
| get the data | `python scripts/download_data.py` | internet | minutes |
| 5, logit (attempt 12 arm) | `python -m fed_news.studies.distill.run_logit` | data | ~5 min |
| 5, logit (attempt 14) | `python -m fed_news.studies.doctype.run_logit` | data, the attempt 12 forecasts in `results/distill/logit/` | ~5 min |
| 6, networks: re-evaluate from the saved seed forecasts | `python -m fed_news.studies.doctype.run_networks forecast` | data | ~1 min |
| 6, networks: tune from scratch | `python -m fed_news.studies.doctype.run_networks tune` | CUDA GPU | hours |
| 3, reader | `python -m fed_news.studies.distill.chronobert_distill tune` / `final` / `tone` | CUDA GPU | 3-4 h on an RTX 2060 |
| 3, teacher (never needed; output is in the data release) | `python -m fed_news.studies.coral.haiku_ordinal project` / `submit` / `collect` | paid Anthropic API key | hours, costs money |
| 1, raw downloads | `python -m fed_news.download` | free FRED API key | hours |
| unit tests | `python -m pytest` | | ~2 min |
| regenerate the results tables in the docs | `python scripts/make_results_tables.py` | | seconds |

`forecast` reuses a seed's saved forecast when the settings and panel fingerprints match, so on the released data it
re-evaluates without training. If anything differs, it trains that seed again (and needs a GPU to finish in reasonable
time).

## Tests

`tests/shared/` covers the shared code; `tests/studies/<study>/` covers each study. Attempt 14's own tests are in
`tests/studies/doctype/` (`test_group_tone.py`, `test_run_logit.py`, `test_run_networks.py`).

## Older documents kept for reference

`docs/studies/` (the design documents, each committed before its results) and `docs/guides/` (the original project's
code guide, data list, deviations and tensor-shape notes) are copied unchanged from the original repository, because
the code headers point to them. They describe the whole original project; the results of studies other than attempts
12 and 14 live in the original repository,
[taminggg1-cmyk/fed-news-forecasting-final](https://github.com/taminggg1-cmyk/fed-news-forecasting-final) (branch
`coral`).
