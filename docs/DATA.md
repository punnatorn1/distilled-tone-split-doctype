# Data: every file attempt 14 reads

[Back to the README](../README.md)

The data are not in git. `python scripts/download_data.py` downloads them from two GitHub releases into `data/`, checks
every file against a sha256 manifest, and `config.DATA` then finds `data/` automatically (or set the environment
variable `FED_NEWS_DATA` to another folder).

| Part | Release | Zip | Files | Manifest |
|---|---|---|---|---|
| 1. economy, market, text | [fed-news-forecasting-final, `data-2026-10-18`](https://github.com/taminggg1-cmyk/fed-news-forecasting-final/releases/tag/data-2026-10-18) | `final_study_data.zip`, 79 MB | 53 | `MANIFEST.txt` (copy: [`guides/DATA_MANIFEST.txt`](guides/DATA_MANIFEST.txt)) |
| 2. attempt 14 tone | [this repository, `data-2026-10-31`](https://github.com/taminggg1-cmyk/distilled-tone-split-doctype/releases/tag/data-2026-10-31) | `split_tone_data.zip`, 40 MB | 7 | `MANIFEST_SPLIT_TONE.txt` |

All files are pandas pickles (pandas 3.0.0). Load them with `pandas.read_pickle`. Only load pickles you trust: the
checksums prove a file is the one released here.

## Part 2: the attempt 14 tone files (this repository's release)

| File | Rows | Columns | What it is | Made by |
|---|---|---|---|---|
| `distill/doc_stance_distill.pkl` | 2,578 documents | `doc_id`, `type`, `release`, `stance`, `n_sent` | **the input of attempt 14**: the distilled reader's tone of every document (`stance` = mean sentence tone, `n_sent` = sentences) | `chronobert_distill tone` |
| `distill/scores_final.pkl` | 199,495 sentences | `row`, `lp1`, `lp2` | the final model's CORAL levels $P_1$, $P_2$ for every sentence from 2010 (`row` points into the corpus's sentence list) | `chronobert_distill final` |
| `distill/scores_foldA.pkl`, `scores_foldB.pkl` | sentences 2002-2009, 1994-2001 | same | the cross-fitted scores before 2010 | `chronobert_distill final` |
| `coral/haiku_sentence_probs.pkl` | 343,742 sentences | `doc_id`, `sentence`, `masked`, `p_dovish`, `p_neutral`, `p_hawkish` | the teacher: Claude Haiku 5.5's probabilities for every sentence (`masked` = the text actually sent, with dates and numbers masked). Remaking it costs money, so it is released. | `haiku_ordinal collect` |
| `coral/doc_stance_chronobert_coral.pkl`, `coral/doc_stance_haiku_ordinal.pkl` | 2,578 documents | as `doc_stance_distill.pkl` | the document tones of two earlier readers (attempts 9-11). Attempt 14 does not use their values; the code builds their panels only to keep the set of meetings identical to the earlier studies (a restriction that drops no meeting) | CORAL study |

sha256 checksums:

```
7344b6f2c4782c99150d3c1ef848360261612827547bb05fa79eccb27a8c4eea       184309  coral/doc_stance_chronobert_coral.pkl
48e1ff183109338c3fdae117ffdf5767bf88c769944fb5f63190184ce8430d96       184309  coral/doc_stance_haiku_ordinal.pkl
3a8fc23e40c31755fdc8f2105a62e55035aa0210974a0026cd9fdd09c73a729c    138515492  coral/haiku_sentence_probs.pkl
27d0d3f2c4aa2207fa44d07473bf8b745f39c4f428f795abcd0ee09f0405123d       184361  distill/doc_stance_distill.pkl
d903bb8f79b9c2fbc26fc85130c2e6789b03f1ac64867ee1c247c9791681695e      3192842  distill/scores_final.pkl
c208fcc43ff2d97ff46ba2c7d152a63090e7a1720abf1be34d020dc7debdf5b6      1443050  distill/scores_foldA.pkl
b44f085b8c020a247ba3429f19f21fab2d6f0700722ebc77bda7ca9c11e19ed5       866740  distill/scores_foldB.pkl
```

## Part 1: the files attempt 14 uses from the original release

| File | What it is | Used by |
|---|---|---|
| `t2.pkl` | **the meeting panel**: one row per FOMC meeting (270 rows, 260 with every input), the 16 S and 8 M columns, the cutoff (`cutoff`), the news-window start (`win_start`), the rate after the meeting (`y_level`), its change (`y_change`), the class (`y_class`), the unemployment rate for a Taylor rule (`t_u3`) and an `unscheduled` flag | every model |
| `t2_full.pkl` | the same panel plus the news columns of Study 4 (`n_stance`, `n_stance_chg`, `n_beige`, `n_news`, `n_gpr`, `n_epu`) | the base every tone panel is built on (`features/swap_tone.py` replaces `n_stance` and `n_stance_chg` with the chosen reader's tone). Attempt 14's forecasts do not use the Study 4 values, but a meeting is kept only if every one of these columns is present (`walk_forward.common_rows`), which is what leaves 260 meetings |
| `doc_stance_haiku.pkl` | an earlier reader's document tone (Study 4) | the common-meetings restriction only |
| `corpus.pkl` | every Fed document and its sentences | the reader (step 3) |
| `alfred/*.pkl`, `market_daily.pkl`, `meetings.pkl`, `calendar.pkl`, `fedfunds.pkl` | the raw downloads behind `t2.pkl` | rebuilding the panel (step 2) |

The other files in part 1 belong to the earlier studies. The full list with sizes and checksums is
[`guides/DATA_MANIFEST.txt`](guides/DATA_MANIFEST.txt); [`guides/DATA.md`](guides/DATA.md) is the original project's
description of each.

## What is not released

- **The distilled model weights** (about 1.7 GB for the final model and the two folds). `chronobert_distill final`
  remakes them on a CUDA GPU from `haiku_sentence_probs.pkl` and the public ChronoBERT 2009 weights
  (`manelalab/chrono-bert-v1-20091231` on Hugging Face). Training on another GPU will not be bit-identical, so the
  released sentence scores are what the saved results used.
- **API keys.** The downloads need a free FRED API key (`<data>/secrets/fred_api_key.txt`) and the teacher step a paid
  Anthropic key; neither is ever needed to reproduce the results from the releases, and none is in this repository.

## Data sources and terms

The economic and market series come from FRED and ALFRED (Federal Reserve Bank of St. Louis); the VIX is © Cboe Global
Markets, redistributed by FRED. Fed documents are public-domain US government works. The tone files are this
project's own model outputs. Check each source's terms before reusing the data for anything beyond reproducing this
study.
