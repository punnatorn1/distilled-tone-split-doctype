# 3. From Fed text to tone columns (Hd and Hs)

[Back to the README](../README.md) · Previous: [2. Economy and market inputs](2_economy_and_market_inputs.md) · Next:
[4. What each forecast sees](4_what_each_forecast_sees.md)

This chapter answers the professor's second question: **how is each sentence scored, and how do the sentence scores
become the numbers the forecaster uses?** Short answer: yes, it is averaging, in three stages (sentence to document,
document to document group, group to meeting), and the forecaster never sees the text itself.

## 3.1 The big picture

```
Fed text, 343,742 sentences (1994-2026)
   |  (1) the reader scores each sentence: distilled ChronoBERT
   v
sentence tone = P(hawkish) - P(dovish)               a number from -1 to +1
   |  (2) mean over the sentences of a document
   v
document tone
   |  (3) mean over the documents released in the meeting's news window, per document group
   v
the meeting's tone columns:  Hd = 2 columns (attempt 12)   or   Hs = 6 columns (attempt 14)
```

**Key fact:** the forecasting models never see the text or ChronoBERT's 768-number vectors. All the news is reduced to
2 (Hd) or 6 (Hs) numbers per meeting.

## 3.2 Words: hawkish, dovish, neutral

- **hawkish**: language pointing towards higher rates or fighting inflation ("inflation remains elevated");
- **dovish**: language pointing towards lower rates or supporting the economy ("the labour market remains weak");
- **neutral**: describes facts without pointing either way.

## 3.3 The text

- Four document types in three **groups**:
  - **policy**: the post-meeting statement and the press-conference transcript;
  - **minutes**: the meeting minutes, released about three weeks after the meeting;
  - **speech**: speeches by Fed Board members.
- A document must have at least 500 characters. It is split into sentences of 30 to 1,000 characters, lower-cased.

Code: `src/fed_news/download/fed_documents.py` (`build_corpus`, `split_sentences`), group map
`src/fed_news/features/news_columns.py` (`GROUP`).

## 3.4 The CORAL head: how one sentence becomes three probabilities

**Why a special output layer?** The classes are ordered: dovish < neutral < hawkish. A normal softmax treats the three
classes as unrelated, so it does not know that neutral sits *between* the other two. CORAL (Cao, Mirjalili and
Raschka, 2020) builds the order into the model.

**How CORAL works.** It turns the three-way question into two nested yes/no questions:

1. "Is this sentence **at least** neutral?" (that is, not dovish)
2. "Is this sentence hawkish?"

Both questions use **the same score $s$** but different thresholds (biases):

$$s = w^\top h,\qquad P_1 = \sigma(s + b_1) = P(\text{at least neutral}),\qquad P_2 = \sigma(s + b_2) = P(\text{hawkish})$$

- $h$ is the sentence's meaning as a vector of 768 numbers, from ChronoBERT;
- $w$ is 768 learned weights that compress $h$ into one score $s$; the higher $s$, the more hawkish;
- $\sigma$ is the sigmoid, $\sigma(x) = 1 / (1 + e^{-x})$, which maps any number to a probability between 0 and 1:
  $\sigma(0) = 0.5$, $\sigma(2) = 0.88$, $\sigma(-2) = 0.12$;
- because $b_1 \ge b_2$, always $P_1 \ge P_2$, as it must be: "at least neutral" can never be less likely than
  "hawkish".

**From $P_1, P_2$ to three probabilities and the tone:**

$$p_{dovish} = 1 - P_1,\qquad p_{neutral} = P_1 - P_2,\qquad p_{hawkish} = P_2$$

$$\text{tone} = p_{hawkish} - p_{dovish}$$

The tone runs from −1 (surely dovish) to +1 (surely hawkish), and is 0 for a neutral sentence or when the two sides
balance.

**Worked example** (the final distilled model, biases $b = (1.54, -1.99)$):

- sentence: "job gains have been strong in recent months, and the unemployment rate has declined substantially."
  (statement of 16 March 2022);
- $s = 2.40$;
- $P_1 = \sigma(2.40 + 1.54) = \sigma(3.94) = 0.98$ and $P_2 = \sigma(2.40 - 1.99) = \sigma(0.41) = 0.60$;
- $p_{dovish} = 0.02$, $p_{neutral} = 0.98 - 0.60 = 0.38$, $p_{hawkish} = 0.60$;
- **tone = 0.60 − 0.02 = +0.58**: fairly hawkish, because a strong labour market means the Fed is ready to hike.

**The CORAL loss** trains the reader. It is the log loss of the two yes/no questions added together:

$$\mathcal{L} = -\sum_{k=1}^{2}\Big[y_k \log P_k + (1-y_k)\log(1-P_k)\Big]$$

$y_k = 1$ if the true class is at rank $k$ or above. A neutral sentence has $(y_1, y_2) = (1, 0)$, so the loss is low
when $P_1$ is near 1 and $P_2$ near 0. (Tested to give the same values as the reference package `coral-pytorch`.)

Code: `src/fed_news/studies/coral/core.py` (`coral_loss`, `level_probs`, `class_probs`, `tone_from_levels`),
`src/fed_news/studies/coral/chronobert_coral.py` (`StanceNetCoral`).

## 3.5 The tensor shapes inside the reader

```
sentence (lower case)
-> tokenizer, at most 256 tokens                        [B, T]
-> ChronoBERT, last layer                               [B, T, 768]   (768 numbers per token)
-> mean over the tokens (padding excluded)              [B, 768]      (one vector per sentence = h)
-> CORAL head: s = w'h, then s + b1 and s + b2          [B, 2]        (P1, P2)
-> three class probabilities                            [B, 3]
-> tone                                                 [B]
```

$B$ = sentences in the batch, $T$ = tokens in the longest sentence of the batch.

## 3.6 The teacher: Claude Haiku 5.5

The reader learns from a teacher, Claude Haiku 5.5, which read every sentence once through Anthropic's paid Message
Batches API (already done; about 23 million tokens). Haiku is **not trained**:

- before sending, dates and numbers are masked as `[MONTH]`, `[NUM]`, `[YEAR]`, so Haiku is less able to recognise
  the period;
- 25 sentences per request; Haiku answers with three percentages [dovish, neutral, hawkish] adding to 100, turned into
  probabilities;
- 100 of the 343,742 sentences got no usable answer (0.03%).

**The problem with using Haiku directly: it may remember the future.** Haiku was trained on text up to mid-2026, which
covers almost every test meeting. In an earlier study Haiku named FOMC outcomes correctly 87.8% of the time from the
date alone. If Haiku "knows" what the Fed did next, its tone could carry that knowledge. Earlier tests found no
evidence of it, but could not rule it out. This is why attempt 12 built the distilled reader.

Code: `src/fed_news/studies/coral/haiku_ordinal.py`. It refuses to run without an API key file and is never needed to
reproduce the results, because its output (`<data>/coral/haiku_sentence_probs.pkl`) is in the data release.

## 3.7 The distilled reader ("Hd" reader): teacher and student

**Idea (knowledge distillation):**

- **teacher** = Haiku (reads well, but may remember the future);
- **student** = ChronoBERT, 2009 vintage (`manelalab/chrono-bert-v1-20091231`): a ModernBERT language model (22
  layers, about 149 million parameters) pre-trained only on text up to the end of 2009;
- the student learns to copy the teacher's probabilities **on sentences from before 2010 only**, then is frozen;
- from 2010 on, the student reads alone. Its knowledge stops at 2009, both from pre-training and from the teacher.
  **So Haiku's possible memory cannot reach the test period from 2010.**

A side benefit: far more training data. There are 144,211 sentences before 2010, about 100 times the human-labelled
sentences available before 2010.

**Soft targets.** Instead of teaching "this sentence is hawkish" (0 or 1), the student is taught the teacher's
probabilities. If Haiku gives $(p_D, p_N, p_H) = (0.1, 0.3, 0.6)$, the CORAL targets are $y_1 = p_N + p_H = 0.9$ and
$y_2 = p_H = 0.6$, with the same CORAL loss (it works for targets between 0 and 1).

**Choosing the settings.** Six settings were tried, training on 55,600 sentences (half of 1994-2006) and validating on
6,000 sentences (2007-2009). The winner has the lowest soft cross-entropy (distance from the teacher's probabilities;
lower is better):

| learning rate | epochs | soft cross-entropy | same class as Haiku | tone correlation with Haiku |
|---|---|---|---|---|
| 1e-5 | 1 | 0.8386 | 83.9% | 0.700 |
| 1e-5 | 2 | 0.8345 | 86.0% | 0.714 |
| 2e-5 | 1 | 0.8379 | 83.6% | 0.713 |
| **2e-5** | **2** | **0.8331 (chosen)** | **86.1%** | **0.721** |
| 4e-5 | 1 | 0.8416 | 83.6% | 0.690 |
| 4e-5 | 2 | failed (NaN) | 16.1% | – |

Fixed: batch 16, max length 256 tokens, weight decay 0.01, bias learning rate 0.01, one seed, fp16 (rows that
overflow in fp16 are recomputed in float32). Source: `results/distill/tuning.csv`, `results/distill/settings.json`.

**Who scores which sentence (cross-fitting).** No model ever scores a sentence it was trained on:

| Sentences from | are scored by the model trained on |
|---|---|
| 2010-2026 | 1994-2009 (the final model) |
| 2002-2009 | 1994-2001 (fold A) |
| 1994-2001 | 2002-2009 (fold B) |

**How well the student copies the teacher** (`results/distill/fidelity.json`): from 2010 (199,431 sentences, final
model) same class as Haiku 83.8%, tone correlation 0.729; before 2010 (144,211 sentences, cross-fitted) 88.0% and
0.714.

Code: `src/fed_news/studies/distill/chronobert_distill.py` (`tune`, `final`, `tone`; needs a CUDA GPU, about 3-4
hours on an RTX 2060).

## 3.8 From sentences to a meeting's numbers: the single tone Hd (2 columns)

**Step 1, document tone** = the mean tone of all sentences in the document.

**Step 2, the meeting's news window** = from the previous meeting's cutoff to just before this meeting's cutoff. So the
previous meeting's own statement (released one day after the previous cutoff) falls into this meeting's window.

**Step 3, mean per group first, then the mean of the groups:**

$$\text{n\_stance} = \text{mean of } \{\bar\tau_{policy},\ \bar\tau_{minutes},\ \bar\tau_{speech}\}\ \text{(groups with at least one document)}$$

Why group first? With one plain mean over all documents, the many speeches would swamp the single statement.
Averaging per group gives each group equal weight.

**Step 4, the change:** `n_stance_chg` = this meeting's `n_stance` − the previous meeting's. (A window with no document
keeps the previous value.)

**Worked example: 16 March 2022** (window 25 January to 14 March 2022)

| Released | Document | Group | Document tone |
|---|---|---|---|
| 26 Jan 2022 | statement (January meeting) | policy | −0.0285 |
| 26 Jan 2022 | press conference | policy | +0.0188 |
| 16 Feb 2022 | minutes of the January meeting | minutes | −0.0440 |
| 18 Feb 2022 | speech | speech | −0.0714 |
| 21 Feb 2022 | speech | speech | +0.0999 |
| 24 Feb 2022 | speech | speech | +0.1133 |

- group means: policy = (−0.0285 + 0.0188) ÷ 2 = −0.0048; minutes = −0.0440; speech = (−0.0714 + 0.0999 + 0.1133) ÷ 3
  = +0.0473;
- `n_stance` = (−0.0048 − 0.0440 + 0.0473) ÷ 3 = **−0.0005**;
- the previous meeting (26 January) had `n_stance` = −0.0213, so `n_stance_chg` = −0.0005 − (−0.0213) = **+0.0208**;
- **Hd for this meeting = [−0.0005, +0.0208]**.

**Why so small?** Most Fed sentences are neutral, and hawkish and dovish sentences cancel. Typical values lie between
−0.05 and +0.05; crises go far lower (10 June 2020: `n_stance` = −0.208). What matters is "above or below normal",
which the z-score (chapter 2.5) takes care of.

Code: `src/fed_news/features/news_columns.py` (`fed_stance`, `build_text_features`),
`src/fed_news/features/swap_tone.py` (`naive_panel`: puts a reader's document tones into the meeting panel).

## 3.9 Hs: six tone columns instead of two (attempt 14)

**The question:** does averaging the three groups into one number throw information away?

**Evidence from the inputs (looked at before any forecast was run):**

- from 2010, the combined `n_stance` nearly copies the statements and minutes (correlation with policy 0.96, minutes
  0.91, speech 0.84);
- the speech tone moves on its own: changes in the speech tone correlate only 0.15 with changes in the statement tone
  and 0.02 with changes in the minutes tone;
- speeches are the most numerous (7.8 per window on average) but get only one third of the weight in the combined
  number.

**The fix:** do not combine. Give each group its own column and let the model choose each group's weight.

| Column | Meaning | 16 Mar 2022 |
|---|---|---|
| `n_stance_policy` | mean tone of statements and press conferences in the window | −0.0048 |
| `n_stance_minutes` | mean tone of minutes | −0.0440 |
| `n_stance_speech` | mean tone of speeches | +0.0473 |
| `n_stance_policy_chg` | change in the policy tone from the previous meeting | +0.0202 |
| `n_stance_minutes_chg` | change in the minutes tone | −0.0469 |
| `n_stance_speech_chg` | change in the speech tone | +0.0892 |

In this example the combined number (−0.0005) says "neutral", but Hs shows the minutes leaning dovish while the speeches
lean clearly hawkish, and much more hawkish than at the previous meeting (+0.0892).

**Gaps.** A group with no document in a window keeps its latest value. Before a group's first document, the combined
`n_stance` is used instead (17 meetings for statements, 1995-1997, and 10 for speeches, 1995-1996; all are training
rows).

Code: `src/fed_news/studies/doctype/group_tone.py` (`GROUPS`, `HS`, `group_tones`, `split_panel`).

## 3.10 Input sizes per meeting

| Arm | S | M | tone | total per meeting |
|---|---|---|---|---|
| S+M (no tone) | 16 | 8 | 0 | **24** |
| S+M+Hd (attempt 12) | 16 | 8 | 2 | **26** |
| S+M+Hs (attempt 14) | 16 | 8 | 6 | **30** |

Chapter 4 shows how many meetings of these rows each model sees.
