# 4. What each forecast sees, and how many meetings

[Back to the README](../README.md) · Previous: [3. From Fed text to tone columns](3_from_text_to_tone.md) · Next:
[5. The models](5_models.md)

This chapter answers the professor's first question: **what does each forecast see, and how many data points before
it predicts?** The question has two different meanings, and they must be kept apart.

| | Meaning | Ordered logit | LSTM / GRU / TFT |
|---|---|---|---|
| **A. input at forecast time** | to forecast meeting $j$, how many meetings' rows go in | **1** (meeting $j$ only) | **L** (meeting $j$ and the $L-1$ before it), $L$ = 2, 4 or 8 in practice |
| **B. training data** | how many meetings the model learned from | **every meeting before $j$**, refit at every meeting | **every meeting before the refit point**, refit every 8 meetings |

## 4.1 A. The input at forecast time

**The ordered logit sees one meeting:** the vector of 24, 26 or 30 numbers for the meeting it forecasts. That does not
mean it knows nothing about the past:

- every `_yoy` column has 12 months of history built in;
- `_mom`, `_chg` and `n_stance_chg` columns carry the previous month or meeting;
- `s_rate_lag` says where the rate stands now.

**A network sees L meetings in time order**, called the lookback:

$$X_j = [\,z_{j-L+1},\ \ldots,\ z_{j-1},\ z_j\,] \qquad \text{shape } [L \times 30] \text{ for S+M+Hs (24 for S+M, 26 for S+M+Hd)}$$

$L$ is not set by hand: tuning picks it from {2, 4, 8, 12} at every tuning point (every 8 meetings is about one year;
the tuning points are the first meeting of 2001, 2005, 2009, 2013, 2017, 2021 and 2025). The values actually chosen:

| Network | Arm | 2001 | 2005 | 2009 | 2013 | 2017 | 2021 | 2025 |
|---|---|---|---|---|---|---|---|---|
| LSTM | S+M | 2 | 4 | 4 | 2 | 2 | 2 | 4 |
| LSTM | S+M+Hd | 4 | 4 | 4 | 2 | 2 | 2 | 4 |
| LSTM | S+M+Hs | 2 | 2 | 4 | 2 | 2 | 2 | 4 |
| GRU | S+M | 4 | 4 | 4 | 4 | 8 | 2 | 2 |
| GRU | S+M+Hd | 4 | 4 | 4 | 4 | 2 | 4 | 2 |
| GRU | S+M+Hs | 4 | 4 | 4 | 2 | 2 | 4 | 4 |
| TFT | S+M | 8 | 8 | 8 | 2 | 2 | 4 | 2 |
| TFT | S+M+Hd | 8 | 2 | 4 | 4 | 2 | 8 | 2 |
| TFT | S+M+Hs | 8 | 8 | 4 | 4 | 4 | 4 | 2 |

- $L = 12$ (one and a half years) was never chosen; mostly 2 to 4 meetings (3 to 6 months).
- $L = 2$ means the model sees this meeting and the one before.
- The value used for a meeting comes from the latest tuning point: a 2022 meeting uses the values tuned at the start
  of 2021.

**Example: the TFT forecasting 16 March 2022.** Both arms use the values tuned at the start of 2021.

- TFT with S+M+Hd chose $L = 8$, so its input is a tensor of shape $[8, 26]$ built from these 8 meetings:

  ```
  28 Apr 2021 | 16 Jun 2021 | 28 Jul 2021 | 22 Sep 2021 | 3 Nov 2021 | 15 Dec 2021 | 26 Jan 2022 | 16 Mar 2022
  each row = 26 z-scored numbers (S 16 + M 8 + Hd 2)
  ```

- TFT with S+M+Hs chose $L = 4$, so its input has shape $[4, 30]$: the last 4 of those meetings (3 Nov 2021 to
  16 Mar 2022), each row 30 numbers (S 16 + M 8 + Hs 6).

In both cases the last row is the meeting being forecast, built only from data dated before its cutoff (chapter 1.5).

## 4.2 B. The training data: walk-forward

**Walk-forward** imitates real use: forecast one meeting at a time in date order, using only meetings whose outcome is
already known.

```
forecast 2001-01-03:  train on meetings 1995-03-28 ... 2000-12-19   (48 meetings)
forecast 2001-01-31:  train on meetings 1995-03-28 ... 2001-01-03   (49 meetings)
...
forecast 2022-03-16:  train on meetings 1995-03-28 ... 2022-01-26   (223 meetings)
```

- The **ordered logit** is refit **at every meeting** (212 fits in the test period).
- The **networks** are refit **every 8 meetings** (and at every tuning point), because training is much slower. The 8
  meetings between refits are forecast by the same model, which still only knows the past.
- Meetings available for training at the tuning points: 48 (2001), 83 (2005), 117 (2009), 149 (2013), 181 (2017),
  214 (2021), 246 (2025).

## 4.3 Very little data for a network

Compare the number of weights to learn with the number of training meetings (input of 26 columns):

| Model | Parameters |
|---|---|
| ordered logit | 28 (26 coefficients + 2 thresholds) |
| LSTM, hidden 16, 2 layers | 5,010 |
| TFT, hidden 8, 1 head | 9,240 |
| GRU, hidden 128, 2 layers | 159,106 |
| LSTM, hidden 256, 2 layers | 817,410 |

All with only 48 to 250 meetings to learn from. This is the main reason the networks do worse than the ordered logit
(chapter 7).

Code: `src/fed_news/models/walk_forward.py` (`train_mask`, `run_spec`), `src/fed_news/models/neural.py`
(`make_sequences`: builds the $[L \times \text{columns}]$ windows), `src/fed_news/studies/coral/run_networks.py`
(`run_tuned_coral`: the refit schedule).
