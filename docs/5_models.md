# 5. The models: ordered logit, LSTM, GRU, TFT

[Back to the README](../README.md) · Previous: [4. What each forecast sees](4_what_each_forecast_sees.md) · Next:
[6. Tests and controls](6_tests_and_controls.md)

The README's main result shows all four models side by side with the single tone Hd. In the pre-registered designs
of both attempts (12 for Hd, 14 for Hs) the **ordered logit** was the primary model (its test was fixed before the
run) and the three networks (12b, 14b) a declared secondary test.

## 5.1 The ordered logit, step by step

### Why this model

The ordered logit is the standard model for ordered classes (cut < hold < hike). It is simple, every number can be
explained, it has few parameters (good for 48 to 250 rows), and it is the same idea as the CORAL head of chapter 3.4.

### The idea: one score and two cut points

Imagine a single "pressure to hike" $\eta$: the higher, the likelier a hike. Two cut points sit on the same axis:

```
          cut            |          hold           |          hike
  -----------------------+-------------------------+----------------------->  eta
                         t1                        t2
```

Far below $t_1$: probably a cut. Between $t_1$ and $t_2$: probably a hold. Far above $t_2$: probably a hike. Near a
cut point, the model is unsure.

### The formulas

**Step 1, the score**, a weighted sum of the z-scored inputs (24, 26 or 30 of them):

$$\eta = \beta_1 z_1 + \beta_2 z_2 + \cdots + \beta_K z_K$$

Each $\beta_k$ says how much input $k$ pushes $\eta$ up ($\beta > 0$) or down ($\beta < 0$) per unit of $z$. The product
$\beta_k z_k$ is input $k$'s **contribution** at this meeting.

**Step 2, the probabilities:**

$$P(\text{cut}) = \sigma(t_1 - \eta),\qquad P(\text{cut or hold}) = \sigma(t_2 - \eta)$$

$$P(\text{hold}) = \sigma(t_2 - \eta) - \sigma(t_1 - \eta),\qquad P(\text{hike}) = 1 - \sigma(t_2 - \eta)$$

If $\eta = t_2$ exactly, $P(\text{hike}) = 0.5$; the further $\eta$ goes above $t_2$, the closer $P(\text{hike})$ gets
to 1.

### How the model learns $\beta$, $t_1$, $t_2$

It picks the values that make the outcomes that really happened in the past as probable as possible, while
penalising large coefficients:

$$\min_{\beta,\,t_1,\,t_2}\ \underbrace{\frac{1}{n}\sum_{i=1}^{n} -\ln P_i(\text{true outcome of meeting } i)}_{\text{training log loss}} \;+\; \underbrace{\frac{\lambda}{2}\sum_k \beta_k^2}_{\text{ridge penalty}}$$

- **The log loss** is low when the model gives high probability to what actually happened (chapter 6.1).
- **The ridge (L2) penalty** pulls the coefficients towards 0. Why: with little data and inputs that move together
  (CPI and PCE, say), a model can fit the training rows perfectly with a huge positive weight on one and a huge
  negative weight on the other, and then forecast badly (overfitting). The penalty prevents that.
- **$\lambda$** sets the penalty's strength. It is chosen from {0.01, 0.03, 0.1, 0.3, 1, 3} again at every fit, by
  **expanding-window cross-validation**:

  ```
  training rows in date order, cut into 6 blocks: [1][2][3][4][5][6]
  fold 1: train [1]            -> score on [2]
  fold 2: train [1][2]         -> score on [3]
  fold 3: train [1][2][3]      -> score on [4]
  fold 4: train [1][2][3][4]   -> score on [5]
  fold 5: train [1]...[5]      -> score on [6]
  ```

  The $\lambda$ with the lowest mean log loss over the 5 folds wins. The split is never random, because the model must
  always use the past to forecast the future.
- The cut points are not penalised, and $t_2 > t_1$ is enforced by writing $t_2 = t_1 + e^{b}$.
- The minimum is found with L-BFGS, the standard optimiser for smooth problems like this.

Code: `src/fed_news/models/linear.py` (`OrderedLogit`, `expanding_folds`, `choose_penalty`), settings
`config.ORDLOGIT_GRID` and `config.CV_FOLDS`.

### Worked example: S+M forecasting 16 March 2022

**Set-up:** trained on 223 meetings (1995-03-28 to 2022-01-26); chosen $\lambda$ = 0.03; cut points $t_1$ = −3.270,
$t_2$ = 3.802. The table lists each input's contribution, largest first.

| Input | raw | z | β | contribution β × z |
|---|---|---|---|---|
| s_cpi_core_yoy | 6.416 | 6.98 | −0.508 | −3.544 |
| m_sp_y6m | 0.735 | 2.19 | +1.334 | **+2.917** |
| m_sp_y1 | 1.155 | 2.54 | +0.990 | **+2.515** |
| s_pce_core_yoy | 5.209 | 6.86 | −0.269 | −1.847 |
| m_sp_y2 | 1.745 | 2.47 | +0.590 | +1.454 |
| m_sp_tbill_3m | 0.325 | 1.52 | +0.845 | +1.284 |
| s_slope_1y6m | 0.420 | 2.37 | −0.331 | −0.784 |
| s_hpi_yoy | 17.612 | 2.62 | +0.293 | +0.768 |
| s_gdp_yoy | 11.784 | 3.22 | +0.207 | +0.666 |
| s_ppi_yoy | 19.266 | 2.84 | +0.209 | +0.592 |
| s_pce_yoy | 6.061 | 4.47 | +0.110 | +0.491 |
| s_earn_yoy | 6.650 | 3.08 | +0.152 | +0.468 |
| s_slope_10y2y | 0.270 | −0.94 | −0.452 | +0.426 |
| s_cpi_yoy | 7.912 | 4.35 | +0.084 | +0.366 |
| s_u6_yoy | −3.900 | −1.63 | −0.208 | +0.339 |
| m_zero_bound | 1 | 1.46 | +0.224 | +0.328 |
| m_target_level | 0.125 | −1.00 | +0.292 | −0.293 |
| s_rate_lag | 0.125 | −1.00 | +0.292 | −0.293 |
| s_payems_chg | 678 | 1.27 | +0.133 | +0.169 |
| s_m2_mom | 1.135 | 0.88 | −0.185 | −0.162 |
| m_wti | 103.22 | 1.76 | +0.072 | +0.126 |
| m_vix | 31.77 | 1.21 | −0.090 | −0.109 |
| s_m2_yoy | 12.648 | 1.42 | +0.060 | +0.085 |
| s_u6_mom | 0.1 | 0.40 | +0.036 | +0.014 |
| **sum = η** | | | | **+5.977** |

**The probabilities:**

- $P(\text{cut}) = \sigma(t_1 - \eta) = \sigma(-3.270 - 5.977) = \sigma(-9.25) = 0.0001$
- $P(\text{cut or hold}) = \sigma(t_2 - \eta) = \sigma(3.802 - 5.977) = \sigma(-2.175) = 0.102$
- $P(\text{hold}) = 0.102 - 0.0001 = 0.102$
- $P(\text{hike}) = 1 - 0.102 = 0.898$

The model gives a hike 0.898, and the Fed hiked.

**What the table teaches:**

1. **The market does most of the work:** the four spreads together contribute +8.17. The market bet on a hike and the
   model believed it.
2. **A coefficient's sign is not "the effect of that input":** core CPI has a negative β (−0.508), which sounds wrong
   since high inflation should push towards a hike. The reason is that many inputs move together (core CPI, core PCE,
   CPI, PCE, and the spreads, which already price inflation), and the model splits the weight among them in whatever
   way fits the training rows best, not by economic logic. Taken together, the inflation inputs still push towards a
   hike. **Do not interpret coefficients one at a time.**
3. **The duplicated pair gets equal weight:** `s_rate_lag` and `m_target_level` are equal at every row, and the ridge
   penalty splits the weight between them exactly (β = 0.292 each).

These numbers come from refitting the S+M ordered logit with this repository's code (`OrderedLogit`, the same
training rows and λ as the walk-forward run); the probabilities match `results/doctype/logit/preds_T2_S+M.csv` to
every printed digit.

### A confident mistake: 29 July 2020

The rate was 0.125% (Covid). The outcome was a **hold**, but S+M gave **P(hike) = 0.992**.

| Input | raw | training mean | z | contribution |
|---|---|---|---|---|
| s_payems_chg | +4,800 (4.8 million jobs) | +111 | **18.6** | **+6.45** |
| s_m2_yoy | 22.9% | 6.1% | 7.9 | +4.03 |
| s_u6_yoy | +10.8 | 0.0 | 5.9 | −1.77 |
| s_u6_mom | −3.2 | 0.0 | −10.9 | −0.94 |

$\eta$ = 9.52, far above $t_2$ = 4.71.

**What happened:** in June 2020, 4.8 million people went back to work after the lockdown, 18.6 typical deviations above
the historical mean. The model had learned in normal times that "many new jobs = hot economy = hike", and extrapolated
far beyond anything it had seen (chapter 2.5). In reality this was a rebound from 20 million lost jobs, not a hot
economy. This meeting is the single largest gain from the tone (chapter 6.6).

### From probabilities to a forecast rate

The Clark-West test (chapter 6.4) needs a forecast **rate**, not probabilities. The rate is the expected value:

$$\hat r = r_{\text{before}} + P(\text{cut}) \times \bar\Delta_{\text{cut}} + P(\text{hike}) \times \bar\Delta_{\text{hike}}$$

$\bar\Delta$ is the average size of past cuts and hikes in the training rows (a hold is 0).

**16 March 2022:** in training the average cut was −0.4228 and the average hike +0.2576, so
$\hat r = 0.125 + 0.0001 \times (-0.4228) + 0.898 \times 0.2576 = 0.356\%$ (the actual rate after the meeting was
0.375%).

Code: `src/fed_news/models/linear.py` (`move_sizes`), `src/fed_news/models/walk_forward.py` (`run_spec`).

## 5.2 Why try networks at all

A network reads a **sequence** of meetings (chapter 4) and can learn non-linear patterns such as "inflation is high
**and** the market expects a hike **and** the rate is still at zero", which the ordered logit cannot. The price is many
parameters, which need much more data than the 48 to 250 meetings available (chapter 4.3).

## 5.3 LSTM: memory with gates

**Idea:** read the meetings one by one ($z_{j-L+1}$ up to $z_j$) and carry a memory $c_t$ along. At each step, three
gates decide what to forget, what to take in and what to output. Each gate is a vector of numbers between 0 (closed)
and 1 (open).

$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \qquad \text{forget gate: how much old memory to keep}$$

$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \qquad \text{input gate: how much new information to take}$$

$$\tilde c_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \qquad \text{the proposed new information}$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t \qquad \text{new memory = kept old + accepted new}$$

$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o),\qquad h_t = o_t \odot \tanh(c_t) \qquad \text{output gate and the step's output}$$

- $x_t$ is the 30-number row of meeting $t$ (S+M+Hs); $h_t$ is the summary state of size $H$ (the hidden size);
  $\odot$ multiplies element by element; $\tanh$ squeezes values into −1 to 1.
- For example, if the market has expected hikes for several meetings in a row, the memory can accumulate that signal.

**In this repository, with the tensor shape at every step:**

```
X  [B, L, 30]
-> LSTM (hidden H, 1 or 2 layers, dropout between layers)        [B, L, H]
-> keep the last step's h (the meeting being forecast)           [B, H]
-> CORAL head: s = w'h, P1 = sigma(s + b1) = P(not a cut),
                        P2 = sigma(s + b2) = P(hike)             [B, 2]
-> P(cut) = 1 - P1, P(hold) = P1 - P2, P(hike) = P2              [B, 3]
```

$B$ = sequences in the batch. The CORAL head is the same formula as chapter 3.4, with the order cut < hold < hike
instead of dovish < neutral < hawkish.

## 5.4 GRU: a slimmer LSTM

The GRU merges the memory and the output into one state and has two gates, so it has about 25% fewer parameters than
an LSTM of the same size.

$$u_t = \sigma(W_u[h_{t-1}, x_t]) \qquad \text{update gate: how much the state changes}$$

$$r_t = \sigma(W_r[h_{t-1}, x_t]) \qquad \text{reset gate: how much of the old state is used to propose the new one}$$

$$\tilde h_t = \tanh(W[r_t \odot h_{t-1}, x_t]),\qquad h_t = (1 - u_t)\odot h_{t-1} + u_t \odot \tilde h_t$$

The rest (last step, CORAL head, shapes) is as for the LSTM.

Code for both: `src/fed_news/studies/coral/networks.py` (`build_coral_model`), `src/fed_news/models/neural.py`
(`RNNRegressor`, `make_sequences`).

## 5.5 TFT: a compact Temporal Fusion Transformer

The TFT is built for multi-variable time series. It is made of parts with clear jobs.

**Two building blocks:**

- **GLU (gated linear unit)**, a gate that can switch a path off:

  $$\text{GLU}(x) = (W_a x + b_a) \odot \sigma(W_b x + b_b)$$

  When the $\sigma$ part is near 0 the path is closed, so the model can skip a part that does not help, which matters
  when data are scarce.

- **GRN (gated residual network)**, two processing layers with a shortcut:

  $$\text{GRN}(x) = \text{LayerNorm}\big(x + \text{GLU}(W_2\,\text{ELU}(W_1 x + b_1) + b_2)\big)$$

  The shortcut "$x +$" means that when the GLU is closed the output is just $x$. LayerNorm keeps the numbers' scale
  steady so training is easier.

**The flow, with shapes (S+M+Hs, 30 inputs):**

```
X [B, L, 30]
(1) variable selection
    - each of the 30 inputs goes through its own GRN              [B, L, 30, H]
    - input weights: softmax(GRN(the whole row))                  [B, L, 30]   (sum to 1)
    - weighted sum                                                [B, L, H]
(2) an LSTM reads the sequence                                    [B, L, H]
    + GLU + shortcut + LayerNorm
(3) attention (1, 2 or 4 heads), looking backwards only           [B, L, H]
    + GLU + shortcut + LayerNorm
(4) another GRN + GLU + shortcut + LayerNorm                      [B, L, H]
(5) last step -> CORAL head                                       [B, 3]
```

- **(1) Variable selection** lets the model decide which inputs matter at this meeting. It is the only place in any of
  the models where each tone column gets its own learned weight.
- **(3) Attention** lets the current step look back at any of the $L$ meetings and weigh the relevant ones:

  $$\text{score}(t, t') = \frac{q_t \cdot k_{t'}}{\sqrt{d}},\qquad \text{weight} = \text{softmax}_{t'}(\text{score}),\qquad \text{out}_t = \sum_{t'} \text{weight}_{t,t'}\, v_{t'}$$

  $q$ (query), $k$ (key) and $v$ (value) are linear maps of $h$. A **causal mask** stops step $t$ looking at later
  steps. All heads share $v$ and their outputs are averaged, so the weights are easy to read.

Code: `src/fed_news/models/tft.py` (`GLU`, `GRN`, `BatchedGRN`, `VariableSelection`,
`InterpretableMultiHeadAttention`, `CompactTFT`).

## 5.6 Training the networks

**Loss:** the CORAL loss (chapter 3.4) on the order cut < hold < hike, against the true outcomes of the training
meetings.

**Optimiser:** Adam with weight decay (not applied to the CORAL biases). The CORAL biases start at the logits of the
cumulative class shares in the training rows, so the model starts from "forecast the past frequencies" and adjusts
from there.

**Settings to choose, and what each does:**

| Setting | Search range | What it does |
|---|---|---|
| hidden size H | {8, 16, 32, 64, 128, 256}; TFT {8, 16, 32, 64} | size of the internal state: bigger can learn more complex patterns but overfits more easily |
| layers (LSTM, GRU) | {1, 2} | stacked layers |
| heads (TFT) | {1, 2, 4} | attention views |
| dropout | 0 to 0.5 | during training, randomly switches off this share of the network so it cannot rely on one path |
| learning rate | 1e-4 to 1e-2 (log scale) | step size of each weight update |
| weight decay | 1e-6 to 1e-2 (log scale) | pulls weights towards 0 at every step (the same idea as the ridge penalty) |
| lookback L | {2, 4, 8, 12} | meetings seen per forecast (chapter 4) |
| batch | {16, 32, 64} | sequences per weight update |
| epochs | not searched directly (from early stopping) | passes over the training data |

**How the settings are chosen (tuning):**

1. **When:** at 7 points, the first meeting of 2001, 2005, 2009, 2013, 2017, 2021 and 2025, each using only data
   before that point.
2. **How many tries:** 20 per point, with **Optuna's TPE sampler**, which is not purely random: it looks at which tries
   did well and proposes the next one near them.
3. **How a try is scored:** the data before the tuning point are cut into 5 blocks in time order; 4 expanding folds
   (as in 5.1); in each fold training stops when the validation log loss has not improved for 20 epochs (**early
   stopping**, at most 300 epochs). Score = mean validation log loss of the 4 folds.
4. **Epochs** used afterwards = the median of the best epoch in the 4 folds.
5. **Every network × arm pair is tuned separately**, the placebo arms included, so every arm gets its own best
   settings (fair to all arms).

**Forecasting:**

- refit every 8 meetings, on all earlier data, with the tuned settings;
- **8 seeds** (8 trainings from different random starting weights), averaging $P_1, P_2$ over the 8 to reduce
  randomness;
- every chosen setting is listed in [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

**The saved settings are guarded.** Each `results/*/networks/settings_*.json` stores a fingerprint of the panel it was
tuned on (a hash of the architecture, column names, dates, values and outcomes). If the data or the column list
change, the forecast step refuses to reuse the settings instead of silently mixing runs. Each seed's forecast file is
also tied to a hash of its settings, so `forecast` reuses the saved seed forecasts and only retrains when something
changed.

Code: `src/fed_news/studies/coral/run_networks.py` (`tune_point_coral`, `cv_score_coral`, `run_tuned_coral`,
`seed_forecast`, `ensemble`, `panel_fingerprint`), `src/fed_news/studies/coral/networks.py` (`fit_validate_coral`,
`fit_fixed_coral`), `src/fed_news/studies/doctype/run_networks.py` (`tune`, `forecast` for the Hs arms).
