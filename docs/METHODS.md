# Methods (pre-registration draft)

Status: **draft**. This document is frozen at the end of M1, before any model is trained on real data (D014). After the freeze, every change is a deviation: it gets a DECISIONS entry with the date, the reason, and whether any result had been seen when it was made. Items still open are listed in section 13. Reference numbers point to the list in [SPEC.md](SPEC.md#12-references).

## 1. Data sources

| Dataset | File used | Fields |
|---|---|---|
| MovieLens-1M | `ml-1m.zip` from files.grouplens.org, `ratings.dat` | user, item, timestamp (s) |
| MovieLens-20M | `ml-20m.zip` from files.grouplens.org, `ratings.csv` | user, item, timestamp (s) |
| Steam | `steam_reviews.json.gz` from mcauleylab.ucsd.edu/public_datasets/data/steam/ | user, product, date (day) |
| Amazon Reviews 2023, Video_Games | 0-core `rating_only` file from the official benchmark release | user_id, parent_asin, timestamp (ms) |

The download script records the SHA-256 of every file. Preprocessing is deterministic given those files. No data file, processed file or per-user output is committed.

## 2. Preprocessing

1. Read every interaction with its row index in the source file.
2. Treat every interaction as a positive event; ratings are ignored.
3. Sort by (user, timestamp, row index). The row index breaks timestamp ties, so order is stable and reproducible.
4. Remove repeated (user, item) pairs, keeping the earliest.
5. For leave-one-out, apply iterative 5-core filtering to users and items on the full deduplicated data, as in [1]. This uses information from the evaluation period, which is inherent to the published protocol; the report says so. The temporal split filters on pre-cutoff data only (section 3.2).
6. Map items to 1..N in sorted order of their original ids. 0 is padding; N+1 is the BERT4Rec mask token.
7. Maximum input length L: 200 for MovieLens-1M and MovieLens-20M, 50 for Steam and Amazon, as in [1] and [5]. Longer histories keep the most recent L items.

Dataset statistics (users, items, interactions, sequence length quantiles, density, share of consecutive same-timestamp pairs) are written to `results/data/`. They are compared with published statistics; a difference above 1% in users, items or interactions is explained in DECISIONS before the freeze. For Amazon, our 5-core counts are also checked against the official 5-core table.

## 3. Splits

### 3.1 Leave-one-out (primary)

For each user with sequence s_1..s_n: training sequence s_1..s_{n-2}; validation input s_1..s_{n-2} with target s_{n-1}; test input s_1..s_{n-1} with target s_n. Models are not retrained on validation data.

### 3.2 Global temporal split with last-item target (secondary)

Following the GTS-Last variant in [12]:
- T is the 0.9 quantile of all deduplicated interaction timestamps. Interactions with timestamp equal to T belong to the test side.
- Training period: interactions with t < T. Iterative 5-core filtering on this period defines the user and item vocabulary.
- Test users: users in the vocabulary with at least one interaction at t ≥ T. The target is the user's last interaction. The input is every earlier interaction of that user in time order, from either period, with out-of-vocabulary items removed. Targets outside the vocabulary stay in the evaluation and count as misses.
- Validation repeats the construction inside the training period with T_val, the 0.9 quantile of training-period timestamps. The model trains on t < T_val, early stops on the validation targets and is then evaluated on the test targets without retraining. Interactions in [T_val, T) therefore serve as test inputs only; this is recorded as a limitation.
- Hyperparameters are not tuned on this split; final runs reuse the configurations selected under leave-one-out (D007).

### 3.3 Shuffled-order control (D017)

For control runs, each user's training sequence and every evaluation input are permuted uniformly at random with a fixed per-user seed, following the shuffle test of [11]. Targets are unchanged. Arms: the final SASRec-CE configuration on all four datasets and the final BERT4Rec configuration on MovieLens-1M and MovieLens-20M, leave-one-out, same seeds as the unshuffled runs. Expected outcome: a drop larger than SESOI on MovieLens-1M and MovieLens-20M. If a sequential model does not drop there, the order-handling code and the evaluation are audited before any other result is reported.

## 4. Models and objectives

### 4.1 SASRec backbone (arms A1 to A4)

Following [1]: learned item and position embeddings, dropout on the embedding sum, B blocks of pre-LayerNorm causal self-attention and a point-wise ReLU feed-forward network with residual connections, and a final LayerNorm. Item embeddings are shared between input and output. The score of item j at position i is the dot product of the hidden state h_i and the item embedding e_j.

Inputs are left-padded with id 0, whose embedding is zero. The attention mask is causal and blocks padded keys; a padded query may attend to itself so no row is empty. Padded positions never enter a loss or a metric.

### 4.2 BERT4Rec (arm A5)

Following [2]: bidirectional Transformer blocks (post-LayerNorm, GELU), learned position embeddings, and an output head (dense, GELU, LayerNorm, then dot product with the shared item embeddings plus an item bias). Training masks each position with probability ρ, with at least one masked position per sequence, and applies full softmax cross-entropy at masked positions. At inference a mask token is appended after the input and scored. The token replacement scheme and the extra last-item-masked samples of the official code are an open item (section 13).

### 4.3 Objectives

For position i with target y_i and score vector s_i:

- **Full cross-entropy (A2):** −log softmax(s_i)[y_i] over items 1..N.
- **Chunked cross-entropy:** the same loss computed over slices of positions so the full logit tensor never exists at once. Used only as an implementation variant in RQ2 and RQ3.
- **Sampled softmax (A3):** K negatives drawn uniformly from 1..N once per batch and shared by all positions. Logits are [s_{i,y_i}, s_{i,n_1}, ..., s_{i,n_K}]; a negative equal to y_i is masked for that position. The uniform proposal makes the log-Q correction constant, so none is applied.
- **BCE (A1):** one negative per position, drawn uniformly from items not in the user's history, as in [1]. Loss −log σ(s⁺) − log(1 − σ(s⁻)).
- **gBCE (A4):** K negatives drawn uniformly with replacement, negatives equal to the positive masked. Loss −β log σ(s⁺) − Σ_k log(1 − σ(s⁻_k)) with α = K/(N − 1) and β = α(t(1 − 1/α) + 1/α) [4]. Computed in fp32.

### 4.4 Floor baselines

- **MostPop (B1):** items ranked by interaction count in the training data.
- **Transition (B2):** score of j is the count of consecutive pairs (last input item, j) in training sequences; ties broken by training popularity.

Both use the same evaluation code, masking and test-access rules as the neural models.

## 5. Training and tuning

### 5.1 Training loop

- One epoch is one pass over all training users in random order, one sequence per user (the most recent L+1 training items for SASRec arms, the most recent L for BERT4Rec).
- Adam (β₁ 0.9, β₂ 0.999), batch size 128, gradient-norm clipping at 5.
- bf16 autocast for all M3 runs; TF32 allowed for fp32 matmuls. H3c tests whether bf16 changes quality.
- Seeds fix Python, NumPy and PyTorch generators and the data order. Fully deterministic kernels are not enforced; their cost is measured instead of assumed.

### 5.2 Early stopping and caps

- Validation NDCG@10 is computed after every epoch. The best checkpoint by validation NDCG@10 is restored at the end.
- On datasets with more than 10,000 users, per-epoch early stopping uses a fixed random subsample of 10,000 validation users; the selected checkpoint is then scored on the full validation set, and that value is the trial objective.
- Caps leave a margin over published training lengths ([5] uses at most 100 epochs with patience 10 for SASRec and 200 with patience 20 for BERT4Rec):

| Arms | Max epochs | Patience |
|---|---|---|
| A1 to A4 | 300 | 30 |
| A5 | 1000 | 100 |

- A wall-clock guard of 8 hours per run protects the schedule. A run that ends by a cap or the guard instead of by patience is marked truncated. Every run records epochs trained, the best epoch and whether it was truncated.
- If more than 20% of an arm's tuning trials on a dataset are truncated, that arm's final runs on that dataset use doubled caps. Truncation rates are reported for every arm and dataset.

### 5.3 Search spaces

Shared by all neural arms: learning rate log-uniform [1e-4, 1e-2]; embedding size {64, 128, 256}; blocks {1, 2, 3}; heads {1, 2}; dropout uniform [0.1, 0.6]; weight decay {0, 1e-4, 1e-2}.

| Arm | Additional dimensions |
|---|---|
| A1 SASRec-BCE | none |
| A2 SASRec-CE | none |
| A3 SASRec-SS | K ∈ {64, 256, 1024, 4096} |
| A4 gSASRec | K ∈ {64, 128, 256}; t uniform [0.5, 1.0] |
| A5 BERT4Rec | ρ uniform [0.1, 0.4] |

### 5.4 Tuning procedure

One Optuna study per (arm, dataset), leave-one-out only, TPE sampler with a fixed seed, the first 5 trials random, a median pruner after 5 complete trials with warm-up equal to 10% of the arm's epoch cap. Trial budget per study: 20 (MovieLens-20M: 10). All trials use training seed 0. The objective is full validation NDCG@10 of the restored checkpoint. Studies are stored in SQLite so tuning resumes after interruption.

### 5.5 Final runs

The best trial per (arm, dataset) is rerun with seeds 1..n, where n follows section 7.5 (at least 5), on leave-one-out and on the temporal split. Floor baselines are deterministic and run once.

## 6. Evaluation

- Every item 1..N is scored. Items in the user's input sequence, the padding id and the mask id are set to −∞.
- Rank of the target: r = 1 + #{j ≠ y : s_j ≥ s_y} over non-masked items, so ties count against the model.
- Single target per user: NDCG@K = 1/log₂(r + 1) if r ≤ K, else 0; HR@K = 1 if r ≤ K, else 0. Metrics are averaged over evaluation users.
- Primary metric: NDCG@10. Secondary, descriptive only: HR@10, NDCG@20, HR@20.
- Model selection, early stopping and tuning see validation data only. Test data are read once per final run, after the checkpoint is frozen (section 11.6).

## 7. Statistical analysis

### 7.1 Unit and scale

The unit of analysis is one training run. For run r, y_r = log(test NDCG@10 averaged over users). Differences between arms are differences in mean y across seeds, so exp(Δ) − 1 is a relative difference. User-level paired tests are not used for inference because they ignore run-to-run variation [17].

### 7.2 Smallest effect size of interest

δ = log(1.02), a 2% relative difference in NDCG@10.

### 7.3 Tests

All tests are Welch t-tests on y between two arms on the same dataset and split, at α = 0.05.

- **Directional:** two-sided test of equal means.
- **Non-inferiority (H1a):** one-sided test of H0: μ_A − μ_B ≤ −δ.
- **Minimum effect (H1b):** one-sided test of H0: μ_other − μ_BCE ≤ δ.
- **Equivalence (TOST):** two one-sided tests against −δ and +δ; equivalent when both reject (the 90% CI lies inside (−δ, δ)).

Published gaps (BERT4Rec over SASRec-BCE [2], gSASRec over BERT4Rec [4], SASRec-CE over BERT4Rec [5]) are classified per dataset:
- confirmed: directional test significant after correction, with the published sign;
- reversed: significant with the opposite sign;
- equivalent: TOST significant;
- confirmed below SESOI: both of the first and third apply;
- inconclusive: otherwise, or the comparison was pre-classified as underpowered (7.5) and nothing rejects.

### 7.4 Families and multiplicity

Holm correction within each family:

| Family | Tests |
|---|---|
| F1 RQ1 directional | 6 arm pairs among A1, A2, A4, A5 × 4 datasets = 24 |
| F1-NI H1a | A2 vs A5 on MovieLens-1M and MovieLens-20M = 2 |
| F1-ME H1b | A1 vs each of A2, A4, A5 × 4 datasets = 12 |
| F2 H2b | A3, A4 and A1 vs A2, TOST × 4 datasets = 12 |
| F3 H3c | bf16 vs fp32, TOST, MovieLens-1M and Video_Games = 2 |
| F4 shuffle | A2 shuffled vs A2 × 4 datasets, A5 shuffled vs A5 × 2 datasets, one-sided = 6 |

H1b holds on a dataset when all three of its F1-ME tests reject. Temporal-split results use the same tests, reported as secondary without confirmatory claims; agreement with leave-one-out is summarised by Kendall's τ between arm rankings per dataset.

### 7.5 Number of seeds (D018)

The rule is fixed now; the M2 pilot only supplies the input σ.

- σ is the pooled standard deviation of y_val (log validation NDCG@10) across seeds within arms, from the pilot (3 seeds per arm A1 to A5 on MovieLens-1M, leave-one-out). Test data are not read in the pilot.
- Required seeds per arm:
  - n_eq = ⌈2 (z₀.₉₅ + z₀.₉₀)² (σ/δ)²⌉ + 1 (TOST with true difference 0, power 0.8)
  - n_dir = ⌈2 (z_{1−α'/2} + z₀.₈₀)² (σ/δ)²⌉ + 1, with α' = 0.05/24 (the strictest Holm step in F1, power 0.8 at an effect of δ)
  - n = min(10, max(5, n_eq, n_dir))
- Example: σ = 0.01 gives (σ/δ)² = 0.255, n_eq = 6, n_dir = 9, so n = 9.
- After the first 5 final seeds per arm on each dataset, σ is re-estimated from their validation values (never test values) and the rule is applied again for that dataset.
- If the rule asks for more than 10, 10 seeds are run and the equivalence, non-inferiority and minimum-effect comparisons on that dataset are pre-classified as underpowered.
- Extra seeds come from the budget reserve (section 12) in the priority order MovieLens-1M, Video_Games, Steam, MovieLens-20M. Comparisons that cannot get their seeds within the cap are pre-classified as underpowered.

### 7.6 Robustness analyses

- Two-level bootstrap (10,000 resamples) over seeds within arm and over users, paired across arms by user, for every F1 and F2 difference.
- Every F1 and F2 analysis repeated on HR@10.
- RQ2 timing analyses repeated without runs flagged for thermal slowdown (section 8).

### 7.7 Reporting

Per arm, dataset and split: mean and SD of every metric across seeds, CIs for differences, test decisions, truncation rate, GPU hours. All runs are listed, including failed and excluded ones with reasons.

## 8. RQ2: cost of training objectives

- Every run logs per epoch: cumulative training time excluding evaluation, validation NDCG@10, peak allocated memory, and NVML samples at 1 Hz.
- Target quality per dataset: 0.95 × the highest mean final validation NDCG@10 among A1 to A5 under leave-one-out.
- Time to target of a run: the first cumulative training time at which validation NDCG@10 reaches the target. A run that never reaches it is censored at infinity.
- **H2a** holds on a dataset if A2 has the lowest median time to target, and for every other arm the 95% bootstrap CI (over seeds) of median(other)/median(A2) lies above 1.
- **H2b** is decided by family F2.
- Peak memory and step time of plain and chunked cross-entropy are reported per dataset; the two compute the same loss, so only cost differs.
- M3 runs are not timed under the controlled protocol of section 9. A run is flagged when NVML reports software or hardware thermal slowdown in more than 10% of its samples.

## 9. RQ3: efficiency measurement protocol

- **Workloads:** SASRec and BERT4Rec train steps and inference on synthetic batches with real shapes. Catalogue size N ∈ {3,416; 13,047; 25,600; 100,000}; length L ∈ {50, 100, 200, 400, 800}; dimension d ∈ {64, 128}; batch 128.
- **Factors:**
  - attention backend: math, memory-efficient, flash, cuDNN, where supported (flash accepts no explicit mask, so it is measured with `is_causal` only);
  - mask form: causal only, or causal plus padding mask;
  - precision: strict fp32, TF32, bf16 autocast;
  - `torch.compile`: off or on, where it works.
- **Measurements:**
  - median step time over 200 steps after at least 30 s of warm-up, timed with CUDA events;
  - peak allocated memory;
  - inference latency at batch 1 and batch 256 (full scoring plus top-10);
  - per-component time (encoder, output scoring, loss, backward, optimizer step) from CUDA events, and attention kernel share from the profiler.
- **Conditions:** native Windows, AC power, power mode recorded, vendor performance profile fixed and recorded, other applications closed. Each configuration is repeated 5 times in randomised order. Between repeats the GPU cools to within 5 °C of its idle temperature or for 120 s, whichever is first. NVML samples SM clock, memory clock, temperature, power and clock event reasons at 10 Hz.
- **Exclusion:** a repeat is excluded and rerun once if any sample shows software or hardware thermal slowdown or power brake, or if its median SM clock deviates more than 5% from the session median. Exclusions are logged.
- **Clock lock:** if `scripts/check_env.py` shows that clocks can be locked and the owner agrees, a second session runs with a locked SM clock; both sessions are reported.
- **Decisions:**
  - H3a holds if, for every configuration with L ≤ 200, the ratio of slowest to fastest fused backend (median step times) has a 95% bootstrap CI over repeats with upper bound below 1.10.
  - H3b holds if, for N ≥ 20,000, L = 200, d = 64, the share of step time spent in output scoring plus loss (forward and backward) has a 95% CI lower bound above 0.5.
  - H3c holds if the bf16 over strict-fp32 throughput ratio has a CI lower bound of at least 1.3 and family F3 shows equivalence. F3 uses the final A2 configuration with 5 seeds per precision on MovieLens-1M and Video_Games.

## 10. RQ4 (conditional, D015)

If approved at the end of M3: item embeddings from the final A1 to A4 runs are indexed with faiss (CPU; flat inner product as exact reference, HNSW, IVF-Flat). User vectors come from the same runs. Measures: recall@10 of the approximate top-10 against the exact top-10, change in NDCG@10, and single-query and batched latency. Index parameters are fixed before any recall is computed. A larger catalogue (Amazon Reviews 2023 Toys_and_Games or Yambda-50M) is added only if the remaining budget allows. All RQ4 analyses are exploratory.

## 11. Correctness safeguards (D016)

Silent errors such as leakage, misaligned labels or test-set tuning often improve metrics instead of crashing. These safeguards are delivery conditions for every milestone that touches data or evaluation.

### 11.1 Invariant assertions

Data move only through typed split objects. The assertions below run after preprocessing and at the start of every run; their results go into the run manifest.

- **Preprocessing:** every user and item meets the 5-core condition where it applies; no repeated (user, item) pairs; item ids are contiguous with 0 and N+1 reserved; timestamps never decrease within a user sequence.
- **Leave-one-out:** training sequence, validation target and test target concatenate to the full sequence with no overlap; timestamps are ordered train ≤ validation ≤ test; validation input equals the training sequence; test input equals the training sequence plus the validation item.
- **Temporal split:** every training interaction has t < T (validation: t < T_val); every test target has t ≥ T; every input contains only interactions before its target; the tie rule at T is asserted.
- **Training-only statistics:** item popularity (MostPop, negative sampling), the transition table and temporal-split filtering accept only the training split type. Each statistic stores a hash of the interactions it was computed from, compared at run time with the training split's hash.
- **Evaluation:**
  - each target equals the item that follows the input in the full sequence, checked by position;
  - each input is a prefix of the user's full sequence;
  - the score matrix covers all item ids, with padding and mask ids excluded;
  - after history masking, the candidate count is N minus the number of distinct history items, and the target's score is finite;
  - the target never appears in the history.
- **Training batches:** SASRec labels equal the inputs shifted by one position; training datasets are built from the training split only; a sampled check confirms no batch contains a validation or test position.

### 11.2 Known-answer tests (synthetic data)

- A planted rule (next item = a fixed permutation of the current item): SASRec and BERT4Rec reach test NDCG@10 ≥ 0.95.
- Test targets replaced by random items: metrics fall to the random level.
- Training sequence order shuffled on the planted-rule data: metrics fall to about the popularity baseline.
- Random scorer: E[HR@K] = K/C and E[NDCG@K] = (1/C) Σ_{r=1..K} 1/log₂(r + 1) for C candidates, matched by Monte Carlo within 4 standard errors.
- All-equal scores give NDCG near 0 under pessimistic ties; hand-computed small cases match.
- Causality: changing inputs after position t leaves SASRec outputs at positions ≤ t unchanged.

### 11.3 Mutation checks

Tests inject each fault below with monkeypatching (no fault switches in production code) and require at least one assertion or test to fail. The mutation-to-detector table lives next to the tests; any undetected mutation gets a new test.

- test item placed in the training sequence; validation item placed in the training sequence;
- labels off by one (target is the current item instead of the next);
- popularity or negative-sampling distribution computed on all data;
- early stopping or tuning reading the test split;
- history masking removed, or applied to the target;
- SASRec causal mask removed;
- temporal cutoff off by one at tied timestamps.

### 11.4 Independent metric implementation

A naive reference (per user, sort all scores, pure Python) and the ranx library (test dependency only) are compared with the vectorised implementation on several thousand random users. Agreement to 1e-6 is required. Inputs for the ranx comparison have no ties, because ranx breaks ties differently.

### 11.5 Anchors

Before M3, each arm is run on MovieLens-1M (leave-one-out, full ranking, history masked) with the hyperparameters of one published source and compared with that source's NDCG@10. The band is ±0.01 around the published value, to absorb preprocessing differences.

| Arm | Configuration and value from | Published NDCG@10 | Band |
|---|---|---|---|
| SASRec-CE | [5] | 0.1821 | 0.172 to 0.192 |
| SASRec-BCE | [5] | 0.1341 | 0.124 to 0.144 |
| BERT4Rec | [5] | 0.1537 | 0.144 to 0.164 |
| gSASRec | gSASRec PyTorch code (256 negatives, t = 0.75) | 0.1726 | 0.163 to 0.183 |

Other published full-ranking values on MovieLens-1M give context: SASRec-CE 0.169, BERT4Rec 0.161 and gSASRec 0.176 in [4]; BERT4Rec 0.1516 in [3].

A result above its band is treated first as leakage: all assertions and mutation tests are rerun and the data flow is inspected by hand. A result below its band is treated as an implementation problem. M3 does not start until every anchor is inside its band or the gap is explained in DECISIONS.

### 11.6 Test-set isolation

- One function reads test data. It requires `phase="final"` and appends a record (time, run id, caller, git commit, checkpoint hash) to the run's `test_access.jsonl`.
- The checkpoint hash is recorded before the test read. Tuning objectives and early stopping use validation data only.
- A test runs a full tuning trial with a test reader that raises on any call.
- The export step checks that every final run read the test data exactly once and every tuning run zero times; the counts appear in the report.

### 11.7 Silent-error review

Before delivering M1, M2, M3 or M5, an independent reviewer that sees only the code and a description of the data flow (no conclusions) traces every place where training, validation and test data are touched and marks each item of 11.1 to 11.6 as pass or fail. Failures are fixed before delivery. This is in addition to the whole-diff review.

## 12. Compute budget and descoping (D012)

| Block | Estimate (GPU h) |
|---|---|
| M2 pilot and anchors | 10 |
| Tuning (5 arms × 4 datasets) | 55 to 90 |
| Final runs, 5 seeds, two splits | 67 |
| Shuffled-order control | 6 |
| RQ3 | 15 |
| RQ4 (conditional) | 15 |
| Reserve for extra seeds (7.5) | remainder up to the cap |

The cap is 200 GPU hours. The M2 pilot replaces the estimates with measured epoch times. If the projected total exceeds the cap, scope is cut in this order: halve MovieLens-20M tuning trials, drop the MovieLens-20M temporal split, drop Steam, drop RQ4. Each cut is a DECISIONS entry.

## 13. Open items before the freeze

- Steam fields used as user and item keys, and handling of reviews without a date.
- BERT4Rec details from the official code [2]: token replacement scheme for masked positions and the additional last-item-masked training samples.
- Confirm the temporal split details against [12] (target choice, validation construction, treatment of out-of-vocabulary items).
- Licence of ranx, and whether it supports Python 3.12 on Windows.
- Validation subsample size for early stopping (10,000 users) once evaluation cost is measured.
