# Methods (pre-registration)

Status: **frozen** at the end of M1, before any model was trained on real data (D014; commit and SHA-256 in D024). Every later change is a deviation: it gets a DECISIONS entry with the date, the reason, and whether any result had been seen when it was made. Reference numbers point to the list in [SPEC.md](SPEC.md#12-references).

## 1. Data sources

| Dataset | File used | Fields |
|---|---|---|
| MovieLens-1M | `ml-1m.zip` from files.grouplens.org, `ratings.dat` | user, item, timestamp (s) |
| MovieLens-20M | `ml-20m.zip` from files.grouplens.org, `ratings.csv` | user, item, timestamp (s) |
| Steam | `steam_reviews.json.gz` from mcauleylab.ucsd.edu/public_datasets/data/steam/ | username, product_id, date (day) |
| Amazon Reviews 2023, Video_Games | 0-core `rating_only` file from the official benchmark release | user_id, parent_asin, timestamp (ms) |

Steam records lack a numeric user id in many lines, so `username` is the user key. Usernames are display names, so distinct accounts with the same name merge; this is a limitation shared with earlier uses of this file. Records without a username, product or date, and malformed lines, are dropped and counted. The download script records the SHA-256 of every file. Preprocessing is deterministic given those files. No data file, processed file or per-user output is committed.

## 2. Preprocessing

1. Read every interaction with its row index in the source file.
2. Treat every interaction as a positive event; ratings are ignored.
3. Sort by (user, timestamp, row index). The row index breaks timestamp ties, so order is stable and reproducible. "Before" and "after" within a user always refer to positions in this order.
4. Remove repeated (user, item) pairs, keeping the earliest. Evaluation masks the user's history (section 6), so a repeated item could never be a valid target.
5. Iterative 5-core filtering on users and items, repeated until stable. For leave-one-out it runs on the full deduplicated data, which uses information from the evaluation period; this is inherent to the published protocol and is reported as such. [1] describes a 5-core filter; its code applies a single pass on raw counts. For the temporal split it runs on the training period only (section 3.2).
6. Map items to 1..N in lexicographic order of their original id strings. 0 is padding; N+1 is the BERT4Rec mask token.
7. Maximum input length L: 200 for MovieLens-1M and MovieLens-20M, 50 for Steam and Amazon, as in [1] and [5]. Longer inputs keep the most recent L items.

Dataset statistics (users, items, interactions, sequence length quantiles, density, share of consecutive same-timestamp pairs) are written to `results/data/`. They are compared with published statistics; a difference above 1% in users, items or interactions is explained in DECISIONS before the freeze. For Amazon, our 5-core counts are also checked against the official 5-core table. Expected differences: Steam against [1] (single-pass filter), and Steam and MovieLens-20M against the BERT4Rec-repository versions used in [3, 4, 5] (user filter only).

## 3. Splits

### 3.1 Leave-one-out (primary)

For each user with sequence s_1..s_n: training sequence s_1..s_{n-2}; validation input s_1..s_{n-2} with target s_{n-1}; test input s_1..s_{n-1} with target s_n. Models are not retrained on validation data.

### 3.2 Global temporal split with last-item target (secondary)

Based on the GTS-Last variant in [12]:
- T is the 0.9 quantile of all deduplicated interaction timestamps. Interactions with t ≤ T form the training period and t > T the test period, as in the code of [12].
- Iterative 5-core filtering on the training period defines the item vocabulary.
- Test users: every user with at least one interaction after T whose input contains at least one vocabulary item. This includes users who first appear after T, as in [12].
- Target: the user's last interaction. Input: all of the user's earlier interactions from both periods, with out-of-vocabulary items removed. Targets outside the vocabulary stay in the evaluation and count as misses.
- Validation repeats the construction inside the training period with T_val, the 0.9 quantile of training-period timestamps ([12] uses a 0.9 validation quantile). Training sequences are each user's vocabulary interactions with t ≤ T_val, for users with at least two of them. The model early stops on the validation targets and is evaluated on the test targets without retraining.
- Hyperparameters are not tuned on this split; final runs reuse the configurations selected under leave-one-out (D007).

Intentional differences from [12]:
- Filtering uses the training period only; [12] filters the whole dataset before splitting.
- All repeated pairs are removed; [12] removes consecutive repeats only.
- No retraining on the full training period after validation; [12] recommends it. This saves GPU time. Interactions in (T_val, T] then serve as test inputs only.
- The vocabulary comes from data up to T, which includes the validation period. This can affect validation scores but not test scores.

### 3.3 Shuffled-order controls (D017)

- **Train-and-evaluate shuffle (arm C1, confirmatory):** each user's training sequence and every evaluation input are permuted uniformly at random. Every position of the full log gets one random key from a generator seeded by the run seed, and each sequence is reordered by its keys, so a training sequence and the validation input built from it are permuted the same way. Targets are unchanged. Runs use the final SASRec-CE configuration on all four datasets and the final BERT4Rec configuration on MovieLens-1M and MovieLens-20M, leave-one-out, with the same seeds as the unshuffled runs. Expected outcome: a drop larger than SESOI (family F4). Without it on the MovieLens datasets, order handling and evaluation are audited before any other result is reported.
- **Test-only shuffle (descriptive):** as in [11], the trained model is evaluated on shuffled test inputs. This is computed for every neural final run inside its single test read (section 11.6). [11] shuffles test inputs only; the train-and-evaluate variant is our adaptation.

## 4. Models and objectives

### 4.1 SASRec backbone (arms A1 to A4)

Following [1] and its code [31]:
- learned item and position embeddings, dropout on their sum;
- B blocks of pre-LayerNorm causal self-attention and a point-wise ReLU feed-forward network, with residual connections;
- a final LayerNorm (in the code [31], not the paper).

The score of item j at position i is the dot product of the hidden state h_i and an output embedding of j. Output embeddings are either shared with the input embeddings or separate; this is a tuned choice (section 5.3).

Inputs are left-padded with id 0, whose embedding is zero. The attention mask is causal and blocks padded keys; a padded query may attend to itself so no row is empty. Padded positions never enter a loss or a metric.

### 4.2 BERT4Rec (arm A5)

Following [2] and its code [32]:
- bidirectional post-LayerNorm Transformer blocks with GELU and learned position embeddings;
- an output head of dense, GELU and LayerNorm (the LayerNorm is in the code [32]), followed by a dot product with the shared item embeddings plus an item bias.

Training masks each position with probability ρ, with at least one masked position per sequence, and applies full softmax cross-entropy at masked positions. A masked position is always replaced by the mask token, as with the official default (`mask_prob = 1.0`). Each epoch uses one sequence per user (the most recent L training items) with fresh random masks. The official code instead precomputes ten masked copies and sliding windows; repeated epochs play the same role here. The official code creates last-item-masked instances only for its test file, so training adds none. The official code also appends the validation item to the training data; here it stays held out (3.1). At inference a mask token is appended after the input and scored.

### 4.3 Objectives

For position i with target y_i and score vector s_i:

- **Full cross-entropy (A2):** −log softmax(s_i)[y_i] over items 1..N.
- **Chunked cross-entropy:** the same loss computed over slices of positions so the full logit tensor never exists at once. Used only as an implementation variant in RQ2 and RQ3.
- **Sampled softmax (A3):** K negatives drawn uniformly from 1..N once per batch and shared by all positions. Logits are [s_{i,y_i}, s_{i,n_1}, ..., s_{i,n_K}]; a negative equal to y_i is masked for that position. The uniform proposal makes the log-Q correction constant, so none is applied.
- **BCE (A1):** one negative per position, drawn uniformly from items not in the user's training sequence. [1] excludes every item the user rated, which includes validation and test items. Loss −log σ(s⁺) − log(1 − σ(s⁻)).
- **gBCE (A4):** K negatives drawn uniformly with replacement, negatives equal to the positive masked. Loss −β log σ(s⁺) − Σ_k log(1 − σ(s⁻_k)) with α = K/(N − 1) and β = α(t(1 − 1/α) + 1/α) [4]. Computed in fp32.

### 4.4 Floor baselines

- **MostPop (B1):** items ranked by interaction count in the training data.
- **Transition (B2):** score of j is the count of consecutive pairs (last input item, j) in training sequences; ties broken by training popularity.

Both use the same evaluation code, masking and test-access rules as the neural models.

## 5. Training and tuning

### 5.1 Training loop

- One epoch is one pass over all training users in random order, one sequence per user (the most recent L+1 training items for SASRec arms, the most recent L for BERT4Rec).
- Adam (β₁ 0.9, β₂ 0.999), batch size 128, gradient-norm clipping at 5.
- Training uses bf16 autocast in all M3 runs, with TF32 allowed for fp32 matmuls. H3c tests whether bf16 changes quality.
- Evaluation scores are always computed in fp32 with autocast disabled (section 6).
- Seeds fix Python, NumPy and PyTorch generators and the data order. Fully deterministic kernels are not enforced; their cost is measured instead of assumed.

### 5.2 Early stopping and caps

- Validation NDCG@10 is computed after every epoch. The best checkpoint by validation NDCG@10 is restored at the end.
- Every validation user is scored after every epoch (D019). The trial objective is validation NDCG@10 of the restored checkpoint.
- Caps leave a margin over published training lengths ([5] uses at most 100 epochs with patience 10 for SASRec and 200 with patience 20 for BERT4Rec):

| Arms | Max epochs | Patience | Wall-clock guard |
|---|---|---|---|
| A1 to A4 | 300 | 30 | 8 h |
| A5 | 1000 | 100 | 8 h |

- A run that ends by the epoch cap or the guard instead of by patience is marked truncated. Every run records epochs trained, the best epoch and whether it was truncated.
- If more than 20% of an arm's tuning trials on a dataset are truncated, that arm's final runs on that dataset double all three limits. Truncation rates are reported for every arm and dataset.

### 5.3 Search spaces

Shared by all neural arms:
- learning rate log-uniform [1e-4, 1e-2];
- embedding size {64, 128, 256};
- blocks {1, 2, 3};
- heads {1, 2};
- dropout uniform [0.1, 0.6];
- weight decay {0, 1e-4, 1e-2}.

| Arm | Additional dimensions |
|---|---|
| A1 SASRec-BCE | output embeddings {shared, separate} |
| A2 SASRec-CE | output embeddings {shared, separate} |
| A3 SASRec-SS | output embeddings {shared, separate}; K ∈ {64, 256, 1024, 4096} |
| A4 gSASRec | output embeddings {shared, separate}; K ∈ {64, 128, 256}; t uniform [0.5, 1.0] |
| A5 BERT4Rec | ρ uniform [0.1, 0.4] |

### 5.4 Tuning procedure

- One Optuna study per (arm, dataset), leave-one-out only, stored in SQLite so tuning resumes after interruption.
- TPE sampler with a fixed seed; the first 5 trials are random.
- Median pruner after 5 complete trials, with warm-up equal to 10% of the arm's epoch cap.
- Trial budget per study: 20 (MovieLens-20M: 10). All trials use training seed 0.
- The objective is full validation NDCG@10 of the restored checkpoint.

### 5.5 Final runs

The best trial per (arm, dataset) is rerun with seeds 1..n on leave-one-out, where n follows section 7.5, and with seeds 1..5 on the temporal split. Test data are read by a separate command (`seqrec test`), which refuses to run until every run of its plan has finished training; seeds added by 7.5 go into the plan before that command runs. Floor baselines are deterministic and run once.

The final runs of a dataset are executed in a randomised order that interleaves arms and seeds. Each batch starts with 180 s of continuous GPU load, so runs start in the thermal steady state rather than from a cool GPU (D011).

## 6. Evaluation

- Scores for every item 1..N are computed in fp32 with autocast disabled.
- Every item the user interacted with before the target is set to −∞, including items older than the truncated input. The padding id and mask id are also set to −∞.
- Rank of the target: r = 1 + #{j ≠ y : s_j ≥ s_y} over non-masked items, so ties count against the model. Each run records its tie rate: the share of evaluation users whose target ties with at least one candidate.
- Single target per user: NDCG@K = 1/log₂(r + 1) if r ≤ K, else 0; HR@K = 1 if r ≤ K, else 0. Metrics are averaged over evaluation users.
- Primary metric: NDCG@10. Secondary, descriptive only: HR@10, NDCG@20, HR@20.
- Model selection, early stopping and tuning see validation data only. Test data are read through one logged function (section 11.6).

## 7. Statistical analysis

### 7.1 Unit and scale

The unit of analysis is one training run. For run r, y_r = log(test NDCG@10 averaged over users). Differences between arms are differences in mean y across seeds, so exp(Δ) − 1 is a relative difference. User-level paired tests are not used for inference because they ignore run-to-run variation [17].

### 7.2 Smallest effect size of interest

δ = log(1.02), a 2% relative difference in NDCG@10.

### 7.3 Tests

All tests are Welch t-tests on y between two arms on the same dataset and split, at α = 0.05 after Holm correction within the family (7.4).

- **Directional:** two-sided test of equal means.
- **Non-inferiority (H1a):** one-sided test of H0: μ_A − μ_B ≤ −δ.
- **Minimum effect (H1b, F4):** one-sided test of H0: μ_A − μ_B ≤ δ, where A is the arm expected to be better.
- **Equivalence (TOST):** two one-sided tests against −δ and +δ. The TOST p-value is the larger of the two, and Holm is applied to it.

Published gaps and the sign of A − B reported by their source (+: A ahead; −: B ahead; none: not reported):

| Gap | A − B | Source | MovieLens-1M | Steam | MovieLens-20M | Video_Games |
|---|---|---|---|---|---|---|
| G1 | A5 − A1 | [2] (sampled metrics) | + | + | + | none |
| G2 | A4 − A5 | [4] | + | − | none | none |
| G3 | A2 − A5 | [5] | + | − | + | none |

Each (gap, dataset) pair with a published sign gets the first class that applies, using the F1 directional test and the F5 TOST:
1. reversed below SESOI: directional significant against the published sign, and TOST significant;
2. reversed: directional significant against the published sign;
3. confirmed below SESOI: directional significant with the published sign, and TOST significant;
4. confirmed: directional significant with the published sign;
5. equivalent: TOST significant;
6. inconclusive.

### 7.4 Families and multiplicity

| Family | Tests | Count |
|---|---|---|
| F1 RQ1 directional | 6 arm pairs among A1, A2, A4, A5 × 4 datasets | 24 |
| F1-NI (H1a) | A2 vs A5, non-inferiority, MovieLens-1M and MovieLens-20M | 2 |
| F1-ME (H1b) | A2, A4 and A5 each vs A1, minimum effect × 4 datasets | 12 |
| F5 published gaps | TOST for each (gap, dataset) pair with a published sign | 8 |
| F2 (H2b) | A3 vs A2 and A4 vs A2, TOST × 4 datasets | 8 |
| F6 (H2a) | A1, A3 and A4 each vs A2, time to target × 4 datasets | 12 |
| F3 (H3c) | bf16 vs strict fp32, TOST, MovieLens-1M and Video_Games | 2 |
| F4 shuffle | A2 vs C1-A2 × 4 datasets, A5 vs C1-A5 × 2 datasets, minimum effect | 6 |

- H1a and H1b are decided per dataset. H1b holds on a dataset when its three F1-ME tests reject. The report gives the count of datasets supporting each.
- The BCE clause of H2b is decided by the F1-ME test of A2 vs A1.
- Temporal-split results use the same tests, reported as secondary without confirmatory claims. Agreement with leave-one-out is summarised by Kendall's τ between arm rankings per dataset.

### 7.5 Number of seeds (D018)

The rule is fixed now; the M2 pilot only supplies the input σ.

- σ is the pooled standard deviation of y_val (log validation NDCG@10) across seeds within arms, from the pilot: 3 seeds per arm A1 to A5 on MovieLens-1M, leave-one-out. The pilot reads no test data.
- With SE = σ√(2/n) and df = 2n − 2, two power functions at the SESOI:
  - directional: P(|T| > t_{1−α'/2, df}) with T noncentral t (df, δ/SE) and α' = 0.05/24, the strictest Holm step in F1;
  - equivalence: max(0, 2·F_t(δ/SE − t_{0.95, df}; df) − 1), the TOST power when the true difference is 0.
- n is the smallest integer from 5 to 10 at which both powers reach 0.8. If none does, n = 10.
- Examples: σ = 0.006 gives n = 6; σ = 0.0075 gives n = 7; σ = 0.01 gives directional power 0.78 at n = 10, so n = 10.
- After the first 5 final seeds per arm on each dataset, σ is re-estimated from their validation values and the rule is applied again for that dataset. Test data are not read before this (5.5).
- Extra seeds apply to leave-one-out only. They come from the budget reserve (section 12) in the priority order MovieLens-1M, Video_Games, Steam, MovieLens-20M.
- Every comparison reports its power at δ, computed with its final n and σ. A comparison with power below 0.8 is labelled underpowered. A non-rejection is never read as absence of an effect; rejections count normally.

### 7.6 Robustness analyses

- Two-level percentile bootstrap (10,000 resamples) over seeds within arm and over users, paired across arms by user, for every F1 and F2 difference. Bootstrap intervals are descriptive; the tests in 7.3 decide. With 5 seeds the seed-level resampling understates uncertainty, and the report says so.
- Every F1 and F2 analysis repeated on HR@10.
- RQ2 timing analyses repeated without runs flagged for thermal slowdown (section 8).

### 7.7 Reporting

Per arm, dataset and split: mean and SD of every metric across seeds, CIs for differences, test decisions, power, truncation rate, tie rate and GPU hours. All runs are listed, including failed and excluded ones with reasons.

## 8. RQ2: cost of training objectives

- Every run logs per epoch: cumulative training time excluding evaluation, validation NDCG@10, peak allocated memory during the training epoch (read before evaluation starts), and NVML samples at 1 Hz.
- Scope: arms A1 to A4, leave-one-out.
- Target quality per dataset: 0.95 × the highest mean (over seeds) best validation NDCG@10 among A1 to A4.
- Time to target of a run: the first cumulative training time at which validation NDCG@10 reaches the target. A run that never reaches it is censored at +∞.
- **H2a:** for each arm X in {A1, A3, A4} and each dataset, a one-sided Mann-Whitney U test that A2's times to target are stochastically smaller than X's (censored values rank last, ties get mid-ranks), in family F6. H2a holds on a dataset when all three tests reject. Medians and the ratio of medians are reported descriptively. With 5 runs per arm the smallest attainable one-sided p is 1/252, so only complete separation passes the strictest Holm step.
- **H2b** is decided by family F2 and, for BCE, by F1-ME.
- Peak memory and step time of plain and chunked cross-entropy are reported per dataset; the two compute the same loss, so only cost differs.
- M3 runs are not timed under the controlled protocol of section 9. Software thermal slowdown and power capping are this GPU's normal state under load (D011), so they do not flag a run. A run is flagged when NVML reports a hardware slowdown, hardware thermal slowdown or power brake event, or when its median SM clock is more than 5% below the median over all M3 runs.

## 9. RQ3: efficiency measurement protocol

### 9.1 Workloads and factors

- **Workloads:** SASRec and BERT4Rec train steps and inference on synthetic batches with real shapes. Catalogue size N ∈ {3,416; 13,047; 18,345; 25,600; 100,000}; length L ∈ {50, 100, 200, 400, 800}; dimension d ∈ {64, 128}; batch 128.
- **Factors:**
  - attention backend: math, memory-efficient, flash, cuDNN, where supported (flash accepts no explicit mask, so it is measured with `is_causal` only);
  - mask form: causal only, or causal plus padding mask;
  - precision: strict fp32 (TF32 off), TF32, bf16 autocast;
  - `torch.compile`: off or on, where it works.
- Plain full cross-entropy at N = 100,000 with batch 128 and L = 200 needs about 29 GiB for logits alone, so chunked cross-entropy is used wherever plain cross-entropy does not fit.

### 9.2 Measurements and conditions

- **Measurements:**
  - median step time over 200 steps after the regime's warm-up, timed with CUDA events;
  - peak allocated memory;
  - inference latency at batch 1 and batch 256 (full scoring plus top-10);
  - per-component time (encoder, output scoring, loss, backward, optimizer step) from CUDA events, and attention kernel share from the profiler.
- **Conditions:** native Windows, AC power, power mode recorded, vendor performance profile fixed and recorded, other applications closed. NVML samples SM clock, memory clock, temperature, power and clock event reasons at 10 Hz. Each configuration is repeated 5 times in randomised order, in two regimes:
  - steady state (primary; all decision rules use it): a session starts with at least 180 s of continuous load, and configurations then run back to back without cooling. On this GPU the clock settles about 100 s into sustained load (D011);
  - cool start (secondary, descriptive): before each repeat the GPU cools to within 5 °C of its idle temperature, then 5 s of warm-up and a measurement of at most 25 s follow, which stays inside the roughly 40 s before thermal slowdown begins.
- **Exclusion:** a repeat is excluded and rerun once if any sample shows a hardware slowdown, hardware thermal slowdown or power brake event, or if its median SM clock deviates more than 5% from the median of its regime in that session. In the cool-start regime, a repeat is also excluded if software thermal slowdown appears. Exclusions are logged.
- **Clock lock:** at the start of M4, locking the SM clock (`nvidia-smi -lgc`, administrator rights) is tried by hand. If it works, a second session runs with a locked clock, and both sessions are reported.

### 9.3 Decision rules

All intervals are percentile bootstraps with 10,000 resamples over the 5 repeats, resampled independently per condition.
- **H3a:**
  - Workload: SASRec train step, full cross-entropy, bf16 autocast, causal-only mask, 2 blocks, 1 head, batch 128.
  - Configurations: L ∈ {50, 100, 200}, d ∈ {64, 128} and N ∈ {3,416; 25,600}, 12 in total.
  - For each configuration, R is the ratio of the slowest to the fastest median step time among the available fused backends (memory-efficient, flash, cuDNN). The backends are re-selected inside every resample.
  - H3a holds if the upper bound of R's interval is below 1.10 in all 12 configurations.
- **H3b:**
  - Workload: SASRec train step, full cross-entropy (plain at N = 25,600, chunked at N = 100,000), bf16 autocast, default backend, causal plus padding mask, 2 blocks, 1 head, d = 64, L = 200, batch 128.
  - Share = (T_full − T_proxy)/T_full. T_proxy is the same step with output scoring and loss replaced by the sum of the final hidden states times a fixed random vector.
  - H3b holds if the interval's lower bound exceeds 0.5 for both N.
- **H3c:**
  - Throughput: the H3b workload at N = 25,600; ratio = median step time in strict fp32 / median step time in bf16.
  - Quality: family F3, comparing test NDCG@10 of the final A2 configuration with 5 seeds per precision. The bf16 runs are the M3 runs with seeds 1 to 5; strict-fp32 runs are added.
  - H3c holds if the throughput ratio's lower bound is at least 1.3 and F3 shows equivalence.

## 10. RQ4 (conditional, D015)

If D015 selects RQ4:
- Indexing: item embeddings from the final A1 to A4 runs go into faiss on CPU: flat inner product as the exact reference, HNSW and IVF-Flat. User vectors come from the same runs.
- Measures: recall@10 of the approximate top-10 against the exact top-10, change in NDCG@10, and single-query and batched latency.
- Index parameters are fixed before any recall is computed.
- A larger catalogue (Amazon Reviews 2023 Toys_and_Games or Yambda-50M) is added only if the remaining budget allows.
- All RQ4 analyses are exploratory.

## 11. Correctness safeguards (D016)

Silent errors such as leakage, misaligned labels or test-set tuning often improve metrics instead of crashing. These safeguards are delivery conditions for every milestone that touches data or evaluation.

### 11.1 Invariant assertions

Data move only through typed split objects. The assertions below run after preprocessing and at the start of every run; their results go into the run manifest.

- **Preprocessing:** every user and item meets the 5-core condition where it applies; no repeated (user, item) pairs; item ids are contiguous with 0 and N+1 reserved; timestamps never decrease within a user sequence.
- **Leave-one-out:** training sequence, validation target and test target concatenate to the full sequence with no overlap; timestamps are ordered train ≤ validation ≤ test; validation input equals the training sequence; test input equals the training sequence plus the validation item.
- **Temporal split:**
  - every training interaction has t ≤ T_val, and every validation target t ≤ T;
  - every test target has t > T;
  - every input contains only interactions at earlier positions than its target.
- **Training-only statistics:** item popularity (MostPop), negative sampling and the transition table accept only the training split type. Each statistic stores a hash of the interactions it was computed from, compared at run time with the training split's hash. Model factories receive the training split only.
- **Temporal-split construction:** the checks recompute both cutoffs, the training-period 5-core vocabulary and each user's last validation-period item from the source log with their own code and compare them with the split.
- **Evaluation:**
  - each target equals the item that follows the input in the full sequence, checked by position;
  - each input holds exactly the user's in-vocabulary items before the target, each once;
  - the score matrix is fp32 and covers all item ids, with padding and mask ids excluded;
  - after masking, the candidate count is N minus the number of distinct items in the user's full history before the target, and the target's score is finite;
  - the target never appears in that history.
- **Training batches:** SASRec labels equal the inputs shifted by one position; training datasets are built from the training split only; a sampled check confirms that no batch contains a validation or test position.

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
- BCE negatives excluding the user's full history, which includes the test item;
- early stopping or tuning reading the test split;
- history masking removed, limited to the truncated input, or applied to the target;
- evaluation scores computed in bf16;
- SASRec causal mask removed;
- temporal cutoff off by one at tied timestamps.

### 11.4 Independent metric implementation

A naive reference (per user, sort all scores, pure Python) and the ranx library (MIT licence, optional `crosscheck` dependency group, D021) are compared with the vectorised implementation on several thousand random users. Agreement to 1e-6 is required. Inputs for the ranx comparison have no ties, because ranx breaks ties differently.

### 11.5 Anchors

Before M3, each arm is run on MovieLens-1M (leave-one-out, full ranking, history masked) with the hyperparameters of one published source, and compared with that source's NDCG@10. The band is ±0.01 around the published value, to absorb preprocessing differences.

| Arm | Configuration and value from | Published NDCG@10 | Band |
|---|---|---|---|
| SASRec-CE | [5] | 0.1821 | 0.172 to 0.192 |
| SASRec-BCE | [5] | 0.1341 | 0.124 to 0.144 |
| BERT4Rec | [5] | 0.1537 | 0.144 to 0.164 |
| gSASRec | ML-1M config of [33]: separate output embeddings, embedding 128, 2 blocks, 1 head, dropout 0.5, 256 negatives, t = 0.75 | 0.1726 | 0.163 to 0.183 |

Other published full-ranking values on MovieLens-1M give context: SASRec-CE 0.169, BERT4Rec 0.161 and gSASRec 0.176 in [4]; BERT4Rec 0.1516 in [3].

Rules:
- Anchors compare against test values, so their reads use the `anchor` phase (11.6). Each arm gets at most three anchor reads.
- A result above its band is treated first as leakage: all assertions and mutation tests are rerun and the data flow is inspected by hand. A result below its band is treated as an implementation problem.
- Any code change made after an anchor read must come with a test that fails before the change, and is logged in DECISIONS.
- If an arm is still outside its band after three reads, it enters M3 flagged "anchor not reproduced". The gap is recorded in DECISIONS, and the flag is carried into every comparison involving that arm.
- Anchor reads touch the MovieLens-1M test targets that the main analysis also uses. The report states this, and every RQ1 conclusion on MovieLens-1M carries the note.

### 11.6 Test-set isolation

- The test split is sealed. Only the logged reader opens it for scores; the invariant checks and the shuffle transform open it for structure, and a static test fails if any other code does. Tuning runs replace it with a stand-in that raises on any access.
- The reader requires a phase, `final` or `anchor`, and appends a record (time, run id, phase, caller, git commit, checkpoint hash) to the run's `test_access.jsonl`; a second record marks that the scores were written. Anchor reads are counted across all runs before data are returned, and a fourth read for the same arm and dataset is refused.
- The checkpoint hash is recorded before the test read. Tuning objectives and early stopping use validation data only.
- The single read of a final run computes every reported test quantity, including the test-only shuffle (3.3).
- A test makes a tuning run try to evaluate on test data and requires it to fail.
- The export step recounts reads from every `test_access.jsonl`, including runs that failed, and the report shows the counts:
  - every final run read test data at most once, and every tested run has a completion record for each read;
  - every tuning run read it zero times;
  - each arm has at most three anchor reads per dataset.

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
| **Core total** | **153 to 188** |
| Reserve: extra seeds (7.5), then RQ4 (about 15) | 12 to 47 |

- The cap is 200 GPU hours.
- One extra leave-one-out seed for every arm on every dataset costs about 7 hours, so the reserve covers one to six extra seeds before RQ4. If the cap binds, datasets late in the priority order of 7.5 keep fewer seeds, and their comparisons are labelled by power as in 7.5.
- The M2 pilot replaces these estimates with measured epoch times. If the projected core total exceeds the cap, scope is cut in this order: halve MovieLens-20M tuning trials, drop the MovieLens-20M temporal split, drop Steam. Each cut is a DECISIONS entry.

## 13. Items resolved before the freeze

Steam fields (section 1); BERT4Rec masking and training instances (4.2); ranx licence (11.4, D021); validation subsample removed (5.2, D019); dataset statistics against published values (D022).
