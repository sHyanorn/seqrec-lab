# Decisions

Numbered decision records (range D001 to D099). Reference numbers point to [SPEC.md](SPEC.md#12-references). A decision changed later keeps its entry and gains a dated amendment.

## D001 Research questions

- Date: 2026-11-08. Status: accepted.
- Decision: RQ1 (robustness of published gaps), RQ2 (cost and quality of training objectives) and RQ3 (where training time goes) are the core questions. RQ4 (approximate nearest-neighbour retrieval on learned item embeddings) is conditional (D015). Hypotheses are in SPEC section 3; decision rules are in METHODS sections 7 to 10.
- Rationale: the direction of the SASRec and BERT4Rec gap and the quality ordering of losses are already reported [3, 4, 5, 13]. What those studies leave open is effect size with run-to-run uncertainty, wall-clock cost on consumer hardware, and a time breakdown at SASRec scale.
- Alternatives: a plain reproduction study (adds little beyond [3, 5]); a new model (outside the project's purpose).

## D002 Native Windows for GPU runs

- Date: 2026-11-08. Status: accepted.
- Decision: all GPU experiments run on native Windows 11. WSL2 is a fallback only for `torch.compile` or FlexAttention work. Timings from the two systems are never mixed in one comparison.
- Rationale: PyTorch 2.15 Windows CUDA wheels are built for `sm_120` [30] and, per the release notes, include the FlashAttention backend [22]. NVML in WSL2 does not support all queries [26]. Clock-lock commands have no effect inside WSL2 [27]. Per-submission launch overhead is higher under WSL2 [28], and small models launch many short kernels.
- Alternatives: WSL2 for everything, which brings official Triton and `torch.compile` but less reliable telemetry and timing.
- Amendment 2026-11-08: confirmed by the environment check (`results/env/check_env_nvidia-geforce-rtx-5080-laptop-gpu_20261108T221127Z.json`). `torch.compile` works natively with triton-windows (speedup 1.075 on the synthetic train step), so no planned work needs WSL2.

## D003 Software stack

- Date: 2026-11-08. Status: accepted.
- Decision: Python 3.12; PyTorch 2.15.x from the `cu132` index on Windows and the CPU index on Linux (development container); NVIDIA driver branch 580 or newer; `triton-windows` 3.9.x as an optional dependency group for `torch.compile`; `nvidia-ml-py` for NVML; uv with a committed `uv.lock`. Later milestones add NumPy, pandas, PyArrow and Optuna; `faiss-cpu` only if RQ4 runs.
- Rationale: PyTorch 2.15 requires Python 3.11 or newer [21]. Its `cu132` Windows wheels exist for Python 3.11 to 3.15 and are built for `sm_120` [30]; 2.15 ships no `cu126` wheels [22]. Python 3.12 is chosen for broad wheel coverage across later dependencies. Official guidance on `torch.compile` for Windows covers CPU and XPU only [23]; GPU compilation relies on the community Triton build [24], so no conclusion depends on it.
- Alternatives: conda (slower, no cross-platform lock); `cu134` wheels (no benefit for this GPU found).
- Amendment 2026-11-08: the environment check confirms Python 3.12.10, torch 2.15.0+cu132 with `sm_120` in its architecture list, cuDNN 9.26, triton-windows 3.9.0.post30, driver 621.01 and bf16 support. On this GPU:
  - math and memory-efficient attention run for every tested dtype, mask and head dimension;
  - flash runs for fp16 and bf16 without an explicit mask;
  - cuDNN runs for fp16 and bf16 up to head dimension 128;
  - all outputs and gradients match the fp32 math backend within tolerance;
  - the default dispatch picks memory-efficient attention whenever a padding mask is present.

## D004 Own implementation of models, losses and evaluation

- Date: 2026-11-08. Status: accepted.
- Decision: models, objectives, evaluation and data pipeline are implemented here in plain PyTorch. Reference repositories are used for protocols and sanity values only. Code from repositories without a licence or under GPL is never copied. MIT or Apache-2.0 code, if copied, keeps its notice and is listed in `THIRD_PARTY_NOTICES`.
- Rationale:
  - RecBole trains SASRec on every prefix, defaults to length 50 and does not mask seen items. In [3], its BERT4Rec had the lowest full-ranking NDCG@10 on MovieLens-1M and stayed far below the best implementation on every dataset.
  - Several PyTorch ports have issues: negatives not masked against the positive (gSASRec PyTorch, RecTools); checkpoints selected on test metrics and no padding mask (SASRec.pytorch).
  - The code of [3] and [5] carries no licence.
- Alternatives: depend on RecBole or RecTools (protocol mismatches above, heavier dependencies).

## D005 Datasets

- Date: 2026-11-08. Status: accepted.
- Decision: MovieLens-1M, Steam, Amazon Reviews 2023 Video_Games and MovieLens-20M. Yelp is excluded. The RQ4 large catalogue is chosen with D015.
- Rationale:
  - MovieLens-1M anchors comparisons with all prior work.
  - Steam is the dataset where BERT4Rec stayed ahead in [4] and [5], and it has weak sequential structure [11].
  - Video_Games adds recent data with short sequences and has the largest catalogue in the core set (25.6k items).
  - MovieLens-20M has strong sequential structure [11] and the most interactions; after 5-core filtering it has about 18.3k items [12].
  - Yelp's terms require Yelp to review findings before publication, forbid sharing, and expire after 12 months.
- Licence notes: MovieLens forbids redistribution and requires citing [18]; Steam and Amazon Reviews 2023 publish no licence and ask for citation [1, 19]. Only download and preprocessing code is committed.

## D006 Preprocessing

- Date: 2026-11-08. Status: accepted.
- Decision: implicit feedback; stable sort by (user, timestamp, source row); keep the earliest of repeated (user, item) pairs; iterative 5-core on users and items for leave-one-out; the same filter on training-period data for the temporal split. Our statistics are compared with published ones, and any difference above 1% in users, items or interactions is explained in this log before the METHODS freeze. Details: METHODS section 2.
- Rationale: a single procedure across datasets. Stable ordering matters because MovieLens and Steam have many tied timestamps. Differences against published statistics are expected where sources filter in a single pass [1] or filter users only [2, 3, 4, 5].

## D007 Splits

- Date: 2026-11-08. Status: accepted.
- Decision: leave-one-out is primary. The secondary split is the global temporal split with last-item target of [12], with three deviations: vocabulary filtering on the training period only, removal of all repeated pairs, and no retraining after validation. Tuning runs on leave-one-out only, and temporal-split final runs reuse those configurations. Details: METHODS section 3.
- Rationale: leave-one-out keeps comparability with published numbers. Temporal splits are more realistic and change model rankings [7, 8, 12]. Filtering on the training period avoids using post-cutoff data. Tuning twice or retraining would cost tens of extra GPU hours.

## D008 Evaluation protocol

- Date: 2026-11-08. Status: accepted.
- Decision: full ranking over all items with scores in fp32; the user's whole history masked, not only the truncated input; ties counted against the model, with the tie rate recorded; primary metric NDCG@10. Checkpoints and hyperparameters are selected on validation data only. Test data are read once per final run through one logged function. Details: METHODS sections 6 and 11.6.
- Rationale: sampled metrics can reorder models [6, 9, 10]. History masking matches the sources of the anchor values (METHODS 11.5). bf16 scores would create ties that depend on each model's logit scale.

## D009 Tuning protocol

- Date: 2026-11-08. Status: accepted.
- Decision:
  - One Optuna TPE study per arm and dataset, with the same trial count for every arm within a dataset: 20 trials, 10 on MovieLens-20M. Median pruning.
  - Early stopping on validation NDCG@10 with patience.
  - Epoch caps with a margin over published training lengths: 300 epochs and patience 30 for SASRec arms; 1000 and 100 for BERT4Rec; an 8-hour guard per run.
  - Every run records epochs trained and whether a limit ended it. Arms with more than 20% truncated trials get all limits doubled in final runs, and truncation rates are reported.
  - SASRec arms tune shared or separate output embeddings, since gSASRec's reference configuration uses separate ones [33].
  - Details: METHODS section 5.
- Rationale: BERT4Rec quality depends strongly on training length [3], so caps must not decide the comparison. A 40-trial budget as in [29] does not fit the GPU cap.
- Alternatives: equal wall-clock budget per arm (would favour fast arms and confound RQ1 with RQ2).

## D010 Statistical analysis

- Date: 2026-11-08. Status: accepted.
- Decision:
  - The unit is the training run.
  - Welch t-tests on log NDCG@10, with Holm correction within each family.
  - SESOI is a 2% relative difference.
  - Equivalence by TOST; non-inferiority and minimum-effect tests for H1a, H1b and the shuffle control.
  - Published gaps classified by an ordered rule against per-dataset published signs.
  - Time to target (H2a) compared with one-sided Mann-Whitney tests, which handle runs that never reach the target.
  - A two-level bootstrap over seeds and users serves as a descriptive robustness check.
  - Details: METHODS section 7.
- Rationale: user-level paired tests are overconfident for SASRec comparisons, while tests across runs keep their nominal error rate [17]. The same source reports that 1 to 2% differences can change winner with the seed, which motivates the 2% margin.

## D011 Efficiency measurement protocol

- Date: 2026-11-08. Status: accepted.
- Decision: native Windows on AC power with the power mode and vendor profile recorded; at least 30 s warm-up; CUDA-event timing; 5 randomised repeats; cool-down between repeats; NVML at 10 Hz; repeats with thermal slowdown or a clock deviation above 5% are excluded and rerun once. A locked-clock session is added only if locking works when tried by hand at the start of M4. Details: METHODS section 9.
- Rationale: laptop GPUs change clocks with temperature and power, which would otherwise dominate small timing differences.
- Amendment 2026-11-08 (from the environment check, before any experiment):
  - Observed behaviour: under sustained bf16 load the GPU runs at its power cap for about 40 s (SM clock about 2,370 MHz, 170 W). It then reaches 86 to 87 °C and software thermal slowdown takes over. The SM clock settles near 2,055 MHz (137 W) after about 100 s, and throughput drops to 85% of its start.
  - Consequence: software thermal slowdown and power capping are the normal operating state and cannot be exclusion criteria.
  - Protocol change: steady-state measurement after at least 180 s of continuous load is primary; short measurements from a cool GPU are secondary. Exclusion and flagging rules use hardware slowdown events and SM clock deviation instead of software thermal events. M3 runs are interleaved across arms after a warm-up so that thermal state does not differ by arm. Details: METHODS sections 8 and 9.2.
  - Clock locking: application-clock queries are not supported on this GPU, but 389 graphics clock levels are reported. Locking is still tried by hand in M4.

## D012 GPU budget

- Date: 2026-11-08. Status: accepted.
- Decision: at most 200 GPU hours in total. Core work is estimated at 153 to 188 hours; the rest is a reserve used first for extra seeds (D018), then for RQ4. Cuts if the projected core exceeds the cap, in order: halve MovieLens-20M tuning trials, drop the MovieLens-20M temporal split, drop Steam.
- Rationale: estimates before measurement (METHODS section 12). About 50 laptop hours per week are available.

## D013 Execution model

- Date: 2026-11-08. Status: accepted.
- Decision: the owner runs one command per batch of runs on the laptop. The runner resumes after interruption and writes configs, metrics, logs and NVML traces to files. An export command writes aggregated CSV files, which the owner pushes. The environment check report (`results/env/`) is committed. Data, checkpoints and per-user outputs stay local. Reports never contain user names, host names or file paths.
- Rationale: keeps the owner's time per batch to a few minutes, and makes every number traceable to committed files.

## D014 Pre-registration timing

- Date: 2026-11-08. Status: accepted.
- Decision: METHODS is frozen at the end of M1, after data preprocessing and before any model is trained on real data. The M2 pilot serves only to verify the implementation (anchors), measure time, and estimate the seed standard deviation for D018 from validation data. Later changes are logged here as deviations.
- Rationale: the analysis plan must not depend on observed results.

## D015 RQ4 decision point

- Date: 2026-11-08. Status: accepted.
- Decision: whether RQ4 runs, and on which large catalogue (candidates: Amazon Reviews 2023 Toys_and_Games, Yambda-50M), is decided at the end of M3 from the remaining GPU budget.
- Rationale: exact search is already cheap for catalogues below about 100k items, so RQ4 needs a larger catalogue to say anything practical, and that costs extra training.

## D016 Correctness safeguards

- Date: 2026-11-08. Status: accepted.
- Decision: METHODS section 11 is part of the protocol: always-on invariant assertions, known-answer tests on synthetic data, mutation checks, an independent metric implementation, anchors on MovieLens-1M, isolated test access, and a separate silent-error review before every milestone that touches data or evaluation. Anchor comparisons need test reads; they use a separate logged phase, at most three reads per arm, and any code change after a read needs a failing test first.
- Rationale: leakage, label misalignment and test-set tuning usually raise metrics instead of failing loudly.

## D017 Shuffled-order control

- Date: 2026-11-08. Status: accepted.
- Decision: two controls.
  - Train-and-evaluate shuffle (confirmatory): final SASRec-CE runs on all four datasets and final BERT4Rec runs on MovieLens-1M and MovieLens-20M are repeated with item order shuffled within each user's sequence, in training and in evaluation inputs.
  - Test-only shuffle (descriptive): every neural final run is also scored on shuffled test inputs, as in [11].
  - Without a drop larger than SESOI on the MovieLens datasets, order handling and evaluation are audited before other results are reported.
- Rationale: a sequential model that does not lose quality without order information is either not using order or not being evaluated correctly. [11] shuffles test inputs only; the train-and-evaluate variant also removes order from training.

## D018 Number of seeds

- Date: 2026-11-08. Status: accepted.
- Decision:
  - The leave-one-out seed count per dataset is the smallest n from 5 to 10 at which exact t-based power reaches 0.8 at the SESOI, for both the strictest F1 directional test and TOST (METHODS 7.5).
  - The input σ comes from validation scores of the M2 pilot, then from the first 5 final seeds per dataset, before any test read.
  - Every comparison reports its power; those below 0.8 are labelled underpowered, and their non-rejections are not interpreted.
- Rationale: with 5 seeds, a 2% margin and plausible seed variance, Welch tests may have too little power. With σ = 0.01, even 10 seeds give 0.78 directional power. A rule fixed before any result avoids choosing the seed count after seeing outcomes.

## D019 Full validation set every epoch

- Date: 2026-11-08. Status: accepted.
- Decision: every validation user is scored after every epoch; the 10,000-user subsample for early stopping is removed. Time to target (RQ2) uses the same full-validation curve.
- Rationale: the environment check measured full-catalogue scoring at about 1.5 ms per 1,000 users for 26,000 items, about 0.2 s for all MovieLens-20M validation users. A subsample would add a second validation curve without saving meaningful time.

## D020 Where datasets are prepared

- Date: 2026-11-08. Status: accepted.
- Decision: the official preprocessing and statistics run once on the GPU machine (`seqrec data prepare`), and only `results/data/` is pushed. The development container prepares Steam and Video_Games for development and checks that their fingerprints match the GPU machine's.
- Rationale: files.grouplens.org is not reachable from the development container. Matching fingerprints across Linux and Windows also checks that preprocessing is deterministic.
- Amendment 2026-11-08: the GPU machine's data run (`results/data/stats.csv`) and the development container produce identical fingerprints for all six Steam and Video_Games products (log, leave-one-out and temporal split): 732bccfb2e720391, 80b6e78d66883bf4, cd343c472c0296b2, b6d5f4b89cbad7fc, 75d6517e6d2ea176, 56efeb4714878af5. For these two datasets, preprocessing is deterministic across Windows and Linux.

## D021 Third-party metric cross-check

- Date: 2026-11-08. Status: accepted.
- Decision: ranx (MIT licence per its PyPI classifiers) cross-checks NDCG@10 and HR@10 in the tests. It lives in the optional `crosscheck` dependency group because it pulls in numba and other heavy packages; the naive reference implementation runs in every test session.
- Rationale: an independent implementation catches errors a self-written reference could share.

## D022 Dataset statistics against published values

- Date: 2026-11-08. Status: accepted.
- Decision: the preprocessing of METHODS section 2 stays as specified. Five rows of `results/data/published_comparison.csv` differ by more than 1%; each comes from a different filtering procedure in the source, not from a defect. MovieLens-1M (6,040 users, 3,416 items) and the MovieLens-20M 5-core counts (138,493 users, 18,345 items [12]) match exactly, and Video_Games matches the official 5-core table within its rounding.
- Explanations:
  - Steam users 281,210 against 334,730 [1] (−16.0%), items 11,961 against 13,047 [1] (−8.3%), interactions 3,484,694 against 3.7M [1] (−5.8%). One pass of the 5-core filter on raw counts (repeated reviews included), followed by removal of repeated pairs, reproduces [1] exactly from the raw file: 334,730 users, 13,047 items, 3,686,172 interactions. A single pass keeps users and items that fall below five once the other side is filtered; the iterative filter used here removes them. Repeated reviews are removed here before filtering (7,793,069 rows to 6,889,728).
  - Steam items 11,961 against 13,044 [2] (−8.3%). Keeping only users with at least five interactions in the single-pass data above reproduces [2] exactly: 281,428 users, 13,044 items, 3,488,885 interactions. The Steam data of [2], also used in [3, 4, 5], therefore differs from ours through the single-pass item filter, not only through its user filter.
  - MovieLens-20M items 18,345 against 26,744 [2] (−31.4%). BERT4Rec does not filter items, and 26,744 is exactly the item count of our deduplicated log before filtering; the item filter removes the 8,399 items with fewer than five ratings.
- Consequence: published numbers on Steam and MovieLens-20M from [2, 3, 4, 5] come from differently filtered data, which the report states next to every comparison with them.

## D023 Fallback for incomplete certificate chains

- Date: 2026-11-08. Status: accepted.
- Decision: when Python reports a certificate verification error, the downloader retries the same URL with `curl`, preferring the `curl.exe` shipped with Windows 10 version 1803 and later; if that also fails, it stops with a manual download command and the target path. Integrity rests on the pinned SHA-256 of every file.
- Rationale: on the GPU machine, Python's `ssl` module rejected files.grouplens.org with "unable to get local issuer certificate", which points to a server that omits an intermediate certificate, while `curl.exe` downloaded the file. Windows' curl validates through Schannel, which can fetch a missing intermediate certificate from the address in the server certificate (AIA); Python's `ssl` module does not. The certificate is still validated against the Windows trust store, so the fallback adds no trust, and it avoids a new dependency such as truststore for one server.

## D024 Methods frozen

- Date: 2026-11-08. Status: accepted.
- Decision: `docs/METHODS.md` is frozen as committed in 569676ffb96a23a7dca0e59207cc4c314651cbb8, SHA-256 b3c26c4f6303b4cd6948ebe7e1bbc25e2e16fbc236b7591c476c47b2a8d0f2b8. It differs from the first freeze commit (b90d6a7) only in its title line. No model had been trained on real data at that point; the only real-data outputs were the dataset statistics in `results/data/`.
- Rationale: D014. Any later edit of the file is a deviation recorded here; `sha256sum docs/METHODS.md` against the value above shows whether the file still matches.

## D025 Reference implementations and model details

- Date: 2026-11-09. Status: accepted. Two items deviate from METHODS (marked below). Decided before any model was trained on real data; no result on real data had been seen.
- Decision:
  - SASRec block (A1 to A4), as in the SASRec code [31], pmixer/SASRec.pytorch before May 2025, [5] and [33]: only the attention query is layer-normalised, keys and values read the un-normalised stream, the attention residual adds the normalised query and the feed-forward residual adds its normalised input. Feed-forward d → d with ReLU, LayerNorm epsilon 1e-8, item embeddings scaled by √d, final LayerNorm. **Deviation from METHODS 4.1**, whose "pre-LayerNorm ... with residual connections" reads as the standard form with residuals on the raw stream, as in the SASRec paper.
  - Attention runs through SDPA with input and output projections, equivalent to `nn.MultiheadAttention`. The mask is causal and blocks padded keys; a padded query attends to itself (METHODS 4.1).
  - Initialisation of every neural arm as in [5]: N(0, 0.02) for linear and embedding weights, zero biases, LayerNorm weight 1 and bias 0, padding rows zero.
  - BERT4Rec (A5): item plus position embedding, LayerNorm, dropout; post-LN blocks with exact GELU, a 4d feed-forward layer without dropout after the GELU, LayerNorm epsilon 1e-12, as BertLayer; output head of dense, GELU and LayerNorm, then the tied item embeddings plus an item bias ([32], METHODS 4.2). Inference scores a mask token appended after the last L−1 items.
  - Full cross-entropy and BERT4Rec use a softmax over items 1..N only (METHODS 4.3).
  - BCE (A1): one negative per position drawn exactly, without rejection, from the items outside the user's whole training sequence; the two terms are summed and averaged over positions, as in [31].
  - gBCE (A4): K negatives per position, uniform with replacement, as in [33]; negatives equal to the positive are dropped (METHODS 4.3); each position's loss is divided by 1 + K, as in [33]. **Deviation from METHODS 4.3**, which writes the per-position sum. The factor is constant, but it changes how gradient clipping at norm 5 and weight decay act relative to the loss.
  - Sampled softmax (A3): K negatives shared by the batch, collisions with the positive masked, no log-Q correction (METHODS 4.3).
  - Chunked cross-entropy recomputes each slice's logits in the backward pass (`torch.utils.checkpoint`); loss and gradients equal plain cross-entropy (tested).
  - Adam's `weight_decay` (L2) applies to all parameters. TF32 is allowed in training and disabled while scoring, so evaluation scores are strict fp32 (METHODS 5.1, 6).
- Borrowed: the SASRec block and embedding scaling from [31]; the initialisation from [5]; gBCE sampling and normalisation from [33]; BERT layer details from [32] and HuggingFace BertModel. No third-party code is in the package. A pinned copy of pmixer's model (Apache-2.0, `THIRD_PARTY_NOTICES`) sits under `tests/reference/` to check our SASRec forward pass against it.
- Intentionally different from the sources: [5] pads on the right and masks no padded keys, and pmixer masks none either; [5]'s softmax includes the padding id; [5] excludes only the user's last 201 items from BCE negatives, while the paper [1] excludes all of the user's items; [33] keeps negatives equal to the positive; [5]'s BERT4Rec has no output transform or bias; [32] trains on sliding windows with ten masked copies (METHODS 4.2).
- Licences: kang205/SASRec, pmixer/SASRec.pytorch, FeiSun/BERT4Rec and gSASRec-pytorch are Apache-2.0; RecBole is MIT; the repositories of [5] and [3] have no licence file.
- Rationale: [31], pmixer before 2025, [5] and [33] share the same SASRec block, and the anchors come from [5] and [33]. Matching their block removes one cause of anchor misses. The gBCE normalisation follows [33] for the same reason.

## D026 Anchor runs

- Date: 2026-11-09. Status: accepted. Deviation from METHODS 5.2 for anchor runs only. Decided before any model was trained on real data.
- Decision:
  - Anchor runs use the source's training length as well as its other hyperparameters. [5]: SASRec-CE and SASRec-BCE at most 100 epochs with patience 10, BERT4Rec at most 200 with patience 20. [33]: gSASRec patience 200 and at most 10,000 epochs, under the 8-hour guard.
  - Configurations, all with batch 128, input length 200 and no weight decay:
    - SASRec-CE and SASRec-BCE [5]: embedding 64, 2 blocks, 1 head, dropout 0.1, learning rate 1e-3, shared output embeddings; BCE with one negative.
    - BERT4Rec [5]: embedding 64, 2 blocks, 2 heads, feed-forward 256, dropout 0.1, mask probability 0.2, learning rate 1e-3.
    - gSASRec [33]: embedding 128, 2 blocks, 1 head, dropout 0.5, separate output embeddings, 256 negatives, t = 0.75, learning rate 1e-3.
  - Bands are those of METHODS 11.5 (±0.01 around the published value), unchanged. Each arm first uses one anchor read (seed 0).
  - The M2 seed-variance runs use the same configurations under the METHODS 5.2 caps and read no test data; A3, which has no published anchor, uses the A2 configuration with K = 256. Comparing their validation NDCG@10 with the anchor runs' shows what the caps cost.
- Rationale: METHODS 11.5 asks for "the hyperparameters of one published source". [5] reports BERT4Rec's best epoch as 197 of 200 (Table 2), so the 1000/100 cap could lift it above its band through longer training alone, and a result above a band is first treated as leakage. The checkpoints shipped with [33] come from about epoch 990 (SASRec) and 1,790 (gSASRec), far beyond the 300-epoch cap.

## D027 Planted-rule data for the known-answer tests

- Date: 2026-11-09. Status: accepted. Deviation from METHODS 11.2 (third item). Decided before any model was trained on real data.
- Decision:
  - The test that every neural arm reaches test NDCG@10 ≥ 0.95 uses the rule as written: every transition follows one fixed permutation (a single cycle over 200 items; 1,000 users with 30 to 50 items each).
  - The shuffled-order test uses a second dataset: training transitions follow the permutation with probability 0.2 and otherwise go to a random unseen item, while validation and test targets always follow it. SASRec-CE must reach test NDCG@10 ≥ 0.95 with order and at most 0.10 with training and evaluation order shuffled.
  - The gSASRec test uses K = 4 and t = 0.5. With K = 16 or more on 200 items, gBCE first settles on a score offset shared by all items and learns the rule only much later; the reference gSASRec code trained on the same data shows the same (validation NDCG@10 about 0.02 after 30 epochs with K = 16, t = 0.75).
- Results on the test data: with the rule as written, test NDCG@10 is 1.000 (SASRec-CE, SASRec-BCE, gSASRec), 0.994 (SASRec-SS) and 0.985 (BERT4Rec), against 0.049 for MostPop. On the second dataset SASRec-CE scores 0.996 with order and 0.026 shuffled, against 0.028 for MostPop.
- Rationale: on walks that always follow the rule, the set of input items reveals the target. In a simulation with the same generator, a scorer that sees only item co-occurrence reached NDCG@10 0.77 against 0.04 for MostPop, so "metrics fall to about the popularity baseline" cannot hold. With probability 0.2 the simulation gave 1.00 for a first-order transition scorer, 0.037 for the co-occurrence scorer and 0.023 for MostPop.

## D028 Tuning harness

- Date: 2026-11-09. Status: accepted. Implements METHODS 5.4 without deviation.
- Decision:
  - `seqrec tune PLAN` runs one Optuna study per (arm, dataset), stored in `runs/<batch>/optuna.sqlite3`. TPE's first 5 trials are random; median pruning starts after 5 complete trials, with a warm-up of 10% of the arm's epoch cap. The objective is the restored checkpoint's validation NDCG@10.
  - Each trial is an ordinary run with phase `tune`, seed 0 and tag `tNN`; its test split is the sealed stand-in, and its config records every parameter.
  - A trial interrupted by a crash resumes from its last checkpoint with its own parameters. TPE is reseeded from (sampler seed, trials so far) before every proposal, so a resumed study proposes the same parameters as an uninterrupted one (tested).
  - Failed and pruned trials count toward the trial budget, so every arm gets the same number of trials. Export writes `tuning_best.csv`: per study, the best trial that finished without pruning, ties to the earlier trial.
- Rationale: METHODS 5.4 fixes the sampler, pruner and budget; these choices make resumption and budget accounting exact.

## D029 Dependencies for M2

- Date: 2026-11-09. Status: accepted.
- Decision: Optuna 5.0 (MIT; released 2026-09-07; with SQLAlchemy and Alembic, both MIT) for tuning, and SciPy 1.18 (BSD-3-Clause) for the noncentral t power of METHODS 7.5.
