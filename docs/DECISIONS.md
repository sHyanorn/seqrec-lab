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
- Amendment 2026-11-09: the cap is 500 GPU hours (D034). The cut order is unchanged and applies above it.

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
- Amendment 2026-11-09 (review fixes, before any GPU run): gBCE scores are computed in fp32 with autocast off, as METHODS 4.3 requires; the sampled-softmax positive and negative logits both come from matrix products and share the autocast precision. Best checkpoints hold network weights only, `last.pt` is removed when training ends, and tuning runs keep no checkpoint, which bounds disk use over about 350 tuning runs.

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
- Limitation: the rule is first order, so these tests cannot show that SASRec uses items before the last one. The exact attention-mask test, the left-padding test and the comparison with pmixer's model cover that.
- Rationale: on walks that always follow the rule, the set of input items reveals the target. In a simulation with the same generator, a scorer that sees only item co-occurrence reached NDCG@10 0.77 against 0.04 for MostPop, so "metrics fall to about the popularity baseline" cannot hold. With probability 0.2 the simulation gave 1.00 for a first-order transition scorer, 0.037 for the co-occurrence scorer and 0.023 for MostPop.

## D028 Tuning harness

- Date: 2026-11-09. Status: accepted. Implements METHODS 5.4 without deviation.
- Decision:
  - `seqrec tune PLAN` runs one Optuna study per (arm, dataset), stored in `runs/<batch>/optuna.sqlite3`. TPE's first 5 trials are random; median pruning starts after 5 complete trials, with a warm-up of 10% of the arm's epoch cap. The objective is the restored checkpoint's validation NDCG@10.
  - Each trial is an ordinary run with phase `tune`, seed 0 and tag `tNN`; its test split is the sealed stand-in, and its config records every parameter.
  - A trial interrupted by a crash resumes from its last checkpoint with its own parameters. TPE is reseeded from (sampler seed, trials so far) before every proposal, so a resumed study proposes the same parameters as an uninterrupted one (tested).
  - Failed and pruned trials count toward the trial budget, so every arm gets the same number of trials. Export writes `tuning_best.csv`: per study, the best trial that finished without pruning, ties to the earlier trial.
- Rationale: METHODS 5.4 fixes the sampler, pruner and budget; these choices make resumption and budget accounting exact.
- Amendment 2026-11-09 (review fixes, before any tuning): a trial interrupted before all its parameters were stored is marked failed rather than trained with defaults; a resumed trial re-reports its finished epochs to the pruner; `seqrec tune` exits with status 1 when a trial fails. The overview gives the truncated share both over all finished trials and over unpruned trials; which one the METHODS 5.2 rule uses is decided before tuning starts.

## D029 Dependencies for M2

- Date: 2026-11-09. Status: accepted.
- Decision: Optuna 5.0 (MIT; released 2026-09-07; with SQLAlchemy, Alembic, Mako and colorlog, all MIT) for tuning, and SciPy 1.18 (BSD-3-Clause) for the noncentral t power of METHODS 7.5.

## D030 Truncation share for the cap-doubling rule

- Date: 2026-11-09. Status: accepted. Clarifies METHODS 5.2. Decided before any tuning trial ran and before any result on real data was seen.
- Decision:
  - "More than 20% of an arm's tuning trials on a dataset are truncated" is computed per study over the trials that stopped by patience, the epoch cap or the 8-hour guard. Pruned, diverged and failed trials are left out of both counts. "More than" is strict: 4 of 20 does not double the limits.
  - Export writes the share per study as `truncated_share` in `tuning_best.csv`, next to the counts `stopped` and `truncated`; the rule reads that column. The overview uses the same definition.
- Rationale: pruned and diverged trials stop before any limit can apply, so including them would make the share depend on how often the pruner fires or training diverges. The rule asks whether the limits cut short the trials that could reach them.
- Amends D028: the overview no longer gives a share over all finished trials.

## D031 Failed attempts

- Date: 2026-11-09. Status: accepted. Not covered by METHODS. Decided before any tuning trial ran and before any result on real data was seen.
- Decision:
  - A run or tuning trial whose training raises an exception is retried once, whatever the cause, and resumes from its last checkpoint. A second failure is final: the run is not trained again, and a trial counts toward the trial budget (D028). A trial that succeeds on its retry uses no extra budget.
  - The retry runs in the same process if the GPU still runs a kernel after the failure. Otherwise (for example after a driver reset) the command stops with status 1 and rerunning it performs the retry; a tuning trial stays running in its study until then.
  - Every failed attempt is appended to the run's `failures.jsonl` (time, step, exception type and first line, commit), and exports count them in a `failures` column. A run that succeeds on its retry keeps the status from its summary.
  - Test reads are not retried. A final run whose test read started but did not complete is refused a second read (METHODS 11.6), and the case is decided individually. A failed anchor read still counts toward the three anchor reads (METHODS 11.5). Exports flag every incomplete read.
  - Interruptions without an exception (a closed terminal, sleep, power loss) are resumptions, not failures (D028).
- Rationale: judging after the fact whether a failure was anomalous (memory fragmentation, a driver reset) or caused by the configuration leaves room for choice; a deterministic error fails again on its retry and is recorded as failed either way. Before this change, a run that succeeded on a rerun was exported as failed, which would have dropped it from the σ estimate (METHODS 7.5) without notice.
- Amendment 2026-11-09 (review fixes, before any tuning trial or pilot run):
  - A plan edited under an existing run directory (a configuration conflict) is not a failed attempt. A dataset that has not been prepared stops the command before any attempt.
  - A run whose summary exists has finished, whatever its failure count; a single call still makes at most two attempts. Stop decisions made inside the epoch loop (pruned, diverged) are saved with that epoch's checkpoint, so a retry never trains past them.
  - `seqrec run` and `seqrec test` refuse a plan whose runs differ from their existing run directories before any attempt or test read, so a run cannot change after other runs of its plan were tested (METHODS 5.5). `seqrec tune` reports such a study and continues with the next.
  - After any failure, a GPU that no longer runs a kernel stops the command, also after a final failure and while testing, so the next run is not charged for it.
  - `seqrec test` treats a run whose training failed twice as finished: it is listed and not tested, and the other runs of the plan are tested. A replacement seed has to enter the plan before `seqrec test` runs (METHODS 5.5). Diverged runs are not tested.
  - Test reads refused before any data are read (anchor limit reached, an earlier final read incomplete, a changed checkpoint) are reported and not recorded as failures; any other error, including one after the read started, is recorded. Export marks a run `test_failed` only when a test read started and did not complete.
  - Run files are flushed to disk before they replace the previous version, and a transient Windows file lock on that replacement is retried.

## D032 Anchor results

- Date: 2026-11-09. Status: accepted. Method fixed before any anchor result was seen; results added after the anchor reads.
- Method: `seqrec anchors results/runs/m2-anchor.csv` lists every anchor run on MovieLens-1M.
  - Stop conditions, each reported before any other step: a run not tested; anything other than one completed anchor read and no final read per run; a commit or parameters other than those of D026; test NDCG@10 outside the band of the METHODS 11.5 table (edges included in the band); test and validation NDCG@10 more than 3·sqrt((v(1 − v) + t(1 − t))/6,040) apart.
  - The last threshold is three standard errors of the difference. Per-user NDCG@10 lies in [0, 1], so m(1 − m)/U bounds the variance of a mean over U users, and the two means are treated as independent; at NDCG@10 0.18 the threshold is about 0.021. Test above validation by more is handled like a result above the band (the leakage checks of METHODS 11.5), below by more like a result below it.
  - Recorded without a stop rule: epochs; best epoch, counted from 1, against [5] (63 for SASRec-CE, 53 for SASRec-BCE, 197 for BERT4Rec) and about 1,790 for the gSASRec checkpoint of [33]; stop reason and truncation; test and validation tie rates, expected near 0 for neural models; test HR@10; NDCG@10 on shuffled test inputs, expected clearly lower (METHODS 3.3); hardware slowdown.
  - No code changes after a stop condition until the deviation is decided; any change then follows METHODS 11.5.
- Amendment 2026-11-09 (before any anchor result was seen): the command prints every stop condition. Parameters, and limits (epoch cap, patience, guard) where the export lists them, are compared with `configs/plans/m2-anchor.toml`; a missing arm is a stop condition. The conditions apply to the latest anchor run of each arm, which must all come from one commit; earlier runs stay in the table as history.
- Results (seed 0, one anchor read per arm, commit 70b2b0f; `seqrec anchors` reports no stop condition):

| Arm | Test NDCG@10 | Band | Validation NDCG@10 | Test − validation (threshold) | Epochs | Best epoch | Stop | Test HR@10 | Shuffled test NDCG@10 | Test tie rate |
|---|---|---|---|---|---|---|---|---|---|---|
| SASRec-CE | 0.1784 | 0.172–0.192 | 0.1882 | −0.0098 (0.0211) | 78 | 68 | patience | 0.3126 | 0.0407 | 0.0000 |
| SASRec-BCE | 0.1361 | 0.124–0.144 | 0.1414 | −0.0053 (0.0189) | 85 | 75 | patience | 0.2535 | 0.0396 | 0.0000 |
| BERT4Rec | 0.1575 | 0.144–0.164 | 0.1643 | −0.0068 (0.0201) | 200 | 200 | epoch cap | 0.2829 | 0.0381 | 0.0000 |
| gSASRec | 0.1691 | 0.163–0.183 | 0.1828 | −0.0137 (0.0208) | 384 | 184 | patience | 0.2924 | 0.0336 | 0.0003 |

  - Every arm is inside its band after one read, so no arm is flagged and no code changed. Test NDCG@10 is below validation for every arm and within the threshold. Shuffled test inputs lower NDCG@10 to 20–29% of its value. Tie rates are at most 0.0003, and no run shows a hardware slowdown.
  - Best epochs against the sources: SASRec-CE 68 (63 in [5]), SASRec-BCE 75 (53), BERT4Rec 200 of 200, so it was still improving at the cap, as in [5] (197 of 200). gSASRec reached its best at epoch 184 and stopped by patience at epoch 384, far earlier than the about 1,790 epochs of the [33] checkpoint, yet inside its band.
  - GPU time 1.13 h, of which gSASRec 1.07 h (1.9 s per training epoch against 0.35 to 0.45 s for the other arms).
  - These reads touched the MovieLens-1M test targets once per arm; RQ1 conclusions on MovieLens-1M carry the note of METHODS 11.5.

## D033 Seed variance and seed count

- Date: 2026-11-09. Status: method fixed before any pilot result was seen; results pending.
- Method:
  - σ and n: `seqrec seeds results/runs/m2-pilot.csv --tag sigma --dataset ml-1m` (METHODS 7.5).
  - What the METHODS 5.2 limits cost: `seqrec anchors results/runs/m2-anchor.csv --pilot results/runs/m2-pilot.csv`. Per arm, Δ is the mean log validation NDCG@10 of the three seed-variance runs (METHODS 5.2 limits) minus the log validation NDCG@10 of the latest anchor run (the source's limits, D026), and z = Δ/(σ·sqrt(1 + 1/3)).
  - An increase of an arm's limits is recommended when Δ < −log(1.02), the SESOI, and z < −2. The limits stay as in METHODS 5.2 until that recommendation is decided.
  - SASRec-SS has no anchor and is reported without Δ. The METHODS 5.2 limits are longer than the source's for SASRec-CE, SASRec-BCE and BERT4Rec, and much shorter for gSASRec (300 epochs against up to 10,000).
- Results: pending.

## D034 GPU budget from the pilot

- Date: 2026-11-09. Status: method fixed before any timing result was seen; results pending.
- Method (`seqrec budget results/runs/m2-pilot.csv`):
  - Epoch time per dataset and arm: median training plus validation seconds per epoch of the pilot timing runs. Tuned and final configurations are unknown, so the central value is the mean of the anchor size (`time-small`) and the largest size in the search space (`time-large`).
  - Epochs per run: the mean best epoch (counted from 1) of the MovieLens-1M seed-variance runs, scaled by the best-epoch ratio of [5] Table 2, plus the patience, at most the epoch cap. Ratios against MovieLens-1M: MovieLens-20M 46/63 for cross-entropy and sampled softmax, 30/53 for BCE and gBCE, 68/197 for BERT4Rec; Steam 16/63, 39/53 and 74/197; Video_Games, absent from [5], 1. Epoch counts do not fall with dataset size: in [5], MovieLens-20M has 23 times the users of MovieLens-1M and a best epoch of 46 against 63 (SASRec+).
  - One run costs its epochs times the epoch time, plus one evaluation of the restored checkpoint.
  - Blocks: tuning (METHODS 5.4 trial budgets, no credit for pruning); final leave-one-out runs with n seeds (D033); temporal-split runs with 5 seeds, timed as leave-one-out; the shuffle control (SASRec-CE on four datasets, BERT4Rec on two, n seeds); 180 s of warm-up for each of 13 batches; RQ3 at 15 h (METHODS 12); GPU hours already spent, from `overview.csv`. Floor baselines and test reads are left out.
  - Two further scenarios: MovieLens-1M epochs on every dataset; the same with the largest configuration's epoch time.
  - Decision: if the central core exceeds 200 GPU hours, the D012 cuts apply in order until it fits, each recorded here. If only the other scenarios exceed it, nothing is cut; the projection is repeated with measured trial times after MovieLens-1M tuning. That repetition also corrects the epoch count if the gSASRec seed-variance runs stop on the early plateau seen on synthetic data (D027).
- Results: pending.
- Amendment 2026-11-09 (before any timing result was seen):
  - The core counts 5 leave-one-out seeds for the final runs and the shuffle control, as METHODS 12 does, and the cut decision uses that core. Seeds beyond 5 from the METHODS 7.5 rule are paid from the reserve in the METHODS 7.5 priority order; one extra seed of a dataset includes its shuffle-control runs. Once a dataset cannot get all its extra seeds, the datasets after it get none.
  - Spent hours are those of the M2 batches only, so a later projection does not count finished work twice.
  - Epoch time includes the checkpoint time per epoch measured by the timing runs, which falls outside the training and validation timers.
  - The BERT4Rec timing runs at the largest size use mask probability 0.4, the upper end of its search range.
  - Warm-ups are recounted after a cut.
- Amendment 2026-11-09 (after the pilot projection was seen, before any tuning run): the owner raised the GPU cap from 200 to 500 hours, all on the laptop, and kept the full scope. The central core is 416.0 h, or 468.9 h with the start rule of D035; the three D012 cuts leave 222.6 h (246.7 h), still above 200 h, and the third cut drops Steam, the only dataset where G2 and G3 have a negative published sign. The D012 cut order stays and applies above 500 h. Before tuning starts on each dataset, the projection is repeated with the measured trial times of the datasets already tuned; if the central core then exceeds 500 h, the D012 cuts apply in order and each is recorded here. `seqrec budget` uses the 500 h cap; METHODS 12 still states 200 h, so this is a deviation from it.

## D035 Start rule for slow-starting runs

- Date: 2026-11-09. Status: accepted; the start limit (200 or 300 epochs) is confirmed before the gSASRec rerun. Deviation from METHODS 5.2. Decided after the pilot's validation results (D033) and the anchor results (D032) were seen, and before any tuning run.
- Trigger: the D033 cap-cost check recommends higher limits for gSASRec (relative −75.9%, z −2.22). Two of its three seed-variance runs stopped by patience at epochs 44 and 54 with validation NDCG@10 0.0216, 1.04 times MostPop's 0.0207; the third reached 0.1841 and the anchor run 0.1828. No run reached the epoch cap, so patience fired before the runs began to learn. The README of the authors' original code (github.com/asash/gsasrec) describes this pattern for gBCE and advises longer patience; [33] uses patience 200.
- Decision:
  - At the start of every run of a neural arm, the runner computes MostPop's validation NDCG@10 on the run's own dataset and split, from its training split (B1, METHODS 4.4), and records it as `start_reference` in the manifest and summary.
  - A run starts at the first epoch whose validation NDCG@10 exceeds 1.5 times that value. Before it starts, patience does not count; afterwards patience and the epoch cap act as in METHODS 5.2.
  - A run that has not started after `start_limit` epochs (default 200) stops with reason `no_start`. It counts as truncated, so the doubling rule of METHODS 5.2 (D030) covers it, and doubling also doubles the start limit. The code does not double it: a plan with doubled limits sets `start_limit` to twice the default by hand, next to `max_epochs`, `patience` and `guard_hours`. It keeps its best checkpoint and enters every analysis like any other run.
  - The rule applies to all neural arms (A1 to A5). The floor baselines do not train and have start limit 0. Median pruning in tuning stays as in METHODS 5.4.
- Effect on existing runs: a run that passed the threshold before patience fired follows the same path under the rule. All twelve seed-variance runs of A1 to A3 and A5, and gSASRec seed 2, ended above 0.14 against a threshold of 0.031, so their results stand. gSASRec seeds 1 to 3 are rerun under the rule in batch `m2-gsasrec`, whose runs replace the pilot's gSASRec seed-variance runs in D033 and D034.
- Analysis inputs: `seqrec seeds`, `seqrec budget` and `seqrec anchors --pilot` accept several exports, and runs in a later file replace earlier runs with the same tag, dataset, split, shuffle and arm. D033 and D034 read `results/runs/m2-pilot.csv` followed by `results/runs/m2-gsasrec.csv`.
- Factor: the stuck runs sat at 1.04 times MostPop; after three epochs SASRec-BCE and SASRec-SS stood at 2.3 times and SASRec-CE at 4.8 times, and converged runs at 7 to 9 times. A wrong factor costs epochs but never removes a run.
- Cost: if every gSASRec run starts as seed 2 and the anchor run did, the central D034 projection grows by about 53 GPU hours (gSASRec on MovieLens-1M from 117 to 234 epochs per run).
- Alternatives: the anchor limits for gSASRec (10,000 epochs, patience 200) add about 164 hours and stretch the pruning warm-up to 1,000 epochs; a lower t is already tuned over [0.5, 1.0], and changing it in the seed-variance runs would leave the anchor configuration; dropping stuck runs selects on the outcome.
- Tests: the start-rule tests in `tests/test_runner.py` and `test_a_later_export_replaces_the_runs_it_repeats` failed before their changes (METHODS 11.5).
- Run directories and checkpoints from before this rule, and a later export with unfinished runs: D040.

## D036 Time counted by the guard

- Date: 2026-11-09. Status: accepted. Clarifies the wall-clock guard of METHODS 5.2. Fixed after the pilot timings were seen (D034) and before any M3 run.
- Decision: the 8-hour guard counts the training and validation time of every epoch, summed over all sessions of a run. Checkpoint writes, the set-up before the first epoch (data loading, invariant checks, model construction), the final evaluation of the restored checkpoint and any time between sessions are not counted. The check runs before each epoch, so a run can pass the guard by at most one epoch. Doubled limits (METHODS 5.2) double this time. The runner has counted this way since M1; `test_the_guard_counts_training_and_validation_time_only` fixes it.
- Rationale: an interrupted run gets the same budget as an uninterrupted one, and disk speed does not decide when a run stops. Checkpoint writes took about 1% of epoch time in the pilot, so the choice moves guard stops by about that much.

## D037 Triton FMA fusion fixed on

- Date: 2026-11-09. Status: accepted. No training run compiles code, so no result depends on it.
- Decision: importing `seqrec` sets `TRITON_DEFAULT_FP_FUSION=1` and overrides any inherited value. Every entry point imports the package before anything is compiled, including the `torch.compile` worker of `scripts/check_env.py`. `tests/test_package.py` failed before the change (METHODS 11.5).
- Rationale: fusion is already the default of the locked triton-windows 3.9 (D003), so behaviour does not change. Fixing it keeps a later Triton release or a stray environment variable from changing the rounding of compiled kernels unnoticed, which the M4 `torch.compile` measurements (METHODS 9.1) would absorb.

## D038 Slow-run flag

- Date: 2026-11-09. Status: accepted. Adds a flag to METHODS 8 and 7.6. Decided after the pilot timings were seen and before any M3 run.
- Decision: export marks `slow_epochs` when a run's median training plus validation time per epoch exceeds 1.5 times the median of the other runs with the same batch, dataset, split, shuffle, arm, parameters and tag, that is, its other seeds. Runs without such runs (tuning trials, timing runs) get no value. RQ2 timing analyses treat a flagged run like one with a hardware slowdown: it stays in the primary analysis and is left out of the repeat of METHODS 7.6. The flag tests failed before the change (METHODS 11.5).
- Rationale: in the pilot, gSASRec seed 2 took 30.6 s per training epoch against 1.7 to 1.9 s for its other seeds and the anchor run, at a higher SM clock (median 2,917 against about 2,000 MHz) and with no hardware slowdown event. Neither flag of METHODS 8 catches such a run, and its times would enter time to target (H2a) unmarked.
- Limitation: tuning trials have no other seeds, so a slowed trial is not flagged; under the guard (D036) it can be marked truncated and count toward the doubling rule of METHODS 5.2.
- `test_export_flags_the_slow_seed` checks the exported values of the flag (D040).

## D039 Step-time profiler

- Date: 2026-11-09. Status: accepted. A measurement tool outside METHODS; no analysis uses its output.
- Decision: `scripts/profile_step.py` trains one arm for a few steps on one prepared dataset through the arm's own epoch loop and prints wall time, process CPU time and GPU kernel time per step. It reads only the training split and writes nothing. `tests/test_profile_step.py` failed before the script existed (METHODS 11.5).
- Rationale: in the pilot timing runs every arm took about 7 ms per step on Steam and Video_Games, at both configuration sizes, which points to a fixed cost per step outside GPU compute. What a change to the training loop could save depends on how much of a step the GPU is busy, which only the GPU machine can measure.
- GPU time counted once per kernel, per-step times without the cost per epoch, and `--from-run`: D040.

## D040 Fixes from the M2 close review

- Date: 2026-11-09. Status: accepted. Code changes after the anchor reads (METHODS 11.5); each came with a test that failed before it. Decided before any M3 run and before the gSASRec rerun of D035.
- Run directories claimed before D035 have no `start_limit` in `config.json`. They compare equal to a plan with the default limit, so interrupted batches resume; such a run continues under the rule. A run whose start limit differs from its directory, including after a change of the default, is refused, and the message names the field (`test_run_directories_from_before_the_start_rule_are_not_conflicts`, `test_a_changed_default_start_limit_stops_an_interrupted_run`). A negative start limit is rejected.
- A `last.pt` saved before D035 has no start record. On resume the run counts as started if an epoch of its `metrics.jsonl` beat 1.5 times the reference, from the first such epoch, or if its best score did; otherwise the rule applies from where it stopped. Summaries record the start limit the run used, and export reads it from there (`test_a_state_saved_before_the_start_rule_resumes_as_started`; `test_an_interrupted_run_keeps_its_start_rule_state` covers a run interrupted before it started).
- `scripts/profile_step.py` (D039) summed the device time of CPU ops and of the kernels they launch, so it counted GPU time about twice. It now sums only GPU events that are not user annotations (`test_gpu_time_counts_each_kernel_once`). Per-step times are the difference between an epoch of n steps and the median of five one-step epochs, divided by n − 1, and the fixed cost per epoch is printed apart (`test_time_per_step_leaves_out_the_cost_per_epoch`); n defaults to 200 and must be at least 2.
- `scripts/profile_step.py --from-run <run directory>` reads dataset, split, arm, seed and parameters from that run's `config.json`, so profiling commands carry no JSON, whose quoting differs between Windows PowerShell 5.1 and PowerShell 7 (`test_a_run_directory_supplies_the_configuration`).
- `seqrec seeds`, `anchors` and `budget` stop with status 1 when an export after the first holds a run that did not finish training. Such a file replaces whole groups (D035), so a failed seed would have lowered n and changed σ without notice (`test_a_rerun_with_an_unfinished_run_is_refused`).
- The NVML sampler also records device memory in use (`nvmlDeviceGetMemoryInfo`, the whole device, other processes and the display included). Run summaries give its distribution and export its peak as `nvml_mem_used_mib_max`, next to the PyTorch allocation peak per epoch, so a run that came close to the 16 GB can be found (`test_the_sampler_records_device_memory_and_its_peak`, `test_export_lists_peak_device_memory`).
- `seqrec test` compares the training and full-data fingerprints with the run's manifest before the test read and refuses a run whose data changed since training; the refusal is not a read and not a failure (D031). Before, only a resumed training session compared the training fingerprint (`test_a_run_is_not_tested_on_data_other_than_it_trained_on`, `test_a_rebuilt_dataset_is_refused_before_the_test_read`).
- `test_the_guard_leaves_out_set_up_and_adds_up_sessions` extends the D036 test to set-up time and an interrupted run.
