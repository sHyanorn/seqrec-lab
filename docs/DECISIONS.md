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
- Amendment 2026-11-10: superseded for all later GPU runs by D045 (Linux container on a rented machine). The comparison with WSL2 no longer applies.

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
- Amendment 2026-11-10: on Linux the lock file carries two mutually exclusive extras, `cpu` (development) and `cu132` (GPU runs, D045); a Linux sync without an extra installs no torch, so a GPU run cannot fall back to a CPU build unnoticed (`tests/test_lock.py`). `triton-windows` is not used on Linux.

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
- Amendment 2026-11-10: the thermal behaviour recorded here is the laptop's. Conditions for RQ3 on the rented machine are in D045; both regimes and the exclusion rule are unchanged.

## D012 GPU budget

- Date: 2026-11-08. Status: accepted.
- Decision: at most 200 GPU hours in total. Core work is estimated at 153 to 188 hours; the rest is a reserve used first for extra seeds (D018), then for RQ4. Cuts if the projected core exceeds the cap, in order: halve MovieLens-20M tuning trials, drop the MovieLens-20M temporal split, drop Steam.
- Rationale: estimates before measurement (METHODS section 12). About 50 laptop hours per week are available.
- Amendment 2026-11-09: the cap is 500 GPU hours (D034). The cut order is unchanged and applies above it.
- Amendment 2026-11-10: the cap stays at 500 GPU hours, now hours of the rented machine (D045). A cap in USD is set when the projection from the pilot rerun exists; until then only M2 work runs there, at most 15 GPU hours. The 50 hours per week no longer limit the calendar.

## D013 Execution model

- Date: 2026-11-08. Status: accepted.
- Decision: the owner runs one command per batch of runs on the laptop. The runner resumes after interruption and writes configs, metrics, logs and NVML traces to files. An export command writes aggregated CSV files, which the owner pushes. The environment check report (`results/env/`) is committed. Data, checkpoints and per-user outputs stay local. Reports never contain user names, host names or file paths.
- Rationale: keeps the owner's time per batch to a few minutes, and makes every number traceable to committed files.
- Amendment 2026-11-10: batches are started through the `batch` workflow on the rented machine, which also commits the exports (D045). Resuming, skipping finished runs and the export are unchanged.

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

- Date: 2026-11-09. Status: accepted. Implements METHODS 5.4 without deviation; D044 later changes the step the pruner counts.
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
- Method: `seqrec anchors results/laptop/m2-anchor.csv` lists every anchor run on MovieLens-1M.
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
- Amendment 2026-11-10: these are the laptop's results; its exports moved to `results/laptop/`. The rented machine repeats the anchor runs (D045), which is the second anchor read of each arm.

## D033 Seed variance and seed count

- Date: 2026-11-09. Status: method fixed before any pilot result was seen; results pending.
- Method:
  - σ and n: `seqrec seeds results/laptop/m2-pilot.csv --tag sigma --dataset ml-1m` (METHODS 7.5).
  - What the METHODS 5.2 limits cost: `seqrec anchors results/laptop/m2-anchor.csv --pilot results/laptop/m2-pilot.csv`. Per arm, Δ is the mean log validation NDCG@10 of the three seed-variance runs (METHODS 5.2 limits) minus the log validation NDCG@10 of the latest anchor run (the source's limits, D026), and z = Δ/(σ·sqrt(1 + 1/3)).
  - An increase of an arm's limits is recommended when Δ < −log(1.02), the SESOI, and z < −2. The limits stay as in METHODS 5.2 until that recommendation is decided.
  - SASRec-SS has no anchor and is reported without Δ. The METHODS 5.2 limits are longer than the source's for SASRec-CE, SASRec-BCE and BERT4Rec, and much shorter for gSASRec (300 epochs against up to 10,000).
- Results, 2026-11-10, from the rented machine (`results/runs/m2-pilot.csv` followed by `results/runs/m2-gsasrec.csv`, and `results/runs/m2-anchor.csv`; all runs on Pod `we8ppstv5rdldt` at `20b8ed2`, all trained, no failed attempt):
  - Standard deviation of log validation NDCG@10 over 3 seeds: SASRec-BCE 0.02797, BERT4Rec 0.01638, SASRec-CE 0.00776, SASRec-SS 0.00321, gSASRec 0.00289. Pooled σ = 0.01503.
  - METHODS 7.5 gives n = 10, the upper end of the rule: at n = 10 the directional power is 0.305 and the equivalence power 0.759, both below 0.8, so comparisons at this σ are labelled underpowered as 7.5 prescribes. σ is estimated again per dataset after the first 5 final seeds.
  - Cost of the METHODS 5.2 limits, relative difference of the seed-variance mean from the anchor run and z: SASRec-CE +2.5% (1.42), SASRec-BCE −1.9% (−1.08), BERT4Rec +9.2% (5.05), gSASRec −0.6% (−0.32). No arm meets the condition for higher limits. One BERT4Rec seed reached the cap of 1,000 epochs; the anchor run of BERT4Rec stops at its source's 200 epochs.
  - gSASRec under the start rule of D035: the three seeds started at epochs 83 to 85 and stopped by patience at 242, 253 and 264 with validation NDCG@10 0.1830, 0.1827 and 0.1820; none ended with `no_start`.

## D034 GPU budget from the pilot

- Date: 2026-11-09. Status: method fixed before any timing result was seen; results pending.
- Method (`seqrec budget results/laptop/m2-pilot.csv`):
  - Epoch time per dataset and arm: median training plus validation seconds per epoch of the pilot timing runs. Tuned and final configurations are unknown, so the central value is the mean of the anchor size (`time-small`) and the largest size in the search space (`time-large`).
  - Epochs per run: the mean best epoch (counted from 1) of the MovieLens-1M seed-variance runs, scaled by the best-epoch ratio of [5] Table 2, plus the patience, at most the epoch cap. Ratios against MovieLens-1M: MovieLens-20M 46/63 for cross-entropy and sampled softmax, 30/53 for BCE and gBCE, 68/197 for BERT4Rec; Steam 16/63, 39/53 and 74/197; Video_Games, absent from [5], 1. Epoch counts do not fall with dataset size: in [5], MovieLens-20M has 23 times the users of MovieLens-1M and a best epoch of 46 against 63 (SASRec+).
  - One run costs its epochs times the epoch time, plus one evaluation of the restored checkpoint.
  - Blocks: tuning (METHODS 5.4 trial budgets, no credit for pruning); final leave-one-out runs with n seeds (D033); temporal-split runs with 5 seeds, timed as leave-one-out; the shuffle control (SASRec-CE on four datasets, BERT4Rec on two, n seeds); 180 s of warm-up for each of 13 batches; RQ3 at 15 h (METHODS 12); GPU hours already spent, from `overview.csv`. Floor baselines and test reads are left out.
  - Two further scenarios: MovieLens-1M epochs on every dataset; the same with the largest configuration's epoch time.
  - Decision: if the central core exceeds 200 GPU hours, the D012 cuts apply in order until it fits, each recorded here. If only the other scenarios exceed it, nothing is cut; the projection is repeated with measured trial times after MovieLens-1M tuning. That repetition also corrects the epoch count if the gSASRec seed-variance runs stop on the early plateau seen on synthetic data (D027).
- Results, 2026-11-10, from the rented machine (the same files as D033; `seqrec budget results/runs/m2-pilot.csv results/runs/m2-gsasrec.csv`). GPU hours, start rule of D035 included:

  | Block | Central | MovieLens-1M epochs | Largest configuration |
  |---|---|---|---|
  | Spent (M2 batches) | 0.7 | 0.7 | 0.7 |
  | Tuning | 123.7 | 196.7 | 256.9 |
  | Final runs, leave-one-out, 5 seeds | 40.4 | 66.2 | 88.7 |
  | Final runs, temporal split | 40.4 | 66.2 | 88.7 |
  | Shuffle control | 13.7 | 26.1 | 37.4 |
  | Warm-up | 0.7 | 0.7 | 0.7 |
  | RQ3 | 15.0 | 15.0 | 15.0 |
  | **Core** | **234.8** | **371.5** | **488.0** |

  - The central core is within the 500 h cap, and so are both other scenarios: no D012 cut applies. These figures replace the laptop's 416.0 h (468.9 h with the start rule).
  - Seeds beyond 5 (D033 gives n = 10): one extra seed costs 0.4 h on MovieLens-1M, 1.9 h on Video_Games, 2.5 h on Steam and 6.0 h on MovieLens-20M; five extra seeds on every dataset take 54.1 h of the central reserve of 265.2 h and leave 211.1 h.
  - At 0.89 USD per hour (D047) the core is about 209, 331 and 434 USD in the three scenarios, and 257 USD in the central one with the extra seeds.
  - What the figures cover: training and validation time, checkpoints, one final evaluation per run and the warm-ups. A Pod is billed for more: in the pilot rerun's batch the runs' 0.75 GPU hours took 1.50 Pod hours, with 57 short runs, 2.5 to 3 minutes of set-up per job and the start of the Pod.
  - Host: the projection holds for the kind of host the pilot ran on, a Ryzen 9 7950X. On the EPYC 7352 Pod of D046 the two CPU-bound configurations measured there took 3.4 to 5.1 times as long per step, and the request for a Pod cannot name the CPU.
- Amendment 2026-11-09 (before any timing result was seen):
  - The core counts 5 leave-one-out seeds for the final runs and the shuffle control, as METHODS 12 does, and the cut decision uses that core. Seeds beyond 5 from the METHODS 7.5 rule are paid from the reserve in the METHODS 7.5 priority order; one extra seed of a dataset includes its shuffle-control runs. Once a dataset cannot get all its extra seeds, the datasets after it get none.
  - Spent hours are those of the M2 batches only, so a later projection does not count finished work twice.
  - Epoch time includes the checkpoint time per epoch measured by the timing runs, which falls outside the training and validation timers.
  - The BERT4Rec timing runs at the largest size use mask probability 0.4, the upper end of its search range.
  - Warm-ups are recounted after a cut.
- Amendment 2026-11-09 (after the pilot projection was seen, before any tuning run): the owner raised the GPU cap from 200 to 500 hours, all on the laptop, and kept the full scope. The central core is 416.0 h, or 468.9 h with the start rule of D035; the three D012 cuts leave 222.6 h (246.7 h), still above 200 h, and the third cut drops Steam, the only dataset where G2 and G3 have a negative published sign. The D012 cut order stays and applies above 500 h. Before tuning starts on each dataset, the projection is repeated with the measured trial times of the datasets already tuned; if the central core then exceeds 500 h, the D012 cuts apply in order and each is recorded here. `seqrec budget` uses the 500 h cap; METHODS 12 still states 200 h, so this is a deviation from it.
- Amendment 2026-11-10: 416.0 h and 468.9 h are laptop hours. The projection is repeated from the pilot rerun on the rented machine (D045) and replaces them; the method is unchanged.

## D035 Start rule for slow-starting runs

- Date: 2026-11-09. Status: accepted; start limit 300 epochs, set before the gSASRec rerun. Deviation from METHODS 5.2. Decided after the pilot's validation results (D033) and the anchor results (D032) were seen, and before any tuning run.
- Trigger: the D033 cap-cost check recommends higher limits for gSASRec (relative −75.9%, z −2.22). Two of its three seed-variance runs stopped by patience at epochs 44 and 54 with validation NDCG@10 0.0216, 1.04 times MostPop's 0.0207; the third reached 0.1841 and the anchor run 0.1828. No run reached the epoch cap, so patience fired before the runs began to learn. The README of the authors' original code (github.com/asash/gsasrec) describes this pattern for gBCE and advises longer patience; [33] uses patience 200.
- Decision:
  - At the start of every run of a neural arm, the runner computes MostPop's validation NDCG@10 on the run's own dataset and split, from its training split (B1, METHODS 4.4), and records it as `start_reference` in the manifest and summary.
  - A run starts at the first epoch whose validation NDCG@10 exceeds 1.5 times that value. Before it starts, patience does not count; afterwards patience and the epoch cap act as in METHODS 5.2.
  - A run that has not started after `start_limit` epochs (default 300) stops with reason `no_start`. It counts as truncated, so the doubling rule of METHODS 5.2 (D030) covers it, and doubling also doubles the start limit. The code does not double it: a plan with doubled limits sets `start_limit` to twice the default by hand, next to `max_epochs`, `patience` and `guard_hours`. It keeps its best checkpoint and enters every analysis like any other run.
  - The rule applies to all neural arms (A1 to A5). The floor baselines do not train and have start limit 0. Median pruning in tuning counts its steps from the start epoch (D044).
- Effect on existing runs: a run that passed the threshold before patience fired follows the same path under the rule. All twelve seed-variance runs of A1 to A3 and A5, and gSASRec seed 2, ended above 0.14 against a threshold of 0.031, so their results stand. gSASRec seeds 1 to 3 are rerun under the rule in batch `m2-gsasrec`, whose runs replace the pilot's gSASRec seed-variance runs in D033 and D034.
- Analysis inputs: `seqrec seeds`, `seqrec budget` and `seqrec anchors --pilot` accept several exports, and runs in a later file replace earlier runs with the same tag, dataset, split, shuffle and arm. D033 and D034 read `results/runs/m2-pilot.csv` followed by `results/runs/m2-gsasrec.csv`.
- Start limit: the anchor run and seed 2 passed the threshold after about 80 epochs; seeds 1 and 3 stopped by patience at epochs 44 and 54 before passing it, so when they would start is unknown. The source configuration tolerates 200 epochs without improvement (patience 200), and 300 leaves a margin over it. Only runs that never start pay for the extra 100 epochs: per run at the central epoch time of D034, 0.13 h on MovieLens-1M, 0.24 h on Video_Games, 0.69 h on Steam and 1.69 h on MovieLens-20M. For A1 to A4 the limit equals the epoch cap, so a run that starts late trains until the cap and is marked truncated. A run stopped before it would have started enters the analysis near MostPop and dominates σ for its arm and dataset. `test_start_limits_default_by_arm_and_can_be_set` failed before the change (METHODS 11.5).
- Factor: the stuck runs sat at 1.04 times MostPop; after three epochs SASRec-BCE and SASRec-SS stood at 2.3 times and SASRec-CE at 4.8 times, and converged runs at 7 to 9 times. A wrong factor costs epochs but never removes a run.
- Cost: if every gSASRec run starts as seed 2 and the anchor run did, the central D034 projection grows by about 53 GPU hours (gSASRec on MovieLens-1M from 117 to 234 epochs per run).
- Alternatives: the anchor limits for gSASRec (10,000 epochs, patience 200) add about 164 hours and stretch the pruning warm-up to 1,000 epochs; a lower t is already tuned over [0.5, 1.0], and changing it in the seed-variance runs would leave the anchor configuration; dropping stuck runs selects on the outcome.
- Tests: the start-rule tests in `tests/test_runner.py` and `test_a_later_export_replaces_the_runs_it_repeats` failed before their changes (METHODS 11.5).
- Run directories and checkpoints from before this rule, and a later export with unfinished runs: D040.
- Amendment 2026-11-10: the gSASRec rerun did not run on the laptop. The pilot and the rerun both run on the rented machine (D045), so the two files read here come from one machine.

## D036 Time counted by the guard

- Date: 2026-11-09. Status: accepted. Clarifies the wall-clock guard of METHODS 5.2. Fixed after the pilot timings were seen (D034) and before any M3 run.
- Decision: the 8-hour guard counts the training and validation time of every epoch, summed over all sessions of a run. Checkpoint writes, the set-up before the first epoch (data loading, invariant checks, model construction), the final evaluation of the restored checkpoint and any time between sessions are not counted. The check runs before each epoch, so a run can pass the guard by at most one epoch. Doubled limits (METHODS 5.2) double this time. The runner has counted this way since M1; `test_the_guard_counts_training_and_validation_time_only` fixes it.
- Rationale: an interrupted run gets the same budget as an uninterrupted one, and disk speed does not decide when a run stops. Checkpoint writes took about 1% of epoch time in the pilot, so the choice moves guard stops by about that much.

## D037 Triton FMA fusion fixed on

- Date: 2026-11-09. Status: accepted. No training run compiles code, so no result depends on it.
- Decision: importing `seqrec` sets `TRITON_DEFAULT_FP_FUSION=1` and overrides any inherited value. Every entry point imports the package before anything is compiled, including the `torch.compile` worker of `scripts/check_env.py`. `tests/test_package.py` failed before the change (METHODS 11.5).
- Rationale: fusion is already the default of the locked triton-windows 3.9 (D003), so behaviour does not change. Fixing it keeps a later Triton release or a stray environment variable from changing the rounding of compiled kernels unnoticed, which the M4 `torch.compile` measurements (METHODS 9.1) would absorb.
- Amendment 2026-11-10: the variable is set the same way on Linux, where `torch.compile` uses the Triton that the `cu132` wheel depends on (D045). That fusion is also that Triton's default is not verified here; it is checked on the rented machine before the M4 `torch.compile` measurements.

## D038 Slow-run flag

- Date: 2026-11-09. Status: accepted. Adds a flag to METHODS 8 and 7.6. Decided after the pilot timings were seen and before any M3 run.
- Decision: export marks `slow_epochs` when a run's median training plus validation time per epoch exceeds 1.5 times the median of the other runs with the same batch, dataset, split, shuffle, arm, parameters and tag, that is, its other seeds. Runs without such runs (tuning trials, timing runs) get no value. RQ2 timing analyses treat a flagged run like one with a hardware slowdown: it stays in the primary analysis and is left out of the repeat of METHODS 7.6. The flag tests failed before the change (METHODS 11.5).
- Rationale: in the pilot, gSASRec seed 2 took 30.6 s per training epoch against 1.7 to 1.9 s for its other seeds and the anchor run, at a higher SM clock (median 2,917 against about 2,000 MHz) and with no hardware slowdown event. Neither flag of METHODS 8 catches such a run, and its times would enter time to target (H2a) unmarked.
- Limitation: tuning trials have no other seeds, so a slowed trial is not flagged; under the guard (D036) it can be marked truncated and count toward the doubling rule of METHODS 5.2.
- `test_export_flags_the_slow_seed` checks the exported values of the flag (D040).
- Amendment 2026-11-10: the flag stays. The slow stretches that motivated it came from the Windows driver (D042) and are not expected on Linux.
- Amendment 2026-11-10, after the owner's decision in D046: the flag compares a run only with its other seeds made on the same Pod, in place of the same host signature. How a run records its Pod is in D046.

## D039 Step-time profiler

- Date: 2026-11-09. Status: accepted. A measurement tool outside METHODS; no analysis uses its output.
- Decision: `scripts/profile_step.py` trains one arm for a few steps on one prepared dataset through the arm's own epoch loop and prints wall time, process CPU time and GPU kernel time per step. It reads only the training split and writes nothing. `tests/test_profile_step.py` failed before the script existed (METHODS 11.5).
- Rationale: in the pilot timing runs every arm took about 7 ms per step on Steam and Video_Games, at both configuration sizes, which points to a fixed cost per step outside GPU compute. What a change to the training loop could save depends on how much of a step the GPU is busy, which only the GPU machine can measure.
- GPU time counted once per kernel, per-step times without the cost per epoch, and `--from-run`: D040.
- The script prints a note above its JSON line when the busy share exceeds 1, since GPU time and wall time come from separate passes and the share then only says that the GPU limits the step, and when the cost per epoch is negative, since a one-step epoch then ran faster than a step of the long epoch and the split between the two is not meaningful (`test_results_outside_the_assumptions_carry_a_note`).
- Amendment 2026-11-10: the step times here were measured on the laptop under Windows. Kernel-launch cost differs on Linux, so the profile is measured again on the rented machine (D045) before the execution mode is decided.

## D040 Fixes from the M2 close review

- Date: 2026-11-09. Status: accepted. Code changes after the anchor reads (METHODS 11.5); each came with a test that failed before it. Decided before any M3 run and before the gSASRec rerun of D035.
- Run directories claimed before D035 have no `start_limit` in `config.json` and trained without the start rule. They match a plan with the arm's default limit (a plan written before D035) or with 0, so interrupted batches resume, and such a run finishes with the rule off; its summary records start limit 0. Any other limit is a conflict. A run whose start limit differs from its directory, including after a change of the default, is refused, and the message names the field. A negative start limit is rejected. `configs/plans/m2-anchor.toml` now states `start_limit = 0`, which its `r1` runs match (`test_run_directories_from_before_the_start_rule_are_not_conflicts`, `test_a_run_from_before_the_start_rule_accepts_only_its_own_limits`, `test_a_run_from_before_the_start_rule_finishes_without_it`, `test_a_changed_default_start_limit_stops_an_interrupted_run`, `test_anchor_runs_train_without_the_start_rule`).
- Every `last.pt` of a run under the rule holds its start record; a resumed checkpoint without one is refused instead of being treated as not started. Summaries record the start limit the run used, and export reads it from there (`test_a_checkpoint_without_its_start_record_is_refused`; `test_an_interrupted_run_keeps_its_start_rule_state` covers a run interrupted before it started).
- `scripts/profile_step.py` (D039) summed the device time of CPU ops and of the kernels they launch, so it counted GPU time about twice. It now sums only GPU events (kernels, copies, memsets) that are not user annotations, printed as `gpu_ms_per_step` (`test_gpu_time_counts_each_gpu_event_once`). Per-step times are the difference between an epoch of n steps and the median of five one-step epochs, divided by n − 1, and the fixed cost per epoch is printed apart; n defaults to 200, must be at least 2 and counts full batches only, and a step time at or below zero stops the script (`test_time_per_step_leaves_out_the_cost_per_epoch`).
- `scripts/profile_step.py --from-run <run directory>` reads dataset, split, arm, seed and parameters from that run's `config.json`, so profiling commands carry no JSON, whose quoting differs between Windows PowerShell 5.1 and PowerShell 7 (`test_a_run_directory_supplies_the_configuration`).
- `seqrec seeds`, `anchors` and `budget` stop with status 1 when an export after the first holds a run that did not finish training. They also stop when a group of a later file holds other seeds than the group it replaces, as when a batch stopped before a seed's run began. Such a file replaces whole groups (D035), so a failed or missing seed would have lowered n and changed σ without notice (`test_a_rerun_with_an_unfinished_run_is_refused`, `test_a_rerun_missing_a_seed_is_refused`).
- The NVML sampler also records device memory in use (`nvmlDeviceGetMemoryInfo`, the whole device, other processes and the display included). Run summaries give its distribution and export its peak as `nvml_mem_used_mib_max`, next to the PyTorch allocation peak per epoch, so a run that came close to the 16 GB can be found. Samples written before this change have no memory field and are skipped in the summary (`test_the_sampler_records_device_memory_and_its_peak`, `test_export_lists_peak_device_memory`, `test_samples_from_before_memory_sampling_still_summarize`).
- `seqrec test` compares the training and full-data fingerprints with the run's manifest before the test read and refuses a run whose data changed since training; the refusal is not a read and not a failure (D031). A resumed training session now compares the full-data fingerprint as well as the training one, so the manifest that `seqrec test` checks holds the data of every session (`test_a_run_is_not_tested_on_data_other_than_it_trained_on`, `test_a_rebuilt_dataset_is_refused_before_the_test_read`, `test_a_resumed_run_refuses_changed_evaluation_data`).
- `test_the_guard_leaves_out_set_up_and_adds_up_sessions` extends the D036 test to set-up time and an interrupted run.

## D041 Memory records and cache release between runs

- Date: 2026-11-09. Status: accepted. Not covered by METHODS; no result depends on it. Code changes after the anchor reads (METHODS 11.5); each came with a test that failed before it. Decided before the gSASRec rerun of D035 and before any M3 run.
- Trigger: gSASRec runs on MovieLens-1M slowed from about 1.9 s to 36–47 s per training epoch for long stretches (anchor run epochs 0–10 and 299–383, seed 2 epochs 44–217). Throughout, the GPU was busy, its SM clock at the maximum, its power near 80 W against 120 W in normal epochs, with no thermal or power-cap event, and validation slowed as much. The PyTorch allocation peak stayed at 4.1–4.6 GiB. One explanation, not yet confirmed, is that part of the run's memory lay in system RAM. gBCE gathers the output embeddings of 256 negatives for every position, about 2 GB in fp32 for the anchor configuration, and the size changes with every batch. The allocator caches such blocks, and on Windows, where the driver by default places memory in system RAM once device memory is full, an allocation past the device does not fail, so the allocator never frees its cache and retries.
- Decision:
  - Every epoch records `train_peak_reserved_gib`, the peak of memory PyTorch reserved, cached blocks included, next to `train_peak_memory_gib`. Summaries give the largest value of each over the run (`train_peak_memory_gib_max`, `train_peak_reserved_gib_max`), and export lists them as `torch_allocated_gib_max` and `torch_reserved_gib_max`, next to `nvml_mem_used_mib_max` (D040). Runs from before this change export no value (`test_epochs_record_the_memory_pytorch_reserved`, `test_runs_summarize_and_export_their_memory_peaks`, `test_runs_without_memory_records_summarize_to_none`).
  - Each run and tuning trial starts by collecting unreachable objects and emptying the CUDA cache, so blocks cached by an earlier run in the same process go back to the driver before the run builds its model; the check after a failed attempt (D031) uses the same step (`test_each_run_releases_cached_device_memory_before_it_builds_its_model`, `test_releasing_device_memory_empties_the_cuda_cache_only`).
- Not decided here: the driver setting that keeps CUDA memory out of system RAM, and a gBCE implementation without the per-position gather. Both wait for a diagnosis on the GPU machine.
- Amendment 2026-11-10: the two open points are closed in D042.

## D042 Device memory spilling into system RAM

- Date: 2026-11-10. Status: closed without the driver-setting test; GPU runs moved to Linux (D045). Code changes after the anchor reads (METHODS 11.5); each came with a test that failed before it. Decided before the gSASRec rerun of D035 and before any M3 run.
- Diagnosis: on the GPU machine the driver's "CUDA - Sysmem Fallback Policy" was at its default, which lets CUDA allocations go to system RAM once device memory is full. Two profiling runs of the gSASRec anchor configuration on MovieLens-1M (`scripts/profile_step.py`) took 410 to 480 ms per step. Within 2 s of the start the process filled the 16 GB of device memory and about 12 GiB of shared system memory, and held both until it ended; meanwhile SM use was 95–100%, memory-controller use 1–4% and PCIe traffic 30–55 GB/s each way, against under 0.2 GB/s at idle. The PyTorch allocation peak of the same configuration in training was 4.6 GiB, so most of the memory held was the allocator's cache: gBCE gathers a (positions × 256 × d) tensor whose size changes with every batch, a request larger than every cached block makes the allocator ask the driver for a new one, and with the fallback that request never fails, so the allocator never frees its cache and retries (D041).
- `scripts/profile_step.py` empties the CUDA cache and resets the allocator's statistics before it builds the model, and prints the peaks of memory PyTorch allocated and reserved, the number of device allocations and the number of retries after a failed one (`torch_allocated_gib_max`, `torch_reserved_gib_max`, `device_allocs`, `alloc_retries`), so a reserved peak far above the allocated one can be read off (`test_memory_readings_separate_memory_in_use_from_cached_blocks`).
- gBCE (A4) no longer gathers the embeddings of every position's negatives at once. `losses.negative_logits` computes the (M, K) negative scores through the smaller of two intermediates: the scores of all items, (M, rows of the output table), from which each position's negatives are picked, when the table has at most K × d rows; otherwise the negatives' embeddings, (M, K, d). Positions are processed in slices whose intermediate holds at most 2²⁶ elements (256 MiB in fp32), recomputed in the backward pass as for chunked cross-entropy. Intermediates still change size from batch to batch below that bound, but never exceed it. The loss of METHODS 4.3 is unchanged and still computed in fp32 with autocast off; only the order of operations, and with it the rounding, differs. Both forms are fp32 matrix products under the TF32 setting of METHODS 5.1. Whether cuBLAS used TF32 for the old form's many small products was not checked; the one large product of the new form will use it, which rounds negative scores at about 1e-3 relative, within what METHODS 5.1 accepts for training.
- Path by dataset over the search space (K ∈ {64, 128, 256}, d ∈ {64, 128, 256}, so K × d from 4,096 to 65,536; the table has N + 1 rows), on leave-one-out: MovieLens-1M (N 3,416) always scores all items; Steam (N 11,961) scores all items when K × d ≥ 16,384 and gathers embeddings below; MovieLens-20M (N 18,345) and Video_Games (N 25,612) score all items when K × d ≥ 32,768 and gather embeddings below. On the temporal split N is the training-period vocabulary (`results/data/stats.csv`), which moves MovieLens-20M (N 14,834) to scoring all items from K × d 16,384; the other datasets keep their thresholds. A batch has at most 128 × L positions (25,600 on the MovieLens sets, 6,400 on Steam and Video_Games); slicing starts above 2²⁶ / min(N + 1, K × d) positions. For the anchor configuration on MovieLens-1M (d 128) the largest intermediate falls from about 3.4 GB to at most 256 MiB.
- Effect on existing results: in fp64 the new and old forms agree in loss and gradients to 1e-12, so the gSASRec anchor run and seed 2, trained with the old form, stand. `test_gbce_never_holds_a_negative_embedding_per_position` failed before the change (METHODS 11.5); `test_gbce_paths_match_the_gathered_form_in_value_and_gradient` covers both paths with one slice and with several; `test_negative_logits_take_the_smaller_intermediate`, `test_negative_logits_slices_stay_within_the_budget` and `test_sliced_gbce_under_bf16_autocast_matches_fp32` fix the path choice, the slice bound with a partial last slice, and autocast staying off when slices are recomputed. The fp64 agreement says nothing about TF32 rounding on the GPU.
- Closure 2026-11-10: this was behaviour of the Windows driver. The old gBCE built large tensors of a different size for every batch, the driver's system-memory fallback let the allocator's cache grow without ever failing, and training took 410 to 480 ms per step. The planned comparison with the fallback switched off was not run, because GPU runs moved to a Linux machine (D045), where an allocation past device memory fails and the allocator frees its cache and retries. The gBCE change above stays: it bounds the intermediate tensors on every platform.

## D043 Execution-mode prototype in the step profiler

- Date: 2026-11-10. Status: accepted as a measurement tool only; no training run uses these modes, and the choice of execution mode for M3 is open (TASKS, before M3 tuning).
- Decision: `scripts/profile_step.py` takes `--fixed-shapes` (the arm's loss over all B × L positions with unlabelled positions weighted 0, so no step reads a shape from the GPU), `--fused-adam` (`torch.optim.Adam(fused=True)`) and `--cuda-graph` (forward, backward, clipping and the Adam step of a full batch captured once and replayed on static copies of each batch, with the model's generator registered with the graph and the last partial batch eager; needs `--fixed-shapes`). They apply to A1 to A4 and live in `scripts/step_modes.py`. `--sync-debug` adds the number of CPU-GPU synchronisations per step, and `--check` runs the same steps on three batches with dropout 0 under the modes and under the arm's own code, from the same initial state, and prints the largest differences of the losses and of the final weights, next to the same differences between two runs of the arm's own code; under the graph two of the batches are replayed on a batch other than the captured one. Under fixed shapes gSASRec draws K negatives for every position, so its check keeps fixed shapes in the reference and tests only fused Adam and the graph; a check whose modes equal its reference is refused.
- What the numbers cover:
  - Fixed shapes compute the loss at all B × L positions, several times the labelled ones on short sequences, so their step time is the cost of this implementation and not of removing the shape read alone. gBCE scores the negatives of all positions in one pass, without the slices of D042, so that the fixed-shape step carries no recomputation the arm's own step lacks and no generator state is stashed inside a captured graph; a non-zero `ce_chunk` is refused.
  - Under a CUDA graph the GPU time covers the kernels the profiler reports; `gpu_events_per_step` is printed so that missing replayed kernels show against the run without the graph. The allocator counts of a graph run include the capture (the cache is emptied when it starts, and the warm-up steps run on a side stream) and are not compared with other modes.
  - The synchronisation count runs the backward pass on the calling thread, where its warnings are caught; a run without modes gives the reference count.
  - The timings and memory readings are printed in a `timing` line before the profiled pass, a timed epoch with a non-finite loss stops the script, and the graph is captured on copies of its first batch. Before a capture the profiler is initialised and CUPTI kept loaded, which is PyTorch's workaround for compiled graphs on CUDA before 12.6, applied here on every version.
  - Readings that are not measurements carry a note: no GPU events for the added steps, a profiler that stopped collecting GPU events early, and a synchronisation count of 0 in an epoch that reads its loss (`syncs_one_step_epoch`). The output names the device. `--warmup 0` skips the warm-up epoch.
  - `--check`: the loss difference is the one to read. Adam turns rounding differences in near-zero gradients into weight differences up to the size of the learning rate, so the weight difference of correct code can exceed that of two runs of the arm's own code.
- Tests: `tests/test_step_modes.py` failed before the modes existed, and the tests of the items above before each change (METHODS 11.5). The losses under fixed shapes equal the arms' own losses given the same negatives; the CUDA graph and the synchronisation count need a GPU and are checked by `--check` and `--sync-debug` on the GPU machine.
- Amendment 2026-11-10: the prototype is measured on the rented machine (D045); no laptop measurement of it enters a decision.

## D044 Pruning steps count from the start epoch

- Date: 2026-11-10. Status: accepted by the owner before any tuning run. Deviation from METHODS 5.4.
- Trigger: METHODS 5.4 prunes from 10% of the epoch cap, epoch 30 for A1 to A4. gSASRec's anchor configuration and seed 2 of its seed-variance runs stayed near 1.04 times MostPop until about epoch 80 (D035). Compared by epoch, trials inside that plateau differ only by noise, and a trial that starts later than the completed ones is pruned for when it starts, which has no known relation to where it ends.
- Decision: a trial reports to the pruner only from its start epoch (D035) on, with the epochs since then as the step, so the start epoch is step 0. Before it starts it reports nothing and cannot be pruned; `start_limit` bounds that time. The pruner stays median pruning after 5 complete trials, with a warm-up of 10% of the epoch cap, now counted in steps, and a step prunes only once 3 complete trials have a value at it (`n_min_trials`). The rule is the same for all neural arms; A1 to A3 and A5 pass the start threshold within their first epochs (D035), so their steps shift by a few epochs. A run without the start rule counts from epoch 0.
- The pruner as used: Optuna 5.0.0 `MedianPruner` (`pruners/_percentile.py`) prunes a trial whose best value so far is below the median, over complete trials, of their values at the same step. Complete trials without a value at that step are left out, and with none the trial is not pruned; pruned trials do not enter the median. The median stopping rule comes from Vizier (Golovin et al., KDD 2017, section 3.2.2), which compares against the median of the completed trials' running averages up to the step; Optuna's single-step values make its rule the stricter one on rising curves. Neither source covers trials that start after a delay.
- Consequences: a trial that never starts runs to its start limit (300 epochs) and ends with `no_start`, as D035 already allows; per such gSASRec trial at the central epoch times of D034 that is about 0.4 h on MovieLens-1M, 0.7 h on Video_Games, 2.1 h on Steam and 5.1 h on MovieLens-20M. It is a complete trial, so it counts toward the 5 that enable pruning, and having reported no step it is in no median. Without a minimum, 4 such trials among the first 5 leave a later trial compared with a single curve and pruned for any shortfall against it at step 30, hence the 3 trials per step. The budget projection is redone with D034 after the execution-mode decision.
- Known and left as they are:
  - Optuna's TPE ranks complete trials ahead of pruned ones, so a `no_start` trial enters its good set with its plateau value before a pruned trial that reached more. Whether to act is decided from the share of trials that start in the first gSASRec study (TASKS, M3).
  - An epoch repeated after a crash is not bit-identical on a GPU. If the crash falls between the report of the start epoch and its checkpoint and the repeated epoch misses the threshold, step 0 keeps the lost epoch's value and later steps shift by one; Optuna keeps the first value of a step, and the tuning code silences its warning about repeated steps.
- Alternatives: a longer warm-up for gSASRec alone removes the noise comparisons but still prunes late starters against early ones, gives one arm its own tuning rule, and at 300 epochs disables pruning for it; Optuna's `PatientPruner` around the median pruner passes a trial on to the inner pruner when it has not improved for some steps, which is the state of a trial in the plateau.
- Tests: `test_pruning_steps_count_from_the_start_epoch`, `test_a_resumed_run_reports_its_earlier_steps_again` and `test_a_trial_that_never_starts_completes_without_pruning_steps` failed before the change (METHODS 11.5). `test_trials_that_never_started_enable_pruning_but_stay_out_of_the_median` fixes the Optuna behaviour the decision relies on and failed before the minimum of 3 trials per step. `test_a_trial_stopped_at_its_start_limit_is_a_complete_trial` covers a trial that ends with `no_start`.

## D045 GPU runs move to a rented machine

- Date: 2026-11-10. Status: accepted by the owner. Deviation from SPEC sections 1, 4, 8, 9 and 10 and from METHODS 9.2 and 12. Made after the M2 pilot and anchor results from the laptop were seen, and before any tuning run, final run or RQ3 measurement. How the machine is released, "One machine for all timing" and the admission test are replaced by D046; the GPU model and its price by D047.
- Decision: every GPU run from here on happens on one rented machine: an RTX 5090 (32 GB, 16 vCPUs in the catalogue) in Runpod Secure Cloud, on demand and not interruptible, in a Linux container, at 1.19 USD per hour [34]. The laptop makes no further measurement.
- Reason: run times on the laptop were unstable under the Windows driver, and diagnosing that took longer than the runs themselves (D042). The remaining plan, about 470 projected laptop hours at about 50 hours per week, would also keep that one machine busy for more than nine weeks.
- What stays: research questions, hypotheses, datasets, models, splits, the tuning and seed rules, the statistical analysis and every decision rule. The GPU is still a consumer card, so the cost claims of RQ2 and RQ3 are still about consumer hardware; they now describe a desktop card in a data centre next to a server CPU, and the report says so.
- Laptop results:
  - Kept as they are, because they do not depend on the machine: the anchor quality inside the published bands (D032), the start rule and its basis (D035), the pruning rule (D044), the data statistics.
  - Measured again on the rented machine: the whole M2 pilot, the anchors and the gSASRec rerun of D035 (57, 4 and 3 runs; the first two took 4.3 GPU hours on the laptop), so that every run of the study comes from one machine. σ and the seed count (D033), the budget projection (D034), the per-step profile (D039) and the execution-mode prototype (D043) are computed from the new runs.
  - The laptop's exports are in `results/laptop/`. `results/runs/` holds the rented machine's; `overview.csv` and `tuning_best.csv` in a directory are computed from the batch files in that directory, so the two machines are never summed.
  - Anchor reads are counted across both directories (METHODS 11.5 allows three per arm): the laptop made one per arm, so the rerun is each arm's second. `seqrec test` and `seqrec export` add the reads in `results/laptop/` to those on the machine and stop if that directory is missing.
  - The exports of a volume that is lost are set aside the same way: on `main`, everything in `results/runs/` (batch files and their summaries) moves to a directory of its own, `results/<name>/`. Every directory under `results/` other than `runs`, `data` and `env` is read as an earlier export, so no job has to name it. Until the files are moved, the job's check after the sync, `seqrec test` and `seqrec export` all stop: a batch file in `results/runs/` that records a test read the volume has no log of would be counted nowhere, and the next export would replace it. A final run read on the lost volume is not read again; an anchor read made there counts toward the three. A read that is both on the volume and in an earlier export is counted twice; that can refuse a read, never allow one.
- Platform (replaces SPEC section 8 for GPU runs): Linux container on the host's NVIDIA driver, branch 580 or newer; PyTorch 2.15.0 `cu132` wheels through the `cu132` extra of the lock file (D003); `torch.compile` through the Triton that the Linux wheel depends on. `scripts/check_env.py` is run once on the machine and its report committed under `results/env/`. The container cannot choose the driver; a machine below branch 580 is not used.
- Execution (replaces SPEC section 9): a batch is one manual run of the `batch` workflow on a self-hosted runner on the machine (`scripts/cloud/`). The workflow takes a plan file from `configs/plans/`, runs it, exports, and commits the batch's file and the summaries only when the plan finished without a failure. Its `export` task commits a batch as it is on the machine, for a plan that cannot finish because a run was given up (D031). Run directories, datasets and checkpoints stay on the machine's volume. The machine stops itself when a job ends, and after 30 minutes without a job.
- Access: the workflow starts by hand only and runs from `main` only. A job on the machine runs as root, so workflows of pull requests from outside contributors require the owner's approval in the repository settings, and that approval is never given.
- One machine for all timing: RQ2 and RQ3 times are compared only within one machine instance (the same GPU, host and CPU). Each job records GPU model, driver, CPU model, vCPU count and kernel. If the instance is lost, timing runs already made on it are repeated on the new one or reported separately, and its exports are set aside as the laptop's were; quality results are not affected.
- Admission test before any timing run, because the CPU is shared with other tenants and small configurations are sensitive to it (D039): two configurations (SASRec with BCE on Steam; gSASRec on MovieLens-1M), 5 processes each of `scripts/profile_step.py`. The spread of the step time, (max − min) / median, must be at most 5% for each; the test is repeated at least 6 hours later after a stop and start, and the two medians must differ by at most 5% with the same CPU model and vCPU count. A machine that fails is not used for timing.
- Deviation from METHODS 9.2 (conditions): "native Windows, AC power, power mode recorded, vendor performance profile fixed and recorded, other applications closed" is replaced by the machine above with nothing else running in the container. NVML sampling, the 5 randomised repeats, both regimes and the exclusion rule are unchanged. The cool-start regime was written for the laptop's thermal slowdown after about 40 s (D011); it is measured as written and reported, whether or not this machine shows such a slowdown. The clock lock is tried as written; a container that may not set it counts as "does not work".
- Deviation from METHODS 12 (budget): hours in METHODS 12 and D034 are laptop hours. The cap is 500 GPU hours on the rented machine. A cap in USD is set when the D034 projection has been repeated from the rerun pilot; until then only M2 work runs, at most 15 GPU hours. The D012 cut order is unchanged and applies above the cap. The 8-hour guard of METHODS 5.2 is unchanged.
- Alternatives: staying on the laptop (no cost, but the instability above and nine weeks of the owner's only machine); an interruptible or community-hosted machine at a lower price (the host can change or disappear between runs, which breaks the one-machine rule); a dedicated server with a workstation GPU (fixed CPU and the cleanest timing, kept as the fallback if the admission test fails); a data-centre GPU (not consumer hardware).
- Limits, known and left as they are:
  - Plan files: a plan with an empty `seeds` list or `trials = 0` expands to no work, and `seqrec export --plan` treats it as finished; `seed` written next to `seeds` is ignored; an `order` other than `"shuffled"` is read as `"listed"`; a plan whose only table is a misspelt `[[studies]]` is refused with the key list of a run plan.
  - No workflow test notices `continue-on-error` on the step that checks the recorded test reads.
  - Earlier exports are found only in directories directly under `results/`. A batch file in `results/` itself, in a nested directory, in `results/env` or `results/data`, or with another extension than `.csv` is not read. Deleting the files of `results/runs/` instead of moving them passes every check and shows only in the git history.
  - A job's read checks see `main` as it was when the job was triggered. A job that waits across a volume replacement can miss a read committed in between, so queued jobs are cancelled before a volume is replaced.
  - A test read that was never committed (a `test` job that failed after reading) is recorded on the volume only and is lost with it.

## D046 A network volume and a new machine for every batch

- Date: 2026-11-10. Status: accepted by the owner. Replaces three parts of D045: how the machine is released, "One machine for all timing" and the admission test. The rest of D045 stands. Made before any run on a rented machine.
- Reason: a Pod with a volume disk stays on the host it was created on, and after a stop it starts again only while that host's GPU is free. The RTX 5090 was at low stock in a single data centre, and three attempts to create such a Pod were refused for lack of capacity.
- Decision:
  - One network volume, mounted at `/workspace`, holds everything that outlives a machine: run directories, datasets, the Python environment, the runner and its registration.
  - Every batch runs on a new Pod, created from `configs/cloud/pod.json` on any free host of the volume's data centre and deleted when its jobs have ended. A stopped Pod would keep its host, so nothing is stopped and started again.
  - One Pod at a time: a Pod is created only while the account has none. Two Pods writing to the volume can corrupt it, and they would share one runner registration.
  - Two guards against a Pod that keeps running. On the Pod, the watchdog (`scripts/cloud/pod_start.sh`) releases it when a job asks for it, after 30 minutes without a job, and when one job outlasts the workflow's limit. The Pod's start command fetches the watchdog from `main` at every start; the volume holds no copy that could go out of date. Off the Pod, whoever creates one schedules a check after the batch's expected end and deletes the Pod if it is still there. The account does not top up by itself, so its balance bounds a loss that passes both.
  - The runner is registered once, on any Pod that mounts the volume; a GPU is not needed for that, and every job checks the driver itself. Later Pods reuse the registration. GitHub removes a runner that was offline for 14 days; it is then registered again.
  - Test reads are counted per volume, as in D045; a new Pod changes nothing there.
- Timing (replaces "One machine for all timing"). A new Pod can be on another host with another CPU, and small configurations are bound by the CPU (D039).
  - RQ3 sessions, and every comparison that is a ratio of step times: all measurements of one comparison are made within the lifetime of one Pod.
  - RQ2 run times are compared only between runs with the same host signature: GPU model, driver branch, CPU model, CPU count and data centre. Every run records it, and the batch file has it in the column `host`. A run that trained on more than one signature lists them all; its times are reported separately. Quality results are not affected. The slow-run flag of D038 compares a run only with its other seeds that have the same signature.
- Admission test (replaces the one in D045), before any timing run: the same two configurations, 5 processes each of `scripts/profile_step.py`.
  - Within one Pod, the spread of the step time, (max − min) / median, must be at most 5% for each configuration.
  - On a second Pod, created at least 6 hours after the first was deleted and with the same CPU model and CPU count, each configuration's median must differ from the first Pod's by at most 5%. A second Pod with another CPU is deleted and created again, at most 3 times; every signature seen is recorded.
  - If two Pods with one signature differ by more than 5%, run times are not compared across Pods in this data centre. The owner then chooses between timing RQ2 within one Pod's lifetime and the dedicated server of D045.
- Costs accepted: time comparisons across batches rest on the signature and the admission test, not on one physical machine; the volume is billed for as long as it exists (50 GB at 0.07 USD per GB per month) and ties the work to its data centre; every batch pays a few minutes of start-up.
- Alternatives: the volume disk of D045 (creation refused; every later start depends on one host); a volume without a data centre (no file locks and no atomic rename, and run files are written by renaming); stopping instead of deleting (the stopped Pod keeps its host).
- Limits, known and left as they are:
  - A stop request is not tied to the job that made it. With three jobs queued and a short one in the middle, the third can be left with 0 to 60 seconds to start before the Pod is deleted.
  - The runner keeps listening until the Pod is gone, so a job that starts while the delete request is under way is cut off.
  - A Pod that cannot reach GitHub at its start runs no watchdog until a fetch succeeds; until then only the check off the Pod guards it.
- Amendment 2026-11-10: the CPU count of a signature is the number of CPUs allocated to the container: its cgroup CPU quota where one is set, and never more than the CPUs it may run on. A container can run on every thread of its host: the first Pod of the admission test had 32, a nominal count of 16 and a quota of 13.6 CPUs, which the signature rounds up to 14. Counting threads would give Pods with different allocations on one kind of host the same signature. "The same CPU count" in the admission test means this number. `test_the_cpu_count_of_a_signature_is_the_containers_quota` failed before the change.
- Admission test, first Pod, 2026-11-10. Pod created 06:16:57Z and gone by 06:30:28Z; workflow runs 11 and 12 at `ef2890f`; signature `NVIDIA GeForce RTX 4090 | 595 | AMD Ryzen 9 7950X 16-Core Processor | 14 | EU-RO-1`, kernel 6.8.0-147. Wall time per step in ms, five processes in the order they ran:
  - Steam, SASRec with BCE, 200 steps per process: 3.003, 3.080, 3.043, 3.075, 3.157. Median 3.075, spread 5.01% from the values as printed. The script printed three decimals, which leaves the spread between 4.97% and 5.04%; at that precision it cannot be decided against 5%, and no verdict is recorded here.
  - MovieLens-1M, gSASRec: 4.507, 4.619, 4.693, 4.712, 4.708. Median 4.693, spread 4.37%. Each process timed 47 steps: the script times full batches only, up to 200, and an epoch of this dataset has 47.
  - Observation, cause not examined: in both configurations the first process was the fastest and the later ones were slower.
  - The script now prints step times to the nanosecond (`test_step_times_keep_the_digits_that_decide_a_five_percent_spread` failed before).
- Owner's decision, 2026-11-10, made after the first Pod's result above was seen and before any further Pod. The criteria are unchanged: both limits stay at 5%, computed as before.
  - The first Pod stays recorded as undecidable at the precision it printed.
  - The test within one Pod is repeated once, in full, on a new Pod: both configurations, with step times printed to the nanosecond. That repeat is the final result for both configurations, whatever it shows, and is not repeated again. Its Pod is the one the second Pod's medians are compared with.
  - The test did not say what follows when a spread within one Pod exceeds 5%. If either configuration does so in the repeat, there is no further attempt. The owner then chooses between adding one warm-up process that is not counted, recorded as a deviation, and comparing run times only within one Pod's lifetime.
  - Both results, the first Pod's and the repeat's, stay in this entry.
- Admission test, the repeat, 2026-11-10: not passed. Pod created 06:57:16Z and gone by 07:12:46Z; workflow runs 13 and 14 at `127907b`; signature `NVIDIA GeForce RTX 4090 | 595 | AMD Ryzen 9 7950X 16-Core Processor | 32 | EU-RO-1`, kernel 6.8.0-138. Wall time per step in ms, five processes in the order they ran:
  - Steam, SASRec with BCE, 200 steps per process: 3.395079, 3.368398, 3.091894, 3.339909, 3.105944. Median 3.339909, spread 9.08%, above 5%.
  - MovieLens-1M, gSASRec, 47 steps per process: 4.813682, 4.821271, 4.899424, 5.040219, 4.784903. Median 4.821271, spread 5.30%, above 5%.
  - Both configurations exceed the limit, and under the owner's decision above this result is final. No second Pod was created. The choice between the two options is the owner's and is open.
  - In neither configuration was the first process the fastest, unlike on the first Pod.
  - The CPU count of this signature is not comparable with the first Pod's 14. This container had no `/sys/fs/cgroup/cpu.max`, so the count is the 32 CPUs it may run on; the platform's nominal count was 16, as on the first Pod. Whether the container was limited in another way was not read.
- Amendment 2026-11-10, after the repeat: the quota of a signature's CPU count is read from cgroup v2 (`cpu.max`), then from cgroup v1 (`cpu.cfs_quota_us` and `cpu.cfs_period_us` under `/sys/fs/cgroup/cpu` or `/sys/fs/cgroup/cpu,cpuacct`), and only without a quota in either is the count the CPUs the container may run on. The platform's nominal count is printed for comparison and not used. Whether the repeat's host limits its containers through cgroup v1 is not known; the `machine` step now prints the source of the count with the raw values and the type of the cgroup filesystem, so the next Pod's job log will show it. No run on a rented machine had recorded a signature before this change. `test_a_cgroup_v1_quota_counts_like_a_v2_quota` failed before it.
- Owner's decision, 2026-11-10, made after the repeat's result above was seen and before any further Pod. The admission test stays recorded as not passed. Its criteria are unchanged and it is not run again.
  - Timing. Run times are compared only between runs made within the lifetime of one Pod. Times from different Pods are reported and not compared. This replaces the same-signature rule for RQ2 and for the slow-run flag of D038; RQ3 already measured each comparison within one Pod. No second-Pod comparison is made. The uncounted warm-up process is not adopted: without the first process the repeat's spreads are 8.58% and 5.25%.
  - Cost accepted. The spread within one Pod exceeded 5%, so the time-based conclusions of RQ2 and RQ3 are less precise than planned. A difference smaller than the measured noise is not reported as a finding, and the report's limits say so.
  - Pilot rerun. The M2 pilot, the anchors and the gSASRec rerun of D035 run on the rented machine as D045 planned. Their quality results are used as planned. Their run times are used for the GPU-hour projection of D034 and for no comparison between arms.
  - Diagnostic measurement, before any timing run of RQ2 or RQ3: one measurement of how the step-time spread within one Pod depends on the length of the timed window. Its design is written into this entry before its Pod is created. It is exploratory and is evidence for no hypothesis. From its result the timing protocol (repeats per comparison, the smallest difference reported as a finding) is written into METHODS before any timing run of RQ2 or RQ3.
- Amendment 2026-11-10, after the owner's decision, before any training run on a rented machine: how the data identify a Pod.
  - Every session of a run records the Pod it ran on (`RUNPOD_POD_ID`, which the platform sets) in the run's manifest. The summary and the batch file list a run's Pods in `pod`, joined with " + " when it trained on more than one. A machine that is not a Pod records an empty value.
  - The slow-run flag of D038 groups by `pod` where it grouped by `host`. A run with no other seed on its Pod gets no value, and so does a run that trained on more than one Pod. A batch that is interrupted and resumed on a new Pod is not rerun for that.
  - The `host` column stays as a description of the machine and enters no comparison.
  - The `machine` step of a job prints the Pod and fails without one, so no job trains runs that record none.
  - `test_an_exported_run_names_every_host_and_pod_it_trained_on`, `test_slow_epochs_compares_only_runs_made_on_one_pod` and `test_the_machine_step_refuses_a_machine_without_a_pod_id` failed before the change (METHODS 11.5).
- Diagnostic measurement, design, 2026-11-10, fixed before its Pod is created and before any of its data exist. It is exploratory and is evidence for no hypothesis.
  - Configurations: the two of the admission test, unchanged: SASRec with BCE on Steam and gSASRec on MovieLens-1M, with the default parameters of `scripts/profile_step.py`.
  - Processes: 5 per configuration, on one Pod that runs nothing else, alternating Steam, MovieLens-1M, Steam and so on, as ten jobs of one process each. The order in which they ran and the time each printed its timing line are recorded.
  - Window: every process times one long window (`--epochs`): 100 epochs of 200 steps on Steam (20,000 steps, about 67 s), 400 epochs of 47 steps on MovieLens-1M (18,800 steps, about 90 s). The step time of an epoch is that of D040, the epoch less the median of five one-step epochs, over its steps less one; the step time of a window is the mean over its epochs. `test_a_window_of_several_epochs_gives_the_step_time_of_each` failed before the option existed.
  - Window lengths: the first 1, 10 and 100 epochs of a Steam process (200, 2,000 and 20,000 steps), and the first 1, 4, 40 and 400 epochs of a MovieLens-1M process (47, 188, 1,880 and 18,800 steps). All lengths are read from the same five processes, so they are not independent of each other; between them only the length of the window differs. The shortest length is the quantity of the admission test.
  - Quantities, for each configuration:
    - at each length, the five step times, their spread (max − min) / median and their coefficient of variation;
    - within each process, the spread and the coefficient of variation over its consecutive, non-overlapping windows of each length below the longest;
    - the variance components of the per-epoch step times between and within processes (one-way, processes as groups, a negative estimate set to 0);
    - the Pod, kernel, type of the cgroup filesystem and source of the CPU count, from the `machine` step.
  - Readings fixed now:
    - The measured noise of the timing protocol is the spread between processes at the window length the protocol uses, the larger of the two configurations. At the shortest length the four spreads of the admission test (5.01%, 4.37%, 9.08% and 5.30%) enter that maximum.
    - For a configuration whose spread between processes at the longest length is below three quarters of that at the shortest, longer windows are read as helping, and the length to use is the shortest whose spread is at most 1.25 times that of the longest. Otherwise the noise is read as a level that belongs to a process, and the protocol relies on repeats in separate processes. Where the two configurations differ, the protocol takes the longer window.
    - Every comparison gets at least 5 repeats, as in METHODS 9.2.
    - The range of five values is a coarse estimate. These readings settle the protocol and are reported as no finding.
  - Left until the data are seen: the time scale and the cause of the two levels seen in the repeat, whether the runs of a comparison are paired or interleaved, whether repeats go above 5, and the value of the smallest difference reported as a finding. The CPU the main thread was on after each epoch is printed with the step times for this use only and enters no reading fixed above.
  - Limit: this is one Pod. The admission test saw two kinds of host, so the result describes this Pod only, and the timing protocol states how it deals with differences between hosts.
  - Record: the step times at every length and every quantity above are written into this entry, none left out. The per-epoch step times and CPU numbers of every process are committed as `results/env/d046-diagnostic.csv`; `results/env` is not read as an export (D045).
  - Cost: about 33 minutes of one Pod, 0.48 USD, at worst about 0.62 USD. The check off the Pod is set at 40 minutes and deletes the Pod at 50.
- Diagnostic measurement, result, 2026-11-10, as designed above. Pod `hv48mixx16ikf9`, created 08:02:45Z and gone by 09:36:17Z; workflow runs 15 to 24 at `38b1610`, one process each, all successful, in the designed order (Steam in runs 15, 17, 19, 21 and 23; MovieLens-1M in the others). Signature `NVIDIA GeForce RTX 4090 | 595 | AMD EPYC 7352 24-Core Processor | 14 | EU-RO-1`, kernel 6.8.0-138, CPU count from a cgroup v1 quota (`1360000 100000`), cgroup filesystem `tmpfs`, nominal count 16. Wall time per step in ms, processes in the order they ran; every epoch's value and the time of each timing line are in `results/env/d046-diagnostic.csv`.
  - The host is of a third kind: both Pods of the admission test had an AMD Ryzen 9 7950X. Medians at the admission test's window (200 steps on Steam, 47 on MovieLens-1M), same request and same GPU model:

    | Pod | CPU, kernel, CPU count | Steam, SASRec with BCE | MovieLens-1M, gSASRec |
    |---|---|---|---|
    | admission test, first | Ryzen 9 7950X, 6.8.0-147, 14 (cgroup v2 quota) | 3.075 | 4.693 |
    | admission test, repeat | Ryzen 9 7950X, 6.8.0-138, 32 (no quota read) | 3.339909 | 4.821271 |
    | diagnostic | EPYC 7352, 6.8.0-138, 14 (cgroup v1 quota) | 15.834245 | 16.408777 |

    The diagnostic Pod's step is 4.7 to 5.1 times the Ryzen Pods' on Steam and 3.4 to 3.5 times on MovieLens-1M. Process CPU time per step equals wall time on this Pod as well (15.633552 against 15.620209 ms in the first process).
  - Steam, between processes:
    - first epoch (200 steps): 15.836222, 15.530501, 15.936126, 15.834245, 15.787577; median 15.834245, spread 2.56%, coefficient of variation 0.96%;
    - first 10 epochs (2,000 steps): 15.622748, 15.911407, 15.857812, 15.949462, 15.768771; median 15.857812, spread 2.06%, coefficient of variation 0.82%;
    - all 100 epochs (20,000 steps): 15.620209, 15.759198, 15.782608, 15.846889, 15.499937; median 15.759198, spread 2.20%, coefficient of variation 0.89%.
  - MovieLens-1M, between processes:
    - first epoch (47 steps): 16.580054, 16.414704, 16.245924, 16.352684, 16.408777; median 16.408777, spread 2.04%, coefficient of variation 0.74%;
    - first 4 epochs (188 steps): 16.536952, 16.224175, 16.314757, 16.274782, 16.318279; median 16.314757, spread 1.92%, coefficient of variation 0.73%;
    - first 40 epochs (1,880 steps): 16.092134, 16.012405, 16.131429, 16.185134, 16.243163; median 16.131429, spread 1.43%, coefficient of variation 0.55%;
    - all 400 epochs (18,800 steps): 16.196095, 16.326716, 16.008951, 16.044149, 16.255774; median 16.196095, spread 1.96%, coefficient of variation 0.84%.
  - Within each process, spread over its consecutive windows, with the coefficient of variation in brackets, processes in the order they ran:
    - Steam, 100 windows of 200 steps: 11.78% (2.15%), 13.90% (1.68%), 11.61% (1.50%), 9.85% (1.36%), 8.24% (1.65%);
    - Steam, 10 windows of 2,000 steps: 3.80% (1.20%), 2.55% (0.96%), 2.26% (0.75%), 2.91% (0.86%), 3.73% (1.29%);
    - MovieLens-1M, 400 windows of 47 steps: 29.32% (3.38%), 21.69% (2.05%), 30.11% (2.82%), 12.47% (1.65%), 26.82% (2.09%);
    - MovieLens-1M, 100 windows of 188 steps: 12.80% (2.32%), 8.75% (1.74%), 17.12% (2.23%), 4.92% (1.33%), 7.67% (1.51%);
    - MovieLens-1M, 10 windows of 1,880 steps: 4.29% (1.21%), 3.31% (1.22%), 3.20% (1.02%), 2.74% (0.98%), 2.14% (0.64%).
  - Variance components of the per-epoch step times: Steam, mean 15.701768 ms, standard deviation between processes 0.137370 ms (0.87%) and within processes 0.264515 ms (1.68%); MovieLens-1M, mean 16.166337 ms, between processes 0.134821 ms (0.83%) and within processes 0.399882 ms (2.47%).
  - Readings fixed in the design:
    - The spread between processes at the longest length is 0.86 times that at the shortest on Steam (2.20% against 2.56%) and 0.96 times on MovieLens-1M (1.96% against 2.04%). Neither is below three quarters, so on this Pod the noise is read as a level that belongs to a process, and the protocol relies on repeats in separate processes.
    - Measured noise, the larger of the two configurations: 2.06% at 2,000 and 1,880 steps, 2.20% at 20,000 and 18,800 steps. At the shortest length the maximum with the admission test's four spreads is 9.08%; this Pod's own spreads there are 2.56% and 2.04%.
  - Observations, causes not examined: the main thread was seen on 13 to 27 of the host's 48 CPUs in each process; single epochs of MovieLens-1M ran as fast as 12.1 ms per step against medians near 16.4 ms.
  - Deviation from the design's cost: the Pod ran for 94 minutes, about 1.39 USD, because each process took about 9 minutes on this host. The deletion set at 50 minutes was moved to 100 minutes while the fifth process ran, so that all ten processes were measured; nothing else in the design changed.
  - Limit, as designed: one Pod of one kind of host. Whether the Ryzen hosts, where the admission test saw spreads of 4.37% to 9.08% at the shortest length, behave the same at longer windows was not measured.
- Owner's decision, 2026-11-10, made after the diagnostic measurement's result above was seen. The diagnostic Pod's host was not of the kind the admission test ran on. The measurement is made once more, with the design above unchanged, on a Pod whose CPU is a Ryzen 9 7950X, after the pilot rerun's batch on that Pod. It stays exploratory and is evidence for no hypothesis. Both results stay in this entry, and the timing protocol is written from both.
- Pilot rerun, the Pod it ran on, 2026-11-10. For this batch only, the Pod was probed before the batch was queued: one process of the admission test's Steam configuration, whose job log names the CPU. A Pod with another CPU than a Ryzen 9 7950X would have been deleted and another created, three Pods at most. The pilot's quality results do not depend on the host, and its run times enter only the projection of D034. The request for a Pod cannot name a CPU, and a Pod's record does not show it.
  - First Pod, `we8ppstv5rdldt`, created 09:54:35Z: AMD Ryzen 9 7950X, kernel 6.8.0-138, CPU count 7 from a cgroup v1 quota (`680000 100000`), nominal count 8; 3.043691 ms per step (workflow run 25). Kept; the batch was queued on it. Its CPU count is half that of the admission test's first Pod.
- Diagnostic measurement, second run, result, 2026-11-10, under the owner's decision above and with the design unchanged. Pod `we8ppstv5rdldt`, created 09:54:35Z and gone by 12:08:26Z; the pilot rerun's batch ran on it first (workflow runs 26 to 28, ended 11:24:37Z), then workflow runs 29 to 38 at `af2ceb0`, one process each, all successful, in the designed order. Signature `NVIDIA GeForce RTX 4090 | 595 | AMD Ryzen 9 7950X 16-Core Processor | 7 | EU-RO-1`, kernel 6.8.0-138, CPU count from a cgroup v1 quota (`680000 100000`), cgroup filesystem `tmpfs`, nominal count 8. Wall time per step in ms, processes in the order they ran; every epoch's value is in `results/env/d046-diagnostic-7950x.csv`.
  - Steam, between processes:
    - first epoch (200 steps): 3.014178, 3.064884, 3.950826, 3.064965, 3.688324; median 3.064965, spread 30.56%, coefficient of variation 12.90%;
    - first 10 epochs (2,000 steps): 3.001803, 3.036150, 3.741892, 3.037183, 3.468088; median 3.037183, spread 24.37%, coefficient of variation 10.20%;
    - all 100 epochs (20,000 steps): 3.121950, 3.160158, 3.136261, 3.176810, 3.337955; median 3.160158, spread 6.84%, coefficient of variation 2.74%.
  - MovieLens-1M, between processes:
    - first epoch (47 steps): 5.027768, 4.741665, 5.453672, 6.416982, 4.550419; median 5.027768, spread 37.13%, coefficient of variation 14.16%;
    - first 4 epochs (188 steps): 5.232700, 4.769902, 5.381650, 6.000174, 4.628666; median 5.232700, spread 26.21%, coefficient of variation 10.46%;
    - first 40 epochs (1,880 steps): 4.742199, 4.707813, 5.390812, 4.937815, 4.554080; median 4.742199, spread 17.64%, coefficient of variation 6.64%;
    - all 400 epochs (18,800 steps): 4.779039, 4.669395, 4.734336, 4.635485, 4.680479; median 4.680479, spread 3.07%, coefficient of variation 1.21%.
  - Within each process, spread over its consecutive windows, with the coefficient of variation in brackets, processes in the order they ran:
    - Steam, 100 windows of 200 steps: 32.77% (5.06%), 37.87% (8.10%), 35.54% (8.87%), 50.17% (10.90%), 38.40% (9.44%);
    - Steam, 10 windows of 2,000 steps: 8.24% (3.09%), 21.06% (6.14%), 24.14% (7.17%), 26.72% (10.30%), 25.45% (7.00%);
    - MovieLens-1M, 400 windows of 47 steps: 54.30% (7.93%), 38.54% (6.90%), 58.80% (8.86%), 57.01% (8.78%), 69.45% (8.31%);
    - MovieLens-1M, 100 windows of 188 steps: 28.41% (5.82%), 22.42% (5.85%), 51.42% (8.24%), 38.43% (7.70%), 48.17% (7.71%);
    - MovieLens-1M, 10 windows of 1,880 steps: 6.89% (2.66%), 10.14% (3.21%), 18.32% (5.12%), 14.13% (4.92%), 15.23% (4.48%).
  - Variance components of the per-epoch step times: Steam, mean 3.186627 ms, standard deviation between processes 0.082646 ms (2.59%) and within processes 0.278200 ms (8.73%); MovieLens-1M, mean 4.699747 ms, between processes 0.053443 ms (1.14%) and within processes 0.384864 ms (8.19%).
  - Readings fixed in the design:
    - The spread between processes at the longest length is 0.22 times that at the shortest on Steam (6.84% against 30.56%) and 0.08 times on MovieLens-1M (3.07% against 37.13%). Both are below three quarters, so on this Pod longer windows are read as helping. No shorter length has a spread within 1.25 times that of the longest (24.37% against 8.55% on Steam, 17.64% against 3.84% on MovieLens-1M), so the length to use is the longest measured: 20,000 and 18,800 steps.
    - Measured noise at that length, the larger of the two configurations: 6.84%. At the shortest length this Pod's spreads, 30.56% and 37.13%, exceed the four of the admission test.
  - The two runs side by side, spread between processes at the shortest and at the longest length: on the EPYC 7352 Pod 2.56% and 2.20% (Steam), 2.04% and 1.96% (MovieLens-1M); on this Pod 30.56% and 6.84%, 37.13% and 3.07%. The readings differ by host: a level per process on the first, noise that long windows average out on the second. Medians over the longest window are 3.160158 and 4.680479 ms here against 15.759198 and 16.196095 ms there.
  - Observations, causes not examined: within a process the per-epoch step times vary by 8% to 9% (standard deviation) here against about 2% on the EPYC Pod; the main thread was seen on 9 to 22 of the host's CPUs in each process; process CPU time equals wall time here too. This Pod had trained the batch for 90 minutes before the measurement, and its CPU quota is half that of the admission test's first Pod; neither effect was separated.
  - Limit: one Pod of each kind of host, five processes per configuration. A measurement at a window longer than 20,000 steps was not made, so whether the spread keeps falling is not known.
- Timing protocol, 2026-11-10: METHODS 9.4, approved by the owner. It was written after both results above and the readings of D048 were seen, and before any timing run of RQ2 or RQ3. Of the choices it left to the owner, he took seeds 1 to 5 for H2a, Ryzen 9 7950X hosts for every later batch, and the 200-step measurement for grid cells outside the decision rules. Its deviations from METHODS 5.5, 8 and 9.2 are listed in 9.4.

## D047 The rented GPU is an RTX 4090

- Date: 2026-11-10. Status: accepted by the owner. Replaces the GPU model, its price and the catalogue figures of D045; the rest of D045 and D046 stands. Made before any run on a rented machine.
- Decision: the rented machine is an RTX 4090 (24 GB) in Runpod Secure Cloud, data centre EU-RO-1, on demand, at 0.89 USD per hour [34]. `configs/cloud/pod.json` requests it. The data centre's RTX 4090 hosts are not all alike: Pods with 12 and with 16 vCPUs have been created from the same request, so the CPU model and count are whatever the signature of each run records (D046).
- Reason: within one day all five requests for an RTX 5090 Pod in EU-RO-1 were refused for lack of capacity while the catalogue listed the card as in stock. With the same data centre, image and CUDA requirement, the first request for an RTX 4090 was granted.
- What stays: the RTX 4090 is a consumer card, so what D045 says about consumer hardware holds. Research questions, hypotheses, decision rules, the rules for test reads and the timing rules of D046 are unchanged.
- Memory: no registered configuration needs more than 24 GB.
  - Full cross-entropy takes about 12 bytes per logit (`scripts/check_env.py`), so its logits need batch × L × N × 12 bytes: 5.3 GiB for the largest training run (MovieLens-20M, N = 18,345, L = 200, batch 128) and 7.3 GiB for the plain workload of H3b (N = 25,600, L = 200). The M2 pilot ran on the laptop's 16 GB without a memory failure.
  - RQ3 grid (METHODS 9.1): chunked cross-entropy is used wherever the plain form does not fit. The cells above 24 GiB are the ones that were above 28 GiB already (N = 25,600 at L = 800; N = 100,000 at L ≥ 200), so the same cells are chunked as on a 32 GB card. The largest plain cell is N = 18,345 at L = 800 with 21.0 GiB of logits; if it does not fit, the same rule applies.
- Binaries: the card's compute capability is 8.9. The `cu132` wheel is built for 7.5, 8.0, 8.6, 9.0, 10.0 and 12.0, and no official wheel is built for 8.9 [36], so `check_env` reports that `sm_89` is not in the wheel's list. Nothing is wrong with that: code built for 8.0 or 8.6 runs on every 8.x card, and the driver picks the translation for each kernel [35]; PyTorch warns only when no entry of the list fits the card [36]. PyTorch's own kernels therefore run as code for 8.6 or 8.0. NVIDIA expects code built for `sm_89` to be better [35]; by how much is not measured here. cuBLAS and cuDNN are NVIDIA's binaries with their own builds, and Triton compiles for the capability of the card it finds [37]. Times in this study are those of the official wheel on this card, and every comparison is between runs of that one binary.
- Budget: the cap of 500 GPU hours and the limit of 15 GPU hours for M2 work (D045) are unchanged; they now count hours on the RTX 4090. The cap in USD is still set after the pilot rerun.
- Alternative: further requests for an RTX 5090, one per hour (no date by which one would be granted).

## D048 Execution mode before M3: measurement of the prototype

- Date: 2026-11-10. Status: design fixed before its Pod is created and before any of its data exist. It is exploratory and is evidence for no hypothesis. It informs one choice, which is the owner's: the execution mode of the training step of every arm from M3 on (TASKS, before M3 tuning).
- Question: in the pilot rerun (D034) a training step on Steam and Video_Games took 3.1 to 5.6 ms in every arm, at both configuration sizes, close to the 3 ms of a step bound by kernel launches (D046). What the modes of D043 remove of that, and what fixed shapes cost where the GPU limits the step, has not been measured on the rented machine.
- Configurations: eight, with the hyperparameters of the pilot's timing runs (`configs/plans/m2-pilot.toml`), so that their step times can be set against the epoch times behind D034. In brackets the step time of the pilot rerun, the median training time of an epoch over its batches:
  1. Steam, SASRec with BCE, anchor size (3.28 ms); the Steam configuration of D046;
  2. Steam, SASRec with full cross-entropy, anchor size (3.16 ms);
  3. Steam, SASRec with sampled softmax, anchor size (3.21 ms);
  4. Steam, gSASRec, anchor size (3.12 ms);
  5. Video_Games, SASRec with full cross-entropy, largest size (5.59 ms);
  6. MovieLens-20M, SASRec with full cross-entropy, anchor size (12.92 ms);
  7. MovieLens-20M, SASRec with full cross-entropy, largest size (19.62 ms, the slowest step of the pilot);
  8. MovieLens-20M, gSASRec, anchor size (10.91 ms).
  - Configurations 1 to 4 are one per arm of A1 to A4 where launches bind the step, which is where H2a is affected most; 5 is the largest size on a dataset of short sequences; 6 to 8 are bound by the GPU. Fixed shapes compute the loss at all B × L positions, and that adds most to full cross-entropy and gBCE, so these two arms are also measured on MovieLens-20M and in configuration 5.
  - Not measured: SASRec with BCE and with sampled softmax on MovieLens-20M (3.1 to 3.3 ms at the anchor size in the pilot), MovieLens-1M (about 3% of the projected hours), and BERT4Rec, which the modes do not cover (D043).
- Modes: four. The arm's own step; fused Adam; fixed shapes with fused Adam; fixed shapes with fused Adam and a CUDA graph. Fused Adam alone leaves the loss code as it is. Fixed shapes alone are not measured; their effect is the third mode against the second.
- Processes: one job per configuration on one Pod that runs nothing else, configurations 1 to 8 in turn, each job running the lines of its schedule, `configs/profiles/d048-<n>-*.txt`, in the file's order. The schedules are committed with this design. Only the last job asks for the Pod to be deleted (`stop`).
  - The Pod's CPU is a Ryzen 9 7950X, the host kind of the pilot rerun and of D034. The Pod is probed as for the pilot rerun (D046); a Pod with another CPU is deleted with nothing else run on it, three Pods at most. The CPU count of the Pod is recorded with the result (7 and 14 have been seen on this CPU), and the readings hold for the count measured.
  - Timing: every process times one window of 200-step epochs (`--epochs`): 20,000 steps in configurations 1 to 4, the longest length of D046, and in configurations 5 to 8 the number of epochs nearest to 65 s of the arm's own step at the pilot's step time, 11,600, 5,000, 3,400 and 6,000 steps. The modes of a configuration time the same number of steps. Five rounds per configuration, each mode once per round, the first mode moving on by one every round: 160 processes.
  - Equivalence: per configuration one `--check` process for each of the three modes other than the arm's own step, before the timing.
  - Synchronisations: in configurations 1 to 4, one `--sync-debug` process per mode.
- Quantities, for each configuration and mode: the five step times with their smallest, median and largest; the speed-up, the median step time of the arm's own step over that of the mode; the five round ratios, one per round, the step time of the arm's own step over the mode's in the same round, with their smallest and largest; the median share of a step the GPU is busy; peak allocated memory; the `--check` readings; the synchronisations per step. With them the Pod, kernel and CPU count from the `machine` step.
  - A timing process counts when it printed its `timing` line, whatever happened after it. A round counts for a mode when both its process and the arm's own counted. With five or four rounds counting, the medians, the speed-up and the round ratios are taken over those rounds. With fewer, the mode has no reading in that configuration.
- Readings fixed now. The thresholds were set with no reading of these modes from the rented machine.
  - A difference from the arm's own step is read as measured when all round ratios lie on one side of 1 and the speed-up is above 1.0684 or below 0.9316, the noise of D046 on this kind of host at this window (6.84%). Otherwise no difference was measured.
  - A mode is read as worth applying to every arm when all three hold: its speed-up is at least 1.3 in at least four of configurations 1 to 5; its speed-up is at least 0.90 in all eight; its equivalence is acceptable in all eight. A mode without a reading in a configuration does not reach 0.90 there.
  - Among modes that are worth applying, the one that changes least is recommended, in the order fused Adam, fixed shapes with fused Adam, CUDA graph. Going through that order, a later mode takes the place of the one recommended so far only if it is worth applying itself and its median step time is at most 1/1.2 of that mode's in at least four of configurations 1 to 5.
  - If no mode is worth applying, the recommendation is the arm's own step, and the report states that time to target (H2a) on small configurations mostly measures launch overhead. Fused Adam alone is then still put to the owner if it has a measured speed-up somewhere and no measured slow-down.
  - For H2a: in configurations 1 to 4, a busy share of at least 0.5 under a mode is read as a step that is mostly GPU work. Under a CUDA graph the GPU time covers only the kernels the profiler reports (D043), and the share is marked so.
  - Equivalence, from `--check` (dropout 0, three batches): the largest loss difference between the mode and its reference, and the same relative to the smallest of the mode's three losses. In this order: not acceptable, if the process printed no result, a loss is not finite or the relative difference is above 1e-2; otherwise acceptable, if the difference is at most 10 times that between two runs of the reference or the relative difference is at most 1e-3; otherwise undecided. An undecided mode counts as not acceptable above, and so is worth applying to no arm.
  - No process is added once the data are seen. Where the step times of the rounds spread further than the modes differ, the configuration is reported as undecided.
  - A rough estimate of the projected hours of D034 under each mode, the speed-ups applied to its blocks, is given as a description and enters no reading.
- Limits:
  - A faster mode's window is shorter in time by its speed-up, and D046 measured the spread only at windows of 63 to 94 s. The round ratios show the spread obtained.
  - Under fixed shapes gSASRec draws negatives for every position, so its random stream differs from the arm's own and its `--check` reference keeps fixed shapes (D043). The fixed-shape step of gSASRec is therefore compared with the arm's own only by the CPU test of D043, given the same negatives.
  - One Pod, one kind of host, five processes per mode. On the EPYC 7352 host of D046 launches cost 3.4 to 5.1 times as much, so the speed-ups there would differ.
- Record: the step times of every epoch of every timing process are committed as `results/env/d048-step-modes.csv`, and every quantity above is written into this entry, none left out.
- Anchors: their test reads on the rented machine (D045) are made after the execution mode is chosen. Each arm has two reads left (METHODS 11.5), and a read made now would test a training step that may be replaced. If the chosen mode changes the step, the four anchors are trained again with it and those runs are read; if the step stays, the four existing runs are read.
- Tooling: the `profile` task of the batch workflow takes a schedule from `configs/profiles/` in its `plan` input and runs each line as one process, once and in order; a process that fails does not stop those after it and fails the job at its end. `--params` is accepted as a flat JSON object without spaces. The profile step may run for 180 minutes, up from 120. `test_profile_runs_every_line_of_a_schedule_once_in_order`, `test_profile_goes_on_after_a_failed_process_and_fails_at_the_end`, `test_a_schedule_takes_the_place_of_profile_arguments`, `test_other_schedule_names_are_rejected`, `test_a_schedule_that_does_not_exist_is_rejected` and the `--params` case of `test_profile_arguments_the_script_knows_are_accepted` failed before the change (METHODS 11.5).
- Cost: 5.2 to 6.3 hours of one Pod, 4.6 to 5.6 USD, the upper end if fixed shapes take two to three times the arm's own step where they add most. Checks off the Pod at 15 minutes, 3.5 hours and 6.5 hours after the first Pod is created; the Pod is deleted at 7.5 hours at the latest, 6.7 USD.
- Amendment 2026-11-10, while configuration 1 ran and before any of its readings was seen: a process took about twice as long to start as the cost above assumed, so the last check moved to 8 hours and the latest deletion of the Pod to 9 hours, 8.1 USD. The design is unchanged.
- Result, 2026-11-10, incomplete: the measurement was stopped during configuration 2 by the owner's decision below. One Pod, `0uswy11tvpk5lx`, created 21:44:17Z and deleted 23:07:40Z; three requests before it were refused for lack of capacity. Probe (workflow run 39): signature `NVIDIA GeForce RTX 4090 | 595 | AMD Ryzen 9 7950X 16-Core Processor | 14 | EU-RO-1`, kernel 6.8.0-138, CPU count 14 from a cgroup v1 quota (`1360000 100000`), nominal count 16; 3.190341 ms per step. Jobs at `4d7e86d`: run 40, configuration 1, all 27 processes, none failed; run 41, configuration 2, cancelled during its tenth timing process; runs 42 to 47, configurations 3 to 8, cancelled before they started. The step times of every epoch are in `results/env/d048-step-modes.csv`.
  - Configuration 1 (Steam, SASRec with BCE, anchor size), wall time per step in ms:

    | Mode | Rounds 1 to 5 | Median | Speed-up | Round ratios | GPU busy share |
    |---|---|---|---|---|---|
    | the arm's own step | 3.206588, 3.069718, 3.230196, 3.236176, 3.259732 | 3.230196 | | | 0.295 |
    | fused Adam | 3.218975, 3.077939, 3.143058, 3.135183, 2.899764 | 3.135183 | 1.030 | 0.996 to 1.124 | 0.292 |
    | fixed shapes, fused Adam | 2.869771, 2.809716, 2.955070, 3.047617, 2.872180 | 2.872180 | 1.125 | 1.062 to 1.135 | 0.806 |
    | fixed shapes, fused Adam, CUDA graph | 2.303574, 2.303939, 2.303986, 2.306694, 2.305098 | 2.303986 | 1.402 | 1.332 to 1.414 | not measured |

    GPU time per step from the profiler: 0.953, 0.913 and 2.316 ms in the first three modes; under the graph the profiler reported no GPU events, so neither it nor the busy share is a measurement there (D043). Peak allocated memory 0.14 GiB in every mode. Synchronisations per step: 4, 4, 0 and 0. `--check`: loss differences 1.19e-7, 2.38e-7 and 2.38e-7 (relative 9.1e-8, 1.8e-7 and 1.8e-7) against 0 between two runs of the reference.
  - Configuration 2 (Steam, SASRec with full cross-entropy, anchor size), stopped after rounds 1 and 2 and the first process of round 3: the arm's own step 3.220046 and 3.167179 ms; fused Adam 3.186587 and 3.232021; fixed shapes with fused Adam 4.409659, 4.411826 and 4.414574; with the CUDA graph 4.277060 and 4.276553. Busy share 0.407 and 0.414, 0.398 and 0.393, 0.973, 0.972 and 0.972; GPU time per step 1.31, 1.27 and 4.29 ms. Peak allocated memory 0.32, 0.32, 0.96 and 0.96 GiB. `--check`: loss differences 5.72e-6, 3.81e-5 and 3.81e-5 (relative 6.2e-7, 4.1e-6 and 4.1e-6) against 0.
  - Outside its window a timing process ran for 47 to 58 s, about 41 s of it before the first timed step.
  - Readings fixed in the design:
    - Configuration 1: fused Adam, no difference measured (round ratios on both sides of 1). Fixed shapes with fused Adam, a measured speed-up of 1.125, below 1.3. With the CUDA graph, a measured speed-up of 1.402. Equivalence acceptable in all three modes. The busy share under fixed shapes, 0.806, is read as a step that is mostly GPU work.
    - Configuration 2: two rounds count, fewer than four, so no mode has a reading. Equivalence acceptable in all three modes.
    - Configurations 3 to 8: no reading.
    - No mode has a reading in all eight configurations, so none is read as worth applying to every arm, and the reading's recommendation is the arm's own step. That outcome comes from the stop: six configurations were not measured.
  - Outside the readings: where it was measured, fixed shapes raised the GPU time of a step, from 0.95 to 2.32 ms in configuration 1 and from 1.31 to 4.29 ms in configuration 2, and in the two complete rounds of configuration 2 the fixed-shape step took 1.38 times the arm's own.
  - Deviation from the design: the measurement ended after 1.39 hours of the Pod, 1.25 USD, with 39 of its 200 processes complete.
- Owner's decision, 2026-11-10, made after the readings of configuration 1 had been reported and before those of configuration 2 were read: the training step of every arm stays the arm's own from M3 on. The measurement is stopped and its Pod deleted, the readings obtained and this decision are recorded here, and M3 starts next. Reason: the speed-ups read so far do not repay changing the training step of every arm and training the anchors again.
  - The modes of D043 stay a measurement tool; no training run uses them.
  - The pilot timings and the projection of D034 stand, since no step changed.
  - The anchors' test reads are made on the four existing runs (above).
  - Time to target (H2a) on small configurations mostly measures launch overhead: the GPU was busy for 0.295 of a step in configuration 1. The report states this.

## D049 Code changes between the anchor reads and M3 tuning

- Date: 2026-11-11. Status: accepted. A log of code changes made after an anchor read, each with a test that failed before it (METHODS 11.5). None changes a registered rule or the training step.
- Anchor-read limit in the reader: `read_test` counts the anchor reads logged for the run's arm and dataset in every batch of the runs directory, adds the reads made on another machine that its caller passes (D045), and refuses the read before any record or data once they reach three (METHODS 11.6). The limit was enforced by `test_run` before, so another caller of the reader was not bound by it. `test_the_reader_itself_refuses_a_fourth_anchor_read` failed before the change.
- Start limit in tuning plans: a study takes `start_limit` and passes it to its trials, so a plan with doubled limits doubles it next to `max_epochs` and `patience` (D035). `test_a_tuning_plan_sets_the_start_limit_of_its_trials` failed before the change. `configs/plans/m3-tune-ml-1m.toml` sets none of the three and so uses the limits of METHODS 5.2 and the start limit of D035.
- CPU model checked by every job: the batch workflow takes a `cpu` input, by default `Ryzen 9 7950X` (METHODS 9.4). The machine step, which precedes all work, fails on a machine whose CPU model does not hold it as whole words (a 7950X3D is not a 7950X) and asks for the machine to be deleted, whatever the job's `stop` input says. A new Pod therefore needs no separate probe: its first job checks the host. Only that job is queued before the host is known, because a Pod deleted while further jobs wait leaves them queued, and they would run first on the next Pod (D046); jobs still waiting are cancelled before another Pod is created. An empty `cpu` accepts any host. `test_the_machine_step_refuses_another_cpu_and_asks_for_the_machine_to_be_deleted`, `test_the_required_cpu_matches_whole_words_of_the_model_name`, `test_other_cpu_names_are_rejected` and `test_every_job_requires_the_registered_host_unless_told_otherwise` failed before the change.
- Items checked against the log: `check_split` requires `items` to equal `full.items[pos]` in the training sequences and in the validation and test inputs (METHODS 11.1). Every other check goes through the positions, so items reordered apart from their positions passed before. `test_items_detached_from_their_log_positions` failed before the change. Each split now passes 34 checks instead of 31; `results/data/stats.csv` carries the new count, and the `setup` job on the first M3 Pod prepares the data again and fails if any dataset gives another file.
- Splits cached per process: `load_splits` keeps the splits of the most recent dataset and split kind together with the checks they passed, so the checks of METHODS 11.1 run once per process and not once per run. This stays. Every run of the process trains and evaluates on the very objects that were checked, records their fingerprints in its manifest, and the test read compares those with the data it loads. A dataset rebuilt on disk while a batch runs would be noticed by none of this, and is ruled out by the workflow: only the `setup` job rebuilds data, and one job runs at a time.
- Read record synced: `read_test` flushes and syncs the record of a read to disk before it returns data, as the scores already are when they are written, and the export reports a run that has test scores and no record of a read. A crash between the two writes could otherwise leave an anchor read that no count sees. `test_the_read_record_is_on_disk_before_any_data_are_returned` and `test_export_reports_test_scores_without_a_record_of_their_read` failed before the change.
- Known and left as they are:
  - Jobs queued behind a `setup` job start whether or not it succeeded. Test reads and tuning are therefore queued only after the setup job of their Pod has succeeded.
  - After a test read that did not complete, every export of the volume reports it and ends with an error, so the workflow commits no further batch of that volume until the read is dealt with by hand. The read stays counted on the volume.
  - Optuna counts only complete and pruned trials toward the five random start-up trials (METHODS 5.4), so a trial that fails among the first five makes the sixth random as well.
  - On a much faster CPU `tests/test_tune.py` could read a median evaluation time of 0.000 s, and the planted-rule test of BERT4Rec passed with 0.969 to 0.994 over six seeds against its threshold of 0.95; neither has failed.

## D050 The RecArena comparison of training objectives

- Date: 2026-11-11. Status: checked before M3 tuning; no change to RQ2.
- What was announced: the README of github.com/amazon-science/RecArena (latest commit 2026-08-28) lists two papers "submitted to RecSys 2026", one of them "Revisiting Negative Sampling for Sequential Recommendation: A Systematic Comparison", described as a comparison of "full cross-entropy, sampled softmax, BCE, and gBCE loss functions across 6 datasets" with 16 to 2,048 negatives. Its command names MovieLens-100K, MovieLens-1M, MovieLens-20M, Amazon Beauty, RateBeer and Twitch; the library evaluates leave-one-out with full ranking.
- Published or not: on 2026-11-11 a search of arXiv and of the accepted contributions of RecSys 2026 found neither the title nor the repository's name, and the README links no paper and reports no result. The comparison counts as announced and not published.
- Overlap with RQ2: the four objectives on a SASRec backbone are the same as arms A1 to A4, and MovieLens-1M and MovieLens-20M are in both. From the README alone, what RQ2 has beyond it is time to a target quality (H2a), equivalence tests against a fixed smallest effect over seeds (H2b), per-arm tuning under one budget, and Steam and Video_Games. This is checked again when the paper appears and before the report is written; until then RQ2 stands as registered.

