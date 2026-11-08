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
