# Specification

## 1. Goal

Measure, on one consumer GPU rented by the hour (D045), three things about Transformer sequential recommenders:

1. how much of the published quality gap between SASRec, BERT4Rec and gSASRec survives an equal tuning budget, several seeds and full-ranking evaluation;
2. what each training objective costs in wall-clock time and memory for the quality it reaches;
3. where training and inference time goes, and how much attention kernels, precision and sequence length matter.

Every confirmatory claim follows the pre-registered analysis in [METHODS.md](METHODS.md). Decisions are logged in [DECISIONS.md](DECISIONS.md) and work is tracked in [TASKS.md](TASKS.md).

## 2. Position relative to prior work

- BERT4Rec [2] was reported to beat SASRec [1]. Later work traced most of that gap to SASRec's training objective (binary cross-entropy with one sampled negative) and to training length. Petrov and Macdonald [3] found BERT4Rec reproduces its reported quality only when trained much longer than its default configuration. gSASRec [4] attributes the gap to overconfidence caused by negative sampling. Klenitskiy and Vasilev [5] show SASRec trained with full cross-entropy beats BERT4Rec on most datasets, Steam being the exception.
- These studies mostly report single training runs, tune with unequal or unstated budgets, and use leave-one-out splits. Short-paper evidence [17] reports that 1 to 2% differences between SASRec runs can change winner with the seed, and that user-level paired tests are overconfident.
- Sampled metrics can reorder models [6, 9, 16], so this project ranks the full catalogue. Leave-one-out splits leak future information and change model rankings [7, 8, 12]; a global temporal split is therefore added as a robustness check.
- Memory-efficient losses for large catalogues are covered by [13, 14, 15]. These works use datacentre GPUs and mostly single runs. Measurements of PyTorch attention backends, mixed precision or `torch.compile` for SASRec-scale models were not found.

The expected direction of RQ1 and the quality ordering in RQ2 are largely known. The contribution is the size of each effect with uncertainty attached, wall-clock costs on a consumer GPU, and a time breakdown that links the loss choice to efficiency.

## 3. Research questions

Hypotheses and decision rules are stated in full in METHODS sections 7 to 10. SESOI (smallest effect size of interest) is a 2% relative difference in NDCG@10.

**RQ1 Robustness of published gaps.** Under an equal tuning budget, at least five seeds and full-ranking evaluation, which pairwise differences among SASRec-BCE, SASRec-CE, gSASRec and BERT4Rec exceed run-to-run variation? Do three published gaps keep their sign and size on the datasets where they were reported: BERT4Rec over SASRec-BCE [2], gSASRec against BERT4Rec [4] (ahead on ML-1M, behind on Steam), and SASRec-CE against BERT4Rec [5] (ahead on ML-1M and ML-20M, behind on Steam)? Does the model ranking under leave-one-out agree with a global temporal split?
- H1a: SASRec-CE is non-inferior to BERT4Rec, tested separately on ML-1M and ML-20M.
- H1b: SASRec-BCE is worse than SASRec-CE, gSASRec and BERT4Rec by more than SESOI, tested per dataset.
- Each published gap is classified as reversed below SESOI, reversed, confirmed below SESOI, confirmed, equivalent or inconclusive by fixed rules.

**RQ2 Cost and quality of training objectives.** On a SASRec backbone, compare full softmax cross-entropy, sampled softmax, BCE with one negative and gBCE [4]. Measure validation quality as a function of training time, time to reach a target quality, peak memory, and how each scales with catalogue size (3.4k to 25.6k items).
- H2a: full cross-entropy reaches the target quality faster than the other three objectives.
- H2b: sampled softmax and gBCE, as tuned, are equivalent to full cross-entropy in final test NDCG@10 within SESOI; BCE with one negative is worse by more than SESOI.

**RQ3 Where the time goes.** How do the attention backend, numeric precision, sequence length (50 to 800) and catalogue size change train step time, peak memory and inference latency? How much of a train step is attention, and how much is output scoring plus loss?
- H3a: at sequence length up to 200, fused attention backends differ by less than 10% in end-to-end train step time.
- H3b: with full cross-entropy, at least 25.6k items, length 200 and dimension 64, output scoring plus loss takes more than half of the train step.
- H3c: bf16 autocast gives at least 1.3x the training throughput of strict fp32 (TF32 off), and its test NDCG@10 is equivalent within SESOI.

**RQ4 (conditional, decided at the end of M3, D015).** For item embeddings trained with each RQ2 objective, compare approximate nearest-neighbour search (faiss HNSW and IVF) with exact top-k on recall@10 relative to exact search, NDCG@10 loss and per-query latency. Exploratory hypothesis: at fixed index parameters, recall@10 differs across training objectives by more than 0.02.

## 4. Scope

In scope:
- Models: SASRec [1], BERT4Rec [2], gSASRec [4], all reimplemented in plain PyTorch (D004).
- Objectives: full cross-entropy (plain and chunked, same objective), sampled softmax, BCE with one negative, gBCE.
- Floor baselines: most popular items, and a first-order item-to-item transition model.
- Controls: item order shuffled inside each user's sequence, in training and evaluation, and in test inputs only (D017).
- Splits: leave-one-out (primary) and a global temporal split (secondary).
- Measurements: full-ranking quality, wall-clock time, peak memory, inference latency, GPU clocks, temperature and power during runs.

Out of scope:
- New model proposals, LLM-based recommenders, HSTU-style long-sequence models, sequences longer than 800.
- Sampled-metric evaluation, except to restate published numbers.
- Multi-GPU, data-centre GPUs, and comparisons between machines or operating systems.
- Datasets whose terms restrict publication (Yelp Open Dataset, D005).
- Redistributing any dataset or derived interaction data.

## 5. Datasets

Only download and preprocessing code is committed. Raw and processed data stay on the local machine.

| Dataset | Role | Size after 5-core (published) | Terms | Timestamps |
|---|---|---|---|---|
| MovieLens-1M [18] | Anchor for published numbers, pilot runs | 6,040 users, 3,416 items, 1.0M interactions [1] | Research use, no redistribution, citation required | Seconds, many ties |
| Steam [1] | Published comparisons; weak sequential structure [11]; BERT4Rec still ahead in [4, 5] | 334,730 users, 13,047 items, 3.7M [1]; [4, 5] use a user-filtered version with 281,428 users and 13,044 items | No license file; citation requested | Days, many ties |
| Amazon Reviews 2023, Video_Games [19] | Recent data, short sequences, largest catalogue in the core set | 94.8k users, 25.6k items, 815k (official 5-core) | No license stated; citation requested | Milliseconds |
| MovieLens-20M [18] | Strong sequential structure [11], most interactions | 138,493 users, 18,345 items, 20M (5-core, [12]); [2] filters users only and keeps 26,744 items | Research use, no redistribution, citation required | Seconds |

Statistics produced by our own preprocessing are compared with the published ones; any difference above 1% is explained in DECISIONS (D006). RQ4 needs a larger catalogue (candidates: Amazon Reviews 2023 Toys_and_Games, Yambda-50M); the choice is made with D015.

## 6. Models, objectives and arms

| Arm | Backbone | Objective | Notes |
|---|---|---|---|
| A1 SASRec-BCE | SASRec | BCE, one uniform negative per position | Original SASRec training [1] |
| A2 SASRec-CE | SASRec | Full softmax cross-entropy | As in [5] |
| A3 SASRec-SS | SASRec | Sampled softmax, K uniform negatives, accidental hits masked | K tuned |
| A4 gSASRec | SASRec | gBCE with K negatives and calibration t [4] | K, t tuned |
| A5 BERT4Rec | Bidirectional encoder | Cloze (masked item) with full softmax [2] | |
| B1 MostPop | None | Training-set popularity | Floor |
| B2 Transition | None | First-order item-to-item counts from training sequences | Floor |
| C1 Shuffled | A2 and A5 configs | Same as source arm, item order shuffled within each sequence in training and evaluation | Order-use check (D017) |

SASRec arms tune whether output embeddings are shared with input embeddings. Chunked cross-entropy computes the same loss as A2 in slices to bound memory. It is an implementation variant measured in RQ2 and RQ3 and is not tuned separately.

## 7. Evaluation protocol (summary)

- Leave-one-out: last item per user is the test target, second to last the validation target. Global temporal split: a global time cutoff with each test user's last item as the target. Details in METHODS section 3.
- Full ranking over all items in fp32, excluding every item already in the user's history, with pessimistic tie handling. Primary metric NDCG@10; HR@10, NDCG@20 and HR@20 are descriptive.
- Hyperparameters and checkpoints are selected on validation data only. The test set is read once per final run, through one logged function.
- Equal Optuna TPE budget per arm within a dataset; five to ten seeds per final configuration, set by a pre-registered power rule (D018).
- Inference uses training runs as the unit (Welch t-test on log NDCG@10), Holm correction within each test family, and TOST for equivalence.

## 8. Platform

| Item | Choice | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB, Ada Lovelace; rented, a new machine of this type for every batch (D045, D046, D047) | [34] |
| OS for GPU runs | Linux container on the provider's host driver (D045) | |
| Python | 3.12 | |
| PyTorch | 2.15.x, CUDA 13.2 wheels (`cu132`) | [21, 22] |
| NVIDIA driver | Branch 580 or newer for CUDA 13 wheels | [22] |
| `torch.compile` on GPU | Triton from the Linux wheel's dependencies | |
| Monitoring | NVML through `nvidia-ml-py` | [25] |
| Environment manager | uv with a committed lock file | |
| Development | Linux container without GPU, CPU wheels, small configurations | |

M0 to the M2 pilot ran on an RTX 5080 Laptop GPU under native Windows (D002); its environment report stays under `results/env/` and its exports under `results/laptop/`. Times from the two machines are never mixed in one comparison.

`scripts/check_env.py` verifies these assumptions on the target machine: wheel architecture, every attention backend by dtype and mask, bf16, matmul throughput, a synthetic SASRec train step at three catalogue sizes, full-ranking scoring speed, `torch.compile`, and two minutes of sustained load with clock, temperature and power logging. Its JSON report and text summary are committed under `results/env/`.

## 9. Execution model

GPU work runs on a rented machine, a new one for every batch (D046). A batch is one manual run of the `batch` workflow: it runs a plan, resumes after interruption, skips finished runs and writes configs, metrics, logs and NVML traces to a network volume that outlives the machine. Its export step writes the batch's aggregated CSV file and the summaries, and commits them when the plan finished without a failure. Datasets, checkpoints and per-user scores stay on the volume.

## 10. Compute budget

The total cap is 500 GPU hours on the rented machine (D045; the hours in D012 and D034 were laptop hours). A cap in USD is set once the pilot rerun there has replaced the estimates; until then only M2 work runs, at most 15 GPU hours. METHODS section 12 has the breakdown before measurement and the descoping order.

## 11. Deliverables

- Source code, configs, lock file and tests in this repository (MIT).
- Aggregated results (CSV) and figures; no raw data.
- `docs/REPORT.md`: a paper-style technical report covering every pre-registered hypothesis, deviations, and limitations.

## 12. References

1. W.-C. Kang, J. McAuley. Self-Attentive Sequential Recommendation. ICDM 2018. https://doi.org/10.1109/ICDM.2018.00035
2. F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, P. Jiang. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. CIKM 2019. https://doi.org/10.1145/3357384.3357895
3. A. Petrov, C. Macdonald. A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. RecSys 2022. https://doi.org/10.1145/3523227.3548487
4. A. V. Petrov, C. Macdonald. gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling. RecSys 2023. https://doi.org/10.1145/3604915.3608783
5. A. Klenitskiy, A. Vasilev. Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? RecSys 2023. https://doi.org/10.1145/3604915.3610644
6. W. Krichene, S. Rendle. On Sampled Metrics for Item Recommendation (extended abstract). IJCAI 2021. https://doi.org/10.24963/ijcai.2021/651
7. Z. Meng, R. McCreadie, C. Macdonald, I. Ounis. Exploring Data Splitting Strategies for the Evaluation of Recommendation Models. RecSys 2020. https://doi.org/10.1145/3383313.3418479
8. Y. Ji, A. Sun, J. Zhang, C. Li. A Critical Study on Data Leakage in Recommender System Offline Evaluation. ACM TOIS 41(3), 2023. https://doi.org/10.1145/3569930
9. A. Dallmann, D. Zoller, A. Hotho. A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models. RecSys 2021. https://doi.org/10.1145/3460231.3475943
10. B. Hidasi, Á. T. Czapp. Widespread Flaws in Offline Evaluation of Recommender Systems. RecSys 2023. https://doi.org/10.1145/3604915.3608839
11. A. Klenitskiy, A. Volodkevich, A. Pembek, A. Vasilev. Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations. RecSys 2024. https://doi.org/10.1145/3640457.3688195
12. D. Gusak, A. Volodkevich, A. Klenitskiy, A. Vasilev, E. Frolov. Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders. RecSys 2025. https://doi.org/10.1145/3705328.3748164
13. G. Mezentsev, D. Gusak, I. Oseledets, E. Frolov. Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs. RecSys 2024. https://doi.org/10.1145/3640457.3688140
14. D. Gusak, G. Mezentsev, I. Oseledets, E. Frolov. RECE: Reduced Cross-Entropy Loss for Large-Catalogue Sequential Recommenders. CIKM 2024. https://doi.org/10.1145/3627673.3679986
15. Zhelnin et al. Faster and Memory-Efficient Training of Sequential Recommendation Models for Large Catalogs. arXiv:2509.09682, 2025 (preprint).
16. R. Cañamares, P. Castells. On Target Item Sampling in Offline Recommender System Evaluation. RecSys 2020. https://doi.org/10.1145/3383313.3412259
17. D. Gusak, A. Volodkevich, E. Frolov. More Test Users, More Overconfidence: Seed Variance in Sequential Recommender Tests. RecSys 2026. https://doi.org/10.1145/3773078.3841281
18. F. M. Harper, J. A. Konstan. The MovieLens Datasets: History and Context. ACM TiiS 5(4), 2015. https://doi.org/10.1145/2827872
19. Y. Hou, J. Li, Z. He, A. Yan, X. Chen, J. McAuley. Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952v1, 2024. Dataset: https://amazon-reviews-2023.github.io/
20. NVIDIA GeForce RTX 50 Series Laptops. https://www.nvidia.com/en-us/geforce/laptops/50-series/
21. PyTorch 2.15 release blog. https://pytorch.org/blog/pytorch-2-15-release-blog/
22. PyTorch v2.15.0 release notes. https://github.com/pytorch/pytorch/releases/tag/v2.15.0
23. How to use torch.compile on Windows CPU/XPU. https://docs.pytorch.org/tutorials/unstable/inductor_windows.html
24. triton-windows. https://github.com/triton-lang/triton-windows
25. nvidia-ml-py on PyPI. https://pypi.org/project/nvidia-ml-py/
26. NVIDIA CUDA on WSL User Guide. https://docs.nvidia.com/cuda/wsl-user-guide/index.html
27. NVIDIA NIM on WSL2, benchmarking guide. https://docs.nvidia.com/nim/wsl2/latest/benchmarking.html
28. NVIDIA Developer Blog. Leveling up CUDA Performance on WSL2 with New Enhancements, 2021. https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements/
29. Shevchenko et al. From Variability to Stability: Advancing RecSys Benchmarking Practices. KDD 2024. https://doi.org/10.1145/3637528.3671655
30. PyTorch v2.15.0 Windows wheel build configuration (CUDA architectures per variant). https://github.com/pytorch/pytorch/blob/v2.15.0/.ci/wheel/windows/build_env_setup.py
31. SASRec official code. https://github.com/kang205/SASRec
32. BERT4Rec official code. https://github.com/FeiSun/BERT4Rec
33. gSASRec PyTorch code. https://github.com/asash/gSASRec-pytorch
34. Runpod pricing, read 2026-11-10. https://www.runpod.io/pricing
35. NVIDIA CUDA Compiler Driver NVCC, v13.5, chapter 5 (GPU compilation: binary compatibility, just-in-time compilation, fatbinaries). https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
36. PyTorch v2.15.0, `torch/cuda/__init__.py` (`DEVICE_REQUIREMENT`, `PYTORCH_RELEASES_CODE_CC`, `_check_capability`). https://github.com/pytorch/pytorch/blob/v2.15.0/torch/cuda/__init__.py
37. Triton v3.9.0, NVIDIA backend driver (`get_current_target`). https://github.com/triton-lang/triton/blob/v3.9.0/third_party/nvidia/backend/driver.py
