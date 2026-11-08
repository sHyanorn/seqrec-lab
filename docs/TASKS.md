# Tasks

Milestones and checklists. Estimates: development hours / GPU hours / owner time.

## M0 Research, specification and feasibility (8 h / under 0.5 h / 15 min)

- [x] Literature, reference code, dataset terms and platform research
- [x] `docs/SPEC.md`, `docs/METHODS.md` (draft), `docs/TASKS.md`, `docs/DECISIONS.md` (D001 to D018)
- [x] uv project with locked PyTorch 2.15 (`cu132` on Windows, CPU on Linux)
- [x] `scripts/check_env.py` with CPU tests
- [x] Owner runs `scripts/check_env.py` on the laptop; JSON and summary committed under `results/env/`
- [x] Review the check results: SDPA backend availability and accuracy, `torch.compile`, memory per catalogue size, thermal behaviour; update D002, D003, D011 or the budget if they contradict assumptions

## M1 Data, evaluation and runner (18 h / 0 h / 30 min)

Data
- [ ] Download scripts with SHA-256 records for the four datasets
- [ ] Preprocessing (METHODS 2) and statistics table in `results/data/`
- [ ] Compare statistics with published ones; explain differences above 1% in DECISIONS
- [ ] Leave-one-out and temporal splits as typed split objects (METHODS 3)
- [ ] Shuffled-order transform (METHODS 3.3)

Evaluation
- [ ] Vectorised full-ranking evaluator with history masking and pessimistic ties
- [ ] Single logged test reader (phases `final` and `anchor`, `test_access.jsonl`)

Correctness (METHODS 11)
- [ ] Invariant assertions for preprocessing, both splits, training-only statistics and evaluation inputs; results stored in the run manifest
- [ ] Known-answer tests: random scorer expectation, tie handling, hand-computed cases
- [ ] Naive reference metrics and ranx cross-check (confirm ranx licence first)
- [ ] Mutation tests for every data and evaluation mutation in 11.3 that does not need a model; mutation-to-detector table

Runner
- [ ] Config schema and run directory layout (config, metrics, logs, NVML trace, manifest)
- [ ] Batch runner: one command per batch, resumable, skips finished runs
- [ ] Export command: aggregated CSV, test-access counts, truncation and GPU-hour summaries
- [ ] NVML sampler shared with `scripts/check_env.py`

Freeze
- [ ] Close METHODS section 13 open items
- [ ] Silent-error review (independent, code and data flow only) and whole-diff review; fixes
- [ ] Freeze METHODS (D014) before any model is trained on real data

## M2 Models, objectives and pilot (19 h / 10 h / 30 min)

- [ ] SASRec backbone with causal and padding masks
- [ ] BERT4Rec with Cloze training and mask-token inference
- [ ] Objectives: full CE, chunked CE, sampled softmax, BCE, gBCE
- [ ] Floor baselines: MostPop, first-order transition
- [ ] Causality perturbation test; planted-rule test (NDCG@10 at least 0.95); shuffled-label and shuffled-order known-answer tests
- [ ] Remaining mutation tests (label shift, causal mask removed, test read during tuning)
- [ ] Tuning harness: Optuna TPE studies in SQLite, pruning, caps and truncation records
- [ ] Pilot on the laptop: anchors on MovieLens-1M (METHODS 11.5), 3 seeds per arm for σ (validation only), epoch-time measurements on all datasets
- [ ] Anchors inside their bands within three anchor reads per arm, or the arm flagged as in METHODS 11.5
- [ ] Seed count from METHODS 7.5; updated GPU projection; descoping decisions if needed
- [ ] Silent-error review and whole-diff review; fixes

## M3 RQ1 and RQ2 runs (6 h / 128 to 163 h plus seed reserve / about 3 weeks of laptop time)

- [ ] Check whether the negative-sampling comparison announced in the amazon-science/RecArena repository has been published; record its overlap with RQ2 in DECISIONS
- [ ] Tuning batches per dataset (leave-one-out)
- [ ] Truncation check per arm; doubled caps where required (METHODS 5.2)
- [ ] Final runs on leave-one-out and the temporal split, seeds per METHODS 7.5
- [ ] Re-estimate σ after 5 seeds per dataset and add seeds if the rule requires
- [ ] Shuffled-order control runs (D017)
- [ ] Export and push aggregated results
- [ ] Silent-error review and whole-diff review
- [ ] RQ4 go or no-go (D015)

## M4 RQ3 efficiency (10 h / 15 h / 1 h)

- [ ] Benchmark harness: backends, mask forms, precisions, lengths, catalogue sizes, `torch.compile`
- [ ] Component timing with CUDA events and profiler attention share
- [ ] Controlled sessions per METHODS 9 (2 to 3 sessions on the laptop)
- [ ] bf16 vs fp32 quality runs for H3c
- [ ] Export results

## M5 RQ4 retrieval, conditional (8 h / 15 h / 15 min)

- [ ] Embedding export from final runs
- [ ] faiss exact, HNSW and IVF indices with parameters fixed in advance
- [ ] Recall@10 against exact search, NDCG@10 change, latency
- [ ] Silent-error review and whole-diff review

## M6 Analysis and report (12 h / 0 h / 0)

- [ ] Statistical analysis exactly as in METHODS 7 to 10; deviations listed
- [ ] Figures and tables
- [ ] `docs/REPORT.md` and README
- [ ] Final whole-repository review
