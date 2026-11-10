# Tasks

Milestones and checklists. Estimates: development hours / GPU hours / owner time.

## M0 Research, specification and feasibility (8 h / under 0.5 h / 15 min)

- [x] Literature, reference code, dataset terms and platform research
- [x] `docs/SPEC.md`, `docs/METHODS.md` (draft), `docs/TASKS.md`, `docs/DECISIONS.md` (D001 to D018)
- [x] uv project with locked PyTorch 2.15 (`cu132` on Windows; `cpu` and `cu132` extras on Linux)
- [x] `scripts/check_env.py` with CPU tests
- [x] Owner runs `scripts/check_env.py` on the laptop; JSON and summary committed under `results/env/`
- [x] Review the check results: SDPA backend availability and accuracy, `torch.compile`, memory per catalogue size, thermal behaviour; update D002, D003, D011 or the budget if they contradict assumptions

## M1 Data, evaluation and runner (18 h / 0 h / 30 min)

Data
- [x] Download scripts with SHA-256 records for the four datasets
- [x] Preprocessing (METHODS 2) and statistics table in `results/data/`
- [x] Compare statistics with published ones; explain differences above 1% in DECISIONS
- [x] Leave-one-out and temporal splits as typed split objects (METHODS 3)
- [x] Shuffled-order transform (METHODS 3.3)

Evaluation
- [x] Vectorised full-ranking evaluator with history masking and pessimistic ties
- [x] Single logged test reader (phases `final` and `anchor`, `test_access.jsonl`)

Correctness (METHODS 11)
- [x] Invariant assertions for preprocessing, both splits, training-only statistics and evaluation inputs; results stored in the run manifest
- [x] Known-answer tests: random scorer expectation, tie handling, hand-computed cases
- [x] Naive reference metrics and ranx cross-check (confirm ranx licence first)
- [x] Mutation tests for every data and evaluation mutation in 11.3 that does not need a model; mutation-to-detector table

Runner
- [x] Config schema and run directory layout (config, metrics, logs, NVML trace, manifest)
- [x] Batch runner: one command per batch, resumable, skips finished runs
- [x] Export command: aggregated CSV, test-access counts, truncation and GPU-hour summaries
- [x] NVML sampler shared with `scripts/check_env.py`

Freeze
- [x] Close METHODS section 13 open items
- [x] Silent-error review (independent, code and data flow only) and whole-diff review; fixes
- [x] Freeze METHODS (D014) before any model is trained on real data

## M2 Models, objectives and pilot (19 h / 10 h / 30 min)

- [x] SASRec backbone with causal and padding masks
- [x] BERT4Rec with Cloze training and mask-token inference
- [x] Objectives: full CE, chunked CE, sampled softmax, BCE, gBCE
- [x] Floor baselines: MostPop, first-order transition
- [x] Causality perturbation test; planted-rule test (NDCG@10 at least 0.95); shuffled-label and shuffled-order known-answer tests (D027)
- [x] Remaining mutation tests (label shift, causal mask removed, test read during tuning)
- [x] Tuning harness: Optuna TPE studies in SQLite, pruning, caps and truncation records (D028)
- [x] Pilot on the laptop: anchors on MovieLens-1M (METHODS 11.5), 3 seeds per arm for σ (validation only), epoch-time measurements on all datasets
- [ ] Rented machine (D045, D046): runner set up, `check_env` report committed, admission test passed
- [ ] Pilot, anchors and the gSASRec rerun of D035 on the rented machine
- [x] Anchors inside their bands within three anchor reads per arm, or the arm flagged as in METHODS 11.5 (D032: all four inside after one read)
- [ ] Seed count from METHODS 7.5 (rule implemented: `seqrec seeds`); updated GPU projection from the rerun; descoping decisions if needed
- [ ] Silent-error review and whole-diff review; fixes

Before M3 tuning starts
- [ ] CPU continuous integration: tests and lint on every push; tests that need prepared real data are skipped there, and the workflow says so
- [ ] Invariant checks run once per process and are cached with the splits (`load_splits`), so a dataset rebuilt while a batch runs is not checked again; key the cache on the stored file, or rerun the checks per run
- [ ] No check that `seqs.items` equals `full.items[seqs.pos]` in every split; add it to the invariants (METHODS 11.1)
- [ ] bf16 training is checked only by the output dtype; also check an intermediate activation
- [ ] The known-answer test for the shuffled-order transform (METHODS 3.3) is weak; test exact orders on a hand-made case
- [ ] The anchor-read limit is enforced in `test_run`, not in `read_test`; move it into the reader so every caller is bound by it
- [x] Median pruning counts its steps from a trial's start epoch, so gSASRec trials are not compared inside the plateau before the start rule fires (D044; deviation from METHODS 5.4)
- [ ] Training steps of small configurations on Steam and Video_Games are bound by kernel launches (about 6.3 ms per step against 1.1 to 1.8 ms of GPU work, D039; laptop numbers, to be measured again on the rented machine, D045), so wall-clock time to target (H2a) mostly measures launch overhead. Decide the execution mode before any M3 run, applied to every arm: fixed shapes (loss over all positions with a mask instead of boolean indexing), fused Adam, then CUDA Graphs, with a prototype in `scripts/profile_step.py` measuring the gain first. Record the effect on RQ2 and repeat the pilot timings and the D034 projection afterwards

## M3 RQ1 and RQ2 runs (6 h / projection pending the pilot rerun, D045 / no owner time per batch)

- [ ] Check whether the negative-sampling comparison announced in the amazon-science/RecArena repository has been published; record its overlap with RQ2 in DECISIONS
- [ ] Tuning plans take a `start_limit` field, doubled together with the caps when a plan is doubled (D035)
- [ ] Tuning batches per dataset (leave-one-out)
- [ ] After the first gSASRec study: read the share of trials that start, and decide whether `no_start` trials need another treatment in TPE (D044)
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
- [ ] Controlled sessions per METHODS 9 and D045 (2 to 3 sessions on the rented machine)
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
