# seqrec-lab

A pre-registered study of Transformer sequential recommenders, SASRec, BERT4Rec and gSASRec, on one consumer GPU. It asks how robust published quality gaps are under equal tuning and several seeds, what training objectives cost in time and memory, and where training time goes. In the leave-one-out analysis of MovieLens-1M, Video_Games and Steam, registered minimum-effect tests find SASRec trained with BCE and one negative worse in NDCG@10 than with full cross-entropy by more than 2% on all three datasets. Registered directional tests place SASRec with full cross-entropy above BERT4Rec and gSASRec in NDCG@10 on all three; on MovieLens-1M these comparisons are labelled underpowered. MovieLens-1M's test targets were read for anchor checks before its final runs. MovieLens-20M has no final runs yet.

## Results

Confirmatory results of milestone M3, from [docs/RESULTS.md](docs/RESULTS.md); every row but the first is the leave-one-out analysis. Differences are in mean log NDCG@10, shown as relative differences with descriptive 95% Welch intervals. The smallest effect of interest (SESOI) is a 2% relative difference.

| | MovieLens-1M | Video_Games | Steam |
|---|---|---|---|
| Final runs | 99 in the batch, 2.88 GPU hours; ten seeds per neural arm in leave-one-out | 65 in the batch, 28.50 GPU hours; six seeds per neural arm in leave-one-out | 59 in the batch, 50.56 GPU hours; five seeds per neural arm in leave-one-out |
| SASRec-CE over SASRec-BCE | +34.3% (+32.4% to +36.2%) | +39.7% (+38.2% to +41.3%) | +26.5% (+25.5% to +27.4%) |
| SASRec-CE over BERT4Rec | +19.5% (+9.6% to +30.3%) | +22.8% (+21.3% to +24.3%) | +2.9% (+2.1% to +3.7%) |
| gSASRec against SASRec-CE | -4.1% (-5.0% to -3.1%) | -8.0% (-8.4% to -7.5%) | -1.3% (-2.0% to -0.7%) |
| H1a: SASRec-CE is not below BERT4Rec by more than the SESOI | supported | not registered | not registered |
| H1b: SASRec-CE, gSASRec and BERT4Rec are each above SASRec-BCE by more than the SESOI | pending | supported | supported |
| H2a: SASRec-CE reaches the target sooner than SASRec-BCE, SASRec-SS and gSASRec, on seeds 1 to 5 trained on one Pod | supported, resting on censored runs: 14 of the 15 runs of the other arms never reach the target | not supported | supported; the five SASRec-BCE runs never reach the target |
| H2b: sampled softmax and gBCE are equivalent to full cross-entropy within the SESOI | not supported | not supported | not supported |
| H2b: BCE with one negative is worse than full cross-entropy by more than the SESOI | supported | supported | supported |

- On MovieLens-1M, every RQ1 conclusion carries the note that its test targets were read for anchor checks, and every directional and equivalence comparison is labelled underpowered.
- On MovieLens-1M the published gaps of SASRec-CE over BERT4Rec and of gSASRec over BERT4Rec are confirmed; the gap of BERT4Rec over SASRec-BCE is pending.
- On Steam the published gap of BERT4Rec over SASRec-BCE is confirmed. The published gaps of gSASRec and of SASRec-CE against BERT4Rec are reversed: both are above BERT4Rec, where the published sign has BERT4Rec ahead.
- H2a is decided on one kind of host, a Ryzen 9 7950X with 14 CPUs.
- A pending result is not a result: it can still change when the other datasets have run (D051).

## How it works

The project is at milestone M3, the RQ1 and RQ2 runs. The [methods](docs/METHODS.md) were registered at commit 569676f, before any model code; section 9.4 was added in 50f0df7, and deviations are recorded in the [decisions](docs/DECISIONS.md). The arms are `sasrec-bce`, `sasrec-ce`, `sasrec-ss`, `gsasrec` and `bert4rec`, with `mostpop` and `transition` as floor baselines, tuned with equal budgets and evaluated by full ranking. Test data are read only by `seqrec test`, once per final run, through a logged reader; `export` lists any test-access count that breaks the protocol (METHODS 11.6) and then exits with status 1.

GPU runs happen on a rented machine through the manual `batch` workflow (D045, D046, D047). `scripts/cloud/` holds the machine's setup and `configs/cloud/pod.json` the request a batch's machine is created from. Exports are in `results/runs/`; those of the laptop used up to the M2 pilot are in `results/laptop/`. The [specification](docs/SPEC.md) and [tasks](docs/TASKS.md) give the scope and progress.

## Run it

Requires [uv](https://docs.astral.sh/uv/). Linux locks two torch builds, so the sync names one; Windows has only the CUDA build:

```
uv sync --locked                  # Windows
uv sync --locked --extra cu132    # Linux with an NVIDIA GPU
uv sync --locked --extra cpu      # Linux without a GPU; as CI runs it
uv run --no-sync pytest -q        # tests
uv run python scripts/check_env.py   # on the GPU machine; --quick runs tiny shapes, also on CPU
uv run seqrec data prepare        # download, preprocess and summarise all four datasets
```

The environment report goes to `results/env/`. Raw and processed data go to `data/` (not committed), and aggregate statistics to `results/data/`. Every raw file is checked against a pinned SHA-256. If Python cannot verify a server's certificate chain (seen with files.grouplens.org on Windows), the download retries with `curl`; failing that, the error message gives the manual download command. Save the file into `data/raw/<dataset>/` and rerun with `--datasets <dataset>`.

```
uv run seqrec run configs/plans/<plan>.toml    # train; resumable, finished runs are skipped
uv run seqrec test configs/plans/<plan>.toml   # read test data once per run, after all are trained
uv run seqrec tune configs/plans/<plan>.toml   # Optuna studies; resumable
uv run seqrec export                           # aggregate to results/runs/
uv run seqrec seeds results/runs/<batch>.csv   # seed count from pilot validation scores
uv run seqrec anchors results/runs/m2-anchor.csv  # anchors against their bands (D032)
uv run seqrec budget results/runs/m2-pilot.csv   # GPU-hour projection (D034)
```

`seeds`, `budget` and `anchors --pilot` accept several exports; runs in a later file replace earlier runs with the same tag, dataset, split, shuffle and arm (D035). If training raises, the run is retried once; if the GPU stops responding, the command exits with status 1 and rerunning it continues (D031).

## Limits

- MovieLens-20M has no final runs, so every registered comparison, hypothesis and gap class on it is pending.
- Before the final runs, the anchor reads of METHODS 11.5 read the MovieLens-1M test targets twice for each of SASRec-CE, SASRec-BCE, BERT4Rec and gSASRec (D032).
- Resampling seeds with replacement, with five to ten seeds per arm, understates the variation between runs (D056).
- On small configurations, time to target mostly measures launch overhead (D048).
- Peak memory and step time of plain against chunked cross-entropy are not reported yet.

## Data and licence

No dataset is redistributed; each keeps its own terms of use (see [SPEC section 5](docs/SPEC.md#5-datasets)). Code: MIT.
