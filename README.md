# seqrec-lab

A study of Transformer sequential recommenders (SASRec, BERT4Rec, gSASRec) on one consumer laptop GPU. It covers how robust published quality gaps are under equal tuning and several seeds, what training objectives cost in time and memory, and where training time goes.

- [Specification](docs/SPEC.md)
- [Methods (pre-registered, frozen)](docs/METHODS.md)
- [Decisions](docs/DECISIONS.md)
- [Tasks](docs/TASKS.md)

Status: milestone M2 (models, objectives and pilot).

## Environment check

Requires [uv](https://docs.astral.sh/uv/). Linux locks two torch builds, so the sync names one; Windows has only the CUDA build:

```
uv sync --locked                  # Windows
uv sync --locked --extra cu132    # Linux with an NVIDIA GPU
uv sync --locked --extra cpu      # Linux without a GPU
```

On the GPU machine:

```
uv run python scripts/check_env.py
```

The report is written to `results/env/`. `--quick` runs tiny shapes and also works on CPU.

## Tests

```
uv run pytest
```

## Data

No dataset is redistributed; each keeps its own terms of use (see [SPEC section 5](docs/SPEC.md#5-datasets)). Download, preprocess and summarise all four datasets:

```
uv run seqrec data prepare
```

Raw and processed files go to `data/` (not committed). Aggregate statistics go to `results/data/`. Every raw file is checked against a pinned SHA-256. If Python cannot verify a server's certificate chain (seen with files.grouplens.org on Windows), the download retries with `curl`; failing that, the error message gives the manual download command. Save the file into `data/raw/<dataset>/` and rerun with `--datasets <dataset>`.

## Runs

```
uv run seqrec run configs/plans/<plan>.toml    # train; resumable, finished runs are skipped
uv run seqrec test configs/plans/<plan>.toml   # read test data once per run, after all are trained
uv run seqrec tune configs/plans/<plan>.toml   # Optuna studies; resumable
uv run seqrec export                           # aggregate to results/runs/
uv run seqrec seeds results/runs/<batch>.csv   # seed count from pilot validation scores
uv run seqrec anchors results/runs/m2-anchor.csv  # anchors against their bands (D032)
uv run seqrec budget results/runs/m2-pilot.csv   # GPU-hour projection (D034)
```

`seeds`, `budget` and `anchors --pilot` accept several exports; runs in a later file replace earlier runs with the same tag, dataset, split, shuffle and arm (D035).

Arms: `sasrec-bce`, `sasrec-ce`, `sasrec-ss`, `gsasrec`, `bert4rec`, `mostpop`, `transition`.

If training raises, the run is retried once; if the GPU stops responding, the command exits with status 1 and rerunning it continues (D031). Test data are read only by `seqrec test`, once per final run, through a logged reader. `export` writes the aggregates, lists any test-access count that breaks the protocol (METHODS 11.6) and then exits with status 1.

## License

MIT
