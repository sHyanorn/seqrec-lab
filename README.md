# seqrec-lab

A study of Transformer sequential recommenders (SASRec, BERT4Rec, gSASRec) on one consumer laptop GPU. It covers how robust published quality gaps are under equal tuning and several seeds, what training objectives cost in time and memory, and where training time goes.

- [Specification](docs/SPEC.md)
- [Methods (pre-registered, frozen)](docs/METHODS.md)
- [Decisions](docs/DECISIONS.md)
- [Tasks](docs/TASKS.md)

Status: milestone M1 (data pipeline, evaluation and runner).

## Environment check

Requires [uv](https://docs.astral.sh/uv/). On the GPU machine:

```
uv sync --locked
uv run python scripts/check_env.py
```

The report is written to `results/env/`. `--quick` runs tiny shapes and also works on CPU.

## Tests

```
uv sync --locked
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
uv run seqrec run plans/<plan>.toml    # train; resumable, finished runs are skipped
uv run seqrec test plans/<plan>.toml   # read test data once per run, after all are trained
uv run seqrec export                   # aggregate to results/runs/
```

Test data are read only by `seqrec test`, once per final run, through a logged reader. `export` writes the aggregates, lists any test-access count that breaks the protocol (METHODS 11.6) and then exits with status 1.

## License

MIT
