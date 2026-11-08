# seqrec-lab

A study of Transformer sequential recommenders (SASRec, BERT4Rec, gSASRec) on one consumer laptop GPU. It covers how robust published quality gaps are under equal tuning and several seeds, what training objectives cost in time and memory, and where training time goes.

- [Specification](docs/SPEC.md)
- [Methods (pre-registration draft)](docs/METHODS.md)
- [Decisions](docs/DECISIONS.md)
- [Tasks](docs/TASKS.md)

Status: milestone M0 (specification and feasibility check).

## Environment check

Requires [uv](https://docs.astral.sh/uv/). On the GPU machine:

```
uv sync --group compile
uv run python scripts/check_env.py
```

The report is written to `results/env/`. `--quick` runs tiny shapes and also works on CPU.

## Tests

```
uv sync
uv run pytest
```

## Data

No dataset is redistributed. Download and preprocessing scripts arrive in M1; each dataset keeps its own terms of use (see [SPEC section 5](docs/SPEC.md#5-datasets)).

## License

MIT
