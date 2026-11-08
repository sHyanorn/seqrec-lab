# Mutation checks

Each row is a silent bug injected by a test in `tests/test_mutations.py` (METHODS 11.3) and the check that catches it. A mutation without a detector gets a new test before the milestone is delivered.

| Mutation | Test | Caught by |
|---|---|---|
| Validation item placed in the training sequence | `test_validation_item_in_training` | train: no evaluation targets |
| Test item placed in the training sequence | `test_test_item_in_training` | train: no evaluation targets |
| Targets off by one position | `test_targets_off_by_one` | train: no evaluation targets |
| Target included in its own input | `test_target_inside_its_input` | train: no evaluation targets |
| Popularity computed on all data | `test_statistic_from_all_data` | statistic was not computed from this training split |
| Temporal cutoff off by one at tied timestamps | `test_temporal_cutoff_off_by_one_at_ties` | gts: validation targets inside validation period |
| History masking removed | `test_history_masking_removed` | candidate count must equal N minus distinct history items |
| History masking limited to the truncated input | `test_history_masking_limited_to_truncated_input` | candidate count must equal N minus distinct history items |
| History masking applied to the target | `test_history_masking_applied_to_target` | target score must stay finite after masking |
| Evaluation scores in bf16 | `test_scores_in_bf16` | evaluation scores must be fp32 |
| Tuning run reads test data | `tests/test_runner.py::test_tuning_run_never_touches_test_data`, `test_test_reader_rejects_other_phases`, export check in `test_batch_and_export_through_cli` | test reader refuses phase `tune`; export flags any tuning run with test reads |

Added in M2, with the models:

| Mutation | Planned detector |
|---|---|
| SASRec labels equal to the inputs (no shift) | label-shift assertion in the training batch builder |
| BCE negatives drawn excluding the full history, test item included | negative sampler accepts only the training split |
| SASRec causal mask removed | causality perturbation test |
