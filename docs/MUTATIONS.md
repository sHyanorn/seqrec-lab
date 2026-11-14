# Mutation checks

Each row is a silent bug injected by a test in `tests/test_mutations.py` (METHODS 11.3) and the check that catches it. A mutation without a detector gets a new test before the milestone is delivered.

| Mutation | Test | Caught by |
|---|---|---|
| Validation item placed in the training sequence | `test_validation_item_in_training` | train: no evaluation targets |
| Test item placed in the training sequence | `test_test_item_in_training` | train: no evaluation targets |
| Targets off by one position | `test_targets_off_by_one` | train: no evaluation targets |
| Target included in its own input (evaluation inputs only) | `test_target_inside_its_input` | validation: inputs precede target |
| Evaluation input drops an earlier item | `test_evaluation_input_drops_an_item` | input holds every earlier in-vocabulary item; loo: validation input equals training |
| Training rows out of step with the validation input (reversed, swapped between two users of equal length, shuffled with other keys) | `test_training_rows_out_of_step_with_the_validation_input` | loo: validation input equals training sequence |
| Items of a sequence reordered apart from their log positions | `test_items_detached_from_their_log_positions` | train, validation, test: items match log |
| Popularity computed on all data | `test_statistic_from_all_data` | statistic was not computed from this training split |
| Temporal cutoff off by one at tied timestamps | `test_temporal_cutoff_off_by_one_at_ties` | gts: validation targets inside validation period |
| Temporal vocabulary taken from the whole log | `test_temporal_vocabulary_from_whole_log` | gts: vocabulary is the 5-core of the training period |
| Temporal vocabulary from a single-pass filter, in the split builder and the shared k-core code alike | `test_temporal_vocabulary_from_a_single_pass_filter` | gts: vocabulary is the 5-core of the training period, recomputed by the check's own code |
| Temporal validation target is the first item of the period | `test_temporal_validation_target_first_not_last` | gts: validation target is the last item up to the cutoff |
| Temporal evaluation sets missing half their users, or holding a user twice (validation and test) | `test_temporal_evaluation_users_differ_from_the_registered_ones` | gts: validation target is the last item up to the cutoff, with the users recomputed from the source log |
| Temporal split's times one unit early inside the training period, so an interaction of the validation period enters training | `test_temporal_split_times_one_unit_early_in_the_training_period` | gts: vocabulary is the 5-core of the training period, which compares the split's rows with the log's by user and time |
| Temporal split's user boundary moved by one row between two users in no evaluation set | `test_temporal_split_rows_moved_between_two_training_users` | gts: vocabulary is the 5-core of the training period, which compares the split's rows with the log's by user and time |
| Temporal training sequences missing half their users, or given a user with one vocabulary interaction | `test_temporal_training_users_differ_from_the_registered_ones` | gts: validation target is the last item up to the cutoff, which also compares the training sequences with those of METHODS 3.2 from the source log |
| Leave-one-out user boundary moved by one row, times one unit early, or items numbered against key order | `test_leave_one_out_rows_differ_from_the_log` | loo: sequence lengths, which compares each user's sequence with the rows the 5-core of the whole log keeps |
| Leave-one-out sequences from a 6-core | `test_leave_one_out_rows_from_a_six_core` | loo: sequence lengths, with the 5-core recomputed by the check's own code |
| Leave-one-out sequences from a single-pass filter | `test_leave_one_out_rows_from_a_single_pass_filter` | items meet 5-core |
| History masking removed | `test_history_masking_removed` | candidate count must equal N minus distinct history items |
| History masking limited to the truncated input | `test_history_masking_limited_to_truncated_input` | candidate count must equal N minus distinct history items |
| History masking applied to the target | `test_history_masking_applied_to_target` | target score must stay finite after masking |
| Evaluation scores in bf16 | `test_scores_in_bf16` | evaluation scores must be fp32 |
| Tuning run evaluates on test data | `tests/test_runner.py::test_tuning_run_cannot_reach_test_data` | the sealed stand-in raises in tuning runs |
| Test data opened outside the approved paths | `tests/test_isolation.py` | static scan of `unseal` calls |
| Test read in a tuning run | `tests/test_runner.py::test_batches_through_cli`, `test_test_reader_rejects_other_phases` | reader refuses phase `tune`; export recounts reads |
| A final run read twice, or an anchor run given a final read | `tests/test_runner.py::test_export_reports_a_final_run_that_read_test_data_twice`, `test_export_reports_an_anchor_run_that_used_a_final_read` | export recounts reads |
| A final plan tested with other leave-one-out seeds than METHODS 7.5 gives, or with a seed record of other runs | `tests/test_seed_record.py` | the seed record check before any read (D064) |
| A fourth anchor read for an arm | `tests/test_runner.py::test_anchor_reads_are_capped`, `test_the_reader_itself_refuses_a_fourth_anchor_read` | the reader's own count over all runs and machines |
| Last checkpoint scored instead of the best | `tests/test_runner.py::test_best_checkpoint_is_the_one_scored` | validation score of the restored checkpoint |

Added in M2, with the models:

| Mutation | Test | Caught by |
|---|---|---|
| SASRec labels equal to the inputs (no shift) | `test_sasrec_labels_not_shifted` | batch check: labels are the inputs shifted by one position |
| BCE negatives drawn excluding the full history, test item included | `test_bce_negatives_exclude_the_full_history` | sampler fingerprint differs from the training split |
| SASRec causal mask removed | `test_sasrec_causal_mask_removed` | causality perturbation (`tests/test_sasrec.py::assert_causal`) |
| Softmax over the padding id as well as the items | `test_softmax_includes_the_padding_id` | naive softmax over items 1..N |
| Sampled-softmax negatives equal to the positive kept | `test_sampled_softmax_keeps_collisions` | loss with the colliding column removed |
| BERT4Rec scored without an appended mask token | `tests/test_known_answers.py::test_bert4rec_scored_without_its_mask_token_fails` | planted-rule NDCG@10 below 0.95 |
| Tuning trial evaluates on test data | `tests/test_tune.py::test_trials_cannot_reach_test_data` | the sealed stand-in raises inside a trial, which fails |
