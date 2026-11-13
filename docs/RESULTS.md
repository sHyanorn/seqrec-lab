# Results

Confirmatory results per dataset, under the tests, families and thresholds of METHODS 7 and 8 and the reporting rules of D051. Every number below comes from the tables in `results/analysis/` (`tests.csv`, `conclusions.csv`, `arms.csv`, `times.csv` and the bootstrap tables `bootstrap-<batch>.csv`) or from the batch files in `results/runs/`. The four tables are written by `seqrec analyze results/runs/m3-tune-ml-1m.csv results/runs/m3-final-ml-1m.csv results/runs/m3-tune-video-games.csv results/runs/m3-final-video-games.csv --out-dir results/analysis`; the tuning files enter only the median SM clock that flags a run.

## Reading the statuses

- Holm's correction uses the registered size of a family over all four datasets (METHODS 7.4). MovieLens-1M and Video_Games have final runs.
- A comparison is **rejected** if Holm rejects it whatever the remaining datasets give, **not rejected** if Holm cannot reject it whatever they give, and **pending** otherwise (D051). A hypothesis or gap class is pending while a comparison it depends on can still change it. Nothing pending is a result.
- A dataset's runs can decide a pending comparison on another dataset of the same family. With the Video_Games runs, one MovieLens-1M comparison went from pending to rejected (HR@10, under Robustness); no other MovieLens-1M status changed.
- Steam and MovieLens-20M have no final runs, so every comparison, hypothesis and gap class on them is pending.
- Counts required by METHODS 7.4: H1a is supported on 1 dataset and pending on 1 (MovieLens-20M); H1b is supported on 1 (Video_Games) and pending on 3.

## MovieLens-1M

### Runs

- Batch `m3-final-ml-1m`: 99 final runs, all on Pod `mee79yy65ukfve` (RTX 4090, Ryzen 9 7950X), 2.88 GPU hours. Every run trained; none failed, none was truncated.
- Leave-one-out has ten seeds per neural arm (D053), the temporal split five, and each floor baseline runs once per split.
- Every run read test data once: 99 runs with one final read, one completion record and no anchor read. The 100 tuning runs of `m3-tune-ml-1m` read test data zero times (METHODS 11.6).

### Notes carried by every statement on this dataset

1. **Anchor reads.** Before these runs, the anchor reads of METHODS 11.5 read the MovieLens-1M test targets twice for each of SASRec-CE, SASRec-BCE, BERT4Rec and gSASRec (D032). Every RQ1 conclusion on MovieLens-1M carries this note.
2. **Power.** At the σ of this dataset (below) and ten seeds, the registered power at the SESOI is 0.005 for a directional test and 0.000 for an equivalence test; with the five seeds of the temporal split it is 0.003 and 0.000. Every directional and equivalence comparison on this dataset is therefore labelled **underpowered** (METHODS 7.5). A non-rejection is never read as the absence of an effect; rejections count normally. `tests.csv` also gives each of these comparisons a supplementary power at the σ of its own two arms; it enters no status or label (D055).
3. **One-sided tests.** METHODS 7.5 defines no power function for the non-inferiority and minimum-effect tests, so they report no power (D051). The time-to-target tests have none either.
4. **Time to target.** D048 found that on small configurations time to target mostly measures launch overhead (the GPU was busy for 0.295 of a step in its configuration 1, SASRec with BCE on Steam). MovieLens-1M was not among the configurations measured there.
5. **Flagged runs.** The repeat of the timing analysis without flagged runs uses the median SM clock of the M3 runs exported so far, so it is provisional until all four datasets have run (D051).

### Seed variance

- Standard deviation of log validation NDCG@10 over seeds 1 to 5: BERT4Rec 0.17128, SASRec-SS 0.01298, gSASRec 0.00931, SASRec-CE 0.00814, SASRec-BCE 0.00359. Pooled σ = 0.07703 (D053).
- BERT4Rec accounts for 98.9% of the pooled variance. Its five validation NDCG@10 values are 0.1692, 0.1174, 0.1724, 0.1729 and 0.1740: one seed, seed 2, lies apart from the other four.
- On test data the same run has NDCG@10 0.1138; the other nine BERT4Rec runs lie between 0.1632 and 0.1695. The run is a finished run under METHODS 5.2 and enters every analysis (D053).

### Quality per arm

Mean over seeds, standard deviation in brackets. Leave-one-out:

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 10 | 0.1425 (0.0025) | 0.2635 (0.0037) | 0.1517 (0.0007) | 0.12 |
| SASRec-CE | 10 | 0.1913 (0.0024) | 0.3291 (0.0037) | 0.2062 (0.0016) | 0.37 |
| SASRec-SS | 10 | 0.1834 (0.0020) | 0.3169 (0.0033) | 0.1937 (0.0024) | 0.31 |
| gSASRec | 10 | 0.1836 (0.0016) | 0.3199 (0.0025) | 0.1950 (0.0013) | 0.26 |
| BERT4Rec | 10 | 0.1611 (0.0167) | 0.2907 (0.0261) | 0.1666 (0.0175) | 0.48 |
| MostPop | 1 | 0.0180 | 0.0369 | 0.0207 | 0.00 |
| Transition | 1 | 0.1259 | 0.2144 | 0.1345 | 0.00 |

Temporal split (secondary):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 5 | 0.0842 (0.0014) | 0.1514 (0.0046) | 0.1219 (0.0024) | 0.03 |
| SASRec-CE | 5 | 0.0969 (0.0035) | 0.1711 (0.0064) | 0.1530 (0.0015) | 0.12 |
| SASRec-SS | 5 | 0.0987 (0.0018) | 0.1654 (0.0031) | 0.1432 (0.0026) | 0.06 |
| gSASRec | 5 | 0.0946 (0.0034) | 0.1646 (0.0046) | 0.1478 (0.0025) | 0.08 |
| BERT4Rec | 5 | 0.0881 (0.0063) | 0.1575 (0.0105) | 0.1417 (0.0032) | 0.16 |
| MostPop | 1 | 0.0192 | 0.0347 | 0.0222 | 0.00 |
| Transition | 1 | 0.0739 | 0.1249 | 0.0956 | 0.00 |

Shuffled-order controls (train-and-evaluate shuffle, leave-one-out):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-CE | 10 | 0.0362 (0.0012) | 0.0751 (0.0024) | 0.0387 (0.0011) | 0.31 |
| BERT4Rec | 10 | 0.0327 (0.0014) | 0.0683 (0.0029) | 0.0382 (0.0014) | 0.58 |

`arms.csv` adds NDCG@20, HR@20, truncation shares (0 for every neural arm; none for MostPop and Transition, whose stop reason `converged` is outside the share, D061) and tie rates (at most 0.0001 for the neural arms; 0.30 to 0.67 for MostPop and Transition).

### RQ1: differences between arms

Primary analysis, leave-one-out. Differences are in mean log NDCG@10, shown as relative differences; the 95% Welch interval is descriptive (D051). Notes 1 and 2 apply to every row and conclusion of this section.

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 10, 10 | +34.3% (+32.4% to +36.2%) | 2.7e-18 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 10, 10 | +28.8% (+27.1% to +30.5%) | 2.9e-15 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.0134 | pending |
| F1 | directional | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.0% to -3.1%) | 1.7e-07 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 0.0011 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +14.6% (+5.1% to +24.9%) | 0.0059 | pending |
| F1-NI | non-inferiority | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 2.8e-04 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 10, 10 | +34.3% (+32.4% to +36.2%) | 4.2e-18 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 10, 10 | +28.8% (+27.1% to +30.5%) | 4.1e-15 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.0158 | pending |
| F5 | equivalence | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.984 | not rejected |
| F5 | equivalence | gSASRec | BERT4Rec | 10, 10 | +14.6% (+5.1% to +24.9%) | 0.993 | not rejected |
| F5 | equivalence | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 0.999 | not rejected |

- **F1.** Four of the six directional comparisons are rejected: SASRec-CE and gSASRec are above SASRec-BCE, SASRec-CE is above gSASRec, and SASRec-CE is above BERT4Rec. BERT4Rec against SASRec-BCE and gSASRec against BERT4Rec are pending.
- **H1a: supported on MovieLens-1M.** The non-inferiority test of SASRec-CE against BERT4Rec is rejected (p = 2.8e-04, below 0.05/2).
- **H1b: pending on MovieLens-1M.** SASRec-BCE is below SASRec-CE and below gSASRec by more than the SESOI (both minimum-effect tests rejected). The test against BERT4Rec (p = 0.0158) is pending, so H1b can still end as supported or as not supported on this dataset.
- **Published gaps** (METHODS 7.3; all three have a published + sign on MovieLens-1M, and all three observed differences have that sign):
  - G1, BERT4Rec − SASRec-BCE: pending, confirmed or inconclusive.
  - G2, gSASRec − BERT4Rec: pending, confirmed or inconclusive.
  - G3, SASRec-CE − BERT4Rec: **confirmed**.
  - No equivalence test of F5 is rejected, so no gap is classed "below SESOI" or "equivalent" on this dataset.

### RQ2: training objectives on the SASRec backbone

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -4.1% (-5.2% to -3.1%) | 1.000 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.0% to -3.1%) | 1.000 | not rejected |

- **H2b, sampled softmax: not supported on MovieLens-1M.** Equivalence with full cross-entropy within the SESOI is not shown (TOST not rejected). The comparison is underpowered (note 2), and the non-rejection alone does not establish a difference. Descriptively, the interval of the difference lies below −2%. No directional test is registered for this pair.
- **H2b, gBCE: not supported on MovieLens-1M.** Equivalence is not shown (TOST not rejected, underpowered). The registered directional test of this pair in F1 is rejected: gSASRec is below SASRec-CE.
- **H2b, BCE: supported on MovieLens-1M.** The minimum-effect test of SASRec-CE against SASRec-BCE is rejected: BCE with one negative is worse by more than the SESOI.

Time to target (F6), the group of METHODS 9.4: seeds 1 to 5 of the four arms, trained on one Pod. The target is validation NDCG@10 0.1960, 0.95 times SASRec-CE's mean best validation NDCG@10 of 0.2063.

| Arm | Runs reaching the target | Times to target (s) | Median (s) |
|---|---|---|---|
| SASRec-CE | 5 of 5 | 30.9, 34.0, 34.5, 36.6, 38.2 | 34.5 |
| SASRec-BCE | 0 of 5 | none | censored |
| SASRec-SS | 0 of 5 | none | censored |
| gSASRec | 1 of 5 | 63.5 | censored |

- **H2a: supported on MovieLens-1M**, on this kind of host (Ryzen 9 7950X, 14 CPUs; METHODS 9.4). Each of the three one-sided Mann-Whitney tests of SASRec-CE against SASRec-BCE, SASRec-SS and gSASRec gives p = 1/252 = 0.0040, the smallest attainable value, and is rejected at the strictest Holm step (0.05/12).
- The three rejections rest on censored runs: 14 of the 15 runs of the other arms never reach the target. Their mean best validation NDCG@10 over seeds 1 to 5 is below it (SASRec-BCE 0.1515, SASRec-SS 0.1921, gSASRec 0.1946). One time is measured outside SASRec-CE, 63.5 s for one gSASRec run against 30.9 to 38.2 s for SASRec-CE.
- The other arms' medians are censored, so the ratio of medians is not finite. Note 4 applies.
- No run of the group is flagged, so the repeat without flagged runs gives the same three results; it is provisional (note 5).

### Shuffled-order control (F4)

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F4 | minimum effect | SASRec-CE | shuffled SASRec-CE | 10, 10 | +428.2% (+415.4% to +441.3%) | 2.1e-20 | rejected |
| F4 | minimum effect | BERT4Rec | shuffled BERT4Rec | 10, 10 | +390.4% (+348.7% to +435.9%) | 1.3e-13 | rejected |

- Both controls show the expected drop larger than the SESOI (METHODS 3.3): the control is **supported** for SASRec-CE and for BERT4Rec.
- Test-only shuffle (descriptive, METHODS 3.3): with shuffled test inputs the unshuffled runs' mean test NDCG@10 is 0.0380 for SASRec-BCE, 0.0415 for SASRec-CE, 0.0412 for SASRec-SS, 0.0412 for gSASRec and 0.0394 for BERT4Rec.

### Secondary: temporal split

Five seeds per arm. These results are secondary and support no confirmatory claim (METHODS 7.4); each family is corrected on its own, and every directional and equivalence comparison is underpowered (note 2).

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +15.0% (+10.0% to +20.2%) | 3.4e-04 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +12.3% (+7.5% to +17.3%) | 7.5e-04 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +4.5% (-4.1% to +13.8%) | 0.237 | not rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -2.3% (-7.3% to +2.9%) | 0.327 | not rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +10.1% (+1.1% to +20.0%) | 0.0333 | not rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +7.5% (-1.3% to +17.1%) | 0.0834 | not rejected |
| F1-NI | non-inferiority | SASRec-CE | BERT4Rec | 5, 5 | +10.1% (+1.1% to +20.0%) | 0.0080 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 5, 5 | +15.0% (+10.0% to +20.2%) | 3.6e-04 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 5, 5 | +12.3% (+7.5% to +17.3%) | 9.5e-04 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 5, 5 | +4.5% (-4.1% to +13.8%) | 0.247 | not rejected |
| F5 | equivalence | BERT4Rec | SASRec-BCE | 5, 5 | +4.5% (-4.1% to +13.8%) | 0.753 | not rejected |
| F5 | equivalence | gSASRec | BERT4Rec | 5, 5 | +7.5% (-1.3% to +17.1%) | 0.909 | not rejected |
| F5 | equivalence | SASRec-CE | BERT4Rec | 5, 5 | +10.1% (+1.1% to +20.0%) | 0.964 | not rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | +1.9% (-2.5% to +6.6%) | 0.480 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -2.3% (-7.3% to +2.9%) | 0.567 | not rejected |

Kendall's τ between the arm rankings by mean log NDCG@10 on leave-one-out and on the temporal split, over the five neural arms, is 0.60. The rankings are SASRec-CE, gSASRec, SASRec-SS, BERT4Rec, SASRec-BCE on leave-one-out and SASRec-SS, SASRec-CE, gSASRec, BERT4Rec, SASRec-BCE on the temporal split.

### Robustness

F1 and F2 repeated on log HR@10 (METHODS 7.6), leave-one-out, corrected within their own families; no confirmatory claim:

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 10, 10 | +24.9% (+23.4% to +26.4%) | 2.4e-18 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 10, 10 | +21.4% (+20.1% to +22.7%) | 1.2e-15 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 10, 10 | +9.9% (+2.1% to +18.2%) | 0.0172 | pending |
| F1 | directional | gSASRec | SASRec-CE | 10, 10 | -2.8% (-3.7% to -1.9%) | 6.9e-06 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +13.7% (+5.7% to +22.3%) | 0.0031 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +10.5% (+2.8% to +18.9%) | 0.0126 | pending |
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -3.7% (-4.7% to -2.7%) | 0.999 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -2.8% (-3.7% to -1.9%) | 0.966 | not rejected |

- All eight statuses match the primary analysis. The comparison of SASRec-CE with BERT4Rec was pending with the MovieLens-1M runs alone and is rejected with the Video_Games runs; its p-value, 0.0031, is unchanged. A comparison is rejected if Holm rejects it with every missing p-value set to 1 (D051). With MovieLens-1M alone this p-value was the 4th smallest of the family's 24 registered tests, with threshold 0.05/21 = 0.0024; the six Video_Games p-values of F1 on HR@10 are all smaller, so it is now the 10th smallest, with threshold 0.05/15 = 0.0033.
- The timing repeat without flagged runs is reported under RQ2.

Two-level percentile bootstrap (METHODS 7.6, D056), leave-one-out, NDCG@10. Each of 10,000 resamples draws the 6,040 test users with replacement once for all arms, so arms are paired by user, and draws each arm's ten runs with replacement. It was computed on the volume from the per-user ranks that the leave-one-out final runs kept from their test reads (D051); the table is `results/analysis/bootstrap-m3-final-ml-1m.csv`. The estimates are those of the primary analysis. The intervals are descriptive; the tests of the primary analysis decide.

| Family | Test | A | B | n | A − B, relative (95% bootstrap interval) |
|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 10, 10 | +34.3% (+29.9% to +38.9%) |
| F1 | directional | gSASRec | SASRec-BCE | 10, 10 | +28.8% (+25.0% to +32.9%) |
| F1 | directional | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+2.8% to +19.9%) |
| F1 | directional | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.8% to -2.3%) |
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+12.7% to +30.3%) |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +14.6% (+8.2% to +25.0%) |
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -4.1% (-5.9% to -2.4%) |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.8% to -2.3%) |

- The seed-level resampling understates uncertainty (METHODS 7.6, written for 5 seeds): with five to ten seeds per arm, here ten, drawing seeds with replacement understates the variation between runs (D056).

### Not reported yet

- Peak memory and step time of plain against chunked cross-entropy (METHODS 8).

## Video_Games

### Runs

- Batch `m3-final-video-games`: 65 final runs, all on Pod `egop2ngfe0fk1i` (RTX 4090, Ryzen 9 7950X), the Pod of the tuning batch (D058), 28.50 GPU hours. Every run trained; none failed, none was truncated.
- Leave-one-out has six seeds per neural arm (D062), the temporal split five, and each floor baseline runs once per split.
- Every run read test data once: 65 runs with one final read, one completion record and no anchor read. The 100 tuning runs of `m3-tune-video-games` read test data zero times (METHODS 11.6). The anchor reads of METHODS 11.5 were made on MovieLens-1M only.

### Notes carried by every statement on this dataset

1. **Power.** At the σ of this dataset (below) and six seeds, the registered power at the SESOI is 0.843 for a directional test and 0.995 for an equivalence test, so no leave-one-out comparison on this dataset is labelled underpowered. With the five seeds of the temporal split it is 0.656 and 0.984: the directional comparisons of the temporal split are labelled **underpowered** (METHODS 7.5) and its equivalence comparisons are not. `tests.csv` also gives each of these comparisons a supplementary power at the σ of its own two arms; it enters no status or label (D055).
2. **One-sided tests.** METHODS 7.5 defines no power function for the minimum-effect tests, so they report no power (D051). The time-to-target tests have none either.
3. **Time to target.** D048 found that on small configurations time to target mostly measures launch overhead (the GPU was busy for 0.295 of a step in its configuration 1, SASRec with BCE on Steam). Its Video_Games configuration was not measured: the measurement stopped before it.
4. **Flagged runs.** The repeat of the timing analysis without flagged runs uses the median SM clock of the M3 runs exported so far, so it is provisional until all four datasets have run (D051).

### Seed variance

- Standard deviation of log validation NDCG@10 over seeds 1 to 5: BERT4Rec 0.01075, SASRec-BCE 0.00769, SASRec-SS 0.00389, SASRec-CE 0.00317, gSASRec 0.00187. Pooled σ = 0.00638, for which METHODS 7.5 gives six seeds (D062).

### Quality per arm

Mean over seeds, standard deviation in brackets. Leave-one-out:

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 6 | 0.0440 (0.0005) | 0.0797 (0.0010) | 0.0494 (0.0004) | 0.34 |
| SASRec-CE | 6 | 0.0615 (0.0002) | 0.1077 (0.0004) | 0.0705 (0.0002) | 0.34 |
| SASRec-SS | 6 | 0.0585 (0.0005) | 0.1036 (0.0010) | 0.0673 (0.0003) | 0.54 |
| gSASRec | 6 | 0.0566 (0.0002) | 0.1008 (0.0006) | 0.0651 (0.0002) | 0.78 |
| BERT4Rec | 6 | 0.0501 (0.0006) | 0.0928 (0.0010) | 0.0578 (0.0006) | 3.79 |
| MostPop | 1 | 0.0126 | 0.0249 | 0.0136 | 0.00 |
| Transition | 1 | 0.0405 | 0.0663 | 0.0465 | 0.00 |

Temporal split (secondary):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 5 | 0.0119 (0.0007) | 0.0220 (0.0012) | 0.0179 (0.0003) | 0.91 |
| SASRec-CE | 5 | 0.0177 (0.0003) | 0.0314 (0.0003) | 0.0280 (0.0002) | 0.94 |
| SASRec-SS | 5 | 0.0169 (0.0002) | 0.0300 (0.0003) | 0.0268 (0.0003) | 1.74 |
| gSASRec | 5 | 0.0170 (0.0003) | 0.0303 (0.0004) | 0.0266 (0.0002) | 3.68 |
| BERT4Rec | 5 | 0.0154 (0.0002) | 0.0279 (0.0006) | 0.0225 (0.0002) | 15.09 |
| MostPop | 1 | 0.0032 | 0.0063 | 0.0044 | 0.00 |
| Transition | 1 | 0.0129 | 0.0217 | 0.0200 | 0.00 |

Shuffled-order control (train-and-evaluate shuffle, leave-one-out):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-CE | 6 | 0.0366 (0.0002) | 0.0683 (0.0006) | 0.0460 (0.0002) | 0.35 |

`arms.csv` adds NDCG@20, HR@20, truncation shares (0 for every neural arm; none for MostPop and Transition, whose stop reason `converged` is outside the share, D061) and tie rates (0.0181 for BERT4Rec on the temporal split and at most 0.0007 for the other neural arms; 0.43 to 0.89 for MostPop and Transition).

### RQ1: differences between arms

Primary analysis, leave-one-out. Differences are in mean log NDCG@10, shown as relative differences; the 95% Welch interval is descriptive (D051). Notes 1 and 2 apply to every row and conclusion of this section.

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 6, 6 | +39.7% (+38.2% to +41.3%) | 3.3e-10 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 6, 6 | +28.6% (+27.1% to +30.0%) | 5.5e-10 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 6, 6 | +13.8% (+12.2% to +15.5%) | 2.4e-09 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 6, 6 | -8.0% (-8.4% to -7.5%) | 1.5e-11 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 6, 6 | +22.8% (+21.3% to +24.3%) | 1.6e-08 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 6, 6 | +13.0% (+11.6% to +14.4%) | 1.7e-07 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 6, 6 | +39.7% (+38.2% to +41.3%) | 2.4e-10 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 6, 6 | +28.6% (+27.1% to +30.0%) | 4.7e-10 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 6, 6 | +13.8% (+12.2% to +15.5%) | 6.1e-09 | rejected |

- **F1.** All six directional comparisons are rejected. Together they order the four arms, from highest to lowest, as SASRec-CE, gSASRec, BERT4Rec, SASRec-BCE.
- **H1a** is registered on the two MovieLens datasets only (METHODS 7.4); Video_Games has no non-inferiority test.
- **H1b: supported on Video_Games.** The three minimum-effect tests are rejected: SASRec-CE, gSASRec and BERT4Rec are each above SASRec-BCE by more than the SESOI.
- **Published gaps:** METHODS 7.3 lists no published sign on Video_Games, so this dataset has no gap class and no F5 test.

### RQ2: training objectives on the SASRec backbone

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F2 | equivalence | SASRec-SS | SASRec-CE | 6, 6 | -4.8% (-5.7% to -4.0%) | 1.000 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 6, 6 | -8.0% (-8.4% to -7.5%) | 1.000 | not rejected |

- **H2b, sampled softmax: not supported on Video_Games.** Equivalence with full cross-entropy within the SESOI is not shown (TOST not rejected), and the test is not underpowered (note 1). Descriptively, the interval of the difference lies below −2%. No directional test is registered for this pair.
- **H2b, gBCE: not supported on Video_Games.** Equivalence is not shown (TOST not rejected, not underpowered). The registered directional test of this pair in F1 is rejected: gSASRec is below SASRec-CE.
- **H2b, BCE: supported on Video_Games.** The minimum-effect test of SASRec-CE against SASRec-BCE is rejected: BCE with one negative is worse by more than the SESOI.

Time to target (F6), the group of METHODS 9.4: seeds 1 to 5 of the four arms, trained on one Pod. The target is validation NDCG@10 0.0670, 0.95 times SASRec-CE's mean best validation NDCG@10 of 0.0705.

| Arm | Runs reaching the target | Times to target (s) | Median (s) |
|---|---|---|---|
| SASRec-CE | 5 of 5 | 26.9, 27.7, 29.7, 30.9, 31.8 | 29.7 |
| SASRec-BCE | 0 of 5 | none | censored |
| SASRec-SS | 4 of 5 | 25.5, 28.0, 33.9, 36.7 | 33.9 |
| gSASRec | 0 of 5 | none | censored |

- **H2a: not supported on Video_Games**, on this kind of host (Ryzen 9 7950X, 14 CPUs; METHODS 9.4). The one-sided Mann-Whitney tests of SASRec-CE against SASRec-BCE and against gSASRec each give p = 1/252 = 0.0040, below the strictest Holm step (0.05/12), and are rejected. The test against SASRec-SS gives p = 0.210, above 0.05, so no Holm step can reject it: it is not rejected.
- No run of SASRec-BCE or gSASRec reaches the target: the best validation NDCG@10 of each of their runs with seeds 1 to 5 is below it (means 0.0494 and 0.0650). Their medians are therefore censored and their ratios of medians are not finite.
- Four of the five SASRec-SS runs reach the target; its mean best validation NDCG@10 over seeds 1 to 5 is 0.0674. Its median time is 33.9 s against 29.7 s for SASRec-CE, a ratio of 1.14, outside the 6.84% within which METHODS 9.4 reports no difference measured. The medians and their ratio are descriptive (METHODS 8). Note 3 applies.
- No run of the group is flagged, so the repeat without flagged runs gives the same three results; it is provisional (note 4).

### Shuffled-order control (F4)

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F4 | minimum effect | SASRec-CE | shuffled SASRec-CE | 6, 6 | +67.8% (+66.7% to +68.9%) | 8.1e-16 | rejected |

- The control shows the expected drop larger than the SESOI (METHODS 3.3): it is **supported** for SASRec-CE. METHODS 3.3 registers the BERT4Rec control on the two MovieLens datasets only.
- Test-only shuffle (descriptive, METHODS 3.3): with shuffled test inputs the unshuffled runs' mean test NDCG@10 is 0.0288 for SASRec-BCE, 0.0414 for SASRec-CE, 0.0415 for SASRec-SS, 0.0402 for gSASRec and 0.0415 for BERT4Rec.

### Secondary: temporal split

Five seeds per arm. These results are secondary and support no confirmatory claim (METHODS 7.4); each family is corrected on its own. Every directional comparison is underpowered and every equivalence comparison is not (note 1).

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +49.4% (+39.0% to +60.6%) | 3.9e-05 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +43.2% (+33.3% to +53.9%) | 6.1e-05 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +29.9% (+20.8% to +39.6%) | 3.9e-04 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -4.2% (-6.7% to -1.5%) | 0.0073 | pending |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +15.1% (+12.3% to +17.8%) | 1.8e-06 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +10.3% (+7.6% to +13.0%) | 2.9e-05 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 5, 5 | +49.4% (+39.0% to +60.6%) | 2.4e-05 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 5, 5 | +43.2% (+33.3% to +53.9%) | 4.0e-05 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 5, 5 | +29.9% (+20.8% to +39.6%) | 2.7e-04 | rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -4.5% (-6.8% to -2.2%) | 0.983 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -4.2% (-6.7% to -1.5%) | 0.953 | not rejected |

Kendall's τ between the arm rankings by mean log NDCG@10 on leave-one-out and on the temporal split, over the five neural arms, is 0.80. The rankings are SASRec-CE, SASRec-SS, gSASRec, BERT4Rec, SASRec-BCE on leave-one-out and SASRec-CE, gSASRec, SASRec-SS, BERT4Rec, SASRec-BCE on the temporal split.

### Robustness

F1 and F2 repeated on log HR@10 (METHODS 7.6), leave-one-out, corrected within their own families; no confirmatory claim:

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 6, 6 | +35.1% (+33.4% to +36.9%) | 2.0e-09 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 6, 6 | +26.4% (+24.7% to +28.1%) | 5.8e-10 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 6, 6 | +16.4% (+14.6% to +18.2%) | 9.3e-10 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 6, 6 | -6.5% (-7.1% to -5.8%) | 7.8e-09 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 6, 6 | +16.1% (+14.8% to +17.5%) | 4.8e-08 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 6, 6 | +8.6% (+7.3% to +9.9%) | 3.5e-07 | rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 6, 6 | -3.8% (-4.8% to -2.9%) | 0.999 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 6, 6 | -6.5% (-7.1% to -5.8%) | 1.000 | not rejected |

- All eight statuses match the primary analysis.
- The timing repeat without flagged runs is reported under RQ2.

Two-level percentile bootstrap (METHODS 7.6, D056), leave-one-out, NDCG@10. Each of 10,000 resamples draws the 94,762 test users with replacement once for all arms, so arms are paired by user, and draws each arm's six runs with replacement. It was computed on the volume from the per-user ranks that the leave-one-out final runs kept from their test reads (D051); the table is `results/analysis/bootstrap-m3-final-video-games.csv`. The estimates are those of the primary analysis. The intervals are descriptive; the tests of the primary analysis decide.

| Family | Test | A | B | n | A − B, relative (95% bootstrap interval) |
|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 6, 6 | +39.7% (+37.5% to +42.0%) |
| F1 | directional | gSASRec | SASRec-BCE | 6, 6 | +28.6% (+26.5% to +30.6%) |
| F1 | directional | BERT4Rec | SASRec-BCE | 6, 6 | +13.8% (+11.5% to +16.2%) |
| F1 | directional | gSASRec | SASRec-CE | 6, 6 | -8.0% (-8.8% to -7.1%) |
| F1 | directional | SASRec-CE | BERT4Rec | 6, 6 | +22.8% (+20.7% to +24.9%) |
| F1 | directional | gSASRec | BERT4Rec | 6, 6 | +13.0% (+11.2% to +14.8%) |
| F2 | equivalence | SASRec-SS | SASRec-CE | 6, 6 | -4.8% (-5.7% to -3.9%) |
| F2 | equivalence | gSASRec | SASRec-CE | 6, 6 | -8.0% (-8.8% to -7.1%) |

- The seed-level resampling understates uncertainty (METHODS 7.6, written for 5 seeds): with five to ten seeds per arm, here six, drawing seeds with replacement understates the variation between runs (D056).

### Not reported yet

- Peak memory and step time of plain against chunked cross-entropy (METHODS 8).
