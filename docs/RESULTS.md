# Results

Confirmatory results per dataset, under the tests, families and thresholds of METHODS 7 and 8 and the reporting rules of D051. Every number below comes from the tables in `results/analysis/` (`tests.csv`, `conclusions.csv`, `arms.csv`, `times.csv`) or from the batch files in `results/runs/`.

## Reading the statuses

- Holm's correction uses the registered size of a family over all four datasets (METHODS 7.4). Only MovieLens-1M has final runs.
- A comparison is **rejected** if Holm rejects it whatever the remaining datasets give, **not rejected** if Holm cannot reject it whatever they give, and **pending** otherwise (D051). A hypothesis or gap class is pending while a comparison it depends on can still change it. Nothing pending is a result.
- Steam, MovieLens-20M and Video_Games have no final runs, so every comparison, hypothesis and gap class on them is pending.
- Counts required by METHODS 7.4: H1a is supported on 1 dataset and pending on 1 (MovieLens-20M); H1b is pending on all 4.

## MovieLens-1M

### Runs

- Batch `m3-final-ml-1m`: 99 final runs, all on Pod `mee79yy65ukfve` (RTX 4090, Ryzen 9 7950X), 2.88 GPU hours. Every run trained; none failed, none was truncated.
- Leave-one-out has ten seeds per neural arm (D053), the temporal split five, and each floor baseline runs once per split.
- Every run read test data once: 99 runs with one final read, one completion record and no anchor read. The 100 tuning runs of `m3-tune-ml-1m` read test data zero times (METHODS 11.6).
- Analysis: `seqrec analyze results/runs/m3-tune-ml-1m.csv results/runs/m3-final-ml-1m.csv --out-dir results/analysis`. The tuning file enters only the median SM clock that flags a run.

### Notes carried by every statement on this dataset

1. **Anchor reads.** Before these runs, the anchor reads of METHODS 11.5 read the MovieLens-1M test targets twice for each of SASRec-CE, SASRec-BCE, BERT4Rec and gSASRec (D032). Every RQ1 conclusion on MovieLens-1M carries this note.
2. **Power.** At the σ of this dataset (below) and ten seeds, the registered power at the SESOI is 0.005 for a directional test and 0.000 for an equivalence test; with the five seeds of the temporal split it is 0.003 and 0.000. Every directional and equivalence comparison on this dataset is therefore labelled **underpowered** (METHODS 7.5). A non-rejection is never read as the absence of an effect; rejections count normally.
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

`arms.csv` adds NDCG@20, HR@20, truncation shares (0 for every arm) and tie rates (at most 0.0001 for the neural arms; 0.30 to 0.67 for MostPop and Transition).

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
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +13.7% (+5.7% to +22.3%) | 0.0031 | pending |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +10.5% (+2.8% to +18.9%) | 0.0126 | pending |
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -3.7% (-4.7% to -2.7%) | 0.999 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -2.8% (-3.7% to -1.9%) | 0.966 | not rejected |

- On HR@10 the comparison of SASRec-CE with BERT4Rec is pending where it is rejected on NDCG@10; the other seven statuses match the primary analysis.
- The timing repeat without flagged runs is reported under RQ2.

### Not reported yet

- The two-level bootstrap of METHODS 7.6: every final run's per-user ranks are kept on the machine's volume (D051); the intervals have not been computed.
- Peak memory and step time of plain against chunked cross-entropy (METHODS 8).
