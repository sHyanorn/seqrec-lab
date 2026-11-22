# Results

## Summary

Confirmatory results of milestone M3 on MovieLens-1M, Video_Games and Steam, under METHODS 7 and 8 and the reporting rules of D051. MovieLens-20M has not run: every registered test on it is pending.

SASRec with full cross-entropy (SASRec-CE) is above SASRec-BCE in mean NDCG@10 by 34.3% on MovieLens-1M, 39.7% on Video_Games and 26.5% on Steam (smallest effect of interest, SESOI: 2%), and above BERT4Rec by 19.5%, 22.8% and 2.9%. The published gap G3, SASRec-CE ahead of BERT4Rec, is confirmed on MovieLens-1M. On Steam its published sign is negative, BERT4Rec ahead; the observed difference is +2.9% and the directional test rejects, so G3 is reversed there, and so is G2 (gSASRec against BERT4Rec, +1.5%). gSASRec is below SASRec-CE on all three datasets (-4.1%, -8.0%, -1.3%), and sampled softmax and gBCE are not shown equivalent to full cross-entropy on any. SASRec-CE reaches the validation target sooner than the other three arms on MovieLens-1M and Steam (H2a) and not on Video_Games (SASRec-SS, p = 0.210).

The main limit is power on MovieLens-1M: one BERT4Rec run lies apart from the others, every directional and equivalence comparison there is underpowered, and a non-rejection does not show the absence of an effect (note 1). Other limits are under Limits.

| Hypothesis | MovieLens-1M | Video_Games | Steam | MovieLens-20M |
|---|---|---|---|---|
| H1a: SASRec-CE not below BERT4Rec by more than the SESOI | supported | not registered | not registered | pending |
| H1b: SASRec-CE, gSASRec, BERT4Rec each above SASRec-BCE by more than the SESOI | pending | supported | supported | pending |
| H2a: SASRec-CE reaches the target before SASRec-BCE, SASRec-SS and gSASRec | supported | not supported | supported | pending |
| H2b: sampled softmax equivalent to full cross-entropy within the SESOI | not supported | not supported | not supported | pending |
| H2b: gBCE equivalent to full cross-entropy within the SESOI | not supported | not supported | not supported | pending |
| H2b: BCE with one negative worse than full cross-entropy by more than the SESOI | supported | supported | supported | pending |
| F4: shuffled order drops SASRec-CE by more than the SESOI | supported | supported | supported | pending |
| F4: shuffled order drops BERT4Rec by more than the SESOI | supported | not registered | not registered | pending |
| G1: BERT4Rec above SASRec-BCE | pending | not registered | confirmed | pending |
| G2: gSASRec above BERT4Rec | confirmed | not registered | reversed | not registered |
| G3: SASRec-CE above BERT4Rec | confirmed | not registered | reversed | pending |

Not registered: METHODS 7.4 registers H1a on the two MovieLens datasets only, METHODS 3.3 the BERT4Rec shuffle control on the two MovieLens datasets only, and METHODS 7.3 lists a published sign for a gap only where its table gives one. H1a is supported on 1 dataset and pending on 1 (MovieLens-20M); H1b is supported on 2 (Video_Games, Steam) and pending on 2 (METHODS 7.4).

## How to read

**Statuses (D051).** Holm's step-down correction for multiple comparisons uses the registered size of each family over all four datasets (METHODS 7.4), so the missing p-values of MovieLens-20M leave some comparisons undecided. A comparison is **rejected** if Holm rejects it whatever the missing p-values are, **not rejected** if Holm cannot reject it whatever they are, and **pending** otherwise. A hypothesis or gap class is pending while a comparison it depends on is. The Holm steps section gives the borderline cases.

**Tables.** Leave-one-out is the primary analysis. Differences are in mean log NDCG@10, shown as relative differences with a descriptive 95% Welch interval. The temporal split and the HR@10 repeats of F1 and F2 (METHODS 7.6) support no confirmatory claim (METHODS 7.4). Quality tables give means over seeds with standard deviations in brackets. `arms.csv` adds NDCG@20, HR@20, truncation shares (0 for every neural arm; none for MostPop and Transition, whose stop reason `converged` is outside the share, D061) and tie rates.

**Time to target (F6).** The group of METHODS 9.4 is seeds 1 to 5 of the four SASRec-backbone arms on one Pod. The target is 0.95 times SASRec-CE's mean best validation NDCG@10; a run that never reaches it is censored.

**Bootstrap.** The two-level percentile bootstrap (METHODS 7.6, D056) runs on leave-one-out NDCG@10 with 10,000 resamples, from the per-user ranks that the final runs kept from their test reads (D051). Each resample draws the test users once for all arms, pairing them by user, and each arm's runs with replacement. The estimates are those of the primary analysis; the intervals are descriptive.

**Sources.** Every number comes from `results/analysis/` (`tests.csv`, `conclusions.csv`, `arms.csv`, `times.csv`, `bootstrap-<batch>.csv`) or from the batch files in `results/runs/`. `seqrec analyze results/runs/m3-tune-ml-1m.csv results/runs/m3-final-ml-1m.csv results/runs/m3-tune-video-games.csv results/runs/m3-final-video-games.csv results/runs/m3-tune-steam.csv results/runs/m3-final-steam.csv --out-dir results/analysis` writes the first four; the tuning files enter only the median SM clock that flags a run. `seqrec bootstrap` writes the bootstrap tables.

## Notes

1. **Power.** The registered power at the SESOI uses the dataset's pooled σ of log validation NDCG@10 (METHODS 7.5). A comparison with power below 0.8 is labelled **underpowered**. A non-rejection is never read as the absence of an effect. `tests.csv` also gives a supplementary power at the σ of each comparison's own two arms; it enters no status or label (D055).

   | Dataset | Pooled σ | Leave-one-out seeds | Leave-one-out power, directional / equivalence | Temporal split power (5 seeds), directional / equivalence | Labelled underpowered |
   |---|---|---|---|---|---|
   | MovieLens-1M | 0.07703 | 10 | 0.005 / 0.000 | 0.003 / 0.000 | every directional and equivalence comparison |
   | Video_Games | 0.00638 | 6 | 0.843 / 0.995 | 0.656 / 0.984 | temporal-split directional comparisons |
   | Steam | 0.00265 | 5 | 1.000 / 1.000 | 1.000 / 1.000 | none |

2. **Anchor reads.** Before the final runs, the anchor reads of METHODS 11.5 read the MovieLens-1M test targets twice for each of SASRec-CE, SASRec-BCE, BERT4Rec and gSASRec (D032). Every RQ1 conclusion on MovieLens-1M carries this note. No other dataset had anchor reads.
3. **Tests without a power function.** METHODS 7.5 defines none for the non-inferiority and minimum-effect tests, and the time-to-target tests have none (D051).
4. **Time to target.** D048 found that on small configurations time to target mostly measures launch overhead: the GPU was busy for 0.295 of a step in its configuration 1, SASRec with BCE on Steam. The configurations of this study were not measured: MovieLens-1M was not among them, the measurement stopped before the Video_Games configuration, and the Steam configurations used the pilot's hyperparameters, not those of D066, and stopped during the second. Medians and ratios of medians are descriptive (METHODS 8).
5. **Flagged runs.** A run is flagged when its median SM clock is more than 5% below the median over all M3 runs (METHODS 8). That median covers the M3 runs exported so far, so the flags are provisional until all four datasets have run (D051). On MovieLens-1M and Video_Games no run of the time-to-target group is flagged. On Steam the five SASRec-BCE runs of the group are flagged, each with a median SM clock of 2550 MHz against the limit of 2650.5 MHz (0.95 × 2790 MHz, the median over the 523 M3 runs exported, D068), and the repeat without flagged runs is not run (D068).

## MovieLens-1M

### Runs

- Batch `m3-final-ml-1m`: 99 final runs on Pod `mee79yy65ukfve` (RTX 4090, Ryzen 9 7950X), 2.88 GPU hours; none failed or was truncated. Neural arms have ten seeds on leave-one-out (D053) and five on the temporal split. Each run has one final read, one completion record and no anchor read; the 100 tuning runs read none (METHODS 11.6).

### Seed variance

- Standard deviation of log validation NDCG@10 over seeds 1 to 5: BERT4Rec 0.17128, SASRec-SS 0.01298, gSASRec 0.00931, SASRec-CE 0.00814, SASRec-BCE 0.00359. Pooled σ = 0.07703 (D053).
- BERT4Rec accounts for 98.9% of the pooled variance. Its validation NDCG@10 is 0.1692, 0.1174, 0.1724, 0.1729 and 0.1740, so seed 2 lies apart. On test data that run has 0.1138 and the other nine BERT4Rec runs lie between 0.1632 and 0.1695. It is a finished run under METHODS 5.2 and enters every analysis (D053).

### Quality per arm

Leave-one-out:

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

Tie rates: at most 0.0001 for the neural arms, 0.30 to 0.67 for MostPop and Transition.

### RQ1: differences between arms

Primary analysis, leave-one-out. Notes 1 and 2 apply to every row and conclusion.

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 10, 10 | +34.3% (+32.4% to +36.2%) | 2.7e-18 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 10, 10 | +28.8% (+27.1% to +30.5%) | 2.9e-15 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.0134 | pending |
| F1 | directional | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.0% to -3.1%) | 1.7e-07 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 0.0011 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +14.6% (+5.1% to +24.9%) | 0.0059 | rejected |
| F1-NI | non-inferiority | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 2.8e-04 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 10, 10 | +34.3% (+32.4% to +36.2%) | 4.2e-18 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 10, 10 | +28.8% (+27.1% to +30.5%) | 4.1e-15 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.0158 | pending |
| F5 | equivalence | BERT4Rec | SASRec-BCE | 10, 10 | +12.4% (+3.1% to +22.6%) | 0.984 | not rejected |
| F5 | equivalence | gSASRec | BERT4Rec | 10, 10 | +14.6% (+5.1% to +24.9%) | 0.993 | not rejected |
| F5 | equivalence | SASRec-CE | BERT4Rec | 10, 10 | +19.5% (+9.6% to +30.3%) | 0.999 | not rejected |

- **F1.** Five of the six directional comparisons are rejected. BERT4Rec against SASRec-BCE (p = 0.0134) is pending.
- **H1a: supported.** The non-inferiority test of SASRec-CE against BERT4Rec is rejected (p = 2.8e-04, below 0.05/2).
- **H1b: pending.** The minimum-effect tests of SASRec-CE and gSASRec against SASRec-BCE are rejected. BERT4Rec's (p = 0.0158) is pending, so H1b can still end as supported or as not supported.
- **Published gaps** (METHODS 7.3). All three have a published + sign here, and all three observed differences are positive. G2 and G3 are **confirmed**; G1 is pending, confirmed or inconclusive. No F5 test is rejected, so no gap is classed "below SESOI" or "equivalent".

### RQ2: training objectives on the SASRec backbone

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -4.1% (-5.2% to -3.1%) | 1.000 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -4.1% (-5.0% to -3.1%) | 1.000 | not rejected |

- **H2b: sampled softmax and gBCE not supported, BCE supported.** Neither TOST is rejected, so equivalence with full cross-entropy is not shown; both are underpowered, so this does not establish a difference. The interval of SASRec-SS against SASRec-CE lies below −2%, and gSASRec is below SASRec-CE in F1. The minimum-effect test of SASRec-CE against SASRec-BCE is rejected.

Time to target. The target is validation NDCG@10 0.1960, 0.95 times SASRec-CE's mean best validation NDCG@10 of 0.2063.

| Arm | Runs reaching the target | Times to target (s) | Median (s) |
|---|---|---|---|
| SASRec-CE | 5 of 5 | 30.9, 34.0, 34.5, 36.6, 38.2 | 34.5 |
| SASRec-BCE | 0 of 5 | none | censored |
| SASRec-SS | 0 of 5 | none | censored |
| gSASRec | 1 of 5 | 63.5 | censored |

- **H2a: supported.** The three one-sided Mann-Whitney tests of SASRec-CE against SASRec-BCE, SASRec-SS and gSASRec each give p = 1/252 = 0.0040, the smallest attainable value, and are rejected at the strictest Holm step (0.05/12). They rest on censored runs: 14 of the 15 runs of the other arms never reach the target, and their mean best validation NDCG@10 over seeds 1 to 5 is below it (SASRec-BCE 0.1515, SASRec-SS 0.1921, gSASRec 0.1946). One run reaches it, a gSASRec run at 63.5 s against 30.9 to 38.2 s for SASRec-CE. Notes 4 and 5 apply.

### Shuffled-order control (F4)

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F4 | minimum effect | SASRec-CE | shuffled SASRec-CE | 10, 10 | +428.2% (+415.4% to +441.3%) | 2.1e-20 | rejected |
| F4 | minimum effect | BERT4Rec | shuffled BERT4Rec | 10, 10 | +390.4% (+348.7% to +435.9%) | 1.3e-13 | rejected |

- Both controls drop by more than the SESOI (METHODS 3.3): **supported** for SASRec-CE and for BERT4Rec. With shuffled test inputs (descriptive, METHODS 3.3) the unshuffled runs' mean test NDCG@10 is 0.0380 for SASRec-BCE, 0.0415 for SASRec-CE, 0.0412 for SASRec-SS, 0.0412 for gSASRec and 0.0394 for BERT4Rec.

### Secondary: temporal split

Every directional and equivalence comparison is underpowered (note 1).

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

Kendall's τ between the arm rankings on leave-one-out and on the temporal split, over the five neural arms, is 0.60.

### Robustness

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 10, 10 | +24.9% (+23.4% to +26.4%) | 2.4e-18 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 10, 10 | +21.4% (+20.1% to +22.7%) | 1.2e-15 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 10, 10 | +9.9% (+2.1% to +18.2%) | 0.0172 | not rejected |
| F1 | directional | gSASRec | SASRec-CE | 10, 10 | -2.8% (-3.7% to -1.9%) | 6.9e-06 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 10, 10 | +13.7% (+5.7% to +22.3%) | 0.0031 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 10, 10 | +10.5% (+2.8% to +18.9%) | 0.0126 | not rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 10, 10 | -3.7% (-4.7% to -2.7%) | 0.999 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 10, 10 | -2.8% (-3.7% to -1.9%) | 0.966 | not rejected |

- Six of the eight statuses match the primary analysis. BERT4Rec against SASRec-BCE (p = 0.0172) is not rejected here and pending there; gSASRec against BERT4Rec (p = 0.0126) is not rejected here and rejected there (Holm steps).
- Bootstrap: 6,040 test users and ten runs per arm, `results/analysis/bootstrap-m3-final-ml-1m.csv`.

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

## Video_Games

### Runs

- Batch `m3-final-video-games`: 65 final runs on Pod `egop2ngfe0fk1i` (RTX 4090, Ryzen 9 7950X), the Pod of the tuning batch (D058), 28.50 GPU hours; none failed or was truncated. Neural arms have six seeds on leave-one-out (D062) and five on the temporal split. Each run has one final read, one completion record and no anchor read; the 100 tuning runs read none (METHODS 11.6).

### Seed variance

- Standard deviation of log validation NDCG@10 over seeds 1 to 5: BERT4Rec 0.01075, SASRec-BCE 0.00769, SASRec-SS 0.00389, SASRec-CE 0.00317, gSASRec 0.00187. Pooled σ = 0.00638, for which METHODS 7.5 gives six seeds (D062).

### Quality per arm

Leave-one-out:

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

Tie rates: 0.0181 for BERT4Rec on the temporal split and at most 0.0007 for the other neural arms, 0.43 to 0.89 for MostPop and Transition.

### RQ1: differences between arms

Primary analysis, leave-one-out. Note 1 applies to every row and conclusion.

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

- **F1.** All six directional comparisons are rejected. They order the four arms, from highest to lowest, as SASRec-CE, gSASRec, BERT4Rec, SASRec-BCE.
- **H1b: supported.** The three minimum-effect tests are rejected.

### RQ2: training objectives on the SASRec backbone

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F2 | equivalence | SASRec-SS | SASRec-CE | 6, 6 | -4.8% (-5.7% to -4.0%) | 1.000 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 6, 6 | -8.0% (-8.4% to -7.5%) | 1.000 | not rejected |

- **H2b: sampled softmax and gBCE not supported, BCE supported.** Neither TOST is rejected, so equivalence with full cross-entropy is not shown, and neither test is underpowered. The interval of SASRec-SS against SASRec-CE lies below −2%, and gSASRec is below SASRec-CE in F1. The minimum-effect test of SASRec-CE against SASRec-BCE is rejected.

Time to target. The target is validation NDCG@10 0.0670, 0.95 times SASRec-CE's mean best validation NDCG@10 of 0.0705.

| Arm | Runs reaching the target | Times to target (s) | Median (s) |
|---|---|---|---|
| SASRec-CE | 5 of 5 | 26.9, 27.7, 29.7, 30.9, 31.8 | 29.7 |
| SASRec-BCE | 0 of 5 | none | censored |
| SASRec-SS | 4 of 5 | 25.5, 28.0, 33.9, 36.7 | 33.9 |
| gSASRec | 0 of 5 | none | censored |

- **H2a: not supported.** The tests of SASRec-CE against SASRec-BCE and against gSASRec each give p = 1/252 = 0.0040, below the strictest Holm step (0.05/12), and are rejected. The test against SASRec-SS gives p = 0.210, above 0.05, so no Holm step can reject it.
- No run of SASRec-BCE or gSASRec reaches the target (mean best validation NDCG@10 over seeds 1 to 5: 0.0494 and 0.0650), so their medians are censored. Four of five SASRec-SS runs reach it (mean 0.0674). Its median time is 33.9 s against 29.7 s for SASRec-CE, a ratio of 1.14, outside the 6.84% within which METHODS 9.4 reports no difference measured. Notes 4 and 5 apply.

### Shuffled-order control (F4)

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F4 | minimum effect | SASRec-CE | shuffled SASRec-CE | 6, 6 | +67.8% (+66.7% to +68.9%) | 8.1e-16 | rejected |

- The control drops by more than the SESOI (METHODS 3.3): **supported** for SASRec-CE. With shuffled test inputs (descriptive, METHODS 3.3) the unshuffled runs' mean test NDCG@10 is 0.0288 for SASRec-BCE, 0.0414 for SASRec-CE, 0.0415 for SASRec-SS, 0.0402 for gSASRec and 0.0415 for BERT4Rec.

### Secondary: temporal split

Every directional comparison is underpowered and no equivalence comparison is (note 1).

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +49.4% (+39.0% to +60.6%) | 3.9e-05 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +43.2% (+33.3% to +53.9%) | 6.1e-05 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +29.9% (+20.8% to +39.6%) | 3.9e-04 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -4.2% (-6.7% to -1.5%) | 0.0073 | not rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +15.1% (+12.3% to +17.8%) | 1.8e-06 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +10.3% (+7.6% to +13.0%) | 2.9e-05 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 5, 5 | +49.4% (+39.0% to +60.6%) | 2.4e-05 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 5, 5 | +43.2% (+33.3% to +53.9%) | 4.0e-05 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 5, 5 | +29.9% (+20.8% to +39.6%) | 2.7e-04 | rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -4.5% (-6.8% to -2.2%) | 0.983 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -4.2% (-6.7% to -1.5%) | 0.953 | not rejected |

Kendall's τ between the arm rankings on leave-one-out and on the temporal split, over the five neural arms, is 0.80. The F1 comparison of gSASRec with SASRec-CE (p = 0.0073) is not rejected (Holm steps).

### Robustness

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
- Bootstrap: 94,762 test users and six runs per arm, `results/analysis/bootstrap-m3-final-video-games.csv`.

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

## Steam

### Runs

- Batch `m3-final-steam`: 59 final runs on the Pods of D065, 50.56 GPU hours. 57 ran on Pod `nivy287l1lt3dn` (RTX 4090, Ryzen 9 7950X), the Pod of the tuning batch. `gsasrec-steam-gts-s4` ran on Pod `m27rj5roizxxv7`, and `bert4rec-steam-loo-s5` resumed there from its checkpoint at epoch 360 and lists both Pods (D067); the two Pods have the same host signature. None failed or was truncated. Neural arms have five seeds on leave-one-out (D067) and on the temporal split. Each run has one final read, one completion record and no anchor read; the 100 tuning runs read none (METHODS 11.6).

### Seed variance

- Standard deviation of log validation NDCG@10 over seeds 1 to 5: SASRec-BCE 0.00315, SASRec-SS 0.00310, SASRec-CE 0.00277, BERT4Rec 0.00244, gSASRec 0.00136. Pooled σ = 0.00265, for which METHODS 7.5 gives five seeds (D067).

### Quality per arm

Leave-one-out:

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 5 | 0.0585 (0.0003) | 0.1132 (0.0004) | 0.0597 (0.0002) | 2.87 |
| SASRec-CE | 5 | 0.0739 (0.0004) | 0.1358 (0.0008) | 0.0775 (0.0002) | 1.34 |
| SASRec-SS | 5 | 0.0726 (0.0002) | 0.1335 (0.0003) | 0.0760 (0.0002) | 1.26 |
| gSASRec | 5 | 0.0730 (0.0003) | 0.1345 (0.0005) | 0.0766 (0.0001) | 1.41 |
| BERT4Rec | 5 | 0.0718 (0.0004) | 0.1334 (0.0006) | 0.0758 (0.0002) | 14.65 |
| MostPop | 1 | 0.0265 | 0.0526 | 0.0285 | 0.00 |
| Transition | 1 | 0.0422 | 0.0765 | 0.0482 | 0.00 |

Temporal split (secondary):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-BCE | 5 | 0.0453 (0.0003) | 0.0889 (0.0003) | 0.0475 (0.0005) | 4.16 |
| SASRec-CE | 5 | 0.0521 (0.0012) | 0.0994 (0.0023) | 0.0535 (0.0005) | 2.50 |
| SASRec-SS | 5 | 0.0514 (0.0013) | 0.0982 (0.0027) | 0.0548 (0.0012) | 2.31 |
| gSASRec | 5 | 0.0520 (0.0005) | 0.0992 (0.0012) | 0.0537 (0.0003) | 2.31 |
| BERT4Rec | 5 | 0.0530 (0.0008) | 0.1008 (0.0015) | 0.0549 (0.0004) | 16.33 |
| MostPop | 1 | 0.0309 | 0.0552 | 0.0312 | 0.00 |
| Transition | 1 | 0.0406 | 0.0751 | 0.0425 | 0.00 |

Shuffled-order control (train-and-evaluate shuffle, leave-one-out):

| Arm | Runs | Test NDCG@10 | Test HR@10 | Validation NDCG@10 | GPU hours |
|---|---|---|---|---|---|
| SASRec-CE | 5 | 0.0421 (0.0003) | 0.0824 (0.0008) | 0.0492 (0.0002) | 1.43 |

Tie rates: 0.0043 for BERT4Rec on the temporal split and at most 0.0003 for the other neural arms, 0.28 to 0.45 for MostPop and Transition.

### RQ1: differences between arms

Primary analysis, leave-one-out. Note 1 applies to every row and conclusion.

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +26.5% (+25.5% to +27.4%) | 1.3e-12 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +24.8% (+24.0% to +25.6%) | 4.7e-12 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +22.9% (+21.9% to +23.9%) | 1.4e-11 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -1.3% (-2.0% to -0.7%) | 0.0019 | rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +2.9% (+2.1% to +3.7%) | 4.4e-05 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +1.5% (+0.8% to +2.3%) | 0.0021 | rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 5, 5 | +26.5% (+25.5% to +27.4%) | 1.3e-12 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 5, 5 | +24.8% (+24.0% to +25.6%) | 4.7e-12 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 5, 5 | +22.9% (+21.9% to +23.9%) | 1.5e-11 | rejected |
| F5 | equivalence | BERT4Rec | SASRec-BCE | 5, 5 | +22.9% (+21.9% to +23.9%) | 1.000 | not rejected |
| F5 | equivalence | gSASRec | BERT4Rec | 5, 5 | +1.5% (+0.8% to +2.3%) | 0.100 | not rejected |
| F5 | equivalence | SASRec-CE | BERT4Rec | 5, 5 | +2.9% (+2.1% to +3.7%) | 0.981 | not rejected |

- **F1.** All six directional comparisons are rejected. They order the four arms, from highest to lowest, as SASRec-CE, gSASRec, BERT4Rec, SASRec-BCE.
- **H1b: supported.** The three minimum-effect tests are rejected.
- **Published gaps** (METHODS 7.3). G1 has a published + sign here, G2 and G3 a published − sign. G1 is **confirmed**. G2 and G3 are **reversed**: both observed differences are positive, against the published − sign, and both directional tests are rejected (p = 0.0021 and 4.4e-05). No F5 test is rejected, so no gap is classed "below SESOI" or "equivalent".

### RQ2: training objectives on the SASRec backbone

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -1.8% (-2.5% to -1.2%) | 0.321 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -1.3% (-2.0% to -0.7%) | 0.0258 | not rejected |

- **H2b: sampled softmax and gBCE not supported, BCE supported.** Neither TOST is rejected, so equivalence with full cross-entropy is not shown, and neither test is underpowered. SASRec-SS is −1.8% against SASRec-CE and its interval, −2.5% to −1.2%, extends below −2%. The gBCE p-value, 0.0258, is below 0.05, and Holm does not reject it whatever the two F2 tests of MovieLens-20M give (Holm steps). gSASRec is below SASRec-CE in F1. The minimum-effect test of SASRec-CE against SASRec-BCE is rejected.

Time to target. The target is validation NDCG@10 0.0736, 0.95 times SASRec-CE's mean best validation NDCG@10 of 0.0775.

| Arm | Runs reaching the target | Times to target (s) | Median (s) |
|---|---|---|---|
| SASRec-CE | 5 of 5 | 54.6, 56.6, 56.8, 58.2, 66.2 | 56.8 |
| SASRec-BCE | 0 of 5 | none | censored |
| SASRec-SS | 5 of 5 | 124.0, 125.2, 127.3, 129.0, 130.7 | 127.3 |
| gSASRec | 5 of 5 | 164.8, 167.6, 168.1, 173.4, 174.5 | 168.1 |

- **H2a: supported.** The three one-sided Mann-Whitney tests of SASRec-CE against SASRec-BCE, SASRec-SS and gSASRec each give p = 1/252 = 0.0040, the smallest attainable value, and are rejected at the strictest Holm step (0.05/12). The test against SASRec-BCE rests on censored runs (mean best validation NDCG@10 over seeds 1 to 5: 0.0597, below the target). The other two rest on measured times: all five SASRec-CE times, 54.6 s to 66.2 s, are below all five of SASRec-SS (124.0 s to 130.7 s) and of gSASRec (164.8 s to 174.5 s).
- The median of SASRec-BCE is censored. SASRec-SS has a median of 127.3 s and gSASRec of 168.1 s against 56.8 s for SASRec-CE, ratios of 2.24 and 2.96, both outside the 6.84% within which METHODS 9.4 reports no difference measured. Notes 4 and 5 apply.

### Shuffled-order control (F4)

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F4 | minimum effect | SASRec-CE | shuffled SASRec-CE | 5, 5 | +75.4% (+73.7% to +77.2%) | 3.7e-13 | rejected |

- The control drops by more than the SESOI (METHODS 3.3): **supported** for SASRec-CE. With shuffled test inputs (descriptive, METHODS 3.3) the unshuffled runs' mean test NDCG@10 is 0.0540 for SASRec-BCE, 0.0634 for SASRec-CE, 0.0619 for SASRec-SS, 0.0631 for gSASRec and 0.0638 for BERT4Rec.

### Secondary: temporal split

No comparison is underpowered (note 1).

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +15.0% (+11.8% to +18.3%) | 8.0e-05 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +14.9% (+13.5% to +16.2%) | 1.0e-08 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +17.1% (+14.8% to +19.3%) | 2.6e-06 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -0.2% (-2.9% to +2.7%) | 0.891 | not rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | -1.7% (-4.6% to +1.2%) | 0.206 | not rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | -1.9% (-3.8% to +0.1%) | 0.0547 | not rejected |
| F1-ME | minimum effect | SASRec-CE | SASRec-BCE | 5, 5 | +15.0% (+11.8% to +18.3%) | 7.9e-05 | rejected |
| F1-ME | minimum effect | gSASRec | SASRec-BCE | 5, 5 | +14.9% (+13.5% to +16.2%) | 1.6e-08 | rejected |
| F1-ME | minimum effect | BERT4Rec | SASRec-BCE | 5, 5 | +17.1% (+14.8% to +19.3%) | 2.7e-06 | rejected |
| F5 | equivalence | BERT4Rec | SASRec-BCE | 5, 5 | +17.1% (+14.8% to +19.3%) | 1.000 | not rejected |
| F5 | equivalence | gSASRec | BERT4Rec | 5, 5 | -1.9% (-3.8% to +0.1%) | 0.462 | not rejected |
| F5 | equivalence | SASRec-CE | BERT4Rec | 5, 5 | -1.7% (-4.6% to +1.2%) | 0.427 | not rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -1.4% (-4.7% to +2.1%) | 0.348 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -0.2% (-2.9% to +2.7%) | 0.0799 | not rejected |

Kendall's τ between the arm rankings on leave-one-out and on the temporal split, over the five neural arms, is 0.40. BERT4Rec is fourth on leave-one-out and first on the temporal split; there SASRec-CE and gSASRec differ from it by −1.7% and −1.9%, and both comparisons are not rejected.

### Robustness

| Family | Test | A | B | n | A − B, relative (95% interval) | p | Status |
|---|---|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +19.9% (+19.0% to +20.8%) | 6.9e-10 | rejected |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +18.7% (+18.1% to +19.4%) | 1.2e-12 | rejected |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +17.8% (+17.1% to +18.5%) | 7.0e-12 | rejected |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -1.0% (-1.7% to -0.2%) | 0.0181 | not rejected |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +1.8% (+1.0% to +2.6%) | 9.1e-04 | rejected |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +0.8% (+0.2% to +1.4%) | 0.0139 | not rejected |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -1.6% (-2.4% to -0.9%) | 0.157 | not rejected |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -1.0% (-1.7% to -0.2%) | 0.0083 | pending |

- Five of the eight statuses match the primary analysis. In F1, gSASRec against SASRec-CE (p = 0.0181) and against BERT4Rec (p = 0.0139) are not rejected here and rejected there. In F2, gSASRec against SASRec-CE (p = 0.00827, 0.0083 in the table) is pending here and not rejected there (Holm steps).
- Bootstrap: 281,210 test users and five runs per arm, `results/analysis/bootstrap-m3-final-steam.csv`.

| Family | Test | A | B | n | A − B, relative (95% bootstrap interval) |
|---|---|---|---|---|---|
| F1 | directional | SASRec-CE | SASRec-BCE | 5, 5 | +26.5% (+25.3% to +27.6%) |
| F1 | directional | gSASRec | SASRec-BCE | 5, 5 | +24.8% (+23.7% to +25.8%) |
| F1 | directional | BERT4Rec | SASRec-BCE | 5, 5 | +22.9% (+21.7% to +24.0%) |
| F1 | directional | gSASRec | SASRec-CE | 5, 5 | -1.3% (-1.9% to -0.7%) |
| F1 | directional | SASRec-CE | BERT4Rec | 5, 5 | +2.9% (+2.1% to +3.7%) |
| F1 | directional | gSASRec | BERT4Rec | 5, 5 | +1.5% (+0.8% to +2.3%) |
| F2 | equivalence | SASRec-SS | SASRec-CE | 5, 5 | -1.8% (-2.4% to -1.3%) |
| F2 | equivalence | gSASRec | SASRec-CE | 5, 5 | -1.3% (-1.9% to -0.7%) |

## Holm steps

Step k of a family of registered size m (24 for F1, 12 for F1-ME, 8 for F2) compares the k-th smallest p-value with 0.05/(m − k + 1), and Holm stops at the first failure. The 6 F1, 3 F1-ME and 2 F2 tests of MovieLens-20M have no p-value (D051). The table sets them to 0 and to 1 for each borderline comparison. A comparison is rejected if it passes with them at 1, not rejected if it fails with them at 0, and pending otherwise.

| Analysis | Dataset | Comparison | p | Missing at 0: step, threshold, Holm | Missing at 1: step, threshold, Holm | Status |
|---|---|---|---|---|---|---|
| Primary F1 | MovieLens-1M | gSASRec − BERT4Rec | 0.005901 | 23, 0.025, passes | 17, 0.00625, passes | rejected |
| Primary F1 | MovieLens-1M | BERT4Rec − SASRec-BCE | 0.01345 | 24, 0.05, passes | 18, 0.00714, fails | pending |
| Primary F1-ME | MovieLens-1M | BERT4Rec − SASRec-BCE | 0.01583 | 12, 0.05, passes | 9, 0.0125, fails | pending |
| Primary F2 | Steam | gSASRec − SASRec-CE | 0.02584 | 3, 0.00833, fails | 1, 0.00625, fails | not rejected |
| Temporal split F1 | Video_Games | gSASRec − SASRec-CE | 0.007335 | 17, 0.00625, fails | 11, 0.00357, fails | not rejected |
| HR@10 F1 | MovieLens-1M | SASRec-CE − BERT4Rec | 0.003135 | 20, 0.01, passes | 14, 0.00455, passes | rejected |
| HR@10 F1 | MovieLens-1M | gSASRec − BERT4Rec | 0.01256 | 21, 0.0125, fails | 15, 0.005, fails | not rejected |
| HR@10 F1 | MovieLens-1M | BERT4Rec − SASRec-BCE | 0.01715 | 23, 0.025, fails | 17, 0.00625, fails | not rejected |
| HR@10 F1 | Steam | gSASRec − SASRec-CE | 0.0181 | 24, 0.05, fails | 18, 0.00714, fails | not rejected |
| HR@10 F1 | Steam | gSASRec − BERT4Rec | 0.01393 | 22, 0.0167, fails | 16, 0.00556, fails | not rejected |
| HR@10 F2 | Steam | gSASRec − SASRec-CE | 0.008274 | 3, 0.00833, passes | 1, 0.00625, fails | pending |

## Limits

- MovieLens-20M has no final runs, so every registered comparison, hypothesis and gap class on it is pending, and a pending status can still change (D051).
- MovieLens-1M: every directional and equivalence comparison is underpowered, and its test targets were read for anchor checks before the final runs (notes 1 and 2). On Video_Games only the temporal-split directional comparisons are underpowered.
- Resampling seeds with replacement, with five to ten seeds per arm (ten, six and five here), understates the variation between runs (METHODS 7.6, D056).
- Time to target mostly measures launch overhead on small configurations (note 4). Conclusions about time hold for this kind of host, a Ryzen 9 7950X with 14 CPUs (METHODS 9.4). The SM-clock flags are provisional (note 5).
- The temporal split and the HR@10 repeats support no confirmatory claim and disagree with the primary analysis in places: two of eight HR@10 statuses on MovieLens-1M and three on Steam differ, and on Steam BERT4Rec is first on the temporal split.
- Peak memory and step time of plain against chunked cross-entropy (METHODS 8) are not reported yet.
