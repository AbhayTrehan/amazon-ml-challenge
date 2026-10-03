# E001 Experimental contract

E001 uses country-equality blocking; hashed character TF-IDF retrieval over name (character 3–5 grams) and address (character trigrams); a compact 49-feature pair representation; and one LightGBM binary classifier. Retrieval is run once at top 20 and K10 is a strict rank-filtered subset of the same cache. E002 and E003 reuse the identical E001 pair scores without retraining. No external data, neural model, GPU, public leaderboard, or supplied-data mutation was used.

# Sampling

Seed: `20260925`. All matched S2/S3 records inherit their unique owner S1 fold. Unmatched auxiliaries receive a deterministic hash fold. The train index contains folds 1–3 only, calibration fold 4 only, and final evaluation fold 0 only, preventing S1/component leakage.

| Role | Folds | S1 | India | US | Singleton | Positive | True edges |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | 1–3 | 30,000 | 11,999 | 18,001 | 1,671 | 28,329 | 103,862 |
| Calibration | 4 | 10,000 | 4,017 | 5,983 | 558 | 9,442 | 34,611 |
| Final evaluation | 0 | 10,000 | 3,992 | 6,008 | 563 | 9,437 | 34,659 |

Train fold counts are 10,007/9,984/10,009 for folds 1/2/3. Cardinality bucket counts `(0,1,2,3,4,5+)` are train `(1,671, 1,623, 5,100, 7,222, 6,578, 7,806)`, calibration `(558, 540, 1,702, 2,412, 2,187, 2,601)`, and evaluation `(563, 536, 1,700, 2,395, 2,203, 2,603)`. Exact selected IDs and complete strata are in `selected_ids/`.

# Candidate results

| Role | Set | Candidate pairs | Mean/S1 | p50/p90/p95/p99/max | Surviving truth | Edge recall | Positive any/all recall | Oracle macro F0.5 |
|---|---|---:|---:|---|---:|---:|---|---:|
| Calibration | K10 | 378,510 | 37.8510 | 38/40/40/40/40 | 33,422/34,611 | 0.965647 | 0.996187 / 0.896208 | 0.987714 |
| Calibration | K20 | 776,717 | 77.6717 | 78/80/80/80/80 | 33,730/34,611 | 0.974546 | 0.997035 / 0.921203 | 0.990746 |
| Evaluation | K10 | 378,511 | 37.8511 | 38/40/40/40/40 | 33,407/34,659 | 0.963877 | 0.996715 / 0.890749 | 0.987456 |
| Evaluation | K20 | 776,668 | 77.6668 | 78/80/80/80/80 | 33,749/34,659 | 0.973744 | 0.997775 / 0.917665 | 0.990993 |

All counts are independent measurements on the indicated complete S1 set, including singletons. `candidate_stats.csv` is the machine-readable source.

# Model

Algorithm: LightGBM 4.7.0 binary classifier, 450 trees, learning rate 0.05, 31 leaves, minimum 100 samples/leaf, 0.85 row subsampling, 0.90 column subsampling, L1 0.1, L2 1.0, deterministic seed 20260925, six CPU threads.

Training rows: 999,874 = 99,874 surviving positive pairs + 900,000 production-blocker hard negatives. The final compact policy keeps up to 15 hard negatives per S1 per source (30 total) to keep S2/S3 representation balanced. A 15-total ablation scored only 0.909073 calibration F0.5 at K10 versus 0.923719 for the retained policy, so it was rejected using calibration only. There are 49 numeric features; the exact schema is `feature_schema.json` and config is `model/model_config.json`.

# Calibration

Only fold 4 selected decision parameters. Coarse-to-fine searches were used; no fold-0 result affected selection.

| Experiment | Candidate set | Threshold | Owner margin | Calibration macro F0.5 | Precision | Recall | Singleton accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| E001 | K10 | 0.780 | — | 0.923719 | 0.946479 | 0.880356 | 0.815412 |
| E001 | K20 | 0.880 | — | 0.908166 | 0.930733 | 0.852157 | 0.829749 |
| E002 | K20 | 0.848 | 0 | 0.911043 | 0.926475 | 0.865390 | 0.810036 |
| E003 | K20 | 0.848 | 0.310 | 0.914791 | 0.936292 | 0.863685 | 0.820789 |

# Final untouched evaluation

| Experiment | Set | Actual macro F0.5 | Oracle F0.5 | Oracle − actual | Precision | Recall | Singleton accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| E001 independent | K10 | **0.924715** | 0.987456 | 0.062741 | 0.942685 | 0.879339 | 0.868561 |
| E001 independent | K20 | **0.908722** | 0.990993 | 0.082271 | 0.926812 | 0.852044 | 0.849023 |
| E002 one-owner | K20 | **0.912014** | 0.990993 | 0.078979 | 0.923115 | 0.864999 | 0.838366 |
| E003 one-owner + margin | K20 | **0.915268** | 0.990993 | 0.075725 | 0.932062 | 0.863326 | 0.847247 |

K10 is the best actual configuration despite its lower candidate ceiling. The K20 ceiling improves by 0.003537, but E001 actual performance drops by 0.015993.

# Decoder ablations

On the same K20 model scores, E002 improves over E001 by **+0.003292** F0.5. E003 improves over E002 by **+0.003254**, and over E001 K20 by **+0.006546**. Neither decoder recovers the K10 E001 score: E003 remains 0.009447 below it. E002/E003 prediction artifacts have zero duplicated target owners; independent E001 appropriately permits them.

# Slice diagnostics

The following are final E001 K10 slices unless noted:

| Slice | F0.5 | Precision | Recall | Note |
|---|---:|---:|---:|---|
| US | 0.941975 | 0.953542 | 0.895430 | Strongest country |
| India | 0.898739 | 0.926134 | 0.855217 | Weakest country |
| S2-related | 0.898237 | 0.927391 | 0.881650 | Weaker source behavior |
| S3-related | 0.911410 | 0.957523 | 0.877177 | Higher precision |
| Cardinality 0 | 0.868561 | — | — | Singleton accuracy 0.868561 |
| Cardinality 1 | 0.829013 | 0.760317 | 0.893657 | Weakest cardinality bucket |
| Cardinality 2 | 0.913020 | 0.892983 | 0.890882 |  |
| Cardinality 3 | 0.929122 | 0.935770 | 0.880028 |  |
| Cardinality 4 | 0.938132 | 0.953411 | 0.880163 |  |
| Cardinality 5+ | 0.948795 | 0.973712 | 0.875323 | Strongest substantive slice |
| Repeated/common normalized name | 0.883287 | 0.874577 | 0.882716 | Major ambiguity weakness |
| Non-ambiguous name | 0.939629 | 0.976039 | 0.877865 |  |
| Missing true-candidate address | 0.912198 | 0.977232 | 0.781502 | Recall-limited |
| No missing true-candidate address | 0.926318 | 0.937994 | 0.895192 |  |

India is also weakest under E003 (0.881657 versus US 0.937600). Complete experiment/slice rows are in `slice_results.csv`.

# Feature importance

The leading gain features are fused name/address similarity, numeric-token Jaccard, address token-set similarity, numeric-conflict indicators, reciprocal address rank, name partial similarity, address length ratio, name token-sort similarity, address character TF-IDF, name length difference, token Jaccard, Jaro-Winkler, numeric overlap count, name character TF-IDF, and normalized-address fanout. These are model-specific associations, not causal effects; correlated similarity features divide gain among themselves.

# Bottleneck diagnosis

**MATCHER limited, with a secondary DECODING/CALIBRATION limitation; not primarily BLOCKING limited at K10/K20.** K10 oracle is 0.987456 while actual is 0.924715, a 0.062741 gap. K20 raises oracle to 0.990993 yet lowers independent actual to 0.908722, widening the gap to 0.082271. Structured decoding recovers 0.006546 on K20 but leaves a 0.075725 gap. Candidate recall is already 96.39%/97.37% at K10/K20, whereas repeated names, singleton decisions, and cardinality-1 entities are visibly weak. Blocking still contributes a small ceiling loss, but more candidates alone currently worsen matching.

# Runtime/resources

The clean primary pipeline took 3,774.66 seconds for sampling/retrieval/features plus 110.17 seconds for the final fit/calibration/evaluation: **3,884.83 seconds (64m 44.83s)**. Retrieval dominated. Measured peak process RSS was 2.75 GiB; OS `ru_maxrss` peaked at **3.88 GiB**. The saved candidate cache is about 132 MiB, features 122 MiB, pair/prediction artifacts 82 MiB, and model artifacts about 3 MiB. The explicitly recorded 15-total-negative ablation added about 75 seconds and is not included in clean reproduction time.

# Reproduction command

From repository root:

```bash
PYTHON=/home/satvik/miniforge3/envs/deepchem/bin/python bash analysis/experiments_v1/e001_classical_baseline/run_e001.sh
```

Existing cached shards are reused. Do not pass `--force` unless intentional regeneration is desired.

# Exact spreadsheet rows

```text
Satvik | E001 lexical+LightGBM K10 | Val F0.5=0.924715 (Oracle=0.987456) | KEEP — Best actual baseline; K10 has a 0.062741 matcher gap and outperforms K20 despite a slightly lower ceiling.
Satvik | E001 lexical+LightGBM K20 | Val F0.5=0.908722 (Oracle=0.990993) | KILL — Extra candidates raise oracle by 0.003537 but reduce actual F0.5 by 0.015993, exposing matcher confusion.
Satvik | E002 K20 one-owner | Val F0.5=0.912014 (Oracle=0.990993) | INCONCLUSIVE — One-owner decoding adds 0.003292 over independent K20 but remains 0.012701 below E001 K10.
Satvik | E003 K20 one-owner+margin | Val F0.5=0.915268 (Oracle=0.990993) | KEEP — Margin decoding adds 0.003254 over E002 and 0.006546 over E001 K20, but matcher quality still dominates.
```
