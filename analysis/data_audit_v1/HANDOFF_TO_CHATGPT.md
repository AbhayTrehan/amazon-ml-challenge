# 1. Executive findings

This is a self-contained empirical handoff for modeling decisions. All exact statements use the full training data unless explicitly labeled as a deterministic sample.

1. Train has 12,527,040 records: S1 2,206,821; S2 5,034,616; S3 5,285,603. Test has 11,702,133: S1 1,732,544; S2 4,887,273; S3 5,082,316.
2. Ground truth has 7,638,365 positive S1→S2/S3 edges and exactly one row per train S1.
3. 123,247 S1 are true no-match entities: 5.5848209%. Predict-empty-for-all therefore scores exactly 0.0558482088 macro F0.5.
4. Matched S1 usually have several records: mean 3.666, median 4, p90/p95 6, p99 8, max 11.
5. Full cardinality counts for 0–11 matches are: 123,247; 119,157; 375,212; 530,841; 484,115; 321,957; 164,868; 63,968; 18,680; 4,205; 534; 37.
6. Every matched auxiliary record belongs to exactly one S1: zero S2/S3 has inverse degree >1. Components are S1-centered stars.
7. S2 matched rate is 3,693,619/5,034,616 = 73.3645%; S3 is 3,944,746/5,285,603 = 74.6319%. About one quarter of each auxiliary file is unmatched.
8. All 7,638,365 positive edges preserve country. Open-set country equality halves the Cartesian space and loses zero train truths.
9. IDs and row order have no detectable leakage: positive and random suffix/row-distance distributions coincide; zero same-suffix positives; modulo equality is chance-like.
10. Raw exact positives are rare: name 4.6360%, address 2.2266%, both only 18 edges. Punctuation normalization raises name to 21.8475%, address to 8.2658%.
11. Legal-suffix removal raises positive name exactness to 35.9585%, but S1 records in name-collision groups rise from 866,894 to 1,012,287 and max frequency from 253 to 539.
12. Even ordinary normalized names are highly ambiguous: 866,894 S1 (39.28%) are in a repeated-name group.
13. Address features are the best defense against same-normalized-name hard negatives: word TF-IDF AUC 0.9727, token Jaccard 0.9723, char TF-IDF 0.9638; numeric Jaccard 0.9002.
14. S3 addresses are noisier: positive median char-TFIDF is 0.884 India/0.815 US versus S2 0.935/0.911.
15. Address top-50 true-edge recall is 92.14% India S2, 83.69% India S3, 90.18% US S2, 89.74% US S3. India S3 is the main lexical-address weak spot.
16. On the common 2,000-query cohort, name+address top-50/source recalls 96.31% of 7,075 edges and has oracle macro F0.5 0.98790 at 197.33 candidates/S1.
17. Adding exact blocks raises recall only to 96.58% and oracle to 0.98871 at 200.19 candidates/S1; exact contributes only 19 unique true links after fuzzy channels.
18. The meaningful Pareto region is K=5/10/20: oracle 0.96215/0.97540/0.98178 at 18.31/37.94/77.65 candidates/S1 for name+address.
19. France is 14.975% of test S1, 14.392% S2, 14.395% S3. It is not a small edge case.
20. France token-occurrence OOV is ~40% against sampled train vocabulary, but character-trigram OOV is only 0.97–1.29%. Character retrieval is far less shifted than word retrieval.
21. France nearest neighbors have high top-1 scores but tiny margins (median ~0.012–0.017), so score magnitude alone may confuse ambiguity with confidence.
22. Exact/normalized `(name,address)` train-test overlap is zero for every source and country. Direct entity memorization cannot solve test.
23. Positive match-count/corruption distributions are almost identical by country, but India strings are longer/multiscript and France is a major lexical-format shift.
24. The official validator does not enforce target membership unless `--check-ids`; candidate/final subset mismatch is only a warning; candidate file is optional to the validator but required in the final ZIP.
25. No supervised baseline was run: the audited blocker projects to 84–435M training pairs depending on K, and a classifier on easy sampled negatives would be misleading.

# 2. Dataset table

| Split/source | Total | US | India | France | Missing name | Empty address |
|---|---:|---:|---:|---:|---:|---:|
| Train S1 | 2,206,821 | 1,323,633 | 883,188 | 0 | 0 | 0 |
| Train S2 | 5,034,616 | 3,016,817 | 2,017,799 | 0 | 0 | 168,967 (3.356%) |
| Train S3 | 5,285,603 | 3,170,056 | 2,115,547 | 0 | 0 | 175,916 (3.328%) |
| Test S1 | 1,732,544 | 663,106 | 809,986 | 259,452 | 0 | 0 |
| Test S2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 | 0 | 129,408 (2.648%) |
| Test S3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 | 0 | 136,098 (2.678%) |

Train is ~60% US/40% India per source. Test is ~38.3% US/~47.3% India/~14.4% France. No both-fields-missing rows exist. Tiny name sentinel counts exist (train S2 5, S3 18; test S2 49, S3 61); they were not treated as null.

France S1 name/address mean lengths are 19.42/50.07 chars, versus US 22.46/34.97 and India 26.37/77.71. France S1 non-ASCII rates are 15.72% name and 28.26% address.

All TSVs are UTF-8, LF-only, no BOM, zero blank/malformed tab-count lines, and parse exactly. IDs and rows are unique. Literal `0x1A` controls occur in some India addresses and must be normalized safely. Complete exact shape statistics are in `artifacts/schema_and_missingness.csv` and `artifacts/text_shape_stats.csv`.

# 3. Ground-truth graph

| True matches | S1 count | Share |
|---:|---:|---:|
| 0 | 123,247 | 5.5848% |
| 1 | 119,157 | 5.3995% |
| 2 | 375,212 | 17.0024% |
| 3 | 530,841 | 24.0546% |
| 4 | 484,115 | 21.9372% |
| 5 | 321,957 | 14.5892% |
| 6 | 164,868 | 7.4708% |
| 7 | 63,968 | 2.8986% |
| 8 | 18,680 | 0.8465% |
| 9 | 4,205 | 0.1905% |
| 10 | 534 | 0.0242% |
| 11 | 37 | 0.0017% |

Composition: S2-only 143,029; S3-only 164,498; both 1,776,047; no match 123,247. Per-S1 S2 count max 5; S3 max 6. India: 883,188 S1, 3,059,843 edges; US: 1,323,633 S1, 4,578,522 edges. Country cardinality distributions are nearly identical.

Inverse mapping: S2 unmatched/exactly-one/>1 = 1,340,997 / 3,693,619 / 0. S3 = 1,340,857 / 3,944,746 / 0. Thus components are only isolated S1 or one S1 plus 1–11 leaves. The empirical one-owner constraint is real in train but unproven for test.

# 4. Metric implications

For truth size `m`, predictions `k`, TP `t`:

`F0.5 = 1.25t / (1.25t + 0.25(m-t) + (k-t))`; empty/empty = 1.

One TP plus one FP when there is one truth scores 0.5556. Two TP plus one FP for two truths scores 0.7143. Two of three truths with no FP scores 0.9091. This is why entity grouping, singleton accuracy, and precision dominate; pair AUC/class imbalance alone are insufficient.

All-empty baseline = **0.0558482088**. Cardinality groups contribute to macro score in direct proportion to their S1 counts, not their edge counts.

# 5. Text/noise structure

| Normalization | Positive name exact | Positive address exact | S1 records in name collisions |
|---|---:|---:|---:|
| Raw | 4.6360% | 2.2266% | 845,385 |
| Lower/whitespace | 15.7710% | 7.4134% | 846,097 |
| Punctuation normalized | 21.8475% | 8.2658% | 866,894 |
| Legal suffix removed | 35.9585% | 8.2658% | 1,012,287 |
| Accent folded | 25.7868% | 8.2659% | 866,894 |

Frequent generic names and shared addresses create hard negatives. The 80k-positive heuristic taxonomy: mixed partial 27.10%, normalized-name-exact-only 18.33%, legal-suffix-only 17.14%, both channels strong 13.60%, name badly corrupt/address recovers 7.77%, normalized-address-exact-only 5.39%, name reordering 4.53%, address missing one side 4.30%, normalized-both exact 1.30%, address bad/name recovers 0.29%, both weak 0.245%.

# 6. Source differences

| Positive median | S2 India | S3 India | S2 US | S3 US |
|---|---:|---:|---:|---:|
| Name char-TFIDF | 0.837 | 0.846 | 0.951 | 0.921 |
| Address char-TFIDF | 0.935 | 0.884 | 0.911 | 0.815 |
| Address Jaccard | 0.800 | 0.636 | 0.667 | 0.444 |

S3 address corruption is materially worse, especially India retrieval recall (83.69%). This supports source-aware calibration/thresholds, chiefly for address, but not automatically separate model architectures.

# 7. Country + France shift

Zero positive country mismatches in 7,638,365 edges. Use generic string equality as a block; do not hard-code known country categories.

Train/test normalized name overlap for S1: US 35.85%, India 44.41%; normalized address overlap 5.54%/4.85%. All exact/normalized name+address pair overlaps are zero. France versus any training country: S1 normalized name overlap 25/259,452 = 0.0096%; S2 1.8947%; S3 3.1519%; address/pair overlap zero.

Sampled OOV:

| Country/source | Token occurrence OOV | Token type OOV | Char-trigram occurrence OOV | Token JS distance |
|---|---:|---:|---:|---:|
| France S1 | 40.68% | 66.87% | 0.966% | 0.926 |
| France S2 | 40.28% | 72.87% | 1.294% | 0.904 |
| France S3 | 40.04% | 71.88% | 1.248% | 0.902 |
| Seen-country range | 2.15–4.96% | 17.1–37.3% | 0.008–0.063% | 0.497–0.574 |

France full-index top-1 name median is 1.0; address 0.941/0.952, but median top1–top2 margins are only 0.0115–0.0172. We can infer char retrieval remains viable, not France accuracy or thresholds.

# 8. Leakage/artifact findings

No suffix, offset, order, percentile, or modulo signal. Positive suffix-distance mean ~333.2–333.6M/median ~292.7–293.5M matches random controls ~330.2–337.0M/~288.3–295.5M. Positive relative row-distance mean ~0.333 and median ~0.293, also random. Same suffix: 0 positives. Mod-100: ~1%; mod-1000 ~0.1%. Train/test ID overlap: zero for every source. Classification: **no suspicious leakage-like feature found**.

# 9. Positive vs hard-negative similarity

Sample: 80,000 positives, 79,387 same-country random negatives, 80,000 exact-normalized-name collision negatives; seed 20260925.

| Feature | Positive mean | Random mean / AUC | Name-collision mean / AUC |
|---|---:|---:|---:|
| Name char-TFIDF | 0.753 | 0.0187 / 0.9294 | 1.000 / poor by construction |
| Address word-TFIDF | 0.683 | 0.0054 / 0.9732 | 0.0071 / 0.9727 |
| Address char-TFIDF | 0.800 | 0.0179 / 0.9642 | 0.0229 / 0.9638 |
| Address Jaccard | 0.610 | 0.0165 / 0.9728 | 0.0187 / 0.9723 |
| Numeric Jaccard | 0.733 | 0.0147 / 0.9005 | 0.0152 / 0.9002 |

Retrieval margins are useful ambiguity features: India S2 address true-top1 median margin 0.0549 versus false-top1 0.0232; US S3 0.0664 versus 0.0299. False name top1s often have score 1 and margin 0 because of repeated names.

# 10. Candidate-generation benchmark

Full Cartesian train pairs: 22,774,876,013,799. Country restriction: 11,839,670,856,657 (51.99%). Full exact normalized name/address union: 22,350,583 pairs, mean 10.128, p50 2, p95 61, p99 176, max 1,042; edge recall 28.8043%; oracle macro F0.5 0.51877.

Common deterministic 2,000-query cohort, full country/source auxiliary indexes:

| Channels | K per source | Mean cand/S1 | p95 | Edge recall | Any true / positive S1 | All true / positive S1 | Oracle F0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Name+address | 1 | 3.63 | 4 | 51.56% | 1,816/1,899 | 324/1,899 | 0.81565 |
| + exact | 1 | 11.72 | 50.05 | 60.10% | 1,852 | 433 | 0.86072 |
| Name+address | 5 | 18.31 | 20 | 89.10% | 1,884 | 1,359 | 0.96215 |
| + exact | 5 | 24.86 | 58.1 | 90.20% | 1,887 | 1,393 | 0.96675 |
| Name+address | 10 | 37.94 | 40 | 92.74% | 1,892 | 1,517 | 0.97540 |
| + exact | 10 | 43.58 | 68.15 | 93.36% | 1,894 | 1,537 | 0.97800 |
| Name+address | 20 | 77.65 | 80 | 94.52% | 1,895 | 1,599 | 0.98178 |
| + exact | 20 | 82.25 | 92.05 | 94.97% | 1,897 | 1,612 | 0.98375 |
| Address only | 50 | 100 | 100 | 88.89% | 1,879 | 1,342 | 0.95882 |
| Name+address | 50 | 197.33 | 200 | 96.31% | 1,899 | 1,689 | 0.98790 |
| + exact | 50 | 200.19 | 207 | 96.58% | 1,899 | 1,701 | 0.98871 |

Runtime/memory examples: India S2 name top-100/5k queries 235.8s and 1.83 GiB; US S3 name top-50/1k 138.7s and 2.61 GiB; address jobs 69–121s/1k and 1.85–1.99 GiB. Sparse matrices have 80.6–169.7M nonzeros.

# 11. Best blocking union + failure modes

At K=50, address recovers 6,289 truths and has 1,771 unique over name+exact; name 4,663/364 unique; exact 2,015/19 unique. Adding exact to fuzzy channels costs 5,720 sample pairs for 19 unique truths and +0.00080 oracle.

Best measured ceiling is all-union K=50: 6,833/7,075 truths, 242 misses. Misses: S2 India 65, S2 US 41, S3 India 101, S3 US 35. Failure patterns: script/transliteration switches, missing address, aggressive shortening, name typos plus truncated address, only weak locality/number fragments, repeated noisy variants, and a tiny both-weak tail. K=10–20 is the more attractive efficiency band; do not assume K=50 is the final choice.

# 12. Validation recommendation

Primary IID: split by S1/star component; all matched S2/S3 stay with owner S1. Assign unmatched auxiliary negatives by their own deterministic ID hash. Score complete S1 groups including empty predictions with macro F0.5. Use `proposed_validation_split.tsv`: seed 20260925, five folds; fold 0 as fixed ~20% validation for fast iteration, another fold for occasional confirmation. Fold country counts are balanced (India 176,021–177,528; US 264,381–265,102).

Cross-country stress: train/tune on US, evaluate full India entities, then reverse. Keep it separate from primary IID score. It probes country sensitivity but cannot identify France threshold behavior.

# 13. Optional supervised baseline

Not run. An honest baseline requires actual blocker hard negatives and complete group scoring. K=10 name+address projects to ~83.7M training candidates; K=50 ~435.5M. A toy model trained on easy sampled negatives would mislead, so the optional stage was skipped.

# 14. Compute constraints

Machine: 12 CPUs, 15 GiB RAM, no swap. Audit DB ~4.3 GiB; complete edge parquet 55 MiB. Sequential sparse indexes peaked at 2.61 GiB process RSS. Full Cartesian and full exhaustive S1 near-neighbor comparisons are impossible locally; source×country sequential sparse retrieval is comfortable. Exact union must be counted/grouped rather than naively materialized. Reproducible successful stages take ~35 minutes on this machine.

# 15. Modeling implications

- Country equality is a justified open-set block.
- Preserve multiple normalization views; do not overwrite raw text.
- Address retrieval should anchor candidates; name char retrieval is complementary.
- Address and numeric features are essential for repeated-name hard negatives.
- Legal suffix/accent views help recall but require ambiguity/fanout features.
- Source-aware address calibration is empirically justified; India S3 needs attention.
- Top1/top2 margins, exact-key fanout, token rarity, rank, and source are evidence-backed features.
- Consider K=10–20 first for iteration; measure full validation oracle before enlarging.
- One-owner decoding is plausible from train but should be ablated, not assumed.
- France argues for character-robust and open-set handling; avoid vocabulary/country one-hot dependence.
- The final threshold must optimize grouped macro F0.5 and singleton accuracy, not pair AUC.

# 16. Open questions

1. Does test preserve country equality and auxiliary inverse uniqueness?
2. What is France singleton/multiplicity/noise behavior?
3. Can a complementary transliteration/learned channel recover the remaining 3.42% K=50 misses without candidate explosion?
4. What source/country thresholds maximize entity macro F0.5?
5. Does one-to-one global assignment improve or harm validation?
6. Does the 2,000-query retrieval ceiling hold on every S1?
7. How do K=10/20 candidate misses interact with a calibrated matcher?

# 17. Artifact paths and reproduction commands

Main report: `analysis/data_audit_v1/REPORT.md`  
Complete positives: `analysis/data_audit_v1/artifacts/positive_edges.parquet`  
Reusable catalog: `analysis/data_audit_v1/artifacts/audit.duckdb`  
Core graph: `analysis/data_audit_v1/artifacts/ground_truth_graph.json`  
Similarity: `analysis/data_audit_v1/artifacts/similarity_positive_stats.csv`, `similarity_negative_stats.csv`, `feature_discriminability.csv`  
Blocking: `analysis/data_audit_v1/artifacts/blocking_benchmarks_all.csv`, `blocking_channel_complementarity.csv`, `blocking_misses.tsv`  
Shift: `analysis/data_audit_v1/artifacts/train_test_overlap.csv`, `train_test_shift.csv`, `france_retrieval_confidence.csv`  
Validation: `analysis/data_audit_v1/artifacts/proposed_validation_split.tsv`

From repository root:

```bash
PYTHON=/home/satvik/miniforge3/envs/deepchem/bin/python \
  bash analysis/data_audit_v1/run_audit.sh
```

Dependencies are pinned in `analysis/data_audit_v1/requirements_analysis.txt`. Retrieval jobs are deliberately sequential to avoid exhausting RAM.
