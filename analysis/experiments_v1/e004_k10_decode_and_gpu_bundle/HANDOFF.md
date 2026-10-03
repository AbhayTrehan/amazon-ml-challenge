# 1. Baseline reproduction

E004-lite used only E001 cached scores with `in_k10=1`. At the frozen E001 threshold 0.780, the exact grouped metric reproduced:

| Role | Expected F0.5 | Reproduced F0.5 | Difference |
|---|---:|---:|---:|
| Fold 4 calibration | 0.923718737208 | 0.923718737208 | 0 |
| Fold 0 evaluation | 0.924715138630 | 0.924715138630 | 0 |

The script aborts before decoder work if either check differs by more than `1e-12`.

# 2. E004A/B/C results

All decoder parameters were selected on fold 4. `frozen_parameters.json` was written before fold 0 was loaded.

| Experiment | Frozen rule | Calibration F0.5 | Fold-0 F0.5 | Delta vs E001 K10 | Precision | Recall | Singleton accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| E004A | one owner; threshold 0.780 | 0.925007 | 0.926509 | +0.001793 | 0.945841 | 0.879281 | 0.875666 |
| E004B | one owner; threshold 0.780; margin 0.485 | 0.926398 | **0.928336** | **+0.003621** | 0.949738 | 0.877752 | 0.886323 |
| E004C | E004B + S2 threshold 0.717 / S3 threshold 0.757 | 0.926785 | 0.927278 | +0.002563 | 0.941773 | 0.887591 | 0.866785 |

Fold-0 diagnostics:

| Experiment | Cardinality-1 F0.5 | Repeated-name F0.5 | India | US | S2 | S3 |
|---|---:|---:|---:|---:|---:|---:|
| E004A | 0.834174 | 0.885884 | 0.902912 | 0.942187 | 0.900434 | 0.911870 |
| E004B | **0.841237** | **0.888798** | **0.908002** | 0.941847 | 0.902293 | 0.911673 |
| E004C | 0.838929 | 0.883352 | 0.905554 | 0.941713 | **0.906078** | **0.913015** |

# 3. Best decoder and delta

**E004B is best: fold-0 macro F0.5 0.928336, +0.003621 over E001 K10 0.924715.** It removes 298 predictions relative to independent E001, raises precision from 0.942685 to 0.949738, gives up 0.001587 recall, and raises singleton accuracy from 0.868561 to 0.886323.

# 4. Decoder branch decision

**NOT CLOSED under the specified stop rule**, because E004B improves by 0.003621, just above the 0.003 boundary. It remains a secondary, tightly bounded branch: simple ownership contributes only 0.001793, and source-specific thresholds fail to beat the single-threshold margin decoder on fold 0. Future attention should be limited to the owner-margin rule rather than broader decoder/HPO work.

# 5. Neural shortlist oracle table

Candidates are ranked across the complete combined K10 union for each S1 using cached LightGBM score. Full K10 contains 378,510/378,511 pairs and has oracle 0.987714/0.987456 on calibration/evaluation.

| Role | Shortlist | Pairs | K10 fraction | Edge recall | Positive any recall | Positive all recall | Oracle macro F0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Calibration | top 1 | 10,000 | 0.026419 | 0.269654 | 0.988456 | 0.052531 | 0.687131 |
| Calibration | top 2 | 20,000 | 0.052839 | 0.518043 | 0.993645 | 0.209701 | 0.861123 |
| Calibration | top 3 | 30,000 | 0.079258 | 0.716044 | 0.995128 | 0.435183 | 0.934915 |
| Calibration | top 5 | 50,000 | 0.132097 | 0.916529 | 0.995975 | 0.779920 | 0.980245 |
| Calibration | top 8 | 80,000 | 0.211355 | 0.962613 | 0.996187 | 0.887100 | 0.987134 |
| Calibration | top 10 | 100,000 | 0.264194 | 0.964462 | 0.996187 | 0.892713 | 0.987443 |
| Calibration | top 5 + score ≥0.05 | 61,621 | 0.162799 | 0.963162 | 0.996187 | 0.887842 | 0.987205 |
| Evaluation | top 1 | 10,000 | 0.026419 | 0.269512 | 0.989827 | 0.052771 | 0.688059 |
| Evaluation | top 2 | 20,000 | 0.052839 | 0.517759 | 0.995020 | 0.212462 | 0.861949 |
| Evaluation | top 3 | 30,000 | 0.079258 | 0.714273 | 0.995655 | 0.433718 | 0.934702 |
| Evaluation | top 5 | 50,000 | 0.132097 | 0.911913 | 0.996397 | 0.763908 | 0.979475 |
| Evaluation | top 8 | 80,000 | 0.211354 | 0.960501 | 0.996609 | 0.880576 | 0.986769 |
| Evaluation | top 10 | 100,000 | 0.264193 | 0.962549 | 0.996609 | 0.887040 | 0.987096 |
| Evaluation | top 5 + score ≥0.05 | 61,913 | 0.163570 | 0.960443 | 0.996609 | 0.879835 | 0.986696 |

# 6. Recommended cross-encoder shortlist

Use **top 5 plus every additional candidate with LightGBM score ≥0.05**. It averages 6.19 candidates/S1 on fold 0, retains only 16.36% of K10 pairs, and gives oracle 0.986696—only 0.000760 below full K10 and 0.000073 below top 8 while scoring 18,087 fewer pairs. If a fixed tensor budget is operationally preferable, top 8 is the simple fallback. Plain top 5 loses 0.007981 oracle and is too aggressive.

# 7. Auxiliary-owner ambiguity

| Role | Distinct auxiliaries | 1 owner | 2 owners | 3 owners | 4+ owners |
|---|---:|---:|---:|---:|---:|
| Calibration | 275,064 | 218,900 (79.58%) | 36,579 (13.30%) | 10,217 (3.71%) | 9,368 (3.41%) |
| Evaluation | 275,561 | 219,186 (79.54%) | 37,044 (13.44%) | 9,814 (3.56%) | 9,517 (3.45%) |

For E001 fold 0, 832/1,853 false positives (**44.9002%**) occur on auxiliaries with more than one candidate S1 owner. Of 4,182 false negatives, **7.8910%** are associated with such ambiguous-owner groups; this denominator includes blocked-out true edges. Across all 34,659 true edges, true-owner recall is **0.963328 top 1**, **0.963848 top 2**, and **0.963877 top 3**. Thus ownership competition explains many false positives but almost no incremental true-owner recovery beyond rank 1; blocking accounts for most of the remaining shortfall.

# 8. GPU bundle

The bundle is self-contained at `gpu_bundle/`; Colab does not need the 12M-record source files.

| File | Rows | Positives | Negatives | Unique S1 | Bytes | SHA256 |
|---|---:|---:|---:|---:|---:|---|
| `train_pairs.parquet` | 398,428 | 98,428 | 300,000 | 30,000 | 17,994,091 | `53ffafc6ecd8fa4931e74dbf4bc5c0a2f58b92e2615dace107d97befc681754c` |
| `calibration_pairs.parquet` | 378,510 | 33,422 | 345,088 | 10,000 | 16,656,129 | `c0b336d32a7ba4b888d56c6fe163c9b26c02b0754f07aeb111f4bd7b8c4a7ecb` |
| `evaluation_pairs.parquet` | 378,511 | 33,407 | 345,104 | 10,000 | 16,657,140 | `394faf61ddfb1c68c282ffbead9bac3516aa84a79b5b4a962e87b4c0eebc25d7` |

Training contains every surviving positive in the cached E001 K10 hard-pair pool and up to five highest-LightGBM-score false pairs per source/S1 (ten total). S2/S3 counts are 198,053/200,375; India/US counts are 158,701/239,727. Calibration and evaluation preserve the complete K10 sets and metadata needed to reconstruct local evaluation. Exact columns, dtypes, source/country counts, byte sizes, provenance, and checksums are in `gpu_bundle/bundle_manifest.json`.

# 9. Exact command

From repository root:

```bash
PYTHON=/home/satvik/miniforge3/envs/deepchem/bin/python bash analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/run_e004.sh
```

Measured runtime: **69.03 seconds**; peak RSS: **2.51 GiB**.

# 10. Copy-paste spreadsheet rows

```text
Satvik | E004A K10 one-owner | Val F0.5=0.926509 | KILL — Simple one-owner decoding adds only 0.001793 and is dominated by the margin decoder.
Satvik | E004B K10 one-owner+margin | Val F0.5=0.928336 | KEEP — A 0.485 owner margin adds 0.003621 over E001 K10, improves precision and singleton accuracy, and clears the decoder stop rule narrowly.
Satvik | E004C K10 source thresholds | Val F0.5=0.927278 | KILL — Separate S2/S3 thresholds improve calibration slightly but generalize worse than E004B on untouched fold 0.
```
