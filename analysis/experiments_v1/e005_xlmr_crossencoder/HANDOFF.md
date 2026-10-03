# E005 XLM-R cross-encoder handoff

# Environment / GPU

Host `sbilabwstsn`; reusable venv `/home/satvik/venvs/amazonml2026`; PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.17.0. Physical GPU 1, RTX 6000 Ada Generation (48 GiB), UUID `GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17`; BF16 and SDPA; one GPU only.

# Input integrity

All three manifest SHA256 hashes and row counts passed before training and after evaluation. Schema/labels passed; train, fold 4, and fold 0 S1 sets are pairwise disjoint. No official data was modified and no external business lookup occurred.

# Dataset / shortlist

Training: 398,428 rows (98,428 positive, 300,000 hard negative), 30,000 S1. Shortlist rule: rank <=5 OR LightGBM >=0.05. Fold 4: 61,621 pairs, oracle 0.987205. Fold 0: 61,913 pairs, oracle 0.986696.

# Model and revision

`FacebookAI/xlm-roberta-base`, revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, MIT, 278,045,186 parameters, two-class cross-entropy, full fine-tuning.

# Token lengths / max_length

p50/p90/p95/p99/p99.5/max = 52/80/88/104/111/170; max_length=104; measured truncation=0.962%.

# Throughput benchmark

Batch 64 BF16: 994.5 train examples/s, 4332.4 inference pairs/s, 5.67 GiB peak allocated.

# Training

```text
 epoch     loss    seconds  examples_per_second  optimizer_steps  learning_rate  peak_allocated_gpu_gb  peak_reserved_gpu_gb
     1 0.068547 405.459270           982.658504             6226       0.000011               5.863324              7.076172
     2 0.024466 393.158473          1013.403061             6226       0.000000               5.874730              6.605469
```

# Checkpoint selection

Epoch 1 fold-4 E005A macro F0.5 was **0.972673**; epoch 2 was **0.975199**. Epoch 2 was therefore selected strictly by fold-4 E005A grouped macro F0.5. Fold 0 was not read until `model/frozen_parameters.json` was written.

# Fair shortlist classical control

Fold 4=0.926398; fold 0=0.928336; full-K10 E004B fold 0=0.928336.

# E005A

Frozen fold-4 configuration: `{"decoder": "one_owner_margin", "experiment": "E005A", "macro_f05": 0.9751993390048544, "margin": 0.2, "precision": 0.9925388978192957, "predicted_edges": 32971, "recall": 0.9455086533183092, "singleton_accuracy": 0.9659498207885304, "stage": "owner_coarse", "threshold": 0.6500000000000001, "tp": 32725, "true_edges": 34611, "weight": null}`. Fold-0 macro F0.5=0.975373, precision=0.992266, recall=0.943997, singleton accuracy=0.976909, delta vs E004B=+0.047037, oracle gap=0.011323. Decision: STRONG KEEP.

# E005B

Frozen fold-4 configuration: `{"decoder": "one_owner_margin", "experiment": "E005B", "macro_f05": 0.9759674430139592, "margin": 0.48000000000000004, "precision": 0.9922963052475756, "predicted_edges": 33101, "recall": 0.9490046517003264, "singleton_accuracy": 0.9623655913978495, "stage": "owner_fine", "threshold": 0.5100000000000001, "tp": 32846, "true_edges": 34611, "weight": 0.6}`. Fold-0 macro F0.5=0.976694, precision=0.992441, recall=0.947084, singleton accuracy=0.978686, delta vs E004B=+0.048358, oracle gap=0.010002. Decision: STRONG KEEP.

# Important slices

```text
experiment                          slice           value  macro_f05
     E005A                        country              US   0.980061
     E005A                        country           India   0.968318
     E005A                         source              S2   0.966192
     E005A                         source              S3   0.963538
     E005A                    cardinality               0   0.976909
     E005A                    cardinality               1   0.929446
     E005A                    cardinality               2   0.971765
     E005A                    cardinality               3   0.976637
     E005A                    cardinality               4   0.980478
     E005A                    cardinality              5+   0.981371
     E005A                 name_ambiguity repeated/common   0.971217
     E005A                 name_ambiguity   non_ambiguous   0.976869
     E005A missing_true_candidate_address            true   0.958627
     E005B                        country              US   0.981447
     E005B                        country           India   0.969541
     E005B                         source              S2   0.968024
     E005B                         source              S3   0.965809
     E005B                    cardinality               0   0.978686
     E005B                    cardinality               1   0.932348
     E005B                    cardinality               2   0.974855
     E005B                    cardinality               3   0.977185
     E005B                    cardinality               4   0.981414
     E005B                    cardinality              5+   0.982149
     E005B                 name_ambiguity repeated/common   0.971271
     E005B                 name_ambiguity   non_ambiguous   0.978646
     E005B missing_true_candidate_address            true   0.966066
```

E005A: India improved vs 0.908002; cardinality-1 improved vs 0.841237; repeated-name improved vs 0.888798.

E005B: India improved vs 0.908002; cardinality-1 improved vs 0.841237; repeated-name improved vs 0.888798.

# Complementarity

```text
best_experiment         slice                                event  count
          E005B       overall  neural_recovers_lgbm_false_negative   2503
          E005B       overall     neural_fixes_lgbm_false_positive   1523
          E005B       overall neural_introduces_new_false_positive    163
          E005B       overall      neural_loses_lgbm_true_positive    100
          E005B         India  neural_recovers_lgbm_false_negative   1121
          E005B         India     neural_fixes_lgbm_false_positive    691
          E005B         India neural_introduces_new_false_positive     75
          E005B         India      neural_loses_lgbm_true_positive     46
          E005B repeated_name     neural_fixes_lgbm_false_positive   1161
          E005B repeated_name  neural_recovers_lgbm_false_negative    620
          E005B repeated_name neural_introduces_new_false_positive     69
          E005B repeated_name      neural_loses_lgbm_true_positive     20
          E005B cardinality_1     neural_fixes_lgbm_false_positive    116
          E005B cardinality_1  neural_recovers_lgbm_false_negative     30
          E005B cardinality_1 neural_introduces_new_false_positive     14
          E005B cardinality_1      neural_loses_lgbm_true_positive      1
```

# Runtime / production projection

Epoch times: 405.5s, 393.2s. Measured evaluation inference: 2483.0 pairs/s. Projected full-test K10: 65,576,790 pairs / 7.34 h. Projected shortlist: 10,724,447 pairs / 1.20 h. Detached pipeline wall time: 18.4 min.

# Decision

Best result: E005B, delta +0.048358; **STRONG KEEP**. E006 was not started.

# Artifact paths

Best model: `model/best_checkpoint/` (`model.safetensors` SHA256 `e7079b798b866b27116a0059c9c7cf1531817e42659ec0c9442654aee5d0c462`); predictions/search/results/slices/disagreements: `outputs/`; runtime and integrity evidence: `runtime.json` and `runtime/`.

# Exact reproduction command

```bash
bash analysis/experiments_v1/e005_xlmr_crossencoder/scripts/run_e005.sh
```

# Spreadsheet rows

Satvik | E005A XLM-R cross-encoder | Val F0.5=0.975373 (Oracle=0.986696) | STRONG KEEP — joint multilingual pair scoring result under the frozen shortlist.
Satvik | E005B XLM-R + LightGBM fusion | Val F0.5=0.976694 (Oracle=0.986696) | STRONG KEEP — simple fold-4-calibrated fusion result under the frozen shortlist.
