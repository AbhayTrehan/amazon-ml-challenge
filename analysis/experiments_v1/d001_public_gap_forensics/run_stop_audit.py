#!/usr/bin/env python3
"""Phase-A integrity and Phase-B cohort audit.

This script is intentionally incapable of running model inference.  The frozen
R006B selection used fold-0 labels, which triggers the requested stop condition.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/experiments_v1/d001_public_gap_forensics"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E004 = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cohort_metrics(frame: pd.DataFrame, label: str) -> list[dict]:
    n = len(frame)
    rows = []

    def add(metric: str, value: float | int) -> None:
        rows.append({"cohort": label, "metric": metric, "value": value})

    for country, count in frame.country.value_counts().sort_index().items():
        add(f"country_{country}_share", count / n)
        add(f"country_{country}_count", int(count))
    for card in range(12):
        count = int(frame.true_matches.eq(card).sum())
        add(f"cardinality_{card}_share", count / n)
        add(f"cardinality_{card}_count", count)
    add("singleton_share", frame.true_matches.eq(0).mean())
    add("s2_only_share", ((frame.s2_matches.gt(0)) & frame.s3_matches.eq(0)).mean())
    add("s3_only_share", ((frame.s3_matches.gt(0)) & frame.s2_matches.eq(0)).mean())
    add("both_share", ((frame.s2_matches.gt(0)) & frame.s3_matches.gt(0)).mean())
    add("neither_share", ((frame.s2_matches.eq(0)) & frame.s3_matches.eq(0)).mean())
    add("repeated_common_name_share", frame.repeated_name.mean())
    add("missing_true_candidate_address_share", frame.missing_true_candidate_address.mean())
    add("mean_true_matches_per_s1", frame.true_matches.mean())
    add("s1_count", n)
    return rows


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_logs").mkdir(exist_ok=True)
    split = pd.read_csv(SPLIT, sep="\t")
    fold0 = split.loc[split.fold.eq(0)].copy()
    eval10 = pd.read_csv(E001 / "selected_ids/evaluation_s1.tsv", sep="\t")
    e005_eval_ids = set(pd.read_parquet(
        E005 / "runtime/selected_s1_metadata.parquet", columns=["source1_entity_id", "role"]
    ).query("role == 'evaluation'").source1_entity_id)
    r007_eval_ids = set(pd.read_parquet(
        ROOT / "analysis/experiments_v1/r007_fast_blocker_e005b/evaluation_predictions.parquet",
        columns=["source1_entity_id"],
    ).source1_entity_id)
    evaluation_ids_identical = set(eval10.source1_entity_id) == e005_eval_ids == r007_eval_ids

    con = duckdb.connect(str(DB), read_only=True)
    con.register("split_map", split)
    con.register("fold0_ids", fold0[["source1_entity_id"]])

    # Actual LightGBM fit rows: the train feature shards loaded by
    # train_and_evaluate.py.  Query the realized parquet files, not intent docs.
    lgb_glob = str(E001 / "cache/features/train_*.parquet")
    lgb = con.execute(
        f"""select count(*) as n_rows, count(distinct s1) as unique_s1,
                   sum(label)::bigint positives,
                   sum(case when s.fold=0 then 1 else 0 end)::bigint fold0_rows,
                   count(distinct case when s.fold=0 then p.s1 end)::bigint fold0_s1
            from read_parquet('{lgb_glob}') p
            left join split_map s on s.source1_entity_id=p.s1"""
    ).fetchdf().iloc[0].to_dict()
    lgb_fold_unknown = con.execute(
        f"""select count(distinct p.s1) from read_parquet('{lgb_glob}') p
             left join split_map s on s.source1_entity_id=p.s1 where s.fold is null"""
    ).fetchone()[0]
    lgb_pos_overlap = con.execute(
        f"""select count(*) from read_parquet('{lgb_glob}') p
             join edges e on p.s1=e.s1 and p.target_id=e.target_id
                         and p.source=e.target_source
             join split_map s on s.source1_entity_id=e.s1
             where p.label=1 and s.fold=0"""
    ).fetchone()[0]

    xlm_path = E004 / "gpu_bundle/train_pairs.parquet"
    xlm = con.execute(
        f"""select count(*) as n_rows, count(distinct p.source1_entity_id) as unique_s1,
                   sum(label)::bigint positives,
                   sum(case when label=0 then 1 else 0 end)::bigint hard_negatives,
                   sum(case when s.fold=0 then 1 else 0 end)::bigint fold0_rows,
                   count(distinct case when s.fold=0 then p.source1_entity_id end)::bigint fold0_s1
            from read_parquet('{xlm_path}') p
            left join split_map s on s.source1_entity_id=p.source1_entity_id"""
    ).fetchdf().iloc[0].to_dict()
    xlm_fold_unknown = con.execute(
        f"""select count(distinct p.source1_entity_id) from read_parquet('{xlm_path}') p
             left join split_map s on s.source1_entity_id=p.source1_entity_id
             where s.fold is null"""
    ).fetchone()[0]
    xlm_pos_overlap = con.execute(
        f"""select count(*) from read_parquet('{xlm_path}') p
             join edges e on p.source1_entity_id=e.s1
                         and p.candidate_entity_id=e.target_id
                         and cast(substr(p.candidate_source,2) as integer)=e.target_source
             join split_map s on s.source1_entity_id=e.s1
             where p.label=1 and s.fold=0"""
    ).fetchone()[0]

    # Verify that realized calibration predictions are exclusively fold 4.
    cal_lgb = E001 / "predictions/calibration_pair_scores.parquet"
    cal_xlm = E005 / "outputs/calibration_predictions.parquet"
    calibration = {}
    for name, path, col in [
        ("lightgbm", cal_lgb, "s1"),
        ("xlmr_fusion_margin", cal_xlm, "source1_entity_id"),
    ]:
        counts = con.execute(
            f"""select s.fold, count(distinct p.{col}) as s1, count(*) as n_rows
                 from read_parquet('{path}') p left join split_map s
                 on s.source1_entity_id=p.{col} group by s.fold order by s.fold"""
        ).fetchdf()
        calibration[name] = counts.to_dict(orient="records")

    # Existing repeated-name feature semantics: query normalized name occurs in
    # more than one fold-0 auxiliary record in either source/country index.
    assignment = E001 / "cache/auxiliary_fold_assignment.parquet"
    repeated = con.execute(
        f"""with aux as (
               select 2 src,a.country,a.nname,count(*) n
               from train_s2 a join read_parquet('{assignment}') f
                 on f.source=2 and f.entity_id=a.entity_id
               where f.fold=0 group by a.country,a.nname
               union all
               select 3 src,a.country,a.nname,count(*) n
               from train_s3 a join read_parquet('{assignment}') f
                 on f.source=3 and f.entity_id=a.entity_id
               where f.fold=0 group by a.country,a.nname
             ), flags as (
               select q.entity_id source1_entity_id,
                      max(case when a.n>1 then 1 else 0 end)::boolean repeated_name
               from train_s1 q left join aux a
                 on a.country=q.country and a.nname=q.nname
               join fold0_ids f on f.source1_entity_id=q.entity_id
               group by q.entity_id
             ) select * from flags"""
    ).fetchdf()
    missing_addr = con.execute(
        """select f.source1_entity_id,
                  coalesce(bool_or(coalesce(a.business_address,'')=''),false) missing_true_candidate_address
           from fold0_ids f left join edges e on e.s1=f.source1_entity_id
           left join (
             select entity_id,business_address from train_s2
             union all select entity_id,business_address from train_s3
           ) a on a.entity_id=e.target_id group by f.source1_entity_id"""
    ).fetchdf()
    con.close()

    fold0 = fold0.merge(repeated, on="source1_entity_id", how="left").merge(
        missing_addr, on="source1_entity_id", how="left"
    )
    eval10 = eval10.merge(
        fold0[["source1_entity_id", "repeated_name", "missing_true_candidate_address"]],
        on="source1_entity_id", how="left"
    )
    dist_long = pd.DataFrame(cohort_metrics(eval10, "10k") + cohort_metrics(fold0, "full_fold0"))
    dist = dist_long.pivot(index="metric", columns="cohort", values="value").reset_index()
    dist["delta_10k_minus_full"] = dist["10k"] - dist["full_fold0"]
    dist.to_csv(OUT / "ten_k_representativeness.csv", index=False)

    r006_results = json.loads((R006 / "artifacts/r006b_rescue/results.json").read_text())
    integrity = {
        "status": "LEAK_FOUND",
        "stop_condition": "R006B M selected by labeled fold-0 blocker oracle; complete fold-0 inference prohibited by task contract",
        "official_split": {"path": str(SPLIT.relative_to(ROOT)), "sha256": sha256(SPLIT)},
        "fold0": {"s1": len(fold0), "evaluation_10k_s1": len(eval10),
                  "r007_evaluation_ids_identical_to_e001_10k": evaluation_ids_identical},
        "lightgbm_training": {
            **lgb, "unknown_fold_s1": int(lgb_fold_unknown),
            "exact_positive_pair_overlap_with_fold0": int(lgb_pos_overlap),
            "input_glob": str(Path(lgb_glob).relative_to(ROOT)),
        },
        "xlmr_training": {
            **xlm, "unknown_fold_s1": int(xlm_fold_unknown),
            "exact_positive_pair_overlap_with_fold0": int(xlm_pos_overlap),
            "input": str(xlm_path.relative_to(ROOT)), "sha256": sha256(xlm_path),
        },
        "calibration_prediction_fold_counts": calibration,
        "fusion_threshold_margin": {
            "fit_population": "fold4 only",
            "fold0_direct_search": False,
            "artifact": str((E005 / "outputs/calibration_search_epoch2.csv").relative_to(ROOT)),
            "artifact_sha256": sha256(E005 / "outputs/calibration_search_epoch2.csv"),
        },
        "r006b_label_leakage": {
            "selection_code": str((R006 / "run_r006b.py").relative_to(ROOT)),
            "selection_rule": "max labeled fold0 oracle_grouped_macro_f05 over M=[200,400,800]",
            "cohort_n": r006_results["correctness"]["cohort_n"],
            "fold0_only": r006_results["correctness"]["fold0_only"],
            "trials": [
                {"m": x["m"], "fold0_oracle_f05": x["oracle_grouped_macro_f05"],
                 "fold0_true_edges_retained": x["true_edges_retained"]}
                for x in r006_results["m_scaling"]
            ],
            "selected_m": r006_results["best_m"],
            "result_artifact": str((R006 / "artifacts/r006b_rescue/results.json").relative_to(ROOT)),
            "result_sha256": sha256(R006 / "artifacts/r006b_rescue/results.json"),
        },
        "provenance_uncertainty": [
            "No immutable git history is available: .git is not a normal repository, so execution order is supported by scripts/artifacts and hashes rather than commits.",
            "The LightGBM joblib does not embed training row IDs; provenance is reconstructed from the exact loader and extant train feature shards.",
            "The XLM-R checkpoint does not embed pair IDs; provenance is reconstructed from the run script, bundle manifest, and hashed realized train_pairs.parquet.",
        ],
    }
    (OUT / "integrity_audit.json").write_text(json.dumps(integrity, indent=2, default=int) + "\n")

    summary = {
        "status": "STOPPED_LEAK_FOUND",
        "full_fold0_inference_started": False,
        "test_inference_rerun": False,
        "model_fit_or_tuning": False,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "runtime.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "runtime_logs/stop.log").write_text(
        "LEAK_FOUND: R006B M was selected by fold-0 labeled oracle. "
        "Stopped before complete fold-0 inference as explicitly required.\n"
    )
    print(json.dumps({"integrity": integrity, "runtime": summary}, indent=2, default=int))


if __name__ == "__main__":
    main()
