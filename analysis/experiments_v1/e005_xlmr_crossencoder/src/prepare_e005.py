#!/usr/bin/env python3
"""Local integrity checks and metadata preparation; never touches challenge sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from common import shortlist
from metric import grouped_f05, self_test

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
BUNDLE = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/gpu_bundle"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    self_test()
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    required = set(next(iter(manifest["files"].values()))["columns_dtypes"])
    report = {"files": {}, "checksums_ok": True, "schema_ok": True, "labels_ok": True}
    roles = {}
    frames = {}
    for name, spec in manifest["files"].items():
        path = BUNDLE / name
        digest = sha256(path)
        d = pd.read_parquet(path)
        item = {
            "sha256": digest,
            "expected_sha256": spec["sha256"],
            "rows": len(d),
            "positives": int(d.label.sum()),
            "negatives": int((d.label == 0).sum()),
            "unique_s1": int(d.source1_entity_id.nunique()),
            "columns": list(d.columns),
            "role_values": sorted(map(str, d.role.unique())),
        }
        item["schema_ok"] = required.issubset(d.columns)
        item["labels_ok"] = set(d.label.unique()).issubset({0, 1})
        item["ok"] = (digest == spec["sha256"] and len(d) == spec["rows"] and
                      item["positives"] == spec["positives"] and item["negatives"] == spec["negatives"] and
                      item["schema_ok"] and item["labels_ok"])
        report["files"][name] = item
        report["checksums_ok"] &= item["ok"]
        report["schema_ok"] &= item["schema_ok"]
        report["labels_ok"] &= item["labels_ok"]
        roles[name.split("_")[0]] = set(d.source1_entity_id.unique())
        frames[name.split("_")[0]] = d
    if not report["checksums_ok"]:
        raise RuntimeError("GPU bundle integrity check failed")
    report["s1_pairwise_disjoint"] = {
        "train_calibration": roles["train"].isdisjoint(roles["calibration"]),
        "train_evaluation": roles["train"].isdisjoint(roles["evaluation"]),
        "calibration_evaluation": roles["calibration"].isdisjoint(roles["evaluation"]),
    }
    if not all(report["s1_pairwise_disjoint"].values()):
        raise RuntimeError("train/calibration/evaluation S1 overlap")

    selected = pd.read_parquet(E001 / "selected_ids/selected_s1.parquet")
    # Name ambiguity is sourced from already-created E001 pair metadata, not raw data.
    ambiguity = {}
    for role in ["calibration", "evaluation"]:
        pairs = pd.read_parquet(E001 / f"predictions/{role}_pair_scores.parquet",
                                columns=["s1", "s1_normalized_name_fanout"])
        ambiguity[role] = (pairs.groupby("s1").s1_normalized_name_fanout.max() > 1)
    meta = selected[selected.role.isin(["calibration", "evaluation"])].copy()
    meta["repeated_name"] = False
    for role, values in ambiguity.items():
        take = meta.role.eq(role)
        meta.loc[take, "repeated_name"] = meta.loc[take, "source1_entity_id"].map(values).fillna(False).astype(bool)
    meta.to_parquet(EXP / "runtime/selected_s1_metadata.parquet", index=False, compression="zstd")

    # Materialize each neural shortlist once and validate its ceiling plus the fair
    # E004B-on-the-same-shortlist control. Fold 4 is processed before fold 0.
    controls = {}
    for role, expected_rows, expected_oracle in [
        ("calibration", 61621, 0.987205), ("evaluation", 61913, 0.986696)
    ]:
        d = frames[role]
        kept = shortlist(d)
        kept.to_parquet(EXP / f"runtime/{role}_shortlist.parquet", index=False, compression="zstd")
        selected_role = meta[meta.role.eq(role)]
        oracle = grouped_f05(kept, selected_role, kept.label.eq(1).to_numpy())

        def e004b_mask(frame):
            x = frame[["candidate_entity_id", "source1_entity_id", "lightgbm_score"]].copy()
            x["_pos"] = range(len(x))
            x = x.sort_values(["candidate_entity_id", "lightgbm_score", "source1_entity_id"],
                              ascending=[True, False, True], kind="mergesort")
            x["_rank"] = x.groupby("candidate_entity_id", sort=False).cumcount()
            win = x[x._rank.eq(0)].copy()
            second = x[x._rank.eq(1)].set_index("candidate_entity_id").lightgbm_score
            win["_margin"] = win.lightgbm_score - win.candidate_entity_id.map(second).fillna(0)
            positions = win[win.lightgbm_score.ge(.780) & win._margin.ge(.485)]._pos.to_numpy()
            mask = pd.Series(False, index=range(len(frame)))
            mask.iloc[positions] = True
            return mask.to_numpy()

        classical = grouped_f05(kept, selected_role, e004b_mask(kept))
        controls[role] = {
            "rows": len(kept), "mean_pairs_per_s1": len(kept) / len(selected_role),
            "oracle": oracle, "e004b_shortlist": classical,
        }
        if len(kept) != expected_rows or abs(oracle["macro_f05"] - expected_oracle) > 1e-6:
            raise RuntimeError(f"{role} shortlist inconsistent with E004: {controls[role]}")
    (EXP / "runtime/shortlist_control.json").write_text(json.dumps(controls, indent=2) + "\n")

    train = pd.read_parquet(BUNDLE / "train_pairs.parquet", columns=["label", "country", "candidate_source"])
    dist = {
        "total": len(train),
        "label": {str(k): int(v) for k, v in train.label.value_counts().sort_index().items()},
        "country": {str(k): int(v) for k, v in train.country.value_counts().sort_index().items()},
        "candidate_source": {str(k): int(v) for k, v in train.candidate_source.value_counts().sort_index().items()},
        "cross": [dict(zip(["country", "candidate_source", "label", "rows"], row))
                  for row in train.groupby(["country", "candidate_source", "label"]).size().reset_index(name="rows").itertuples(index=False, name=None)],
    }
    (EXP / "runtime/training_distribution.json").write_text(json.dumps(dist, indent=2) + "\n")
    (EXP / "runtime/integrity_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"integrity": report, "shortlist_control": controls,
                      "training_distribution": dist}, indent=2))


if __name__ == "__main__":
    main()
