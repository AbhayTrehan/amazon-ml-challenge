from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/experiments_v1/e008_final_robust_matcher"
P003 = ROOT / "analysis/production/p003_e008"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
ASSIGNMENT = ROOT / "analysis/experiments_v1/e001_classical_baseline/cache/auxiliary_fold_assignment.parquet"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
P002 = ROOT / "analysis/production/p002_final"
P001 = ROOT / "analysis/production/p001_e005b"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
VENV = Path("/home/satvik/venvs/amazonml2026/bin/python")
SEED = 20260925
MODEL_ID = "FacebookAI/xlm-roberta-base"
MODEL_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
MAX_LENGTH = 160
KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]
LEGAL = {"llc", "llp", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "pvt", "private", "plc", "lp", "sarl", "sas", "sa", "gmbh", "bv", "pte", "firm", "enterprises", "enterprise"}


def mkdirs():
    for x in ["cache/index", "cache/blocker", "cache/features", "cache/neural", "cache/shortlist", "data", "model/lightgbm", "model/xlmr", "calibration", "validation", "logs", "runtime", "freeze"]:
        (EXP / x).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def stable_hash(text: str, seed: str) -> int:
    return int.from_bytes(hashlib.blake2b((seed + "\0" + str(text)).encode(), digest_size=8).digest(), "big")


def normalize_existing(text) -> str:
    """Exact logical equivalent of audit DuckDB lower + punctuation/space normalization."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(re.sub(r"[^\w]+", " ", str(text).lower(), flags=re.UNICODE).split())


def numeric_tokens(text) -> str:
    return " ".join(x for x in normalize_existing(text).split() if x.isdigit())


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def grouped_metrics(frame: pd.DataFrame, selected: pd.DataFrame, mask) -> dict:
    ids = selected.source1_entity_id.to_numpy()
    idx = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
    take = np.asarray(mask, dtype=bool)
    gi = idx.loc[frame.source1_entity_id].to_numpy()
    pred = np.bincount(gi[take], minlength=len(ids))
    tp = np.bincount(gi[take], weights=frame.label.to_numpy()[take], minlength=len(ids))
    true = selected.true_matches.to_numpy(dtype=np.float64)
    den = 1.25 * tp + .25 * (true - tp) + (pred - tp)
    f = np.divide(1.25 * tp, den, out=np.zeros_like(den), where=den > 0)
    f[(true == 0) & (pred == 0)] = 1.0
    return {"macro_f05": float(f.mean()), "precision": float(tp.sum()/pred.sum()) if pred.sum() else 1.0,
            "recall": float(tp.sum()/true.sum()) if true.sum() else 1.0,
            "singleton_accuracy": float(np.mean(pred[true == 0] == 0)) if np.any(true == 0) else None,
            "tp": int(tp.sum()), "predicted_edges": int(pred.sum()), "true_edges": int(true.sum())}


def expected_utility_decode(frame: pd.DataFrame, probability="effective_probability") -> np.ndarray:
    out = np.zeros(len(frame), dtype=bool)
    for _, positions in frame.groupby("source1_entity_id", sort=False).indices.items():
        ix = np.asarray(positions, dtype=np.int64); p = frame.iloc[ix][probability].to_numpy(float)
        order = np.lexsort((frame.iloc[ix]["candidate_entity_id"].astype(str).to_numpy(), -p)); so = ix[order]; ps = p[order]
        total = ps.sum(); utilities = np.r_[np.prod(1 - ps), 1.25 * np.cumsum(ps) / (np.arange(1, len(ps)+1) + .25 * total)]
        k = int(np.flatnonzero(utilities == utilities.max())[0])
        out[so[:k]] = True
    return out
