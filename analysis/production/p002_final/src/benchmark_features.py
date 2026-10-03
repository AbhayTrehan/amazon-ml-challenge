#!/usr/bin/env python3
"""Profile and parity-check the optimized frozen 49-feature builder on R007."""
from __future__ import annotations

import cProfile
import hashlib
import io
import json
import pstats
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "diagnostics"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
R007 = ROOT / "analysis/experiments_v1/r007_fast_blocker_e005b"
sys.path.insert(0, str(E001 / "scripts"))
sys.path.insert(0, str(R006))
sys.path.insert(0, str(R007))
import prepare_baseline as historical  # noqa: E402
import run_r006a  # noqa: E402
import run_r007  # noqa: E402
from feature_builder import add_context, build_features  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def frames():
    selected = run_r006a.cohort(10000)
    cand = pd.read_parquet(R007 / "r006b_candidates.parquet")
    for (country, source), block in cand.groupby(["country", "source"], sort=False):
        sc = selected[selected.country.eq(country)].sort_values("source1_entity_id", kind="mergesort")
        yield country, int(source), block.reset_index(drop=True), run_r007.full_query(sc), run_r007.full_aux(country, int(source))


def profile_one(cand, q, aux):
    profiler = cProfile.Profile()
    profiler.enable(); start = time.perf_counter()
    old = historical.build_features(cand.copy(), q, aux)
    elapsed = time.perf_counter() - start; profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "historical_feature_profile.txt").write_text(stream.getvalue())
    return old, elapsed


def compare(old, new, features):
    discrete = [c for c in features if np.issubdtype(old[c].dtype, np.integer) or old[c].dtype == bool]
    numeric = [c for c in features if c not in discrete]
    exact_bad = {c: int(np.count_nonzero(old[c].to_numpy() != new[c].to_numpy())) for c in discrete}
    exact_bad = {k: v for k, v in exact_bad.items() if v}
    max_abs = {c: float(np.max(np.abs(old[c].to_numpy(dtype=np.float64) - new[c].to_numpy(dtype=np.float64)))) for c in numeric}
    return discrete, numeric, exact_bad, max_abs


def main():
    model = joblib.load(E001 / "model/lightgbm_model.joblib")
    features = json.loads((E001 / "model/model_config.json").read_text())["features"]
    groups = list(frames())
    context_rows = []
    context_old_seconds = context_new_seconds = 0.0
    for country, source, cand, _, _ in groups:
        # The saved R007 artifact already contains historical context columns.
        raw_cols = ["qpos", "apos", "name_rank", "name_retrieval_score", "address_rank", "address_retrieval_score",
                    "s1", "target_id", "source", "country", "role", "name_char_tfidf", "address_char_tfidf",
                    "s1_normalized_name_fanout", "s1_normalized_address_fanout", "candidate_normalized_name_fanout",
                    "candidate_normalized_address_fanout", "nname_exact", "naddr_exact", "label"]
        raw = cand[raw_cols].copy()
        for channel in ["name", "address"]:
            missing = raw[f"{channel}_rank"].eq(99)
            raw.loc[missing, f"{channel}_rank"] = np.nan
            raw.loc[missing, f"{channel}_retrieval_score"] = np.nan
        t = time.perf_counter(); old_ctx = historical.add_context(raw.copy()); context_old_seconds += time.perf_counter() - t
        t = time.perf_counter(); new_ctx = add_context(raw.copy()); context_new_seconds += time.perf_counter() - t
        cols = [c for c in old_ctx if c not in raw_cols]
        bad = {c: float(np.max(np.abs(cand[c].to_numpy(dtype=float) - new_ctx[c].to_numpy(dtype=float)))) for c in cols}
        context_rows.append({"country": country, "source": source, "rows": len(raw), "max_abs": max(bad.values()),
                             "bad_columns": {k: v for k, v in bad.items() if v > 1e-7}})
    first = groups[0]
    sample = first[2].iloc[:50000].copy()
    old_sample, old_sample_s = profile_one(sample, first[3], first[4])
    t = time.perf_counter(); new_sample = build_features(sample.copy(), first[3], first[4]); new_sample_s = time.perf_counter() - t
    discrete, numeric, exact_bad, max_abs = compare(old_sample, new_sample, features)
    old_pred = model.predict_proba(old_sample[features])[:, 1]
    new_pred = model.predict_proba(new_sample[features])[:, 1]
    sample_report = {
        "rows": len(sample), "historical_seconds": old_sample_s, "optimized_seconds": new_sample_s,
        "speedup": old_sample_s / new_sample_s, "feature_order_exact": list(old_sample[features]) == features == list(new_sample[features]),
        "discrete_columns": discrete, "numeric_columns": numeric, "discrete_mismatches": exact_bad,
        "numeric_max_abs": max_abs, "numeric_global_max_abs": max(max_abs.values()),
        "lightgbm_max_abs": float(np.max(np.abs(old_pred - new_pred))),
    }
    sample_report["ok"] = (not exact_bad and sample_report["numeric_global_max_abs"] <= 1e-6
                           and sample_report["lightgbm_max_abs"] <= 1e-7)
    if not sample_report["ok"]:
        atomic_json(OUT / "feature_parity.json", {"sample": sample_report})
        raise RuntimeError(f"50k parity failed: {sample_report}")

    full_old, full_new = [], []
    old_seconds = new_seconds = 0.0
    for _, _, cand, q, aux in groups:
        t = time.perf_counter(); full_old.append(historical.build_features(cand.copy(), q, aux)); old_seconds += time.perf_counter() - t
        t = time.perf_counter(); full_new.append(build_features(cand.copy(), q, aux)); new_seconds += time.perf_counter() - t
    old = pd.concat(full_old, ignore_index=True); new = pd.concat(full_new, ignore_index=True)
    discrete, numeric, exact_bad, max_abs = compare(old, new, features)
    old_pred = model.predict_proba(old[features])[:, 1]
    new_pred = model.predict_proba(new[features])[:, 1]
    key = ["s1", "target_id", "source"]
    a = old[key].copy(); a["score"] = old_pred
    b = new[key].copy(); b["score"] = new_pred
    def shortlist(x):
        x = x.sort_values(["s1", "score", "target_id"], ascending=[True, False, True], kind="mergesort").copy()
        x["rank"] = x.groupby("s1", sort=False).cumcount() + 1
        return set(map(tuple, x.loc[x["rank"].le(5) | x.score.ge(.05), key].itertuples(index=False, name=None)))
    full_report = {
        "rows": len(old), "historical_seconds": old_seconds, "optimized_seconds": new_seconds,
        "speedup": old_seconds / new_seconds, "old_pairs_per_second": len(old) / old_seconds,
        "new_pairs_per_second": len(new) / new_seconds,
        "projected_full_minutes": new_seconds / len(new) * 65_626_000 / 60,
        "feature_order_exact": list(old[features]) == features == list(new[features]),
        "discrete_mismatches": exact_bad, "numeric_max_abs": max_abs,
        "numeric_global_max_abs": max(max_abs.values()),
        "lightgbm_max_abs": float(np.max(np.abs(old_pred - new_pred))),
        "shortlist_identity_exact": shortlist(a) == shortlist(b),
    }
    full_report["ok"] = (not exact_bad and full_report["numeric_global_max_abs"] <= 1e-6
                         and full_report["lightgbm_max_abs"] <= 1e-7 and full_report["shortlist_identity_exact"])
    report = {"context": {"historical_seconds": context_old_seconds, "optimized_seconds": context_new_seconds,
                           "speedup": context_old_seconds / context_new_seconds, "groups": context_rows,
                           "ok": all(not x["bad_columns"] for x in context_rows)},
              "sample_50000": sample_report, "full_r007": full_report,
              "historical_profile": str(OUT / "historical_feature_profile.txt")}
    atomic_json(OUT / "feature_parity.json", report)
    print(json.dumps(report, indent=2))
    if not full_report["ok"] or not report["context"]["ok"]: raise RuntimeError("complete R007 parity failed")


if __name__ == "__main__":
    main()
