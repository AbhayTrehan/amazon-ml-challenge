"""Exact S1-macro F0.5 used by E001/E004, including zero-prediction S1s."""
from __future__ import annotations

import numpy as np
import pandas as pd


def grouped_f05(frame: pd.DataFrame, selected: pd.DataFrame, mask, *, true_col="true_matches") -> dict:
    s1_col = "source1_entity_id"
    ids = selected[s1_col].astype(str).to_numpy()
    if len(ids) != len(set(ids)):
        raise ValueError("selected S1 IDs are not unique")
    index = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
    row_ids = frame[s1_col].astype(str)
    if not row_ids.isin(index.index).all():
        raise ValueError("pair frame contains S1 outside selected universe")
    gi = index.loc[row_ids].to_numpy()
    use = np.asarray(mask, dtype=bool)
    if len(use) != len(frame):
        raise ValueError("mask length mismatch")
    labels = frame.label.to_numpy(dtype=np.int8)
    pred = np.bincount(gi[use], minlength=len(ids))
    tp = np.bincount(gi[use], weights=labels[use], minlength=len(ids))
    true = selected[true_col].to_numpy(dtype=np.float64)
    if np.any(tp > true):
        raise ValueError("true positives exceed selected truth counts")
    den = 1.25 * tp + 0.25 * (true - tp) + (pred - tp)
    scores = np.divide(1.25 * tp, den, out=np.zeros_like(den), where=den > 0)
    scores[(true == 0) & (pred == 0)] = 1.0
    return {
        "macro_f05": float(scores.mean()),
        "precision": float(tp.sum() / pred.sum()) if pred.sum() else 1.0,
        "recall": float(tp.sum() / true.sum()) if true.sum() else 1.0,
        "singleton_accuracy": float(np.mean(pred[true == 0] == 0)) if np.any(true == 0) else float("nan"),
        "predicted_edges": int(pred.sum()),
        "true_edges": int(true.sum()),
        "tp": int(tp.sum()),
    }


def self_test() -> None:
    selected = pd.DataFrame({
        "source1_entity_id": ["a", "b", "c", "d"],
        "true_matches": [0, 1, 2, 0],
    })
    pairs = pd.DataFrame({
        "source1_entity_id": ["a", "b", "b", "c", "c", "c"],
        "label": [0, 1, 0, 1, 1, 0],
    })
    # a correct singleton=1; b exact=1; c one of two truth => 1.25/(1.25+.25)=5/6;
    # d has no pair row but is still a correct singleton=1.
    got = grouped_f05(pairs, selected, [False, True, False, True, False, False])
    expected = (1 + 1 + 5 / 6 + 1) / 4
    assert abs(got["macro_f05"] - expected) < 1e-15, (got, expected)
    assert got["tp"] == 2 and got["predicted_edges"] == 2


if __name__ == "__main__":
    self_test()
    print("metric self-test: OK")

