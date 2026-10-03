#!/usr/bin/env python3
"""Parity-preserving optimized implementation of the frozen E001 features."""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler


LEGAL = {
    "llc", "llp", "inc", "incorporated", "corp", "corporation", "co",
    "company", "ltd", "limited", "pvt", "private", "plc", "lp", "sarl",
    "sas", "sa", "gmbh", "bv", "pte", "firm", "enterprises", "enterprise",
}


@dataclass(frozen=True)
class TextViews:
    text: np.ndarray
    tokens: tuple[tuple[str, ...], ...]
    token_sets: tuple[frozenset[str], ...]
    numeric_sets: tuple[frozenset[str], ...]
    suffixes: np.ndarray
    accents: np.ndarray
    lengths: np.ndarray
    token_counts: np.ndarray


def _text_array(values) -> np.ndarray:
    # Frozen inputs are strings after fillna; preserve the old str() fallback.
    return np.asarray([str(x) for x in values], dtype=object)


def precompute(values, decorations: bool = True) -> TextViews:
    text = _text_array(values)
    toks = tuple(tuple(x.split()) if x else tuple() for x in text)
    sets = tuple(frozenset(x) for x in toks)
    nums = tuple(frozenset(z for z in x if z.isdigit()) for x in toks)
    suffixes = []
    if decorations:
        for x in toks:
            end = len(x)
            while end and x[end - 1] in LEGAL:
                end -= 1
            suffixes.append(" ".join(x[:end]))
        accents = np.asarray([
            "".join(c for c in unicodedata.normalize("NFKD", x)
                    if unicodedata.category(c) != "Mn")
            for x in text
        ], dtype=object)
    else:
        suffixes = [""] * len(text)
        accents = np.full(len(text), "", dtype=object)
    return TextViews(
        text=text,
        tokens=toks,
        token_sets=sets,
        numeric_sets=nums,
        suffixes=np.asarray(suffixes, dtype=object),
        accents=accents,
        lengths=np.fromiter((len(x) for x in text), np.int32, len(text)),
        token_counts=np.fromiter((len(x) for x in toks), np.int16, len(toks)),
    )


def _jaccard(left, right, li, ri) -> np.ndarray:
    def one(i, j):
        x, y = left[i], right[j]
        return len(x & y) / len(x | y) if x or y else 1.0
    return np.fromiter((one(i, j) for i, j in zip(li, ri)), np.float32, len(li))


def _rapid(left, right, scorer, workers: int, scale: float = 0.01) -> np.ndarray:
    # cpdist applies the same scorer to corresponding pairs. float32 output is
    # identical to the historical per-pair result after its explicit cast.
    out = process.cpdist(left, right, scorer=scorer, dtype=np.float32, workers=workers)
    if scale != 1.0:
        out *= np.float32(scale)
    return out


def add_context(cand: pd.DataFrame) -> pd.DataFrame:
    """Exact NumPy equivalent of the frozen pandas group transforms."""
    cand = cand.copy()
    cand["retrieval_channels"] = cand.name_rank.notna().astype(np.int8) + cand.address_rank.notna().astype(np.int8)
    cand["retrieved_by_name"] = cand.name_rank.notna().astype(np.int8)
    cand["retrieved_by_address"] = cand.address_rank.notna().astype(np.int8)
    cand["name_rank"] = cand.name_rank.fillna(99).astype(np.int16)
    cand["address_rank"] = cand.address_rank.fillna(99).astype(np.int16)
    cand["name_retrieval_score"] = cand.name_retrieval_score.fillna(0).astype(np.float32)
    cand["address_retrieval_score"] = cand.address_retrieval_score.fillna(0).astype(np.float32)
    cand["reciprocal_name_rank"] = np.where(cand.name_rank < 99, 1 / cand.name_rank, 0).astype(np.float32)
    cand["reciprocal_address_rank"] = np.where(cand.address_rank < 99, 1 / cand.address_rank, 0).astype(np.float32)
    cand["fused_max_score"] = cand[["name_char_tfidf", "address_char_tfidf"]].max(axis=1).astype(np.float32)
    cand["fused_sum_score"] = cand[["name_char_tfidf", "address_char_tfidf"]].sum(axis=1).astype(np.float32)
    qpos = cand.qpos.to_numpy(dtype=np.intp, copy=False)
    ngroups = int(qpos.max()) + 1 if len(qpos) else 0
    counts = np.bincount(qpos, minlength=ngroups)
    cand["candidate_count_k20_source"] = counts[qpos].astype(np.int16)
    for col, prefix in [("name_char_tfidf", "name"), ("address_char_tfidf", "address"), ("fused_max_score", "fused")]:
        values = cand[col].to_numpy(dtype=np.float32, copy=False)
        top = np.full(ngroups, -np.inf, np.float32)
        np.maximum.at(top, qpos, values)
        top_counts = np.bincount(qpos, weights=(values == top[qpos]), minlength=ngroups)
        second_less = np.zeros(ngroups, np.float32)
        mask = values < top[qpos]
        np.maximum.at(second_less, qpos[mask], values[mask])
        second = np.where(top_counts >= 2, top, second_less)
        cand[f"query_{prefix}_top12_margin"] = (top[qpos] - second[qpos]).astype(np.float32)
    topf = np.full(ngroups, -np.inf, np.float32)
    fused = cand.fused_max_score.to_numpy(dtype=np.float32, copy=False)
    np.maximum.at(topf, qpos, fused)
    near = np.bincount(qpos, weights=fused >= topf[qpos] - np.float32(.02), minlength=ngroups)
    cand["near_tie_count_002"] = near[qpos].astype(np.int16)
    cand["in_k10"] = ((cand.name_rank <= 10) | (cand.address_rank <= 10)).astype(np.int8)
    return cand


def build_features(cand: pd.DataFrame, q: pd.DataFrame, aux: pd.DataFrame,
                   workers: int | None = None) -> pd.DataFrame:
    """Return the historical frame plus all 49 frozen features, unchanged."""
    workers = int(workers or os.environ.get("P002_FEATURE_THREADS", "20"))
    qi0 = cand.qpos.to_numpy(dtype=np.intp, copy=False)
    ai0 = cand.apos.to_numpy(dtype=np.intp, copy=False)
    qu, qi = np.unique(qi0, return_inverse=True)
    au, ai = np.unique(ai0, return_inverse=True)

    qn = precompute(q.nname.fillna("").to_numpy()[qu])
    qa = precompute(q.naddr.fillna("").to_numpy()[qu], decorations=False)
    an = precompute(aux.nname.fillna("").to_numpy()[au])
    aa = precompute(aux.naddr.fillna("").to_numpy()[au], decorations=False)

    qnt, ant = qn.text[qi], an.text[ai]
    qat, aat = qa.text[qi], aa.text[ai]
    out = cand.copy()
    qs, ass = qn.suffixes[qi], an.suffixes[ai]
    qacc, aacc = qn.accents[qi], an.accents[ai]
    out["legal_suffix_name_exact"] = ((qs != "") & (qs == ass)).astype(np.int8)
    out["accent_name_exact"] = ((qacc != "") & (qacc == aacc)).astype(np.int8)
    out["name_token_jaccard"] = _jaccard(qn.token_sets, an.token_sets, qi, ai)
    out["address_token_jaccard"] = _jaccard(qa.token_sets, aa.token_sets, qi, ai)

    qnl, anl = qn.lengths[qi], an.lengths[ai]
    qal, aal = qa.lengths[qi], aa.lengths[ai]
    nmax, amax = np.maximum(qnl, anl), np.maximum(qal, aal)
    out["name_length_ratio"] = np.divide(np.minimum(qnl, anl), nmax,
        out=np.ones(len(out), np.float32), where=nmax != 0).astype(np.float32)
    out["address_length_ratio"] = np.divide(np.minimum(qal, aal), amax,
        out=np.ones(len(out), np.float32), where=amax != 0).astype(np.float32)
    out["name_length_diff"] = np.abs(qnl - anl).astype(np.int16)
    out["address_length_diff"] = np.abs(qal - aal).astype(np.int16)
    out["name_token_count_diff"] = np.abs(qn.token_counts[qi] - an.token_counts[ai]).astype(np.int16)
    out["address_token_count_diff"] = np.abs(qa.token_counts[qi] - aa.token_counts[ai]).astype(np.int16)

    out["name_rf_ratio"] = _rapid(qnt, ant, fuzz.ratio, workers)
    out["name_rf_partial_ratio"] = _rapid(qnt, ant, fuzz.partial_ratio, workers)
    out["name_rf_token_set"] = _rapid(qnt, ant, fuzz.token_set_ratio, workers)
    out["name_rf_token_sort"] = _rapid(qnt, ant, fuzz.token_sort_ratio, workers)
    out["name_jaro_winkler"] = _rapid(qnt, ant, JaroWinkler.normalized_similarity, workers, 1.0)
    out["address_rf_ratio"] = _rapid(qat, aat, fuzz.ratio, workers)
    out["address_rf_token_set"] = _rapid(qat, aat, fuzz.token_set_ratio, workers)

    # Keep the historical set semantics, but reuse entity-level numeric sets.
    numeric_rows = ((qa.numeric_sets[i], aa.numeric_sets[j]) for i, j in zip(qi, ai))
    nj = np.empty(len(out), np.float32)
    ni = np.empty(len(out), np.int16)
    nq = np.empty(len(out), np.int16)
    na = np.empty(len(out), np.int16)
    nc = np.empty(len(out), np.int8)
    for k, (x, y) in enumerate(numeric_rows):
        inter = len(x & y); union = len(x | y)
        nj[k] = inter / union if union else 1.0
        ni[k] = inter; nq[k] = len(x); na[k] = len(y)
        nc[k] = int(bool(x) and bool(y) and inter == 0)
    out["numeric_token_jaccard"] = nj
    out["numeric_exact_overlap_count"] = ni
    out["s1_numeric_token_count"] = nq
    out["candidate_numeric_token_count"] = na
    out["obvious_numeric_conflict"] = nc

    qraw_addr = q.business_address.fillna("").to_numpy(dtype=object)[qu][qi]
    araw_addr = aux.business_address.fillna("").to_numpy(dtype=object)[au][ai]
    out["s1_address_missing"] = np.fromiter((not bool(x) for x in qraw_addr), np.int8, len(out))
    out["candidate_address_missing"] = np.fromiter((not bool(x) for x in araw_addr), np.int8, len(out))
    out["source3"] = out.source.eq(3).astype(np.int8)
    out.drop(columns=["qpos", "apos", "_priority"], errors="ignore", inplace=True)
    return out
