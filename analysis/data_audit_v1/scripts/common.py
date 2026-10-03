#!/usr/bin/env python3
"""Shared, dependency-free helpers for the ML Challenge 2026 audit."""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

SEED = 20260925
SENTINELS = {"na", "n/a", "null", "none", "nil", "-", "--", "unknown"}
LEGAL_SUFFIXES = {
    "llc", "llp", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "pvt", "private", "plc", "lp", "sarl", "sas", "sa",
    "gmbh", "bv", "pte", "firm", "enterprises", "enterprise",
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
NUM_RE = re.compile(r"\d+")


def nfkc(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def punct_norm(text: str) -> str:
    s = nfkc(text)
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(punct_norm(text))


def token_norm(text: str) -> str:
    return " ".join(tokens(text))


def suffix_norm(text: str) -> str:
    ts = tokens(text)
    while ts and ts[-1] in LEGAL_SUFFIXES:
        ts.pop()
    return " ".join(ts)


def accent_fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", nfkc(text))
                   if unicodedata.category(c) != "Mn")


def numeric_suffix(entity_id: str):
    try:
        return int(entity_id.split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def quantiles_from_counter(hist: Counter, probs=(0, .01, .05, .5, .95, .99, 1)):
    n = sum(hist.values())
    if not n:
        return {str(p): None for p in probs}
    out = {}
    ordered = sorted(hist.items())
    for p in probs:
        rank = max(0, min(n - 1, math.ceil(p * n) - 1))
        cum = 0
        for value, count in ordered:
            cum += count
            if cum > rank:
                out[str(p)] = value
                break
    return out


def stable_bucket(value: str, modulo: int = 1_000_000) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") % modulo


def script_bucket(text: str) -> str:
    scripts = Counter()
    for ch in text:
        if not ch.isalpha() or ord(ch) < 128:
            continue
        name = unicodedata.name(ch, "")
        key = next((x for x in ("DEVANAGARI", "LATIN", "CYRILLIC", "ARABIC", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "GURMUKHI") if x in name), "OTHER")
        scripts[key] += 1
    return scripts.most_common(1)[0][0] if scripts else "ASCII"


def levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        cur = [j]
        for i, ca in enumerate(a, 1):
            cur.append(min(cur[-1] + 1, prev[i] + 1, prev[i-1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def jaccard(a: str, b: str) -> float:
    x, y = set(tokens(a)), set(tokens(b))
    return len(x & y) / len(x | y) if x or y else 1.0


def dice_ngrams(a: str, b: str, n: int = 3) -> float:
    def grams(s):
        s = punct_norm(s)
        return Counter(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    x, y = grams(a), grams(b)
    den = sum(x.values()) + sum(y.values())
    return 2 * sum((x & y).values()) / den if den else 1.0


def fbeta(true_ids, pred_ids, beta=.5) -> float:
    true_ids, pred_ids = set(true_ids), set(pred_ids)
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * (len(true_ids)-tp) + (len(pred_ids)-tp))
