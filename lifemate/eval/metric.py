"""LifeMate scorer: the official per-sample formula (hash gate, then F1 by maximum one-to-one call matching).

This is the ONLY place call matching is implemented. It is pure: calls, hashes and the fuzzy-key map go in, a score
comes out. Three variants cover what the official description leaves open (final-plan §4.4):
  primary      fuzzy tokens = NFC + lower + [^\\W_]+ (diacritics kept, multiset); non-fuzzy compared strictly
  pessimistic  fuzzy tokens = NFC + lower + whitespace split (multiset); strict
  optimistic   fuzzy tokens = zaic_tools-style folded set; lenient non-fuzzy comparison (upper bound)
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from fractions import Fraction

FUZZY_MIN = Fraction(3, 5)  # a fuzzy argument matches only when token F1 > 0.6 (strictly)
EMPTY = (None, "", [], {})
VARIANTS = {"primary": ("word", False), "pessimistic": ("space", False), "optimistic": ("folded", True)}
ASK = "sys.ask_missing_info"


def fuzzy_keys(catalog: dict) -> dict[str, frozenset[str]]:
    """tool name -> top-level argument names marked x-match: fuzzy in this split's catalog"""
    return {t["name"]: frozenset(k for k, p in t["parameters"].get("properties", {}).items() if p.get("x-match") == "fuzzy")
            for app in catalog["apps"] for t in app["tools"]}


def _fold(s: str) -> str:
    """zaic_tools.utils._fold: lower-case, diacritics removed, đ -> d"""
    t = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def tokens(text: str, mode: str) -> Counter:
    s = unicodedata.normalize("NFC", text).lower()
    if mode == "word":
        return Counter(re.findall(r"[^\W_]+", s))
    if mode == "space":
        return Counter(s.split())
    if mode == "folded":
        return Counter(set(re.findall(r"[^\W_]+", _fold(s))))
    raise ValueError(mode)


def token_f1(pred: str, gold: str, mode: str) -> Fraction:
    p, g = tokens(pred, mode), tokens(gold, mode)
    n = sum(p.values()) + sum(g.values())
    return Fraction(1) if n == 0 else Fraction(2 * sum((p & g).values()), n)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _order(x):
    return (type(x).__name__, _norm(str(x)))


def same(a, b, lenient: bool) -> bool:
    """non-fuzzy identity, type-aware: True is never 1; 2 vs 2.0 differ unless lenient"""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b and (lenient or type(a) is type(b))
    if isinstance(a, str) and isinstance(b, str):
        return a == b or (lenient and _norm(a) == _norm(b))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        if lenient and not any(isinstance(x, (dict, list)) for x in a + b):
            a, b = sorted(a, key=_order), sorted(b, key=_order)
        return all(same(x, y, lenient) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], lenient) for k in a)
    return a is None and b is None


def _strip_empty(args: dict) -> dict:
    return {k: v for k, v in args.items() if v not in EMPTY}


def call_match(s: dict, g: dict, fuzzy: dict[str, frozenset[str]], mode: str, lenient: bool) -> bool:
    if s.get("name") != g.get("name"):
        return False
    sa, ga = s.get("arguments") or {}, g.get("arguments") or {}
    if lenient:
        sa, ga = _strip_empty(sa), _strip_empty(ga)
    if sa.keys() != ga.keys():
        return False
    fz = fuzzy.get(g["name"], frozenset())
    for k, gv in ga.items():
        sv = sa[k]
        if k in fz and isinstance(sv, str) and isinstance(gv, str):
            if not token_f1(sv, gv, mode) > FUZZY_MIN:
                return False
        elif not same(sv, gv, lenient):
            return False
    return True


def max_matching(sub: list, gold: list, ok) -> list[int | None]:
    """Maximum one-to-one matching (Kuhn's augmenting paths). Returns owner[j] = index of the submitted call matched
    to gold call j, or None."""
    adj = [[j for j, g in enumerate(gold) if ok(s, g)] for s in sub]
    owner: list[int | None] = [None] * len(gold)

    def augment(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] is None or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(len(sub)):
        augment(i, set())
    return owner


def score_sample(sub_calls: list, sub_hash: str, gold_calls: list, gold_hash: str,
                 fuzzy: dict[str, frozenset[str]], variant: str = "primary") -> Fraction:
    return explain_sample(sub_calls, sub_hash, gold_calls, gold_hash, fuzzy, variant)["score"]


def gold_type(gold_calls: list, kinds: dict[str, str] | None = None) -> str:
    """empty / ask / write / read-only. `kinds`: tool name -> catalog kind (read/write/meta)."""
    if not gold_calls:
        return "empty"
    if gold_calls[-1]["name"] == ASK:
        return "ask"
    if kinds is not None:
        return "write" if any(kinds.get(c["name"]) == "write" for c in gold_calls) else "read-only"
    return "act"


def explain_sample(sub_calls: list, sub_hash: str, gold_calls: list, gold_hash: str,
                   fuzzy: dict[str, frozenset[str]], variant: str = "primary", n_rejected: int = 0,
                   kinds: dict[str, str] | None = None) -> dict:
    """The score plus the diagnostic record that reports and error analysis read."""
    mode, lenient = VARIANTS[variant]
    rec = {"variant": variant, "hash_ok": sub_hash == gold_hash, "n_sub": len(sub_calls), "n_gold": len(gold_calls),
           "n_rejected": n_rejected, "gold_type": gold_type(gold_calls, kinds), "m": 0,
           "unmatched_sub": [], "unmatched_gold": []}
    if not gold_calls:
        rec["score"] = Fraction(int(not sub_calls))
        rec["unmatched_sub"] = list(sub_calls)
        return rec
    owner = max_matching(sub_calls, gold_calls, lambda s, g: call_match(s, g, fuzzy, mode, lenient))
    m = sum(o is not None for o in owner)
    matched_sub = {o for o in owner if o is not None}
    rec["m"] = m
    rec["unmatched_sub"] = [c for i, c in enumerate(sub_calls) if i not in matched_sub]
    rec["unmatched_gold"] = [g for j, g in enumerate(gold_calls) if owner[j] is None]
    # = 2PR/(P+R) with P = M/|S|, R = M/|G|; 0 when M == 0; gated by the state hash
    rec["score"] = Fraction(0) if not rec["hash_ok"] else Fraction(2 * m, len(sub_calls) + len(gold_calls))
    return rec


def final_score(per_sample: list[Fraction]) -> float:
    return float(100 * sum(per_sample, Fraction(0)) / len(per_sample))
