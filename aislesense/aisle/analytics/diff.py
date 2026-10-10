"""Before/after: compare the tips of two sessions of one site, matched by (kind, zone).

cleared = in A (before) but not in B (after), added = only in B, kept = in both (text from B).
"""
from __future__ import annotations


def compare(a_tips: list, b_tips: list) -> dict:
    key = lambda t: (t["kind"], t.get("zone"))  # noqa: E731
    A = {key(t): t for t in a_tips}
    B = {key(t): t for t in b_tips}
    item = lambda t: {"kind": t["kind"], "zone": t.get("zone"), "text": t["text"]}  # noqa: E731
    return {"cleared": [item(A[k]) for k in A if k not in B],
            "added": [item(B[k]) for k in B if k not in A],
            "kept": [item(B[k]) for k in B if k in A]}


def payload(a: str, b: str, a_tips: list, b_tips: list) -> dict:
    return {"a": a, "b": b, **compare(a_tips, b_tips)}
