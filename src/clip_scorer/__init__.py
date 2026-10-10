"""Rank candidate short-form clips cut from long recordings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .features import HOOK_TYPES, HookFeatures, extract_features
from .heuristic import HeuristicScorer, HookScorer, trapezoid
from .labels import creator_medians, extract_views, success_label

__all__ = [
    "HOOK_TYPES",
    "HeuristicScorer",
    "HookFeatures",
    "HookScorer",
    "creator_medians",
    "extract_features",
    "extract_views",
    "rank_hooks",
    "success_label",
    "trapezoid",
]


def rank_hooks(
    hooks: Iterable[Mapping[str, Any]],
    *,
    recording_duration_s: float | None,
    scorer: HookScorer | None = None,
) -> list[tuple[Mapping[str, Any], int]]:
    """Score every hook and return `(hook, score)` pairs, best first.

    Ties keep their input order, so the result is deterministic.
    """
    scorer = scorer or HeuristicScorer()
    scored = [(h, scorer.score(extract_features(h, recording_duration_s))) for h in hooks]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
