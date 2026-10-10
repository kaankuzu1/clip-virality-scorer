"""Feature extraction for candidate clips ("hooks").

A hook is a time window in a long recording that an upstream LLM pass has
flagged as a possible short-form clip. Each hook arrives as a plain dict with
an LLM virality estimate, a hook type, timestamps, the transcript quote and an
optional promise-style title. This module turns that dict into a small,
typed feature record that every scorer consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The eight hook archetypes the upstream LLM is allowed to emit.
# Order is load-bearing: it fixes the one-hot column order in `xgb.feature_vector`.
HOOK_TYPES: tuple[str, ...] = (
    "hook",
    "punchline",
    "reveal",
    "setup",
    "callback",
    "reaction",
    "controversy",
    "quote",
)


@dataclass(frozen=True, slots=True)
class HookFeatures:
    virality_score: int  # LLM estimate, 0-100
    hook_type: str  # one of HOOK_TYPES (unknown values allowed)
    duration_s: float  # effective clip duration in seconds
    chapter_position: float  # 0..1, hook start relative to recording length
    has_caption_suggestion: bool  # did the LLM propose a promise-style title?
    quote_length_chars: int  # length of the transcript quote


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_features(
    hook: Mapping[str, Any],
    recording_duration_s: float | None,
) -> HookFeatures:
    """Build `HookFeatures` from a raw hook dict.

    Total function: malformed or missing fields fall back to neutral values
    instead of raising, because a single bad LLM row must never break the
    scoring of a whole recording.
    """
    start_s = _to_float(hook.get("start_s"))
    end_s = _to_float(hook.get("end_s"))

    # Prefer an explicit effective duration when the caller has one
    # (e.g. multi-segment clips); otherwise use the window length.
    explicit = hook.get("duration_s")
    if explicit is not None:
        duration = _to_float(explicit, default=end_s - start_s)
    else:
        duration = end_s - start_s
    duration = max(0.0, duration)

    if recording_duration_s and recording_duration_s > 0:
        position = max(0.0, min(1.0, start_s / recording_duration_s))
    else:
        position = 0.0

    virality = int(_to_float(hook.get("virality_score")))
    virality = max(0, min(100, virality))

    return HookFeatures(
        virality_score=virality,
        hook_type=str(hook.get("hook_type") or "").strip().lower(),
        duration_s=duration,
        chapter_position=position,
        has_caption_suggestion=bool(str(hook.get("caption_suggestion") or "").strip()),
        quote_length_chars=len(str(hook.get("quote") or "")),
    )
