"""Heuristic baseline scorer.

A deterministic weighted blend of calibrated priors. It needs no training
data, which is the point: it ranks clips from day one and stays as the
fallback whenever a trained model is missing or fails.
"""

from __future__ import annotations

from typing import Protocol

from .features import HookFeatures


class HookScorer(Protocol):
    """Every scorer maps features to an int in [0, 100] and names its version."""

    version: str

    def score(self, features: HookFeatures) -> int: ...


def trapezoid(x: float, *, plateau_low: float, plateau_high: float, ramp_high: float) -> float:
    """Piecewise-linear membership in [0, 1].

    Rises 0 -> 1 over [0, plateau_low], stays 1 over [plateau_low, plateau_high],
    falls 1 -> 0 over [plateau_high, ramp_high], and is 0 beyond.
    """
    if not 0 < plateau_low <= plateau_high < ramp_high:
        raise ValueError("expected 0 < plateau_low <= plateau_high < ramp_high")
    if x <= 0:
        return 0.0
    if x < plateau_low:
        return x / plateau_low
    if x <= plateau_high:
        return 1.0
    if x < ramp_high:
        return 1.0 - (x - plateau_high) / (ramp_high - plateau_high)
    return 0.0


class HeuristicScorer:
    """LLM estimate + structural priors.

    The LLM virality estimate carries most of the weight. The other terms are
    cheap tie-breakers for the LLM's noisy upper range:

    * hook type prior: some archetypes (punchlines, reveals) perform better
      on short-form feeds than others (setups, callbacks);
    * duration: penalise fragments under 15 s and windows over 150 s, with
      no bias in between, so the topic decides the length;
    * caption suggestion: a clip with an obvious promise to lead with;
    * quote length: very short quotes are usually fragments without setup,
      very long ones usually sprawl.
    """

    version = "heuristic-v7"

    W_LLM = 0.75
    W_HOOK_TYPE = 0.10
    W_DURATION = 0.05
    W_CAPTION = 0.05
    W_QUOTE = 0.05

    HOOK_TYPE_PRIOR: dict[str, float] = {
        "punchline": 1.00,
        "reveal": 0.95,
        "controversy": 0.90,
        "reaction": 0.85,
        "quote": 0.80,
        "hook": 0.80,
        "callback": 0.70,
        "setup": 0.65,
    }
    UNKNOWN_TYPE_PRIOR = 0.60

    DURATION = dict(plateau_low=15.0, plateau_high=150.0, ramp_high=180.0)
    QUOTE = dict(plateau_low=80.0, plateau_high=600.0, ramp_high=1200.0)

    def score(self, f: HookFeatures) -> int:
        llm = max(0, min(100, f.virality_score)) / 100.0
        htype = self.HOOK_TYPE_PRIOR.get(f.hook_type, self.UNKNOWN_TYPE_PRIOR)
        duration = trapezoid(f.duration_s, **self.DURATION)
        caption = 1.0 if f.has_caption_suggestion else 0.0
        quote = trapezoid(float(f.quote_length_chars), **self.QUOTE)

        total = (
            self.W_LLM * llm
            + self.W_HOOK_TYPE * htype
            + self.W_DURATION * duration
            + self.W_CAPTION * caption
            + self.W_QUOTE * quote
        )
        return int(round(max(0.0, min(1.0, total)) * 100))
