"""Gradient-boosted scorer and its training loop.

Heavy dependencies (numpy, xgboost, joblib) are imported inside functions so
the heuristic path works with the standard library alone. Install them with
`pip install clip-virality-scorer[xgb]`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import HOOK_TYPES, HookFeatures
from .heuristic import HeuristicScorer, HookScorer

CORE_COLUMNS: tuple[str, ...] = (
    "virality_score",
    "duration_s",
    "chapter_position",
    "has_caption_suggestion",
    "quote_length_chars",
)


def feature_columns() -> list[str]:
    """Human-readable column names, in exactly the order of `feature_vector`."""
    return list(CORE_COLUMNS) + [f"hook_type_{ht}" for ht in HOOK_TYPES]


def feature_vector(f: HookFeatures) -> list[float]:
    """Flat numeric vector. Hook type is one-hot encoded."""
    vec = [
        f.virality_score / 100.0,
        f.duration_s,
        f.chapter_position,
        1.0 if f.has_caption_suggestion else 0.0,
        float(f.quote_length_chars),
    ]
    vec.extend(1.0 if f.hook_type == ht else 0.0 for ht in HOOK_TYPES)
    return vec


def stable_seed(text: str) -> int:
    """Seed derived from a string, identical across processes.

    Python's built-in `hash()` is salted per process, so it cannot be used
    for a reproducible train/validation split.
    """
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def roc_auc(y_true: Sequence[bool], y_score: Sequence[float]) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count half).

    Returns 0.5 when one class is absent. O(n log n), no sklearn needed.
    """
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda p: p[0])
    n_pos = sum(1 for _, t in pairs if t)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks over tied scores, then apply the rank-sum formula.
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum_pos += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    n_train: int
    n_val: int
    train_mse: float
    val_mse: float | None
    val_auc: float | None
    columns: list[str]


def train(
    features: Sequence[HookFeatures],
    labels: Sequence[float],
    out_path: str | Path,
    *,
    run_id: str = "default",
    val_fraction: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 4,
) -> TrainResult:
    """Fit an XGBoost regressor on labels in [0, 1] and save it with joblib.

    The shuffle seed comes from `run_id`, so the same run id always produces
    the same split. AUC is computed on "above the creator's median" (label > 0.5).
    """
    import joblib
    import numpy as np
    from xgboost import XGBRegressor

    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if len(features) < 2:
        raise ValueError("need at least two rows to train")

    X = np.asarray([feature_vector(f) for f in features], dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)

    idx = np.arange(len(y))
    np.random.default_rng(stable_seed(run_id)).shuffle(idx)
    split = max(1, int(round(len(idx) * (1.0 - val_fraction))))
    tr, va = idx[:split], idx[split:]

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X[tr], y[tr])

    train_mse = float(np.mean((model.predict(X[tr]) - y[tr]) ** 2))
    val_mse = val_auc = None
    if len(va) > 0:
        pred = model.predict(X[va])
        val_mse = float(np.mean((pred - y[va]) ** 2))
        val_auc = roc_auc((y[va] > 0.5).tolist(), pred.tolist())

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": feature_columns()}, out)
    return TrainResult(out, len(tr), len(va), train_mse, val_mse, val_auc, feature_columns())


class XGBScorer:
    """`HookScorer` backed by a trained model; falls back to the heuristic per call."""

    version = "xgb-v1"

    def __init__(self, model_path: str | Path):
        import joblib

        bundle: Any = joblib.load(Path(model_path))
        if not isinstance(bundle, dict) or bundle.get("columns") != feature_columns():
            # A model trained on a different column layout would silently
            # mis-predict, so refuse it outright.
            raise ValueError("model was trained with a different feature layout")
        self._model = bundle["model"]
        self._fallback = HeuristicScorer()

    def score(self, f: HookFeatures) -> int:
        try:
            import numpy as np

            raw = float(self._model.predict(np.asarray([feature_vector(f)], dtype=np.float32))[0])
        except Exception:  # noqa: BLE001 - any inference failure degrades to the baseline
            return self._fallback.score(f)
        if not 0.0 <= raw <= 1.0:
            raw = 1.0 / (1.0 + math.exp(-raw))
        return int(round(raw * 100))


def load_scorer(model_path: str | Path | None) -> HookScorer:
    """Return an `XGBScorer` if the model loads, else the heuristic baseline. Never raises."""
    if model_path:
        try:
            return XGBScorer(model_path)
        except Exception:  # noqa: BLE001
            pass
    return HeuristicScorer()
