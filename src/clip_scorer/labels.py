"""Training labels from post-publish engagement.

Raw view counts are not comparable across creators: 2,000 views is a hit for
a small channel and a flop for a large one. Labels are therefore normalised
per creator before training.

Engagement arrives as a platform-keyed dict of time-bucketed snapshots::

    {"tiktok": {"snapshots": [{"bucket": "24h", "views": 1200, "dry_run": False}, ...]}}

Dry-run snapshots (fake numbers produced when publishing is simulated) are
always excluded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from statistics import median
from typing import Any

LABEL_BUCKET = "24h"


def extract_views(engagement: Any, bucket: str = LABEL_BUCKET) -> float:
    """Sum real (non-dry-run) views in `bucket` across all platforms."""
    if not isinstance(engagement, Mapping):
        return 0.0
    total = 0.0
    for block in engagement.values():
        if not isinstance(block, Mapping):
            continue
        for snap in block.get("snapshots") or []:
            if not isinstance(snap, Mapping):
                continue
            if snap.get("dry_run") is True or snap.get("bucket") != bucket:
                continue
            try:
                total += float(snap.get("views") or 0)
            except (TypeError, ValueError):
                continue
    return total


def creator_medians(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Median positive view count per creator (`user_id`)."""
    per_user: dict[str, list[float]] = {}
    for row in rows:
        views = extract_views(row.get("engagement"))
        if views > 0:
            per_user.setdefault(str(row.get("user_id")), []).append(views)
    return {uid: float(median(v)) for uid, v in per_user.items()}


def success_label(engagement: Any, creator_median: float) -> float | None:
    """Map a clip's views to a [0, 1] label relative to its creator's median.

    label = sigmoid(log2(views / median)): 0.5x median -> 0.27, at median -> 0.5,
    2x -> 0.73, 4x -> 0.88. The log keeps one viral outlier from dominating.
    Returns None when there is nothing to learn from.
    """
    views = extract_views(engagement)
    if views <= 0 or creator_median <= 0:
        return None
    z = math.log2(views / creator_median)
    return 1.0 / (1.0 + math.exp(-z))
