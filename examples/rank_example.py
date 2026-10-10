"""Rank the sample hooks in examples/hooks.json.

Run from the repository root:  python examples/rank_example.py
"""

import json
from pathlib import Path

from clip_scorer import rank_hooks

data = json.loads((Path(__file__).parent / "hooks.json").read_text(encoding="utf-8"))

for hook, score in rank_hooks(data["hooks"], recording_duration_s=data["recording_duration_s"]):
    print(f"{score:>3}  {hook['hook_type']:<10} {hook['start_s']:>5}s-{hook['end_s']}s")
