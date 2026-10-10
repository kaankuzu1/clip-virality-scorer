"""Command-line interface: `clip-scorer rank hooks.json`.

Reads candidate hooks as JSON and prints them ranked, best first. Accepted
input shapes:

* a JSON array of hook objects;
* an object ``{"recording_duration_s": 7200, "hooks": [...]}``;
* JSON Lines, one hook object per line.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from . import rank_hooks
from .heuristic import HeuristicScorer, HookScorer


def parse_hooks(text: str) -> tuple[list[dict[str, Any]], float | None]:
    """Return `(hooks, recording_duration_s)` from any accepted input shape."""
    text = text.strip()
    if not text:
        return [], None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Not a single JSON document: treat it as JSON Lines.
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    duration = None
    if isinstance(data, dict):
        duration = data.get("recording_duration_s")
        data = data.get("hooks", [])
    if not isinstance(data, list) or not all(isinstance(h, dict) for h in data):
        raise ValueError("expected a list of hook objects")
    return data, (float(duration) if duration is not None else None)


def _load_scorer(model_path: str | None) -> HookScorer:
    if not model_path:
        return HeuristicScorer()
    try:
        from .xgb import load_scorer
    except ImportError as exc:  # xgb extras not installed
        raise SystemExit(f"--model needs the xgb extras: pip install 'clip-virality-scorer[xgb]' ({exc})") from exc
    return load_scorer(model_path)


def _print_table(ranked: list[tuple[dict[str, Any], int]], out: TextIO) -> None:
    out.write(f"{'rank':>4}  {'score':>5}  {'start':>8}  {'end':>8}  {'type':<12} quote\n")
    for i, (hook, score) in enumerate(ranked, 1):
        quote = str(hook.get("quote") or "").replace("\n", " ")
        if len(quote) > 50:
            quote = quote[:47] + "..."
        out.write(
            f"{i:>4}  {score:>5}  {hook.get('start_s', ''):>8}  {hook.get('end_s', ''):>8}  "
            f"{str(hook.get('hook_type') or ''):<12} {quote}\n"
        )


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    parser = argparse.ArgumentParser(prog="clip-scorer", description="Rank candidate short-form clips.")
    sub = parser.add_subparsers(dest="command", required=True)
    rank = sub.add_parser("rank", help="score and rank hooks from a JSON file or stdin")
    rank.add_argument("input", help="path to a JSON / JSON Lines file, or - for stdin")
    rank.add_argument("--duration", type=float, help="recording length in seconds (overrides the file)")
    rank.add_argument("--model", help="trained XGBoost model (.joblib); falls back to the heuristic if it can't load")
    rank.add_argument("--top", type=int, help="only print the best N hooks")
    rank.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args(argv)

    try:
        if args.input == "-":
            text = stdin.read()
        else:
            with open(args.input, encoding="utf-8") as fh:
                text = fh.read()
        hooks, duration = parse_hooks(text)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.duration is not None:
        duration = args.duration
    ranked = rank_hooks(hooks, recording_duration_s=duration, scorer=_load_scorer(args.model))
    if args.top is not None:
        ranked = ranked[: max(0, args.top)]

    try:
        if args.format == "json":
            json.dump([{**hook, "score": score} for hook, score in ranked], stdout, ensure_ascii=False, indent=2)
            stdout.write("\n")
        else:
            _print_table(ranked, stdout)
        stdout.flush()
    except BrokenPipeError:
        # Output piped into e.g. `head`, which closed early: not an error.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
