import io
import json
from pathlib import Path

import pytest

from clip_scorer.cli import main, parse_hooks

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "hooks.json"


def run(argv, stdin_text=""):
    out = io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin_text), stdout=out)
    return code, out.getvalue()


def test_parse_accepts_array_object_and_jsonl():
    hook = {"start_s": 0, "end_s": 30, "virality_score": 70}
    assert parse_hooks(json.dumps([hook])) == ([hook], None)
    assert parse_hooks(json.dumps({"recording_duration_s": 600, "hooks": [hook]})) == ([hook], 600.0)
    assert parse_hooks(json.dumps(hook) + "\n\n" + json.dumps(hook)) == ([hook, hook], None)
    assert parse_hooks("   ") == ([], None)


def test_parse_rejects_non_hook_lists():
    with pytest.raises(ValueError):
        parse_hooks("[1, 2]")


def test_rank_example_file_as_json():
    code, out = run(["rank", str(EXAMPLE), "--format", "json"])
    assert code == 0
    ranked = json.loads(out)
    scores = [h["score"] for h in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == 4
    # The 5-second "no way" fragment has the highest LLM estimate but must not win.
    assert ranked[0]["start_s"] != 900


def test_rank_from_stdin_with_top_and_table():
    hooks = [{"start_s": i * 100, "end_s": i * 100 + 40, "virality_score": 60 + i} for i in range(5)]
    code, out = run(["rank", "-", "--top", "2", "--duration", "1000"], json.dumps(hooks))
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].split()[:2] == ["rank", "score"]
    assert len(lines) == 3


def test_missing_model_falls_back_to_heuristic(tmp_path):
    pytest.importorskip("xgboost")
    code, out = run(["rank", str(EXAMPLE), "--model", str(tmp_path / "nope.joblib"), "--format", "json"])
    assert code == 0
    assert len(json.loads(out)) == 4
