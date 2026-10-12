# clip-virality-scorer

[![CI](https://github.com/kaankuzu1/clip-virality-scorer/actions/workflows/ci.yml/badge.svg)](https://github.com/kaankuzu1/clip-virality-scorer/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Rank candidate short-form clips cut from long recordings (streams, podcasts,
VODs) by how likely they are to perform on TikTok / Shorts / Reels.

This is the scoring stage of [Audience Labs](https://audiencelabs.net) (in
private beta), an AI pipeline that turns multi-hour livestreams into vertical clips, extracted
into a small standalone package with its own tests. The rest of the pipeline
(transcription, diarization, LLM analysis, rendering) stays private.

## Quickstart

Try it on the bundled example:

```bash
git clone https://github.com/kaankuzu1/clip-virality-scorer && cd clip-virality-scorer
pip install .
clip-scorer rank examples/hooks.json
```

```
rank  score     start       end  type         quote
   1     84       120       148  punchline    So I asked him why the server was down, and he ...
   2     79      2400      2470  reveal       After three weeks of testing, the cheapest setu...
   3     74       900       905  reaction     no way
   4     67      5100      5340  setup        Okay so before we start the next round, let me ...
```

The 5-second `no way` reaction has the highest LLM estimate (85) but ranks
third: it is a fragment with no setup and no title.

To use it in your own project, install straight from GitHub. The base
install has no dependencies (standard library only); training and loading
XGBoost models needs the `xgb` extra:

```bash
pip install "git+https://github.com/kaankuzu1/clip-virality-scorer"
pip install "clip-virality-scorer[xgb] @ git+https://github.com/kaankuzu1/clip-virality-scorer"
```

Requires Python 3.10+.

## Where it sits in the pipeline

```
recording → transcript → LLM proposes candidate "hooks" → [ this package ] → ranked clips → render
```

An upstream LLM pass marks time windows that might make a good clip and gives
each one a virality estimate (0-100), a hook type (`punchline`, `reveal`,
`setup`, …), the transcript quote and an optional promise-style title. LLM
estimates are noisy and cluster in the 70-85 range, so this package adds
cheap structural signals on top and, once real engagement data exists, learns
a correction from it.

## Input format

A hook is one candidate clip: a time window in a long recording that your
own pipeline (an LLM pass, a highlight detector, a human editor) flagged as
worth posting. Every field is optional; missing or malformed values fall back
to neutral ones instead of raising.

| Field | Type | Meaning |
|---|---|---|
| `start_s`, `end_s` | number | Window in the recording, in seconds. |
| `duration_s` | number | Effective clip length, if it differs from `end_s - start_s` (e.g. multi-segment clips). |
| `virality_score` | 0-100 | Your upstream estimate of how well it will do. Carries 75% of the heuristic score. |
| `hook_type` | string | One of `hook`, `punchline`, `reveal`, `setup`, `callback`, `reaction`, `controversy`, `quote`. Unknown types get a lower prior. |
| `quote` | string | Transcript text of the clip. |
| `caption_suggestion` | string | Optional promise-style title for the clip. |

Any other fields are kept and passed through to the output, so you can carry
your own ids along.

## Command line

```bash
clip-scorer rank hooks.json                    # table, best first
clip-scorer rank hooks.json --format json      # same hooks plus a "score" field
clip-scorer rank hooks.json --top 3 --duration 7200
cat hooks.jsonl | clip-scorer rank -           # read from stdin
clip-scorer rank hooks.json --model model.joblib   # use a trained model
```

The input file can be a JSON array of hooks, an object
`{"recording_duration_s": 7200, "hooks": [...]}`, or JSON Lines with one hook
per line. `python -m clip_scorer` works the same as `clip-scorer`.

## Use it in your own app

Call `rank_hooks` right after the step that produces candidate clips, and
render or publish the top results:

```python
from clip_scorer import rank_hooks

candidates = my_pipeline.find_candidate_clips(recording)  # list of dicts, see "Input format"

ranked = rank_hooks(candidates, recording_duration_s=recording.duration_s)
for hook, score in ranked[:5]:
    my_pipeline.render_clip(hook["start_s"], hook["end_s"], score=score)
```

Ranking is deterministic (ties keep their input order) and never raises on a
bad row, so it is safe to run inside a batch job.

**Bring your own scorer.** Anything with a `score(features) -> int` method
works, so you can swap in your own model without changing the call site:

```python
from clip_scorer import HookFeatures, rank_hooks


class LengthAwareScorer:
    def score(self, f: HookFeatures) -> int:
        return f.virality_score - (20 if f.duration_s < 10 else 0)


rank_hooks(candidates, recording_duration_s=3600, scorer=LengthAwareScorer())
```

**Learn from real results.** Once your clips are published and you collect
view counts, train an XGBoost model on them (see [Training](#training)) and
pass it with `--model` or `load_scorer("model.joblib")`. Until then the
heuristic is the default.

## Design

| Module | What it does |
|---|---|
| `features.py` | Turns a raw hook dict into a typed `HookFeatures`. Total function: a malformed LLM row degrades to neutral values instead of raising. |
| `heuristic.py` | `HeuristicScorer`, a deterministic weighted blend that works from day one with no training data. Also the fallback for every other scorer. |
| `labels.py` | Training labels from post-publish engagement, normalised per creator: `sigmoid(log2(views_24h / creator_median))`. |
| `xgb.py` | Feature vector, XGBoost training loop with a reproducible split and validation AUC, and `XGBScorer`. |

**Heuristic weights:** LLM estimate 0.75, hook-type prior 0.10, duration
0.05, has-title 0.05, quote length 0.05. Duration and quote length use a
trapezoid: fragments under 15 s and windows over 150 s are penalised, with no
bias in between, so the topic decides the length rather than a preferred
target.

**Why per-creator labels:** 2,000 views is a hit for a small channel and a
flop for a large one. Dividing by the creator's own median makes a hit look
like a hit regardless of audience size; the log keeps one viral outlier from
dominating. Snapshots marked `dry_run` (simulated publishes) are always
excluded.

**Safety rails:**
- The train/validation split is seeded with `sha256(run_id)`. Python's
  built-in `hash()` is salted per process, so it would give a different split
  on every run.
- A saved model stores its column layout. Loading a model trained on a
  different layout is refused instead of silently mis-predicting.
- `load_scorer()` never raises: a missing or broken model falls back to the
  heuristic, and inference errors fall back per call.

## Training

Each training row is one published clip: the hook, the recording length, the
creator's id and the clip's engagement snapshots.

```python
row = {
    "user_id": "creator-42",
    "recording_duration_s": 7200,
    "hook": {"start_s": 120, "end_s": 148, "virality_score": 78, "hook_type": "punchline"},
    "engagement": {"tiktok": {"snapshots": [{"bucket": "24h", "views": 1800, "dry_run": False}]}},
}
```

```python
from clip_scorer import creator_medians, extract_features, success_label
from clip_scorer.xgb import load_scorer, train

medians = creator_medians(rows)
pairs = [
    (
        extract_features(r["hook"], r["recording_duration_s"]),
        success_label(r["engagement"], medians.get(str(r["user_id"]), 0)),
    )
    for r in rows
]
pairs = [(f, y) for f, y in pairs if y is not None]

result = train([f for f, _ in pairs], [y for _, y in pairs], "model.joblib", run_id="2026-10-10")
print(result.val_auc)

scorer = load_scorer("model.joblib")  # XGBScorer, or the heuristic if loading fails
```

## Development

```bash
git clone https://github.com/kaankuzu1/clip-virality-scorer && cd clip-virality-scorer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[xgb,dev]"
ruff check .
pytest -q
python examples/rank_example.py
```

## Tests

CI runs `ruff check` and `pytest` on Python 3.10 and 3.12. Tests cover the heuristic's shape
(monotone in the LLM estimate, no duration bias inside the plateau, tie
breaking), feature extraction on malformed input, label normalisation, the
AUC implementation against a brute-force reference, the command line, and a full
train → save → load round trip on synthetic data.

## Limitations

- The weights and priors are hand-calibrated, not learned. There is no real
  engagement data yet, so the XGBoost path is tested only on synthetic data.
- Validation AUC is reported on "above the creator's median", which is a
  proxy for success, not a ground truth.

## Contributing

Issues and pull requests are welcome. Please run `ruff check .` and
`pytest -q` before opening a PR; CI runs both.

## How it was built

Built solo with AI-assisted development (Claude Code).

## License

MIT
