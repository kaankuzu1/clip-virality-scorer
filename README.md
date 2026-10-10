# clip-virality-scorer

Rank candidate short-form clips cut from long recordings (streams, podcasts,
VODs) by how likely they are to perform on TikTok / Shorts / Reels.

This is the scoring stage of [Audience Labs](https://audiencelabs.net), an
AI pipeline that turns multi-hour livestreams into vertical clips, extracted
into a small standalone package with its own tests. The rest of the pipeline
(transcription, diarization, LLM analysis, rendering) stays private.

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

## Install

```bash
pip install -e .            # heuristic only, standard library
pip install -e ".[xgb]"     # + numpy, xgboost, scikit-learn, joblib for training
pip install -e ".[xgb,dev]" # + pytest, ruff
```

Requires Python 3.10+.

## Usage

```python
from clip_scorer import rank_hooks

hooks = [
    {"start_s": 120, "end_s": 148, "virality_score": 78, "hook_type": "punchline",
     "quote": "…", "caption_suggestion": "He Did Not See That Coming"},
    {"start_s": 900, "end_s": 905, "virality_score": 85, "hook_type": "reaction",
     "quote": "no way"},
]

for hook, score in rank_hooks(hooks, recording_duration_s=7200):
    print(score, hook["hook_type"])
```

The 5-second reaction has the higher LLM estimate but ranks lower: it is a
fragment with no setup and no title.

Training once you have engagement data:

```python
from clip_scorer import creator_medians, extract_features, success_label
from clip_scorer.xgb import load_scorer, train

medians = creator_medians(rows)  # rows: {"user_id", "engagement", "hook", ...}
pairs = [
    (extract_features(r["hook"], r["recording_duration_s"]),
     success_label(r["engagement"], medians.get(str(r["user_id"]), 0)))
    for r in rows
]
pairs = [(f, y) for f, y in pairs if y is not None]

result = train([f for f, _ in pairs], [y for _, y in pairs], "model.joblib", run_id="2026-10-10")
print(result.val_auc)

scorer = load_scorer("model.joblib")  # XGBScorer, or the heuristic if loading fails
```

## Tests

```bash
ruff check .
pytest -q
```

CI runs both on Python 3.10 and 3.12. Tests cover the heuristic's shape
(monotone in the LLM estimate, no duration bias inside the plateau, tie
breaking), feature extraction on malformed input, label normalisation, the
AUC implementation against a brute-force reference, and a full
train → save → load round trip on synthetic data.

## Limitations

- The weights and priors are hand-calibrated, not learned. There is no real
  engagement data yet, so the XGBoost path is tested only on synthetic data.
- Validation AUC is reported on "above the creator's median", which is a
  proxy for success, not a ground truth.

## How it was built

Built solo with AI-assisted development (Claude Code).

## License

MIT
