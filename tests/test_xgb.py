import random

import pytest

from clip_scorer import HeuristicScorer, HookFeatures
from clip_scorer.features import HOOK_TYPES
from clip_scorer.xgb import feature_columns, feature_vector, load_scorer, roc_auc, stable_seed


def test_feature_vector_matches_column_names():
    f = HookFeatures(50, "reveal", 20.0, 0.3, True, 200)
    vec = feature_vector(f)
    cols = feature_columns()
    assert len(vec) == len(cols)
    assert vec[cols.index("hook_type_reveal")] == 1.0
    assert sum(vec[cols.index(f"hook_type_{h}")] for h in HOOK_TYPES) == 1.0


def test_stable_seed_is_deterministic():
    assert stable_seed("run-1") == stable_seed("run-1")
    assert stable_seed("run-1") != stable_seed("run-2")


def brute_auc(y, s):
    pos = [b for b, t in zip(s, y, strict=True) if t]
    neg = [b for b, t in zip(s, y, strict=True) if not t]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_brute_force_with_ties():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(2, 40)
        y = [rng.random() < 0.4 for _ in range(n)]
        s = [rng.choice([0.1, 0.2, 0.5, 0.9]) for _ in range(n)]
        if 0 < sum(y) < n:
            assert roc_auc(y, s) == pytest.approx(brute_auc(y, s))


def test_roc_auc_edge_cases():
    assert roc_auc([True, True], [0.1, 0.2]) == 0.5
    assert roc_auc([False, True], [0.1, 0.9]) == 1.0
    assert roc_auc([True, False], [0.1, 0.9]) == 0.0


def test_load_scorer_falls_back_to_heuristic(tmp_path):
    assert isinstance(load_scorer(None), HeuristicScorer)
    assert isinstance(load_scorer(tmp_path / "missing.joblib"), HeuristicScorer)


xgboost = pytest.importorskip("xgboost")


def synthetic(n, seed=1):
    """Labels follow a hidden rule so the model has something real to learn."""
    rng = random.Random(seed)
    feats, labels = [], []
    for _ in range(n):
        f = HookFeatures(
            virality_score=rng.randint(0, 100),
            hook_type=rng.choice(HOOK_TYPES),
            duration_s=rng.uniform(5, 170),
            chapter_position=rng.random(),
            has_caption_suggestion=rng.random() < 0.6,
            quote_length_chars=rng.randint(20, 1200),
        )
        signal = 0.7 * f.virality_score / 100 + 0.3 * (f.hook_type in ("punchline", "reveal"))
        feats.append(f)
        labels.append(min(1.0, max(0.0, signal + rng.gauss(0, 0.05))))
    return feats, labels


def test_train_save_load_roundtrip(tmp_path):
    from clip_scorer.xgb import XGBScorer, train

    feats, labels = synthetic(400)
    res = train(feats, labels, tmp_path / "m.joblib", run_id="test", n_estimators=50)
    assert res.n_train == 320 and res.n_val == 80
    assert res.val_auc is not None and res.val_auc > 0.8

    scorer = load_scorer(res.model_path)
    assert isinstance(scorer, XGBScorer)
    hi = scorer.score(HookFeatures(95, "punchline", 30, 0.5, True, 300))
    lo = scorer.score(HookFeatures(5, "setup", 30, 0.5, True, 300))
    assert 0 <= lo < hi <= 100


def test_same_run_id_gives_same_split(tmp_path):
    from clip_scorer.xgb import train

    feats, labels = synthetic(200)
    a = train(feats, labels, tmp_path / "a.joblib", run_id="same", n_estimators=20)
    b = train(feats, labels, tmp_path / "b.joblib", run_id="same", n_estimators=20)
    assert a.val_mse == pytest.approx(b.val_mse)


def test_model_with_wrong_layout_is_refused(tmp_path):
    import joblib

    path = tmp_path / "old.joblib"
    joblib.dump({"model": object(), "columns": ["virality_score"]}, path)
    assert isinstance(load_scorer(path), HeuristicScorer)
