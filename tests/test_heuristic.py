import pytest

from clip_scorer import HeuristicScorer, HookFeatures, rank_hooks, trapezoid


def feats(**kw):
    base = dict(
        virality_score=70,
        hook_type="punchline",
        duration_s=30.0,
        chapter_position=0.5,
        has_caption_suggestion=True,
        quote_length_chars=300,
    )
    base.update(kw)
    return HookFeatures(**base)


S = HeuristicScorer()


def test_weights_sum_to_one():
    total = S.W_LLM + S.W_HOOK_TYPE + S.W_DURATION + S.W_CAPTION + S.W_QUOTE
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x, expected",
    [(-1, 0.0), (0, 0.0), (7.5, 0.5), (15, 1.0), (100, 1.0), (150, 1.0), (165, 0.5), (180, 0.0), (500, 0.0)],
)
def test_trapezoid_duration_shape(x, expected):
    assert trapezoid(x, **S.DURATION) == pytest.approx(expected)


def test_trapezoid_rejects_bad_params():
    with pytest.raises(ValueError):
        trapezoid(1, plateau_low=10, plateau_high=5, ramp_high=20)


def test_score_is_always_an_int_in_range():
    for v in (-50, 0, 50, 100, 500):
        for d in (0.0, 5.0, 60.0, 1000.0):
            for q in (0, 50, 5000):
                s = S.score(feats(virality_score=v, duration_s=d, quote_length_chars=q))
                assert isinstance(s, int) and 0 <= s <= 100


def test_perfect_and_empty_hooks_hit_the_bounds():
    assert S.score(feats(virality_score=100)) == 100
    empty = feats(virality_score=0, hook_type="", duration_s=0, has_caption_suggestion=False, quote_length_chars=0)
    assert S.score(empty) == 6  # only the unknown-type prior (0.60 * 0.10) remains


def test_score_is_monotone_in_llm_virality():
    scores = [S.score(feats(virality_score=v)) for v in range(0, 101, 5)]
    assert scores == sorted(scores)


def test_duration_has_no_bias_inside_plateau():
    assert S.score(feats(duration_s=20)) == S.score(feats(duration_s=140))


def test_fragments_and_oversized_clips_are_penalised():
    good = S.score(feats(duration_s=30))
    assert S.score(feats(duration_s=5)) < good
    assert S.score(feats(duration_s=175)) < good


def test_caption_suggestion_breaks_ties():
    assert S.score(feats(has_caption_suggestion=True)) > S.score(feats(has_caption_suggestion=False))


def test_hook_type_prior_ordering():
    assert S.score(feats(hook_type="punchline")) > S.score(feats(hook_type="setup"))
    assert S.score(feats(hook_type="mystery")) <= S.score(feats(hook_type="setup"))


def test_rank_hooks_orders_best_first_and_is_stable():
    hooks = [
        {"id": "a", "virality_score": 40, "hook_type": "setup", "start_s": 0, "end_s": 30},
        {"id": "b", "virality_score": 90, "hook_type": "punchline", "start_s": 60, "end_s": 90},
        {"id": "c", "virality_score": 40, "hook_type": "setup", "start_s": 120, "end_s": 150},
    ]
    ranked = [h["id"] for h, _ in rank_hooks(hooks, recording_duration_s=600)]
    assert ranked == ["b", "a", "c"]
