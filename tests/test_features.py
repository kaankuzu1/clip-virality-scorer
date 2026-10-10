from clip_scorer import extract_features


def test_basic_extraction():
    f = extract_features(
        {
            "virality_score": 82,
            "hook_type": " Reveal ",
            "start_s": 300,
            "end_s": 330,
            "quote": "x" * 120,
            "caption_suggestion": "He did not see it coming",
        },
        recording_duration_s=1200,
    )
    assert f.virality_score == 82
    assert f.hook_type == "reveal"
    assert f.duration_s == 30
    assert f.chapter_position == 0.25
    assert f.has_caption_suggestion is True
    assert f.quote_length_chars == 120


def test_explicit_duration_wins_over_window():
    f = extract_features({"start_s": 0, "end_s": 100, "duration_s": 24.5}, 600)
    assert f.duration_s == 24.5


def test_malformed_rows_never_raise():
    f = extract_features(
        {"virality_score": "n/a", "start_s": None, "end_s": "oops", "duration_s": "bad", "caption_suggestion": "   "},
        recording_duration_s=None,
    )
    assert f.virality_score == 0
    assert f.duration_s == 0.0
    assert f.chapter_position == 0.0
    assert f.has_caption_suggestion is False


def test_values_are_clamped():
    f = extract_features({"virality_score": 250, "start_s": 900, "end_s": 800}, 600)
    assert f.virality_score == 100
    assert f.duration_s == 0.0
    assert f.chapter_position == 1.0
