import pytest

from clip_scorer import creator_medians, extract_views, success_label


def snap(bucket, views, dry=False):
    return {"bucket": bucket, "views": views, "dry_run": dry}


ENG = {
    "tiktok": {"snapshots": [snap("1h", 100), snap("24h", 1000)]},
    "youtube_shorts": {"snapshots": [snap("24h", 500), snap("24h", 99999, dry=True)]},
}


def test_extract_views_sums_24h_real_snapshots_only():
    assert extract_views(ENG) == 1500


GARBAGE = [None, [], "x", {"tiktok": None}, {"tiktok": {"snapshots": [None, {"bucket": "24h", "views": "?"}]}}]


@pytest.mark.parametrize("bad", GARBAGE)
def test_extract_views_tolerates_garbage(bad):
    assert extract_views(bad) == 0.0


def test_label_is_half_at_the_median_and_monotone():
    at = success_label({"t": {"snapshots": [snap("24h", 1000)]}}, 1000)
    double = success_label({"t": {"snapshots": [snap("24h", 2000)]}}, 1000)
    half = success_label({"t": {"snapshots": [snap("24h", 500)]}}, 1000)
    assert at == pytest.approx(0.5)
    assert half < at < double < 1.0


def test_label_is_none_without_signal():
    assert success_label({}, 1000) is None
    assert success_label(ENG, 0) is None


def test_creator_medians_are_per_user():
    rows = [{"user_id": "a", "engagement": {"t": {"snapshots": [snap("24h", v)]}}} for v in (100, 200, 900)] + [
        {"user_id": "b", "engagement": {"t": {"snapshots": [snap("24h", 10_000)]}}},
        {"user_id": "c", "engagement": {}},
    ]
    assert creator_medians(rows) == {"a": 200.0, "b": 10_000.0}
