from scripts.canary_check import share_within_range


def test_rollback_must_have_zero_canary_answers():
    assert share_within_range(0, 400, 0)
    assert not share_within_range(1, 400, 0)


def test_full_rollout_must_be_all_canary():
    assert share_within_range(400, 400, 100)
    assert not share_within_range(399, 400, 100)


def test_partial_share_uses_statistical_range():
    assert share_within_range(15, 400, 5)       # 3.75%, normal sampling noise
    assert share_within_range(24, 400, 5)
    assert not share_within_range(60, 400, 5)   # far too many
    assert not share_within_range(0, 400, 5)    # canary receiving nothing is also wrong
