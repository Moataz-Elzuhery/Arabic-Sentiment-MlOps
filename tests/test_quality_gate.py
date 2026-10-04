from scripts.quality_gate import check

BASE = {"f1_macro": 0.8474}
GOOD = {"f1_macro": 0.8474, "per_class_f1": {"negative": 0.83, "neutral": 0.76, "positive": 0.95}}


def test_passes_when_equal_or_better():
    assert check(GOOD, BASE, tolerance=0.005, min_class_f1=0.70) == []
    better = {**GOOD, "f1_macro": 0.86}
    assert check(better, BASE, tolerance=0.005, min_class_f1=0.70) == []


def test_fails_when_f1_drops():
    worse = {**GOOD, "f1_macro": 0.80}
    failures = check(worse, BASE, tolerance=0.005, min_class_f1=0.70)
    assert len(failures) == 1 and "below baseline" in failures[0]


def test_tolerance_absorbs_tiny_drop():
    tiny = {**GOOD, "f1_macro": 0.8450}
    assert check(tiny, BASE, tolerance=0.005, min_class_f1=0.70) == []


def test_fails_when_one_class_collapses():
    bad = {"f1_macro": 0.85, "per_class_f1": {"negative": 0.9, "neutral": 0.40, "positive": 0.95}}
    failures = check(bad, BASE, tolerance=0.005, min_class_f1=0.70)
    assert len(failures) == 1 and "neutral" in failures[0]
