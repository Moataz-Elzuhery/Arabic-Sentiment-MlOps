from src.preprocessing import normalize_arabic


def test_removes_diacritics_and_tatweel():
    assert normalize_arabic("مُمْتَازٌ") == "ممتاز"
    assert normalize_arabic("جـــميل") == "جميل"


def test_squeezes_elongation():
    assert normalize_arabic("جمييييييل") == "جمييل"


def test_urls_and_spaces():
    assert normalize_arabic("  المنتج   http://x.com  رائع ") == "المنتج رائع"


def test_non_string():
    assert normalize_arabic(None) == ""
