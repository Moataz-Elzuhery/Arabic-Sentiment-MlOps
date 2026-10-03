"""Light Arabic text normalization used both at training and inference time.

Keep this function identical in both places, otherwise you create train/serve skew.
"""
import re

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
_TATWEEL = re.compile(r"\u0640")
_ELONGATION = re.compile(r"(.)\1{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Remove diacritics/tatweel/URLs, squeeze letter elongation, normalize spaces."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ELONGATION.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()
