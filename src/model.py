from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import ID2LABEL, LABEL2ID


def build_model(base_model: str):
    """Pretrained Arabic BERT + fresh classification head (3 classes)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model


def load_model(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model
