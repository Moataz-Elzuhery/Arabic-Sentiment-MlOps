.PHONY: test lint smoke train serve
test:
	pytest -q
lint:
	ruff check .
smoke:
	python -m src.train --max-samples 200 --epochs 1 --output-dir models/smoke
train:
	python -m src.train --config configs/train.yaml
serve:
	MODEL_DIR=models/v1 uvicorn src.api:app --reload
