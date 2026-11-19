.PHONY: install test clean-data split cards sft dpo
install:
	pip install -e ".[dev,nlp,train,llm]"
test:
	pytest -q
# Example pipeline once data/interim/*.jsonl exists:
clean-data:
	python -m fluous.cli clean --inp data/interim/*.jsonl --out data/processed/clean.jsonl --config configs/data.yaml
split:
	python -m fluous.cli split --inp data/processed/clean.jsonl --out data/processed/split.jsonl
cards:
	python -m fluous.cli cards --inp data/processed/split.jsonl --out data/processed/carded.jsonl
sft:
	python -m fluous.train.sft --config configs/sft.yaml
dpo:
	python -m fluous.train.dpo --config configs/dpo.yaml
