.PHONY: install test train demo monitor

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

# Dados reais: baixe cs-training.csv (ver data/README.md) e salve em data/raw/
train:
	python -m credit_scoring.train --data data/raw/cs-training.csv

# Demonstração com dados sintéticos (NÃO usar os números como resultado)
demo:
	python -m credit_scoring.train --synthetic

monitor:
	python -m credit_scoring.monitor --data data/raw/cs-training.csv
