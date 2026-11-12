.PHONY: install lint test

install:
	pip install -e ".[dev,ingest]"

lint:
	ruff check .

test:
	pytest -q