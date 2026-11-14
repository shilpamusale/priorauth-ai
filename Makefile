.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent]"

lint:
	ruff check .

test:
	pytest -q