.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent,ui]"

lint:
	ruff check .

test:
	pytest -q