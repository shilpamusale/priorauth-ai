.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent,ui,track]"

lint:
	ruff check .

test:
	pytest -q
