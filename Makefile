.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent,ui,track,api]"

lint:
	ruff check .

test:
	pytest -q
