.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent,ui,track,api,retrieval]"

lint:
	ruff check .

test:
	pytest -q
