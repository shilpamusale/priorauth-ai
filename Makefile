.PHONY: install lint test

install:
	pip install -e ".[dev,ingest,agent,ui,track,api,retrieval,mcp]"

lint:
	ruff check .

test:
	pytest -q
