.PHONY: install lint test

install: ## Install the package with dev tools
	pip install -e ".[dev]"

lint: ## Ruff lint
	ruff check .

test: ## Unit tests and gold-set consistency checks (no API calls)
	pytest -q
