.PHONY: setup install

## Run once after cloning: install deps + git hooks
setup: install
	uv run --directory src pre-commit install
	@echo "Git hooks installed via pre-commit."

## Install Python dependencies
install:
	uv sync --directory src
