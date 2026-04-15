.PHONY: test lint format typecheck check

test:
	uv run pytest tests/ -x

lint:
	uv run ruff check pyweight/ tests/

format:
	uv run ruff format pyweight/ tests/

typecheck:
	uv run pyright pyweight/

check: lint typecheck test
