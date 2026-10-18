.PHONY: dev-sync test build

dev-sync:  ## create .venv and install pymust with all extras (editable)
	uv sync --all-extras

test:
	uv run pytest tests

build:  ## sdist + wheel in dist/
	uv build
