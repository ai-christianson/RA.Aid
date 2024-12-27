.PHONY: test setup-dev setup-hooks

test:
	python -m pytest

setup-dev:
	pip install -e ".[dev]"

setup-hooks: setup-dev
	pre-commit install
