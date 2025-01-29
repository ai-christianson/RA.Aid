.PHONY: test setup-dev setup-hooks

test:
	# for future consideration append  --cov-fail-under=80 to fail test coverage if below 80%
	python -m pytest --cov=ra_aid --cov-report=term-missing --cov-report=html

setup-dev:
	pip install -e ".[dev]"

setup-hooks: setup-dev
	pre-commit install

check:
	ruff check

fix:
	ruff check . --select I --fix # First sort imports
	ruff format .
	ruff check --fix

fix-basic:
	ruff check --fix
