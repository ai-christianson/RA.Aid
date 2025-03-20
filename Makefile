.PHONY: test setup-dev setup-hooks last-cost help

help:
	@echo "Available targets:"
	@echo "  help        - Display this help message"
	@echo "  test        - Run tests with coverage reporting"
	@echo "  setup-dev   - Install development dependencies"
	@echo "  setup-hooks - Install git pre-commit hooks"
	@echo "  check       - Run code quality checks with ruff"
	@echo "  fix         - Fix code style issues automatically"
	@echo "  fix-basic   - Fix basic code style issues"
	@echo "  last-cost   - Display cost and token usage for the latest session"

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

last-cost:
	python ra_aid/scripts/cli.py latest

all-costs:
	python ra_aid/scripts/cli.py all
