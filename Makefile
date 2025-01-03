.PHONY: test setup-dev setup-hooks

test:
	# for future consideration append  --cov-fail-under=80 to fail test coverage if below 80%
	python -m pytest --cov=ra_aid --cov-report=term-missing --cov-report=html

setup-dev:
	pip install -e ".[dev]"

setup-hooks: setup-dev
	pre-commit install
