.PHONY: test setup-dev setup-hooks test-version docker-test-version

test:
	# for future consideration append  --cov-fail-under=80 to fail test coverage if below 80%
	python -m pytest --cov=ra_aid --cov-report=term-missing --cov-report=html

test-version:
	ra-aid --version

docker-test-version:
	docker build -f Dockerfile.test.runtime -t ra-aid-version-test .
	docker run --rm ra-aid-version-test

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
