COV_REPORT := html
PYTHON := uv run --frozen

default: qa unit-tests check-typing

qa:
	$(PYTHON) -m pre_commit run --all-files

unit-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

check-typing:
	$(PYTHON) -m mypy .

docs-build:
	cp README.md docs/. && cd docs && rm -fr _api && make clean && make html

doc-tests:
	$(PYTHON) -m pytest -vv --doctest-glob="*.md" --doctest-glob="*.rst" README.md

integration-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --log-cli-level=INFO tests/integration*.py

.env: .env.in
	-mv $@ $@.bck
	cp $^ $@

lab: .env
	$(PYTHON) --extra lab --env-file .env -m jupyter lab
