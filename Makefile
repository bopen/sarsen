COV_REPORT := html
PYTHON := uv run --frozen

default: qa unit-tests check-typing

check-code:
	$(PYTHON) -m pre_commit run --all-files
	$(PYTHON) -m deptry .

qa: check-code

unit-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

check-typing:
	$(PYTHON) -m mypy .

build-docs:
	cp README.md docs/. && rm -fr docs/_api docs/_build
	$(PYTHON) -m sphinx -b html docs docs/_build/html

doc-tests:
	$(PYTHON) -m pytest -vv --doctest-glob="*.md" --doctest-glob="*.rst" README.md

test-minimum-versions:
	uv run --isolated --resolution lowest-direct -p python3.11 -m pytest .

minver-tests: test-minimum-versions

.env: .env.in
	-mv $@ $@~
	cp $^ $@

lab: .env
	$(PYTHON) --group lab --env-file .env -m jupyter lab
