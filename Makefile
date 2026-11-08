COV_REPORT := html
PYTHON := uv run --frozen

default: qa unit-tests check-typing

qa:
	$(PYTHON) -m pre_commit run --all-files
	$(PYTHON) -m deptry .

unit-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

check-typing:
	$(PYTHON) -m mypy .

docs-build:
	cp README.md docs/. && rm -fr docs/_api docs/_build
	$(PYTHON) -m sphinx -b html docs docs/_build/html

doc-tests:
	$(PYTHON) -m pytest -vv --doctest-glob="*.md" --doctest-glob="*.rst" README.md

minver-tests:
	uv run --isolated --resolution lowest-direct -p python3.11 -m pytest .

.env: .env.in
	-mv $@ $@~
	cp $^ $@

lab: .env
	$(PYTHON) --group lab --env-file .env -m jupyter lab
