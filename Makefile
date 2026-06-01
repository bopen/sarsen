PROJECT := sarsen
COV_REPORT := html
PYTHON := uv run --frozen

default: qa unit-tests type-check

qa:
	$(PYTHON) -m pre_commit run --all-files

unit-tests:
	$(PYTHON) -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	$(PYTHON) -m mypy .

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run --rm -ti -v $(PWD):/srv $(PROJECT)

docs-build:
	cp README.md docs/. && cd docs && rm -fr _api && make clean && make html

doc-test:
	$(PYTHON) -m pytest -vv --doctest-glob='*.md' README.md
