PROJECT := sarsen
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests type-check

qa:
	uv run --frozen -m pre_commit run --all-files

unit-tests:
	uv run --frozen -m pytest -vv --cov=. --cov-report=$(COV_REPORT) --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	uv run --frozen -m mypy .

conda-env-update:
	$(CONDA) install -y -c conda-forge conda-merge
	$(CONDA) run conda-merge environment.yml ci/environment-ci.yml > ci/combined-environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f ci/combined-environment-ci.yml

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run --rm -ti -v $(PWD):/srv $(PROJECT)

template-update:
	pre-commit run --all-files cruft -c .pre-commit-config-cruft.yaml

docs-build:
	cp README.md docs/. && cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW


doc-test:
	uv run --frozen -m pytest -vv --doctest-glob='*.md' README.md
