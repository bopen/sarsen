PROJECT := sarsen
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa test type-check

qa:
	pre-commit run --all-files

test:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

doc-test:
	python -m pytest -vv --doctest-glob='*.md' README.md

type-check:
	python -m mypy --strict .

conda-env-update:
	$(CONDA) env update $(CONDAFLAGS) -f environment.yml

conda-env-update-all: conda-env-update
	$(CONDA) env update $(CONDAFLAGS) -f environment-dev.yml
