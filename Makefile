PROJECT := sarsen
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa test type-check

qa:
	pre-commit run --all-files

test:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

type-check:
	python -m mypy --strict .

doc-test:
	python -m pytest -v README.md

conda-env-update:
	$(CONDA) env update $(CONDAFLAGS) -f environment.yml

conda-env-update-all: conda-env-update
	$(CONDA) env update $(CONDAFLAGS) -f environment-dev.yml
