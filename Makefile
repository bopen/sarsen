ENVIRONMENT := SARSEN
COV_REPORT := html
CONDA := conda

default: fix-code-style test code-quality

fix-code-style:
	black .
	isort .
	mdformat .

test: unit-test

unit-test:
	python -m pytest -v --cov=. --cov-report=$(COV_REPORT) tests/

doc-test:
	python -m pytest -v README.md

code-quality:
	flake8 . --max-complexity=10 --max-line-length=127
	mypy --strict .

code-style:
	black --check .
	isort --check .
	mdformat --check .

# deploy

conda-env-create:
	$(CONDA) env create -n $(ENVIRONMENT) -f environment.yml

conda-env-update:
	$(CONDA) env update -n $(ENVIRONMENT) -f environment.yml
