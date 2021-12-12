ENVIRONMENT := SARSEN
COV_REPORT := html

default: fix-code-style test

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
	conda env create -n $(ENVIRONMENT) -f environment.yml

conda-env-update:
	conda env update -n $(ENVIRONMENT) -f environment.yml

