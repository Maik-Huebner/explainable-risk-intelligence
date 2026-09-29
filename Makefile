PYTHON := .venv/bin/python

.PHONY: install quality test smoke reproduce notebook-check

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-lock.txt
	.venv/bin/pip install --no-deps -e .

quality:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
	$(PYTHON) scripts/audit_repository.py

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m explainable_risk_intelligence.cli smoke-test

reproduce:
	$(PYTHON) -m explainable_risk_intelligence.cli reproduce

notebook-check:
	$(PYTHON) scripts/execute_notebook.py
