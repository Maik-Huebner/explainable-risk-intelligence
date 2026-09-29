PYTHON := .venv/bin/python
PYTHON_MINOR := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
LOCKFILE ?= $(if $(filter 3.13,$(PYTHON_MINOR)),requirements-lock-py313.txt,$(if $(filter 3.11 3.12,$(PYTHON_MINOR)),requirements-lock-py311.txt,))

.PHONY: install quality test smoke reproduce notebook-check

install:
	@test -n "$(LOCKFILE)" || (echo "Unsupported Python $(PYTHON_MINOR); expected 3.11-3.13" && exit 1)
	python3 -m venv .venv
	.venv/bin/pip install -r $(LOCKFILE)
	.venv/bin/pip install --no-deps --no-build-isolation -e .

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
