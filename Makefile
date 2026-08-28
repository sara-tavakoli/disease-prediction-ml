.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python

# LightGBM needs libomp on macOS; harmless elsewhere.
ifeq ($(shell uname), Darwin)
export DYLD_LIBRARY_PATH := $(shell brew --prefix libomp 2>/dev/null)/lib:$(DYLD_LIBRARY_PATH)
endif

.PHONY: help setup lint format test test-fast cov smoke synth data data-full \
        train-all serve docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create a venv and install the package with dev extras
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/pre-commit install

lint: ## ruff check
	ruff check src tests

format: ## ruff format
	ruff format src tests
	ruff check --fix src tests

test: ## Full test suite (incl. slow end-to-end)
	pytest -q

test-fast: ## Unit tests only
	pytest -q -m "not slow"

cov: ## Coverage report
	pytest -q --cov=sepsis --cov-report=term-missing

smoke: ## 2-epoch transformer run on synthetic data
	sepsis train --config configs/base.yaml configs/smoke.yaml configs/model_transformer.yaml \
		--set output_dir=artifacts/smoke

synth: ## Materialise a synthetic cohort under data/raw
	sepsis synth --n 6000 --prevalence 0.08 --out data/raw

data: ## Download a small real PhysioNet sample into data/raw
	sepsis download --limit 400

data-full: ## Mirror the complete PhysioNet/CinC 2019 training corpus (~2.6 GB)
	sepsis download --full

train-all: ## Train every model on the configured data source
	@for m in lightgbm lstm gru tcn transformer; do \
		echo "=== $$m ==="; \
		sepsis train --config configs/base.yaml configs/model_$$m.yaml || exit 1; \
	done

serve: ## Run the FastAPI service against artifacts/transformer
	sepsis serve --run-dir artifacts/transformer --port 8000

docker: ## Build the inference image
	docker build -t sepsis-early-warning:latest .

clean: ## Remove caches and run artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -rf artifacts/smoke* artifacts/ci_* mlruns
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
