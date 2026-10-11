# Generative Process Design — build entry points (BLUEPRINT.md §2.2)
# All targets run through uv so the venv is implicit.

PROFILE ?= commercial          # research | commercial  (BLUEPRINT D5)
SOURCE  ?= all

.DEFAULT_GOAL := help

.PHONY: help install lint type test test-property test-integration test-data-quality check \
        fetch extract build-corpus train eval smoke clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-20s %s\n", $$1, $$2}'

install: ## Create venv and install core + dev deps
	uv sync --group dev

lint: ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## ruff format (writes)
	uv run ruff format .
	uv run ruff check --fix .

type: ## mypy (strict on canonical/ and validity/)
	uv run mypy

contracts: ## import-linter: core must not import isolated extractor deps
	uv run lint-imports

test: ## Unit + property tests with coverage (PR budget: <5 min)
	uv run pytest tests/unit tests/property -n auto \
		--cov --cov-report=term-missing --cov-fail-under=80

test-property: ## Property tests only, more examples (nightly)
	HYPOTHESIS_PROFILE=nightly uv run pytest tests/property -m property

test-integration: ## End-to-end smoke on fixtures
	uv run pytest tests/integration -m integration

test-data-quality: ## Gates against data/corpus/$(PROFILE)/stats.json
	GPD_PROFILE=$(PROFILE) uv run pytest tests/data_quality -m data_quality

check: lint type contracts test ## Everything a PR must pass

fetch: ## Download raw sources (SOURCE=<name>|all) with checksum verification
	uv run python scripts/fetch.py --source $(SOURCE)

# ---------------------------------------------------------------- PoC-0 (BLUEPRINT §8)
HIRT_DIR   = data/interim/hirtreiter/standard
TRAIN_SIZE ?= 10k
RUN        ?= runs/pfd2pid_$(TRAIN_SIZE)

poc0-data: ## Fetch Hirtreiter and write PairRecord JSONL (TRAIN_SIZE=10k)
	uv run python scripts/fetch.py --source hirtreiter
	uv run python extractors/hirtreiter/extract.py --train-size $(TRAIN_SIZE) --variant standard

poc0-train: ## Train PFD→P&ID seq2seq on Hirtreiter (needs `uv sync --extra models`)
	uv run python -m gpd.models.train --train $(HIRT_DIR)/train_$(TRAIN_SIZE).jsonl \
		--val $(HIRT_DIR)/val.jsonl --out $(RUN)

poc0-eval: ## Beam-5 evaluation on the upstream test split vs gpd/eval/baselines.json
	uv run python -m gpd.eval.benchmarks pfd2pid --checkpoint $(RUN)/best.pt --test $(HIRT_DIR)/test.jsonl

poc0: poc0-data poc0-train poc0-eval ## Full PoC-0 chain

# ---------------------------------------------------------------- WS1+ (not implemented yet)
extract: ## [WS1] Run all extractors → data/interim/<source>/*.ir.json
	@echo "not implemented yet (BLUEPRINT WS1)"; exit 1

build-corpus: ## [WS4] interim IR → canonical → validity → dedup → splits → data/corpus/$(PROFILE)/
	@echo "not implemented yet (BLUEPRINT WS4)"; exit 1

smoke: ## fixtures → extract → corpus → 20 train steps → eval → API call
	uv run pytest tests/integration/test_smoke.py -m integration

clean: ## Remove build artifacts (never touches data/raw)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
