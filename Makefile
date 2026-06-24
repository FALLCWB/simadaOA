.PHONY: test lint type-check format sim analyze clean

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-all:
	uv run pytest tests/ -v --tb=long

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/simada/

sim:
	uv run simada run configs/scenarios/7day_cohort.yaml

analyze:
	uv run simada analyze results/

clean:
	rm -rf results/ export/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
