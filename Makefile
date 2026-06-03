.PHONY: test lint fmt gauntlet install

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

# Run the full evaluation gauntlet.
# Usage: make gauntlet DATASET=tabformer SYNTH=dp_ctgan_eps3
gauntlet:
	uv run python -m mirrorbank.evaluate.gauntlet \
	  --real data/raw/$(DATASET)/transactions.csv \
	  --synth outputs/$(SYNTH)/synthetic_transactions.csv \
	  --instrument $(DATASET)

# List available instruments
instruments:
	uv run python -c "from mirrorbank.instruments.registry import REGISTRY; print('\n'.join(sorted(REGISTRY)))"
