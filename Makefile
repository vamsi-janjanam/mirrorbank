.PHONY: install install-generate install-release test lint fmt gauntlet instruments sample-data ui

install:
	pipenv install

install-dev:
	pipenv install --dev

# Full DP generation stack (torch, opacus, sdv)
install-generate:
	pipenv install --categories generate

# Release artifact stack (reportlab, jinja2)
install-release:
	pipenv install --categories release

test:
	pipenv run pytest

lint:
	pipenv run ruff check src/ tests/

fmt:
	pipenv run ruff format src/ tests/

# Run the full evaluation gauntlet.
# Usage: make gauntlet DATASET=tabformer SYNTH=dp_ctgan_eps3
gauntlet:
	pipenv run python -m mirrorbank.evaluate.gauntlet \
	  --real data/raw/$(DATASET)/transactions.csv \
	  --synth outputs/$(SYNTH)/synthetic_transactions.csv \
	  --instrument $(DATASET)

# Generate example CSVs into data/sample/ (real + synthetic per instrument)
sample-data:
	pipenv run python scripts/generate_sample_data.py

# List available instruments
instruments:
	pipenv run python -c "from mirrorbank.instruments.registry import REGISTRY; print('\n'.join(sorted(REGISTRY)))"

# Launch the Streamlit UI
ui:
	pipenv run streamlit run src/mirrorbank/ui/app.py
