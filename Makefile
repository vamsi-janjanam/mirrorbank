.PHONY: install install-generate install-release test lint fmt gauntlet instruments sample-data ui generate audit hooks

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

# Generate synthetic data via the CLI
generate:
	pipenv run mirrorbank generate $(CSV) --instrument $(INSTRUMENT) --rows $(ROWS)

# Launch the Streamlit UI
ui:
	pipenv run streamlit run src/mirrorbank/ui/app.py

# Security audit: dependency CVEs (pip-audit) + static analysis (bandit, medium+ severity)
audit:
	uv export --no-hashes --no-emit-project --format requirements-txt -o /tmp/mirrorbank-requirements-audit.txt
	uvx pip-audit -r /tmp/mirrorbank-requirements-audit.txt --no-deps --disable-pip
	uvx bandit -r src/ -ll

# Install pre-commit hooks (ruff, secrets scanning, large-file guard)
hooks:
	uvx pre-commit install
