# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mirrorbank is a differentially private synthetic financial data generator. Given a sensitive tabular dataset (e.g., credit card transactions), it produces a synthetic CSV, a privacy certificate (ε, δ values + MIA audit result), and a utility scorecard (KS tests, TSTR AUC). The hero metric: generate 1M synthetic transactions at ε=3 in under 10 minutes, with a fraud classifier trained on synthetic data reaching ≥90% of real-data AUC, and membership inference attack accuracy below 53%.

## Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit UI
uv run streamlit run src/mirrorbank/ui/app.py

# Run the full evaluation gauntlet on a dataset
make gauntlet DATASET=tabformer SYNTH=dp_tabddpm_eps3

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_budget.py

# Run a single test
uv run pytest tests/test_budget.py::test_name

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Docker (recommended for demo)
docker compose up
```

## Architecture

Mirrorbank is a five-stage pipeline. Each stage produces an artifact consumed by the next.

### Stage 1 — Ingest and profile (`src/mirrorbank/profile/`)
`SchemaProfiler` reads the input dataset and infers column types (continuous, categorical, datetime, free-text, identifier), detects likely PII fields, computes per-column statistics, and builds a correlation graph. The profile is the input contract for all downstream stages.

### Stage 2 — Dual-track generation (`src/mirrorbank/generate/`)
Two models run in parallel because no single architecture handles both numerical and text data well.

- **Track A (DP-TabDDPM):** `tabddpm.py` + `dp_trainer.py`. Handles all structured fields (amounts, timestamps, MCC codes, geographic/account/velocity features). Trained with DP-SGD via Opacus — every gradient step injects calibrated Gaussian noise and logs to the privacy accountant.
- **Track B (LLM narrative engine):** `llm_narrator.py`. Handles free-text fields (merchant names, descriptions). Takes synthetic structured rows from Track A as seeds and prompts the LLM to produce coherent text. **The LLM never sees real data** — it only sees synthetic seeds, so it consumes zero privacy budget. This is a key architectural invariant.

`pipeline.py` orchestrates both tracks and merges their outputs.

### Stage 3 — Privacy budget accountant (`src/mirrorbank/privacy/budget.py`)
`PrivacyBudget` wraps the training loop. Uses Rényi Differential Privacy (RDP) accounting (tighter than basic composition). Every gradient step charges the budget; training halts when ε is exhausted. Supported presets: ε ∈ {1, 3, 5, 10}, δ = 1e-5.

### Stage 4 — Evaluation gauntlet (`src/mirrorbank/evaluate/`)
Three independent test batteries, all run via `gauntlet.py`:
- **Fidelity** (`fidelity.py`): per-column KS tests (target: p > 0.05 for ≥80% of columns), pairwise correlation matrix distance (target: Frobenius norm < 0.15).
- **Utility** (`utility.py`): TSTR — train XGBoost on synthetic, test on held-out real. Target: `AUC_TSTR / AUC_TRTR ≥ 0.90`.
- **Privacy** (`mia_audit.py`): shadow-model membership inference attack (Shokri et al. 2017). Train N=10 shadow generative models, build attack classifier, report attack AUC. Target: ≤ 0.55.

### Stage 5 — Release (`src/mirrorbank/release/`)
Bundles synthetic CSV, privacy certificate PDF, and utility scorecard HTML into a versioned release artifact. The Streamlit UI (`src/mirrorbank/ui/app.py`) provides one-click download.

## Key design decisions

- **RDP accounting over basic composition.** Rényi DP gives tighter ε bounds for iterative mechanisms like DP-SGD. Opacus's RDP accountant is the implementation.
- **LLM outside the privacy boundary.** The narrative engine operates on synthetic seeds only. It does not consume privacy budget. Never pass real records to the LLM.
- **Diffusion over GANs.** TabDDPM outperforms CTGAN/TVAE on mixed-type tabular data and is more stable under DP-SGD (no adversarial training instability).
- **MIA is the minimum privacy audit.** The shadow-model attack is implemented from scratch (not a library call) for interview defensibility and extensibility.

## Datasets

- **IBM TabFormer** (default): 24M credit card transactions. Download instructions in `data/README.md`.
- **Amex Default Prediction**: 5.5M rows, 190 features. Available on Kaggle.
- Raw data goes in `data/raw/` (gitignored). Generated outputs go in `data/synthetic/`.

## Config files

Dataset-specific hyperparameters (batch size, noise multiplier, number of training steps, ε target, column type overrides) live in `configs/tabformer.yaml` and `configs/amex.yaml`.

## Observability

Training runs are tracked in Weights & Biases. Every run logs: privacy budget consumed per step, final (ε, δ), fidelity metrics, TSTR AUC, and MIA attack AUC.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs lint, tests, and the evaluation gauntlet on every PR.
