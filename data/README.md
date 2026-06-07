# Data Directory

Raw datasets go in `data/raw/` and are gitignored. Generated synthetic outputs go in `data/synthetic/` and are also gitignored. Only the `data/sample/` directory (small demo files) is committed to the repo.

---

## IBM TabFormer — Credit Card Transactions (default)

**Size:** 24M rows · **Instrument:** `credit_card`

```bash
# Download from Hugging Face
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='IBM/TabFormer',
    filename='card_transaction.v1.csv',
    local_dir='data/raw/tabformer/'
)
"
```

Place the file at: `data/raw/tabformer/card_transaction.v1.csv`

Config: `configs/tabformer.yaml`

---

## Amex Default Prediction (alternative credit card dataset)

**Size:** 5.5M rows · 190 features · **Instrument:** `credit_card`

```bash
# Requires a Kaggle account
kaggle competitions download -c amex-default-prediction -p data/raw/amex/
unzip data/raw/amex/amex-default-prediction.zip -d data/raw/amex/
```

---

## Synthetic ACH / Wire / Zelle / Check datasets

No public reference dataset exists for these instruments. To generate a synthetic
dataset to test the pipeline, use the sample generator script (planned):

```bash
# Planned — not yet implemented
pipenv run python scripts/generate_sample.py \
  --instrument ach \
  --rows 100000 \
  --output data/sample/ach_100k.csv
```

In the meantime, use the test fixtures in `tests/conftest.py` as reference for
the expected schema of each instrument.

---

## Directory layout

```
data/
├── README.md            # this file
├── sample/              # small demo files committed to repo
├── raw/                 # full datasets — gitignored, never commit
└── synthetic/           # generated outputs — gitignored
```
