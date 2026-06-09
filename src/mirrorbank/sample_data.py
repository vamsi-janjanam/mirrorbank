"""Example datasets — tiny, schema-valid synthetic DataFrames per instrument.

Single source of truth for sample data used by the test fixtures
(`tests/conftest.py`), the UI "Load sample" buttons, and the
`scripts/generate_sample_data.py` CSV generator. These are *not* DP-generated —
they are hand-crafted fixtures whose distributions and business rules match each
`InstrumentSchema` (e.g. wires carry zero weekend volume).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timedelta

import polars as pl

from mirrorbank.instruments.registry import REGISTRY


def _random_dt(start: datetime, days: int, rng: random.Random) -> datetime:
    return start + timedelta(days=rng.randint(0, days), hours=rng.randint(0, 23))


# ── Per-instrument builders ─────────────────────────────────────────────────────


def _build_ach(n: int, rng: random.Random) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(10, 5000), 2) for _ in range(n)],
            "transaction_type": [rng.choice(["credit", "debit"]) for _ in range(n)],
            "sec_code": [rng.choice(["PPD", "CCD", "WEB", "TEL"]) for _ in range(n)],
            "settlement_date": [_random_dt(start, 365, rng) for _ in range(n)],
            "effective_date": [_random_dt(start, 365, rng) for _ in range(n)],
            "days_to_settle": [rng.choice([1, 2]) for _ in range(n)],
            "company_name": [
                rng.choice(["Acme Corp", "Netflix", "Employer Inc"]) for _ in range(n)
            ],
            "company_entry_desc": [
                rng.choice(["PAYROLL", "INSURANCE PMT", "SUBSCRIPTION"])
                for _ in range(n)
            ],
            "odfi_routing": ["021000021"] * n,
            "rdfi_routing": ["021000021"] * n,
            "receiver_account_type": [
                rng.choice(["checking", "savings"]) for _ in range(n)
            ],
            "is_returned": [rng.random() < 0.015 for _ in range(n)],
            "return_code": [None] * n,
            "risk_score": [rng.randint(0, 1000) for _ in range(n)],
            "is_fraud": [rng.random() < 0.002 for _ in range(n)],
        }
    )


def _build_wire(n: int, rng: random.Random) -> pl.DataFrame:
    # Wires only settle on weekdays during Fedwire hours (9am-6pm ET).
    # Generate strictly by picking weekday offsets from a Monday baseline.
    base = datetime(2024, 1, 8, 9, 0)  # Monday 2024-01-08, 09:00
    weekday_dates = [
        base + timedelta(weeks=i // 5, days=i % 5, hours=rng.randint(0, 8))
        for i in range(n)
    ]

    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(10_000, 500_000), 2) for _ in range(n)],
            "currency": ["USD"] * n,
            "amount_usd": [round(rng.uniform(10_000, 500_000), 2) for _ in range(n)],
            "timestamp": weekday_dates,
            "settlement_time_mins": [rng.randint(5, 120) for _ in range(n)],
            "network": [rng.choice(["fedwire", "chips"]) for _ in range(n)],
            "wire_type": ["domestic"] * n,
            "originator_name": ["Acme Corp LLC"] * n,
            "originator_country": ["US"] * n,
            "beneficiary_name": ["Vendor Inc"] * n,
            "beneficiary_country": ["US"] * n,
            "imad": [f"20240102BANKID0{i:06d}" for i in range(n)],  # fingerprint column
            "purpose_code": [
                rng.choice(["TRAD", "GDDS", "SVCS", "SALA"]) for _ in range(n)
            ],
            "ofac_screened": [True] * n,
            "aml_alert": [rng.random() < 0.01 for _ in range(n)],
            "is_suspicious": [rng.random() < 0.001 for _ in range(n)],
        }
    )


def _build_zelle(n: int, rng: random.Random) -> pl.DataFrame:
    start = datetime(2024, 1, 1, 18, 0)
    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(5, 500), 2) for _ in range(n)],
            "timestamp": [_random_dt(start, 180, rng) for _ in range(n)],
            "direction": [rng.choice(["send", "receive"]) for _ in range(n)],
            "sender_token": [f"tok_{i:06d}" for i in range(n)],  # fingerprint column
            "receiver_token": [f"tok_{i + 1:06d}" for i in range(n)],
            "payment_category": [
                rng.choice(["rent", "food", "utilities", "personal"]) for _ in range(n)
            ],
            "device_type": [
                rng.choice(["mobile_ios", "mobile_android", "web"]) for _ in range(n)
            ],
            "is_new_recipient": [rng.random() < 0.15 for _ in range(n)],
            "recipient_enrolled": [rng.random() < 0.95 for _ in range(n)],
            "sender_txn_count_24h": [rng.randint(0, 5) for _ in range(n)],
            "sender_amount_24h": [round(rng.uniform(0, 1000), 2) for _ in range(n)],
            "account_age_days": [rng.randint(1, 1500) for _ in range(n)],
            "is_disputed": [rng.random() < 0.015 for _ in range(n)],
            "dispute_reason": [None] * n,
            "dispute_outcome": [None] * n,
        }
    )


def _build_check(n: int, rng: random.Random) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(50, 5000), 2) for _ in range(n)],
            "date_written": [_random_dt(start, 365, rng) for _ in range(n)],
            "date_cleared": [_random_dt(start, 365, rng) for _ in range(n)],
            "days_to_clear": [rng.randint(1, 5) for _ in range(n)],
            "check_type": [
                rng.choice(["personal", "business", "cashier"]) for _ in range(n)
            ],
            "payee_name": [
                rng.choice(["Landlord LLC", "Acme Supplies", "John Smith"])
                for _ in range(n)
            ],
            "bank_name": [
                rng.choice(["First National Bank", "City Bank", "Trust Bank"])
                for _ in range(n)
            ],
            "bank_routing": ["021000021"] * n,
            "check_number": [str(rng.randint(1001, 9999)) for _ in range(n)],
            "micr_line": [
                f"|021000021|  {rng.randint(100000000, 999999999)}  {rng.randint(1001, 9999)}"
                for _ in range(n)
            ],
            "is_returned": [rng.random() < 0.005 for _ in range(n)],
            "return_reason": [None] * n,
            "is_fraud": [rng.random() < 0.003 for _ in range(n)],
            "fraud_type": [None] * n,
        }
    )


def _build_debit_card(n: int, rng: random.Random) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    mccs = ["5411", "5812", "5814", "7011", "5912", None]
    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(1, 300), 2) for _ in range(n)],
            "timestamp": [_random_dt(start, 365, rng) for _ in range(n)],
            "transaction_type": [
                rng.choice(["purchase", "refund", "atm_withdrawal"]) for _ in range(n)
            ],
            "mcc_code": [rng.choice(mccs) for _ in range(n)],
            "merchant_city": [
                rng.choice(["Dallas", "Phoenix", "Seattle"]) for _ in range(n)
            ],
            "merchant_state": [rng.choice(["TX", "AZ", "WA"]) for _ in range(n)],
            "merchant_country": ["US"] * n,
            "entry_mode": [
                rng.choice(["pin", "signature", "contactless", "atm"]) for _ in range(n)
            ],
            "pin_used": [rng.random() < 0.55 for _ in range(n)],  # fingerprint column
            "card_present": [rng.random() < 0.85 for _ in range(n)],
            "is_international": [rng.random() < 0.02 for _ in range(n)],
            "is_overdraft": [rng.random() < 0.03 for _ in range(n)],
            "overdraft_fee": [None] * n,
            "response_code": [rng.choice(["00", "00", "00", "51"]) for _ in range(n)],
            "txn_count_1h": [rng.randint(0, 3) for _ in range(n)],
            "txn_amount_1h": [round(rng.uniform(0, 300), 2) for _ in range(n)],
            "is_fraud": [rng.random() < 0.003 for _ in range(n)],
        }
    )


def _build_credit_card(n: int, rng: random.Random) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    mccs = ["5411", "5812", "5814", "4111", "7011", "5912"]
    return pl.DataFrame(
        {
            "amount": [round(rng.uniform(1, 500), 2) for _ in range(n)],
            "timestamp": [_random_dt(start, 365, rng) for _ in range(n)],
            "transaction_type": [rng.choice(["purchase", "refund"]) for _ in range(n)],
            "mcc_code": [rng.choice(mccs) for _ in range(n)],
            "merchant_city": [
                rng.choice(["New York", "Chicago", "Austin"]) for _ in range(n)
            ],
            "merchant_state": [rng.choice(["NY", "IL", "TX"]) for _ in range(n)],
            "merchant_country": ["US"] * n,
            "entry_mode": [
                rng.choice(["chip", "contactless", "ecommerce"]) for _ in range(n)
            ],
            "card_present": [rng.random() < 0.75 for _ in range(n)],
            "is_international": [rng.random() < 0.05 for _ in range(n)],
            "response_code": [rng.choice(["00", "00", "00", "51"]) for _ in range(n)],
            "txn_count_1h": [rng.randint(0, 4) for _ in range(n)],
            "txn_amount_1h": [round(rng.uniform(0, 500), 2) for _ in range(n)],
            "distinct_merchants_24h": [rng.randint(1, 8) for _ in range(n)],
            "distinct_countries_30d": [rng.randint(1, 3) for _ in range(n)],
            "is_fraud": [rng.random() < 0.002 for _ in range(n)],
        }
    )


# ── Registry: instrument → (builder, default row count) ─────────────────────────

_BUILDERS: dict[str, tuple[Callable[[int, random.Random], pl.DataFrame], int]] = {
    "ach": (_build_ach, 200),
    "wire": (_build_wire, 100),
    "zelle": (_build_zelle, 300),
    "check": (_build_check, 150),
    "debit_card": (_build_debit_card, 400),
    "credit_card": (_build_credit_card, 500),
}


def sample_dataset(
    instrument: str, n: int | None = None, seed: int = 42
) -> pl.DataFrame:
    """Return a schema-valid example DataFrame for the given instrument.

    Args:
        instrument: one of the keys in the instrument registry (e.g. "wire").
        n: number of rows. Defaults to a sensible per-instrument size.
        seed: RNG seed. Use a different seed to get a statistically similar but
            non-identical frame (handy for a "real vs synthetic" fidelity demo).
    """
    if instrument not in _BUILDERS:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown instrument {instrument!r}. Available: {available}"
        )
    builder, default_n = _BUILDERS[instrument]
    rng = random.Random(seed)
    return builder(n if n is not None else default_n, rng)
