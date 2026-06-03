"""Credit card transaction instrument schema."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema


class CreditCardSchema(InstrumentSchema):
    """
    Credit card transaction schema.

    Statistical fingerprint:
    - Amount: right-skewed, most transactions <$200.
    - Fraud rate: ~0.17 % (severe class imbalance — handle with oversampling or weighted loss).
    - Entry mode mix: chip ~60 %, ecommerce ~25 %, contactless ~10 %, swipe ~5 %.
    - Card-not-present (CNP/ecommerce) transactions have ~10× higher fraud rate.
    """

    name = "credit_card"
    display_name = "Credit Card Transaction"
    fraud_label = "is_fraud"

    columns = [
        # ── Core ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "amount",
            ColumnKind.CONTINUOUS,
            description="Purchase amount in USD. Right-skewed, most <$200.",
        ),
        ColumnSpec(
            "timestamp",
            ColumnKind.DATETIME,
            description="24/7 — includes nights and weekends",
        ),
        ColumnSpec(
            "transaction_type",
            ColumnKind.CATEGORICAL,
            description="'purchase', 'refund', 'cash_advance', 'balance_transfer'",
        ),
        # ── Merchant ──────────────────────────────────────────────────────────
        ColumnSpec(
            "mcc_code",
            ColumnKind.CATEGORICAL,
            description="Merchant Category Code — ISO 18245 four-digit code",
        ),
        ColumnSpec(
            "merchant_name",
            ColumnKind.FREE_TEXT,
            description="Conditioned on mcc_code by vocab sampler",
        ),
        ColumnSpec("merchant_city", ColumnKind.CATEGORICAL),
        ColumnSpec("merchant_state", ColumnKind.CATEGORICAL),
        ColumnSpec("merchant_country", ColumnKind.CATEGORICAL),
        # ── Channel ───────────────────────────────────────────────────────────
        ColumnSpec(
            "entry_mode",
            ColumnKind.CATEGORICAL,
            description="'chip', 'swipe', 'contactless', 'manual_keyed', 'ecommerce'",
        ),
        ColumnSpec(
            "card_present",
            ColumnKind.CATEGORICAL,
            description="Boolean — CNP (card not present / ecommerce) has ~10× higher fraud rate",
        ),
        ColumnSpec("is_international", ColumnKind.CATEGORICAL),
        # ── Authorization ─────────────────────────────────────────────────────
        ColumnSpec(
            "response_code",
            ColumnKind.CATEGORICAL,
            description="'00'=approved, '05'=do not honor, '14'=invalid card number, '51'=insufficient funds",
        ),
        # ── Velocity features (important fraud signals) ────────────────────────
        ColumnSpec(
            "txn_count_1h",
            ColumnKind.CONTINUOUS,
            description="# transactions on this card in the last 1 hour",
        ),
        ColumnSpec(
            "txn_amount_1h",
            ColumnKind.CONTINUOUS,
            description="Total amount charged on this card in the last 1 hour",
        ),
        ColumnSpec(
            "distinct_merchants_24h",
            ColumnKind.CONTINUOUS,
            description="# unique merchants in the last 24 hours",
        ),
        ColumnSpec(
            "distinct_countries_30d",
            ColumnKind.CONTINUOUS,
            description="# distinct countries in the last 30 days — sudden jump is fraud signal",
        ),
        # ── Fraud ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "is_fraud",
            ColumnKind.CATEGORICAL,
            description="Boolean. ~0.17 % fraud rate — severe class imbalance.",
        ),
    ]
