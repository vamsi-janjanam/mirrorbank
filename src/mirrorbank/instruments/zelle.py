"""Zelle P2P payment instrument schema."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema


class ZelleSchema(InstrumentSchema):
    """
    Zelle P2P payment schema.

    Statistical fingerprint:
    - Amount: bimodal — small ($20–$150 social P2P) and medium ($500–$1 500 rent/services).
      Hard cap $2 500 per transaction (most banks).
    - Timing: peak 6 pm–10 pm weekdays, afternoon on weekends. Real-time settlement.
    - Fraud: scam-driven (not account takeover). is_new_recipient=True and
      account_age_days<30 are the top features.
    - ~1.5 % dispute rate; ~40 % of disputes are authorized push payment scams.
    """

    name = "zelle"
    display_name = "Zelle P2P Payment"
    fraud_label = "is_disputed"

    columns = [
        # ── Core ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "amount",
            ColumnKind.CONTINUOUS,
            description="Typically $1–2 500. Hard cap varies by bank.",
        ),
        ColumnSpec(
            "timestamp",
            ColumnKind.DATETIME,
            description="Real-time settlement — no batch. Strong evening/weekend patterns.",
        ),
        ColumnSpec(
            "direction",
            ColumnKind.CATEGORICAL,
            description="'send' or 'receive'",
        ),
        # ── Tokens (hashed phone/email — not raw PII) ─────────────────────────
        ColumnSpec(
            "sender_token",
            ColumnKind.IDENTIFIER,
            description="SHA-256 of sender phone or email — not reversible",
        ),
        ColumnSpec("receiver_token", ColumnKind.IDENTIFIER),
        # ── Context ───────────────────────────────────────────────────────────
        ColumnSpec(
            "memo",
            ColumnKind.FREE_TEXT,
            nullable=True,
            description="'Rent', 'Dinner split', 'Dog walker June'",
        ),
        ColumnSpec(
            "payment_category",
            ColumnKind.CATEGORICAL,
            description="Inferred category: rent, food, utilities, services, personal",
        ),
        ColumnSpec(
            "device_type",
            ColumnKind.CATEGORICAL,
            description="'mobile_ios', 'mobile_android', 'web'",
        ),
        # ── Risk signals ──────────────────────────────────────────────────────
        ColumnSpec(
            "is_new_recipient",
            ColumnKind.CATEGORICAL,
            description="Boolean — first time sending to this receiver. Top fraud signal.",
        ),
        ColumnSpec(
            "recipient_enrolled",
            ColumnKind.CATEGORICAL,
            description="Boolean — receiver has a Zelle account. False = unclaimed payment.",
        ),
        ColumnSpec(
            "sender_txn_count_24h",
            ColumnKind.CONTINUOUS,
            description="# transactions sent by this sender in the last 24 h",
        ),
        ColumnSpec(
            "sender_amount_24h",
            ColumnKind.CONTINUOUS,
            description="Total USD sent by this sender in the last 24 h",
        ),
        ColumnSpec(
            "account_age_days",
            ColumnKind.CONTINUOUS,
            description="Days since Zelle enrollment. <30 days is a strong fraud indicator.",
        ),
        # ── Dispute / fraud ───────────────────────────────────────────────────
        ColumnSpec(
            "is_disputed",
            ColumnKind.CATEGORICAL,
            description="Consumer disputed the transaction",
        ),
        ColumnSpec(
            "dispute_reason",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="unauthorized, scam_impersonation, goods_not_received, wrong_person, other",
        ),
        ColumnSpec(
            "dispute_outcome",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="approved, denied, pending",
        ),
    ]

    def validate(self, df) -> list[str]:
        errors: list[str] = []
        if "amount" in df.columns:
            over_cap = df.filter(df["amount"] > 2500).height
            if over_cap > 0:
                errors.append(
                    f"Zelle: {over_cap} transactions exceed $2 500 hard cap"
                )
        if "is_disputed" in df.columns and "dispute_reason" in df.columns:
            bad_dispute = df.filter(
                (df["is_disputed"]) & (df["dispute_reason"].is_null())
            ).height
            if bad_dispute > 0:
                errors.append(
                    f"Zelle: {bad_dispute} disputed transactions are missing dispute_reason"
                )
        return errors
