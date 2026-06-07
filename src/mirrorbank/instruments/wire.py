"""Wire transfer instrument schema (Fedwire, CHIPS, SWIFT)."""

from __future__ import annotations

from mirrorbank.instruments.base import ColumnKind, ColumnSpec, InstrumentSchema
from mirrorbank.reference.identifiers import generate_imad
from mirrorbank.reference.routing_numbers import generate_routing_number
from mirrorbank.reference.swift_codes import generate_swift_code


class WireSchema(InstrumentSchema):
    """
    Wire transfer schema covering domestic Fedwire, CHIPS, and international SWIFT.

    Statistical fingerprint:
    - Amount: log-normal, very right-skewed. Median $50K–$500K; long tail to $10M+.
    - Volume: very low vs ACH/Zelle — wires are rare but high-value.
    - Timing: strictly weekday 9 am–6 pm ET (Fedwire hours). Zero weekend volume.
    - Fraud: BEC (business email compromise) dominates — a legitimate wire instruction
      intercepted and rerouted at the last step.
    """

    name = "wire"
    display_name = "Wire Transfer"
    fraud_label = "is_suspicious"

    columns = [
        # ── Core ─────────────────────────────────────────────────────────────
        ColumnSpec(
            "amount",
            ColumnKind.CONTINUOUS,
            description="Transfer amount in original currency. Median $50K–$500K.",
        ),
        ColumnSpec(
            "currency",
            ColumnKind.CATEGORICAL,
            description="ISO 4217: 'USD' for domestic; other codes for international",
        ),
        ColumnSpec(
            "amount_usd",
            ColumnKind.CONTINUOUS,
            description="FX-converted amount for cross-currency comparison",
        ),
        ColumnSpec(
            "timestamp",
            ColumnKind.DATETIME,
            description="Fedwire hours: 9 am–6 pm ET weekdays only. No weekend wires.",
        ),
        ColumnSpec(
            "settlement_time_mins",
            ColumnKind.CONTINUOUS,
            description="Fedwire: same-day RTGS, minutes. CHIPS: end-of-day netting.",
        ),
        ColumnSpec(
            "network",
            ColumnKind.CATEGORICAL,
            description="'fedwire' (domestic RTGS), 'chips' (large-value netting), 'swift' (international)",
        ),
        ColumnSpec(
            "wire_type",
            ColumnKind.CATEGORICAL,
            description="'domestic' or 'international'",
        ),
        # ── Originator ────────────────────────────────────────────────────────
        ColumnSpec(
            "originator_name",
            ColumnKind.FREE_TEXT,
            description="Sending entity name — individual or corporate",
        ),
        ColumnSpec(
            "originator_bank_aba",
            ColumnKind.REFERENCE,
            reference_generator=generate_routing_number,
        ),
        ColumnSpec(
            "originator_bank_swift",
            ColumnKind.REFERENCE,
            nullable=True,
            reference_generator=generate_swift_code,
        ),
        ColumnSpec("originator_country", ColumnKind.CATEGORICAL),
        # ── Beneficiary ───────────────────────────────────────────────────────
        ColumnSpec("beneficiary_name", ColumnKind.FREE_TEXT),
        ColumnSpec(
            "beneficiary_bank_aba",
            ColumnKind.REFERENCE,
            nullable=True,
            reference_generator=generate_routing_number,
        ),
        ColumnSpec(
            "beneficiary_bank_swift",
            ColumnKind.REFERENCE,
            nullable=True,
            reference_generator=generate_swift_code,
        ),
        ColumnSpec("beneficiary_country", ColumnKind.CATEGORICAL),
        # ── Fedwire tracking ──────────────────────────────────────────────────
        ColumnSpec(
            "imad",
            ColumnKind.REFERENCE,
            description="Input Message Accountability Data — unique Fedwire identifier",
            reference_generator=generate_imad,
        ),
        # ── Purpose / compliance ──────────────────────────────────────────────
        ColumnSpec(
            "remittance_info",
            ColumnKind.FREE_TEXT,
            description="Payment purpose: 'Invoice 2024-0492', 'Real estate closing', 'Payroll'",
        ),
        ColumnSpec(
            "purpose_code",
            ColumnKind.CATEGORICAL,
            description="ISO 20022 purpose codes: SALA, TRAD, GDDS, SVCS, INTC, etc.",
        ),
        ColumnSpec(
            "ofac_screened",
            ColumnKind.CATEGORICAL,
            description="Boolean — passed OFAC sanctions screening",
        ),
        ColumnSpec(
            "aml_alert",
            ColumnKind.CATEGORICAL,
            description="Boolean — triggered an AML monitoring rule",
        ),
        # ── SAR / fraud signals ───────────────────────────────────────────────
        ColumnSpec(
            "is_suspicious",
            ColumnKind.CATEGORICAL,
            description="Boolean — SAR filed or pending",
        ),
        ColumnSpec(
            "sar_reason",
            ColumnKind.CATEGORICAL,
            nullable=True,
            description="bec_fraud, structuring, layering, unusual_activity, sanctions_hit",
        ),
    ]

    def validate(self, df) -> list[str]:
        """Wires must not settle on weekends (Fedwire is closed)."""
        import polars as pl

        errors: list[str] = []
        try:
            # Polars dt.weekday(): 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun
            weekend_volume = df.filter(
                pl.col("timestamp").dt.weekday().is_in([6, 7])
            ).height
            if weekend_volume > 0:
                errors.append(
                    f"Wire: {weekend_volume} transactions on weekends — Fedwire is closed Sat/Sun"
                )
        except Exception:
            pass  # timestamp column may not exist in test stubs
        return errors
