"""Tests for instrument schemas and registry."""

import pytest

from mirrorbank.instruments.ach import ACHSchema
from mirrorbank.instruments.base import ColumnKind
from mirrorbank.instruments.check import CheckSchema
from mirrorbank.instruments.credit_card import CreditCardSchema
from mirrorbank.instruments.debit_card import DebitCardSchema
from mirrorbank.instruments.registry import REGISTRY, detect_instrument, get_schema
from mirrorbank.instruments.wire import WireSchema
from mirrorbank.instruments.zelle import ZelleSchema


# ── Registry ──────────────────────────────────────────────────────────────────

def test_all_instruments_in_registry():
    expected = {"ach", "check", "zelle", "wire", "credit_card", "debit_card"}
    assert expected == set(REGISTRY.keys())


def test_get_schema_returns_correct_type():
    assert isinstance(get_schema("ach"), ACHSchema)
    assert isinstance(get_schema("wire"), WireSchema)
    assert isinstance(get_schema("zelle"), ZelleSchema)
    assert isinstance(get_schema("check"), CheckSchema)
    assert isinstance(get_schema("credit_card"), CreditCardSchema)
    assert isinstance(get_schema("debit_card"), DebitCardSchema)


def test_get_schema_unknown_raises():
    with pytest.raises(ValueError, match="Unknown instrument"):
        get_schema("bitcoin")


# ── ACH ───────────────────────────────────────────────────────────────────────

def test_ach_has_fraud_label():
    assert ACHSchema.fraud_label == "is_fraud"


def test_ach_training_columns_exclude_pii():
    schema = ACHSchema()
    training = schema.training_columns()
    assert "originator_account" not in training
    assert "receiver_account" not in training
    assert "amount" in training
    assert "sec_code" in training


def test_ach_reference_columns_have_generators():
    schema = ACHSchema()
    for col in schema.reference_columns():
        assert col.reference_generator is not None, f"{col.name} missing reference_generator"


def test_ach_validate_catches_bad_returns(ach_df):
    import polars as pl
    # Force some returned transactions to have null return_code
    bad_df = ach_df.with_columns(
        pl.when(pl.col("is_returned")).then(True).otherwise(False).alias("is_returned"),
        pl.lit(None).cast(pl.Utf8).alias("return_code"),
    )
    schema = ACHSchema()
    errors = schema.validate(bad_df)
    # If any rows have is_returned=True, errors should be non-empty
    returned_count = bad_df.filter(pl.col("is_returned")).height
    if returned_count > 0:
        assert len(errors) > 0


# ── Wire ──────────────────────────────────────────────────────────────────────

def test_wire_has_suspicious_label():
    assert WireSchema.fraud_label == "is_suspicious"


def test_wire_validate_rejects_weekend_volume(wire_df):
    from datetime import datetime, timedelta
    import polars as pl

    # Inject weekend timestamps
    weekend_dt = datetime(2024, 1, 6, 10, 0)  # Saturday
    tampered = wire_df.with_columns(
        pl.concat_list([
            pl.Series("timestamp", [weekend_dt] + [None] * (len(wire_df) - 1))
        ]).explode().alias("timestamp")
    )
    # Use the simpler approach: just add a weekend row
    weekend_row = wire_df.head(1).with_columns(
        pl.lit(weekend_dt).alias("timestamp")
    )
    df_with_weekend = pl.concat([wire_df, weekend_row])
    errors = WireSchema().validate(df_with_weekend)
    assert len(errors) > 0
    assert "weekend" in errors[0].lower()


def test_wire_validate_passes_clean_data(wire_df):
    errors = WireSchema().validate(wire_df)
    assert errors == []


# ── Zelle ─────────────────────────────────────────────────────────────────────

def test_zelle_validate_rejects_over_cap():
    import polars as pl

    df = pl.DataFrame({"amount": [500.0, 2600.0, 100.0], "is_disputed": [False, False, False]})
    errors = ZelleSchema().validate(df)
    assert any("2 500" in e or "cap" in e.lower() for e in errors)


def test_zelle_validate_passes_under_cap():
    import polars as pl

    df = pl.DataFrame({"amount": [50.0, 500.0, 2499.99], "is_disputed": [False, False, False]})
    errors = ZelleSchema().validate(df)
    assert not any("cap" in e.lower() for e in errors)


# ── Auto-detection ────────────────────────────────────────────────────────────

def test_detect_ach(ach_df):
    assert detect_instrument(ach_df) == "ach"


def test_detect_wire(wire_df):
    assert detect_instrument(wire_df) == "wire"


def test_detect_zelle(zelle_df):
    assert detect_instrument(zelle_df) == "zelle"


def test_detect_credit_card(credit_card_df):
    assert detect_instrument(credit_card_df) == "credit_card"


def test_detect_ambiguous_raises():
    import polars as pl

    df = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        detect_instrument(df)
