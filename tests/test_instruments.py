"""Tests for instrument schemas and registry."""

import polars as pl
import pytest

from mirrorbank.instruments.ach import ACHSchema
from mirrorbank.instruments.base import ColumnKind
from mirrorbank.instruments.check import CheckSchema
from mirrorbank.instruments.credit_card import CreditCardSchema
from mirrorbank.instruments.debit_card import DebitCardSchema
from mirrorbank.instruments.registry import (
    REGISTRY,
    InstrumentRegistry,
    detect_instrument,
    get_schema,
)
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


def test_instrument_registry_class():
    """InstrumentRegistry wrapper behaves identically to module-level functions."""
    registry = InstrumentRegistry()
    assert set(registry.list_instruments()) == set(REGISTRY.keys())
    assert isinstance(registry.get("ach"), ACHSchema)
    assert isinstance(registry.get("wire"), WireSchema)


def test_instrument_registry_custom_registry():
    """InstrumentRegistry accepts a custom dict for dependency injection."""
    custom = {"ach": ACHSchema}
    reg = InstrumentRegistry(registry=custom)
    assert reg.list_instruments() == ["ach"]
    with pytest.raises(ValueError):
        reg.get("wire")


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


def test_ach_text_columns():
    schema = ACHSchema()
    text = schema.text_columns()
    assert "company_name" in text
    assert "company_entry_desc" in text


def test_ach_reference_columns_have_generators():
    schema = ACHSchema()
    for col in schema.reference_columns():
        assert col.reference_generator is not None, f"{col.name} missing reference_generator"


def test_ach_validate_catches_bad_returns(ach_df):
    """Returned transactions missing return_code must produce a validation error."""
    # Force at least one returned row to ensure the assertion is always exercised
    forced = ach_df.with_columns(
        pl.when(pl.arange(0, len(ach_df)) == 0)
        .then(True)
        .otherwise(pl.col("is_returned"))
        .alias("is_returned"),
        pl.lit(None).cast(pl.String).alias("return_code"),
    )
    errors = ACHSchema().validate(forced)
    assert len(errors) > 0
    assert "return_code" in errors[0]


def test_ach_validate_passes_clean_data(ach_df):
    """Clean fixture (no returned rows with codes) should produce no errors."""
    clean = ach_df.with_columns(pl.lit(False).alias("is_returned"))
    errors = ACHSchema().validate(clean)
    assert errors == []


def test_ach_validate_rejects_zero_amount(ach_df):
    df = ach_df.with_columns(pl.lit(0.0).alias("amount"))
    errors = ACHSchema().validate(df)
    assert any("zero" in e.lower() for e in errors)


# ── Check ─────────────────────────────────────────────────────────────────────

def test_check_has_fraud_label():
    assert CheckSchema.fraud_label == "is_fraud"


def test_check_training_columns():
    schema = CheckSchema()
    training = schema.training_columns()
    assert "amount" in training
    assert "days_to_clear" in training
    assert "payor_account" not in training   # PII


def test_check_text_columns():
    schema = CheckSchema()
    text = schema.text_columns()
    assert "payee_name" in text
    assert "bank_name" in text


def test_check_reference_columns_have_generators():
    schema = CheckSchema()
    for col in schema.reference_columns():
        assert col.reference_generator is not None, f"{col.name} missing reference_generator"


def test_check_validate_catches_bad_returns(check_df):
    """Returned checks without a return_reason must produce a validation error."""
    forced = check_df.with_columns(
        pl.when(pl.arange(0, len(check_df)) == 0)
        .then(True)
        .otherwise(pl.col("is_returned"))
        .alias("is_returned"),
        pl.lit(None).cast(pl.String).alias("return_reason"),
    )
    errors = CheckSchema().validate(forced)
    assert len(errors) > 0


def test_check_validate_passes_clean_data(check_df):
    clean = check_df.with_columns(pl.lit(False).alias("is_returned"))
    errors = CheckSchema().validate(clean)
    assert errors == []


# ── Wire ──────────────────────────────────────────────────────────────────────

def test_wire_has_suspicious_label():
    assert WireSchema.fraud_label == "is_suspicious"


def test_wire_validate_rejects_weekend_volume(wire_df):
    from datetime import datetime

    weekend_dt = datetime(2024, 1, 6, 10, 0)  # Saturday
    weekend_row = wire_df.head(1).with_columns(pl.lit(weekend_dt).alias("timestamp"))
    df_with_weekend = pl.concat([wire_df, weekend_row])
    errors = WireSchema().validate(df_with_weekend)
    assert len(errors) > 0
    assert "weekend" in errors[0].lower()


def test_wire_validate_passes_clean_data(wire_df):
    errors = WireSchema().validate(wire_df)
    assert errors == []


# ── Zelle ─────────────────────────────────────────────────────────────────────

def test_zelle_has_disputed_label():
    assert ZelleSchema.fraud_label == "is_disputed"


def test_zelle_validate_rejects_over_cap():
    df = pl.DataFrame({"amount": [500.0, 2600.0, 100.0], "is_disputed": [False, False, False]})
    errors = ZelleSchema().validate(df)
    assert any("2 500" in e or "cap" in e.lower() for e in errors)


def test_zelle_validate_passes_under_cap():
    df = pl.DataFrame({"amount": [50.0, 500.0, 2499.99], "is_disputed": [False, False, False]})
    errors = ZelleSchema().validate(df)
    assert not any("cap" in e.lower() for e in errors)


def test_zelle_validate_catches_dispute_missing_reason():
    df = pl.DataFrame({
        "amount": [100.0, 200.0],
        "is_disputed": [True, False],
        "dispute_reason": [None, None],
    })
    errors = ZelleSchema().validate(df)
    assert any("dispute_reason" in e for e in errors)


# ── Debit Card ────────────────────────────────────────────────────────────────

def test_debit_card_has_fraud_label():
    assert DebitCardSchema.fraud_label == "is_fraud"


def test_debit_card_training_columns():
    schema = DebitCardSchema()
    training = schema.training_columns()
    assert "amount" in training
    assert "pin_used" in training
    assert "is_overdraft" in training


def test_debit_card_text_columns():
    schema = DebitCardSchema()
    text = schema.text_columns()
    assert "merchant_name" in text


# ── Credit Card ───────────────────────────────────────────────────────────────

def test_credit_card_has_fraud_label():
    assert CreditCardSchema.fraud_label == "is_fraud"


def test_credit_card_training_columns():
    schema = CreditCardSchema()
    training = schema.training_columns()
    assert "amount" in training
    assert "mcc_code" in training
    assert "card_present" in training


def test_credit_card_text_columns():
    schema = CreditCardSchema()
    text = schema.text_columns()
    assert "merchant_name" in text


# ── Auto-detection ────────────────────────────────────────────────────────────

def test_detect_ach(ach_df):
    assert detect_instrument(ach_df) == "ach"


def test_detect_wire(wire_df):
    assert detect_instrument(wire_df) == "wire"


def test_detect_zelle(zelle_df):
    assert detect_instrument(zelle_df) == "zelle"


def test_detect_credit_card(credit_card_df):
    assert detect_instrument(credit_card_df) == "credit_card"


def test_detect_debit_card(debit_card_df):
    assert detect_instrument(debit_card_df) == "debit_card"


def test_detect_check(check_df):
    assert detect_instrument(check_df) == "check"


def test_detect_ambiguous_raises():
    df = pl.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    with pytest.raises(ValueError, match="Cannot auto-detect"):
        detect_instrument(df)
