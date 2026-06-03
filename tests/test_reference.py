"""Tests for reference data generators — check-digit math and format rules."""

import re

import pytest

from mirrorbank.reference.identifiers import (
    generate_check_number,
    generate_imad,
    generate_micr_line,
    generate_trace_number,
)
from mirrorbank.reference.routing_numbers import generate_routing_number, is_valid_routing_number
from mirrorbank.reference.swift_codes import generate_swift_code, is_valid_swift_code


# ── Routing numbers ───────────────────────────────────────────────────────────

def test_routing_number_length():
    for _ in range(50):
        rn = generate_routing_number()
        assert len(rn) == 9, f"Expected 9 digits, got {len(rn)}: {rn!r}"


def test_routing_number_is_digits():
    for _ in range(50):
        rn = generate_routing_number()
        assert rn.isdigit(), f"Non-digit character in {rn!r}"


def test_routing_number_passes_check_digit():
    for _ in range(100):
        rn = generate_routing_number()
        assert is_valid_routing_number(rn), f"Check digit failed for {rn!r}"


def test_routing_number_known_invalid():
    # 000000000 passes the check-digit math (all zeros → checksum 0 → check 0)
    # but wrong length and non-digits are always invalid
    assert not is_valid_routing_number("12345678")   # wrong length (8 digits)
    assert not is_valid_routing_number("12345678x")  # non-digit character
    assert not is_valid_routing_number("1234567890") # wrong length (10 digits)


# ── SWIFT codes ───────────────────────────────────────────────────────────────

def test_swift_code_length():
    for _ in range(50):
        bic = generate_swift_code()
        assert len(bic) == 11, f"Expected 11 chars, got {len(bic)}: {bic!r}"


def test_swift_code_valid_format():
    for _ in range(100):
        bic = generate_swift_code()
        assert is_valid_swift_code(bic), f"Invalid SWIFT format: {bic!r}"


def test_swift_code_country_us():
    for _ in range(20):
        bic = generate_swift_code(country="US")
        assert bic[4:6] == "US", f"Expected US country code, got {bic[4:6]} in {bic!r}"


def test_swift_code_ends_xxx():
    for _ in range(20):
        bic = generate_swift_code()
        assert bic.endswith("XXX"), f"Expected XXX branch, got {bic[-3:]} in {bic!r}"


# ── ACH trace number ──────────────────────────────────────────────────────────

def test_trace_number_length():
    tn = generate_trace_number(routing="021000021", sequence=42)
    assert len(tn) == 15, f"Expected 15 chars: {tn!r}"


def test_trace_number_format():
    # ACH trace uses first 8 digits of routing: "021000021"[:8] = "02100002"
    tn = generate_trace_number(routing="021000021", sequence=42)
    assert tn == "021000020000042"


def test_trace_number_random_valid():
    for _ in range(20):
        tn = generate_trace_number()
        assert len(tn) == 15
        assert tn.isdigit()


# ── Fedwire IMAD ──────────────────────────────────────────────────────────────

def test_imad_format():
    from datetime import date
    imad = generate_imad(txn_date=date(2024, 3, 15), sequence=42)
    assert imad.startswith("20240315"), f"Expected date prefix: {imad!r}"
    assert len(imad) == 8 + 8 + 6  # date + bank_id + sequence


def test_imad_random_valid():
    for _ in range(20):
        imad = generate_imad()
        assert len(imad) == 22


# ── MICR line ─────────────────────────────────────────────────────────────────

def test_micr_line_contains_routing():
    micr = generate_micr_line(routing="021000021", account="1234567890", check_number="1042")
    assert "021000021" in micr


def test_micr_line_structure():
    micr = generate_micr_line(routing="021000021", account="1234567890", check_number="1042")
    assert micr.startswith("|021000021|")
