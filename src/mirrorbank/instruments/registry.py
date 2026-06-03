"""Instrument registry — maps instrument names to schema classes and auto-detects types."""

from __future__ import annotations

from mirrorbank.instruments.ach import ACHSchema
from mirrorbank.instruments.base import InstrumentSchema
from mirrorbank.instruments.check import CheckSchema
from mirrorbank.instruments.credit_card import CreditCardSchema
from mirrorbank.instruments.debit_card import DebitCardSchema
from mirrorbank.instruments.wire import WireSchema
from mirrorbank.instruments.zelle import ZelleSchema

# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, type[InstrumentSchema]] = {
    "ach": ACHSchema,
    "check": CheckSchema,
    "zelle": ZelleSchema,
    "wire": WireSchema,
    "credit_card": CreditCardSchema,
    "debit_card": DebitCardSchema,
}

# Column fingerprints used by detect_instrument() — order matters (more specific first)
_FINGERPRINTS: list[tuple[frozenset[str], str]] = [
    (frozenset({"sec_code", "odfi_routing"}), "ach"),
    (frozenset({"sec_code"}), "ach"),
    (frozenset({"imad"}), "wire"),
    (frozenset({"sender_token", "recipient_enrolled"}), "zelle"),
    (frozenset({"sender_token"}), "zelle"),
    (frozenset({"micr_line"}), "check"),
    (frozenset({"check_number", "payee_name"}), "check"),
    (frozenset({"mcc_code", "cash_advance"}), "credit_card"),
    (frozenset({"mcc_code", "pin_used"}), "debit_card"),
    (frozenset({"mcc_code", "card_present"}), "credit_card"),
]


class InstrumentRegistry:
    """Thin wrapper around the registry dict for dependency injection / testing."""

    def __init__(self, registry: dict[str, type[InstrumentSchema]] | None = None):
        self._registry = registry or REGISTRY

    def get(self, name: str) -> InstrumentSchema:
        return get_schema(name, registry=self._registry)

    def list_instruments(self) -> list[str]:
        return sorted(self._registry.keys())


def get_schema(
    instrument: str,
    registry: dict[str, type[InstrumentSchema]] | None = None,
) -> InstrumentSchema:
    """Return an instantiated schema for the given instrument name."""
    reg = registry or REGISTRY
    if instrument not in reg:
        available = ", ".join(sorted(reg.keys()))
        raise ValueError(
            f"Unknown instrument {instrument!r}. Available: {available}"
        )
    return reg[instrument]()


def detect_instrument(df) -> str:  # df: polars.DataFrame
    """
    Auto-detect instrument type from column names.

    Uses column fingerprints — e.g. 'sec_code' → ACH, 'imad' → wire.
    Pass --instrument explicitly if auto-detection is ambiguous.
    """
    cols = set(df.columns)
    for required_cols, instrument in _FINGERPRINTS:
        if required_cols.issubset(cols):
            return instrument
    raise ValueError(
        "Cannot auto-detect instrument from column names. "
        "Pass --instrument {" + ", ".join(sorted(REGISTRY)) + "} explicitly."
    )
