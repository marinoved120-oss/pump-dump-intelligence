from __future__ import annotations

import pytest

from research.data.binance import BinanceDataError, resolve_symbol, validate_symbols


def test_exact_contract_is_preferred() -> None:
    active = ["KOMAUSDT", "1000PEPEUSDT"]
    assert resolve_symbol("KOMA", active) == "KOMAUSDT"


def test_quantity_prefixed_contract_alias() -> None:
    active = ["KOMAUSDT", "1000PEPEUSDT", "1000SHIBUSDT"]
    assert resolve_symbol("PEPE", active) == "1000PEPEUSDT"
    assert resolve_symbol("pepeusdt", active) == "1000PEPEUSDT"
    assert validate_symbols(["PEPE", "KOMA"], active) == [
        "1000PEPEUSDT",
        "KOMAUSDT",
    ]


def test_unknown_symbol_has_clean_error() -> None:
    with pytest.raises(BinanceDataError, match="Not an active"):
        resolve_symbol("NOTREAL", ["KOMAUSDT", "1000PEPEUSDT"])


def test_ambiguous_scaled_alias_is_rejected() -> None:
    with pytest.raises(BinanceDataError, match="Ambiguous"):
        resolve_symbol("ABC", ["1000ABCUSDT", "1000000ABCUSDT"])
