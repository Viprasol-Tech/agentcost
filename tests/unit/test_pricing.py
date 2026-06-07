from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcost.pricing import DEFAULT_PRICES, ModelPrice, PriceTable


def test_modelprice_cost_hand_computed_gpt4o():
    # gpt-4o: $2.50/1M in, $10.00/1M out.
    # 1000 in -> 1000/1e6 * 2.50 = 0.0025 ; 500 out -> 500/1e6 * 10 = 0.005
    price = ModelPrice(input_per_1m=2.50, output_per_1m=10.00)
    assert price.cost(1000, 500) == pytest.approx(0.0075)


def test_modelprice_cost_hand_computed_round_million():
    price = ModelPrice(input_per_1m=3.0, output_per_1m=15.0)
    # exactly 1M in + 1M out -> 3 + 15 = 18
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(18.0)


def test_modelprice_cost_zero_tokens():
    price = ModelPrice(input_per_1m=2.5, output_per_1m=10.0)
    assert price.cost(0, 0) == 0.0


def test_modelprice_cost_input_only():
    price = ModelPrice(input_per_1m=10.0, output_per_1m=30.0)
    assert price.cost(2_000_000, 0) == pytest.approx(20.0)


def test_modelprice_negative_tokens_raise():
    price = ModelPrice(input_per_1m=2.5, output_per_1m=10.0)
    with pytest.raises(ValueError):
        price.cost(-1, 0)


def test_modelprice_is_frozen():
    price = ModelPrice(input_per_1m=1.0, output_per_1m=2.0)
    with pytest.raises(ValidationError):
        price.input_per_1m = 5.0  # type: ignore[misc]


def test_pricetable_builtin_lookup():
    table = PriceTable()
    assert table.has("gpt-4o")
    assert table.get("gpt-4o").input_per_1m == DEFAULT_PRICES["gpt-4o"].input_per_1m


def test_pricetable_cost_matches_modelprice():
    table = PriceTable()
    expected = DEFAULT_PRICES["claude-3-5-sonnet"].cost(1000, 1000)
    assert table.cost("claude-3-5-sonnet", 1000, 1000) == pytest.approx(expected)


def test_pricetable_register_and_override():
    table = PriceTable(include_builtins=False)
    table.register("my-model", ModelPrice(input_per_1m=1.0, output_per_1m=2.0))
    assert table.cost("my-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    table.override("my-model", input_per_1m=5.0, output_per_1m=5.0)
    assert table.cost("my-model", 1_000_000, 0) == pytest.approx(5.0)


def test_pricetable_override_builtin():
    table = PriceTable()
    table.override("gpt-4o", input_per_1m=0.0, output_per_1m=0.0)
    assert table.cost("gpt-4o", 10_000, 10_000) == 0.0


def test_pricetable_unknown_raises_when_no_default():
    table = PriceTable()
    with pytest.raises(KeyError):
        table.get("does-not-exist")


def test_pricetable_unknown_uses_default():
    default = ModelPrice(input_per_1m=1.0, output_per_1m=1.0)
    table = PriceTable(default=default)
    assert table.get("unknown-model") is default
    assert table.cost("unknown-model", 1_000_000, 0) == pytest.approx(1.0)


def test_pricetable_strict_ignores_default():
    default = ModelPrice(input_per_1m=1.0, output_per_1m=1.0)
    table = PriceTable(default=default, strict=True)
    with pytest.raises(KeyError):
        table.get("unknown-model")


def test_pricetable_register_empty_name_raises():
    table = PriceTable()
    with pytest.raises(ValueError):
        table.register("", ModelPrice(input_per_1m=1.0, output_per_1m=1.0))


def test_pricetable_contains_and_len():
    table = PriceTable(include_builtins=False)
    assert len(table) == 0
    assert "gpt-4o" not in table
    table.register("gpt-4o", ModelPrice(input_per_1m=1.0, output_per_1m=1.0))
    assert "gpt-4o" in table
    assert len(table) == 1


def test_pricetable_models_sorted():
    table = PriceTable(include_builtins=False)
    table.register("b", ModelPrice(input_per_1m=1.0, output_per_1m=1.0))
    table.register("a", ModelPrice(input_per_1m=1.0, output_per_1m=1.0))
    assert table.models() == ["a", "b"]


def test_pricetable_custom_prices_override_builtins():
    custom = {"gpt-4o": ModelPrice(input_per_1m=99.0, output_per_1m=99.0)}
    table = PriceTable(custom)
    assert table.get("gpt-4o").input_per_1m == 99.0
