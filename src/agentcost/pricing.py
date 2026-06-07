"""Model pricing: dollars per 1M input/output tokens.

The :class:`PriceTable` ships with a sensible built-in default table covering common
frontier and mid-tier models. Prices are expressed as USD per 1,000,000 tokens, which
is how every major vendor publishes them, so the numbers in :data:`DEFAULT_PRICES` map
directly to public price sheets.

Prices change frequently. Treat the built-in table as a convenience starting point and
override or register your own rates with :meth:`PriceTable.register` whenever you need
authoritative numbers.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

# Tokens are billed per million; keeping a named constant avoids magic numbers.
TOKENS_PER_MILLION = 1_000_000


class ModelPrice(BaseModel):
    """Per-1M-token pricing for a single model."""

    model_config = ConfigDict(frozen=True)

    input_per_1m: float = Field(ge=0.0, description="USD per 1M input (prompt) tokens.")
    output_per_1m: float = Field(ge=0.0, description="USD per 1M output (completion) tokens.")

    def cost(self, in_tokens: int, out_tokens: int) -> float:
        """Return the USD cost for the given token counts."""
        if in_tokens < 0 or out_tokens < 0:
            raise ValueError("token counts must be non-negative")
        input_cost = in_tokens / TOKENS_PER_MILLION * self.input_per_1m
        output_cost = out_tokens / TOKENS_PER_MILLION * self.output_per_1m
        return input_cost + output_cost


# Built-in default rates (USD / 1M tokens). Indicative public list prices.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4-turbo": ModelPrice(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-3.5-turbo": ModelPrice(input_per_1m=0.50, output_per_1m=1.50),
    "o1": ModelPrice(input_per_1m=15.00, output_per_1m=60.00),
    "o1-mini": ModelPrice(input_per_1m=1.10, output_per_1m=4.40),
    "claude-3-5-sonnet": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "claude-3-5-haiku": ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
    "claude-3-opus": ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    "claude-3-haiku": ModelPrice(input_per_1m=0.25, output_per_1m=1.25),
    "gemini-1.5-pro": ModelPrice(input_per_1m=1.25, output_per_1m=5.00),
    "gemini-1.5-flash": ModelPrice(input_per_1m=0.075, output_per_1m=0.30),
    "mistral-large": ModelPrice(input_per_1m=2.00, output_per_1m=6.00),
    "llama-3.1-70b": ModelPrice(input_per_1m=0.35, output_per_1m=0.40),
}


class PriceTable:
    """A mutable registry of model prices with a built-in default table.

    Unknown models can either fall back to a configurable default price or raise,
    depending on ``strict``.
    """

    def __init__(
        self,
        prices: Mapping[str, ModelPrice] | None = None,
        *,
        default: ModelPrice | None = None,
        strict: bool = False,
        include_builtins: bool = True,
    ) -> None:
        self._prices: dict[str, ModelPrice] = {}
        if include_builtins:
            self._prices.update(DEFAULT_PRICES)
        if prices:
            self._prices.update(prices)
        self._default = default
        self._strict = strict

    def register(self, model: str, price: ModelPrice) -> None:
        """Add or override the price for ``model``."""
        if not model:
            raise ValueError("model name must be non-empty")
        self._prices[model] = price

    def override(self, model: str, *, input_per_1m: float, output_per_1m: float) -> None:
        """Convenience wrapper around :meth:`register` for raw numbers."""
        self.register(model, ModelPrice(input_per_1m=input_per_1m, output_per_1m=output_per_1m))

    def get(self, model: str) -> ModelPrice:
        """Return the :class:`ModelPrice` for ``model``.

        Falls back to the configured default when the model is unknown and the table is
        not strict; raises :class:`KeyError` otherwise.
        """
        price = self._prices.get(model)
        if price is not None:
            return price
        if self._default is not None and not self._strict:
            return self._default
        raise KeyError(f"unknown model {model!r}; register it or set a default price")

    def has(self, model: str) -> bool:
        """Return ``True`` if ``model`` has an explicit registered price."""
        return model in self._prices

    def cost(self, model: str, in_tokens: int, out_tokens: int) -> float:
        """Return the USD cost of a call to ``model`` with the given token counts."""
        return self.get(model).cost(in_tokens, out_tokens)

    def models(self) -> list[str]:
        """Return the sorted list of known model names."""
        return sorted(self._prices)

    def __contains__(self, model: object) -> bool:
        return isinstance(model, str) and self.has(model)

    def __len__(self) -> int:
        return len(self._prices)
