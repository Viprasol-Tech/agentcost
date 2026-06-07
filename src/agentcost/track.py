"""The ``track_llm`` decorator: wrap an LLM call and record its cost.

This is the headline feature. Decorate any function that performs an LLM call and
agentcost will record the spend under whatever tags are active (set via
:func:`agentcost.track`). The decorator works in two modes:

* **Reported usage** -- if the wrapped function returns ``(text, Usage)`` (or a Usage on
  its own), the exact token counts from the provider are used.
* **Estimated usage** -- if the function returns only a string, agentcost estimates the
  output tokens with the heuristic tokenizer and the prompt tokens from the call's
  arguments (when a ``prompt``/``text`` keyword or first positional string is present).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agentcost.context import current_tags
from agentcost.pricing import PriceTable
from agentcost.recorder import CostRecord, CostStore
from agentcost.tokens import HeuristicTokenizer, Tokenizer

# A single process-wide store and price table used when callers do not pass their own.
_default_store = CostStore()
_default_prices = PriceTable()

R = TypeVar("R")


class Usage(BaseModel):
    """Token usage reported by (or estimated for) an LLM call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


def get_default_store() -> CostStore:
    """Return the process-wide default :class:`CostStore`."""
    return _default_store


def reset_default_store() -> None:
    """Clear the process-wide default store (useful in tests)."""
    _default_store.clear()


def _coerce_usage(result: Any, *, tokenizer: Tokenizer, prompt: str | None) -> Usage:
    """Derive a :class:`Usage` from a wrapped function's return value.

    The original return value is preserved by the caller; this only extracts (or
    estimates) token counts from whatever shape the function returned.
    """
    # (text, Usage) tuple -- the recommended return shape.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Usage):
        return result[1]
    # (text, mapping) where mapping carries token counts.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return _usage_from_mapping(result[1])
    # A bare Usage.
    if isinstance(result, Usage):
        return result
    # A mapping with usage keys.
    if isinstance(result, Mapping):
        return _usage_from_mapping(result)
    # A bare string -- estimate input from the prompt and output from the reply.
    if isinstance(result, str):
        in_tokens = tokenizer.count(prompt) if prompt else 0
        return Usage(input_tokens=in_tokens, output_tokens=tokenizer.count(result))
    raise TypeError(
        "track_llm expected the wrapped function to return str, Usage, a mapping, or "
        f"(text, Usage); got {type(result).__name__}"
    )


def _usage_from_mapping(data: Mapping[str, Any]) -> Usage:
    """Extract a :class:`Usage` from a provider-style usage mapping."""
    in_tokens = data.get("input_tokens", data.get("prompt_tokens", 0))
    out_tokens = data.get("output_tokens", data.get("completion_tokens", 0))
    return Usage(input_tokens=int(in_tokens), output_tokens=int(out_tokens))


def _extract_prompt(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str | None:
    """Best-effort extraction of the prompt text for estimating input tokens."""
    for key in ("prompt", "text", "input", "messages", "content"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _flatten_messages(value)
    for arg in args:
        if isinstance(arg, str):
            return arg
    return None


def _flatten_messages(messages: list[Any]) -> str:
    """Flatten a chat ``messages`` list into a single string for estimation."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, str):
            parts.append(msg)
        elif isinstance(msg, Mapping):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def track_llm(
    model: str,
    *,
    store: CostStore | None = None,
    prices: PriceTable | None = None,
    tokenizer: Tokenizer | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator factory that records the cost of each wrapped LLM call.

    Args:
        model: The model name used to price the call.
        store: Where to record cost (defaults to the process-wide store).
        prices: The price table to use (defaults to the built-in table).
        tokenizer: Tokenizer for estimating tokens when usage is not reported.

    The recorded :class:`~agentcost.recorder.CostRecord` is tagged with whatever tags
    are active at call time (set via :func:`agentcost.track`).
    """
    active_store = store if store is not None else _default_store
    active_prices = prices if prices is not None else _default_prices
    active_tokenizer: Tokenizer = tokenizer if tokenizer is not None else HeuristicTokenizer()

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            result = func(*args, **kwargs)
            prompt = _extract_prompt(args, kwargs)
            usage = _coerce_usage(result, tokenizer=active_tokenizer, prompt=prompt)
            cost = active_prices.cost(model, usage.input_tokens, usage.output_tokens)
            active_store.record(
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost=cost,
                tags=current_tags(),
            )
            # Preserve the wrapped function's original return value untouched.
            return result

        return wrapper

    return decorator


def record_call(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    store: CostStore | None = None,
    prices: PriceTable | None = None,
) -> CostRecord:
    """Imperatively record a single call's cost under the active tags.

    A lower-level alternative to :func:`track_llm` for code paths where a decorator does
    not fit (e.g. streaming responses where usage arrives at the end).
    """
    active_store = store if store is not None else _default_store
    active_prices = prices if prices is not None else _default_prices
    cost = active_prices.cost(model, input_tokens, output_tokens)
    return active_store.record(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        tags=current_tags(),
    )
