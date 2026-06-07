from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcost.context import track
from agentcost.pricing import ModelPrice, PriceTable
from agentcost.recorder import CostStore
from agentcost.tokens import HeuristicTokenizer
from agentcost.track import (
    Usage,
    get_default_store,
    record_call,
    reset_default_store,
    track_llm,
)


@pytest.fixture
def store() -> CostStore:
    return CostStore()


@pytest.fixture
def prices() -> PriceTable:
    p = PriceTable(include_builtins=False)
    p.register("test-model", ModelPrice(input_per_1m=1.0, output_per_1m=2.0))
    return p


def test_decorator_records_with_reported_usage(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> tuple[str, Usage]:
        return "hi", Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    text, _ = call()
    assert text == "hi"
    assert len(store) == 1
    rec = store.records[0]
    # 1M in * 1.0 + 1M out * 2.0 = 3.0
    assert rec.cost == pytest.approx(3.0)
    assert rec.input_tokens == 1_000_000


def test_decorator_records_under_active_tags(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> tuple[str, Usage]:
        return "x", Usage(input_tokens=10, output_tokens=10)

    with track(feature="search", user="alice"):
        call()
    assert store.records[0].tags == {"feature": "search", "user": "alice"}


def test_decorator_string_return_estimates(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call(prompt: str) -> str:
        return "abcd" * 10  # 40 chars -> 10 tokens

    out = call("abcd" * 5)  # 20-char prompt -> 5 tokens
    assert out == "abcd" * 10
    rec = store.records[0]
    assert rec.input_tokens == 5
    assert rec.output_tokens == 10


def test_decorator_mapping_usage(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> tuple[str, dict[str, int]]:
        return "text", {"input_tokens": 100, "output_tokens": 200}

    call()
    rec = store.records[0]
    assert rec.input_tokens == 100
    assert rec.output_tokens == 200


def test_decorator_openai_style_usage_keys(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> tuple[str, dict[str, int]]:
        return "text", {"prompt_tokens": 30, "completion_tokens": 40}

    call()
    rec = store.records[0]
    assert rec.input_tokens == 30
    assert rec.output_tokens == 40


def test_decorator_bare_usage(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> Usage:
        return Usage(input_tokens=5, output_tokens=5)

    call()
    assert len(store) == 1


def test_decorator_mapping_return(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> dict[str, int]:
        return {"input_tokens": 7, "output_tokens": 8, "other": 1}

    call()
    rec = store.records[0]
    assert rec.input_tokens == 7
    assert rec.output_tokens == 8


def test_decorator_invalid_return_raises(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> int:
        return 42  # type: ignore[return-value]

    with pytest.raises(TypeError):
        call()


def test_decorator_preserves_metadata(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def my_func() -> Usage:
        """My docstring."""
        return Usage(input_tokens=1, output_tokens=1)

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


def test_decorator_prompt_from_kwarg(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call(*, prompt: str) -> str:
        return "yo"

    call(prompt="a" * 16)  # 16 chars -> 4 tokens
    assert store.records[0].input_tokens == 4


def test_decorator_prompt_from_messages(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call(*, messages: list[dict[str, str]]) -> str:
        return "ok"

    call(messages=[{"role": "user", "content": "a" * 8}])  # 8 chars -> 2 tokens
    assert store.records[0].input_tokens == 2


def test_decorator_custom_tokenizer(store, prices):
    @track_llm("test-model", store=store, prices=prices, tokenizer=HeuristicTokenizer(2))
    def call(prompt: str) -> str:
        return "abcd"  # 4 chars / 2 = 2 tokens

    call("ab")  # 2 chars / 2 = 1 token
    rec = store.records[0]
    assert rec.input_tokens == 1
    assert rec.output_tokens == 2


def test_multiple_calls_accumulate(store, prices):
    @track_llm("test-model", store=store, prices=prices)
    def call() -> Usage:
        return Usage(input_tokens=1_000_000, output_tokens=0)

    call()
    call()
    assert len(store) == 2
    assert store.total_cost() == pytest.approx(2.0)


def test_record_call_imperative(store, prices):
    with track(feature="stream"):
        rec = record_call(
            model="test-model", input_tokens=1_000_000, output_tokens=0, store=store, prices=prices
        )
    assert rec.cost == pytest.approx(1.0)
    assert rec.tags == {"feature": "stream"}
    assert len(store) == 1


def test_default_store_used_when_none():
    reset_default_store()

    @track_llm("gpt-4o")
    def call() -> Usage:
        return Usage(input_tokens=1000, output_tokens=500)

    call()
    assert len(get_default_store()) == 1
    reset_default_store()
    assert len(get_default_store()) == 0


def test_usage_frozen():
    u = Usage(input_tokens=1, output_tokens=1)
    with pytest.raises(ValidationError):
        u.input_tokens = 5  # type: ignore[misc]
