from __future__ import annotations

import asyncio

import pytest

from agentcost.context import current_tags, track, use_tags


def test_no_tags_by_default():
    assert current_tags() == {}


def test_track_sets_tags():
    with track(feature="search", user="alice"):
        tags = current_tags()
        assert tags["feature"] == "search"
        assert tags["user"] == "alice"


def test_tags_cleared_after_scope():
    with track(feature="search"):
        assert current_tags() == {"feature": "search"}
    assert current_tags() == {}


def test_extra_tags():
    with track(feature="x", region="eu", tier="pro"):
        tags = current_tags()
        assert tags["region"] == "eu"
        assert tags["tier"] == "pro"


def test_agent_run_tag():
    with track(agent_run="run-7"):
        assert current_tags()["agent_run"] == "run-7"


def test_nested_scopes_merge():
    with track(feature="search", user="alice"):
        with track(agent_run="run-1"):
            tags = current_tags()
            assert tags == {"feature": "search", "user": "alice", "agent_run": "run-1"}
        # inner scope popped
        assert current_tags() == {"feature": "search", "user": "alice"}


def test_nested_scope_overrides_key():
    with track(feature="search", user="alice"):
        with track(user="bob"):
            assert current_tags()["user"] == "bob"
        assert current_tags()["user"] == "alice"


def test_deeply_nested():
    with track(feature="a"), track(user="u"), track(agent_run="r"):
        assert current_tags() == {"feature": "a", "user": "u", "agent_run": "r"}


def test_current_tags_is_copy():
    with track(feature="x"):
        tags = current_tags()
        tags["feature"] = "mutated"
        assert current_tags()["feature"] == "x"


def test_none_tag_value_rejected():
    with pytest.raises(ValueError), track(custom=None):
        pass


def test_empty_tag_key_rejected():
    with pytest.raises(ValueError), track(**{"": "value"}):
        pass


def test_non_string_tag_coerced():
    with track(run_index=5):
        assert current_tags()["run_index"] == "5"


def test_track_as_decorator():
    @track(feature="summarize", user="carol")
    def do_work() -> dict[str, str]:
        return current_tags()

    result = do_work()
    assert result == {"feature": "summarize", "user": "carol"}
    assert current_tags() == {}


def test_decorator_nested_with_context():
    @track(agent_run="inner")
    def inner() -> dict[str, str]:
        return current_tags()

    with track(feature="outer"):
        assert inner() == {"feature": "outer", "agent_run": "inner"}


def test_use_tags_mapping():
    with use_tags({"feature": "f", "user": "u"}):
        assert current_tags() == {"feature": "f", "user": "u"}
    assert current_tags() == {}


def test_use_tags_merges():
    with track(feature="base"), use_tags({"user": "z"}):
        assert current_tags() == {"feature": "base", "user": "z"}


def test_exception_restores_tags():
    with pytest.raises(RuntimeError), track(feature="x"):
        raise RuntimeError("boom")
    assert current_tags() == {}


def test_async_isolation():
    async def main() -> tuple[dict[str, str], dict[str, str]]:
        async def task(name: str) -> dict[str, str]:
            with track(user=name):
                await asyncio.sleep(0)
                return current_tags()

        return await asyncio.gather(task("alice"), task("bob"))

    results = asyncio.run(main())
    assert {r["user"] for r in results} == {"alice", "bob"}
