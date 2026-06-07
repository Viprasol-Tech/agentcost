"""Contextvars-based tagging.

The whole point of agentcost is that you can answer "how much did *this feature* cost?"
without threading a request id through every function. :func:`track` sets tags in a
:class:`contextvars.ContextVar`, so every LLM call recorded inside its scope (even deep
in the call stack, or in async tasks) inherits those tags.

Scopes are nestable: an inner ``track`` merges its tags on top of the outer scope's
tags, and the previous tags are restored on exit. The implementation is async-safe
because ``contextvars`` copies the context per task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar

# Reserved keys that map onto report dimensions. Any other key is kept as a free-form tag.
FEATURE = "feature"
USER = "user"
AGENT_RUN = "agent_run"

# Empty-mapping default is safe here: callers only ever read it via dict(...) copies and
# every mutation goes through ContextVar.set with a fresh dict, so the default is never
# mutated in place.
_EMPTY_TAGS: dict[str, str] = {}
_tags: ContextVar[dict[str, str]] = ContextVar("agentcost_tags", default=_EMPTY_TAGS)

F = TypeVar("F", bound=Callable[..., Any])


def current_tags() -> dict[str, str]:
    """Return a copy of the tags currently in scope."""
    return dict(_tags.get())


def _normalise(value: object) -> str:
    """Coerce a tag value to a string, rejecting ``None``."""
    if value is None:
        raise ValueError("tag values must not be None")
    return value if isinstance(value, str) else str(value)


def _merge(
    base: Mapping[str, str],
    *,
    feature: str | None,
    user: str | None,
    agent_run: str | None,
    extra: Mapping[str, object],
) -> dict[str, str]:
    merged = dict(base)
    if feature is not None:
        merged[FEATURE] = _normalise(feature)
    if user is not None:
        merged[USER] = _normalise(user)
    if agent_run is not None:
        merged[AGENT_RUN] = _normalise(agent_run)
    for key, value in extra.items():
        if not key:
            raise ValueError("tag keys must be non-empty")
        merged[key] = _normalise(value)
    return merged


class track:  # lowercase reads naturally as a verb at call sites
    """Tag all spend recorded within this scope.

    Usable as both a context manager and a decorator::

        with track(feature="search", user="u-42"):
            ...                       # spend tagged feature=search, user=u-42

        @track(feature="summarize")
        def summarize(doc): ...

    Nested scopes merge: an inner scope's tags override matching keys from the outer
    scope while leaving the rest intact.
    """

    def __init__(
        self,
        *,
        feature: str | None = None,
        user: str | None = None,
        agent_run: str | None = None,
        **tags: object,
    ) -> None:
        self._feature = feature
        self._user = user
        self._agent_run = agent_run
        self._extra = tags
        self._token: Any = None

    def __enter__(self) -> dict[str, str]:
        merged = _merge(
            _tags.get(),
            feature=self._feature,
            user=self._user,
            agent_run=self._agent_run,
            extra=self._extra,
        )
        self._token = _tags.set(merged)
        return merged

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _tags.reset(self._token)
            self._token = None

    def __call__(self, func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with track(
                feature=self._feature,
                user=self._user,
                agent_run=self._agent_run,
                **self._extra,
            ):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


@contextmanager
def use_tags(tags: Mapping[str, object]) -> Iterator[dict[str, str]]:
    """Context manager that merges an arbitrary tag mapping into the current scope."""
    merged = _merge(_tags.get(), feature=None, user=None, agent_run=None, extra=tags)
    token = _tags.set(merged)
    try:
        yield merged
    finally:
        _tags.reset(token)
