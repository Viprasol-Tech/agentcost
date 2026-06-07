"""Recording and persisting cost events.

:class:`CostRecord` is one tagged LLM call. :class:`CostStore` is an append-only,
in-memory list of records with JSON load/save so you can persist a run and report on it
later (the CLI ``report`` subcommand reads exactly this format).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CostRecord(BaseModel):
    """A single recorded, tagged LLM call."""

    model_config = ConfigDict(frozen=True)

    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0.0)
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: time.time())

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def tag(self, key: str, default: str = "") -> str:
        """Return the value of tag ``key`` or ``default`` when absent."""
        return self.tags.get(key, default)


class CostStore:
    """An in-memory, append-only collection of :class:`CostRecord` objects."""

    def __init__(self, records: Iterable[CostRecord] | None = None) -> None:
        self._records: list[CostRecord] = list(records) if records is not None else []

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        tags: Mapping[str, str] | None = None,
        timestamp: float | None = None,
    ) -> CostRecord:
        """Create, store, and return a :class:`CostRecord`."""
        resolved_ts = timestamp if timestamp is not None else time.time()
        rec = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            tags=dict(tags) if tags else {},
            timestamp=resolved_ts,
        )
        self._records.append(rec)
        return rec

    def add(self, record: CostRecord) -> None:
        """Append an existing :class:`CostRecord`."""
        self._records.append(record)

    @property
    def records(self) -> list[CostRecord]:
        """Return a shallow copy of the stored records."""
        return list(self._records)

    def total_cost(self) -> float:
        """Return the summed cost across all records."""
        return sum(r.cost for r in self._records)

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CostRecord]:
        return iter(self._records)

    # ------------------------------------------------------------------ JSON I/O
    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the store to a JSON string."""
        payload = [r.model_dump() for r in self._records]
        return json.dumps(payload, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> CostStore:
        """Deserialise a store from a JSON string."""
        raw = json.loads(data)
        if not isinstance(raw, list):
            raise ValueError("expected a JSON array of records")
        return cls(CostRecord.model_validate(item) for item in raw)

    def save(self, path: str | Path, *, indent: int | None = 2) -> None:
        """Write the store to ``path`` as JSON (UTF-8)."""
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CostStore:
        """Read a store from a JSON file at ``path``."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
