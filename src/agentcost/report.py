"""Aggregation and reporting.

Roll a :class:`~agentcost.recorder.CostStore` up by any dimension -- ``feature``,
``user``, ``agent_run``, ``model``, or any custom tag -- into per-group totals, then
render the result as a Rich table or extract the top-N spenders.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table

from agentcost.context import AGENT_RUN, FEATURE, USER
from agentcost.recorder import CostRecord

# Records missing the grouping key are bucketed under this label.
UNTAGGED = "(untagged)"


class ReportRow(BaseModel):
    """Aggregated cost for one group (e.g. one feature)."""

    model_config = ConfigDict(frozen=True)

    key: str
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0.0)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens for the group."""
        return self.input_tokens + self.output_tokens


class CostReport(BaseModel):
    """A full aggregation along a single dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    rows: list[ReportRow]

    @property
    def total_cost(self) -> float:
        """Total cost across all rows."""
        return sum(r.cost for r in self.rows)

    @property
    def total_calls(self) -> int:
        """Total number of calls across all rows."""
        return sum(r.calls for r in self.rows)

    def top(self, n: int) -> list[ReportRow]:
        """Return the ``n`` highest-cost rows (already cost-sorted)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.rows[:n]

    def to_table(self, *, title: str | None = None) -> Table:
        """Render this report as a Rich :class:`~rich.table.Table`."""
        table = Table(title=title or f"Cost by {self.dimension}", header_style="bold")
        table.add_column(self.dimension.capitalize(), style="cyan", no_wrap=True)
        table.add_column("Calls", justify="right")
        table.add_column("In tok", justify="right")
        table.add_column("Out tok", justify="right")
        table.add_column("Cost (USD)", justify="right", style="green")
        for row in self.rows:
            table.add_row(
                row.key,
                f"{row.calls:,}",
                f"{row.input_tokens:,}",
                f"{row.output_tokens:,}",
                f"${row.cost:,.6f}",
            )
        table.add_section()
        table.add_row(
            "TOTAL",
            f"{self.total_calls:,}",
            "",
            "",
            f"${self.total_cost:,.6f}",
            style="bold",
        )
        return table


def _key_for(record: CostRecord, dimension: str) -> str:
    """Resolve the grouping key for a record along ``dimension``."""
    if dimension == "model":
        return record.model
    return record.tags.get(dimension) or UNTAGGED


def build_report(
    records: Iterable[CostRecord],
    *,
    by: str = FEATURE,
) -> CostReport:
    """Aggregate ``records`` by the ``by`` dimension into a :class:`CostReport`.

    ``by`` may be ``"model"``, one of the reserved tag names (``feature``, ``user``,
    ``agent_run``), or any custom tag key. Rows are sorted by descending cost, with the
    group key as a stable tiebreaker.
    """
    buckets: dict[str, dict[str, float | int]] = {}
    for record in records:
        key = _key_for(record, by)
        bucket = buckets.setdefault(
            key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        bucket["calls"] = int(bucket["calls"]) + 1
        bucket["input_tokens"] = int(bucket["input_tokens"]) + record.input_tokens
        bucket["output_tokens"] = int(bucket["output_tokens"]) + record.output_tokens
        bucket["cost"] = float(bucket["cost"]) + record.cost

    rows = [
        ReportRow(
            key=key,
            calls=int(b["calls"]),
            input_tokens=int(b["input_tokens"]),
            output_tokens=int(b["output_tokens"]),
            cost=float(b["cost"]),
        )
        for key, b in buckets.items()
    ]
    rows.sort(key=lambda r: (-r.cost, r.key))
    return CostReport(dimension=by, rows=rows)


# Convenience aliases for the common dimensions.
def by_feature(records: Iterable[CostRecord]) -> CostReport:
    """Aggregate by feature."""
    return build_report(records, by=FEATURE)


def by_user(records: Iterable[CostRecord]) -> CostReport:
    """Aggregate by user."""
    return build_report(records, by=USER)


def by_agent_run(records: Iterable[CostRecord]) -> CostReport:
    """Aggregate by agent run."""
    return build_report(records, by=AGENT_RUN)


def by_model(records: Iterable[CostRecord]) -> CostReport:
    """Aggregate by model."""
    return build_report(records, by="model")
