"""Render the ``agentcost demo`` output to a colored SVG for the README hero.

This reuses the real CLI demo logic -- the same deterministic simulation, the same
``build_report`` tables, and the same top-spenders panel -- but renders them into a
``rich`` recording console so the image matches actual terminal output. Run with::

    PYTHONPATH=src python docs/make_demo.py
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agentcost.cli import _DIMENSIONS, _simulate
from agentcost.pricing import PriceTable
from agentcost.recorder import CostStore
from agentcost.report import build_report

OUT = Path(__file__).resolve().parent / "assets" / "demo.svg"


def render(console: Console) -> None:
    """Reproduce ``agentcost demo`` output on the given console."""
    store = CostStore()
    prices = PriceTable()
    _simulate(store, prices)

    console.print(
        Panel(
            f"Simulated {len(store)} LLM calls across 2 features and 2 users "
            f"(total ${store.total_cost():,.6f}).",
            title="agentcost demo",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    records = store.records
    for dim in _DIMENSIONS:
        console.print(build_report(records, by=dim).to_table())
        console.print()

    top = build_report(records, by="feature").top(2)
    lines = [f"{i + 1}. {row.key}: ${row.cost:,.6f}" for i, row in enumerate(top)]
    console.print(
        Panel(
            "\n".join(lines),
            title="Top spenders (by feature)",
            border_style="magenta",
            expand=False,
        )
    )


def main() -> None:
    console = Console(record=True, width=100)
    render(console)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT), title="agentcost demo")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
