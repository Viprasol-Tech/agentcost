"""Command-line interface and offline demo.

``agentcost demo`` simulates several tagged LLM calls across two features and two users
with a deterministic fake LLM (no network, no keys), then prints a full cost-attribution
breakdown. ``agentcost report`` re-renders the same breakdowns from a saved JSON store.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from agentcost.context import track
from agentcost.pricing import PriceTable
from agentcost.recorder import CostStore
from agentcost.report import build_report
from agentcost.tokens import estimate_tokens
from agentcost.track import record_call, track_llm

app = typer.Typer(
    name="agentcost",
    help="Zero-infra LLM cost attribution: per feature, per agent-run, per user.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# Dimensions shown by the demo and report commands.
_DIMENSIONS = ("feature", "user", "agent_run", "model")


def fake_llm(prompt: str, *, reply: str) -> tuple[str, dict[str, int]]:
    """A deterministic, offline stand-in for a real LLM call.

    Returns the canned reply plus a usage mapping derived from the prompt and reply, so
    the demo produces stable, reproducible numbers with no network access.
    """
    return reply, {
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": estimate_tokens(reply),
    }


def _simulate(store: CostStore, prices: PriceTable) -> None:
    """Run a deterministic workload that exercises every dimension."""

    @track_llm("gpt-4o", store=store, prices=prices)
    def chat(prompt: str, *, reply: str) -> tuple[str, dict[str, int]]:
        return fake_llm(prompt, reply=reply)

    @track_llm("gpt-4o-mini", store=store, prices=prices)
    def cheap_chat(prompt: str, *, reply: str) -> tuple[str, dict[str, int]]:
        return fake_llm(prompt, reply=reply)

    @track_llm("claude-3-5-sonnet", store=store, prices=prices)
    def summarize(prompt: str, *, reply: str) -> tuple[str, dict[str, int]]:
        return fake_llm(prompt, reply=reply)

    # Feature "search", user alice, one agent run.
    with track(feature="search", user="alice", agent_run="run-001"):
        chat(
            "Find the cheapest flights from NYC to SFO next Friday.", reply="Found 7 options. " * 12
        )
        cheap_chat("Rank those flights by price.", reply="Ranked. " * 6)

    # Feature "search", user bob, another agent run.
    with track(feature="search", user="bob", agent_run="run-002"):
        chat("Search recent papers on retrieval-augmented generation.", reply="Top 5 papers. " * 20)

    # Feature "summarize", user alice.
    with track(feature="summarize", user="alice", agent_run="run-003"):
        summarize("Summarize this 30-page contract for me.", reply="Summary of key clauses. " * 40)

    # Feature "summarize", user bob -- nested scope overrides the model-level tag.
    with track(feature="summarize", user="bob", agent_run="run-004"):
        summarize(
            "Summarize the quarterly earnings call transcript.", reply="Earnings summary. " * 30
        )
        with track(agent_run="run-004-retry"):
            cheap_chat("Shorten that summary to three bullet points.", reply="- A\n- B\n- C\n" * 2)

    # A streaming-style imperative record, also tagged.
    with track(feature="search", user="alice", agent_run="run-005"):
        record_call(
            model="gpt-4o", input_tokens=1500, output_tokens=600, store=store, prices=prices
        )


def _print_reports(store: CostStore) -> None:
    """Render the cost-attribution tables for every dimension."""
    records = store.records
    for dim in _DIMENSIONS:
        report = build_report(records, by=dim)
        console.print(report.to_table())
        console.print()

    # Top spenders across features.
    feature_report = build_report(records, by="feature")
    top = feature_report.top(2)
    lines = [f"{i + 1}. {row.key}: ${row.cost:,.6f}" for i, row in enumerate(top)]
    console.print(
        Panel(
            "\n".join(lines),
            title="Top spenders (by feature)",
            border_style="magenta",
            expand=False,
        )
    )


@app.command()
def demo(
    save: Path | None = typer.Option(
        None, "--save", "-s", help="Write the simulated cost store to this JSON path."
    ),
) -> None:
    """Simulate tagged LLM calls and print the cost-attribution breakdown."""
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
    _print_reports(store)

    if save is not None:
        store.save(save)
        console.print(f"[green]Saved cost store to {save}[/green]")


@app.command()
def report(
    path: Path = typer.Argument(..., help="Path to a JSON cost store saved by `demo --save`."),
    by: str = typer.Option(
        "feature", "--by", "-b", help="Dimension: feature, user, agent_run, or model."
    ),
) -> None:
    """Render a cost-attribution report from a saved JSON store."""
    if not path.exists():
        console.print(f"[red]No such file: {path}[/red]")
        raise typer.Exit(code=1)
    store = CostStore.load(path)
    report_obj = build_report(store.records, by=by)
    console.print(report_obj.to_table())


@app.command()
def models() -> None:
    """List the models in the built-in price table with their rates."""
    prices = PriceTable()
    from rich.table import Table

    table = Table(title="Built-in price table (USD / 1M tokens)", header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    for name in prices.models():
        price = prices.get(name)
        table.add_row(name, f"${price.input_per_1m:,.4f}", f"${price.output_per_1m:,.4f}")
    console.print(table)


if __name__ == "__main__":
    app()
