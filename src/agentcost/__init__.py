"""agentcost: zero-infra LLM cost attribution.

Track LLM spend per feature, per agent-run, and per user with nothing more than a
decorator or context manager. No proxy, no gateway, no extra service to run.
"""

from __future__ import annotations

from agentcost.context import current_tags, track
from agentcost.pricing import DEFAULT_PRICES, ModelPrice, PriceTable
from agentcost.recorder import CostRecord, CostStore
from agentcost.report import CostReport, ReportRow, build_report
from agentcost.tokens import HeuristicTokenizer, Tokenizer, estimate_tokens
from agentcost.track import Usage, track_llm

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PRICES",
    "CostRecord",
    "CostReport",
    "CostStore",
    "HeuristicTokenizer",
    "ModelPrice",
    "PriceTable",
    "ReportRow",
    "Tokenizer",
    "Usage",
    "__version__",
    "build_report",
    "current_tags",
    "estimate_tokens",
    "track",
    "track_llm",
]
