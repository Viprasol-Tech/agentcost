<p align="center">
  <img src="docs/assets/logo.png" width="120" alt="agentcost logo">
</p>

<h1 align="center">agentcost</h1>

<p align="center">
  <strong>Zero-infrastructure LLM cost attribution — track spend per feature, per agent-run, and per user without a gateway.</strong>
</p>

<p align="center">
  <em>Built and maintained by <a href="https://viprasol.com">Viprasol Tech</a> — Fintech Experts. Full-Stack Builders.</em>
</p>

<p align="center">
  <a href="https://github.com/Viprasol-Tech/agentcost/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Viprasol-Tech/agentcost/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Viprasol-Tech/agentcost?style=flat-square&color=blue" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-103%20passing-brightgreen?style=flat-square" alt="Tests">
  <a href="https://t.me/viprasol_help"><img src="https://img.shields.io/badge/Telegram-support-26A5E4?style=flat-square&logo=telegram&logoColor=white" alt="Telegram"></a>
</p>

---

You usually discover an LLM cost overrun **when the bill arrives** — and agentic apps burn 5–30× more tokens than a single chat call. The existing fix is to stand up a **gateway/proxy** (LiteLLM, Bifrost). `agentcost` is the opposite: a tiny decorator + context manager you drop into your code that attributes every dollar to a **feature**, an **agent-run**, and a **user** — no proxy, no account, no infra.

## 📊 See where the money goes (real demo output)

```text
                   Cost by feature
| Feature   | Calls | In tok | Out tok | Cost (USD) |
| search    |     4 |  1,535 |     733 |  $0.011038 |
| summarize |     3 |     34 |     381 |  $0.005699 |
| TOTAL     |     7 |        |         |  $0.016737 |

                  Cost by user
| User  | Calls | In tok | Out tok | Cost (USD) |
| alice |     4 |  1,531 |     903 |  $0.013933 |
| bob   |     3 |     38 |     211 |  $0.002804 |
```

```bash
pip install -e .
agentcost demo      # simulated tagged calls -> cost-attribution tables
```

## ✨ Features

- 🏷️ **Tag spend by context** — `with track(feature=..., user=..., agent_run=...)`, nestable, contextvars-based.
- 🎀 **One-line instrumentation** — wrap any LLM call with the `@track_llm` decorator.
- 💵 **Real pricing** — built-in per-model price table (input/output $/1M tokens), fully overridable.
- 🔢 **Token estimation** — heuristic out of the box, or inject a real `Tokenizer`.
- 📈 **Attribution reports** — totals by feature / user / agent-run / model + top-N spenders.
- 🪶 **Zero infra** — no gateway, no proxy, no account; pure Python, MIT.

## 🚀 Usage

```python
from agentcost import track, track_llm, CostStore

store = CostStore()

@track_llm(store=store, model="gpt-4o")
def chat(prompt: str) -> tuple[str, dict]:
    # call your LLM; return (text, {"input_tokens": .., "output_tokens": ..})
    ...

with track(feature="search", user="alice", agent_run="run-001"):
    chat("find me the cheapest flight")

from agentcost.report import build_report
print(build_report(store).by_feature)   # spend attributed to "search"
```

## 🏗️ Architecture

```mermaid
flowchart LR
    CALL[LLM call] -->|@track_llm| REC[CostRecord]
    CTX[track context tags] --> REC
    PRICE[PriceTable] --> REC
    REC --> STORE[CostStore]
    STORE --> REPORT[Reports: feature / user / agent-run / model]
```

## 🗺️ Roadmap

- [x] Decorator + context-manager attribution, pricing table, reports
- [ ] Built-in adapters for OpenAI / Anthropic usage objects
- [ ] Budget alerts + hard caps per tag
- [ ] Export to CSV / Prometheus

## ❓ FAQ

**Does it need a proxy?** No — that's the point. It records in-process.
**Real token counts?** Inject a `Tokenizer`; otherwise a heuristic is used.

> ⭐ **Star agentcost if it helped you find your LLM spend leak.**

## Contact — Viprasol Tech Private Limited

- 🌐 Website: [viprasol.com](https://viprasol.com)
- ✉️ Email: [support@viprasol.com](mailto:support@viprasol.com)
- 💬 Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) · 📱 WhatsApp: +91 96336 52112
- 🐙 GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) · 💼 [LinkedIn](https://www.linkedin.com/in/viprasol/) · 𝕏 [@viprasol](https://twitter.com/viprasol)

> *Viprasol Tech — fintech software, AI agents, algorithmic trading systems, and B2B SaaS. Need a custom build? [Get in touch](mailto:support@viprasol.com).*

## License

[MIT](LICENSE) © 2025 Viprasol Tech Private Limited
