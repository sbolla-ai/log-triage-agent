# Log Triage Agent

An educational, production-style LLM agent for incident response. Built level-by-level
through an Anthropic-API learning journey, from a basic tool-calling loop to a
self-correcting, RAG-grounded, injection-defended autonomous agent.

## What it does

Given an incident query like "investigate database timeouts in logs/sample.log",
the agent plans a sequence of log-analysis and runbook-retrieval steps, executes
them with self-correction on failure, and produces a cited incident report.

## Architecture at a glance

- **Stateless LLM + stateful orchestration** (Anthropic Claude Sonnet 4.5 + Haiku 4.5)
- **Layered guardrails**: path whitelisting, input validation, untrusted-data framing
- **Structured plans** with risk-based human-in-the-loop approval
- **Self-correcting executor** with evaluator + revisor + retry budgets
- **RAG** with semantic retrieval over markdown runbooks (Voyage `voyage-3.5`)
- **Cost observability** with per-role tracking and JSONL telemetry
- **Three-layer indirect injection defense** (pattern sanitization + framing + output guardrail)
- **Eval harness** with LLM-as-judge scoring

## Project layout

| File | Purpose |
|---|---|
| `phase3_react_loop.py` | ReAct loop baseline (Level 1) |
| `phase6_routing.py` | LLM router + dispatch (Level 2) |
| `phase7_multiagent.py` | Parallel multi-agent orchestration (Level 3) |
| `phase9_self_correct.py` | Self-correcting autonomous planner (Level 4) |
| `phase10_evals.py` | LLM-as-judge eval harness (Level 4) |
| `agent_observability.py` | Per-role cost tracking module |
| `rag_module.py` | RAG: chunking + embedding + retrieval (Level 5) |
| `phase12_rag_agent.py` | Instrumented agent with RAG runbooks |
| `injection_defense.py` | Three-layer prompt-injection defense |
| `phase14_defended_agent.py` | Defended agent (Level 5) |
| `LIMITATIONS.md` | Honest documentation of known gaps |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install anthropic voyageai numpy python-dotenv
```

Create a `.env` file:
Run any phase directly:

```bash
python phase14_defended_agent.py
```

## Level progression (git tags)

- `v1-foundation` — ReAct loop with tool calling
- `v2-routing` — LLM router + memory + guardrails
- `v3-multi-agent` — Parallel fan-out orchestration
- `v4-autonomous` — Planner + HITL + self-correction + evals
- `v5-evolution` — Cost observability + RAG + injection defense

## Known limitations

See `LIMITATIONS.md`. The project is a learning artifact, not a product. Backlog
items (fallback handler, secrets scanning, full production readiness review)
are documented there.
