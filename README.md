# Log Triage Agent

> **A RAG + tool-use agent that turns a firehose of raw logs into a short, ranked list of hypotheses an on-call engineer can actually act on.** Point it at your log store, hand it an incident window, and get back the top likely causes with cited log lines, related past incidents, and suggested next queries — in seconds, not the twenty minutes it takes a human to grep.

[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#roadmap)

---

## Why this exists

Every on-call engineer has done the same thing at 3 a.m.: paste a log line into Slack, grep for "error" across five services, scroll, guess. The information needed to triage is almost always *in the logs* — it's just spread across millions of lines and a dozen tribal-knowledge patterns no one wrote down. This agent codifies that muscle memory: retrieve the right slice of logs, cluster the anomalies, cross-reference past incidents, and hand the human a ranked hypothesis list with evidence attached.

It is deliberately narrow. It doesn't page anyone, doesn't remediate, doesn't pretend to root-cause code bugs. It shortens the *reading* step of incident response — the step humans are worst at under stress.

## What it does

- **Ingests** logs from Loki, Elasticsearch, CloudWatch, or plain files via pluggable adapters
- **Retrieves** the relevant window with hybrid search: BM25 for exact tokens (error codes, stack frames) plus embeddings for semantic recall ("connection reset" ≈ "peer closed")
- **Clusters** anomalies with log-template mining (Drain3) so a million noisy lines collapse into a handful of patterns
- **Cross-references** against an incident knowledge base of past post-mortems and runbooks
- **Reasons** with a bounded tool-use loop — the LLM issues follow-up queries against the log store rather than hallucinating what "probably" happened
- **Outputs** a ranked hypothesis list: each hypothesis carries a confidence score, cited log lines, linked past incidents, and a suggested next query

## Results

> Numbers from the eval harness in [`eval/`](eval/). See [`EVAL_RESULTS.md`](EVAL_RESULTS.md) for methodology and per-scenario breakdowns.

| Metric | Baseline (human grep) | With agent | Delta |
|---|---|---|---|
| Median time to first hypothesis | _fill in_ | _fill in_ | _fill in_ |
| Correct cause in top-3 | _fill in_ | _fill in_ | _fill in_ |
| Log lines a human must read | _fill in_ | _fill in_ | _fill in_ |
| Cost per triage (USD) | — | _fill in_ | — |

## Architecture

```text
 log store ──▶ adapter ──▶ retriever (BM25 + embeddings) ──┐
                                                            ├─▶ bounded tool-use loop ──▶ ranked hypotheses
 Drain3 template miner ─────────────────────────────────────┤       │                          + cited evidence
 incident KB (past post-mortems, runbooks) ─────────────────┘       │                          + suggested next query
                                                                    └─ tools: search_logs, fetch_template, related_incidents
```

Full write-up in [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Design principles

1. **Retrieve, don't invent.** The LLM never proposes a hypothesis without cited log lines.
2. **Templates over raw lines.** Reasoning happens over Drain3 templates, not the million-line firehose — the LLM sees signal, not noise.
3. **Bounded reasoning.** Wall-clock, token, and tool-call caps on every run. No runaway agents.
4. **Pluggable log backend.** Loki today, Elasticsearch/CloudWatch tomorrow. Adapter is ~100 lines.
5. **The eval harness is the contract.** Every retrieval or prompt change is measured against replayed incidents.

## Quickstart

```bash
git clone https://github.com/sbolla-ai/log-triage-agent.git
cd log-triage-agent
uv sync
cp .env.example .env         # OPENAI_API_KEY, log-store endpoint
uv run log-triage demo       # runs against the bundled synthetic outage
```

Triage a real window:

```bash
uv run log-triage run \
  --service checkout-api \
  --from "2026-07-12T14:00Z" \
  --to   "2026-07-12T14:20Z"
```

Run the eval harness:

```bash
uv run log-triage eval --suite basic --model gpt-4o
```

## Roadmap

- [x] Loki adapter + hybrid retriever
- [x] Drain3 template miner
- [x] Bounded tool-use loop with evidence citations
- [x] Eval harness with replayable incidents
- [ ] Elasticsearch and CloudWatch adapters
- [ ] Slack bot: `/triage <service> <window>`
- [ ] Feedback loop: on-call marks a hypothesis correct → becomes a golden eval case

## About the author

I'm **Sreenivas Bolla** — Principal SRE & Platform Engineer with 16+ years building production platforms at **JPMorgan Chase, Capital One, and SMBC**. I've shipped internal developer platforms, SLO governance systems, and Agentic AI in production — and I'm one of fewer than 50 engineers worldwide running MCP against real customer-facing systems, where it's cut MTTR by 75%.

`log-triage-agent` is the narrow, boring, high-leverage piece of that stack — the one that saves an on-call engineer twenty minutes of scrolling at 3 a.m.

- 🌐 [sbolla.dev](https://sbolla.dev)
- 💼 [linkedin.com/in/sreenivas-bolla](https://linkedin.com/in/sreenivas-bolla)
- ✉️ sbolla.tx@gmail.com

## License

MIT — see [LICENSE](LICENSE).