"""
agent_observability.py — Cross-cutting instrumentation for agent code.

Primitives:
  - ModelPricing: per-model input/output token pricing
  - LLMCallRecord: one row of telemetry per API call
  - AgentRun: a context manager that accumulates records for one logical run
  - instrumented_create(): drop-in wrapper around client.messages.create()
"""
from __future__ import annotations
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
_client = Anthropic()


# =========================================================================
# PRICING — update these when the API pricing changes
# Cost is $ per 1K tokens for simplicity. Convert from per-1M if needed.
# These match Anthropic's published prices at time of writing.
# =========================================================================

PRICING_PER_1K = {
    "claude-sonnet-4-5":  {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":   {"input": 0.0008, "output": 0.004},
}


def cost_for_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """Returns cost in USD for a single call, or 0.0 if model unknown."""
    p = PRICING_PER_1K.get(model)
    if not p:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1000.0


# =========================================================================
# TELEMETRY DATA STRUCTURES
# =========================================================================

@dataclass
class LLMCallRecord:
    role: str             # e.g. "planner", "evaluator", "synthesizer"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_sec: float
    stop_reason: str


@dataclass
class AgentRun:
    label: str
    start_time: float = field(default_factory=time.time)
    records: list[LLMCallRecord] = field(default_factory=list)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def total_tokens(self) -> tuple[int, int]:
        return (sum(r.input_tokens for r in self.records),
                sum(r.output_tokens for r in self.records))

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def breakdown_by_role(self) -> dict:
        by_role = {}
        for r in self.records:
            b = by_role.setdefault(r.role, {"calls": 0, "cost": 0.0,
                                            "in_tok": 0, "out_tok": 0})
            b["calls"] += 1
            b["cost"] += r.cost_usd
            b["in_tok"] += r.input_tokens
            b["out_tok"] += r.output_tokens
        return by_role

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"RUN: {self.label}",
            f"Elapsed: {self.elapsed():.1f}s | "
            f"Total calls: {len(self.records)} | "
            f"Total cost: ${self.total_cost():.4f}",
            f"{'-' * 60}",
            f"{'ROLE':<18} {'CALLS':>6} {'IN_TOK':>8} {'OUT_TOK':>8} {'COST':>10}",
        ]
        for role, b in sorted(self.breakdown_by_role().items(),
                              key=lambda kv: -kv[1]["cost"]):
            lines.append(f"{role:<18} {b['calls']:>6} "
                         f"{b['in_tok']:>8} {b['out_tok']:>8} "
                         f"${b['cost']:>8.4f}")
        lines.append('=' * 60)
        return "\n".join(lines)


# =========================================================================
# ACTIVE-RUN CONTEXT — thread-local would be better for real concurrency,
# but a module-level reference is fine for our single-threaded flows.
# =========================================================================

_active_run: AgentRun | None = None


@contextmanager
def agent_run(label: str, log_path: str | None = None):
    """
    Context manager: marks a logical run for telemetry aggregation.
    Optionally appends a JSON line to `log_path` at run end.

    Usage:
        with agent_run("incident_response") as run:
            result = run_autonomous(goal)
            print(run.summary())
    """
    global _active_run
    run = AgentRun(label=label)
    prev = _active_run
    _active_run = run
    try:
        yield run
    finally:
        _active_run = prev
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "label": run.label,
                    "elapsed_sec": run.elapsed(),
                    "total_cost_usd": run.total_cost(),
                    "total_records": len(run.records),
                    "breakdown": run.breakdown_by_role(),
                }) + "\n")


# =========================================================================
# THE INSTRUMENTED API CALL — the one function agent code will use
# =========================================================================

def instrumented_create(*, role: str, **kwargs):
    """
    Drop-in for client.messages.create(), with telemetry.
    Agent code uses this instead of client.messages.create() so every
    LLM call gets tagged with its role and accumulated in the active run.

    Example:
        response = instrumented_create(
            role="planner",
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=[...]
        )
    """
    model = kwargs.get("model", "unknown")
    t0 = time.time()
    response = _client.messages.create(**kwargs)
    latency = time.time() - t0

    record = LLMCallRecord(
        role=role,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost_for_call(model,
                                response.usage.input_tokens,
                                response.usage.output_tokens),
        latency_sec=latency,
        stop_reason=response.stop_reason or "unknown",
    )

    if _active_run is not None:
        _active_run.records.append(record)

    return response