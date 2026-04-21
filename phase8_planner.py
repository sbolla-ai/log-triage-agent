"""
Phase 8 — Level 4 Autonomous Planner with Risk-Based HITL.

Flow:
  goal → planner LLM → JSON plan → risk review → executor → synthesizer
  
Key concepts:
  - Tools carry a `risk` tag: 'low' (auto), 'medium' (log), 'high' (HITL)
  - The plan is validated against a schema before any execution
  - Unknown tool names are rejected before they reach the executor
  - Each step's output is captured and passed forward as context
"""
import json
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()


# =========================================================================
# GUARDRAILS (same as before)
# =========================================================================

ALLOWED_LOG_DIR = Path("./logs").resolve()
ALLOWED_RUNBOOK_DIR = Path("./runbooks").resolve()


class SecurityError(Exception):
    pass


def validate_log_path(log_file):
    p = Path(log_file).expanduser().resolve()
    if not p.is_relative_to(ALLOWED_LOG_DIR):
        raise SecurityError(f"'{log_file}' outside ./logs/")
    if not p.is_file():
        raise SecurityError(f"'{log_file}' not found")
    if p.suffix not in {".log", ".txt"}:
        raise SecurityError(f"'{p.suffix}' not allowed")
    return p


# =========================================================================
# TOOLS — now with RISK TIERS
# =========================================================================

def search_logs(pattern, log_file):
    try:
        sp = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(sp, "r") as f:
            matches = [f"{i}: {l.rstrip()}" for i, l in enumerate(f, 1)
                       if re.search(pattern, l)]
        return "\n".join(matches[:50]) if matches else "No matches"
    except Exception as e:
        return f"ERROR: {e}"


def get_log_stats(log_file):
    try:
        sp = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        sev, total, ts = Counter(), 0, []
        ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        sv_re = re.compile(r"\b(ERROR|WARN|INFO|DEBUG)\b")
        with open(sp, "r") as f:
            for line in f:
                total += 1
                if m := ts_re.match(line): ts.append(m.group(1))
                if m := sv_re.search(line): sev[m.group(1)] += 1
        tr = f"{ts[0]} to {ts[-1]}" if ts else "unknown"
        return f"Total: {total}, Range: {tr}, Severity: {dict(sev)}"
    except Exception as e:
        return f"ERROR: {e}"


def extract_errors(log_file, limit=10):
    try:
        sp = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(sp, "r") as f:
            errs = [l.rstrip() for l in f if "ERROR" in l]
        return "\n".join(errs[-limit:]) if errs else "No ERRORs"
    except Exception as e:
        return f"ERROR: {e}"


def search_runbooks(query):
    try:
        results = []
        for rb in sorted(ALLOWED_RUNBOOK_DIR.glob("*.md")):
            with open(rb, "r") as f:
                content = f.read()
            for sec in re.split(r"^## ", content, flags=re.MULTILINE):
                if re.search(query, sec, re.IGNORECASE):
                    results.append(f"--- {rb.name} ---\n## {sec.strip()}")
        return "\n\n".join(results) if results else f"No match for '{query}'"
    except Exception as e:
        return f"ERROR: {e}"


def page_oncall(team: str, severity: str, message: str):
    """
    SIMULATED paging. In a real system this would call PagerDuty.
    For Level 4 we just print — but it's classified as high-risk to
    demonstrate the HITL flow.
    """
    # DON'T actually page anyone. This is a simulation.
    return f"[SIMULATED PAGE] team={team}, severity={severity}, msg={message!r}"


# THE IMPORTANT BIT: per-tool risk + metadata
TOOL_SPECS = {
    "search_logs": {
        "fn": search_logs, "risk": "low",
        "schema": {"pattern": "string (regex)", "log_file": "string (path)"},
        "description": "Regex search in a log file under ./logs/",
    },
    "get_log_stats": {
        "fn": get_log_stats, "risk": "low",
        "schema": {"log_file": "string (path)"},
        "description": "Stats for a log file: counts, time range",
    },
    "extract_errors": {
        "fn": extract_errors, "risk": "low",
        "schema": {"log_file": "string (path)", "limit": "integer (default 10)"},
        "description": "Recent ERROR lines from a log file",
    },
    "search_runbooks": {
        "fn": search_runbooks, "risk": "low",
        "schema": {"query": "string"},
        "description": "Search company runbooks under ./runbooks/",
    },
    "page_oncall": {
        "fn": page_oncall, "risk": "high",
        "schema": {"team": "string", "severity": "string (P1/P2/P3)", "message": "string"},
        "description": "Page an on-call team via PagerDuty. USE SPARINGLY.",
    },
}

ALLOWED_TOOLS = set(TOOL_SPECS.keys())


# =========================================================================
# THE PLANNER — LLM call that emits a structured plan
# =========================================================================

PLANNER_SYSTEM = f"""\
You are an incident-response planner. Given a goal, produce a step-by-step \
plan as VALID JSON matching this schema:

{{
  "goal": "<restate the goal>",
  "steps": [
    {{"id": 1, "tool": "<tool_name>", "args": {{...}}, "why": "<one line>"}},
    ...
  ]
}}

Rules:
- Only use tools from this list: {sorted(ALLOWED_TOOLS)}
- Use the EXACT argument names from the tool reference below.
- Keep plans short (usually 2-5 steps).
- A final "synthesize" step is implicit — do NOT include one.
- Return ONLY the JSON, no prose before or after, no markdown fences.

Tool reference:
""" + "\n".join(
    f"- {n}({', '.join(f'{k}: {v}' for k, v in s['schema'].items())})\n"
    f"  {s['description']} [risk: {s['risk']}]"
    for n, s in TOOL_SPECS.items()
)


def plan(goal: str) -> dict:
    """Ask the planner LLM to emit a structured plan. Returns a dict."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": goal}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    # Defensive parsing — strip markdown fences if the model added them
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}\nRaw output:\n{raw}")


# =========================================================================
# PLAN VALIDATION — cheap checks before we execute anything
# =========================================================================

def validate_plan(plan_dict: dict) -> tuple[bool, str]:
    if "steps" not in plan_dict or not isinstance(plan_dict["steps"], list):
        return False, "Plan missing 'steps' list"
    if not plan_dict["steps"]:
        return False, "Plan has no steps"
    for i, step in enumerate(plan_dict["steps"]):
        if not isinstance(step, dict):
            return False, f"Step {i} is not a dict"
        for key in ("tool", "args"):
            if key not in step:
                return False, f"Step {i} missing '{key}'"
        if step["tool"] not in ALLOWED_TOOLS:
            return False, f"Step {i} uses unknown tool: {step['tool']!r}"
        if not isinstance(step["args"], dict):
            return False, f"Step {i} 'args' is not a dict"
    return True, "OK"


# =========================================================================
# RISK-BASED HITL — the heart of Level 4's safety layer
# =========================================================================

def risk_review(plan_dict: dict) -> tuple[bool, str]:
    """
    Returns (approved, reason). Auto-approves low-risk plans.
    Prompts human approval for any plan containing medium-or-high-risk steps.
    """
    risks = [TOOL_SPECS[s["tool"]]["risk"] for s in plan_dict["steps"]]
    max_risk = max(risks, key=lambda r: {"low": 0, "medium": 1, "high": 2}[r])

    if max_risk == "low":
        print(f"[risk-review] all steps low-risk → auto-approved")
        return True, "auto-approved"

    # We have at least one medium/high risk step. Show the plan, prompt.
    print(f"\n[risk-review] plan contains {max_risk}-risk steps. Human review required.")
    print("-" * 70)
    print(json.dumps(plan_dict, indent=2))
    print("-" * 70)
    for i, step in enumerate(plan_dict["steps"]):
        tag = TOOL_SPECS[step["tool"]]["risk"].upper()
        print(f"  Step {i + 1}: [{tag:>6}] {step['tool']}({step['args']})")
    print("-" * 70)

    while True:
        answer = input("Approve this plan? [y/n/abort]: ").strip().lower()
        if answer == "y":
            return True, "human-approved"
        if answer in {"n", "abort"}:
            return False, "human-rejected"


# =========================================================================
# THE EXECUTOR — runs the approved plan step-by-step
# =========================================================================

def execute_plan(plan_dict: dict) -> list[dict]:
    """Runs each step in order. Returns list of {step, result, error} dicts."""
    results = []
    for step in plan_dict["steps"]:
        tool_name = step["tool"]
        args = step["args"]
        fn = TOOL_SPECS[tool_name]["fn"]
        print(f"\n[executor] step {step.get('id', '?')}: {tool_name}({args})")
        try:
            out = fn(**args)
            is_error = isinstance(out, str) and out.startswith("ERROR:")
            results.append({"step": step, "result": out, "error": is_error})
            preview = (out[:200] + "...") if len(out) > 200 else out
            print(f"[executor] result: {preview}")
        except Exception as e:
            results.append({"step": step, "result": f"EXCEPTION: {e}", "error": True})
            print(f"[executor] EXCEPTION: {e}")
    return results


# =========================================================================
# SYNTHESIZER — combines step results into a final answer
# =========================================================================

SYNTH_SYSTEM = """\
You are an incident commander. Given the user's original goal and the results \
from each executed step, write a concise incident report with these sections:

1. Summary (1-2 sentences)
2. Evidence (from the step results)
3. Recommended actions (from runbook results)
4. Severity (P1/P2/P3) with one-sentence justification
5. Escalation path

Cite the step numbers where evidence came from.
"""


def synthesize(goal: str, results: list[dict]) -> str:
    summary_input = f"GOAL: {goal}\n\nSTEP RESULTS:\n"
    for r in results:
        step = r["step"]
        summary_input += (
            f"\nStep {step.get('id', '?')} — {step['tool']}({step['args']})\n"
            f"  Result: {r['result']}\n"
        )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": summary_input}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


# =========================================================================
# THE ORCHESTRATOR — the Level 4 entry point
# =========================================================================

def run_autonomous(goal: str) -> str:
    print(f"\n{'=' * 70}\n[agent] goal: {goal}\n{'=' * 70}")

    # 1. Plan
    print(f"\n[planner] generating plan...")
    plan_dict = plan(goal)
    print(f"[planner] proposed plan:\n{json.dumps(plan_dict, indent=2)}")

    # 2. Validate (schema)
    ok, reason = validate_plan(plan_dict)
    if not ok:
        return f"ABORTED: invalid plan ({reason})"

    # 3. Risk review (HITL if needed)
    approved, note = risk_review(plan_dict)
    if not approved:
        return f"ABORTED by human: {note}"

    # 4. Execute
    results = execute_plan(plan_dict)

    # 5. Synthesize
    print(f"\n[synth] writing final report...")
    return synthesize(goal, results)


# =========================================================================
# DRIVER
# =========================================================================

if __name__ == "__main__":
    goals = [
        # Low-risk: read-only investigation, should auto-approve
        "Investigate database timeouts in logs/sample.log and propose "
        "troubleshooting steps. DO NOT page anyone.",

        # High-risk: asks agent to consider paging, triggers HITL
        "Analyze logs/sample.log for the NullPointerException and, if severity "
        "is P1, page the backend team on-call.",
    ]
    for g in goals:
        print(run_autonomous(g))
        print("\n" + "#" * 70 + "\n")