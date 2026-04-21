"""
Phase 9 — Level 4 Self-Correcting Executor.

Adds to Phase 8:
  - evaluate_step(): a small LLM call that judges whether a step's result
    is acceptable for its intent. Returns {ok: bool, reason: str}.
  - revise_step(): asks the planner to rewrite a failed step, given the
    error and the evaluator's reason.
  - Executor loop with per-step retries + a global retry budget.

The whole point: the agent recovers from tool failures and weak results
within a single run, without human intervention (for technical failures).
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
# GUARDRAILS, TOOLS, TOOL_SPECS — IDENTICAL TO PHASE 8
# (Copy from phase8_planner.py, no changes)
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


def page_oncall(team, severity, message):
    return f"[SIMULATED PAGE] team={team}, severity={severity}, msg={message!r}"


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
# PLANNER — same as Phase 8
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
- Return ONLY the JSON, no prose before or after, no markdown fences.

Tool reference:
""" + "\n".join(
    f"- {n}({', '.join(f'{k}: {v}' for k, v in s['schema'].items())})\n"
    f"  {s['description']} [risk: {s['risk']}]"
    for n, s in TOOL_SPECS.items()
)


def plan(goal: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": goal}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def validate_plan(plan_dict):
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


def risk_review(plan_dict):
    risks = [TOOL_SPECS[s["tool"]]["risk"] for s in plan_dict["steps"]]
    max_risk = max(risks, key=lambda r: {"low": 0, "medium": 1, "high": 2}[r])
    if max_risk == "low":
        print(f"[risk-review] all steps low-risk → auto-approved")
        return True, "auto-approved"
    print(f"\n[risk-review] plan contains {max_risk}-risk steps. Human review required.")
    print("-" * 70)
    for i, step in enumerate(plan_dict["steps"]):
        tag = TOOL_SPECS[step["tool"]]["risk"].upper()
        print(f"  Step {i + 1}: [{tag:>6}] {step['tool']}({step['args']})")
    print("-" * 70)
    while True:
        ans = input("Approve this plan? [y/n]: ").strip().lower()
        if ans == "y": return True, "human-approved"
        if ans == "n": return False, "human-rejected"


# =========================================================================
# NEW: EVALUATOR — judges whether a step's result is acceptable
# =========================================================================

EVALUATOR_SYSTEM = """\
You are a strict evaluator. You will be given:
- A planned step (tool name, args, intent)
- The actual result returned by executing that step

Judge whether the result successfully fulfilled the step's intent.

Return ONLY valid JSON in this exact shape:
{"ok": true|false, "reason": "<one short sentence>"}

Mark ok=false if:
- The result starts with "ERROR:" or "EXCEPTION:" (technical failure)
- The result is empty, "No matches", or "No match" when the intent expected data
- The result is structurally wrong for what was asked

Mark ok=true if:
- The tool returned real data that addresses the step's intent
- "No matches" is acceptable when the intent was to verify absence

Be lenient: not-finding something can still be a valid answer. Only fail
when the result is broken or unusable.
"""


def evaluate_step(step: dict, result: str) -> dict:
    """Returns {'ok': bool, 'reason': str}. Single small LLM call."""
    prompt = (
        f"STEP:\n  tool: {step['tool']}\n  args: {step['args']}\n"
        f"  intent: {step.get('why', '(not stated)')}\n\n"
        f"RESULT:\n{result}\n"
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=EVALUATOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Defensive: if evaluator emits bad JSON, treat as ok to avoid infinite loops
        return {"ok": True, "reason": f"evaluator returned invalid JSON: {raw}"}


# =========================================================================
# NEW: REVISOR — asks the planner to rewrite a single broken step
# =========================================================================

REVISOR_SYSTEM = f"""\
You are a step revisor. A single step in a plan failed. Your job: rewrite \
just that one step so it has a chance of succeeding.

Return ONLY valid JSON: a single step object in the form:
{{"id": <id>, "tool": "<tool_name>", "args": {{...}}, "why": "<one line>"}}

Constraints:
- Use ONLY tools from: {sorted(ALLOWED_TOOLS)}
- Use EXACT argument names from the tool reference.
- Do not change the tool to a higher-risk one (e.g. don't switch a search to page_oncall).

Tool reference:
""" + "\n".join(
    f"- {n}({', '.join(f'{k}: {v}' for k, v in s['schema'].items())})\n"
    f"  {s['description']} [risk: {s['risk']}]"
    for n, s in TOOL_SPECS.items()
)


def revise_step(step: dict, bad_result: str, eval_reason: str) -> dict:
    """LLM call to rewrite a single failed step."""
    prompt = (
        f"FAILED STEP:\n{json.dumps(step, indent=2)}\n\n"
        f"RESULT IT PRODUCED:\n{bad_result}\n\n"
        f"WHY IT FAILED:\n{eval_reason}\n\n"
        f"Provide a revised step that has a better chance of succeeding."
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        system=REVISOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return step  # fallback: return original; outer loop will give up


# =========================================================================
# NEW: SELF-CORRECTING EXECUTOR
# =========================================================================

MAX_RETRIES_PER_STEP = 2
MAX_RETRIES_GLOBAL = 5


def execute_one(step: dict) -> tuple[str, bool]:
    """Run a single tool call. Returns (result_string, is_error)."""
    tool_name = step["tool"]
    args = step["args"]
    fn = TOOL_SPECS[tool_name]["fn"]
    try:
        out = fn(**args)
        is_err = isinstance(out, str) and out.startswith("ERROR:")
        return out, is_err
    except Exception as e:
        return f"EXCEPTION: {type(e).__name__}: {e}", True


def execute_plan_with_correction(plan_dict: dict) -> list[dict]:
    """
    Execute steps in order. Each step may retry if the evaluator marks it bad.
    Total retries across the run are capped by MAX_RETRIES_GLOBAL.
    """
    results = []
    global_retries = 0

    for original_step in plan_dict["steps"]:
        step = original_step  # may be revised in the loop
        local_attempts = 0

        while True:
            local_attempts += 1
            print(f"\n[executor] step {step.get('id', '?')} attempt {local_attempts}: "
                  f"{step['tool']}({step['args']})")
            result, is_err = execute_one(step)
            preview = (result[:200] + "...") if len(result) > 200 else result
            print(f"[executor] result: {preview}")

            # Evaluator decides if we keep this result
            verdict = evaluate_step(step, result)
            print(f"[evaluator] ok={verdict['ok']} reason={verdict['reason']}")

            if verdict["ok"]:
                results.append({"step": step, "result": result, "attempts": local_attempts})
                break

            # Bad result. Can we retry?
            if global_retries >= MAX_RETRIES_GLOBAL:
                print(f"[executor] global retry budget exhausted ({MAX_RETRIES_GLOBAL}). Giving up.")
                results.append({"step": step, "result": result, "attempts": local_attempts,
                                "gave_up": True})
                break
            if local_attempts >= MAX_RETRIES_PER_STEP + 1:
                print(f"[executor] per-step retries exhausted ({MAX_RETRIES_PER_STEP}). Giving up on this step.")
                results.append({"step": step, "result": result, "attempts": local_attempts,
                                "gave_up": True})
                break

            # Revise and retry
            global_retries += 1
            print(f"[revisor] requesting a corrected step (global retries: {global_retries}/{MAX_RETRIES_GLOBAL})")
            step = revise_step(step, result, verdict["reason"])
            print(f"[revisor] new step: {step['tool']}({step['args']})")

            # Safety: re-validate the revised step against the schema
            ok, reason = validate_plan({"steps": [step]})
            if not ok:
                print(f"[executor] revised step is invalid ({reason}). Giving up.")
                results.append({"step": step, "result": f"ERROR: revised step invalid: {reason}",
                                "attempts": local_attempts, "gave_up": True})
                break

    return results


# =========================================================================
# SYNTHESIZER — same shape as Phase 8
# =========================================================================

SYNTH_SYSTEM = """\
You are an incident commander. Given the user's original goal and the results \
from each executed step, write a concise incident report with these sections:
1. Summary (1-2 sentences)
2. Evidence (cite step numbers)
3. Recommended actions
4. Severity (P1/P2/P3) with one-sentence justification
5. Escalation path

If a step gave_up=True, note that in the Evidence section.
"""


def synthesize(goal, results):
    text = f"GOAL: {goal}\n\nSTEP RESULTS:\n"
    for r in results:
        step = r["step"]
        gave_up = " (GAVE UP)" if r.get("gave_up") else ""
        text += (f"\nStep {step.get('id', '?')}{gave_up} — {step['tool']}({step['args']})\n"
                 f"  Attempts: {r['attempts']}\n  Result: {r['result']}\n")
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


# =========================================================================
# ORCHESTRATOR
# =========================================================================

def run_autonomous(goal: str) -> str:
    print(f"\n{'=' * 70}\n[agent] goal: {goal}\n{'=' * 70}")
    print(f"\n[planner] generating plan...")
    plan_dict = plan(goal)
    print(f"[planner] proposed plan:\n{json.dumps(plan_dict, indent=2)}")

    ok, reason = validate_plan(plan_dict)
    if not ok:
        return f"ABORTED: invalid plan ({reason})"

    approved, note = risk_review(plan_dict)
    if not approved:
        return f"ABORTED by human: {note}"

    results = execute_plan_with_correction(plan_dict)

    print(f"\n[synth] writing final report...")
    return synthesize(goal, results)


# =========================================================================
# DRIVER — including a goal designed to TRIGGER self-correction
# =========================================================================

if __name__ == "__main__":
    goals = [
        # Goal 1: same low-risk investigation, should NOT need any retries
        "Investigate database timeouts in logs/sample.log and propose "
        "troubleshooting steps. DO NOT page anyone.",

        # Goal 2: deliberately likely to trigger self-correction.
        # The runbook search will probably miss on the first phrasing,
        # forcing the revisor to try a different query.
        "Find the runbook procedure for resolving SSL certificate expiry. "
        "If no runbook exists, just say so clearly.",
    ]
    for g in goals:
        print(run_autonomous(g))
        print("\n" + "#" * 70 + "\n")