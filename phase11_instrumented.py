"""
Phase 11 — Phase 9's self-correcting agent, now instrumented.

Every LLM call is tagged with its role. Run summaries print cost breakdown
by role. Optional JSONL logging to disk for later analysis.
"""
import json
import re
from pathlib import Path
from collections import Counter
from agent_observability import instrumented_create, agent_run


# =========================================================================
# GUARDRAILS + TOOLS — unchanged from Phase 9
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
    "search_logs": {"fn": search_logs, "risk": "low",
                    "schema": {"pattern": "string (regex)", "log_file": "string (path)"},
                    "description": "Regex search in a log file under ./logs/"},
    "get_log_stats": {"fn": get_log_stats, "risk": "low",
                      "schema": {"log_file": "string (path)"},
                      "description": "Stats for a log file: counts, time range"},
    "extract_errors": {"fn": extract_errors, "risk": "low",
                       "schema": {"log_file": "string (path)", "limit": "integer (default 10)"},
                       "description": "Recent ERROR lines from a log file"},
    "search_runbooks": {"fn": search_runbooks, "risk": "low",
                        "schema": {"query": "string"},
                        "description": "Search company runbooks under ./runbooks/"},
    "page_oncall": {"fn": page_oncall, "risk": "high",
                    "schema": {"team": "string", "severity": "string (P1/P2/P3)", "message": "string"},
                    "description": "Page an on-call team via PagerDuty."},
}
ALLOWED_TOOLS = set(TOOL_SPECS.keys())


# =========================================================================
# PLANNER / EVALUATOR / REVISOR / SYNTHESIZER — now instrumented
# =========================================================================

PLANNER_SYSTEM = f"""\
You are an incident-response planner. Given a goal, produce a step-by-step \
plan as VALID JSON: {{"goal": "...", "steps": [{{"id": 1, "tool": "...", "args": {{...}}, "why": "..."}}]}}.
Only use tools from: {sorted(ALLOWED_TOOLS)}. Use EXACT argument names from below.
Return ONLY the JSON, no markdown fences.

Tool reference:
""" + "\n".join(
    f"- {n}({', '.join(f'{k}: {v}' for k, v in s['schema'].items())}): {s['description']} [risk: {s['risk']}]"
    for n, s in TOOL_SPECS.items()
)


def plan(goal):
    response = instrumented_create(
        role="planner",
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
        return False, "missing steps"
    if not plan_dict["steps"]:
        return False, "empty plan"
    for i, step in enumerate(plan_dict["steps"]):
        if not isinstance(step, dict): return False, f"step {i} not dict"
        for k in ("tool", "args"):
            if k not in step: return False, f"step {i} missing {k}"
        if step["tool"] not in ALLOWED_TOOLS:
            return False, f"step {i} unknown tool {step['tool']!r}"
        if not isinstance(step["args"], dict):
            return False, f"step {i} args not dict"
    return True, "OK"


def risk_review(plan_dict):
    risks = [TOOL_SPECS[s["tool"]]["risk"] for s in plan_dict["steps"]]
    max_risk = max(risks, key=lambda r: {"low": 0, "medium": 1, "high": 2}[r])
    if max_risk == "low":
        print("[risk-review] all steps low-risk → auto-approved")
        return True, "auto-approved"
    print(f"\n[risk-review] plan contains {max_risk}-risk steps. Human review required.")
    for i, s in enumerate(plan_dict["steps"]):
        tag = TOOL_SPECS[s["tool"]]["risk"].upper()
        print(f"  Step {i+1}: [{tag:>6}] {s['tool']}({s['args']})")
    while True:
        ans = input("Approve this plan? [y/n]: ").strip().lower()
        if ans == "y": return True, "human-approved"
        if ans == "n": return False, "human-rejected"


EVALUATOR_SYSTEM = """\
You are a strict evaluator. Given a step (tool, args, intent) and its result,
return ONLY {"ok": true|false, "reason": "..."}.
Mark ok=false for technical errors or empty-when-data-expected results.
Mark ok=true if the tool returned usable data addressing the intent, OR if
"not found" is a valid answer for a verify-absence intent.
"""


def evaluate_step(step, result):
    prompt = (f"STEP:\n  tool: {step['tool']}\n  args: {step['args']}\n"
              f"  intent: {step.get('why', '(not stated)')}\n\n"
              f"RESULT:\n{result}\n")
    response = instrumented_create(
        role="evaluator",
        model="claude-haiku-4-5",
        max_tokens=200,
        system=EVALUATOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": True, "reason": "invalid json from evaluator"}


REVISOR_SYSTEM = f"""\
You are a step revisor. A step failed. Rewrite it as a single JSON object:
{{"id": <id>, "tool": "...", "args": {{...}}, "why": "..."}}
Use ONLY: {sorted(ALLOWED_TOOLS)}. Use exact argument names. Do NOT upgrade risk.

Tool reference:
""" + "\n".join(
    f"- {n}({', '.join(f'{k}: {v}' for k, v in s['schema'].items())}): {s['description']} [risk: {s['risk']}]"
    for n, s in TOOL_SPECS.items()
)


def revise_step(step, bad_result, reason):
    prompt = (f"FAILED STEP:\n{json.dumps(step, indent=2)}\n\n"
              f"RESULT:\n{bad_result}\n\nREASON:\n{reason}\n\n"
              f"Provide a revised step.")
    response = instrumented_create(
        role="revisor",
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
        return step


MAX_RETRIES_PER_STEP = 2
MAX_RETRIES_GLOBAL = 5


def execute_one(step):
    fn = TOOL_SPECS[step["tool"]]["fn"]
    try:
        out = fn(**step["args"])
        return out, (isinstance(out, str) and out.startswith("ERROR:"))
    except Exception as e:
        return f"EXCEPTION: {type(e).__name__}: {e}", True


def execute_plan_with_correction(plan_dict):
    results = []
    global_retries = 0
    for original_step in plan_dict["steps"]:
        step = original_step
        attempts = 0
        while True:
            attempts += 1
            print(f"\n[executor] step {step.get('id','?')} attempt {attempts}: "
                  f"{step['tool']}({step['args']})")
            result, is_err = execute_one(step)
            print(f"[executor] result: {result[:200]}{'...' if len(result) > 200 else ''}")
            verdict = evaluate_step(step, result)
            print(f"[evaluator] ok={verdict['ok']} reason={verdict['reason']}")
            if verdict["ok"]:
                results.append({"step": step, "result": result, "attempts": attempts})
                break
            if global_retries >= MAX_RETRIES_GLOBAL or attempts >= MAX_RETRIES_PER_STEP + 1:
                results.append({"step": step, "result": result, "attempts": attempts, "gave_up": True})
                break
            global_retries += 1
            step = revise_step(step, result, verdict["reason"])
            print(f"[revisor] new step: {step['tool']}({step['args']})")
            ok, reason = validate_plan({"steps": [step]})
            if not ok:
                results.append({"step": step, "result": f"ERROR: revised step invalid: {reason}",
                                "attempts": attempts, "gave_up": True})
                break
    return results


SYNTH_SYSTEM = """\
You are an incident commander. Given a goal and step results, write an incident
report with sections: Summary, Evidence (cite steps), Recommended actions,
Severity (P1/P2/P3 with justification), Escalation path.
"""


def synthesize(goal, results):
    text = f"GOAL: {goal}\n\nSTEP RESULTS:\n"
    for r in results:
        s = r["step"]
        g = " (GAVE UP)" if r.get("gave_up") else ""
        text += f"\nStep {s.get('id','?')}{g} — {s['tool']}({s['args']})\n  Result: {r['result']}\n"
    response = instrumented_create(
        role="synthesizer",
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=SYNTH_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def run_autonomous(goal):
    print(f"\n[agent] goal: {goal}")
    plan_dict = plan(goal)
    print(f"[planner] plan:\n{json.dumps(plan_dict, indent=2)}")
    ok, reason = validate_plan(plan_dict)
    if not ok: return f"ABORTED: invalid plan ({reason})"
    approved, note = risk_review(plan_dict)
    if not approved: return f"ABORTED: {note}"
    results = execute_plan_with_correction(plan_dict)
    return synthesize(goal, results)


# =========================================================================
# DRIVER — runs a single incident and prints the cost breakdown
# =========================================================================

if __name__ == "__main__":
    goal = (
        "Investigate database timeouts in logs/sample.log and propose "
        "troubleshooting steps. DO NOT page anyone."
    )
    # The context manager collects telemetry; at exit, it writes one JSONL line
    # to runs.log so you can analyze later.
    with agent_run(label="db_timeout_investigation", log_path="runs.log") as run:
        report = run_autonomous(goal)
        print(f"\n--- INCIDENT REPORT ---\n{report}")
        print(run.summary())