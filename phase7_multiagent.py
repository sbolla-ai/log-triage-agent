"""
Phase 7: Level 3 Multi-Agent Incident Response.

Architecture:
  Alert → Orchestrator
            ├── Log Agent       (in parallel)
            └── Runbook Agent   (in parallel)
          → Incident Agent (synthesis)
          → Final incident report

Key Level 3 concepts:
  - Each agent has its own tools, system prompt, and ReAct loop
  - Agents share state only through the orchestrator (Mediator pattern)
  - concurrent.futures.ThreadPoolExecutor gives us parallelism
"""
import re
import concurrent.futures
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()


# =========================================================================
# GUARDRAILS (copied from Phase 6, unchanged)
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
# TOOLS (unchanged — same three log tools + runbook search)
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
        return "\n".join(matches[:50]) if matches else f"No matches in {log_file}"
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
        return f"File: {log_file}\nTotal lines: {total}\nTime range: {tr}\nSeverity: {dict(sev)}"
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
        return "\n".join(errs[-limit:]) if errs else f"No ERRORs in {log_file}"
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
        return "\n\n".join(results) if results else f"No runbook match for '{query}'"
    except Exception as e:
        return f"ERROR: {e}"


# =========================================================================
# TOOL SCHEMAS + REGISTRIES (per-agent isolation)
# =========================================================================

LOG_TOOLS = [
    {"name": "search_logs",
     "description": "Regex search in a log file under ./logs/",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"},
                                     "log_file": {"type": "string"}},
                      "required": ["pattern", "log_file"]}},
    {"name": "get_log_stats",
     "description": "Stats for a log file: counts, time range",
     "input_schema": {"type": "object",
                      "properties": {"log_file": {"type": "string"}},
                      "required": ["log_file"]}},
    {"name": "extract_errors",
     "description": "Recent ERROR lines from a log file",
     "input_schema": {"type": "object",
                      "properties": {"log_file": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["log_file"]}},
]

RUNBOOK_TOOLS = [
    {"name": "search_runbooks",
     "description": "Search company runbooks under ./runbooks/ by keyword",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
]

LOG_TOOL_REGISTRY = {"search_logs": search_logs,
                     "get_log_stats": get_log_stats,
                     "extract_errors": extract_errors}
RUNBOOK_TOOL_REGISTRY = {"search_runbooks": search_runbooks}


# =========================================================================
# GENERIC REACT LOOP — parameterized for any agent
# =========================================================================

def react_loop(user_message, system_prompt, tools_list, tool_registry, max_iter=5):
    """
    Runs a ReAct loop for one agent. Returns just the final text answer.
    Stateless: each call starts fresh. This is perfect for worker agents
    that don't need inter-turn memory — the orchestrator holds conversation state.
    """
    messages = [{"role": "user", "content": user_message}]
    
    for _ in range(max_iter):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=tools_list,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if b.type == "text")
        
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = tool_registry.get(block.name)
                try:
                    result = fn(**block.input) if fn else f"ERROR: unknown tool"
                    is_error = result.startswith("ERROR:")
                except Exception as e:
                    result, is_error = f"ERROR: {e}", True
                tool_results.append({"type": "tool_result",
                                     "tool_use_id": block.id,
                                     "content": result,
                                     "is_error": is_error})
        messages.append({"role": "user", "content": tool_results})
    
    return "Agent hit max iterations without a final answer."


# =========================================================================
# THE THREE AGENTS — each is just a configured react_loop
# =========================================================================

LOG_AGENT_SYSTEM = """\
You are a log analysis specialist. Given an alert description, your job is to:
1. Read the relevant log file using the available tools
2. Identify the error pattern, frequency, and timeline
3. Return a concise factual summary (5-10 lines max)

Do NOT speculate about causes or fixes. Just report what you find.
Do NOT access files outside ./logs/.
"""


def log_agent(alert_description: str) -> str:
    """The log-analysis specialist. One ReAct loop, returns a factual summary."""
    print(f"  [log_agent] starting...")
    result = react_loop(
        user_message=alert_description,
        system_prompt=LOG_AGENT_SYSTEM,
        tools_list=LOG_TOOLS,
        tool_registry=LOG_TOOL_REGISTRY,
    )
    print(f"  [log_agent] done.")
    return result


RUNBOOK_AGENT_SYSTEM = """\
You are a runbook retrieval specialist. Given an alert description, your job is to:
1. Identify the relevant troubleshooting topic
2. Search runbooks using the available tool
3. Return the matching procedure(s) verbatim, with source file named

Do NOT invent procedures not in the runbooks. If no match, say so clearly.
"""


def runbook_agent(alert_description: str) -> str:
    """The runbook specialist. One ReAct loop, returns a procedure."""
    print(f"  [runbook_agent] starting...")
    result = react_loop(
        user_message=alert_description,
        system_prompt=RUNBOOK_AGENT_SYSTEM,
        tools_list=RUNBOOK_TOOLS,
        tool_registry=RUNBOOK_TOOL_REGISTRY,
    )
    print(f"  [runbook_agent] done.")
    return result


INCIDENT_AGENT_SYSTEM = """\
You are an incident commander. You will be given:
- An alert description
- A log analysis from the log-analysis specialist
- A runbook excerpt from the runbook specialist

Your job is to produce a structured incident report with exactly these sections:
1. **Summary** (1-2 sentences)
2. **Evidence** (key log findings)
3. **Recommended actions** (from the runbook)
4. **Severity** (P1/P2/P3) with a one-sentence justification
5. **Escalation path** (who to page, if anyone)

Be concise. No speculation beyond what the inputs support.
"""


def incident_agent(alert: str, log_summary: str, runbook_excerpt: str) -> str:
    """
    The synthesis specialist. No tools — pure reasoning over inputs.
    Takes three strings, returns an incident report.
    """
    print(f"  [incident_agent] synthesizing...")
    combined = (
        f"ALERT: {alert}\n\n"
        f"--- LOG ANALYSIS ---\n{log_summary}\n\n"
        f"--- RUNBOOK EXCERPT ---\n{runbook_excerpt}\n"
    )
    # No tools, so we can use a simpler message call — no ReAct loop needed
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=INCIDENT_AGENT_SYSTEM,
        messages=[{"role": "user", "content": combined}],
    )
    print(f"  [incident_agent] done.")
    return "".join(b.text for b in response.content if b.type == "text")


# =========================================================================
# THE ORCHESTRATOR — the brain of the multi-agent system
# =========================================================================

def orchestrate_incident_response(alert_description: str) -> str:
    """
    Fan-out to log_agent + runbook_agent in parallel, then fan-in to incident_agent.
    This is THE orchestration pattern of Level 3.
    """
    print(f"\n[orchestrator] received alert: {alert_description!r}")
    print(f"[orchestrator] fanning out to log + runbook agents in parallel...")
    
    # ThreadPoolExecutor gives us actual parallelism.
    # Network I/O (Anthropic API calls) is what's slow; threads are perfect here.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        log_future = executor.submit(log_agent, alert_description)
        runbook_future = executor.submit(runbook_agent, alert_description)
        
        log_summary = log_future.result()
        runbook_excerpt = runbook_future.result()
    
    print(f"[orchestrator] both agents returned. Synthesizing...")
    report = incident_agent(alert_description, log_summary, runbook_excerpt)
    print(f"[orchestrator] complete.")
    return report


# =========================================================================
# DRIVER
# =========================================================================

if __name__ == "__main__":
    import time
    
    alerts = [
        "ALERT: Database connection timeouts detected in logs/sample.log at 09:03. "
        "Multiple retries observed. Investigate and produce incident report.",
        
        "ALERT: NullPointerException in UserService.getProfile in logs/sample.log. "
        "Users reporting profile page failures. Produce incident report.",
    ]
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n{'=' * 70}")
        print(f"INCIDENT #{i}")
        print('=' * 70)
        t0 = time.time()
        report = orchestrate_incident_response(alert)
        elapsed = time.time() - t0
        print(f"\n{'--- INCIDENT REPORT ---':^70}")
        print(report)
        print(f"\n[completed in {elapsed:.1f}s]")