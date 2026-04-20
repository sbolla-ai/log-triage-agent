"""
Phase 6: Level 2 Routing.
Adds an LLM-based intent classifier that dispatches to specialized handlers.

Architecture:
  user_input
    → Guardrail 2 (input validation)
      → classify_intent() LLM call → returns a category string
        → dispatch via ROUTER dict:
           - 'log_analysis'  → log_triage_handler (uses 3 log tools)
           - 'runbook'       → runbook_handler    (uses 1 runbook tool)
           - 'unknown'       → fallback_handler   (no LLM call, decline)
"""
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()


# =========================================================================
# GUARDRAIL 1: Path whitelisting (unchanged from Phase 4)
# =========================================================================

ALLOWED_LOG_DIR = Path("./logs").resolve()
ALLOWED_RUNBOOK_DIR = Path("./runbooks").resolve()


class SecurityError(Exception):
    pass


def validate_log_path(log_file: str) -> Path:
    requested = Path(log_file).expanduser().resolve()
    if not requested.is_relative_to(ALLOWED_LOG_DIR):
        raise SecurityError(f"Access denied: '{log_file}' outside ./logs/")
    if not requested.is_file():
        raise SecurityError(f"Access denied: '{log_file}' does not exist")
    if requested.suffix not in {".log", ".txt"}:
        raise SecurityError(f"Access denied: '{requested.suffix}' not allowed")
    return requested


def validate_runbook_path(runbook_file: str) -> Path:
    requested = Path(runbook_file).expanduser().resolve()
    if not requested.is_relative_to(ALLOWED_RUNBOOK_DIR):
        raise SecurityError(f"Access denied: '{runbook_file}' outside ./runbooks/")
    if not requested.is_file():
        raise SecurityError(f"Access denied: '{runbook_file}' does not exist")
    if requested.suffix not in {".md", ".txt"}:
        raise SecurityError(f"Access denied: '{requested.suffix}' not allowed")
    return requested


# =========================================================================
# LOG TOOLS (unchanged from Phase 5)
# =========================================================================

def search_logs(pattern: str, log_file: str) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(safe_path, "r") as f:
            matches = [f"{i}: {l.rstrip()}" for i, l in enumerate(f, start=1)
                       if re.search(pattern, l)]
        return "\n".join(matches[:50]) if matches else f"No matches in {log_file}."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def get_log_stats(log_file: str) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        severities, total, timestamps = Counter(), 0, []
        ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        sev_re = re.compile(r"\b(ERROR|WARN|INFO|DEBUG)\b")
        with open(safe_path, "r") as f:
            for line in f:
                total += 1
                if m := ts_re.match(line): timestamps.append(m.group(1))
                if m := sev_re.search(line): severities[m.group(1)] += 1
        tr = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"
        return (f"File: {log_file}\nTotal lines: {total}\nTime range: {tr}\n"
                f"Severity counts: {dict(severities)}")
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def extract_errors(log_file: str, limit: int = 10) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(safe_path, "r") as f:
            errors = [l.rstrip() for l in f if "ERROR" in l]
        return "\n".join(errors[-limit:]) if errors else f"No ERRORs in {log_file}."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =========================================================================
# RUNBOOK TOOL — new for this phase
# =========================================================================

def search_runbooks(query: str) -> str:
    """Search all runbooks for lines matching the query (case-insensitive)."""
    try:
        results = []
        for rb in sorted(ALLOWED_RUNBOOK_DIR.glob("*.md")):
            with open(rb, "r") as f:
                content = f.read()
            # Return sections (separated by ##) that mention the query
            sections = re.split(r"^## ", content, flags=re.MULTILINE)
            for sec in sections:
                if re.search(query, sec, re.IGNORECASE):
                    results.append(f"--- {rb.name} ---\n## {sec.strip()}")
        return "\n\n".join(results) if results else f"No runbook sections matched '{query}'."
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# =========================================================================
# TOOL DISPATCH — each handler has its OWN tool registry
# =========================================================================

LOG_TOOLS = [
    {"name": "search_logs", "description": "Regex search in a log file. For finding specific patterns.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}, "log_file": {"type": "string"}},
                      "required": ["pattern", "log_file"]}},
    {"name": "get_log_stats", "description": "Stats for a log file: counts, time range.",
     "input_schema": {"type": "object", "properties": {"log_file": {"type": "string"}},
                      "required": ["log_file"]}},
    {"name": "extract_errors", "description": "Recent ERROR lines from a log file.",
     "input_schema": {"type": "object",
                      "properties": {"log_file": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["log_file"]}},
]

RUNBOOK_TOOLS = [
    {"name": "search_runbooks",
     "description": "Search company runbooks (under ./runbooks/) for a topic. "
                    "Use for questions about how to fix or investigate infrastructure issues.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string",
                                               "description": "Keyword or phrase to find in runbooks"}},
                      "required": ["query"]}},
]

LOG_TOOL_REGISTRY = {"search_logs": search_logs, "get_log_stats": get_log_stats,
                     "extract_errors": extract_errors}
RUNBOOK_TOOL_REGISTRY = {"search_runbooks": search_runbooks}


# =========================================================================
# GUARDRAIL 2: Input validation
# =========================================================================

MAX_INPUT_LENGTH = 2000
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
]


def validate_user_input(user_message: str):
    if not user_message or not user_message.strip():
        return False, "Empty input."
    if len(user_message) > MAX_INPUT_LENGTH:
        return False, f"Input too long ({len(user_message)} chars)."
    for p in INJECTION_PATTERNS:
        if p.search(user_message):
            return False, f"Matched pattern: {p.pattern}"
    return True, "OK"


# =========================================================================
# THE CLASSIFIER — a small LLM call that returns ONLY a category string
# =========================================================================

VALID_CATEGORIES = {"log_analysis", "runbook", "unknown"}

CLASSIFIER_SYSTEM_PROMPT = """\
You are an intent classifier. Your ONLY job is to read the user's message and \
return EXACTLY one of these three strings:

- log_analysis    : The user wants to analyze, search, or summarize log files.
- runbook         : The user wants to know how to fix something, troubleshoot \
infrastructure, or find documented procedures.
- unknown         : Anything else. Off-topic, unclear, or outside your scope.

Return ONLY the single category string. No prose, no punctuation, no quotes.
"""


def classify_intent(user_message: str, history: list | None = None) -> str:
    """Single LLM call, returns a category string from VALID_CATEGORIES."""
    # We pass history too, so follow-ups like "Tell me more" can be classified
    # in context. The classifier sees the same context the handlers will.
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=20,             # category strings are short
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=messages,
    )
    
    raw = "".join(b.text for b in response.content if b.type == "text").strip().lower()
    # Defensive parse: only accept known categories; fall back to 'unknown'.
    if raw in VALID_CATEGORIES:
        return raw
    return "unknown"


# =========================================================================
# HANDLERS — specialized agents per category
# =========================================================================

LOG_SYSTEM_PROMPT = """\
You are a log triage assistant. Your ONLY job is to help users analyze log files \
under ./logs/ using the tools provided. Never access files outside ./logs/.
If a tool returns an error, relay it clearly.
"""

RUNBOOK_SYSTEM_PROMPT = """\
You are a runbook assistant. Your ONLY job is to help users find information from \
company runbooks under ./runbooks/ using the search_runbooks tool. Quote relevant \
sections directly. Never invent procedures not in the runbooks.
"""


def _react_loop(user_message, history, system_prompt, tools_list, tool_registry, max_iter=5):
    """Generic ReAct loop. Same body as before; parameterized for reuse."""
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})
    
    for _ in range(max_iter):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            tools=tools_list,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            answer = "".join(b.text for b in response.content if b.type == "text")
            return answer, messages
        
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = tool_registry.get(block.name)
                if fn is None:
                    result, is_error = f"ERROR: unknown tool {block.name}", True
                else:
                    try:
                        result = fn(**block.input)
                        is_error = result.startswith("ERROR:")
                    except Exception as e:
                        result, is_error = f"ERROR: {type(e).__name__}: {e}", True
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": result, "is_error": is_error})
        messages.append({"role": "user", "content": tool_results})
    
    return "Agent stopped: hit max iterations.", messages


def log_triage_handler(user_message, history):
    return _react_loop(user_message, history, LOG_SYSTEM_PROMPT,
                       LOG_TOOLS, LOG_TOOL_REGISTRY)


def runbook_handler(user_message, history):
    return _react_loop(user_message, history, RUNBOOK_SYSTEM_PROMPT,
                       RUNBOOK_TOOLS, RUNBOOK_TOOL_REGISTRY)


def fallback_handler(user_message, history):
    """No LLM call. Deterministic decline."""
    decline = (
        "I can help with two things: analyzing log files under ./logs/, and "
        "looking up troubleshooting steps in the runbooks under ./runbooks/. "
        "Your question doesn't seem to fit either. Could you rephrase?"
    )
    # We still append to history so the conversation stays coherent
    new_history = list(history or [])
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": decline})
    return decline, new_history


# =========================================================================
# THE ROUTER — dict-based dispatch (the answer from Q3)
# =========================================================================

ROUTER = {
    "log_analysis": log_triage_handler,
    "runbook":      runbook_handler,
    "unknown":      fallback_handler,
}


# =========================================================================
# THE TOP-LEVEL AGENT
# =========================================================================

def run_agent(user_message: str, history: list | None = None):
    """
    The full Level 2 flow: guardrail → classify → dispatch → return.
    """
    # Guardrail 2
    is_ok, reason = validate_user_input(user_message)
    if not is_ok:
        return f"[Guardrail 2 rejected input: {reason}]", history or []
    
    # Classify
    category = classify_intent(user_message, history)
    print(f"[router] classified as: {category}")
    
    # Dispatch (dict lookup, with fallback_handler as the default)
    handler = ROUTER.get(category, fallback_handler)
    return handler(user_message, history)


# =========================================================================
# DRIVER — scripted conversation that exercises both handlers
# =========================================================================

if __name__ == "__main__":
    history = []
    turns = [
        "What errors are in logs/sample.log?",           # → log_analysis
        "Which one looks most urgent?",                  # → log_analysis (follow-up)
        "How do I troubleshoot database timeouts?",      # → runbook
        "What about nginx 502 errors?",                  # → runbook
        "What's the capital of France?",                 # → unknown (fallback)
    ]
    
    for i, q in enumerate(turns, start=1):
        print(f"\n{'=' * 70}")
        print(f"TURN {i} — USER: {q}")
        print("=" * 70)
        answer, history = run_agent(q, history)
        print(f"\nAGENT: {answer}")
        print(f"\n[history now has {len(history)} messages]")