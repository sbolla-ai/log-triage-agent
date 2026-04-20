"""
Phase 5: Level 2 — Memory.
Same guardrails as Phase 4, but now run_agent() is a pure transformation:
takes messages in, returns messages out. The caller owns the conversation.
"""
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()


# =========================================================================
# GUARDRAIL 1: PATH WHITELISTING — unchanged from Phase 4
# =========================================================================

ALLOWED_LOG_DIR = Path("./logs").resolve()


class SecurityError(Exception):
    pass


def validate_log_path(log_file: str) -> Path:
    requested = Path(log_file).expanduser().resolve()
    if not requested.is_relative_to(ALLOWED_LOG_DIR):
        raise SecurityError(
            f"Access denied: '{log_file}' is outside the allowed log directory."
        )
    if not requested.is_file():
        raise SecurityError(f"Access denied: '{log_file}' does not exist.")
    if requested.suffix not in {".log", ".txt"}:
        raise SecurityError(
            f"Access denied: '{requested.suffix}' files are not allowed."
        )
    return requested


# =========================================================================
# TOOL IMPLEMENTATIONS — unchanged from Phase 4
# =========================================================================

def search_logs(pattern: str, log_file: str) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(safe_path, "r") as f:
            matches = []
            for i, line in enumerate(f, start=1):
                if re.search(pattern, line):
                    matches.append(f"{i}: {line.rstrip()}")
        if not matches:
            return f"No matches for '{pattern}' in {log_file}."
        return "\n".join(matches[:50])
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def get_log_stats(log_file: str) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        severities = Counter()
        total = 0
        timestamps = []
        ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        sev_re = re.compile(r"\b(ERROR|WARN|INFO|DEBUG)\b")
        with open(safe_path, "r") as f:
            for line in f:
                total += 1
                if m := ts_re.match(line):
                    timestamps.append(m.group(1))
                if m := sev_re.search(line):
                    severities[m.group(1)] += 1
        time_range = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"
        return (
            f"File: {log_file}\nTotal lines: {total}\n"
            f"Time range: {time_range}\nSeverity counts: {dict(severities)}"
        )
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def extract_errors(log_file: str, limit: int = 10) -> str:
    try:
        safe_path = validate_log_path(log_file)
    except SecurityError as e:
        return f"ERROR: {e}"
    try:
        with open(safe_path, "r") as f:
            errors = [line.rstrip() for line in f if "ERROR" in line]
        if not errors:
            return f"No ERROR lines found in {log_file}."
        return "\n".join(errors[-limit:])
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


TOOL_REGISTRY = {
    "search_logs": search_logs,
    "get_log_stats": get_log_stats,
    "extract_errors": extract_errors,
}


# =========================================================================
# TOOL SCHEMAS — unchanged
# =========================================================================

TOOLS = [
    {
        "name": "search_logs",
        "description": (
            "Search a log file for a regex pattern. Returns matching lines "
            "with line numbers. Use for finding specific things in logs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "log_file": {"type": "string", "description": "Path to log file"},
            },
            "required": ["pattern", "log_file"],
        },
    },
    {
        "name": "get_log_stats",
        "description": (
            "Returns total line count, severity counts, and time range for "
            "a log file. Use for overviews and summaries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {"type": "string", "description": "Path to log file"},
            },
            "required": ["log_file"],
        },
    },
    {
        "name": "extract_errors",
        "description": (
            "Returns the most recent ERROR-level lines. Use when asked "
            "about errors, failures, or what went wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {"type": "string", "description": "Path to log file"},
                "limit": {"type": "integer", "description": "Max errors (default 10)"},
            },
            "required": ["log_file"],
        },
    },
]


# =========================================================================
# GUARDRAILS 2 & 3 — unchanged
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
    for pattern in INJECTION_PATTERNS:
        if pattern.search(user_message):
            return False, f"Matched suspicious pattern: {pattern.pattern}"
    return True, "OK"


SYSTEM_PROMPT = """\
You are a log triage assistant. Your ONLY job is to help users analyze log \
files that live under the ./logs/ directory using the tools provided.

Rules:
- Never try to access files outside ./logs/.
- Never attempt to execute shell commands or restart services.
- If the user asks for something outside log analysis, politely decline.
- If a tool returns an error, relay it clearly instead of guessing.
"""


# =========================================================================
# THE REACT LOOP — now a pure transformation on messages
# =========================================================================

MAX_HISTORY_MESSAGES = 20  # Growth policy: cap the list


def prune_history(messages: list) -> list:
    """
    Keep at most MAX_HISTORY_MESSAGES. Critical rule: never split a
    tool_use from its tool_result. If we'd land mid-pair, drop one more.
    """
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    # Simple approach: keep the last N, but ensure we don't start mid-pair.
    # A tool_result message (role=user, content is list of tool_result blocks)
    # must always follow a tool_use. If the first kept message is a tool_result,
    # drop one more.
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    while trimmed and _starts_with_orphan_tool_result(trimmed[0]):
        trimmed = trimmed[1:]
    return trimmed


def _starts_with_orphan_tool_result(msg: dict) -> bool:
    """A tool_result as the first message has nothing to reply to — drop it."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def run_agent(user_message: str, history: list | None = None, max_iterations: int = 5):
    """
    Run one conversational turn through the ReAct loop.
    
    Args:
        user_message: what the user just said
        history: prior messages from earlier turns (None = fresh conversation)
        max_iterations: safety cap on tool-calling loop
    
    Returns:
        (answer_text, updated_history) — caller should reuse updated_history
        for the next turn.
    """
    # Guardrail 2: validate input
    is_ok, reason = validate_user_input(user_message)
    if not is_ok:
        return (f"[Guardrail 2 rejected input: {reason}]", history or [])

    # Start from the caller's history, plus this turn's user message.
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Append Claude's final answer to history before returning
            messages.append({"role": "assistant", "content": response.content})
            answer = "".join(b.text for b in response.content if b.type == "text")
            return (answer, prune_history(messages))

        # Tool use — append Claude's response, execute tools, append results
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_REGISTRY.get(block.name)
                if fn is None:
                    result = f"ERROR: unknown tool {block.name}"
                    is_error = True
                else:
                    try:
                        result = fn(**block.input)
                        is_error = result.startswith("ERROR:")
                    except Exception as e:
                        result = f"ERROR: {type(e).__name__}: {e}"
                        is_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })

        messages.append({"role": "user", "content": tool_results})

    return ("Agent stopped: hit max_iterations.", prune_history(messages))


# =========================================================================
# DRIVER — multi-turn conversation demo
# =========================================================================

def scripted_conversation():
    """Demonstrates memory with a scripted 3-turn conversation."""
    print("=" * 70)
    print("SCRIPTED CONVERSATION (proves memory works)")
    print("=" * 70)
    
    history = []
    turns = [
        "What errors are in logs/sample.log?",
        "Which one looks most urgent?",
        "Tell me more about the database issue you mentioned.",
    ]
    
    for i, q in enumerate(turns, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"USER: {q}")
        answer, history = run_agent(q, history)
        print(f"\nAGENT: {answer}")
        print(f"\n[history now has {len(history)} messages]")


def interactive_chat():
    """Simple REPL you can actually talk to. Type 'quit' to exit."""
    print("=" * 70)
    print("INTERACTIVE CHAT (type 'quit' to exit, 'reset' to clear memory)")
    print("=" * 70)
    
    history = []
    while True:
        try:
            user_input = input("\nYOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        
        if user_input.lower() in {"quit", "exit"}:
            print("Bye.")
            break
        if user_input.lower() == "reset":
            history = []
            print("[memory cleared]")
            continue
        if not user_input:
            continue
        
        answer, history = run_agent(user_input, history)
        print(f"\nAGENT: {answer}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_chat()
    else:
        scripted_conversation()