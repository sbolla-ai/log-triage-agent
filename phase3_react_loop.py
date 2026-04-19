"""
Phase 3: The full ReAct loop.
Claude picks tools, we execute them, Claude picks more tools or answers.
"""
import re
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# =========================================================================
# TOOL IMPLEMENTATIONS — real Python functions that do real work
# =========================================================================

def search_logs(pattern: str, log_file: str) -> str:
    """Grep-like search. Returns matching lines with line numbers."""
    try:
        with open(log_file, "r") as f:
            matches = []
            for i, line in enumerate(f, start=1):
                if re.search(pattern, line):
                    matches.append(f"{i}: {line.rstrip()}")
        if not matches:
            return f"No matches found for pattern '{pattern}' in {log_file}."
        return "\n".join(matches[:50])  # cap output for token safety
    except FileNotFoundError:
        return f"ERROR: File not found: {log_file}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def get_log_stats(log_file: str) -> str:
    """Returns severity counts, total lines, and time range."""
    try:
        severities = Counter()
        total = 0
        timestamps = []
        ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        sev_re = re.compile(r"\b(ERROR|WARN|INFO|DEBUG)\b")

        with open(log_file, "r") as f:
            for line in f:
                total += 1
                if m := ts_re.match(line):
                    timestamps.append(m.group(1))
                if m := sev_re.search(line):
                    severities[m.group(1)] += 1

        time_range = (
            f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"
        )
        return (
            f"File: {log_file}\n"
            f"Total lines: {total}\n"
            f"Time range: {time_range}\n"
            f"Severity counts: {dict(severities)}"
        )
    except FileNotFoundError:
        return f"ERROR: File not found: {log_file}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def extract_errors(log_file: str, limit: int = 10) -> str:
    """Returns the most recent ERROR lines, up to `limit`."""
    try:
        with open(log_file, "r") as f:
            errors = [line.rstrip() for line in f if "ERROR" in line]
        if not errors:
            return f"No ERROR lines found in {log_file}."
        selected = errors[-limit:]
        return "\n".join(selected)
    except FileNotFoundError:
        return f"ERROR: File not found: {log_file}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# Dispatch table — maps tool names to real Python functions.
# Classic Strategy pattern: the LLM picks the strategy by name.
TOOL_REGISTRY = {
    "search_logs": search_logs,
    "get_log_stats": get_log_stats,
    "extract_errors": extract_errors,
}


# =========================================================================
# TOOL SCHEMAS — the CONTRACT we show Claude
# =========================================================================

TOOLS = [
    {
        "name": "search_logs",
        "description": (
            "Search a log file for a regex pattern. Returns matching lines "
            "with line numbers (up to 50 matches). Use this when the user "
            "asks to find, search, or grep for something specific."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "log_file": {
                    "type": "string",
                    "description": "Path to the log file",
                },
            },
            "required": ["pattern", "log_file"],
        },
    },
    {
        "name": "get_log_stats",
        "description": (
            "Returns high-level stats about a log file: total line count, "
            "severity counts (ERROR, WARN, INFO, DEBUG), and time range. "
            "Use this when the user asks for an overview or summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {"type": "string", "description": "Path to the log file"}
            },
            "required": ["log_file"],
        },
    },
    {
        "name": "extract_errors",
        "description": (
            "Returns the most recent ERROR-level lines from a log file. "
            "Use this when the user asks about errors, failures, or what "
            "went wrong."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {"type": "string", "description": "Path to the log file"},
                "limit": {
                    "type": "integer",
                    "description": "Max number of error lines to return (default 10)",
                },
            },
            "required": ["log_file"],
        },
    },
]


# =========================================================================
# THE REACT LOOP — the heart of the agent
# =========================================================================

def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """Run the ReAct loop until Claude gives a final answer."""
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        print(f"stop_reason: {response.stop_reason}")

        # If Claude is done, return the final text answer
        if response.stop_reason == "end_turn":
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return final_text

        # Otherwise, Claude wants tools run.
        # Append Claude's WHOLE response to the history
        # (including any text blocks AND tool_use blocks).
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool_use block and collect results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → executing {block.name}({block.input})")
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

                print(f"  → result (first 200 chars): {result[:200]}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )

        # Append the tool results as a user message — yes, user.
        # Tool results always come back on the "user" turn by convention.
        messages.append({"role": "user", "content": tool_results})

    return "Agent stopped: hit max_iterations without finishing."


# =========================================================================
# DRIVER
# =========================================================================

if __name__ == "__main__":
    question = "What's going wrong in sample.log? Give me a summary of the errors."
    print(f"USER: {question}")
    answer = run_agent(question)
    print("\n" + "=" * 60)
    print(f"AGENT: {answer}")
    print("=" * 60)