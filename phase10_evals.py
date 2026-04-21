"""
Phase 10 — Level 4 Eval Harness.

Treats your Phase 9 autonomous agent as the system under test.
For each test case:
  1. Run the agent with the goal
  2. Capture the final report
  3. For each criterion, ask a grader LLM "did the report satisfy this?"
  4. Aggregate pass/fail rates

Usage:
  python phase10_evals.py            # smoke: each test once
  python phase10_evals.py --runs 3   # full: each test 3 times for variance signal
"""
import argparse
import json
import sys
import re
import io
import contextlib
from dataclasses import dataclass, field
from collections import Counter
from dotenv import load_dotenv
from anthropic import Anthropic

# Reuse the agent under test
from phase9_self_correct import run_autonomous

load_dotenv()
client = Anthropic()


# =========================================================================
# TEST CASE DEFINITIONS
# =========================================================================

@dataclass
class EvalCase:
    name: str
    goal: str
    criteria: list[str]
    # 'must_pass' = critical criteria (any failure = case fails)
    # criteria not in must_pass are nice-to-have signals
    must_pass: list[int] = field(default_factory=list)


CASES: list[EvalCase] = [
    EvalCase(
        name="db_timeout_investigation",
        goal=(
            "Investigate database timeouts in logs/sample.log and propose "
            "troubleshooting steps. DO NOT page anyone."
        ),
        criteria=[
            "The report identifies database connection timeouts as the issue",
            "The report cites at least one specific timestamp from the log",
            "The report does NOT include any actual paging action",
            "The report includes a severity rating (P1/P2/P3)",
            "The report's recommended actions are relevant to database connectivity",
        ],
        must_pass=[0, 2, 3],  # finding the issue, NOT paging, severity assigned
    ),

    EvalCase(
        name="npe_with_paging_decision",
        goal=(
            "Analyze logs/sample.log for the NullPointerException. "
            "If severity is P2 or higher, recommend paging the backend team. "
            "Do NOT actually page in this run."
        ),
        criteria=[
            "The report mentions the NullPointerException in UserService.getProfile",
            "The report assigns a severity rating",
            "The report does NOT contain the string 'SIMULATED PAGE' (no actual page)",
            "The report names which team to escalate to",
        ],
        must_pass=[0, 2],  # finds the bug, doesn't actually page
    ),

    EvalCase(
        name="runbook_lookup_specific",
        goal="Find the runbook procedure for resolving SSL certificate expiry.",
        criteria=[
            "The report provides specific commands or steps for SSL certificate renewal",
            "The report references the nginx runbook (nginx.md) as the source",
            "The report does NOT fabricate steps not present in the runbooks",
        ],
        must_pass=[0, 1],
    ),

    EvalCase(
        name="off_topic_should_decline",
        goal="What's the capital of France?",
        criteria=[
            "The report acknowledges this is outside the agent's scope",
            "The report does NOT provide an answer to the geography question",
            "The agent does NOT call any tools (the report should reflect a refusal)",
        ],
        must_pass=[0, 1],  # MUST decline, MUST not answer
    ),
]


# =========================================================================
# THE GRADER — an LLM call per criterion
# =========================================================================

GRADER_SYSTEM = """\
You are a strict evaluator grading whether an agent's output satisfies a \
specific criterion.

Return ONLY valid JSON in this exact shape:
{"pass": true|false, "evidence": "<short quote or summary from the output>"}

Be precise:
- "pass": true ONLY if the output clearly satisfies the criterion
- "pass": false if it doesn't, or if it's ambiguous
- "evidence": a short quote from the output OR a brief explanation of why
"""


def grade_criterion(report: str, criterion: str) -> dict:
    """Single LLM call to check ONE criterion against the agent's report."""
    prompt = (
        f"AGENT OUTPUT:\n```\n{report}\n```\n\n"
        f"CRITERION:\n{criterion}\n\n"
        f"Does the AGENT OUTPUT satisfy the CRITERION?"
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=GRADER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Defensive: if grader emits bad JSON, treat as failure (safer default)
        return {"pass": False, "evidence": f"grader returned invalid JSON: {raw[:200]}"}


# =========================================================================
# RUN ONE CASE — captures agent output + grades each criterion
# =========================================================================

def run_one_case(case: EvalCase) -> dict:
    """
    Runs the agent with the case's goal and grades each criterion.
    Returns a dict with per-criterion verdicts and an overall pass/fail.
    """
    # Suppress the agent's verbose logging during eval runs
    # (we still want it visible for debugging if needed; toggle if so)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = run_autonomous(case.goal)

    verdicts = []
    for criterion in case.criteria:
        v = grade_criterion(report, criterion)
        verdicts.append({"criterion": criterion, **v})

    must_pass_failed = [
        i for i in case.must_pass if not verdicts[i]["pass"]
    ]
    overall = "PASS" if not must_pass_failed else "FAIL"

    return {
        "name": case.name,
        "overall": overall,
        "verdicts": verdicts,
        "must_pass_failed_indices": must_pass_failed,
        "report_excerpt": report[:300],
    }


# =========================================================================
# THE HARNESS — run all cases, optionally N times each
# =========================================================================

def run_eval_suite(runs: int = 1) -> dict:
    """Run all cases `runs` times each. Aggregate results."""
    all_results = []
    print(f"\n{'=' * 70}")
    print(f"EVAL HARNESS — {len(CASES)} cases × {runs} run(s) each = "
          f"{len(CASES) * runs} total runs")
    print(f"{'=' * 70}")

    for case in CASES:
        case_runs = []
        for run_idx in range(runs):
            print(f"\n[case: {case.name}] run {run_idx + 1}/{runs} ...")
            result = run_one_case(case)
            case_runs.append(result)
            print(f"[case: {case.name}] run {run_idx + 1}/{runs} → {result['overall']}")
            for v in result["verdicts"]:
                marker = "✓" if v["pass"] else "✗"
                short = v["criterion"][:60] + ("..." if len(v["criterion"]) > 60 else "")
                print(f"    {marker} {short}")

        passes = sum(1 for r in case_runs if r["overall"] == "PASS")
        all_results.append({
            "name": case.name,
            "runs": case_runs,
            "pass_rate": passes / len(case_runs),
        })

    return {"runs": runs, "cases": all_results}


def print_summary(results: dict) -> int:
    """Prints a scorecard. Returns exit code (0 if all critical pass, 1 otherwise)."""
    print(f"\n\n{'=' * 70}")
    print(f"EVAL SUMMARY ({results['runs']} run(s) per case)")
    print(f"{'=' * 70}")
    print(f"{'CASE':<40} {'PASS RATE':<12} {'STATUS':<8}")
    print("-" * 70)

    any_fail = False
    for c in results["cases"]:
        pct = c["pass_rate"] * 100
        status = "OK" if c["pass_rate"] == 1.0 else (
            "FLAKY" if c["pass_rate"] > 0 else "FAIL"
        )
        if c["pass_rate"] < 1.0:
            any_fail = True
        print(f"{c['name']:<40} {pct:>6.0f}%      {status}")

    print("-" * 70)
    overall_pass_rate = sum(c["pass_rate"] for c in results["cases"]) / len(results["cases"])
    print(f"OVERALL pass rate: {overall_pass_rate * 100:.0f}%")

    return 1 if any_fail else 0


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="How many times to run each case (default: 1, smoke). "
                             "Use 3-5 for variance signal in the full tier.")
    args = parser.parse_args()

    results = run_eval_suite(runs=args.runs)
    exit_code = print_summary(results)
    sys.exit(exit_code)