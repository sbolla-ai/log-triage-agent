# Known Limitations (as of v4-autonomous)

The eval harness in phase10_evals.py surfaced the following gaps. These are
intentionally deferred to Level 5 (production hardening), not bugs.

## 1. Criteria need refinement (cases 1 & 2)
The grader treats text mentions of "paging" as actual paging actions.
Real fix: criterion should check for absence of "[SIMULATED PAGE]" string
specifically, not the topic of paging in escalation discussions.

## 2. Agent fabricates runbook content (case 3)
The synthesizer occasionally adds steps not present in the actual runbooks
(e.g., generic SSL renewal commands beyond what nginx.md contains).
Real fix: RAG pattern with strict grounding — the synthesizer must cite
which retrieved chunk supports each claim.

## 3. No off-topic refusal (case 4)
Phase 9's planner has no fallback handler. Off-topic prompts get a planned
sequence of nonsensical tool calls. Phase 6 had this; we lost it in the
Level 4 refactor.
Real fix: re-introduce the routing/scope check as a pre-planning step,
OR teach the planner to emit an empty plan with a refusal message.
