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

## Pass 3 (Phase 14) — Known limitations of injection defense

The three-layer defense in injection_defense.py validates against the
cartoon "IMPORTANT SYSTEM INSTRUCTION" payload. Known gaps for production:

1. Pattern-based sanitization (Layer 1) cannot catch semantically equivalent
   payloads that avoid trigger words. A sneakier payload using "established
   practice" or "audit norm" framing would bypass current patterns.
   Mitigation: add an LLM-as-judge guardrail (Haiku call per chunk) before
   reaching the synthesizer. Estimated cost: +25% per RAG-using run.

2. Layer 3 output scan is regex-only; production systems use either an LLM
   judge or a fine-tuned safety classifier (Llama Guard, NeMo Guardrails).

3. No capability sandboxing per data source. All retrieved content currently
   runs through the same agent with full toolset. Production systems should
   scope tools by source trust (internal vs. customer-uploaded vs. web).

4. No content provenance signing on the corpus. Compliance-heavy contexts
   (healthcare, finance) typically require this for forensic audit trail.

These limitations are documented for future work. The three-layer defense
as built is a reasonable starting point for low-to-medium-risk applications.
