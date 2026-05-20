# ADR: StrictEvidence Citation-Grade Policy Gate Design

Date: 2026-05-20

## Status

Accepted for the post-eligibility StrictEvidence promotion chain.

## Context

The StrictEvidence eligibility pipeline has 99 readback-validated eligibility
records and a post-apply promotion hold. Those rows are eligible candidates for
a later citation-grade policy gate, but they are still not runtime evidence and
must not be visible to answer integration.

The next decision is whether citation-grade status should mutate an existing
StrictEvidence or eligibility row, or be represented as a separate promotion
artifact.

## Decision

1. **Citation-grade status is a separate append-only artifact.**
2. **StrictEvidence rows, SourceSpan rows, and eligibility rows must not be
   mutated in place** to mark citation-grade status.
3. **The legacy `citationGrade` boolean remains false** on existing
   StrictEvidence and eligibility records during this promotion chain.
4. **Citation-grade is not runtime binding.** A citation-grade record only means
   the evidence passed a scoped citation policy gate and may enter later runtime
   binding gates.
5. **Runtime evidence, parser routing, and answer integration remain separate
   gates.**
6. **No-answer safety evaluation is required before runtime or answer
   integration can see citation-grade records.**

## Consequences

- Citation-grade promotion remains auditable and rollbackable by run or record
  id without rewriting earlier evidence artifacts.
- Policy changes do not require rewriting StrictEvidence or eligibility JSONL.
- Future helpers should define a citation-grade record contract before any
  apply path.
- Answerability still requires later no-answer, runtime-binding, and answer
  integration gates.

## Rejected Alternatives

### Mutate `citationGrade=true` on existing records

Rejected because it obscures when and why the record became citation-grade,
makes rollback ambiguous, and turns legacy booleans into runtime authority.

### Treat citation-grade as runtime evidence

Rejected because citation-grade is only a promotion tier. Runtime visibility and
answer integration need separate binding and no-answer safety gates.

## References

- `docs/adr/2026-05-19-source-span-strict-evidence-separation.md`
- `docs/adr/2026-05-20-strict-evidence-strict-eligible-mutation-semantics.md`
- `knowledge_hub/papers/strict_evidence_eligibility_post_apply_promotion_hold_review.py`
