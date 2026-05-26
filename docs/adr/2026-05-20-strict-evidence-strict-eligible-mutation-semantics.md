# ADR: StrictEvidence strictEligible Mutation Semantics

Date: 2026-05-20

## Status

Accepted for the post-pilot StrictEvidence promotion chain.

## Context

The StrictEvidence pilot now has 99 readback-validated records and a completed
manifest-only section/figure-caption pilot. The post-pilot hold gate keeps
`strictEligible` mutation, citation-grade evidence, runtime evidence, parser
routing, answer integration, DB/index/reembed, and vault access blocked.

The next decision is whether a validated StrictEvidence record should become
eligible for later promotion by mutating a boolean field on the existing record
or by writing a separate promotion/eligibility artifact.

## Decision

1. **StrictEvidence records must not be mutated in place to become eligible.**
2. **`strictEligible` on existing StrictEvidence records remains a legacy
   compatibility flag and must stay `false` in this promotion chain.**
3. **Eligibility is represented by a later append-only eligibility record** that
   references the immutable StrictEvidence id, SourceSpan id, candidate id,
   gate/run ids, and promotion policy version.
4. **The eligibility record is not runtime evidence.** It only means the
   StrictEvidence record passed a scoped eligibility gate and may enter later
   citation-grade/runtime-binding gates.
5. **Citation-grade evidence, runtime evidence, parser routing, and answer
   integration remain separate gates.** This ADR does not allow any of those
   surfaces.
6. **Rollback is run-scoped and record-scoped.** A later eligibility executor
   must be able to remove or invalidate eligibility records for an explicit run
   without editing the underlying StrictEvidence or SourceSpan records.

## Consequences

- StrictEvidence remains an immutable audit record.
- Eligibility decisions are independently reviewable, idempotent, and
  rollbackable.
- Policy changes do not require rewriting the existing strict-evidence JSONL
  rows.
- Future helpers should introduce a separate eligibility record contract before
  any apply path.
- Runtime answerability still requires later citation-grade and runtime-binding
  gates.

## Rejected Alternatives

### Mutate `strictEligible=true` on StrictEvidence JSONL

Rejected because it obscures when and why the record became eligible, makes
rollback ambiguous, and turns a legacy boolean into policy authority.

### Treat strict eligibility as answerability

Rejected because strict eligibility is only a promotion prerequisite. It is not
citation-grade evidence, runtime binding, or answer integration.

## References

- `docs/adr/2026-05-19-source-span-strict-evidence-separation.md`
- `docs/adr/2026-05-19-strict-evidence-artifact-type-authority-matrix.md`
- `knowledge_hub/papers/strict_evidence_post_pilot_promotion_hold_review.py`
- `knowledge_hub/papers/strict_evidence_strict_eligible_mutation_decision_record.py`
