# ADR: StrictEvidence Runtime Binding Gate Design

Date: 2026-05-20

## Status

Accepted for the post-citation-grade StrictEvidence promotion chain.

## Context

The citation-grade pipeline has 99 readback-validated citation-grade records and
a post-apply promotion hold. Those records are suitable candidates for a later
runtime binding gate, but they are not yet runtime-visible evidence and must not
be exposed to answer integration.

The next decision is whether runtime visibility should be represented by
mutating citation-grade or StrictEvidence records, or by a separate binding
artifact.

## Decision

1. **Runtime binding is a separate append-only artifact.**
2. **Citation-grade, eligibility, StrictEvidence, and SourceSpan records must
   not be mutated in place** to make evidence runtime-visible.
3. **Runtime binding is not answer integration.** A runtime binding record only
   makes an evidence item eligible for a later controlled runtime surface.
4. **Answer integration remains a separate gate** after runtime binding
   readback and no-answer safety checks.
5. **No-answer safety evaluation is required before any runtime binding apply
   path can make records visible to answer flows.**
6. **Rollback must be scoped by run id / runtime binding id** before downstream
   answer bindings reference the runtime record.

## Consequences

- Runtime visibility remains auditable and reversible without rewriting earlier
  evidence artifacts.
- Citation-grade records remain promotion metadata, not runtime authority.
- Future helpers should define a runtime binding record contract before any
  apply path.
- Answerability still requires a later answer-integration gate that can enforce
  no-answer behavior when binding evidence is absent or insufficient.

## Rejected Alternatives

### Mutate `runtimeEvidence=true` on existing records

Rejected because it obscures when and why the record became runtime-visible,
makes rollback ambiguous, and couples storage promotion to answer behavior.

### Let citation-grade records be directly visible to answer integration

Rejected because citation-grade is only a quality tier. Runtime binding and
answer integration need separate gates, readback, and safety evaluation.

## References

- `docs/adr/2026-05-19-source-span-strict-evidence-separation.md`
- `docs/adr/2026-05-20-strict-evidence-citation-grade-policy-gate-design.md`
- `knowledge_hub/papers/strict_evidence_citation_grade_post_apply_promotion_hold_review.py`
