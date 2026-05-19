# ADR: SourceSpan And StrictEvidence Separation

Date: 2026-05-19

## Status

Accepted for the parsed-artifact SourceSpan strict-evidence tranche.

## Context

Applied SourceSpan JSONL records now exist under local `papers_dir` structured-evidence stores. A later tranche must decide whether strict evidence is a mutation of SourceSpan rows or a separate artifact type with its own promotion lifecycle.

In-place promotion would couple locator provenance to strict-evidence lifecycle, make rollback ambiguous, and encourage reuse of legacy boolean flags (`strictEligible`, `citationGrade`, `runtimeEvidence`) as runtime authority.

## Decision

1. **SourceSpan and StrictEvidence are separate artifact types.**
2. **SourceSpan records are locator/provenance records only.** They preserve candidate linkage, source hash, locator signals, idempotency keys, and non-strict policy markers.
3. **SourceSpan records must not be mutated in-place into strict evidence.** Strict promotion creates new StrictEvidence records; it does not rewrite SourceSpan JSONL rows into strict form.
4. **StrictEvidence is an append-only record type** that references one or more SourceSpan records by stable identifiers (`sourceSpanId`, `candidateRecordId`, `idempotencyKey`).
5. **SourceSpan remains valid** even if strict promotion fails, is blocked by policy gates, or is rolled back. Rollback removes StrictEvidence rows for an explicit run; it does not invalidate surviving SourceSpan provenance.
6. **Legacy compatibility flags stay false until schema cleanup.** SourceSpan `strictEligible`, `citationGrade`, and `runtimeEvidence` remain legacy compatibility markers and must stay `false` during report-only gates and design tranches. They are not promotion outputs for this pipeline stage.

## Consequences

- Report-only gates may classify SourceSpan rows as strict-policy candidates without creating StrictEvidence.
- Apply/readback helpers continue to write and validate SourceSpan JSONL only.
- StrictEvidence creation requires a later explicit executor tranche with its own schema, store contract, and rollback semantics.
- Parser routing, answer integration, DB/index/reembed mutation, and vault writes remain out of scope for SourceSpan strict-policy classification.

## References

- `knowledge_hub/papers/parsed_artifact_source_span_store_contract.py`
- `knowledge_hub/papers/parsed_artifact_source_span_promotion_readback_review.py`
- `docs/adr/2026-05-19-strict-evidence-artifact-type-authority-matrix.md`
