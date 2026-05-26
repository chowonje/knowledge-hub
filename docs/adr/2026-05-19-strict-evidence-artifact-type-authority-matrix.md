# ADR: Strict Evidence Artifact-Type Authority Matrix

Date: 2026-05-19

## Status

Accepted for the parsed-artifact SourceSpan strict-evidence typed policy gate.

## Context

SourceSpan rows currently carry heterogeneous locator shapes (`page`, `bbox`, `blockIndexes`, optional `chars`). Strict evidence cannot treat all locator shapes as equivalent. Text, captions, tables, equations, and figure regions require different authority fields before a StrictEvidence record may be planned.

## Decision

### Text (`section` and text-bearing spans)

Strict text evidence requires:

- `sourceContentHash`
- `locator.chars.start` and `locator.chars.end`
- `locator.chars.basis == "sourceContentHash"`
- `locator.chars.normalization`
- `locator.chars.expectedSubstringSha256`

Page/bbox-only locators are **not** strict text evidence. Page/bbox may be recorded as auxiliary verification metadata for text, but cannot substitute for char-offset authority.

### Figure captions

Caption strict evidence follows text strict rules plus future `figureId` linkage when caption text is promoted.

Current store `figure` rows without char-offset authority are not strict evidence. Page/bbox-only figure rows remain blocked until caption offset authority exists or figure-region structured authority is defined.

### Figure regions

Figure-region strict evidence requires figure-native authority:

- `figureId`
- region `bbox`
- `extractionMethod`
- `regionContentHash`

### Tables

Table strict evidence requires table/cell-native authority such as:

- `tableId`
- row/column coordinates
- `cellRawText`
- `cellNormalizedValue`
- `cellContentHash`
- header linkage metadata when available

Page/bbox-only table locators are insufficient.

### Equations

Equation strict evidence requires equation-native authority such as:

- `equationId`
- `equationTeXHash` or `mathmlHash`

PDF-region-only equation anchors are design inputs, not strict evidence.

### Aggregated claims

Aggregated claims require multi-evidence composition. A single SourceSpan cannot be promoted directly into a composite strict claim without an explicit composition gate.

## Consequences

- Typed policy gates classify SourceSpan rows by artifact type and authority mode without creating StrictEvidence.
- Missing char-offset authority blocks text/caption strict candidacy even when readback validation succeeded for SourceSpan storage.
- Structured artifact types block on missing table/equation/figure-region fields until dedicated extractors populate them.
- The recommended next design tranche is original-source offset authority for text/caption spans before any StrictEvidence executor.

## References

- `docs/adr/2026-05-19-source-span-strict-evidence-separation.md`
- `knowledge_hub/papers/parsed_artifact_source_span_strict_evidence_policy_gate_v2_typed.py`
