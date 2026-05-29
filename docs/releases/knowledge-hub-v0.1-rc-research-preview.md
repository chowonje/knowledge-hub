# Knowledge Hub v0.1 RC Research Preview (2026-05-29)

## Highlights

- Knowledge Hub is framed as a local-first, evidence-first research knowledge runtime for auditable AI research workflows.
- The supported public path remains intentionally narrow: `discover -> index -> search/ask -> evidence review`.
- The v0.1 RC scope is section/paragraph evidence-first paper QA and comparison over a local AI-paper corpus.
- Parsed-artifact evidence chunks now have a labs-only answer preview path with source hash, `chars:start-end` locators, citation spans, and conservative no-answer behavior.
- public/default promotion remains held. Public `khub ask`, default MCP activation, and default-on parsed-artifact evidence chunk behavior are not enabled by this release candidate.

## Release Scope

This release candidate is a Research Preview for local research workflows:

- manage a local AI-paper corpus
- inspect source and parsed-artifact coverage
- run grounded search and ask flows
- review evidence traces before trusting an answer
- test section/paragraph evidence chunks through labs-only opt-in surfaces

The canonical product shape is CLI + MCP + local stores + evidence reports + docs. GUI and Obsidian integrations remain consumers or projections, not the product authority.

## User-Facing Paths

- Default: `khub doctor`, `khub discover`, `khub index`, `khub search`, `khub ask`, `khub trace`
- Labs preview: `khub labs paper evidence-chunk-ask`
- MCP: default retrieval tools remain conservative; evidence-chunk answer preview stays in labs/all profiles only

## What Is Deliberately Held

- No default `khub ask` activation for parsed-artifact evidence chunks
- No default MCP activation for evidence-chunk answer preview
- No broad corpus-scale answer-quality claim
- No table, equation, or figure-caption default evidence promise
- No runtime promotion of visual retrieval hints as answer evidence
- No external model call requirement for the release-note gate

## Breaking Changes

- None recorded for the supported public path in this release-note tranche.

## Upgrade Notes

- Existing local stores are not migrated by this release-note tranche.
- Labs evidence-chunk preview requires explicit paper ids and local parsed-artifact/candidate data.
- Operators should continue to treat generated reports as audit artifacts, not as runtime evidence.

## Features

- Canonical product definition for KnowledgeOS / Knowledge Hub.
- Post-merge convergence review for the v0.1 RC labs evidence chunk path.
- Public/default promotion decision gate that keeps default surfaces closed while allowing Research Preview release-note preparation.
- Labs-only parsed-artifact evidence chunk answer preview for section/paragraph evidence.
- Local report chain for answerability, no-answer safety, public hygiene, and release smoke evidence.

## Known Issues

- Current quality evidence is tranche-scale, not corpus-scale.
- Table, equation, and figure-caption evidence remain limited or labs-only until their evidence gates pass.
- Source quality is uneven across source families.
- Some operator and eval surfaces are intentionally broader than the public promise.
- General release language remains blocked until corpus-scale quality evidence and default-surface promotion gates pass.

## Verification

- `parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate.v1.json`
  - `status=ready`
  - `releaseNotesPathAllowedRows=1`
  - `publicDefaultPromotionReadyRows=0`
  - `publicDefaultPromotionHeldRows=1`
  - `generalRcReadyRows=0`
  - `corpusScaleClaimProvenRows=0`
- `parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.json`
  - `researchPreviewRcCandidateRows=1`
  - `publicDefaultPromotionHeldRows=1`
  - `schemaViolationCount=0`
- Public hygiene and release smoke remain the relevant public checks for this Research Preview posture.

## Risk / Notes

- This release candidate is suitable for controlled Research Preview use, not for broad default-surface promotion.
- The next product-quality gate should either prepare the external-facing release package or broaden corpus-scale answer quality evidence.
