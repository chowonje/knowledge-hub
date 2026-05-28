# Paper Retrieval Lane Split Parsed Artifact Evidence Chunk Contract Review

- schema: `knowledge-hub.paper.paper-retrieval-lane-split-parsed-artifact-evidence-chunk-contract-review.v1`
- status: `ready`
- decision: `paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_candidate_dry_run`
- contractRows: `5`
- allowedArtifactTypeRows: `5`
- disallowedSourceRows: `5`
- answerEvidenceEligibleContractRows: `5`
- answerabilityEligibleContractRows: `5`
- visualHintAnswerEvidenceAllowedRows: `0`
- fallbackChunkAnswerEvidenceAllowedRows: `0`
- locatorOnlyAnswerEvidenceAllowedRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`
- runtimeRouteWriteRows: `0`
- answerVisibleRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`
- parsedArtifactEvidenceChunkCreatedRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`

## Gate

- passed: `True`
- allArtifactTypesCovered: `True`
- visualHintCannotBeAnswerEvidence: `True`
- noMutationOrRuntimeExposure: `True`

## Allowed Artifact Contracts

- `section`: `chars:start-end_required`, structuredReadback=`False`
- `paragraph`: `chars:start-end_required`, structuredReadback=`False`
- `table`: `page_bbox_plus_table_cell_or_caption_required`, structuredReadback=`True`
- `equation`: `chars_or_page_bbox_plus_equation_identity_required`, structuredReadback=`True`
- `figure_caption`: `chars_or_page_bbox_plus_caption_identity_required`, structuredReadback=`True`

## Disallowed Sources

- `visual_retrieval_hint_text`: `candidate_discovery_only_not_answer_evidence`
- `fallback_chunk`: `fallback_chunk_missing_strict_provenance`
- `locator_only_anchor`: `locator_without_verbatim_text_cannot_satisfy_answerability`
- `memory_unit_locator`: `memory_unit_locator_is_not_original_paper_evidence`
- `korean_summary_or_paraphrase`: `derived_language_or_paraphrase_is_not_original_evidence`

## Non-Scope

- No parsed-artifact evidence chunk creation.
- No SourceSpan, StrictEvidence, citation-grade, or runtime evidence creation.
- No answer-visible exposure or answer generation.
- No runtime route write, operational search query, DB/index mutation, parser execution, reindex, vault scan, or external download.
