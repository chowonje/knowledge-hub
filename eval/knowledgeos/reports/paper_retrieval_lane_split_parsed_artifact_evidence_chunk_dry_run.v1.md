# Paper Retrieval Lane Split Parsed Artifact Evidence Chunk Dry Run

- schema: `knowledge-hub.paper.paper-retrieval-lane-split-parsed-artifact-evidence-chunk-dry-run.v1`
- status: `ready`
- decision: `paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_ready`
- nextRecommendedTranche: `paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review`
- sourceCandidateDiscoveryOnlyRows: `125`
- plannedLaneSplitRows: `2`
- parsedArtifactEvidenceChunkLaneRows: `1`
- visualCandidateDiscoveryLaneRows: `1`
- answerEvidenceEligibleLaneRows: `1`
- visualHintRowsQuarantinedFromAnswerEvidence: `125`
- fallbackToVisualHintAsEvidenceAllowedRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`
- runtimeRouteWriteRows: `0`
- operationalSearchIndexQueryRows: `0`
- answerVisibleRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`

## Gate

- passed: `True`
- onlyParsedLaneMaySupplyAnswerEvidence: `True`
- visualHintsQuarantinedFromAnswerEvidence: `True`
- noMutationOrRuntimeExposure: `True`

## Lane Split

- `visual_retrieval_hint_candidate_discovery`: can expand candidates only.
- `parsed_artifact_evidence_chunk`: the only planned lane that may later satisfy answerability.

## Non-Scope

- No runtime route write.
- No operational search index query.
- No vector DB apply.
- No answer-visible exposure.
- No strict or citation-grade evidence creation.
- No parser execution, reindex, DB/index mutation, vault scan, or external download.
