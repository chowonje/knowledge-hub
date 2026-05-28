# Parsed Artifact Evidence Chunk Candidate Dry Run

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-dry-run.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_candidate_dry_run_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_candidate_canary_apply_readback`
- parsedArtifactRows: `449`
- paperRowsWithCandidates: `313`
- selectedCandidateRows: `1200`
- heldCandidateRows: `52`
- paragraphCandidateRows: `1097`
- sectionCandidateRows: `103`
- answerEvidenceCandidateRows: `1200`
- answerabilityCandidateRows: `1200`
- blockedMissingSourceHashRows: `136`
- blockedSourceHashMismatchRows: `0`
- blockedMissingLocatorRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`
- answerVisibleRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`

## Gate

- passed: `True`
- hasCandidateRows: `True`
- allSelectedRowsCandidateOnly: `True`
- noMutationOrRuntimeExposure: `True`

## Non-Scope

- No SourceSpan creation.
- No StrictEvidence, citation-grade, runtime evidence, or answer-visible exposure.
- No parser execution, reindex, DB/index mutation, vault scan, or external download.
