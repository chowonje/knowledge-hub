# Parsed Artifact Evidence Chunk Real Answer Quality Smoke

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-real-answer-quality-smoke.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_real_answer_quality_smoke_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_opt_in_route_review`
- inputCaseRows: `2`
- passRows: `2`
- failRows: `0`
- blockedRows: `0`
- answerContractCitationRows: `4`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Policy

Report-only deterministic answer smoke. It invokes the runtime adapter and checks term support, citations, and provenance, but does not call an LLM or judge model and does not include raw excerpts or answer text in the report.

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- sourceSpanCreatedRows: `0`
- strictEvidenceRows: `0`
- parserExecutionRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- canonicalParsedArtifactWriteRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- llmCallRows: `0`
- judgeModelCallRows: `0`

## Cases

### `alphafold_casp14_accuracy`

- status: `pass`
- resolvedPaperIds: `['1207.0580']`
- adapterRowsAdded: `2`
- citations: `2`
- coverage: `complete` / `1.0`
- observedEvidenceTerms: `['AlphaFold', 'CASP14', 'accuracy']`
- observedAnswerTerms: `['AlphaFold', 'CASP14', 'accuracy']`
- failureReasons: `[]`

### `word_vectors_representation`

- status: `pass`
- resolvedPaperIds: `['1301.3781']`
- adapterRowsAdded: `2`
- citations: `2`
- coverage: `complete` / `1.0`
- observedEvidenceTerms: `['continuous', 'representations', 'word', 'similarity']`
- observedAnswerTerms: `['continuous', 'representations', 'word', 'similarity']`
- failureReasons: `[]`
