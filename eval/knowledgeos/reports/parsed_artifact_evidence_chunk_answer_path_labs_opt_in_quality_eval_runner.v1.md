# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Quality Eval Runner

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-quality-eval-runner.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet`
- inputCaseRows: `4`
- qualityPassRows: `4`
- qualityPartialRows: `0`
- qualityFailRows: `0`
- averageQualityScore: `1.0`
- minQualityScore: `1.0`
- expectedNoEvidenceRows: `1`
- noEvidenceLlmCallRows: `0`
- citationCount: `8`
- evidencePacketContractSpanRows: `8`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Policy

Report-only quality runner for the labs preview surface. It scores evidence support, source coverage, citations/spans, and no-evidence safety; it does not judge final answer prose.

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- sourceSpanCreatedRows: `0`
- strictEvidenceRows: `0`
- citationGradeEvidenceRows: `0`
- runtimeEvidenceRows: `0`
- parserExecutionRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- canonicalParsedArtifactWriteRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- publicCliFlagRows: `0`
- defaultOnRows: `0`
- externalLlmCallRows: `0`
- modelApiCallRows: `0`
- judgeModelCallRows: `0`

## Cases

### `alphafold_protein_structure_seed`

- pass: `True`
- qualityGrade: `pass`
- qualityScore: `1.0`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- observedEvidenceTerms: `['AlphaFold', 'protein', 'structure']`
- citations: `2`
- spans: `2`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `word_vectors_similarity_seed`

- pass: `True`
- qualityGrade: `pass`
- qualityScore: `1.0`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- observedEvidenceTerms: `['continuous', 'vector', 'word', 'similarity']`
- citations: `2`
- spans: `2`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `resolved_pair_compare_seed`

- pass: `True`
- qualityGrade: `pass`
- qualityScore: `1.0`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- observedEvidenceTerms: `['AlphaFold', 'CASP14', 'continuous', 'similarity']`
- citations: `4`
- spans: `4`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `missing_candidate_store_no_evidence_seed`

- pass: `True`
- qualityGrade: `pass`
- qualityScore: `1.0`
- expected/observed status: `no_evidence` / `no_evidence`
- expected/observed answerable: `False` / `False`
- observedEvidenceTerms: `[]`
- citations: `0`
- spans: `0`
- localFakeLlmCallRows: `0`
- failureReasons: `[]`
