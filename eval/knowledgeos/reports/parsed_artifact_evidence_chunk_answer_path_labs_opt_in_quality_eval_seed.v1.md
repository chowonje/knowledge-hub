# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Quality Eval Seed

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-quality-eval-seed.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner`
- inputCaseRows: `4`
- passRows: `4`
- failRows: `0`
- expectedAnswerableRows: `3`
- expectedNoEvidenceRows: `1`
- adapterRowsAdded: `8`
- citationCount: `8`
- evidencePacketContractSpanRows: `8`
- localFakeLlmCallRows: `3`
- noEvidenceLlmCallRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Policy

Report-only seed for the next labs quality eval runner. Raw answer text, citations, sources, and excerpts are excluded from this report.

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
- focus: `single_paper_section_paragraph_evidence`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- paperIds: `['1207.0580']`
- observedEvidenceTerms: `['AlphaFold', 'protein', 'structure']`
- missingEvidenceTerms: `[]`
- citations: `2`
- spans: `2`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `word_vectors_similarity_seed`

- pass: `True`
- focus: `single_paper_section_paragraph_evidence`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- paperIds: `['1301.3781']`
- observedEvidenceTerms: `['continuous', 'vector', 'word', 'similarity']`
- missingEvidenceTerms: `[]`
- citations: `2`
- spans: `2`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `resolved_pair_compare_seed`

- pass: `True`
- focus: `two_paper_compare_seed`
- expected/observed status: `ok` / `ok`
- expected/observed answerable: `True` / `True`
- paperIds: `['1207.0580', '1301.3781']`
- observedEvidenceTerms: `['AlphaFold', 'CASP14', 'continuous', 'similarity']`
- missingEvidenceTerms: `[]`
- citations: `4`
- spans: `4`
- localFakeLlmCallRows: `1`
- failureReasons: `[]`

### `missing_candidate_store_no_evidence_seed`

- pass: `True`
- focus: `expected_no_evidence_safety`
- expected/observed status: `no_evidence` / `no_evidence`
- expected/observed answerable: `False` / `False`
- paperIds: `['missing-paper-id']`
- observedEvidenceTerms: `[]`
- missingEvidenceTerms: `[]`
- citations: `0`
- spans: `0`
- localFakeLlmCallRows: `0`
- failureReasons: `[]`
