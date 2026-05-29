# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Output Capture

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-output-capture.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review`
- capturedCommandRows: `5`
- outputCapturePassRows: `5`
- outputCaptureFailRows: `0`
- jsonAssertionRows: `47`
- jsonAssertionPassRows: `47`
- externalRejectionPassRows: `1`
- rawOutputPersistedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Captured Rows

- `alphafold_protein_structure_seed`: pass=`True`, status=`ok`, answerable=`True`, citations=`2`, spans=`2`, payloadHash=`sha256:4985ed2158a2b02755f45f2d417459ff67c2e90c86b03d653de79292ff06853b`
- `word_vectors_similarity_seed`: pass=`True`, status=`ok`, answerable=`True`, citations=`2`, spans=`2`, payloadHash=`sha256:97276e1de9678c04885420a037c291ceec69fd7da09a31c6c95159d204a7ae68`
- `resolved_pair_compare_seed`: pass=`True`, status=`ok`, answerable=`True`, citations=`4`, spans=`4`, payloadHash=`sha256:fa5a9979e28ae95968db9dfc9970bce998b8c39880e003bda26ad776597eeee3`
- `missing_candidate_store_no_evidence_seed`: pass=`True`, status=`no_evidence`, answerable=`False`, citations=`0`, spans=`0`, payloadHash=`sha256:f3d1ddc43100beb6b66a9966521f8c541a2d5e723928f6af757ef97b86c104f3`

## External Rejection

- pass: `True`
- observedExitCode: `1`
- observedErrorHash: `sha256:e570dceefdfad6535f9c5b42a93edbe9c5819195aada216a2086ad6374f34f0f`

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
