# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Packet

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-packet.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture`
- packetRows: `4`
- expectedAnswerableCommandRows: `3`
- expectedNoEvidenceCommandRows: `1`
- externalRejectionCommandRows: `1`
- jsonAssertionRows: `47`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Instructions

- Run from the product repo root.
- Use the labs CLI surface only: `khub labs paper evidence-chunk-ask`.
- MCP testing requires the `labs` or `all` profile; the default MCP profile should not expose this tool.
- Do not use `--allow-external` except for the explicit rejection check.

## Commands

### `alphafold_protein_structure_seed`

```bash
khub labs paper evidence-chunk-ask 'What parsed section or paragraph evidence is available about AlphaFold protein structure prediction?' --paper-id 1207.0580 --json
```

- expectedStatus: `ok`
- expectedAnswerable: `True`
- expectedMinCitations: `2`
- expectedMinSpanRows: `2`
- expectedEvidenceTerms: `['AlphaFold', 'protein', 'structure']`

### `word_vectors_similarity_seed`

```bash
khub labs paper evidence-chunk-ask 'What parsed section or paragraph evidence is available about continuous word representations?' --paper-id 1301.3781 --json
```

- expectedStatus: `ok`
- expectedAnswerable: `True`
- expectedMinCitations: `2`
- expectedMinSpanRows: `2`
- expectedEvidenceTerms: `['continuous', 'vector', 'word', 'similarity']`

### `resolved_pair_compare_seed`

```bash
khub labs paper evidence-chunk-ask 'Compare available parsed evidence for the AlphaFold and word vector papers.' --paper-id 1207.0580 --paper-id 1301.3781 --json
```

- expectedStatus: `ok`
- expectedAnswerable: `True`
- expectedMinCitations: `4`
- expectedMinSpanRows: `4`
- expectedEvidenceTerms: `['AlphaFold', 'CASP14', 'continuous', 'similarity']`

### `missing_candidate_store_no_evidence_seed`

```bash
khub labs paper evidence-chunk-ask 'What parsed section or paragraph evidence is available for this missing paper?' --paper-id missing-paper-id --json
```

- expectedStatus: `no_evidence`
- expectedAnswerable: `False`
- expectedMinCitations: `0`
- expectedMinSpanRows: `0`
- expectedEvidenceTerms: `[]`

### `external_rejection`

```bash
khub labs paper evidence-chunk-ask 'What parsed section or paragraph evidence is available about AlphaFold protein structure prediction?' --paper-id 1207.0580 --allow-external --json
```

- expectedExitCode: `1`
- expectedErrorContains: `--allow-external is not enabled`

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
