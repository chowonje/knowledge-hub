# Parsed Artifact Evidence Chunk Answer Path Opt-in Route Review

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-opt-in-route-review.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design`
- answerQualitySmokePassRows: `2`
- routeReviewRows: `8`
- routeReviewPassRows: `7`
- routeReviewGapRows: `1`
- routeReviewFailRows: `0`
- publicSearcherIngressGap: `True`
- publicCliDefaultUnchanged: `True`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Policy

Report-only route review. It inspects the current answer path and records where the opt-in can flow next; it does not change runtime behavior, add public CLI flags, generate answers, call LLMs, mutate stores, or scan the vault.

## Route Rows

### `answer_runtime_request_query_plan`

- layer: `runtime_request`
- status: `pass`
- file: `knowledge_hub/ai/rag_answer_runtime.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `runtime_wrapper_query_plan`

- layer: `runtime_wrapper`
- status: `pass`
- file: `knowledge_hub/ai/rag_answer_runtime.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `legacy_execution_passes_query_plan_to_evidence_assembly`

- layer: `legacy_runtime_execution`
- status: `pass`
- file: `knowledge_hub/ai/rag_answer_runtime.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `ask_v2_execution_passes_query_plan_to_evidence_assembly`

- layer: `ask_v2_runtime_execution`
- status: `pass`
- file: `knowledge_hub/ai/ask_v2.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `evidence_assembly_adapter_wired`

- layer: `evidence_assembly`
- status: `pass`
- file: `knowledge_hub/ai/evidence_assembly.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `answer_payload_exposes_contracts`

- layer: `answer_payload`
- status: `pass`
- file: `knowledge_hub/ai/answer_payload_builder.py`
- observed: `True`
- gapReason: ``
- recommendation: ``

### `public_searcher_generate_answer_query_plan_ingress`

- layer: `public_searcher_api`
- status: `gap`
- file: `knowledge_hub/ai/rag.py`
- observed: `False`
- gapReason: `public_searcher_generate_answer_does_not_accept_query_plan`
- recommendation: `Add an internal/labs-only opt-in ingress before attempting end-to-end khub ask route smoke.`

### `khub_ask_has_no_public_strict_chunk_flag`

- layer: `public_cli`
- status: `pass`
- file: `knowledge_hub/interfaces/cli/commands/search_cmd.py`
- observed: `True`
- gapReason: ``
- recommendation: ``
