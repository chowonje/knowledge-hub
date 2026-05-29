# Parsed Artifact Evidence Chunk Answer Path Opt-in Implementation Design

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-opt-in-implementation-design.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_opt_in_searcher_ingress_implementation`
- designRows: `7`
- designReadyRows: `7`
- blockedRows: `0`
- plannedSearcherIngressRows: `2`
- plannedRuntimeForwardingRows: `2`
- publicSearcherIngressGapConfirmed: `True`
- plannedPublicCliChange: `False`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Design

- ingressMode: `internal_python_api_opt_in`
- targetClass: `knowledge_hub.ai.rag.RAGSearcher`
- targetMethods: `generate_answer, stream_answer`
- queryPlanParameter: `query_plan`
- defaultBehavior: `unchanged_when_query_plan_is_none`
- publicCliFlag: `None`

## Policy

Report-only implementation design. It fixes the future searcher ingress contract; it does not change runtime code, add public CLI flags, generate answers, call LLMs, mutate stores, or scan the vault.

## Rows

### `rag_searcher_generate_answer_add_query_plan_param`

- layer: `public_python_searcher_api`
- status: `design_ready`
- file: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- plannedChange: Add keyword-only query_plan: Optional[Dict[str, Any]] = None to RAGSearcher.generate_answer.
- blockers: ``
- nextCheck: Signature keeps default None so existing callers remain compatible.

### `rag_searcher_generate_answer_forward_query_plan`

- layer: `runtime_forwarding`
- status: `design_ready`
- file: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- plannedChange: Forward query_plan=query_plan from RAGSearcher.generate_answer to rag_answer_runtime.generate_answer.
- blockers: ``
- nextCheck: Unit test should monkeypatch generate_answer_runtime and assert object identity or equality for query_plan.

### `rag_searcher_stream_answer_add_query_plan_param`

- layer: `public_python_searcher_api`
- status: `design_ready`
- file: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- plannedChange: Add keyword-only query_plan: Optional[Dict[str, Any]] = None to RAGSearcher.stream_answer for parity.
- blockers: ``
- nextCheck: Streaming remains opt-in and default behavior is unchanged when query_plan is None.

### `rag_searcher_stream_answer_forward_query_plan`

- layer: `runtime_forwarding`
- status: `design_ready`
- file: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- plannedChange: Forward query_plan=query_plan from RAGSearcher.stream_answer to rag_answer_runtime.stream_answer.
- blockers: ``
- nextCheck: Unit test should monkeypatch stream_answer_runtime and assert query_plan is passed through.

### `khub_ask_no_public_flag`

- layer: `public_cli`
- status: `design_ready`
- file: `knowledge_hub/interfaces/cli/commands/search_cmd.py`
- currentObserved: `True`
- plannedChange: Keep khub ask public flags unchanged; no parsed-artifact evidence chunk CLI flag in this tranche.
- blockers: ``
- nextCheck: CLI tests should verify no public opt-in flag is introduced by the searcher ingress implementation.

### `default_none_preserves_behavior`

- layer: `compatibility`
- status: `design_ready`
- file: `knowledge_hub/ai/rag.py`
- currentObserved: `True`
- plannedChange: Keep query_plan optional and default None so absent opt-in preserves existing search/ask behavior.
- blockers: ``
- nextCheck: Regression test should call generate_answer without query_plan and assert forwarded value is None.

### `activation_remains_adapter_owned`

- layer: `answerability_policy`
- status: `design_ready`
- file: `knowledge_hub/ai/parsed_artifact_evidence_chunk_runtime_adapter.py`
- currentObserved: `True`
- plannedChange: Do not duplicate activation policy in RAGSearcher; adapter remains responsible for opt-in, source type, resolved-paper, hash, locator, and row-cap checks.
- blockers: ``
- nextCheck: Implementation should only pass query_plan through and leave adapter gating unchanged.
