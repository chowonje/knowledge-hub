# Limited Visual Retrieval Hint Runtime Candidate Discovery Route Design 005

- schema: `knowledge-hub.paper.limited-visual-retrieval-hint-runtime-candidate-discovery-route-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_final_merge_review`
- nextRecommendedTranche: `final_merge_review_and_branch_cleanup`
- plannedRouteBindingRows: `125`
- candidateDiscoveryOnlyRows: `125`
- qualityEvalProductionVectorHitAt5Rows: `234`
- qualityEvalHybridHitAt5Rows: `241`
- qualityEvalHybridHitAt5LiftRows: `108`
- qualityEvalRankRegressedRows: `0`
- blockedRows: `0`
- productionVectorIndexWriteRows: `0`
- operationalSearchIndexQueryRows: `0`
- runtimeRouteWriteRows: `0`
- runtimeVisibleRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`
- answerableWithoutTextEvidenceRows: `0`

## Gate

- passed: `True`
- plannedRouteBindingsExactly125: `True`
- noMutationOrRuntimeExposure: `True`

## Runtime Placement

- routeName: `visual_retrieval_hint_candidate_discovery`
- routeMode: `candidate_discovery_only`
- targetRuntimeBoundary: `knowledge_hub.ai.rag_search_runtime.RAGSearchRuntime.search_with_diagnostics`
- targetCollectionName: `knowledge_hub_visual_retrieval_hints`

## Non-Scope

- No runtime route write.
- No operational search index query.
- No answer-visible exposure.
- No evidence promotion.
- No graph DB, ontology, memory card, or clustering write.
