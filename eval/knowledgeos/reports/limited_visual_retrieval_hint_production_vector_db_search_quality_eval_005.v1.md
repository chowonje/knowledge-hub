# Limited Visual Retrieval Hint Production Vector DB Search Quality Eval 005

- schema: `knowledge-hub.paper.limited-visual-retrieval-hint-production-vector-db-search-quality-eval.v1`
- status: `ready`
- decision: `ready_for_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design`
- nextRecommendedTranche: `limited_visual_retrieval_hint_runtime_candidate_discovery_route_design`
- sourceProductionVectorRecordRows: `125`
- queryRows: `250`
- textOnlyHitAt5Rows: `133`
- productionVectorHitAt5Rows: `234`
- hybridHitAt5Rows: `241`
- hybridHitAt5LiftRows: `108`
- rankRegressedRows: `0`
- productionVectorIndexWriteRows: `0`
- runtimeVisibleRows: `0`
- strictEvidenceRows: `0`
- schemaViolationCount: `0`

## Quality Gate

- passed: `True`
- thresholds: `{'minProductionVectorHitAt5Rows': 200, 'minHybridHitAt5LiftRows': 25, 'maxRankRegressedRows': 0}`
- observed: `{'productionVectorHitAt5Rows': 234, 'hybridHitAt5LiftRows': 108, 'rankRegressedRows': 0}`
