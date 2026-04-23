from __future__ import annotations

from typing import Any

from knowledge_hub.ai.rag_decision_log import emit_rag_decision_log
from knowledge_hub.ai.retrieval_pipeline import RetrievalPipelineService


class RAGSearchRuntime:
    def __init__(self, searcher: Any):
        self.searcher = searcher

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
        semantic_top_k: int | None = None,
        lexical_top_k: int | None = None,
        expand_parent_context: bool = False,
        use_ontology_expansion: bool = True,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            return {"results": [], "diagnostics": {}}

        pipeline_result = RetrievalPipelineService(self.searcher).execute(
            query=query,
            top_k=top_k,
            source_type=source_type,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            semantic_top_k=semantic_top_k,
            lexical_top_k=lexical_top_k,
            use_ontology_expansion=use_ontology_expansion,
            metadata_filter=metadata_filter,
        )
        final = list(pipeline_result.results)
        if expand_parent_context:
            final = self.searcher._apply_parent_context(final, include_parent_text=True)
        diagnostics = pipeline_result.diagnostics()
        retrieval_plan = dict(diagnostics.get("retrievalPlan") or {})
        emit_rag_decision_log(
            stage="search_with_diagnostics",
            query=query,
            source_type=source_type,
            query_plan=retrieval_plan.get("queryPlan"),
            query_frame=retrieval_plan.get("queryFrame"),
            memory_route=diagnostics.get("memoryRoute"),
            memory_prefilter=diagnostics.get("memoryPrefilter"),
        )
        return {"results": final, "diagnostics": diagnostics}
