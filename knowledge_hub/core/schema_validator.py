"""Centralized JSON schema validation helpers.

This module keeps CLI/Service outputs contract-safe without forcing validation
into every call site. It maps contract IDs to docs/schemas/*.json and returns
non-blocking diagnostics when schema files are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = PROJECT_ROOT / "docs" / "schemas"

SCHEMA_NAME_BY_ID = {
    "knowledge-hub.document-memory.build.result.v1": "document-memory-build-result.v1.json",
    "knowledge-hub.document-memory.card.result.v1": "document-memory-card-result.v1.json",
    "knowledge-hub.document-memory.search.result.v1": "document-memory-search-result.v1.json",
    "knowledge-hub.memory-route.result.v1": "memory-route-result.v1.json",
    "knowledge-hub.document-memory.eval.prepare.result.v1": "document-memory-eval-prepare-result.v1.json",
    "knowledge-hub.document-memory.eval.report.v1": "document-memory-eval-report.v1.json",
    "knowledge-hub.memory-router.eval.report.v1": "memory-router-eval-report.v1.json",
    "knowledge-hub.ask-v2.eval.report.v1": "ask-v2-eval-report.v1.json",
    "knowledge-hub.claim-synthesis.eval.prepare.result.v1": "claim-synthesis-eval-prepare-result.v1.json",
    "knowledge-hub.claim-synthesis.eval.report.v1": "claim-synthesis-eval-report.v1.json",
    "knowledge-hub.paper-summary.build.result.v1": "paper-summary-build-result.v1.json",
    "knowledge-hub.paper-summary.card.result.v1": "paper-summary-card-result.v1.json",
    "knowledge-hub.paper-summary.eval.prepare.result.v1": "paper-summary-eval-prepare-result.v1.json",
    "knowledge-hub.paper.public.summary.v1": "paper-public-summary-result.v1.json",
    "knowledge-hub.paper.public.evidence.v1": "paper-public-evidence-result.v1.json",
    "knowledge-hub.paper.public.memory.v1": "paper-public-memory-result.v1.json",
    "knowledge-hub.paper.public.related.v1": "paper-public-related-result.v1.json",
    "knowledge-hub.paper.extraction-report.v1": "paper-extraction-report-result.v1.json",
    "knowledge-hub.paper.parsed-materialization.result.v1": "paper-parsed-materialization-result.v1.json",
    "knowledge-hub.paper.layout-parser-pilot.result.v1": "paper-layout-parser-pilot-result.v1.json",
    "knowledge-hub.paper.arxiv-source-tex-availability-audit.v1": "paper-arxiv-source-tex-availability-audit.v1.json",
    "knowledge-hub.paper.tex-structure-candidate-alignment-audit.v1": "paper-tex-structure-candidate-alignment-audit.v1.json",
    "knowledge-hub.paper.tex-sectionspan-candidate-report.v1": "paper-tex-sectionspan-candidate-report.v1.json",
    "knowledge-hub.paper.tex-figure-caption-candidate-report.v1": "paper-tex-figure-caption-candidate-report.v1.json",
    "knowledge-hub.paper.tex-equation-quote-candidate-report.v1": "paper-tex-equation-quote-candidate-report.v1.json",
    "knowledge-hub.paper.tex-equation-canonical-alignment-diagnostic-audit.v1": "paper-tex-equation-canonical-alignment-diagnostic-audit.v1.json",
    "knowledge-hub.paper.tex-equation-normalization-bridge-audit.v1": "paper-tex-equation-normalization-bridge-audit.v1.json",
    "knowledge-hub.paper.tex-equation-canonical-text-normalizer-design.v1": "paper-tex-equation-canonical-text-normalizer-design.v1.json",
    "knowledge-hub.paper.tex-equation-line-local-anchor-audit.v1": "paper-tex-equation-line-local-anchor-audit.v1.json",
    "knowledge-hub.paper.tex-equation-pdf-region-anchor-audit.v1": "paper-tex-equation-pdf-region-anchor-audit.v1.json",
    "knowledge-hub.paper.parsed-artifact-structured-evidence-execution-plan.v1": "paper-parsed-artifact-structured-evidence-execution-plan.v1.json",
    "knowledge-hub.paper.parsed-artifact-structured-evidence-write-target-contract-audit.v1": "paper-parsed-artifact-structured-evidence-write-target-contract-audit.v1.json",
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-store-contract.v1": "paper-parsed-artifact-source-span-candidate-store-contract.v1.json",
    "knowledge-hub.paper.parsed-artifact-source-span-candidate-record.v1": "paper-parsed-artifact-source-span-candidate-record.v1.json",
    "knowledge-hub.paper.structured-evidence-candidate-record.v1": "paper-structured-evidence-candidate-record.v1.json",
    "knowledge-hub.paper.tex-equation-quote-candidate-v2-design.v1": "paper-tex-equation-quote-candidate-v2-design.v1.json",
    "knowledge-hub.paper.tex-equation-remaining-window-diagnostic.v1": "paper-tex-equation-remaining-window-diagnostic.v1.json",
    "knowledge-hub.paper.tex-equation-segmented-multiline-matching-design.v1": "paper-tex-equation-segmented-multiline-matching-design.v1.json",
    "knowledge-hub.paper.tex-equation-rendered-macro-term-profile-design.v1": "paper-tex-equation-rendered-macro-term-profile-design.v1.json",
    "knowledge-hub.paper.tex-equation-label-number-pdf-region-disambiguation-design.v1": "paper-tex-equation-label-number-pdf-region-disambiguation-design.v1.json",
    "knowledge-hub.paper.tex-equation-source-span-promotion-readiness-audit.v1": "paper-tex-equation-source-span-promotion-readiness-audit.v1.json",
    "knowledge-hub.paper.mineru-normalizer-audit.v1": "paper-mineru-normalizer-audit.v1.json",
    "knowledge-hub.paper.mineru-source-alignment-audit.v1": "paper-mineru-source-alignment-audit.v1.json",
    "knowledge-hub.paper.mineru-potential-review-pack.v1": "paper-mineru-potential-review-pack.v1.json",
    "knowledge-hub.paper.sectionspan-candidate-report.v1": "paper-sectionspan-candidate-report.v1.json",
    "knowledge-hub.paper.figure-caption-candidate-report.v1": "paper-figure-caption-candidate-report.v1.json",
    "knowledge-hub.paper.equation-quote-candidate-report.v1": "paper-equation-quote-candidate-report.v1.json",
    "knowledge-hub.paper.table-region-candidate-report.v1": "paper-table-region-candidate-report.v1.json",
    "knowledge-hub.paper.structured-candidate-summary.v1": "paper-structured-candidate-summary.v1.json",
    "knowledge-hub.paper.parsed-artifact-structured-evidence-readiness-audit.v1": "paper-parsed-artifact-structured-evidence-readiness-audit.v1.json",
    "knowledge-hub.paper.complex-qa-eval-design.v1": "paper-complex-qa-eval-design.v1.json",
    "knowledge-hub.paper.candidate-layer-review-gate.v1": "paper-candidate-layer-review-gate.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-backlog.v1": "paper-candidate-layer-blocker-backlog.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-review-pack.v1": "paper-candidate-layer-blocker-review-pack.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-template.v1": "paper-candidate-layer-blocker-decision-template.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1": "paper-candidate-layer-blocker-decision-record.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-input-pack.v1": "paper-candidate-layer-blocker-decision-input-pack.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-edit-plan.v1": "paper-candidate-layer-blocker-decision-edit-plan.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-file-validation.v1": "paper-candidate-layer-blocker-decision-file-validation.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-decision-file-draft.v1": "paper-candidate-layer-blocker-decision-file-draft.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-manual-decision-review-sheet.v1": "paper-candidate-layer-blocker-manual-decision-review-sheet.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-manual-decision-command-packet.v1": "paper-candidate-layer-blocker-manual-decision-command-packet.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-nonbinding-decision-recommendations.v1": "paper-candidate-layer-blocker-nonbinding-decision-recommendations.v1.json",
    "knowledge-hub.paper.candidate-layer-blocker-resolution-preview.v1": "paper-candidate-layer-blocker-resolution-preview.v1.json",
    "knowledge-hub.paper.candidate-layer-remaining-manual-blocker-review-pack.v1": "paper-candidate-layer-remaining-manual-blocker-review-pack.v1.json",
    "knowledge-hub.paper.non-sectionspan-pdf-offset-feasibility-audit.v1": "paper-non-sectionspan-pdf-offset-feasibility-audit.v1.json",
    "knowledge-hub.paper.source-span-offset-authority-audit.v1": "paper-source-span-offset-authority-audit.v1.json",
    "knowledge-hub.paper.equation-alignment-feasibility-audit.v1": "paper-equation-alignment-feasibility-audit.v1.json",
    "knowledge-hub.paper.equation-quote-next-action-gate.v1": "paper-equation-quote-next-action-gate.v1.json",
    "knowledge-hub.paper.equation-quote-manual-review-sheet.v1": "paper-equation-quote-manual-review-sheet.v1.json",
    "knowledge-hub.paper.equation-quote-decision-file-draft.v1": "paper-equation-quote-decision-file-draft.v1.json",
    "knowledge-hub.paper.equation-quote-decision-file-validation.v1": "paper-equation-quote-decision-file-validation.v1.json",
    "knowledge-hub.paper.equation-quote-decision-record.v1": "paper-equation-quote-decision-record.v1.json",
    "knowledge-hub.paper.equation-quote-decision-next-action-brief.v1": "paper-equation-quote-decision-next-action-brief.v1.json",
    "knowledge-hub.paper.equation-quote-decision-recommendation-pack.v1": "paper-equation-quote-decision-recommendation-pack.v1.json",
    "knowledge-hub.paper.equation-quote-decision-edit-plan.v1": "paper-equation-quote-decision-edit-plan.v1.json",
    "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1": "paper-table-cell-provenance-feasibility-audit.v1.json",
    "knowledge-hub.paper.figure-region-link-feasibility-audit.v1": "paper-figure-region-link-feasibility-audit.v1.json",
    "knowledge-hub.paper.candidate-layer-promotion-policy-draft.v1": "paper-candidate-layer-promotion-policy-draft.v1.json",
    "knowledge-hub.paper.sectionspan-contract-review.v1": "paper-sectionspan-contract-review.v1.json",
    "knowledge-hub.paper.sectionspan-contract-review-pack.v1": "paper-sectionspan-contract-review-pack.v1.json",
    "knowledge-hub.paper.sectionspan-strict-promotion-design.v1": "paper-sectionspan-strict-promotion-design.v1.json",
    "knowledge-hub.paper.sectionspan-source-authority-options.v1": "paper-sectionspan-source-authority-options.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-design.v1": "paper-sectionspan-pdf-offset-recovery-design.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-dry-run.v1": "paper-sectionspan-pdf-offset-recovery-dry-run.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1": "paper-sectionspan-pdf-offset-recovery-review-pack.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1": "paper-sectionspan-pdf-offset-human-review-gate.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-template.v1": "paper-sectionspan-pdf-offset-review-decision-template.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-record.v1": "paper-sectionspan-pdf-offset-review-decision-record.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-review-priority-pack.v1": "paper-sectionspan-pdf-offset-review-priority-pack.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-packet.v1": "paper-sectionspan-pdf-offset-selected-review-packet.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1": "paper-sectionspan-pdf-offset-selected-review-decision-template.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-draft.v1": "paper-sectionspan-pdf-offset-selected-review-decision-file-draft.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-file-validation.v1": "paper-sectionspan-pdf-offset-selected-review-decision-file-validation.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-manual-sheet.v1": "paper-sectionspan-pdf-offset-selected-review-manual-sheet.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-next-action-brief.v1": "paper-sectionspan-pdf-offset-selected-review-next-action-brief.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-record.v1": "paper-sectionspan-pdf-offset-selected-review-decision-record.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-evidence-pack.v1": "paper-sectionspan-pdf-offset-selected-review-evidence-pack.v1.json",
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-proposal.v1": "paper-sectionspan-pdf-offset-selected-review-decision-proposal.v1.json",
    "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1": "paper-figure-caption-pdf-offset-feasibility.v1.json",
    "knowledge-hub.paper.figure-caption-region-link-review-pack.v1": "paper-figure-caption-region-link-review-pack.v1.json",
    "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1": "paper-equation-quote-pdf-offset-feasibility.v1.json",
    "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1": "paper-table-region-pdf-offset-feasibility.v1.json",
    "knowledge-hub.paper.table-cell-provenance-review-pack.v1": "paper-table-cell-provenance-review-pack.v1.json",
    "knowledge-hub.paper.table-cell-bbox-source-span-authority-design.v1": "paper-table-cell-bbox-source-span-authority-design.v1.json",
    "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1": "paper-table-cell-bbox-source-span-extractor-pilot.v1.json",
    "knowledge-hub.paper.table-cell-probe-result-review-pack.v1": "paper-table-cell-probe-result-review-pack.v1.json",
    "knowledge-hub.paper.table-cell-pymupdf-overlay-review-pack.v1": "paper-table-cell-pymupdf-overlay-review-pack.v1.json",
    "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1": "paper-table-cell-pymupdf-pairing-diagnostic.v1.json",
    "knowledge-hub.paper.table-cell-next-action-gate.v1": "paper-table-cell-next-action-gate.v1.json",
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-plan.v1": "paper-table-cell-isolated-extractor-pilot-plan.v1.json",
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-result.v1": "paper-table-cell-isolated-extractor-pilot-result.v1.json",
    "knowledge-hub.paper.canon-quality-audit.result.v1": "paper-canon-quality-audit-result.v1.json",
    "knowledge-hub.paper.board-export.v1": "paper-board-export-result.v1.json",
    "knowledge-hub.paper-memory.build.result.v1": "paper-memory-build-result.v1.json",
    "knowledge-hub.paper-memory.card.result.v1": "paper-memory-card-result.v1.json",
    "knowledge-hub.paper-memory.search.result.v1": "paper-memory-search-result.v1.json",
    "knowledge-hub.paper-knowledge-slots.v1": "paper-knowledge-slots.v1.json",
    "knowledge-hub.paper-topic-synthesis.result.v1": "paper-topic-synthesis-result.v1.json",
    "knowledge-hub.paper-memory.eval.report.v1": "paper-memory-eval-report.v1.json",
    "knowledge-hub.paper-topic-synthesis.result.v1": "paper-topic-synthesis-result.v1.json",
    "knowledge-hub.claim-normalize.result.v1": "claim-normalize-result.v1.json",
    "knowledge-hub.claim-compare.result.v1": "claim-compare-result.v1.json",
    "knowledge-hub.claim-synthesis.result.v1": "claim-synthesis-result.v1.json",
    "knowledge-hub.ai-watchlist-batch.v2": "ai-watchlist-batch.v2.json",
    "knowledge-hub.foundry.agent.run.result.v1": "agent-run-result.v1.json",
    "knowledge-hub.agent.writeback.request.result.v1": "agent-writeback-request-result.v1.json",
    "knowledge-hub.agent.discover.result.v1": "agent-discover-result.v1.json",
    "knowledge-hub.foundry.connector.sync.result.v2": "connector-sync-result.v2.json",
    "knowledge-hub.authority.result-envelope.v1": "authority-result-envelope.v1.json",
    "knowledge-hub.learning.map.result.v1": "learning-map-result.v1.json",
    "knowledge-hub.learning.start-resume.result.v1": "learning-start-resume-result.v1.json",
    "knowledge-hub.learning.session-state.result.v1": "learning-session-state-result.v1.json",
    "knowledge-hub.learning.explain.result.v1": "learning-explain-result.v1.json",
    "knowledge-hub.learning.checkpoint.result.v1": "learning-checkpoint-result.v1.json",
    "knowledge-hub.learning.grade.result.v1": "learning-grade-result.v1.json",
    "knowledge-hub.learning.next.result.v1": "learning-next-result.v1.json",
    "knowledge-hub.learning.template.result.v1": "learning-template-result.v1.json",
    "knowledge-hub.learning.run.result.v1": "learning-run-result.v1.json",
    "knowledge-hub.learning.gap.result.v1": "learning-gap-result.v1.json",
    "knowledge-hub.learning.quiz.generate.result.v1": "learning-quiz-generate-result.v1.json",
    "knowledge-hub.learning.quiz.grade.result.v1": "learning-quiz-grade-result.v1.json",
    "knowledge-hub.learning.patch.suggest.result.v1": "learning-patch-suggest-result.v1.json",
    "knowledge-hub.learning.reinforcement.result.v1": "learning-reinforcement-result.v1.json",
    "knowledge-hub.learning.graph.build.result.v1": "learning-graph-build-result.v1.json",
    "knowledge-hub.learning.graph.pending.result.v1": "learning-graph-pending-result.v1.json",
    "knowledge-hub.learning.path.result.v1": "learning-path-result.v1.json",
    "knowledge-hub.learning.review.result.v1": "learning-review-result.v1.json",
    "knowledge-hub.crawl.ingest.result.v1": "crawl-ingest-result.v1.json",
    "knowledge-hub.crawl.collect.result.v1": "crawl-collect-result.v1.json",
    "knowledge-hub.dinger.ingest.result.v1": "dinger-ingest-result.v1.json",
    "knowledge-hub.dinger.ask.result.v1": "dinger-ask-result.v1.json",
    "knowledge-hub.dinger.capture.result.v1": "dinger-capture-result.v1.json",
    "knowledge-hub.dinger.capture-list.result.v1": "dinger-capture-list-result.v1.json",
    "knowledge-hub.dinger.capture-status.result.v1": "dinger-capture-status-result.v1.json",
    "knowledge-hub.dinger.capture-show.result.v1": "dinger-capture-show-result.v1.json",
    "knowledge-hub.dinger.capture-process.result.v1": "dinger-capture-process-result.v1.json",
    "knowledge-hub.dinger.capture-cleanup.result.v1": "dinger-capture-cleanup-result.v1.json",
    "knowledge-hub.dinger.capture-requeue.result.v1": "dinger-capture-requeue-result.v1.json",
    "knowledge-hub.dinger.capture-retry.result.v1": "dinger-capture-retry-result.v1.json",
    "knowledge-hub.dinger.file.result.v1": "dinger-file-result.v1.json",
    "knowledge-hub.dinger.recent.result.v1": "dinger-recent-result.v1.json",
    "knowledge-hub.dinger.lint.result.v1": "dinger-lint-result.v1.json",
    "knowledge-hub.normalized.web-record.v2": "normalized-web-record.v2.json",
    "knowledge-hub.indexed.web-record.v2": "indexed-web-record.v2.json",
    "knowledge-hub.prepared-source-record.v1": "prepared-source-record.v1.json",
    "knowledge-hub.source-ledger-record.v1": "source-ledger-record.v1.json",
    "knowledge-hub.crawl.pending.list.result.v1": "crawl-pending-list-result.v1.json",
    "knowledge-hub.crawl.pending.apply.result.v1": "crawl-pending-apply-result.v1.json",
    "knowledge-hub.crawl.pending.reject.result.v1": "crawl-pending-reject-result.v1.json",
    "knowledge-hub.crawl.pipeline.run.result.v1": "crawl-pipeline-run-result.v1.json",
    "knowledge-hub.crawl.domain.policy.result.v1": "crawl-domain-policy-result.v1.json",
    "knowledge-hub.crawl.benchmark.result.v1": "crawl-benchmark-result.v1.json",
    "knowledge-hub.entity.merge.list.result.v1": "entity-merge-list-result.v1.json",
    "knowledge-hub.entity.merge.apply.result.v1": "entity-merge-apply-result.v1.json",
    "knowledge-hub.entity.merge.reject.result.v1": "entity-merge-reject-result.v1.json",
    "knowledge-hub.ko-note.generate.result.v1": "ko-note-generate-result.v1.json",
    "knowledge-hub.ko-note.status.result.v1": "ko-note-status-result.v1.json",
    "knowledge-hub.ko-note.report.result.v1": "ko-note-report-result.v1.json",
    "knowledge-hub.ko-note.apply.result.v1": "ko-note-apply-result.v1.json",
    "knowledge-hub.ko-note.enrich.result.v1": "ko-note-enrich-result.v1.json",
    "knowledge-hub.ko-note.review.list.result.v1": "ko-note-review-list-result.v1.json",
    "knowledge-hub.ko-note.review.result.v1": "ko-note-review-result.v1.json",
    "knowledge-hub.ko-note.remediate.result.v1": "ko-note-remediate-result.v1.json",
    "knowledge-hub.rag.report.result.v1": "rag-report-result.v1.json",
    "knowledge-hub.rag.corrective-report.result.v1": "rag-corrective-report-result.v1.json",
    "knowledge-hub.rag.corrective-eval.report.v1": "rag-corrective-eval-report.v1.json",
    "knowledge-hub.rag.adaptive-plan.result.v1": "rag-adaptive-plan-result.v1.json",
    "knowledge-hub.rag.corrective-run.result.v1": "rag-corrective-run-result.v1.json",
    "knowledge-hub.rag.corrective-execution-review.result.v1": "rag-corrective-execution-review-result.v1.json",
    "knowledge-hub.rag.answerability-rerank.result.v1": "rag-answerability-rerank-result.v1.json",
    "knowledge-hub.rag.answerability-rerank-eval.report.v1": "rag-answerability-rerank-eval-report.v1.json",
    "knowledge-hub.rag.graph-global-plan.result.v1": "rag-graph-global-plan-result.v1.json",
    "knowledge-hub.rag.vnext-observation.report.v1": "rag-vnext-observation-report.v1.json",
    "knowledge-hub.rag.visualization.result.v1": "rag-visualization-result.v1.json",
    "knowledge-hub.runtime.diagnostics.v1": "runtime-diagnostics.v1.json",
    "knowledge-hub.doctor.result.v1": "doctor-result.v1.json",
    "knowledge-hub.retrieval.eval.report.v1": "retrieval-eval-report.v1.json",
    "knowledge-hub.eval.gate.result.v1": "eval-gate-result.v1.json",
    "knowledge-hub.answer-eval.packet.v1": "answer-eval-packet.v1.json",
    "knowledge-hub.answer-eval.result.v1": "answer-eval-result.v1.json",
    "knowledge-hub.evidence-packet.v1": "evidence-packet.v1.json",
    "knowledge-hub.answer-contract.v1": "answer-contract.v1.json",
    "knowledge-hub.verification-verdict.v1": "verification-verdict.v1.json",
    "knowledge-hub.compare-packet.v1": "compare-packet.v1.json",
    "knowledge-hub.evidence-substrate.contract.v1": "evidence-substrate-contract.v1.json",
    "knowledge-hub.inspect.result.v1": "inspect-result.v1.json",
    "knowledge-hub.compare.result.v1": "compare-result.v1.json",
    "knowledge-hub.answer-trace.result.v1": "answer-trace-result.v1.json",
    "knowledge-hub.mcp.resource.result.v1": "mcp-resource-result.v1.json",
    "knowledge-hub.evidence-registry.record.v1": "evidence-registry-record.v1.json",
    "knowledge-hub.evidence-registry.lookup.result.v1": "evidence-registry-lookup-result.v1.json",
    "knowledge-hub.evidence-registry.write.result.v1": "evidence-registry-write-result.v1.json",
    "knowledge-hub.answer-loop.collect.result.v1": "answer-loop-collect-result.v1.json",
    "knowledge-hub.answer-loop.judge.result.v1": "answer-loop-judge-result.v1.json",
    "knowledge-hub.answer-loop.summary.result.v1": "answer-loop-summary-result.v1.json",
    "knowledge-hub.answer-loop.autofix.result.v1": "answer-loop-autofix-result.v1.json",
    "knowledge-hub.answer-loop.run.result.v1": "answer-loop-run-result.v1.json",
    "knowledge-hub.answer-loop.optimize.result.v1": "answer-loop-optimize-result.v1.json",
    "knowledge-hub.ops.report.run.result.v1": "ops-report-run-result.v1.json",
    "knowledge-hub.ops.action.list.result.v1": "ops-action-list-result.v1.json",
    "knowledge-hub.ops.action.result.v1": "ops-action-result.v1.json",
    "knowledge-hub.ops.action.execute.result.v1": "ops-action-execute-result.v1.json",
    "knowledge-hub.ops.action.receipts.result.v1": "ops-action-receipts-result.v1.json",
    "knowledge-hub.ko-note.enrich.status.result.v1": "ko-note-enrich-status-result.v1.json",
    "knowledge-hub.vault.topology.snapshot.v1": "vault-topology-snapshot.v1.json",
    "knowledge-hub.vault.cluster.materialize.manifest.v1": "vault-cluster-materialize-manifest.v1.json",
    "knowledge-hub.task-context.result.v1": "task-context-result.v1.json",
    "knowledge-hub.context-pack.result.v1": "context-pack-result.v1.json",
    "knowledge-hub.eval-center.summary.result.v1": "eval-center-summary-result.v1.json",
    "knowledge-hub.eval-case.v1": "eval-case.v1.json",
    "knowledge-hub.eval-case-registry.result.v1": "eval-case-registry-result.v1.json",
    "knowledge-hub.failure-bank.sync.result.v1": "failure-bank-sync-result.v1.json",
    "knowledge-hub.failure-bank.list.result.v1": "failure-bank-list-result.v1.json",
    "knowledge-hub.failure-bank.link-eval-cases.result.v1": "failure-bank-link-eval-cases-result.v1.json",
    "knowledge-hub.os.project.create.result.v1": "os-project-create-result.v1.json",
    "knowledge-hub.os.project.list.result.v1": "os-project-list-result.v1.json",
    "knowledge-hub.os.project.update.result.v1": "os-project-update-result.v1.json",
    "knowledge-hub.os.project.show.result.v1": "os-project-show-result.v1.json",
    "knowledge-hub.os.project.evidence.result.v1": "os-project-evidence-result.v1.json",
    "knowledge-hub.os.evidence.show.result.v1": "os-evidence-show-result.v1.json",
    "knowledge-hub.os.evidence.review.result.v1": "os-evidence-review-result.v1.json",
    "knowledge-hub.os.capture.result.v1": "os-capture-result.v1.json",
    "knowledge-hub.os.goal.add.result.v1": "os-goal-add-result.v1.json",
    "knowledge-hub.os.goal.update.result.v1": "os-goal-update-result.v1.json",
    "knowledge-hub.os.task.add.result.v1": "os-task-add-result.v1.json",
    "knowledge-hub.os.task.update.result.v1": "os-task-update-result.v1.json",
    "knowledge-hub.os.task.update-status.result.v1": "os-task-update-status-result.v1.json",
    "knowledge-hub.os.task.start.result.v1": "os-task-start-result.v1.json",
    "knowledge-hub.os.task.block.result.v1": "os-task-block-result.v1.json",
    "knowledge-hub.os.task.complete.result.v1": "os-task-complete-result.v1.json",
    "knowledge-hub.os.task.cancel.result.v1": "os-task-cancel-result.v1.json",
    "knowledge-hub.os.inbox.triage.result.v1": "os-inbox-triage-result.v1.json",
    "knowledge-hub.os.decide.result.v1": "os-decide-result.v1.json",
    "knowledge-hub.os.decision.add.result.v1": "os-decision-add-result.v1.json",
    "knowledge-hub.os.decision.list.result.v1": "os-decision-list-result.v1.json",
    "knowledge-hub.os.inbox.list.result.v1": "os-inbox-list-result.v1.json",
    "knowledge-hub.os.inbox.resolve.result.v1": "os-inbox-resolve-result.v1.json",
    "knowledge-hub.os.next.result.v1": "os-next-result.v1.json",
    "knowledge-hub.os.project.export.obsidian.result.v1": "os-project-export-obsidian-result.v1.json",
    "knowledge-hub.notebook.provider.list.result.v1": "notebook-provider-list-result.v1.json",
    "knowledge-hub.notebook.provider.local-models.result.v1": "notebook-provider-local-models-result.v1.json",
    "knowledge-hub.notebook.topic.preview.result.v1": "notebook-topic-preview-result.v1.json",
    "knowledge-hub.notebook.topic.create.result.v1": "notebook-topic-create-result.v1.json",
    "knowledge-hub.notebook.topic.sync.result.v1": "notebook-topic-sync-result.v1.json",
    "knowledge-hub.transform.list.result.v1": "transform-list-result.v1.json",
    "knowledge-hub.transform.preview.result.v1": "transform-preview-result.v1.json",
    "knowledge-hub.transform.run.result.v1": "transform-run-result.v1.json",
    "knowledge-hub.ask-graph.result.v1": "ask-graph-result.v1.json",
    "knowledge-hub.workbench.search.result.v1": "workbench-search-result.v1.json",
    "knowledge-hub.workbench.chat.result.v1": "workbench-chat-result.v1.json",
    "knowledge-hub.paper.judge.result.v1": "paper-judge-result.v1.json",
    "knowledge-hub.paper.discover.result.v1": "paper-discover-result.v1.json",
}


@dataclass
class SchemaValidationResult:
    ok: bool
    errors: list[str]
    schema: str
    schema_found: bool


def _schema_file_for(schema_id: str) -> Path | None:
    file_name = SCHEMA_NAME_BY_ID.get(schema_id)
    if not file_name:
        return None
    candidate = SCHEMA_ROOT / file_name
    return candidate if candidate.exists() else None


def validate_payload(payload: dict[str, Any], schema_id: str, *, strict: bool = False) -> SchemaValidationResult:
    """Validate payload against registered JSON schema.

    Returns non-blocking validation failures by default (strict=False), so existing
    call sites can decide whether to escalate.
    """
    schema_path = _schema_file_for(schema_id)
    if schema_path is None:
        message = f"schema not found for id: {schema_id}"
        return SchemaValidationResult(ok=not strict, errors=[message], schema=schema_id, schema_found=False)

    return _validate_loaded(payload, schema_id, schema_path, strict=strict)


def _validate_loaded(payload: dict[str, Any], schema_id: str, schema_path: Path, strict: bool) -> SchemaValidationResult:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as error:
        return SchemaValidationResult(
            ok=not strict,
            errors=[f"schema load failed: {schema_path}: {error}"],
            schema=schema_id,
            schema_found=False,
        )

    try:
        validator = Draft202012Validator(schema)
        def _format_path(error_path) -> str:
            parts = list(error_path)
            return "/" + "/".join(str(part) for part in parts) if parts else "(root)"

        errors = [f"{_format_path(item.path)}: {item.message}" for item in sorted(validator.iter_errors(payload), key=lambda item: item.path)]
    except Exception as error:
        return SchemaValidationResult(
            ok=not strict,
            errors=[f"schema validate failed: {error}"],
            schema=schema_id,
            schema_found=True,
        )

    return SchemaValidationResult(ok=(not errors or not strict), errors=errors, schema=schema_id, schema_found=True)


def annotate_schema_errors(payload: dict[str, Any], schema_id: str, *, strict: bool = False) -> SchemaValidationResult:
    result = validate_payload(payload, schema_id, strict=strict)
    if result.errors:
        existing = list(payload.get("schemaErrors", []))
        for error in result.errors:
            if error not in existing:
                existing.append(error)
        payload["schemaErrors"] = existing
    return result
