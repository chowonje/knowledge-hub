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
    "knowledge-hub.paper.canon-quality-audit.result.v1": "paper-canon-quality-audit-result.v1.json",
    "knowledge-hub.paper.board-export.v1": "paper-board-export-result.v1.json",
    "knowledge-hub.paper-memory.build.result.v1": "paper-memory-build-result.v1.json",
    "knowledge-hub.paper-memory.card.result.v1": "paper-memory-card-result.v1.json",
    "knowledge-hub.paper-memory.search.result.v1": "paper-memory-search-result.v1.json",
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
    "knowledge-hub.runtime.diagnostics.v1": "runtime-diagnostics.v1.json",
    "knowledge-hub.doctor.result.v1": "doctor-result.v1.json",
    "knowledge-hub.retrieval.eval.report.v1": "retrieval-eval-report.v1.json",
    "knowledge-hub.eval.gate.result.v1": "eval-gate-result.v1.json",
    "knowledge-hub.eval-center.summary.result.v1": "eval-center-summary-result.v1.json",
    "knowledge-hub.failure-bank.sync.result.v1": "failure-bank-sync-result.v1.json",
    "knowledge-hub.failure-bank.list.result.v1": "failure-bank-list-result.v1.json",
    "knowledge-hub.answer-eval.packet.v1": "answer-eval-packet.v1.json",
    "knowledge-hub.answer-eval.result.v1": "answer-eval-result.v1.json",
    "knowledge-hub.evidence-packet.v1": "evidence-packet.v1.json",
    "knowledge-hub.answer-contract.v1": "answer-contract.v1.json",
    "knowledge-hub.verification-verdict.v1": "verification-verdict.v1.json",
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
