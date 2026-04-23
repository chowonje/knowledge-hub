"""Combined scheduled ops report runner."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.application.ko_note_reports import build_ko_note_report
from knowledge_hub.application.ops_actions import queue_item_view
from knowledge_hub.application.paper_reports import build_paper_source_ops_report
from knowledge_hub.application.rag_reports import build_rag_ops_report
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.notes.materializer import KoNoteMaterializer

_OPS_NOTE_BLOCK_START = "<!-- KHUB_OPS_REPORT:start -->"
_OPS_NOTE_BLOCK_END = "<!-- KHUB_OPS_REPORT:end -->"
_OPS_NOTE_BLOCK_PATTERN = re.compile(
    re.escape(_OPS_NOTE_BLOCK_START) + r"[\s\S]*?" + re.escape(_OPS_NOTE_BLOCK_END),
    re.MULTILINE,
)
_OPS_REPORT_SCHEMA = "knowledge-hub.ops.report.run.result.v1"
_OPS_REPORT_POLICY_VERSION = "scheduled-ops-report-v1"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snapshot_slug(ts: datetime | None = None) -> str:
    current = ts or _now_utc()
    return current.strftime("%Y%m%dT%H%M%S.%fZ")


def _dedupe_dicts(items: list[dict[str, Any]], *, key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _combined_status(*, ko_status: str, rag_status: str, alerts: list[dict[str, Any]], warnings: list[str]) -> str:
    if ko_status != "ok" or rag_status != "ok":
        return "failed"
    if warnings:
        return "warning"
    for alert in alerts:
        severity = str(alert.get("severity") or "").strip().lower()
        if severity in {"warning", "critical"}:
            return "warning"
    return "ok"


def _alert_counts(alerts: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"info": 0, "warning": 0, "critical": 0}
    for alert in alerts:
        severity = str(alert.get("severity") or "").strip().lower()
        if severity in counts:
            counts[severity] += 1
    counts["total"] = sum(counts.values())
    return counts


def _action_target(*, action: dict[str, Any], run_id: str, rag_days: int, rag_limit: int) -> tuple[str, str]:
    scope = str(action.get("scope") or "").strip()
    if scope == "ko_note":
        return "ko_note_run", f"run:{run_id}"
    if scope == "rag":
        return "rag_window", f"window:days={int(rag_days)};limit={int(rag_limit)}"
    if scope == "paper":
        paper_id = str(action.get("paperId") or "").strip()
        args = [str(item) for item in (action.get("args") or [])]
        if not paper_id and "--paper-id" in args:
            index = args.index("--paper-id")
            if index + 1 < len(args):
                paper_id = str(args[index + 1]).strip()
        return "paper", f"paper:{paper_id or 'unknown'}"
    return "ops_scope", f"scope:{scope or 'unknown'}"


def _upsert_managed_block(original: str, block: str) -> str:
    body = str(original or "")
    replacement = f"\n\n{block}\n"
    if _OPS_NOTE_BLOCK_PATTERN.search(body):
        updated = _OPS_NOTE_BLOCK_PATTERN.sub(replacement, body).rstrip()
        return updated + "\n"
    stripped = body.rstrip()
    if stripped:
        return f"{stripped}{replacement}"
    return block + "\n"


class OpsReportRunner:
    def __init__(
        self,
        config,
        *,
        sqlite_db=None,
        materializer: KoNoteMaterializer | None = None,
        searcher: RAGSearcher | None = None,
        artifact_root: str | Path | None = None,
        note_path: str | Path | None = None,
    ):
        self.config = config
        self.sqlite_db = sqlite_db or SQLiteDatabase(self.config.sqlite_path)
        self.materializer = materializer or KoNoteMaterializer(self.config, sqlite_db=self.sqlite_db)
        self.searcher = searcher or RAGSearcher(
            embedder=None,
            database=VectorDatabase(self.config.vector_db_path, self.config.collection_name),
            llm=None,
            sqlite_db=self.sqlite_db,
            config=self.config,
        )
        self._artifact_root_override = Path(artifact_root).expanduser().resolve() if artifact_root else None
        self._note_path_override = Path(note_path).expanduser().resolve() if note_path else None

    def artifact_root(self) -> Path:
        return self._artifact_root_override or (Path.home() / ".khub" / "ops-reports").resolve()

    def note_path(self) -> Path | None:
        if self._note_path_override is not None:
            return self._note_path_override
        vault_path = str(getattr(self.config, "vault_path", "") or "").strip()
        if not vault_path:
            return None
        return (Path(vault_path).expanduser().resolve() / "LearningHub" / "ops" / "Knowledge Hub Ops Report.md").resolve()

    def _resolve_run_id(self, explicit_run_id: str | None) -> tuple[str, list[str]]:
        if str(explicit_run_id or "").strip():
            return str(explicit_run_id).strip(), []
        getter = getattr(self.sqlite_db, "get_latest_ko_note_run", None)
        if not callable(getter):
            return "", ["latest ko note run lookup unavailable"]
        latest = getter() or {}
        run_id = str(latest.get("run_id") or "").strip()
        if run_id:
            return run_id, []
        return "", ["latest completed ko note run not found"]

    def _fallback_ko_note_report(self, run_id: str, warnings: list[str]) -> dict[str, Any]:
        return {
            "schema": "knowledge-hub.ko-note.report.result.v1",
            "status": "failed",
            "runId": str(run_id),
            "run": {},
            "recentRuns": [],
            "recentSummary": {},
            "alerts": [],
            "recommendedActions": [],
            "warnings": list(warnings),
            "ts": _now_iso(),
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _render_note(self, payload: dict[str, Any]) -> str:
        ko_report = dict(payload.get("koNoteReport") or {})
        rag_report = dict(payload.get("ragReport") or {})
        run = dict(ko_report.get("run") or {})
        rag_counts = dict(rag_report.get("counts") or {})
        rag_rates = dict(rag_report.get("rates") or {})
        alert_counts = dict(payload.get("alertCounts") or {})
        action_queue = dict(payload.get("actionQueue") or {})
        lines = [
            _OPS_NOTE_BLOCK_START,
            f"## Snapshot",
            f"- Timestamp: `{payload.get('ts')}`",
            f"- Status: `{payload.get('status')}`",
            f"- Policy: `{payload.get('policyVersion')}`",
            f"- ko-note run: `{payload.get('koNoteRunId') or 'none'}`",
            f"- Alerts: total={alert_counts.get('total', 0)} warning={alert_counts.get('warning', 0)} critical={alert_counts.get('critical', 0)}",
            f"- Action queue: pending={((action_queue.get('counts') or {}).get('pending', 0))} acked={((action_queue.get('counts') or {}).get('acked', 0))} resolved={((action_queue.get('counts') or {}).get('resolved', 0))}",
            "",
            "## Ko-Note",
            f"- Generated: source={run.get('sourceGenerated', 0)} concept={run.get('conceptGenerated', 0)}",
            f"- Statuses: staged={((run.get('counts') or {}).get('staged', 0))} approved={((run.get('counts') or {}).get('approved', 0))} applied={((run.get('counts') or {}).get('applied', 0))} rejected={((run.get('counts') or {}).get('rejected', 0))}",
            f"- Review queued: {(((run.get('reviewQueue') or {}).get('combined') or {}).get('total', 0))}",
            f"- Auto approved: {((run.get('autoApproved') or {}).get('total', 0))}",
            "",
            "## RAG",
            f"- Window: days={rag_report.get('days', 0)} limit={rag_report.get('limit', 0)} total={rag_counts.get('total', 0)}",
            f"- Verification: needsCaution={rag_counts.get('needsCaution', 0)} unsupportedLogs={rag_counts.get('unsupportedClaimLogs', 0)}",
            f"- Rewrite/Fallback: rewriteApplied={rag_counts.get('rewriteApplied', 0)} fallback={rag_counts.get('conservativeFallback', 0)}",
            f"- Rates: caution={rag_rates.get('needsCautionRate', 0.0)} unsupported={rag_rates.get('unsupportedClaimRate', 0.0)} fallback={rag_rates.get('conservativeFallbackRate', 0.0)}",
            "",
            "## Alerts",
        ]
        alerts = list(payload.get("alerts") or [])
        if alerts:
            for alert in alerts[:10]:
                lines.append(f"- [{alert.get('severity')}] `{alert.get('code')}`: {alert.get('summary')}")
        else:
            lines.append("- none")
        lines.extend(["", "## Recommended Actions"])
        actions = list(payload.get("recommendedActions") or [])
        if actions:
            for action in actions[:5]:
                command = " ".join(
                    [str(action.get("command") or ""), *[str(item) for item in (action.get("args") or [])]]
                ).strip()
                lines.append(f"- {action.get('summary')}")
                if command:
                    lines.append(f"  - `{command}`")
        else:
            lines.append("- none")
        pending_actions = list((action_queue.get("pendingActions") or []))
        lines.extend(["", "## Pending Queue"])
        if pending_actions:
            for action in pending_actions[:5]:
                command = " ".join([str(action.get("command") or ""), *[str(item) for item in (action.get("args") or [])]]).strip()
                suffix_bits = []
                if str(action.get("lastExecutionStatus") or "").strip():
                    suffix_bits.append(f"last={action.get('lastExecutionStatus')}")
                if str(action.get("status") or "").strip():
                    suffix_bits.append(f"queue={action.get('status')}")
                suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
                lines.append(f"- `{action.get('actionId')}` {action.get('summary')}{suffix}")
                if command:
                    lines.append(f"  - `{command}`")
                if str(action.get("lastResultSummary") or "").strip():
                    lines.append(f"  - {action.get('lastResultSummary')}")
        else:
            lines.append("- none")
        artifact_paths = dict(payload.get("artifactPaths") or {})
        lines.extend(["", "## Artifacts"])
        for key in ("snapshotDir", "opsReportJson", "koNoteReportJson", "ragReportJson"):
            value = str(artifact_paths.get(key) or "").strip()
            if value:
                lines.append(f"- {key}: `{value}`")
        paper_report_path = str(artifact_paths.get("paperSourceReportJson") or "").strip()
        if paper_report_path:
            lines.append(f"- paperSourceReportJson: `{paper_report_path}`")
        warnings = [str(item).strip() for item in (payload.get("warnings") or []) if str(item).strip()]
        if warnings:
            lines.extend(["", "## Warnings"])
            for warning in warnings[:10]:
                lines.append(f"- {warning}")
        lines.append(_OPS_NOTE_BLOCK_END)
        return "\n".join(lines)

    def _write_note(self, payload: dict[str, Any]) -> tuple[str, list[str]]:
        note_path = self.note_path()
        if note_path is None:
            return "", ["vault_path not configured; ops note skipped"]
        note_path.parent.mkdir(parents=True, exist_ok=True)
        original = note_path.read_text(encoding="utf-8") if note_path.exists() else "# Knowledge Hub Ops Report\n"
        updated = _upsert_managed_block(original, self._render_note(payload))
        note_path.write_text(updated, encoding="utf-8")
        return str(note_path), []

    def _enqueue_actions(
        self,
        *,
        actions: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
        run_id: str,
        rag_days: int,
        rag_limit: int,
        ts: str,
    ) -> dict[str, Any]:
        upsert = getattr(self.sqlite_db, "upsert_ops_action", None)
        lister = getattr(self.sqlite_db, "list_ops_actions", None)
        counter = getattr(self.sqlite_db, "count_ops_actions", None)
        if not callable(upsert) or not callable(lister) or not callable(counter):
            return {
                "created": 0,
                "updated": 0,
                "reopened": 0,
                "counts": {"pending": 0, "acked": 0, "resolved": 0, "total": 0},
                "pendingActions": [],
                "warnings": ["ops action queue unavailable"],
            }
        summary = {"created": 0, "updated": 0, "reopened": 0}
        latest_receipt_getter = getattr(self.sqlite_db, "get_latest_ops_action_receipt", None)
        for action in actions:
            target_kind, target_key = _action_target(
                action=action,
                run_id=run_id,
                rag_days=rag_days,
                rag_limit=rag_limit,
            )
            reason_codes = [str(item) for item in (action.get("reasonCodes") or []) if str(item).strip()]
            linked_alerts = [
                dict(alert)
                for alert in alerts
                if str(alert.get("scope") or "") == str(action.get("scope") or "")
                and str(alert.get("code") or "") in set(reason_codes)
            ]
            result = upsert(
                scope=str(action.get("scope") or ""),
                action_type=str(action.get("actionType") or ""),
                target_kind=target_kind,
                target_key=target_key,
                summary=str(action.get("summary") or ""),
                reason_codes=reason_codes,
                command=str(action.get("command") or ""),
                args=[str(item) for item in (action.get("args") or [])],
                alerts=linked_alerts,
                action=dict(action),
                seen_at=ts,
            )
            operation = str((result or {}).get("operation") or "")
            if operation in summary:
                summary[operation] += 1
        counts = counter()
        pending_actions = []
        for item in lister(status="pending", scope=None, limit=3):
            latest_receipt = (
                latest_receipt_getter(str(item.get("action_id") or "")) or {}
                if callable(latest_receipt_getter)
                else {}
            )
            view = queue_item_view(item, latest_receipt=latest_receipt)
            pending_actions.append(
                {
                    "actionId": str(view.get("actionId") or ""),
                    "scope": str(view.get("scope") or ""),
                    "actionType": str(view.get("actionType") or ""),
                    "status": str(view.get("status") or ""),
                    "summary": str(view.get("summary") or ""),
                    "command": str(view.get("command") or ""),
                    "args": [str(arg) for arg in (view.get("args") or [])],
                    "targetKind": str(view.get("targetKind") or ""),
                    "targetKey": str(view.get("targetKey") or ""),
                    "seenCount": int(view.get("seenCount") or 0),
                    "lastSeenAt": str(view.get("lastSeenAt") or ""),
                    "lastExecutionStatus": str(view.get("lastExecutionStatus") or ""),
                    "lastResultSummary": str(view.get("lastResultSummary") or ""),
                }
            )
        return {
            **summary,
            "counts": counts,
            "pendingActions": pending_actions,
            "warnings": [],
        }

    def _apply_retention(self, *, keep: int) -> None:
        root = self.artifact_root()
        if not root.exists():
            return
        keep_count = max(1, int(keep))
        candidates = [path for path in root.iterdir() if path.is_dir()]
        if len(candidates) <= keep_count:
            return
        candidates.sort(key=lambda item: item.name)
        latest_successful: Path | None = None
        for candidate in reversed(candidates):
            combined_path = candidate / "ops-report.json"
            if not combined_path.exists():
                continue
            try:
                payload = json.loads(combined_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("status") or "") != "failed":
                latest_successful = candidate
                break
        removable = list(candidates)
        while len(removable) > keep_count:
            candidate = removable.pop(0)
            if latest_successful is not None and candidate == latest_successful:
                removable.append(candidate)
                continue
            for child in sorted(candidate.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            candidate.rmdir()

    def run(
        self,
        *,
        run_id: str | None = None,
        recent_runs: int = 10,
        rag_days: int = 7,
        rag_limit: int = 100,
        retention: int = 30,
    ) -> dict[str, Any]:
        selected_run_id, run_id_warnings = self._resolve_run_id(run_id)
        ko_report = (
            build_ko_note_report(self.sqlite_db, run_id=selected_run_id, recent_runs=max(1, int(recent_runs)))
            if selected_run_id
            else self._fallback_ko_note_report(selected_run_id, run_id_warnings)
        )
        rag_report = build_rag_ops_report(self.sqlite_db, limit=max(1, int(rag_limit)), days=max(0, int(rag_days)))
        paper_report = build_paper_source_ops_report(self.sqlite_db)
        combined_alerts = _dedupe_dicts(
            [
                *list(ko_report.get("alerts") or []),
                *list(rag_report.get("alerts") or []),
                *list(paper_report.get("alerts") or []),
            ],
            key_fields=("scope", "code", "summary"),
        )
        combined_actions = _dedupe_dicts(
            [
                *list(ko_report.get("recommendedActions") or []),
                *list(rag_report.get("recommendedActions") or []),
                *list(paper_report.get("recommendedActions") or []),
            ],
            key_fields=("scope", "actionType", "summary", "command"),
        )
        warnings = [
            *run_id_warnings,
            *[str(item).strip() for item in (ko_report.get("warnings") or []) if str(item).strip()],
            *[str(item).strip() for item in (rag_report.get("warnings") or []) if str(item).strip()],
            *[str(item).strip() for item in (paper_report.get("warnings") or []) if str(item).strip()],
        ]
        warnings = list(dict.fromkeys(warnings))
        status = _combined_status(
            ko_status=str(ko_report.get("status") or ""),
            rag_status=str(rag_report.get("status") or ""),
            alerts=combined_alerts,
            warnings=warnings,
        )

        ts_dt = _now_utc()
        ts = ts_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        snapshot_dir = self.artifact_root() / _snapshot_slug(ts_dt)
        artifact_paths = {
            "root": str(self.artifact_root()),
            "snapshotDir": str(snapshot_dir),
            "opsReportJson": str(snapshot_dir / "ops-report.json"),
            "koNoteReportJson": str(snapshot_dir / "ko-note-report.json"),
            "ragReportJson": str(snapshot_dir / "rag-report.json"),
        }
        payload = {
            "schema": _OPS_REPORT_SCHEMA,
            "status": status,
            "policyVersion": _OPS_REPORT_POLICY_VERSION,
            "ts": ts,
            "koNoteRunId": str(selected_run_id or ""),
            "koNoteReport": ko_report,
            "ragReport": rag_report,
            "alertCounts": _alert_counts(combined_alerts),
            "alerts": combined_alerts,
            "recommendedActions": combined_actions[:10],
            "artifactPaths": artifact_paths,
            "notePath": "",
            "actionQueue": {},
            "warnings": warnings,
        }
        action_queue = self._enqueue_actions(
            actions=combined_actions,
            alerts=combined_alerts,
            run_id=str(selected_run_id or ""),
            rag_days=max(0, int(rag_days)),
            rag_limit=max(1, int(rag_limit)),
            ts=ts,
        )
        payload["actionQueue"] = action_queue
        queue_warnings = [str(item).strip() for item in (action_queue.get("warnings") or []) if str(item).strip()]
        if queue_warnings:
            payload["warnings"] = list(dict.fromkeys([*payload["warnings"], *queue_warnings]))
            if payload["status"] == "ok":
                payload["status"] = "warning"
        self._write_json(Path(artifact_paths["koNoteReportJson"]), ko_report)
        self._write_json(Path(artifact_paths["ragReportJson"]), rag_report)
        note_path, note_warnings = self._write_note(payload)
        if note_warnings:
            payload["warnings"] = list(dict.fromkeys([*payload["warnings"], *note_warnings]))
            if payload["status"] == "ok":
                payload["status"] = "warning"
        payload["notePath"] = note_path
        self._write_json(Path(artifact_paths["opsReportJson"]), payload)
        self._apply_retention(keep=retention)
        return payload
