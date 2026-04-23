"""OpenAI Batch API support for paper-memory rebuilds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemorySchemaExtractor
from knowledge_hub.providers.policy_guard import evaluate_outbound_policy_batch


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(value or "").strip())
    token = "-".join(part for part in token.split("-") if part)
    return token or "batch"


def _custom_id_for_paper(paper_id: str) -> str:
    return f"paper-memory::{_clean_text(paper_id)}"


def _paper_id_from_custom_id(custom_id: Any) -> str:
    token = str(custom_id or "").strip()
    if token.startswith("paper-memory::"):
        return token.split("::", 1)[1].strip()
    return token


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _chat_completion_kwargs(model: str, *, max_completion_tokens: int = 520) -> dict[str, Any]:
    token = str(model or "").strip()
    if token.startswith("gpt-5"):
        return {"max_completion_tokens": int(max_completion_tokens)}
    return {"max_tokens": int(max_completion_tokens), "temperature": 0.0}


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\d*\b", re.IGNORECASE)
_LATEX_AUTHOR_RE = re.compile(r"\\(?:correspondingauthor|author|affil|thanks)\{[^{}]*\}", re.IGNORECASE)


def _sanitize_external_value(value: Any) -> Any:
    if isinstance(value, str):
        body = _LATEX_AUTHOR_RE.sub(" ", value)
        body = _EMAIL_RE.sub("[redacted-email]", body)
        return _clean_text(redact_p0(body))
    if isinstance(value, dict):
        return {str(key): _sanitize_external_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_external_value(item) for item in value]
    return value


@dataclass
class BatchPrepareResult:
    manifest_path: Path
    requests_path: Path
    summary_path: Path
    paper_count: int
    blocked_count: int


class OpenAIPaperMemoryBatchService:
    endpoint = "/v1/chat/completions"

    def __init__(self, khub, *, model: str = "gpt-5.4") -> None:
        self.khub = khub
        self.sqlite_db = khub.sqlite_db()
        self.model = _clean_text(model) or "gpt-5.4"
        self._builder = PaperMemoryBuilder(self.sqlite_db)
        self._extractor = PaperMemorySchemaExtractor(llm=None, model=self.model)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            cfg = dict(self.khub.config.get_provider_config("openai") or {})
            api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not configured for OpenAI Batch API")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def default_output_dir(self, *, root: str | None = None, model: str | None = None) -> Path:
        chosen_model = _clean_text(model or self.model) or "gpt-5.4"
        if root:
            path = Path(str(root)).expanduser()
        else:
            db_path = Path(str(self.khub.config.sqlite_path)).expanduser()
            stamp = datetime.now().strftime("%Y-%m-%d")
            path = db_path.parent / "runs" / f"paper_memory_openai_batch_{stamp}_{_safe_slug(chosen_model)}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepare(
        self,
        *,
        output_dir: str | Path | None = None,
        paper_ids: list[str] | None = None,
        paper_ids_file: str | None = None,
        limit: int = 50,
        model: str | None = None,
    ) -> BatchPrepareResult:
        chosen_model = _clean_text(model or self.model) or "gpt-5.4"
        output_root = self.default_output_dir(root=str(output_dir) if output_dir else None, model=chosen_model)
        resolved_ids = self._resolve_paper_ids(paper_ids=paper_ids, paper_ids_file=paper_ids_file, limit=limit)
        requests_path = output_root / "requests.jsonl"
        manifest_path = output_root / "manifest.json"
        summary_path = output_root / "prepare_summary.json"
        items: list[dict[str, Any]] = []
        policy_inputs: list[str] = []
        request_lines: list[dict[str, Any]] = []

        for paper_id in resolved_ids:
            row = self.sqlite_db.get_paper(paper_id)
            if not row:
                continue
            compact_input = self._builder.build_compact_extraction_input(paper_id=paper_id)
            external_input = _sanitize_external_value(compact_input)
            policy_inputs.append(json.dumps(external_input, ensure_ascii=False))
            items.append(
                {
                    "paperId": paper_id,
                    "title": str(row.get("title") or ""),
                    "year": str(row.get("year") or ""),
                    "compactInput": external_input,
                    "customId": _custom_id_for_paper(paper_id),
                }
            )

        report = evaluate_outbound_policy_batch(provider="openai", model=chosen_model, texts=policy_inputs)
        blocked = set(int(index) for index in report.blocked_indices)

        for index, item in enumerate(items):
            item["policy"] = {
                "blocked": index in blocked,
                "traceId": report.trace_id,
                "warnings": list(report.warnings),
            }
            if index in blocked:
                item["requestStatus"] = "blocked"
                continue
            prompt = self._extractor.build_prompt(paper=item["compactInput"])
            request_lines.append(
                {
                    "custom_id": item["customId"],
                    "method": "POST",
                    "url": self.endpoint,
                    "body": {
                        "model": chosen_model,
                        "messages": [{"role": "user", "content": prompt}],
                        **_chat_completion_kwargs(chosen_model),
                    },
                }
            )
            item["requestStatus"] = "queued"

        with requests_path.open("w", encoding="utf-8") as handle:
            for line in request_lines:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")

        manifest = {
            "schema": "knowledge-hub.paper-memory.openai-batch.manifest.v1",
            "createdAt": _utc_now_iso(),
            "model": chosen_model,
            "endpoint": self.endpoint,
            "outputDir": str(output_root),
            "requestsPath": str(requests_path),
            "outputPath": str(output_root / "output.jsonl"),
            "errorPath": str(output_root / "errors.jsonl"),
            "items": items,
            "policy": report.to_dict(),
            "batch": {
                "status": "prepared",
                "inputFileId": "",
                "batchId": "",
                "outputFileId": "",
                "errorFileId": "",
            },
        }
        _dump_json(manifest_path, manifest)
        prepare_summary = {
            "paperCount": len(items),
            "queuedCount": sum(1 for item in items if item.get("requestStatus") == "queued"),
            "blockedCount": sum(1 for item in items if item.get("requestStatus") == "blocked"),
            "requestsPath": str(requests_path),
            "manifestPath": str(manifest_path),
            "traceId": report.trace_id,
        }
        _dump_json(summary_path, prepare_summary)
        return BatchPrepareResult(
            manifest_path=manifest_path,
            requests_path=requests_path,
            summary_path=summary_path,
            paper_count=len(items),
            blocked_count=prepare_summary["blockedCount"],
        )

    def submit(self, *, manifest_path: str | Path) -> dict[str, Any]:
        manifest = self._load_manifest(manifest_path)
        requests_path = Path(str(manifest.get("requestsPath") or "")).expanduser()
        with requests_path.open("rb") as handle:
            input_file = self.client.files.create(file=handle, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint=self.endpoint,
            completion_window="24h",
            metadata={"kind": "paper-memory-openai-batch", "model": str(manifest.get("model") or self.model)},
        )
        manifest["batch"] = {
            "status": str(batch.status),
            "inputFileId": str(input_file.id),
            "batchId": str(batch.id),
            "outputFileId": str(getattr(batch, "output_file_id", "") or ""),
            "errorFileId": str(getattr(batch, "error_file_id", "") or ""),
            "submittedAt": _utc_now_iso(),
        }
        self._write_manifest(manifest_path, manifest)
        return dict(manifest["batch"])

    def status(self, *, manifest_path: str | Path) -> dict[str, Any]:
        manifest = self._load_manifest(manifest_path)
        batch_id = str(((manifest.get("batch") or {}).get("batchId")) or "").strip()
        if not batch_id:
            return dict(manifest.get("batch") or {})
        batch = self.client.batches.retrieve(batch_id)
        manifest["batch"] = {
            **dict(manifest.get("batch") or {}),
            "status": str(batch.status),
            "inputFileId": str(getattr(batch, "input_file_id", "") or ""),
            "outputFileId": str(getattr(batch, "output_file_id", "") or ""),
            "errorFileId": str(getattr(batch, "error_file_id", "") or ""),
            "requestCounts": dict(getattr(batch, "request_counts", None) or {}),
            "completedAt": str(getattr(batch, "completed_at", "") or ""),
            "failedAt": str(getattr(batch, "failed_at", "") or ""),
            "expiredAt": str(getattr(batch, "expired_at", "") or ""),
        }
        self._write_manifest(manifest_path, manifest)
        return dict(manifest["batch"])

    def download(self, *, manifest_path: str | Path) -> dict[str, Any]:
        manifest = self._load_manifest(manifest_path)
        batch_info = dict(manifest.get("batch") or {})
        output_path = Path(str(manifest.get("outputPath") or "")).expanduser()
        error_path = Path(str(manifest.get("errorPath") or "")).expanduser()
        output_file_id = str(batch_info.get("outputFileId") or "").strip()
        error_file_id = str(batch_info.get("errorFileId") or "").strip()
        if output_file_id:
            content = self.client.files.content(output_file_id)
            output_path.write_text(content.text, encoding="utf-8")
        if error_file_id:
            content = self.client.files.content(error_file_id)
            error_path.write_text(content.text, encoding="utf-8")
        batch_info["downloadedAt"] = _utc_now_iso()
        manifest["batch"] = batch_info
        self._write_manifest(manifest_path, manifest)
        return {
            "outputPath": str(output_path),
            "errorPath": str(error_path),
            "outputExists": output_path.exists(),
            "errorExists": error_path.exists(),
        }

    def apply(self, *, manifest_path: str | Path) -> dict[str, Any]:
        manifest = self._load_manifest(manifest_path)
        output_path = Path(str(manifest.get("outputPath") or "")).expanduser()
        if not output_path.exists():
            batch_info = dict(manifest.get("batch") or {})
            if str(batch_info.get("outputFileId") or "").strip():
                self.download(manifest_path=manifest_path)
        output_path = Path(str(manifest.get("outputPath") or "")).expanduser()
        if not output_path.exists():
            raise RuntimeError("batch output file is not available yet")

        extractor = PaperMemorySchemaExtractor(llm=None, model=str(manifest.get("model") or self.model))
        applied_rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []
        for raw_line in output_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            custom_id = str(payload.get("custom_id") or "")
            paper_id = _paper_id_from_custom_id(custom_id)
            response = dict(payload.get("response") or {})
            if int(response.get("status_code") or 0) != 200:
                failed_rows.append({"paperId": paper_id, "customId": custom_id, "reason": "non_200_status"})
                continue
            body = dict(response.get("body") or {})
            choices = list(body.get("choices") or [])
            message = dict((choices[0] or {}).get("message") or {}) if choices else {}
            content = message.get("content") or ""
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict) and str(item.get("type") or "") == "text":
                        chunks.append(str(item.get("text") or ""))
                content = "\n".join(chunks)
            try:
                raw_payload = extractor.coerce_payload(content)
                stored = self._builder.apply_extraction_payload_and_store(
                    paper_id=paper_id,
                    raw_payload=raw_payload,
                    extractor_model=str(manifest.get("model") or self.model),
                )
                applied_rows.append(
                    {
                        "paperId": paper_id,
                        "title": str(stored.get("title") or ""),
                        "qualityFlag": str(stored.get("quality_flag") or ""),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                failed_rows.append({"paperId": paper_id, "customId": custom_id, "reason": str(exc)})

        output_root = Path(str(manifest.get("outputDir") or "")).expanduser()
        apply_csv_path = output_root / "apply_results.csv"
        apply_summary_path = output_root / "apply_summary.json"
        fieldnames = ["paperId", "title", "qualityFlag", "status"]
        with apply_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(applied_rows)
        apply_summary = {
            "appliedCount": len(applied_rows),
            "failedCount": len(failed_rows),
            "applyResultsCsv": str(apply_csv_path),
            "failed": failed_rows,
            "appliedAt": _utc_now_iso(),
        }
        _dump_json(apply_summary_path, apply_summary)
        manifest["apply"] = apply_summary
        self._write_manifest(manifest_path, manifest)
        return apply_summary

    def _resolve_paper_ids(
        self,
        *,
        paper_ids: list[str] | None,
        paper_ids_file: str | None,
        limit: int,
    ) -> list[str]:
        explicit = [_clean_text(item) for item in list(paper_ids or []) if _clean_text(item)]
        if explicit:
            return list(dict.fromkeys(explicit))
        file_token = str(paper_ids_file or "").strip()
        if file_token:
            rows = Path(file_token).expanduser().read_text(encoding="utf-8", errors="ignore").splitlines()
            cleaned = [_clean_text(row) for row in rows if _clean_text(row) and not _clean_text(row).startswith("#")]
            return list(dict.fromkeys(cleaned))
        result: list[str] = []
        for row in self.sqlite_db.list_papers(limit=max(1, int(limit))):
            paper_id = _clean_text(row.get("arxiv_id"))
            if paper_id:
                result.append(paper_id)
        return result

    def _load_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        path = Path(str(manifest_path)).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest_path: str | Path, manifest: dict[str, Any]) -> None:
        _dump_json(Path(str(manifest_path)).expanduser(), manifest)


__all__ = ["BatchPrepareResult", "OpenAIPaperMemoryBatchService"]
