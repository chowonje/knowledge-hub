"""Optional MinerU PDF adapter for labs-only paper parsing."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


_MINERU_COMMAND = ("mineru",)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_path_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    token = _clean_text(value)
    if not token:
        return []
    return [part.strip() for part in token.split(">") if part.strip()]


def _normalize_bbox(value: Any) -> list[float] | dict[str, float] | None:
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for key in ("x0", "y0", "x1", "y1", "left", "top", "right", "bottom", "width", "height"):
            if key in value:
                try:
                    out[str(key)] = float(value[key])
                except Exception:
                    continue
        return out or None
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value[:4]:
            try:
                out.append(float(item))
            except Exception:
                return None
        return out if out else None
    return None


def _normalize_page(value: Any) -> int | None:
    try:
        page = int(value)
    except Exception:
        return None
    return page if page >= 0 else None


def _normalize_reading_order(value: Any) -> int | None:
    try:
        order = int(value)
    except Exception:
        return None
    return order if order >= 0 else None


def _element_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    text = _clean_text(
        node.get("text")
        or node.get("content")
        or node.get("markdown")
        or node.get("label")
        or node.get("caption")
    )
    element_type = _clean_text(
        node.get("type")
        or node.get("kind")
        or node.get("label_type")
        or node.get("category")
        or node.get("semantic_type")
    ).lower()
    page = _normalize_page(
        node.get("page")
        or node.get("page_number")
        or node.get("page_index")
    )
    bbox = _normalize_bbox(
        node.get("bbox")
        or node.get("bounding_box")
        or node.get("box")
        or node.get("rect")
        or node.get("coordinates")
    )
    heading_path = _clean_path_list(
        node.get("heading_path")
        or node.get("section_path")
        or node.get("path")
    )
    reading_order = _normalize_reading_order(
        node.get("reading_order")
        or node.get("order")
        or node.get("order_index")
        or node.get("position")
        or node.get("index")
    )
    if not text and not element_type and page is None and bbox is None:
        return None
    return {
        "type": element_type or "element",
        "text": text,
        "page": page,
        "bbox": bbox,
        "heading_path": heading_path,
        "reading_order": reading_order,
    }


def _walk_elements(value: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        item = _element_from_node(value)
        if item:
            out.append(item)
        for nested in value.values():
            _walk_elements(nested, out)
        return
    if isinstance(value, list):
        for nested in value:
            _walk_elements(nested, out)


def _dedupe_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int | None, str, int | None]] = set()
    result: list[dict[str, Any]] = []
    for item in elements:
        bbox = item.get("bbox")
        bbox_key = json.dumps(bbox, sort_keys=True, ensure_ascii=False) if bbox is not None else ""
        key = (
            str(item.get("type") or ""),
            str(item.get("text") or ""),
            item.get("page"),
            bbox_key,
            item.get("reading_order"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_generated_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (-path.stat().st_size, str(path)))[0]


@dataclass(slots=True)
class MinerUParseResult:
    markdown_text: str
    elements: list[dict[str, Any]]
    parser_meta: dict[str, Any]
    artifact_dir: str
    markdown_path: str
    json_path: str
    manifest_path: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "markdown_text": self.markdown_text,
            "elements": list(self.elements),
            "parser_meta": dict(self.parser_meta),
            "artifact_dir": self.artifact_dir,
            "markdown_path": self.markdown_path,
            "json_path": self.json_path,
            "manifest_path": self.manifest_path,
        }


class MinerUPDFAdapter:
    def __init__(self, *, papers_dir: str):
        self.papers_dir = Path(str(papers_dir)).expanduser()

    def artifact_dir_for(self, *, paper_id: str) -> Path:
        return self.papers_dir / "parsed" / str(paper_id).strip()

    def _manifest_path(self, *, paper_id: str) -> Path:
        return self.artifact_dir_for(paper_id=paper_id) / "manifest.json"

    def _load_existing(self, *, paper_id: str) -> MinerUParseResult | None:
        manifest_path = self._manifest_path(paper_id=paper_id)
        if not manifest_path.exists():
            return None
        payload = _load_json(manifest_path)
        markdown_path = Path(str(payload.get("markdown_path") or "").strip())
        json_path = Path(str(payload.get("json_path") or "").strip())
        if not markdown_path.exists() or not json_path.exists():
            return None
        markdown_text = markdown_path.read_text(encoding="utf-8")
        document_payload = _load_json(json_path)
        return MinerUParseResult(
            markdown_text=markdown_text,
            elements=list(document_payload.get("elements") or []),
            parser_meta=dict(document_payload.get("parser_meta") or payload.get("parser_meta") or {}),
            artifact_dir=str(self.artifact_dir_for(paper_id=paper_id)),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            manifest_path=str(manifest_path),
        )

    def ensure_artifacts(self, *, paper_id: str, pdf_path: str, refresh: bool = False) -> MinerUParseResult:
        token = str(paper_id).strip()
        existing = None if refresh else self._load_existing(paper_id=token)
        if existing is not None:
            return existing

        source_pdf = Path(str(pdf_path).strip())
        if not source_pdf.exists():
            raise RuntimeError(f"paper pdf not found: {source_pdf}")
        try:
            version = str(importlib.metadata.version("mineru"))
        except Exception as error:
            raise RuntimeError("mineru is not installed; install it to use --paper-parser mineru") from error

        artifact_dir = self.artifact_dir_for(paper_id=token)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if refresh:
            for child in artifact_dir.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child, ignore_errors=True)

        with tempfile.TemporaryDirectory(prefix=f"khub-mineru-{token}-") as tmp_dir:
            temp_root = Path(tmp_dir)
            command = [*_MINERU_COMMAND, "-p", str(source_pdf), "-o", str(temp_root), "-b", "pipeline"]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except FileNotFoundError as error:
                raise RuntimeError("mineru CLI not found; install MinerU to use --paper-parser mineru") from error
            except subprocess.CalledProcessError as error:
                stderr = _clean_text(error.stderr or error.stdout or "")
                detail = stderr or "unknown MinerU parser error"
                raise RuntimeError(f"mineru parse failed: {detail}") from error

            generated_markdown = _find_generated_file(temp_root, (".md", ".markdown"))
            generated_json = _find_generated_file(temp_root, (".json",))
            if generated_markdown is None or generated_json is None:
                raise RuntimeError("mineru conversion did not produce markdown/json outputs")
            markdown_text = generated_markdown.read_text(encoding="utf-8")
            raw_payload = _load_json(generated_json)

        elements: list[dict[str, Any]] = []
        _walk_elements(raw_payload, elements)
        normalized_elements = _dedupe_elements(elements)
        parser_meta = {
            "parser": "mineru",
            "mode": "local",
            "version": version,
            "source_pdf": str(source_pdf),
            "command": "mineru -p <pdf> -o <dir> -b pipeline",
        }
        markdown_path = artifact_dir / "document.md"
        json_path = artifact_dir / "document.json"
        manifest_path = artifact_dir / "manifest.json"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "markdown_text": markdown_text,
                    "elements": normalized_elements,
                    "parser_meta": parser_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "paper_id": token,
                    "parser_meta": parser_meta,
                    "markdown_path": str(markdown_path),
                    "json_path": str(json_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return MinerUParseResult(
            markdown_text=markdown_text,
            elements=normalized_elements,
            parser_meta=parser_meta,
            artifact_dir=str(artifact_dir),
            markdown_path=str(markdown_path),
            json_path=str(json_path),
            manifest_path=str(manifest_path),
        )


__all__ = ["MinerUPDFAdapter", "MinerUParseResult"]
