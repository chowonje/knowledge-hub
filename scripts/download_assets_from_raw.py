#!/usr/bin/env python3
"""Download high-volume assets linked from already crawled raw HTML files.

Usage example:
  python scripts/download_assets_from_raw.py \
    --raw-dir /Volumes/T9/khub/raw_web \
    --out-dir /Volumes/T9/khub/assets \
    --workers 24 \
    --max-total-gb 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "KnowledgeHub-AssetDownloader/1.0"
DEFAULT_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".ppt",
    ".pptx",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".parquet",
    ".jsonl",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
}
SKIP_CONTENT_TYPE_PREFIXES = (
    "text/html",
    "text/plain",
    "application/json",
    "application/javascript",
    "text/javascript",
    "text/css",
    "application/xml",
    "text/xml",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(value: str, fallback: str = "file") -> str:
    value = value.strip()
    if not value:
        return fallback
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip().replace(" ", "_")
    return result or fallback


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return ""
    clean = parsed._replace(fragment="")
    return urlunparse(clean)


def _guess_ext(url: str, content_type: str = "") -> str:
    path = urlparse(url).path or ""
    ext = Path(path).suffix.lower()
    if ext:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed.lower()
    return ".bin"


def _target_path(out_dir: Path, url: str, content_type: str = "") -> Path:
    parsed = urlparse(url)
    domain = _sanitize_name(parsed.netloc.lower().lstrip("www."), fallback="unknown-domain")
    path = parsed.path or "/"
    stem = _sanitize_name(Path(path).stem, fallback="file")
    ext = _guess_ext(url, content_type=content_type)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    rel = Path(domain) / f"{stem}-{digest}{ext}"
    return out_dir / rel


def _extract_links_from_html(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    attrs = ("href", "src", "data-src", "data-href", "content", "poster")

    for tag in soup.find_all(True):
        for attr in attrs:
            value = tag.get(attr)
            if not value or not isinstance(value, str):
                continue
            url = _normalize_url(urljoin(base_url, value.strip()))
            if url:
                links.add(url)
    return links


def _candidate_from_url(url: str, allow_all: bool) -> bool:
    if allow_all:
        return True
    path = urlparse(url).path.lower()
    ext = Path(path).suffix.lower()
    return ext in DEFAULT_EXTENSIONS


@dataclass
class DownloadResult:
    url: str
    status: str
    bytes_written: int = 0
    path: str = ""
    error: str = ""
    content_type: str = ""
    source_url: str = ""
    note_id: str = ""


class Budget:
    def __init__(self, max_total_bytes: int):
        self.max_total_bytes = max_total_bytes
        self.used_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, amount: int) -> bool:
        if self.max_total_bytes <= 0:
            return True
        with self.lock:
            if self.used_bytes + amount > self.max_total_bytes:
                return False
            self.used_bytes += amount
            return True

    def add_used(self, amount: int) -> None:
        if amount <= 0:
            return
        with self.lock:
            self.used_bytes += amount

    def release(self, amount: int) -> None:
        if amount <= 0:
            return
        with self.lock:
            self.used_bytes = max(0, self.used_bytes - amount)


def _download_one(
    task: tuple[str, str, str],
    out_dir: Path,
    timeout: int,
    max_file_bytes: int,
    min_file_bytes: int,
    budget: Budget,
    retries: int,
) -> DownloadResult:
    url, source_url, note_id = task
    last_error = ""
    headers = {"User-Agent": USER_AGENT}

    for _attempt in range(retries + 1):
        bytes_written = 0
        reserved_bytes = 0
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as response:
                if response.status_code >= 400:
                    return DownloadResult(
                        url=url,
                        status="http_error",
                        error=f"status={response.status_code}",
                        source_url=source_url,
                        note_id=note_id,
                    )
                content_type = (response.headers.get("Content-Type") or "").lower().strip()
                content_length = int(response.headers.get("Content-Length") or "0")

                if any(content_type.startswith(prefix) for prefix in SKIP_CONTENT_TYPE_PREFIXES):
                    return DownloadResult(
                        url=url,
                        status="skip_text",
                        content_type=content_type,
                        source_url=source_url,
                        note_id=note_id,
                    )

                if max_file_bytes > 0 and content_length > max_file_bytes > 0:
                    return DownloadResult(
                        url=url,
                        status="skip_too_large",
                        error=f"content_length={content_length}",
                        content_type=content_type,
                        source_url=source_url,
                        note_id=note_id,
                    )

                if content_length > 0:
                    if not budget.reserve(content_length):
                        return DownloadResult(
                            url=url,
                            status="skip_budget",
                            error="max_total_bytes exceeded",
                            content_type=content_type,
                            source_url=source_url,
                            note_id=note_id,
                        )
                    reserved_bytes = content_length

                dest = _target_path(out_dir, url, content_type=content_type)
                dest.parent.mkdir(parents=True, exist_ok=True)

                if dest.exists() and dest.stat().st_size > 0:
                    if reserved_bytes > 0:
                        budget.release(reserved_bytes)
                    return DownloadResult(
                        url=url,
                        status="exists",
                        bytes_written=dest.stat().st_size,
                        path=str(dest),
                        content_type=content_type,
                        source_url=source_url,
                        note_id=note_id,
                    )

                with dest.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if max_file_bytes > 0 and bytes_written > max_file_bytes:
                            f.close()
                            dest.unlink(missing_ok=True)
                            if reserved_bytes > 0:
                                budget.release(reserved_bytes)
                            return DownloadResult(
                                url=url,
                                status="skip_too_large",
                                error=f"stream_exceeded={bytes_written}",
                                content_type=content_type,
                                source_url=source_url,
                                note_id=note_id,
                            )
                        if reserved_bytes == 0 and not budget.reserve(len(chunk)):
                            f.close()
                            dest.unlink(missing_ok=True)
                            return DownloadResult(
                                url=url,
                                status="skip_budget",
                                error="max_total_bytes exceeded",
                                content_type=content_type,
                                source_url=source_url,
                                note_id=note_id,
                            )
                        f.write(chunk)

                if bytes_written < min_file_bytes:
                    dest.unlink(missing_ok=True)
                    if reserved_bytes > 0:
                        budget.release(reserved_bytes)
                    else:
                        budget.release(bytes_written)
                    return DownloadResult(
                        url=url,
                        status="skip_too_small",
                        bytes_written=bytes_written,
                        content_type=content_type,
                        source_url=source_url,
                        note_id=note_id,
                    )

                if reserved_bytes > bytes_written:
                    budget.release(reserved_bytes - bytes_written)
                elif reserved_bytes == 0:
                    budget.add_used(bytes_written)

                sidecar = {
                    "url": url,
                    "source_url": source_url,
                    "note_id": note_id,
                    "saved_at": _now_iso(),
                    "content_type": content_type,
                    "bytes": bytes_written,
                    "path": str(dest),
                }
                Path(str(dest) + ".meta.json").write_text(
                    json.dumps(sidecar, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return DownloadResult(
                    url=url,
                    status="downloaded",
                    bytes_written=bytes_written,
                    path=str(dest),
                    content_type=content_type,
                    source_url=source_url,
                    note_id=note_id,
                )
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
    return DownloadResult(
        url=url,
        status="error",
        error=last_error or "unknown error",
        source_url=source_url,
        note_id=note_id,
    )


def _iter_tasks(raw_dir: Path, allow_all: bool) -> Iterable[tuple[str, str, str]]:
    seen: set[str] = set()
    for html_path in sorted(raw_dir.glob("web_*.html")):
        note_id = html_path.stem
        meta_path = raw_dir / f"{note_id}.json"
        source_url = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                source_url = str(meta.get("url") or "").strip()
            except Exception:
                source_url = ""
        if not source_url:
            continue
        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for link in _extract_links_from_html(html, base_url=source_url):
            if link in seen:
                continue
            if not _candidate_from_url(link, allow_all=allow_all):
                continue
            seen.add(link)
            yield (link, source_url, note_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download linked assets from raw HTML archive")
    parser.add_argument("--raw-dir", default="/Volumes/T9/khub/raw_web")
    parser.add_argument("--out-dir", default="/Volumes/T9/khub/assets")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--min-file-kb", type=int, default=4)
    parser.add_argument("--max-file-gb", type=int, default=4)
    parser.add_argument("--max-total-gb", type=int, default=300)
    parser.add_argument("--allow-all-links", action="store_true", help="download non-extension links too (higher volume, less precision)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.strip()
    if not manifest:
        manifest = str(
            (Path("/Volumes/T9/khub/runs") / f"asset_download_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").resolve()
        )
    manifest_path = Path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    max_total_bytes = max(0, int(args.max_total_gb) * 1024 * 1024 * 1024)
    max_file_bytes = max(0, int(args.max_file_gb) * 1024 * 1024 * 1024)
    min_file_bytes = max(0, int(args.min_file_kb) * 1024)

    tasks = list(_iter_tasks(raw_dir, allow_all=args.allow_all_links))
    print(f"[asset] candidates={len(tasks)} raw_dir={raw_dir}")
    if args.dry_run:
        preview = [{"url": u, "source_url": s, "note_id": n} for u, s, n in tasks[:50]]
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": _now_iso(),
                    "mode": "dry_run",
                    "candidate_count": len(tasks),
                    "preview": preview,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[asset] dry-run manifest={manifest_path}")
        return 0

    budget = Budget(max_total_bytes=max_total_bytes)
    results: list[DownloadResult] = []
    status_count: dict[str, int] = {}
    lock = threading.Lock()

    def _collect(result: DownloadResult) -> None:
        with lock:
            results.append(result)
            status_count[result.status] = status_count.get(result.status, 0) + 1

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                _download_one,
                task,
                out_dir,
                args.timeout,
                max_file_bytes,
                min_file_bytes,
                budget,
                args.retries,
            ): task
            for task in tasks
        }
        completed = 0
        total = len(future_map)
        for future in as_completed(future_map):
            result = future.result()
            _collect(result)
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"[asset] progress={completed}/{total} downloaded={status_count.get('downloaded', 0)} used={budget.used_bytes}")

    total_downloaded = sum(item.bytes_written for item in results if item.status in ("downloaded", "exists"))
    summary = {
        "created_at": _now_iso(),
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "candidate_count": len(tasks),
        "status_count": status_count,
        "total_downloaded_bytes": total_downloaded,
        "budget_used_bytes": budget.used_bytes,
        "results": [item.__dict__ for item in results],
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[asset] manifest={manifest_path}")
    print(f"[asset] status_count={status_count}")
    print(f"[asset] total_downloaded_bytes={total_downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
