from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from knowledge_hub.web.crawl4ai_adapter import CrawlDocument

_YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}
_TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+(?P<end>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class YouTubeExtractionResult:
    document: CrawlDocument
    warnings: list[str]


def is_youtube_url(url: str) -> bool:
    try:
        host = (urlparse(str(url or "").strip()).netloc or "").lower()
    except Exception:
        return False
    return host in _YOUTUBE_DOMAINS


def youtube_video_id_from_url(url: str) -> str:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host == "youtu.be":
        return parsed.path.strip("/").split("/", 1)[0]
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return str(parse_qs(parsed.query).get("v", [""])[0]).strip()
        if parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed.path.strip("/").split("/", 1)[1].split("/", 1)[0]
    return ""


def youtube_watch_url(url: str) -> str:
    video_id = youtube_video_id_from_url(url)
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else str(url or "").strip()


def extract_youtube_document(
    url: str,
    *,
    timeout: int = 15,
    transcript_language: str | None = None,
    asr_model: str = "tiny",
) -> YouTubeExtractionResult:
    requested_url = str(url or "").strip()
    warnings: list[str] = []
    video_id = youtube_video_id_from_url(requested_url)
    canonical_url = youtube_watch_url(requested_url)
    fetched_at = datetime.now(timezone.utc).isoformat()

    if not video_id:
        return YouTubeExtractionResult(
            document=CrawlDocument(
                url=requested_url,
                title=requested_url,
                content="",
                markdown="",
                raw_html="",
                source_metadata={"media_platform": "youtube", "media_type": "video"},
                fetched_at=fetched_at,
                engine="youtube",
                ok=False,
                error="youtube_ingest_blocked_no_text",
            ),
            warnings=["youtube_ingest_blocked_no_text"],
        )

    metadata = _load_video_metadata(canonical_url, timeout=timeout)
    if not metadata:
        warnings.append("youtube_metadata_partial")
        metadata = {}

    title = str(metadata.get("title") or canonical_url).strip()
    channel_name = str(
        metadata.get("channel")
        or metadata.get("uploader")
        or metadata.get("channel_name")
        or ""
    ).strip()
    description = str(metadata.get("description") or "").strip()
    chapters = _normalize_chapters(metadata.get("chapters"))
    published_at = _published_at_from_metadata(metadata)
    language = _preferred_language(transcript_language, metadata)
    tags = _normalize_tags(metadata.get("tags"))
    duration_sec = _safe_int(metadata.get("duration"))
    thumbnail_url = str(metadata.get("thumbnail") or "").strip()
    channel_id = str(metadata.get("channel_id") or "").strip()

    transcript_text = ""
    transcript_segments: list[dict[str, Any]] = []
    transcript_source = ""

    transcript_segments, detected_language, caption_warnings = _load_caption_segments(
        canonical_url,
        video_id=video_id,
        transcript_language=transcript_language,
        timeout=timeout,
    )
    warnings.extend(caption_warnings)
    if transcript_segments:
        transcript_text = "\n".join(str(item.get("text") or "").strip() for item in transcript_segments).strip()
        transcript_source = "caption"
        if description or chapters:
            transcript_source = "caption_plus_metadata"
        if detected_language:
            language = detected_language

    if not transcript_text:
        asr_text, asr_warnings = _run_local_asr(
            canonical_url,
            video_id=video_id,
            transcript_language=transcript_language,
            asr_model=asr_model,
            timeout=timeout,
        )
        warnings.extend(asr_warnings)
        if asr_text:
            transcript_text = asr_text
            transcript_source = "asr"

    if not transcript_text and not description and not chapters:
        warnings.append("youtube_ingest_blocked_no_text")

    content = _build_linear_content(
        title=title,
        canonical_url=canonical_url,
        channel_name=channel_name,
        description=description,
        chapters=chapters,
        transcript_segments=transcript_segments,
        transcript_text=transcript_text,
    )
    if not content.strip():
        content = transcript_text.strip()

    if not description and not chapters:
        warnings.append("youtube_metadata_partial")

    warning_list = list(dict.fromkeys([item for item in warnings if str(item).strip()]))
    source_metadata = {
        "source_name": "YouTube",
        "source_type": "web",
        "source_vendor": "youtube",
        "source_channel": _youtube_source_channel(channel_name, channel_id, video_id),
        "source_channel_type": "youtube_video",
        "source_item_id": video_id,
        "media_platform": "youtube",
        "media_type": "video",
        "video_id": video_id,
        "channel_name": channel_name,
        "channel_id": channel_id,
        "duration_sec": duration_sec,
        "language": language,
        "transcript_source": transcript_source,
        "transcript_segments": transcript_segments,
        "chapters": chapters,
        "thumbnail_url": thumbnail_url,
        "warnings": warning_list,
    }

    return YouTubeExtractionResult(
        document=CrawlDocument(
            url=canonical_url,
            title=title,
            content=content,
            markdown=content,
            raw_html=json.dumps(
                {
                    "video_id": video_id,
                    "metadata": metadata,
                    "chapters": chapters,
                    "transcript_segments": transcript_segments,
                    "transcript_source": transcript_source,
                },
                ensure_ascii=False,
                indent=2,
            ),
            description=description,
            author=channel_name,
            published_at=published_at,
            tags=tags,
            source_metadata=source_metadata,
            fetched_at=fetched_at,
            engine="youtube",
            ok=bool(content.strip()),
            error="" if content.strip() else "youtube_ingest_blocked_no_text",
        ),
        warnings=warning_list,
    )


def _youtube_source_channel(channel_name: str, channel_id: str, video_id: str) -> str:
    if channel_id:
        return f"youtube_{_slug(channel_id)}"
    if channel_name:
        return f"youtube_{_slug(channel_name)}"
    if video_id:
        return f"youtube_video_{video_id}"
    return "youtube_video"


def _slug(value: str) -> str:
    lowered = str(value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9가-힣]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "unknown"


def _load_video_metadata(url: str, *, timeout: int) -> dict[str, Any]:
    yt_dlp = _resolve_binary("yt-dlp")
    if yt_dlp is None:
        return {}
    command = [
        yt_dlp,
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--no-progress",
        url,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(5, int(timeout)),
        check=False,
    )
    if completed.returncode != 0:
        return {}
    stdout = str(completed.stdout or "").strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except Exception:
        return {}


def _load_caption_segments(
    url: str,
    *,
    video_id: str,
    transcript_language: str | None,
    timeout: int,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    yt_dlp = _resolve_binary("yt-dlp")
    if yt_dlp is None:
        return [], "", ["youtube_caption_unavailable"]
    with tempfile.TemporaryDirectory(prefix="khub_youtube_caption_") as tmp_dir:
        command = [
            yt_dlp,
            "--skip-download",
            "--no-warnings",
            "--no-progress",
            "--write-sub",
            "--write-auto-sub",
            "--convert-subs",
            "vtt",
            "--sub-langs",
            _subtitle_languages(transcript_language),
            "-o",
            f"{video_id}.%(ext)s",
            url,
        ]
        completed = subprocess.run(
            command,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout)),
            check=False,
        )
        files = sorted(Path(tmp_dir).glob(f"{video_id}*.vtt"))
        if not files:
            return [], "", ["youtube_caption_unavailable"]
        best = _pick_caption_file(files, transcript_language=transcript_language)
        segments = _parse_vtt_file(best)
        language = _language_from_caption_file(best, transcript_language=transcript_language)
        if not segments:
            return [], language, ["youtube_caption_unavailable"]
        warnings: list[str] = []
        if completed.returncode != 0:
            warnings.append("youtube_caption_partial")
        return segments, language, warnings


def _subtitle_languages(transcript_language: str | None) -> str:
    requested = str(transcript_language or "").strip()
    if requested:
        return f"{requested}.*, {requested}, en.*, en, ko.*, ko"
    return "ko.*,ko,en.*,en"


def _pick_caption_file(files: list[Path], *, transcript_language: str | None) -> Path:
    requested = str(transcript_language or "").strip().lower()
    ordered: list[tuple[int, Path]] = []
    for path in files:
        name = path.name.lower()
        score = 50
        if requested and f".{requested}." in name:
            score = 0
        elif ".ko." in name:
            score = 10
        elif ".en." in name:
            score = 20
        ordered.append((score, path))
    ordered.sort(key=lambda item: (item[0], item[1].name))
    return ordered[0][1]


def _language_from_caption_file(path: Path, *, transcript_language: str | None) -> str:
    requested = str(transcript_language or "").strip()
    if requested:
        return requested
    parts = path.name.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return ""


def _parse_vtt_file(path: Path) -> list[dict[str, Any]]:
    return _parse_vtt_text(path.read_text(encoding="utf-8"))


def _parse_vtt_text(raw: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start_sec: float | None = None
    end_sec: float | None = None
    text_lines: list[str] = []

    def flush() -> None:
        nonlocal start_sec, end_sec, text_lines
        text = _clean_caption_text(" ".join(text_lines))
        if start_sec is not None and end_sec is not None and text:
            segments.append(
                {
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "text": text,
                }
            )
        start_sec = None
        end_sec = None
        text_lines = []

    for raw_line in raw.splitlines():
        line = str(raw_line or "").strip("\ufeff").strip()
        if not line:
            flush()
            continue
        if line.upper().startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE"):
            continue
        match = _TIMESTAMP_RE.search(line)
        if match:
            flush()
            start_sec = _timestamp_to_seconds(match.group("start"))
            end_sec = _timestamp_to_seconds(match.group("end"))
            continue
        if start_sec is None:
            continue
        text_lines.append(line)
    flush()
    return segments


def _timestamp_to_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_caption_text(value: str) -> str:
    cleaned = _TAG_RE.sub(" ", str(value or ""))
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_youtube_segments_for_indexing(
    transcript_segments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in transcript_segments or []:
        if not isinstance(item, dict):
            continue
        text = _clean_caption_text(str(item.get("text") or ""))
        if not text:
            continue
        start_sec = _safe_float(item.get("start_sec"))
        end_sec = _safe_float(item.get("end_sec"))
        if start_sec is None:
            continue
        if end_sec is None or end_sec < start_sec:
            end_sec = start_sec

        current = {
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "text": text,
        }
        if not normalized:
            normalized.append(current)
            continue

        previous = normalized[-1]
        previous_text = str(previous.get("text") or "")
        if text == previous_text:
            previous["end_sec"] = max(float(previous.get("end_sec") or 0.0), current["end_sec"])
            continue

        longer, shorter = (text, previous_text) if len(text) >= len(previous_text) else (previous_text, text)
        if shorter and (
            longer.startswith(shorter)
            or longer.endswith(shorter)
            or shorter in longer
        ):
            previous_start = _safe_float(previous.get("start_sec"))
            previous_end = _safe_float(previous.get("end_sec"))
            previous["text"] = longer
            previous["start_sec"] = min(
                previous_start if previous_start is not None else current["start_sec"],
                current["start_sec"],
            )
            previous["end_sec"] = max(
                previous_end if previous_end is not None else current["end_sec"],
                current["end_sec"],
            )
            continue

        normalized.append(current)

    return normalized


def _normalize_chapters(raw: Any) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return chapters
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        start_sec = _safe_float(item.get("start_time"))
        end_sec = _safe_float(item.get("end_time"))
        if not title and start_sec is None and end_sec is None:
            continue
        chapters.append(
            {
                "title": title,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return chapters


def _normalize_tags(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _preferred_language(transcript_language: str | None, metadata: dict[str, Any]) -> str:
    requested = str(transcript_language or "").strip()
    if requested:
        return requested
    return str(metadata.get("language") or metadata.get("language_preference") or "").strip()


def _published_at_from_metadata(metadata: dict[str, Any]) -> str:
    raw = metadata.get("release_timestamp") or metadata.get("timestamp")
    if raw:
        try:
            return datetime.fromtimestamp(int(raw), tz=timezone.utc).isoformat()
        except Exception:
            pass
    upload_date = str(metadata.get("upload_date") or "").strip()
    if len(upload_date) == 8 and upload_date.isdigit():
        try:
            return datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            return ""
    return ""


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _format_seconds(value: float | int | None) -> str:
    if value is None:
        return ""
    total = int(max(0, float(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_linear_content(
    *,
    title: str,
    canonical_url: str,
    channel_name: str,
    description: str,
    chapters: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    transcript_text: str,
) -> str:
    parts = [f"# {title}"]
    meta_lines = [f"URL: {canonical_url}"]
    if channel_name:
        meta_lines.append(f"Channel: {channel_name}")
    if meta_lines:
        parts.append("\n".join(meta_lines))
    if description:
        parts.append("## Description\n" + description.strip())
    if chapters:
        chapter_lines = []
        for item in chapters:
            stamp = _format_seconds(item.get("start_sec"))
            label = str(item.get("title") or "").strip()
            chapter_lines.append(f"- {stamp} {label}".strip())
        parts.append("## Chapters\n" + "\n".join(chapter_lines))
    if transcript_segments:
        transcript_lines = []
        for item in transcript_segments:
            stamp = _format_seconds(item.get("start_sec"))
            text = str(item.get("text") or "").strip()
            if text:
                transcript_lines.append(f"[{stamp}] {text}")
        if transcript_lines:
            parts.append("## Transcript\n" + "\n".join(transcript_lines))
    elif transcript_text:
        parts.append("## Transcript\n" + transcript_text.strip())
    return "\n\n".join(part for part in parts if str(part).strip()).strip()


def _run_local_asr(
    url: str,
    *,
    video_id: str,
    transcript_language: str | None,
    asr_model: str,
    timeout: int,
) -> tuple[str, list[str]]:
    yt_dlp = _resolve_binary("yt-dlp")
    ffmpeg = _resolve_binary("ffmpeg")
    if yt_dlp is None or ffmpeg is None:
        return "", ["youtube_asr_dependency_missing"]
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "transcribe_media.py"
    if not script_path.exists():
        return "", ["youtube_asr_dependency_missing"]

    with tempfile.TemporaryDirectory(prefix="khub_youtube_asr_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        audio_path = _download_audio(url, temp_dir=temp_dir, video_id=video_id, timeout=timeout, yt_dlp=yt_dlp)
        if audio_path is None:
            return "", ["youtube_asr_failed"]
        output_path = temp_dir / f"{video_id}.txt"
        command = [
            sys.executable,
            str(script_path),
            "--media-path",
            str(audio_path),
            "--media-type",
            "audio",
            "--output-path",
            str(output_path),
            "--model",
            str(asr_model or "tiny"),
        ]
        language = str(transcript_language or "").strip()
        if language:
            command.extend(["--language", language])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(60, int(timeout) * 8),
            check=False,
        )
        payload = _parse_json_line(completed.stdout)
        if completed.returncode != 0 or not payload.get("ok"):
            return "", ["youtube_asr_failed"]
        try:
            text = output_path.read_text(encoding="utf-8").strip()
        except Exception:
            text = ""
        if not text:
            return "", ["youtube_asr_failed"]
        return text, []


def _download_audio(url: str, *, temp_dir: Path, video_id: str, timeout: int, yt_dlp: str) -> Path | None:
    command = [
        yt_dlp,
        "-x",
        "--audio-format",
        "mp3",
        "--no-warnings",
        "--no-progress",
        "-o",
        f"{video_id}.%(ext)s",
        url,
    ]
    completed = subprocess.run(
        command,
        cwd=str(temp_dir),
        capture_output=True,
        text=True,
        timeout=max(15, int(timeout) * 4),
        check=False,
    )
    if completed.returncode != 0:
        return None
    for ext in ("mp3", "m4a", "webm", "wav"):
        candidate = temp_dir / f"{video_id}.{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(temp_dir.glob(f"{video_id}*"))
    return matches[0] if matches else None


def _parse_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        token = line.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_binary(name: str) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidates = [
        Path(sys.executable).parent / name,
        Path(sys.prefix) / "bin" / name,
        Path(sys.executable).resolve().parent / name,
    ]
    for local in candidates:
        if local.exists():
            return str(local)
    return None
