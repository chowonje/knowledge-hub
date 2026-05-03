#!/usr/bin/env python3
"""Local STT helper for Learning OS media ingest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe local media using local Whisper model.")
    parser.add_argument("--media-path", required=True, help="Input media file path")
    parser.add_argument("--media-type", required=True, choices=["audio", "video"], help="Media type")
    parser.add_argument("--output-path", required=True, help="Transcript output path (.txt)")
    parser.add_argument("--model", default="tiny", help="Whisper model name (tiny/base/small/...)")
    parser.add_argument("--language", default=None, help="Optional language code")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    media_path = Path(args.media_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not media_path.exists() or not media_path.is_file():
        emit({
            "ok": False,
            "error": f"media file not found: {media_path}",
        })
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import whisper  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        emit({
            "ok": False,
            "error": (
                "openai-whisper unavailable; install with "
                "'pip install openai-whisper' and ensure ffmpeg is installed"
            ),
            "detail": str(exc),
        })
        return 0

    try:
        model = whisper.load_model(args.model)
        transcribe_options: dict[str, object] = {}
        if isinstance(args.language, str) and args.language.strip():
            transcribe_options["language"] = args.language.strip()
        result = model.transcribe(str(media_path), **transcribe_options)
        text = str(result.get("text", "")).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        emit({
            "ok": False,
            "error": f"whisper transcription failed: {exc}",
        })
        return 0

    if not text:
        emit({
            "ok": False,
            "error": "automatic transcription produced empty transcript",
        })
        return 0

    output_path.write_text(text + "\n", encoding="utf-8")

    emit({
        "ok": True,
        "engine": "openai-whisper",
        "model": args.model,
        "media_type": args.media_type,
        "output_path": str(output_path),
        "chars": len(text),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
