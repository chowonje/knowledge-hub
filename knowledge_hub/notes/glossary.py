"""Glossary helpers for Korean technical note generation."""

from __future__ import annotations

from pathlib import Path

import yaml


FALLBACK_GLOSSARY: dict[str, str] = {
    "Transformer": "Transformer",
    "Diffusion": "Diffusion",
    "Retrieval-Augmented Generation": "Retrieval-Augmented Generation",
    "Adam": "Adam",
    "Batch Normalization": "Batch Normalization",
}


def load_glossary(path: str | None = None) -> dict[str, str]:
    glossary = dict(FALLBACK_GLOSSARY)
    if not path:
        return glossary
    target = Path(path).expanduser()
    if not target.exists():
        return glossary
    try:
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return glossary
    if isinstance(loaded, dict):
        for key, value in loaded.items():
            token = str(key or "").strip()
            if not token:
                continue
            glossary[token] = str(value or token).strip() or token
    return glossary


def protect_terms(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    protected = str(text or "")
    replacements: dict[str, str] = {}
    for index, term in enumerate(sorted(glossary.keys(), key=len, reverse=True)):
        if not term:
            continue
        placeholder = f"__KHUB_TERM_{index}__"
        if term not in protected:
            continue
        protected = protected.replace(term, placeholder)
        replacements[placeholder] = glossary.get(term, term)
    return protected, replacements


def restore_terms(text: str, replacements: dict[str, str]) -> str:
    restored = str(text or "")
    for placeholder, value in replacements.items():
        restored = restored.replace(placeholder, value)
    return restored
