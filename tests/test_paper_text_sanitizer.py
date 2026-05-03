from __future__ import annotations

from knowledge_hub.papers.text_sanitizer import normalize_paper_texts, sanitize_paper_text


def test_sanitize_paper_text_strips_latex_preamble_and_starts_at_semantic_section():
    raw_text = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\title{Agent Memory}\n"
        "\\author{Test Author}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "Abstract\n"
        "This paper studies long-term agent memory retrieval across multi-session tasks. "
        "It proposes a compact paper-memory representation that keeps salient evidence.\n\n"
        "Introduction\n"
        "The approach improves retrieval quality while reducing repeated context stuffing.\n"
    )

    result = sanitize_paper_text(raw_text)

    assert result.starts_with_latex is True
    assert result.semantic_start_reason == "section_heading"
    assert "\\documentclass" not in result.sanitized_text
    assert result.sanitized_text.startswith("Abstract")
    assert "long-term agent memory retrieval" in result.sanitized_text
    assert "starts_with_latex" in result.warnings


def test_normalize_paper_texts_prefers_nonweak_raw_when_translated_is_empty():
    normalized = normalize_paper_texts(
        translated_text="",
        raw_text=(
            "\\documentclass{article}\n"
            "Introduction\n"
            "This paper describes a retrieval benchmark with sufficient prose to survive sanitation. "
            "It includes detailed evaluation protocol and longitudinal measurements across sessions.\n"
        ),
    )

    assert normalized.preferred_source == "raw"
    assert normalized.raw.starts_with_latex is True
    assert "raw_starts_with_latex" in normalized.warnings
    assert normalized.preferred_text.startswith("Introduction")


def test_sanitize_paper_text_removes_references_tail_and_command_heavy_lines():
    raw_text = (
        "\\documentclass{article}\n"
        "\\title{Agent Memory}\n"
        "Abstract\n"
        "This paper studies agent memory retrieval and provides sufficient prose for extraction.\n"
        "\\section{Method}\n"
        "$E=mc^2$\n"
        "References\n"
        "[1] Prior work\n"
    )

    result = sanitize_paper_text(raw_text)

    assert result.removed_references_tail is True
    assert result.dropped_latex_line_count >= 2
    assert "References" not in result.sanitized_text
    assert "references_tail_removed" in result.warnings
