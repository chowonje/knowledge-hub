from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnswerInitialGenerationDeps:
    build_answer_generation_fallback_fn: Any


@dataclass(frozen=True)
class AnswerInitialGenerationResult:
    initial_answer: str | None = None
    fallback_kind: str | None = None
    fallback_answer: str | None = None
    generation_meta: dict[str, Any] | None = None
    fallback_verification: dict[str, Any] | None = None
    fallback_rewrite: dict[str, Any] | None = None
    generation_warnings: list[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_kind is not None


class AnswerInitialGeneration:
    def __init__(self, deps: AnswerInitialGenerationDeps) -> None:
        self._deps = deps

    def run(
        self,
        *,
        query: str,
        selected_llm: Any,
        answer_prompt: str,
        safe_context: str,
        evidence_packet: Any,
        routing_meta: dict[str, Any],
        stage: str,
        stream: bool = False,
        answer_max_tokens: int | None = None,
    ) -> AnswerInitialGenerationResult:
        if selected_llm is None:
            (
                fallback_answer,
                generation_meta,
                fallback_verification,
                fallback_rewrite,
                generation_warnings,
            ) = self._deps.build_answer_generation_fallback_fn(
                query=query,
                error=RuntimeError("No available LLM route"),
                stage=stage,
                evidence=evidence_packet.evidence,
                citations=evidence_packet.citations,
                routing_meta=routing_meta,
            )
            return AnswerInitialGenerationResult(
                fallback_kind="no_route",
                fallback_answer=fallback_answer,
                generation_meta=generation_meta,
                fallback_verification=fallback_verification,
                fallback_rewrite=fallback_rewrite,
                generation_warnings=list(generation_warnings),
            )

        try:
            if stream:
                # Stream providers do not expose a consistent per-call output cap yet.
                initial_answer = "".join(
                    str(chunk or "") for chunk in selected_llm.stream_generate(answer_prompt, safe_context)
                )
            elif answer_max_tokens is not None:
                initial_answer = selected_llm.generate(answer_prompt, safe_context, max_tokens=answer_max_tokens)
            else:
                initial_answer = selected_llm.generate(answer_prompt, safe_context)
        except Exception as error:
            (
                fallback_answer,
                generation_meta,
                fallback_verification,
                fallback_rewrite,
                generation_warnings,
            ) = self._deps.build_answer_generation_fallback_fn(
                query=query,
                error=error,
                stage=stage,
                evidence=evidence_packet.evidence,
                citations=evidence_packet.citations,
                routing_meta=routing_meta,
            )
            return AnswerInitialGenerationResult(
                fallback_kind="generation_error",
                fallback_answer=fallback_answer,
                generation_meta=generation_meta,
                fallback_verification=fallback_verification,
                fallback_rewrite=fallback_rewrite,
                generation_warnings=list(generation_warnings),
            )

        return AnswerInitialGenerationResult(initial_answer=initial_answer)


__all__ = [
    "AnswerInitialGeneration",
    "AnswerInitialGenerationDeps",
    "AnswerInitialGenerationResult",
]
