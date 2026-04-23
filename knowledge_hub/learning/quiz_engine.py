"""Quiz generation and grading helpers."""

from __future__ import annotations

import re
from typing import Any


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _coerce_target_trunks(
    target_trunk_ids: list[str],
    target_trunks: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    raw_items = target_trunks if isinstance(target_trunks, list) and target_trunks else [
        {"canonical_id": str(item), "display_name": str(item)}
        for item in target_trunk_ids
    ]
    items: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        canonical_id = str(
            raw_item.get("canonical_id")
            or raw_item.get("canonicalId")
            or raw_item.get("concept_id")
            or raw_item.get("conceptId")
            or ""
        ).strip()
        display_name = str(
            raw_item.get("display_name")
            or raw_item.get("displayName")
            or canonical_id
        ).strip()
        if not canonical_id or not display_name:
            continue
        label_key = _normalize_label(display_name) or canonical_id
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        items.append({"canonical_id": canonical_id, "display_name": display_name})
    return items


def _build_mixed_question_set(
    topic: str,
    target_trunk_ids: list[str],
    target_trunks: list[dict[str, Any]] | None,
    missing_trunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved_targets = _coerce_target_trunks(target_trunk_ids, target_trunks)
    questions: list[dict[str, Any]] = []
    for idx, trunk in enumerate(resolved_targets[:2], start=1):
        trunk_id = trunk["canonical_id"]
        label = trunk["display_name"]
        questions.append(
            {
                "id": f"q{idx}",
                "type": "mcq",
                "prompt": f"[{topic}] 개념 `{label}`와 가장 관련이 깊은 설명을 고르세요.",
                "choices": [
                    f"{label}는 핵심 개념 맥락을 연결한다.",
                    f"{label}는 무관한 랜덤 표식이다.",
                    f"{label}는 항상 관계가 없다.",
                    f"{label}는 삭제 대상이다.",
                ],
                "answer": "A",
                "targetConceptId": trunk_id,
                "difficulty": "easy",
            }
        )
    if resolved_targets:
        first_trunk = resolved_targets[0]
        questions.append(
            {
                "id": f"q{len(questions)+1}",
                "type": "short",
                "prompt": f"`{first_trunk['display_name']}`와 연결된 관계 1개를 relation_norm 형태로 쓰세요.",
                "answer": "related_to",
                "targetConceptId": first_trunk["canonical_id"],
                "difficulty": "medium",
            }
        )
    if missing_trunks:
        focus = missing_trunks[0]
        focus_id = str(focus.get("canonical_id") or focus.get("canonicalId") or "").strip() or "core-concept"
        focus_label = str(focus.get("display_name") or focus.get("displayName") or focus_id).strip() or focus_id
    elif resolved_targets:
        focus_id = resolved_targets[0]["canonical_id"]
        focus_label = resolved_targets[0]["display_name"]
    else:
        focus_id = "core-concept"
        focus_label = "core-concept"
    questions.append(
        {
            "id": f"q{len(questions)+1}",
            "type": "essay",
            "prompt": f"`{focus_label}`가 빠졌을 때 학습 그래프에 생기는 공백을 3~5문장으로 설명하세요.",
            "answer_keywords": [focus_label, "관계", "근거"],
            "targetConceptId": focus_id,
            "difficulty": "hard",
        }
    )
    return questions


def generate_quiz(
    topic: str,
    *,
    target_trunk_ids: list[str],
    target_trunks: list[dict[str, Any]] | None = None,
    missing_trunks: list[dict[str, Any]],
    mix: str = "mixed",
    question_count: int = 6,
) -> dict[str, Any]:
    if mix not in {"mixed", "mcq", "essay"}:
        mix = "mixed"

    questions = _build_mixed_question_set(topic, target_trunk_ids, target_trunks, missing_trunks)
    if mix == "mcq":
        questions = [item for item in questions if item.get("type") == "mcq"]
    elif mix == "essay":
        questions = [item for item in questions if item.get("type") == "essay"]
    questions = questions[: max(1, question_count)]

    return {
        "mix": mix,
        "questionCount": len(questions),
        "questions": questions,
    }


def grade_quiz(
    quiz_payload: dict[str, Any],
    answers: list[dict[str, Any]],
) -> dict[str, Any]:
    quiz = quiz_payload.get("quiz") if isinstance(quiz_payload.get("quiz"), dict) else {}
    questions = quiz.get("questions") if isinstance(quiz.get("questions"), list) else []
    by_id = {str(item.get("id")): item for item in questions}
    submitted = {str(item.get("id")): str(item.get("answer", "")).strip() for item in answers if isinstance(item, dict)}

    details: list[dict[str, Any]] = []
    correct = 0
    essay_count = 0

    for qid, question in by_id.items():
        answer = submitted.get(qid, "")
        qtype = str(question.get("type", "mcq"))
        is_correct = False
        feedback = ""

        if qtype == "mcq":
            expected = str(question.get("answer", "")).strip().upper()
            is_correct = answer.upper() == expected and bool(expected)
            feedback = "정답입니다." if is_correct else f"정답은 {expected}입니다."
        elif qtype == "short":
            expected = str(question.get("answer", "")).strip().lower()
            is_correct = answer.lower() == expected and bool(expected)
            feedback = "정답입니다." if is_correct else f"핵심 relation은 `{expected}`입니다."
        else:
            essay_count += 1
            keywords = [str(item).lower() for item in question.get("answer_keywords", []) if str(item).strip()]
            matched = sum(1 for kw in keywords if kw in answer.lower())
            ratio = matched / max(1, len(keywords))
            is_correct = ratio >= 0.5 and len(answer.split()) >= 10
            feedback = f"핵심 키워드 매칭: {matched}/{len(keywords)}"

        if is_correct:
            correct += 1
        details.append(
            {
                "id": qid,
                "type": qtype,
                "isCorrect": is_correct,
                "feedback": feedback,
                "targetConceptId": question.get("targetConceptId"),
            }
        )

    total = len(by_id)
    score = round((correct / max(1, total)) * 100, 2)
    return {
        "score": score,
        "total": total,
        "correct": correct,
        "passed": score >= 70,
        "essayCount": essay_count,
        "details": details,
        "feedback": [
            "약한 개념 연결을 우선 보완하세요." if score < 70 else "핵심 개념 연결이 안정적입니다.",
            "서술형 답안은 근거 포인터(path/heading/block_id)를 포함하면 더 좋습니다.",
        ],
    }
