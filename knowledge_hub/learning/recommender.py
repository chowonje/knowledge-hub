"""Next-branch recommendation engine for Learning Coach MVP v2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from uuid import uuid4

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.learning.models import NEXT_SCHEMA
from knowledge_hub.learning.policy import evaluate_policy_for_payload


def _read_dynamic_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        try:
            parsed = json.loads(row)
            if isinstance(parsed, dict):
                rows.append(parsed)
        except Exception:
            continue
    return rows


def build_load_signal(dynamic_dir: Path) -> dict:
    expenses = _read_dynamic_jsonl(dynamic_dir / "expense.jsonl")
    sleeps = _read_dynamic_jsonl(dynamic_dir / "sleep.jsonl")

    sleep_hours: list[float] = []
    for item in sleeps:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        duration_minutes = metadata.get("durationMinutes") or item.get("durationMinutes")
        sleep_hour = metadata.get("sleepHours") or item.get("sleepHours")
        if isinstance(duration_minutes, (int, float)):
            sleep_hours.append(float(duration_minutes) / 60.0)
        elif isinstance(sleep_hour, (int, float)):
            sleep_hours.append(float(sleep_hour))

    expense_values: list[float] = []
    for item in expenses:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        amount = metadata.get("amount") or item.get("amount") or metadata.get("value") or item.get("value")
        if isinstance(amount, (int, float)):
            expense_values.append(float(amount))

    avg_sleep = mean(sleep_hours) if sleep_hours else None
    avg_expense = mean(expense_values) if expense_values else None

    level = "normal"
    reasons: list[str] = []
    if avg_sleep is not None and avg_sleep < 6.0:
        level = "high"
        reasons.append("low_sleep")
    if avg_expense is not None and avg_expense > 200000:
        level = "high"
        reasons.append("high_expense")

    return {
        "level": level,
        "avgSleepHours": round(avg_sleep, 3) if avg_sleep is not None else None,
        "avgExpense": round(avg_expense, 3) if avg_expense is not None else None,
        "reasons": reasons,
    }


def recommend_next(
    db: SQLiteDatabase,
    topic: str,
    session_id: str,
    map_result: dict,
    dynamic_dir: Path,
    allow_external: bool = False,
    run_id: str | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    run_id = str(run_id or f"learn_next_{uuid4().hex[:12]}")

    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=[topic],
        mode="external-allowed" if allow_external else "local-only",
    )
    if not policy.allowed:
        db.append_learning_event(
            event_id=f"evt_{uuid4().hex}",
            event_type="learning.policy.blocked",
            logical_step="next",
            session_id=session_id,
            run_id=run_id,
            request_id=run_id,
            source="learning",
            payload={
                "topic": topic,
                "reason": policy.blocked_reason,
                "errors": policy.policy_errors,
            },
            policy_class=policy.classification,
        )
        return {
            "schema": NEXT_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "blocked",
            "policy": policy.to_dict(),
            "session": {"sessionId": session_id},
            "unlockPlan": [],
            "remediationPlan": [],
            "loadSignal": build_load_signal(dynamic_dir),
            "reasoning": policy.policy_errors,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }

    progress = db.get_learning_progress(session_id)
    if not progress:
        load_signal = build_load_signal(dynamic_dir)
        db.append_learning_event(
            event_id=f"evt_{uuid4().hex}",
            event_type="learning.next.generated",
            logical_step="next",
            session_id=session_id,
            run_id=run_id,
            request_id=run_id,
            source="learning",
            payload={
                "gatePassed": False,
                "unlockCount": 0,
                "remediationCount": 0,
                "loadLevel": load_signal.get("level"),
                "error": "progress-missing",
            },
            policy_class=policy.classification,
        )
        return {
            "schema": NEXT_SCHEMA,
            "runId": run_id,
            "topic": topic,
            "status": "error",
            "policy": policy.to_dict(),
            "session": {"sessionId": session_id},
            "unlockPlan": [],
            "remediationPlan": [],
            "loadSignal": load_signal,
            "reasoning": ["learning progress not found; run grade first"],
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
        }

    load_signal = build_load_signal(dynamic_dir)
    high_load = load_signal.get("level") == "high"

    gate_passed = bool(progress.get("gate_passed", 0))
    gate_status = str(progress.get("gate_status", "failed"))
    weaknesses = progress.get("weaknesses_json") if isinstance(progress.get("weaknesses_json"), list) else []

    unlock_plan: list[dict] = []
    remediation_plan: list[dict] = []
    reasoning: list[str] = []

    if gate_passed:
        raw_branches = map_result.get("branches") if isinstance(map_result.get("branches"), list) else []
        unlock_limit = 1 if high_load else 3
        for branch in raw_branches[:unlock_limit]:
            canonical_id = str(branch.get("canonical_id", ""))
            display_name = str(branch.get("display_name", canonical_id))
            unlock_plan.append(
                {
                    "canonical_id": canonical_id,
                    "display_name": display_name,
                    "reason": "passed gate; branch unlocked",
                    "miniMission": f"{display_name} 관련 연결 엣지 3개 작성",
                }
            )
        reasoning.append("gate passed; unlocking next branches")
        if high_load:
            reasoning.append("high load signal detected; unlock limit reduced to 1")
    else:
        for item in weaknesses[:3]:
            src = str(item.get("source", "unknown"))
            tgt = str(item.get("target", "unknown"))
            reason = str(item.get("reason", "insufficient mastery"))
            remediation_plan.append(
                {
                    "focus": [src, tgt],
                    "reason": reason,
                    "task": "핵심 관계 2개를 근거 포인터와 함께 다시 작성",
                }
            )
        if not remediation_plan:
            remediation_plan.append(
                {
                    "focus": ["core-trunk"],
                    "reason": gate_status,
                    "task": "트렁크 핵심 개념 3개를 다시 연결해 최소 엣지 수를 충족",
                }
            )
        reasoning.append("gate not passed; remediation plan generated")

    db.append_learning_event(
        event_id=f"evt_{uuid4().hex}",
        event_type="learning.next.generated",
        logical_step="next",
        session_id=session_id,
        run_id=run_id,
        request_id=run_id,
        source="learning",
        payload={
            "gatePassed": gate_passed,
            "unlockCount": len(unlock_plan),
            "remediationCount": len(remediation_plan),
            "loadLevel": load_signal.get("level"),
        },
        policy_class=policy.classification,
    )

    status = "ok" if unlock_plan or remediation_plan else "error"
    return {
        "schema": NEXT_SCHEMA,
        "runId": run_id,
        "topic": topic,
        "status": status,
        "policy": policy.to_dict(),
        "session": {"sessionId": session_id},
        "unlockPlan": unlock_plan,
        "remediationPlan": remediation_plan,
        "loadSignal": load_signal,
        "reasoning": reasoning,
        "createdAt": now.isoformat(),
        "updatedAt": now.isoformat(),
    }
