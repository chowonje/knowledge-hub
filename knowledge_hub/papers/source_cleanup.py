from __future__ import annotations

import csv
import json
from pathlib import Path


RELINK_TO_CANONICAL: dict[str, dict[str, str]] = {
    "Batch_Normalization_c72acd36": {
        "canonicalPaperId": "1502.03167",
        "reason": "local imported Batch Normalization record points to unrelated ANN textbook content; relink to canonical batch normalization source",
    },
    "localpdf_Attention_is_All_you_Need_AI_Papers_lega_7398ae8b16": {
        "canonicalPaperId": "1706.03762",
        "reason": "legacy localpdf points to GLU Variants content; relink to canonical Attention is All you Need source",
    },
    "Deep_Residual_Learning_efbb7871": {
        "canonicalPaperId": "1512.03385",
        "reason": "local imported Deep Residual Learning record points to unrelated LoFTR content; relink to canonical ResNet source",
    },
    "localpdf_Deep_Residual_Learning_2ad82ba486": {
        "canonicalPaperId": "1512.03385",
        "reason": "localpdf points to unrelated LoFTR content; relink to canonical Deep Residual Learning source",
    },
    "localpdf_Learning_Transferable_Visual_Models_from_ce5210d41b": {
        "canonicalPaperId": "2103.00020",
        "reason": "localpdf points to unrelated facial-expression paper; relink to canonical CLIP source",
    },
    "Sequence_to_Sequence_Learning_0e5054e7": {
        "canonicalPaperId": "1409.3215",
        "reason": "local imported Sequence to Sequence Learning record points to unrelated sentiment-transfer paper; relink to canonical seq2seq source",
    },
}

EXCLUDE_UNTIL_MANUAL_FIX: dict[str, str] = {
    "2503.07891": "Gemini Embedding record points to DINOv3 source; keep out of rebuild until the correct Gemini source is restored",
    "localpdf_R_Graphics_Output_8b745a311c": "R Graphics Output localpdf points to unrelated Deep Sound Change source; keep out of rebuild until a correct source is available",
}

REVIEWED_KEEP_CURRENT_SOURCE: dict[str, str] = {
    "localpdf_AI_Safety_Alignment_and_Ethics_AI_SAE_40b586cf84": "source preview matches title closely enough; keep current source and rely on parser escalation if needed",
    "localpdf_Fine_tuning_with_RAG_for_Improving_LLM_L_3f50546cf0": "source preview matches title; keep current source and rebuild with improved sanitation",
    "localpdf_LLM_Powered_AI_Agent_Systems_and_Their_A_fe831f32fd": "source preview matches title; keep current source and rebuild with improved sanitation",
    "NeRF_1b9d0d11": "source preview matches canonical NeRF paper; keep current source",
    "2601.09668": "source preview matches title but starts inside a later section; keep current source and rely on parser escalation if needed",
    "2005.14165": "source preview matches GPT-3 paper but extraction begins in figure/examples; keep current source and rely on parser escalation if needed",
}


def load_source_cleanup_queue(path_value: str | Path) -> list[dict[str, str]]:
    path = Path(path_value).expanduser()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve_relink(sqlite_db, paper_id: str, title: str, row: dict[str, str]) -> dict[str, str]:
    decision = {
        **row,
        "paperId": paper_id,
        "title": title,
        "action": "relink_to_canonical",
        "status": "blocked_missing_canonical",
        "canonicalPaperId": "",
        "canonicalTitle": "",
        "newPdfPath": "",
        "newTextPath": "",
        "resolutionReason": "",
    }
    rule = RELINK_TO_CANONICAL.get(paper_id) or {}
    canonical_paper_id = str(rule.get("canonicalPaperId") or "").strip()
    if not canonical_paper_id:
        decision["resolutionReason"] = "missing canonical mapping"
        return decision
    canonical = sqlite_db.get_paper(canonical_paper_id)
    if not canonical:
        decision["canonicalPaperId"] = canonical_paper_id
        decision["resolutionReason"] = "canonical paper missing from sqlite store"
        return decision
    decision.update(
        {
            "status": "resolved",
            "canonicalPaperId": canonical_paper_id,
            "canonicalTitle": str(canonical.get("title") or ""),
            "newPdfPath": str(canonical.get("pdf_path") or ""),
            "newTextPath": str(canonical.get("text_path") or ""),
            "resolutionReason": str(rule.get("reason") or ""),
        }
    )
    return decision


def build_source_cleanup_plan(
    queue_rows: list[dict[str, str]],
    *,
    sqlite_db,
) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    for row in queue_rows:
        paper_id = str(row.get("paperId") or "").strip()
        title = str(row.get("title") or "").strip()
        if not paper_id:
            continue
        if paper_id in RELINK_TO_CANONICAL:
            decision = _resolve_relink(sqlite_db, paper_id, title, row)
        elif paper_id in EXCLUDE_UNTIL_MANUAL_FIX:
            decision = {
                **row,
                "paperId": paper_id,
                "title": title,
                "action": "exclude_until_manual_fix",
                "status": "blocked_manual_source_needed",
                "canonicalPaperId": "",
                "canonicalTitle": "",
                "newPdfPath": "",
                "newTextPath": "",
                "resolutionReason": EXCLUDE_UNTIL_MANUAL_FIX[paper_id],
            }
        elif paper_id in REVIEWED_KEEP_CURRENT_SOURCE:
            decision = {
                **row,
                "paperId": paper_id,
                "title": title,
                "action": "keep_current_source",
                "status": "reviewed_no_change",
                "canonicalPaperId": "",
                "canonicalTitle": "",
                "newPdfPath": str(row.get("oldPdfPath") or ""),
                "newTextPath": str(row.get("oldTextPath") or ""),
                "resolutionReason": REVIEWED_KEEP_CURRENT_SOURCE[paper_id],
            }
        else:
            decision = {
                **row,
                "paperId": paper_id,
                "title": title,
                "action": "manual_review_required",
                "status": "blocked_manual_review",
                "canonicalPaperId": "",
                "canonicalTitle": "",
                "newPdfPath": "",
                "newTextPath": "",
                "resolutionReason": "no cleanup rule available",
            }
        decisions.append(decision)
    decisions.sort(key=lambda item: (str(item.get("status") or ""), str(item.get("paperId") or "")))
    return decisions


def apply_source_cleanup_plan(*, sqlite_db, decisions: list[dict[str, str]]) -> dict[str, int]:
    applied = 0
    skipped = 0
    for decision in decisions:
        if str(decision.get("action") or "") != "relink_to_canonical" or str(decision.get("status") or "") != "resolved":
            skipped += 1
            continue
        existing = sqlite_db.get_paper(str(decision.get("paperId") or ""))
        if not existing:
            skipped += 1
            continue
        payload = dict(existing)
        payload["pdf_path"] = str(decision.get("newPdfPath") or "")
        payload["text_path"] = str(decision.get("newTextPath") or "")
        sqlite_db.upsert_paper(payload)
        applied += 1
    return {"applied": applied, "skipped": skipped}


def _clean_pass_b_ids(pass_b_ids: list[str], decisions: list[dict[str, str]]) -> list[str]:
    blocked = {
        str(item.get("paperId") or "")
        for item in decisions
        if str(item.get("action") or "") in {"exclude_until_manual_fix", "manual_review_required"}
    }
    return [paper_id for paper_id in pass_b_ids if paper_id and paper_id not in blocked]


def write_source_cleanup_artifacts(
    *,
    artifact_dir: str | Path,
    decisions: list[dict[str, str]],
    pass_b_ids: list[str] | None = None,
) -> dict[str, str]:
    path = Path(artifact_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / "paper_memory_source_cleanup_resolutions.csv"
    summary_path = path / "paper_memory_source_cleanup_summary.json"
    result: dict[str, str] = {
        "resolutionsCsv": str(csv_path),
        "summary": str(summary_path),
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paperId",
                "title",
                "action",
                "status",
                "oldPdfPath",
                "oldTextPath",
                "newPdfPath",
                "newTextPath",
                "canonicalPaperId",
                "canonicalTitle",
                "resolutionReason",
                "recommendedParser",
            ],
        )
        writer.writeheader()
        for decision in decisions:
            writer.writerow({key: str(decision.get(key) or "") for key in writer.fieldnames or []})
    summary = {
        "schema": "knowledge-hub.paper-memory-source-cleanup.v1",
        "total": len(decisions),
        "statusCounts": {},
        "actionCounts": {},
    }
    for decision in decisions:
        status = str(decision.get("status") or "")
        action = str(decision.get("action") or "")
        summary["statusCounts"][status] = int(summary["statusCounts"].get(status) or 0) + 1
        summary["actionCounts"][action] = int(summary["actionCounts"].get(action) or 0) + 1
    if pass_b_ids is not None:
        clean_ids = _clean_pass_b_ids(pass_b_ids, decisions)
        clean_path = path / "paper_memory_pass_b_clean_ids.txt"
        clean_path.write_text("\n".join(clean_ids) + ("\n" if clean_ids else ""), encoding="utf-8")
        summary["passBInputCount"] = len(pass_b_ids)
        summary["passBCleanCount"] = len(clean_ids)
        summary["passBExcludedCount"] = len(pass_b_ids) - len(clean_ids)
        result["passBCleanIds"] = str(clean_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
