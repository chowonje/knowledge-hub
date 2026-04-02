"""Paper CLI runtime helpers for concept/vault maintenance flows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from rich.table import Table


def run_paper_sync_keywords(
    *,
    khub: Any,
    force: bool,
    limit: int,
    claims: bool,
    allow_external: bool,
    llm_mode: str,
    console: Any,
    resolve_vault_papers_dir_fn: Callable[[str], Path | None],
    sqlite_db_fn: Callable[..., Any],
    extract_note_concepts_fn: Callable[[str], list[str]],
    extract_summary_text_fn: Callable[[str, str, Any], str],
    resolve_routed_llm_fn: Callable[..., tuple[Any, Any, list[str]]],
    extract_keywords_with_evidence_fn: Callable[[Any, str, str, Any], list[dict[str, Any]]],
    concept_id_fn: Callable[[str], str],
    upsert_ai_concept_fn: Callable[..., Any],
    update_note_concepts_fn: Callable[[str, list[str]], str],
    regenerate_concept_index_fn: Callable[[Path, dict[str, list[str]]], None],
    entity_resolver_cls: type[Any],
    estimate_evidence_quality_fn: Callable[[str], float],
    extract_claim_candidates_fn: Callable[..., list[Any]],
    score_claim_with_breakdown_fn: Callable[..., tuple[float, dict[str, Any]]],
) -> None:
    config = khub.config
    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    papers_dir = resolve_vault_papers_dir_fn(vault_path)
    if not papers_dir or not papers_dir.exists():
        console.print("[red]Obsidian 논문 폴더를 찾을 수 없습니다.[/red]")
        console.print("[dim]khub config set obsidian.vault_path 로 vault 경로를 확인하세요.[/dim]")
        return

    sqlite_db = sqlite_db_fn(config, khub=khub)
    md_files = [file for file in sorted(papers_dir.glob("*.md")) if file.name != "00_Concept_Index.md"]

    console.print(f"[bold]Obsidian 논문 노트 {len(md_files)}개 스캔 중...[/bold]\n")

    all_concepts: dict[str, list[str]] = {}
    updated = 0
    skipped = 0
    relations_added = 0
    claim_auto = 0
    claim_pending = 0
    claim_dropped = 0

    for index, md_path in enumerate(md_files, 1):
        content = md_path.read_text(encoding="utf-8")

        arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        existing_concepts = extract_note_concepts_fn(content)
        has_good_concepts = bool(existing_concepts)

        if has_good_concepts and not force:
            for concept in existing_concepts:
                all_concepts.setdefault(concept, []).append(md_path.stem)
            skipped += 1
            continue

        if limit > 0 and updated >= limit:
            break

        title = md_path.stem
        summary_text = extract_summary_text_fn(content, title, sqlite_db)
        if not summary_text or len(summary_text) < 20:
            console.print(f"  [{index}/{len(md_files)}] {title[:50]}... [dim]텍스트 부족, 스킵[/dim]")
            skipped += 1
            continue

        console.print(f"  [{index}/{len(md_files)}] {title[:50]}...", end=" ")

        try:
            keyword_llm, _keyword_decision, _keyword_warnings = resolve_routed_llm_fn(
                config,
                task_type="predicate_reasoning",
                allow_external=allow_external,
                llm_mode=llm_mode,
                query=title,
                context=summary_text[:3500],
                source_count=1,
            )
            if keyword_llm is None:
                console.print("[red]실패: 사용 가능한 키워드 추출 LLM 없음[/red]")
                continue
            evidence_results = extract_keywords_with_evidence_fn(keyword_llm, title, summary_text, sqlite_db)
        except Exception as error:
            console.print(f"[red]실패: {error}[/red]")
            continue

        if not evidence_results:
            console.print("[yellow]키워드 없음[/yellow]")
            continue

        concepts = [item["concept"] for item in evidence_results]
        for concept in concepts:
            all_concepts.setdefault(concept, []).append(title)

        if arxiv_id:
            paper_entity_id = f"paper:{arxiv_id}"
            sqlite_db.upsert_ontology_entity(
                entity_id=paper_entity_id,
                entity_type="paper",
                canonical_name=title,
                properties={"arxiv_id": arxiv_id},
                source="paper_sync_keywords",
            )
            for evidence in evidence_results:
                concept_name = evidence["concept"]
                concept_id = concept_id_fn(concept_name)
                upsert_ai_concept_fn(
                    sqlite_db,
                    entity_id=concept_id,
                    canonical_name=concept_name,
                    source="paper_sync_keywords",
                    title=title,
                    relation_predicates=["uses"],
                )
                sqlite_db.add_relation(
                    source_type="paper",
                    source_id=paper_entity_id,
                    relation="uses",
                    target_type="concept",
                    target_id=concept_id,
                    evidence_text=json.dumps(
                        {
                            "source": "paper_sync_keywords",
                            "relation_norm": "uses",
                            "evidence_ptrs": [
                                {
                                    "type": "note",
                                    "path": str(md_path),
                                    "snippet_hash": hashlib.sha1(
                                        str(evidence.get("evidence", "")).encode("utf-8")
                                    ).hexdigest()[:16],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    confidence=evidence.get("confidence", 0.7),
                )
                relations_added += 1

            if claims:
                resolver = entity_resolver_cls(sqlite_db)
                claim_llm, _claim_decision, claim_warnings = resolve_routed_llm_fn(
                    config,
                    task_type="claim_extraction",
                    allow_external=allow_external,
                    llm_mode=llm_mode,
                    query=title,
                    context=summary_text[:3500],
                    source_count=1,
                )
                if claim_llm is None:
                    claim_warnings.append("claim route unavailable")
                    claim_candidates = []
                else:
                    claim_candidates = extract_claim_candidates_fn(
                        claim_llm,
                        title=title,
                        text=summary_text,
                        max_claims=8,
                    )

                for claim_index, candidate in enumerate(claim_candidates):
                    subject_resolution = resolver.resolve(candidate.subject, entity_type="concept")
                    subject_entity_id = (
                        subject_resolution.canonical_id if subject_resolution else paper_entity_id
                    )
                    subject_conf = float(subject_resolution.resolve_confidence) if subject_resolution else 0.65

                    object_entity_id = None
                    object_literal = None
                    object_conf = 0.6
                    if candidate.object_value:
                        object_resolution = resolver.resolve(candidate.object_value, entity_type="concept")
                        if object_resolution:
                            object_entity_id = object_resolution.canonical_id
                            object_conf = float(object_resolution.resolve_confidence)
                        else:
                            object_literal = candidate.object_value

                    entity_resolve_conf = (subject_conf + object_conf) / 2.0
                    evidence_quality = estimate_evidence_quality_fn(candidate.evidence)
                    claim_score, score_breakdown = score_claim_with_breakdown_fn(
                        candidate.llm_confidence,
                        entity_resolve_conf,
                        evidence_quality,
                        claim_text=candidate.claim_text,
                        evidence=candidate.evidence,
                        subject=candidate.subject,
                        predicate=candidate.predicate,
                        object_value=candidate.object_value,
                    )
                    claim_id = (
                        f"claim:{arxiv_id}:{claim_index}:"
                        f"{hashlib.sha1((candidate.claim_text + candidate.predicate).encode('utf-8')).hexdigest()[:10]}"
                    )
                    evidence_ptrs = [
                        {
                            "type": "note",
                            "path": str(md_path),
                            "snippet_hash": hashlib.sha1(candidate.evidence.encode("utf-8")).hexdigest()[:16],
                            "score_breakdown": score_breakdown,
                        }
                    ]

                    if claim_score >= 0.82:
                        evidence_ptrs[0]["claim_decision"] = "accepted"
                        sqlite_db.upsert_claim(
                            claim_id=claim_id,
                            claim_text=candidate.claim_text,
                            subject_entity_id=subject_entity_id,
                            predicate=candidate.predicate,
                            object_entity_id=object_entity_id,
                            object_literal=object_literal,
                            confidence=claim_score,
                            evidence_ptrs=evidence_ptrs,
                            source="paper_claim_extractor",
                        )
                        claim_auto += 1
                    elif claim_score >= 0.65:
                        evidence_ptrs[0]["claim_decision"] = "pending"
                        sqlite_db.add_ontology_pending(
                            pending_type="claim",
                            run_id=f"paper_claim_{arxiv_id}",
                            topic_slug="paper",
                            note_id=md_path.stem,
                            source_url="",
                            source_entity_id=subject_entity_id,
                            predicate_id=candidate.predicate,
                            target_entity_id=object_entity_id or "",
                            confidence=claim_score,
                            evidence_ptrs=evidence_ptrs,
                            reason={
                                "claim_id": claim_id,
                                "claim_text": candidate.claim_text,
                                "subject_entity_id": subject_entity_id,
                                "predicate": candidate.predicate,
                                "object_entity_id": object_entity_id or "",
                                "object_literal": object_literal or "",
                                "llm_confidence": candidate.llm_confidence,
                                "entity_resolve_conf": entity_resolve_conf,
                                "evidence_quality": evidence_quality,
                                "entity_resolution_confidence": score_breakdown.get(
                                    "entity_resolution_confidence", entity_resolve_conf
                                ),
                                "generic_claim_penalty": score_breakdown.get("generic_claim_penalty", 0.0),
                                "contradiction_hint": score_breakdown.get("contradiction_hint", 0.0),
                                "score_breakdown": score_breakdown,
                            },
                            status="pending",
                        )
                        claim_pending += 1
                    else:
                        claim_dropped += 1

        new_content = update_note_concepts_fn(content, concepts)
        md_path.write_text(new_content, encoding="utf-8")
        updated += 1
        console.print(f"[green]{len(concepts)}개 키워드[/green]")

    console.print(
        f"\n[bold]업데이트: {updated}개 | 스킵(기존): {skipped}개 | 관계: {relations_added}개"
        f" | claims(auto/pending/drop): {claim_auto}/{claim_pending}/{claim_dropped}[/bold]"
    )

    concept_index_path = papers_dir / "00_Concept_Index.md"
    regenerate_concept_index_fn(concept_index_path, all_concepts)
    console.print(f"[bold green]Concept Index 갱신 완료 ({len(all_concepts)}개 개념)[/bold green]")


def run_paper_build_concepts(
    *,
    khub: Any,
    force: bool,
    console: Any,
    resolve_vault_papers_dir_fn: Callable[[str], Path | None],
    resolve_vault_concepts_dir_fn: Callable[[str], Path],
    build_llm_fn: Callable[..., Any],
    sqlite_db_fn: Callable[..., Any],
    extract_note_concepts_fn: Callable[[str], list[str]],
    batch_describe_concepts_fn: Callable[[Any, list[str], list[str]], dict[str, dict[str, Any]]],
    concept_id_fn: Callable[[str], str],
    upsert_ai_concept_fn: Callable[..., Any],
    build_concept_note_fn: Callable[[str, str, list[str], list[str]], str],
    rebuild_concept_index_with_relations_fn: Callable[[Path, Path, dict[str, list[str]]], None],
) -> None:
    config = khub.config
    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다.[/red]")
        return

    papers_dir = resolve_vault_papers_dir_fn(vault_path)
    concepts_dir = resolve_vault_concepts_dir_fn(vault_path)
    concepts_dir.mkdir(parents=True, exist_ok=True)

    llm = build_llm_fn(config, config.summarization_provider, config.summarization_model, khub=khub)
    sqlite_db = sqlite_db_fn(config, khub=khub)

    concept_papers: dict[str, list[str]] = {}
    for md_path in sorted(papers_dir.glob("*.md")):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        for concept in extract_note_concepts_fn(content):
            concept_papers.setdefault(concept, []).append(md_path.stem)

    all_concept_names = sorted(concept_papers.keys())
    console.print(f"[bold]{len(all_concept_names)}개 개념 발견[/bold]")

    if not force:
        existing = {file.stem for file in concepts_dir.glob("*.md")}
        to_process = [concept for concept in all_concept_names if concept not in existing]
    else:
        to_process = list(all_concept_names)

    if not to_process:
        console.print("[green]모든 개념 노트가 이미 생성되어 있습니다. --force로 재생성 가능.[/green]")
        rebuild_concept_index_with_relations_fn(papers_dir, concepts_dir, concept_papers)
        return

    console.print(f"[bold]{len(to_process)}개 개념 노트 생성 시작[/bold]\n")

    batch_size = 15
    created = 0
    relations_stored = 0

    for offset in range(0, len(to_process), batch_size):
        batch = to_process[offset:offset + batch_size]
        console.print(f"  배치 [{offset + 1}~{offset + len(batch)}/{len(to_process)}]...", end=" ")

        try:
            results = batch_describe_concepts_fn(llm, batch, all_concept_names)
        except Exception as error:
            console.print(f"[red]API 오류: {error}[/red]")
            continue

        for concept_name, info in results.items():
            description = info.get("description", "")
            related = info.get("related", [])
            papers = concept_papers.get(concept_name, [])

            concept_id = concept_id_fn(concept_name)
            upsert_ai_concept_fn(
                sqlite_db,
                entity_id=concept_id,
                canonical_name=concept_name,
                source="paper_build_concepts",
                description=description,
                title=" ".join([concept_name, *papers[:3]]),
                related_names=related,
                relation_predicates=["related_to"],
            )

            for related_name in related:
                related_id = concept_id_fn(related_name)
                upsert_ai_concept_fn(
                    sqlite_db,
                    entity_id=related_id,
                    canonical_name=related_name,
                    source="paper_build_concepts",
                    title=concept_name,
                    related_names=[concept_name],
                    relation_predicates=["related_to"],
                )
                sqlite_db.add_relation(
                    source_type="concept",
                    source_id=concept_id,
                    relation="concept_related_to",
                    target_type="concept",
                    target_id=related_id,
                    evidence_text=f"LLM이 {concept_name}의 관련 개념으로 식별",
                    confidence=0.6,
                )
                relations_stored += 1

            note_content = build_concept_note_fn(concept_name, description, related, papers)
            safe_name = re.sub(r'[\\/:*?"<>|]', "", concept_name).strip()
            (concepts_dir / f"{safe_name}.md").write_text(note_content, encoding="utf-8")
            created += 1

        console.print(f"[green]{len(results)}개 생성[/green]")

    rebuild_concept_index_with_relations_fn(papers_dir, concepts_dir, concept_papers)

    console.print(f"\n[bold green]{created}개 개념 노트 생성 완료[/bold green]")
    console.print(f"[dim]concept_related_to 관계: {relations_stored}개 저장[/dim]")
    console.print(f"[dim]위치: {concepts_dir}[/dim]")


def run_paper_normalize_concepts(
    *,
    khub: Any,
    dry_run: bool,
    console: Any,
    resolve_vault_papers_dir_fn: Callable[[str], Path | None],
    resolve_vault_concepts_dir_fn: Callable[[str], Path],
    build_llm_fn: Callable[..., Any],
    sqlite_db_fn: Callable[..., Any],
    detect_synonym_groups_fn: Callable[[Any, list[str]], list[dict[str, Any]]],
    concept_id_fn: Callable[[str], str],
    upsert_ai_concept_fn: Callable[..., Any],
    merge_obsidian_concept_fn: Callable[[Path, Path, str, str], int],
    replace_in_paper_notes_fn: Callable[[Path, str, str], None],
    rebuild_concept_index_with_relations_fn: Callable[[Path, Path, dict[str, list[str]]], None],
) -> None:
    config = khub.config
    vault_path = config.vault_path
    if not vault_path:
        console.print("[red]Obsidian vault 경로가 설정되지 않았습니다.[/red]")
        return

    llm = build_llm_fn(config, config.summarization_provider, config.summarization_model, khub=khub)
    papers_dir = resolve_vault_papers_dir_fn(vault_path)
    concepts_dir = resolve_vault_concepts_dir_fn(vault_path)

    concept_names = sorted({file.stem for file in concepts_dir.glob("*.md")}) if concepts_dir.exists() else []
    for md_path in sorted(papers_dir.glob("*.md")):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        for concept in re.findall(r"\[\[([^\]]+)\]\]", content):
            if concept != "00_Concept_Index" and concept not in concept_names:
                concept_names.append(concept)

    concept_names = sorted(set(concept_names))
    console.print(f"[bold]{len(concept_names)}개 개념 스캔 완료[/bold]\n")

    if len(concept_names) < 2:
        console.print("[green]정규화할 개념이 부족합니다.[/green]")
        return

    console.print("[bold]동의어/복수형/약어 그룹 탐지 중...[/bold]")
    all_groups: list[dict[str, Any]] = []
    batch_size = 80
    for offset in range(0, len(concept_names), batch_size):
        batch = concept_names[offset:offset + batch_size]
        console.print(f"  배치 [{offset + 1}~{offset + len(batch)}/{len(concept_names)}]...", end=" ")
        try:
            groups = detect_synonym_groups_fn(llm, batch)
            all_groups.extend(groups)
            console.print(f"[green]{len(groups)}개 그룹[/green]")
        except Exception as error:
            console.print(f"[red]실패: {error}[/red]")

    if not all_groups:
        console.print("[green]동의어 그룹이 발견되지 않았습니다.[/green]")
        return

    table = Table(title=f"동의어 그룹 ({len(all_groups)}개)")
    table.add_column("정규 이름", style="cyan")
    table.add_column("별칭 (병합 대상)", style="yellow")
    for group in all_groups:
        table.add_row(group["canonical"], ", ".join(group["aliases"]))
    console.print(table)

    if dry_run:
        console.print("\n[dim]--dry-run: 변경 없이 종료[/dim]")
        return

    sqlite_db = sqlite_db_fn(config, khub=khub)
    registered = 0
    for concept_name in concept_names:
        concept_id = concept_id_fn(concept_name)
        upsert_ai_concept_fn(
            sqlite_db,
            entity_id=concept_id,
            canonical_name=concept_name,
            source="paper_normalize_concepts",
        )

    for group in all_groups:
        canonical = group["canonical"]
        canonical_id = concept_id_fn(canonical)
        upsert_ai_concept_fn(
            sqlite_db,
            entity_id=canonical_id,
            canonical_name=canonical,
            source="paper_normalize_concepts",
            related_names=group["aliases"],
        )

        for alias in group["aliases"]:
            sqlite_db.add_entity_alias(alias, canonical_id)
            alias_id = concept_id_fn(alias)
            existing = sqlite_db.get_ontology_entity(alias_id)
            if existing and existing["canonical_name"] != canonical:
                sqlite_db.delete_ontology_entity(alias_id)
        registered += 1

    console.print(f"\n[green]{registered}개 정규화 그룹 DB 등록[/green]")

    merged = 0
    for group in all_groups:
        canonical = group["canonical"]
        for alias in group["aliases"]:
            merged += merge_obsidian_concept_fn(papers_dir, concepts_dir, alias, canonical)
            replace_in_paper_notes_fn(papers_dir, alias, canonical)

    console.print(f"[green]Obsidian 노트 {merged}개 병합 완료[/green]")

    concept_papers: dict[str, list[str]] = {}
    for md_path in sorted(papers_dir.glob("*.md")):
        if md_path.name == "00_Concept_Index.md":
            continue
        content = md_path.read_text(encoding="utf-8")
        for concept in re.findall(r"\[\[([^\]]+)\]\]", content):
            if concept != "00_Concept_Index":
                concept_papers.setdefault(concept, []).append(md_path.stem)

    rebuild_concept_index_with_relations_fn(papers_dir, concepts_dir, concept_papers)
    console.print(f"[bold green]정규화 완료 — {len(all_groups)}개 그룹, {merged}개 노트 병합[/bold green]")


def run_paper_resummary_vault(
    *,
    khub: Any,
    bad_only: bool,
    resummary_all: bool,
    threshold: int,
    provider: str | None,
    model: str | None,
    limit: int,
    dry_run: bool,
    console: Any,
    resolve_vault_papers_dir_fn: Callable[[str], Path | None],
    assess_vault_note_quality_fn: Callable[[str], dict[str, Any]],
    build_llm_fn: Callable[..., Any],
    collect_vault_note_text_fn: Callable[[Path, Path], str],
    update_vault_note_summary_fn: Callable[[Path, str], None],
    log: Any,
) -> None:
    _ = bad_only
    config = khub.config

    candidates: list[Path | None] = []
    if config.vault_path:
        candidates.append(resolve_vault_papers_dir_fn(config.vault_path.strip("'\"")))
    candidates.append(Path(config.papers_dir.strip("'\"")))

    papers_dir = None
    for candidate in candidates:
        if candidate and candidate.exists() and list(candidate.glob("*.md")):
            papers_dir = candidate
            break

    if not papers_dir:
        console.print("[red]논문 노트가 있는 폴더를 찾을 수 없습니다.[/red]")
        console.print(f"[dim]검색 경로: {[str(candidate) for candidate in candidates]}[/dim]")
        return

    md_files = [file for file in sorted(papers_dir.glob("*.md")) if file.name != "00_Concept_Index.md"]
    assessments = []
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8")
        assessments.append((md_path, assess_vault_note_quality_fn(content)))

    if resummary_all:
        targets = assessments
    else:
        targets = [(md_path, quality) for md_path, quality in assessments if quality["score"] < threshold]

    targets.sort(key=lambda item: item[1]["score"])
    if limit > 0:
        targets = targets[:limit]

    total = len(md_files)
    bad_count = len(targets)
    good_count = total - sum(1 for _, quality in assessments if quality["score"] < threshold)

    console.print("\n[bold]Obsidian Vault 논문 요약 리뷰[/bold]")
    console.print(f"  전체: {total}편 | 양호: {good_count}편 | 재요약 대상: {bad_count}편\n")

    if not targets:
        console.print("[green]모든 노트의 요약이 양호합니다.[/green]")
        return

    table = Table(title=f"재요약 대상 ({len(targets)}편)")
    table.add_column("노트", max_width=50)
    table.add_column("점수", width=5, justify="right")
    table.add_column("등급", width=10)
    table.add_column("문제점", max_width=25)
    for md_path, quality in targets:
        table.add_row(
            md_path.stem[:50],
            str(quality["score"]),
            f"[{quality['color']}]{quality['label']}[/{quality['color']}]",
            quality["reason"],
        )
    console.print(table)

    if dry_run:
        console.print("\n[dim]--dry-run: 변경 없이 종료. 실행하려면 --dry-run 제거[/dim]")
        return

    prov = provider or config.summarization_provider
    mdl = model or config.summarization_model
    console.print(f"\n[bold]{len(targets)}편 재요약 시작[/bold]")
    console.print(f"[dim]프로바이더: {prov}/{mdl}[/dim]\n")

    llm = build_llm_fn(config, prov, mdl, khub=khub)

    success = 0
    failed: list[dict[str, str]] = []
    for index, (md_path, _quality) in enumerate(targets, 1):
        title = md_path.stem
        console.print(f"  [{index}/{len(targets)}] {title[:50]}...", end=" ")

        text = collect_vault_note_text_fn(md_path, papers_dir)
        if len(text) < 50:
            console.print("[yellow]텍스트 부족, 스킵[/yellow]")
            continue

        source = "전문" if len(text) > 2000 else "abstract/노트"
        try:
            summary = llm.summarize_paper(text, title=title, language="ko")
            if not summary or len(summary) < 50:
                console.print("[yellow]요약 생성 실패[/yellow]")
                failed.append({"title": title, "error": "빈 응답"})
                continue
            update_vault_note_summary_fn(md_path, summary)
            success += 1
            console.print(f"[green]OK ({source}, {len(summary)}자)[/green]")
        except Exception as error:
            log.error("vault 재요약 실패 %s: %s", title, error)
            failed.append({"title": title, "error": str(error)})
            console.print(f"[red]FAIL ({error})[/red]")

    console.print(f"\n[bold green]{success}/{len(targets)}편 재요약 완료[/bold green]")
    if failed:
        console.print(f"[bold red]실패: {len(failed)}편[/bold red]")
        for item in failed:
            console.print(f"  {item['title'][:50]}: {item['error'][:80]}")
