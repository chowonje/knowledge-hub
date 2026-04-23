"""khub learn - Personal Learning Coach MVP v2 CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _service(ctx):
    khub = ctx.obj["khub"]
    if hasattr(khub, "learning_service"):
        return khub.learning_service()
    from knowledge_hub.learning import LearningCoachService

    return LearningCoachService(khub.config)


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]schema:[/bold] {payload.get('schema')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    if payload.get("topic"):
        console.print(f"[bold]topic:[/bold] {payload.get('topic')}")
    if payload.get("runId"):
        console.print(f"[bold]runId:[/bold] {payload.get('runId')}")
    if payload.get("warnings"):
        for item in payload["warnings"]:
            console.print(f"[yellow]- {item}[/yellow]")
    if payload.get("error"):
        console.print(f"[red]error:[/red] {payload.get('error')}")


def _load_answers(answers_json: str | None, answers_file: str | None) -> list[dict]:
    if answers_json:
        parsed = json.loads(answers_json)
        if not isinstance(parsed, list):
            raise click.BadParameter("--answers-json must be a JSON array")
        return [item for item in parsed if isinstance(item, dict)]
    if answers_file:
        path = Path(answers_file).expanduser().resolve()
        if not path.exists():
            raise click.BadParameter(f"answers file not found: {path}", param_hint="--answers-file")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise click.BadParameter("--answers-file JSON must be an array")
        return [item for item in parsed if isinstance(item, dict)]
    return []


@click.group("learn")
def learn_group():
    """학습 코치 (줄기 개념 맵/평가/다음 가지 추천)"""


@learn_group.command("map")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False, help="writeback 없이 결과만 계산")
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--canvas/--no-canvas", default=False, show_default=True, help="Obsidian Canvas 파일(05_Topic_Canvas.canvas) 생성")
@click.option("--allow-external", is_flag=True, default=False, help="P1/P2 sanitized facts 외부 호출 허용")
@click.pass_context
def learn_map(ctx, topic, source, days, top_k, as_json, dry_run, writeback, canvas, allow_external):
    """주제별 trunk/branch 맵 생성"""
    svc = _service(ctx)
    payload = svc.map(
        topic=topic,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        write_canvas=canvas and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("assess-template")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--concept-count", default=6, type=int, show_default=True)
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_assess_template(
    ctx,
    topic,
    session_id,
    concept_count,
    source,
    days,
    top_k,
    as_json,
    dry_run,
    writeback,
    allow_external,
):
    """세션 평가 템플릿 생성"""
    svc = _service(ctx)
    payload = svc.assess_template(
        topic=topic,
        session_id=session_id,
        concept_count=max(1, concept_count),
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("grade")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_grade(ctx, topic, session_id, as_json, dry_run, writeback, allow_external):
    """세션 연결도 채점"""
    svc = _service(ctx)
    payload = svc.grade(
        topic=topic,
        session_id=session_id,
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("next")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_next(ctx, topic, session_id, source, days, top_k, as_json, dry_run, writeback, allow_external):
    """통과/미통과 기반 다음 학습 가지 추천"""
    svc = _service(ctx)
    payload = svc.next(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("run")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--concept-count", default=6, type=int, show_default=True)
@click.option("--auto-next", is_flag=True, default=False)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--canvas/--no-canvas", default=False, show_default=True, help="run 시 map 단계 캔버스 생성")
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_run(
    ctx,
    topic,
    session_id,
    source,
    days,
    top_k,
    concept_count,
    auto_next,
    as_json,
    dry_run,
    writeback,
    canvas,
    allow_external,
):
    """map -> template -> (grade) -> (next) 오케스트레이션"""
    svc = _service(ctx)
    payload = svc.run(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        concept_count=max(1, concept_count),
        auto_next=auto_next,
        writeback=writeback and not dry_run,
        allow_external=allow_external,
        canvas=canvas and not dry_run,
    )
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]schema:[/bold] {payload.get('schema')}")
    console.print(f"[bold]status:[/bold] {payload.get('status')}")
    steps = payload.get("steps") if isinstance(payload.get("steps"), dict) else {}
    for name in ["map", "assessTemplate", "grade", "next"]:
        item = steps.get(name)
        if isinstance(item, dict):
            console.print(f"- {name}: {item.get('status')}")
        else:
            console.print(f"- {name}: skipped")


@learn_group.command("start-or-resume")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--force-new-session", is_flag=True, default=False)
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--concept-count", default=6, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_start_or_resume(ctx, topic, force_new_session, source, days, top_k, concept_count, as_json):
    """같은 주제의 최근 세션을 자동 이어서 시작"""
    svc = _service(ctx)
    payload = svc.start_or_resume_topic(
        topic=topic,
        force_new_session=force_new_session,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        concept_count=max(1, concept_count),
    )
    _emit(payload, as_json=as_json)


@learn_group.command("session-state")
@click.option("--topic", default="", help="주제")
@click.option("--session-id", default="", help="세션 ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_session_state(ctx, topic, session_id, as_json):
    """최근 세션 상태/약점/퀴즈 이력 조회"""
    svc = _service(ctx)
    payload = svc.get_session_state(topic=topic or None, session_id=session_id or None)
    _emit(payload, as_json=as_json)


@learn_group.command("explain-topic")
@click.option("--topic", required=True, help="주제")
@click.option("--question", required=True, help="설명 질문")
@click.option("--session-id", default="", help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--top-k", default=5, type=int, show_default=True)
@click.option("--min-score", default=0.3, type=float, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_explain_topic(ctx, topic, question, session_id, source, top_k, min_score, as_json):
    """근거 우선 설명 + 부족 시 모델 보완 설명"""
    svc = _service(ctx)
    payload = svc.explain_topic(
        topic=topic,
        question=question,
        session_id=session_id or None,
        source=source,
        top_k=max(1, top_k),
        min_score=float(min_score),
    )
    _emit(payload, as_json=as_json)


@learn_group.command("checkpoint")
@click.option("--topic", required=True, help="주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--summary", required=True, help="체크포인트 요약")
@click.option("--known-item", "known_items", multiple=True)
@click.option("--shaky-item", "shaky_items", multiple=True)
@click.option("--unknown-item", "unknown_items", multiple=True)
@click.option("--misconception", "misconceptions", multiple=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.pass_context
def learn_checkpoint(ctx, topic, session_id, summary, known_items, shaky_items, unknown_items, misconceptions, as_json, writeback):
    """평가 외 수동 체크포인트 저장"""
    svc = _service(ctx)
    payload = svc.checkpoint(
        topic=topic,
        session_id=session_id,
        summary=summary,
        known_items=list(known_items),
        shaky_items=list(shaky_items),
        unknown_items=list(unknown_items),
        misconceptions=list(misconceptions),
        writeback=writeback,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("analyze-gaps")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_analyze_gaps(ctx, topic, session_id, source, days, top_k, as_json, dry_run, writeback, allow_external):
    """세션/온톨로지 기반 부족 개념/근거/약한 엣지 진단"""
    svc = _service(ctx)
    payload = svc.analyze_gaps(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("quiz-generate")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--mix", default=None, type=click.Choice(["mixed", "mcq", "essay"]))
@click.option("--question-count", default=6, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_quiz_generate(
    ctx,
    topic,
    session_id,
    source,
    days,
    top_k,
    mix,
    question_count,
    as_json,
    dry_run,
    writeback,
    allow_external,
):
    """혼합형 퀴즈 생성"""
    svc = _service(ctx)
    payload = svc.generate_quiz(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        mix=mix,
        question_count=max(1, question_count),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("quiz-grade")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--answers-json", default=None, help="답안 JSON 배열 문자열")
@click.option("--answers-file", default=None, help="답안 JSON 파일 경로")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_quiz_grade(
    ctx,
    topic,
    session_id,
    source,
    days,
    top_k,
    answers_json,
    answers_file,
    as_json,
    dry_run,
    writeback,
    allow_external,
):
    """퀴즈 답안 채점 및 약점 피드백"""
    answers = _load_answers(answers_json, answers_file)
    svc = _service(ctx)
    payload = svc.grade_quiz(
        topic=topic,
        session_id=session_id,
        answers=answers,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("reinforce")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--top-k-per-gap", default=3, type=int, show_default=True, help="각 gap당 추천 소스 수")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_reinforce(ctx, topic, session_id, source, days, top_k, top_k_per_gap, as_json, dry_run, writeback, allow_external):
    """gap 분석 기반 지식 보강 추천 (소스 매칭 + 우선순위)"""
    svc = _service(ctx)
    payload = svc.reinforce(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        top_k_per_gap=max(1, top_k_per_gap),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    if as_json:
        console.print_json(data=payload)
        return
    _emit(payload, as_json=False)
    actions = payload.get("actions") or []
    if actions:
        console.print(f"\n[bold]추천 보강 액션 ({len(actions)}개):[/bold]")
        for i, action in enumerate(actions, 1):
            console.print(f"  {i}. [{action.get('priority', '?')}] {action.get('actionType', '?')}: {action.get('targetLabel', '?')}")
            for src in (action.get("sources") or [])[:2]:
                console.print(f"     - {src.get('title', '?')} (score: {src.get('score', 0):.2f})")
    summary = payload.get("summary") or {}
    if summary:
        console.print(
            f"\n[dim]총 {summary.get('totalActions', 0)}건 | "
            f"fill_concept: {summary.get('fillConcept', 0)}, "
            f"strengthen: {summary.get('strengthenEvidence', 0)}, "
            f"add_relation: {summary.get('addRelation', 0)}[/dim]"
        )


@learn_group.command("suggest-patch")
@click.option("--topic", required=True, help="관심 분야 주제")
@click.option("--session-id", required=True, help="세션 ID")
@click.option("--source", default="all", type=click.Choice(["all", "note", "paper", "web"]))
@click.option("--days", default=180, type=int, show_default=True)
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_suggest_patch(ctx, topic, session_id, source, days, top_k, as_json, dry_run, writeback, allow_external):
    """부족 영역 보완 초안(diff 텍스트) 제안 생성"""
    svc = _service(ctx)
    payload = svc.suggest_patch(
        topic=topic,
        session_id=session_id,
        source=source,
        days=max(1, days),
        top_k=max(1, top_k),
        writeback=writeback and not dry_run,
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("graph-build")
@click.option("--topic", required=True, help="학습 그래프를 생성할 주제")
@click.option("--top-k", default=12, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.option("--allow-external", is_flag=True, default=False)
@click.pass_context
def learn_graph_build(ctx, topic, top_k, as_json, allow_external):
    """온톨로지 기반 학습 그래프 후보를 생성하고 pending 큐에 적재"""
    svc = _service(ctx)
    payload = svc.build_learning_graph(
        topic=topic,
        top_k=max(1, top_k),
        allow_external=allow_external,
    )
    _emit(payload, as_json=as_json)


@learn_group.group("graph-pending")
def learn_graph_pending():
    """학습 그래프 pending 큐 관리"""


@learn_graph_pending.command("list")
@click.option("--topic", default=None, help="주제 필터")
@click.option("--item-type", default="all", type=click.Choice(["all", "edge", "path", "difficulty", "resource_link"]))
@click.option("--limit", default=100, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_graph_pending_list(ctx, topic, item_type, limit, as_json):
    svc = _service(ctx)
    payload = svc.list_learning_graph_pending(topic=topic, item_type=item_type, limit=max(1, limit))
    _emit(payload, as_json=as_json)


@learn_graph_pending.command("apply")
@click.option("--pending-id", required=True, type=int, help="적용할 pending ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_graph_pending_apply(ctx, pending_id, as_json):
    svc = _service(ctx)
    payload = svc.apply_learning_graph_pending(pending_id=pending_id)
    _emit(payload, as_json=as_json)


@learn_graph_pending.command("reject")
@click.option("--pending-id", required=True, type=int, help="거절할 pending ID")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_graph_pending_reject(ctx, pending_id, as_json):
    svc = _service(ctx)
    payload = svc.reject_learning_graph_pending(pending_id=pending_id)
    _emit(payload, as_json=as_json)


@learn_group.command("path-generate")
@click.option("--topic", required=True, help="학습 경로를 생성할 주제")
@click.option("--approved-only/--include-pending", default=True, show_default=True)
@click.option("--writeback/--no-writeback", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_path_generate(ctx, topic, approved_only, writeback, as_json):
    """승인된 학습 엣지 기반으로 학습 경로 생성"""
    svc = _service(ctx)
    payload = svc.generate_learning_path(
        topic=topic,
        approved_only=approved_only,
        writeback=writeback,
    )
    _emit(payload, as_json=as_json)


@learn_group.command("review-writeback")
@click.option("--topic", required=True, help="리뷰 인덱스 노트를 생성할 주제")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def learn_review_writeback(ctx, topic, limit, as_json):
    """Claims to Review / Top Concepts / Reading Queue 노트 생성"""
    svc = _service(ctx)
    payload = svc.write_learning_review_surfaces(topic=topic, limit=max(1, limit))
    _emit(payload, as_json=as_json)
