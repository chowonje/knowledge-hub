"""khub vault - vault organization helpers."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.vault.ai_organizer import AIVaultOrganizer
from knowledge_hub.vault.cluster_materializer import (
    ClusterMaterializationError,
    ClusterMaterializationOptions,
    VaultClusterMaterializer,
)
from knowledge_hub.vault.organizer import VaultOrganizer
from knowledge_hub.vault.topology import TopologyBuildError, TopologyBuildOptions, VaultTopologyBuilder

console = Console()


def _sqlite_db(khub):
    if hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    return SQLiteDatabase(khub.config.sqlite_path)


def _vector_db(khub):
    if hasattr(khub, "vector_db"):
        return khub.vector_db()
    config = khub.config
    return VectorDatabase(config.vector_db_path, config.collection_name)


@click.group("vault")
def vault_group():
    """Obsidian vault 정리/재구성 도구"""


@vault_group.command("organize")
@click.option("--vault-path", default=None, help="Obsidian vault 경로 (기본: 설정값)")
@click.option("--apply", is_flag=True, default=False, help="실제 파일/인덱스 반영")
@click.option("--relocate", is_flag=True, default=False, help="노트를 컬렉션 폴더로 실제 이동 (--apply 필요)")
@click.option("--output-dir", default="LearningHub/Collections", show_default=True, help="결과 저장 상대 경로(vault 기준)")
@click.option("--similarity-threshold", default=0.16, type=click.FloatRange(0.0, 1.0), show_default=True, help="노트 연결 임계치")
@click.option("--min-cluster-size", default=2, type=click.IntRange(1, 999), show_default=True, help="singleton 판단 최소 크기")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vault_organize(
    ctx,
    vault_path,
    apply,
    relocate,
    output_dir,
    similarity_threshold,
    min_cluster_size,
    as_json,
):
    """섞여 있는 노트를 주제별 컬렉션 + 복습 순서로 정리"""
    config = ctx.obj["khub"].config
    resolved_vault = vault_path or config.vault_path
    if not resolved_vault:
        console.print("[red]vault path가 없습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    if relocate and not apply:
        console.print("[yellow]--relocate는 --apply와 함께만 동작합니다. preview로 진행합니다.[/yellow]")
        relocate = False

    organizer = VaultOrganizer(
        vault_path=resolved_vault,
        exclude_folders=config.vault_excludes,
    )
    result = organizer.organize(
        apply=apply,
        relocate=relocate,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )

    if as_json:
        console.print_json(data=result)
        return

    console.print(
        f"[bold]mode:[/bold] {result['mode']} | "
        f"[bold]clusters:[/bold] {result['clusterCount']} | "
        f"[bold]notes:[/bold] {result['noteCount']} | "
        f"[bold]singletons:[/bold] {result['singletonCount']}"
    )
    if apply:
        console.print(
            f"[bold]summary:[/bold] {result['summaryPath']} | "
            f"[bold]manifest:[/bold] {result['manifestPath']} | "
            f"[bold]moved:[/bold] {result['movedCount']}"
        )
    else:
        console.print("[dim]preview 모드: 변경 없음. 적용하려면 --apply[/dim]")

    table = Table(title="Collections (top 12)")
    table.add_column("ID", style="dim", width=11)
    table.add_column("Label", max_width=36)
    table.add_column("Notes", justify="right", width=6)
    table.add_column("Singleton", width=9)
    table.add_column("Path", max_width=44)

    for cluster in result["clusters"][:12]:
        table.add_row(
            cluster["id"],
            cluster["label"],
            str(cluster["size"]),
            "Y" if cluster["singleton"] else "",
            cluster["collectionPath"],
        )
    console.print(table)


@vault_group.command("organize-ai")
@click.option("--vault-path", default=None, help="Obsidian vault 경로 (기본: 설정값)")
@click.option("--scope", type=click.Choice(["projects-ai"]), default="projects-ai", show_default=True)
@click.option("--dry-run/--apply", "dry_run", default=True, show_default=True, help="미리보기 또는 실제 적용")
@click.option("--report-json", is_flag=True, default=False, help="JSON 리포트 출력")
@click.pass_context
def vault_organize_ai(ctx, vault_path, scope, dry_run, report_json):
    """Projects/AI를 paper/concept/web/agent 기준으로 재정리"""
    config = ctx.obj["khub"].config
    resolved_vault = vault_path or config.vault_path
    if not resolved_vault:
        console.print("[red]vault path가 없습니다. khub config set obsidian.vault_path <경로>[/red]")
        return
    if scope != "projects-ai":
        raise click.BadParameter(f"unsupported scope: {scope}")

    organizer = AIVaultOrganizer(resolved_vault)
    result = organizer.organize_projects_ai(apply=not dry_run)

    if report_json:
        console.print_json(data=result)
        return

    console.print(
        f"[bold]organize-ai[/bold] mode={result['mode']} planned={result['plannedMoveCount']} "
        f"applied={result['appliedMoveCount']} rewrittenLinks={result['rewrittenLinks']}"
    )
    console.print(
        f"duplicates={len(result.get('duplicateGroups', []))} "
        f"manualReview={len(result.get('manualReview', []))}"
    )

    move_table = Table(title="AI Organize Moves (top 15)")
    move_table.add_column("From", max_width=48)
    move_table.add_column("To", max_width=48)
    for move in (result.get("fileMoves", []) + result.get("directoryMoves", []))[:15]:
        move_table.add_row(move["from"], move["to"])
    console.print(move_table)

    review_items = result.get("manualReview", [])
    if review_items:
        review_table = Table(title="Manual Review (top 15)")
        review_table.add_column("Path", max_width=48)
        review_table.add_column("Reason", max_width=28)
        review_table.add_column("Target", max_width=48)
        for item in review_items[:15]:
            review_table.add_row(item.get("path", ""), item.get("reason", ""), item.get("target", ""))
        console.print(review_table)


@vault_group.command("topology-build")
@click.option("--vault-path", default=None, help="Obsidian vault 경로 (기본: 설정값)")
@click.option("--projection", type=click.Choice(["auto", "umap", "pca"]), default="auto", show_default=True)
@click.option("--neighbors", default=15, type=click.IntRange(1, 500), show_default=True)
@click.option("--min-dist", "min_dist", default=0.08, type=click.FloatRange(0.0, 1.0), show_default=True)
@click.option("--similarity-threshold", default=0.30, type=click.FloatRange(0.0, 1.0), show_default=True)
@click.option("--min-cluster-size", default=2, type=click.IntRange(1, 999), show_default=True)
@click.option("--spherize/--no-spherize", default=False, show_default=True)
@click.option("--output", default=None, help="snapshot 출력 경로 (기본: <vault>/.obsidian/khub/topology/latest.json)")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vault_topology_build(
    ctx,
    vault_path,
    projection,
    neighbors,
    min_dist,
    similarity_threshold,
    min_cluster_size,
    spherize,
    output,
    as_json,
):
    """Vault note topology snapshot 생성"""
    config = ctx.obj["khub"].config
    resolved_vault = vault_path or config.vault_path
    if not resolved_vault:
        console.print("[red]vault path가 없습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    khub = ctx.obj["khub"]
    vector_db = _vector_db(khub)
    sqlite_db = _sqlite_db(khub)
    builder = VaultTopologyBuilder(resolved_vault, vector_db=vector_db, sqlite_db=sqlite_db)

    try:
        payload = builder.build(
            TopologyBuildOptions(
                projection=projection,
                neighbors=neighbors,
                min_dist=min_dist,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                spherize=spherize,
                output_path=output,
            )
        )
        output_path = builder.write_snapshot(payload, output_path=output)
    except TopologyBuildError as error:
        console.print(f"[red]topology build failed:[/red] {error}")
        return

    payload["outputPath"] = output_path
    if as_json:
        console.print_json(data=payload)
        return

    coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {}
    console.print(
        f"[bold]topology[/bold] notes={coverage.get('embeddedNotes', 0)}/{coverage.get('totalVaultNotes', 0)} "
        f"clusters={len(payload.get('clusters', []))} edges={len(payload.get('edges', []))}"
    )
    console.print(
        f"[bold]projection:[/bold] {payload.get('projection')} | "
        f"[bold]neighbors:[/bold] {payload.get('neighbors')} | "
        f"[bold]minDist:[/bold] {payload.get('minDist')} | "
        f"[bold]spherize:[/bold] {payload.get('spherize')}"
    )
    console.print(f"[bold]output:[/bold] {output_path}")


@vault_group.command("cluster-materialize")
@click.option("--vault-path", default=None, help="Obsidian vault 경로 (기본: 설정값)")
@click.option("--snapshot", default=None, help="topology snapshot 경로 (기본: <vault>/.obsidian/khub/topology/latest.json)")
@click.option("--apply/--dry-run", "apply", default=False, show_default=True)
@click.option("--output-dir", default="LearningHub/Cluster_Views", show_default=True, help="생성 cluster note 저장 상대 경로(vault 기준)")
@click.option("--max-cluster-links", default=2, type=click.IntRange(0, 20), show_default=True)
@click.option("--max-bridge-links", default=1, type=click.IntRange(0, 20), show_default=True)
@click.option("--include-singletons/--no-singletons", default=False, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vault_cluster_materialize(
    ctx,
    vault_path,
    snapshot,
    apply,
    output_dir,
    max_cluster_links,
    max_bridge_links,
    include_singletons,
    as_json,
):
    """Topology snapshot을 기반으로 reversible cluster overlay 생성"""
    config = ctx.obj["khub"].config
    resolved_vault = vault_path or config.vault_path
    if not resolved_vault:
        console.print("[red]vault path가 없습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    materializer = VaultClusterMaterializer(resolved_vault)
    try:
        payload = materializer.materialize(
            ClusterMaterializationOptions(
                snapshot_path=snapshot,
                apply=apply,
                output_dir=output_dir,
                max_cluster_links=max_cluster_links,
                max_bridge_links=max_bridge_links,
                include_singletons=include_singletons,
            )
        )
    except ClusterMaterializationError as error:
        console.print(f"[red]cluster materialization failed:[/red] {error}")
        return

    if as_json:
        console.print_json(data=payload)
        return

    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    console.print(
        f"[bold]cluster overlay[/bold] mode={payload.get('mode')} "
        f"clusters={counts.get('clustersGenerated', 0)} "
        f"touched={counts.get('notesTouched', 0)} "
        f"singletonsSkipped={counts.get('singletonsSkipped', 0)}"
    )
    if apply:
        console.print(f"[bold]manifest:[/bold] {payload.get('manifestPath', '')}")
    else:
        console.print("[dim]preview 모드: 변경 없음. 적용하려면 --apply[/dim]")

    table = Table(title="Cluster Notes (top 12)")
    table.add_column("Cluster", style="dim", width=12)
    table.add_column("Label", max_width=32)
    table.add_column("Members", justify="right", width=7)
    table.add_column("Bridge", justify="right", width=7)
    table.add_column("Path", max_width=44)
    for cluster_note in payload.get("clusterNotes", [])[:12]:
        table.add_row(
            cluster_note.get("clusterId", ""),
            cluster_note.get("label", ""),
            str(cluster_note.get("memberCount", "")),
            str(cluster_note.get("bridgeCount", "")),
            cluster_note.get("path", ""),
        )
    console.print(table)


@vault_group.command("cluster-revert")
@click.option("--vault-path", default=None, help="Obsidian vault 경로 (기본: 설정값)")
@click.option("--manifest", default=None, help="되돌릴 manifest 경로 (기본: latest)")
@click.option("--latest/--no-latest", default=True, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def vault_cluster_revert(ctx, vault_path, manifest, latest, as_json):
    """가장 최근 또는 지정된 cluster overlay를 되돌림"""
    config = ctx.obj["khub"].config
    resolved_vault = vault_path or config.vault_path
    if not resolved_vault:
        console.print("[red]vault path가 없습니다. khub config set obsidian.vault_path <경로>[/red]")
        return

    materializer = VaultClusterMaterializer(resolved_vault)
    try:
        payload = materializer.revert(manifest_path=manifest, latest=latest)
    except ClusterMaterializationError as error:
        console.print(f"[red]cluster revert failed:[/red] {error}")
        return

    if as_json:
        console.print_json(data=payload)
        return

    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    console.print(
        f"[bold]cluster revert[/bold] deleted={counts.get('generatedDeleted', 0)} "
        f"cleaned={counts.get('touchedCleaned', 0)} "
        f"missing={counts.get('generatedMissing', 0) + counts.get('touchedMissing', 0)}"
    )
    console.print(f"[bold]manifest:[/bold] {payload.get('manifestPath', '')}")
