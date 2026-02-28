"""Obsidian writeback helpers for Learning Coach."""

from __future__ import annotations

import re
import json
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from knowledge_hub.learning.mapper import slugify_topic


@dataclass
class LearningHubPaths:
    topic_slug: str
    base_dir: Path
    hub_file: Path
    trunk_map_file: Path
    next_branches_file: Path
    web_sources_file: Path
    web_concepts_file: Path
    canvas_file: Path
    sessions_dir: Path

    def session_file(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.md"


class VaultWriteAdapter:
    """Adapter interface for vault read/write operations."""

    def read_text(self, path: Path) -> str:
        raise NotImplementedError

    def write_text(self, path: Path, content: str) -> None:
        raise NotImplementedError


class FileSystemVaultAdapter(VaultWriteAdapter):
    def read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        _ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")


class ObsidianCliPreferredAdapter(VaultWriteAdapter):
    """Try Obsidian CLI write first, then fall back to filesystem write."""

    def __init__(self, vault_path: str, cli_binary: str = "obsidian", vault_name: str = ""):
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.cli_binary = cli_binary
        self._fs = FileSystemVaultAdapter()
        self._cli_available = shutil.which(self.cli_binary) is not None
        self._cli_vault_name = vault_name.strip()
        self._cli_vault_root = self.vault_path
        self._cli_path_prefix = Path(".")
        if self._cli_available:
            self._resolve_cli_vault_binding()

    def read_text(self, path: Path) -> str:
        return self._fs.read_text(path)

    def write_text(self, path: Path, content: str) -> None:
        if self._cli_available and self._try_cli_write(path, content):
            return
        self._fs.write_text(path, content)

    def _try_cli_write(self, path: Path, content: str) -> bool:
        try:
            rel_path = path.resolve().relative_to(self.vault_path)
        except ValueError:
            return False

        cli_rel_path = (self._cli_path_prefix / rel_path).as_posix()
        command = [
            self.cli_binary,
            f"vault={self._cli_vault_name or self.vault_path.name}",
            "create",
            f"path={cli_rel_path}",
            f"content={content}",
            "overwrite",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = f"{result.stdout}\n{result.stderr}".lower()
            if "vault not found" in output:
                return False
            if result.returncode != 0:
                return False
            if not self._verify_cli_write(cli_rel_path, content):
                return False
            try:
                if path.exists():
                    return path.read_text(encoding="utf-8").rstrip() == content.rstrip()
                return False
            except Exception:
                return False
        except Exception:
            return False

    def _verify_cli_write(self, cli_rel_path: str, expected: str) -> bool:
        try:
            verify = subprocess.run(
                [
                    self.cli_binary,
                    f"vault={self._cli_vault_name or self.vault_path.name}",
                    "read",
                    f"path={cli_rel_path}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception:
            return False

        if verify.returncode != 0:
            return False

        text = self._strip_cli_noise((verify.stdout or "").splitlines())
        if not text:
            return False
        return text.rstrip() == expected.rstrip()

    @staticmethod
    def _strip_cli_noise(lines: list[str]) -> str:
        cleaned: list[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                cleaned.append("")
                continue
            if s.startswith("20") and "Loading updated app package" in s:
                continue
            if s.startswith("Your Obsidian installer is out of date"):
                continue
            if s == "Obsidian CLI":
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip("\n")

    def _resolve_cli_vault_binding(self) -> None:
        if self._cli_vault_name:
            return
        try:
            result = subprocess.run(
                [self.cli_binary, "vaults", "verbose"],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception:
            return

        if result.returncode != 0:
            return

        candidates: list[tuple[str, Path]] = []
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("your obsidian installer is out of date"):
                continue
            if line.lower().startswith("loading updated app package"):
                continue
            if "\t" not in line:
                continue
            name, raw_path = line.split("\t", 1)
            vault_root = Path(raw_path.strip()).expanduser().resolve()
            candidates.append((name.strip(), vault_root))

        best: tuple[str, Path] | None = None
        best_depth = -1
        for name, root in candidates:
            if self.vault_path == root or root in self.vault_path.parents:
                depth = len(root.parts)
                if depth > best_depth:
                    best = (name, root)
                    best_depth = depth

        if not best:
            return

        self._cli_vault_name, self._cli_vault_root = best
        try:
            self._cli_path_prefix = self.vault_path.relative_to(self._cli_vault_root)
        except ValueError:
            self._cli_path_prefix = Path(".")


def resolve_vault_write_adapter(
    vault_path: str,
    backend: str = "filesystem",
    cli_binary: str = "obsidian",
    vault_name: str = "",
) -> VaultWriteAdapter:
    mode = (backend or "filesystem").strip().lower()
    if mode in {"cli-preferred", "cli", "obsidian-cli"}:
        return ObsidianCliPreferredAdapter(vault_path=vault_path, cli_binary=cli_binary, vault_name=vault_name)
    return FileSystemVaultAdapter()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read(path: Path, adapter: VaultWriteAdapter | None = None) -> str:
    writer = adapter or FileSystemVaultAdapter()
    return writer.read_text(path)


def _write(path: Path, content: str, adapter: VaultWriteAdapter | None = None) -> None:
    writer = adapter or FileSystemVaultAdapter()
    writer.write_text(path, content)


def _upsert_marked_section(content: str, key: str, body: str) -> str:
    start = f"<!-- SECTION:{key}:start -->"
    end = f"<!-- SECTION:{key}:end -->"
    block = f"{start}\n{body.rstrip()}\n{end}"

    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}",
        re.DOTALL,
    )

    if pattern.search(content):
        return pattern.sub(block, content)

    if content and not content.endswith("\n"):
        content += "\n"
    return f"{content}\n{block}\n"


def _split_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        return {}, content
    lines = content.splitlines()
    if len(lines) < 3:
        return {}, content
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}, content

    frontmatter_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    try:
        parsed = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def _safe_canvas_id(value: str, fallback: str = "node") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "-", str(value).strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or fallback


def _text_nodes_and_edges_from_map(map_result: dict) -> tuple[list[dict], list[dict]]:
    trunks = map_result.get("trunks") if isinstance(map_result.get("trunks"), list) else []
    branches = map_result.get("branches") if isinstance(map_result.get("branches"), list) else []

    nodes: list[dict] = []
    edges: list[dict] = []
    topic = str(map_result.get("topic", "topic"))
    topic_node_id = _safe_canvas_id(topic, fallback="topic")

    nodes.append(
        {
            "id": topic_node_id,
            "type": "text",
            "x": 40,
            "y": 220,
            "width": 300,
            "height": 120,
            "text": f"# {topic}\n\n큰줄기 탐색 주제",
        }
    )

    trunk_node_ids: set[str] = set()
    for index, trunk in enumerate(trunks):
        trunk_id = _safe_canvas_id(str(trunk.get("canonical_id", f"trunk-{index}")), fallback=f"trunk-{index}")
        trunk_node_ids.add(trunk_id)
        nodes.append(
            {
                "id": trunk_id,
                "type": "text",
                "x": 420,
                "y": 80 + index * 180,
                "width": 320,
                "height": 140,
                "text": (
                    f"## {trunk.get('display_name', trunk_id)}\n"
                    f"canonical_id: `{trunk.get('canonical_id', '-')}`\n"
                    f"score: `{trunk.get('trunkScore', 0):.4f}`\n"
                    f"aliases: {', '.join(trunk.get('aliases', []) or [])}"
                ),
            }
        )
        edges.append(
            {
                "id": f"{topic_node_id}-{trunk_id}",
                "fromNode": topic_node_id,
                "toNode": trunk_id,
                "fromSide": "right",
                "toSide": "left",
            }
        )

    for index, branch in enumerate(branches):
        branch_id = _safe_canvas_id(
            str(branch.get("canonical_id", f"branch-{index}")),
            fallback=f"branch-{index}",
        )
        nodes.append(
            {
                "id": branch_id,
                "type": "text",
                "x": 820,
                "y": 100 + index * 160,
                "width": 300,
                "height": 150,
                "text": (
                    f"## {branch.get('display_name', branch_id)}\n"
                    f"canonical_id: `{branch.get('canonical_id', '-')}`\n"
                    f"confidence: `{branch.get('confidence', 0):.4f}`\n"
                    f"parents: {', '.join(branch.get('parentTrunkIds', []) or [])}"
                ),
            }
        )

        for parent in branch.get("parentTrunkIds", []) if isinstance(branch.get("parentTrunkIds", []), list) else []:
            parent_id = _safe_canvas_id(str(parent), fallback="topic")
            if parent_id not in trunk_node_ids:
                parent_id = topic_node_id
            edges.append(
                {
                    "id": f"{parent_id}-{branch_id}",
                    "fromNode": parent_id,
                    "toNode": branch_id,
                    "fromSide": "right",
                    "toSide": "left",
                }
            )

    return nodes, edges


def build_paths(vault_path: str, topic: str) -> LearningHubPaths:
    vault = Path(vault_path).expanduser().resolve()
    topic_slug = slugify_topic(topic)
    base = vault / "LearningHub" / topic_slug
    sessions = base / "sessions"
    return LearningHubPaths(
        topic_slug=topic_slug,
        base_dir=base,
        hub_file=base / "00_Hub.md",
        trunk_map_file=base / "01_Trunk_Map.md",
        next_branches_file=base / "02_Next_Branches.md",
        web_sources_file=base / "03_Web_Sources.md",
        web_concepts_file=base / "04_Web_Concepts.md",
        canvas_file=base / "05_Topic_Canvas.canvas",
        sessions_dir=sessions,
    )


def write_hub(
    paths: LearningHubPaths,
    topic: str,
    session_id: str,
    status_line: str,
    adapter: VaultWriteAdapter | None = None,
) -> None:
    content = _read(paths.hub_file, adapter=adapter)
    if not content:
        content = f"# Learning Hub: {topic}\n"

    content = _upsert_marked_section(
        content,
        "active-session",
        "\n".join(
            [
                "## Active Session",
                f"- session_id: `{session_id}`",
                f"- status: {status_line}",
            ]
        ),
    )
    content = _upsert_marked_section(
        content,
        "recent-scores",
        "\n".join(
            [
                "## Recent Scores",
                "- latest: (run grade 후 자동 갱신)",
            ]
        ),
    )
    content = _upsert_marked_section(
        content,
        "unlocked-branches",
        "\n".join(
            [
                "## Unlocked Branches",
                "- (run next 후 자동 갱신)",
            ]
        ),
    )
    content = _upsert_marked_section(
        content,
        "next-action",
        "\n".join(
            [
                "## Next Action",
                "- 세션 노트에서 개념 연결도를 작성한 뒤 `khub learn grade`를 실행하세요.",
            ]
        ),
    )

    _write(paths.hub_file, content, adapter=adapter)


def write_trunk_map(
    paths: LearningHubPaths,
    topic: str,
    map_result: dict,
    adapter: VaultWriteAdapter | None = None,
) -> None:
    trunks = map_result.get("trunks") if isinstance(map_result.get("trunks"), list) else []
    branches = map_result.get("branches") if isinstance(map_result.get("branches"), list) else []

    lines: list[str] = [
        f"# Trunk Map: {topic}",
        "",
        f"- status: {map_result.get('status')}",
        f"- suggestedTopK: {map_result.get('suggestedTopK')}",
        "",
        "## Trunks",
    ]

    for idx, trunk in enumerate(trunks, start=1):
        lines.extend(
            [
                f"{idx}. **{trunk.get('display_name')}** (`{trunk.get('canonical_id')}`)",
                f"   - trunkScore: {trunk.get('trunkScore')}",
                f"   - breakdown: {trunk.get('scoreBreakdown')}",
                f"   - evidenceSources: {', '.join(trunk.get('evidenceSources', []))}",
            ]
        )

    lines.append("")
    lines.append("## Branch Candidates")
    for idx, branch in enumerate(branches[:20], start=1):
        lines.append(
            f"{idx}. **{branch.get('display_name')}** (`{branch.get('canonical_id')}`)"
            f" - confidence={branch.get('confidence')}"
            f" - parents={', '.join(branch.get('parentTrunkIds', []))}"
        )

    _write(paths.trunk_map_file, "\n".join(lines).rstrip() + "\n", adapter=adapter)


def write_session_template(
    paths: LearningHubPaths,
    topic: str,
    session_id: str,
    target_trunk_ids: Iterable[str],
    policy_mode: str = "local-only",
    overwrite: bool = False,
    adapter: VaultWriteAdapter | None = None,
) -> Path:
    session_path = paths.session_file(session_id)

    trunks = [str(item).strip() for item in target_trunk_ids if str(item).strip()]
    now_iso = datetime.now(timezone.utc).isoformat()
    created_at = now_iso
    body = "\n".join(
        [
            "# Session Concept Map",
            "",
            "아래 형식으로 엣지를 작성하세요:",
            "```text",
            "source -> relation -> target | evidence_ptr: path=...;heading=...;block_id=...;snippet=... | confidence: 1-5",
            "```",
            "",
            "예시:",
            "```text",
            "Transformer -> enables -> In-Context Learning | evidence_ptr: path=Papers/notes.md;heading=Attention;block_id=^attn;snippet=attention supports context learning | confidence: 4",
            "```",
            "",
            "## Concept Map Edges",
            "",
            "- ",
        ]
    )

    if session_path.exists() and not overwrite:
        existing = _read(session_path, adapter=adapter)
        parsed_frontmatter, existing_body = _split_frontmatter(existing)
        created_at = str(parsed_frontmatter.get("created_at") or created_at)
        if existing_body.strip():
            body = existing_body.rstrip() + "\n"

    frontmatter_lines = [
        "---",
        f"topic: {topic!r}",
        f"topic_slug: {paths.topic_slug!r}",
        f"session_id: {session_id!r}",
        "target_trunk_ids:",
    ]
    for trunk_id in trunks:
        frontmatter_lines.append(f"  - {trunk_id!r}")
    frontmatter_lines.extend(
        [
            f"policy_mode: {policy_mode!r}",
            f"created_at: {created_at!r}",
            f"updated_at: {now_iso!r}",
        ]
    )
    frontmatter_lines.extend(
        [
        "---",
            "",
        ]
    )

    _write(session_path, "\n".join(frontmatter_lines) + body, adapter=adapter)
    return session_path


def write_next_branches(
    paths: LearningHubPaths,
    topic: str,
    next_result: dict,
    adapter: VaultWriteAdapter | None = None,
) -> None:
    lines = [
        f"# Next Branches: {topic}",
        "",
        f"- status: {next_result.get('status')}",
        f"- loadSignal: {next_result.get('loadSignal')}",
        "",
        "## Unlock Plan",
    ]

    unlock_plan = next_result.get("unlockPlan") if isinstance(next_result.get("unlockPlan"), list) else []
    if unlock_plan:
        for idx, item in enumerate(unlock_plan, start=1):
            lines.append(
                f"{idx}. **{item.get('display_name', item.get('canonical_id'))}** ({item.get('canonical_id')})"
                f" - {item.get('reason')}"
            )
            lines.append(f"   - mini mission: {item.get('miniMission')}")
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("## Remediation Plan")
    remediation = next_result.get("remediationPlan") if isinstance(next_result.get("remediationPlan"), list) else []
    if remediation:
        for idx, item in enumerate(remediation, start=1):
            lines.append(f"{idx}. focus={item.get('focus')} - {item.get('reason')}")
            lines.append(f"   - task: {item.get('task')}")
    else:
        lines.append("- 없음")

    lines.append("")
    lines.append("## Reasoning")
    for item in (next_result.get("reasoning") or []):
        lines.append(f"- {item}")

    _write(paths.next_branches_file, "\n".join(lines).rstrip() + "\n", adapter=adapter)


def write_web_sources(
    paths: LearningHubPaths,
    topic: str,
    run_summary: dict,
    adapter: VaultWriteAdapter | None = None,
) -> None:
    content = _read(paths.web_sources_file, adapter=adapter)
    if not content:
        content = f"# Web Sources: {topic}\n"

    docs = run_summary.get("docs") if isinstance(run_summary.get("docs"), list) else []
    failed = run_summary.get("failed") if isinstance(run_summary.get("failed"), list) else []

    overview = "\n".join(
        [
            "## Run Summary",
            f"- runId: `{run_summary.get('runId', '-')}`",
            f"- engine: {run_summary.get('engine', '-')}",
            f"- requested: {run_summary.get('requested', 0)}",
            f"- crawled: {run_summary.get('crawled', 0)}",
            f"- stored: {run_summary.get('stored', 0)}",
        ]
    )
    content = _upsert_marked_section(content, "run-summary", overview)

    source_lines = ["## Sources"]
    if docs:
        for item in docs[:200]:
            source_lines.append(
                f"- `{item.get('note_id', '-')}` | {item.get('url', '-')}"
                f" | local: `{item.get('file_path', '-')}`"
            )
    else:
        source_lines.append("- 없음")
    content = _upsert_marked_section(content, "source-list", "\n".join(source_lines))

    failed_lines = ["## Failures"]
    if failed:
        for item in failed[:100]:
            failed_lines.append(f"- {item.get('url', '-')}: {item.get('error', '-')}")
    else:
        failed_lines.append("- 없음")
    content = _upsert_marked_section(content, "failed-list", "\n".join(failed_lines))

    _write(paths.web_sources_file, content, adapter=adapter)


def write_web_concepts(
    paths: LearningHubPaths,
    topic: str,
    ontology_summary: dict,
    adapter: VaultWriteAdapter | None = None,
) -> None:
    content = _read(paths.web_concepts_file, adapter=adapter)
    if not content:
        content = f"# Web Concepts: {topic}\n"

    summary = "\n".join(
        [
            "## Ontology Summary",
            f"- runId: `{ontology_summary.get('runId', '-')}`",
            f"- conceptsAccepted: {ontology_summary.get('conceptsAccepted', 0)}",
            f"- relationsAccepted: {ontology_summary.get('relationsAccepted', 0)}",
            f"- pendingCount: {ontology_summary.get('pendingCount', 0)}",
            f"- aliasesAdded: {ontology_summary.get('aliasesAdded', 0)}",
        ]
    )
    content = _upsert_marked_section(content, "ontology-summary", summary)

    concept_lines = ["## Accepted Concepts"]
    accepted_concepts = (
        ontology_summary.get("acceptedConcepts")
        if isinstance(ontology_summary.get("acceptedConcepts"), list)
        else []
    )
    if accepted_concepts:
        for item in accepted_concepts[:150]:
            concept_lines.append(
                f"- `{item.get('canonical_id', '-')}` {item.get('display_name', '-')}"
                f" (confidence={item.get('confidence', 0):.3f})"
            )
    else:
        concept_lines.append("- 없음")
    content = _upsert_marked_section(content, "accepted-concepts", "\n".join(concept_lines))

    relation_lines = ["## Accepted Relations"]
    accepted_relations = (
        ontology_summary.get("acceptedRelations")
        if isinstance(ontology_summary.get("acceptedRelations"), list)
        else []
    )
    if accepted_relations:
        for item in accepted_relations[:200]:
            relation_lines.append(
                f"- `{item.get('source_canonical_id', '-')}` -> {item.get('relation_norm', '-')}"
                f" -> `{item.get('target_canonical_id', '-')}`"
                f" (confidence={item.get('confidence', 0):.3f})"
            )
    else:
        relation_lines.append("- 없음")
    content = _upsert_marked_section(content, "accepted-relations", "\n".join(relation_lines))

    pending_lines = ["## Pending Queue", "승인/거절은 `khub crawl pending apply|reject --id <id>`로 처리"]
    pending_items = (
        ontology_summary.get("pendingSamples")
        if isinstance(ontology_summary.get("pendingSamples"), list)
        else []
    )
    if pending_items:
        for item in pending_items[:150]:
            pending_lines.append(
                f"- id={item.get('id')} kind={item.get('kind')} confidence={item.get('confidence')}"
            )
    else:
        pending_lines.append("- 없음")
    content = _upsert_marked_section(content, "pending-queue", "\n".join(pending_lines))

    _write(paths.web_concepts_file, content, adapter=adapter)


def write_canvas(
    paths: LearningHubPaths,
    topic: str,
    map_result: dict,
    adapter: VaultWriteAdapter | None = None,
) -> Path:
    nodes, edges = _text_nodes_and_edges_from_map(map_result)
    payload = {"nodes": nodes, "edges": edges}
    _write(paths.canvas_file, json.dumps(payload, ensure_ascii=False, indent=2), adapter=adapter)
    return paths.canvas_file
