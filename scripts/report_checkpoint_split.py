#!/usr/bin/env python3
"""Report current worktree changes by checkpoint bucket."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal


DefinitionSource = Literal["json", "pathspec"]


@dataclass(frozen=True)
class ChangedPath:
    path: str
    status: str


@dataclass(frozen=True)
class PathspecRule:
    pattern: str
    exclude: bool = False
    glob: bool = False


@dataclass(frozen=True)
class BucketDefinition:
    order: int
    slug: str
    title: str
    description: str
    source_kind: DefinitionSource
    source_path: Path
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    rules: list[PathspecRule] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report current git changes by ordered checkpoint bucket."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the git worktree.",
    )
    parser.add_argument(
        "--definition-source",
        choices=("auto", "json", "pathspec"),
        default="auto",
        help="Where to load bucket definitions from.",
    )
    parser.add_argument(
        "--checkpoints-json",
        type=Path,
        default=None,
        help="Checkpoint bucket definition file.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=None,
        help="Fallback directory containing ordered *.pathspec manifests.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Show bucket paths in text mode.",
    )
    parser.add_argument(
        "--write-pathspec-dir",
        type=Path,
        help="Write exact current changed paths per bucket to this directory.",
    )
    parser.add_argument(
        "--max-paths-per-bucket",
        type=int,
        default=20,
        help="Maximum paths to print per bucket in text mode when --show-paths is set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON.",
    )
    parser.add_argument(
        "--fail-on-unmatched",
        action="store_true",
        help="Exit non-zero when any changed paths are not covered by a bucket.",
    )
    return parser.parse_args()


def parse_pathspec_line(raw_line: str) -> PathspecRule:
    line = raw_line.strip()
    if line.startswith(":("):
        end = line.find(")")
        if end == -1:
            raise ValueError(f"invalid pathspec magic: {raw_line!r}")
        magic = {part.strip() for part in line[2:end].split(",") if part.strip()}
        pattern = line[end + 1 :]
        return PathspecRule(
            pattern=pattern,
            exclude="exclude" in magic,
            glob="glob" in magic,
        )
    return PathspecRule(pattern=line)


def match_pattern(path: str, pattern: str) -> bool:
    if any(token in pattern for token in "*?[]"):
        return fnmatch(path, pattern)
    if pattern.endswith("/"):
        return path.startswith(pattern)
    return path == pattern or path.startswith(f"{pattern}/")


def match_rule(path: str, rule: PathspecRule) -> bool:
    if rule.glob:
        return fnmatch(path, rule.pattern)
    return match_pattern(path, rule.pattern)


def parse_changed_paths(repo_root: Path) -> list[ChangedPath]:
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    changed_paths: list[ChangedPath] = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path = raw_line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = bytes(path[1:-1], "utf-8").decode("unicode_escape")
        changed_paths.append(ChangedPath(path=path, status=status))
    return changed_paths


def status_category(status: str) -> str:
    if status == "??":
        return "untracked"
    if "D" in status:
        return "deleted"
    if "R" in status:
        return "renamed"
    if "C" in status:
        return "copied"
    if "A" in status:
        return "added"
    if "M" in status:
        return "modified"
    return "other"


def relative_path(base: Path, path: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_definitions(checkpoints_json: Path) -> list[BucketDefinition]:
    payload = json.loads(checkpoints_json.read_text(encoding="utf-8"))
    buckets = []
    for bucket in sorted(payload.get("buckets", []), key=lambda item: item.get("order", 0)):
        buckets.append(
            BucketDefinition(
                order=int(bucket.get("order", len(buckets) + 1)),
                slug=str(bucket["slug"]),
                title=str(bucket.get("title", bucket["slug"])),
                description=str(bucket.get("description", "")),
                source_kind="json",
                source_path=checkpoints_json,
                include_patterns=list(bucket.get("patterns", [])),
                exclude_patterns=list(bucket.get("exclude_patterns", [])),
            )
        )
    return buckets


def load_pathspec_definitions(manifests_dir: Path) -> list[BucketDefinition]:
    buckets: list[BucketDefinition] = []
    for index, manifest_path in enumerate(sorted(manifests_dir.glob("*.pathspec")), start=1):
        stem = manifest_path.stem
        order = index
        slug = stem
        if "-" in stem and stem.split("-", 1)[0].isdigit():
            order = int(stem.split("-", 1)[0])
            slug = stem.split("-", 1)[1]
        rules: list[PathspecRule] = []
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rules.append(parse_pathspec_line(raw_line))
        buckets.append(
            BucketDefinition(
                order=order,
                slug=slug,
                title=slug.replace("-", " "),
                description="",
                source_kind="pathspec",
                source_path=manifest_path,
                rules=rules,
            )
        )
    return sorted(buckets, key=lambda bucket: bucket.order)


def load_bucket_definitions(
    definition_source: str,
    checkpoints_json: Path,
    manifests_dir: Path,
) -> tuple[DefinitionSource, Path, list[BucketDefinition]]:
    if definition_source in {"auto", "json"} and checkpoints_json.exists():
        return "json", checkpoints_json, load_json_definitions(checkpoints_json)
    if definition_source in {"auto", "pathspec"} and manifests_dir.exists():
        return "pathspec", manifests_dir, load_pathspec_definitions(manifests_dir)
    raise FileNotFoundError("no checkpoint definitions found")


def bucket_matches(path: str, bucket: BucketDefinition) -> bool:
    if bucket.source_kind == "json":
        include_match = any(match_pattern(path, pattern) for pattern in bucket.include_patterns)
        exclude_match = any(match_pattern(path, pattern) for pattern in bucket.exclude_patterns)
        return include_match and not exclude_match

    include_rules = [rule for rule in bucket.rules if not rule.exclude]
    exclude_rules = [rule for rule in bucket.rules if rule.exclude]
    if not any(match_rule(path, rule) for rule in include_rules):
        return False
    return not any(match_rule(path, rule) for rule in exclude_rules)


def build_report(
    repo_root: Path,
    definition_kind: DefinitionSource,
    definition_path: Path,
    buckets: list[BucketDefinition],
) -> dict[str, object]:
    changed_paths = parse_changed_paths(repo_root)
    report_buckets: list[dict[str, object]] = []
    matched_paths: set[str] = set()

    for bucket in buckets:
        bucket_paths = [
            changed
            for changed in changed_paths
            if changed.path not in matched_paths and bucket_matches(changed.path, bucket)
        ]
        matched_paths.update(changed.path for changed in bucket_paths)
        report_buckets.append(
            {
                "order": bucket.order,
                "slug": bucket.slug,
                "name": f"{bucket.order:02d}-{bucket.slug}",
                "title": bucket.title,
                "description": bucket.description,
                "definitionSource": relative_path(repo_root, bucket.source_path),
                "changedCount": len(bucket_paths),
                "statusBreakdown": dict(
                    Counter(status_category(changed.status) for changed in bucket_paths)
                ),
                "paths": [
                    {
                        "path": changed.path,
                        "status": changed.status,
                        "statusCategory": status_category(changed.status),
                    }
                    for changed in bucket_paths
                ],
            }
        )

    unmatched = [
        {
            "path": changed.path,
            "status": changed.status,
            "statusCategory": status_category(changed.status),
        }
        for changed in changed_paths
        if changed.path not in matched_paths
    ]

    return {
        "repoRoot": repo_root.as_posix(),
        "definitionSourceKind": definition_kind,
        "definitionSourcePath": relative_path(repo_root, definition_path),
        "totalChangedPaths": len(changed_paths),
        "matchedChangedPaths": len(changed_paths) - len(unmatched),
        "unmatchedChangedPaths": len(unmatched),
        "buckets": report_buckets,
        "unmatched": unmatched,
    }


def format_status_breakdown(breakdown: dict[str, int]) -> str:
    if not breakdown:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(breakdown.items()))


def write_exact_pathspecs(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket in report["buckets"]:
        pathspec_path = output_dir / f"{bucket['order']:02d}-{bucket['slug']}.pathspec"
        lines = [
            f"# Exact changed paths for {bucket['name']}",
            "# Generated by scripts/report_checkpoint_split.py",
            "",
        ]
        if bucket["paths"]:
            lines.extend(item["path"] for item in bucket["paths"])
        else:
            lines.append("# no changed paths")
        pathspec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        bucket["exactPathspecFile"] = pathspec_path.as_posix()

    unmatched_path = output_dir / "unmatched.pathspec"
    lines = [
        "# Exact changed paths that did not match any checkpoint bucket",
        "# Generated by scripts/report_checkpoint_split.py",
        "",
    ]
    if report["unmatched"]:
        lines.extend(item["path"] for item in report["unmatched"])
    else:
        lines.append("# no unmatched paths")
    unmatched_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["unmatchedPathspecFile"] = unmatched_path.as_posix()


def emit_text(
    report: dict[str, object],
    show_paths: bool,
    max_paths_per_bucket: int,
) -> str:
    lines = [
        "Checkpoint split report",
        f"Repo: {report['repoRoot']}",
        "Definitions: "
        f"{report['definitionSourcePath']} ({report['definitionSourceKind']})",
        "Changed paths: "
        f"{report['totalChangedPaths']} total, "
        f"{report['matchedChangedPaths']} matched, "
        f"{report['unmatchedChangedPaths']} unmatched",
    ]

    for bucket in report["buckets"]:
        lines.append("")
        lines.append(
            f"{bucket['name']}: {bucket['changedCount']} changed "
            f"({format_status_breakdown(bucket['statusBreakdown'])})"
        )
        if bucket.get("exactPathspecFile"):
            lines.append(f"  exact pathspec: {bucket['exactPathspecFile']}")
        if show_paths:
            visible_items = bucket["paths"][:max_paths_per_bucket]
            for item in visible_items:
                lines.append(f"  {item['status']} {item['path']}")
            remaining = bucket["changedCount"] - len(visible_items)
            if remaining > 0:
                lines.append(f"  ... {remaining} more")

    lines.append("")
    lines.append(f"Unmatched: {report['unmatchedChangedPaths']}")
    if report.get("unmatchedPathspecFile"):
        lines.append(f"  exact pathspec: {report['unmatchedPathspecFile']}")
    if show_paths:
        visible_unmatched = report["unmatched"][:max_paths_per_bucket]
        for item in visible_unmatched:
            lines.append(f"  {item['status']} {item['path']}")
        remaining = report["unmatchedChangedPaths"] - len(visible_unmatched)
        if remaining > 0:
            lines.append(f"  ... {remaining} more")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoints_json = (
        args.checkpoints_json.resolve()
        if args.checkpoints_json
        else repo_root / "ops" / "checkpoints" / "checkpoints.json"
    )
    manifests_dir = (
        args.manifests_dir.resolve()
        if args.manifests_dir
        else repo_root / "ops" / "checkpoints"
    )
    definition_kind, definition_path, buckets = load_bucket_definitions(
        definition_source=args.definition_source,
        checkpoints_json=checkpoints_json,
        manifests_dir=manifests_dir,
    )
    report = build_report(
        repo_root=repo_root,
        definition_kind=definition_kind,
        definition_path=definition_path,
        buckets=buckets,
    )
    if args.write_pathspec_dir:
        write_exact_pathspecs(report, args.write_pathspec_dir.resolve())
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(
            emit_text(
                report,
                show_paths=args.show_paths,
                max_paths_per_bucket=args.max_paths_per_bucket,
            )
        )
    if args.fail_on_unmatched and report["unmatchedChangedPaths"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
