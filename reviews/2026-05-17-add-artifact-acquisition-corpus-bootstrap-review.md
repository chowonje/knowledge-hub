# Review: Add artifact acquisition corpus bootstrap

## Findings

- The helper is hidden and explicitly selected through `khub paper corpus-bootstrap`; root/public help is not widened.
- Dry-run is the default, and network acquisition requires both `--apply` and `--allow-network`.
- The helper writes only a selected verified file into `papers_dir`; it does not write SQLite rows, attach paper source paths, rebuild derivatives, or invoke `repair-source`.
- Existing local hash mismatches are blocked and not overwritten automatically.
- Downloaded hash or byte-length mismatches are blocked and the temporary download is removed instead of being promoted.
- `repo_fixture` entries are skipped as repo/CI-owned artifacts; local corpus resolution stays under `papers_dir`.
- Result payloads use safe path refs such as `papers_dir/<filename>` and include repair dry-run hints rather than performing repair.

## Risks

- The helper trusts manifest `provenanceUrl` ownership; future manifest updates should review URL legitimacy and licensing before acquisition.
- `--all --apply --allow-network` can download many local-corpus artifacts, so operators should prefer explicit artifact/source selectors.
- The command is hidden/operator-only. If it becomes public later, public help, docs, MCP scope, and schema registration should be reviewed together.
- Successful acquisition only proves the local file matches the manifest; strict answerability still depends on downstream source hash/span propagation and `repair-source`/materialization steps.

## Missing Tests

- Focused tests cover dry-run planning without network, explicit network gating, successful hash-verified promotion, downloaded mismatch cleanup, existing mismatch no-overwrite behavior, repo-fixture skipping, repo-vs-local tier isolation, multi-candidate fail-closed behavior, and hidden CLI invocation.
- Focused repair/live-compare regression, changed Python `py_compile`, release smoke, public release hygiene, full pytest, and diff hygiene passed after docs/record updates.
