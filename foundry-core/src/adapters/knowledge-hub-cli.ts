/*
Authority contract:
- Python owns final validation, policy gating, normalized payload acceptance,
  and product-facing outputs.
- TypeScript owns delegated orchestration and `.khub/personal-foundry/*` state.
- This adapter stays subprocess + JSON stdout only. Inner TS->Python bridge calls
  are single-shot; retry ownership stays with the outer Python bridge.
*/
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const KHUB_CLI_MODULE = "knowledge_hub.interfaces.cli.main";
const MODULE_DIR = dirname(fileURLToPath(import.meta.url));
const KNOWLEDGE_HUB_REPO_ROOT = resolve(MODULE_DIR, "../../..");

export function resolveKnowledgeHubCliEntrypoint(projectRoot: string): string[] {
  const legacyCliPath = resolve(projectRoot, "cli.py");
  if (existsSync(legacyCliPath)) {
    return [legacyCliPath];
  }
  return ["-m", KHUB_CLI_MODULE];
}

export function resolveKnowledgeHubCliWorkingDirectory(): string {
  return KNOWLEDGE_HUB_REPO_ROOT;
}

export function runKnowledgeHubCli(projectRoot: string, pythonPath: string, args: string[]): string {
  return execFileSync(
    pythonPath,
    [...resolveKnowledgeHubCliEntrypoint(projectRoot), ...args],
    {
      // Keep module execution anchored to the product repo root so `python -m`
      // works even when the Node bridge itself is launched from `foundry-core/`.
      cwd: resolveKnowledgeHubCliWorkingDirectory(),
      encoding: "utf8",
    }
  );
}
