import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { KnowledgeHubCLICommand } from "../src/adapters/knowledge-hub-connector";
import {
  KHUB_CLI_MODULE,
  resolveKnowledgeHubCliEntrypoint,
  resolveKnowledgeHubCliWorkingDirectory,
} from "../src/adapters/knowledge-hub-cli";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test("knowledge hub CLI adapter defaults to canonical module entrypoint", () => {
  const entrypoint = resolveKnowledgeHubCliEntrypoint("/tmp/nonexistent-project-root");
  assert.deepEqual(entrypoint, ["-m", KHUB_CLI_MODULE]);
  assert.equal(KHUB_CLI_MODULE, "knowledge_hub.interfaces.cli.main");
});

test("knowledge hub CLI adapter runs module entrypoint from the product repo root", () => {
  const expectedRepoRoot = path.resolve(__dirname, "..", "..");
  assert.equal(resolveKnowledgeHubCliWorkingDirectory(), expectedRepoRoot);
});

test("knowledge hub CLI command uses the resolved canonical entrypoint by default", async () => {
  let captured: { cmd: string; args: string[] } | null = null;
  const executor = new KnowledgeHubCLICommand(
    { projectRoot: "/tmp/nonexistent-project-root" },
    async (cmd, args) => {
      captured = { cmd, args };
      return "ok";
    }
  );

  await executor.run("khub", ["search", "rag"]);

  assert.deepEqual(captured, {
    cmd: "python",
    args: ["-m", KHUB_CLI_MODULE, "search", "rag"],
  });
});
