import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  AGENT_RUN_PLAYBOOK_SCHEMA,
  buildAgentPlaybook,
  normalizeFoundryRunEnvelope,
} from "../src/cli-agent.js";

describe("cli-agent contract", () => {
  it("uses the canonical playbook schema and includes task context for coding goals", () => {
    const { planTools, playbook } = buildAgentPlaybook(
      "Implement task context for the agent runtime",
      "planner",
      "adaptive",
      2,
      {
        repoPath: "/tmp/repo",
        includeWorkspace: true,
        maxWorkspaceFiles: 4,
      }
    );

    assert.equal(playbook.schema, AGENT_RUN_PLAYBOOK_SCHEMA);
    assert.deepEqual(planTools, ["build_task_context", "ask_knowledge"]);
    assert.equal(playbook.steps[0]?.tool, "build_task_context");
  });

  it("redacts blocked artifacts during local envelope normalization", () => {
    const { playbook } = buildAgentPlaybook("Summarize recent notes", "planner", "adaptive", 2);
    const normalized = normalizeFoundryRunEnvelope({
      schema: "knowledge-hub.foundry.agent.run.result.v1",
      source: "foundry-core/cli-agent",
      runId: "run_001",
      status: "completed",
      goal: "Summarize recent notes",
      role: "planner",
      orchestratorMode: "adaptive",
      stage: "DONE",
      plan: ["ask_knowledge"],
      playbook,
      transitions: [],
      verify: {
        allowed: true,
        schemaValid: true,
        policyAllowed: false,
        schemaErrors: [],
      },
      artifact: {
        jsonContent: { email: "alice@example.com" },
        classification: "P0",
        generatedAt: new Date().toISOString(),
        metadata: { trace: "private@example.com" },
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      dryRun: false,
    } as any);

    assert.equal(normalized.status, "blocked");
    assert.equal(normalized.stage, "VERIFY");
    assert.equal(normalized.verify?.allowed, false);
    assert.equal(normalized.verify?.policyAllowed, false);
    assert.equal(normalized.artifact?.jsonContent, "[REDACTED_BY_POLICY]");
    assert.deepEqual(normalized.artifact?.metadata, {});
  });
});
