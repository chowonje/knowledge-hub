import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { JsonlAuditLog } from "../src/personal-foundry/audit-log.js";
import { PlanActVerifyRuntime } from "../src/personal-foundry/agent-runtime.js";
import {
  DefaultConnectorRunner,
  JsonFileCursorStore,
  JsonFileIdempotencyStore,
} from "../src/personal-foundry/connector-runner.js";
import { emitOntologyBatchToEventBus } from "../src/personal-foundry/emitters.js";
import { JsonlEventBus } from "../src/personal-foundry/event-bus.js";
import { LocalPolicyEngine } from "../src/personal-foundry/policy-engine.js";
import type {
  ConnectorAuthInput,
  ConnectorAuthResult,
  ConnectorEmitInput,
  ConnectorEmitOutput,
  ConnectorMapContext,
  ConnectorRecord,
  ConnectorSDK,
  ConnectorSyncInput,
  ConnectorSyncOutput,
  OntologyBatch,
} from "../src/personal-foundry/interfaces.js";

function createNow() {
  return () => new Date().toISOString();
}

describe("LocalPolicyEngine", () => {
  it("denies outbound P0 payload and allows sanitized P2 payload", async () => {
    const engine = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const deny = await engine.evaluate({
      actorId: "u1",
      action: "external_llm_call",
      resourceType: "llm",
      resourceId: "provider-x",
      payload: { email: "alice@example.com", password: "secret" },
      outbound: true,
    });

    assert.equal(deny.allowed, false);
    assert.equal(deny.classification, "P0");

    const allow = await engine.evaluate({
      actorId: "u1",
      action: "external_llm_call",
      resourceType: "llm",
      resourceId: "provider-x",
      payload: { summary: "weekly focus report" },
      outbound: true,
    });

    assert.equal(allow.allowed, true);
    assert.equal(allow.classification, "P2");
    assert.equal(allow.requiresSanitization, true);
  });

  it("treats declared classification as advisory when payload is more sensitive", async () => {
    const engine = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const decision = await engine.evaluate({
      actorId: "u1",
      action: "agent_writeback",
      resourceType: "artifact",
      resourceId: "artifact-1",
      payload: { answer: "contact private@example.com" },
      classification: "P2",
    });

    assert.equal(decision.allowed, false);
    assert.equal(decision.classification, "P0");
    assert.equal(decision.policyCode, "WRITEBACK_CLASSIFICATION_DENY");
  });
});

describe("DefaultConnectorRunner", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "khub-personal-foundry-"));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("supports idempotency and incremental cursor", async () => {
    const now = createNow();
    const eventBus = new JsonlEventBus({ logPath: path.join(tmpDir, "events.jsonl"), now });
    const audit = new JsonlAuditLog({ logPath: path.join(tmpDir, "audit.jsonl"), now });
    const policy = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const cursorStore = new JsonFileCursorStore(path.join(tmpDir, "cursor.json"));
    const idempotencyStore = new JsonFileIdempotencyStore({ path: path.join(tmpDir, "idempotency.json"), now });
    let ontologyAppendCalls = 0;

    const seenCursors: Array<string | undefined> = [];

    const connector: ConnectorSDK = {
      id: "mock-khub",
      version: "0.0.1",
      sourceSystem: "knowledge_hub",
      supportsIncrementalSync: true,
      async authorize(_input: ConnectorAuthInput): Promise<ConnectorAuthResult> {
        return {
          credentialId: "cred_1",
          accountId: "acc_1",
          scopes: ["read"],
          issuedAt: now(),
        };
      },
      async sync(input: ConnectorSyncInput): Promise<ConnectorSyncOutput> {
        seenCursors.push(input.cursor);
        return {
          connectorRunId: `crun_${Date.now()}`,
          records: [
            {
              sourceRecordId: "record-1",
              sourceUpdatedAt: now(),
              payload: { title: "Test Note" },
              classification: "P1",
            },
          ],
          cursor: input.cursor,
          nextCursor: "cursor-next",
          hasMore: false,
          extractedAt: now(),
        };
      },
      async mapToOntology(records: ConnectorRecord[], context: ConnectorMapContext): Promise<OntologyBatch> {
        return {
          entities: records.map((record) => ({
            id: `ent-${record.sourceRecordId}`,
            type: "KnowledgeItem",
            properties: record.payload as Record<string, unknown>,
            classification: "P1",
            sourceSystem: "knowledge_hub",
            updatedAt: context.now,
          })),
          relations: [],
          events: [
            {
              aggregateId: "ent-record-1",
              aggregateType: "KnowledgeItem",
              type: "DocumentIngested",
              payload: { sourceRecordId: "record-1" },
              classification: "P1",
              sourceSystem: "knowledge_hub",
              occurredAt: context.now,
            },
          ],
        };
      },
      async emitEvents(input: ConnectorEmitInput): Promise<ConnectorEmitOutput> {
        return emitOntologyBatchToEventBus(input, eventBus, now);
      },
    };

    const runner = new DefaultConnectorRunner({
      eventBus,
      audit,
      policy,
      ontologyStore: {
        async appendBatch() {
          ontologyAppendCalls += 1;
          return {
            eventIds: [],
            entityCount: 0,
            relationCount: 0,
            eventCount: 0,
            timeSeriesCount: 0,
          };
        },
        async readEntity() { return null; },
        async readEvents() { return []; },
        async appendSnapshot(input) { return { id: "snap_1", ...input }; },
        async readLatestSnapshot() { return null; },
        async appendTimeSeries() {},
        async getTimeSeries() { return []; },
      },
      cursorStore,
      idempotencyStore,
      now,
      classificationGate: "P1",
    });

    const first = await runner.run({
      connector,
      actorId: "user-1",
      requestId: "req-1",
      source: "note",
    });

    assert.equal(first.status, "done");
    assert.equal(first.emittedEventCount > 0, true);
    assert.equal(seenCursors[0], undefined);

    const duplicate = await runner.run({
      connector,
      actorId: "user-1",
      requestId: "req-1",
      source: "note",
    });

    assert.equal(duplicate.status, "deduped");

    const second = await runner.run({
      connector,
      actorId: "user-1",
      requestId: "req-2",
      source: "note",
    });

    assert.equal(second.status, "done");
    assert.equal(seenCursors[1], "cursor-next");
    assert.equal(ontologyAppendCalls, 2);
  });
});

describe("PlanActVerifyRuntime", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "khub-agent-runtime-"));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("runs plan-act-verify-writeback successfully", async () => {
    const now = createNow();
    const audit = new JsonlAuditLog({ logPath: path.join(tmpDir, "audit-ok.jsonl"), now });
    const policy = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const runtime = new PlanActVerifyRuntime({ policy, audit, now });

    const result = await runtime.run({
      actorId: "user-1",
      requestId: "request-ok",
      goal: "summarize weekly knowledge",
      maxRounds: 1,
      planner: async () => [
        {
          order: 1,
          toolName: "ask_knowledge",
          objective: "compose answer",
          input: { question: "weekly summary" },
        },
      ],
      tools: {
        ask_knowledge: {
          name: "ask_knowledge",
          async execute() {
            return {
              artifact: {
                jsonContent: { summary: "ok" },
                classification: "P2",
                generatedAt: now(),
              },
            };
          },
        },
      },
      schemaValidate: async (artifact) => {
        const obj = artifact as Record<string, unknown>;
        return {
          ok: typeof obj?.summary === "string",
          errors: typeof obj?.summary === "string" ? [] : ["missing summary"],
        };
      },
      writeback: async () => ({ ok: true, detail: "stored" }),
    });

    assert.equal(result.status, "completed");
    assert.equal(result.stage, "DONE");
    assert.equal(result.verify?.schemaValid, true);
    assert.equal(result.writeback?.ok, true);
  });

  it("blocks P0 artifact on writeback policy", async () => {
    const now = createNow();
    const audit = new JsonlAuditLog({ logPath: path.join(tmpDir, "audit-blocked.jsonl"), now });
    const policy = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const runtime = new PlanActVerifyRuntime({ policy, audit, now });

    const result = await runtime.run({
      actorId: "user-1",
      requestId: "request-block",
      goal: "attempt writeback with p0",
      maxRounds: 1,
      planner: async () => [
        {
          order: 1,
          toolName: "ask_knowledge",
          objective: "compose answer",
          input: { question: "contains sensitive" },
        },
      ],
      tools: {
        ask_knowledge: {
          name: "ask_knowledge",
          async execute() {
            return {
              artifact: {
                jsonContent: { email: "alice@example.com" },
                classification: "P0",
                generatedAt: now(),
              },
            };
          },
        },
      },
      schemaValidate: async () => ({ ok: true, errors: [] }),
      writeback: async () => ({ ok: true, detail: "should not happen" }),
    });

    assert.equal(result.status, "blocked");
    assert.equal(result.verify?.policyAllowed, false);
    assert.equal(result.artifact?.classification, "P0");
    assert.equal(result.artifact?.jsonContent, "[REDACTED_BY_POLICY]");
    assert.equal(result.transitions.some((transition) => transition.stage === "VERIFY" && transition.status === "blocked"), true);
  });

  it("reclassifies sensitive artifacts even when tools declare P2", async () => {
    const now = createNow();
    const audit = new JsonlAuditLog({ logPath: path.join(tmpDir, "audit-advisory.jsonl"), now });
    const policy = new LocalPolicyEngine({
      outboundMaxClassification: "P2",
      writebackMaxClassification: "P1",
    });

    const runtime = new PlanActVerifyRuntime({ policy, audit, now });

    const result = await runtime.run({
      actorId: "user-1",
      requestId: "request-advisory",
      goal: "attempt writeback with mislabeled p0",
      maxRounds: 1,
      planner: async () => [
        {
          order: 1,
          toolName: "ask_knowledge",
          objective: "compose answer",
          input: { question: "contains sensitive" },
        },
      ],
      tools: {
        ask_knowledge: {
          name: "ask_knowledge",
          async execute() {
            return {
              artifact: {
                jsonContent: { answer: "contact private@example.com" },
                classification: "P2",
                generatedAt: now(),
              },
            };
          },
        },
      },
      schemaValidate: async () => ({ ok: true, errors: [] }),
      writeback: async () => ({ ok: true, detail: "should not happen" }),
    });

    assert.equal(result.status, "blocked");
    assert.equal(result.verify?.policyAllowed, false);
    assert.equal(result.artifact?.classification, "P0");
    assert.equal(result.artifact?.jsonContent, "[REDACTED_BY_POLICY]");
    assert.equal(result.transitions.some((transition) => transition.stage === "VERIFY" && transition.status === "blocked"), true);
  });
});
