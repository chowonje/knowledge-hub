import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

import {
  runConnectorSync,
  emitOntologyBatchToBus,
  createConnectorRegistry,
  InMemorySyncStateStore,
  InMemoryIdempotencyStore,
  FileSyncStateStore,
  FileIdempotencyStore,
} from "../src/connector-sdk.js";

import type {
  ConnectorContract,
  ConnectorSyncInput,
  ConnectorSyncResult,
  ConnectorMappingContext,
  ConnectorEmitInput,
  ConnectorEmitResult,
  ConnectorAuthInput,
  ConnectorAuthResult,
  ConnectorRuntimeContext,
  DataRecord,
  MappedOntologyBatch,
  FoundryEventBus,
  EventBusEvent,
} from "../src/types.js";

// ── Test fixtures ──

function createMockConnector(id = "test-connector"): ConnectorContract {
  return {
    id,
    version: "1.0.0",
    name: "Test Connector",
    sourceSystem: "test",
    supportedScopes: ["read"],
    supportsIncrementalSync: true,

    async authorize(_input: ConnectorAuthInput): Promise<ConnectorAuthResult> {
      return {
        credentialId: "cred_1",
        accountId: "acc_1",
        scopes: ["read"],
        issuedAt: new Date().toISOString(),
        tokenFingerprint: "fp_test",
      };
    },

    async sync(_input: ConnectorSyncInput): Promise<ConnectorSyncResult> {
      return {
        runId: "sync_run_1",
        connectorRunId: `crun_${crypto.randomUUID()}`,
        rawRecords: [
          {
            sourceRecordId: "rec_1",
            sourceUpdatedAt: new Date().toISOString(),
            payload: { title: "Test Paper", field: "AI" },
            classification: "P2",
          },
        ],
        nextCursor: "cursor_2",
        hasMore: false,
        extractedAt: new Date().toISOString(),
        metric: { scanned: 1, returned: 1, retries: 0 },
      };
    },

    async mapToOntology(
      rawRecords: DataRecord[],
      context: ConnectorMappingContext
    ): Promise<MappedOntologyBatch> {
      return {
        entities: rawRecords.map((r, i) => ({
          entityType: "Paper",
          entityId: `paper_${i}`,
          properties: r.payload as Record<string, unknown>,
          classification: r.classification,
          schemaVersion: "1.0",
          actorId: context.actorId,
        })),
        relations: [],
        events: [
          {
            eventType: "PaperIngested",
            aggregateId: "paper_0",
            aggregateType: "Paper",
            payload: { source: "test" },
            occurredAt: new Date().toISOString(),
            classification: "P2",
            schemaVersion: "1.0",
          },
        ],
      };
    },

    async emitEvents(
      input: ConnectorEmitInput,
      runtime: ConnectorRuntimeContext
    ): Promise<ConnectorEmitResult> {
      return emitOntologyBatchToBus(input, runtime);
    },
  };
}

function createTestRuntime(overrides: Partial<ConnectorRuntimeContext> = {}): ConnectorRuntimeContext {
  const published: EventBusEvent[] = [];
  const audits: unknown[] = [];

  const bus: FoundryEventBus = {
    async publish<T>(event: EventBusEvent<T>) {
      published.push(event as EventBusEvent);
    },
    subscribe() {},
  };

  return {
    now: () => new Date().toISOString(),
    syncStateStore: new InMemorySyncStateStore(),
    idempotencyStore: new InMemoryIdempotencyStore(),
    bus,
    audit: async (evt) => { audits.push(evt); },
    classificationGate: "P2",
    ...overrides,
    _published: published,
    _audits: audits,
  } as ConnectorRuntimeContext & { _published: EventBusEvent[]; _audits: unknown[] };
}

// ── Tests ──

describe("InMemoryIdempotencyStore", () => {
  it("first run is fresh", async () => {
    const store = new InMemoryIdempotencyStore();
    const fresh = await store.isFreshRun("c1", "r1");
    assert.equal(fresh, true);
  });

  it("started run is not fresh", async () => {
    const store = new InMemoryIdempotencyStore();
    await store.markStarted("c1", "r1");
    const fresh = await store.isFreshRun("c1", "r1");
    assert.equal(fresh, false);
  });

  it("completed run is not fresh", async () => {
    const store = new InMemoryIdempotencyStore();
    const runId = await store.markStarted("c1", "r1");
    await store.markCompleted("c1", "r1", runId);
    const fresh = await store.isFreshRun("c1", "r1");
    assert.equal(fresh, false);
  });

  it("failed run is fresh (can retry)", async () => {
    const store = new InMemoryIdempotencyStore();
    const runId = await store.markStarted("c1", "r1");
    await store.markFailed("c1", "r1", runId);
    const fresh = await store.isFreshRun("c1", "r1");
    assert.equal(fresh, true);
  });

  it("markStarted returns existing runId for duplicate", async () => {
    const store = new InMemoryIdempotencyStore();
    const runId1 = await store.markStarted("c1", "r1");
    const runId2 = await store.markStarted("c1", "r1");
    assert.equal(runId1, runId2);
  });
});

describe("FileSyncStateStore", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "khub-test-"));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("returns null for unknown connector", async () => {
    const store = new FileSyncStateStore(tmpDir);
    const state = await store.get("nonexistent");
    assert.equal(state, null);
  });

  it("persists and retrieves cursor", async () => {
    const store = new FileSyncStateStore(tmpDir);
    await store.upsert("c1", { cursor: "page_5", updatedAt: "2026-01-01T00:00:00Z" });

    const store2 = new FileSyncStateStore(tmpDir);
    const state = await store2.get("c1");
    assert.equal(state?.cursor, "page_5");
  });
});

describe("FileIdempotencyStore", () => {
  let tmpDir: string;

  before(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "khub-test-idemp-"));
  });

  after(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("persists across new instances", async () => {
    const store1 = new FileIdempotencyStore(tmpDir);
    assert.equal(await store1.isFreshRun("c1", "r1"), true);
    await store1.markStarted("c1", "r1");

    const store2 = new FileIdempotencyStore(tmpDir);
    assert.equal(await store2.isFreshRun("c1", "r1"), false);
  });
});

describe("runConnectorSync", () => {
  it("completes a fresh sync successfully", async () => {
    const connector = createMockConnector();
    const registry = createConnectorRegistry([connector]);
    const runtime = createTestRuntime();

    const result = await runConnectorSync({
      connector,
      input: { connectorId: connector.id, actorId: "user_1" },
      runtime,
      registry,
    });

    assert.equal(result.status, "done");
    assert.equal(result.deduplicated, false);
    assert.ok(result.emittedEventCount > 0);
    assert.ok(result.emittedEventIds.length > 0);
    assert.equal(result.nextCursor, "cursor_2");
    assert.equal(result.hasMore, false);
  });

  it("deduplicates a repeated requestId", async () => {
    const connector = createMockConnector();
    const registry = createConnectorRegistry([connector]);
    const runtime = createTestRuntime();
    const requestId = crypto.randomUUID();

    const result1 = await runConnectorSync({
      connector,
      input: { connectorId: connector.id, actorId: "user_1", requestId },
      runtime,
      registry,
    });
    assert.equal(result1.status, "done");

    const result2 = await runConnectorSync({
      connector,
      input: { connectorId: connector.id, actorId: "user_1", requestId },
      runtime,
      registry,
    });
    assert.equal(result2.status, "deduped");
    assert.equal(result2.deduplicated, true);
    assert.equal(result2.emittedEventCount, 0);
  });

  it("returns failed for unknown connector", async () => {
    const connector = createMockConnector();
    const registry = createConnectorRegistry([connector]);
    const runtime = createTestRuntime();

    const result = await runConnectorSync({
      connector,
      input: { connectorId: "nonexistent", actorId: "user_1" },
      runtime,
      registry,
    });

    assert.equal(result.status, "failed");
    assert.ok(result.errorMessage?.includes("not found"));
  });

  it("marks failed on sync error", async () => {
    const failing = createMockConnector("fail-connector");
    (failing as any).sync = async () => { throw new Error("network timeout"); };
    const registry = createConnectorRegistry([failing]);
    const runtime = createTestRuntime();

    const result = await runConnectorSync({
      connector: failing,
      input: { connectorId: failing.id, actorId: "user_1" },
      runtime,
      registry,
    });

    assert.equal(result.status, "failed");
    assert.ok(result.errorMessage?.includes("network timeout"));
  });
});

describe("emitOntologyBatchToBus", () => {
  it("emits entities and events to the bus", async () => {
    const runtime = createTestRuntime();

    const mapped: MappedOntologyBatch = {
      entities: [
        {
          entityType: "Paper",
          entityId: "p1",
          properties: { title: "Test" },
          classification: "P2",
          schemaVersion: "1.0",
        },
      ],
      relations: [],
      events: [
        {
          eventType: "PaperCreated",
          aggregateId: "p1",
          aggregateType: "Paper",
          payload: {},
          occurredAt: new Date().toISOString(),
          classification: "P2",
          schemaVersion: "1.0",
        },
      ],
    };

    const result = await emitOntologyBatchToBus(
      { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
      runtime
    );

    assert.equal(result.eventIds.length, 2);
    assert.ok(result.emittedAt);
  });

  it("generates snapshotId when events are emitted", async () => {
    const runtime = createTestRuntime();

    const mapped: MappedOntologyBatch = {
      entities: [
        {
          entityType: "Paper",
          entityId: "p1",
          properties: {},
          classification: "P2",
          schemaVersion: "1.0",
        },
      ],
      relations: [],
      events: [],
    };

    const result = await emitOntologyBatchToBus(
      { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
      runtime
    );

    assert.ok(result.snapshotId, "snapshotId should be generated");
    assert.ok(result.snapshotId!.startsWith("snap_"));
    assert.ok(result.snapshotVersion);
  });

  it("returns no snapshot for empty batch", async () => {
    const runtime = createTestRuntime();

    const mapped: MappedOntologyBatch = {
      entities: [],
      relations: [],
      events: [],
    };

    const result = await emitOntologyBatchToBus(
      { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
      runtime
    );

    assert.equal(result.eventIds.length, 0);
    assert.equal(result.snapshotId, undefined);
  });

  it("rejects batch exceeding classification gate", async () => {
    const runtime = createTestRuntime({ classificationGate: "P3" });

    const mapped: MappedOntologyBatch = {
      entities: [
        {
          entityType: "Secret",
          entityId: "s1",
          properties: {},
          classification: "P0",
          schemaVersion: "1.0",
        },
      ],
      relations: [],
      events: [],
    };

    await assert.rejects(
      () => emitOntologyBatchToBus(
        { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
        runtime
      ),
      /classification P0 exceeds gate P3/
    );
  });
});

describe("relation evidence tracking", () => {
  it("emits relations with evidence attached to bus payload", async () => {
    const published: EventBusEvent[] = [];
    const bus: FoundryEventBus = {
      async publish<T>(event: EventBusEvent<T>) { published.push(event as EventBusEvent); },
      subscribe() {},
    };
    const runtime = createTestRuntime({ bus });

    const mapped: MappedOntologyBatch = {
      entities: [],
      relations: [
        {
          relationType: "paper_uses_concept",
          sourceEntityId: "paper_1",
          targetEntityId: "concept_42",
          properties: {},
          evidence: [
            {
              text: "We propose a novel attention mechanism",
              chunkId: "chunk_3",
              score: 0.92,
              model: "gpt-4o-mini",
              extractedAt: "2026-02-19T00:00:00Z",
            },
          ],
          confidence: 0.92,
          classification: "P2",
          schemaVersion: "1.0",
        },
      ],
      events: [],
    };

    const result = await emitOntologyBatchToBus(
      { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
      runtime
    );

    assert.equal(result.eventIds.length, 1);
    const relEvent = published.find(e => e.type === "relation:paper_uses_concept");
    assert.ok(relEvent, "relation event should be published");
    const payload = relEvent!.payload as Record<string, unknown>;
    assert.equal(payload["kind"], "relation");
    assert.equal(payload["confidence"], 0.92);
    const evidence = payload["evidence"] as Array<Record<string, unknown>>;
    assert.equal(evidence.length, 1);
    assert.equal(evidence[0]["text"], "We propose a novel attention mechanism");
    assert.equal(evidence[0]["score"], 0.92);
  });

  it("defaults to empty evidence and confidence 1.0 when not provided", async () => {
    const published: EventBusEvent[] = [];
    const bus: FoundryEventBus = {
      async publish<T>(event: EventBusEvent<T>) { published.push(event as EventBusEvent); },
      subscribe() {},
    };
    const runtime = createTestRuntime({ bus });

    const mapped: MappedOntologyBatch = {
      entities: [],
      relations: [
        {
          relationType: "concept_related_to",
          sourceEntityId: "c1",
          targetEntityId: "c2",
          properties: {},
          classification: "P2",
          schemaVersion: "1.0",
        },
      ],
      events: [],
    };

    await emitOntologyBatchToBus(
      { actorId: "u1", requestId: "r1", connectorRunId: "crun_1", mapped },
      runtime
    );

    const payload = published[0].payload as Record<string, unknown>;
    assert.deepEqual(payload["evidence"], []);
    assert.equal(payload["confidence"], 1.0);
  });
});

describe("createConnectorRegistry", () => {
  it("retrieves registered connectors", () => {
    const c = createMockConnector("my-conn");
    const registry = createConnectorRegistry([c]);
    assert.ok(registry.has("my-conn"));
    assert.equal(registry.get("my-conn")?.id, "my-conn");
    assert.equal(registry.has("other"), false);
    assert.equal(registry.get("other"), undefined);
  });
});
