import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  isMoreSensitive,
  ConnectorAuditEvent,
  ConnectorContract,
  ConnectorEmitInput,
  ConnectorEmitResult,
  ConnectorRunInput,
  ConnectorRunResult,
  ConnectorSyncInput,
  ConnectorSyncStateStore,
  ConnectorRuntimeContext,
  ConnectorSyncResult,
  ConnectorIdempotencyStore,
  ConnectorMappingContext,
  DataRecord,
  DataClassification,
  MappedOntologyBatch,
  FoundryEventBus,
  EventBusEvent,
  OntologyEntityDraft,
  OntologyRelationDraft,
  OntologyEventDraft,
  ConnectorRegistry,
  ConnectorContract as Contract,
} from "./types.js";

export interface PythonCommand {
  run(cmd: string, args: string[]): Promise<string>;
}

const DEFAULT_STATE_DIR = path.join(
  process.env["KHUB_STATE_DIR"] ?? path.join(process.env["HOME"] ?? ".", ".khub"),
  "foundry-state",
);

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function readJson<T>(filePath: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
  } catch {
    return null;
  }
}

function writeJson(filePath: string, data: unknown): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
}

// ── Persistent file-backed SyncStateStore ──

export class FileSyncStateStore implements ConnectorSyncStateStore {
  private readonly dir: string;

  constructor(stateDir: string = DEFAULT_STATE_DIR) {
    this.dir = path.join(stateDir, "sync-state");
    ensureDir(this.dir);
  }

  private filePath(connectorId: string): string {
    const safe = connectorId.replace(/[^a-zA-Z0-9_-]/g, "_");
    return path.join(this.dir, `${safe}.json`);
  }

  async get(connectorId: string): Promise<{ cursor?: string; updatedAt?: string } | null> {
    return readJson(this.filePath(connectorId));
  }

  async upsert(connectorId: string, nextState: { cursor?: string; updatedAt: string }): Promise<void> {
    writeJson(this.filePath(connectorId), nextState);
  }
}

// ── Persistent file-backed IdempotencyStore ──

type IdempotencyEntry = { runId: string; status: "started" | "completed" | "failed"; updatedAt: string };
type IdempotencyIndex = Record<string, IdempotencyEntry>;

export class FileIdempotencyStore implements ConnectorIdempotencyStore {
  private readonly filePath: string;
  private cache: IdempotencyIndex;

  constructor(stateDir: string = DEFAULT_STATE_DIR) {
    ensureDir(stateDir);
    this.filePath = path.join(stateDir, "idempotency.json");
    this.cache = readJson<IdempotencyIndex>(this.filePath) ?? {};
  }

  private persist(): void {
    writeJson(this.filePath, this.cache);
  }

  async isFreshRun(connectorId: string, requestId: string): Promise<boolean> {
    const entry = this.cache[`${connectorId}::${requestId}`];
    return entry?.status !== "started" && entry?.status !== "completed";
  }

  async markStarted(connectorId: string, requestId: string): Promise<string> {
    const key = `${connectorId}::${requestId}`;
    const existing = this.cache[key];
    if (existing?.status === "started" || existing?.status === "completed") {
      return existing.runId;
    }
    const runId = `run_${crypto.randomUUID()}`;
    this.cache[key] = { runId, status: "started", updatedAt: new Date().toISOString() };
    this.persist();
    return runId;
  }

  async markCompleted(connectorId: string, requestId: string, runId: string): Promise<void> {
    this.cache[`${connectorId}::${requestId}`] = { runId, status: "completed", updatedAt: new Date().toISOString() };
    this.persist();
  }

  async markFailed(connectorId: string, requestId: string, runId: string): Promise<void> {
    this.cache[`${connectorId}::${requestId}`] = { runId, status: "failed", updatedAt: new Date().toISOString() };
    this.persist();
  }
}

// ── Legacy in-memory stores (kept for tests / ephemeral use) ──

export class InMemorySyncStateStore implements ConnectorSyncStateStore {
  private readonly state = new Map<string, { cursor?: string; updatedAt?: string }>();

  async get(connectorId: string): Promise<{ cursor?: string; updatedAt?: string } | null> {
    return this.state.get(connectorId) ?? null;
  }

  async upsert(connectorId: string, nextState: { cursor?: string; updatedAt: string }): Promise<void> {
    this.state.set(connectorId, nextState);
  }
}

export class InMemoryIdempotencyStore implements ConnectorIdempotencyStore {
  private readonly runs = new Map<
    string,
    { runId: string; status: "started" | "completed" | "failed"; updatedAt: string }
  >();

  async isFreshRun(connectorId: string, requestId: string): Promise<boolean> {
    const entry = this.runs.get(`${connectorId}::${requestId}`);
    return entry?.status !== "started" && entry?.status !== "completed";
  }

  async markStarted(connectorId: string, requestId: string): Promise<string> {
    const key = `${connectorId}::${requestId}`;
    const existing = this.runs.get(key);
    if (existing?.status === "started" || existing?.status === "completed") {
      return existing.runId;
    }

    const runId = `run_${crypto.randomUUID()}`;
    this.runs.set(key, { runId, status: "started", updatedAt: new Date().toISOString() });
    return runId;
  }

  async markCompleted(connectorId: string, requestId: string, runId: string): Promise<void> {
    this.runs.set(`${connectorId}::${requestId}`, {
      runId,
      status: "completed",
      updatedAt: new Date().toISOString(),
    });
  }

  async markFailed(connectorId: string, requestId: string, runId: string): Promise<void> {
    this.runs.set(`${connectorId}::${requestId}`, {
      runId,
      status: "failed",
      updatedAt: new Date().toISOString(),
    });
  }
}

function fallbackRuntime(overrides: Partial<ConnectorRuntimeContext>): ConnectorRuntimeContext {
  const bus: FoundryEventBus = {
    async publish<T>(event: EventBusEvent<T>) {
      void event;
    },
    subscribe() {
      return;
    },
  };

  return {
    now: () => new Date().toISOString(),
    syncStateStore: new FileSyncStateStore(),
    idempotencyStore: new FileIdempotencyStore(),
    bus,
    audit: async () => {},
    classificationGate: "P2",
    ...overrides,
  };
}

function validatePayloadSensitivity(
  mapped: MappedOntologyBatch,
  gate: DataClassification
): void {
  const allDrafts = [
    ...mapped.entities.map((e: OntologyEntityDraft) => ({ classification: e.classification })),
    ...mapped.relations.map((r: OntologyRelationDraft) => ({ classification: r.classification })),
    ...mapped.events.map((e: OntologyEventDraft) => ({ classification: e.classification })),
  ];

  for (const draft of allDrafts) {
    if (isMoreSensitive(draft.classification, gate)) {
      throw new Error(`classification ${draft.classification} exceeds gate ${gate}`);
  }
}
}

export interface ConnectorSyncOptions {
  connector: ConnectorContract;
  input: ConnectorRunInput;
  runtime: ConnectorRuntimeContext;
  registry: ConnectorRegistry;
}

export async function emitOntologyBatchToBus(
  input: ConnectorEmitInput,
  runtime: ConnectorRuntimeContext
): Promise<ConnectorEmitResult> {
  const emittedAt = runtime.now();
  validatePayloadSensitivity(input.mapped, runtime.classificationGate ?? "P2");

  const emitted: string[] = [];
  let idx = 0;

  for (const entity of input.mapped.entities) {
    const eventId = `${input.connectorRunId}.ent.${idx++}`;
    await runtime.bus.publish({
      id: eventId,
      type: `entity:${entity.entityType}`,
      occurredAt: emittedAt,
      actorId: entity.actorId ?? undefined,
      payload: { ...entity, kind: "entity" },
    });
    emitted.push(eventId);
  }

  for (const relation of input.mapped.relations) {
    const eventId = `${input.connectorRunId}.rel.${idx++}`;
    await runtime.bus.publish({
      id: eventId,
      type: `relation:${relation.relationType}`,
      occurredAt: emittedAt,
      actorId: relation.actorId ?? undefined,
      payload: {
        ...relation,
        kind: "relation",
        evidence: relation.evidence ?? [],
        confidence: relation.confidence ?? 1.0,
      },
    });
    emitted.push(eventId);
  }

  for (const event of input.mapped.events) {
    const eventId = `${input.connectorRunId}.evt.${idx++}`;
    await runtime.bus.publish({
      id: eventId,
      type: `event:${event.eventType}`,
      occurredAt: event.occurredAt,
      actorId: event.actorId ?? undefined,
      payload: { ...event, kind: "event" },
    });
    emitted.push(eventId);
  }

  let snapshotId: string | undefined;
  let snapshotVersion: string | undefined;

  if (emitted.length > 0) {
    const hash = crypto.createHash("sha256");
    for (const eid of emitted) hash.update(eid);
    snapshotId = `snap_${hash.digest("hex").slice(0, 16)}`;
    snapshotVersion = `${emitted.length}@${emittedAt}`;
  }

  return {
    eventIds: emitted,
    emittedAt,
    snapshotId,
    snapshotVersion,
  };
}

export async function runConnectorSync(params: ConnectorSyncOptions): Promise<ConnectorRunResult> {
  const runtime = fallbackRuntime(params.runtime);
  const requestId = params.input.requestId ?? crypto.randomUUID();
  const connector = params.registry.get(params.input.connectorId);

  if (!connector) {
    return {
      connectorId: params.input.connectorId,
      requestId,
      runId: `failed_${crypto.randomUUID()}`,
      status: "failed",
      deduplicated: false,
      emittedEventCount: 0,
      emittedEventIds: [],
      errorMessage: `connector not found: ${params.input.connectorId}`,
    };
  }

  const fresh = await runtime.idempotencyStore.isFreshRun(connector.id, requestId);
  const runId = await runtime.idempotencyStore.markStarted(connector.id, requestId);

  if (!fresh) {
    await runtime.audit({
      eventType: "connector.synced",
      connectorId: connector.id,
      actorId: params.input.actorId,
      requestId,
      runId,
      message: `connector sync deduped (${requestId})`,
      severity: "info",
      metadata: { reason: "duplicate_request" },
    });
    return {
      connectorId: connector.id,
      requestId,
      runId,
      status: "deduped",
      deduplicated: true,
      emittedEventCount: 0,
      emittedEventIds: [],
      errorMessage: "duplicate requestId",
    };
  }

  try {
    const previous = await runtime.syncStateStore.get(connector.id);
    const input = params.input as ConnectorSyncInput & { source?: string; since?: string };
    const syncInput: ConnectorSyncInput = {
      actorId: params.input.actorId,
      requestId,
      cursor: input.cursor ?? input.since ?? previous?.cursor,
      includeDeleted: input.includeDeleted,
      pageLimit: input.pageLimit,
      dryRun: input.dryRun,
      source: input.source,
    };

    await runtime.audit({
      eventType: "connector.sync.started",
      connectorId: connector.id,
      actorId: params.input.actorId,
      requestId,
      runId,
      message: `connector sync started (${connector.id})`,
      severity: "info",
      metadata: { cursor: syncInput.cursor },
    });

    const syncResult: ConnectorSyncResult = await connector.sync(syncInput);
    const mappingContext: ConnectorMappingContext = {
      actorId: params.input.actorId,
      requestId,
      connectorId: connector.id,
      connectorRunId: syncResult.connectorRunId,
      runStartedAt: runtime.now(),
    };

    const mapped = await connector.mapToOntology(syncResult.rawRecords as DataRecord[], mappingContext);
    const emitResult = await connector.emitEvents({
      actorId: params.input.actorId,
      requestId,
      connectorRunId: syncResult.connectorRunId,
      mapped,
      sourceCursor: {
        cursor: syncInput.cursor,
        nextCursor: syncResult.nextCursor,
        hasMore: syncResult.hasMore,
      },
      metadata: params.input.metadata,
    }, runtime);

    await runtime.syncStateStore.upsert(connector.id, {
      cursor: syncResult.nextCursor,
      updatedAt: runtime.now(),
    });

    await runtime.audit({
      eventType: "connector.sync.completed",
      connectorId: connector.id,
      actorId: params.input.actorId,
      requestId,
      runId,
      message: `connector sync completed (${emitResult.eventIds.length} events)`,
      severity: "info",
      metadata: {
        emitted: emitResult.eventIds.length,
        hasMore: syncResult.hasMore,
      },
    });

    await runtime.idempotencyStore.markCompleted(connector.id, requestId, runId);
    return {
      connectorId: connector.id,
      requestId,
      runId,
      status: "done",
      deduplicated: false,
      emittedEventCount: emitResult.eventIds.length,
      emittedEventIds: emitResult.eventIds,
      nextCursor: syncResult.nextCursor,
      hasMore: syncResult.hasMore,
    };
  } catch (error) {
    const errorObj = error as Error | unknown;
    const message = errorObj instanceof Error ? errorObj.message : "connector sync failed";
    await runtime.audit({
      eventType: "connector.sync.failed",
      connectorId: connector.id,
      actorId: params.input.actorId,
      requestId,
      runId,
      message,
      severity: "error",
      metadata: {
        error: message,
        errorType: errorObj instanceof Error ? errorObj.name : "UnknownError",
        stack: errorObj instanceof Error ? errorObj.stack : undefined,
      },
    });
    await runtime.idempotencyStore.markFailed(connector.id, requestId, runId);
    return {
      connectorId: connector.id,
      requestId,
      runId,
      status: "failed",
      deduplicated: false,
      emittedEventCount: 0,
      emittedEventIds: [],
      errorMessage: message,
    };
  }
}

export function createConnectorRegistry(connectors: Contract[]): ConnectorRegistry {
  const map = new Map<string, ConnectorContract>();
  for (const connector of connectors) {
    map.set(connector.id, connector);
  }
  return {
    get(connectorId: string) {
      return map.get(connectorId);
    },
    has(connectorId: string) {
      return map.has(connectorId);
    },
  };
}
