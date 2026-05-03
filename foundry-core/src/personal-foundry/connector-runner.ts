import crypto from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { isMoreSensitive } from "../types.js";
import type { DataClassification } from "../types.js";
import type {
  ConnectorCursorStore,
  ConnectorIdempotencyStore,
  ConnectorRunInput,
  ConnectorRunResult,
  IdempotencyEntry,
  PersonalAuditLog,
  PersonalEventBus,
  PersonalOntologyStore,
  PersonalPolicyEngine,
  SourceCursor,
} from "./interfaces.js";

interface CursorIndex {
  [key: string]: SourceCursor;
}

interface IdempotencyIndex {
  [key: string]: IdempotencyEntry;
}

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const raw = await readFile(path, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2), "utf8");
}

function cursorKey(connectorId: string, source?: string): string {
  return `${connectorId}::${source ?? "default"}`;
}

function idempotencyKey(connectorId: string, requestId: string): string {
  return `${connectorId}::${requestId}`;
}

export class JsonFileCursorStore implements ConnectorCursorStore {
  private readonly path: string;

  constructor(path: string) {
    this.path = path;
  }

  async get(connectorId: string, source?: string): Promise<SourceCursor | null> {
    const state = await readJson<CursorIndex>(this.path, {});
    return state[cursorKey(connectorId, source)] ?? null;
  }

  async set(entry: SourceCursor): Promise<void> {
    const state = await readJson<CursorIndex>(this.path, {});
    state[cursorKey(entry.connectorId, entry.source)] = entry;
    await writeJson(this.path, state);
  }
}

export class JsonFileIdempotencyStore implements ConnectorIdempotencyStore {
  private readonly path: string;
  private readonly now: () => string;

  constructor(input: { path: string; now?: () => string }) {
    this.path = input.path;
    this.now = input.now ?? (() => new Date().toISOString());
  }

  async isFreshRun(connectorId: string, requestId: string): Promise<boolean> {
    const state = await readJson<IdempotencyIndex>(this.path, {});
    const entry = state[idempotencyKey(connectorId, requestId)];
    return entry?.status !== "started" && entry?.status !== "completed";
  }

  async markStarted(connectorId: string, requestId: string): Promise<string> {
    const state = await readJson<IdempotencyIndex>(this.path, {});
    const key = idempotencyKey(connectorId, requestId);
    const existing = state[key];
    if (existing?.status === "started" || existing?.status === "completed") {
      return existing.runId;
    }

    const runId = `run_${crypto.randomUUID()}`;
    state[key] = {
      connectorId,
      requestId,
      runId,
      status: "started",
      updatedAt: this.now(),
    };
    await writeJson(this.path, state);
    return runId;
  }

  async markCompleted(connectorId: string, requestId: string, runId: string): Promise<void> {
    const state = await readJson<IdempotencyIndex>(this.path, {});
    state[idempotencyKey(connectorId, requestId)] = {
      connectorId,
      requestId,
      runId,
      status: "completed",
      updatedAt: this.now(),
    };
    await writeJson(this.path, state);
  }

  async markFailed(connectorId: string, requestId: string, runId: string): Promise<void> {
    const state = await readJson<IdempotencyIndex>(this.path, {});
    state[idempotencyKey(connectorId, requestId)] = {
      connectorId,
      requestId,
      runId,
      status: "failed",
      updatedAt: this.now(),
    };
    await writeJson(this.path, state);
  }
}

export interface ConnectorRunnerDependencies {
  eventBus: PersonalEventBus;
  audit: PersonalAuditLog;
  policy: PersonalPolicyEngine;
  ontologyStore?: PersonalOntologyStore;
  cursorStore: ConnectorCursorStore;
  idempotencyStore: ConnectorIdempotencyStore;
  now?: () => string;
  classificationGate?: DataClassification;
}

function validateClassificationGate(input: {
  gate: DataClassification;
  mapped: {
    entities: Array<{ classification: DataClassification }>;
    relations: Array<{ classification: DataClassification }>;
    events: Array<{ classification: DataClassification }>;
  };
}): void {
  const records = [
    ...input.mapped.entities,
    ...input.mapped.relations,
    ...input.mapped.events,
  ];

  for (const record of records) {
    if (isMoreSensitive(record.classification, input.gate)) {
      throw new Error(`classification gate blocked (${record.classification} > ${input.gate})`);
    }
  }
}

export class DefaultConnectorRunner {
  private readonly eventBus: PersonalEventBus;
  private readonly audit: PersonalAuditLog;
  private readonly policy: PersonalPolicyEngine;
  private readonly ontologyStore?: PersonalOntologyStore;
  private readonly cursorStore: ConnectorCursorStore;
  private readonly idempotencyStore: ConnectorIdempotencyStore;
  private readonly now: () => string;
  private readonly classificationGate: DataClassification;

  constructor(deps: ConnectorRunnerDependencies) {
    this.eventBus = deps.eventBus;
    this.audit = deps.audit;
    this.policy = deps.policy;
    this.ontologyStore = deps.ontologyStore;
    this.cursorStore = deps.cursorStore;
    this.idempotencyStore = deps.idempotencyStore;
    this.now = deps.now ?? (() => new Date().toISOString());
    this.classificationGate = deps.classificationGate ?? "P1";
  }

  async run(input: ConnectorRunInput): Promise<ConnectorRunResult> {
    const connector = input.connector;
    const fresh = await this.idempotencyStore.isFreshRun(connector.id, input.requestId);
    const runId = await this.idempotencyStore.markStarted(connector.id, input.requestId);

    if (!fresh) {
      await this.audit.append({
        actorId: input.actorId,
        action: "connector_sync",
        resourceType: "connector",
        resourceId: connector.id,
        allowed: true,
        reason: "deduped request",
        classification: "P2",
        requestId: input.requestId,
        runId,
      });

      return {
        connectorId: connector.id,
        runId,
        requestId: input.requestId,
        status: "deduped",
        emittedEventCount: 0,
        emittedEventIds: [],
      };
    }

    try {
      const cursorState = await this.cursorStore.get(connector.id, input.source);
      const cursor = input.cursor ?? cursorState?.cursor;

      const syncResult = await connector.sync({
        actorId: input.actorId,
        requestId: input.requestId,
        source: input.source,
        cursor,
        pageLimit: input.pageLimit,
        dryRun: input.dryRun,
      });

      const mapped = await connector.mapToOntology(syncResult.records, {
        actorId: input.actorId,
        requestId: input.requestId,
        connectorId: connector.id,
        connectorRunId: syncResult.connectorRunId,
        now: this.now(),
        source: input.source,
      });

      validateClassificationGate({
        gate: this.classificationGate,
        mapped,
      });

      if (this.ontologyStore) {
        await this.ontologyStore.appendBatch(mapped);
      }

      const policy = await this.policy.evaluate({
        actorId: input.actorId,
        action: "connector_sync",
        resourceType: "connector",
        resourceId: connector.id,
        payload: mapped,
      });

      await this.audit.append({
        actorId: input.actorId,
        action: "connector_sync",
        resourceType: "connector",
        resourceId: connector.id,
        allowed: policy.allowed,
        reason: policy.reason,
        classification: policy.classification,
        runId,
        requestId: input.requestId,
      });

      if (!policy.allowed) {
        await this.idempotencyStore.markFailed(connector.id, input.requestId, runId);
        return {
          connectorId: connector.id,
          runId,
          requestId: input.requestId,
          status: "failed",
          emittedEventCount: 0,
          emittedEventIds: [],
          errorMessage: policy.reason,
        };
      }

      const emitted = await connector.emitEvents(
        {
          actorId: input.actorId,
          requestId: input.requestId,
          connectorRunId: syncResult.connectorRunId,
          mapped,
          sourceCursor: {
            cursor,
            nextCursor: syncResult.nextCursor,
            hasMore: syncResult.hasMore,
          },
        },
        {
          now: this.now,
          eventBus: this.eventBus,
          audit: this.audit,
        }
      );

      await this.cursorStore.set({
        connectorId: connector.id,
        source: input.source,
        cursor: syncResult.nextCursor,
        updatedAt: this.now(),
        hasMore: syncResult.hasMore,
      });

      await this.idempotencyStore.markCompleted(connector.id, input.requestId, runId);

      return {
        connectorId: connector.id,
        runId,
        requestId: input.requestId,
        status: "done",
        emittedEventCount: emitted.eventIds.length,
        emittedEventIds: emitted.eventIds,
        nextCursor: syncResult.nextCursor,
        hasMore: syncResult.hasMore,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "connector run failed";
      await this.idempotencyStore.markFailed(connector.id, input.requestId, runId);

      await this.audit.append({
        actorId: input.actorId,
        action: "connector_sync",
        resourceType: "connector",
        resourceId: connector.id,
        allowed: false,
        reason: message,
        classification: "P1",
        runId,
        requestId: input.requestId,
      });

      return {
        connectorId: connector.id,
        runId,
        requestId: input.requestId,
        status: "failed",
        emittedEventCount: 0,
        emittedEventIds: [],
        errorMessage: message,
      };
    }
  }
}
