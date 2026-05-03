import { randomUUID } from "node:crypto";
import { resolveKnowledgeHubCliEntrypoint } from "./knowledge-hub-cli.js";
import { emitOntologyBatchToBus } from "../connector-sdk.js";
import {
  ConnectorAuthInput,
  ConnectorAuthResult,
  ConnectorContract,
  ConnectorEmitInput,
  ConnectorEmitResult,
  ConnectorRuntimeContext,
  DataClassification,
  DataRecord,
  ConnectorMappingContext,
  ConnectorSyncInput,
  ConnectorSyncResult,
  MappedOntologyBatch,
  OntologyEntityDraft,
  OntologyEventDraft,
  OntologyRelationDraft,
} from "../types.js";

export interface KnowledgeHubConnectorConfig {
  projectRoot: string;
  command?: string;
  pythonPath?: string;
  scopes?: string[];
  connectorId?: string;
  fixedSource?: "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";
}

export interface KnowledgeHubRawRecord {
  id: string;
  source: "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";
  title: string;
  content?: string;
  filePath?: string;
  metadata?: Record<string, unknown>;
  updatedAt?: string;
  tags?: string[];
  classification?: DataClassification;
}

export interface KnowledgeHubOntologyDeltaRaw {
  entities: Array<Record<string, unknown>>;
  relations: Array<Record<string, unknown>>;
  claims: Array<Record<string, unknown>>;
  events: Array<Record<string, unknown>>;
}

function normalizeClassification(row: KnowledgeHubRawRecord & { classification?: string }): DataClassification {
  if (row.classification === "P0" || row.classification === "P1" || row.classification === "P2" || row.classification === "P3") {
    return row.classification;
  }
  if (row.source === "paper" || row.source === "expense" || row.source === "sleep" || row.source === "schedule" || row.source === "behavior") return "P1";
  return "P2";
}

function mapSourceToEntityType(source: KnowledgeHubRawRecord["source"]): string {
  switch (source) {
    case "paper":
      return "Paper";
    case "expense":
      return "Expense";
    case "sleep":
      return "SleepLog";
    case "schedule":
      return "Schedule";
    case "behavior":
      return "BehaviorEvent";
    case "web":
      return "KnowledgeItem";
    case "note":
      return "KnowledgeItem";
    default:
      return "KnowledgeItem";
  }
}

function mapSourceToEventType(source: KnowledgeHubRawRecord["source"]): string {
  switch (source) {
    case "note":
      return "KnowledgeItemIngested";
    case "paper":
      return "PaperIngested";
    case "web":
      return "WebDocumentIngested";
    case "expense":
      return "ExpenseLogged";
    case "sleep":
      return "SleepLogged";
    case "schedule":
      return "ScheduleCreated";
    case "behavior":
      return "BehaviorEventLogged";
    default:
      return "DocumentIngested";
  }
}

function coerceSource(input: unknown): "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior" {
  if (typeof input !== "string") {
    return "all";
  }
  const normalized = input.trim().toLowerCase();
  if (normalized === "notes") {
    return "note";
  }
  if (
    normalized === "note" ||
    normalized === "paper" ||
    normalized === "web" ||
    normalized === "expense" ||
    normalized === "sleep" ||
    normalized === "schedule" ||
    normalized === "behavior"
  ) {
    return normalized;
  }
  return "all";
}

function asString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function asSource(input: unknown): KnowledgeHubRawRecord["source"] | null {
  if (typeof input !== "string") {
    return null;
  }
  const normalized = input.trim().toLowerCase();
  if (normalized === "notes") {
    return "note";
  }
  if (
    normalized === "note" ||
    normalized === "paper" ||
    normalized === "web" ||
    normalized === "expense" ||
    normalized === "sleep" ||
    normalized === "schedule" ||
    normalized === "behavior"
  ) {
    return normalized;
  }
  return null;
}

function parseCursor(raw: unknown): { nextRecordTs?: string; nextEventTs?: string; hasMore?: boolean } | undefined {
  if (typeof raw !== "object" || raw === null) {
    return undefined;
  }
  const cursor = raw as Record<string, unknown>;
  const nextRecordTs = asString(cursor.next_record_ts)
    ? cursor.next_record_ts
    : asString(cursor.next)
      ? cursor.next
      : undefined;
  const nextEventTs = asString(cursor.next_event_ts)
    ? cursor.next_event_ts
    : asString(cursor.next)
      ? cursor.next
      : undefined;
  const hasMore = typeof cursor.hasMore === "boolean" ? cursor.hasMore : undefined;
  if (!nextRecordTs && !nextEventTs && hasMore === undefined) {
    return undefined;
  }
  return {
    nextRecordTs,
    nextEventTs,
    hasMore,
  };
}

function parseObjectList(raw: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

function parseOntologyDelta(raw: unknown): KnowledgeHubOntologyDeltaRaw | undefined {
  if (typeof raw !== "object" || raw === null) {
    return undefined;
  }
  const value = raw as Record<string, unknown>;
  const entities = parseObjectList(value.entities);
  const relations = parseObjectList(value.relations);
  const claims = parseObjectList(value.claims);
  const events = parseObjectList(value.events);
  if (entities.length === 0 && relations.length === 0 && claims.length === 0 && events.length === 0) {
    return undefined;
  }
  return {
    entities,
    relations,
    claims,
    events,
  };
}

function parseSyncPayload(raw: string): {
  runId?: string;
  connectorId?: string;
  ts?: string;
  items: KnowledgeHubRawRecord[];
  cursor?: { nextRecordTs?: string; nextEventTs?: string; hasMore?: boolean };
  sourceFilter?: string;
  ontologyDelta?: KnowledgeHubOntologyDeltaRaw;
} {
  const parsed = JSON.parse(raw || "{}");
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("agent sync payload must be object");
  }
  const parsedPayload = parsed as Record<string, unknown>;
  const rawItems = Array.isArray(parsedPayload.items) ? parsedPayload.items : [];
  const cursor = parseCursor(parsedPayload.cursor);
  const sourceFilter = asString(parsedPayload.source_filter) ? parsedPayload.source_filter : undefined;
  const ontologyDelta = parseOntologyDelta(parsedPayload.ontologyDelta);
  const filtered: KnowledgeHubRawRecord[] = [];

  for (const item of rawItems) {
    if (typeof item !== "object" || item === null) {
      continue;
    }
    const row = item as Record<string, unknown>;
    const source = asSource(row.source);
    const rawId = row.id;
    if (!asString(rawId) || !source) {
      continue;
    }
    filtered.push({
      id: rawId,
      source,
      title: asString(row.title) ? String(row.title) : "",
      content: asString(row.content) ? String(row.content) : undefined,
      filePath: asString(row.filePath) ? String(row.filePath) : undefined,
      metadata: typeof row.metadata === "object" && row.metadata !== null ? (row.metadata as Record<string, unknown>) : undefined,
      updatedAt: asString(row.updatedAt) ? String(row.updatedAt) : undefined,
      tags: Array.isArray(row.tags) ? (row.tags as string[]) : undefined,
      classification: row.classification as DataClassification | undefined,
    });
  }

  return {
    runId: asString(parsedPayload.runId) ? String(parsedPayload.runId) : undefined,
    connectorId: asString(parsedPayload.connectorId) ? String(parsedPayload.connectorId) : undefined,
    ts: asString(parsedPayload.ts) ? String(parsedPayload.ts) : undefined,
    items: filtered,
    cursor,
    sourceFilter,
    ontologyDelta,
  };
}

function rowAsRecord(row: KnowledgeHubRawRecord): DataRecord {
  return {
    sourceRecordId: row.id,
    sourceUpdatedAt: row.updatedAt ?? new Date().toISOString(),
    sourceHash: row.id,
    payload: {
      ...row,
      source: row.source,
    },
    classification: normalizeClassification(row),
  };
}

export class KnowledgeHubCLICommand {
  constructor(
    private readonly config: KnowledgeHubConnectorConfig,
    private readonly runner: (cmd: string, args: string[]) => Promise<string>
  ) {}

  run(_cmd: string, args: string[]): Promise<string> {
    const base = this.config.pythonPath ?? "python";
    const entrypoint = this.config.command
      ? [this.config.command]
      : resolveKnowledgeHubCliEntrypoint(this.config.projectRoot);
    const command = [...entrypoint, ...args];
    return this.runner(base, command);
  }
}

export class KnowledgeHubConnector implements ConnectorContract {
  readonly id: string;
  readonly version = "1.0.0";
  readonly name: string;
  readonly sourceSystem = "knowledge_hub";
  readonly description = "Obsidian + paper + web knowledge sources";
  readonly supportedScopes = [
    "read:notes",
    "read:papers",
    "read:web",
    "read:expenses",
    "read:sleep",
    "read:schedule",
    "read:behavior",
    "read:graph",
  ] as const;
  readonly supportsIncrementalSync = true;
  private readonly fixedSource: "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";

  constructor(
    private readonly executor: KnowledgeHubCLICommand,
    private readonly defaultScope: ReadonlyArray<string> = [
      "read:notes",
      "read:papers",
      "read:web",
      "read:expenses",
      "read:sleep",
      "read:schedule",
      "read:behavior",
    ],
    options: {
      connectorId?: string;
      fixedSource?: "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";
    } = {}
  ) {
    this.id = options.connectorId ?? "knowledge-hub";
    this.name = this.id;
    this.fixedSource = options.fixedSource ?? "all";
  }

  async authorize(input: ConnectorAuthInput): Promise<ConnectorAuthResult> {
    return {
      credentialId: `khub_${randomUUID()}`,
      accountId: input.actorId,
      scopes: input.scopes.length ? input.scopes : [...this.defaultScope],
      issuedAt: new Date().toISOString(),
      tokenFingerprint: `fp_${randomUUID()}`,
    };
  }

  async sync(input: ConnectorSyncInput): Promise<ConnectorSyncResult> {
    const limit = Math.max(1, Math.min(input.pageLimit ?? 400, 2000));
    const source = this.fixedSource !== "all" ? this.fixedSource : coerceSource((input as { source?: string }).source);
    const args = [
      "agent",
      "sync",
      "--json",
      "--limit",
      String(limit),
    ];
    if (source !== "all") {
      args.push("--source", source);
    }
    const cursor = input.cursor;
    if (cursor) {
      args.push("--cursor", cursor);
    }

    const payload = await this.executor.run("khub", args);
    try {
      const parsed = parseSyncPayload(payload);
      const rawItems = parsed.sourceFilter
        ? parsed.items.filter((item) => item.source === parsed.sourceFilter)
        : parsed.items;
      const records: DataRecord[] = rawItems.map(rowAsRecord);

      if (rawItems.length > 0 && records.length === 0) {
        throw new Error("agent sync payload has no usable records");
      }

      const cursorPayload = parsed.cursor
        ? JSON.stringify({
            next_record_ts: parsed.cursor.nextRecordTs ?? "",
            next_event_ts: parsed.cursor.nextEventTs ?? "",
            hasMore: Boolean(parsed.cursor.hasMore),
          })
        : undefined;

      return {
        runId: parsed.runId ?? `khub_${randomUUID()}`,
        connectorRunId: parsed.runId ?? `sync_${input.requestId ?? randomUUID()}`,
        rawRecords: records,
        connectorId: parsed.connectorId ?? this.id,
        ontologyDelta: parsed.ontologyDelta,
        cursor: input.cursor,
        nextCursor: cursorPayload,
        hasMore: Boolean(parsed.cursor?.hasMore),
        extractedAt: parsed.ts ?? new Date().toISOString(),
        metric: {
          scanned: rawItems.length,
          returned: records.length,
          retries: 0,
        },
      };
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(`agent sync payload json parse error: ${error.message}`);
      }
      if (error instanceof Error) {
        throw new Error(`agent sync failed: ${error.message}`);
      }
      throw new Error("agent sync failed");
    }
  }

  async mapToOntology(
    rawRecords: DataRecord[],
    context: ConnectorMappingContext
  ): Promise<MappedOntologyBatch> {
    const entities: OntologyEntityDraft[] = [];
    const relations: OntologyRelationDraft[] = [];
    const events: OntologyEventDraft[] = [];

    for (const raw of rawRecords) {
      const payload = raw.payload as Record<string, unknown>;
      const source = asSource(payload.source) ?? "note";
      const payloadId = asString(payload.id) ? payload.id : raw.sourceRecordId;
      const title = asString(payload.title) ? payload.title : payloadId;
      const filePath = asString(payload.filePath) ? payload.filePath : undefined;
      const updatedAt = asString(payload.updatedAt) ? payload.updatedAt : undefined;
      const metadata =
        typeof payload.metadata === "object" && payload.metadata !== null
          ? payload.metadata as Record<string, unknown>
          : {};
      const tags = Array.isArray(payload.tags)
        ? payload.tags.filter(asString)
        : [];
      const entityId = `kh:${payloadId}`;
      const now = updatedAt ?? new Date().toISOString();
      const entityType = mapSourceToEntityType(source);
      const eventType = mapSourceToEventType(source);

      entities.push({
        entityType,
        entityId,
        properties: {
          title,
          source,
          sourceType: source,
          filePath: filePath ?? "",
          tags,
          metadata,
        },
        classification: raw.classification,
        schemaVersion: "khub-1.0",
        actorId: context.actorId,
      });

      events.push({
        eventType,
        aggregateId: entityId,
        aggregateType: entityType,
        payload: {
          sourceRecordId: raw.sourceRecordId,
          source,
          title,
          runId: context.connectorRunId,
          occurredAt: now,
          metadata,
        },
        actorId: context.actorId,
        occurredAt: now,
        sourceRecordId: raw.sourceRecordId,
        classification: raw.classification,
        schemaVersion: "khub-1.0",
      });
    }

    return { entities, relations, events };
  }

  async emitEvents(input: ConnectorEmitInput, runtime: ConnectorRuntimeContext): Promise<ConnectorEmitResult> {
    return emitOntologyBatchToBus(input, runtime);
  }
}
