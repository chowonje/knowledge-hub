import type {
  ConnectorMappingContext as LegacyConnectorMappingContext,
  DataRecord as LegacyDataRecord,
  DataClassification as LegacyDataClassification,
} from "../types.js";
import type {
  ConnectorEmitContext,
  ConnectorEmitInput,
  ConnectorEmitOutput,
  ConnectorMapContext,
  ConnectorRecord,
  ConnectorSDK,
  ConnectorSyncInput,
  ConnectorSyncOutput,
  OntologyBatch,
  SourceSystem,
} from "../personal-foundry/interfaces.js";
import { emitOntologyBatchToEventBus } from "../personal-foundry/emitters.js";
import { KnowledgeHubConnector } from "./knowledge-hub-connector.js";
import type { KnowledgeHubOntologyDeltaRaw } from "./knowledge-hub-connector.js";

function normalizeSourceSystem(value: string): SourceSystem {
  if (
    value === "knowledge_hub" ||
    value === "calendar" ||
    value === "finance" ||
    value === "sleep" ||
    value === "behavior" ||
    value === "manual"
  ) {
    return value;
  }
  return "knowledge_hub";
}

function relationId(input: {
  relationType: string;
  sourceEntityId: string;
  targetEntityId: string;
}): string {
  const safeType = input.relationType.replace(/[^a-zA-Z0-9_-]/g, "_");
  return `rel:${input.sourceEntityId}:${safeType}:${input.targetEntityId}`;
}

function toClassification(value: unknown, fallback: LegacyDataClassification = "P2"): LegacyDataClassification {
  if (value === "P0" || value === "P1" || value === "P2" || value === "P3") {
    return value;
  }
  return fallback;
}

function toIso(value: unknown, fallback: string): string {
  if (typeof value === "string" && value.trim().length > 0) {
    return value;
  }
  return fallback;
}

function stripRecord(input: Record<string, unknown>, excluded: string[]): Record<string, unknown> {
  const skipped = new Set(excluded);
  const output: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (!skipped.has(key)) {
      output[key] = value;
    }
  }
  return output;
}

function mapDeltaToBatch(
  delta: KnowledgeHubOntologyDeltaRaw | undefined,
  sourceSystem: SourceSystem,
  fallbackNow: string
): OntologyBatch | undefined {
  if (!delta) {
    return undefined;
  }

  const entities = delta.entities
    .map((item: Record<string, unknown>) => {
      const id = String(item.entity_id ?? item.id ?? "").trim();
      const type = String(item.entity_type ?? item.type ?? "Entity").trim();
      if (!id || !type) {
        return null;
      }
      return {
        id,
        type,
        properties: stripRecord(item, [
          "entity_id",
          "id",
          "entity_type",
          "type",
          "classification",
          "updated_at",
          "created_at",
        ]),
        classification: toClassification(item.classification, "P2"),
        sourceSystem,
        updatedAt: toIso(item.updated_at ?? item.created_at, fallbackNow),
      };
    })
    .filter((item): item is NonNullable<typeof item> => item !== null);

  const relationRows = delta.relations.map((row: Record<string, unknown>) => {
    const sourceEntityId = String(row.source_entity_id ?? row.source_id ?? "").trim();
    const targetEntityId = String(row.target_entity_id ?? row.target_id ?? "").trim();
    const type = String(row.relation_type ?? row.relation ?? row.type ?? "related_to").trim();
    if (!sourceEntityId || !targetEntityId || !type) {
      return null;
    }
    return {
      id:
        String(row.id ?? "").trim() ||
        relationId({
          relationType: type,
          sourceEntityId,
          targetEntityId,
        }),
      type,
      sourceEntityId,
      targetEntityId,
      properties: stripRecord(row, [
        "id",
        "source_entity_id",
        "source_id",
        "target_entity_id",
        "target_id",
        "relation_type",
        "relation",
        "type",
        "classification",
        "created_at",
      ]),
      classification: toClassification(row.classification, "P2"),
      sourceSystem,
      updatedAt: toIso(row.created_at, fallbackNow),
    };
  });

  const claimAsRelations = delta.claims
    .map((row: Record<string, unknown>) => {
      const claimId = String(row.claim_id ?? "").trim();
      const sourceEntityId = String(row.subject_entity_id ?? "").trim();
      const targetEntityId = String(row.object_entity_id ?? "").trim();
      const predicate = String(row.predicate ?? "asserts").trim();
      if (!sourceEntityId || !targetEntityId) {
        return null;
      }
      return {
        id: claimId ? `claim-rel:${claimId}` : relationId({ relationType: predicate, sourceEntityId, targetEntityId }),
        type: `claim:${predicate}`,
        sourceEntityId,
        targetEntityId,
        properties: stripRecord(row, ["classification", "created_at"]),
        classification: toClassification(row.classification, "P1"),
        sourceSystem,
        updatedAt: toIso(row.created_at, fallbackNow),
      };
    })
    .filter((item): item is NonNullable<typeof item> => item !== null);

  const relations = [...relationRows, ...claimAsRelations].filter(
    (item): item is NonNullable<typeof item> => item !== null
  );

  const claimEvents = delta.claims
    .map((row: Record<string, unknown>) => {
      const claimId = String(row.claim_id ?? "").trim();
      if (!claimId) {
        return null;
      }
      return {
        aggregateId: claimId,
        aggregateType: "Claim",
        type: "ClaimObserved",
        payload: row,
        classification: toClassification(row.classification, "P1"),
        sourceSystem,
        occurredAt: toIso(row.created_at, fallbackNow),
        actorId: typeof row.source === "string" ? row.source : undefined,
        sourceRecordId: claimId,
      };
    })
    .filter((item): item is NonNullable<typeof item> => item !== null);

  const events = [
    ...delta.events
      .map((row: Record<string, unknown>) => {
        const aggregateId = String(row.entity_id ?? row.aggregate_id ?? "").trim();
        const eventType = String(row.event_type ?? row.type ?? "").trim();
        if (!aggregateId || !eventType) {
          return null;
        }
        return {
          aggregateId,
          aggregateType: String(row.entity_type ?? row.aggregate_type ?? "OntologyEntity"),
          type: eventType,
          payload: row,
          classification: toClassification(row.classification ?? row.policy_class, "P2"),
          sourceSystem,
          occurredAt: toIso(row.created_at ?? row.occurred_at, fallbackNow),
          actorId: typeof row.actor === "string" ? row.actor : undefined,
          sourceRecordId: typeof row.event_id === "string" ? row.event_id : undefined,
        };
      })
      .filter((item): item is NonNullable<typeof item> => item !== null),
    ...claimEvents,
  ];

  if (entities.length === 0 && relations.length === 0 && events.length === 0) {
    return undefined;
  }

  return {
    entities,
    relations,
    events,
    timeSeries: [],
  };
}

export class KnowledgeHubPersonalConnectorBridge implements ConnectorSDK {
  readonly id: string;
  readonly version: string;
  readonly sourceSystem: SourceSystem;
  readonly supportsIncrementalSync: boolean;
  private readonly syncDeltaByRunId = new Map<string, OntologyBatch>();

  constructor(private readonly legacy: KnowledgeHubConnector) {
    this.id = legacy.id;
    this.version = legacy.version;
    this.sourceSystem = normalizeSourceSystem(legacy.sourceSystem);
    this.supportsIncrementalSync = legacy.supportsIncrementalSync;
  }

  async authorize(input: {
    actorId: string;
    requestId: string;
    scopes: string[];
  }) {
    return this.legacy.authorize(input);
  }

  async sync(input: ConnectorSyncInput): Promise<ConnectorSyncOutput> {
    const synced = await this.legacy.sync({
      actorId: input.actorId,
      requestId: input.requestId,
      cursor: input.cursor,
      pageLimit: input.pageLimit,
      dryRun: input.dryRun,
      source: input.source,
    } as {
      actorId: string;
      requestId: string;
      cursor?: string;
      pageLimit?: number;
      dryRun?: boolean;
      source?: string;
    });

    const records: ConnectorRecord[] = synced.rawRecords.map((record: LegacyDataRecord) => ({
      sourceRecordId: record.sourceRecordId,
      sourceUpdatedAt: record.sourceUpdatedAt,
      payload: record.payload,
      classification: record.classification,
    }));

    const ontologyDelta = mapDeltaToBatch(
      synced.ontologyDelta
        ? {
            entities: synced.ontologyDelta.entities ?? [],
            relations: synced.ontologyDelta.relations ?? [],
            claims: synced.ontologyDelta.claims ?? [],
            events: synced.ontologyDelta.events ?? [],
          }
        : undefined,
      this.sourceSystem,
      synced.extractedAt
    );
    if (ontologyDelta) {
      this.syncDeltaByRunId.set(synced.connectorRunId, ontologyDelta);
    }

    return {
      connectorRunId: synced.connectorRunId,
      records,
      cursor: synced.cursor,
      nextCursor: synced.nextCursor,
      hasMore: synced.hasMore,
      extractedAt: synced.extractedAt,
      ontologyDelta,
    };
  }

  async mapToOntology(records: ConnectorRecord[], context: ConnectorMapContext): Promise<OntologyBatch> {
    const stagedDelta = this.syncDeltaByRunId.get(context.connectorRunId);
    if (stagedDelta) {
      this.syncDeltaByRunId.delete(context.connectorRunId);
      return stagedDelta;
    }

    const legacyRecords: LegacyDataRecord[] = records.map((record) => ({
      sourceRecordId: record.sourceRecordId,
      sourceUpdatedAt: record.sourceUpdatedAt,
      payload: record.payload,
      classification: record.classification,
    }));

    const mapped = await this.legacy.mapToOntology(legacyRecords, {
      actorId: context.actorId,
      requestId: context.requestId,
      connectorId: context.connectorId,
      connectorRunId: context.connectorRunId,
      runStartedAt: context.now,
    } as LegacyConnectorMappingContext);

    return {
      entities: mapped.entities.map((entity: typeof mapped.entities[number]) => ({
        id: entity.entityId,
        type: entity.entityType,
        properties: entity.properties,
        classification: entity.classification,
        sourceSystem: this.sourceSystem,
        updatedAt: context.now,
      })),
      relations: mapped.relations.map((relation: typeof mapped.relations[number]) => ({
        id: relationId({
          relationType: relation.relationType,
          sourceEntityId: relation.sourceEntityId,
          targetEntityId: relation.targetEntityId,
        }),
        type: relation.relationType,
        sourceEntityId: relation.sourceEntityId,
        targetEntityId: relation.targetEntityId,
        properties: relation.properties,
        classification: relation.classification,
        sourceSystem: this.sourceSystem,
        updatedAt: context.now,
      })),
      events: mapped.events.map((event: typeof mapped.events[number]) => ({
        aggregateId: event.aggregateId,
        aggregateType: event.aggregateType,
        type: event.eventType,
        payload: event.payload,
        classification: event.classification,
        sourceSystem: this.sourceSystem,
        occurredAt: event.occurredAt,
        actorId: event.actorId ?? undefined,
        sourceRecordId: event.sourceRecordId,
      })),
      timeSeries: [],
    };
  }

  async emitEvents(input: ConnectorEmitInput, context: ConnectorEmitContext): Promise<ConnectorEmitOutput> {
    const emitted = await emitOntologyBatchToEventBus(input, context.eventBus, context.now);

    await context.audit.append({
      actorId: input.actorId,
      action: "connector_sync",
      resourceType: "connector",
      resourceId: this.id,
      allowed: true,
      reason: `emitted ${emitted.eventIds.length} ontology events`,
      classification: "P1",
      requestId: input.requestId,
      runId: input.connectorRunId,
      metadata: {
        eventCount: emitted.eventIds.length,
      },
    });

    return emitted;
  }
}
