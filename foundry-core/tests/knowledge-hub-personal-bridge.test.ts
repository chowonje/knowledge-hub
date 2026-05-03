import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { KnowledgeHubPersonalConnectorBridge } from "../src/adapters/knowledge-hub-personal-connector.js";
import type { KnowledgeHubConnector } from "../src/adapters/knowledge-hub-connector.js";
import type { BusEvent, PersonalAuditLog, PersonalEventBus } from "../src/personal-foundry/interfaces.js";

describe("KnowledgeHubPersonalConnectorBridge", () => {
  it("maps legacy connector sync/map/emit to personal connector contract", async () => {
    const now = new Date().toISOString();

    const legacy = {
      id: "knowledge-hub",
      version: "1.0.0",
      sourceSystem: "knowledge_hub",
      supportsIncrementalSync: true,
      async authorize() {
        return {
          credentialId: "cred",
          accountId: "acc",
          scopes: ["read"],
          issuedAt: now,
          tokenFingerprint: "fp",
        };
      },
      async sync() {
        return {
          runId: "legacy-run",
          connectorRunId: "legacy-connector-run",
          rawRecords: [
            {
              sourceRecordId: "rec-1",
              sourceUpdatedAt: now,
              payload: { source: "note", title: "hello" },
              classification: "P1",
            },
          ],
          cursor: "cursor-1",
          nextCursor: "cursor-2",
          hasMore: false,
          extractedAt: now,
          metric: {
            scanned: 1,
            returned: 1,
            retries: 0,
          },
        };
      },
      async mapToOntology() {
        return {
          entities: [
            {
              entityType: "KnowledgeItem",
              entityId: "kh:1",
              properties: { title: "hello" },
              classification: "P1",
              schemaVersion: "khub-1.0",
              actorId: "user-1",
            },
          ],
          relations: [],
          events: [
            {
              eventType: "DocumentIngested",
              aggregateId: "kh:1",
              aggregateType: "KnowledgeItem",
              payload: { sourceRecordId: "rec-1" },
              actorId: "user-1",
              occurredAt: now,
              sourceRecordId: "rec-1",
              classification: "P1",
              schemaVersion: "khub-1.0",
            },
          ],
        };
      },
    } as unknown as KnowledgeHubConnector;

    const bridge = new KnowledgeHubPersonalConnectorBridge(legacy);

    const synced = await bridge.sync({
      actorId: "user-1",
      requestId: "req-1",
      source: "note",
      cursor: "cursor-1",
      pageLimit: 10,
    });

    assert.equal(synced.records.length, 1);
    assert.equal(synced.records[0].sourceRecordId, "rec-1");

    const mapped = await bridge.mapToOntology(synced.records, {
      actorId: "user-1",
      requestId: "req-1",
      connectorId: bridge.id,
      connectorRunId: synced.connectorRunId,
      now,
      source: "note",
    });

    assert.equal(mapped.entities.length, 1);
    assert.equal(mapped.events.length, 1);

    const published: BusEvent[] = [];
    const eventBus: PersonalEventBus = {
      async publish(event) {
        const normalized: BusEvent = {
          id: event.id ?? `evt_${published.length + 1}`,
          type: event.type,
          occurredAt: event.occurredAt,
          sourceSystem: event.sourceSystem,
          actorId: event.actorId,
          classification: event.classification,
          payload: event.payload,
        };
        published.push(normalized);
        return normalized;
      },
      async read() {
        return published;
      },
      subscribe() {
        return () => {};
      },
    };

    const auditRecords: unknown[] = [];
    const audit: PersonalAuditLog = {
      async append(input) {
        const record = {
          id: input.id ?? `audit_${auditRecords.length + 1}`,
          at: input.at ?? now,
          ...input,
        };
        auditRecords.push(record);
        return record;
      },
      async query() {
        return [];
      },
    };

    const emitted = await bridge.emitEvents(
      {
        actorId: "user-1",
        requestId: "req-1",
        connectorRunId: synced.connectorRunId,
        mapped,
      },
      {
        now: () => now,
        eventBus,
        audit,
      }
    );

    assert.equal(emitted.eventIds.length, 2);
    assert.equal(published.length, 2);
    assert.equal(auditRecords.length > 0, true);
  });
});
