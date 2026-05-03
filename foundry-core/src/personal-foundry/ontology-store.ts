import crypto from "node:crypto";
import { appendFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type {
  OntologyBatch,
  OntologyBatchAppendResult,
  OntologyEntityRecord,
  OntologyRelationRecord,
  OntologySnapshot,
  PersonalOntologyStore,
  StoredOntologyEvent,
  TimeSeriesBatchRecord,
  TimeSeriesPoint,
} from "./interfaces.js";

export interface LocalOntologyStoreOptions {
  baseDir: string;
  now?: () => string;
}

function keyForSeries(key: string, entityId: string): string {
  return `${key}::${entityId}`;
}

export class LocalOntologyStore implements PersonalOntologyStore {
  private readonly baseDir: string;
  private readonly now: () => string;

  private readonly entityById = new Map<string, OntologyEntityRecord>();
  private readonly relationById = new Map<string, OntologyRelationRecord>();
  private readonly eventsByAggregate = new Map<string, StoredOntologyEvent[]>();
  private readonly sequenceByAggregate = new Map<string, number>();
  private readonly snapshotByStream = new Map<string, OntologySnapshot>();
  private readonly pointsBySeries = new Map<string, TimeSeriesPoint[]>();

  private readonly entityLogPath: string;
  private readonly relationLogPath: string;
  private readonly eventLogPath: string;
  private readonly snapshotLogPath: string;
  private readonly timeSeriesLogPath: string;

  constructor(options: LocalOntologyStoreOptions) {
    this.baseDir = options.baseDir;
    this.now = options.now ?? (() => new Date().toISOString());

    this.entityLogPath = join(this.baseDir, "ontology.entities.jsonl");
    this.relationLogPath = join(this.baseDir, "ontology.relations.jsonl");
    this.eventLogPath = join(this.baseDir, "ontology.events.jsonl");
    this.snapshotLogPath = join(this.baseDir, "ontology.snapshots.jsonl");
    this.timeSeriesLogPath = join(this.baseDir, "ontology.timeseries.jsonl");
  }

  private async appendJsonl(path: string, payload: unknown): Promise<void> {
    await mkdir(this.baseDir, { recursive: true });
    await appendFile(path, `${JSON.stringify(payload)}\n`, "utf8");
  }

  async appendBatch(batch: OntologyBatch): Promise<OntologyBatchAppendResult> {
    const eventIds: string[] = [];

    for (const entity of batch.entities) {
      this.entityById.set(entity.id, entity);
      await this.appendJsonl(this.entityLogPath, entity);
    }

    for (const relation of batch.relations) {
      this.relationById.set(relation.id, relation);
      await this.appendJsonl(this.relationLogPath, relation);
    }

    for (const event of batch.events) {
      const nextSequence = (this.sequenceByAggregate.get(event.aggregateId) ?? 0) + 1;
      this.sequenceByAggregate.set(event.aggregateId, nextSequence);

      const stored: StoredOntologyEvent = {
        ...event,
        id: event.id ?? `oevt_${crypto.randomUUID()}`,
        sequence: nextSequence,
      };

      const list = this.eventsByAggregate.get(event.aggregateId) ?? [];
      list.push(stored);
      this.eventsByAggregate.set(event.aggregateId, list);
      eventIds.push(stored.id);
      await this.appendJsonl(this.eventLogPath, stored);
    }

    const timeSeries = batch.timeSeries ?? [];
    for (const entry of timeSeries) {
      await this.appendTimeSeries(entry);
    }

    return {
      eventIds,
      entityCount: batch.entities.length,
      relationCount: batch.relations.length,
      eventCount: batch.events.length,
      timeSeriesCount: timeSeries.length,
    };
  }

  async readEntity(entityId: string): Promise<OntologyEntityRecord | null> {
    return this.entityById.get(entityId) ?? null;
  }

  async readEvents(input: { aggregateId: string; fromSequence?: number; toSequence?: number }): Promise<StoredOntologyEvent[]> {
    const list = this.eventsByAggregate.get(input.aggregateId) ?? [];
    return list.filter((event) => {
      if (input.fromSequence !== undefined && event.sequence < input.fromSequence) {
        return false;
      }
      if (input.toSequence !== undefined && event.sequence > input.toSequence) {
        return false;
      }
      return true;
    });
  }

  async appendSnapshot(input: Omit<OntologySnapshot, "id">): Promise<OntologySnapshot> {
    const snapshot: OntologySnapshot = {
      ...input,
      id: `snap_${crypto.randomUUID()}`,
      createdAt: input.createdAt || this.now(),
    };

    this.snapshotByStream.set(snapshot.streamId, snapshot);
    await this.appendJsonl(this.snapshotLogPath, snapshot);
    return snapshot;
  }

  async readLatestSnapshot(streamId: string): Promise<OntologySnapshot | null> {
    return this.snapshotByStream.get(streamId) ?? null;
  }

  async appendTimeSeries(entry: TimeSeriesBatchRecord): Promise<void> {
    const key = keyForSeries(entry.key, entry.entityId);
    const existing = this.pointsBySeries.get(key) ?? [];
    existing.push(...entry.points);
    existing.sort((a, b) => a.at.localeCompare(b.at));
    this.pointsBySeries.set(key, existing);

    await this.appendJsonl(this.timeSeriesLogPath, {
      ...entry,
      appendedAt: this.now(),
    });
  }

  async getTimeSeries(input: { key: string; entityId: string; from?: string; to?: string }): Promise<TimeSeriesPoint[]> {
    const key = keyForSeries(input.key, input.entityId);
    const existing = this.pointsBySeries.get(key) ?? [];
    return existing.filter((point) => {
      if (input.from && point.at < input.from) {
        return false;
      }
      if (input.to && point.at > input.to) {
        return false;
      }
      return true;
    });
  }
}
