import crypto from "node:crypto";
import { mkdir, appendFile, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { AuditQuery, AuditRecord, PersonalAuditLog } from "./interfaces.js";

function parseJsonl<T>(raw: string): T[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as T];
      } catch {
        return [];
      }
    });
}

export class JsonlAuditLog implements PersonalAuditLog {
  private readonly logPath: string;
  private readonly now: () => string;

  constructor(input: { logPath: string; now?: () => string }) {
    this.logPath = input.logPath;
    this.now = input.now ?? (() => new Date().toISOString());
  }

  async append(input: Omit<AuditRecord, "id" | "at"> & { id?: string; at?: string }): Promise<AuditRecord> {
    const record: AuditRecord = {
      ...input,
      id: input.id ?? `audit_${crypto.randomUUID()}`,
      at: input.at ?? this.now(),
    };

    await mkdir(dirname(this.logPath), { recursive: true });
    await appendFile(this.logPath, `${JSON.stringify(record)}\n`, "utf8");
    return record;
  }

  async query(filter: AuditQuery = {}): Promise<AuditRecord[]> {
    let raw = "";
    try {
      raw = await readFile(this.logPath, "utf8");
    } catch {
      return [];
    }

    const parsed = parseJsonl<AuditRecord>(raw);
    const filtered = parsed.filter((record) => {
      if (filter.actorId && record.actorId !== filter.actorId) {
        return false;
      }
      if (filter.action && record.action !== filter.action) {
        return false;
      }
      if (filter.runId && record.runId !== filter.runId) {
        return false;
      }
      if (filter.from && record.at < filter.from) {
        return false;
      }
      if (filter.to && record.at > filter.to) {
        return false;
      }
      return true;
    });

    if (filter.limit && filter.limit > 0) {
      return filtered.slice(-filter.limit);
    }
    return filtered;
  }
}
