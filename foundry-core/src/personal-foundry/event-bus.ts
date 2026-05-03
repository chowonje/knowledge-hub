import crypto from "node:crypto";
import { mkdir, appendFile, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { BusEvent, EventReadInput, EventSubscriber, PersonalEventBus } from "./interfaces.js";

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

export class JsonlEventBus implements PersonalEventBus {
  private readonly logPath: string;
  private readonly now: () => string;
  private readonly subscribers = new Set<EventSubscriber>();

  constructor(input: { logPath: string; now?: () => string }) {
    this.logPath = input.logPath;
    this.now = input.now ?? (() => new Date().toISOString());
  }

  async publish(event: Omit<BusEvent, "id"> & { id?: string }): Promise<BusEvent> {
    const normalized: BusEvent = {
      ...event,
      id: event.id ?? `evt_${crypto.randomUUID()}`,
      occurredAt: event.occurredAt || this.now(),
    };

    await mkdir(dirname(this.logPath), { recursive: true });
    await appendFile(this.logPath, `${JSON.stringify(normalized)}\n`, "utf8");

    for (const subscriber of this.subscribers) {
      await subscriber(normalized);
    }

    return normalized;
  }

  async read(input: EventReadInput = {}): Promise<BusEvent[]> {
    let raw = "";
    try {
      raw = await readFile(this.logPath, "utf8");
    } catch {
      return [];
    }

    const parsed = parseJsonl<BusEvent>(raw);
    const filtered = parsed.filter((event) => {
      if (input.type && event.type !== input.type) {
        return false;
      }
      if (input.from && event.occurredAt < input.from) {
        return false;
      }
      if (input.to && event.occurredAt > input.to) {
        return false;
      }
      return true;
    });

    if (input.limit && input.limit > 0) {
      return filtered.slice(-input.limit);
    }
    return filtered;
  }

  subscribe(subscriber: EventSubscriber): () => void {
    this.subscribers.add(subscriber);
    return () => {
      this.subscribers.delete(subscriber);
    };
  }
}
