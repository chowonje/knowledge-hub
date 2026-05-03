import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createFeatureRuntime } from "../src/features.js";

describe("createFeatureRuntime with ontology event log", () => {
  it("builds daily_coach from ontology-store events", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "khub-feature-ontology-"));
    const filePath = path.join(tmpDir, "ontology.events.jsonl");

    const rows = [
      {
        id: "oe_1",
        sequence: 1,
        aggregateId: "expense:1",
        aggregateType: "Expense",
        type: "ExpenseLogged",
        payload: {
          source: "expense",
          amount: 310000,
          metadata: { tags: ["spend"] },
        },
        occurredAt: "2026-02-26T09:00:00.000Z",
      },
      {
        id: "oe_2",
        sequence: 1,
        aggregateId: "sleep:1",
        aggregateType: "SleepLog",
        type: "SleepLogged",
        payload: {
          source: "sleep",
          durationMinutes: 300,
          metadata: { tags: ["health"] },
        },
        occurredAt: "2026-02-26T23:00:00.000Z",
      },
      {
        id: "oe_3",
        sequence: 1,
        aggregateId: "note:1",
        aggregateType: "KnowledgeItem",
        type: "KnowledgeItemIngested",
        payload: {
          source: "note",
          title: "Deep Work Notes",
          metadata: { tags: ["focus"] },
        },
        occurredAt: "2026-02-27T08:00:00.000Z",
      },
    ];

    fs.writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");

    const runtime = createFeatureRuntime({
      eventLogPath: filePath,
      now: () => "2026-02-28T00:00:00.000Z",
    });

    const result = await runtime.execute({
      name: "daily_coach",
      intent: "analyze",
      params: {
        source: "all",
        days: 7,
        top_k: 5,
      },
    });

    assert.equal(result.featureName, "daily_coach");
    assert.equal(result.status, "ok");

    const payload = result.payload as Record<string, unknown>;
    const sourceCount = payload.sourceCount as Record<string, number>;
    assert.equal(sourceCount.expense > 0, true);
    assert.equal(sourceCount.sleep > 0, true);
    assert.equal(sourceCount.note > 0, true);

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});
