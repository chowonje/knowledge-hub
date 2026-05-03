import { join } from "node:path";
import type { DataClassification } from "../types.js";
import { JsonlAuditLog } from "./audit-log.js";
import { PlanActVerifyRuntime } from "./agent-runtime.js";
import { JsonFileCursorStore, JsonFileIdempotencyStore, DefaultConnectorRunner } from "./connector-runner.js";
import { JsonlEventBus } from "./event-bus.js";
import { InMemoryFeatureLayer } from "./feature-layer.js";
import { LocalOntologyStore } from "./ontology-store.js";
import { LocalPolicyEngine } from "./policy-engine.js";

export interface LocalStackOptions {
  baseDir: string;
  now?: () => string;
  outboundMaxClassification?: DataClassification;
  writebackMaxClassification?: DataClassification;
  classificationGate?: DataClassification;
}

export class PersonalFoundryStack {
  readonly eventBus: JsonlEventBus;
  readonly audit: JsonlAuditLog;
  readonly policy: LocalPolicyEngine;
  readonly ontologyStore: LocalOntologyStore;
  readonly featureLayer: InMemoryFeatureLayer;
  readonly connectorRunner: DefaultConnectorRunner;
  readonly agentRuntime: PlanActVerifyRuntime;

  constructor(options: LocalStackOptions) {
    const now = options.now ?? (() => new Date().toISOString());

    this.eventBus = new JsonlEventBus({
      logPath: join(options.baseDir, "event-bus.jsonl"),
      now,
    });

    this.audit = new JsonlAuditLog({
      logPath: join(options.baseDir, "audit.jsonl"),
      now,
    });

    this.policy = new LocalPolicyEngine({
      outboundMaxClassification: options.outboundMaxClassification,
      writebackMaxClassification: options.writebackMaxClassification,
    });

    this.ontologyStore = new LocalOntologyStore({
      baseDir: join(options.baseDir, "ontology"),
      now,
    });

    this.featureLayer = new InMemoryFeatureLayer();

    this.connectorRunner = new DefaultConnectorRunner({
      eventBus: this.eventBus,
      audit: this.audit,
      policy: this.policy,
      cursorStore: new JsonFileCursorStore(join(options.baseDir, "connector-cursors.json")),
      idempotencyStore: new JsonFileIdempotencyStore({
        path: join(options.baseDir, "connector-idempotency.json"),
        now,
      }),
      now,
      classificationGate: options.classificationGate ?? "P1",
    });

    this.agentRuntime = new PlanActVerifyRuntime({
      policy: this.policy,
      audit: this.audit,
      now,
    });
  }
}
