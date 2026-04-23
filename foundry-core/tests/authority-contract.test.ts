import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";

import { normalizeFoundryRunEnvelope } from "../src/cli-agent.js";
import { loadSharedPolicyPatternConfig } from "../src/personal-foundry/p0-detection-config.js";
import { LocalPolicyEngine } from "../src/personal-foundry/policy-engine.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");
const fixturesRoot = path.resolve(repoRoot, "docs", "schemas", "fixtures");
const schemasRoot = path.resolve(repoRoot, "docs", "schemas");
const policyCorpusPath = path.resolve(repoRoot, "docs", "policy", "p0-sample-corpus.json");
const policyConformancePath = path.resolve(repoRoot, "docs", "policy", "policy-conformance-cases.json");
const p0PatternConfigPath = path.resolve(repoRoot, "docs", "policy", "p0-detection-patterns.json");
const authorityEnvelopeSchemaPath = path.resolve(schemasRoot, "authority-result-envelope.v1.json");

function loadJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
}

function validateAgainstSchema(schema: object, payload: unknown): void {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate = ajv.compile(schema);
  const ok = validate(payload);
  assert.equal(ok, true, ajv.errorsText(validate.errors, { separator: "\n" }));
}

function validateAgainstSchemaExpectingFailure(schema: object, payload: unknown): string {
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  const validate = ajv.compile(schema);
  const ok = validate(payload);
  assert.equal(ok, false, "schema validation unexpectedly passed");
  return ajv.errorsText(validate.errors, { separator: "\n" });
}

function pointerFromTraceability(traceability: Record<string, unknown>): string | undefined {
  if (typeof traceability.packetPath === "string" && traceability.packetPath) {
    return traceability.packetPath;
  }
  if (typeof traceability.runtimePointer === "string" && traceability.runtimePointer) {
    return traceability.runtimePointer;
  }
  return undefined;
}

function patternSignature(pattern: RegExp): string {
  return `${pattern.source}::${pattern.flags}`;
}

describe("authority contract", () => {
  it("loads p0 detection config from the shared docs policy source", () => {
    const source = loadJson<{
      version?: unknown;
      patterns?: Array<{ regex?: unknown; flags?: unknown }>;
      redactKeys?: unknown[];
    }>(p0PatternConfigPath);

    assert.equal(typeof source.version, "string");
    assert.notEqual(String(source.version || "").trim(), "");
    assert.ok(Array.isArray(source.patterns) && source.patterns.length > 0);
    assert.ok(Array.isArray(source.redactKeys) && source.redactKeys.length > 0);

    const allowedFlags = new Set(["i", "m", "s"]);
    const expectedPatterns = source.patterns!.map((item, index) => {
      const regex = String(item.regex || "").trim();
      assert.notEqual(regex, "", `patterns[${index}].regex must be non-empty`);
      const rawFlags = String(item.flags || "");
      for (const flag of rawFlags) {
        assert.ok(allowedFlags.has(flag), `unsupported regex flag in shared P0 config: ${flag}`);
      }
      return patternSignature(new RegExp(regex, rawFlags));
    });
    const expectedRedactKeys = Array.from(
      new Set(source.redactKeys!.map((key) => String(key || "").trim().toLowerCase()).filter(Boolean))
    ).sort();
    assert.ok(expectedRedactKeys.length > 0);

    const loaded = loadSharedPolicyPatternConfig();
    const loadedPatterns = loaded.patterns.map((pattern) => patternSignature(pattern));
    const loadedRedactKeys = Array.from(new Set(loaded.redactKeys.map((key) => key.trim().toLowerCase()).filter(Boolean))).sort();

    assert.equal(loaded.version, source.version);
    assert.deepEqual(loadedPatterns, expectedPatterns);
    assert.deepEqual(loadedRedactKeys, expectedRedactKeys);
  });

  it("keeps LocalPolicyEngine P0 detection in parity with the shared sample corpus", () => {
    const corpus = loadJson<{ cases: Array<{ id: string; text: string; expectedP0: boolean }> }>(policyCorpusPath);
    const engine = new LocalPolicyEngine();

    for (const item of corpus.cases) {
      const isP0 = engine.classify({ text: item.text }) === "P0";
      assert.equal(isP0, item.expectedP0, item.id);
    }
  });

  it("keeps LocalPolicyEngine decisions in parity with the shared Python authority fixture", async () => {
    const fixture = loadJson<{
      cases: Array<{
        id: string;
        action: any;
        resourceType: string;
        resourceId: string;
        classification?: any;
        payload: unknown;
        expected: {
          allowed: boolean;
          classification: string;
          tsPolicyCode: string;
        };
      }>;
    }>(policyConformancePath);
    const engine = new LocalPolicyEngine();

    for (const item of fixture.cases) {
      const decision = await engine.evaluate({
        actorId: "fixture-user",
        action: item.action,
        resourceType: item.resourceType,
        resourceId: item.resourceId,
        payload: item.payload,
        classification: item.classification,
      });
      assert.equal(decision.allowed, item.expected.allowed, item.id);
      assert.equal(decision.classification, item.expected.classification, item.id);
      assert.equal(decision.policyCode, item.expected.tsPolicyCode, item.id);
    }
  });

  it("validates the shared playbook fixture against the authority sub-schema", () => {
    const runSchema = loadJson<{ properties: { playbook: object } }>(path.resolve(schemasRoot, "agent-run-result.v1.json"));
    const playbookFixture = loadJson<object>(path.resolve(fixturesRoot, "agent-run-playbook.v1.fixture.json"));
    validateAgainstSchema(runSchema.properties.playbook, playbookFixture);
  });

  it("validates capture flow fixtures against the shared authority envelope", () => {
    const schema = loadJson<object>(authorityEnvelopeSchemaPath);
    const fixtures = [
      "authority-result-envelope.v1.fixture.json",
      "dinger-capture-result.v1.fixture.json",
      "dinger-file-result.v1.fixture.json",
      "os-capture-result.v1.fixture.json",
    ] as const;

    for (const fixtureName of fixtures) {
      const fixture = loadJson<object>(path.resolve(fixturesRoot, fixtureName));
      validateAgainstSchema(schema, fixture);
    }
  });

  it("pins capture-flow lifecycle stage, policy, and traceability progression", () => {
    const captureFixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "dinger-capture-result.v1.fixture.json"));
    const filedFixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "dinger-file-result.v1.fixture.json"));
    const osFixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "os-capture-result.v1.fixture.json"));
    const expectedPolicy = {
      capturePacket: "input",
      dingerFiling: "projection_only",
      osBridge: "inbox_evidence_candidate_only",
      canonicalStore: "no_new_store",
    };

    assert.equal(captureFixture.stage, "captured");
    assert.equal(filedFixture.stage, "filed");
    assert.equal(osFixture.stage, "linked_to_os");
    assert.deepEqual(captureFixture.flowSemantics, expectedPolicy);
    assert.deepEqual(filedFixture.flowSemantics, expectedPolicy);
    assert.deepEqual(osFixture.flowSemantics, expectedPolicy);

    assert.equal(captureFixture.traceability.captureId, filedFixture.traceability.captureId);
    assert.equal(filedFixture.traceability.captureId, osFixture.traceability.captureId);
    assert.equal(captureFixture.traceability.packetPath, filedFixture.traceability.packetPath);
    assert.equal(filedFixture.traceability.packetPath, osFixture.traceability.packetPath);
    assert.equal("filingOutputPointer" in captureFixture.traceability, false);
    assert.equal(filedFixture.traceability.filingOutputPointer, "KnowledgeOS/Dinger/Pages/rag-capture.md");
    assert.equal(osFixture.traceability.filingOutputPointer, filedFixture.traceability.filingOutputPointer);
    assert.equal(osFixture.traceability.osBridgeTrace.bridge, "os_capture");
    assert.equal(osFixture.traceability.osBridgeTrace.schema, "knowledge-hub.os.capture.result.v1");
    assert.equal(osFixture.traceability.osBridgeTrace.itemId, "inbox_123");
    assert.equal(osFixture.traceability.osBridgeTrace.projectId, "proj_123");
    assert.equal(osFixture.traceability.osBridgeTrace.captureTraceBridge, "dinger");
    assert.equal(osFixture.traceability.osBridgeTrace.relativePath, filedFixture.traceability.filingOutputPointer);
  });

  it("keeps authority timeout classification-only and out of canonical promotion", () => {
    const fixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "authority-result-envelope.v1.fixture.json"));
    assert.equal(fixture.stage, "failed");
    assert.equal(fixture.authorityTimeout.handledAs, "classification_only");
    assert.deepEqual(fixture.flowSemantics, {
      capturePacket: "input",
      dingerFiling: "projection_only",
      osBridge: "inbox_evidence_candidate_only",
      canonicalStore: "no_new_store",
    });
  });

  it("requires an error message for failed authority envelopes", () => {
    const schema = loadJson<object>(authorityEnvelopeSchemaPath);
    const fixture = loadJson<Record<string, unknown>>(path.resolve(fixturesRoot, "authority-result-envelope.v1.fixture.json"));
    delete fixture.error;
    const errors = validateAgainstSchemaExpectingFailure(schema, fixture);
    assert.match(errors, /error/);
  });

  it("pins flow semantics and traceability across capture -> dinger -> OS smoke fixtures", () => {
    const cases = [
      ["authority-result-envelope.v1.fixture.json", "failed", true, false],
      ["dinger-capture-result.v1.fixture.json", "captured", false, false],
      ["dinger-file-result.v1.fixture.json", "filed", true, false],
      ["os-capture-result.v1.fixture.json", "linked_to_os", true, true],
    ] as const;

    for (const [fixtureName, expectedStage, expectsProjectionPointer, expectsOsBridgeTrace] of cases) {
      const fixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, fixtureName));
      assert.equal(fixture.stage, expectedStage, fixtureName);
      assert.deepEqual(fixture.flowSemantics, {
        capturePacket: "input",
        dingerFiling: "projection_only",
        osBridge: "inbox_evidence_candidate_only",
        canonicalStore: "no_new_store",
      });

      assert.equal(typeof fixture.traceability.captureId, "string", fixtureName);
      assert.ok(pointerFromTraceability(fixture.traceability), fixtureName);

      if (expectsProjectionPointer) {
        assert.equal(fixture.traceability.filingOutputPointer, fixture.projectionRelativePath, fixtureName);
      } else {
        assert.equal("filingOutputPointer" in fixture.traceability, false, fixtureName);
      }

      if (expectsOsBridgeTrace) {
        assert.equal(fixture.traceability.osBridgeTrace.bridge, "os_capture", fixtureName);
        assert.equal(fixture.traceability.osBridgeTrace.schema, "knowledge-hub.os.capture.result.v1", fixtureName);
        assert.equal(fixture.traceability.osBridgeTrace.captureTraceBridge, "dinger", fixtureName);
        assert.equal(fixture.traceability.osBridgeTrace.itemId, fixture.item.id, fixtureName);
        assert.equal(fixture.traceability.osBridgeTrace.projectId, fixture.item.projectId, fixtureName);
        assert.equal(fixture.traceability.osBridgeTrace.relativePath, fixture.projectionRelativePath, fixtureName);
      } else {
        assert.equal("osBridgeTrace" in fixture.traceability, false, fixtureName);
      }

      if (fixtureName === "authority-result-envelope.v1.fixture.json") {
        assert.equal(fixture.authorityTimeout.handledAs, "classification_only");
      }
    }
  });

  it("keeps runtime statuses command-specific while docs-only stage pins the lifecycle position", () => {
    const dingerFileFixture = loadJson<Record<string, unknown>>(path.resolve(fixturesRoot, "dinger-file-result.v1.fixture.json"));
    const osCaptureFixture = loadJson<Record<string, unknown>>(path.resolve(fixturesRoot, "os-capture-result.v1.fixture.json"));
    assert.equal(dingerFileFixture.status, "ok");
    assert.equal(dingerFileFixture.stage, "filed");
    assert.equal(osCaptureFixture.status, "ok");
    assert.equal(osCaptureFixture.stage, "linked_to_os");
  });

  it("pins note-first dedupe and replay semantics for os capture fixtures", () => {
    const fixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "os-capture-result.v1.fixture.json"));
    assert.deepEqual(fixture.dedupeKey, {
      kind: "dinger_file",
      strategy: "vault_note",
      markers: [{ sourceType: "vault", primary: "KnowledgeOS/Dinger/Pages/rag-capture.md" }],
      fingerprint: "dinger_file|vault_note|vault:KnowledgeOS/Dinger/Pages/rag-capture.md",
    });
    assert.deepEqual(fixture.replay, {
      policy: { open: "reuse", resolved: "create_new", triaged: "create_new" },
      dedupeKey: fixture.dedupeKey,
      action: "created_new_without_prior_match",
      matchedItemIds: [],
      matchedStates: [],
    });
    assert.equal(fixture.linkAction, "created");
    assert.equal(fixture.duplicateSourceRefsSkipped, 1);
    assert.deepEqual(fixture.reason, {
      dedupeKeySummary: {
        kind: "dinger_file",
        strategy: "vault_note",
        markerCount: 1,
        markers: [{ sourceType: "vault", primary: "KnowledgeOS/Dinger/Pages/rag-capture.md" }],
        fingerprint: "dinger_file|vault_note|vault:KnowledgeOS/Dinger/Pages/rag-capture.md",
      },
      replayAction: "created_new_without_prior_match",
      matchedOpenItems: [],
      matchedResolvedItems: [],
      matchedOtherItems: [],
      bridgeTraceSummary: {
        bridge: "dinger",
        sourceSchema: "knowledge-hub.dinger.file.result.v1",
        kind: "web_capture",
        relativePath: "KnowledgeOS/Dinger/Pages/rag-capture.md",
        title: "RAG Capture",
        captureUrl: "https://example.com/rag",
        captureId: "cap_123456789abc",
      },
    });
    assert.match(fixture.explanation, /created_new_without_prior_match/);
  });

  it("accepts and preserves the shared agent-run fixture through local normalization", () => {
    const runSchema = loadJson<object>(path.resolve(schemasRoot, "agent-run-result.v1.json"));
    const playbookFixture = loadJson<object>(path.resolve(fixturesRoot, "agent-run-playbook.v1.fixture.json"));
    const runFixture = loadJson<Record<string, unknown>>(path.resolve(fixturesRoot, "agent-run-result.v1.fixture.json"));
    runFixture.playbook = playbookFixture;

    validateAgainstSchema(runSchema, runFixture);
    const normalized = normalizeFoundryRunEnvelope(runFixture as any);
    assert.equal(normalized.runId, runFixture.runId);
    assert.equal(normalized.playbook?.schema, "knowledge-hub.foundry.agent.run.playbook.v1");
    assert.equal(normalized.playbook?.steps[0]?.tool, "search_knowledge");
  });

  it("validates the shared bridge fixtures against the authority schemas", () => {
    const fixtures = [
      ["connector-sync-result.v2.json", "connector-sync-result.v2.fixture.json"],
      ["os-project-create-result.v1.json", "os-project-create-result.v1.fixture.json"],
      ["os-project-update-result.v1.json", "os-project-update-result.v1.fixture.json"],
      ["os-project-show-result.v1.json", "os-project-show-result.v1.fixture.json"],
      ["os-project-evidence-result.v1.json", "os-project-evidence-result.v1.fixture.json"],
      ["dinger-ingest-result.v1.json", "dinger-ingest-result.v1.fixture.json"],
      ["dinger-ask-result.v1.json", "dinger-ask-result.v1.fixture.json"],
      ["dinger-capture-result.v1.json", "dinger-capture-result.v1.fixture.json"],
      ["dinger-file-result.v1.json", "dinger-file-result.v1.fixture.json"],
      ["dinger-recent-result.v1.json", "dinger-recent-result.v1.fixture.json"],
      ["dinger-lint-result.v1.json", "dinger-lint-result.v1.fixture.json"],
      ["os-project-export-obsidian-result.v1.json", "os-project-export-obsidian-result.v1.fixture.json"],
      ["os-capture-result.v1.json", "os-capture-result.v1.fixture.json"],
      ["os-goal-update-result.v1.json", "os-goal-update-result.v1.fixture.json"],
      ["os-task-update-result.v1.json", "os-task-update-result.v1.fixture.json"],
      ["os-task-start-result.v1.json", "os-task-start-result.v1.fixture.json"],
      ["os-task-block-result.v1.json", "os-task-block-result.v1.fixture.json"],
      ["os-task-complete-result.v1.json", "os-task-complete-result.v1.fixture.json"],
      ["os-task-cancel-result.v1.json", "os-task-cancel-result.v1.fixture.json"],
      ["os-inbox-triage-result.v1.json", "os-inbox-triage-result.v1.fixture.json"],
      ["os-decide-result.v1.json", "os-decide-result.v1.fixture.json"],
      ["os-decision-add-result.v1.json", "os-decision-add-result.v1.fixture.json"],
      ["os-decision-list-result.v1.json", "os-decision-list-result.v1.fixture.json"],
      ["os-next-result.v1.json", "os-next-result.v1.fixture.json"],
    ] as const;

    for (const [schemaName, fixtureName] of fixtures) {
      const schema = loadJson<object>(path.resolve(schemasRoot, schemaName));
      const fixture = loadJson<object>(path.resolve(fixturesRoot, fixtureName));
      validateAgainstSchema(schema, fixture);
      const roundTripped = JSON.parse(JSON.stringify(fixture));
      assert.equal((roundTripped as { schema?: string }).schema, (fixture as { schema?: string }).schema, fixtureName);
    }
  });

  it("rejects dinger file source refs that do not carry a primary identifier", () => {
    const schema = loadJson<object>(path.resolve(schemasRoot, "dinger-file-result.v1.json"));
    const fixture = loadJson<Record<string, unknown>>(path.resolve(fixturesRoot, "dinger-file-result.v1.fixture.json"));
    fixture.sourceRefs = [{ sourceType: "paper" }];
    const errors = validateAgainstSchemaExpectingFailure(schema, fixture);
    assert.match(errors, /paperId/);
  });

  it("rejects os capture items that lose project scope or severity typing", () => {
    const schema = loadJson<object>(path.resolve(schemasRoot, "os-capture-result.v1.json"));
    const fixture = loadJson<Record<string, any>>(path.resolve(fixturesRoot, "os-capture-result.v1.fixture.json"));
    fixture.item.severity = "urgent-ish";
    delete fixture.item.projectId;
    const errors = validateAgainstSchemaExpectingFailure(schema, fixture);
    assert.match(errors, /projectId|severity|urgent-ish/);
  });
});
