/*
Authority contract:
- Python is the final authority for validation, policy gating, normalized bridge
  payload acceptance, and product-facing outputs.
- TypeScript owns runtime orchestration and personal-foundry state for
  self-protection inside `foundry-core`.
- This schema registry is intentionally a runtime subset, not a parallel copy of
  Python's full contract registry.
*/
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { Ajv, type ErrorObject, type ValidateFunction } from "ajv";
import type {
  AgentRunRecord,
  AgentToolInput,
  AgentToolExecutionContext,
  AgentRuntimeInput,
  AgentRuntimeResult,
  FoundryToolRegistry,
  FoundryToolResult,
  PolicyContext,
  PolicyDecision,
  RunArtifact,
  RunStage,
  StageTransition,
  DataClassification,
  DataSanitizer,
} from "./types.js";

const SCHEMA_NAME_BY_ID: Record<string, string> = {
  "knowledge-hub.learning.map.result.v1": "learning-map-result.v1.json",
  "knowledge-hub.learning.template.result.v1": "learning-map-result.v1.json",
  "knowledge-hub.learning.grade.result.v1": "learning-grade-result.v1.json",
  "knowledge-hub.learning.next.result.v1": "learning-next-result.v1.json",
  "knowledge-hub.crawl.ingest.result.v1": "crawl-ingest-result.v1.json",
};

const SCHEMA_SEARCH_ROOTS = [
  path.resolve(process.cwd(), "docs", "schemas"),
  path.resolve(process.cwd(), "..", "..", "docs", "schemas"),
  path.resolve(__dirname, "../../docs/schemas"),
];

const ajv = new Ajv({ allErrors: true, strict: false });
const schemaValidators = new Map<string, ValidateFunction>();

function resolveSchemaFile(schemaId: string): string | null {
  const fileName = SCHEMA_NAME_BY_ID[schemaId];
  if (!fileName) {
    return null;
  }
  for (const root of SCHEMA_SEARCH_ROOTS) {
    const candidate = path.join(root, fileName);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function getSchemaValidator(schemaId: string): ValidateFunction | null {
  if (schemaValidators.has(schemaId)) {
    return schemaValidators.get(schemaId) ?? null;
  }

  const schemaPath = resolveSchemaFile(schemaId);
  if (!schemaPath) {
    return null;
  }
  try {
    const schemaSource = fs.readFileSync(schemaPath, "utf-8");
    const schema = JSON.parse(schemaSource);
    const validate = ajv.compile(schema);
    schemaValidators.set(schemaId, validate);
    return validate;
  } catch (error) {
    console.error(`schema load error for ${schemaId}:`, error);
    return null;
  }
}

function summarizeAjvErrors(errors: ErrorObject[] | null | undefined): string[] {
  if (!errors?.length) {
    return [];
  }
  return errors.flatMap((entry: ErrorObject, index) => {
    const itemPath = entry.instancePath || "root";
    const message = entry.message ?? "invalid";
    return [`${index}: ${itemPath || "root"} ${message}`];
  });
}

export interface AgentRuntimeDependencies {
  now: () => string;
}

function defaultNow(): string {
  return new Date().toISOString();
}

function createRunRecord(actorId: string, goal: string, tool: string, now: () => string): AgentRunRecord {
  return {
    id: `run_${crypto.randomUUID()}`,
    actorId,
    goal,
    stage: "PLAN",
    status: "running",
    tool,
    createdAt: now(),
    updatedAt: now(),
    transitions: [],
    verify: undefined,
    writeback: undefined,
  };
}

function pushTransition(
  run: AgentRunRecord,
  stage: RunStage,
  status: StageTransition["status"],
  message: string,
  now: () => string
): void {
  run.transitions.push({ stage, status, message, at: now() });
  run.stage = stage;
  run.updatedAt = now();
  if (status === "failed") {
    run.status = "failed";
    return;
  }
  if (status === "blocked") {
    run.status = "blocked";
    return;
  }
  if (stage === "DONE" && status === "completed") {
    run.status = "completed";
  }
}

async function defaultPlanner(goal: string): Promise<string[]> {
  return [`synthesize:${goal}`];
}

function defaultPolicy(
  context: PolicyContext
): Promise<PolicyDecision> {
  const action = context.action;
  const classification = context.classification ?? "P3";
  const sensitive = classification === "P0";
  const allowed = action !== "llm_execute" || !sensitive;
  const result = allowed ? "ALLOW" : "DENY";
  return Promise.resolve({
    action,
    resource: context.resource,
    result,
    allowed,
    reason: allowed ? "policy allow" : "policy deny: P0 payload",
    severity: allowed ? "info" : "error",
    classification,
    requiresWriteback: false,
  });
}

async function defaultValidator(artifact: RunArtifact): Promise<{ ok: boolean; errors: string[] }> {
  if (!artifact?.jsonContent) {
    return { ok: false, errors: ["missing artifact jsonContent"] };
  }
  if (typeof artifact.jsonContent !== "object") {
    return { ok: false, errors: ["artifact jsonContent must be an object"] };
  }

  const jsonObj = artifact.jsonContent as Record<string, unknown>;
  const schemaId = typeof jsonObj.schema === "string" ? jsonObj.schema.trim() : "";
  if (!schemaId) {
    return { ok: true, errors: [] };
  }

  const validate = getSchemaValidator(schemaId);
  if (!validate) {
    return { ok: true, errors: [] };
  }

  if (validate(artifact.jsonContent)) {
    return { ok: true, errors: [] };
  }
  return { ok: false, errors: summarizeAjvErrors(validate.errors) };
}

function createSanitizer(): DataSanitizer {
  return {
    sanitize(value: unknown, profile: "facts" | "summary") {
      const serialized = JSON.stringify(value ?? {});
      const original = serialized.length > 2000 ? `${serialized.slice(0, 2000)}...` : serialized;
      const sanitized = original
        .replace(/"password"\s*:\s*".+?"/gi, "\"password\":\"[REDACTED]\"")
        .replace(/"token"\s*:\s*".+?"/gi, "\"token\":\"[REDACTED]\"")
        .replace(/"secret"\s*:\s*".+?"/gi, "\"secret\":\"[REDACTED]\"");
      return {
        output: value,
        profile,
        original: sanitized,
      };
    },
  };
}

export async function runAgentRuntime(input: AgentRuntimeInput): Promise<AgentRuntimeResult> {
  const now = input.now ?? defaultNow;
  const run = createRunRecord(input.actorId, input.goal, "planner", now);
  const maxRounds = input.maxRounds ?? 3;
  const planner = input.planner ?? defaultPlanner;
  const policyEngine = input.policyEngine
    ? { evaluate: input.policyEngine.evaluate.bind(input.policyEngine) }
    : { evaluate: defaultPolicy };
  const validate = input.validateOutput ?? defaultValidator;
  const toolRegistry: FoundryToolRegistry = input.tools || {};

  const runId = run.id;
  const requestId = `req_${crypto.randomUUID()}`;
  const execContext: AgentToolExecutionContext = {
    actorId: input.actorId,
    requestId,
    runId,
    step: "plan",
  };
  const sanitizer = createSanitizer();
  let currentArtifact: RunArtifact | undefined;

  const publish = async (evt: { stage: RunStage; type: "state" | "result"; payload: unknown }) => {
    await input.bus.publish({
      id: `${runId}_${evt.type}_${Date.now()}`,
      type: `agent.${evt.stage}.${evt.type}`,
      occurredAt: now(),
      actorId: input.actorId,
      payload: evt.payload,
    });
  };

  const writeAudit = async (result: {
    action: string;
    resourceType: string;
    resourceId: string;
    decision: "allow" | "deny" | "warn";
    reason: string;
  }) => {
    await input.audit.append({
      id: `audit_${crypto.randomUUID()}`,
      at: now(),
      actorId: input.actorId,
      action: result.action,
      resourceType: result.resourceType,
      resourceId: result.resourceId,
      result: result.decision,
      reason: result.reason,
      severity: result.decision === "allow" ? "info" : "warn",
    });
  };

  try {
    await publish({ stage: "PLAN", type: "state", payload: { event: "started" } });
    pushTransition(run, "PLAN", "started", `plan start: ${input.goal}`, now);
    const planned = await planner(input.goal, { toolCountHint: Object.keys(toolRegistry).length });
    const executeOrder = input.toolSequence?.length ? input.toolSequence : planned;
    run.tool = planned.join(",");
    run.planTools = executeOrder;
    pushTransition(run, "PLAN", "completed", `plan produced ${executeOrder.length} tools`, now);

    for (let round = 0; round < maxRounds; round += 1) {
      currentArtifact = undefined;
      for (const toolName of executeOrder) {
        const tool = toolRegistry[toolName];
        if (!tool) {
          throw new Error(`tool not found: ${toolName}`);
        }

        pushTransition(run, "ACT", "started", `act ${toolName}`, now);
        run.tool = toolName;
        run.verify = undefined;
        await publish({ stage: "ACT", type: "state", payload: { tool: toolName, step: "start" } });

        const policy = await policyEngine.evaluate({
          actorId: input.actorId,
          action: "agent_tool",
          resource: { type: "agent_tool", id: toolName },
          payload: input.initialInput,
          classification: "P2",
        });
        await writeAudit({
          action: "agent_tool",
          resourceType: "agent_tool",
          resourceId: toolName,
          decision: policy.allowed ? "allow" : "deny",
          reason: policy.reason,
        });
        if (!policy.allowed) {
          pushTransition(run, "ACT", "blocked", policy.reason, now);
          continue;
        }

        const result: FoundryToolResult | void = await tool.execute(input.initialInput ?? {}, {
          ...execContext,
          step: `act:${toolName}`,
          maxWritebackRows: 2000,
        });
        if (!result) {
          pushTransition(run, "ACT", "completed", `tool no output: ${toolName}`, now);
          continue;
        }
        if (result.artifact) {
          const sanitized = result.artifact;
          if (result.structuredFacts) {
            const redacted = sanitizer.sanitize(result.structuredFacts, "facts");
            sanitized.metadata = {
              ...(sanitized.metadata ?? {}),
              sanitizedForExternal: redacted,
            };
          }
          currentArtifact = sanitized;
          run.proposal = sanitized;
        }
        if (result.errors?.length) {
          await writeAudit({
            action: "agent_tool",
            resourceType: "agent_tool",
            resourceId: toolName,
            decision: "warn",
            reason: result.errors.join(", "),
          });
        }
        pushTransition(run, "ACT", "completed", `tool completed: ${toolName}`, now);
      }

      if (!currentArtifact) {
        pushTransition(run, "VERIFY", "blocked", "no artifact generated", now);
        continue;
      }

      pushTransition(run, "VERIFY", "started", "verify start", now);
      const validation = await validate(currentArtifact);
      const policy = await policyEngine.evaluate({
        actorId: input.actorId,
        action: "agent_artifact_write",
        resource: {
          type: "agent_artifact",
          id: currentArtifact.id ?? run.id,
        },
        payload: currentArtifact.jsonContent,
        classification: (currentArtifact.classification ?? "P2") as DataClassification,
      });

      run.verify = {
        allowed: validation.ok && policy.allowed,
        schemaValid: validation.ok,
        policyAllowed: policy.allowed,
        schemaErrors: validation.errors,
      };
      await writeAudit({
        action: "agent_artifact_write",
        resourceType: "agent_artifact",
        resourceId: currentArtifact.id ?? run.id,
        decision: policy.allowed && validation.ok ? "allow" : "deny",
        reason: validation.errors.length ? validation.errors.join(", ") : policy.reason,
      });

      if (!run.verify.allowed) {
        pushTransition(run, "VERIFY", "blocked", "verify failed", now);
        continue;
      }
      pushTransition(run, "VERIFY", "completed", "verify passed", now);

      pushTransition(run, "WRITEBACK", "started", "writeback start", now);
      const writebackResult = input.writeback
        ? await input.writeback({ actorId: input.actorId, goal: input.goal, artifact: currentArtifact })
        : { ok: true, detail: "noop writeback" };

      run.writeback = writebackResult;
      if (!writebackResult.ok) {
        pushTransition(run, "WRITEBACK", "failed", writebackResult.detail ?? "writeback failed", now);
        break;
      }
      pushTransition(run, "WRITEBACK", "completed", writebackResult.detail ?? "writeback done", now);
      pushTransition(run, "DONE", "completed", "run done", now);

      await publish({ stage: "WRITEBACK", type: "result", payload: { runId, artifact: currentArtifact } });
      await publish({ stage: "DONE", type: "state", payload: { status: "completed" } });
      return { run, final: currentArtifact };
    }

    pushTransition(run, "FAILED", "failed", "max rounds exhausted", now);
    return { run };
  } catch (error) {
    const message = error instanceof Error ? error.message : "agent runtime error";
    await writeAudit({
      action: "agent_run",
      resourceType: "agent_run",
      resourceId: run.id,
      decision: "deny",
      reason: message,
    });
    pushTransition(run, "FAILED", "failed", message, now);
    return { run };
  }
}

export function sanitizeForExternal(value: unknown, profile: "facts" | "summary" = "facts"): {
  output: unknown;
  profile: "facts" | "summary";
  original: string;
} {
  const sanitized = createSanitizer().sanitize(value, profile);
  return {
    output: sanitized.output,
    profile: sanitized.profile,
    original: sanitized.original ?? "",
  };
}
