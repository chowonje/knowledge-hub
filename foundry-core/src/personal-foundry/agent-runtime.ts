import crypto from "node:crypto";
import type { DataClassification } from "../types.js";
import type {
  AgentPlanStep,
  AgentRunEnvelope,
  AgentRuntimeInput,
  AgentTransition,
  PersonalAgentRuntime,
  PersonalAuditLog,
  PersonalPolicyEngine,
} from "./interfaces.js";

const POLICY_REDACTION_TEXT = "[REDACTED_BY_POLICY]";

export interface AgentRuntimeDependencies {
  policy: PersonalPolicyEngine;
  audit: PersonalAuditLog;
  now?: () => string;
}

function defaultNow(): string {
  return new Date().toISOString();
}

function defaultPlanner(goal: string): Promise<AgentPlanStep[]> {
  return Promise.resolve([
    {
      order: 1,
      toolName: "search_knowledge",
      objective: "collect evidence",
      input: { query: goal, top_k: 5 },
    },
    {
      order: 2,
      toolName: "ask_knowledge",
      objective: "synthesize answer",
      input: { question: goal, top_k: 5 },
    },
  ]);
}

function defaultSchemaValidator(artifact: unknown): Promise<{ ok: boolean; errors: string[] }> {
  if (!artifact || typeof artifact !== "object") {
    return Promise.resolve({ ok: false, errors: ["artifact must be an object"] });
  }
  return Promise.resolve({ ok: true, errors: [] });
}

function pushTransition(
  transitions: AgentTransition[],
  stage: AgentTransition["stage"],
  status: AgentTransition["status"],
  message: string,
  at: string
): void {
  transitions.push({ stage, status, message, at });
}

export class PlanActVerifyRuntime implements PersonalAgentRuntime {
  private readonly policy: PersonalPolicyEngine;
  private readonly audit: PersonalAuditLog;
  private readonly now: () => string;

  constructor(deps: AgentRuntimeDependencies) {
    this.policy = deps.policy;
    this.audit = deps.audit;
    this.now = deps.now ?? defaultNow;
  }

  async run(input: AgentRuntimeInput): Promise<AgentRunEnvelope> {
    const runId = `run_${crypto.randomUUID()}`;
    const createdAt = this.now();
    const transitions: AgentTransition[] = [];
    const maxRounds = input.maxRounds ?? 2;
    const planner = input.planner ?? defaultPlanner;
    const schemaValidator = input.schemaValidate ?? defaultSchemaValidator;

    let stage: AgentTransition["stage"] = "PLAN";
    let status: AgentRunEnvelope["status"] = "running";
    let plan: AgentPlanStep[] = [];
    let latestArtifact: { jsonContent: unknown; classification: DataClassification } | null = null;
    let verify: AgentRunEnvelope["verify"];
    let writeback: AgentRunEnvelope["writeback"];

    pushTransition(transitions, "PLAN", "started", "planning started", this.now());
    plan = await planner(input.goal);
    pushTransition(transitions, "PLAN", "completed", `planned ${plan.length} steps`, this.now());

    for (let round = 0; round < maxRounds; round += 1) {
      let roundProducedArtifact = false;

      for (const step of plan) {
        stage = "ACT";
        pushTransition(transitions, "ACT", "started", `tool ${step.toolName} start`, this.now());

        const tool = input.tools[step.toolName];
        if (!tool) {
          pushTransition(transitions, "ACT", "failed", `tool not found: ${step.toolName}`, this.now());
          status = "failed";
          stage = "FAILED";
          return {
            runId,
            status,
            goal: input.goal,
            stage,
            transitions,
            plan,
            artifact: latestArtifact,
            verify,
            writeback,
            createdAt,
            updatedAt: this.now(),
          };
        }

        const actPolicy = await this.policy.evaluate({
          actorId: input.actorId,
          action: "agent_act",
          resourceType: "tool",
          resourceId: step.toolName,
          payload: step.input,
          runId,
          requestId: input.requestId,
        });

        await this.audit.append({
          actorId: input.actorId,
          action: "agent_act",
          resourceType: "tool",
          resourceId: step.toolName,
          allowed: actPolicy.allowed,
          reason: actPolicy.reason,
          classification: actPolicy.classification,
          runId,
          requestId: input.requestId,
          metadata: { stepOrder: step.order },
        });

        if (!actPolicy.allowed) {
          pushTransition(transitions, "ACT", "blocked", actPolicy.reason, this.now());
          continue;
        }

        const result = await tool.execute(step.input, {
          actorId: input.actorId,
          runId,
          requestId: input.requestId,
          step: `${step.order}:${step.toolName}`,
        });

        if (result?.artifact) {
          const classification = result.artifact.classification ?? this.policy.classify(result.artifact.jsonContent);
          latestArtifact = {
            jsonContent: result.artifact.jsonContent,
            classification,
          };
          roundProducedArtifact = true;
        }

        if (result?.errors?.length) {
          pushTransition(transitions, "ACT", "completed", `tool completed with warnings: ${result.errors.join(", ")}`, this.now());
        } else {
          pushTransition(transitions, "ACT", "completed", `tool ${step.toolName} completed`, this.now());
        }
      }

      stage = "VERIFY";
      pushTransition(transitions, "VERIFY", "started", "verify started", this.now());

      if (!roundProducedArtifact || !latestArtifact) {
        verify = {
          policyAllowed: false,
          schemaValid: false,
          schemaErrors: ["no artifact generated"],
        };
        pushTransition(transitions, "VERIFY", "blocked", "no artifact generated", this.now());
        continue;
      }

      const schema = await schemaValidator(latestArtifact.jsonContent);
      const verifyPolicy = await this.policy.evaluate({
        actorId: input.actorId,
        action: "agent_verify",
        resourceType: "artifact",
        resourceId: runId,
        payload: latestArtifact.jsonContent,
        classification: latestArtifact.classification,
        runId,
        requestId: input.requestId,
      });
      latestArtifact.classification = verifyPolicy.classification;

      verify = {
        policyAllowed: verifyPolicy.allowed,
        schemaValid: schema.ok,
        schemaErrors: schema.errors,
      };

      await this.audit.append({
        actorId: input.actorId,
        action: "agent_verify",
        resourceType: "artifact",
        resourceId: runId,
        allowed: verifyPolicy.allowed && schema.ok,
        reason: schema.ok ? verifyPolicy.reason : `schema invalid: ${schema.errors.join("; ")}`,
        classification: latestArtifact.classification,
        runId,
        requestId: input.requestId,
      });

      if (!verifyPolicy.allowed || !schema.ok) {
        if (verifyPolicy.classification === "P0") {
          latestArtifact.jsonContent = POLICY_REDACTION_TEXT;
        }
        pushTransition(transitions, "VERIFY", "blocked", "verify failed", this.now());
        continue;
      }

      pushTransition(transitions, "VERIFY", "completed", "verify passed", this.now());

      stage = "WRITEBACK";
      pushTransition(transitions, "WRITEBACK", "started", "writeback started", this.now());

      const writebackPolicy = await this.policy.evaluate({
        actorId: input.actorId,
        action: "agent_writeback",
        resourceType: "artifact",
        resourceId: runId,
        payload: latestArtifact.jsonContent,
        classification: latestArtifact.classification,
        runId,
        requestId: input.requestId,
      });
      latestArtifact.classification = writebackPolicy.classification;

      if (!writebackPolicy.allowed) {
        writeback = {
          ok: false,
          detail: writebackPolicy.reason,
        };
        await this.audit.append({
          actorId: input.actorId,
          action: "agent_writeback",
          resourceType: "artifact",
          resourceId: runId,
          allowed: false,
          reason: writebackPolicy.reason,
          classification: latestArtifact.classification,
          runId,
          requestId: input.requestId,
        });

        pushTransition(transitions, "WRITEBACK", "blocked", writebackPolicy.reason, this.now());
        continue;
      }

      writeback = input.writeback
        ? await input.writeback({
            actorId: input.actorId,
            goal: input.goal,
            artifact: latestArtifact.jsonContent,
            classification: latestArtifact.classification,
            runId,
          })
        : { ok: true, detail: "noop" };

      await this.audit.append({
        actorId: input.actorId,
        action: "agent_writeback",
        resourceType: "artifact",
        resourceId: runId,
        allowed: writeback.ok,
        reason: writeback.detail ?? "writeback done",
        classification: latestArtifact.classification,
        runId,
        requestId: input.requestId,
      });

      if (!writeback.ok) {
        pushTransition(transitions, "WRITEBACK", "failed", writeback.detail ?? "writeback failed", this.now());
        status = "failed";
        stage = "FAILED";
        return {
          runId,
          status,
          goal: input.goal,
          stage,
          transitions,
          plan,
          artifact: latestArtifact,
          verify,
          writeback,
          createdAt,
          updatedAt: this.now(),
        };
      }

      pushTransition(transitions, "WRITEBACK", "completed", writeback.detail ?? "writeback done", this.now());
      pushTransition(transitions, "DONE", "completed", "run completed", this.now());

      status = "completed";
      stage = "DONE";

      return {
        runId,
        status,
        goal: input.goal,
        stage,
        transitions,
        plan,
        artifact: latestArtifact,
        verify,
        writeback,
        createdAt,
        updatedAt: this.now(),
      };
    }

    status = "blocked";
    stage = "VERIFY";

    return {
      runId,
      status,
      goal: input.goal,
      stage,
      transitions,
      plan,
      artifact: latestArtifact,
      verify,
      writeback,
      createdAt,
      updatedAt: this.now(),
    };
  }
}
