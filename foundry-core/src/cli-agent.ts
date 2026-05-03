#!/usr/bin/env node
/*
Authority contract:
- Python remains the final authority for validation, policy gating, normalized
  run payload acceptance, and product-facing CLI/MCP outputs.
- TypeScript remains the authority for delegated execution flow and
  `.khub/personal-foundry/*` state in `foundry-core`.
- The bridge remains subprocess + JSON stdout only; this entrypoint normalizes
  local envelopes for self-protection but does not take ownership of Python's
  stores or add inner retry loops around Python bridge calls.
*/
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { runKnowledgeHubCli } from "./adapters/knowledge-hub-cli.js";
import { createFeatureRuntime } from "./features.js";
import { KnowledgeHubConnector, KnowledgeHubCLICommand } from "./adapters/knowledge-hub-connector.js";
import { KnowledgeHubPersonalConnectorBridge } from "./adapters/knowledge-hub-personal-connector.js";
import { PlanActVerifyRuntime } from "./personal-foundry/agent-runtime.js";
import { JsonlAuditLog } from "./personal-foundry/audit-log.js";
import { DefaultConnectorRunner, JsonFileCursorStore, JsonFileIdempotencyStore } from "./personal-foundry/connector-runner.js";
import { JsonlEventBus } from "./personal-foundry/event-bus.js";
import { LocalPolicyEngine } from "./personal-foundry/policy-engine.js";
import { LocalOntologyStore } from "./personal-foundry/ontology-store.js";
import type { FeatureQuery } from "./contracts/index.js";
import {
  AgentRuntimeInput as PersonalAgentRuntimeInput,
  AgentPlanStep as PersonalAgentPlanStep,
  AgentRunEnvelope as PersonalAgentRunEnvelope,
  AgentTool as PersonalAgentTool,
  AgentToolResult as PersonalAgentToolResult,
  ConnectorCursorStore as PersonalConnectorCursorStore,
} from "./personal-foundry/interfaces.js";

const argv = process.argv.slice(2);

const KNOWN_COMMANDS = new Set(["run", "sync", "feature", "help"]);

function parseTopLevel(args: string[]): {
  projectRoot: string;
  cliCommand: string;
  command: string;
  commandArgs: string[];
} {
  if (args.length === 0) {
    return { projectRoot: process.cwd(), cliCommand: "python", command: "help", commandArgs: [] };
  }

  let offset = 0;
  let projectRoot = process.cwd();
  let cliCommand = "python";

  if (!KNOWN_COMMANDS.has(args[0]) && !args[0].startsWith("-")) {
    projectRoot = args[0];
    offset = 1;
  }

  if (offset < args.length && !KNOWN_COMMANDS.has(args[offset]) && !args[offset].startsWith("-")) {
    cliCommand = args[offset];
    offset += 1;
  }

  const command = args[offset] ?? "help";
  const commandArgs = args.slice(offset + 1);
  return { projectRoot, cliCommand, command, commandArgs };
}

const { projectRoot, cliCommand, command, commandArgs } = parseTopLevel(argv);

type AgentRole = "planner" | "researcher" | "analyst" | "summarizer" | "auditor" | "coach";
type OrchestratorMode = "single-pass" | "adaptive" | "strict";

const DEFAULT_AGENT_ROLE: AgentRole = "planner";
const DEFAULT_ORCHESTRATOR_MODE: OrchestratorMode = "adaptive";
const POLICY_REDATION_TEXT = "[REDACTED_BY_POLICY]";
export const AGENT_RUN_PLAYBOOK_SCHEMA = "knowledge-hub.foundry.agent.run.playbook.v1";

function runCli(args: string[]): string {
  return runKnowledgeHubCli(projectRoot, cliCommand, args);
}

function supportsAgentSyncCommand(): boolean {
  try {
    const help = runCli(["--help"]);
    return /(?:^|\n)\s+agent\s+/m.test(help);
  } catch {
    return false;
  }
}

function parseCliJsonOutput(raw: string): Record<string, unknown> {
  const text = String(raw ?? "").trim();
  if (!text) return {};
  try {
    const parsed = JSON.parse(text);
    return typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : { value: parsed };
  } catch {
    const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        const parsed = JSON.parse(lines[index]);
        if (typeof parsed === "object" && parsed !== null) {
          return parsed as Record<string, unknown>;
        }
      } catch {
        continue;
      }
    }
  }
  return { raw: text };
}

function classifyTaskMode(goal: string): "coding" | "design" | "debug" | "knowledge" {
  const normalized = goal.trim().toLowerCase();
  if (!normalized) return "knowledge";
  if (/(?:^|\s)(?:[\w.-]+\/[\w./-]+|[\w.-]+\.(?:md|py|ts|tsx|js|jsx|json|yaml|yml|toml))(?:\s|$)/.test(normalized)) {
    return "coding";
  }
  if (["bug", "debug", "error", "exception", "fail", "fix", "traceback", "버그", "오류", "실패"].some((token) => normalized.includes(token))) {
    return "debug";
  }
  if (["architecture", "design", "interface", "migration", "pattern", "refactor", "spec", "구조", "설계", "리팩터"].some((token) => normalized.includes(token))) {
    return "design";
  }
  if (["change", "code", "implement", "patch", "script", "test", "update", "write", "코드", "구현", "수정", "테스트"].some((token) => normalized.includes(token))) {
    return "coding";
  }
  return "knowledge";
}

function parseAgentRunCommand(inputArgs: string[]): {
  goal: string;
  maxRounds: number;
  dryRun: boolean;
  dumpJson: boolean;
  role: AgentRole;
  reportPath: string | undefined;
  orchestratorMode: OrchestratorMode;
  repoPath: string | undefined;
  includeWorkspace: boolean | undefined;
  maxWorkspaceFiles: number;
} {
  let maxRounds = 3;
  let dryRun = false;
  let dumpJson = false;
  let role = DEFAULT_AGENT_ROLE;
  let reportPath: string | undefined;
  let orchestratorMode = DEFAULT_ORCHESTRATOR_MODE;
  let repoPath: string | undefined;
  let includeWorkspace: boolean | undefined;
  let maxWorkspaceFiles = 8;
  const goalParts: string[] = [];

  for (let i = 0; i < inputArgs.length; i += 1) {
    const token = inputArgs[i];

    if (token === "--goal") {
      if (i + 1 < inputArgs.length) {
        goalParts.push(inputArgs[i + 1]);
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--goal=")) {
      goalParts.push(token.slice(7));
      continue;
    }

    if (token === "--max-rounds") {
      const value = inputArgs[i + 1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        maxRounds = parsed;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--max-rounds=")) {
      const value = token.split("=", 2)[1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        maxRounds = parsed;
      }
      continue;
    }

    if (token === "--dry-run") {
      dryRun = true;
      continue;
    }

    if (token === "--dump-json") {
      dumpJson = true;
      continue;
    }

    if (token === "--role") {
      const value = inputArgs[i + 1];
      if (value) {
        role = value.toLowerCase() as AgentRole;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--role=")) {
      role = token.slice(7).toLowerCase() as AgentRole;
      continue;
    }

    if (token === "--orchestrator-mode") {
      const value = inputArgs[i + 1];
      if (value) {
        orchestratorMode = value.toLowerCase() as OrchestratorMode;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--orchestrator-mode=")) {
      orchestratorMode = token.split("=", 2)[1].toLowerCase() as OrchestratorMode;
      continue;
    }

    if (token === "--report-path") {
      reportPath = inputArgs[i + 1];
      i += 1;
      continue;
    }

    if (token.startsWith("--report-path=")) {
      reportPath = token.split("=", 2)[1];
      continue;
    }

    if (token === "--repo-path") {
      repoPath = inputArgs[i + 1];
      i += 1;
      continue;
    }

    if (token.startsWith("--repo-path=")) {
      repoPath = token.split("=", 2)[1];
      continue;
    }

    if (token === "--include-workspace") {
      includeWorkspace = true;
      continue;
    }

    if (token === "--no-include-workspace") {
      includeWorkspace = false;
      continue;
    }

    if (token === "--max-workspace-files") {
      const value = inputArgs[i + 1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        maxWorkspaceFiles = parsed;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--max-workspace-files=")) {
      const value = token.split("=", 2)[1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        maxWorkspaceFiles = parsed;
      }
      continue;
    }

    if (!token.startsWith("-")) {
      goalParts.push(token);
    }
  }

  return {
    goal: goalParts.join(" ").trim(),
    maxRounds,
    dryRun,
    dumpJson,
    role,
    reportPath,
    orchestratorMode,
    repoPath,
    includeWorkspace,
    maxWorkspaceFiles,
  };
}

function normalizeAgentRole(value: string | undefined): AgentRole {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "researcher" || normalized === "analyst" || normalized === "summarizer" || normalized === "auditor" || normalized === "coach") {
    return normalized;
  }
  return DEFAULT_AGENT_ROLE;
}

function normalizeOrchestratorMode(value: string | undefined): OrchestratorMode {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "single-pass" || normalized === "single" || normalized === "singlepass") {
    return "single-pass";
  }
  if (normalized === "strict") {
    return "strict";
  }
  return DEFAULT_ORCHESTRATOR_MODE;
}

export function normalizeFoundryRunEnvelope(input: AgentRunEnvelope): AgentRunEnvelope {
  if (!input.verify) {
    return input;
  }
  const policyAllowed = input.verify.policyAllowed;
  if (policyAllowed) {
    return input;
  }

  const errors = [...(input.verify.schemaErrors ?? [])];
  const policyError = "policy deny: P0 artifact blocked by local policy";
  if (!errors.includes(policyError)) {
    errors.push(policyError);
  }

  return {
    ...input,
    status: "blocked",
    stage: "VERIFY",
    verify: {
      ...input.verify,
      allowed: false,
      schemaValid: false,
      schemaErrors: errors,
      policyAllowed: false,
    },
    writeback: {
      ok: false,
      detail: "policy gate blocked",
    },
    artifact: input.artifact ? {
      ...input.artifact,
      jsonContent: POLICY_REDATION_TEXT,
      classification: "P0",
      generatedAt: input.artifact.generatedAt ?? new Date().toISOString(),
      metadata: {},
    } : null,
  };
}

interface SyncCommandInput {
  source?: SyncSource;
  limit?: number;
  cursor?: string;
  fresh?: boolean;
  stateFile?: string;
  eventLogPath?: string;
  persistState?: boolean;
  iterateSources?: boolean;
  failFast?: boolean;
  classificationGate?: "P0" | "P1" | "P2" | "P3";
}

const SYNC_STATE_FILE = ".foundry-sync-state.json";
const EVENT_LOG_FILE = ".foundry-ontology-events.jsonl";
const SYNC_SOURCES = ["all", "note", "paper", "web", "expense", "sleep", "schedule", "behavior"] as const;

type SyncSource = (typeof SYNC_SOURCES)[number];

function parseSyncSource(value: string | undefined): SyncSource | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value === "notes" ? "note" : value;
  return (SYNC_SOURCES as readonly string[]).includes(normalized) ? normalized as SyncSource : undefined;
}

function getSyncStatePath(stateFile?: string): string {
  const raw = stateFile ?? SYNC_STATE_FILE;
  return isAbsolute(raw) ? raw : resolve(projectRoot, raw);
}

function getEventLogPath(eventLogPath?: string): string {
  const raw = eventLogPath ?? EVENT_LOG_FILE;
  return isAbsolute(raw) ? raw : resolve(projectRoot, raw);
}

function parseSyncCommand(inputArgs: string[]): SyncCommandInput {
  let source: SyncSource = "all";
  let limit: number | undefined;
  let cursor: string | undefined;
  let fresh = false;
  let stateFile: string | undefined;
  let eventLogPath: string | undefined;
  let persistState = true;
  let iterateSources = false;
  let failFast = false;
  let classificationGate: "P0" | "P1" | "P2" | "P3" | undefined;

  for (let i = 0; i < inputArgs.length; i += 1) {
    const token = inputArgs[i];

    if (token === "--classification-gate") {
      const value = (inputArgs[i + 1] ?? "").toUpperCase();
      if (value === "P0" || value === "P1" || value === "P2" || value === "P3") {
        classificationGate = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--classification-gate=")) {
      const value = token.split("=", 2)[1].toUpperCase();
      if (value === "P0" || value === "P1" || value === "P2" || value === "P3") {
        classificationGate = value;
      }
      continue;
    }

    if (token === "--source") {
      const value = parseSyncSource(inputArgs[i + 1]);
      if (value) {
        source = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--source=")) {
      const value = parseSyncSource(token.split("=", 2)[1]);
      if (value) {
        source = value;
      }
      continue;
    }

    if (token === "--cursor") {
      const value = inputArgs[i + 1];
      if (value) cursor = value;
      i += 1;
      continue;
    }

    if (token.startsWith("--cursor=")) {
      cursor = token.split("=", 2)[1];
      continue;
    }

    if (token === "--fresh") {
      fresh = true;
      continue;
    }

    if (token === "--no-save-state") {
      persistState = false;
      continue;
    }

    if (token === "--all-sources") {
      iterateSources = true;
      continue;
    }

    if (token === "--fail-fast") {
      failFast = true;
      continue;
    }

    if (token === "--state-file") {
      const value = inputArgs[i + 1];
      if (value) stateFile = value;
      i += 1;
      continue;
    }

    if (token.startsWith("--state-file=")) {
      stateFile = token.split("=", 2)[1];
      continue;
    }

    if (token === "--event-log" || token === "--event-log-file" || token === "--event-log-path") {
      const value = inputArgs[i + 1];
      if (value) {
        eventLogPath = value;
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--event-log=") || token.startsWith("--event-log-file=") || token.startsWith("--event-log-path=")) {
      eventLogPath = token.split("=", 2)[1];
      continue;
    }

    if (token === "--limit") {
      const value = inputArgs[i + 1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        limit = parsed;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--limit=")) {
      const value = token.split("=", 2)[1];
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed) && parsed > 0) {
        limit = parsed;
      }
    }
  }

  return {
    source,
    limit,
    cursor,
    fresh,
    stateFile,
    eventLogPath,
    persistState,
    iterateSources,
    failFast,
    classificationGate,
  };
}

interface FeatureCommandInput {
  feature?: string;
  intent: FeatureQuery["intent"];
  source?: string;
  days?: number;
  from?: string;
  to?: string;
  topK?: number;
  limit?: number;
  expenseThreshold?: number;
  minSleepHours?: number;
  stateFile?: string;
  ontologyEventsPath?: string;
  eventLogPath?: string;
  dumpJson: boolean;
}

function parseFeatureCommand(inputArgs: string[]): FeatureCommandInput {
  let feature: string | undefined;
  let intent: FeatureQuery["intent"] = "analyze";
  let source: string | undefined;
  let days: number | undefined;
  let from: string | undefined;
  let to: string | undefined;
  let topK: number | undefined;
  let limit: number | undefined;
  let expenseThreshold: number | undefined;
  let minSleepHours: number | undefined;
  let stateFile: string | undefined;
  let ontologyEventsPath: string | undefined;
  let eventLogPath: string | undefined;
  let dumpJson = true;

  for (let i = 0; i < inputArgs.length; i += 1) {
    const token = inputArgs[i];

    if (token === "--intent") {
      const value = (inputArgs[i + 1] ?? "").toLowerCase();
      if (["read", "analyze", "forecast", "summarize", "compare", "alert"].includes(value)) {
        intent = value as FeatureQuery["intent"];
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--intent=")) {
      const value = token.split("=", 2)[1].toLowerCase();
      if (["read", "analyze", "forecast", "summarize", "compare", "alert"].includes(value)) {
        intent = value as FeatureQuery["intent"];
      }
      continue;
    }

    if (token === "--source") {
      const value = inputArgs[i + 1];
      if (value) source = value;
      i += 1;
      continue;
    }

    if (token.startsWith("--source=")) {
      source = token.split("=", 2)[1];
      continue;
    }

    if (token === "--days") {
      const value = Number.parseInt(inputArgs[i + 1], 10);
      if (!Number.isNaN(value) && value > 0) {
        days = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--days=")) {
      const value = Number.parseInt(token.split("=", 2)[1], 10);
      if (!Number.isNaN(value) && value > 0) {
        days = value;
      }
      continue;
    }

    if (token === "--from") {
      const value = inputArgs[i + 1];
      if (value) from = value;
      i += 1;
      continue;
    }

    if (token.startsWith("--from=")) {
      from = token.split("=", 2)[1];
      continue;
    }

    if (token === "--to") {
      const value = inputArgs[i + 1];
      if (value) to = value;
      i += 1;
      continue;
    }

    if (token.startsWith("--to=")) {
      to = token.split("=", 2)[1];
      continue;
    }

    if (token === "--top-k" || token === "--top_k") {
      const value = Number.parseInt(inputArgs[i + 1], 10);
      if (!Number.isNaN(value) && value > 0) {
        topK = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--top-k=") || token.startsWith("--top_k=")) {
      const value = Number.parseInt(token.split("=", 2)[1], 10);
      if (!Number.isNaN(value) && value > 0) {
        topK = value;
      }
      continue;
    }

    if (token === "--limit") {
      const value = Number.parseInt(inputArgs[i + 1], 10);
      if (!Number.isNaN(value) && value > 0) {
        limit = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--limit=")) {
      const value = Number.parseInt(token.split("=", 2)[1], 10);
      if (!Number.isNaN(value) && value > 0) {
        limit = value;
      }
      continue;
    }

    if (token === "--expense-threshold") {
      const value = Number.parseFloat(inputArgs[i + 1]);
      if (!Number.isNaN(value)) {
        expenseThreshold = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--expense-threshold=")) {
      const value = Number.parseFloat(token.split("=", 2)[1]);
      if (!Number.isNaN(value)) {
        expenseThreshold = value;
      }
      continue;
    }

    if (token === "--min-sleep-hours") {
      const value = Number.parseFloat(inputArgs[i + 1]);
      if (!Number.isNaN(value)) {
        minSleepHours = value;
      }
      i += 1;
      continue;
    }

    if (token.startsWith("--min-sleep-hours=")) {
      const value = Number.parseFloat(token.split("=", 2)[1]);
      if (!Number.isNaN(value)) {
        minSleepHours = value;
      }
      continue;
    }

    if (token === "--event-log" || token === "--event-log-file" || token === "--event-log-path") {
      const value = inputArgs[i + 1];
      if (value) {
        eventLogPath = value;
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--event-log=") || token.startsWith("--event-log-file=") || token.startsWith("--event-log-path=")) {
      eventLogPath = token.split("=", 2)[1];
      continue;
    }

    if (token === "--state-file") {
      const value = inputArgs[i + 1];
      if (value) {
        stateFile = value;
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--state-file=")) {
      stateFile = token.split("=", 2)[1];
      continue;
    }

    if (token === "--ontology-events-path") {
      const value = inputArgs[i + 1];
      if (value) {
        ontologyEventsPath = value;
        i += 1;
      }
      continue;
    }

    if (token.startsWith("--ontology-events-path=")) {
      ontologyEventsPath = token.split("=", 2)[1];
      continue;
    }

    if (token === "--no-json") {
      dumpJson = false;
      continue;
    }

    if (token === "--json") {
      dumpJson = true;
      continue;
    }

    if (!token.startsWith("-") && !feature) {
      feature = token;
    }
  }

  return {
    feature,
    intent,
    source,
    days,
    from,
    to,
    topK,
    limit,
    expenseThreshold,
    minSleepHours,
    stateFile,
    ontologyEventsPath,
    eventLogPath,
    dumpJson,
  };
}

interface AgentPlaybookStep {
  order: number;
  tool: "ask_knowledge" | "search_knowledge" | "build_task_context";
  objective: string;
  rationale: string;
  inputs?: Record<string, unknown>;
}

interface AgentPlaybook {
  schema: typeof AGENT_RUN_PLAYBOOK_SCHEMA;
  source: "foundry-core/cli-agent";
  goal: string;
  role: AgentRole;
  orchestratorMode: OrchestratorMode;
  maxRounds: number;
  assumptions: string[];
  warnings: string[];
  steps: AgentPlaybookStep[];
  generatedAt: string;
}

export function buildAgentPlaybook(
  goal: string,
  role: AgentRole,
  orchestratorMode: OrchestratorMode,
  maxRounds: number,
  options: {
    repoPath?: string;
    includeWorkspace?: boolean;
    maxWorkspaceFiles?: number;
  } = {},
) {
  const plannerGoal = goal.toLowerCase();
  const mode = classifyTaskMode(goal);
  const workspaceEnabled = options.includeWorkspace ?? Boolean(options.repoPath && mode !== "knowledge");
  const warnings: string[] = [];
  const needsSearch =
    role === "researcher" ||
    role === "analyst" ||
    role === "coach" ||
    role === "planner" ||
    ["찾아", "검색", "search", "추천", "리스트", "목록", "비교", "compare", "차이", "대조"].some((token) => plannerGoal.includes(token));

  const planTools =
    mode === "coding" || mode === "design" || mode === "debug"
      ? ["build_task_context", "ask_knowledge"]
      : needsSearch
        ? ["search_knowledge", "ask_knowledge"]
        : ["ask_knowledge"];

  if (orchestratorMode === "strict" && planTools.length < 2) {
    warnings.push("strict 모드 권장: 증거 검색 단계를 위해 목표에 '검색' 키워드를 넣어 주세요.");
  }
  if ((mode === "coding" || mode === "design" || mode === "debug") && !workspaceEnabled) {
    warnings.push("workspace context unavailable; task context will use persistent knowledge only");
  }

  const assumptions = [
    `role=${role}`,
    `orchestratorMode=${orchestratorMode}`,
    `taskMode=${mode}`,
    `workspaceIncluded=${workspaceEnabled ? "true" : "false"}`,
    "모든 도구 호출은 읽기 전용",
    "검증은 schema + policy gate 기준 적용",
  ];

  const steps = planTools.map((tool, index) => ({
    order: index + 1,
    tool: tool as AgentPlaybookStep["tool"],
    objective:
      tool === "build_task_context"
        ? "읽기 전용 작업 컨텍스트 조합"
        : tool === "search_knowledge"
          ? "증거 기반 후보 수집"
          : "증거 기반 최종 답변 생성",
    rationale:
      tool === "build_task_context"
        ? "Obsidian/논문/웹 근거와 현재 repo 컨텍스트를 한 번에 조합"
        : tool === "search_knowledge"
          ? "search-first로 참조 근거를 선행 수집"
          : "요약/답변 품질을 위한 최종 합성 단계",
    ...(tool === "build_task_context"
      ? {
          inputs: {
            repoPath: options.repoPath ?? "",
            includeWorkspace: workspaceEnabled,
            maxWorkspaceFiles: options.maxWorkspaceFiles ?? 8,
          },
        }
      : {}),
  }));

  return {
    planTools,
    playbook: {
      schema: AGENT_RUN_PLAYBOOK_SCHEMA,
      source: "foundry-core/cli-agent",
      goal,
      role,
      orchestratorMode,
      maxRounds,
      assumptions,
      warnings,
      steps,
      generatedAt: new Date().toISOString(),
    } as AgentPlaybook,
  };
}

function writeAgentRunReport(reportPath: string | undefined, runPayload: AgentRunEnvelope): void {
  if (!reportPath) return;

  const resolved = isAbsolute(reportPath) ? reportPath : resolve(projectRoot, reportPath);
  const dir = dirname(resolved);
  if (dir && dir !== "." && !existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const output = {
    schema: "knowledge-hub.foundry.agent.run.report.v1",
    generatedAt: new Date().toISOString(),
    source: "foundry-core/cli-agent",
    run: runPayload,
  };
  writeFileSync(resolved, JSON.stringify(output, null, 2), "utf8");
}

function getFeatureEventSourcePath(input: FeatureCommandInput): string {
  if (input.ontologyEventsPath) {
    return isAbsolute(input.ontologyEventsPath)
      ? input.ontologyEventsPath
      : resolve(projectRoot, input.ontologyEventsPath);
  }

  // Backward compatibility: explicit event log override still supported.
  if (input.eventLogPath) {
    return getEventLogPath(input.eventLogPath);
  }

  const statePath = getSyncStatePath(input.stateFile);
  return resolve(dirname(statePath), "ontology-store", "ontology.events.jsonl");
}

function buildFeatureQuery(input: FeatureCommandInput): FeatureQuery {
  const params: Record<string, unknown> = {};
  if (input.source) params.source = input.source;
  if (input.topK !== undefined) params.top_k = input.topK;
  if (input.limit !== undefined) params.limit = input.limit;
  if (input.expenseThreshold !== undefined) params.expenseThreshold = input.expenseThreshold;
  if (input.minSleepHours !== undefined) params.minSleepHours = input.minSleepHours;
  if (input.days !== undefined) params.days = input.days;

  return {
    name: input.feature ?? "",
    intent: input.intent,
    params,
    timeframe: input.from || input.to ? { from: input.from, to: input.to } : undefined,
  };
}

function formatHumanSummary(input: {
  runId: string;
  goal: string;
  status: string;
  stage: string;
  transitions: Array<{ stage: string; status: string; message: string; at: string; code?: string }>;
  summaryMessage: string;
  artifact?: string;
  errors?: string[];
}): string {
  const lines = [
    `[runId] ${input.runId}`,
    `[goal] ${input.goal}`,
    `[status] ${input.status}`,
    `[stage] ${input.stage}`,
    `[summary] ${input.summaryMessage}`,
  ];

  if (input.artifact) {
    lines.push(`[artifact] ${input.artifact}`);
  }
  if (input.errors?.length) {
    lines.push("[errors]");
    lines.push(...input.errors);
  }
  if (input.transitions.length) {
    lines.push("[trace]");
    for (const t of input.transitions) {
      lines.push(`- ${t.at} ${t.stage}.${t.status}: ${t.message}`);
    }
  }
  return lines.join("\n");
}

interface AgentRunEnvelope {
  schema: "knowledge-hub.foundry.agent.run.result.v1";
  source: "foundry-core/cli-agent";
  runId: string;
  status: "running" | "completed" | "blocked" | "failed";
  goal: string;
  role: AgentRole;
  orchestratorMode: OrchestratorMode;
  stage: string;
  tool?: string;
  plan?: string[];
  playbook?: AgentPlaybook;
  transitions: Array<{ stage: string; status: string; message: string; at: string; code?: string }>;
  verify?: {
    allowed: boolean;
    schemaValid: boolean;
    policyAllowed: boolean;
    schemaErrors: string[];
  };
  writeback?: {
    ok: boolean;
    detail?: string;
  };
  artifact?: {
    id?: string;
    jsonContent?: unknown;
    classification?: string;
    generatedAt?: string;
    metadata?: Record<string, unknown>;
  } | null;
  createdAt: string;
  updatedAt: string;
  dryRun: boolean;
}

function buildEnvelope(result: PersonalAgentRunEnvelope, options: {
  dryRun?: boolean;
  role?: AgentRole;
  orchestratorMode?: OrchestratorMode;
  playbook?: AgentPlaybook;
}): AgentRunEnvelope {
  const planTools = result.plan.map((step) => step.toolName);
  const latestTool = planTools.length > 0 ? planTools[planTools.length - 1] : undefined;
  const policyAllowed = result.verify?.policyAllowed ?? false;
  const schemaValid = result.verify?.schemaValid ?? false;
  const schemaErrors = result.verify?.schemaErrors ?? [];

  return {
    schema: "knowledge-hub.foundry.agent.run.result.v1",
    source: "foundry-core/cli-agent",
    runId: result.runId,
    status: result.status,
    goal: result.goal,
    role: options.role ?? DEFAULT_AGENT_ROLE,
    orchestratorMode: options.orchestratorMode ?? DEFAULT_ORCHESTRATOR_MODE,
    stage: result.stage,
    tool: latestTool,
    plan: planTools,
    playbook: options.playbook,
    transitions: result.transitions.map((transition) => ({
      stage: transition.stage,
      status: transition.status,
      message: transition.message,
      at: transition.at,
      code: `${transition.stage}.${transition.status}`,
    })),
    verify: result.verify ? {
      allowed: policyAllowed && schemaValid,
      schemaValid,
      policyAllowed,
      schemaErrors,
    } : undefined,
    writeback: result.writeback,
    artifact: result.artifact
      ? {
          jsonContent: result.artifact.jsonContent,
          classification: result.artifact.classification,
          generatedAt: result.updatedAt,
        }
      : null,
    createdAt: result.createdAt,
    updatedAt: result.updatedAt,
    dryRun: !!options.dryRun,
  };
}

function buildPlannerInput(
  toolName: string,
  goal: string,
  options: {
    repoPath?: string;
    includeWorkspace?: boolean;
    maxWorkspaceFiles?: number;
  } = {},
): Record<string, unknown> {
  if (toolName === "build_task_context") {
    return {
      goal,
      repo_path: options.repoPath ?? undefined,
      include_workspace: options.includeWorkspace ?? false,
      max_workspace_files: options.maxWorkspaceFiles ?? 8,
      max_knowledge_hits: 5,
    };
  }
  if (toolName === "search_knowledge") {
    return { query: goal, top_k: 5 };
  }
  return { question: goal, top_k: 5 };
}

function buildPlannerSteps(
  goal: string,
  planTools: string[],
  options: {
    repoPath?: string;
    includeWorkspace?: boolean;
    maxWorkspaceFiles?: number;
  } = {},
): PersonalAgentPlanStep[] {
  return planTools.map((toolName, index) => ({
    order: index + 1,
    toolName,
    objective:
      toolName === "build_task_context"
        ? "assemble task context"
        : toolName === "search_knowledge"
          ? "collect evidence"
          : "synthesize answer",
    input: buildPlannerInput(toolName, goal, options),
  }));
}

async function makeKnowledgeTools(): Promise<Record<string, PersonalAgentTool>> {
  return {
    build_task_context: {
      name: "build_task_context",
      async execute(input: Record<string, unknown>): Promise<PersonalAgentToolResult | void> {
        const goal = String(input.goal ?? input.query ?? input.question ?? "");
        if (!goal) return;
        const args = ["agent", "context", goal, "--json"];
        const repoPath = String(input.repo_path ?? "");
        if (repoPath) {
          args.push("--repo-path", repoPath);
        }
        if (input.include_workspace === true) {
          args.push("--include-workspace");
        } else if (input.include_workspace === false) {
          args.push("--no-include-workspace");
        }
        if (input.max_workspace_files !== undefined) {
          args.push("--max-workspace-files", String(input.max_workspace_files));
        }
        if (input.max_knowledge_hits !== undefined) {
          args.push("--max-knowledge-hits", String(input.max_knowledge_hits));
        }
        const result = parseCliJsonOutput(runCli(args));
        return {
          artifact: {
            jsonContent: result,
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
    ask_knowledge: {
      name: "ask_knowledge",
      async execute(input: Record<string, unknown>): Promise<PersonalAgentToolResult | void> {
        const question = String(input.question ?? "");
        if (!question) return;
        const result = runCli(["ask", question, "--top-k", String(input.top_k ?? 5)]);
        return {
          artifact: {
            jsonContent: { answer: result, source: "knowledge-hub" },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
    search_knowledge: {
      name: "search_knowledge",
      async execute(input: Record<string, unknown>): Promise<PersonalAgentToolResult | void> {
        const query = String(input.query ?? "");
        if (!query) return;
        const result = runCli(["search", query, "--top-k", String(input.top_k ?? 5)]);
        return {
          artifact: {
            jsonContent: { query, result },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
  };
}

async function runAgentGoal(
  goal: string,
  options: {
    maxRounds?: number;
    dryRun?: boolean;
    dumpJson?: boolean;
    role?: string;
    reportPath?: string;
    orchestratorMode?: string;
    repoPath?: string;
    includeWorkspace?: boolean;
    maxWorkspaceFiles?: number;
  }
) {
  const tools = await makeKnowledgeTools();
  const role = normalizeAgentRole(options.role);
  const orchestratorMode = normalizeOrchestratorMode(options.orchestratorMode);
  const taskMode = classifyTaskMode(goal);
  const workspaceEnabled = options.includeWorkspace ?? Boolean(options.repoPath && taskMode !== "knowledge");
  const { planTools, playbook } = buildAgentPlaybook(
    goal,
    role,
    orchestratorMode,
    options.maxRounds ?? 3,
    {
      repoPath: options.repoPath,
      includeWorkspace: workspaceEnabled,
      maxWorkspaceFiles: options.maxWorkspaceFiles ?? 8,
    }
  );
  const plannerSteps = buildPlannerSteps(goal, planTools, {
    repoPath: options.repoPath,
    includeWorkspace: workspaceEnabled,
    maxWorkspaceFiles: options.maxWorkspaceFiles ?? 8,
  });
  const policy = new LocalPolicyEngine({
    outboundMaxClassification: "P2",
    writebackMaxClassification: "P1",
  });
  const audit = new JsonlAuditLog({
    logPath: resolve(projectRoot, ".khub", "foundry-runtime", "agent-audit.jsonl"),
  });
  const runtime = new PlanActVerifyRuntime({
    policy,
    audit,
  });

  const result = await runtime.run({
    actorId: "cli-user",
    requestId: `req_${Date.now()}`,
    goal,
    tools,
    maxRounds: options.maxRounds ?? 3,
    planner: async () => plannerSteps,
    schemaValidate: async (artifact: unknown) => {
      if (!artifact || typeof artifact !== "object") {
        return { ok: false, errors: ["artifact must be an object"] };
      }
      return { ok: true, errors: [] };
    },
    writeback: async ({ artifact }: Parameters<NonNullable<PersonalAgentRuntimeInput["writeback"]>>[0]) => {
      if (options.dryRun) {
        return { ok: true, detail: "dry-run: writeback skipped" };
      }
      if (!options.dryRun && !options.dumpJson) {
        console.log(`writeback: ${JSON.stringify(artifact).slice(0, 500)}...`);
      }
      return { ok: true, detail: "cli writeback complete" };
    },
  });

  const envelope = buildEnvelope(result, {
    dryRun: options.dryRun,
    role,
    orchestratorMode,
    playbook,
  });
  const normalizedEnvelope = normalizeFoundryRunEnvelope(envelope);
  if (options.reportPath) {
    writeAgentRunReport(options.reportPath, normalizedEnvelope);
  }
  if (options.dumpJson) {
    console.log(JSON.stringify(normalizedEnvelope));
    return;
  }

  const hasFinal = Boolean(normalizedEnvelope.artifact);
  const summaryMessage = hasFinal ? "artifact available" : "no artifact generated";
  const artifactText = normalizedEnvelope.artifact ? JSON.stringify(normalizedEnvelope.artifact.jsonContent).slice(0, 500) : undefined;
  const errors = normalizedEnvelope.verify?.schemaErrors?.length
    ? normalizedEnvelope.verify.schemaErrors
    : [];

  console.log(
    formatHumanSummary({
      runId: normalizedEnvelope.runId,
      goal: normalizedEnvelope.goal,
      status: normalizedEnvelope.status,
      stage: normalizedEnvelope.stage,
      transitions: normalizedEnvelope.transitions,
      summaryMessage,
      artifact: artifactText,
      errors,
    })
  );
}

async function runFeature(commandArgs: string[]) {
  const input = parseFeatureCommand(commandArgs);
  const runtime = createFeatureRuntime({
    eventLogPath: getFeatureEventSourcePath(input),
  });

  if (!input.feature) {
    console.log("usage:");
    console.log("  node cli-agent.ts <projectRoot> <pythonPath> feature <name|list> [--source <source>] [--days N] [--from ISO8601] [--to ISO8601] [--top-k N] [--limit N] [--intent read|analyze|forecast|summarize|compare|alert] [--state-file <path>] [--ontology-events-path <path>]");
    return;
  }

  if (input.feature === "list") {
    const featureNames = runtime.list();
    if (input.dumpJson) {
      console.log(JSON.stringify({
        schema: "knowledge-hub.foundry.feature.list.result.v1",
        source: "foundry-core/cli-agent",
        featureNames,
      }, null, 2));
    } else {
      console.log("available features:");
      for (const name of featureNames) {
        console.log(`- ${name}`);
      }
    }
    return;
  }

  const query = buildFeatureQuery(input);
  const result = await runtime.execute(query);
  if (input.dumpJson) {
    console.log(JSON.stringify(result));
    return;
  }

  console.log(JSON.stringify(result, null, 2));
}

async function syncKnowledgeHub(commandArgs: string[]) {
  const syncInput = parseSyncCommand(commandArgs);
  const source: SyncSource = syncInput.source ?? "all";
  if (!supportsAgentSyncCommand()) {
    const skipped = {
      schema: "knowledge-hub.foundry.connector.sync.result.v2",
      source: "foundry-core/cli-agent",
      status: "skipped",
      reason: "knowledge_hub CLI does not expose 'agent sync' command in this environment",
      requestedSource: source,
    };
    console.log(JSON.stringify(skipped, null, 2));
    return;
  }
  const statePath = getSyncStatePath(syncInput.stateFile);
  const eventLogPath = getEventLogPath(syncInput.eventLogPath);
  const stateDir = dirname(statePath);
  const idempotencyPath = resolve(stateDir, ".foundry-idempotency.json");
  const auditPath = resolve(stateDir, ".foundry-audit.jsonl");
  const sourcesToSync: SyncSource[] = syncInput.iterateSources
    ? ["note", "web", "paper"]
    : [source];

  if (syncInput.iterateSources && source !== "all") {
    console.error(`--all-sources was set; ignoring --source ${source} and syncing all sources.`);
  }

  const executor = new KnowledgeHubCLICommand(
    { projectRoot },
    (_cmd: string, args: string[]) => Promise.resolve(runCli(args.slice(1)))
  );
  const connectorForSource = (currentSource: SyncSource) => {
    const normalized = currentSource;
    const scoped: Record<string, { connectorId: string; fixedSource: string }> = {
      note: { connectorId: "knowledge-hub-vault", fixedSource: "note" },
      web: { connectorId: "knowledge-hub-web", fixedSource: "web" },
      paper: { connectorId: "knowledge-hub-arxiv", fixedSource: "paper" },
    };
    const current = scoped[normalized] ?? {
      connectorId: normalized === "all" ? "knowledge-hub" : `knowledge-hub-${normalized}`,
      fixedSource: normalized === "all" ? "all" : normalized,
    };
    return new KnowledgeHubPersonalConnectorBridge(
      new KnowledgeHubConnector(executor, undefined, {
        connectorId: current.connectorId,
        fixedSource: current.fixedSource as "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior",
      })
    );
  };

  const eventBus = new JsonlEventBus({
    logPath: eventLogPath,
  });
  const audit = new JsonlAuditLog({
    logPath: auditPath,
  });
  const policy = new LocalPolicyEngine({
    outboundMaxClassification: "P2",
    writebackMaxClassification: "P1",
  });
  const ontologyStore = new LocalOntologyStore({
    baseDir: resolve(stateDir, "ontology-store"),
  });
  const fileCursorStore = new JsonFileCursorStore(statePath);
  const cursorStore: PersonalConnectorCursorStore = {
    async get(connectorId, sourceKey) {
      if (syncInput.fresh) {
        return null;
      }
      return fileCursorStore.get(connectorId, sourceKey);
    },
    async set(entry) {
      if (!syncInput.persistState) {
        return;
      }
      await fileCursorStore.set(entry);
    },
  };

  const runner = new DefaultConnectorRunner({
    eventBus,
    audit,
    policy,
    ontologyStore,
    cursorStore,
    idempotencyStore: new JsonFileIdempotencyStore({ path: idempotencyPath }),
    classificationGate: syncInput.classificationGate ?? "P1",
  });

  const runs = [];

  for (const currentSource of sourcesToSync) {
    const connector = connectorForSource(currentSource);
    const result = await runner.run({
      connector,
      actorId: "cli-user",
      requestId: `cli_${Date.now()}_${connector.id}_${currentSource}`,
      cursor: syncInput.cursor,
      source: currentSource,
      pageLimit: syncInput.limit,
      dryRun: false,
    });

    runs.push({
      ...result,
      source: currentSource,
      deduplicated: result.status === "deduped",
    });

    if (syncInput.failFast && result.status === "failed") {
      break;
    }
  }

  if (runs.length === 1) {
    console.log(JSON.stringify(runs[0], null, 2));
    return;
  }

  const totalEmitted = runs.reduce((acc, run) => acc + (run.emittedEventCount ?? 0), 0);
  const dedupedCount = runs.filter((run) => run.status === "deduped").length;
  const failedCount = runs.filter((run) => run.status === "failed").length;
  const hasFailed = failedCount > 0;
  const overallStatus = hasFailed ? "failed" : dedupedCount === runs.length ? "deduped" : "done";
  console.log(JSON.stringify({
    mode: "multi-source",
    requestedSource: source,
    runs: sourcesToSync,
    totalEmitted,
    hasMore: runs.some((run) => run.hasMore),
    dedupedCount,
    failedCount,
    overallStatus,
    results: runs,
  }, null, 2));
}

async function main() {
  if (command === "run") {
    const parsed = parseAgentRunCommand(commandArgs);
    if (!parsed.goal) {
      console.log("usage:");
      console.log("  node cli-agent.ts [projectRoot] [pythonPath] run --goal \"질문\" [--role planner|researcher|analyst|summarizer|auditor|coach] [--orchestrator-mode single-pass|adaptive|strict] [--repo-path ./repo] [--include-workspace|--no-include-workspace] [--max-workspace-files 8] [--report-path ./path] [--max-rounds 3] [--dry-run] [--dump-json]");
      console.log("  node cli-agent.ts [projectRoot] [pythonPath] sync");
      process.exit(1);
      return;
    }
    await runAgentGoal(parsed.goal, {
      maxRounds: parsed.maxRounds,
      dryRun: parsed.dryRun,
      dumpJson: parsed.dumpJson,
      role: parsed.role,
      reportPath: parsed.reportPath,
      orchestratorMode: parsed.orchestratorMode,
      repoPath: parsed.repoPath,
      includeWorkspace: parsed.includeWorkspace,
      maxWorkspaceFiles: parsed.maxWorkspaceFiles,
    });
    return;
  }

  if (command === "sync") {
    await syncKnowledgeHub(commandArgs);
    return;
  }

  if (command === "feature") {
    await runFeature(commandArgs);
    return;
  }

  console.log("usage:");
  console.log("  node cli-agent.ts <projectRoot> <pythonPath> run --goal \"질문\" [--role planner|researcher|analyst|summarizer|auditor|coach] [--orchestrator-mode single-pass|adaptive|strict] [--repo-path ./repo] [--include-workspace|--no-include-workspace] [--max-workspace-files 8] [--report-path ./path] [--max-rounds 3] [--dry-run] [--dump-json]");
  console.log("  node cli-agent.ts <projectRoot> <pythonPath> sync [--source all|note|paper|web|expense|sleep|schedule|behavior] [--limit 200] [--cursor <ts>] [--fresh] [--state-file <path>] [--event-log <path>] [--no-save-state] [--all-sources] [--fail-fast] [--classification-gate P0|P1|P2|P3]");
  console.log("  node cli-agent.ts <projectRoot> <pythonPath> feature <name|list> [--source <source>] [--days N] [--from ISO8601] [--to ISO8601] [--top-k N] [--limit N] [--intent read|analyze|forecast|summarize|compare|alert] [--state-file <path>] [--ontology-events-path <path>] [--json|--no-json]");
  console.log("    --event-log는 동기화 이벤트를 저장할 jsonl 경로를 지정합니다 (기본: .foundry-ontology-events.jsonl).");
  console.log("    --cursor 생략 시 소스별 마지막 nextCursor를 상태 파일에서 이어받습니다.");
  console.log("    기본 상태 파일: .foundry-sync-state.json");
  console.log("    --state-file은 projectRoot 기준 상대경로를 지원합니다.");
  console.log("    --fresh는 커서를 무시하고 전체 동기화를 강제로 실행합니다.");
  console.log("    --no-save-state는 동기화 후 상태 저장을 건너뜁니다.");
  console.log("    --all-sources는 note/web/paper(1차 번들) 순서로 순차 동기화합니다.");
  console.log("    --fail-fast는 all-sources 실행 중 한 소스 실패 시 즉시 중단합니다.");
  console.log("    feature는 기본적으로 <state-file 기준>/ontology-store/ontology.events.jsonl 를 읽습니다.");
  console.log("    --ontology-events-path 또는 (호환용) --event-log로 이벤트 소스를 명시할 수 있습니다.");
  console.log(`예: node cli-agent.ts ${projectRoot} ${cliCommand} run --goal \"최근 논문 요약\" --dump-json`);
}

const isDirectExecution = (() => {
  const entry = process.argv[1];
  if (!entry) {
    return false;
  }
  try {
    return import.meta.url === pathToFileURL(entry).href;
  } catch {
    return false;
  }
})();

if (isDirectExecution) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
