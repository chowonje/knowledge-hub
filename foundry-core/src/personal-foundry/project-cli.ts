#!/usr/bin/env node
import { resolve } from "node:path";

import { runKnowledgeHubCli } from "../adapters/knowledge-hub-cli.js";
import { createFeatureRuntime } from "../features.js";
import { KnowledgeHubConnector, KnowledgeHubCLICommand } from "../adapters/knowledge-hub-connector.js";
import { KnowledgeHubPersonalConnectorBridge } from "../adapters/knowledge-hub-personal-connector.js";
import { PlanActVerifyRuntime } from "./agent-runtime.js";
import { JsonlAuditLog } from "./audit-log.js";
import { DefaultConnectorRunner, JsonFileCursorStore, JsonFileIdempotencyStore } from "./connector-runner.js";
import { emitOntologyBatchToEventBus } from "./emitters.js";
import { JsonlEventBus } from "./event-bus.js";
import type { AgentRuntimeInput, AgentTool, AgentToolResult, OntologyBatch } from "./interfaces.js";
import type { FeatureQuery } from "../contracts/index.js";
import { LocalOntologyStore } from "./ontology-store.js";
import {
  type OpsAlertInput,
  type SourceRef,
  enrichOpsAlertsForSelector,
  PersonalFoundryOsService,
} from "./os-state.js";
import { parsePipelineInput, type PipelineInput, type SourceKind } from "./pipeline-input.js";
import { LocalPolicyEngine } from "./policy-engine.js";
import {
  getPersonalFoundryProjectStatus,
  initPersonalFoundryProject,
  resolvePersonalFoundryPaths,
} from "./project.js";

interface ParsedTopLevel {
  projectRoot: string;
  pythonPath: string;
  command: "init" | "status" | "pipeline" | "project" | "goal" | "task" | "decision" | "inbox" | "next" | "capture" | "decide" | "help";
  args: string[];
}

function parseTopLevel(argv: string[]): ParsedTopLevel {
  if (argv.length === 0) {
    return {
      projectRoot: process.cwd(),
      pythonPath: "python",
      command: "help",
      args: [],
    };
  }

  let offset = 0;
  let projectRoot = process.cwd();
  let pythonPath = "python";

  const first = argv[0];
  const second = argv[1];
  const known = new Set(["init", "status", "pipeline", "project", "goal", "task", "decision", "inbox", "next", "capture", "decide", "help"]);

  if (first && !known.has(first) && !first.startsWith("-")) {
    projectRoot = first;
    offset = 1;
  }
  if (second && offset === 1 && !known.has(second) && !second.startsWith("-")) {
    pythonPath = second;
    offset = 2;
  }

  const commandRaw = argv[offset] ?? "help";
  const command = known.has(commandRaw) ? (commandRaw as ParsedTopLevel["command"]) : "help";
  return {
    projectRoot,
    pythonPath,
    command,
    args: argv.slice(offset + 1),
  };
}

function runKhubCli(projectRoot: string, pythonPath: string, args: string[]): string {
  return runKnowledgeHubCli(projectRoot, pythonPath, args);
}

function hasFlag(args: string[], flag: string): boolean {
  return args.some((item) => item === flag);
}

function flagValue(args: string[], flag: string): string {
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (token === flag) {
      return index + 1 < args.length ? String(args[index + 1] || "") : "";
    }
    if (token.startsWith(`${flag}=`)) {
      return token.split("=", 2)[1] || "";
    }
  }
  return "";
}

function flagValues(args: string[], flag: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const token = args[index];
    if (token === flag && index + 1 < args.length) {
      values.push(String(args[index + 1] || ""));
      index += 1;
      continue;
    }
    if (token.startsWith(`${flag}=`)) {
      values.push(token.split("=", 2)[1] || "");
    }
  }
  return values;
}

function compact(value: string | undefined): string {
  return String(value || "").trim();
}

function parseSourceRefs(args: string[]): SourceRef[] {
  const refs: SourceRef[] = [];
  for (const raw of flagValues(args, "--source-ref")) {
    if (!compact(raw)) {
      continue;
    }
    const parsed = JSON.parse(raw) as SourceRef;
    refs.push(parsed);
  }
  for (const value of flagValues(args, "--paper-id")) {
    if (compact(value)) {
      refs.push({ sourceType: "paper", paperId: compact(value) });
    }
  }
  for (const value of flagValues(args, "--url")) {
    if (compact(value)) {
      refs.push({ sourceType: "web", url: compact(value) });
    }
  }
  for (const value of flagValues(args, "--note-id")) {
    if (compact(value)) {
      refs.push({ sourceType: "vault", noteId: compact(value) });
    }
  }
  for (const value of flagValues(args, "--stable-scope-id")) {
    if (compact(value)) {
      refs.push({ sourceType: "scope", stableScopeId: compact(value) });
    }
  }
  for (const value of flagValues(args, "--document-scope-id")) {
    if (compact(value)) {
      refs.push({ sourceType: "document", documentScopeId: compact(value) });
    }
  }
  return refs;
}

function parseOpsAlerts(args: string[]): OpsAlertInput[] {
  const raw = compact(flagValue(args, "--ops-alerts-json"));
  if (!raw) {
    return [];
  }
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? (parsed as OpsAlertInput[]) : [];
}

function supportsAgentSync(projectRoot: string, pythonPath: string): boolean {
  try {
    const help = runKhubCli(projectRoot, pythonPath, ["--help"]);
    return /(?:^|\n)\s+agent\s+/m.test(help);
  } catch {
    return false;
  }
}

interface PaperListRow {
  arxivId: string;
  title: string;
  year?: string;
  field?: string;
  pdf?: string;
  summary?: string;
  translation?: string;
  vector?: string;
}

function parsePaperListOutput(raw: string): PaperListRow[] {
  const rows = raw.split(/\r?\n/).filter((line) => line.includes("│"));
  const parsed: PaperListRow[] = [];
  let current: PaperListRow | null = null;

  for (const line of rows) {
    const cols = line.split("│").map((value) => value.trim());
    if (cols.length < 9) {
      continue;
    }

    const arxivId = cols[1];
    const title = cols[2];
    const year = cols[3];
    const field = cols[4];
    const pdf = cols[5];
    const summary = cols[6];
    const translation = cols[7];
    const vector = cols[8];

    if (arxivId === "arXiv ID" || arxivId.length === 0 && title === "제목") {
      continue;
    }

    const isNewRow = /^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$/.test(arxivId);
    if (isNewRow) {
      const next: PaperListRow = {
        arxivId,
        title,
        year: year || undefined,
        field: field || undefined,
        pdf: pdf || undefined,
        summary: summary || undefined,
        translation: translation || undefined,
        vector: vector || undefined,
      };
      parsed.push(next);
      current = next;
      continue;
    }

    if (!arxivId && current && title) {
      current.title = `${current.title} ${title}`.trim();
    }
  }

  return parsed;
}

async function fallbackIngestFromPaperList(input: {
  projectRoot: string;
  pythonPath: string;
  actorId: string;
  limit?: number;
  ontologyStore: LocalOntologyStore;
  eventBus: JsonlEventBus;
  audit: JsonlAuditLog;
}): Promise<{ status: "done" | "failed"; emittedEventCount: number; reason?: string }> {
  try {
    const limit = Math.max(1, Math.min(input.limit ?? 30, 200));
    const raw = runKhubCli(input.projectRoot, input.pythonPath, ["paper", "list", "-n", String(limit)]);
    const papers = parsePaperListOutput(raw);

    if (papers.length === 0) {
      return { status: "failed", emittedEventCount: 0, reason: "no parsable papers from paper list output" };
    }

    const now = new Date().toISOString();
    const batch: OntologyBatch = {
      entities: papers.map((paper) => ({
        id: `kh:paper:${paper.arxivId}`,
        type: "Paper",
        properties: {
          arxivId: paper.arxivId,
          title: paper.title,
          year: paper.year ?? "",
          field: paper.field ?? "",
          pdf: paper.pdf ?? "",
          summary: paper.summary ?? "",
          translation: paper.translation ?? "",
          vector: paper.vector ?? "",
          source: "paper-list-fallback",
        },
        classification: "P1",
        sourceSystem: "knowledge_hub",
        updatedAt: now,
      })),
      relations: [],
      events: papers.map((paper) => ({
        aggregateId: `kh:paper:${paper.arxivId}`,
        aggregateType: "Paper",
        type: "PaperListedFallbackIngested",
        payload: {
          arxivId: paper.arxivId,
          title: paper.title,
          source: "paper-list-fallback",
        },
        classification: "P1",
        sourceSystem: "knowledge_hub",
        occurredAt: now,
        actorId: input.actorId,
        sourceRecordId: paper.arxivId,
      })),
      timeSeries: [],
    };

    await input.ontologyStore.appendBatch(batch);
    const emitted = await emitOntologyBatchToEventBus(
      {
        actorId: input.actorId,
        requestId: `fallback_${Date.now()}`,
        connectorRunId: `fallback_paper_list_${Date.now()}`,
        mapped: batch,
      },
      input.eventBus,
      () => now
    );

    await input.audit.append({
      actorId: input.actorId,
      action: "connector_sync",
      resourceType: "connector",
      resourceId: "knowledge-hub-paper-list-fallback",
      allowed: true,
      reason: `fallback ingest from paper list: ${papers.length} papers`,
      classification: "P1",
      metadata: {
        paperCount: papers.length,
        eventCount: emitted.eventIds.length,
      },
    });

    return {
      status: "done",
      emittedEventCount: emitted.eventIds.length,
    };
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return {
      status: "failed",
      emittedEventCount: 0,
      reason,
    };
  }
}

function createKnowledgeTools(projectRoot: string, pythonPath: string): Record<string, AgentTool> {
  return {
    search_knowledge: {
      name: "search_knowledge",
      async execute(input: Record<string, unknown>): Promise<AgentToolResult | void> {
        const query = String(input.query ?? "").trim();
        if (!query) {
          return;
        }
        const topK = Number(input.top_k ?? 5);
        const raw = runKhubCli(projectRoot, pythonPath, ["search", query, "--top-k", String(topK)]);
        return {
          artifact: {
            jsonContent: {
              mode: "search_knowledge",
              query,
              raw,
            },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
    ask_knowledge: {
      name: "ask_knowledge",
      async execute(input: Record<string, unknown>): Promise<AgentToolResult | void> {
        const question = String(input.question ?? "").trim();
        if (!question) {
          return;
        }
        const topK = Number(input.top_k ?? 5);
        const raw = runKhubCli(projectRoot, pythonPath, ["ask", question, "--top-k", String(topK)]);
        return {
          artifact: {
            jsonContent: {
              mode: "ask_knowledge",
              question,
              raw,
            },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
  };
}

function createDryRunTools(): Record<string, AgentTool> {
  return {
    search_knowledge: {
      name: "search_knowledge",
      async execute(input: Record<string, unknown>): Promise<AgentToolResult> {
        return {
          artifact: {
            jsonContent: {
              mode: "search_knowledge",
              dryRun: true,
              input,
              summary: "dry-run tool output",
            },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
    ask_knowledge: {
      name: "ask_knowledge",
      async execute(input: Record<string, unknown>): Promise<AgentToolResult> {
        return {
          artifact: {
            jsonContent: {
              mode: "ask_knowledge",
              dryRun: true,
              input,
              answer: "dry-run: agent execution skipped external query",
            },
            generatedAt: new Date().toISOString(),
            classification: "P2",
          },
        };
      },
    },
  };
}

async function runPipeline(projectRoot: string, pythonPath: string, input: PipelineInput): Promise<Record<string, unknown>> {
  const init = initPersonalFoundryProject({ projectRoot });
  const paths = init.paths;

  const eventBus = new JsonlEventBus({ logPath: paths.syncEventLogPath });
  const audit = new JsonlAuditLog({ logPath: paths.auditLogPath });
  const policy = new LocalPolicyEngine({
    outboundMaxClassification: "P2",
    writebackMaxClassification: "P1",
  });
  const ontologyStore = new LocalOntologyStore({ baseDir: paths.ontologyDir });
  const cursorStore = new JsonFileCursorStore(paths.stateFilePath);
  const idempotencyStore = new JsonFileIdempotencyStore({ path: paths.idempotencyPath });

  const executor = new KnowledgeHubCLICommand(
    { projectRoot },
    (_cmd: string, args: string[]) => Promise.resolve(runKhubCli(projectRoot, pythonPath, args.slice(1)))
  );
  const connectorForSource = (source: string) => {
    const scoped: Record<string, { connectorId: string; fixedSource: string }> = {
      note: { connectorId: "knowledge-hub-vault", fixedSource: "note" },
      web: { connectorId: "knowledge-hub-web", fixedSource: "web" },
      paper: { connectorId: "knowledge-hub-arxiv", fixedSource: "paper" },
    };
    const current = scoped[source] ?? {
      connectorId: source === "all" ? "knowledge-hub" : `knowledge-hub-${source}`,
      fixedSource: source === "all" ? "all" : source,
    };
    return new KnowledgeHubPersonalConnectorBridge(
      new KnowledgeHubConnector(executor, undefined, {
        connectorId: current.connectorId,
        fixedSource: current.fixedSource as "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior",
      })
    );
  };

  const runner = new DefaultConnectorRunner({
    eventBus,
    audit,
    policy,
    ontologyStore,
    cursorStore,
    idempotencyStore,
    classificationGate: "P1",
  });

  const sources = input.source === "all"
    ? (["note", "web", "paper"] as const)
    : ([input.source] as const);

  const syncResults: Record<string, unknown>[] = [];
  const canSync = supportsAgentSync(projectRoot, pythonPath);
  if (!canSync) {
    const fallback = await fallbackIngestFromPaperList({
      projectRoot,
      pythonPath,
      actorId: "cli-user",
      limit: input.limit,
      ontologyStore,
      eventBus,
      audit,
    });
    syncResults.push({
      source: input.source,
      mode: "fallback-paper-list",
      ...fallback,
      reason: fallback.reason ?? "agent sync not available; used paper list fallback",
    });
  } else {
    for (const source of sources) {
      const connector = connectorForSource(source);
      const sync = await runner.run({
        connector,
        actorId: "cli-user",
        requestId: `pipeline_${Date.now()}_${source}`,
        source,
        pageLimit: input.limit,
        dryRun: false,
      });
      syncResults.push({ source, ...sync });
    }
  }

  const featureRuntime = createFeatureRuntime({
    eventLogPath: paths.ontologyEventsPath,
  });
  const featureQuery: FeatureQuery = {
    name: input.featureName,
    intent: "analyze",
    params: {
      source: "all",
      days: input.days ?? 7,
      top_k: 8,
    },
  };
  const featureResult = await featureRuntime.execute(featureQuery);

  const runtime = new PlanActVerifyRuntime({
    policy,
    audit,
  });
  const tools = input.dryRun ? createDryRunTools() : createKnowledgeTools(projectRoot, pythonPath);
  const agent = await runtime.run({
    actorId: "cli-user",
    requestId: `agent_${Date.now()}`,
    goal: input.goal,
    maxRounds: input.maxRounds,
    tools,
    planner: async (goal) => [
      { order: 1, toolName: "search_knowledge", objective: "collect evidence", input: { query: goal, top_k: 5 } },
      { order: 2, toolName: "ask_knowledge", objective: "synthesize answer", input: { question: goal, top_k: 5 } },
    ],
    schemaValidate: async (artifact: unknown) => {
      if (!artifact || typeof artifact !== "object") {
        return { ok: false, errors: ["artifact must be object"] };
      }
      return { ok: true, errors: [] };
    },
    writeback: async ({ artifact }: Parameters<NonNullable<AgentRuntimeInput["writeback"]>>[0]) => {
      if (input.dryRun) {
        return { ok: true, detail: "dry-run: writeback skipped" };
      }
      const payload = {
        id: `pipeline_writeback_${Date.now()}`,
        type: "agent:writeback",
        occurredAt: new Date().toISOString(),
        sourceSystem: "manual" as const,
        classification: "P2" as const,
        actorId: "cli-user",
        payload: artifact as Record<string, unknown>,
      };
      await eventBus.publish(payload);
      return { ok: true, detail: "writeback appended to event bus" };
    },
  });

  return {
    schema: "knowledge-hub.personal-foundry.project.pipeline.result.v1",
    createdAt: new Date().toISOString(),
    paths,
    sync: syncResults,
    feature: featureResult,
    agent,
  };
}

function createOsService(projectRoot: string): PersonalFoundryOsService {
  const init = initPersonalFoundryProject({ projectRoot });
  const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
  return new PersonalFoundryOsService({ paths: init.paths, eventBus });
}

async function runProjectCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const subcommand = args[0] ?? "help";

  if (subcommand === "create") {
    const result = await service.createProject({
      title: flagValue(args, "--title"),
      slug: flagValue(args, "--slug") || undefined,
      status: flagValue(args, "--status") || undefined,
      priority: flagValue(args, "--priority") || undefined,
      summary: flagValue(args, "--summary") || undefined,
      owner: flagValue(args, "--owner") || undefined,
    });
    return {
      schema: "knowledge-hub.os.project.create.result.v1",
      status: "ok",
      created: result.created,
      project: result.project,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "list") {
    const items = await service.listProjects();
    return {
      schema: "knowledge-hub.os.project.list.result.v1",
      status: "ok",
      count: items.length,
      items,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "update") {
    const updated = await service.updateProject({
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
      title: flagValue(args, "--title") || undefined,
      status: flagValue(args, "--status") || undefined,
      priority: flagValue(args, "--priority") || undefined,
      summary: flagValue(args, "--summary") || undefined,
      owner: flagValue(args, "--owner") || undefined,
    });
    return {
      schema: "knowledge-hub.os.project.update.result.v1",
      status: "ok",
      project: updated,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "show") {
    const selector = {
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
    };
    const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
    const shown = await service.showProject(selector, { opsAlerts });
    return {
      schema: "knowledge-hub.os.project.show.result.v1",
      status: "ok",
      project: shown.project,
      goals: shown.goals,
      tasks: shown.tasks,
      inbox: shown.inbox,
      decisions: shown.decisions,
      projectEvidence: shown.projectEvidence,
      nextActionableTasks: shown.nextActionableTasks,
      blockedTasks: shown.blockedTasks,
      recentDecisions: shown.recentDecisions,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "evidence") {
    const selector = {
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
    };
    const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
    const result = await service.projectEvidence(selector, { opsAlerts });
    return {
      schema: "knowledge-hub.os.project.evidence.result.v1",
      status: "ok",
      project: result.project,
      projectEvidence: result.projectEvidence,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "export") {
    const selector = {
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
    };
    const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
    const result = await service.buildProjectProjection(selector, { opsAlerts });
    return {
      schema: "knowledge-hub.os.project.export.obsidian.result.v1",
      status: "ok",
      project: result.project,
      goals: result.goals,
      tasks: result.tasks,
      inbox: result.inbox,
      decisions: result.decisions,
      projectEvidence: result.projectEvidence,
      nextActionableTasks: result.nextActionableTasks,
      blockedTasks: result.blockedTasks,
      recentDecisions: result.recentDecisions,
      projections: result.projections,
      createdAt: new Date().toISOString(),
    };
  }

  throw new Error(`unknown project subcommand: ${subcommand}`);
}

async function runGoalCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const subcommand = args[0] ?? "help";
  if (subcommand === "add") {
    const result = await service.addGoal({
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
      title: flagValue(args, "--title"),
      status: flagValue(args, "--status") || undefined,
      successCriteria: flagValue(args, "--success-criteria") || undefined,
    });
    return {
      schema: "knowledge-hub.os.goal.add.result.v1",
      status: "ok",
      created: result.created,
      goal: result.goal,
      createdAt: new Date().toISOString(),
    };
  }
  if (subcommand === "update") {
    const goal = await service.updateGoal({
      goalId: flagValue(args, "--goal-id"),
      title: flagValue(args, "--title") || undefined,
      status: flagValue(args, "--status") || undefined,
      successCriteria: flagValue(args, "--success-criteria") || undefined,
    });
    return {
      schema: "knowledge-hub.os.goal.update.result.v1",
      status: "ok",
      goal,
      createdAt: new Date().toISOString(),
    };
  }
  throw new Error(`unknown goal subcommand: ${subcommand}`);
}

async function runTaskCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const subcommand = args[0] ?? "help";
  if (subcommand === "add") {
    const selector = {
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
    };
    const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
    const result = await service.addTask({
      ...selector,
      goalId: flagValue(args, "--goal-id") || undefined,
      title: flagValue(args, "--title"),
      kind: flagValue(args, "--kind") || undefined,
      status: flagValue(args, "--status") || undefined,
      priority: flagValue(args, "--priority") || undefined,
      assignee: flagValue(args, "--assignee") || undefined,
      dueAt: flagValue(args, "--due-at") || undefined,
      blockedBy: flagValues(args, "--blocked-by"),
      sourceRefs: parseSourceRefs(args),
      opsAlerts,
    });
    return {
      schema: "knowledge-hub.os.task.add.result.v1",
      status: "ok",
      created: result.created,
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "update-status") {
    const result = await service.updateTaskStatus({
      taskId: flagValue(args, "--task-id"),
      status: flagValue(args, "--status"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.update-status.result.v1",
      status: "ok",
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "start") {
    const result = await service.startTask({
      taskId: flagValue(args, "--task-id"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.start.result.v1",
      status: "ok",
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "block") {
    const result = await service.blockTask({
      taskId: flagValue(args, "--task-id"),
      reason: flagValue(args, "--reason"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.block.result.v1",
      status: "ok",
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "complete") {
    const result = await service.completeTask({
      taskId: flagValue(args, "--task-id"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.complete.result.v1",
      status: "ok",
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "cancel") {
    const result = await service.cancelTask({
      taskId: flagValue(args, "--task-id"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.cancel.result.v1",
      status: "ok",
      task: result.task,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "update") {
    const task = await service.updateTask({
      taskId: flagValue(args, "--task-id"),
      title: flagValue(args, "--title") || undefined,
      kind: flagValue(args, "--kind") || undefined,
      priority: flagValue(args, "--priority") || undefined,
      assignee: flagValue(args, "--assignee") || undefined,
      dueAt: flagValue(args, "--due-at") || undefined,
      blockedBy: flagValues(args, "--blocked-by"),
      sourceRefs: parseSourceRefs(args),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.task.update.result.v1",
      status: "ok",
      task: task.task,
      inbox: task.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  throw new Error(`unknown task subcommand: ${subcommand}`);
}

async function runDecisionCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const subcommand = args[0] ?? "help";
  if (subcommand === "add") {
    const decision = await service.addDecision({
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
      goalId: flagValue(args, "--goal-id") || undefined,
      taskId: flagValue(args, "--task-id") || undefined,
      kind: flagValue(args, "--kind"),
      summary: flagValue(args, "--summary"),
      rationale: flagValue(args, "--rationale") || undefined,
      sourceRefs: parseSourceRefs(args),
      createdByType: flagValue(args, "--created-by-type") || undefined,
      createdById: flagValue(args, "--created-by-id") || undefined,
      supersedesDecisionId: flagValue(args, "--supersedes-decision-id") || undefined,
    });
    return {
      schema: "knowledge-hub.os.decision.add.result.v1",
      status: "ok",
      decision,
      createdAt: new Date().toISOString(),
    };
  }
  if (subcommand === "list") {
    const items = await service.listDecisions({
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
      goalId: flagValue(args, "--goal-id") || undefined,
      taskId: flagValue(args, "--task-id") || undefined,
    });
    return {
      schema: "knowledge-hub.os.decision.list.result.v1",
      status: "ok",
      count: items.length,
      items,
      createdAt: new Date().toISOString(),
    };
  }
  throw new Error(`unknown decision subcommand: ${subcommand}`);
}

async function runInboxCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const subcommand = args[0] ?? "help";
  if (subcommand === "list") {
    const stateFilter = (flagValue(args, "--state") || "open") as "open" | "resolved" | "all";
    const selector = {
      projectId: flagValue(args, "--project-id") || undefined,
      slug: flagValue(args, "--slug") || undefined,
    };
    const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
    const items = await service.listInbox({
      state: stateFilter,
      ...selector,
      opsAlerts,
    });
    return {
      schema: "knowledge-hub.os.inbox.list.result.v1",
      status: "ok",
      count: items.length,
      items,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "resolve") {
    const result = await service.resolveInboxItem({
      itemId: flagValue(args, "--item-id"),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.inbox.resolve.result.v1",
      status: "ok",
      item: result.item,
      inbox: result.inbox,
      createdAt: new Date().toISOString(),
    };
  }

  if (subcommand === "triage") {
    const toTask = hasFlag(args, "--to-task");
    const toDecision = hasFlag(args, "--to-decision");
    const resolveOnly = hasFlag(args, "--resolve-only");
    const modeCount = Number(toTask) + Number(toDecision) + Number(resolveOnly);
    if (modeCount !== 1) {
      throw new Error("exactly one of --to-task, --to-decision, or --resolve-only is required");
    }
    const result = await service.triageInboxItem({
      itemId: flagValue(args, "--item-id"),
      action: toTask ? "to_task" : toDecision ? "to_decision" : "resolve_only",
      taskTitle: flagValue(args, "--title") || undefined,
      taskKind: flagValue(args, "--kind") || undefined,
      taskPriority: flagValue(args, "--priority") || undefined,
      taskAssignee: flagValue(args, "--assignee") || undefined,
      taskDueAt: flagValue(args, "--due-at") || undefined,
      taskGoalId: flagValue(args, "--goal-id") || undefined,
      taskBlockedBy: flagValues(args, "--blocked-by"),
      taskSourceRefs: parseSourceRefs(args),
      decisionKind: flagValue(args, "--kind") || undefined,
      decisionSummary: flagValue(args, "--summary") || undefined,
      decisionRationale: flagValue(args, "--rationale") || undefined,
      decisionGoalId: flagValue(args, "--goal-id") || undefined,
      decisionTaskId: flagValue(args, "--task-id") || undefined,
      decisionCreatedByType: flagValue(args, "--created-by-type") || undefined,
      decisionCreatedById: flagValue(args, "--created-by-id") || undefined,
      decisionSupersedesDecisionId: flagValue(args, "--supersedes-decision-id") || undefined,
      decisionSourceRefs: parseSourceRefs(args),
      opsAlerts: parseOpsAlerts(args),
    });
    return {
      schema: "knowledge-hub.os.inbox.triage.result.v1",
      status: "ok",
      item: result.item,
      inbox: result.inbox,
      createdTask: result.createdTask,
      createdDecision: result.createdDecision,
      createdAt: new Date().toISOString(),
    };
  }

  throw new Error(`unknown inbox subcommand: ${subcommand}`);
}

async function runCaptureCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const selector = {
    projectId: flagValue(args, "--project-id") || undefined,
    slug: flagValue(args, "--slug") || undefined,
  };
  const opsAlerts = await enrichOpsAlertsForSelector(service, selector, parseOpsAlerts(args));
  const result = await service.captureInboxItem({
    ...selector,
    summary: flagValue(args, "--summary"),
    kind: flagValue(args, "--kind") || undefined,
    severity: flagValue(args, "--severity") || undefined,
    sourceRefs: parseSourceRefs(args),
    opsAlerts,
  });
  return {
    schema: "knowledge-hub.os.capture.result.v1",
    status: "ok",
    item: result.item,
    inbox: result.inbox,
    createdAt: new Date().toISOString(),
  };
}

async function runDecideCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const decision = await service.addDecision({
    projectId: flagValue(args, "--project-id") || undefined,
    slug: flagValue(args, "--slug") || undefined,
    goalId: flagValue(args, "--goal-id") || undefined,
    taskId: flagValue(args, "--task-id") || undefined,
    kind: flagValue(args, "--kind"),
    summary: flagValue(args, "--summary"),
    rationale: flagValue(args, "--rationale") || undefined,
    sourceRefs: parseSourceRefs(args),
    createdByType: flagValue(args, "--created-by-type") || undefined,
    createdById: flagValue(args, "--created-by-id") || undefined,
    supersedesDecisionId: flagValue(args, "--supersedes-decision-id") || undefined,
  });
  return {
    schema: "knowledge-hub.os.decide.result.v1",
    status: "ok",
    decision,
    createdAt: new Date().toISOString(),
  };
}

async function runNextCommand(projectRoot: string, args: string[]): Promise<Record<string, unknown>> {
  const service = createOsService(projectRoot);
  const result = await service.next({
    projectId: flagValue(args, "--project-id") || undefined,
    slug: flagValue(args, "--slug") || undefined,
    opsAlerts: parseOpsAlerts(args),
  });
  return {
    schema: "knowledge-hub.os.next.result.v1",
    status: "ok",
    project: result.project,
    activeProjects: result.activeProjects,
    actionableTasks: result.actionableTasks,
    blockedTasks: result.blockedTasks,
    openInbox: result.openInbox,
    recentDecisions: result.recentDecisions,
    createdAt: new Date().toISOString(),
  };
}

function printHelp(): void {
  console.log("usage:");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] init");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] status");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] pipeline [--goal text] [--source all|note|paper|web|expense|sleep|schedule|behavior] [--feature daily_coach|focus_analytics|risk_alert] [--days N] [--limit N] [--max-rounds N] [--dry-run]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project create --title text [--slug text] [--summary text]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project update (--project-id id | --slug slug) [--title text] [--status status] [--priority level]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project list");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project show (--project-id id | --slug slug)");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project evidence (--project-id id | --slug slug)");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] project export (--project-id id | --slug slug)");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] goal add (--project-id id | --slug slug) --title text");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] goal update --goal-id id [--title text] [--status status]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task add (--project-id id | --slug slug) --title text [--paper-id id] [--url url]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task update --task-id id [--title text] [--priority level] [--source-ref json]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task update-status --task-id id --status open|in_progress|blocked|completed|cancelled");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task start --task-id id");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task block --task-id id --reason text");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task complete --task-id id");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] task cancel --task-id id");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] decision add (--project-id id | --slug slug) --kind type --summary text");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] decide (--project-id id | --slug slug) --kind type --summary text");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] decision list [--project-id id | --slug slug] [--goal-id id] [--task-id id]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] capture (--project-id id | --slug slug) --summary text [--kind captured] [--paper-id id] [--url url]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] inbox list [--state open|resolved|all]");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] inbox triage --item-id id (--to-task | --to-decision | --resolve-only)");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] inbox resolve --item-id id");
  console.log("  node project-cli.ts [projectRoot] [pythonPath] next");
}

async function main(): Promise<void> {
  const parsed = parseTopLevel(process.argv.slice(2));

  if (parsed.command === "help") {
    printHelp();
    return;
  }

  if (parsed.command === "init") {
    const result = initPersonalFoundryProject({ projectRoot: parsed.projectRoot });
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "status") {
    const result = getPersonalFoundryProjectStatus({ projectRoot: parsed.projectRoot });
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "pipeline") {
    const input = parsePipelineInput(parsed.args);
    const result = await runPipeline(parsed.projectRoot, parsed.pythonPath, input);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "project") {
    const result = await runProjectCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "goal") {
    const result = await runGoalCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "task") {
    const result = await runTaskCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "decision") {
    const result = await runDecisionCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "inbox") {
    const result = await runInboxCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "capture") {
    const result = await runCaptureCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "decide") {
    const result = await runDecideCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (parsed.command === "next") {
    const result = await runNextCommand(parsed.projectRoot, parsed.args);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  printHelp();
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
});
