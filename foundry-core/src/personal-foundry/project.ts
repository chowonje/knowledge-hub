import { existsSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";

export interface PersonalFoundryPaths {
  projectRoot: string;
  baseDir: string;
  stateFilePath: string;
  idempotencyPath: string;
  eventBusLogPath: string;
  syncEventLogPath: string;
  auditLogPath: string;
  projectsPath: string;
  goalsPath: string;
  tasksPath: string;
  decisionsPath: string;
  inboxPath: string;
  ontologyDir: string;
  ontologyEntitiesPath: string;
  ontologyRelationsPath: string;
  ontologyEventsPath: string;
  ontologySnapshotsPath: string;
  ontologyTimeSeriesPath: string;
  agentAuditPath: string;
  manifestPath: string;
}

export interface ResolvePathsInput {
  projectRoot: string;
  baseDir?: string;
  stateFile?: string;
  eventLogPath?: string;
}

export interface InitProjectInput extends ResolvePathsInput {
  overwriteManifest?: boolean;
}

export interface InitProjectResult {
  schema: "knowledge-hub.personal-foundry.project.init.result.v1";
  ok: boolean;
  createdAt: string;
  paths: PersonalFoundryPaths;
}

function ensureDir(dir: string): void {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function ensureJsonFile(path: string, fallback: unknown): void {
  ensureDir(dirname(path));
  if (!existsSync(path)) {
    writeFileSync(path, JSON.stringify(fallback, null, 2), "utf8");
  }
}

function ensureJsonlFile(path: string): void {
  ensureDir(dirname(path));
  if (!existsSync(path)) {
    writeFileSync(path, "", "utf8");
  }
}

export function resolvePersonalFoundryPaths(input: ResolvePathsInput): PersonalFoundryPaths {
  const projectRoot = resolve(input.projectRoot);
  const baseDirRaw = input.baseDir ?? resolve(projectRoot, ".khub", "personal-foundry");
  const baseDir = isAbsolute(baseDirRaw) ? baseDirRaw : resolve(projectRoot, baseDirRaw);

  const stateFileRaw = input.stateFile ?? resolve(baseDir, ".foundry-sync-state.json");
  const stateFilePath = isAbsolute(stateFileRaw) ? stateFileRaw : resolve(projectRoot, stateFileRaw);

  const syncEventLogRaw = input.eventLogPath ?? resolve(baseDir, ".foundry-ontology-events.jsonl");
  const syncEventLogPath = isAbsolute(syncEventLogRaw) ? syncEventLogRaw : resolve(projectRoot, syncEventLogRaw);

  const ontologyDir = resolve(baseDir, "ontology-store");

  return {
    projectRoot,
    baseDir,
    stateFilePath,
    idempotencyPath: resolve(baseDir, ".foundry-idempotency.json"),
    eventBusLogPath: resolve(baseDir, "event-bus.jsonl"),
    syncEventLogPath,
    auditLogPath: resolve(baseDir, ".foundry-audit.jsonl"),
    projectsPath: resolve(baseDir, "projects.json"),
    goalsPath: resolve(baseDir, "goals.json"),
    tasksPath: resolve(baseDir, "tasks.json"),
    decisionsPath: resolve(baseDir, "decisions.json"),
    inboxPath: resolve(baseDir, "inbox.json"),
    ontologyDir,
    ontologyEntitiesPath: resolve(ontologyDir, "ontology.entities.jsonl"),
    ontologyRelationsPath: resolve(ontologyDir, "ontology.relations.jsonl"),
    ontologyEventsPath: resolve(ontologyDir, "ontology.events.jsonl"),
    ontologySnapshotsPath: resolve(ontologyDir, "ontology.snapshots.jsonl"),
    ontologyTimeSeriesPath: resolve(ontologyDir, "ontology.timeseries.jsonl"),
    agentAuditPath: resolve(baseDir, "agent-audit.jsonl"),
    manifestPath: resolve(baseDir, "project-manifest.json"),
  };
}

export function initPersonalFoundryProject(input: InitProjectInput): InitProjectResult {
  const now = new Date().toISOString();
  const paths = resolvePersonalFoundryPaths(input);

  ensureDir(paths.baseDir);
  ensureDir(paths.ontologyDir);

  ensureJsonFile(paths.stateFilePath, {});
  ensureJsonFile(paths.idempotencyPath, {});
  ensureJsonFile(paths.projectsPath, []);
  ensureJsonFile(paths.goalsPath, []);
  ensureJsonFile(paths.tasksPath, []);
  ensureJsonFile(paths.decisionsPath, []);
  ensureJsonFile(paths.inboxPath, []);
  ensureJsonlFile(paths.syncEventLogPath);
  ensureJsonlFile(paths.eventBusLogPath);
  ensureJsonlFile(paths.auditLogPath);
  ensureJsonlFile(paths.ontologyEntitiesPath);
  ensureJsonlFile(paths.ontologyRelationsPath);
  ensureJsonlFile(paths.ontologyEventsPath);
  ensureJsonlFile(paths.ontologySnapshotsPath);
  ensureJsonlFile(paths.ontologyTimeSeriesPath);
  ensureJsonlFile(paths.agentAuditPath);

  const manifest = {
    schema: "knowledge-hub.personal-foundry.project.manifest.v1",
    createdAt: now,
    updatedAt: now,
    projectRoot: paths.projectRoot,
    baseDir: paths.baseDir,
    files: {
      stateFilePath: paths.stateFilePath,
      idempotencyPath: paths.idempotencyPath,
      syncEventLogPath: paths.syncEventLogPath,
      eventBusLogPath: paths.eventBusLogPath,
      auditLogPath: paths.auditLogPath,
      projectsPath: paths.projectsPath,
      goalsPath: paths.goalsPath,
      tasksPath: paths.tasksPath,
      decisionsPath: paths.decisionsPath,
      inboxPath: paths.inboxPath,
      ontologyEventsPath: paths.ontologyEventsPath,
      agentAuditPath: paths.agentAuditPath,
    },
  };

  if (!existsSync(paths.manifestPath) || input.overwriteManifest) {
    writeFileSync(paths.manifestPath, JSON.stringify(manifest, null, 2), "utf8");
  } else {
    try {
      const previous = JSON.parse(readFileSync(paths.manifestPath, "utf8"));
      writeFileSync(
        paths.manifestPath,
        JSON.stringify(
          {
            ...previous,
            updatedAt: now,
            files: manifest.files,
          },
          null,
          2
        ),
        "utf8"
      );
    } catch {
      writeFileSync(paths.manifestPath, JSON.stringify(manifest, null, 2), "utf8");
    }
  }

  return {
    schema: "knowledge-hub.personal-foundry.project.init.result.v1",
    ok: true,
    createdAt: now,
    paths,
  };
}

export interface ProjectStatusResult {
  schema: "knowledge-hub.personal-foundry.project.status.result.v1";
  ok: boolean;
  paths: PersonalFoundryPaths;
  files: Array<{
    path: string;
    exists: boolean;
    bytes: number;
    lines?: number;
  }>;
}

function fileStatus(path: string): { path: string; exists: boolean; bytes: number; lines?: number } {
  if (!existsSync(path)) {
    return { path, exists: false, bytes: 0 };
  }
  const raw = readFileSync(path, "utf8");
  const lines = raw.length === 0 ? 0 : raw.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
  return {
    path,
    exists: true,
    bytes: Buffer.byteLength(raw),
    lines,
  };
}

export function getPersonalFoundryProjectStatus(input: ResolvePathsInput): ProjectStatusResult {
  const paths = resolvePersonalFoundryPaths(input);
  const files = [
    paths.stateFilePath,
    paths.idempotencyPath,
    paths.syncEventLogPath,
    paths.eventBusLogPath,
    paths.auditLogPath,
    paths.projectsPath,
    paths.goalsPath,
    paths.tasksPath,
    paths.decisionsPath,
    paths.inboxPath,
    paths.decisionsPath,
    paths.ontologyEntitiesPath,
    paths.ontologyRelationsPath,
    paths.ontologyEventsPath,
    paths.ontologySnapshotsPath,
    paths.ontologyTimeSeriesPath,
    paths.agentAuditPath,
    paths.manifestPath,
  ].map(fileStatus);

  return {
    schema: "knowledge-hub.personal-foundry.project.status.result.v1",
    ok: files.every((file) => file.exists),
    paths,
    files,
  };
}
