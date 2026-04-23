import crypto from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import type { PersonalEventBus } from "./interfaces.js";
import type { PersonalFoundryPaths } from "./project.js";

export type ProjectStatus = "active" | "on_hold" | "completed" | "archived";
export type GoalStatus = "open" | "in_progress" | "blocked" | "completed" | "dropped";
export type TaskStatus = "open" | "in_progress" | "blocked" | "completed" | "cancelled";
export type Priority = "low" | "medium" | "high" | "urgent";
export type InboxSeverity = "low" | "medium" | "high" | "critical";
export type InboxState = "open" | "resolved";
export type InboxKind = "blocked_task" | "due_soon_task" | "ops_alert" | "captured" | (string & {});
export type SourceRefType = "paper" | "web" | "vault" | "scope" | "document";
export type DecisionActorType = "human" | "agent" | "system";

const SOURCE_REF_PRIMARY_KEYS = [
  "paperId",
  "url",
  "noteId",
  "stableScopeId",
  "documentScopeId",
] as const;

const OPEN_TASK_STATUSES = new Set<TaskStatus>(["open", "in_progress", "blocked"]);
const ACTIONABLE_TASK_STATUSES = new Set<TaskStatus>(["open", "in_progress"]);

const PROJECT_STATUS_VALUES = ["active", "on_hold", "completed", "archived"] as const;
const GOAL_STATUS_VALUES = ["open", "in_progress", "blocked", "completed", "dropped"] as const;
const TASK_STATUS_VALUES = ["open", "in_progress", "blocked", "completed", "cancelled"] as const;
const PRIORITY_VALUES = ["low", "medium", "high", "urgent"] as const;
const DECISION_ACTOR_TYPE_VALUES = ["human", "agent", "system"] as const;

const PROJECT_TRANSITIONS: Record<ProjectStatus, ProjectStatus[]> = {
  active: ["on_hold", "completed", "archived"],
  on_hold: ["active", "completed", "archived"],
  completed: ["archived"],
  archived: [],
};

const GOAL_TRANSITIONS: Record<GoalStatus, GoalStatus[]> = {
  open: ["in_progress", "blocked", "dropped"],
  in_progress: ["blocked", "completed", "dropped"],
  blocked: ["open", "in_progress", "dropped"],
  completed: [],
  dropped: [],
};

const TASK_TRANSITIONS: Record<TaskStatus, TaskStatus[]> = {
  open: ["in_progress", "blocked", "cancelled"],
  in_progress: ["blocked", "completed", "cancelled"],
  blocked: ["open", "in_progress", "cancelled"],
  completed: [],
  cancelled: [],
};

const MANAGED_INBOX_KINDS = new Set<InboxKind>(["blocked_task", "due_soon_task", "ops_alert"]);

export interface SourceRef {
  sourceType: SourceRefType;
  paperId?: string;
  url?: string;
  noteId?: string;
  stableScopeId?: string;
  documentScopeId?: string;
  title?: string;
}

export interface ProjectRecord {
  id: string;
  title: string;
  slug: string;
  status: ProjectStatus;
  priority: Priority;
  summary: string;
  owner: string;
  createdAt: string;
  updatedAt: string;
}

export interface GoalRecord {
  id: string;
  projectId: string;
  title: string;
  status: GoalStatus;
  successCriteria: string;
  createdAt: string;
  updatedAt: string;
}

export interface TaskRecord {
  id: string;
  projectId: string;
  goalId?: string;
  title: string;
  kind: string;
  status: TaskStatus;
  priority: Priority;
  assignee: string;
  dueAt?: string;
  blockedBy: string[];
  sourceRefs: SourceRef[];
  createdAt: string;
  updatedAt: string;
}

export interface InboxItem {
  id: string;
  projectId?: string;
  taskId?: string;
  kind: InboxKind;
  severity: InboxSeverity;
  state: InboxState;
  summary: string;
  sourceRefs: SourceRef[];
  createdAt: string;
  updatedAt: string;
}

export interface DecisionRecord {
  id: string;
  projectId: string;
  goalId?: string;
  taskId?: string;
  kind: string;
  summary: string;
  rationale: string;
  sourceRefs: SourceRef[];
  createdByType: DecisionActorType;
  createdById: string;
  supersedesDecisionId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface OpsAlertInput {
  alertId: string;
  projectId?: string;
  kind?: string;
  severity?: InboxSeverity;
  summary: string;
  sourceRefs?: SourceRef[];
}

export interface ProjectSelector {
  projectId?: string;
  slug?: string;
}

export interface OsStateSnapshot {
  projects: ProjectRecord[];
  goals: GoalRecord[];
  tasks: TaskRecord[];
  inbox: InboxItem[];
  decisions: DecisionRecord[];
}

export interface ProjectionSection {
  key: string;
  body: string;
}

export interface MarkdownProjection {
  relativePath: string;
  title: string;
  sections: ProjectionSection[];
}

export interface ProjectEvidenceTaskLink {
  taskId: string;
  title: string;
  sourceRef: SourceRef;
}

export interface ProjectEvidenceDecisionLink {
  decisionId: string;
  summary: string;
  sourceRef: SourceRef;
}

export interface ProjectEvidenceInboxLink {
  inboxId: string;
  summary: string;
  sourceRef: SourceRef;
}

export interface ProjectEvidenceDecisionSummary {
  decisionId: string;
  summary: string;
  sourceRefs: SourceRef[];
}

export interface ProjectEvidenceTaskSummary {
  taskId: string;
  title: string;
  sourceRefs: SourceRef[];
}

export interface ProjectEvidenceGroupedBySourceType {
  paper: SourceRef[];
  web: SourceRef[];
  vault: SourceRef[];
  scope: SourceRef[];
  document: SourceRef[];
}

export interface ProjectEvidenceView {
  projectId: string;
  sourceRefs: SourceRef[];
  taskLinkedRefs: ProjectEvidenceTaskLink[];
  decisionLinkedRefs: ProjectEvidenceDecisionLink[];
  inboxLinkedRefs: ProjectEvidenceInboxLink[];
  groupedBySourceType: ProjectEvidenceGroupedBySourceType;
  recentDecisionEvidence: ProjectEvidenceDecisionSummary[];
  nextTaskEvidence: ProjectEvidenceTaskSummary[];
}

export interface ProjectProjectionResult {
  project: ProjectRecord;
  goals: GoalRecord[];
  tasks: TaskRecord[];
  inbox: InboxItem[];
  decisions: DecisionRecord[];
  projectEvidence: ProjectEvidenceView;
  nextActionableTasks: TaskRecord[];
  blockedTasks: TaskRecord[];
  recentDecisions: DecisionRecord[];
  projections: MarkdownProjection[];
}

export interface OsNextResult {
  project: ProjectRecord | null;
  activeProjects: ProjectRecord[];
  actionableTasks: TaskRecord[];
  blockedTasks: TaskRecord[];
  openInbox: InboxItem[];
  recentDecisions: DecisionRecord[];
}

export interface CaptureInboxResult {
  item: InboxItem;
  inbox: InboxItem[];
}

export interface TriageInboxResult {
  item: InboxItem;
  inbox: InboxItem[];
  createdTask?: TaskRecord;
  createdDecision?: DecisionRecord;
}

function defaultSnapshot(): OsStateSnapshot {
  return {
    projects: [],
    goals: [],
    tasks: [],
    inbox: [],
    decisions: [],
  };
}

async function readJsonArray<T>(path: string): Promise<T[]> {
  try {
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as T[]) : [];
  } catch {
    return [];
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2), "utf8");
}

function nowIso(now: (() => string) | undefined): string {
  return (now ?? (() => new Date().toISOString()))();
}

function makeId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
}

function slugify(value: string): string {
  return String(value || "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "project";
}

function parsePriority(value: string | undefined, fallback: Priority = "medium"): Priority {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if ((PRIORITY_VALUES as readonly string[]).includes(normalized)) {
    return normalized as Priority;
  }
  throw new Error(`invalid priority: ${value}`);
}

function parseProjectStatus(value: string | undefined, fallback: ProjectStatus = "active"): ProjectStatus {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if ((PROJECT_STATUS_VALUES as readonly string[]).includes(normalized)) {
    return normalized as ProjectStatus;
  }
  throw new Error(`invalid project status: ${value}`);
}

function parseGoalStatus(value: string | undefined, fallback: GoalStatus = "open"): GoalStatus {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if ((GOAL_STATUS_VALUES as readonly string[]).includes(normalized)) {
    return normalized as GoalStatus;
  }
  throw new Error(`invalid goal status: ${value}`);
}

function parseTaskStatus(value: string | undefined, fallback: TaskStatus = "open"): TaskStatus {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if ((TASK_STATUS_VALUES as readonly string[]).includes(normalized)) {
    return normalized as TaskStatus;
  }
  throw new Error(`invalid task status: ${value}`);
}

function parseDecisionActorType(value: string | undefined, fallback: DecisionActorType = "human"): DecisionActorType {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return fallback;
  }
  if ((DECISION_ACTOR_TYPE_VALUES as readonly string[]).includes(normalized)) {
    return normalized as DecisionActorType;
  }
  throw new Error(`invalid decision actor type: ${value}`);
}

function normalizeInboxSeverity(value: string | undefined, fallback: InboxSeverity = "medium"): InboxSeverity {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "low" || normalized === "medium" || normalized === "high" || normalized === "critical") {
    return normalized;
  }
  return fallback;
}

function assertAllowedTransition<T extends string>(
  label: string,
  current: T,
  next: T,
  transitions: Record<T, T[]>,
): void {
  if (current === next) {
    return;
  }
  if ((transitions[current] || []).includes(next)) {
    return;
  }
  throw new Error(`invalid ${label} transition: ${current} -> ${next}`);
}

function normalizedDueAt(value: string | undefined): string | undefined {
  const raw = String(value || "").trim();
  if (!raw) {
    return undefined;
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return undefined;
  }
  return parsed.toISOString();
}

function sourceRefPrimary(ref: SourceRef): { key: string; value: string } {
  for (const key of SOURCE_REF_PRIMARY_KEYS) {
    const value = String(ref[key] || "").trim();
    if (value) {
      return { key, value };
    }
  }
  throw new Error("sourceRef requires one primary identifier");
}

function normalizeSourceType(value: string | undefined): SourceRefType {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "paper" || normalized === "web" || normalized === "vault" || normalized === "scope" || normalized === "document") {
    return normalized;
  }
  if (normalized === "note") {
    return "vault";
  }
  return "web";
}

export function normalizeSourceRef(input: SourceRef): SourceRef {
  const ref: SourceRef = {
    sourceType: normalizeSourceType(input.sourceType),
  };
  for (const key of SOURCE_REF_PRIMARY_KEYS) {
    const value = String(input[key] || "").trim();
    if (value) {
      ref[key] = value;
    }
  }
  if (String(input.title || "").trim()) {
    ref.title = String(input.title).trim();
  }
  sourceRefPrimary(ref);
  return ref;
}

function refKey(ref: SourceRef): string {
  const primary = sourceRefPrimary(ref);
  return `${ref.sourceType}:${primary.key}:${primary.value}`;
}

function dedupeSourceRefs(refs: SourceRef[]): SourceRef[] {
  const seen = new Set<string>();
  const items: SourceRef[] = [];
  for (const item of refs) {
    const normalized = normalizeSourceRef(item);
    const key = refKey(normalized);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push(normalized);
  }
  return items;
}

function sameSourceRefs(left: SourceRef[], right: SourceRef[]): boolean {
  const a = dedupeSourceRefs(left).map(refKey).sort();
  const b = dedupeSourceRefs(right).map(refKey).sort();
  return JSON.stringify(a) === JSON.stringify(b);
}

function summarizeSourceRef(ref: SourceRef): string {
  if (ref.sourceType === "web" && ref.url) {
    return ref.title ? `[${ref.title}](${ref.url})` : ref.url;
  }
  if (ref.sourceType === "paper" && ref.paperId) {
    return ref.title ? `paper:${ref.paperId} (${ref.title})` : `paper:${ref.paperId}`;
  }
  if (ref.sourceType === "vault" && ref.noteId) {
    return ref.title ? `note:${ref.noteId} (${ref.title})` : `note:${ref.noteId}`;
  }
  if (ref.sourceType === "scope" && ref.stableScopeId) {
    return ref.title ? `scope:${ref.stableScopeId} (${ref.title})` : `scope:${ref.stableScopeId}`;
  }
  if (ref.sourceType === "document" && ref.documentScopeId) {
    return ref.title ? `document:${ref.documentScopeId} (${ref.title})` : `document:${ref.documentScopeId}`;
  }
  const primary = sourceRefPrimary(ref);
  return ref.title ? `${primary.key}:${primary.value} (${ref.title})` : `${primary.key}:${primary.value}`;
}

function uniqueProjectRefs(tasks: TaskRecord[], inbox: InboxItem[], decisions: DecisionRecord[] = []): SourceRef[] {
  const refs = [
    ...tasks.flatMap((task) => task.sourceRefs),
    ...inbox.flatMap((item) => item.sourceRefs),
    ...decisions.flatMap((item) => item.sourceRefs),
  ];
  return dedupeSourceRefs(refs);
}

function emptyProjectEvidenceGroups(): ProjectEvidenceGroupedBySourceType {
  return {
    paper: [],
    web: [],
    vault: [],
    scope: [],
    document: [],
  };
}

function groupSourceRefsByType(refs: SourceRef[]): ProjectEvidenceGroupedBySourceType {
  const groups = emptyProjectEvidenceGroups();
  for (const ref of dedupeSourceRefs(refs)) {
    groups[ref.sourceType].push(ref);
  }
  return groups;
}

function taskIdentity(input: {
  projectId: string;
  goalId?: string;
  title: string;
  kind: string;
}): string {
  return JSON.stringify({
    projectId: input.projectId,
    goalId: input.goalId || "",
    title: input.title.trim().toLowerCase(),
    kind: input.kind.trim().toLowerCase(),
  });
}

function goalIdentity(projectId: string, title: string): string {
  return `${projectId}::${title.trim().toLowerCase()}`;
}

function dueSoonSeverity(dueAt: string | undefined, now: string): InboxSeverity | null {
  if (!dueAt) {
    return null;
  }
  const due = new Date(dueAt);
  const current = new Date(now);
  if (Number.isNaN(due.getTime()) || Number.isNaN(current.getTime())) {
    return null;
  }
  const delta = due.getTime() - current.getTime();
  if (delta > 72 * 60 * 60 * 1000) {
    return null;
  }
  if (delta <= 0) {
    return "critical";
  }
  if (delta <= 24 * 60 * 60 * 1000) {
    return "high";
  }
  return "medium";
}

function normalizeInboxKind(value: string | undefined, fallback: InboxKind = "captured"): InboxKind {
  const text = emptyIfBlank(value).toLowerCase().replace(/[\s-]+/g, "_");
  if (!text) {
    return fallback;
  }
  return text as InboxKind;
}

function renderList(lines: string[], emptyText: string): string {
  if (!lines.length) {
    return `- ${emptyText}`;
  }
  return lines.join("\n");
}

function renderTaskLine(task: TaskRecord): string {
  const bits = [`- [${task.status}] ${task.title}`];
  if (task.kind) {
    bits.push(`kind=${task.kind}`);
  }
  if (task.priority) {
    bits.push(`priority=${task.priority}`);
  }
  if (task.assignee) {
    bits.push(`assignee=${task.assignee}`);
  }
  if (task.dueAt) {
    bits.push(`due=${task.dueAt}`);
  }
  if (task.blockedBy.length) {
    bits.push(`blockedBy=${task.blockedBy.join(", ")}`);
  }
  return bits.join(" | ");
}

function renderDecisionLine(decision: DecisionRecord): string {
  const bits = [`- [${decision.kind}] ${decision.summary}`];
  if (decision.taskId) {
    bits.push(`task=${decision.taskId}`);
  }
  if (decision.goalId) {
    bits.push(`goal=${decision.goalId}`);
  }
  if (decision.createdByType || decision.createdById) {
    bits.push(`actor=${decision.createdByType}:${decision.createdById}`);
  }
  if (decision.supersedesDecisionId) {
    bits.push(`supersedes=${decision.supersedesDecisionId}`);
  }
  return bits.join(" | ");
}

function renderSourceRefs(refs: SourceRef[], emptyText: string): string {
  return renderList(refs.map((ref) => `- ${summarizeSourceRef(ref)}`), emptyText);
}

function emptyIfBlank(value: string | undefined): string {
  return String(value || "").trim();
}

const PRIORITY_SORT_ORDER: Record<Priority, number> = {
  urgent: 0,
  high: 1,
  medium: 2,
  low: 3,
};

const SEVERITY_SORT_ORDER: Record<InboxSeverity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

function compareIsoAscending(left: string | undefined, right: string | undefined): number {
  if (!left && !right) {
    return 0;
  }
  if (!left) {
    return 1;
  }
  if (!right) {
    return -1;
  }
  return left.localeCompare(right);
}

function compareIsoDescending(left: string | undefined, right: string | undefined): number {
  return compareIsoAscending(right, left);
}

function compareTaskForNext(left: TaskRecord, right: TaskRecord): number {
  const priorityDelta = PRIORITY_SORT_ORDER[left.priority] - PRIORITY_SORT_ORDER[right.priority];
  if (priorityDelta !== 0) {
    return priorityDelta;
  }
  const dueDelta = compareIsoAscending(left.dueAt, right.dueAt);
  if (dueDelta !== 0) {
    return dueDelta;
  }
  const updatedDelta = compareIsoDescending(left.updatedAt, right.updatedAt);
  if (updatedDelta !== 0) {
    return updatedDelta;
  }
  return left.id.localeCompare(right.id);
}

function compareBlockedTask(left: TaskRecord, right: TaskRecord): number {
  const updatedDelta = compareIsoDescending(left.updatedAt, right.updatedAt);
  if (updatedDelta !== 0) {
    return updatedDelta;
  }
  return left.id.localeCompare(right.id);
}

function compareInboxItem(left: InboxItem, right: InboxItem): number {
  const severityDelta = SEVERITY_SORT_ORDER[left.severity] - SEVERITY_SORT_ORDER[right.severity];
  if (severityDelta !== 0) {
    return severityDelta;
  }
  const updatedDelta = compareIsoDescending(left.updatedAt, right.updatedAt);
  if (updatedDelta !== 0) {
    return updatedDelta;
  }
  return left.id.localeCompare(right.id);
}

function compareDecision(left: DecisionRecord, right: DecisionRecord): number {
  const createdDelta = compareIsoDescending(left.createdAt, right.createdAt);
  if (createdDelta !== 0) {
    return createdDelta;
  }
  return left.id.localeCompare(right.id);
}

function sortProjectsForNext(left: ProjectRecord, right: ProjectRecord): number {
  const priorityDelta = PRIORITY_SORT_ORDER[left.priority] - PRIORITY_SORT_ORDER[right.priority];
  if (priorityDelta !== 0) {
    return priorityDelta;
  }
  const updatedDelta = compareIsoDescending(left.updatedAt, right.updatedAt);
  if (updatedDelta !== 0) {
    return updatedDelta;
  }
  return left.id.localeCompare(right.id);
}

export class PersonalFoundryOsService {
  private readonly paths: PersonalFoundryPaths;
  private readonly eventBus: PersonalEventBus;
  private readonly now: () => string;

  constructor(input: { paths: PersonalFoundryPaths; eventBus: PersonalEventBus; now?: () => string }) {
    this.paths = input.paths;
    this.eventBus = input.eventBus;
    this.now = input.now ?? (() => new Date().toISOString());
  }

  private async loadState(): Promise<OsStateSnapshot> {
    const [projects, goals, tasks, inbox, decisions] = await Promise.all([
      readJsonArray<ProjectRecord>(this.paths.projectsPath),
      readJsonArray<GoalRecord>(this.paths.goalsPath),
      readJsonArray<TaskRecord>(this.paths.tasksPath),
      readJsonArray<InboxItem>(this.paths.inboxPath),
      readJsonArray<DecisionRecord>(this.paths.decisionsPath),
    ]);
    return { projects, goals, tasks, inbox, decisions };
  }

  private async writeState(state: OsStateSnapshot): Promise<void> {
    await Promise.all([
      writeJson(this.paths.projectsPath, state.projects),
      writeJson(this.paths.goalsPath, state.goals),
      writeJson(this.paths.tasksPath, state.tasks),
      writeJson(this.paths.inboxPath, state.inbox),
      writeJson(this.paths.decisionsPath, state.decisions),
    ]);
  }

  private async emit(type: string, payload: Record<string, unknown>): Promise<void> {
    await this.eventBus.publish({
      type,
      occurredAt: nowIso(this.now),
      sourceSystem: "manual",
      classification: "P1",
      payload,
    });
  }

  private findProject(state: OsStateSnapshot, selector: ProjectSelector): ProjectRecord {
    const projectId = emptyIfBlank(selector.projectId);
    const slug = emptyIfBlank(selector.slug);
    const project = state.projects.find((item) => (projectId && item.id === projectId) || (slug && item.slug === slug));
    if (!project) {
      throw new Error("project not found");
    }
    return project;
  }

  private projectScopedSummary(state: OsStateSnapshot, projectId: string): {
    goals: GoalRecord[];
    tasks: TaskRecord[];
    inbox: InboxItem[];
    decisions: DecisionRecord[];
    projectEvidence: ProjectEvidenceView;
    nextActionableTasks: TaskRecord[];
    blockedTasks: TaskRecord[];
    recentDecisions: DecisionRecord[];
  } {
    const goals = state.goals.filter((item) => item.projectId === projectId);
    const tasks = state.tasks.filter((item) => item.projectId === projectId);
    const inbox = state.inbox
      .filter((item) => item.projectId === projectId && item.state === "open")
      .sort(compareInboxItem);
    const decisions = state.decisions
      .filter((item) => item.projectId === projectId)
      .sort(compareDecision);
    const nextActionableTasks = tasks
      .filter((task) => ACTIONABLE_TASK_STATUSES.has(task.status))
      .sort(compareTaskForNext);
    const blockedTasks = tasks
      .filter((task) => task.status === "blocked")
      .sort(compareBlockedTask);
    const projectEvidence: ProjectEvidenceView = {
      projectId,
      sourceRefs: uniqueProjectRefs(tasks, inbox, decisions),
      taskLinkedRefs: tasks.flatMap((task) =>
        dedupeSourceRefs(task.sourceRefs).map((sourceRef) => ({
          taskId: task.id,
          title: task.title,
          sourceRef,
        }))
      ),
      decisionLinkedRefs: decisions.flatMap((decision) =>
        dedupeSourceRefs(decision.sourceRefs).map((sourceRef) => ({
          decisionId: decision.id,
          summary: decision.summary,
          sourceRef,
        }))
      ),
      inboxLinkedRefs: inbox.flatMap((item) =>
        dedupeSourceRefs(item.sourceRefs).map((sourceRef) => ({
          inboxId: item.id,
          summary: item.summary,
          sourceRef,
        }))
      ),
      groupedBySourceType: groupSourceRefsByType(uniqueProjectRefs(tasks, inbox, decisions)),
      recentDecisionEvidence: decisions.slice(0, 5).map((decision) => ({
        decisionId: decision.id,
        summary: decision.summary,
        sourceRefs: dedupeSourceRefs(decision.sourceRefs),
      })),
      nextTaskEvidence: nextActionableTasks.map((task) => ({
        taskId: task.id,
        title: task.title,
        sourceRefs: dedupeSourceRefs(task.sourceRefs),
      })),
    };
    return {
      goals,
      tasks,
      inbox,
      decisions,
      projectEvidence,
      nextActionableTasks,
      blockedTasks,
      recentDecisions: decisions.slice(0, 5),
    };
  }

  private async syncInbox(state: OsStateSnapshot, input: { opsAlerts?: OpsAlertInput[] } = {}): Promise<OsStateSnapshot> {
    const now = nowIso(this.now);
    const desired = new Map<string, InboxItem>();

    for (const task of state.tasks) {
      if (task.status === "blocked") {
        const id = `inbox_blocked_${task.id}`;
        desired.set(id, {
          id,
          projectId: task.projectId,
          taskId: task.id,
          kind: "blocked_task",
          severity: "high",
          state: "open",
          summary: `Blocked task: ${task.title}`,
          sourceRefs: task.sourceRefs,
          createdAt: now,
          updatedAt: now,
        });
      }
      if (OPEN_TASK_STATUSES.has(task.status)) {
        const severity = dueSoonSeverity(task.dueAt, now);
        if (severity) {
          const id = `inbox_due_soon_${task.id}`;
          desired.set(id, {
            id,
            projectId: task.projectId,
            taskId: task.id,
            kind: "due_soon_task",
            severity,
            state: "open",
            summary: `Due soon: ${task.title}`,
            sourceRefs: task.sourceRefs,
            createdAt: now,
            updatedAt: now,
          });
        }
      }
    }

    for (const alert of input.opsAlerts ?? []) {
      const alertId = emptyIfBlank(alert.alertId);
      const summary = emptyIfBlank(alert.summary);
      if (!alertId || !summary) {
        continue;
      }
      const id = `inbox_ops_${alertId}`;
      desired.set(id, {
        id,
        projectId: emptyIfBlank(alert.projectId) || undefined,
        kind: "ops_alert",
        severity: normalizeInboxSeverity(alert.severity, "medium"),
        state: "open",
        summary,
        sourceRefs: dedupeSourceRefs(alert.sourceRefs ?? []),
        createdAt: now,
        updatedAt: now,
      });
    }

    const nextInbox: InboxItem[] = [];
    for (const existing of state.inbox) {
      const desiredItem = desired.get(existing.id);
      if (!desiredItem) {
        if (existing.state === "resolved" || !MANAGED_INBOX_KINDS.has(existing.kind)) {
          nextInbox.push(existing);
        }
        continue;
      }

      if (existing.state === "resolved") {
        nextInbox.push(existing);
        desired.delete(existing.id);
        continue;
      }

      const changed = (
        existing.summary !== desiredItem.summary
        || existing.severity !== desiredItem.severity
        || existing.projectId !== desiredItem.projectId
        || existing.taskId !== desiredItem.taskId
        || !sameSourceRefs(existing.sourceRefs, desiredItem.sourceRefs)
      );
      if (changed) {
        nextInbox.push({
          ...desiredItem,
          createdAt: existing.createdAt,
          updatedAt: now,
        });
      } else {
        nextInbox.push(existing);
      }
      desired.delete(existing.id);
    }

    for (const item of desired.values()) {
      nextInbox.push(item);
      await this.emit("inbox.item_added", {
        inboxId: item.id,
        kind: item.kind,
        projectId: item.projectId ?? "",
        taskId: item.taskId ?? "",
        severity: item.severity,
      });
    }

    state.inbox = nextInbox.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
    return state;
  }

  async captureInboxItem(input: {
    projectId?: string;
    slug?: string;
    summary: string;
    kind?: string;
    severity?: string;
    sourceRefs?: SourceRef[];
    opsAlerts?: OpsAlertInput[];
  }): Promise<CaptureInboxResult> {
    const state = await this.syncInbox(await this.loadState(), {
      opsAlerts: input.opsAlerts,
    });
    const project = this.findProject(state, input);
    const summary = emptyIfBlank(input.summary);
    if (!summary) {
      throw new Error("inbox summary is required");
    }
    const now = nowIso(this.now);
    const item: InboxItem = {
      id: makeId("inbox"),
      projectId: project.id,
      kind: normalizeInboxKind(input.kind, "captured"),
      severity: normalizeInboxSeverity(input.severity, "medium"),
      state: "open",
      summary,
      sourceRefs: dedupeSourceRefs(input.sourceRefs ?? []),
      createdAt: now,
      updatedAt: now,
    };
    state.inbox.push(item);
    await this.writeState(state);
    await this.emit("inbox.item_added", {
      inboxId: item.id,
      kind: item.kind,
      projectId: item.projectId ?? "",
      taskId: "",
      severity: item.severity,
    });
    return {
      item,
      inbox: state.inbox.filter((entry) => entry.projectId === project.id && entry.state === "open"),
    };
  }

  async createProject(input: {
    title: string;
    slug?: string;
    status?: string;
    priority?: string;
    summary?: string;
    owner?: string;
  }): Promise<{ project: ProjectRecord; created: boolean }> {
    const state = await this.loadState();
    const now = nowIso(this.now);
    const title = emptyIfBlank(input.title);
    if (!title) {
      throw new Error("project title is required");
    }
    const slug = slugify(input.slug || title);
    const existing = state.projects.find((item) => item.slug === slug);
    if (existing) {
      const nextStatus = parseProjectStatus(input.status, existing.status);
      assertAllowedTransition("project status", existing.status, nextStatus, PROJECT_TRANSITIONS);
      const next: ProjectRecord = {
        ...existing,
        title,
        status: nextStatus,
        priority: parsePriority(input.priority, existing.priority),
        summary: emptyIfBlank(input.summary) || existing.summary,
        owner: emptyIfBlank(input.owner) || existing.owner,
      };
      const changed = JSON.stringify(existing) !== JSON.stringify(next);
      if (changed) {
        next.updatedAt = now;
        state.projects = state.projects.map((item) => (item.id === existing.id ? next : item));
        await this.writeState(state);
        await this.emit("project.updated", { projectId: next.id, slug: next.slug });
        return { project: next, created: false };
      }
      return { project: existing, created: false };
    }

    const project: ProjectRecord = {
      id: makeId("proj"),
      title,
      slug,
      status: parseProjectStatus(input.status),
      priority: parsePriority(input.priority),
      summary: emptyIfBlank(input.summary),
      owner: emptyIfBlank(input.owner),
      createdAt: now,
      updatedAt: now,
    };
    state.projects.push(project);
    await this.writeState(state);
    await this.emit("project.created", { projectId: project.id, slug: project.slug });
    return { project, created: true };
  }

  async listProjects(): Promise<ProjectRecord[]> {
    const state = await this.loadState();
    return [...state.projects].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async updateProject(input: {
    projectId?: string;
    slug?: string;
    title?: string;
    status?: string;
    priority?: string;
    summary?: string;
    owner?: string;
  }): Promise<ProjectRecord> {
    const state = await this.loadState();
    const project = this.findProject(state, input);
    const nextStatus = parseProjectStatus(input.status, project.status);
    assertAllowedTransition("project status", project.status, nextStatus, PROJECT_TRANSITIONS);
    const now = nowIso(this.now);
    const title = emptyIfBlank(input.title) || project.title;
    const next: ProjectRecord = {
      ...project,
      title,
      slug: project.slug,
      status: nextStatus,
      priority: parsePriority(input.priority, project.priority),
      summary: emptyIfBlank(input.summary) || project.summary,
      owner: emptyIfBlank(input.owner) || project.owner,
      updatedAt: now,
    };
    state.projects = state.projects.map((item) => (item.id === project.id ? next : item));
    await this.writeState(state);
    await this.emit("project.updated", { projectId: next.id, slug: next.slug });
    return next;
  }

  async showProject(selector: ProjectSelector, input: { opsAlerts?: OpsAlertInput[] } = {}): Promise<{
    project: ProjectRecord;
    goals: GoalRecord[];
    tasks: TaskRecord[];
    inbox: InboxItem[];
    decisions: DecisionRecord[];
    projectEvidence: ProjectEvidenceView;
    nextActionableTasks: TaskRecord[];
    blockedTasks: TaskRecord[];
    recentDecisions: DecisionRecord[];
  }> {
    const state = await this.syncInbox(await this.loadState(), input);
    await this.writeState(state);
    const project = this.findProject(state, selector);
    const scoped = this.projectScopedSummary(state, project.id);
    return {
      project,
      goals: scoped.goals,
      tasks: scoped.tasks,
      inbox: scoped.inbox,
      decisions: scoped.decisions,
      projectEvidence: scoped.projectEvidence,
      nextActionableTasks: scoped.nextActionableTasks,
      blockedTasks: scoped.blockedTasks,
      recentDecisions: scoped.recentDecisions,
    };
  }

  async projectEvidence(selector: ProjectSelector, input: { opsAlerts?: OpsAlertInput[] } = {}): Promise<{
    project: ProjectRecord;
    projectEvidence: ProjectEvidenceView;
  }> {
    const shown = await this.showProject(selector, input);
    return {
      project: shown.project,
      projectEvidence: shown.projectEvidence,
    };
  }

  async addGoal(input: {
    projectId?: string;
    slug?: string;
    title: string;
    status?: string;
    successCriteria?: string;
  }): Promise<{ goal: GoalRecord; created: boolean }> {
    const state = await this.loadState();
    const project = this.findProject(state, input);
    const title = emptyIfBlank(input.title);
    if (!title) {
      throw new Error("goal title is required");
    }
    const now = nowIso(this.now);
    const existing = state.goals.find((item) => goalIdentity(project.id, item.title) === goalIdentity(project.id, title));
    if (existing) {
      const nextStatus = parseGoalStatus(input.status, existing.status);
      assertAllowedTransition("goal status", existing.status, nextStatus, GOAL_TRANSITIONS);
      const next: GoalRecord = {
        ...existing,
        title,
        status: nextStatus,
        successCriteria: emptyIfBlank(input.successCriteria) || existing.successCriteria,
      };
      const changed = JSON.stringify(existing) !== JSON.stringify(next);
      if (changed) {
        next.updatedAt = now;
        state.goals = state.goals.map((item) => (item.id === existing.id ? next : item));
        await this.writeState(state);
        await this.emit("goal.updated", { goalId: next.id, projectId: project.id });
        return { goal: next, created: false };
      }
      return { goal: existing, created: false };
    }

    const goal: GoalRecord = {
      id: makeId("goal"),
      projectId: project.id,
      title,
      status: parseGoalStatus(input.status),
      successCriteria: emptyIfBlank(input.successCriteria),
      createdAt: now,
      updatedAt: now,
    };
    state.goals.push(goal);
    await this.writeState(state);
    await this.emit("goal.created", { goalId: goal.id, projectId: project.id });
    return { goal, created: true };
  }

  async updateGoal(input: {
    goalId: string;
    title?: string;
    status?: string;
    successCriteria?: string;
  }): Promise<GoalRecord> {
    const state = await this.loadState();
    const goalId = emptyIfBlank(input.goalId);
    const goal = state.goals.find((item) => item.id === goalId);
    if (!goal) {
      throw new Error("goal not found");
    }
    const nextStatus = parseGoalStatus(input.status, goal.status);
    assertAllowedTransition("goal status", goal.status, nextStatus, GOAL_TRANSITIONS);
    const now = nowIso(this.now);
    const next: GoalRecord = {
      ...goal,
      title: emptyIfBlank(input.title) || goal.title,
      status: nextStatus,
      successCriteria: emptyIfBlank(input.successCriteria) || goal.successCriteria,
      updatedAt: now,
    };
    state.goals = state.goals.map((item) => (item.id === goal.id ? next : item));
    await this.writeState(state);
    await this.emit("goal.updated", { goalId: next.id, projectId: next.projectId });
    return next;
  }

  async addTask(input: {
    projectId?: string;
    slug?: string;
    goalId?: string;
    title: string;
    kind?: string;
    status?: string;
    priority?: string;
    assignee?: string;
    dueAt?: string;
    blockedBy?: string[];
    sourceRefs?: SourceRef[];
    opsAlerts?: OpsAlertInput[];
  }): Promise<{ task: TaskRecord; created: boolean; inbox: InboxItem[] }> {
    const state = await this.loadState();
    const project = this.findProject(state, input);
    const title = emptyIfBlank(input.title);
    if (!title) {
      throw new Error("task title is required");
    }
    if (input.goalId && !state.goals.some((goal) => goal.id === input.goalId && goal.projectId === project.id)) {
      throw new Error("goal not found for task");
    }
    const now = nowIso(this.now);
    const identity = taskIdentity({
      projectId: project.id,
      goalId: input.goalId,
      title,
      kind: emptyIfBlank(input.kind) || "task",
    });
    const existing = state.tasks.find((item) => taskIdentity(item) === identity);
    const refs = dedupeSourceRefs(input.sourceRefs ?? []);
    const blockedBy = [...new Set((input.blockedBy ?? []).map((item) => emptyIfBlank(item)).filter(Boolean))];
    const dueAt = normalizedDueAt(input.dueAt);

    if (existing) {
      const nextStatus = parseTaskStatus(input.status, existing.status);
      assertAllowedTransition("task status", existing.status, nextStatus, TASK_TRANSITIONS);
      const next: TaskRecord = {
        ...existing,
        title,
        kind: emptyIfBlank(input.kind) || existing.kind,
        status: nextStatus,
        priority: parsePriority(input.priority, existing.priority),
        assignee: emptyIfBlank(input.assignee) || existing.assignee,
        dueAt: dueAt ?? existing.dueAt,
        blockedBy: blockedBy.length ? blockedBy : existing.blockedBy,
        sourceRefs: refs.length ? refs : existing.sourceRefs,
      };
      const changed = JSON.stringify(existing) !== JSON.stringify(next);
      if (changed) {
        next.updatedAt = now;
        state.tasks = state.tasks.map((item) => (item.id === existing.id ? next : item));
        await this.syncInbox(state, { opsAlerts: input.opsAlerts });
        await this.writeState(state);
        await this.emit("task.updated", { taskId: next.id, projectId: project.id });
        return {
          task: next,
          created: false,
          inbox: state.inbox.filter((item) => item.projectId === project.id && item.state === "open"),
        };
      }
      return {
        task: existing,
        created: false,
        inbox: state.inbox.filter((item) => item.projectId === project.id && item.state === "open"),
      };
    }

    const task: TaskRecord = {
      id: makeId("task"),
      projectId: project.id,
      goalId: emptyIfBlank(input.goalId) || undefined,
      title,
      kind: emptyIfBlank(input.kind) || "task",
      status: parseTaskStatus(input.status),
      priority: parsePriority(input.priority),
      assignee: emptyIfBlank(input.assignee),
      dueAt,
      blockedBy,
      sourceRefs: refs,
      createdAt: now,
      updatedAt: now,
    };
    state.tasks.push(task);
    await this.syncInbox(state, { opsAlerts: input.opsAlerts });
    await this.writeState(state);
    await this.emit("task.created", { taskId: task.id, projectId: project.id });
    return {
      task,
      created: true,
      inbox: state.inbox.filter((item) => item.projectId === project.id && item.state === "open"),
    };
  }

  async updateTask(input: {
    taskId: string;
    title?: string;
    kind?: string;
    priority?: string;
    assignee?: string;
    dueAt?: string;
    blockedBy?: string[];
    sourceRefs?: SourceRef[];
    opsAlerts?: OpsAlertInput[];
  }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    const state = await this.loadState();
    const taskId = emptyIfBlank(input.taskId);
    const existing = state.tasks.find((item) => item.id === taskId);
    if (!existing) {
      throw new Error("task not found");
    }
    const now = nowIso(this.now);
    const hasBlockedBy = Array.isArray(input.blockedBy) && input.blockedBy.length > 0;
    const hasSourceRefs = Array.isArray(input.sourceRefs) && input.sourceRefs.length > 0;
    const next: TaskRecord = {
      ...existing,
      title: emptyIfBlank(input.title) || existing.title,
      kind: emptyIfBlank(input.kind) || existing.kind,
      priority: parsePriority(input.priority, existing.priority),
      assignee: emptyIfBlank(input.assignee) || existing.assignee,
      dueAt: normalizedDueAt(input.dueAt) ?? existing.dueAt,
      blockedBy: hasBlockedBy
        ? [...new Set((input.blockedBy ?? []).map((item) => emptyIfBlank(item)).filter(Boolean))]
        : existing.blockedBy,
      sourceRefs: hasSourceRefs ? dedupeSourceRefs(input.sourceRefs ?? []) : existing.sourceRefs,
      updatedAt: now,
    };
    state.tasks = state.tasks.map((item) => (item.id === existing.id ? next : item));
    const opsAlerts = enrichOpsAlertsWithProjectId(input.opsAlerts, existing.projectId);
    await this.syncInbox(state, { opsAlerts });
    await this.writeState(state);
    await this.emit("task.updated", { taskId: next.id, projectId: next.projectId });
    return {
      task: next,
      inbox: state.inbox.filter((item) => item.projectId === next.projectId && item.state === "open"),
    };
  }

  async updateTaskStatus(input: {
    taskId: string;
    status: string;
    opsAlerts?: OpsAlertInput[];
  }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    const state = await this.loadState();
    const taskId = emptyIfBlank(input.taskId);
    const existing = state.tasks.find((item) => item.id === taskId);
    if (!existing) {
      throw new Error("task not found");
    }
    const opsAlerts = enrichOpsAlertsWithProjectId(input.opsAlerts, existing.projectId);
    const nextStatus = parseTaskStatus(input.status, existing.status);
    assertAllowedTransition("task status", existing.status, nextStatus, TASK_TRANSITIONS);
    if (nextStatus === existing.status) {
      await this.syncInbox(state, { opsAlerts });
      await this.writeState(state);
      return {
        task: existing,
        inbox: state.inbox.filter((item) => item.projectId === existing.projectId && item.state === "open"),
      };
    }
    const now = nowIso(this.now);
    const next: TaskRecord = {
      ...existing,
      status: nextStatus,
      updatedAt: now,
    };
    state.tasks = state.tasks.map((item) => (item.id === existing.id ? next : item));
    await this.syncInbox(state, { opsAlerts });
    await this.writeState(state);
    await this.emit("task.status_changed", {
      taskId: next.id,
      projectId: next.projectId,
      fromStatus: existing.status,
      toStatus: next.status,
    });
    await this.emit("task.updated", { taskId: next.id, projectId: next.projectId });
    return {
      task: next,
      inbox: state.inbox.filter((item) => item.projectId === next.projectId && item.state === "open"),
    };
  }

  async startTask(input: { taskId: string; opsAlerts?: OpsAlertInput[] }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    return this.updateTaskStatus({
      taskId: input.taskId,
      status: "in_progress",
      opsAlerts: input.opsAlerts,
    });
  }

  async completeTask(input: { taskId: string; opsAlerts?: OpsAlertInput[] }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    return this.updateTaskStatus({
      taskId: input.taskId,
      status: "completed",
      opsAlerts: input.opsAlerts,
    });
  }

  async cancelTask(input: { taskId: string; opsAlerts?: OpsAlertInput[] }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    return this.updateTaskStatus({
      taskId: input.taskId,
      status: "cancelled",
      opsAlerts: input.opsAlerts,
    });
  }

  async blockTask(input: {
    taskId: string;
    reason: string;
    opsAlerts?: OpsAlertInput[];
  }): Promise<{ task: TaskRecord; inbox: InboxItem[] }> {
    const taskId = emptyIfBlank(input.taskId);
    const reason = emptyIfBlank(input.reason);
    if (!reason) {
      throw new Error("block reason is required");
    }
    const state = await this.loadState();
    const existing = state.tasks.find((item) => item.id === taskId);
    if (!existing) {
      throw new Error("task not found");
    }
    const blockedBy = [...new Set([...(existing.blockedBy ?? []), reason])];
    await this.updateTask({
      taskId,
      blockedBy,
      opsAlerts: input.opsAlerts,
    });
    return this.updateTaskStatus({
      taskId,
      status: "blocked",
      opsAlerts: input.opsAlerts,
    });
  }

  async addDecision(input: {
    projectId?: string;
    slug?: string;
    goalId?: string;
    taskId?: string;
    kind: string;
    summary: string;
    rationale?: string;
    sourceRefs?: SourceRef[];
    createdByType?: string;
    createdById?: string;
    supersedesDecisionId?: string;
  }): Promise<DecisionRecord> {
    const state = await this.loadState();
    const project = this.findProject(state, input);
    const kind = emptyIfBlank(input.kind);
    const summary = emptyIfBlank(input.summary);
    if (!kind) {
      throw new Error("decision kind is required");
    }
    if (!summary) {
      throw new Error("decision summary is required");
    }
    const goalId = emptyIfBlank(input.goalId) || undefined;
    const taskId = emptyIfBlank(input.taskId) || undefined;
    if (goalId && !state.goals.some((item) => item.id === goalId && item.projectId === project.id)) {
      throw new Error("goal not found for decision");
    }
    if (taskId && !state.tasks.some((item) => item.id === taskId && item.projectId === project.id)) {
      throw new Error("task not found for decision");
    }
    const supersedesDecisionId = emptyIfBlank(input.supersedesDecisionId) || undefined;
    if (supersedesDecisionId && !state.decisions.some((item) => item.id === supersedesDecisionId && item.projectId === project.id)) {
      throw new Error("supersedes decision not found");
    }
    const now = nowIso(this.now);
    const decision: DecisionRecord = {
      id: makeId("decision"),
      projectId: project.id,
      goalId,
      taskId,
      kind,
      summary,
      rationale: emptyIfBlank(input.rationale),
      sourceRefs: dedupeSourceRefs(input.sourceRefs ?? []),
      createdByType: parseDecisionActorType(input.createdByType),
      createdById: emptyIfBlank(input.createdById) || "cli-user",
      supersedesDecisionId,
      createdAt: now,
      updatedAt: now,
    };
    state.decisions.push(decision);
    await this.writeState(state);
    await this.emit("decision.created", {
      decisionId: decision.id,
      projectId: decision.projectId,
      goalId: decision.goalId ?? "",
      taskId: decision.taskId ?? "",
      kind: decision.kind,
      createdByType: decision.createdByType,
    });
    return decision;
  }

  async listDecisions(input: {
    projectId?: string;
    slug?: string;
    goalId?: string;
    taskId?: string;
  } = {}): Promise<DecisionRecord[]> {
    const state = await this.loadState();
    let items = [...state.decisions];
    if (input.projectId || input.slug) {
      const project = this.findProject(state, input);
      items = items.filter((item) => item.projectId === project.id);
    }
    const goalId = emptyIfBlank(input.goalId);
    if (goalId) {
      items = items.filter((item) => item.goalId === goalId);
    }
    const taskId = emptyIfBlank(input.taskId);
    if (taskId) {
      items = items.filter((item) => item.taskId === taskId);
    }
    return items.sort(compareDecision);
  }

  async next(input: ProjectSelector & { opsAlerts?: OpsAlertInput[] } = {}): Promise<OsNextResult> {
    const state = await this.syncInbox(await this.loadState(), input);
    await this.writeState(state);
    const activeProjects = state.projects
      .filter((item) => item.status === "active")
      .sort(sortProjectsForNext);
    const activeIds = new Set(activeProjects.map((item) => item.id));
    const project = (input.projectId || input.slug)
      ? this.findProject(state, input)
      : (activeProjects[0] ?? null);
    const scopedProjectIds = project ? new Set([project.id]) : activeIds;
    const actionableTasks = state.tasks
      .filter((item) => scopedProjectIds.has(item.projectId) && ACTIONABLE_TASK_STATUSES.has(item.status))
      .sort(compareTaskForNext);
    const blockedTasks = state.tasks
      .filter((item) => scopedProjectIds.has(item.projectId) && item.status === "blocked")
      .sort(compareBlockedTask);
    const openInbox = state.inbox
      .filter((item) => item.state === "open" && (!item.projectId || scopedProjectIds.has(item.projectId)))
      .sort(compareInboxItem);
    const recentDecisions = state.decisions
      .filter((item) => scopedProjectIds.has(item.projectId))
      .sort(compareDecision);
    return {
      project,
      activeProjects,
      actionableTasks,
      blockedTasks,
      openInbox,
      recentDecisions,
    };
  }

  async listInbox(input: {
    state?: InboxState | "all";
    projectId?: string;
    slug?: string;
    opsAlerts?: OpsAlertInput[];
  } = {}): Promise<InboxItem[]> {
    const state = await this.syncInbox(await this.loadState(), { opsAlerts: input.opsAlerts });
    await this.writeState(state);
    let items = [...state.inbox];
    if (input.projectId || input.slug) {
      const project = this.findProject(state, input);
      items = items.filter((item) => item.projectId === project.id);
    }
    const filter = input.state ?? "open";
    if (filter !== "all") {
      items = items.filter((item) => item.state === filter);
    }
    return items.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async resolveInboxItem(input: {
    itemId: string;
    opsAlerts?: OpsAlertInput[];
  }): Promise<{ item: InboxItem; inbox: InboxItem[] }> {
    const state0 = await this.loadState();
    const itemId = emptyIfBlank(input.itemId);
    const seedItem = state0.inbox.find((item) => item.id === itemId);
    const opsAlerts = enrichOpsAlertsWithProjectId(input.opsAlerts, seedItem?.projectId);
    const state = await this.syncInbox(state0, { opsAlerts });
    const existing = state.inbox.find((item) => item.id === itemId);
    if (!existing) {
      throw new Error("inbox item not found");
    }
    if (existing.state !== "resolved") {
      existing.state = "resolved";
      existing.updatedAt = nowIso(this.now);
      await this.emit("inbox.item_resolved", {
        inboxId: existing.id,
        projectId: existing.projectId ?? "",
        taskId: existing.taskId ?? "",
      });
      await this.writeState(state);
    }
    const pid = String(existing.projectId || "").trim();
    const inboxOpen = state.inbox.filter((item) => item.state === "open");
    return {
      item: existing,
      inbox: pid !== "" ? inboxOpen.filter((item) => item.projectId === pid) : inboxOpen,
    };
  }

  async triageInboxItem(input: {
    itemId: string;
    action: "to_task" | "to_decision" | "resolve_only";
    taskTitle?: string;
    taskKind?: string;
    taskPriority?: string;
    taskAssignee?: string;
    taskDueAt?: string;
    taskGoalId?: string;
    taskBlockedBy?: string[];
    taskSourceRefs?: SourceRef[];
    decisionKind?: string;
    decisionSummary?: string;
    decisionRationale?: string;
    decisionGoalId?: string;
    decisionTaskId?: string;
    decisionCreatedByType?: string;
    decisionCreatedById?: string;
    decisionSupersedesDecisionId?: string;
    decisionSourceRefs?: SourceRef[];
    opsAlerts?: OpsAlertInput[];
  }): Promise<TriageInboxResult> {
    const state = await this.syncInbox(await this.loadState(), {
      opsAlerts: input.opsAlerts,
    });
    await this.writeState(state);
    const itemId = emptyIfBlank(input.itemId);
    const existing = state.inbox.find((item) => item.id === itemId);
    if (!existing) {
      throw new Error("inbox item not found");
    }
    if (existing.state === "resolved") {
      throw new Error("inbox item already resolved");
    }
    if (input.action === "resolve_only") {
      return this.resolveInboxItem({ itemId: existing.id, opsAlerts: input.opsAlerts });
    }
    if (!existing.projectId) {
      throw new Error("project-scoped inbox item required for triage");
    }

    if (input.action === "to_task") {
      const taskResult = await this.addTask({
        projectId: existing.projectId,
        goalId: emptyIfBlank(input.taskGoalId) || undefined,
        title: input.taskTitle || "",
        kind: emptyIfBlank(input.taskKind) || "task",
        priority: emptyIfBlank(input.taskPriority) || undefined,
        assignee: emptyIfBlank(input.taskAssignee) || undefined,
        dueAt: emptyIfBlank(input.taskDueAt) || undefined,
        blockedBy: input.taskBlockedBy ?? [],
        sourceRefs: dedupeSourceRefs([...(existing.sourceRefs ?? []), ...(input.taskSourceRefs ?? [])]),
        opsAlerts: input.opsAlerts,
      });
      const resolved = await this.resolveInboxItem({ itemId: existing.id, opsAlerts: input.opsAlerts });
      return {
        item: resolved.item,
        inbox: resolved.inbox,
        createdTask: taskResult.task,
      };
    }

    const decision = await this.addDecision({
      projectId: existing.projectId,
      goalId: emptyIfBlank(input.decisionGoalId) || undefined,
      taskId: emptyIfBlank(input.decisionTaskId) || undefined,
      kind: input.decisionKind || "",
      summary: input.decisionSummary || "",
      rationale: emptyIfBlank(input.decisionRationale) || undefined,
      sourceRefs: dedupeSourceRefs([...(existing.sourceRefs ?? []), ...(input.decisionSourceRefs ?? [])]),
      createdByType: emptyIfBlank(input.decisionCreatedByType) || undefined,
      createdById: emptyIfBlank(input.decisionCreatedById) || undefined,
      supersedesDecisionId: emptyIfBlank(input.decisionSupersedesDecisionId) || undefined,
    });
    const resolved = await this.resolveInboxItem({ itemId: existing.id, opsAlerts: input.opsAlerts });
    return {
      item: resolved.item,
      inbox: resolved.inbox,
      createdDecision: decision,
    };
  }

  async buildProjectProjection(selector: ProjectSelector, input: { opsAlerts?: OpsAlertInput[] } = {}): Promise<ProjectProjectionResult> {
    const shown = await this.showProject(selector, input);
    const state = await this.loadState();
    const project = shown.project;
    const tasksByStatus = new Map<TaskStatus, TaskRecord[]>();
    for (const status of ["open", "in_progress", "blocked", "completed", "cancelled"] as TaskStatus[]) {
      tasksByStatus.set(status, shown.tasks.filter((task) => task.status === status));
    }
    const projectRefs = shown.projectEvidence.sourceRefs;
    const activeProjects = state.projects
      .filter((item) => item.status === "active")
      .sort(sortProjectsForNext);
    const projections: MarkdownProjection[] = [
      {
        relativePath: `KnowledgeOS/Projects/${project.slug}/Project Brief.md`,
        title: `${project.title} Project Brief`,
        sections: [
          {
            key: "project-overview",
            body: [
              "## Project Overview",
              "",
              `- Status: ${project.status}`,
              `- Priority: ${project.priority}`,
              `- Owner: ${project.owner || "unassigned"}`,
              `- Goals: ${shown.goals.length}`,
              `- Tasks: ${shown.tasks.length}`,
              project.summary ? "" : "",
              project.summary ? project.summary : "_No summary yet._",
            ].filter(Boolean).join("\n"),
          },
          {
            key: "next-actionable-summary",
            body: [
              "## Next Actionable Tasks",
              "",
              renderList(shown.nextActionableTasks.slice(0, 10).map(renderTaskLine), "No actionable tasks."),
            ].join("\n"),
          },
          {
            key: "blocked-tasks-summary",
            body: [
              "## Blocked Tasks",
              "",
              renderList(shown.blockedTasks.slice(0, 10).map(renderTaskLine), "No blocked tasks."),
            ].join("\n"),
          },
          {
            key: "recent-decisions-summary",
            body: [
              "## Recent Decisions",
              "",
              renderList(shown.recentDecisions.map(renderDecisionLine), "No decisions yet."),
            ].join("\n"),
          },
          {
            key: "linked-sources",
            body: [
              "## Linked Sources",
              "",
              renderSourceRefs(projectRefs, "No linked sources."),
            ].join("\n"),
          },
        ],
      },
      {
        relativePath: `KnowledgeOS/Projects/${project.slug}/Goals.md`,
        title: `${project.title} Goals`,
        sections: [
          {
            key: "goal-list",
            body: [
              "## Goals",
              "",
              renderList(
                shown.goals.map((goal) =>
                  `- [${goal.status}] ${goal.title}${goal.successCriteria ? ` | success=${goal.successCriteria}` : ""}`
                ),
                "No goals."
              ),
            ].join("\n"),
          },
        ],
      },
      {
        relativePath: `KnowledgeOS/Projects/${project.slug}/Task Board.md`,
        title: `${project.title} Task Board`,
        sections: [
          {
            key: "next-actionable",
            body: [
              "## NEXT ACTIONABLE",
              "",
              renderList(shown.nextActionableTasks.map(renderTaskLine), "No actionable tasks."),
            ].join("\n"),
          },
          ...(["open", "in_progress", "blocked", "completed", "cancelled"] as TaskStatus[]).map((status) => ({
            key: `tasks-${status}`,
            body: [
              `## ${status.replace("_", " ").toUpperCase()}`,
              "",
              renderList((tasksByStatus.get(status) ?? []).map(renderTaskLine), "No tasks."),
            ].join("\n"),
          })),
        ],
      },
      {
        relativePath: `KnowledgeOS/Projects/${project.slug}/Decisions.md`,
        title: `${project.title} Decisions`,
        sections: [
          {
            key: "decision-log",
            body: [
              "## Decisions",
              "",
              renderList(shown.decisions.map(renderDecisionLine), "No decisions."),
            ].join("\n"),
          },
          {
            key: "decision-evidence",
            body: [
              "## Decision Evidence",
              "",
              renderSourceRefs(
                dedupeSourceRefs(shown.decisions.flatMap((decision) => decision.sourceRefs)),
                "No linked decision evidence."
              ),
            ].join("\n"),
          },
        ],
      },
      {
        relativePath: `KnowledgeOS/Projects/${project.slug}/Evidence.md`,
        title: `${project.title} Evidence`,
        sections: [
          {
            key: "grouped-evidence",
            body: [
              "## Grouped Evidence",
              "",
              "### Paper",
              renderSourceRefs(shown.projectEvidence.groupedBySourceType.paper, "No paper refs."),
              "",
              "### Web",
              renderSourceRefs(shown.projectEvidence.groupedBySourceType.web, "No web refs."),
              "",
              "### Vault",
              renderSourceRefs(shown.projectEvidence.groupedBySourceType.vault, "No vault refs."),
              "",
              "### Scope",
              renderSourceRefs(shown.projectEvidence.groupedBySourceType.scope, "No scope refs."),
              "",
              "### Document",
              renderSourceRefs(shown.projectEvidence.groupedBySourceType.document, "No document refs."),
            ].join("\n"),
          },
          {
            key: "next-task-evidence",
            body: [
              "## Next Task Evidence",
              "",
              renderList(
                shown.projectEvidence.nextTaskEvidence.map((item) => {
                  const refs = item.sourceRefs.length
                    ? item.sourceRefs.map(summarizeSourceRef).join(", ")
                    : "no refs";
                  return `- ${item.title} | refs=${refs}`;
                }),
                "No next-task evidence."
              ),
            ].join("\n"),
          },
          {
            key: "recent-decision-evidence",
            body: [
              "## Recent Decision Evidence",
              "",
              renderList(
                shown.projectEvidence.recentDecisionEvidence.map((item) => {
                  const refs = item.sourceRefs.length
                    ? item.sourceRefs.map(summarizeSourceRef).join(", ")
                    : "no refs";
                  return `- ${item.summary} | refs=${refs}`;
                }),
                "No recent decision evidence."
              ),
            ].join("\n"),
          },
          {
            key: "inbox-linked-evidence",
            body: [
              "## Inbox-linked Evidence",
              "",
              renderList(
                shown.projectEvidence.inboxLinkedRefs.map((item) =>
                  `- ${item.summary} -> ${summarizeSourceRef(item.sourceRef)}`
                ),
                "No inbox-linked evidence."
              ),
            ].join("\n"),
          },
        ],
      },
      {
        relativePath: "KnowledgeOS/Projects/Index.md",
        title: "KnowledgeOS Projects Index",
        sections: [
          {
            key: "active-projects",
            body: [
              "## Active Projects",
              "",
              renderList(
                activeProjects.map((item) => {
                  const scoped = this.projectScopedSummary(state, item.id);
                  return (
                    `- [${item.status}] ${item.title} (${item.slug})`
                    + ` | owner=${item.owner || "unassigned"}`
                    + ` | priority=${item.priority}`
                    + ` | goals=${scoped.goals.length}`
                    + ` | tasks=${scoped.tasks.length}`
                    + ` | inbox=${scoped.inbox.length}`
                    + ` | decisions=${scoped.recentDecisions.length}`
                  );
                }),
                "No active projects."
              ),
            ].join("\n"),
          },
        ],
      },
      {
        relativePath: "KnowledgeOS/Inbox.md",
        title: "KnowledgeOS Inbox",
        sections: [
          {
            key: `project-${project.slug}-inbox`,
            body: [
              `## ${project.title}`,
              "",
              renderList(
                shown.inbox.map((item) => {
                  const refs = item.sourceRefs.length ? ` | refs=${item.sourceRefs.map(summarizeSourceRef).join(", ")}` : "";
                  return `- [${item.severity}] ${item.summary}${refs}`;
                }),
                "No open inbox items."
              ),
            ].join("\n"),
          },
        ],
      },
    ];
    await this.emit("projection.refreshed", { projectId: project.id, slug: project.slug, projectionCount: projections.length });
    return {
      project,
      goals: shown.goals,
      tasks: shown.tasks,
      inbox: shown.inbox,
      decisions: shown.decisions,
      projectEvidence: shown.projectEvidence,
      nextActionableTasks: shown.nextActionableTasks,
      blockedTasks: shown.blockedTasks,
      recentDecisions: shown.recentDecisions,
      projections,
    };
  }
}

/** Attach projectId when khub passes ops JSON without projectId (matches task add / show / export enrichment). */
function enrichOpsAlertsWithProjectId(
  alerts: OpsAlertInput[] | undefined,
  projectId: string | undefined,
): OpsAlertInput[] | undefined {
  const pid = String(projectId || "").trim();
  if (!alerts?.length || !pid) {
    return alerts;
  }
  if (!alerts.some((item) => !String(item.projectId || "").trim())) {
    return alerts;
  }
  return alerts.map((item) => (String(item.projectId || "").trim() ? item : { ...item, projectId: pid }));
}

/** Attach projectId to ops alerts when the CLI is scoped to a project but SQLite payloads omit it. */
export async function enrichOpsAlertsForSelector(
  service: PersonalFoundryOsService,
  selector: ProjectSelector,
  alerts: OpsAlertInput[],
): Promise<OpsAlertInput[]> {
  if (!alerts.length) {
    return alerts;
  }
  const missing = alerts.some((item) => !String(item.projectId || "").trim());
  if (!missing) {
    return alerts;
  }
  let resolvedId = String(selector.projectId || "").trim();
  if (!resolvedId && String(selector.slug || "").trim()) {
    const projects = await service.listProjects();
    const slug = String(selector.slug || "").trim();
    resolvedId = projects.find((item) => item.slug === slug)?.id || "";
  }
  if (!resolvedId) {
    return alerts;
  }
  return alerts.map((item) => (String(item.projectId || "").trim() ? item : { ...item, projectId: resolvedId }));
}
