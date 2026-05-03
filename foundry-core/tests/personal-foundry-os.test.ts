import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { JsonlEventBus } from "../src/personal-foundry/event-bus.js";
import { enrichOpsAlertsForSelector, PersonalFoundryOsService } from "../src/personal-foundry/os-state.js";
import { initPersonalFoundryProject } from "../src/personal-foundry/project.js";

describe("personal-foundry project os", () => {
  it("persists project/goal/task/decision state and emits expected events without duplicates", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-project-os-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-07T00:00:00.000Z",
    });

    const created = await service.createProject({
      title: "KnowledgeOS MVP",
      slug: "knowledgeos-mvp",
      summary: "first project",
      owner: "won",
    });
    assert.equal(created.created, true);

    const repeated = await service.createProject({
      title: "KnowledgeOS MVP",
      slug: "knowledgeos-mvp",
      summary: "first project",
      owner: "won",
    });
    assert.equal(repeated.created, false);

    const goal = await service.addGoal({
      projectId: created.project.id,
      title: "Ship Project OS",
      successCriteria: "Project, task, inbox all visible",
    });
    assert.equal(goal.created, true);

    const task = await service.addTask({
      projectId: created.project.id,
      goalId: goal.goal.id,
      title: "Implement blocked flow",
      kind: "delivery",
      status: "blocked",
      dueAt: "2026-04-08T00:00:00.000Z",
      sourceRefs: [
        { sourceType: "paper", paperId: "1706.03762" },
        { sourceType: "web", url: "https://example.com/os-mvp" },
      ],
    });
    assert.equal(task.created, true);
    assert.equal(task.inbox.length, 2);

    const decision = await service.addDecision({
      projectId: created.project.id,
      goalId: goal.goal.id,
      taskId: task.task.id,
      kind: "architecture",
      summary: "Keep Obsidian projection-only.",
      rationale: "Canonical state should remain local and structured.",
      sourceRefs: [{ sourceType: "document", documentScopeId: "knowledge_os_definition_v1" }],
      createdByType: "human",
      createdById: "won",
    });
    assert.equal(decision.projectId, created.project.id);

    const repeatedTask = await service.addTask({
      projectId: created.project.id,
      goalId: goal.goal.id,
      title: "Implement blocked flow",
      kind: "delivery",
      status: "blocked",
    });
    assert.equal(repeatedTask.created, false);

    const listed = await service.listProjects();
    assert.equal(listed.length, 1);

    const shown = await service.showProject({ projectId: created.project.id });
    assert.equal(shown.goals.length, 1);
    assert.equal(shown.tasks.length, 1);
    assert.equal(shown.inbox.length, 2);
    assert.equal(shown.decisions.length, 1);
    assert.equal(shown.recentDecisions.length, 1);
    assert.equal(shown.projectEvidence.sourceRefs.length, 3);
    assert.equal(shown.projectEvidence.taskLinkedRefs.length, 2);
    assert.equal(shown.projectEvidence.decisionLinkedRefs.length, 1);

    const rawProjects = JSON.parse(fs.readFileSync(init.paths.projectsPath, "utf8"));
    const rawGoals = JSON.parse(fs.readFileSync(init.paths.goalsPath, "utf8"));
    const rawTasks = JSON.parse(fs.readFileSync(init.paths.tasksPath, "utf8"));
    const rawInbox = JSON.parse(fs.readFileSync(init.paths.inboxPath, "utf8"));
    const rawDecisions = JSON.parse(fs.readFileSync(init.paths.decisionsPath, "utf8"));
    assert.equal(rawProjects.length, 1);
    assert.equal(rawGoals.length, 1);
    assert.equal(rawTasks.length, 1);
    assert.equal(rawInbox.length, 2);
    assert.equal(rawDecisions.length, 1);

    const events = await eventBus.read();
    const eventTypes = events.map((event) => event.type);
    assert.equal(eventTypes.includes("project.created"), true);
    assert.equal(eventTypes.includes("goal.created"), true);
    assert.equal(eventTypes.includes("task.created"), true);
    assert.equal(eventTypes.includes("decision.created"), true);
    assert.equal(eventTypes.includes("inbox.item_added"), true);

    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("builds obsidian projections and resolved inbox items stay out of open inbox", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-project-os-export-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-07T00:00:00.000Z",
    });

    const project = await service.createProject({ title: "KnowledgeOS MVP", slug: "knowledgeos-mvp" });
    await service.addTask({
      projectId: project.project.id,
      title: "Link sources",
      kind: "research",
      status: "blocked",
      sourceRefs: [
        { sourceType: "paper", paperId: "1706.03762", title: "Attention Is All You Need" },
        { sourceType: "web", url: "https://example.com/os-mvp", title: "OS MVP note" },
      ],
    });
    await service.addDecision({
      projectId: project.project.id,
      kind: "scope",
      summary: "Keep Obsidian projection-only.",
      rationale: "Projection should stay one-way.",
      sourceRefs: [{ sourceType: "document", documentScopeId: "knowledge_os_definition_v1" }],
      createdById: "won",
    });

    const before = await service.listInbox({ projectId: project.project.id });
    assert.equal(before.length >= 1, true);

    const projection = await service.buildProjectProjection({ projectId: project.project.id });
    assert.equal(projection.projections.length, 7);
    assert.equal(projection.projections.every((item) => item.relativePath.startsWith("KnowledgeOS/")), true);
    assert.equal(projection.projections.some((item) => item.relativePath.startsWith("Projects/")), false);
    const brief = projection.projections.find((item) => item.relativePath.endsWith("Project Brief.md"));
    assert.ok(brief);
    const linkedSources = brief?.sections.find((section) => section.key === "linked-sources");
    const blockedSummary = brief?.sections.find((section) => section.key === "blocked-tasks-summary");
    const decisionsSummary = brief?.sections.find((section) => section.key === "recent-decisions-summary");
    assert.equal(String(linkedSources?.body || "").includes("paper:1706.03762"), true);
    assert.equal(String(linkedSources?.body || "").includes("https://example.com/os-mvp"), true);
    assert.equal(String(blockedSummary?.body || "").includes("Link sources"), true);
    assert.equal(String(decisionsSummary?.body || "").includes("Keep Obsidian projection-only."), true);

    const decisionsDoc = projection.projections.find((item) => item.relativePath.endsWith("Decisions.md"));
    assert.ok(decisionsDoc);
    assert.equal(String(decisionsDoc?.sections[0]?.body || "").includes("Keep Obsidian projection-only."), true);
    const evidenceDoc = projection.projections.find((item) => item.relativePath.endsWith("Evidence.md"));
    assert.ok(evidenceDoc);
    assert.equal(String(evidenceDoc?.sections[0]?.body || "").includes("document:knowledge_os_definition_v1"), true);
    const indexDoc = projection.projections.find((item) => item.relativePath === "KnowledgeOS/Projects/Index.md");
    assert.ok(indexDoc);
    assert.equal(String(indexDoc?.sections[0]?.body || "").includes("KnowledgeOS MVP"), true);

    await service.resolveInboxItem({ itemId: before[0].id });
    const after = await service.listInbox({ projectId: project.project.id });
    assert.equal(after.some((item) => item.id === before[0].id), false);

    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("updateTaskStatus attaches task projectId to ops alerts so show/export stay consistent with task add", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-ops-update-status-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-07T00:00:00.000Z",
    });
    const project = await service.createProject({ title: "P", slug: "p" });
    const added = await service.addTask({
      projectId: project.project.id,
      title: "Do work",
      status: "open",
      opsAlerts: [],
    });
    const updated = await service.updateTaskStatus({
      taskId: added.task.id,
      status: "in_progress",
      opsAlerts: [{ alertId: "guard-1", summary: "SQLite guardrail", severity: "medium" }],
    });
    const ops = updated.inbox.filter((item) => item.kind === "ops_alert");
    assert.equal(ops.length, 1);
    assert.equal(ops[0]?.projectId, project.project.id);
    assert.equal(ops[0]?.summary.includes("SQLite guardrail"), true);
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("enrichOpsAlertsForSelector attaches projectId so ops alerts appear in project-scoped show", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-ops-enrich-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-07T00:00:00.000Z",
    });
    const created = await service.createProject({ title: "Scoped", slug: "scoped-slug" });
    const rawAlerts = [{ alertId: "sqlite1", summary: "Ops from SQLite", severity: "high" as const }];
    const enriched = await enrichOpsAlertsForSelector(service, { slug: "scoped-slug" }, rawAlerts);
    assert.equal(enriched[0]?.projectId, created.project.id);
    const shown = await service.showProject({ slug: "scoped-slug" }, { opsAlerts: enriched });
    const ops = shown.inbox.filter((item) => item.kind === "ops_alert");
    assert.equal(ops.length, 1);
    assert.equal(ops[0]?.summary.includes("Ops from SQLite"), true);
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("enforces explicit lifecycle transitions for project, goal, and task", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-os-transitions-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-09T00:00:00.000Z",
    });

    const project = await service.createProject({ title: "Decision OS", slug: "decision-os" });
    const goal = await service.addGoal({ projectId: project.project.id, title: "Ship lifecycle rules" });
    const task = await service.addTask({ projectId: project.project.id, goalId: goal.goal.id, title: "Enforce transitions" });

    await assert.rejects(
      service.updateGoal({ goalId: goal.goal.id, status: "completed" }),
      /invalid goal status transition: open -> completed/
    );
    await assert.rejects(
      service.updateTaskStatus({ taskId: task.task.id, status: "completed" }),
      /invalid task status transition: open -> completed/
    );

    const movedGoal = await service.updateGoal({ goalId: goal.goal.id, status: "in_progress" });
    assert.equal(movedGoal.status, "in_progress");
    const movedTask = await service.updateTaskStatus({ taskId: task.task.id, status: "in_progress" });
    assert.equal(movedTask.task.status, "in_progress");
    const movedProject = await service.updateProject({ projectId: project.project.id, status: "on_hold" });
    assert.equal(movedProject.status, "on_hold");
    const completedProject = await service.updateProject({ projectId: project.project.id, status: "completed" });
    assert.equal(completedProject.status, "completed");
    await assert.rejects(
      service.updateProject({ projectId: project.project.id, status: "active" }),
      /invalid project status transition: completed -> active/
    );

    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("persists decisions, supports supersession, and ranks next view by priority then due date", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-os-decisions-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    let tick = 0;
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => new Date(Date.UTC(2026, 3, 9, 0, 0, tick++)).toISOString(),
    });

    const project = await service.createProject({ title: "Decision OS", slug: "decision-os", priority: "urgent" });
    const goal = await service.addGoal({ projectId: project.project.id, title: "Ship decision model", status: "in_progress" });
    const first = await service.addTask({
      projectId: project.project.id,
      goalId: goal.goal.id,
      title: "Implement decision add",
      kind: "delivery",
      priority: "high",
      dueAt: "2026-04-11T00:00:00.000Z",
      status: "open",
    });
    const second = await service.addTask({
      projectId: project.project.id,
      goalId: goal.goal.id,
      title: "Implement next view",
      kind: "delivery",
      priority: "urgent",
      dueAt: "2026-04-12T00:00:00.000Z",
      status: "open",
    });
    await service.addTask({
      projectId: project.project.id,
      goalId: goal.goal.id,
      title: "Investigate bridge edge cases",
      kind: "research",
      priority: "medium",
      status: "blocked",
    });

    const decision = await service.addDecision({
      projectId: project.project.id,
      goalId: goal.goal.id,
      taskId: first.task.id,
      kind: "implementation",
      summary: "Add explicit decision records",
      rationale: "Tasks alone are insufficient",
      createdById: "won",
      sourceRefs: [{ sourceType: "web", url: "https://example.com/decision-os" }],
    });
    const superseding = await service.addDecision({
      projectId: project.project.id,
      goalId: goal.goal.id,
      taskId: second.task.id,
      kind: "implementation",
      summary: "Prioritize the next view after decision persistence",
      rationale: "Read model closes the operating loop",
      createdByType: "agent",
      createdById: "planner",
      supersedesDecisionId: decision.id,
    });

    const decisions = await service.listDecisions({ projectId: project.project.id });
    assert.equal(decisions.length, 2);
    assert.equal(decisions[0]?.id, superseding.id);
    assert.equal(decisions[0]?.supersedesDecisionId, decision.id);

    const next = await service.next();
    assert.equal(next.activeProjects[0]?.id, project.project.id);
    assert.equal(next.actionableTasks[0]?.id, second.task.id);
    assert.equal(next.actionableTasks[1]?.id, first.task.id);
    assert.equal(next.blockedTasks.length, 1);
    assert.equal(next.recentDecisions[0]?.id, superseding.id);

    const shown = await service.showProject({ projectId: project.project.id });
    assert.equal(shown.decisions.length, 2);
    assert.equal(shown.projectEvidence.sourceRefs.length, 1);
    assert.equal(shown.projectEvidence.recentDecisionEvidence[0]?.decisionId, superseding.id);
    assert.equal(shown.nextActionableTasks[0]?.id, second.task.id);
    assert.equal(shown.recentDecisions[0]?.id, superseding.id);

    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("supports capture, inbox triage, and task workflow verbs without new storage", async () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-os-workflow-"));
    const init = initPersonalFoundryProject({ projectRoot: tmp });
    const eventBus = new JsonlEventBus({ logPath: init.paths.eventBusLogPath });
    const service = new PersonalFoundryOsService({
      paths: init.paths,
      eventBus,
      now: () => "2026-04-09T00:00:00.000Z",
    });

    const project = await service.createProject({ title: "Workflow OS", slug: "workflow-os" });
    const captured = await service.captureInboxItem({
      projectId: project.project.id,
      summary: "Turn source review into tracked work",
      kind: "captured",
      severity: "high",
      sourceRefs: [{ sourceType: "paper", paperId: "1706.03762" }],
    });
    assert.equal(captured.item.kind, "captured");
    assert.equal(captured.inbox.some((item) => item.id === captured.item.id), true);

    const triagedTask = await service.triageInboxItem({
      itemId: captured.item.id,
      action: "to_task",
      taskTitle: "Review source review notes",
      taskKind: "research",
      taskPriority: "high",
    });
    assert.equal(triagedTask.item.state, "resolved");
    assert.equal(triagedTask.createdTask?.title, "Review source review notes");
    assert.equal(triagedTask.createdTask?.sourceRefs[0]?.paperId, "1706.03762");

    const started = await service.startTask({ taskId: triagedTask.createdTask!.id });
    assert.equal(started.task.status, "in_progress");

    const blocked = await service.blockTask({
      taskId: triagedTask.createdTask!.id,
      reason: "waiting_for_source_fix",
    });
    assert.equal(blocked.task.status, "blocked");
    assert.equal(blocked.task.blockedBy.includes("waiting_for_source_fix"), true);

    const completed = await service.updateTaskStatus({
      taskId: triagedTask.createdTask!.id,
      status: "in_progress",
    });
    assert.equal(completed.task.status, "in_progress");
    const done = await service.completeTask({ taskId: triagedTask.createdTask!.id });
    assert.equal(done.task.status, "completed");

    const capturedDecision = await service.captureInboxItem({
      projectId: project.project.id,
      summary: "Record a scope decision",
      sourceRefs: [{ sourceType: "document", documentScopeId: "knowledge_os_definition_v1" }],
    });
    const triagedDecision = await service.triageInboxItem({
      itemId: capturedDecision.item.id,
      action: "to_decision",
      decisionKind: "scope",
      decisionSummary: "Keep CLI as the canonical operator interface",
      decisionCreatedById: "won",
    });
    assert.equal(triagedDecision.item.state, "resolved");
    assert.equal(triagedDecision.createdDecision?.kind, "scope");
    assert.equal(triagedDecision.createdDecision?.sourceRefs[0]?.documentScopeId, "knowledge_os_definition_v1");

    const cancelledCapture = await service.captureInboxItem({
      projectId: project.project.id,
      summary: "Cancel a non-essential cleanup task",
    });
    const cancelledTask = await service.triageInboxItem({
      itemId: cancelledCapture.item.id,
      action: "to_task",
      taskTitle: "Cleanup legacy notes",
    });
    const cancelled = await service.cancelTask({ taskId: cancelledTask.createdTask!.id });
    assert.equal(cancelled.task.status, "cancelled");

    const next = await service.next({ projectId: project.project.id });
    assert.equal(next.recentDecisions[0]?.id, triagedDecision.createdDecision?.id);

    fs.rmSync(tmp, { recursive: true, force: true });
  });
});
