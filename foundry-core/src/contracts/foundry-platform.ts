import type {
  DataClassification,
  DataRecord,
  DataClassificationRule,
  OntologyEntity,
  OntologyEvent,
  OntologyRelation,
  OntologySnapshot,
  OntologySourceRecord,
  ConnectorContract,
  ConnectorEmitInput,
  ConnectorEmitResult,
  ConnectorRunInput,
  ConnectorRunResult,
  ConnectorSyncInput,
  ConnectorSyncStateStore,
  ConnectorIdempotencyStore,
  ConnectorRuntimeContext,
  FoundryEventBus,
  AuditSink,
  PolicyAction,
  PolicyDecision,
  PolicyContext,
  AgentRuntimeInput,
  AgentRuntimeResult,
  FoundryToolRegistry,
  RunArtifact,
  TimeSeriesPoint,
  FoundryTool,
  StageTransition,
} from "../types.js";

export type SourceSystem = "knowledge_hub" | "calendar" | "finance" | "sleep" | "behavior" | "manual";

export type PrivacyClass = DataClassification;
export type SanitizationProfile = "raw" | "facts" | "summary" | "public";
export type CanonicalEntityType =
  | "KnowledgeItem"
  | "Paper"
  | "Task"
  | "Schedule"
  | "Expense"
  | "SleepLog"
  | "BehaviorEvent"
  | "Person"
  | "Metric";

export interface ClassificationContext {
  userId?: string;
  sourceSystem?: SourceSystem;
  tenant?: string;
  project?: string;
  tags?: string[];
}

export interface RawPayloadEnvelope {
  schema: "knowledge-hub.foundry.raw.payload.v1";
  source: "knowledge-hub.cli" | "connector" | "manual";
  sourceSystem: SourceSystem;
  actorId: string;
  receivedAt: string;
  correlationId: string;
  payload: Record<string, unknown>;
  classification: PrivacyClass;
}

export interface OntologyEnvelope {
  schema: "knowledge-hub.foundry.ontology.batch.v1";
  source: "connector" | "agent" | "manual";
  sourceSystem: SourceSystem;
  actorId: string;
  receivedAt: string;
  cursor?: string;
  entityBatch: OntologyEntity[];
  relationBatch: OntologyRelation[];
  eventBatch: OntologyEvent[];
  provenance: OntologySourceRecord[];
}

export interface TimeSeriesEnvelope {
  schema: "knowledge-hub.foundry.timeseries.batch.v1";
  sourceSystem: SourceSystem;
  actorId: string;
  key: string;
  entityId: string;
  points: TimeSeriesPoint[];
  measuredAt: string;
}

export interface FeatureQuery {
  name: string;
  intent: "read" | "analyze" | "forecast" | "summarize" | "compare" | "alert";
  params: Record<string, unknown>;
  filters?: Record<string, string | number | boolean | null>;
  timeframe?: {
    from?: string;
    to?: string;
  };
}

export interface FeatureResult {
  featureName: string;
  status: "ok" | "partial" | "denied" | "error";
  payload: Record<string, unknown>;
  generatedAt: string;
  provenance: OntologySourceRecord[];
  classification: PrivacyClass;
}

export interface FeatureFunction<TInput = Record<string, unknown>, TOutput extends FeatureResult = FeatureResult> {
  id: string;
  name: string;
  description?: string;
  version: string;
  classificationPolicy?: {
    maxInputClass: PrivacyClass;
    maxOutputClass: PrivacyClass;
  };
  execute(input: TInput, context: ClassificationContext): Promise<TOutput>;
}

export interface FeatureRegistry {
  register(feature: FeatureFunction): Promise<void>;
  get(id: string): FeatureFunction | undefined;
  list(): FeatureFunction[];
}

export interface LocalFirstPolicyEngine {
  evaluate(context: PolicyContext & { sanitization?: SanitizationProfile }): Promise<PolicyDecision>;
  classify(data: unknown): PrivacyClass;
  requiresExternalCall(context: {
    actorId: string;
    action: PolicyAction;
    resourceId: string;
    classification: PrivacyClass;
  }): boolean;
}

export interface ExternalCallContract {
  action: PolicyAction;
  name: string;
  allowedProfiles: SanitizationProfile[];
  requestSchema?: Record<string, unknown>;
  responseSchema?: Record<string, unknown>;
  maxPayloadBytes?: number;
}

export interface AuditLogWriter {
  append(event: {
    eventId: string;
    eventType: string;
    at: string;
    actorId: string;
    requestId?: string;
    runId?: string;
    action: PolicyAction | string;
    resourceType: string;
    resourceId: string;
    allowed: boolean;
    classification?: PrivacyClass;
    metadata?: Record<string, unknown>;
  }): Promise<void>;
}

export interface OntologyStoreAdapter {
  appendEvent(event: Omit<OntologyEvent, "id" | "sequence">): Promise<OntologyEvent>;
  readEvents(input: {
    aggregateId: string;
    fromSequence?: number;
    toSequence?: number;
  }): Promise<OntologyEvent[]>;
  upsertEntity(entity: OntologyEntity): Promise<void>;
  upsertRelation(relation: OntologyRelation): Promise<void>;
  appendSnapshot(snapshot: Omit<OntologySnapshot, "id">): Promise<string>;
  readLatestSnapshot(streamId: string): Promise<OntologySnapshot | null>;
  appendTimeSeries(payload: { key: string; entityId: string; points: TimeSeriesPoint[] }): Promise<void>;
  getTimeSeries(input: { key: string; entityId: string; from?: string; to?: string }): Promise<TimeSeriesPoint[]>;
  readEntity(entityId: string): Promise<OntologyEntity | null>;
}

export interface OntologyBus {
  publish(event: {
    type: string;
    payload: OntologyEntity | OntologyRelation | OntologyEvent | Record<string, unknown>;
    occurredAt: string;
    actorId?: string;
    requestId?: string;
  }): Promise<void>;
  subscribe(type: string, handler: (evt: Record<string, unknown>) => Promise<void> | void): void;
}

export type AgentRole = "planner" | "researcher" | "analyst" | "summarizer" | "auditor" | "coach";
export type OrchestratorMode = "single-pass" | "adaptive" | "strict";

export interface SyncCommandInput {
  projectRoot: string;
  actorId: string;
  source: "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";
  limit?: number;
  cursor?: string;
  stateFile?: string;
  persistState?: boolean;
  iterateSources?: boolean;
  fresh?: boolean;
}

export interface SyncCommandResult {
  schema: "knowledge-hub.foundry.connector.sync.result.v2";
  source: string;
  runId: string;
  status: "done" | "deduped" | "failed";
  emittedEventCount: number;
  emittedEventIds: string[];
  hasMore?: boolean;
  nextCursor?: string;
  errorMessage?: string;
}

export interface DiscoverFeatureExecution {
  feature: string;
  ok: boolean;
  result?: Record<string, unknown> | string | null;
  error?: string;
}

export interface DiscoverRequestResolution {
  source: "cli" | "resume";
  days: "cli" | "resume";
  from: "cli" | "resume";
  to: "cli" | "resume";
  topK: "cli" | "resume";
  limit: "cli" | "resume";
  intent: "cli" | "resume";
  features: "cli" | "resume";
  expenseThreshold: "cli" | "resume";
  minSleepHours: "cli" | "resume";
  eventLogPath: "cli" | "resume";
  stateFile: "cli" | "resume";
  saveState: "cli" | "resume";
}

export interface DiscoverRequestSnapshot {
  source: string;
  days: number;
  from?: string | null;
  to?: string | null;
  topK: number;
  limit?: number | null;
  intent: string;
  features: string[];
  expenseThreshold?: number | null;
  minSleepHours?: number | null;
  eventLogPath?: string | null;
  stateFile?: string | null;
  saveState: boolean;
  resumeSource: boolean;
  resolution: DiscoverRequestResolution;
}

export interface DiscoverOutputEnvelope {
  schema: "knowledge-hub.agent.discover.result.v1";
  runId: string;
  source: string;
  status: "ok" | "partial" | "error";
  sync: Record<string, unknown> | string;
  errors?: string[];
  features: DiscoverFeatureExecution[];
  request: DiscoverRequestSnapshot;
}

export interface AgentCommandInput {
  projectRoot: string;
  actorId: string;
  goal: string;
  maxRounds?: number;
  role?: AgentRole;
  orchestratorMode?: OrchestratorMode;
  dryRun?: boolean;
  dumpJson?: boolean;
  compact?: boolean;
  reportPath?: string;
}

export interface AgentPlaybookStep {
  order: number;
  tool: "ask_knowledge" | "search_knowledge" | "build_task_context";
  objective: string;
  rationale: string;
}

export interface AgentPlaybookEnvelope {
  schema: "knowledge-hub.foundry.agent.run.playbook.v1";
  source: "foundry-core/cli-agent" | "knowledge-hub/cli.py.fallback";
  goal: string;
  role: AgentRole;
  orchestratorMode: OrchestratorMode;
  maxRounds: number;
  assumptions: string[];
  warnings: string[];
  steps: AgentPlaybookStep[];
  generatedAt: string;
}

export interface AgentCommandEnvelope {
  schema: "knowledge-hub.foundry.agent.run.result.v1";
  source: "foundry-core/cli-agent" | "knowledge-hub/cli.py.fallback";
  runId: string;
  status: "running" | "completed" | "blocked" | "failed";
  goal: string;
  role: AgentRole;
  orchestratorMode: OrchestratorMode;
  stage: string;
  playbook?: AgentPlaybookEnvelope;
  tool?: string;
  plan?: string[];
  maxRounds?: number;
  transitions: StageTransition[];
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
    classification?: PrivacyClass;
    generatedAt?: string;
    metadata?: Record<string, unknown>;
  } | null;
  createdAt: string;
  updatedAt: string;
  dryRun: boolean;
}

export interface FoundryRuntimeContract {
  runAgent(input: AgentRuntimeInput): Promise<AgentRuntimeResult>;
  runConnector(input: ConnectorRunInput): Promise<ConnectorRunResult>;
  evaluatePolicy(context: PolicyContext): Promise<PolicyDecision>;
  buildTools(): Promise<FoundryToolRegistry>;
  sanitize(value: unknown, profile: SanitizationProfile): {
    output: unknown;
    profile: SanitizationProfile;
    original: string;
  };
}

export interface RuntimeDependencies {
  bus: FoundryEventBus;
  audit: AuditSink;
  ontology: OntologyStoreAdapter;
  eventBus: OntologyBus;
  policy: LocalFirstPolicyEngine;
  auditWriter: AuditLogWriter;
  tools: Record<string, FoundryTool>;
  connectorRegistry: Map<string, ConnectorContract>;
  syncStateStore: ConnectorSyncStateStore;
  idempotencyStore: ConnectorIdempotencyStore;
}

export interface DefaultPolicyRule {
  action: PolicyAction;
  allowAll: boolean;
  maxClassification: PrivacyClass;
  requiresExternalSanitization?: SanitizationProfile;
}

export interface ClassificationGate {
  maxAllowed: PrivacyClass;
  allowP0ToExternal: boolean;
  maskPatterns: string[];
  tags: string[];
}

export interface SyncStateEntry {
  cursor?: string;
  updatedAt: string;
  hasMore?: boolean;
}

export type SyncStateBySource = Record<string, SyncStateEntry>;

export interface RunAuditContext {
  actorId: string;
  requestId: string;
  runId: string;
  toolName?: string;
  step?: string;
  sourceSystem?: SourceSystem;
}

export interface ConnectorRuntimeAdapter extends Omit<ConnectorRuntimeContext, "bus"> {
  bus: OntologyBus;
  auditBridge: (context: RunAuditContext, message: string, severity: "info" | "warn" | "error") => Promise<void>;
}

export interface IngestAdapter {
  pull(input: ConnectorSyncInput): Promise<DataRecord[]>;
  mapToOntology(raw: DataRecord[], requestId: string): Promise<{
    entities: Pick<OntologyEntity, "id" | "type" | "properties" | "classification" | "sanitizationLevel" | "schemaVersion" | "source" | "createdAt" | "updatedAt" | "canonicalRefs">[];
    relations: Pick<OntologyRelation, "id" | "type" | "sourceEntityId" | "targetEntityId" | "properties" | "classification" | "sanitizationLevel" | "schemaVersion">[];
    events: Pick<OntologyEvent, "id" | "aggregateId" | "aggregateType" | "kind" | "type" | "payload" | "occurredAt" | "actorId" | "source" | "classification" | "sanitizationLevel" | "schemaVersion">[];
  }>;
  emit(input: ConnectorEmitInput): Promise<ConnectorEmitResult>;
}

export interface PlannerStep {
  order: number;
  name: string;
  toolName: string;
  input: Record<string, unknown>;
  retryable: boolean;
}

export interface RunSummary {
  runId: string;
  status: "ok" | "blocked" | "error";
  stagesCompleted: string[];
  policyDeniedCount: number;
  schemaFailedCount: number;
  writtenArtifacts: number;
  createdAt: string;
  updatedAt: string;
}
