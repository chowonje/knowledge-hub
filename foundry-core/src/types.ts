export type DataClassification = "P0" | "P1" | "P2" | "P3";
export type SanitizationLevel = 0 | 1 | 2 | 3;
export type OntologyKind = "entity" | "relation" | "event";
export type OntologyId = string;
export type RunStage = "PLAN" | "ACT" | "VERIFY" | "WRITEBACK" | "DONE" | "FAILED";
export type RunStatus = "running" | "completed" | "blocked" | "failed";
export type PolicyAction =
  | "read"
  | "write"
  | "agent_run"
  | "agent_tool"
  | "llm_execute"
  | "agent_artifact_write";
export type PolicySeverity = "info" | "warn" | "error";
export type SanitizeProfile = "facts" | "summary" | "public";

export type PolicyDecisionResult = "ALLOW" | "DENY" | "WARN";

export interface ClassificationContext {
  [key: string]: unknown;
}

export interface DataClassificationRule {
  key: string;
  classification: DataClassification;
}

export interface OntologySourceRecord {
  sourceSystem: "knowledge_hub" | "connector" | "agent" | "manual";
  sourceId: string;
  sourceType: string;
}

export interface OntologyEntity {
  id: OntologyId;
  type: string;
  properties: Record<string, unknown>;
  canonicalRefs: OntologyId[];
  createdAt: string;
  updatedAt: string;
  schemaVersion: string;
  classification: DataClassification;
  sanitizationLevel: SanitizationLevel;
  source: OntologySourceRecord;
}

export interface RelationEvidence {
  text: string;
  chunkId?: string;
  score: number;
  threshold?: number;
  model?: string;
  extractedAt: string;
}

export interface OntologyRelation {
  id: OntologyId;
  type: string;
  sourceEntityId: OntologyId;
  targetEntityId: OntologyId;
  properties: Record<string, unknown>;
  evidence?: RelationEvidence[];
  confidence?: number;
  createdAt: string;
  updatedAt?: string;
  schemaVersion: string;
  classification: DataClassification;
  sanitizationLevel: SanitizationLevel;
}

export interface OntologyEvent {
  id: OntologyId;
  aggregateId: OntologyId;
  sequence: number;
  aggregateType: string;
  kind: OntologyKind;
  type: string;
  payload: Record<string, unknown>;
  occurredAt: string;
  actorId: OntologyId | null;
  schemaVersion: string;
  source: OntologySourceRecord;
  sourceRecordId?: string;
  classification: DataClassification;
  sanitizationLevel: SanitizationLevel;
}

export interface OntologySnapshot {
  id: OntologyId;
  entityType: string;
  streamId: string;
  version: number;
  payload: Record<string, unknown>;
  fromSequence: number;
  toSequence: number;
  createdAt: string;
}

export interface TimeSeriesPoint {
  at: string;
  metric: string;
  value: number;
  unit?: string;
  labels?: Record<string, string>;
}

export interface TimeSeriesRecord {
  key: string;
  entityId: OntologyId;
  points: TimeSeriesPoint[];
  createdAt: string;
  updatedAt: string;
}

export interface OntologyStore {
  appendEvent(event: Omit<OntologyEvent, "id" | "sequence">): Promise<OntologyEvent>;
  readEvents(input: { aggregateId: string; fromSequence?: number; toSequence?: number }): Promise<OntologyEvent[]>;
  readEntity(entityId: OntologyId): Promise<OntologyEntity | null>;
  upsertEntity(entity: OntologyEntity): Promise<void>;
  upsertRelation(relation: OntologyRelation): Promise<void>;
  appendSnapshot(snapshot: Omit<OntologySnapshot, "id">): Promise<string>;
  readLatestSnapshot(streamId: string): Promise<OntologySnapshot | null>;
  appendTimeSeries(point: TimeSeriesRecord): Promise<void>;
  getTimeSeries(input: { key: string; entityId: OntologyId; from?: string; to?: string }): Promise<TimeSeriesPoint[]>;
}

export interface EventBusEvent<T = Record<string, unknown>> {
  id: string;
  type: string;
  occurredAt: string;
  actorId?: string;
  payload: T;
}

export type EventBusHandler = (evt: EventBusEvent) => Promise<void> | void;

export interface FoundryEventBus {
  publish<T = Record<string, unknown>>(event: EventBusEvent<T>): Promise<void>;
  subscribe(type: string, handler: EventBusHandler): void;
}

export interface ConnectorAuditEvent {
  eventType:
    | "connector.authorized"
    | "connector.sync.started"
    | "connector.sync.completed"
    | "connector.sync.failed"
    | "connector.synced"
    | "connector.deduped";
  connectorId: string;
  actorId?: string;
  requestId?: string;
  runId?: string;
  message: string;
  severity: PolicySeverity;
  metadata?: Record<string, unknown>;
}

export interface ConnectorAuthInput {
  actorId: string;
  requestId: string;
  scopes: string[];
  redirectUri?: string;
  callbackCode?: string;
}

export interface ConnectorAuthResult {
  credentialId: string;
  accountId: string;
  scopes: string[];
  issuedAt: string;
  expiresAt?: string;
  tokenFingerprint: string;
}

export interface ConnectorSyncInput {
  actorId: string;
  requestId: string;
  cursor?: string;
  since?: string;
  source?: string;
  includeDeleted?: boolean;
  pageLimit?: number;
  dryRun?: boolean;
}

export interface DataRecord<TPayload = Record<string, unknown>> {
  sourceRecordId: string;
  sourceUpdatedAt: string;
  sourceHash?: string;
  payload: TPayload;
  classification: DataClassification;
}

export interface ConnectorSyncResult<TPayload = Record<string, unknown>> {
  runId: string;
  connectorRunId: string;
  rawRecords: DataRecord<TPayload>[];
  cursor?: string;
  nextCursor?: string;
  hasMore: boolean;
  extractedAt: string;
  connectorId?: string;
  ontologyDelta?: {
    entities?: Array<Record<string, unknown>>;
    relations?: Array<Record<string, unknown>>;
    claims?: Array<Record<string, unknown>>;
    events?: Array<Record<string, unknown>>;
  };
  metric: {
    scanned: number;
    returned: number;
    retries: number;
  };
}

export interface OntologyEntityDraft {
  entityType: string;
  entityId: OntologyId;
  properties: Record<string, unknown>;
  classification: DataClassification;
  schemaVersion: string;
  actorId?: string | null;
}

export interface OntologyRelationDraft {
  relationType: string;
  sourceEntityId: OntologyId;
  targetEntityId: OntologyId;
  properties: Record<string, unknown>;
  evidence?: RelationEvidence[];
  confidence?: number;
  classification: DataClassification;
  schemaVersion: string;
  actorId?: string | null;
}

export interface OntologyEventDraft {
  eventType: string;
  aggregateId: OntologyId;
  aggregateType: string;
  payload: Record<string, unknown>;
  actorId?: string | null;
  occurredAt: string;
  sourceRecordId?: string;
  classification: DataClassification;
  schemaVersion: string;
}

export interface MappedOntologyBatch {
  entities: OntologyEntityDraft[];
  relations: OntologyRelationDraft[];
  events: OntologyEventDraft[];
}

export interface ConnectorMappingContext {
  actorId: string;
  requestId: string;
  connectorId: string;
  connectorRunId: string;
  runStartedAt: string;
}

export interface ConnectorEmitInput {
  actorId: string;
  requestId: string;
  connectorRunId: string;
  mapped: MappedOntologyBatch;
  metadata?: Record<string, unknown>;
  sourceCursor?: {
    cursor?: string;
    nextCursor?: string;
    hasMore?: boolean;
  };
}

export interface ConnectorEmitResult {
  eventIds: string[];
  emittedAt: string;
  snapshotId?: string;
  snapshotVersion?: string;
}

export interface ConnectorSyncStateStore {
  get(connectorId: string): Promise<{ cursor?: string; updatedAt?: string } | null>;
  upsert(connectorId: string, state: { cursor?: string; updatedAt: string }): Promise<void>;
}

export interface ConnectorIdempotencyStore {
  isFreshRun(connectorId: string, requestId: string): Promise<boolean>;
  markStarted(connectorId: string, requestId: string): Promise<string>;
  markCompleted(connectorId: string, requestId: string, runId: string): Promise<void>;
  markFailed(connectorId: string, requestId: string, runId: string): Promise<void>;
}

export interface ConnectorRuntimeContext {
  now: () => string;
  syncStateStore: ConnectorSyncStateStore;
  idempotencyStore: ConnectorIdempotencyStore;
  bus: FoundryEventBus;
  audit: (event: ConnectorAuditEvent) => Promise<void>;
  classificationGate?: DataClassification;
  maxPublishedEvents?: number;
}

export interface ConnectorRunInput {
  connectorId: string;
  actorId: string;
  requestId?: string;
  cursor?: string;
  includeDeleted?: boolean;
  dryRun?: boolean;
  pageLimit?: number;
  metadata?: Record<string, unknown>;
}

export interface ConnectorRunResult {
  connectorId: string;
  requestId: string;
  runId: string;
  status: "done" | "deduped" | "failed";
  deduplicated: boolean;
  emittedEventCount: number;
  emittedEventIds: string[];
  nextCursor?: string;
  hasMore?: boolean;
  errorMessage?: string;
}

export interface ConnectorContract {
  readonly id: string;
  readonly version: string;
  readonly name: string;
  readonly description?: string;
  readonly sourceSystem: string;
  readonly supportedScopes: readonly string[];
  readonly supportsIncrementalSync: boolean;

  authorize(input: ConnectorAuthInput): Promise<ConnectorAuthResult>;
  sync(input: ConnectorSyncInput): Promise<ConnectorSyncResult>;
  mapToOntology(rawRecords: DataRecord[], context: ConnectorMappingContext): Promise<MappedOntologyBatch>;
  emitEvents(input: ConnectorEmitInput, runtime: ConnectorRuntimeContext): Promise<ConnectorEmitResult>;
}

export interface ConnectorRegistry {
  get(connectorId: string): ConnectorContract | undefined;
  has(connectorId: string): boolean;
}

export interface PolicyContext {
  actorId?: string;
  action: PolicyAction;
  resource: {
    type: string;
    id: string;
    tags?: string[];
  };
  payload?: unknown;
  classification?: DataClassification;
  projectId?: string;
}

export interface PolicyDecision {
  action: PolicyAction;
  resource: PolicyContext["resource"];
  result: PolicyDecisionResult;
  allowed: boolean;
  reason: string;
  severity: PolicySeverity;
  classification: DataClassification;
  requiresWriteback?: boolean;
}

export interface PolicyEngine {
  evaluate(context: PolicyContext): Promise<PolicyDecision>;
}

export interface AuditEvent {
  id: string;
  at: string;
  actorId?: string;
  action: PolicyAction | string;
  resourceType: string;
  resourceId: string;
  result: "allow" | "deny" | "warn";
  reason: string;
  severity: PolicySeverity;
  correlationId?: string;
  metadata?: Record<string, unknown>;
}

export interface AuditSink {
  append(event: AuditEvent): Promise<void>;
  query(filter: { action?: string; actorId?: string; from?: string; to?: string }): Promise<AuditEvent[]>;
}

export interface StageTransition {
  stage: RunStage;
  status: "started" | "completed" | "blocked" | "failed";
  message: string;
  at: string;
}

export interface RunArtifact {
  id?: string;
  jsonContent: unknown;
  humanSummaryMd?: string;
  classification?: DataClassification;
  metadata?: Record<string, unknown>;
  provenance?: OntologySourceRecord[];
  generatedAt: string;
}

export interface AgentRunRecord {
  id: string;
  actorId: string;
  goal: string;
  stage: RunStage;
  status: RunStatus;
  tool: string;
  createdAt: string;
  updatedAt: string;
  transitions: StageTransition[];
  planTools?: string[];
  proposal?: RunArtifact;
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
}

export interface AgentToolInput {
  [key: string]: unknown;
}

export interface AgentToolExecutionContext {
  actorId: string;
  requestId: string;
  runId: string;
  step: string;
  maxWritebackRows?: number;
}

export type FoundryToolValidationAction = "read" | "write";

export interface FoundryTool {
  name: string;
  action: FoundryToolValidationAction;
  description?: string;
  execute(input: AgentToolInput, context: AgentToolExecutionContext): Promise<FoundryToolResult | void>;
}

export interface FoundryToolResult {
  artifact?: RunArtifact;
  structuredFacts?: Record<string, unknown>;
  errors?: string[];
}

export type FoundryToolRegistry = Record<string, FoundryTool>;

export interface AgentRuntimeInput {
  actorId: string;
  goal: string;
  tools: FoundryToolRegistry;
  toolSequence?: string[];
  initialInput?: AgentToolInput;
  maxRounds?: number;
  now?: () => string;
  planner?: (goal: string, context: Record<string, unknown>) => Promise<string[]>;
  policyEngine?: PolicyEngine;
  validateOutput?: (artifact: RunArtifact) => Promise<{ ok: boolean; errors: string[] }>;
  writeback?: (payload: { actorId: string; goal: string; artifact: RunArtifact }) => Promise<{ ok: boolean; detail?: string }>;
  bus: FoundryEventBus;
  audit: AuditSink;
}

export interface AgentRuntimeResult {
  run: AgentRunRecord;
  final?: RunArtifact;
}

export interface DataSanitizer {
  sanitize(value: unknown, profile: "facts" | "summary"): { output: unknown; profile: "facts" | "summary"; original?: string };
}

export const DATA_CLASSIFICATION_ORDER: Record<DataClassification, number> = {
  P0: 0,
  P1: 1,
  P2: 2,
  P3: 3,
};

export function isMoreSensitive(current: DataClassification, maxAllowed: DataClassification): boolean {
  return DATA_CLASSIFICATION_ORDER[current] < DATA_CLASSIFICATION_ORDER[maxAllowed];
}
