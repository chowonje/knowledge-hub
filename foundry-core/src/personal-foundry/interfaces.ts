import type { DataClassification } from "../types.js";

export type SourceSystem =
  | "knowledge_hub"
  | "calendar"
  | "finance"
  | "sleep"
  | "behavior"
  | "manual";

export type SanitizationProfile = "facts" | "summary" | "public";

export type PersonalAction =
  | "connector_authorize"
  | "connector_sync"
  | "agent_plan"
  | "agent_act"
  | "agent_verify"
  | "agent_writeback"
  | "external_llm_call"
  | "ui_api_read"
  | "ui_api_write"
  | "learning_read"
  | "learning_write"
  | "learning_external_call";

export interface SourceCursor {
  connectorId: string;
  source?: string;
  cursor?: string;
  updatedAt: string;
  hasMore?: boolean;
}

export interface ConnectorCursorStore {
  get(connectorId: string, source?: string): Promise<SourceCursor | null>;
  set(entry: SourceCursor): Promise<void>;
}

export interface IdempotencyEntry {
  connectorId: string;
  requestId: string;
  runId: string;
  status: "started" | "completed" | "failed";
  updatedAt: string;
}

export interface ConnectorIdempotencyStore {
  isFreshRun(connectorId: string, requestId: string): Promise<boolean>;
  markStarted(connectorId: string, requestId: string): Promise<string>;
  markCompleted(connectorId: string, requestId: string, runId: string): Promise<void>;
  markFailed(connectorId: string, requestId: string, runId: string): Promise<void>;
}

export interface ConnectorAuthInput {
  actorId: string;
  requestId: string;
  scopes: string[];
}

export interface ConnectorAuthResult {
  credentialId: string;
  accountId: string;
  scopes: string[];
  issuedAt: string;
  expiresAt?: string;
}

export interface ConnectorRecord<T = Record<string, unknown>> {
  sourceRecordId: string;
  sourceUpdatedAt: string;
  payload: T;
  classification: DataClassification;
}

export interface ConnectorSyncInput {
  actorId: string;
  requestId: string;
  source?: string;
  cursor?: string;
  pageLimit?: number;
  dryRun?: boolean;
}

export interface ConnectorSyncOutput<T = Record<string, unknown>> {
  connectorRunId: string;
  records: ConnectorRecord<T>[];
  cursor?: string;
  nextCursor?: string;
  hasMore: boolean;
  extractedAt: string;
  ontologyDelta?: OntologyBatch;
}

export interface OntologyEntityRecord {
  id: string;
  type: string;
  properties: Record<string, unknown>;
  classification: DataClassification;
  sourceSystem: SourceSystem;
  updatedAt: string;
}

export interface OntologyRelationRecord {
  id: string;
  type: string;
  sourceEntityId: string;
  targetEntityId: string;
  properties: Record<string, unknown>;
  classification: DataClassification;
  sourceSystem: SourceSystem;
  updatedAt: string;
}

export interface OntologyEventRecord {
  id?: string;
  aggregateId: string;
  aggregateType: string;
  type: string;
  payload: Record<string, unknown>;
  classification: DataClassification;
  sourceSystem: SourceSystem;
  occurredAt: string;
  actorId?: string;
  sourceRecordId?: string;
}

export interface TimeSeriesBatchRecord {
  key: string;
  entityId: string;
  points: TimeSeriesPoint[];
}

export interface OntologyBatch {
  entities: OntologyEntityRecord[];
  relations: OntologyRelationRecord[];
  events: OntologyEventRecord[];
  timeSeries?: TimeSeriesBatchRecord[];
}

export interface ConnectorMapContext {
  actorId: string;
  requestId: string;
  connectorId: string;
  connectorRunId: string;
  now: string;
  source?: string;
}

export interface ConnectorEmitInput {
  actorId: string;
  requestId: string;
  connectorRunId: string;
  mapped: OntologyBatch;
  sourceCursor?: {
    cursor?: string;
    nextCursor?: string;
    hasMore?: boolean;
  };
}

export interface ConnectorEmitOutput {
  eventIds: string[];
  emittedAt: string;
  snapshotId?: string;
}

export interface ConnectorEmitContext {
  now: () => string;
  eventBus: PersonalEventBus;
  audit: PersonalAuditLog;
}

export interface ConnectorSDK<T = Record<string, unknown>> {
  readonly id: string;
  readonly version: string;
  readonly sourceSystem: SourceSystem;
  readonly supportsIncrementalSync: boolean;

  authorize(input: ConnectorAuthInput): Promise<ConnectorAuthResult>;
  sync(input: ConnectorSyncInput): Promise<ConnectorSyncOutput<T>>;
  mapToOntology(records: ConnectorRecord<T>[], context: ConnectorMapContext): Promise<OntologyBatch>;
  emitEvents(input: ConnectorEmitInput, context: ConnectorEmitContext): Promise<ConnectorEmitOutput>;
}

export interface ConnectorRunInput {
  connector: ConnectorSDK;
  actorId: string;
  requestId: string;
  source?: string;
  cursor?: string;
  pageLimit?: number;
  dryRun?: boolean;
}

export interface ConnectorRunResult {
  connectorId: string;
  runId: string;
  requestId: string;
  status: "done" | "deduped" | "failed";
  emittedEventCount: number;
  emittedEventIds: string[];
  nextCursor?: string;
  hasMore?: boolean;
  errorMessage?: string;
}

export interface BusEvent<T = Record<string, unknown>> {
  id: string;
  type: string;
  occurredAt: string;
  sourceSystem: SourceSystem;
  actorId?: string;
  classification: DataClassification;
  payload: T;
}

export interface EventReadInput {
  type?: string;
  from?: string;
  to?: string;
  limit?: number;
}

export type EventSubscriber = (event: BusEvent) => Promise<void> | void;

export interface PersonalEventBus {
  publish(event: Omit<BusEvent, "id"> & { id?: string }): Promise<BusEvent>;
  read(input?: EventReadInput): Promise<BusEvent[]>;
  subscribe(subscriber: EventSubscriber): () => void;
}

export interface OntologySnapshot {
  id: string;
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

export interface StoredOntologyEvent extends OntologyEventRecord {
  id: string;
  sequence: number;
}

export interface OntologyBatchAppendResult {
  eventIds: string[];
  entityCount: number;
  relationCount: number;
  eventCount: number;
  timeSeriesCount: number;
}

export interface PersonalOntologyStore {
  appendBatch(batch: OntologyBatch): Promise<OntologyBatchAppendResult>;
  readEntity(entityId: string): Promise<OntologyEntityRecord | null>;
  readEvents(input: { aggregateId: string; fromSequence?: number; toSequence?: number }): Promise<StoredOntologyEvent[]>;
  appendSnapshot(input: Omit<OntologySnapshot, "id">): Promise<OntologySnapshot>;
  readLatestSnapshot(streamId: string): Promise<OntologySnapshot | null>;
  appendTimeSeries(entry: TimeSeriesBatchRecord): Promise<void>;
  getTimeSeries(input: { key: string; entityId: string; from?: string; to?: string }): Promise<TimeSeriesPoint[]>;
}

export interface AuditRecord {
  id: string;
  at: string;
  actorId: string;
  action: PersonalAction;
  resourceType: string;
  resourceId: string;
  allowed: boolean;
  reason: string;
  classification: DataClassification;
  runId?: string;
  requestId?: string;
  metadata?: Record<string, unknown>;
}

export interface AuditQuery {
  actorId?: string;
  action?: PersonalAction;
  runId?: string;
  from?: string;
  to?: string;
  limit?: number;
}

export interface PersonalAuditLog {
  append(input: Omit<AuditRecord, "id" | "at"> & { id?: string; at?: string }): Promise<AuditRecord>;
  query(filter?: AuditQuery): Promise<AuditRecord[]>;
}

export interface PolicyEvaluationInput {
  actorId: string;
  action: PersonalAction;
  resourceType: string;
  resourceId: string;
  payload?: unknown;
  classification?: DataClassification;
  outbound?: boolean;
  runId?: string;
  requestId?: string;
}

export interface PolicyDecision {
  allowed: boolean;
  reason: string;
  classification: DataClassification;
  policyCode: string;
  requiresSanitization: boolean;
  sanitizedPayload?: unknown;
}

export interface PersonalPolicyEngine {
  classify(payload: unknown): DataClassification;
  sanitize(payload: unknown, profile: SanitizationProfile): unknown;
  evaluate(input: PolicyEvaluationInput): Promise<PolicyDecision>;
}

export interface FeatureQuery {
  featureName: string;
  params?: Record<string, unknown>;
  actorId: string;
}

export interface FeatureResult {
  featureName: string;
  classification: DataClassification;
  payload: Record<string, unknown>;
  createdAt: string;
  provenance: Array<{
    sourceSystem: SourceSystem;
    sourceId: string;
    sourceType: string;
  }>;
}

export interface FeatureFunction {
  id: string;
  description?: string;
  execute(input: FeatureQuery): Promise<FeatureResult>;
}

export interface FeatureLayer {
  register(feature: FeatureFunction): Promise<void>;
  execute(input: FeatureQuery): Promise<FeatureResult>;
  list(): FeatureFunction[];
}

export interface AgentPlanStep {
  order: number;
  toolName: string;
  objective: string;
  input: Record<string, unknown>;
}

export interface AgentToolContext {
  actorId: string;
  runId: string;
  requestId: string;
  step: string;
}

export interface AgentToolResult {
  artifact?: {
    id?: string;
    jsonContent: unknown;
    classification?: DataClassification;
    generatedAt: string;
    metadata?: Record<string, unknown>;
  };
  errors?: string[];
}

export interface AgentTool {
  name: string;
  execute(input: Record<string, unknown>, context: AgentToolContext): Promise<AgentToolResult | void>;
}

export interface AgentRuntimeInput {
  actorId: string;
  requestId: string;
  goal: string;
  maxRounds?: number;
  planner?: (goal: string) => Promise<AgentPlanStep[]>;
  tools: Record<string, AgentTool>;
  schemaValidate?: (artifact: unknown) => Promise<{ ok: boolean; errors: string[] }>;
  writeback?: (payload: {
    actorId: string;
    goal: string;
    artifact: unknown;
    classification: DataClassification;
    runId: string;
  }) => Promise<{ ok: boolean; detail?: string }>;
}

export interface AgentTransition {
  stage: "PLAN" | "ACT" | "VERIFY" | "WRITEBACK" | "DONE" | "FAILED";
  status: "started" | "completed" | "blocked" | "failed";
  message: string;
  at: string;
}

export interface AgentRunEnvelope {
  runId: string;
  status: "running" | "completed" | "blocked" | "failed";
  goal: string;
  stage: AgentTransition["stage"];
  transitions: AgentTransition[];
  plan: AgentPlanStep[];
  artifact?: {
    jsonContent: unknown;
    classification: DataClassification;
  } | null;
  verify?: {
    policyAllowed: boolean;
    schemaValid: boolean;
    schemaErrors: string[];
  };
  writeback?: {
    ok: boolean;
    detail?: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface PersonalAgentRuntime {
  run(input: AgentRuntimeInput): Promise<AgentRunEnvelope>;
}
