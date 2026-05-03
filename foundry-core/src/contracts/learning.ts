export type LearningRelationNorm =
  | "causes"
  | "enables"
  | "part_of"
  | "contrasts"
  | "example_of"
  | "requires"
  | "improves"
  | "related_to"
  | "unknown_relation";

export interface EvidencePointer {
  type: "note" | "paper" | "web" | "manual";
  path: string;
  heading?: string;
  block_id?: string;
  snippet_hash: string;
}

export interface ConceptIdentity {
  canonical_id: string;
  display_name: string;
  aliases: string[];
  resolveConfidence: number;
  resolveMethod: "exact" | "alias" | "normalized" | "fuzzy" | "canonical";
}

export interface TrunkConcept extends ConceptIdentity {
  trunkScore: number;
  scoreBreakdown: {
    topicRelevance: number;
    graphCentrality: number;
    evidenceCoverage: number;
    recency: number;
  };
  evidenceSources?: string[];
}

export interface LearningPolicyStatus {
  mode: "local-only" | "external-allowed";
  allowed: boolean;
  classification: "P0" | "P1" | "P2" | "P3";
  blockedReason?: string | null;
  policyErrors: string[];
}

export interface LearningMapResultEnvelope {
  schema: "knowledge-hub.learning.map.result.v1";
  runId: string;
  topic: string;
  topicSlug?: string;
  status: "ok" | "fallback" | "blocked" | "error";
  policy: LearningPolicyStatus;
  trunks: TrunkConcept[];
  branches: Array<ConceptIdentity & { parentTrunkIds: string[]; confidence: number }>;
  scoringDetail: Record<string, unknown>;
  suggestedTopK: number;
  warnings?: string[];
  createdAt: string;
  updatedAt: string;
}

export interface LearningProgressGate {
  passed: boolean;
  status: "passed" | "failed" | "insufficient" | "blocked";
  reasons: string[];
}

export interface LearningGradeResultEnvelope {
  schema: "knowledge-hub.learning.grade.result.v1";
  runId: string;
  topic: string;
  status: "ok" | "failed" | "insufficient" | "blocked" | "error";
  policy: LearningPolicyStatus;
  session: {
    sessionId: string;
    path?: string;
    targetTrunkIds: string[];
  };
  targetTrunkIds: string[];
  scores: {
    coverage: number;
    edgeAccuracy: number;
    explanationQuality: number;
    final: number;
    totalEdges: number;
    validEdges: number;
    minEdges: number;
  };
  gateDecision: LearningProgressGate;
  weaknesses: Array<Record<string, unknown>>;
  edges?: Array<{
    sourceCanonicalId: string;
    relationRaw: string;
    relationNorm: LearningRelationNorm;
    targetCanonicalId: string;
    evidencePtrs: EvidencePointer[];
    confidence: number;
    isValid: boolean;
    issues: string[];
  }>;
  policyErrors?: string[];
  createdAt: string;
  updatedAt: string;
}

export interface LearningNextResultEnvelope {
  schema: "knowledge-hub.learning.next.result.v1";
  runId: string;
  topic: string;
  status: "ok" | "blocked" | "error";
  policy: LearningPolicyStatus;
  session: {
    sessionId: string;
  };
  unlockPlan: Array<{
    canonical_id: string;
    display_name: string;
    reason: string;
    miniMission?: string;
  }>;
  remediationPlan: Array<{
    focus: string[];
    reason: string;
    task: string;
  }>;
  loadSignal: {
    level: "normal" | "high";
    avgSleepHours?: number | null;
    avgExpense?: number | null;
    reasons?: string[];
  };
  reasoning: string[];
  createdAt: string;
  updatedAt: string;
}

export interface ConceptResolverContract {
  resolve(input: string): Promise<ConceptIdentity | null>;
}
