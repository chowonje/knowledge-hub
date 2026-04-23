import { isMoreSensitive } from "../types.js";
import type {
  DataClassification,
} from "../types.js";
import { loadSharedPolicyPatternConfig } from "./p0-detection-config.js";
import type {
  PersonalPolicyEngine,
  PolicyDecision,
  PolicyEvaluationInput,
  SanitizationProfile,
} from "./interfaces.js";

export interface LocalPolicyEngineOptions {
  outboundMaxClassification?: DataClassification;
  writebackMaxClassification?: DataClassification;
  p0Patterns?: RegExp[];
  redactKeys?: string[];
}

function normalizeString(input: unknown): string {
  return typeof input === "string" ? input : JSON.stringify(input ?? {});
}

function matchesAnyPattern(value: string, patterns: RegExp[]): boolean {
  for (const pattern of patterns) {
    if (pattern.test(value)) {
      return true;
    }
  }
  return false;
}

function inferClassificationFromShape(payload: unknown): DataClassification {
  if (!payload || typeof payload !== "object") {
    return "P3";
  }

  const data = payload as Record<string, unknown>;
  if (data["summary"] || data["abstract"] || data["insight"]) {
    return "P2";
  }

  if (data["entity"] || data["relation"] || data["event"] || data["facts"]) {
    return "P1";
  }

  return "P3";
}

const PUBLIC_HINT_PATTERNS = [
  /\bhttps?:\/\//i,
  /\barxiv\b/i,
  /\bdoi\b/i,
  /\bwikipedia\b/i,
  /\bgithub\b/i,
  /\bopen source\b/i,
];

const P1_STRUCTURED_HINT_PATTERNS = [
  /^\s*[\{\[]/,
  /^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.+$/m,
  /\b(id|uuid|arxiv_id|paper_id|source_id|entity_id|claim_id)\b/i,
  /\b(status|score|confidence|count|timestamp|created_at|updated_at)\b/i,
];

const POLICY_METADATA_KEYS = new Set(["classification", "policyclass", "policy_class"]);

function mostSensitiveClassification(left: DataClassification, right: DataClassification): DataClassification {
  return isMoreSensitive(left, right) ? left : right;
}

function normalizeClassification(value: unknown): DataClassification | undefined {
  if (value === "P0" || value === "P1" || value === "P2" || value === "P3") {
    return value;
  }
  if (typeof value === "string") {
    const token = value.trim().toUpperCase();
    if (token === "P0" || token === "P1" || token === "P2" || token === "P3") {
      return token;
    }
  }
  return undefined;
}

function policyMetadataKey(value: string): string {
  return value.replace(/[-\s]/g, "_").toLowerCase();
}

function declaredClassificationsFromPayload(payload: unknown): DataClassification[] {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return [];
  }

  const data = payload as Record<string, unknown>;
  const declarations: DataClassification[] = [];
  const direct = normalizeClassification(data["classification"]);
  if (direct) {
    declarations.push(direct);
  }

  const policyNode = data["policy"];
  if (policyNode && typeof policyNode === "object" && !Array.isArray(policyNode)) {
    const policyClassification = normalizeClassification((policyNode as Record<string, unknown>)["classification"]);
    if (policyClassification) {
      declarations.push(policyClassification);
    }
  }

  return declarations;
}

function extractTextFragmentsForPolicy(value: unknown): string[] {
  if (value === null || value === undefined) {
    return [];
  }
  if (typeof value === "string") {
    return [value];
  }
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return [String(value)];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => extractTextFragmentsForPolicy(item));
  }
  if (typeof value === "object") {
    const fragments: string[] = [];
    for (const [key, nestedValue] of Object.entries(value as Record<string, unknown>)) {
      if (POLICY_METADATA_KEYS.has(policyMetadataKey(key))) {
        continue;
      }
      fragments.push(...extractTextFragmentsForPolicy(nestedValue));
    }
    return fragments;
  }
  return [String(value)];
}

function classifyTextLevel(value: string, patterns: RegExp[]): DataClassification {
  if (!value) {
    return "P3";
  }
  if (matchesAnyPattern(value, patterns)) {
    return "P0";
  }

  const raw = value.trim();
  if (!raw) {
    return "P3";
  }
  const lowered = raw.toLowerCase();
  if ((raw.length <= 400 && PUBLIC_HINT_PATTERNS.some((pattern) => pattern.test(raw))) || lowered.startsWith("http://") || lowered.startsWith("https://")) {
    return "P3";
  }
  if (P1_STRUCTURED_HINT_PATTERNS.some((pattern) => pattern.test(raw))) {
    return "P1";
  }
  return "P2";
}

function classifyPayloadLevel(payload: unknown, patterns: RegExp[]): DataClassification {
  const fragments = extractTextFragmentsForPolicy(payload);
  let best: DataClassification = "P3";
  for (const fragment of fragments) {
    best = mostSensitiveClassification(best, classifyTextLevel(fragment, patterns));
    if (best === "P0") {
      break;
    }
  }
  return best;
}

function redactObject(
  input: unknown,
  redactKeys: Set<string>,
  patterns: RegExp[],
  visited: WeakSet<object>
): unknown {
  if (input === null || input === undefined) {
    return input;
  }

  if (typeof input === "string") {
    return matchesAnyPattern(input, patterns) ? "[REDACTED]" : input;
  }

  if (typeof input !== "object") {
    return input;
  }

  if (visited.has(input as object)) {
    return "[CIRCULAR]";
  }
  visited.add(input as object);

  if (Array.isArray(input)) {
    return input.map((item) => redactObject(item, redactKeys, patterns, visited));
  }

  const output: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    if (redactKeys.has(key.toLowerCase())) {
      output[key] = "[REDACTED]";
      continue;
    }
    output[key] = redactObject(value, redactKeys, patterns, visited);
  }
  return output;
}

export class LocalPolicyEngine implements PersonalPolicyEngine {
  private readonly outboundMaxClassification: DataClassification;
  private readonly writebackMaxClassification: DataClassification;
  private readonly p0Patterns: RegExp[];
  private readonly redactKeys: Set<string>;

  constructor(options: LocalPolicyEngineOptions = {}) {
    const sharedConfig = loadSharedPolicyPatternConfig();
    this.outboundMaxClassification = options.outboundMaxClassification ?? "P2";
    this.writebackMaxClassification = options.writebackMaxClassification ?? "P1";
    this.p0Patterns = options.p0Patterns ?? sharedConfig.patterns;
    this.redactKeys = new Set((options.redactKeys ?? sharedConfig.redactKeys).map((key) => key.toLowerCase()));
  }

  classify(payload: unknown): DataClassification {
    const serialized = normalizeString(payload);
    if (matchesAnyPattern(serialized, this.p0Patterns)) {
      return "P0";
    }
    return mostSensitiveClassification(inferClassificationFromShape(payload), classifyPayloadLevel(payload, this.p0Patterns));
  }

  sanitize(payload: unknown, profile: SanitizationProfile): unknown {
    const redacted = redactObject(payload, this.redactKeys, this.p0Patterns, new WeakSet<object>());

    if (profile === "public") {
      return { summary: "sanitized-public", data: "[REDACTED]" };
    }

    if (profile === "summary") {
      return {
        summary: "sanitized-summary",
        data: redacted,
      };
    }

    return redacted;
  }

  async evaluate(input: PolicyEvaluationInput): Promise<PolicyDecision> {
    const detectedClassification = this.classify(input.payload);
    const declaredClassifications = [
      ...(input.classification ? [input.classification] : []),
      ...declaredClassificationsFromPayload(input.payload),
    ];
    let classification = detectedClassification;
    if (declaredClassifications.length === 0) {
      classification = mostSensitiveClassification("P2", classification);
    }
    for (const declaredClassification of declaredClassifications) {
      classification = mostSensitiveClassification(declaredClassification, classification);
    }
    const isOutbound = Boolean(input.outbound) || input.action === "external_llm_call";

    if (isOutbound && isMoreSensitive(classification, this.outboundMaxClassification)) {
      return {
        allowed: false,
        reason: `policy deny: outbound classification ${classification} exceeds ${this.outboundMaxClassification}`,
        classification,
        policyCode: "OUTBOUND_CLASSIFICATION_DENY",
        requiresSanitization: true,
      };
    }

    if (input.action === "agent_verify" && classification === "P0") {
      return {
        allowed: false,
        reason: "policy deny: P0 artifact blocked by local policy",
        classification,
        policyCode: "AGENT_VERIFY_P0_DENY",
        requiresSanitization: true,
      };
    }

    if (input.action === "agent_writeback" && isMoreSensitive(classification, this.writebackMaxClassification)) {
      return {
        allowed: false,
        reason: `policy deny: writeback classification ${classification} exceeds ${this.writebackMaxClassification}`,
        classification,
        policyCode: "WRITEBACK_CLASSIFICATION_DENY",
        requiresSanitization: false,
      };
    }

    if (isOutbound) {
      return {
        allowed: true,
        reason: "policy allow: outbound payload sanitized",
        classification,
        policyCode: "OUTBOUND_ALLOW_SANITIZED",
        requiresSanitization: true,
        sanitizedPayload: this.sanitize(input.payload, "summary"),
      };
    }

    return {
      allowed: true,
      reason: "policy allow",
      classification,
      policyCode: "ALLOW",
      requiresSanitization: false,
    };
  }
}
