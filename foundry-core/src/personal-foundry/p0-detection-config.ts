import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface RawPatternDefinition {
  id?: string;
  regex?: string;
  flags?: string;
}

interface RawPolicyPatternConfig {
  version?: string;
  patterns?: RawPatternDefinition[];
  redactKeys?: string[];
}

export interface SharedPolicyPatternConfig {
  version: string;
  patterns: RegExp[];
  redactKeys: string[];
}

const DEFAULT_PATTERN_CONFIG: Required<RawPolicyPatternConfig> = {
  version: "fallback-v1",
  patterns: [
    { id: "email", regex: String.raw`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` },
    { id: "phone_like", regex: String.raw`\b(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?)\d{3,4}[-\s]?\d{4}\b` },
    { id: "card_like", regex: String.raw`\b(?:\d[ -]*?){13,19}\b` },
    { id: "ssn_like", regex: String.raw`\b\d{3}-\d{2}-\d{4}\b` },
    { id: "secret_keyword", regex: String.raw`\b(password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|secret(?:[_-]?key)?|주민등록번호|계좌번호)\b`, flags: "i" },
  ],
  redactKeys: [
    "password",
    "passwd",
    "secret",
    "secret_key",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "ssn",
    "card_number",
    "account_number",
    "phone",
    "email",
  ],
};

const POLICY_PATTERN_PATH_CANDIDATES = [
  path.resolve(process.cwd(), "docs", "policy", "p0-detection-patterns.json"),
  path.resolve(process.cwd(), "..", "docs", "policy", "p0-detection-patterns.json"),
  path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../docs/policy/p0-detection-patterns.json"),
];

let cachedConfig: SharedPolicyPatternConfig | null = null;

function compilePattern(definition: RawPatternDefinition): RegExp | null {
  const regex = String(definition.regex || "").trim();
  if (!regex) {
    return null;
  }
  try {
    return new RegExp(regex, definition.flags || "");
  } catch {
    return null;
  }
}

function normalizeConfig(raw: RawPolicyPatternConfig): SharedPolicyPatternConfig {
  const patterns = (raw.patterns || [])
    .map((item) => compilePattern(item))
    .filter((item): item is RegExp => item instanceof RegExp);
  const redactKeys = Array.from(
    new Set((raw.redactKeys || []).map((key) => String(key || "").trim().toLowerCase()).filter(Boolean))
  );

  return {
    version: String(raw.version || DEFAULT_PATTERN_CONFIG.version),
    patterns: patterns.length > 0
      ? patterns
      : DEFAULT_PATTERN_CONFIG.patterns
          .map((item) => compilePattern(item))
          .filter((item): item is RegExp => item instanceof RegExp),
    redactKeys: redactKeys.length > 0
      ? redactKeys
      : DEFAULT_PATTERN_CONFIG.redactKeys.map((key) => key.toLowerCase()),
  };
}

export function loadSharedPolicyPatternConfig(): SharedPolicyPatternConfig {
  if (cachedConfig) {
    return cachedConfig;
  }

  for (const candidate of POLICY_PATTERN_PATH_CANDIDATES) {
    try {
      if (!fs.existsSync(candidate)) {
        continue;
      }
      const parsed = JSON.parse(fs.readFileSync(candidate, "utf8")) as RawPolicyPatternConfig;
      cachedConfig = normalizeConfig(parsed);
      return cachedConfig;
    } catch {
      continue;
    }
  }

  cachedConfig = normalizeConfig(DEFAULT_PATTERN_CONFIG);
  return cachedConfig;
}
