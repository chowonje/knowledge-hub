export type SourceKind = "all" | "note" | "paper" | "web" | "expense" | "sleep" | "schedule" | "behavior";

export interface PipelineInput {
  goal: string;
  source: SourceKind;
  limit?: number;
  days?: number;
  featureName: "daily_coach" | "focus_analytics" | "risk_alert";
  dryRun: boolean;
  maxRounds: number;
}

function parsePositiveInt(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return undefined;
  }
  return parsed;
}

export function parsePipelineInput(args: string[]): PipelineInput {
  let goal = "recent knowledge activity summary";
  let source: SourceKind = "all";
  let limit: number | undefined;
  let days: number | undefined;
  let featureName: PipelineInput["featureName"] = "daily_coach";
  let dryRun = false;
  let maxRounds = 2;

  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];

    if (token === "--goal" && args[i + 1]) {
      goal = args[i + 1];
      i += 1;
      continue;
    }
    if (token.startsWith("--goal=")) {
      goal = token.split("=", 2)[1];
      continue;
    }

    if (token === "--source" && args[i + 1]) {
      const value = args[i + 1] as SourceKind;
      if (["all", "note", "paper", "web", "expense", "sleep", "schedule", "behavior"].includes(value)) {
        source = value;
      }
      i += 1;
      continue;
    }
    if (token.startsWith("--source=")) {
      const value = token.split("=", 2)[1] as SourceKind;
      if (["all", "note", "paper", "web", "expense", "sleep", "schedule", "behavior"].includes(value)) {
        source = value;
      }
      continue;
    }

    if (token === "--limit" && args[i + 1]) {
      limit = parsePositiveInt(args[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith("--limit=")) {
      limit = parsePositiveInt(token.split("=", 2)[1]);
      continue;
    }

    if (token === "--days" && args[i + 1]) {
      days = parsePositiveInt(args[i + 1]);
      i += 1;
      continue;
    }
    if (token.startsWith("--days=")) {
      days = parsePositiveInt(token.split("=", 2)[1]);
      continue;
    }

    if (token === "--feature" && args[i + 1]) {
      const value = args[i + 1];
      if (value === "daily_coach" || value === "focus_analytics" || value === "risk_alert") {
        featureName = value;
      }
      i += 1;
      continue;
    }
    if (token.startsWith("--feature=")) {
      const value = token.split("=", 2)[1];
      if (value === "daily_coach" || value === "focus_analytics" || value === "risk_alert") {
        featureName = value;
      }
      continue;
    }

    if (token === "--max-rounds" && args[i + 1]) {
      const parsed = parsePositiveInt(args[i + 1]);
      if (parsed) {
        maxRounds = parsed;
      }
      i += 1;
      continue;
    }
    if (token.startsWith("--max-rounds=")) {
      const parsed = parsePositiveInt(token.split("=", 2)[1]);
      if (parsed) {
        maxRounds = parsed;
      }
      continue;
    }

    if (token === "--dry-run") {
      dryRun = true;
    }
  }

  return {
    goal,
    source,
    limit,
    days,
    featureName,
    dryRun,
    maxRounds,
  };
}

export function pipelineInputToCliArgs(input: PipelineInput): string[] {
  const args = [
    "--goal",
    input.goal,
    "--source",
    input.source,
    "--feature",
    input.featureName,
    "--max-rounds",
    String(input.maxRounds),
  ];

  if (typeof input.days === "number") {
    args.push("--days", String(input.days));
  }
  if (typeof input.limit === "number") {
    args.push("--limit", String(input.limit));
  }
  if (input.dryRun) {
    args.push("--dry-run");
  }
  return args;
}
