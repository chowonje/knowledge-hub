import { existsSync, readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import type {
  ClassificationContext,
  FeatureFunction,
  FeatureQuery,
  FeatureResult,
} from "./contracts/index.js";
import type { OntologySourceRecord } from "./types.js";

interface FeatureRuntimeDependencies {
  eventLogPath: string;
  now?: () => string;
}

interface EventRecord {
  id: string;
  type: string;
  occurredAt: string;
  actorId?: string;
  payload: Record<string, unknown>;
}

interface FeatureWindow {
  from?: string;
  to?: string;
  sourceFilter: Set<string>;
  topK: number;
  limit?: number;
  expenseThreshold?: number;
  minSleepHours?: number;
}

type RuntimeFeatureQuery = FeatureQuery & Record<string, unknown>;

function toStringOrUndefined(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return undefined;
}

function toNumberOrUndefined(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function parseDate(value: unknown): number | undefined {
  const raw = toStringOrUndefined(value);
  if (!raw) {
    return undefined;
  }
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function dateKey(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

function toLowerCaseString(value: unknown): string | undefined {
  const raw = toStringOrUndefined(value);
  return raw?.trim().toLowerCase();
}

function readEvents(path: string): EventRecord[] {
  if (!existsSync(path)) {
    return [];
  }

  const content = readFileSync(path, "utf8").trim();
  if (!content) {
    return [];
  }

  const rows = content.split(/\r?\n/);
  const events: EventRecord[] = [];

  for (const row of rows) {
    if (!row.trim()) {
      continue;
    }

    try {
      const parsed = JSON.parse(row);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        continue;
      }
      const item = parsed as Record<string, unknown>;
      const id = toStringOrUndefined(item.id) ?? `event_${randomUUID()}`;
      const type = toStringOrUndefined(item.type) ?? "event:unknown";
      const occurredAt = toStringOrUndefined(item.occurredAt) ?? new Date().toISOString();
      const payload = typeof item.payload === "object" && item.payload !== null
        ? item.payload as Record<string, unknown>
        : {};
      events.push({
        id,
        type,
        occurredAt,
        actorId: toStringOrUndefined(item.actorId),
        payload,
      });
    } catch {
      continue;
    }
  }

  return events;
}

function resolveSource(value: string | undefined, eventType: string): string {
  const candidate = toLowerCaseString(value);
  if (candidate && candidate.length > 0) {
    return candidate;
  }

  const lowered = eventType.toLowerCase();
  if (lowered.includes("expense")) {
    return "expense";
  }
  if (lowered.includes("sleep")) {
    return "sleep";
  }
  if (lowered.includes("schedule")) {
    return "schedule";
  }
  if (lowered.includes("behavior")) {
    return "behavior";
  }
  if (lowered.includes("paper")) {
    return "paper";
  }
  if (lowered.includes("web")) {
    return "web";
  }
  return "note";
}

function extractSource(event: EventRecord): string {
  return resolveSource(toLowerCaseString(event.payload.source), event.type);
}

function buildWindow(query: FeatureQuery, now: () => string): FeatureWindow {
  const paramSource = query.params?.source ?? query.params?.sources;
  const sourceFilter = new Set<string>();

  if (Array.isArray(paramSource)) {
    for (const raw of paramSource) {
      const source = toLowerCaseString(raw);
      if (source) {
        sourceFilter.add(source);
      }
    }
  } else {
    const source = toLowerCaseString(paramSource);
    if (source) {
      for (const item of source.split(",")) {
        const normalized = item.trim();
        if (normalized.length > 0) {
          sourceFilter.add(normalized);
        }
      }
    }
  }

  if (sourceFilter.size === 0) {
    sourceFilter.add("all");
  }

  const nowMs = Date.parse(now());
  const requestedFrom = parseDate(query.timeframe?.from);
  const requestedTo = parseDate(query.timeframe?.to);
  const requestedDays = query.params ? toNumberOrUndefined(query.params.days) : undefined;
  const fallbackDays = Number.isFinite(requestedDays ?? NaN) ? Math.max(1, Math.trunc(requestedDays!)) : 7;

  const toMs = Number.isFinite(requestedTo ?? NaN) ? requestedTo! : nowMs;
  const fromMs = Number.isFinite(requestedFrom ?? NaN)
    ? requestedFrom!
    : toMs - fallbackDays * 24 * 60 * 60 * 1000;

  const from = new Date(fromMs).toISOString();
  const to = new Date(toMs).toISOString();

  return {
    from,
    to,
    sourceFilter,
    topK: Math.max(1, Math.trunc(toNumberOrUndefined(query.params?.top_k) || 8)),
    limit: query.params?.limit ? Math.max(1, Math.trunc(toNumberOrUndefined(query.params.limit) || 0)) : undefined,
    expenseThreshold: toNumberOrUndefined(query.params?.expenseThreshold),
    minSleepHours: toNumberOrUndefined(query.params?.minSleepHours),
  };
}

function filterEvents(records: EventRecord[], query: FeatureQuery, now: () => string): EventRecord[] {
  const config = buildWindow(query, now);
  const fromMs = parseDate(config.from);
  const toMs = parseDate(config.to);
  const onlySources = config.sourceFilter;
  const useAllSources = onlySources.has("all");

  const scoped = records.filter((event) => {
    const eventMs = parseDate(event.occurredAt);
    if (!eventMs || eventMs < (fromMs ?? 0) || eventMs > (toMs ?? Number.POSITIVE_INFINITY)) {
      return false;
    }
    const source = extractSource(event);
    return useAllSources || onlySources.has(source);
  });

  scoped.sort((left, right) => Date.parse(left.occurredAt) - Date.parse(right.occurredAt));
  if (config.limit) {
    return scoped.slice(-config.limit);
  }
  return scoped;
}

function buildProvenance(events: EventRecord[]): OntologySourceRecord[] {
  const seen = new Set<string>();
  const result: OntologySourceRecord[] = [];
  for (const event of events) {
    const source = extractSource(event);
    const sourceId = toStringOrUndefined(event.payload.sourceRecordId) ?? event.id;
    const signature = `${sourceId}:${source}`;
    if (seen.has(signature)) {
      continue;
    }
    seen.add(signature);
    result.push({
      sourceSystem: "knowledge_hub",
      sourceId,
      sourceType: source,
    });
  }
  return result;
}

function buildDailyCoachPayload(
  events: EventRecord[],
  query: FeatureQuery,
  now: () => string
): FeatureResult {
  const window = buildWindow(query, now);
  const totals: Record<string, number> = {
    note: 0,
    paper: 0,
    web: 0,
    expense: 0,
    sleep: 0,
    schedule: 0,
    behavior: 0,
  };
  const expenseByDate: Record<string, number> = {};
  const sleepByDate: Record<string, number> = {};
  const tags: Record<string, number> = {};
  const dayWithEvents = new Set<string>();
  let expenseSum = 0;
  let sleepMinutes = 0;
  let expenseCount = 0;

  for (const event of events) {
    const source = extractSource(event);
    const ts = parseDate(event.occurredAt);
    if (!ts) {
      continue;
    }
    if (totals[source] !== undefined) {
      totals[source] += 1;
    }
    const key = dateKey(ts);
    dayWithEvents.add(key);

    const metadata = event.payload.metadata && typeof event.payload.metadata === "object"
      ? event.payload.metadata as Record<string, unknown>
      : {};
    const eventTags = metadata.tags;
    if (Array.isArray(eventTags)) {
      for (const item of eventTags) {
        const tag = toLowerCaseString(item);
        if (tag) {
          tags[tag] = (tags[tag] ?? 0) + 1;
        }
      }
    }

    if (source === "expense") {
      const amount =
        toNumberOrUndefined(event.payload.amount) ??
        toNumberOrUndefined(metadata.amount) ??
        toNumberOrUndefined(event.payload.value) ??
        toNumberOrUndefined(metadata.value) ??
        toNumberOrUndefined(metadata.total);
      if (amount !== undefined) {
        expenseSum += amount;
        expenseCount += 1;
        expenseByDate[key] = (expenseByDate[key] ?? 0) + amount;
      }
    }

    if (source === "sleep") {
      const sleepHours = toNumberOrUndefined(metadata.sleepHours);
      const durationMinutes =
        toNumberOrUndefined(event.payload.durationMinutes) ??
        toNumberOrUndefined(metadata.durationMinutes) ??
        toNumberOrUndefined(metadata.duration) ??
        (sleepHours !== undefined ? sleepHours * 60 : 0);
      if (durationMinutes > 0) {
        sleepMinutes += durationMinutes;
        sleepByDate[key] = (sleepByDate[key] ?? 0) + durationMinutes;
      }
    }

    if (source === "schedule") {
      const durationMinutes =
        toNumberOrUndefined(event.payload.durationMinutes) ??
        toNumberOrUndefined(metadata.durationMinutes) ??
        toNumberOrUndefined(metadata.duration);
      if (durationMinutes && durationMinutes > 0) {
        // no-op, currently schedule 집중도만 추적
      }
    }
  }

  const days = Math.max(1, Math.round((parseDate(window.to)! - parseDate(window.from)!) / 86400000));
  const expensePerDay = expenseSum / days;
  const sleepHours = sleepMinutes / 60;
  const sleepAvgByDay = (sleepHours / Math.max(1, Object.keys(sleepByDate).length));
  const topTags = Object.entries(tags)
    .sort((left, right) => right[1] - left[1])
    .slice(0, window.topK)
    .map(([label, count]) => ({ label, count }));

  const activeSourceKeys = Object.entries(totals)
    .filter(([, count]) => count > 0)
    .map(([source, count]) => ({ source, count }));

  const recommendations: string[] = [];
  if (expensePerDay > 150000) {
    recommendations.push("최근 비용 지출이 높습니다. 내역 상위 출처를 점검해보세요.");
  }
  if (sleepHours / days < 6) {
    recommendations.push("수면량이 낮습니다. 오늘/내일부터 수면 기록 우선순위를 올려보세요.");
  }
  if (totals.schedule < totals.note) {
    recommendations.push("일정 등록량 대비 노트 생성이 많습니다. 계획-실행 페어링 강화를 추천합니다.");
  }

  return {
    featureName: "daily_coach",
    status: "ok",
    payload: {
      window: {
        from: window.from,
        to: window.to,
        dayCount: days,
        sampleCount: events.length,
      },
      sourceCount: totals,
      activeSourceKeys,
      summary: {
        expense: {
          count: expenseCount,
          total: Math.round(expenseSum * 100) / 100,
          avgPerDay: Math.round(expensePerDay * 100) / 100,
        },
        sleep: {
          totalHours: Math.round(sleepHours * 100) / 100,
          avgHoursPerEventDay: Math.round(sleepAvgByDay * 100) / 100,
        },
        coverageDays: dayWithEvents.size,
      },
      focus: {
        topTags,
      },
      recommendations,
    },
    generatedAt: now(),
    provenance: buildProvenance(events),
    classification: "P2",
  };
}

function buildFocusAnalyticsPayload(
  events: EventRecord[],
  query: FeatureQuery,
  now: () => string
): FeatureResult {
  const window = buildWindow(query, now);
  const sourceCount: Record<string, number> = {};
  const titleWeight: Record<string, number> = {};
  const dayBuckets: Record<string, number> = {};
  let totalEvents = 0;

  for (const event of events) {
    const source = extractSource(event);
    const ts = parseDate(event.occurredAt);
    if (!ts) {
      continue;
    }
    totalEvents += 1;
    sourceCount[source] = (sourceCount[source] ?? 0) + 1;
    const day = dateKey(ts);
    dayBuckets[day] = (dayBuckets[day] ?? 0) + 1;

    const title = toStringOrUndefined(event.payload.title);
    if (title) {
      titleWeight[title] = (titleWeight[title] ?? 0) + 1;
    }
  }

  const sourceRank = Object.entries(sourceCount)
    .sort((left, right) => right[1] - left[1])
    .map(([source, count]) => ({ source, count }));
  const topSources = sourceRank.slice(0, window.topK);
  const topTitles = Object.entries(titleWeight)
    .sort((left, right) => right[1] - left[1])
    .slice(0, window.topK)
    .map(([title, count]) => ({ title, count }));
  const timeline = Object.entries(dayBuckets)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([date, count]) => ({ date, count }));
  const focusScore = Math.min(
    100,
    Math.max(
      0,
      Math.round(
        ((sourceCount.paper ?? 0) * 3 + (sourceCount.note ?? 0) * 2 + (sourceCount.schedule ?? 0) * 1.5 + (sourceCount.web ?? 0) * 1) /
          Math.max(1, totalEvents) *
          100
      )
    )
  );

  return {
    featureName: "focus_analytics",
    status: "ok",
    payload: {
      window: {
        from: window.from,
        to: window.to,
        sampleCount: events.length,
      },
      focusScore,
      sourceBreakdown: topSources,
      topTitles,
      timeline,
      trend: timeline.length >= 2 ? timeline[timeline.length - 1].count - timeline[0].count : 0,
      suggestions: [
        "focusScore가 높을수록 문서/일정 기반의 작업 집중도가 높은 편입니다.",
        "저점 구간이 연속된 경우엔 집중 블록(예: 90분 단위) 정책을 추가하세요.",
      ],
    },
    generatedAt: now(),
    provenance: buildProvenance(events),
    classification: "P2",
  };
}

function buildRiskAlertPayload(
  events: EventRecord[],
  query: FeatureQuery,
  now: () => string
): FeatureResult {
  const window = buildWindow(query, now);
  const expenseThreshold = window.expenseThreshold ?? 200000;
  const minSleepHours = window.minSleepHours ?? 6;
  const dailyExpense: Record<string, number> = {};
  const dailySleep: Record<string, number> = {};
  const dailySchedule: Record<string, number> = {};

  for (const event of events) {
    const ts = parseDate(event.occurredAt);
    if (!ts) {
      continue;
    }
    const source = extractSource(event);
    const day = dateKey(ts);
    const metadata = event.payload.metadata && typeof event.payload.metadata === "object"
      ? event.payload.metadata as Record<string, unknown>
      : {};

    if (source === "expense") {
      const amount =
        toNumberOrUndefined(event.payload.amount) ??
        toNumberOrUndefined(metadata.amount) ??
        toNumberOrUndefined(event.payload.value) ??
        toNumberOrUndefined(metadata.total);
      if (amount !== undefined) {
        dailyExpense[day] = (dailyExpense[day] ?? 0) + amount;
      }
    }

    if (source === "sleep") {
      const minutes =
        toNumberOrUndefined(event.payload.durationMinutes) ??
        toNumberOrUndefined(metadata.durationMinutes) ??
        toNumberOrUndefined(metadata.duration) ??
        (toNumberOrUndefined(metadata.sleepHours) ?? 0) * 60;
      if (minutes !== undefined) {
        dailySleep[day] = (dailySleep[day] ?? 0) + minutes;
      }
    }

    if (source === "schedule") {
      dailySchedule[day] = (dailySchedule[day] ?? 0) + 1;
    }
  }

  const alerts: { type: string; severity: "critical" | "warning"; message: string; date?: string; value?: number }[] = [];
  const fromMs = parseDate(window.from);
  const toMs = parseDate(window.to);
  const today = new Date(now());
  const fromDate = typeof fromMs === "number" ? new Date(fromMs) : new Date(today);
  const toDate = typeof toMs === "number" ? new Date(toMs) : today;
  const toDayNum = Math.floor(toDate.getTime() / 86400000);
  const fromDayNum = Math.floor(fromDate.getTime() / 86400000);

  const zeroScheduleDays: string[] = [];
  let streak = 0;
  let maxStreak = 0;

  for (let d = fromDayNum; d <= toDayNum; d += 1) {
    const key = dateKey(d * 86400000);
    const sleepHours = (dailySleep[key] ?? 0) / 60;
    const expense = dailyExpense[key] ?? 0;
    const scheduled = dailySchedule[key] ?? 0;

    if (expense > expenseThreshold) {
      alerts.push({
        type: "high_expense_day",
        severity: "critical",
        message: `일별 지출이 임계값(${expenseThreshold})을 초과했습니다.`,
        date: key,
        value: expense,
      });
    }
    if (dailySleep[key] !== undefined && sleepHours < minSleepHours) {
      alerts.push({
        type: "low_sleep_day",
        severity: "warning",
        message: `수면 시간이 임계치(${minSleepHours}h) 미만입니다.`,
        date: key,
        value: sleepHours,
      });
    }

    if (scheduled === 0) {
      streak += 1;
      zeroScheduleDays.push(key);
      maxStreak = Math.max(maxStreak, streak);
    } else {
      streak = 0;
    }
  }

  if (maxStreak >= 3) {
    alerts.push({
      type: "schedule_inactive_streak",
      severity: "warning",
      message: `3일 이상 일정이 없는 구간이 있습니다. (${maxStreak}일 연속)`,
      value: maxStreak,
    });
  }

  const status = alerts.length ? "partial" : "ok";
  const warningCount = alerts.filter((item) => item.severity === "warning").length;
  const criticalCount = alerts.filter((item) => item.severity === "critical").length;

  return {
    featureName: "risk_alert",
    status,
    payload: {
      window: {
        from: window.from,
        to: window.to,
        sampleCount: events.length,
      },
      alerts,
      health: {
        totalAlerts: alerts.length,
        critical: criticalCount,
        warning: warningCount,
        zeroScheduleDays,
        maxNoScheduleStreak: maxStreak,
      },
      actions: criticalCount
        ? [
            "지출 상한선/카테고리별 한도를 먼저 조정하세요.",
            "수면이 짧은 일자와 일정 비우는 구간을 우선 점검하세요.",
          ]
        : [
            "현재 위험 신호는 없습니다. 기존 규칙을 유지해도 됩니다.",
          ],
    },
    generatedAt: now(),
    provenance: buildProvenance(events),
    classification: "P2",
  };
}

function buildFeatureResult(query: FeatureQuery, events: EventRecord[], now: () => string): FeatureResult {
  if (!events.length) {
    return {
      featureName: query.name,
      status: "error",
      payload: {
        reason: "no ontology events in configured window",
        sourceFilter: Array.from(buildWindow(query, now).sourceFilter),
      },
      generatedAt: now(),
      provenance: [],
      classification: "P2",
    };
  }

  if (query.name === "focus_analytics") {
    return buildFocusAnalyticsPayload(events, query, now);
  }
  if (query.name === "risk_alert") {
    return buildRiskAlertPayload(events, query, now);
  }

  return buildDailyCoachPayload(events, query, now);
}

export function createFeatureRuntime(config: FeatureRuntimeDependencies) {
  const now = config.now ?? (() => new Date().toISOString());

  const features = new Map<string, FeatureFunction<RuntimeFeatureQuery, FeatureResult>>(([
    "daily_coach",
    "focus_analytics",
    "risk_alert",
  ] as const).map((id) => [id, {
    id,
    name: id,
    version: "1.0.0",
    description: `${id} feature function`,
    classificationPolicy: {
      maxInputClass: "P1",
      maxOutputClass: "P2",
    },
    async execute(query: RuntimeFeatureQuery, _context: ClassificationContext): Promise<FeatureResult> {
      const allEvents = readEvents(config.eventLogPath);
      const scoped = filterEvents(allEvents, query, now);
      return buildFeatureResult(query, scoped, now);
    },
  }]));

  return {
    list() {
      return Array.from(features.keys());
    },
    async execute(query: FeatureQuery): Promise<FeatureResult> {
      const typedQuery = query as RuntimeFeatureQuery;
      const feature = features.get(typedQuery.name);
      if (!feature) {
        return {
          featureName: typedQuery.name,
          status: "error",
          payload: {
            reason: `unsupported feature: ${typedQuery.name}`,
            available: this.list(),
          },
          generatedAt: now(),
          provenance: [],
          classification: "P2",
        };
      }
      return feature.execute(typedQuery, { userId: "cli-user" });
    },
  } as const;
}
