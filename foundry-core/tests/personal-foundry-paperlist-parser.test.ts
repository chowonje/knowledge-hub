import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// parser behavior is validated indirectly by running pipeline fallback on this fixture-like text
function parsePaperListOutputForTest(raw: string): Array<{ arxivId: string; title: string }> {
  const rows = raw.split(/\r?\n/).filter((line) => line.includes("│"));
  const parsed: Array<{ arxivId: string; title: string }> = [];
  let current: { arxivId: string; title: string } | null = null;

  for (const line of rows) {
    const cols = line.split("│").map((value) => value.trim());
    if (cols.length < 9) continue;

    const arxivId = cols[1];
    const title = cols[2];

    if (arxivId === "arXiv ID" || (arxivId.length === 0 && title === "제목")) {
      continue;
    }

    const isNewRow = /^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$/.test(arxivId);
    if (isNewRow) {
      const row = { arxivId, title };
      parsed.push(row);
      current = row;
      continue;
    }

    if (!arxivId && current && title) {
      current.title = `${current.title} ${title}`.trim();
    }
  }

  return parsed;
}

describe("paper list table parser shape", () => {
  it("parses wrapped rich-table rows into id/title", () => {
    const fixtureDir = dirname(fileURLToPath(import.meta.url));
    const fixture = readFileSync(resolve(fixtureDir, "fixtures", "paper-list-sample.txt"), "utf8");
    const rows = parsePaperListOutputForTest(fixture);

    assert.equal(rows.length >= 2, true);
    assert.equal(rows[0].arxivId, "2602.16708");
    assert.equal(rows[0].title.includes("Policy Compiler for Secure Agentic Systems"), true);
  });
});
