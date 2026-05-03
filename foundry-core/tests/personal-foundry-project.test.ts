import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  getPersonalFoundryProjectStatus,
  initPersonalFoundryProject,
  resolvePersonalFoundryPaths,
} from "../src/personal-foundry/project.js";

describe("personal-foundry project", () => {
  it("resolves default paths under .khub/personal-foundry", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-project-paths-"));
    const paths = resolvePersonalFoundryPaths({ projectRoot: tmp });

    assert.equal(paths.baseDir.startsWith(path.join(tmp, ".khub", "personal-foundry")), true);
    assert.equal(paths.ontologyEventsPath.endsWith(path.join("ontology-store", "ontology.events.jsonl")), true);

    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("initializes required files and returns healthy status", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "khub-project-init-"));

    const init = initPersonalFoundryProject({ projectRoot: tmp });
    assert.equal(init.ok, true);
    assert.equal(fs.existsSync(init.paths.manifestPath), true);
    assert.equal(fs.existsSync(init.paths.stateFilePath), true);
    assert.equal(fs.existsSync(init.paths.decisionsPath), true);
    assert.equal(fs.existsSync(init.paths.ontologyEventsPath), true);
    assert.equal(fs.existsSync(init.paths.decisionsPath), true);

    const status = getPersonalFoundryProjectStatus({ projectRoot: tmp });
    assert.equal(status.ok, true);
    assert.equal(status.files.every((file) => file.exists), true);

    fs.rmSync(tmp, { recursive: true, force: true });
  });
});
