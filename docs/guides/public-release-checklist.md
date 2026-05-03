# Public Release Checklist

Last updated: 2026-04-20

## Current verdict

`knowledge-hub` is ready for a **public prototype / research preview** release, but not yet for a **clean first-time-user stable release**.

Why:

- the core product surface is real and demonstrable: `doctor`, `search`, `ask`, paper/document-memory flows, and answer-loop eval all run
- targeted regression checks are passing
- local-first / policy-first positioning is clear
- but the active worktree is still too broad and mixed for a clean public snapshot
- answer quality is still uneven by source: `paper` and `project` are credible, while at least one `vault` compare question still collapses to `0 source`

## Recommended public posture

Use one of these labels in the GitHub repo before opening it:

1. `Research Preview`
2. `Active Prototype`
3. `Build in Public`

Recommended status line:

> `Status: Research Preview — the supported default path is discover -> index -> search/ask -> evidence review. APIs, quality bars, and experimental surfaces may change without notice.`

Do not present the current state as:

1. `Production Ready`
2. `Stable API`
3. `One-command setup for all workflows`

## Public wording contract

Prefer these phrases in the public branch:

- `Research Preview`
- `Public Prototype`
- `Experimental`
- `Subject to change without notice`
- `Narrow smoke gate`

Avoid these phrases entirely:

- `stable`
- `production-ready`
- `GA`
- `1.0`
- `fully tested`
- `enterprise-ready`
- `secure by default`

## Minimum release goal

The public branch should satisfy all of the following:

1. no secrets or local-only credentials are tracked
2. no private vault content or personal runtime artifacts are tracked
3. README first-run path matches the actual supported setup
4. one narrow smoke gate passes on a clean checkout
5. known gaps are documented instead of hidden

## Public branch checklist

### 1. Scope the public snapshot

- cut a dedicated `public-preview` branch from a known commit, not from the current mixed worktree
- decide the public promise:
  - CLI-only retrieval assistant
  - local-first knowledge runtime
  - eval / answer-loop demo
- remove or defer unfinished surfaces that make the README noisier without helping the first run

### 2. Remove non-public material

Review and exclude these categories before pushing:

- local config files:
  - `config.yaml`
  - `config.yaml.bak_*`
- local databases and runtime state:
  - `*.db`
  - `*.db-shm`
  - `*.db-wal`
  - `.khub/` runtime outputs if they are copied into the repo
- generated eval runs and local experiment dumps unless they are intentionally curated examples:
  - `eval/knowledgeos/runs/*`
  - repo-local compatibility symlinks that point generated eval artifacts outside the checkout
  - ad hoc benchmark outputs
- personal work management notes unless you explicitly want them public:
  - `tasks/*`
  - `worklog/*`
  - `reviews/*`
- vault-derived private content or copied Obsidian pages

### 3. Run a secrets check

Minimum scan:

```bash
cd knowledge-hub
python scripts/check_public_release_hygiene.py --json
rg -n --hidden -g '!node_modules' -g '!.git' '(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|PERPLEXITY_API_KEY|PPLX_API_KEY|sk-[A-Za-z0-9]|ghp_[A-Za-z0-9]|github_pat_)' .
```

Interpretation:

- `check_public_release_hygiene.py` is the repo-local gate for tracked local files, literal high-confidence secrets, generated eval runs, and absolute user-path leaks
- environment variable names in docs/examples are fine
- real token values are not
- backup configs deserve a second look even if they only contain `${ENV_VAR}` placeholders
- a full commit-history secret audit is strongly recommended, but the minimum acceptance gate for this tranche is a clean `public-preview` snapshot branch

### 4. Make the first-run story smaller

README should answer four questions fast:

1. what this project is
2. what is stable today
3. how to install the smallest supported profile
4. which three commands prove it works

Recommended first-run command set:

```bash
pip install -e ".[ollama]"
khub doctor
khub search "attention mechanism"
khub ask "Transformer의 핵심 아이디어는?"
```

Optional advanced sections can stay below that:

- paper ingestion
- MCP
- answer-loop eval
- labs surfaces

### 5. Verify the public smoke gate

Minimum public release verification:

```bash
cd knowledge-hub
khub doctor
python scripts/check_release_smoke.py
pytest tests/test_answer_loop.py tests/test_runtime_diagnostics.py tests/test_pymupdf_adapter.py -q
```

If the public branch promises answer-loop demos, also run:

```bash
cd knowledge-hub
khub labs eval answer-loop collect --queries eval/knowledgeos/queries/user_answer_eval_queries_v1.csv --out-dir /tmp/khub-answer-loop-smoke --answer-backend codex_mcp --backend-model codex_mcp=gpt-5.4 --json
```

Use a reduced query set if cost or latency matters.

### 6. Document the current limits honestly

Keep a short `Known Limits` section in `README.md` or a linked guide:

- some source families are stronger than others
- `paper` and `project` eval quality are ahead of `vault`
- at least one known `vault` compare path can still collapse to `0 source`
- local model quality and latency vary by machine
- heavy labs flows are additive, not the default stable surface
- some Python / TypeScript boundary areas and operator surfaces remain in transition

### 7. Curate examples, do not dump artifacts

Public examples should be small and intentional:

- one example answer-loop query set
- one example judged CSV or summary
- one example parser workflow

Do not ship large local run directories as if they were canonical product assets.

## Current blocker summary

These are the main things still preventing a clean public snapshot:

1. **Mixed worktree**
   - too many unrelated changes are staged only as local progress, not as a coherent public slice

2. **Source-quality unevenness**
   - `paper` and `project` answer quality are demonstrably improving
   - at least one `vault` comparison path still produces `0 source`

3. **Public narrative drift**
   - the repo contains many implemented surfaces
   - the first-time story needs a smaller stable subset

4. **Additive surfaces are still too visible**
   - labs, agent, audit, and operator flows exist for real internal use
   - they still compete with the core public story unless the snapshot is curated

5. **Verification is still narrow**
   - targeted smoke checks pass
   - a clean full-repo release gate is not yet the claim this repository can honestly make

## Current unfinished areas

- **Clean branch cut is unfinished**
  - the active worktree is still mixed, so the release branch should be cut from a reviewed subset rather than pushed as-is

- **First-run packaging is unfinished**
  - the README still needs to behave like a small product entrypoint, not a map of every subsystem

- **Source-family quality remains uneven**
  - `paper` and `project` are the strongest paths today, while `vault` still has weaker comparison reliability

- **Public scope reduction is unfinished**
  - the repo still contains more labs/operator surfaces than a first external user needs

- **Artifact hygiene is unfinished**
  - local tasks, reviews, worklogs, generated runs, and machine-specific paths still need a deliberate public/public-not-public decision

## Suggested branch strategy

Recommended sequence:

1. cut `public-preview`
2. keep only the stable CLI/runtime/eval slice you want to show
3. prune local artifacts and private notes
4. tighten README to the small supported path
5. run the smoke gate on a clean checkout
6. open the repository with `Research Preview` wording

## Release decision rule

Open the repo when all of the following are true:

- secret scan is clean
- public branch has a coherent README and install path
- smoke gate passes
- generated/private/local artifacts are pruned
- known gaps are written down

If one of those is still false, do not block forever; just do not call the release stable.
