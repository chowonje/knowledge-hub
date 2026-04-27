# Public Release Checklist

Last updated: 2026-04-27

## Current verdict

`knowledge-hub` is ready to publish as a **Research Preview** from a reviewed branch, but not as a stable release.

Why:

- the supported public path is now explicit: `add -> index -> search/ask -> evidence review`
- the core setup and proof surfaces are real and demonstrable: `doctor`, `provider`, `add`, `index`, `search`, and `ask`
- full Python tests, Foundry checks, release smoke, weekly core-loop smoke, and public-release hygiene pass on the current public-preview candidate branch
- local-first / policy-first positioning is clear, and provider setup stores API keys as environment-variable references rather than raw secrets
- remaining risk is now about preview scope, live provider variance, corpus/source variance, and the absence of OPF model-based PII scanning in the local check environment

## Recommended public posture

Use one of these labels in the GitHub repo before opening it:

1. `Research Preview`
2. `Active Prototype`
3. `Build in Public`

Recommended status line:

> `Status: Research Preview — the supported default path is add -> index -> search/ask -> evidence review. APIs, quality bars, and experimental surfaces may change without notice.`

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

- cut a dedicated `public-preview` branch from a reviewed commit or PR branch, not from an unreviewed local workspace
- decide the public promise:
  - CLI-only retrieval assistant
  - local-first knowledge runtime
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
- generated verification runs and local experiment dumps unless they are intentionally curated examples:
  - ignored run-output directories
  - repo-local compatibility symlinks that point generated verification artifacts outside the checkout
  - ad hoc comparison outputs
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

- `check_public_release_hygiene.py` is the repo-local gate for tracked local files, literal high-confidence secrets, generated verification runs, and absolute user-path leaks
- environment variable names in docs/examples are fine
- real token values are not
- backup configs deserve a second look even if they only contain `${ENV_VAR}` placeholders
- a full commit-history secret audit is strongly recommended, but the minimum acceptance gate for this tranche is a clean `public-preview` snapshot branch

### 4. Make the first-run story smaller

README should answer four questions fast:

1. what this project is
2. what is supported today
3. how to install the smallest supported profile
4. which small command sequence proves it works

Recommended first-run command set:

```bash
pip install -e ".[ollama]"
khub doctor
khub add "large language model agent" --type paper -n 3
khub index
khub search "attention mechanism"
khub ask "Transformer의 핵심 아이디어는?"
```

Optional advanced sections can stay below that:

- paper ingestion
- provider setup
- MCP
- labs surfaces

### 5. Verify the public smoke gate

Minimum public release verification:

```bash
cd knowledge-hub
python -m pytest -q
cd foundry-core && npm ci && npm run check && npm test
cd ..
python scripts/check_release_smoke.py --json
python scripts/check_release_smoke.py --mode weekly_core_loop --json
python scripts/check_public_release_hygiene.py
```

### 6. Document the current limits honestly

Keep a short `Known Limits` section in `README.md` or a linked guide:

- some source families are stronger than others
- `paper` and `project` quality signals are ahead of `vault`
- at least one known `vault` compare path can still collapse to `0 source`
- local model quality and latency vary by machine
- heavy labs flows are additive, not the default stable surface
- some Python / TypeScript boundary areas and operator surfaces remain in transition

### 7. Curate examples, do not dump artifacts

Public examples should be small and intentional:

- one example parser workflow

Do not ship large local run directories as if they were canonical product assets.

## Remaining release risks

These are the main risks that keep the release in Research Preview language:

1. **Model-based PII scanning**
   - deterministic privacy/hygiene scans can pass while OPF is unavailable
   - report this as a residual risk unless OPF has been installed and run

2. **Live provider variance**
   - custom OpenAI-compatible aliases depend on external service behavior, model ids, auth scopes, and regional API differences
   - do not imply every provider preset has full live coverage

3. **Source-quality unevenness**
   - `paper` and `project` answer quality are demonstrably improving
   - at least one `vault` comparison path still produces `0 source`

4. **Public narrative drift**
   - the repo contains many implemented surfaces
   - the first-time story must stay limited to the supported core loop

5. **Additive surfaces are still too visible**
   - labs, agent, audit, and operator flows exist for real internal use
   - they still compete with the core public story unless the snapshot is curated

## Follow-up areas

- **Release branch cut**
  - cut or merge from the reviewed public-preview PR branch rather than publishing a mixed local workspace

- **Source-family quality remains uneven**
  - `paper` and `project` are the strongest paths today, while `vault` still has weaker comparison reliability

- **Public scope reduction**
  - the repo still contains more labs/operator surfaces than a first external user needs

- **Provider live checks**
  - add a small opt-in live-provider matrix for OpenAI-compatible aliases before claiming broad provider support

## Suggested branch strategy

Recommended sequence:

1. cut `public-preview`
2. keep only the stable CLI/runtime slice you want to show
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
