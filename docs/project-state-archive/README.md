# Project State Archive

This directory preserves detailed historical project-state records that no longer belong in the top-level current-state surface.

`docs/PROJECT_STATE.md` should answer current operational questions:

- What is the public posture?
- What is currently green?
- What is blocked or held?
- Which tranche should run next?
- Which work should stop because it is repeating the same report-only pattern?

Long dated implementation ledgers, phase-by-phase eval notes, and prior local-run details should live here instead.

## Archived snapshots

- [2026-06-03 pre-current-summary project state](2026-06-03-pre-current-summary-project-state.md): full `docs/PROJECT_STATE.md` content before the current-summary restructure.

## Archiving rules

- Preserve historical detail rather than deleting it.
- Prefer topic-specific archive files when adding new history.
- Link archive files from `docs/PROJECT_STATE.md` only when they still inform a current blocker, hold, promotion decision, or default-path metric.
- Do not use this archive as a second source of truth for product behavior; current behavior belongs in code, tests, `CHANGELOG.md`, and the current `docs/PROJECT_STATE.md` summary.
