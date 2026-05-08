# Claude Code Instructions

These instructions apply to all work inside `Q_Drop_Intergration/`.

## Working Style

- Favor small, readable, compatibility-aware refactors.
- Keep existing training entry points stable unless a task explicitly calls for API changes.
- When a rename would ripple through docs or scripts, prefer aliases or local cleanup instead of broad breakage.

## Naming Rules

- `snake_case`: Python modules, functions, methods, local variables, non-constant attributes
- `PascalCase`: classes
- `UPPER_SNAKE_CASE`: constants
- Boolean names should read clearly, for example `use_scheduler`, `is_training`, or `enable_dropout`

## Code Preferences

- Break complex training logic into helper methods instead of long monolithic functions.
- Prefer descriptive tensor and gradient names over abbreviations.
- Avoid introducing new `sys.path` workarounds unless there is no safer packaging option.
- Keep optimizer application logic easy to trace, especially in custom `train_step()` implementations.

## Project Safety

- Preserve reproducibility, default experiment settings, and output paths unless the user asks otherwise.
- Be careful with quantum-training code paths: sanitize `NaN` values and keep tensor shapes unchanged.
- Update nearby docs or references when changing a public symbol or workflow-facing behavior.

## Validation

- Run a lightweight syntax check after edits.
- Call out any behavior fix, compatibility shim, or remaining risk in the final handoff.
