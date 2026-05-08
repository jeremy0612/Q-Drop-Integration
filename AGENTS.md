# Agent Instructions

These instructions apply to all work inside `Q_Drop_Intergration/`.

## Primary Goals

- Preserve experiment behavior unless the task explicitly asks for a behavior change.
- Prefer compatibility-aware refactors over broad renames.
- Keep training scripts and documented entry points stable.

## Naming Convention

- Use `snake_case` for Python files, functions, methods, local variables, and module-level variables.
- Use `PascalCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants.
- Use descriptive boolean names such as `is_*`, `has_*`, `should_*`, `use_*`, or `enable_*`.
- Do not introduce new mixed-case Python module filenames. Existing legacy filenames may remain when renaming them would break imports or documentation.

## Refactor Rules

- Keep public class names and import paths stable unless you also update all references.
- Prefer extracting helper methods over growing large `train_step()` or `main()` implementations.
- Use explicit ML names such as `inputs`, `labels`, `predictions`, `gradients`, and `quantum_weights`.
- Avoid single-letter variable names except for short-lived math indices.
- Keep one optimizer update per batch unless the algorithm intentionally requires staged updates.

## ML and Research Safety

- Preserve seed-setting and reproducibility behavior.
- Sanitize `NaN` values in gradients, quantum outputs, and trainable weights when working in unstable quantum-training paths.
- Do not silently change dataset selection, default hyperparameters, output locations, or checkpoint formats.
- Treat workflow files, DVC files, and training scripts as user-facing interfaces: rename cautiously and update docs when needed.

## Validation

- Run at least a lightweight syntax check after code edits.
- If you make a compatibility tradeoff or behavior fix, mention it in the final handoff.
