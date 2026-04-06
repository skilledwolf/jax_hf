# AGENTS.md

## Purpose

This repository contains `jax_hf`, a JAX Hartree-Fock package. Optimize for correctness, reproducibility, and small reviewable changes.

## Workflow

1. Start by reading `TASKS.md` and `git status --short`.
2. Keep `TASKS.md` current using:
   - `[ ]` not started
   - `[~]` in progress
   - `[x]` done
3. When you start a task, flip it to `[~]`.
4. Mark a task `[x]` only after the code, tests, and relevant docs are updated, or after leaving a short note explaining what remains.
5. Add newly discovered work to `TASKS.md` as soon as you confirm it.

## Commits

- Save work in sensible units. Each commit should cover one coherent change that can be reviewed or reverted independently.
- Do not mix unrelated fixes, refactors, and documentation churn in the same commit.
- Commit after finishing a meaningful task or a tightly coupled pair of tasks.
- Run the narrowest useful verification before each commit and mention it in your summary.
- Never revert or overwrite user changes you did not make.

## Review Strategy

- For broad scrutiny or larger refactors, use subagents in parallel when available.
- Split work into sensible slices such as core solvers, public API/tests, and packaging/docs.
- Convert validated findings into actionable entries in `TASKS.md`.
- Prefer adding or updating regression tests alongside bug fixes.

## Quality Bar

- Preserve public API compatibility unless the task explicitly permits a breaking change.
- Call out numerical, convergence, or solver-behavior changes explicitly.
- Update README or public API docs when user-facing behavior changes.
