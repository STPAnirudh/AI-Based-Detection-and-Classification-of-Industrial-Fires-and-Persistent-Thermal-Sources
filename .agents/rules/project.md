\---

trigger: model\_decision

description: Core rules for the industrial fire and persistent thermal source detection project. Apply when modifying project architecture, Python/ML code, datasets, training, inference, or documentation.

\---



\# Project Rules



\## General

\- Inspect the existing project before making changes.

\- Make the smallest necessary change.

\- Do not rewrite working code unnecessarily.

\- Preserve the existing architecture unless a change is explicitly requested.

\- Do not delete files or functionality unless explicitly requested.



\## AI/ML

\- Preserve the existing dataset and ML pipeline unless the user requests a change.

\- Keep training, validation, and test data properly separated.

\- Do not claim that a model works without testing it.

\- Report errors, assumptions, and limitations instead of guessing.



\## Dependencies

\- Reuse existing dependencies where possible.

\- Do not add a new framework or library when the existing project can accomplish the task without it.



\## Verification

\- After significant code changes, check for syntax/import errors.

\- Run the relevant tests or validation commands when available.

\- Summarize what changed and what was verified.

