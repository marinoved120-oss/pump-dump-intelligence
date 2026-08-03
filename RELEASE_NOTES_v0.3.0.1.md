# v0.3.0.1 — Patch and Recovery Hardening

This hotfix addresses the first live autonomous-development failure observed during `V030-001`.

The model returned text that looked like a diff but was not accepted by `git apply`. The orchestrator previously stopped with `No valid patches in input`, marked the roadmap task failed, and used a broad cleanup command that could remove unrelated untracked files.

v0.3.0.1 now:

- normalizes Markdown-wrapped or prefixed patches;
- requires complete unified-diff headers and `@@` hunks;
- asks the model to regenerate structurally invalid output;
- repairs a patch using the exact Git error when applicability checks fail;
- allows up to three patch attempts before failing safely;
- treats `FAILED` roadmap attempts as retryable;
- returns to the base branch and deletes failed proposal branches;
- restores tracked files and deletes only untracked files explicitly named by the proposal;
- never runs broad `git clean -fd` during proposal failure handling.

Validation: 48 automated tests passed.
