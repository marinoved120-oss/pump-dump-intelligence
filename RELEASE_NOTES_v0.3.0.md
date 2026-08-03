# v0.3.0 — Controlled Autonomous Development

This release establishes the governance and approval layer before live market-data collection begins.

Implemented:

- fixed project constitution;
- approved development roadmap;
- isolated Git proposal branches;
- OpenAI Responses API developer provider;
- patch scope and protected-file validation;
- full test execution before approval;
- Telegram user allowlist;
- approval, rejection, diff and test commands;
- SQLite audit log;
- continuous next-task processing after an approved merge;
- no automatic trading and no Docker-socket access.

Validation performed in the release workspace:

- `41 passed`;
- Python package wheel built successfully without build isolation;
- orchestrator version and doctor commands executed.

Docker image build could not be executed in the release environment because Docker is not installed there. The included Dockerfile and Compose configuration must be validated on the target Windows Docker Desktop installation.
