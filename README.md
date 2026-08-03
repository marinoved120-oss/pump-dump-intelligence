# Pump/Dump Intelligence Engine v0.3.0.1

This release keeps the v0.2.4.1 prospective research pipeline and adds a controlled development orchestrator. The orchestrator can prepare one roadmap change at a time in an isolated Git branch, run the complete test suite, and request approval in Telegram before merging.

It does **not** place trades, hold exchange trading keys, mount the Docker socket, or silently alter the project constitution.

## Fixed project contract

`PROJECT_CONSTITUTION.yaml` defines mandatory analysis components and forbidden shortcuts. Protected governance files cannot be changed by an ordinary task. `ROADMAP.yaml` contains the approved sequence for the live recorder, spot/futures books, whale-wall tracking, spoofing/iceberg evidence, derivatives context, causal evidence and paper monitoring.

## Orchestrator safety model

- one pending serious change at a time;
- code generated only inside the task's allowed paths;
- `git apply --check` before patch application;
- full tests before approval;
- Telegram allowlist by numeric user ID;
- explicit approval before merge;
- audit trail in SQLite;
- no automatic trading;
- no Docker socket by default;
- no secrets in repository context or Telegram reports.

## Install v0.3.0.1

Copy this release over the existing project, then rebuild:

```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue
docker compose build --no-cache
```

Research pipeline check:

```powershell
docker compose run --rm research version
docker compose run --rm --entrypoint python research -m pytest
```

Expected version:

```text
Pump/Dump Research v0.3.0.1
```


## Patch-generation resilience in v0.3.0.1

The orchestrator now normalizes Markdown-wrapped diffs, rejects file headers without valid hunks, and asks the developer model to regenerate an invalid patch. If `git apply --check` still fails, it performs up to two repair attempts using the exact Git error and repository context. Failed roadmap attempts remain retryable. Cleanup is limited to files named by the generated proposal; unrelated local files and `.env` are never removed by a broad `git clean`.

After installing this hotfix, an earlier `FAILED` record for `V030-001` does not need to be deleted. Running `run-next` retries the same roadmap task with a new change ID.

## One-time orchestrator configuration

Create a separate Telegram bot for development approvals and put the following into `.env`:

```dotenv
TELEGRAM_DEV_BOT_TOKEN=...
TELEGRAM_ALLOWED_USER_ID=
TELEGRAM_DEV_CHAT_ID=
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5
ORCHESTRATOR_WORKER_ENABLED=false
```

Keep `ORCHESTRATOR_WORKER_ENABLED=false` until `doctor`, Git initialization and Telegram authorization have been verified. Do not send tokens or API keys through Telegram or this chat.

Send `/start` to the new development bot, then discover your numeric IDs:

```powershell
docker compose --profile orchestrator run --rm orchestrator telegram-id
```

Copy the displayed private `User ID` into `TELEGRAM_ALLOWED_USER_ID` and `Chat ID` into `TELEGRAM_DEV_CHAT_ID`, then save `.env`.

Initialize the controlled Git baseline:

```powershell
docker compose --profile orchestrator run --rm orchestrator init
```

Check configuration:

```powershell
docker compose --profile orchestrator run --rm orchestrator doctor
```

Show the fixed roadmap:

```powershell
docker compose --profile orchestrator run --rm orchestrator roadmap
```

Start only the Telegram approval gateway:

```powershell
docker compose --profile orchestrator up -d orchestrator
```

The worker remains idle while `ORCHESTRATOR_WORKER_ENABLED=false`.

## Controlled first proposal

Run exactly one roadmap task:

```powershell
docker compose --profile orchestrator run --rm orchestrator run-next
```

The orchestrator creates an isolated branch, requests a patch from the configured developer model, validates allowed paths, runs tests, commits the proposal branch and sends the change to Telegram. It does not merge until approval.

Telegram commands:

```text
/status
/changes
/approve CHANGE-XXXXXXXX
/reject CHANGE-XXXXXXXX reason
/diff CHANGE-XXXXXXXX
/tests CHANGE-XXXXXXXX
```

After the first proposal has been inspected and the approval flow works, enable continuous roadmap processing:

```dotenv
ORCHESTRATOR_WORKER_ENABLED=true
```

Then restart:

```powershell
docker compose --profile orchestrator up -d --force-recreate orchestrator
```

The worker stops whenever a proposal waits for approval or validation fails.

## Important operational limitation

The local orchestrator uses an API model configured by `OPENAI_API_KEY`; it is not connected to this chat session. This chat cannot continue working after the conversation is closed or send Telegram messages by itself. The persistent Docker service performs those functions locally.

## Research command retained

```powershell
docker compose run --rm research prospective KOMA PEPE WIF DOGE TURBO BLESS 1000RATS BICO BAN BEAT ICNT UAI VIC SKYAI TAKE BTW --target dump_8_15m --model random_forest
```

This remains historical research. It is not a trading recommendation and does not prove live profitability.
