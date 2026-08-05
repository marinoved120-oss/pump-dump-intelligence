from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from orchestrator import __version__
from orchestrator.config import ConfigError, OrchestratorConfig
from orchestrator.db import OrchestratorDB
from orchestrator.pipeline import DevelopmentPipeline
from orchestrator.roadmap import load_roadmap
from orchestrator.telegram import TelegramGateway, discover_recent_users
from orchestrator.worker import RoadmapWorker

app = typer.Typer(no_args_is_help=True, help="Pump/Dump Development Orchestrator v0.3.0.2")
console = Console()


def _services() -> tuple[OrchestratorConfig, OrchestratorDB, DevelopmentPipeline]:
    config = OrchestratorConfig.from_env()
    config.ensure_directories()
    db = OrchestratorDB(config.state_dir / "orchestrator.sqlite3")
    pipeline = DevelopmentPipeline(config, db)
    return config, db, pipeline


@app.command()
def version() -> None:
    console.print(f"Pump/Dump Development Orchestrator v{__version__}")


@app.command("init")
def initialize() -> None:
    config, db, pipeline = _services()
    pipeline.prepare_repository()
    console.print("[green]Repository and orchestrator state initialized.[/green]")
    console.print(f"State: {config.state_dir}")


@app.command()
def doctor() -> None:
    config, db, pipeline = _services()
    rows = [
        ("Project root", str(config.project_root), config.project_root.exists()),
        ("Constitution", str(pipeline.constitution.path), pipeline.constitution.path.exists()),
        ("Roadmap", str(config.project_root / "ROADMAP.yaml"), (config.project_root / "ROADMAP.yaml").exists()),
        ("Git", "repository" if pipeline.repo.is_repository() else "not initialized", pipeline.repo.is_repository()),
        ("Telegram token", "configured" if config.telegram_bot_token else "missing", bool(config.telegram_bot_token)),
        ("Telegram user", str(config.telegram_allowed_user_id or "missing"), config.telegram_allowed_user_id is not None),
        ("OpenAI API", "configured" if config.openai_api_key else "missing", bool(config.openai_api_key)),
    ]
    table = Table("Check", "Value", "OK")
    for name, value, ok in rows:
        table.add_row(name, value, "yes" if ok else "no")
    console.print(table)


@app.command("telegram-id")
def telegram_id() -> None:
    """Show recent Telegram sender and chat IDs after the user messages the bot."""
    config = OrchestratorConfig.from_env()
    if not config.telegram_bot_token:
        raise ConfigError("TELEGRAM_DEV_BOT_TOKEN is missing")
    rows = asyncio.run(discover_recent_users(config.telegram_bot_token))
    table = Table("User ID", "Chat ID", "Username", "Name")
    for row in rows:
        table.add_row(
            str(row["user_id"]),
            str(row["chat_id"]),
            str(row["username"]),
            str(row["first_name"]),
        )
    console.print(table)
    if not rows:
        console.print("Send /start to the development bot, then run this command again.")


@app.command()
def roadmap() -> None:
    config, _, _ = _services()
    tasks = load_roadmap(config.project_root / "ROADMAP.yaml")
    table = Table("ID", "Risk", "Approval", "Title")
    for task in tasks:
        table.add_row(task.task_id, task.risk_level.value, str(task.requires_approval), task.title)
    console.print(table)


@app.command()
def status() -> None:
    _, db, _ = _services()
    rows = db.list_changes(limit=20)
    table = Table("Change", "Task", "Risk", "Status", "Title")
    for row in rows:
        table.add_row(
            row["change_id"], row["task_id"], row["risk_level"], row["status"], row["title"]
        )
    console.print(table)


@app.command("run-next")
def run_next() -> None:
    config, db, pipeline = _services()
    telegram = None
    if config.telegram_bot_token and config.telegram_allowed_user_id is not None:
        telegram = TelegramGateway(config, db, pipeline)
    worker = RoadmapWorker(pipeline, db, telegram, config.project_root / "ROADMAP.yaml")
    change_id = asyncio.run(worker.run_once())
    console.print(change_id or "No pending roadmap tasks.")


@app.command()
def approve(change_id: str, user_id: Annotated[int | None, typer.Option()] = None) -> None:
    _, _, pipeline = _services()
    commit = pipeline.approve(change_id.upper(), user_id)
    console.print(f"[green]Merged[/green] {change_id.upper()} -> {commit}")


@app.command()
def reject(
    change_id: str,
    reason: str,
    user_id: Annotated[int | None, typer.Option()] = None,
) -> None:
    _, _, pipeline = _services()
    pipeline.reject(change_id.upper(), user_id, reason)
    console.print(f"[yellow]Rejected[/yellow] {change_id.upper()}")


@app.command()
def show(change_id: str) -> None:
    _, db, _ = _services()
    change = db.get_change(change_id.upper())
    if not change:
        raise typer.BadParameter("Unknown change")
    console.print_json(json.dumps(change, ensure_ascii=False, default=str))


@app.command("telegram")
def telegram_service() -> None:
    config, db, pipeline = _services()
    gateway = TelegramGateway(config, db, pipeline)
    asyncio.run(gateway.run_forever())


@app.command("service")
def service() -> None:
    config, db, pipeline = _services()
    pipeline.prepare_repository()
    gateway = TelegramGateway(config, db, pipeline)
    worker = RoadmapWorker(pipeline, db, gateway, config.project_root / "ROADMAP.yaml")

    async def main() -> None:
        jobs = [asyncio.create_task(gateway.run_forever())]
        if config.worker_enabled:
            jobs.append(asyncio.create_task(worker.run_forever()))
        await asyncio.gather(*jobs)

    asyncio.run(main())


if __name__ == "__main__":
    app()
