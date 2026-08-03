from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.db import OrchestratorDB
from orchestrator.models import ChangeStatus
from orchestrator.pipeline import DevelopmentPipeline
from orchestrator.roadmap import load_roadmap
from orchestrator.telegram import TelegramGateway


class RoadmapWorker:
    def __init__(
        self,
        pipeline: DevelopmentPipeline,
        db: OrchestratorDB,
        telegram: TelegramGateway | None,
        roadmap_path: Path,
    ):
        self.pipeline = pipeline
        self.db = db
        self.telegram = telegram
        self.roadmap_path = roadmap_path

    def _completed_task_ids(self) -> set[str]:
        terminal_or_active = {
            ChangeStatus.QUEUED.value,
            ChangeStatus.GENERATING.value,
            ChangeStatus.VALIDATING.value,
            ChangeStatus.PENDING_APPROVAL.value,
            ChangeStatus.APPROVED.value,
            ChangeStatus.MERGED.value,
        }
        return {
            str(row["task_id"])
            for row in self.db.list_changes(limit=10_000)
            if row["status"] in terminal_or_active
        }

    async def run_once(self) -> str | None:
        completed = self._completed_task_ids()
        for task in load_roadmap(self.roadmap_path):
            if task.task_id in completed:
                continue
            if task.requires_approval and self.telegram is None:
                raise RuntimeError("Telegram approval gateway is required for this task")
            change_id = await asyncio.to_thread(self.pipeline.create_proposal, task)
            if self.telegram:
                await self.telegram.notify_change(change_id)
            return change_id
        return None

    async def run_forever(self, interval_seconds: int = 60) -> None:
        while True:
            pending = self.db.list_changes(limit=100)
            blocked = any(row["status"] == ChangeStatus.PENDING_APPROVAL.value for row in pending)
            if not blocked:
                try:
                    await self.run_once()
                except Exception:
                    await asyncio.sleep(interval_seconds)
            await asyncio.sleep(interval_seconds)
