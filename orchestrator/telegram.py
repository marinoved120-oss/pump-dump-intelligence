from __future__ import annotations

import asyncio
import html
import json
from dataclasses import dataclass

import httpx

from orchestrator.config import OrchestratorConfig
from orchestrator.db import OrchestratorDB
from orchestrator.models import ChangeStatus
from orchestrator.pipeline import DevelopmentPipeline, PipelineError


class TelegramError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramCommand:
    name: str
    args: str


def parse_command(text: str) -> TelegramCommand | None:
    value = text.strip()
    if not value.startswith("/"):
        return None
    head, _, args = value.partition(" ")
    name = head.split("@", 1)[0].lower()
    return TelegramCommand(name=name, args=args.strip())


def _truncate(text: str, limit: int = 3500) -> str:
    return text if len(text) <= limit else text[: limit - 40] + "\n…[truncated]"


async def discover_recent_users(bot_token: str) -> list[dict[str, object]]:
    """Return unique private-chat senders from recent bot updates for one-time setup."""
    base_url = f"https://api.telegram.org/bot{bot_token}"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{base_url}/getUpdates",
            json={"timeout": 0, "limit": 100, "allowed_updates": ["message"]},
        )
    if response.status_code >= 400:
        raise TelegramError(f"Telegram API error {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if not payload.get("ok"):
        raise TelegramError(str(payload))
    users: dict[tuple[int, int], dict[str, object]] = {}
    for update in payload.get("result", []):
        message = update.get("message", {})
        sender = message.get("from", {})
        chat = message.get("chat", {})
        user_id = sender.get("id")
        chat_id = chat.get("id")
        if user_id is None or chat_id is None:
            continue
        users[(int(user_id), int(chat_id))] = {
            "user_id": int(user_id),
            "chat_id": int(chat_id),
            "username": sender.get("username") or "",
            "first_name": sender.get("first_name") or "",
        }
    return list(users.values())


class TelegramGateway:
    def __init__(self, config: OrchestratorConfig, db: OrchestratorDB, pipeline: DevelopmentPipeline):
        config.require_telegram()
        self.config = config
        self.db = db
        self.pipeline = pipeline
        self.base_url = f"https://api.telegram.org/bot{config.telegram_bot_token}"
        self.offset = 0

    async def _call(self, method: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.config.polling_timeout_seconds + 10) as client:
            response = await client.post(f"{self.base_url}/{method}", json=payload)
        if response.status_code >= 400:
            raise TelegramError(f"Telegram API error {response.status_code}: {response.text[:500]}")
        data = response.json()
        if not data.get("ok"):
            raise TelegramError(str(data))
        return data

    async def send(self, chat_id: int, text: str, reply_markup: dict | None = None) -> None:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": _truncate(text),
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        await self._call("sendMessage", payload)

    def _authorized(self, user_id: int | None) -> bool:
        return user_id == self.config.telegram_allowed_user_id

    async def notify_change(self, change_id: str) -> None:
        change = self.db.get_change(change_id)
        if not change:
            return
        chat_id = self.config.telegram_chat_id or self.config.telegram_allowed_user_id
        text = (
            f"🛠 <b>{html.escape(change_id)} — {html.escape(change['title'])}</b>\n\n"
            f"Риск: <b>{html.escape(change['risk_level'])}</b>\n"
            f"Статус: <b>{html.escape(change['status'])}</b>\n\n"
            f"{html.escape(change['description'])}\n\n"
            f"Проверки:\n<pre>{html.escape(change.get('validation_summary') or 'нет данных')}</pre>"
        )
        keyboard = None
        if change["status"] == ChangeStatus.PENDING_APPROVAL.value:
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "✅ Подтвердить", "callback_data": f"approve:{change_id}"},
                        {"text": "❌ Отклонить", "callback_data": f"reject:{change_id}"},
                    ],
                    [
                        {"text": "📄 Diff", "callback_data": f"diff:{change_id}"},
                        {"text": "🧪 Tests", "callback_data": f"tests:{change_id}"},
                    ],
                ]
            }
        await self.send(int(chat_id), text, keyboard)

    async def _handle_command(self, chat_id: int, user_id: int, command: TelegramCommand) -> None:
        if not self._authorized(user_id):
            await self.send(chat_id, "⛔ Доступ запрещён.")
            return
        if command.name in {"/start", "/help"}:
            await self.send(
                chat_id,
                "Команды:\n"
                "/status — состояние\n"
                "/changes — последние изменения\n"
                "/approve CHANGE-ID\n"
                "/reject CHANGE-ID причина\n"
                "/diff CHANGE-ID\n"
                "/tests CHANGE-ID",
            )
            return
        if command.name == "/status":
            pending = self.db.list_changes(ChangeStatus.PENDING_APPROVAL, limit=20)
            await self.send(chat_id, f"Ожидают подтверждения: <b>{len(pending)}</b>")
            return
        if command.name == "/changes":
            rows = self.db.list_changes(limit=10)
            text = "\n".join(
                f"<code>{html.escape(row['change_id'])}</code> — {html.escape(row['status'])} — "
                f"{html.escape(row['title'])}"
                for row in rows
            ) or "Изменений пока нет."
            await self.send(chat_id, text)
            return
        if command.name == "/approve":
            change_id = command.args.split()[0] if command.args else ""
            commit = self.pipeline.approve(change_id, user_id)
            await self.send(chat_id, f"✅ {html.escape(change_id)} merged: <code>{commit[:12]}</code>")
            return
        if command.name == "/reject":
            change_id, _, reason = command.args.partition(" ")
            if not change_id or not reason:
                await self.send(chat_id, "Формат: /reject CHANGE-ID причина")
                return
            self.pipeline.reject(change_id, user_id, reason)
            await self.send(chat_id, f"❌ {html.escape(change_id)} отклонён.")
            return
        if command.name in {"/diff", "/tests"}:
            change_id = command.args.split()[0] if command.args else ""
            change = self.db.get_change(change_id)
            if not change:
                await self.send(chat_id, "Изменение не найдено.")
                return
            key = "diff_text" if command.name == "/diff" else "validation_summary"
            await self.send(chat_id, f"<pre>{html.escape(change.get(key) or 'нет данных')}</pre>")
            return
        await self.send(chat_id, "Неизвестная команда. /help")

    async def _handle_callback(self, callback: dict) -> None:
        callback_id = callback.get("id")
        user_id = callback.get("from", {}).get("id")
        message = callback.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        data = str(callback.get("data", ""))
        if not self._authorized(user_id):
            await self._call(
                "answerCallbackQuery",
                {"callback_query_id": callback_id, "text": "Доступ запрещён", "show_alert": True},
            )
            return
        action, _, change_id = data.partition(":")
        try:
            if action == "approve":
                commit = self.pipeline.approve(change_id, user_id)
                text = f"Подтверждено, commit {commit[:12]}"
            elif action == "reject":
                self.pipeline.reject(change_id, user_id, "Rejected from Telegram button")
                text = "Отклонено"
            elif action == "diff":
                change = self.db.get_change(change_id)
                await self.send(chat_id, f"<pre>{html.escape((change or {}).get('diff_text') or 'нет данных')}</pre>")
                text = "Diff отправлен"
            elif action == "tests":
                change = self.db.get_change(change_id)
                await self.send(
                    chat_id,
                    f"<pre>{html.escape((change or {}).get('validation_summary') or 'нет данных')}</pre>",
                )
                text = "Результаты отправлены"
            else:
                text = "Неизвестное действие"
        except PipelineError as exc:
            text = str(exc)[:180]
        await self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})

    async def run_forever(self) -> None:
        while True:
            try:
                data = await self._call(
                    "getUpdates",
                    {
                        "offset": self.offset,
                        "timeout": self.config.polling_timeout_seconds,
                        "allowed_updates": ["message", "callback_query"],
                    },
                )
                for update in data.get("result", []):
                    self.offset = max(self.offset, int(update["update_id"]) + 1)
                    if "callback_query" in update:
                        await self._handle_callback(update["callback_query"])
                        continue
                    message = update.get("message", {})
                    text = message.get("text")
                    if not text:
                        continue
                    command = parse_command(str(text))
                    if command:
                        await self._handle_command(
                            int(message["chat"]["id"]),
                            int(message.get("from", {}).get("id", 0)),
                            command,
                        )
            except (httpx.HTTPError, TelegramError, asyncio.TimeoutError):
                await asyncio.sleep(5)
