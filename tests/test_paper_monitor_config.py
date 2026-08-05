from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import websockets
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from research.cli import app
from research.monitor.config import (
    MonitorConfigError,
    build_paper_monitor,
    load_monitor_config,
)
from research.monitor.preflight import (
    MonitorPreflightReport,
    run_monitor_preflight,
)


class MemorySink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def _safe_payload() -> dict[str, object]:
    return {
        "paper_monitor": {
            "mode": "paper_only",
            "alerts": {
                "cooldown_seconds": 900,
                "red_min_independent_groups": 3,
            },
            "telegram": {
                "parse_mode": "HTML",
                "max_message_chars": 4096,
                "redact_secrets": True,
                "disable_web_page_preview": True,
            },
            "outcomes": {
                "offsets_minutes": [
                    5,
                    15,
                    30,
                    60,
                    240,
                ],
                "storage_format": "jsonl",
                "path": (
                    "artifacts/"
                    "paper-outcomes.jsonl"
                ),
                "production_model_updates_enabled": (
                    False
                ),
            },
            "safety": {
                "exchange_trading_enabled": False,
                "exchange_trading_credentials_allowed": (
                    False
                ),
                "order_placement_enabled": False,
                "order_cancellation_enabled": False,
                "withdrawals_enabled": False,
                "automatic_trading_enabled": False,
            },
        }
    }


def _paper_section(
    payload: dict[str, object],
) -> dict[str, object]:
    paper = payload["paper_monitor"]
    assert isinstance(paper, dict)
    return paper


def _nested_section(
    payload: dict[str, object],
    name: str,
) -> dict[str, object]:
    section = _paper_section(payload)[name]
    assert isinstance(section, dict)
    return section


def _write_config(
    tmp_path: Path,
    payload: dict[str, object] | None = None,
) -> Path:
    path = tmp_path / "monitor.yaml"
    path.write_text(
        yaml.safe_dump(
            _safe_payload()
            if payload is None
            else payload,
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _check_by_name(
    report: MonitorPreflightReport,
    name: str,
) -> object:
    return next(
        check
        for check in report.checks
        if check.name == name
    )


def _is_binance_credential_name(
    name: str,
) -> bool:
    normalized = name.upper()

    if not normalized.startswith("BINANCE_"):
        return False

    if normalized in {
        "BINANCE_KEY",
        "BINANCE_SECRET",
    }:
        return True

    return normalized.endswith(
        (
            "API_KEY",
            "API_SECRET",
            "SECRET_KEY",
            "PRIVATE_KEY",
        )
    )


def test_safe_config_is_typed_frozen_and_used(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    document = load_monitor_config(path)
    settings = document.paper_monitor

    assert settings.mode == "paper_only"
    assert settings.outcomes.offsets_minutes == (
        5,
        15,
        30,
        60,
        240,
    )

    with pytest.raises(
        ValidationError,
        match="frozen",
    ):
        settings.mode = "paper_only"

    sink = MemorySink()
    monitor = build_paper_monitor(
        document,
        sink,
        project_root=tmp_path,
    )

    assert (
        monitor.config.cooldown_seconds
        == 900
    )
    assert (
        monitor.config
        .red_min_independent_groups
        == 3
    )
    assert monitor.outcomes.store is not None
    assert monitor.outcomes.store.path == (
        tmp_path
        / "artifacts"
        / "paper-outcomes.jsonl"
    )
    assert not monitor.outcomes.store.path.exists()
    assert sink.messages == []


def test_missing_config_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        MonitorConfigError,
        match="file not found",
    ):
        load_monitor_config(
            tmp_path / "missing.yaml"
        )


def test_malformed_yaml_fails_without_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "monitor.yaml"
    secret = "must-not-leak"
    path.write_text(
        "paper_monitor: [\n"
        f"  {secret}\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        MonitorConfigError,
        match="invalid YAML syntax",
    ) as captured:
        load_monitor_config(path)

    assert secret not in str(captured.value)


def test_missing_required_field_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _safe_payload()
    safety = _nested_section(
        payload,
        "safety",
    )
    del safety["withdrawals_enabled"]

    path = _write_config(
        tmp_path,
        payload,
    )

    with pytest.raises(
        MonitorConfigError,
        match="withdrawals_enabled",
    ):
        load_monitor_config(path)


def test_unknown_field_fails_without_value(
    tmp_path: Path,
) -> None:
    payload = _safe_payload()
    paper = _paper_section(payload)
    secret = "unknown-value-secret"
    paper["unexpected"] = {
        "token": secret,
    }

    path = _write_config(
        tmp_path,
        payload,
    )

    with pytest.raises(
        MonitorConfigError,
        match="unexpected",
    ) as captured:
        load_monitor_config(path)

    assert secret not in str(captured.value)


def test_strict_types_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _safe_payload()
    alerts = _nested_section(
        payload,
        "alerts",
    )
    alerts["cooldown_seconds"] = "900"

    path = _write_config(
        tmp_path,
        payload,
    )

    with pytest.raises(
        MonitorConfigError,
        match="cooldown_seconds",
    ):
        load_monitor_config(path)


@pytest.mark.parametrize(
    (
        "section_name",
        "field_name",
        "unsafe_value",
        "expected",
    ),
    [
        (
            "safety",
            "exchange_trading_enabled",
            True,
            "exchange_trading_enabled",
        ),
        (
            "safety",
            (
                "exchange_trading_"
                "credentials_allowed"
            ),
            True,
            (
                "exchange_trading_"
                "credentials_allowed"
            ),
        ),
        (
            "safety",
            "order_placement_enabled",
            True,
            "order_placement_enabled",
        ),
        (
            "safety",
            "order_cancellation_enabled",
            True,
            "order_cancellation_enabled",
        ),
        (
            "safety",
            "withdrawals_enabled",
            True,
            "withdrawals_enabled",
        ),
        (
            "safety",
            "automatic_trading_enabled",
            True,
            "automatic_trading_enabled",
        ),
        (
            "outcomes",
            (
                "production_model_"
                "updates_enabled"
            ),
            True,
            "production model updates",
        ),
        (
            "telegram",
            "redact_secrets",
            False,
            "redact_secrets",
        ),
    ],
)
def test_every_unsafe_flag_fails_closed(
    tmp_path: Path,
    section_name: str,
    field_name: str,
    unsafe_value: bool,
    expected: str,
) -> None:
    payload = deepcopy(_safe_payload())
    section = _nested_section(
        payload,
        section_name,
    )
    section[field_name] = unsafe_value

    path = _write_config(
        tmp_path,
        payload,
    )

    with pytest.raises(
        MonitorConfigError,
        match=expected,
    ):
        load_monitor_config(path)


@pytest.mark.parametrize(
    "outcome_path",
    [
        "/tmp/paper-outcomes.jsonl",
        "../paper-outcomes.jsonl",
        "data/paper-outcomes.jsonl",
        r"artifacts\paper-outcomes.jsonl",
        "artifacts/paper-outcomes.txt",
        " artifacts/paper-outcomes.jsonl",
    ],
)
def test_preflight_rejects_unsafe_outcome_paths(
    tmp_path: Path,
    outcome_path: str,
) -> None:
    payload = _safe_payload()
    outcomes = _nested_section(
        payload,
        "outcomes",
    )
    outcomes["path"] = outcome_path
    path = _write_config(
        tmp_path,
        payload,
    )

    report = run_monitor_preflight(
        path,
        project_root=tmp_path,
        environ={},
    )
    check = _check_by_name(
        report,
        "outcome_storage",
    )

    assert report.status == "failed"
    assert check.status == "failed"


def test_preflight_rejects_directory_target(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    target = (
        tmp_path
        / "artifacts"
        / "paper-outcomes.jsonl"
    )
    target.mkdir(parents=True)

    report = run_monitor_preflight(
        path,
        project_root=tmp_path,
        environ={},
    )
    check = _check_by_name(
        report,
        "outcome_storage",
    )

    assert report.status == "failed"
    assert check.status == "failed"
    assert "must target a file" in check.detail


def test_preflight_report_is_deterministic_and_redacted(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    secret = "do-not-print-this-value"
    environ = {
        "BINANCE_API_KEY": secret,
        "BINANCE_FUTURES_BASE_URL": (
            "https://fapi.binance.com"
        ),
    }

    first = run_monitor_preflight(
        path,
        project_root=tmp_path,
        environ=environ,
    )
    second = run_monitor_preflight(
        path,
        project_root=tmp_path,
        environ=environ,
    )
    output = first.to_json()

    assert first.status == "failed"
    assert output == second.to_json()
    assert secret not in output
    assert "BINANCE_API_KEY" in output
    assert "BINANCE_FUTURES_BASE_URL" not in output

    parsed = json.loads(output)
    assert parsed["schema_version"] == 1
    assert parsed["status"] == "failed"


def test_safe_preflight_has_zero_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(tmp_path)
    outcome_path = (
        tmp_path
        / "artifacts"
        / "paper-outcomes.jsonl"
    )

    def fail_network(
        *_args: object,
        **_kwargs: object,
    ) -> None:
        raise AssertionError(
            "preflight attempted a network request"
        )

    monkeypatch.setattr(
        httpx.Client,
        "request",
        fail_network,
    )
    monkeypatch.setattr(
        httpx.AsyncClient,
        "request",
        fail_network,
    )
    monkeypatch.setattr(
        websockets,
        "connect",
        fail_network,
    )

    report = run_monitor_preflight(
        path,
        project_root=tmp_path,
        environ={},
    )

    assert report.status == "passed"
    assert report.outcome_path == (
        "artifacts/paper-outcomes.jsonl"
    )
    assert not outcome_path.exists()
    assert not outcome_path.parent.exists()


def test_cli_returns_machine_readable_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(tmp_path)

    for name in tuple(os.environ):
        if _is_binance_credential_name(name):
            monkeypatch.delenv(
                name,
                raising=False,
            )

    runner = CliRunner()
    arguments = [
        "paper-preflight",
        "--config",
        str(path),
        "--project-root",
        str(tmp_path),
    ]

    safe = runner.invoke(
        app,
        arguments,
    )

    assert safe.exit_code == 0
    assert json.loads(
        safe.stdout.strip()
    )["status"] == "passed"

    secret = "cli-secret-value"
    unsafe = runner.invoke(
        app,
        arguments,
        env={
            "BINANCE_API_KEY": secret,
        },
    )

    assert unsafe.exit_code == 2
    assert secret not in unsafe.stdout
    assert json.loads(
        unsafe.stdout.strip()
    )["status"] == "failed"
