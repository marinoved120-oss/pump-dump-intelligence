from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets
from pydantic import ValidationError
from typer.testing import CliRunner

from research.cli import app
from research.monitor.runtime import (
    AlertReplayEvent,
    PaperAnalysisInput,
    run_paper_replay,
)

runner = CliRunner()


def _analysis_payload() -> dict[str, Any]:
    return {
        "market_phase": "markup",
        "likely_pump_causes": [
            "spot demand",
            "short covering",
        ],
        "spot_role": "spot-led demand",
        "futures_role": "confirming but not leading",
        "open_interest_state": "OI falling",
        "funding_and_leverage": "funding neutral",
        "liquidation_state": "short liquidations elevated",
        "spot_order_book_evidence": [
            "persistent bid support",
        ],
        "futures_order_book_evidence": [
            "moderate ask liquidity",
        ],
        "whale_limit_orders": [
            "large bid persisted",
        ],
        "spoofing_evidence": [
            "insufficient for spoofing-like hypothesis",
        ],
        "iceberg_evidence": [
            "visible refill observed",
        ],
        "absorption_evidence": [
            "buy flow with weak price response",
        ],
        "liquidity_pull_evidence": [
            "ask liquidity pulled",
        ],
        "spot_futures_divergence": "spot moved before futures",
        "technical_structure": "range breakout",
        "onchain_and_news_context": (
            "token=abc123 no verified catalyst"
        ),
        "supporting_evidence": [
            "spot aggressive demand",
            "short liquidations",
            "price breakout",
        ],
        "contradictions": [
            "futures basis did not expand",
        ],
        "missing_data": [
            "onchain data",
        ],
        "alternative_scenarios": [
            "cross-venue repricing",
        ],
        "invalidation_conditions": [
            "full return below breakout level",
        ],
        "observed_outcome": "pending",
    }


def _alert_payload(
    alert_id: str = "paper-001",
    *,
    created_ts_ms: int = 0,
) -> dict[str, Any]:
    return {
        "alert_id": alert_id,
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "created_ts_ms": created_ts_ms,
        "reference_price": 100.0,
        "alert_level": "red",
        "score": 0.91,
        "confidence": 0.82,
        "data_quality_score": 0.88,
        "independent_evidence_groups": [
            "derivatives",
            "order_flow",
            "price_structure",
        ],
        "analysis": _analysis_payload(),
    }


def _alert_event(
    alert_id: str = "paper-001",
    *,
    event_ts_ms: int = 0,
    created_ts_ms: int = 0,
) -> dict[str, Any]:
    return {
        "type": "alert",
        "event_ts_ms": event_ts_ms,
        "alert": _alert_payload(
            alert_id,
            created_ts_ms=created_ts_ms,
        ),
    }


def _price_event(
    event_ts_ms: int,
    price: float,
) -> dict[str, Any]:
    return {
        "type": "price",
        "event_ts_ms": event_ts_ms,
        "prices_by_symbol": {
            "BTCUSDT": price,
        },
    }


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                record,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    config_path = root / "configs" / "monitor.yaml"
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(
        Path("configs/monitor.yaml"),
        config_path,
    )
    (root / "artifacts").mkdir()
    return root, config_path


def _run(
    root: Path,
    config_path: Path,
):
    return run_paper_replay(
        "artifacts/alerts.jsonl",
        "artifacts/prices.jsonl",
        "artifacts/messages.jsonl",
        config_path=config_path,
        project_root=root,
        environ={},
    )


def _outcome_lines(root: Path) -> list[str]:
    return (
        root / "artifacts" / "paper-outcomes.jsonl"
    ).read_text(encoding="utf-8").splitlines()


def _message_lines(root: Path) -> list[str]:
    return (
        root / "artifacts" / "messages.jsonl"
    ).read_text(encoding="utf-8").splitlines()


def test_replay_publishes_initial_alert_and_redacts_message(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)
    _write_jsonl(
        root / "artifacts" / "alerts.jsonl",
        [_alert_event()],
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [_price_event(1, 100.0)],
    )

    report = _run(root, config_path)

    assert report.status == "passed"
    assert report.preflight_status == "passed"
    assert report.events_processed == 2
    assert report.alerts_published == 1
    assert report.alerts_suppressed == 0
    assert report.messages_appended == 1
    assert report.errors == ()

    messages = _message_lines(root)
    assert len(messages) == 1
    assert "PAPER MONITOR" in messages[0]
    assert "abc123" not in messages[0]
    assert "[REDACTED]" in messages[0]


def test_replay_uses_existing_cooldown_and_invalidation_rules(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)
    _write_jsonl(
        root / "artifacts" / "alerts.jsonl",
        [
            _alert_event(),
            _alert_event(
                "paper-002",
                event_ts_ms=1_000,
                created_ts_ms=1_000,
            ),
            {
                "type": "invalidation",
                "event_ts_ms": 2_000,
                "alert_id": "paper-001",
                "reason": "Full return below breakout level",
            },
            {
                "type": "invalidation",
                "event_ts_ms": 3_000,
                "alert_id": "paper-001",
                "reason": "Full return below breakout level",
            },
        ],
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [_price_event(4_000, 100.0)],
    )

    report = _run(root, config_path)

    assert report.status == "passed"
    assert report.alerts_published == 1
    assert report.alerts_suppressed == 1
    assert report.invalidations_published == 1
    assert report.invalidations_suppressed == 1
    assert report.messages_appended == 2

    messages = _message_lines(root)
    assert len(messages) == 2
    assert "INVALIDATION UPDATE" in messages[1]


def test_replay_sorts_events_and_records_due_outcomes(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)
    _write_jsonl(
        root / "artifacts" / "alerts.jsonl",
        [_alert_event()],
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [
            _price_event(30 * 60_000, 99.0),
            _price_event(5 * 60_000, 101.0),
            _price_event(15 * 60_000, 102.0),
        ],
    )

    report = _run(root, config_path)

    assert report.status == "passed"
    assert report.events_processed == 4
    assert report.outcomes_recorded == 3

    records = [
        json.loads(line)
        for line in _outcome_lines(root)
    ]
    assert [
        record["offset_minutes"]
        for record in records
    ] == [5, 15, 30]
    assert [
        record["observed_price"]
        for record in records
    ] == [101.0, 102.0, 99.0]


def test_restart_does_not_duplicate_stored_outcomes(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)
    _write_jsonl(
        root / "artifacts" / "alerts.jsonl",
        [_alert_event()],
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [
            _price_event(5 * 60_000, 101.0),
            _price_event(15 * 60_000, 102.0),
            _price_event(30 * 60_000, 99.0),
        ],
    )

    first = _run(root, config_path)
    first_lines = _outcome_lines(root)
    second = _run(root, config_path)
    second_lines = _outcome_lines(root)

    assert first.status == "passed"
    assert first.outcomes_recorded == 3
    assert second.status == "passed"
    assert second.outcomes_recorded == 0
    assert second_lines == first_lines


def test_strict_event_models_reject_unknown_fields_and_mutation() -> None:
    raw = _alert_event()
    raw["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        AlertReplayEvent.model_validate(raw)

    analysis = PaperAnalysisInput.model_validate(
        _analysis_payload()
    )
    with pytest.raises(ValidationError, match="frozen"):
        analysis.market_phase = "distribution"


def test_malformed_input_fails_closed_with_machine_report(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)
    alerts_path = root / "artifacts" / "alerts.jsonl"
    alerts_path.write_text(
        '{"type":"alert"\n',
        encoding="utf-8",
        newline="\n",
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [_price_event(1, 100.0)],
    )

    report = _run(root, config_path)

    assert report.status == "failed"
    assert report.preflight_status == "passed"
    assert report.events_processed == 0
    assert report.messages_appended == 0
    assert report.errors == (
        "alerts input JSON syntax invalid at line 1",
    )
    assert json.loads(report.to_json()) == report.to_dict()
    assert not (
        root / "artifacts" / "messages.jsonl"
    ).exists()


def test_preflight_runs_before_runtime_files_are_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    config_path = root / "configs" / "monitor.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "paper_monitor: invalid\n",
        encoding="utf-8",
        newline="\n",
    )

    report = _run(root, config_path)

    assert report.status == "failed"
    assert report.preflight_status == "failed"
    assert report.events_processed == 0
    assert "invalid monitor config" in report.errors[0]
    assert "alerts input file not found" not in report.errors[0]


def test_unsafe_runtime_path_fails_closed_without_storage(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)

    report = run_paper_replay(
        "../alerts.jsonl",
        "artifacts/prices.jsonl",
        "artifacts/messages.jsonl",
        config_path=config_path,
        project_root=root,
        environ={},
    )

    assert report.status == "failed"
    assert report.preflight_status == "passed"
    assert report.events_processed == 0
    assert report.errors == (
        "alerts input path cannot contain parent traversal",
    )
    assert not (
        root / "artifacts" / "messages.jsonl"
    ).exists()


def test_replay_performs_no_network_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path = _project(tmp_path)
    _write_jsonl(
        root / "artifacts" / "alerts.jsonl",
        [_alert_event()],
    )
    _write_jsonl(
        root / "artifacts" / "prices.jsonl",
        [_price_event(1, 100.0)],
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

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

    report = _run(root, config_path)

    assert report.status == "passed"


def test_cli_emits_one_json_report_and_nonzero_on_failure(
    tmp_path: Path,
) -> None:
    root, config_path = _project(tmp_path)

    result = runner.invoke(
        app,
        [
            "paper-replay",
            "--alerts",
            "artifacts/missing-alerts.jsonl",
            "--prices",
            "artifacts/missing-prices.jsonl",
            "--messages",
            "artifacts/messages.jsonl",
            "--config",
            str(config_path),
            "--project-root",
            str(root),
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout.strip())
    assert payload["status"] == "failed"
    assert payload["preflight_status"] == "passed"
    assert payload["events_processed"] == 0
