from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1260px; margin: 36px auto; padding: 0 18px; color: #17202a; }
h1, h2, h3 { color: #102a43; }
.card { border: 1px solid #d9e2ec; border-radius: 12px; padding: 16px; margin: 14px 0; background: #f8fbff; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; }
th, td { border: 1px solid #d9e2ec; padding: 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.good { color: #137333; font-weight: 700; }
.warn { color: #b06000; font-weight: 700; }
code { background: #eef2f6; padding: 2px 5px; border-radius: 4px; }
.small { color: #52606d; font-size: 0.92rem; }
.scroll { overflow-x: auto; }
"""


def _format_timestamp(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _interval(values: dict[str, Any], name: str) -> str:
    value = _number(values.get(name))
    low = values.get(f"{name}_ci_low")
    high = values.get(f"{name}_ci_high")
    if low is None or high is None:
        return value
    return f"{value} [{float(low):.3f}–{float(high):.3f}]"


def _provenance_html(provenance: dict[str, Any]) -> str:
    symbols = provenance.get("symbols", [])
    symbols_text = ", ".join(map(str, symbols)) if symbols else "—"
    source = str(provenance.get("source", "UNKNOWN"))
    source_label = {
        "REAL_BINANCE_FUTURES": "Реальные исторические данные Binance USD-M Futures",
        "SYNTHETIC": "Детерминированные синтетические данные",
    }.get(source, source)
    return f"""
<div class="card">
<h2>Происхождение данных</h2>
<p><strong>Источник:</strong> {html.escape(source_label)}</p>
<p><strong>Контракты:</strong> {html.escape(symbols_text)}</p>
<p><strong>Период:</strong> {html.escape(_format_timestamp(provenance.get('start')))} — {html.escape(_format_timestamp(provenance.get('end')))}</p>
<p><strong>Строк/снимков:</strong> {int(provenance.get('rows', 0)):,}; <strong>независимых событий:</strong> {int(provenance.get('events', 0)):,}</p>
<p><strong>Режим пампа:</strong> {html.escape(str(provenance.get('regime', 'ALL')))}</p>
<p class="small">Сформировано: {html.escape(str(provenance.get('generated_at', '—')))}</p>
</div>
"""


def _split_html(split_summary: list[dict[str, Any]]) -> str:
    if not split_summary:
        return ""
    rows = []
    for item in split_summary:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('split', '')))}</td>"
            f"<td>{int(item.get('rows', 0)):,}</td>"
            f"<td>{int(item.get('positives', 0)):,}</td>"
            f"<td>{float(item.get('positive_rate', 0)):.2%}</td>"
            f"<td>{int(item.get('events', 0)):,}</td>"
            f"<td>{int(item.get('positive_events', 0)):,}</td>"
            f"<td>{html.escape(_format_timestamp(item.get('start')))}</td>"
            f"<td>{html.escape(_format_timestamp(item.get('end')))}</td>"
            "</tr>"
        )
    return f"""
<h2>Временное разделение</h2>
<div class="scroll"><table><thead><tr><th>Раздел</th><th>Строки</th><th>Положительные</th><th>Доля</th><th>События</th><th>+События</th><th>Начало</th><th>Конец</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
"""


def _snapshot_metrics_html(metrics: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, values in metrics.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_number(values.get('average_precision'))}</td>"
            f"<td>{_number(values.get('roc_auc'))}</td>"
            f"<td>{_number(values.get('precision'))}</td>"
            f"<td>{_number(values.get('recall'))}</td>"
            f"<td>{_number(values.get('f1'))}</td>"
            f"<td>{_number(values.get('brier'))}</td>"
            f"<td>{_number(values.get('false_alerts_per_day'), 2)}</td>"
            f"<td>{_number(values.get('threshold'), 2)}</td>"
            "</tr>"
        )
    return f"""
<h2>Snapshot-метрики на test</h2>
<div class="scroll"><table><thead><tr><th>Модель</th><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier</th><th>Ложных/день</th><th>Порог</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
"""


def _event_metrics_html(metrics: dict[str, dict[str, Any]]) -> str:
    rows = []
    for name, values in metrics.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(_interval(values, 'event_precision'))}</td>"
            f"<td>{html.escape(_interval(values, 'event_recall'))}</td>"
            f"<td>{html.escape(_interval(values, 'event_f1'))}</td>"
            f"<td>{int(values.get('true_positive_events', 0))}</td>"
            f"<td>{int(values.get('false_positive_events', 0))}</td>"
            f"<td>{_number(values.get('false_events_per_day'), 2)}</td>"
            f"<td>{_number(values.get('alerts_per_predicted_event'), 2)}</td>"
            f"<td>{_number(values.get('median_lead_to_first_positive_minutes'), 1)}</td>"
            f"<td>{_number(values.get('median_first_alert_minutes_before_peak'), 1)}</td>"
            "</tr>"
        )
    return f"""
<h2>Event-level метрики на test</h2>
<p class="small">Доверительные интервалы рассчитаны grouped bootstrap по целым <code>event_id</code>. Lead time измеряется до первого snapshot, для которого целевое падение уже попадает в будущий горизонт.</p>
<div class="scroll"><table><thead><tr><th>Модель</th><th>Event precision [95% CI]</th><th>Event recall [95% CI]</th><th>Event F1 [95% CI]</th><th>TP events</th><th>FP events</th><th>Ложных events/день</th><th>Alerts/event</th><th>Lead до первого positive, мин</th><th>Первый alert до peak, мин</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
"""


def _availability_html(availability: pd.DataFrame, dropped: list[str]) -> str:
    if availability.empty:
        return ""
    display = availability.copy()
    percentage_columns = [column for column in display if column.endswith("_available")]
    for column in percentage_columns:
        display[column] = display[column].map(lambda value: f"{float(value):.1%}")
    dropped_text = ", ".join(f"<code>{html.escape(name)}</code>" for name in dropped) or "нет"
    return f"""
<h2>Доступность признаков</h2>
<p>Полностью пустые признаки в train автоматически исключены: {dropped_text}.</p>
<div class="scroll">{display.to_html(index=False, border=0, escape=True)}</div>
"""


def render_report(
    metrics: dict[str, dict[str, Any]],
    events: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    output_path: str | Path,
    version: str = "0.2.4",
    provenance: dict[str, Any] | None = None,
    split_summary: list[dict[str, Any]] | None = None,
    feature_availability: pd.DataFrame | None = None,
    dropped_features: list[str] | None = None,
) -> Path:
    events_html = "<p>Подходящие памп-события не найдены.</p>"
    if not events.empty:
        preferred = [
            "event_id",
            "symbol",
            "pump_start",
            "peak_time",
            "pump_regime",
            "merged_event_count",
            "pump_return_to_peak",
            "event_pump_threshold",
            "minutes_to_peak",
            "peak_drawdown_5m",
            "peak_drawdown_15m",
            "peak_drawdown_30m",
            "peak_drawdown_60m",
            "max_event_drawdown_class",
        ]
        columns = [column for column in preferred if column in events]
        display = events.sort_values("pump_start").tail(40)[columns].copy()
        events_html = display.to_html(
            index=False,
            border=0,
            float_format=lambda value: f"{value:.4f}",
        )

    provenance = dict(provenance or {})
    provenance.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    source = provenance.get("source", "UNKNOWN")
    caution = (
        "Высокий результат на синтетике проверяет код и протокол, но не доказывает прогнозную силу на реальном рынке."
        if source == "SYNTHETIC"
        else "Это исторический бэктест на реальных данных. Он не является торговой рекомендацией и ещё не доказывает устойчивость в онлайне."
    )

    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Pump/Dump Research v{html.escape(version)}</title><style>{STYLE}</style></head>
<body>
<h1>Pump/Dump Research v{html.escape(version)}</h1>
<div class="card">
<h2>Эксперимент</h2>
<p>Целевая метка: <code>{html.escape(target)}</code></p>
<p>Пересекающиеся pump-to-peak интервалы объединяются в одно событие. Цель каждой строки измеряется от цены собственного snapshot timestamp. Event-level оценка объединяет все тревоги одного пампа.</p>
</div>
{_provenance_html(provenance)}
{_split_html(split_summary or [])}
{_snapshot_metrics_html(metrics)}
{_event_metrics_html(metrics)}
<div class="card"><strong>Важно:</strong> {html.escape(caution)}</div>
{_availability_html(feature_availability if feature_availability is not None else pd.DataFrame(), dropped_features or [])}
<h2>Очищенные события</h2>
<div class="scroll">{events_html}</div>
<h2>Эффективные признаки модели</h2>
<p>{', '.join(f'<code>{html.escape(name)}</code>' for name in feature_columns)}</p>
<h2>Машиночитаемые метрики</h2>
<pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination


def render_loso_report(
    fold_metrics: pd.DataFrame,
    pooled_metrics: dict[str, dict[str, Any]],
    output_path: str | Path,
    target: str,
    version: str = "0.2.4",
) -> Path:
    fold_html = (
        fold_metrics.to_html(index=False, border=0, float_format=lambda value: f"{value:.4f}")
        if not fold_metrics.empty
        else "<p>Подходящие folds не сформированы.</p>"
    )
    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>LOSO Evaluation v{html.escape(version)}</title><style>{STYLE}</style></head>
<body>
<h1>Leave-One-Symbol-Out — v{html.escape(version)}</h1>
<div class="card"><p>Цель: <code>{html.escape(target)}</code>. Каждый контракт по очереди полностью исключается из обучения и используется только как внешний test.</p></div>
{_snapshot_metrics_html(pooled_metrics)}
{_event_metrics_html(pooled_metrics)}
<h2>Результаты по удержанным контрактам</h2>
<div class="scroll">{fold_html}</div>
<h2>Машиночитаемый pooled summary</h2>
<pre>{html.escape(json.dumps(pooled_metrics, ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination


def _metric_or_dash(value: Any, digits: int = 3) -> str:
    return _number(value, digits)


def render_prospective_report(
    summary: dict[str, Any],
    symbol_metrics: pd.DataFrame,
    fold_log: pd.DataFrame,
    exposure: pd.DataFrame,
    output_path: str | Path,
    target: str,
    model_name: str,
    version: str = "0.2.4",
    calibration_start: Any = None,
    evaluation_start: Any = None,
) -> Path:
    """Render the purged walk-forward LOSO prospective protocol report."""
    symbol_html = (
        symbol_metrics.to_html(
            index=False,
            border=0,
            float_format=lambda value: f"{value:.4f}",
        )
        if not symbol_metrics.empty
        else "<p>Нет валидных символов.</p>"
    )
    folds = fold_log.copy()
    if not folds.empty:
        preferred = [
            "fold_type",
            "event_id",
            "holdout_symbol",
            "event_start",
            "train_events",
            "train_positive_events",
            "train_symbols",
            "train_max_event_end",
            "features_used",
            "status",
        ]
        folds = folds[[column for column in preferred if column in folds]].tail(80)
    fold_html = (
        folds.to_html(index=False, border=0, escape=True)
        if not folds.empty
        else "<p>Нет fold-записей.</p>"
    )
    exposure_html = (
        exposure.to_html(
            index=False,
            border=0,
            float_format=lambda value: f"{value:.3f}",
        )
        if not exposure.empty
        else "<p>Экспозиция не рассчитана.</p>"
    )
    macro_positive = summary.get("macro_positive_symbols", {}) or {}
    macro_all = summary.get("macro_all_symbols", {}) or {}
    body = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Prospective Protocol v{html.escape(version)}</title><style>{STYLE}</style></head>
<body>
<h1>Purged Walk-Forward LOSO — v{html.escape(version)}</h1>
<div class="card">
<p><strong>Цель:</strong> <code>{html.escape(target)}</code></p>
<p><strong>Модель:</strong> <code>{html.escape(model_name)}</code></p>
<p><strong>Один глобальный порог:</strong> {_number(summary.get('global_threshold'), 2)} — выбран только на ранних out-of-symbol calibration событиях.</p>
<p><strong>Purging:</strong> {int(summary.get('purge_minutes', 0))} минут. Для каждого evaluation-события обучение использует только завершившиеся события других контрактов из прошлого.</p>
<p><strong>Начало calibration:</strong> {html.escape(_format_timestamp(calibration_start))}; <strong>начало prospective evaluation:</strong> {html.escape(_format_timestamp(evaluation_start))}.</p>
</div>
<h2>Pooled snapshot-метрики</h2>
<div class="scroll"><table><thead><tr><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier</th><th>Глобальный порог</th></tr></thead>
<tbody><tr><td>{_number(summary.get('average_precision'))}</td><td>{_number(summary.get('roc_auc'))}</td><td>{_number(summary.get('precision'))}</td><td>{_number(summary.get('recall'))}</td><td>{_number(summary.get('f1'))}</td><td>{_number(summary.get('brier'))}</td><td>{_number(summary.get('global_threshold'), 2)}</td></tr></tbody></table></div>
<h2>Pooled event-level метрики</h2>
<div class="scroll"><table><thead><tr><th>Event precision [95% CI]</th><th>Event recall [95% CI]</th><th>Event F1 [95% CI]</th><th>TP</th><th>FP</th><th>False events / 100 symbol-days</th><th>Symbols detected</th></tr></thead>
<tbody><tr><td>{html.escape(_interval(summary, 'event_precision'))}</td><td>{html.escape(_interval(summary, 'event_recall'))}</td><td>{html.escape(_interval(summary, 'event_f1'))}</td><td>{int(summary.get('true_positive_events', 0))}</td><td>{int(summary.get('false_positive_events', 0))}</td><td>{_number(summary.get('false_events_per_100_symbol_days'), 2)}</td><td>{int(summary.get('symbols_with_detected_positive_event', 0))}/{int(summary.get('symbols_with_positive_events', 0))} ({float(summary.get('symbol_detection_coverage', 0)):.1%})</td></tr></tbody></table></div>
<h2>Macro-метрики по контрактам</h2>
<p class="small">Macro positive считает каждый контракт с хотя бы одним положительным event одинаково, поэтому крупная выборка одной монеты не доминирует.</p>
<div class="scroll"><table><thead><tr><th>Группа</th><th>Macro PR-AUC</th><th>Macro event precision</th><th>Macro event recall</th><th>Macro event F1</th></tr></thead>
<tbody>
<tr><td>Все evaluation-контракты</td><td>{_metric_or_dash(macro_all.get('average_precision'))}</td><td>{_metric_or_dash(macro_all.get('event_precision'))}</td><td>{_metric_or_dash(macro_all.get('event_recall'))}</td><td>{_metric_or_dash(macro_all.get('event_f1'))}</td></tr>
<tr><td>Контракты с положительными events</td><td>{_metric_or_dash(macro_positive.get('average_precision'))}</td><td>{_metric_or_dash(macro_positive.get('event_precision'))}</td><td>{_metric_or_dash(macro_positive.get('event_recall'))}</td><td>{_metric_or_dash(macro_positive.get('event_f1'))}</td></tr>
</tbody></table></div>
<h2>Распределение упреждения</h2>
<div class="scroll"><table><thead><tr><th>P25, мин</th><th>Медиана, мин</th><th>P75, мин</th><th>Доля ≥5 мин</th><th>Доля ≥10 мин</th><th>Первый alert до peak, медиана</th></tr></thead>
<tbody><tr><td>{_number(summary.get('lead_minutes_p25'), 1)}</td><td>{_number(summary.get('lead_minutes_median'), 1)}</td><td>{_number(summary.get('lead_minutes_p75'), 1)}</td><td>{_number(summary.get('lead_share_ge_5m'), 2)}</td><td>{_number(summary.get('lead_share_ge_10m'), 2)}</td><td>{_number(summary.get('median_first_alert_minutes_before_peak'), 1)}</td></tr></tbody></table></div>
<h2>Результаты по неизвестным контрактам</h2>
<div class="scroll">{symbol_html}</div>
<h2>Экспозиция</h2>
<p>Всего: <strong>{_number(summary.get('total_symbol_days'), 1)}</strong> symbol-days; ложных snapshot-alerts на 100 symbol-days: <strong>{_number(summary.get('false_alerts_per_100_symbol_days'), 2)}</strong>.</p>
<div class="scroll">{exposure_html}</div>
<h2>Аудит walk-forward folds</h2>
<p class="small">В каждой строке <code>train_max_event_end</code> должен быть раньше <code>event_start</code> минимум на purge-интервал, а holdout-контракт полностью отсутствует в train.</p>
<div class="scroll">{fold_html}</div>
<div class="card"><strong>Важно:</strong> Это строгий исторический prospective-style тест. Он не заменяет живой paper-monitoring и не является торговой рекомендацией.</div>
<h2>Машиночитаемый summary</h2>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))}</pre>
</body></html>"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination
