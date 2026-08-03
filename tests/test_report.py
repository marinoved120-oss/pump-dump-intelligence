import pandas as pd

from research.reports.html import render_report


def test_report_distinguishes_real_data_from_synthetic(tmp_path) -> None:
    output = render_report(
        {"model": {"average_precision": 0.2}},
        pd.DataFrame(),
        "correction_3_15m",
        ["return_1m"],
        tmp_path / "report.html",
        provenance={
            "source": "REAL_BINANCE_FUTURES",
            "symbols": ["TESTUSDT"],
            "rows": 10,
            "events": 2,
            "regime": "FAST",
        },
        split_summary=[],
    )
    text = output.read_text(encoding="utf-8")
    assert "Реальные исторические данные Binance" in text
    assert "высокий результат на синтетике" not in text.lower()
