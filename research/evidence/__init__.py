"""Evidence-compatible market-behaviour hypotheses."""

from .manipulation import (
    AbsorptionWindow,
    EvidenceConfig,
    EvidenceReport,
    ManipulationEvidenceAnalyzer,
)
from .spot_futures import (
    SpotFuturesEvidenceAnalyzer,
    SpotFuturesEvidenceConfig,
    SpotFuturesEvidenceReport,
)

__all__ = [
    "AbsorptionWindow",
    "EvidenceConfig",
    "EvidenceReport",
    "ManipulationEvidenceAnalyzer",
    "SpotFuturesEvidenceAnalyzer",
    "SpotFuturesEvidenceConfig",
    "SpotFuturesEvidenceReport",
]
