"""Evidence-compatible market-behaviour hypotheses."""

from .causal import (
    CausalAnalysisBundle,
    CausalAssessment,
    CausalEngineConfig,
    CausalEvidenceEngine,
    EvidenceItem,
    HypothesisSpec,
    InvalidationRule,
)
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
    "CausalAnalysisBundle",
    "CausalAssessment",
    "CausalEngineConfig",
    "CausalEvidenceEngine",
    "EvidenceConfig",
    "EvidenceItem",
    "EvidenceReport",
    "HypothesisSpec",
    "InvalidationRule",
    "ManipulationEvidenceAnalyzer",
    "SpotFuturesEvidenceAnalyzer",
    "SpotFuturesEvidenceConfig",
    "SpotFuturesEvidenceReport",
]
