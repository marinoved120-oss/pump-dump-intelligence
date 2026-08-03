from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .schemas import DepthUpdate


@dataclass(frozen=True)
class SequenceState:
    last_sequence: Optional[int] = None
    expected_next: Optional[int] = None
    in_sync: bool = False
    gap_detected: bool = False
    applied_count: int = 0
    duplicate_count: int = 0
    gap_count: int = 0
    out_of_order_count: int = 0
    last_exchange_ts: Optional[int] = None
    data_quality_score: float = 1.0
    issues: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    reason: Optional[str]
    state: SequenceState


class SequenceApplier:
    """Manages sequence-integrity for order book depth updates.

    Enforces that incrementals cannot cross a detected sequence gap, and that a
    snapshot is required to resynchronize. Maintains a conservative data-quality
    score that is lowered by gaps and staleness.
    """

    def __init__(
        self,
        require_snapshot_first: bool = True,
        staleness_warn_ms: int = 5_000,
        staleness_critical_ms: int = 15_000,
    ) -> None:
        self._state = SequenceState()
        self._require_snapshot_first = require_snapshot_first
        self._staleness_warn_ms = staleness_warn_ms
        self._staleness_critical_ms = staleness_critical_ms

    @property
    def state(self) -> SequenceState:
        return self._state

    def _with(self, **kwargs) -> SequenceState:
        data = self._state.__dict__.copy()
        data.update(kwargs)
        if isinstance(data.get("issues"), list):
            data["issues"] = tuple(data["issues"])  # type: ignore[index]
        return SequenceState(**data)

    def _bump_issue(self, issue: str) -> None:
        issues = list(self._state.issues)
        issues.append(issue)
        self._state = self._with(issues=tuple(issues))

    def _set_quality_for_staleness(self, now_ms: Optional[int]) -> None:
        if now_ms is None or self._state.last_exchange_ts is None:
            return
        delay = max(0, now_ms - self._state.last_exchange_ts)
        score = self._state.data_quality_score
        if delay >= self._staleness_critical_ms:
            score = min(score, 0.3)
            self._bump_issue("stale_critical")
        elif delay >= self._staleness_warn_ms:
            score = min(score, 0.7)
            self._bump_issue("stale_warn")
        self._state = self._with(data_quality_score=score)

    def _set_quality_for_gap(self) -> None:
        # Gap forces out-of-sync and worst quality until resynced
        self._state = self._with(in_sync=False, gap_detected=True, data_quality_score=0.0)

    def apply_depth(self, update: DepthUpdate, now_ms: Optional[int] = None) -> ApplyResult:
        # Update staleness quality relative to previous exchange_ts (if any)
        self._set_quality_for_staleness(now_ms)

        # Initial state requirements
        if self._state.last_sequence is None:
            if self._require_snapshot_first and not update.is_snapshot:
                self._bump_issue("requires_snapshot")
                return ApplyResult(False, "requires_snapshot", self._state)
            # Accept first snapshot
            if update.is_snapshot:
                self._state = self._with(
                    last_sequence=update.sequence,
                    expected_next=update.sequence + 1,
                    in_sync=True,
                    gap_detected=False,
                    applied_count=self._state.applied_count + 1,
                    last_exchange_ts=update.exchange_ts,
                    data_quality_score=1.0,
                )
                return ApplyResult(True, None, self._state)

        # If out of sync, only allow a snapshot to resynchronize
        if not self._state.in_sync:
            if update.is_snapshot:
                self._state = self._with(
                    last_sequence=update.sequence,
                    expected_next=update.sequence + 1,
                    in_sync=True,
                    gap_detected=False,
                    applied_count=self._state.applied_count + 1,
                    last_exchange_ts=update.exchange_ts,
                    data_quality_score=1.0,
                    issues=tuple(i for i in self._state.issues if i != "sequence_gap"),
                )
                return ApplyResult(True, None, self._state)
            self._bump_issue("out_of_sync")
            return ApplyResult(False, "out_of_sync", self._state)

        # At this point we have a last_sequence and are in sync
        assert self._state.last_sequence is not None
        last = self._state.last_sequence

        # Duplicates (including re-sent snapshot with same sequence)
        if update.sequence == last:
            self._state = self._with(
                duplicate_count=self._state.duplicate_count + 1,
                last_exchange_ts=max(self._state.last_exchange_ts or 0, update.exchange_ts),
            )
            self._bump_issue("duplicate")
            return ApplyResult(False, "duplicate", self._state)

        # Ordered next incremental
        if update.sequence == last + 1:
            self._state = self._with(
                last_sequence=update.sequence,
                expected_next=update.sequence + 1,
                in_sync=True,
                applied_count=self._state.applied_count + 1,
                last_exchange_ts=update.exchange_ts,
            )
            return ApplyResult(True, None, self._state)

        # Out-of-order older packet
        if update.sequence < last:
            self._state = self._with(
                out_of_order_count=self._state.out_of_order_count + 1,
                last_exchange_ts=max(self._state.last_exchange_ts or 0, update.exchange_ts),
            )
            self._bump_issue("stale_sequence")
            return ApplyResult(False, "stale_sequence", self._state)

        # Gap: sequence advanced by >1
        if update.sequence > last + 1:
            self._state = self._with(
                gap_count=self._state.gap_count + 1,
                expected_next=last + 1,
            )
            self._bump_issue("sequence_gap")
            self._set_quality_for_gap()
            return ApplyResult(False, "gap", self._state)

        # Fallback (should not reach)
        return ApplyResult(False, "unknown", self._state)

    def mark_missing_source(self) -> None:
        """Mark source as missing to reduce data quality without applying updates."""
        self._bump_issue("missing_source")
        self._state = self._with(data_quality_score=min(self._state.data_quality_score, 0.5))

    def recompute_quality(self, now_ms: Optional[int]) -> SequenceState:
        """Recompute data-quality score from staleness and sync status.

        If out of sync, score is forced to 0.0. Otherwise staleness thresholds
        downgrade to 0.7 (warn) or 0.3 (critical); pristine in-sync state restores to 1.0.
        """
        if not self._state.in_sync:
            return self._with(data_quality_score=0.0)
        # start from perfect, then apply staleness
        self._state = self._with(data_quality_score=1.0)
        self._set_quality_for_staleness(now_ms)
        return self._state

    def reset(self) -> None:
        """Hard reset of the state (e.g., on manual intervention)."""
        self._state = SequenceState()

__all__ = ["SequenceApplier", "SequenceState", "ApplyResult"]
