from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConstitutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Constitution:
    path: Path
    payload: dict
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "Constitution":
        if not path.exists():
            raise ConstitutionError(f"Constitution not found: {path}")
        raw = path.read_bytes()
        payload = yaml.safe_load(raw) or {}
        if not isinstance(payload, dict):
            raise ConstitutionError("PROJECT_CONSTITUTION.yaml must contain a mapping")
        return cls(path=path, payload=payload, sha256=hashlib.sha256(raw).hexdigest())

    @property
    def protected_paths(self) -> tuple[str, ...]:
        governance = self.payload.get("governance", {})
        values = governance.get("protected_paths", [])
        return tuple(str(v).replace("\\", "/") for v in values)

    @property
    def forbidden_shortcuts(self) -> tuple[str, ...]:
        values = self.payload.get("forbidden_shortcuts", [])
        return tuple(str(v) for v in values)

    def validate_changed_paths(self, changed_paths: list[str], *, critical_approved: bool = False) -> None:
        normalized = [p.replace("\\", "/").lstrip("./") for p in changed_paths]
        protected = {p.lstrip("./") for p in self.protected_paths}
        violations = sorted(path for path in normalized if path in protected)
        if violations and not critical_approved:
            raise ConstitutionError(
                "Protected project files require CRITICAL approval: " + ", ".join(violations)
            )

    def verify_hash(self, expected: str) -> None:
        if self.sha256 != expected:
            raise ConstitutionError("Project constitution hash changed unexpectedly")
