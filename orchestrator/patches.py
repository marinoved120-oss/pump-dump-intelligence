from __future__ import annotations

import re
import textwrap
from pathlib import Path


class PatchError(RuntimeError):
    pass


_DIFF_PATH = re.compile(r"^(?:\+\+\+|---)\s+(?:a/|b/)?([^\t\n]+)", re.MULTILINE)
_DIFF_GIT_PATH = re.compile(r"^diff --git\s+a/(.+?)\s+b/(.+?)$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)
_FENCE = re.compile(r"^```(?:diff|patch|text)?\s*$", re.IGNORECASE)


def normalize_unified_diff(diff_text: str) -> str:
    """Remove common model wrappers without changing patch semantics."""
    value = str(diff_text or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    value = textwrap.dedent(value).strip()
    if not value:
        return ""

    lines = value.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if lines and _FENCE.match(lines[0].strip()):
        lines.pop(0)
    while lines and _FENCE.match(lines[-1].strip()):
        lines.pop()

    start = None
    for index, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- a/") or line == "--- /dev/null":
            start = index
            break
    if start is None:
        return "\n".join(lines).strip()

    lines = lines[start:]
    while lines and _FENCE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def validate_patch_structure(diff_text: str) -> str:
    normalized = normalize_unified_diff(diff_text)
    if not normalized:
        raise PatchError("Generated response contains an empty patch")
    if not (normalized.startswith("diff --git ") or normalized.startswith("--- ")):
        raise PatchError("Patch must start with a git or unified-diff file header")
    if "--- " not in normalized or "+++ " not in normalized:
        raise PatchError("Patch is missing ---/+++ file headers")
    if not _HUNK_HEADER.search(normalized) and "GIT binary patch" not in normalized:
        raise PatchError("Patch has file headers but no valid @@ hunk")
    return normalized


def extract_changed_paths(diff_text: str) -> list[str]:
    normalized = normalize_unified_diff(diff_text)
    paths: set[str] = set()
    for match in _DIFF_PATH.finditer(normalized):
        value = match.group(1).strip()
        if value == "/dev/null":
            continue
        paths.add(value.replace("\\", "/").lstrip("./"))
    for match in _DIFF_GIT_PATH.finditer(normalized):
        for value in match.groups():
            if value != "/dev/null":
                paths.add(value.replace("\\", "/").lstrip("./"))
    return sorted(paths)


def validate_patch_paths(diff_text: str, allowed_paths: tuple[str, ...]) -> list[str]:
    normalized = validate_patch_structure(diff_text)
    paths = extract_changed_paths(normalized)
    if not paths:
        raise PatchError("Generated response does not contain file paths")
    allowed = [p.replace("\\", "/").rstrip("/") for p in allowed_paths]
    rejected = [
        path
        for path in paths
        if not any(path == prefix or path.startswith(prefix + "/") for prefix in allowed)
    ]
    if rejected:
        raise PatchError("Patch touches paths outside task scope: " + ", ".join(rejected))
    return paths


def write_patch(state_dir: Path, change_id: str, diff_text: str) -> Path:
    normalized = validate_patch_structure(diff_text)
    path = state_dir / "proposals" / f"{change_id}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")
    return path
