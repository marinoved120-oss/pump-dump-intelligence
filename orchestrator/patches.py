from __future__ import annotations

import re
import textwrap
from pathlib import Path


class PatchError(RuntimeError):
    pass


_DIFF_PATH = re.compile(r"^(?:\+\+\+|---)\s+(?:a/|b/)?([^\t\n]+)", re.MULTILINE)
_DIFF_GIT_PATH = re.compile(r"^diff --git\s+a/(.+?)\s+b/(.+?)$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@", re.MULTILINE)
_HUNK_HEADER_FULL = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$"
)
_FENCE = re.compile(r"^```(?:diff|patch|text)?\s*$", re.IGNORECASE)
_FILE_METADATA_PREFIXES = (
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
)


def _looks_like_file_boundary(lines: list[str], index: int) -> bool:
    line = lines[index]
    if not line.startswith("+diff --git a/") or " b/" not in line:
        return False
    lookahead = lines[index + 1 : index + 6]
    return any(
        candidate.startswith(_FILE_METADATA_PREFIXES)
        or (
            candidate.startswith("+")
            and candidate[1:].startswith(_FILE_METADATA_PREFIXES)
        )
        for candidate in lookahead
    )


def _repair_prefixed_file_boundaries(lines: list[str]) -> list[str]:
    """Recover file headers accidentally emitted as added hunk lines by a model."""
    repaired = list(lines)
    for index, line in enumerate(repaired):
        if not _looks_like_file_boundary(repaired, index):
            continue
        repaired[index] = line[1:]
        cursor = index + 1
        while cursor < len(repaired):
            candidate = repaired[cursor]
            if candidate.startswith("diff --git "):
                break
            if candidate.startswith("+") and candidate[1:].startswith(_FILE_METADATA_PREFIXES):
                repaired[cursor] = candidate[1:]
                candidate = repaired[cursor]
            if candidate.startswith("@@ "):
                break
            if not candidate.startswith(_FILE_METADATA_PREFIXES):
                break
            cursor += 1
    return repaired


def _recount_hunks(lines: list[str]) -> list[str]:
    """Recompute hunk lengths so recovered file boundaries cannot remain hidden in a hunk."""
    recounted = list(lines)
    index = 0
    while index < len(recounted):
        match = _HUNK_HEADER_FULL.match(recounted[index])
        if not match:
            index += 1
            continue

        old_count = 0
        new_count = 0
        cursor = index + 1
        while (
            cursor < len(recounted)
            and not recounted[cursor].startswith("@@ ")
            and not recounted[cursor].startswith("diff --git ")
        ):
            line = recounted[cursor]
            if line.startswith("\\ No newline at end of file"):
                pass
            elif line.startswith("+"):
                new_count += 1
            elif line.startswith("-"):
                old_count += 1
            elif line.startswith(" "):
                old_count += 1
                new_count += 1
            else:
                break
            cursor += 1

        old_start = int(match.group(1))
        new_start = int(match.group(3))
        suffix = match.group(5)
        old_range = str(old_start) if old_count == 1 else f"{old_start},{old_count}"
        new_range = str(new_start) if new_count == 1 else f"{new_start},{new_count}"
        recounted[index] = f"@@ -{old_range} +{new_range} @@{suffix}"
        index = max(cursor, index + 1)
    return recounted


def normalize_unified_diff(diff_text: str) -> str:
    """Remove common model wrappers and repair deterministic patch formatting defects."""
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
    lines = _repair_prefixed_file_boundaries(lines)
    lines = _recount_hunks(lines)
    return "\n".join(lines).rstrip() + "\n"


def validate_patch_structure(diff_text: str) -> str:
    normalized = normalize_unified_diff(diff_text)
    if not normalized:
        raise PatchError("Generated response contains an empty patch")
    if not (normalized.startswith("diff --git ") or normalized.startswith("--- ")):
        raise PatchError("Patch must start with a git or unified-diff file header")
    if "\n+diff --git " in normalized:
        raise PatchError("Patch contains a file header hidden inside an added hunk line")
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


def validate_changed_paths(paths: list[str], allowed_paths: tuple[str, ...]) -> list[str]:
    normalized_paths = sorted(
        {
            path.replace("\\", "/").lstrip("./").rstrip("/")
            for path in paths
            if path.strip()
        }
    )
    if not normalized_paths:
        raise PatchError("Generated response does not contain file paths")
    unsafe = [
        path
        for path in normalized_paths
        if path.startswith("/") or path == ".." or "/../" in f"/{path}/"
    ]
    if unsafe:
        raise PatchError("Patch contains unsafe paths: " + ", ".join(unsafe))
    allowed = [p.replace("\\", "/").lstrip("./").rstrip("/") for p in allowed_paths]
    rejected = [
        path
        for path in normalized_paths
        if not any(path == prefix or path.startswith(prefix + "/") for prefix in allowed)
    ]
    if rejected:
        raise PatchError("Patch touches paths outside task scope: " + ", ".join(rejected))
    return normalized_paths


def validate_patch_paths(diff_text: str, allowed_paths: tuple[str, ...]) -> list[str]:
    normalized = validate_patch_structure(diff_text)
    return validate_changed_paths(extract_changed_paths(normalized), allowed_paths)


def write_patch(state_dir: Path, change_id: str, diff_text: str) -> Path:
    normalized = validate_patch_structure(diff_text)
    path = state_dir / "proposals" / f"{change_id}.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")
    return path
