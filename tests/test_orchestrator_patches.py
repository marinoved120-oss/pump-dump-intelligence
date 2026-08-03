import pytest
from pathlib import Path

from orchestrator.patches import (
    PatchError,
    extract_changed_paths,
    repair_missing_context_prefixes,
    validate_patch_paths,
)


PATCH = """diff --git a/research/live/schema.py b/research/live/schema.py
new file mode 100644
--- /dev/null
+++ b/research/live/schema.py
@@ -0,0 +1 @@
+VALUE = 1
"""


def test_extract_changed_paths() -> None:
    assert extract_changed_paths(PATCH) == ["research/live/schema.py"]


def test_patch_paths_are_limited_to_task_scope() -> None:
    assert validate_patch_paths(PATCH, ("research/live",)) == ["research/live/schema.py"]


def test_patch_outside_scope_is_rejected() -> None:
    with pytest.raises(PatchError):
        validate_patch_paths(PATCH, ("research/evidence",))


def test_hidden_file_boundary_is_recovered_before_scope_validation() -> None:
    malformed = """diff --git a/research/live/schema.py b/research/live/schema.py
new file mode 100644
--- /dev/null
+++ b/research/live/schema.py
@@ -0,0 +1,2 @@
+VALUE = 1
+diff --git a/tests/test_live_collectors.py b/tests/test_live_collectors.py
new file mode 100644
--- /dev/null
+++ b/tests/test_live_collectors.py
@@ -0,0 +1,9 @@
+def test_value():
+    assert True
"""
    with pytest.raises(PatchError, match="outside task scope"):
        validate_patch_paths(malformed, ("research/live",))


def test_incorrect_hunk_counts_are_recounted() -> None:
    malformed = """diff --git a/research/live/schema.py b/research/live/schema.py
new file mode 100644
--- /dev/null
+++ b/research/live/schema.py
@@ -0,0 +1,99 @@
+VALUE = 1
"""
    assert validate_patch_paths(malformed, ("research/live",)) == [
        "research/live/schema.py"
    ]


def test_actual_changed_paths_are_limited_to_task_scope() -> None:
    from orchestrator.patches import validate_changed_paths

    with pytest.raises(PatchError, match="outside task scope"):
        validate_changed_paths(
            ["research/live/schema.py", "tests/test_live_collectors.py"],
            ("research/live",),
        )

def test_missing_context_prefixes_are_repaired_from_source(tmp_path: Path) -> None:
    target = tmp_path / "research/live/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        '"""Live recorder utilities."""\n'
        "\n"
        "from .schemas import DepthUpdate\n",
        encoding="utf-8",
    )

    malformed = (
        "diff --git a/research/live/__init__.py b/research/live/__init__.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/research/live/__init__.py\n"
        "+++ b/research/live/__init__.py\n"
        "@@ -1 +1 @@\n"
        '"""Live recorder utilities."""\n'
        "\n"
        "from .schemas import DepthUpdate\n"
        "+from .collector import BinanceCollector\n"
    )

    repaired = repair_missing_context_prefixes(malformed, tmp_path)

    assert '@@ -1,3 +1,4 @@' in repaired
    assert '\n """Live recorder utilities."""\n' in repaired
    assert "\n \n from .schemas import DepthUpdate\n" in repaired


def test_missing_context_prefix_repair_rejects_source_mismatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "research/live/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("original line\n", encoding="utf-8")

    malformed = (
        "diff --git a/research/live/__init__.py b/research/live/__init__.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/research/live/__init__.py\n"
        "+++ b/research/live/__init__.py\n"
        "@@ -1 +1 @@\n"
        "different line\n"
        "+added line\n"
    )

    with pytest.raises(PatchError, match="does not exactly match source"):
        repair_missing_context_prefixes(malformed, tmp_path)

def test_bare_blank_before_next_file_boundary_is_removed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "research/live/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_text("existing line\n", encoding="utf-8")

    malformed = (
        "diff --git a/research/live/__init__.py b/research/live/__init__.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/research/live/__init__.py\n"
        "+++ b/research/live/__init__.py\n"
        "@@ -1 +1,2 @@\n"
        "existing line\n"
        "+added line\n"
        "\n"
        "diff --git a/research/live/new.py b/research/live/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/research/live/new.py\n"
        "@@ -0,0 +1 @@\n"
        "+VALUE = 1\n"
    )

    repaired = repair_missing_context_prefixes(malformed, tmp_path)

    assert "+added line\ndiff --git a/research/live/new.py" in repaired
    assert "+added line\n\ndiff --git a/research/live/new.py" not in repaired

