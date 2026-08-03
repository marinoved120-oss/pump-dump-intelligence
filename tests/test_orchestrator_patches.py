import pytest

from orchestrator.patches import PatchError, extract_changed_paths, validate_patch_paths


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
