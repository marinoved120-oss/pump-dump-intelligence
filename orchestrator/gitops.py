from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _safe_branch(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", value).strip("-/.")
    if not cleaned:
        raise GitError("Invalid branch name")
    return cleaned


class GitRepo:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def run(self, *args: str, check: bool = True, timeout: int = 120) -> CommandResult:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
        if check and completed.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return result

    def is_repository(self) -> bool:
        return self.run("rev-parse", "--is-inside-work-tree", check=False).returncode == 0

    def initialize(self, user_name: str, user_email: str) -> None:
        if not self.is_repository():
            self.run("init")
        self.run("config", "user.name", user_name)
        self.run("config", "user.email", user_email)

    def current_branch(self) -> str:
        return self.run("branch", "--show-current").stdout.strip()

    def ensure_clean(self) -> None:
        if self.run("status", "--porcelain").stdout.strip():
            raise GitError("Repository has uncommitted changes")

    def create_branch(self, branch_name: str) -> str:
        safe = _safe_branch(branch_name)
        self.run("switch", "-c", safe)
        return safe

    def switch(self, branch_name: str) -> None:
        self.run("switch", _safe_branch(branch_name))

    def delete_branch(self, branch_name: str) -> None:
        self.run("branch", "-D", _safe_branch(branch_name), check=False)

    def changed_paths(self) -> list[str]:
        output = self.run("diff", "--name-only", "HEAD").stdout
        untracked = self.run("ls-files", "--others", "--exclude-standard").stdout
        return sorted({line.strip() for line in (output + "\n" + untracked).splitlines() if line.strip()})

    def diff(self) -> str:
        tracked = self.run("diff", "--binary", "HEAD").stdout
        untracked_paths = self.run("ls-files", "--others", "--exclude-standard").stdout.splitlines()
        parts = [tracked]
        for path in untracked_paths:
            result = self.run("diff", "--no-index", "--binary", "/dev/null", path, check=False)
            if result.returncode not in {0, 1}:
                raise GitError(result.stderr)
            parts.append(result.stdout)
        return "\n".join(parts).strip()

    def apply_patch(self, patch_path: Path) -> None:
        self.run("apply", "--check", str(patch_path))
        self.run("apply", "--whitespace=fix", str(patch_path))

    def cleanup_generated_paths(self, candidate_paths: list[str]) -> None:
        """Restore tracked files and delete only untracked files named by the proposal."""
        self.run("reset", "--hard", "HEAD", check=False)
        untracked = {
            line.strip().replace("\\", "/")
            for line in self.run("ls-files", "--others", "--exclude-standard", check=False).stdout.splitlines()
            if line.strip()
        }
        normalized_candidates = [p.replace("\\", "/").lstrip("./").rstrip("/") for p in candidate_paths]
        removable = sorted(
            path
            for path in untracked
            if any(path == candidate or path.startswith(candidate + "/") for candidate in normalized_candidates)
        )
        for relative in removable:
            target = (self.root / relative).resolve()
            try:
                target.relative_to(self.root)
            except ValueError:
                continue
            if target.is_file() or target.is_symlink():
                target.unlink(missing_ok=True)
        for relative in sorted(removable, key=lambda value: value.count("/"), reverse=True):
            parent = (self.root / relative).parent
            while parent != self.root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    def commit_all(self, message: str) -> str:
        self.run("add", "-A")
        self.run("commit", "-m", message)
        return self.run("rev-parse", "HEAD").stdout.strip()

    def merge_no_ff(self, branch_name: str, message: str) -> str:
        self.run("merge", "--no-ff", _safe_branch(branch_name), "-m", message)
        return self.run("rev-parse", "HEAD").stdout.strip()
