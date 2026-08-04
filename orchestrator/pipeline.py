from __future__ import annotations

import uuid
from pathlib import Path

from orchestrator.config import OrchestratorConfig
from orchestrator.constitution import Constitution
from orchestrator.db import OrchestratorDB
from orchestrator.developer import OpenAIDeveloper
from orchestrator.gitops import GitError, GitRepo
from orchestrator.models import ChangeStatus, TaskSpec
from orchestrator.patches import (
    PatchError,
    repair_missing_context_prefixes,
    validate_changed_paths,
    validate_patch_paths,
    write_patch,
)
from orchestrator.validation import run_validation


class PipelineError(RuntimeError):
    pass


def new_change_id() -> str:
    return f"CHANGE-{uuid.uuid4().hex[:8].upper()}"


class DevelopmentPipeline:
    def __init__(self, config: OrchestratorConfig, db: OrchestratorDB):
        self.config = config
        self.db = db
        self.repo = GitRepo(config.project_root)
        self.constitution = Constitution.load(config.project_root / "PROJECT_CONSTITUTION.yaml")

    def prepare_repository(self) -> None:
        self.repo.initialize(self.config.git_user_name, self.config.git_user_email)
        if not self.repo.run("rev-parse", "--verify", "HEAD", check=False).returncode == 0:
            self.repo.run("add", "-A")
            self.repo.run("commit", "-m", "chore: initialize controlled project baseline")
        self.db.set_setting("constitution_sha256", self.constitution.sha256)
        current = self.repo.current_branch()
        if current:
            self.db.set_setting("base_branch", current)

    def _latest_rejection_feedback(self, task_id: str) -> str | None:
        """Return the newest non-empty rejection reason for this task."""
        for row in self.db.list_changes(limit=10_000):
            if str(row.get("task_id") or "") != task_id:
                continue
            if str(row.get("status") or "") != ChangeStatus.REJECTED.value:
                continue

            reason = str(row.get("rejection_reason") or "").strip()
            if reason:
                return reason

        return None

    def create_proposal(self, task: TaskSpec) -> str:
        if not self.config.openai_api_key:
            raise PipelineError("OPENAI_API_KEY is required for autonomous code generation")
        self.repo.ensure_clean()
        expected_hash = self.db.get_setting("constitution_sha256")
        if expected_hash:
            self.constitution.verify_hash(expected_hash)

        change_id = new_change_id()
        self.db.create_change(change_id, task)
        self.db.update_change(change_id, status=ChangeStatus.GENERATING)
        base_branch = self.repo.current_branch()
        branch = self.repo.create_branch(f"orchestrator/{change_id.lower()}")
        keep_branch = False
        candidate_paths: list[str] = []

        try:
            constitution_text = self.constitution.path.read_text(encoding="utf-8")
            developer = OpenAIDeveloper(self.config.openai_api_key, self.config.openai_model)
            reviewer_feedback = self._latest_rejection_feedback(
                task.task_id
            )
            generated = developer.generate(
                self.config.project_root,
                task,
                constitution_text,
                reviewer_feedback=reviewer_feedback,
            )

            patch_path: Path | None = None
            for attempt in range(3):
                try:
                    candidate_diff = repair_missing_context_prefixes(
                        generated.diff,
                        self.config.project_root,
                    )
                    candidate_paths = validate_patch_paths(
                        candidate_diff,
                        task.allowed_paths,
                    )
                    self.constitution.validate_changed_paths(
                        candidate_paths,
                        critical_approved=False,
                    )
                    patch_path = write_patch(
                        self.config.state_dir,
                        change_id,
                        candidate_diff,
                    )
                    self.repo.apply_patch(patch_path)
                    break
                except (GitError, PatchError) as exc:
                    self.repo.cleanup_generated_paths(candidate_paths)
                    if attempt == 2:
                        raise PipelineError(
                            f"Generated patch remained invalid after 3 attempts: {exc}"
                        ) from exc
                    generated = developer.repair(
                        self.config.project_root,
                        task,
                        constitution_text,
                        generated.diff,
                        str(exc),
                    )
            else:
                raise PipelineError("No patch could be applied")

            actual_paths = validate_changed_paths(
                self.repo.changed_paths(),
                task.allowed_paths,
            )
            self.constitution.validate_changed_paths(
                actual_paths,
                critical_approved=False,
            )
            self.db.update_change(change_id, status=ChangeStatus.VALIDATING)
            validation = run_validation(
                self.config.project_root,
                self.config.state_dir,
                change_id,
                self.config.test_command,
            )
            diff_text = self.repo.diff()
            self.db.update_change(
                change_id,
                branch_name=branch,
                patch_path=str(patch_path) if patch_path else None,
                diff_text=diff_text,
                changed_paths_json=self.db.encode_paths(actual_paths),
                validation_summary=validation.summary,
                validation_log_path=str(validation.log_path) if validation.log_path else None,
                status=(ChangeStatus.PENDING_APPROVAL if validation.ok else ChangeStatus.FAILED),
            )
            if validation.ok:
                self.repo.commit_all(f"{change_id}: {generated.summary}")
                keep_branch = True
            else:
                self.repo.cleanup_generated_paths(actual_paths)
            return change_id
        except Exception:
            self.db.update_change(change_id, status=ChangeStatus.FAILED, branch_name=branch)
            self.repo.cleanup_generated_paths(candidate_paths)
            raise
        finally:
            if self.repo.current_branch() != base_branch:
                self.repo.switch(base_branch)
            if not keep_branch:
                self.repo.delete_branch(branch)

    def approve(self, change_id: str, actor_user_id: int | None) -> str:
        change = self.db.get_change(change_id)
        if not change:
            raise PipelineError(f"Unknown change: {change_id}")
        if change["status"] != ChangeStatus.PENDING_APPROVAL.value:
            raise PipelineError(f"Change is not pending approval: {change['status']}")
        branch = change.get("branch_name")
        if not branch:
            raise PipelineError("Change does not have a branch")
        self.repo.ensure_clean()
        base_branch = self.db.get_setting("base_branch") or self.repo.current_branch()
        self.repo.switch(base_branch)
        commit = self.repo.merge_no_ff(branch, f"Approve {change_id}: {change['title']}")
        self.db.record_decision(change_id, "APPROVE", actor_user_id)
        self.db.update_change(change_id, status=ChangeStatus.MERGED)
        return commit

    def reject(self, change_id: str, actor_user_id: int | None, reason: str) -> None:
        change = self.db.get_change(change_id)
        if not change:
            raise PipelineError(f"Unknown change: {change_id}")
        if change["status"] != ChangeStatus.PENDING_APPROVAL.value:
            raise PipelineError(f"Change is not pending approval: {change['status']}")
        self.db.record_decision(change_id, "REJECT", actor_user_id, reason)
        self.db.update_change(
            change_id,
            status=ChangeStatus.REJECTED,
            rejection_reason=reason,
        )

    def cleanup_proposal_files(self, change_id: str) -> None:
        change = self.db.get_change(change_id)
        if not change:
            return
        patch_path = change.get("patch_path")
        if patch_path:
            Path(patch_path).unlink(missing_ok=True)
        branch = change.get("branch_name")
        if branch:
            base_branch = self.db.get_setting("base_branch") or self.repo.current_branch()
            self.repo.switch(base_branch)
            self.repo.delete_branch(branch)
