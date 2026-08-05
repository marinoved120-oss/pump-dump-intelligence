from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_PATTERNS = (
    ("create_order", re.compile(r"\bcreate_order\b", re.IGNORECASE)),
    ("place_order", re.compile(r"\bplace_order\b", re.IGNORECASE)),
    ("send_order", re.compile(r"\bsend_order\b", re.IGNORECASE)),
    ("cancel_order", re.compile(r"\bcancel_order\b", re.IGNORECASE)),
    ("withdraw", re.compile(r"\bwithdraw(?:al|als)?\b", re.IGNORECASE)),
    (
        "private order or withdrawal endpoint",
        re.compile(
            r"/(?:api|fapi|sapi)/v\d+/[^\s\"']*(?:order|withdraw)",
            re.IGNORECASE,
        ),
    ),
)


def run(
    arguments: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = shlex.join(str(argument) for argument in arguments)
    print(f"$ {printable}")

    subprocess.run(
        [str(argument) for argument in arguments],
        cwd=cwd,
        env=env,
        check=True,
    )


def verify_read_only_research() -> None:
    research_root = ROOT / "research"
    matches: list[str] = []

    for path in sorted(research_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")

        for line_number, line in enumerate(text.splitlines(), start=1):
            for description, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    relative = path.relative_to(ROOT)
                    matches.append(
                        f"{relative}:{line_number}: "
                        f"{description}: {line.strip()}"
                    )

    if matches:
        print("FORBIDDEN TRADING CAPABILITY FOUND", file=sys.stderr)
        for match in matches:
            print(match, file=sys.stderr)
        raise SystemExit(1)

    print("TRADING ENDPOINT SCAN: OK")


def prepare_build_source(source_directory: Path) -> None:
    source_directory.mkdir(parents=True)

    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(
            ROOT / filename,
            source_directory / filename,
        )

    for package in ("research", "orchestrator"):
        shutil.copytree(
            ROOT / package,
            source_directory / package,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
            ),
        )


def build_wheel(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)

    source_directory = output_directory.parent / "source"
    prepare_build_source(source_directory)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(output_directory),
        ],
        cwd=source_directory,
    )

    wheels = sorted(output_directory.glob("*.whl"))

    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected exactly one wheel, found {len(wheels)}: {wheels}"
        )

    wheel = wheels[0]
    print(f"WHEEL BUILD: OK ({wheel.name})")
    return wheel


def inspect_wheel(wheel: Path) -> None:
    expected_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    required_members = {
        "research/config.py",
        "research/default_research.yaml",
        "orchestrator/cli.py",
    }

    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

        missing = sorted(required_members - members)
        if missing:
            raise RuntimeError(
                "Wheel is missing required members: " + ", ".join(missing)
            )

        metadata_names = sorted(
            name
            for name in members
            if name.endswith(".dist-info/METADATA")
        )
        entry_point_names = sorted(
            name
            for name in members
            if name.endswith(".dist-info/entry_points.txt")
        )

        if len(metadata_names) != 1:
            raise RuntimeError(
                f"Expected one METADATA file, found {metadata_names}"
            )

        if len(entry_point_names) != 1:
            raise RuntimeError(
                f"Expected one entry_points.txt file, found {entry_point_names}"
            )

        metadata = archive.read(metadata_names[0]).decode("utf-8")
        entry_points = archive.read(entry_point_names[0]).decode("utf-8")

    if f"Version: {expected_version}" not in metadata:
        raise RuntimeError(
            f"Wheel metadata does not contain version {expected_version}"
        )

    if "pd-research = research.cli:app" not in entry_points:
        raise RuntimeError("pd-research console entry point is missing")

    if "pd-orchestrator = orchestrator.cli:app" not in entry_points:
        raise RuntimeError("pd-orchestrator console entry point is missing")

    print("WHEEL CONTENTS: OK")
    print(f"WHEEL VERSION: {expected_version}")


def verify_installed_config_fallback(wheel: Path, temporary_root: Path) -> None:
    target = temporary_root / "site"
    empty_working_directory = temporary_root / "empty"
    empty_working_directory.mkdir(parents=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ]
    )

    environment = os.environ.copy()
    previous_pythonpath = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = (
        str(target)
        if not previous_pythonpath
        else str(target) + os.pathsep + previous_pythonpath
    )
    environment["CI_EXPECTED_SITE"] = str(target)

    verification_code = """
import os
from pathlib import Path

import research
from research.config import load_config

expected_site = Path(os.environ["CI_EXPECTED_SITE"]).resolve()
module_path = Path(research.__file__).resolve()

if expected_site not in module_path.parents:
    raise SystemExit(
        f"research imported from unexpected location: {module_path}"
    )

config = load_config()

if not config.binance.base_url:
    raise SystemExit("Packaged config has no Binance base URL")

if not config.labels.forward_horizons:
    raise SystemExit("Packaged config has no forward horizons")

print(f"INSTALLED MODULE: {module_path}")
print("INSTALLED WHEEL CONFIG FALLBACK: OK")
"""

    run(
        [sys.executable, "-c", verification_code],
        cwd=empty_working_directory,
        env=environment,
    )


def main() -> None:
    verify_read_only_research()

    with tempfile.TemporaryDirectory(
        prefix="pump-dump-ci-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        wheel = build_wheel(temporary_root / "dist")
        inspect_wheel(wheel)
        verify_installed_config_fallback(wheel, temporary_root)

    print("RELEASE INTEGRITY: OK")


if __name__ == "__main__":
    main()
