from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from amd_ai.image.profile import TorchProfile


TORCH_COMPONENTS = ("torch", "torchvision", "torchaudio", "triton")


class DependencyError(RuntimeError):
    pass


def render_torch_constraints(requirements_lock: str | Path) -> str:
    path = Path(requirements_lock)
    try:
        lock_text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DependencyError(f"cannot read Torch requirements lock: {error}") from error
    versions: dict[str, str] = {}
    for name in TORCH_COMPONENTS:
        match = re.search(rf"^{name}==([^\s\\]+)", lock_text, re.MULTILINE)
        if match is None:
            raise DependencyError(f"Torch requirements lock is missing {name}")
        versions[name] = match.group(1).split("+", 1)[0]
    return "".join(f"{name}=={versions[name]}\n" for name in TORCH_COMPONENTS)


def render_profile_constraints(profile: TorchProfile) -> str:
    return "".join(
        f"{name}=={profile.wheels[name].version.split('+', 1)[0]}\n"
        for name in TORCH_COMPONENTS
    )


def lock_project_dependencies(project_dir: Path) -> Path:
    project_dir = project_dir.resolve()
    requirements_input = project_dir / "requirements.in"
    constraints_path = project_dir / "torch-constraints.txt"
    output = project_dir / "requirements.lock"
    try:
        input_text = requirements_input.read_text(encoding="utf-8")
        constraints = constraints_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DependencyError(f"cannot read project dependency input: {error}") from error
    _constraint_versions(constraints)
    if not any(
        line.strip() and not line.lstrip().startswith("#")
        for line in input_text.splitlines()
    ):
        _write_text(output, "")
        return output

    temporary = project_dir / ".requirements.lock.tmp"
    temporary.unlink(missing_ok=True)
    argv = (
        "uv",
        "pip",
        "compile",
        "--python-version",
        "3.12",
        "--constraint",
        constraints_path.name,
        "--generate-hashes",
        "--output-file",
        temporary.name,
        requirements_input.name,
    )
    completed = subprocess.run(
        argv,
        check=False,
        cwd=project_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        evidence = completed.stderr.strip() or completed.stdout.strip()
        raise DependencyError(
            f"dependency lock failed ({completed.returncode}): {evidence}"
        )
    try:
        lock_text = temporary.read_text(encoding="utf-8")
        validate_project_lock(lock_text, constraints)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_project_lock(lock_text: str, constraints_text: str) -> None:
    expected = _constraint_versions(constraints_text)
    component_pattern = "|".join(re.escape(name) for name in TORCH_COMPONENTS)
    for line_number, line in enumerate(lock_text.splitlines(), start=1):
        stripped = line.strip()
        match = re.match(
            rf"^({component_pattern})(?:\[[^]]+\])?\s*(==|@)\s*([^\s\\]+)?",
            stripped,
            re.IGNORECASE,
        )
        if match is None:
            if re.match(rf"^({component_pattern})(?:\b|\[)", stripped, re.IGNORECASE):
                raise DependencyError(
                    f"protected Torch requirement is not exactly pinned on line {line_number}"
                )
            continue
        name = match.group(1).lower()
        operator = match.group(2)
        version = match.group(3) or ""
        if operator == "@":
            raise DependencyError(f"direct URL for protected package {name} is forbidden")
        if version.split("+", 1)[0] != expected[name]:
            raise DependencyError(
                f"project lock changes {name}: expected {expected[name]}, got {version}"
            )


def _constraint_versions(text: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = re.fullmatch(r"(torch|torchvision|torchaudio|triton)==([^\s]+)", stripped)
        if match is None:
            raise DependencyError(f"invalid Torch constraint on line {line_number}")
        name, version = match.groups()
        if name in versions:
            raise DependencyError(f"duplicate Torch constraint: {name}")
        versions[name] = version
    if tuple(versions) != TORCH_COMPONENTS:
        raise DependencyError("Torch constraints must pin all four components in order")
    return versions


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
