from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    ProtectedProfile,
    canonicalize_protected_name,
)
from amd_ai.overlay.packaging_compat import (
    InvalidRequirement,
    InvalidWheelFilename,
    Requirement,
    Version,
    parse_wheel_filename,
)
from amd_ai.overlay.policy import PipPolicyError, parse_pip_request


GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_OPTIONS = frozenset(
    {
        "--index-url",
        "--extra-index-url",
        "--index",
        "--find-links",
        "--no-index",
        "--pre",
    }
)


class RequirementPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class InspectedRequirements:
    resolver_inputs: tuple[str, ...]
    external: tuple[str, ...]
    local_inputs: tuple[Path, ...]
    vcs_inputs: tuple[str, ...]


def inspect_requirements(
    requirements: tuple[str, ...],
    requirements_files: tuple[str, ...],
    *,
    project: Path,
    profile: ProtectedProfile,
) -> InspectedRequirements:
    project_root = project.resolve(strict=True)
    if not project_root.is_dir():
        raise RequirementPolicyError(
            f"project path is not a directory: {project_root}"
        )

    resolver_inputs: list[str] = []
    external: set[str] = set()
    local_inputs: list[Path] = []
    vcs_inputs: list[str] = []

    def inspect_one(value: str, base_dir: Path) -> None:
        stripped = value.strip()
        if not stripped:
            return
        local = _possible_local_path(stripped, base_dir)
        if local is not None:
            _record_local(
                local,
                project_root=project_root,
                local_inputs=local_inputs,
            )
            return

        try:
            requirement = Requirement(stripped)
        except InvalidRequirement as error:
            raise RequirementPolicyError(
                f"invalid requirement: {stripped}"
            ) from error

        protected = canonicalize_protected_name(requirement.name)
        is_protected = protected in PROTECTED_DISTRIBUTIONS
        if requirement.marker is not None and not requirement.marker.evaluate():
            return
        if requirement.url is not None:
            if is_protected:
                raise RequirementPolicyError(
                    f"direct source for protected distribution is forbidden: {protected}"
                )
            if requirement.url.lower().startswith("git+"):
                _validate_git_requirement(stripped, requirement.url)
                vcs_inputs.append(stripped)
                return
            parsed = urlsplit(requirement.url)
            if parsed.scheme == "file":
                local_path = Path(unquote(parsed.path)).resolve(strict=True)
                _record_local(
                    local_path,
                    project_root=project_root,
                    local_inputs=local_inputs,
                )
                return
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or not parsed.path.lower().endswith(".whl")
            ):
                raise RequirementPolicyError(
                    "direct requirements must be credential-free HTTPS wheels"
                )
            resolver_inputs.append(stripped)
            return

        if is_protected:
            expected = Version(profile.version_for(protected))
            if requirement.specifier and not requirement.specifier.contains(
                expected, prereleases=True
            ):
                raise RequirementPolicyError(
                    f"protected requirement conflicts with verified parent: "
                    f"{protected}{requirement.specifier} does not accept {expected}"
                )
            external.add(protected)
            return
        resolver_inputs.append(stripped)

    active_files: set[Path] = set()

    def inspect_file(path_text: str, base_dir: Path) -> None:
        path = _resolve_inside_project(path_text, base_dir, project_root)
        if not path.is_file():
            raise RequirementPolicyError(
                f"requirements path is not a file: {path}"
            )
        if path in active_files:
            raise RequirementPolicyError(f"requirements include cycle: {path}")
        active_files.add(path)
        try:
            for line in _logical_lines(path):
                tokens = _split_option(line)
                if tokens and tokens[0] in {"-r", "--requirement"}:
                    if len(tokens) != 2:
                        raise RequirementPolicyError(
                            f"invalid requirements include in {path}: {line}"
                        )
                    inspect_file(tokens[1], path.parent)
                    continue
                if line.startswith("--requirement="):
                    inspect_file(line.partition("=")[2], path.parent)
                    continue
                if tokens and tokens[0].partition("=")[0] in SOURCE_OPTIONS:
                    _record_source_option(tokens, resolver_inputs)
                    continue
                if line.startswith("-"):
                    raise RequirementPolicyError(
                        f"unsupported requirements option in {path}: {line}"
                    )
                inspect_one(line, path.parent)
        finally:
            active_files.remove(path)

    for requirement in requirements:
        inspect_one(requirement, project_root)
    for requirements_file in requirements_files:
        inspect_file(requirements_file, project_root)

    return InspectedRequirements(
        resolver_inputs=tuple(resolver_inputs),
        external=tuple(sorted(external)),
        local_inputs=tuple(local_inputs),
        vcs_inputs=tuple(vcs_inputs),
    )


def _logical_lines(path: Path) -> tuple[str, ...]:
    try:
        physical = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RequirementPolicyError(
            f"cannot read requirements file {path}: {error}"
        ) from error
    logical: list[str] = []
    pending = ""
    for raw_line in physical:
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        logical.append((pending + line).strip())
        pending = ""
    if pending:
        raise RequirementPolicyError(
            f"unterminated line continuation in requirements file: {path}"
        )
    return tuple(logical)


def _strip_comment(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def _split_option(line: str) -> list[str]:
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError as error:
        raise RequirementPolicyError(
            f"invalid requirements option quoting: {line}"
        ) from error


def _record_source_option(tokens: list[str], output: list[str]) -> None:
    try:
        request = parse_pip_request(["install", *tokens, "overlay-placeholder"])
    except PipPolicyError as error:
        raise RequirementPolicyError(str(error)) from error
    if request.resolver_options:
        option = " ".join(shlex.quote(value) for value in request.resolver_options)
        output.append(option)


def _possible_local_path(value: str, base_dir: Path) -> Path | None:
    if value.startswith(("./", "../", "/")):
        return _resolve_existing_path(value, base_dir)
    candidate = base_dir / value
    if candidate.exists() or candidate.is_symlink():
        return candidate.resolve(strict=True)
    return None


def _resolve_existing_path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as error:
        raise RequirementPolicyError(
            f"local requirement does not exist: {candidate}"
        ) from error


def _resolve_inside_project(
    value: str, base_dir: Path, project_root: Path
) -> Path:
    path = _resolve_existing_path(value, base_dir)
    if not path.is_relative_to(project_root):
        raise RequirementPolicyError(
            f"requirements path is outside project: {path}"
        )
    return path


def _record_local(
    path: Path,
    *,
    project_root: Path,
    local_inputs: list[Path],
) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(project_root):
        raise RequirementPolicyError(
            f"local requirement is outside project: {resolved}"
        )
    if resolved.is_dir():
        local_inputs.append(resolved)
        return
    if not resolved.is_file() or resolved.suffix.lower() != ".whl":
        raise RequirementPolicyError(
            f"local requirement must be a source directory or wheel: {resolved}"
        )
    try:
        name, _, _, _ = parse_wheel_filename(resolved.name)
    except InvalidWheelFilename as error:
        raise RequirementPolicyError(
            f"invalid local wheel filename: {resolved.name}"
        ) from error
    protected = canonicalize_protected_name(str(name))
    if protected in PROTECTED_DISTRIBUTIONS:
        raise RequirementPolicyError(
            f"local wheel is a protected distribution: {protected}"
        )
    local_inputs.append(resolved)


def _validate_git_requirement(requirement: str, url: str) -> None:
    parsed = urlsplit(url.removeprefix("git+"))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
    ):
        raise RequirementPolicyError(
            "Git requirements must use credential-free HTTPS"
        )
    _, separator, commit = parsed.path.rpartition("@")
    if not separator or GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise RequirementPolicyError(
            f"Git requirement must use an exact commit: {requirement}"
        )
