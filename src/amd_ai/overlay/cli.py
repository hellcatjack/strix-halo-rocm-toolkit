from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from amd_ai.overlay.lock import LockError, render_lock
from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    OverlayError,
    OverlayPaths,
    ProtectedProfile,
    canonicalize_name,
    canonicalize_protected_name,
)
from amd_ai.overlay.packaging_compat import (
    InvalidRequirement,
    InvalidWheelFilename,
    Requirement,
    canonicalize_name as packaging_canonicalize_name,
    parse_wheel_filename,
)
from amd_ai.overlay.policy import (
    QUERY_COMMANDS,
    PipPolicyError,
    PipRequest,
    parse_pip_request,
)
from amd_ai.overlay.requirements import (
    InspectedRequirements,
    RequirementPolicyError,
    inspect_requirements,
)
from amd_ai.overlay.resolver import (
    ProcessRunner,
    ResolverError,
    SubprocessProcessRunner,
    WheelArtifact,
    resolve_and_materialize,
)
from amd_ai.overlay.transaction import (
    TransactionError,
    build_generation,
    initialize_overlay,
    new_transaction_id,
    overlay_transaction,
)
from amd_ai.overlay.verify import (
    OverlayVerificationError,
    VerifiedGeneration,
    load_protected_profile,
    verify_base_manifest,
    verify_candidate_overlay,
    verify_current_generation,
)
from amd_ai.runner import CommandResult


URL_USERINFO_PATTERN = re.compile(r"(?P<scheme>https?://)[^\s/@]+@")
SENSITIVE_ENV_PARTS = ("TOKEN", "SECRET", "PASSWORD", "KEY")


class LoggedProcessRunner:
    def __init__(self, delegate: ProcessRunner, log_path: Path) -> None:
        self._delegate = delegate
        self._log_path = log_path

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        result = self._delegate.run(args, environment=environment, cwd=cwd)
        secret_values = tuple(
            value
            for name, value in environment.items()
            if value
            and any(part in name.upper() for part in SENSITIVE_ENV_PARTS)
        )

        def redact(value: str) -> str:
            return _redact_text(value, secret_values=secret_values)

        lines = [
            "command="
            + json.dumps(
                [redact(value) for value in args], ensure_ascii=True
            ),
            f"cwd={redact(str(cwd)) if cwd is not None else ''}",
        ]
        for name, value in sorted(environment.items()):
            rendered = (
                "<redacted>"
                if any(part in name.upper() for part in SENSITIVE_ENV_PARTS)
                else redact(value)
            )
            lines.append(f"env {name}={rendered}")
        lines.extend(
            (
                f"returncode={result.returncode}",
                "stdout=" + redact(result.stdout),
                "stderr=" + redact(result.stderr),
                "",
            )
        )
        _append_private_log(self._log_path, "\n".join(lines))
        return result


def main(
    argv: list[str] | None = None,
    *,
    project: Path = Path("/workspace"),
    runner: ProcessRunner | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        request = parse_pip_request(arguments)
        project_root = _require_managed_project(project)
        paths = OverlayPaths.for_project(project_root)
        base_environment = dict(
            os.environ if environment is None else environment
        )
        profile = load_protected_profile(environment=base_environment)
        if request.command == "uninstall":
            _reject_protected_uninstall(request)

        transaction_id = new_transaction_id()
        process_runner = LoggedProcessRunner(
            runner or SubprocessProcessRunner(),
            paths.logs / f"{transaction_id}.log",
        )
        with overlay_transaction(paths):
            verify_base_manifest(
                runner=process_runner,
                base_environment=base_environment,
            )
            initialize_overlay(
                paths,
                profile=profile,
                acquire_lock=False,
            )
            current = verify_current_generation(
                paths,
                profile=profile,
                runner=process_runner,
                base_environment=base_environment,
            )
            if request.command in QUERY_COMMANDS:
                return _run_query(
                    request,
                    paths=paths,
                    runner=process_runner,
                    base_environment=base_environment,
                )
            if request.command == "install":
                return _run_install(
                    request,
                    paths=paths,
                    profile=profile,
                    current=current,
                    runner=process_runner,
                    base_environment=base_environment,
                    transaction_id=transaction_id,
                )
            return _run_uninstall(
                request,
                paths=paths,
                profile=profile,
                current=current,
                runner=process_runner,
                base_environment=base_environment,
                transaction_id=transaction_id,
            )
    except KeyboardInterrupt:
        print("protected pip: interrupted", file=sys.stderr)
        return 1
    except (
        InvalidRequirement,
        LockError,
        OSError,
        OverlayError,
        OverlayVerificationError,
        PipPolicyError,
        RequirementPolicyError,
        ResolverError,
        TransactionError,
        ValueError,
    ) as error:
        print(f"protected pip: {error}", file=sys.stderr)
        return 2


def _run_install(
    request: PipRequest,
    *,
    paths: OverlayPaths,
    profile: ProtectedProfile,
    current: VerifiedGeneration,
    runner: ProcessRunner,
    base_environment: Mapping[str, str],
    transaction_id: str,
) -> int:
    requested = inspect_requirements(
        request.requirements,
        request.requirements_files,
        project=paths.project,
        profile=profile,
    )
    if requested.external:
        print(
            f"{', '.join(requested.external)}: "
            "already satisfied by verified parent"
        )
    if not _has_overlay_inputs(requested):
        return 0

    current_roots = _input_lines(current.input_text)
    replaced_names = _known_requirement_names(requested)
    retained_roots = tuple(
        line
        for line in current_roots
        if _requirement_name(line) not in replaced_names
    )
    retained = inspect_requirements(
        retained_roots,
        (),
        project=paths.project,
        profile=profile,
    )
    combined = _combine_inspected(retained, requested)
    _resolve_and_build(
        combined,
        resolver_options=request.resolver_options,
        paths=paths,
        profile=profile,
        runner=runner,
        base_environment=base_environment,
        transaction_id=transaction_id,
    )
    return 0


def _run_uninstall(
    request: PipRequest,
    *,
    paths: OverlayPaths,
    profile: ProtectedProfile,
    current: VerifiedGeneration,
    runner: ProcessRunner,
    base_environment: Mapping[str, str],
    transaction_id: str,
) -> int:
    requested_names = {
        canonicalize_name(name) for name in request.names
    }
    roots = _input_lines(current.input_text)
    by_name = {_requirement_name(line): line for line in roots}
    missing = sorted(requested_names.difference(by_name))
    if missing:
        raise PipPolicyError(
            "pip uninstall only accepts top-level overlay packages; missing: "
            + ", ".join(missing)
        )
    if not request.assume_yes:
        answer = input(
            "Remove " + ", ".join(sorted(requested_names)) + "? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("protected pip: uninstall cancelled", file=sys.stderr)
            return 1
    retained_roots = tuple(
        line for name, line in by_name.items() if name not in requested_names
    )
    retained = inspect_requirements(
        retained_roots,
        (),
        project=paths.project,
        profile=profile,
    )
    _resolve_and_build(
        retained,
        resolver_options=(),
        paths=paths,
        profile=profile,
        runner=runner,
        base_environment=base_environment,
        transaction_id=transaction_id,
    )
    return 0


def _resolve_and_build(
    inspected: InspectedRequirements,
    *,
    resolver_options: tuple[str, ...],
    paths: OverlayPaths,
    profile: ProtectedProfile,
    runner: ProcessRunner,
    base_environment: Mapping[str, str],
    transaction_id: str,
) -> None:
    transaction_dir = paths.root / "transactions" / transaction_id
    try:
        artifacts = resolve_and_materialize(
            inspected,
            profile=profile,
            artifacts_root=paths.artifacts,
            transaction_dir=transaction_dir,
            resolver_options=resolver_options,
            runner=runner,
            base_environment=base_environment,
        )
        lock_text = render_lock(artifacts, project=paths.project)
        input_text = _render_input_roots(artifacts)
        build_generation(
            paths,
            profile=profile,
            input_text=input_text,
            lock_text=lock_text,
            runner=runner,
            verifier=lambda site_packages: verify_candidate_overlay(
                site_packages,
                runner=runner,
                profile=profile,
                base_environment=base_environment,
            ),
            transaction_id=transaction_id,
            base_environment=base_environment,
            acquire_lock=False,
        )
    finally:
        shutil.rmtree(transaction_dir, ignore_errors=True)


def _run_query(
    request: PipRequest,
    *,
    paths: OverlayPaths,
    runner: ProcessRunner,
    base_environment: Mapping[str, str],
) -> int:
    environment = dict(base_environment)
    environment.update(
        {
            "PYTHONPATH": f"{paths.current}/site-packages:/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    command = [
        "/opt/venv/bin/python",
        "-m",
        "pip",
        request.command,
        *request.query_options,
        *request.names,
    ]
    result = runner.run(
        command,
        environment=environment,
        cwd=paths.project,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        evidence = result.stderr.strip() or "pip query failed"
        raise PipPolicyError(evidence)
    return 0


def _render_input_roots(artifacts: tuple[WheelArtifact, ...]) -> str:
    roots = sorted(
        (
            packaging_canonicalize_name(artifact.name),
            artifact.path.resolve(strict=True).as_uri(),
        )
        for artifact in artifacts
        if artifact.requested
    )
    return "".join(f"{name} @ {uri}\n" for name, uri in roots)


def _input_lines(text: str) -> tuple[str, ...]:
    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    names: set[str] = set()
    for line in lines:
        name = _requirement_name(line)
        if name in names:
            raise OverlayVerificationError(
                f"current top-level input contains duplicate package: {name}"
            )
        names.add(name)
    return lines


def _requirement_name(value: str) -> str:
    try:
        return packaging_canonicalize_name(Requirement(value).name)
    except InvalidRequirement as error:
        raise OverlayVerificationError(
            f"current top-level input is invalid; run amd-ai doctor and repair: {value}"
        ) from error


def _known_requirement_names(
    inspected: InspectedRequirements,
) -> frozenset[str]:
    names: set[str] = set()
    for value in (*inspected.resolver_inputs, *inspected.vcs_inputs):
        if value.startswith("-"):
            continue
        try:
            names.add(packaging_canonicalize_name(Requirement(value).name))
        except InvalidRequirement:
            continue
    for path in inspected.local_inputs:
        if not path.is_file() or path.suffix.lower() != ".whl":
            continue
        try:
            name, _, _, _ = parse_wheel_filename(path.name)
        except InvalidWheelFilename as error:
            raise RequirementPolicyError(
                f"invalid local wheel filename: {path.name}"
            ) from error
        names.add(packaging_canonicalize_name(str(name)))
    return frozenset(names)


def _combine_inspected(
    first: InspectedRequirements,
    second: InspectedRequirements,
) -> InspectedRequirements:
    return InspectedRequirements(
        resolver_inputs=(*first.resolver_inputs, *second.resolver_inputs),
        external=tuple(sorted(set(first.external).union(second.external))),
        local_inputs=(*first.local_inputs, *second.local_inputs),
        vcs_inputs=(*first.vcs_inputs, *second.vcs_inputs),
    )


def _has_overlay_inputs(inspected: InspectedRequirements) -> bool:
    return bool(
        inspected.resolver_inputs
        or inspected.local_inputs
        or inspected.vcs_inputs
    )


def _reject_protected_uninstall(request: PipRequest) -> None:
    protected = sorted(
        {
            canonicalize_protected_name(name)
            for name in request.names
            if canonicalize_protected_name(name) in PROTECTED_DISTRIBUTIONS
        }
    )
    if protected:
        raise PipPolicyError(
            "cannot uninstall protected parent distributions: "
            + ", ".join(protected)
        )


def _require_managed_project(project: Path) -> Path:
    try:
        root = project.resolve(strict=True)
    except OSError as error:
        raise OverlayError(f"project directory does not exist: {project}") from error
    if not root.is_dir():
        raise OverlayError(f"project path is not a directory: {root}")
    marker = root / "amd-ai-project.toml"
    if marker.is_symlink() or not marker.is_file():
        raise OverlayError(
            f"managed project marker is missing: {marker}"
        )
    return root


def _redact_text(
    value: str, *, secret_values: tuple[str, ...] = ()
) -> str:
    redacted = value
    for secret in sorted(secret_values, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return URL_USERINFO_PATTERN.sub(
        r"\g<scheme><redacted>@", redacted
    )


def _append_private_log(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise TransactionError(f"overlay log path is unsafe: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, content.encode("utf-8", errors="replace"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
