from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


FORBIDDEN_OPTIONS = frozenset(
    {
        "--user",
        "--target",
        "--prefix",
        "--root",
        "--editable",
        "-e",
        "--break-system-packages",
        "--ignore-installed",
        "--force-reinstall",
        "--python",
    }
)
VALUE_OPTIONS = frozenset(
    {"--index-url", "--extra-index-url", "--index", "--find-links"}
)
FLAG_OPTIONS = frozenset({"--no-index", "--pre"})
QUERY_COMMANDS = frozenset({"list", "show", "check", "freeze"})
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PINNED_GIT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]+\])?\s+@\s+"
    r"git\+https://[^@\s]+@(?P<commit>[0-9a-f]{40})(?:#\S+)?$"
)
VCS_PREFIXES = ("git+", "hg+", "svn+", "bzr+")


class PipPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class PipRequest:
    command: str
    requirements: tuple[str, ...] = ()
    requirements_files: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    resolver_options: tuple[str, ...] = ()
    upgrade: bool = False
    assume_yes: bool = False
    query_options: tuple[str, ...] = ()


def parse_pip_request(argv: list[str]) -> PipRequest:
    if not argv:
        raise PipPolicyError("pip command is required")
    for value in argv:
        if not value or any(character in value for character in ("\0", "\n", "\r")):
            raise PipPolicyError("pip argument contains a forbidden character")

    command = argv[0]
    if command == "install":
        return _parse_install(argv[1:])
    if command == "uninstall":
        return _parse_uninstall(argv[1:])
    if command in QUERY_COMMANDS:
        return _parse_query(command, argv[1:])
    raise PipPolicyError(f"unsupported pip command: {command}")


def _parse_install(argv: list[str]) -> PipRequest:
    requirements: list[str] = []
    requirement_files: list[str] = []
    resolver_options: list[str] = []
    upgrade = False
    index = 0
    while index < len(argv):
        value = argv[index]
        option = value.split("=", 1)[0]
        if option in FORBIDDEN_OPTIONS or value.startswith("-e"):
            raise PipPolicyError(f"pip option is forbidden: {option}")
        if value == "--":
            raise PipPolicyError("pip option terminator is forbidden")
        if value in {"-U", "--upgrade"}:
            upgrade = True
            index += 1
            continue
        if value in FLAG_OPTIONS:
            resolver_options.append(value)
            index += 1
            continue
        if value in {"-r", "--requirement"}:
            path, index = _next_value(argv, index, value)
            requirement_files.append(path)
            continue
        if value.startswith("--requirement="):
            requirement_files.append(_assignment_value(value))
            index += 1
            continue
        if value.startswith("-r") and value != "-r":
            requirement_files.append(value[2:])
            index += 1
            continue
        if value in VALUE_OPTIONS:
            option_value, index = _next_value(argv, index, value)
            _validate_resolver_option(value, option_value)
            resolver_options.extend((value, option_value))
            continue
        if option in VALUE_OPTIONS and "=" in value:
            option_value = _assignment_value(value)
            _validate_resolver_option(option, option_value)
            resolver_options.extend((option, option_value))
            index += 1
            continue
        if value in {"-i", "-f"}:
            option_value, index = _next_value(argv, index, value)
            expanded = "--index-url" if value == "-i" else "--find-links"
            _validate_resolver_option(expanded, option_value)
            resolver_options.extend((expanded, option_value))
            continue
        if value.startswith("-"):
            raise PipPolicyError(f"unsupported pip install option: {value}")
        _validate_requirement_argument(value)
        requirements.append(value)
        index += 1

    if not requirements and not requirement_files:
        raise PipPolicyError("pip install requires a package or requirements file")
    return PipRequest(
        command="install",
        requirements=tuple(requirements),
        requirements_files=tuple(requirement_files),
        resolver_options=tuple(resolver_options),
        upgrade=upgrade,
    )


def _parse_uninstall(argv: list[str]) -> PipRequest:
    names: list[str] = []
    assume_yes = False
    for value in argv:
        if value in {"-y", "--yes"}:
            assume_yes = True
            continue
        if value.startswith("-"):
            raise PipPolicyError(f"unsupported pip uninstall option: {value}")
        if NAME_PATTERN.fullmatch(value) is None:
            raise PipPolicyError(f"pip uninstall requires a package name: {value}")
        names.append(value)
    if not names:
        raise PipPolicyError("pip uninstall requires at least one package name")
    return PipRequest(
        command="uninstall",
        names=tuple(names),
        assume_yes=assume_yes,
    )


def _parse_query(command: str, argv: list[str]) -> PipRequest:
    options: list[str] = []
    names: list[str] = []
    value_options = {
        "list": frozenset({"--format", "--exclude"}),
        "show": frozenset(),
        "check": frozenset(),
        "freeze": frozenset(),
    }[command]
    flag_options = {
        "list": frozenset(
            {"--outdated", "--editable", "--exclude-editable", "--strict"}
        ),
        "show": frozenset({"--files", "-f", "--strict"}),
        "check": frozenset(),
        "freeze": frozenset({"--exclude-editable", "--strict"}),
    }[command]
    index = 0
    while index < len(argv):
        value = argv[index]
        option = value.split("=", 1)[0]
        if value in flag_options:
            options.append(value)
            index += 1
            continue
        if value in value_options:
            option_value, index = _next_value(argv, index, value)
            options.extend((value, option_value))
            continue
        if option in value_options and "=" in value:
            options.extend((option, _assignment_value(value)))
            index += 1
            continue
        if value.startswith("-"):
            raise PipPolicyError(f"unsupported pip {command} option: {value}")
        if command not in {"show"}:
            raise PipPolicyError(f"pip {command} does not accept package names")
        if NAME_PATTERN.fullmatch(value) is None:
            raise PipPolicyError(f"invalid package name: {value}")
        names.append(value)
        index += 1
    if command == "show" and not names:
        raise PipPolicyError("pip show requires at least one package name")
    return PipRequest(
        command=command,
        names=tuple(names),
        query_options=tuple(options),
    )


def _next_value(
    argv: list[str], index: int, option: str
) -> tuple[str, int]:
    if index + 1 >= len(argv) or argv[index + 1].startswith("-"):
        raise PipPolicyError(f"pip option requires a value: {option}")
    return argv[index + 1], index + 2


def _assignment_value(value: str) -> str:
    _, _, assigned = value.partition("=")
    if not assigned:
        raise PipPolicyError("pip option value is empty")
    return assigned


def _validate_resolver_option(option: str, value: str) -> None:
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise PipPolicyError("package index URL must not contain credentials")
    if option in {"--index-url", "--extra-index-url", "--index"}:
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
        ):
            raise PipPolicyError(
                "package index URL must be credential-free HTTPS"
            )
    elif parsed.scheme and parsed.scheme not in {"https", "file"}:
        raise PipPolicyError("find-links must be HTTPS, file, or a local path")


def _validate_requirement_argument(value: str) -> None:
    lowered = value.lower()
    if lowered.startswith(VCS_PREFIXES):
        raise PipPolicyError("VCS requirements require an explicit package name")
    if "git+" in lowered and PINNED_GIT_PATTERN.fullmatch(value) is None:
        raise PipPolicyError(
            "Git requirements require HTTPS and an exact 40-character commit"
        )
    if any(prefix in lowered for prefix in ("hg+", "svn+", "bzr+")):
        raise PipPolicyError("only exact-commit Git requirements are supported")
