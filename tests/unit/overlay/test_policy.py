from __future__ import annotations

import pytest

from amd_ai.overlay.policy import PipPolicyError, parse_pip_request


@pytest.mark.parametrize(
    "argv",
    [
        ["install", "--user", "requests"],
        ["install", "--target=/tmp/site", "requests"],
        ["install", "--prefix", "/tmp/prefix", "requests"],
        ["install", "--root", "/tmp/root", "requests"],
        ["install", "-e", "."],
        ["install", "git+https://github.com/pallets/flask.git"],
        ["install", "--trusted-host", "packages.example", "requests"],
        ["install", "--", "--target", "/tmp/site", "requests"],
    ],
)
def test_install_rejects_nontransactional_targets(argv: list[str]) -> None:
    with pytest.raises(PipPolicyError):
        parse_pip_request(argv)


def test_install_keeps_requirements_and_secret_free_index_options() -> None:
    request = parse_pip_request(
        [
            "install",
            "--index-url",
            "https://packages.example/simple",
            "-r",
            "deps.txt",
        ]
    )

    assert request.command == "install"
    assert request.requirements_files == ("deps.txt",)
    assert request.requirements == ()
    assert request.resolver_options == (
        "--index-url",
        "https://packages.example/simple",
    )


def test_install_accepts_exact_commit_named_git_requirement() -> None:
    requirement = (
        "demo @ git+https://github.com/example/demo.git@" + "a" * 40
    )

    request = parse_pip_request(["install", requirement])

    assert request.requirements == (requirement,)


@pytest.mark.parametrize(
    "value",
    [
        "http://packages.example/simple",
        "https://user:secret@packages.example/simple",
    ],
)
def test_index_url_must_be_https_and_credential_free(value: str) -> None:
    with pytest.raises(PipPolicyError, match="index URL"):
        parse_pip_request(["install", "--index-url", value, "requests"])


def test_query_and_uninstall_commands_have_narrow_grammar() -> None:
    assert parse_pip_request(["show", "torch"]).names == ("torch",)
    assert parse_pip_request(["freeze"]).command == "freeze"
    uninstall = parse_pip_request(["uninstall", "-y", "requests"])
    assert uninstall.assume_yes is True
    assert uninstall.names == ("requests",)


def test_unknown_subcommand_and_nul_are_rejected() -> None:
    with pytest.raises(PipPolicyError, match="unsupported pip command"):
        parse_pip_request(["download", "requests"])
    with pytest.raises(PipPolicyError, match="forbidden character"):
        parse_pip_request(["install", "requests\0bad"])
