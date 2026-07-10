import json
from pathlib import Path

from amd_ai import cli
from amd_ai.cli import main
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import FakeRunner


def test_preflight_fixture_writes_json_without_using_host_commands(tmp_path, capsys):
    output = tmp_path / "preflight.json"

    code = main(
        [
            "host-preflight",
            "--fixture-root",
            "tests/fixtures/host/healthy",
            "--json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["command"] == "host-preflight"
    assert payload["status"] == "unverified"
    assert payload["facts"]["kernel"] == "6.17.0-1025-oem"
    assert "HOST.UPSTREAM_UNVERIFIED" in capsys.readouterr().out


def test_preflight_fixture_option_is_hidden_from_help(capsys):
    try:
        main(["host-preflight", "--help"])
    except SystemExit as error:
        assert error.code == 0

    assert "--fixture-root" not in capsys.readouterr().out


def test_prepare_plan_fixture_writes_actions_without_applying(tmp_path, capsys):
    output = tmp_path / "prepare.json"

    code = main(
        [
            "host-prepare",
            "plan",
            "--target-user",
            "customer",
            "--fixture-root",
            "tests/fixtures/host/healthy",
            "--json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["command"] == "host-prepare"
    assert payload["facts"]["mode"] == "plan"
    assert payload["facts"]["actions"][0]["code"] == "BACKUP.SNAPSHOT"
    assert "APT.INSTALL_OEM_KERNEL" in capsys.readouterr().out


def test_prepare_apply_rejects_any_confirmation_except_apply(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    code = main(
        [
            "host-prepare",
            "apply",
            "--target-user",
            "customer",
            "--fixture-root",
            "tests/fixtures/host/healthy",
        ]
    )

    assert code == 2
    assert "confirmation" in capsys.readouterr().err.lower()


def test_prepare_resolves_groups_for_target_user_not_sudo_process(monkeypatch):
    class User:
        pw_gid = 1000

    monkeypatch.setattr(cli.pwd, "getpwnam", lambda name: User())
    monkeypatch.setattr(
        cli.os,
        "getgrouplist",
        lambda name, primary_gid: [primary_gid, 109, 110],
    )

    assert cli._target_user_group_ids(
        "customer",
        fixture_root=None,
        fixture_group_ids=(0,),
    ) == (109, 110, 1000)
    assert cli._target_user_group_ids(
        "fixture-user",
        fixture_root=Path("fixture"),
        fixture_group_ids=(109,),
    ) == (109,)


def test_docker_prefix_detection_falls_back_to_noninteractive_sudo():
    direct = ("docker", "version", "--format", "{{.Server.Version}}")
    fallback = (
        "sudo",
        "-n",
        "docker",
        "version",
        "--format",
        "{{.Server.Version}}",
    )
    runner = FakeRunner(
        {
            direct: CommandResult(direct, 1, "", "permission denied"),
            fallback: CommandResult(fallback, 0, "27.5.1\n", ""),
        }
    )

    assert cli._detect_docker_prefix(runner) == ("sudo", "-n", "docker")


def test_verify_fixture_requires_recorded_kernel_even_when_probe_passes(tmp_path):
    output = tmp_path / "verify.json"

    code = main(
        [
            "host-verify",
            "--fixture-root",
            "tests/fixtures/host/healthy",
            "--json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["command"] == "host-verify"
    assert payload["status"] == "unverified"
    assert payload["facts"]["probe"]["image_id"] == "sha256:fixture-probe"
