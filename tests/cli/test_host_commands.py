import json

from amd_ai.cli import main


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
