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

