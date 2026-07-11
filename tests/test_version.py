import hashlib
import json
from pathlib import Path

import pytest

from amd_ai import __version__
from amd_ai.cli import main


def test_version_constant_and_cli(capsys):
    assert __version__ == "0.2.3"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "amd-ai 0.2.3"


def test_installer_only_release_keeps_stable_image_baseline() -> None:
    path = Path("profiles/releases/stable.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "4226d04bf995c9c253c6a978f08bdbb9466ccd47119f967ebd39f0c08b7bfe2d"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["release_id"] == "0.2.0"
    assert payload["base"]["manifest_digest"] == (
        "sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12"
    )
    assert payload["torch"]["manifest_digest"] == (
        "sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b"
    )
