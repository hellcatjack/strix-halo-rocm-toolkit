import pytest

from amd_ai import __version__
from amd_ai.cli import main


def test_version_constant_and_cli(capsys):
    assert __version__ == "0.1.0"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "amd-ai 0.1.0"
