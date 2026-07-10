import pytest

from amd_ai.runner import CommandError, SubprocessRunner


def test_runner_captures_stdout_without_shell():
    result = SubprocessRunner().run(["printf", "%s", "ok"])

    assert result.args == ("printf", "%s", "ok")
    assert result.stdout == "ok"
    assert result.returncode == 0


def test_runner_raises_typed_error():
    with pytest.raises(CommandError) as error:
        SubprocessRunner().run(["python3.12", "-c", "import sys; sys.exit(7)"])

    assert error.value.result.returncode == 7
