import pytest

from amd_ai.runner import CommandError, CommandResult, SubprocessRunner


def test_runner_captures_stdout_without_shell():
    result = SubprocessRunner().run(["printf", "%s", "ok"])

    assert result.args == ("printf", "%s", "ok")
    assert result.stdout == "ok"
    assert result.returncode == 0


def test_runner_raises_typed_error():
    with pytest.raises(CommandError) as error:
        SubprocessRunner().run(["python3.12", "-c", "import sys; sys.exit(7)"])

    assert error.value.result.returncode == 7


def test_command_result_truncation_defaults_are_backward_compatible() -> None:
    result = CommandResult(("true",), 0, "", "")

    assert result.stdout_truncated is False
    assert result.stderr_truncated is False
