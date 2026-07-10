from pathlib import Path

import pytest

from amd_ai.project.dependencies import (
    DependencyError,
    lock_project_dependencies,
    render_torch_constraints,
    validate_project_lock,
)


def test_constraints_pin_complete_verified_stack():
    text = render_torch_constraints("profiles/torch/stable.requirements.lock")

    assert text.splitlines() == [
        "torch==2.9.1",
        "torchvision==0.24.0",
        "torchaudio==2.9.0",
        "triton==3.5.1",
    ]


def test_empty_project_input_produces_an_empty_lock(tmp_path):
    (tmp_path / "requirements.in").write_text(
        "# Project dependencies only.\n", encoding="utf-8"
    )
    (tmp_path / "torch-constraints.txt").write_text(
        render_torch_constraints("profiles/torch/stable.requirements.lock"),
        encoding="utf-8",
    )

    lock_project_dependencies(tmp_path)

    assert (tmp_path / "requirements.lock").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "lock_text",
    [
        "torch==2.8.0 --hash=sha256:" + "a" * 64 + "\n",
        "torch @ https://example.com/torch.whl --hash=sha256:" + "a" * 64 + "\n",
    ],
)
def test_project_lock_rejects_replacement_or_direct_torch(lock_text):
    constraints = render_torch_constraints(
        Path("profiles/torch/stable.requirements.lock")
    )

    with pytest.raises(DependencyError):
        validate_project_lock(lock_text, constraints)
