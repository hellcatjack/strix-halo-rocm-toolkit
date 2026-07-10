from __future__ import annotations

import json

import pytest

from tests.container.test_readonly_overlay import project_factory


pytestmark = pytest.mark.container


def test_readonly_root_blocks_base_mutation_but_protected_pip_works(
    project_factory,
) -> None:
    project = project_factory("readonly-root")

    root_write = project.run("touch", "/opt/venv/forbidden")
    base_pip = project.run(
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "six==1.17.0",
    )
    protected_pip = project.run("pip", "install", "six==1.17.0")

    assert root_write.returncode != 0
    assert base_pip.returncode != 0
    assert protected_pip.returncode == 0, protected_pip.stderr


def test_shadow_is_blocked_and_offline_lock_replay_restores_overlay(
    project_factory,
) -> None:
    project = project_factory("shadow-repair")
    installed = project.run("pip", "install", "six==1.17.0")
    assert installed.returncode == 0, installed.stderr
    current_before = project.current_target()
    shadow = project.path / ".amd-ai/current/site-packages/torch.py"
    shadow.write_text("shadow = True\n", encoding="utf-8")

    blocked = project.run(
        "container-check",
        "--mode",
        "torch",
        "--metadata-only",
        "--json",
        "-",
    )

    assert blocked.returncode == 2, blocked.stderr or blocked.stdout
    report = json.loads(blocked.stdout)
    assert "TORCH.SHADOWED" in {
        finding["code"] for finding in report["findings"]
    }

    repaired = project.repair_overlay("TORCH.SHADOWED")
    verified = project.run(
        "python",
        "-c",
        "import six, torch; print(six.__version__); print(torch.__version__)",
    )

    assert repaired.returncode == 0, repaired.stderr or repaired.stdout
    assert project.current_target() != current_before
    assert verified.returncode == 0, verified.stderr or verified.stdout
    assert "1.17.0" in verified.stdout
