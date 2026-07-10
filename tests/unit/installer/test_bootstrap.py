from __future__ import annotations

import os
from pathlib import Path

import pytest

from amd_ai.installer import bootstrap
from amd_ai.installer.bootstrap import (
    BootstrapError,
    LAUNCHER_CONTENT,
    install_user_runtime,
)


def toolkit_fixture(root: Path) -> Path:
    files = {
        "src/amd_ai/cli.py": "def main(argv=None):\n    return 0\n",
        "src/amd_ai/__init__.py": '__version__ = "0.2.0"\n',
        "profiles/torch/stable.env": "ROCM_VERSION=7.2.1\n",
        "templates/project/requirements.in": "\n",
        "images/common/Dockerfile": "FROM scratch\n",
        "bin/_dispatch": "#!/usr/bin/env bash\nexit 0\n",
        "pyproject.toml": "[project]\nname = 'fixture'\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "bin/_dispatch").chmod(0o755)
    return root


def test_install_runtime_copies_required_payload_and_switches_current(
    tmp_path: Path,
) -> None:
    source = toolkit_fixture(tmp_path / "source")
    home = tmp_path / "home"

    result = install_user_runtime(
        source_root=source,
        home=home,
        version="0.2.0",
        installer_source_revision="a" * 40,
    )

    assert (result.runtime / "src/amd_ai/cli.py").is_file()
    assert (result.runtime / "profiles/torch/stable.env").is_file()
    assert (
        os.readlink(home / ".local/share/strix-halo-rocm-toolkit/current")
        == "releases/0.2.0-" + "a" * 12
    )
    assert os.access(home / ".local/bin/strix-halo-rocm", os.X_OK)
    assert result.launcher.read_text(encoding="ascii") == LAUNCHER_CONTENT
    assert result.launcher.stat().st_mode & 0o777 == 0o755


def test_source_payload_symlink_is_rejected(tmp_path: Path) -> None:
    source = toolkit_fixture(tmp_path / "source")
    (source / "profiles/unsafe").symlink_to(tmp_path / "outside")

    with pytest.raises(BootstrapError, match="symlink"):
        install_user_runtime(
            source_root=source,
            home=tmp_path / "home",
            version="0.2.0",
            installer_source_revision="a" * 40,
        )


def test_symlinked_destination_control_directory_is_rejected(
    tmp_path: Path,
) -> None:
    source = toolkit_fixture(tmp_path / "source")
    home = tmp_path / "home"
    (home / ".local/share").mkdir(parents=True)
    (home / ".local/share/strix-halo-rocm-toolkit").symlink_to(
        tmp_path / "attacker"
    )

    with pytest.raises(BootstrapError, match="symlink"):
        install_user_runtime(
            source_root=source,
            home=home,
            version="0.2.0",
            installer_source_revision="a" * 40,
        )


def test_partial_copy_never_creates_release_or_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = toolkit_fixture(tmp_path / "source")
    home = tmp_path / "home"
    real_copy2 = bootstrap.shutil.copy2
    calls = 0

    def fail_second_copy(source_path: Path, target_path: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated copy failure")
        return Path(real_copy2(source_path, target_path))

    monkeypatch.setattr(bootstrap.shutil, "copy2", fail_second_copy)

    with pytest.raises(BootstrapError, match="copy"):
        install_user_runtime(
            source_root=source,
            home=home,
            version="0.2.0",
            installer_source_revision="a" * 40,
        )

    toolkit = home / ".local/share/strix-halo-rocm-toolkit"
    assert not (toolkit / f"releases/0.2.0-{'a' * 12}").exists()
    assert not (toolkit / "current").exists()
    assert not list((toolkit / "releases").glob(".staging-*"))


def test_failed_current_switch_preserves_previous_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = toolkit_fixture(tmp_path / "source")
    home = tmp_path / "home"
    install_user_runtime(
        source_root=source,
        home=home,
        version="0.2.0",
        installer_source_revision="a" * 40,
    )
    current = home / ".local/share/strix-halo-rocm-toolkit/current"
    previous = os.readlink(current)
    real_replace = os.replace

    def fail_current_switch(source_path: str | Path, target_path: str | Path) -> None:
        if Path(target_path) == current:
            raise OSError("simulated current switch failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(os, "replace", fail_current_switch)

    with pytest.raises(BootstrapError, match="current"):
        install_user_runtime(
            source_root=source,
            home=home,
            version="0.2.0",
            installer_source_revision="b" * 40,
        )

    assert os.readlink(current) == previous
    assert not list(current.parent.glob(".current-*"))


def test_main_installs_then_forwards_remaining_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = toolkit_fixture(tmp_path / "source")
    captured: dict[str, object] = {}
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        bootstrap, "discover_installer_source_revision", lambda path: "a" * 40
    )

    def forward(arguments: list[str]) -> int:
        captured["arguments"] = arguments
        return 7

    monkeypatch.setattr(bootstrap, "_forward_install", forward)

    result = bootstrap.main(
        [
            "--source-root",
            str(source),
            "--mode",
            "container",
            "--dry-run",
        ]
    )

    assert result == 7
    assert captured["arguments"] == [
        "--source-root",
        str(source.resolve()),
        "--mode",
        "container",
        "--dry-run",
    ]
