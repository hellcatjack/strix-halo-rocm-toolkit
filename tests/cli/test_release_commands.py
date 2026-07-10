from __future__ import annotations

from pathlib import Path

import pytest

from amd_ai import cli


def test_release_verify_loads_and_checks_manifest(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "verify_release_command",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    code = cli.main(
        [
            "release",
            "verify",
            "--manifest",
            "tests/fixtures/releases/stable.json",
        ]
    )

    assert code == 0
    assert captured["manifest_path"] == Path(
        "tests/fixtures/releases/stable.json"
    )


def test_release_publish_requires_all_evidence_paths() -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(
            ["release", "publish", "--release-id", "0.2.0"]
        )

    assert error.value.code == 2


def test_release_publish_forwards_stage_and_paths(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "publish_release_command",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    qualification = tmp_path / "release.json"
    sbom = tmp_path / "release.spdx.json"

    code = cli.main(
        [
            "release",
            "publish",
            "--release-id",
            "0.2.0",
            "--qualification",
            str(qualification),
            "--sbom",
            str(sbom),
            "--dry-run",
        ]
    )

    assert code == 0
    assert captured == {
        "release_id": "0.2.0",
        "qualification_path": qualification,
        "sbom_path": sbom,
        "output": Path("profiles/releases/stable.json"),
        "publish_report": Path("reports/publish-candidate.json"),
        "dry_run": True,
        "push_only": False,
    }


def test_release_publish_stages_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(
            [
                "release",
                "publish",
                "--release-id",
                "0.2.0",
                "--qualification",
                "release.json",
                "--sbom",
                "release.spdx.json",
                "--dry-run",
                "--push-only",
            ]
        )

    assert error.value.code == 2
