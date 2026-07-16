from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_push_only_publication_requires_buildx(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeDocker:
        prefix = ("docker",)

        def image_id(self, reference):
            return "sha256:" + ("a" if "python" in reference else "b") * 64

        def require_buildx(self):
            calls.append("buildx")
            return "buildx 0.30.1"

    candidate = SimpleNamespace(source_revision="c" * 40, torch_local_id="id")
    observed = SimpleNamespace(
        base=SimpleNamespace(reference="ghcr.io/example/base@sha256:" + "d" * 64),
        torch=SimpleNamespace(reference="ghcr.io/example/torch@sha256:" + "e" * 64),
    )
    monkeypatch.setattr(cli.Docker, "detect", classmethod(lambda cls: FakeDocker()))
    monkeypatch.setattr(cli, "_clean_release_revision", lambda: "c" * 40)
    monkeypatch.setattr(cli, "validate_publish_inputs", lambda **kwargs: candidate)
    monkeypatch.setattr(
        cli,
        "verify_publish_candidate_local_images",
        lambda value, registry: value,
    )
    monkeypatch.setattr(cli, "publish_images", lambda value, registry: observed)
    monkeypatch.setattr(cli, "write_observed_release", lambda path, value: None)

    code = cli.publish_release_command(
        release_id="0.2.0",
        qualification_path=tmp_path / "qualification.json",
        sbom_path=tmp_path / "release.spdx.json",
        output=tmp_path / "stable.json",
        publish_report=tmp_path / "publish.json",
        dry_run=False,
        push_only=True,
    )

    assert code == 0
    assert calls == ["buildx"]
