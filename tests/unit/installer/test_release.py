from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_ai.installer.release import ReleaseError, load_stable_release


FIXTURE = Path("tests/fixtures/releases/stable.json")


def test_valid_release_distinguishes_manifest_and_config_digest() -> None:
    release = load_stable_release(FIXTURE)

    assert release.torch.reference.startswith(
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:"
    )
    assert release.torch.manifest_digest != release.torch.config_digest
    assert release.supported_host_adapter_ids == ("ubuntu-24.04",)


@pytest.mark.parametrize(
    "mutation", ("missing", "unknown", "bad-digest", "mutable-image")
)
def test_release_schema_rejects_ambiguous_payload(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["source_revision"]
    elif mutation == "unknown":
        payload["latest"] = True
    elif mutation == "bad-digest":
        payload["torch"]["manifest_digest"] = payload["torch"][
            "config_digest"
        ]
    else:
        payload["torch"]["image"] += ":latest"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError):
        load_stable_release(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("release_id", "latest"),
        ("gpu_arch", "gfx1100"),
        ("rocm_version", "7.2.4"),
        ("python_version", "3.13"),
        ("torch_version", "2.10.0"),
        ("published_at", "2026-07-10T12:00:00+00:00"),
    ),
)
def test_release_schema_rejects_wrong_fixed_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError):
        load_stable_release(path)


def test_release_schema_rejects_duplicate_json_key(tmp_path: Path) -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    path = tmp_path / "release.json"
    path.write_text(
        text.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError, match="duplicate"):
        load_stable_release(path)


def test_release_schema_rejects_duplicate_adapter(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["supported_host_adapter_ids"] = [
        "ubuntu-24.04",
        "ubuntu-24.04",
    ]
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError, match="adapter"):
        load_stable_release(path)
