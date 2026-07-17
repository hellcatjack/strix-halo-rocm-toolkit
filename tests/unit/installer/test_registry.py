import json
from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.installer.registry import (
    DEFAULT_REGISTRY_POLICY_PATH,
    RegistryChoice,
    RegistryPolicyError,
    load_registry_policy,
    registry_candidates,
)
from amd_ai.installer.release import load_stable_release


FIXTURE = Path("tests/fixtures/releases/stable.json")
SWR = "swr.cn-east-3.myhuaweicloud.com/hellcat-home"


def test_default_registry_policy_is_strict_and_deterministic() -> None:
    policy = load_registry_policy(DEFAULT_REGISTRY_POLICY_PATH)

    assert policy.schema_version == 1
    assert policy.default_order == ("swr", "ghcr")
    assert tuple(policy.sources) == ("swr", "ghcr")
    assert policy.sources["swr"].repositories == {
        "base": f"{SWR}/strix-halo-rocm-python",
        "torch": f"{SWR}/strix-halo-rocm-pytorch",
    }
    assert policy.sources["ghcr"].repositories == {
        "base": "ghcr.io/hellcatjack/strix-halo-rocm-python",
        "torch": "ghcr.io/hellcatjack/strix-halo-rocm-pytorch",
    }


def test_auto_registry_prefers_swr_then_canonical_ghcr() -> None:
    release = load_stable_release(FIXTURE)

    candidates = registry_candidates(release, RegistryChoice.AUTO)

    assert tuple(candidate.name for candidate in candidates) == ("swr", "ghcr")
    assert candidates[0].release.base.image == (
        f"{SWR}/strix-halo-rocm-python"
    )
    assert candidates[0].release.torch.image == (
        f"{SWR}/strix-halo-rocm-pytorch"
    )
    assert candidates[1].release == release


def test_registry_candidate_changes_only_repository_names() -> None:
    release = load_stable_release(FIXTURE)

    mirrored = registry_candidates(release, "swr")[0].release

    assert mirrored.base.manifest_digest == release.base.manifest_digest
    assert mirrored.base.config_digest == release.base.config_digest
    assert mirrored.base.artifact_digests == release.base.artifact_digests
    assert mirrored.torch.manifest_digest == release.torch.manifest_digest
    assert mirrored.torch.config_digest == release.torch.config_digest
    assert mirrored.torch.artifact_digests == release.torch.artifact_digests


@pytest.mark.parametrize(
    ("choice", "names"),
    (("swr", ("swr",)), ("ghcr", ("ghcr",))),
)
def test_explicit_registry_has_one_candidate(
    choice: str, names: tuple[str, ...]
) -> None:
    release = load_stable_release(FIXTURE)

    assert tuple(
        candidate.name for candidate in registry_candidates(release, choice)
    ) == names


def test_unmapped_custom_release_uses_canonical_source_in_auto() -> None:
    release = load_stable_release(FIXTURE)
    custom = replace(
        release,
        base=replace(release.base, image="ghcr.io/example/base"),
        torch=replace(release.torch, image="ghcr.io/example/torch"),
    )

    candidates = registry_candidates(custom, "auto")

    assert len(candidates) == 1
    assert candidates[0].name == "ghcr"
    assert candidates[0].release == custom


def test_explicit_swr_rejects_unmapped_release() -> None:
    release = load_stable_release(FIXTURE)
    custom = replace(
        release,
        base=replace(release.base, image="ghcr.io/example/base"),
    )

    with pytest.raises(RegistryPolicyError, match="SWR mapping"):
        registry_candidates(custom, "swr")


def test_trusted_references_include_swr_then_ghcr() -> None:
    release = load_stable_release(FIXTURE)

    references = tuple(
        candidate.release.torch.reference
        for candidate in registry_candidates(release)
    )

    assert references[0].startswith(f"{SWR}/")
    assert references[1] == release.torch.reference


def test_registry_policy_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "registries.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,'
        '"default_order":["ghcr"],"sources":[]}',
        encoding="utf-8",
    )

    with pytest.raises(RegistryPolicyError, match="duplicate key"):
        load_registry_policy(path)


def test_registry_policy_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = _policy_payload()
    payload["unexpected"] = True

    with pytest.raises(RegistryPolicyError, match="keys"):
        _load_policy_payload(tmp_path, payload)


def test_registry_policy_rejects_duplicate_source_ids(
    tmp_path: Path,
) -> None:
    payload = _policy_payload()
    payload["sources"].append(dict(payload["sources"][0]))

    with pytest.raises(RegistryPolicyError, match="duplicate source"):
        _load_policy_payload(tmp_path, payload)


def test_registry_policy_requires_complete_repository_kinds(
    tmp_path: Path,
) -> None:
    payload = _policy_payload()
    del payload["sources"][0]["repositories"]["torch"]

    with pytest.raises(RegistryPolicyError, match="repositories keys"):
        _load_policy_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "repository",
    (
        "swr.cn-east-3.myhuaweicloud.com/hellcat-home/image:latest",
        "swr.cn-east-3.myhuaweicloud.com/hellcat-home/image@sha256:"
        + "a" * 64,
        "https://swr.cn-east-3.myhuaweicloud.com/hellcat-home/image",
    ),
)
def test_registry_policy_rejects_mutable_or_malformed_repositories(
    tmp_path: Path,
    repository: str,
) -> None:
    payload = _policy_payload()
    payload["sources"][0]["repositories"]["base"] = repository

    with pytest.raises(RegistryPolicyError, match="repository"):
        _load_policy_payload(tmp_path, payload)


def test_registry_policy_requires_canonical_source_last(
    tmp_path: Path,
) -> None:
    payload = _policy_payload()
    payload["default_order"] = ["ghcr", "swr"]

    with pytest.raises(RegistryPolicyError, match="end with ghcr"):
        _load_policy_payload(tmp_path, payload)


def _policy_payload() -> dict:
    return {
        "schema_version": 1,
        "default_order": ["swr", "ghcr"],
        "sources": [
            {
                "id": "swr",
                "label": "华为 SWR",
                "repositories": {
                    "base": f"{SWR}/strix-halo-rocm-python",
                    "torch": f"{SWR}/strix-halo-rocm-pytorch",
                },
            },
            {
                "id": "ghcr",
                "label": "GHCR",
                "repositories": {
                    "base": "ghcr.io/hellcatjack/strix-halo-rocm-python",
                    "torch": "ghcr.io/hellcatjack/strix-halo-rocm-pytorch",
                },
            },
        ],
    }


def _load_policy_payload(tmp_path: Path, payload: object):
    path = tmp_path / "registries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_registry_policy(path)
