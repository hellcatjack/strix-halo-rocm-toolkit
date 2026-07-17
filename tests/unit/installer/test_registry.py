from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.installer.registry import (
    RegistryChoice,
    RegistryPolicyError,
    registry_candidates,
)
from amd_ai.installer.release import load_stable_release


FIXTURE = Path("tests/fixtures/releases/stable.json")
SWR = "swr.cn-east-3.myhuaweicloud.com/hellcat-home"


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
