from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from amd_ai.installer.models import StableRelease


class RegistryPolicyError(ValueError):
    pass


class RegistryChoice(StrEnum):
    AUTO = "auto"
    SWR = "swr"
    GHCR = "ghcr"


@dataclass(frozen=True)
class RegistryCandidate:
    name: str
    label: str
    release: StableRelease


SWR_REPOSITORIES: Mapping[str, str] = MappingProxyType(
    {
        "ghcr.io/hellcatjack/strix-halo-rocm-python": (
            "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
            "strix-halo-rocm-python"
        ),
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch": (
            "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
            "strix-halo-rocm-pytorch"
        ),
    }
)


def registry_candidates(
    release: StableRelease,
    choice: RegistryChoice | str = RegistryChoice.AUTO,
) -> tuple[RegistryCandidate, ...]:
    selected = _registry_choice(choice)
    candidates: list[RegistryCandidate] = []
    mirrored = _swr_release(release)
    if selected in (RegistryChoice.AUTO, RegistryChoice.SWR):
        if mirrored is None:
            if selected is RegistryChoice.SWR:
                raise RegistryPolicyError(
                    "stable release has no trusted SWR mapping"
                )
        else:
            candidates.append(
                RegistryCandidate("swr", "华为 SWR", mirrored)
            )
    if selected in (RegistryChoice.AUTO, RegistryChoice.GHCR):
        candidates.append(RegistryCandidate("ghcr", "GHCR", release))
    return tuple(candidates)


def registry_plan_label(choice: RegistryChoice | str) -> str:
    selected = _registry_choice(choice)
    if selected is RegistryChoice.AUTO:
        return "auto（华为 SWR 优先，GHCR 回退）"
    if selected is RegistryChoice.SWR:
        return "swr（仅华为 SWR，不回退）"
    return "ghcr（仅 GHCR，不回退）"


def trusted_image_references(
    release: StableRelease,
    *,
    kind: str,
) -> tuple[str, ...]:
    if kind not in {"base", "torch"}:
        raise RegistryPolicyError("release image kind is invalid")
    return tuple(
        getattr(candidate.release, kind).reference
        for candidate in registry_candidates(release)
    )


def _registry_choice(value: RegistryChoice | str) -> RegistryChoice:
    try:
        return RegistryChoice(value)
    except (TypeError, ValueError) as error:
        raise RegistryPolicyError("registry choice is invalid") from error


def _swr_release(release: StableRelease) -> StableRelease | None:
    base = SWR_REPOSITORIES.get(release.base.image)
    torch = SWR_REPOSITORIES.get(release.torch.image)
    if base is None or torch is None:
        return None
    return replace(
        release,
        base=replace(release.base, image=base),
        torch=replace(release.torch, image=torch),
    )
