from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ReleaseImage:
    image: str
    manifest_digest: str
    config_digest: str
    artifact_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_digests",
            MappingProxyType(dict(self.artifact_digests)),
        )

    @property
    def reference(self) -> str:
        return f"{self.image}@{self.manifest_digest}"


@dataclass(frozen=True)
class StableRelease:
    schema_version: int
    release_id: str
    source_repository: str
    source_revision: str
    qualification_profile_digest: str
    qualification_report_digest: str
    sbom_digest: str
    gpu_arch: str
    supported_host_adapter_ids: tuple[str, ...]
    rocm_version: str
    python_version: str
    torch_version: str
    torch_profile_id: str
    torch_profile_digest: str
    base: ReleaseImage
    torch: ReleaseImage
    published_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_host_adapter_ids",
            tuple(self.supported_host_adapter_ids),
        )
