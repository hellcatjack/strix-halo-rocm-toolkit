from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from amd_ai.installer.models import StableRelease


DEFAULT_REGISTRY_POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "profiles"
    / "releases"
    / "registries.json"
)
ROOT_KEYS = frozenset({"schema_version", "default_order", "sources"})
SOURCE_KEYS = frozenset({"id", "label", "repositories"})
REPOSITORY_KEYS = frozenset({"base", "torch"})
SOURCE_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,31}")
REPOSITORY_PATTERN = re.compile(
    r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?"
    r"(?:/[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?){2,}"
)


class RegistryPolicyError(ValueError):
    pass


class RegistryChoice(StrEnum):
    AUTO = "auto"
    SWR = "swr"
    GHCR = "ghcr"


@dataclass(frozen=True)
class RegistrySource:
    name: str
    label: str
    repositories: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repositories",
            MappingProxyType(dict(self.repositories)),
        )


@dataclass(frozen=True)
class RegistryPolicy:
    schema_version: int
    default_order: tuple[str, ...]
    sources: Mapping[str, RegistrySource]

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_order", tuple(self.default_order))
        object.__setattr__(
            self,
            "sources",
            MappingProxyType(dict(self.sources)),
        )


@dataclass(frozen=True)
class RegistryCandidate:
    name: str
    label: str
    release: StableRelease


def load_registry_policy(path: Path) -> RegistryPolicy:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RegistryPolicyError(
            f"cannot read registry policy: {error}"
        ) from error
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise RegistryPolicyError(
            f"cannot parse registry policy: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RegistryPolicyError("registry policy must be an object")
    _require_keys("registry policy", payload, ROOT_KEYS)
    if type(payload["schema_version"]) is not int or payload[
        "schema_version"
    ] != 1:
        raise RegistryPolicyError(
            "registry policy schema_version must be integer 1"
        )

    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RegistryPolicyError(
            "registry policy sources must be a nonempty list"
        )
    sources: dict[str, RegistrySource] = {}
    for index, raw_source in enumerate(raw_sources):
        label = f"registry policy source {index}"
        if not isinstance(raw_source, dict):
            raise RegistryPolicyError(f"{label} must be an object")
        _require_keys(label, raw_source, SOURCE_KEYS)
        source_id = _required_string(raw_source, "id", label)
        if SOURCE_ID_PATTERN.fullmatch(source_id) is None:
            raise RegistryPolicyError(f"{label} id is invalid")
        if source_id in sources:
            raise RegistryPolicyError(
                f"registry policy has duplicate source id: {source_id}"
            )
        source_label = _required_string(raw_source, "label", label)
        if not source_label.isprintable():
            raise RegistryPolicyError(f"{label} label is invalid")
        repositories = raw_source["repositories"]
        if not isinstance(repositories, dict):
            raise RegistryPolicyError(
                f"{label} repositories must be an object"
            )
        _require_keys(
            f"{label} repositories",
            repositories,
            REPOSITORY_KEYS,
        )
        parsed_repositories = {
            kind: _repository(
                repositories[kind],
                f"{label} {kind} repository",
            )
            for kind in ("base", "torch")
        }
        if len(set(parsed_repositories.values())) != 2:
            raise RegistryPolicyError(
                f"{label} repositories must be distinct"
            )
        sources[source_id] = RegistrySource(
            source_id,
            source_label,
            parsed_repositories,
        )

    raw_order = payload["default_order"]
    if (
        not isinstance(raw_order, list)
        or not raw_order
        or any(not isinstance(item, str) for item in raw_order)
        or len(set(raw_order)) != len(raw_order)
    ):
        raise RegistryPolicyError(
            "registry policy default_order is invalid"
        )
    order = tuple(raw_order)
    if set(order) != set(sources):
        raise RegistryPolicyError(
            "registry policy default_order must list every source"
        )
    if order[-1] != RegistryChoice.GHCR.value:
        raise RegistryPolicyError(
            "registry policy default_order must end with ghcr"
        )
    if RegistryChoice.SWR.value not in sources:
        raise RegistryPolicyError(
            "registry policy must define swr"
        )
    canonical = sources.get(RegistryChoice.GHCR.value)
    if canonical is None or any(
        not repository.startswith("ghcr.io/")
        for repository in canonical.repositories.values()
    ):
        raise RegistryPolicyError(
            "registry policy ghcr source must use ghcr.io"
        )
    return RegistryPolicy(1, order, sources)


def registry_candidates(
    release: StableRelease,
    choice: RegistryChoice | str = RegistryChoice.AUTO,
    *,
    policy: RegistryPolicy | None = None,
) -> tuple[RegistryCandidate, ...]:
    selected = _registry_choice(choice)
    resolved_policy = policy or load_registry_policy(
        DEFAULT_REGISTRY_POLICY_PATH
    )
    source_names = (
        resolved_policy.default_order
        if selected is RegistryChoice.AUTO
        else (selected.value,)
    )
    canonical = resolved_policy.sources[RegistryChoice.GHCR.value]
    canonical_matches = all(
        getattr(release, kind).image == canonical.repositories[kind]
        for kind in ("base", "torch")
    )
    candidates: list[RegistryCandidate] = []
    for source_name in source_names:
        source = resolved_policy.sources.get(source_name)
        if source is None:
            raise RegistryPolicyError(
                f"registry policy has no source: {source_name}"
            )
        if source_name == RegistryChoice.GHCR.value:
            if any(
                not getattr(release, kind).image.startswith("ghcr.io/")
                for kind in ("base", "torch")
            ):
                raise RegistryPolicyError(
                    "stable release canonical images are not GHCR repositories"
                )
            candidate_release = release
        else:
            if not canonical_matches:
                if selected is not RegistryChoice.AUTO:
                    raise RegistryPolicyError(
                        f"stable release has no trusted "
                        f"{source.label} mapping"
                    )
                continue
            candidate_release = replace(
                release,
                base=replace(
                    release.base,
                    image=source.repositories["base"],
                ),
                torch=replace(
                    release.torch,
                    image=source.repositories["torch"],
                ),
            )
        candidates.append(
            RegistryCandidate(
                source.name,
                source.label,
                candidate_release,
            )
        )
    return tuple(candidates)


def registry_plan_label(
    choice: RegistryChoice | str,
    *,
    policy: RegistryPolicy | None = None,
) -> str:
    selected = _registry_choice(choice)
    resolved_policy = policy or load_registry_policy(
        DEFAULT_REGISTRY_POLICY_PATH
    )
    if selected is RegistryChoice.AUTO:
        preferred = resolved_policy.sources[
            resolved_policy.default_order[0]
        ].label
        canonical = resolved_policy.sources[
            RegistryChoice.GHCR.value
        ].label
        return f"auto（{preferred} 优先，{canonical} 回退）"
    source = resolved_policy.sources.get(selected.value)
    if source is None:
        raise RegistryPolicyError(
            f"registry policy has no source: {selected.value}"
        )
    return f"{selected.value}（仅{source.label}，不回退）"


def trusted_image_references(
    release: StableRelease,
    *,
    kind: str,
    policy: RegistryPolicy | None = None,
) -> tuple[str, ...]:
    if kind not in {"base", "torch"}:
        raise RegistryPolicyError("release image kind is invalid")
    return tuple(
        getattr(candidate.release, kind).reference
        for candidate in registry_candidates(
            release,
            policy=policy,
        )
    )


def _registry_choice(value: RegistryChoice | str) -> RegistryChoice:
    try:
        return RegistryChoice(value)
    except (TypeError, ValueError) as error:
        raise RegistryPolicyError("registry choice is invalid") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise RegistryPolicyError(
                f"registry policy contains duplicate key: {name}"
            )
        result[name] = value
    return result


def _require_keys(
    label: str,
    payload: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    if set(payload) != expected:
        raise RegistryPolicyError(
            f"{label} keys must be exactly {sorted(expected)}"
        )


def _required_string(
    payload: Mapping[str, object],
    name: str,
    label: str,
) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise RegistryPolicyError(f"{label} {name} must be a string")
    return value


def _repository(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or REPOSITORY_PATTERN.fullmatch(value) is None
    ):
        raise RegistryPolicyError(f"{label} is invalid")
    return value
