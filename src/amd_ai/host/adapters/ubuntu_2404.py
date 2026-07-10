from __future__ import annotations

from dataclasses import dataclass

from amd_ai.host.models import HostSnapshot
from amd_ai.report import Report


@dataclass(frozen=True)
class Ubuntu2404Adapter:
    adapter_id: str = "ubuntu-24.04"

    def matches(self, snapshot: HostSnapshot) -> bool:
        return (
            snapshot.os_id == "ubuntu"
            and snapshot.os_version.startswith("24.04")
            and snapshot.architecture == "x86_64"
        )

    def evaluate(self, snapshot: HostSnapshot) -> Report:
        from amd_ai.host.policy import evaluate_preflight

        return evaluate_preflight(snapshot)

    def create_prepare_plan(
        self,
        snapshot: HostSnapshot,
        target_user: str,
        memory_gib: int | None,
    ):
        from amd_ai.host.prepare import create_ubuntu_prepare_plan

        return create_ubuntu_prepare_plan(snapshot, target_user, memory_gib)

