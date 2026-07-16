from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from amd_ai.host.models import HostSnapshot
from amd_ai.report import Report

if TYPE_CHECKING:
    from amd_ai.host.models import HostPlanPhase, PreparePlan


class HostAdapter(Protocol):
    adapter_id: str

    def matches(self, snapshot: HostSnapshot) -> bool: ...

    def evaluate(self, snapshot: HostSnapshot) -> Report: ...

    def create_prepare_plan(
        self,
        snapshot: HostSnapshot,
        target_user: str,
        phase: HostPlanPhase,
    ) -> PreparePlan: ...


def select_adapter(snapshot: HostSnapshot) -> HostAdapter | None:
    from amd_ai.host.adapters.ubuntu_2404 import Ubuntu2404Adapter

    adapter = Ubuntu2404Adapter()
    return adapter if adapter.matches(snapshot) else None
