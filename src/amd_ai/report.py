from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Status(StrEnum):
    PASS = "pass"
    CHANGE_REQUIRED = "change-required"
    REBOOT_REQUIRED = "reboot-required"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    summary: str
    evidence: str
    remediation: str


@dataclass(frozen=True)
class Report:
    command: str
    status: Status
    generated_at: str
    facts: dict[str, Any]
    findings: tuple[Finding, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
