from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


DIAGNOSTIC_CODES = frozenset(
    {
        "PLATFORM.PASS",
        "RELEASE.INVALID",
        "HOST.PREFLIGHT_FAILED",
        "IMAGE.PARENT_MISSING",
        "IMAGE.DIGEST_DRIFT",
        "IMAGE.PROJECT_CHANGED",
        "PROJECT.CONFIG_INVALID",
        "TORCH.BASE_CHANGED",
        "TORCH.SHADOWED",
        "OVERLAY.LOCK_INVALID",
        "OVERLAY.TRANSACTION_INCOMPLETE",
        "GPU.RUNTIME_FAILED",
        "KERNEL.LOG_FAILED",
    }
)
ALLOWED_ACTION_KINDS = frozenset(
    {
        "quarantine-overlay",
        "remove-project-image",
        "pull-parent",
        "retag-parent",
        "build-project-image",
        "rebuild-overlay",
        "verify-project",
    }
)
SENSITIVE_ENVIRONMENT_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
URL_USERINFO_PATTERN = re.compile(r"(?P<scheme>https?://)[^\s/@]+@")
UNREDACTED_URL_USERINFO_PATTERN = re.compile(
    r"https?://(?!<redacted>@)[^\s/@]+@"
)


class DoctorModelError(ValueError):
    pass


class DiagnosticDisposition(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    REPAIRABLE = "repairable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    disposition: DiagnosticDisposition
    summary: str
    evidence: str
    remediation: str

    def __post_init__(self) -> None:
        if self.code not in DIAGNOSTIC_CODES:
            raise DoctorModelError(f"unknown diagnostic code: {self.code}")
        if not isinstance(self.disposition, DiagnosticDisposition):
            raise DoctorModelError("diagnostic disposition is invalid")
        for name in ("summary", "evidence", "remediation"):
            value = getattr(self, name)
            if not isinstance(value, str) or "\0" in value:
                raise DoctorModelError(f"diagnostic {name} is invalid")


@dataclass(frozen=True)
class RepairAction:
    kind: str
    exact_target: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_ACTION_KINDS:
            raise DoctorModelError(f"unknown repair action kind: {self.kind}")
        if (
            not isinstance(self.exact_target, str)
            or not self.exact_target
            or "\0" in self.exact_target
            or "*" in self.exact_target
            or "?" in self.exact_target
        ):
            raise DoctorModelError("repair action target is not exact")
        if self.reason_code not in DIAGNOSTIC_CODES:
            raise DoctorModelError(
                f"unknown repair reason code: {self.reason_code}"
            )


@dataclass(frozen=True)
class DoctorReport:
    schema_version: int
    generated_at: str
    project: str | None
    facts: Mapping[str, object]
    diagnostics: tuple[Diagnostic, ...]
    status: str

    @classmethod
    def create(
        cls,
        *,
        project: str | Path | None,
        diagnostics: tuple[Diagnostic, ...],
        facts: Mapping[str, object],
        environment: Mapping[str, str] | None = None,
    ) -> DoctorReport:
        values = {} if environment is None else dict(environment)
        safe_diagnostics = tuple(
            replace(
                diagnostic,
                evidence=redact_evidence(diagnostic.evidence, values),
            )
            for diagnostic in diagnostics
        )
        project_path = (
            None
            if project is None
            else str(Path(project).expanduser().resolve(strict=False))
        )
        rank = {
            DiagnosticDisposition.PASS: 0,
            DiagnosticDisposition.WARNING: 1,
            DiagnosticDisposition.REPAIRABLE: 2,
            DiagnosticDisposition.BLOCKED: 3,
        }
        highest = max(
            (diagnostic.disposition for diagnostic in safe_diagnostics),
            key=lambda value: rank[value],
            default=DiagnosticDisposition.PASS,
        )
        return cls(
            schema_version=1,
            generated_at=datetime.now(UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            project=project_path,
            facts=MappingProxyType(dict(facts)),
            diagnostics=safe_diagnostics,
            status=highest.value,
        )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DoctorModelError("doctor report schema is invalid")
        if not self.generated_at.endswith("Z"):
            raise DoctorModelError("doctor report timestamp is not UTC")
        if self.status not in {value.value for value in DiagnosticDisposition}:
            raise DoctorModelError("doctor report status is invalid")
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "project": self.project,
            "status": self.status,
            "facts": dict(self.facts),
            "diagnostics": [
                {
                    **asdict(diagnostic),
                    "disposition": diagnostic.disposition.value,
                }
                for diagnostic in self.diagnostics
            ],
        }


def redact_evidence(
    value: str, environment: Mapping[str, str]
) -> str:
    secrets = sorted(
        {
            secret
            for name, secret in environment.items()
            if secret
            and any(
                token in name.upper()
                for token in SENSITIVE_ENVIRONMENT_TOKENS
            )
        },
        key=len,
        reverse=True,
    )
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    redacted = URL_USERINFO_PATTERN.sub(
        r"\g<scheme><redacted>@", redacted
    )
    if UNREDACTED_URL_USERINFO_PATTERN.search(redacted) is not None or any(
        secret in redacted for secret in secrets
    ):
        raise DoctorModelError("diagnostic evidence contains a secret")
    return redacted
