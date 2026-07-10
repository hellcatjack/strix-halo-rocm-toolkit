# AMD AI Host Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the testable management CLI and safe Ubuntu 24.04 host audit, preparation, TTM configuration, and post-reboot verification workflow for Ryzen AI Max+ 395.

**Architecture:** Thin `bin/` wrappers dispatch to a Python 3.12 standard-library package. Read-only probes produce immutable snapshots and findings; a separate planner converts an eligible snapshot into explicit actions; only the apply layer can execute privileged commands. Host logic accepts injected filesystems and command runners so tests never alter the development machine.

**Tech Stack:** Python 3.12, pytest, Bash, Ubuntu APT/systemd, Docker Engine, Linux sysfs/procfs, `amd-debug-tools` 0.2.19.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, Python floor and pytest configuration |
| `.python-version` | Pin development Python 3.12 |
| `src/amd_ai/__init__.py` | CLI version |
| `src/amd_ai/cli.py` | Argument parsing and command dispatch only |
| `src/amd_ai/runner.py` | Shell-free subprocess abstraction |
| `src/amd_ai/report.py` | Versioned JSON report and finding types |
| `src/amd_ai/host/parsers.py` | Pure parsers for procfs, os-release, DMI, dpkg and lspci text |
| `src/amd_ai/host/models.py` | Immutable host snapshot/action dataclasses |
| `src/amd_ai/host/probe.py` | Read-only host fact collection |
| `src/amd_ai/host/ttm.py` | AI Max memory normalization and TTM calculation |
| `src/amd_ai/host/policy.py` | Ubuntu/gfx1151 support classification |
| `src/amd_ai/host/adapters/base.py` | Distribution adapter protocol and registry |
| `src/amd_ai/host/adapters/ubuntu_2404.py` | Ubuntu 24.04 package, kernel and service policy |
| `src/amd_ai/host/prepare.py` | Safe cleanup/install/change plan generation |
| `src/amd_ai/host/apply.py` | Backup, confirmation and privileged action execution |
| `src/amd_ai/host/verify.py` | Post-reboot and Docker probe validation |
| `bin/_dispatch` | Shared Python path bootstrap |
| `bin/host-preflight` | Read-only preflight wrapper |
| `bin/host-prepare` | Plan/apply wrapper |
| `bin/host-verify` | Post-reboot wrapper |
| `tests/unit/host/` | Pure parser, policy, TTM, planner and apply tests |
| `tests/cli/test_host_commands.py` | End-to-end CLI tests with fixture roots/fake runners |
| `tests/fixtures/host/` | Ubuntu 24.04 healthy, ROCm 6.4 residue and unsupported-host facts |
| `docs/host-operations.md` | Exact audit/apply/reboot/recovery workflow |

### Task 1: Bootstrap the Python package and executable dispatcher

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/amd_ai/__init__.py`
- Create: `src/amd_ai/cli.py`
- Create: `bin/_dispatch`
- Test: `tests/test_version.py`

- [ ] **Step 1: Add the package metadata and failing version test**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"

[project]
name = "amd-ai-container-platform"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = []

[dependency-groups]
dev = [
  "pytest==8.4.1",
  "pytest-cov==6.2.1",
]

[tool.pytest.ini_options]
addopts = "--strict-config --strict-markers"
testpaths = ["tests"]
markers = [
  "container: requires a local Docker daemon and built images",
  "hardware: requires /dev/kfd and the target Radeon 8060S host",
]

[tool.hatch.build.targets.wheel]
packages = ["src/amd_ai"]
```

```text
# .python-version
3.12
```

```python
# tests/test_version.py
import pytest

from amd_ai import __version__
from amd_ai.cli import main


def test_version_constant_and_cli(capsys):
    assert __version__ == "0.1.0"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "amd-ai 0.1.0"
```

Create an empty `src/amd_ai/__init__.py` before `uv sync`; this lets the editable package build while the missing `__version__` import keeps the test red for the intended reason.

- [ ] **Step 2: Run the test to verify it fails before package code exists**

Run:

```bash
uv sync --dev
uv run pytest tests/test_version.py -q
```

Expected: collection fails with `ImportError: cannot import name '__version__' from 'amd_ai'`.

- [ ] **Step 3: Implement the minimal package and dispatcher**

```python
# src/amd_ai/__init__.py
__version__ = "0.1.0"
```

```python
# src/amd_ai/cli.py
from __future__ import annotations

import argparse
from collections.abc import Sequence

from amd_ai import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amd-ai")
    parser.add_argument(
        "--version",
        action="version",
        version=f"amd-ai {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
#!/usr/bin/env bash
# bin/_dispatch
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m amd_ai.cli "$@"
```

Run `chmod +x bin/_dispatch`.

- [ ] **Step 4: Verify the bootstrap test passes**

Run: `uv run pytest tests/test_version.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the package bootstrap**

```bash
git add pyproject.toml uv.lock .python-version src/amd_ai bin/_dispatch tests/test_version.py
git commit -m "chore: bootstrap host management CLI"
```

### Task 2: Add shell-free command execution and report models

**Files:**
- Create: `src/amd_ai/runner.py`
- Create: `src/amd_ai/report.py`
- Test: `tests/unit/test_runner.py`
- Test: `tests/unit/test_report.py`

- [ ] **Step 1: Write failing runner and report tests**

```python
# tests/unit/test_runner.py
import pytest

from amd_ai.runner import CommandError, SubprocessRunner


def test_runner_captures_stdout_without_shell():
    result = SubprocessRunner().run(["printf", "%s", "ok"])
    assert result.args == ("printf", "%s", "ok")
    assert result.stdout == "ok"
    assert result.returncode == 0


def test_runner_raises_typed_error():
    with pytest.raises(CommandError) as error:
        SubprocessRunner().run(["python3", "-c", "import sys; sys.exit(7)"])
    assert error.value.result.returncode == 7
```

```python
# tests/unit/test_report.py
from amd_ai.report import Finding, Report, Severity, Status


def test_report_serializes_stably():
    report = Report(
        command="host-preflight",
        status=Status.CHANGE_REQUIRED,
        generated_at="2026-07-09T12:00:00Z",
        facts={"kernel": "6.17.0-1025-oem"},
        findings=(
            Finding(
                code="HOST.REBOOT",
                severity=Severity.WARNING,
                summary="Reboot required",
                evidence="OEM kernel installed",
                remediation="Reboot and run host-verify",
            ),
        ),
    )
    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert payload["status"] == "change-required"
    assert payload["findings"][0]["code"] == "HOST.REBOOT"
```

- [ ] **Step 2: Run both tests and verify missing modules fail**

Run: `uv run pytest tests/unit/test_runner.py tests/unit/test_report.py -q`

Expected: collection fails for `amd_ai.runner` and `amd_ai.report`.

- [ ] **Step 3: Implement the runner and immutable report types**

```python
# src/amd_ai/runner.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult) -> None:
        super().__init__(f"command failed ({result.returncode}): {' '.join(result.args)}")
        self.result = result


class Runner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            args,
            check=False,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result = CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)
        if check and result.returncode != 0:
            raise CommandError(result)
        return result
```

```python
# src/amd_ai/report.py
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
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_runner.py tests/unit/test_report.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the core types**

```bash
git add src/amd_ai/runner.py src/amd_ai/report.py tests/unit/test_runner.py tests/unit/test_report.py
git commit -m "feat: add command runner and report schema"
```

### Task 3: Implement pure host fact parsers

**Files:**
- Create: `src/amd_ai/host/__init__.py`
- Create: `src/amd_ai/host/parsers.py`
- Test: `tests/unit/host/test_parsers.py`

- [ ] **Step 1: Write parser tests with exact target-host samples**

```python
# tests/unit/host/test_parsers.py
from amd_ai.host.parsers import (
    parse_cmdline,
    parse_dmi_memory_bytes,
    parse_lspci_gpu,
    parse_meminfo,
    parse_os_release,
    parse_vram_mib,
)


def test_parse_target_host_facts():
    assert parse_os_release('ID=ubuntu\nVERSION_ID="24.04"\n') == {
        "ID": "ubuntu",
        "VERSION_ID": "24.04",
    }
    assert parse_meminfo("MemTotal:       131015488 kB\nSwapTotal:              0 kB\n") == {
        "MemTotal": 131015488,
        "SwapTotal": 0,
    }
    args = parse_cmdline(
        "quiet splash amdgpu.gttsize=131072 ttm.pages_limit=33554432 "
        "amdgpu.mcbp=0 amdgpu.gpu_recovery=1 amdgpu.cwsr_enable=0"
    )
    assert args["ttm.pages_limit"] == "33554432"
    assert args["amdgpu.gttsize"] == "131072"


def test_parse_dmi_and_gpu():
    dmi = "Size: 64 GB\nSize: 64 GB\nSize: No Module Installed\n"
    assert parse_dmi_memory_bytes(dmi) == 128 * 1024**3
    gpu = parse_lspci_gpu(
        "c5:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. "
        "Device [1002:1586]\n\tKernel driver in use: amdgpu\n"
    )
    assert gpu.pci_id == "1002:1586"
    assert gpu.driver == "amdgpu"
    assert parse_vram_mib("amdgpu: 512M of VRAM memory ready") == 512
```

- [ ] **Step 2: Run the parser tests and verify imports fail**

Run: `uv run pytest tests/unit/host/test_parsers.py -q`

Expected: collection fails because `amd_ai.host.parsers` does not exist.

- [ ] **Step 3: Implement strict parsers without shell evaluation**

```python
# src/amd_ai/host/parsers.py
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuPciInfo:
    pci_id: str | None
    driver: str | None
    raw: str


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z_()]+):\s+(\d+)\s+kB", line.strip())
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def parse_cmdline(text: str) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {}
    for token in shlex.split(text.strip()):
        key, separator, value = token.partition("=")
        parsed[key] = value if separator else None
    return parsed


def parse_dmi_memory_bytes(text: str) -> int | None:
    total = 0
    matched = False
    multipliers = {"MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for size, unit in re.findall(r"^\s*Size:\s+(\d+)\s+(MB|GB|TB)\s*$", text, re.MULTILINE):
        total += int(size) * multipliers[unit]
        matched = True
    return total if matched else None


def parse_lspci_gpu(text: str) -> GpuPciInfo:
    pci_match = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", text)
    driver_match = re.search(r"Kernel driver in use:\s+(\S+)", text)
    return GpuPciInfo(
        pci_id=pci_match.group(1).lower() if pci_match else None,
        driver=driver_match.group(1) if driver_match else None,
        raw=text,
    )


def parse_vram_mib(text: str) -> int | None:
    matches = re.findall(r"VRAM:\s*(\d+)M|(?:amdgpu:\s*)?(\d+)M(?:iB)?\s+of VRAM", text, re.IGNORECASE)
    values = [first or second for first, second in matches]
    return int(values[-1]) if values else None
```

Create an empty `src/amd_ai/host/__init__.py`.

- [ ] **Step 4: Run parser tests**

Run: `uv run pytest tests/unit/host/test_parsers.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit parsers**

```bash
git add src/amd_ai/host tests/unit/host/test_parsers.py
git commit -m "feat: parse Ubuntu GPU host facts"
```

### Task 4: Implement deterministic AI Max TTM planning

**Files:**
- Create: `src/amd_ai/host/ttm.py`
- Test: `tests/unit/host/test_ttm.py`

- [ ] **Step 1: Write failing tests for 128 GiB, fallback, conflict and override**

```python
# tests/unit/host/test_ttm.py
import pytest

from amd_ai.host.ttm import MemoryConflict, compute_ttm_plan


def test_target_host_maps_124_95_gib_to_128_gib():
    plan = compute_ttm_plan(mem_total_kib=131015488, page_size=4096)
    assert plan.nominal_gib == 128
    assert plan.pages_limit == 33554432
    assert plan.legacy_gttsize_mib == 131072
    assert plan.source == "meminfo"


def test_dmi_is_preferred_when_consistent():
    plan = compute_ttm_plan(
        mem_total_kib=131015488,
        page_size=4096,
        dmi_memory_bytes=128 * 1024**3,
    )
    assert plan.source == "dmi"


def test_large_dmi_disagreement_requires_override():
    with pytest.raises(MemoryConflict):
        compute_ttm_plan(
            mem_total_kib=63 * 1024**2,
            page_size=4096,
            dmi_memory_bytes=128 * 1024**3,
        )


def test_explicit_capacity_is_auditable():
    plan = compute_ttm_plan(mem_total_kib=63 * 1024**2, page_size=4096, explicit_gib=64)
    assert plan.nominal_gib == 64
    assert plan.source == "explicit"
```

- [ ] **Step 2: Run and verify the missing implementation fails**

Run: `uv run pytest tests/unit/host/test_ttm.py -q`

Expected: collection fails for `amd_ai.host.ttm`.

- [ ] **Step 3: Implement the exact normalization formula**

```python
# src/amd_ai/host/ttm.py
from __future__ import annotations

import math
from dataclasses import dataclass


GIB = 1024**3
KIB = 1024


class MemoryConflict(ValueError):
    pass


@dataclass(frozen=True)
class TtmPlan:
    nominal_gib: int
    page_size: int
    pages_limit: int
    legacy_gttsize_mib: int
    source: str


def _normalize_gib(byte_count: int) -> int:
    return math.ceil((byte_count / GIB) / 8) * 8


def compute_ttm_plan(
    *,
    mem_total_kib: int,
    page_size: int,
    dmi_memory_bytes: int | None = None,
    explicit_gib: int | None = None,
) -> TtmPlan:
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("page size must be a positive power of two")
    meminfo_gib = _normalize_gib(mem_total_kib * KIB)
    if explicit_gib is not None:
        if explicit_gib <= 0 or explicit_gib % 8:
            raise ValueError("explicit capacity must be a positive multiple of 8 GiB")
        nominal_gib, source = explicit_gib, "explicit"
    elif dmi_memory_bytes is not None:
        dmi_gib = _normalize_gib(dmi_memory_bytes)
        if abs(dmi_gib - meminfo_gib) > 8:
            raise MemoryConflict(f"DMI={dmi_gib} GiB, MemTotal={meminfo_gib} GiB")
        nominal_gib, source = dmi_gib, "dmi"
    else:
        nominal_gib, source = meminfo_gib, "meminfo"
    return TtmPlan(
        nominal_gib=nominal_gib,
        page_size=page_size,
        pages_limit=(nominal_gib * GIB) // page_size,
        legacy_gttsize_mib=nominal_gib * 1024,
        source=source,
    )
```

- [ ] **Step 4: Run focused TTM tests**

Run: `uv run pytest tests/unit/host/test_ttm.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit TTM planning**

```bash
git add src/amd_ai/host/ttm.py tests/unit/host/test_ttm.py
git commit -m "feat: calculate AI Max TTM limits"
```

### Task 5: Collect a read-only host snapshot

**Files:**
- Create: `src/amd_ai/host/models.py`
- Create: `src/amd_ai/host/probe.py`
- Create: `tests/unit/host/fakes.py`
- Test: `tests/unit/host/test_probe.py`
- Create fixtures under: `tests/fixtures/host/healthy/`

- [ ] **Step 1: Build a fixture root and write the failing collection test**

Fixture files must contain the observed host values:

```text
tests/fixtures/host/healthy/etc/os-release
ID=ubuntu
VERSION_ID="24.04"

tests/fixtures/host/healthy/proc/meminfo
MemTotal:       131015488 kB
SwapTotal:              0 kB

tests/fixtures/host/healthy/proc/cmdline
quiet splash amdgpu.gttsize=131072 ttm.pages_limit=33554432 amdgpu.mcbp=0 amdgpu.gpu_recovery=1 amdgpu.cwsr_enable=0

tests/fixtures/host/healthy/sys/module/ttm/parameters/pages_limit
33554432
```

```python
# tests/unit/host/test_probe.py
from pathlib import Path

from amd_ai.host.probe import HostProbe
from tests.unit.host.fakes import FakeRunner


def test_probe_collects_target_snapshot():
    runner = FakeRunner.healthy_target()
    snapshot = HostProbe(
        root=Path("tests/fixtures/host/healthy"),
        runner=runner,
        device_gids={"/dev/kfd": 0, "/dev/dri/renderD128": 128},
    ).collect()
    assert snapshot.os_id == "ubuntu"
    assert snapshot.kernel == "6.17.0-1025-oem"
    assert snapshot.gpu.pci_id == "1002:1586"
    assert snapshot.gpu.driver == "amdgpu"
    assert snapshot.mem_total_kib == 131015488
    assert snapshot.ttm_pages_limit == 33554432
    assert snapshot.device_gids == {"/dev/kfd": 0, "/dev/dri/renderD128": 128}
```

- [ ] **Step 2: Run and verify snapshot imports fail**

Run: `uv run pytest tests/unit/host/test_probe.py -q`

Expected: collection fails because `amd_ai.host.probe` is absent.

- [ ] **Step 3: Implement immutable models, fake runner and probe**

Define `HostSnapshot` in `models.py` with these exact fields and types:

```python
from dataclasses import dataclass

from amd_ai.host.parsers import GpuPciInfo


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    version: str
    origin: str | None = None


@dataclass(frozen=True)
class HostSnapshot:
    os_id: str
    os_version: str
    architecture: str
    kernel: str
    gpu: GpuPciInfo
    mem_total_kib: int
    swap_total_kib: int
    page_size: int
    kernel_args: dict[str, str | None]
    ttm_pages_limit: int | None
    dmi_memory_bytes: int | None
    device_gids: dict[str, int]
    current_group_ids: tuple[int, ...]
    packages: tuple[InstalledPackage, ...]
    apt_sources: tuple[str, ...]
    dkms_status: str
    docker_version: str | None
    dmesg: str
    dedicated_vram_mib: int | None
```

`HostProbe.collect()` must perform only these reads/commands, all with `check=False` where absence is a fact rather than an exception:

```python
commands = {
    "architecture": ["uname", "-m"],
    "kernel": ["uname", "-r"],
    "gpu": ["lspci", "-Dnnk", "-d", "1002:1586"],
    "dmi": ["dmidecode", "--type", "memory"],
    "packages": ["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"],
    "dkms": ["dkms", "status"],
    "docker": ["docker", "version", "--format", "{{.Server.Version}}"],
    "dmesg": ["dmesg", "--level=err,warn"],
    "page_size": ["getconf", "PAGESIZE"],
}
```

The probe reads APT source files under `etc/apt/sources.list` and `etc/apt/sources.list.d/*`, ignores unreadable optional files, and takes real device GIDs from `os.stat` unless the test supplies `device_gids`. For installed ROCm-prefix packages and `amdgpu-dkms`, query `apt-cache policy <package>` and retain a `repo.radeon.com` origin line in `InstalledPackage.origin`; strip a dpkg architecture suffix such as `:amd64` from the package name. The fake runner maps the commands above to deterministic `CommandResult` objects and raises on an unregistered command.

Fixture mode additionally reads `tests/fixtures/host/healthy/commands.json` and `devices.json`. `commands.json` maps the exact argv joined with NUL characters to return code/stdout/stderr; `devices.json` maps `/dev/kfd` and `/dev/dri/renderD128` to numeric GIDs. This makes the final fixture CLI command independent of the machine running the tests.

- [ ] **Step 4: Run probe and parser tests**

Run: `uv run pytest tests/unit/host/test_probe.py tests/unit/host/test_parsers.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit host snapshot collection**

```bash
git add src/amd_ai/host/models.py src/amd_ai/host/probe.py tests/unit/host tests/fixtures/host/healthy
git commit -m "feat: collect read-only host snapshot"
```

### Task 6: Classify support and expose `host-preflight`

**Files:**
- Create: `src/amd_ai/host/policy.py`
- Create: `src/amd_ai/host/adapters/__init__.py`
- Create: `src/amd_ai/host/adapters/base.py`
- Create: `src/amd_ai/host/adapters/ubuntu_2404.py`
- Create: `profiles/host/tested-kernels.json`
- Modify: `src/amd_ai/cli.py`
- Create: `bin/host-preflight`
- Test: `tests/unit/host/test_policy.py`
- Test: `tests/cli/test_host_commands.py`

- [ ] **Step 1: Write policy and CLI tests**

```python
# tests/unit/host/test_policy.py
from amd_ai.host.policy import evaluate_preflight
from tests.unit.host.fakes import healthy_snapshot


def test_healthy_gfx1151_oem_host_passes():
    report = evaluate_preflight(healthy_snapshot(kernel="6.14.0-1018-oem"))
    assert report.status.value == "pass"
    assert not [finding for finding in report.findings if finding.severity.value == "error"]


def test_newer_unrecorded_oem_kernel_is_explicitly_unverified():
    report = evaluate_preflight(healthy_snapshot(kernel="6.17.0-1025-oem"))
    assert report.status.value == "unverified"
    assert any(finding.code == "HOST.UPSTREAM_UNVERIFIED" for finding in report.findings)


def test_unknown_distribution_is_blocked_for_apply():
    snapshot = healthy_snapshot(os_id="fedora", os_version="42")
    report = evaluate_preflight(snapshot)
    assert report.status.value == "blocked"
    assert any(finding.code == "HOST.UNSUPPORTED_OS" for finding in report.findings)


def test_unknown_distribution_has_no_write_adapter():
    from amd_ai.host.adapters.base import select_adapter

    assert select_adapter(healthy_snapshot(os_id="fedora", os_version="42")) is None
```

```python
# tests/cli/test_host_commands.py
from amd_ai.cli import main


def test_preflight_fixture_writes_json(tmp_path):
    output = tmp_path / "preflight.json"
    code = main([
        "host-preflight",
        "--fixture-root",
        "tests/fixtures/host/healthy",
        "--json",
        str(output),
    ])
    assert code == 0
    assert '"command": "host-preflight"' in output.read_text()
```

- [ ] **Step 2: Run and verify the new command fails**

Run: `uv run pytest tests/unit/host/test_policy.py tests/cli/test_host_commands.py -q`

Expected: failures for the missing policy and unrecognized `host-preflight` command.

- [ ] **Step 3: Implement support policy and CLI wiring**

`evaluate_preflight()` must add exact findings for these predicates:

```text
HOST.UNSUPPORTED_OS      ID != ubuntu or VERSION_ID not starting with 24.04; blocked
HOST.UNSUPPORTED_ARCH    uname -m not x86_64; blocked
HOST.OEM_KERNEL          kernel does not end in -oem or is older than 6.14.0-1018; blocked
GPU.NOT_FOUND            PCI ID 1002:1586 absent; blocked
GPU.WRONG_DRIVER         kernel driver is not amdgpu; blocked
GPU.KFD_MISSING          /dev/kfd absent; change-required
GPU.RENDER_MISSING       no /dev/dri/render* node; change-required
GPU.PERMISSION           current groups do not cover device GIDs; change-required
HOST.SWAP_DISABLED       SwapTotal is zero; informational warning only
GPU.BIOS_VRAM_HIGH       parsed dedicated VRAM is greater than 512 MiB; change-required with BIOS-only remediation
HOST.UPSTREAM_UNVERIFIED OEM kernel meets minimum but is outside the recorded AMD-tested patch set; unverified
```

Kernel comparison uses `re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-(\d+)-oem", kernel)` and compares the four integers to `(6, 14, 0, 1018)`. The current `6.17.0-1025-oem` passes the minimum and is reported as upstream-unverified unless included in a versioned tested-kernel data file later by the release plan. Status precedence is `blocked`, `reboot-required`, `change-required`, `unverified`, then `pass`.

Create `profiles/host/tested-kernels.json` with schema version 1, kernel list containing only `6.14.0-1018-oem`, and the AMD ROCm 7.2.1 Ryzen documentation URL as its source. Load this versioned file for `HOST.UPSTREAM_UNVERIFIED`; adding another kernel is a reviewed release change, never an automatic side effect of running qualification.

Define a `HostAdapter` protocol with immutable `adapter_id`, `matches(snapshot)`, `evaluate(snapshot)`, and `create_prepare_plan(snapshot, target_user, memory_gib)` members. Register only `Ubuntu2404Adapter(adapter_id="ubuntu-24.04")`. `select_adapter()` returns it only for `ID=ubuntu`, `VERSION_ID` beginning `24.04`, and `x86_64`; all other hosts retain common read-only facts but receive no write adapter. Task 7 places every APT, OEM-kernel, systemd and source-file action behind this adapter.

Extend `cli.py` with a subparser named `host-preflight`. It accepts `--json PATH` and hidden test-only `--fixture-root PATH`; production defaults to `/`. It prints one line per finding and returns 2 for `blocked`, 1 for `change-required`/`reboot-required`, and 0 for `pass`/`unverified`.

```bash
#!/usr/bin/env bash
# bin/host-preflight
set -euo pipefail
exec "$(dirname "$0")/_dispatch" host-preflight "$@"
```

Run `chmod +x bin/host-preflight`.

- [ ] **Step 4: Run policy and CLI tests**

Run: `uv run pytest tests/unit/host/test_policy.py tests/cli/test_host_commands.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit preflight**

```bash
git add src/amd_ai/host/policy.py src/amd_ai/host/adapters profiles/host/tested-kernels.json src/amd_ai/cli.py bin/host-preflight tests/unit/host/test_policy.py tests/cli/test_host_commands.py
git commit -m "feat: add Ubuntu host preflight"
```

### Task 7: Generate a safe Ubuntu preparation plan

**Files:**
- Create: `src/amd_ai/host/prepare.py`
- Add fixture: `tests/fixtures/host/rocm64-residue/`
- Test: `tests/unit/host/test_prepare.py`

- [ ] **Step 1: Write failing cleanup and install-plan tests**

```python
# tests/unit/host/test_prepare.py
from amd_ai.host.models import InstalledPackage
from amd_ai.host.prepare import cleanup_candidates, create_prepare_plan
from tests.unit.host.fakes import healthy_snapshot


def test_cleanup_only_selects_rocm_64_and_amdgpu_dkms():
    packages = (
        InstalledPackage("rocm-core", "6.4.43483-1", "repo.radeon.com/rocm/apt/6.4"),
        InstalledPackage("hip-runtime-amd", "6.4.43483-1", "repo.radeon.com/rocm/apt/6.4"),
        InstalledPackage("amdgpu-dkms", "6.12.12.60402-1", "repo.radeon.com/graphics/6.4"),
        InstalledPackage("dkms", "3.0.11-1ubuntu13"),
        InstalledPackage("zfs-dkms", "2.2.2-0ubuntu9"),
    )
    assert cleanup_candidates(packages) == (
        "amdgpu-dkms",
        "hip-runtime-amd",
        "rocm-core",
    )


def test_plan_never_removes_generic_dkms():
    plan = create_prepare_plan(healthy_snapshot(with_rocm64=True), target_user="customer")
    flattened = " ".join(arg for action in plan.actions for arg in action.argv)
    assert "amdgpu-dkms" in flattened
    assert " zfs-dkms " not in f" {flattened} "
    assert " autoremove " not in f" {flattened} "
    assert plan.reboot_required is True
```

- [ ] **Step 2: Run and verify planning fails before implementation**

Run: `uv run pytest tests/unit/host/test_prepare.py -q`

Expected: collection fails for `amd_ai.host.prepare`.

- [ ] **Step 3: Implement action types, cleanup whitelist and ordered plan**

Add these dataclasses to `models.py`:

```python
@dataclass(frozen=True)
class PlannedAction:
    code: str
    summary: str
    argv: tuple[str, ...]
    privileged: bool
    input_text: str | None = None


@dataclass(frozen=True)
class PreparePlan:
    supported: bool
    target_user: str
    actions: tuple[PlannedAction, ...]
    reboot_required: bool
```

Implement `cleanup_candidates()` with this exact rule: select `amdgpu-dkms`; additionally select a package only when its version begins with `6.4`, its recorded origin contains `repo.radeon.com`, and its name begins with one of `rocm`, `hip-`, `hsa-`, `hsakmt`, `comgr`, `miopen`, `rocblas`, `rocfft`, `rocrand`, `rocsolver`, `rocsparse`, `rccl`. Sort and deduplicate. Never select `dkms`, `zfs-dkms`, `virtualbox-dkms` or names that merely contain `amd`.

`create_prepare_plan()` delegates to the selected `Ubuntu2404Adapter`, must refuse when no adapter exists, and emits actions in this order:

```text
BACKUP.SNAPSHOT
APT.DISABLE_OLD_ROCM_SOURCES
APT.REMOVE_OLD_ROCM_PACKAGES
APT.INSTALL_OEM_KERNEL
APT.INSTALL_HOST_TOOLS
DOCKER.INSTALL_IF_MISSING
GROUPS.ADD_DEVICE_GROUPS
TTM.INSTALL_AMD_DEBUG_TOOLS
TTM.SET_AI_MAX
HOST.REBOOT (only with an explicit --reboot at apply time)
```

Use argv arrays, not shell strings. The kernel action is exactly:

```python
("apt-get", "install", "-y", "linux-oem-24.04", "linux-headers-oem-24.04", "linux-firmware")
```

The host-tools action installs `ca-certificates`, `curl`, `gnupg`, `pciutils`, `python3-pip`, and `pipx`. The group action derives names with `grp.getgrgid()` from actual device GIDs and uses `usermod -a -G comma,separated target_user` only when groups are missing.

Disable only source files containing a noncomment `repo.radeon.com` line whose URL path matches `/6\.4(?:[./]|$)`; rename each to `<name>.amd-ai-disabled` after backup. The package-removal action is `apt-get remove --purge -y <sorted candidates>` and is omitted when the list is empty. Internal actions such as backup/source rename use an empty argv and a handler selected by the fixed action code; unknown internal action codes are blocking errors.

- [ ] **Step 4: Run planner tests**

Run: `uv run pytest tests/unit/host/test_prepare.py -q`

Expected: all tests pass and no generated argv contains `autoremove`.

- [ ] **Step 5: Commit preparation planning**

```bash
git add src/amd_ai/host/models.py src/amd_ai/host/prepare.py tests/unit/host/test_prepare.py tests/fixtures/host/rocm64-residue
git commit -m "feat: plan safe Ubuntu host preparation"
```

### Task 8: Add backups, pinned `amd-ttm`, Docker setup and privileged apply

**Files:**
- Create: `src/amd_ai/host/apply.py`
- Modify: `src/amd_ai/cli.py`
- Create: `bin/host-prepare`
- Test: `tests/unit/host/test_apply.py`
- Extend: `tests/cli/test_host_commands.py`

- [ ] **Step 1: Write failing dry-run, backup and confirmation tests**

```python
# tests/unit/host/test_apply.py
from pathlib import Path

import pytest

from amd_ai.host.apply import ApplyRefused, backup_host_state, execute_plan, ttm_input_text
from tests.unit.host.fakes import FakeRunner, healthy_snapshot, prepare_plan


def test_backup_copies_config_and_records_commands(tmp_path):
    backup = backup_host_state(
        snapshot=healthy_snapshot(),
        destination=tmp_path,
        root=Path("tests/fixtures/host/healthy"),
        runner=FakeRunner.healthy_target(),
        timestamp="20260709T120000Z",
    )
    assert (backup / "manifest.json").is_file()
    assert (backup / "proc/cmdline").is_file()


def test_execute_requires_root_and_confirmation():
    with pytest.raises(ApplyRefused, match="root"):
        execute_plan(prepare_plan(), FakeRunner(), effective_uid=1000, confirmed=True)
    with pytest.raises(ApplyRefused, match="confirmation"):
        execute_plan(prepare_plan(), FakeRunner(), effective_uid=0, confirmed=False)


def test_ai_max_accepts_memory_warning_but_declines_reboot():
    assert ttm_input_text(nominal_gib=128, mem_total_kib=131015488) == "y\nn\n"
    assert ttm_input_text(nominal_gib=100, mem_total_kib=131015488) == "n\n"
```

- [ ] **Step 2: Run and verify apply tests fail**

Run: `uv run pytest tests/unit/host/test_apply.py -q`

Expected: collection fails for `amd_ai.host.apply`.

- [ ] **Step 3: Implement backup and exact privileged execution rules**

`backup_host_state()` writes under `/var/backups/amd-ai/<UTC timestamp>/` by default. It copies, when present, `/etc/os-release`, `/etc/default/grub`, `/etc/modprobe.d/ttm.conf`, `/etc/apt/sources.list`, all `/etc/apt/sources.list.d/*`, all `/etc/apt/preferences.d/*`, `/proc/cmdline`, and `/sys/module/ttm/parameters/pages_limit`. It stores JSON command outputs for `dpkg-query`, `dkms status`, `uname -a`, `lspci -Dnnk -d 1002:1586`, and `docker version`. It creates files with mode `0600` and directories with `0700`.

The pinned AMD tool installation action runs:

```text
python3 -m pip download --no-deps --only-binary=:all: --dest /var/cache/amd-ai amd-debug-tools==0.2.19
sha256sum verification against 7c77875c2fa71c8f10d151bbd24583fd8f16f3fa31dca7cd38026f1e1768bb2f
env PIPX_HOME=/opt/amd-ai/pipx PIPX_BIN_DIR=/usr/local/bin pipx install --force /var/cache/amd-ai/amd_debug_tools-0.2.19-py3-none-any.whl
```

Do not install the wheel until the SHA-256 comparison passes. If the live/configured page limit already equals the computed target, omit `TTM.SET_AI_MAX`. Otherwise run `/usr/local/bin/amd-ttm --set <nominal_gib>`. When the request exceeds 90% of visible `MemTotal`, pass `y\nn\n`: `y` explicitly accepts the AI Max allocation warning and `n` declines the tool's reboot prompt. For lower values pass only `n\n`. The wrapper owns reboot policy.

After `amd-ttm`, parse `/etc/modprobe.d/ttm.conf` and require the expected page count. If version 0.2.19 rejects only because the normalized nominal tier is slightly above visible `MemTotal`, write the same setting atomically using `options ttm pages_limit=<pages_limit>` when `/sys/module/ttm` exists, or `options amdttm pages_limit=<pages_limit>` when only `/sys/module/amdttm` exists; any other failure stops apply. Preserve existing `amdgpu.gttsize`, `amdgpu.cwsr_enable`, `amdgpu.mcbp`, and `amdgpu.gpu_recovery` command-line arguments unchanged, and never add legacy `amdgpu.gttsize` during this fallback.

When Docker is absent, download the official Docker Ubuntu signing key and require fingerprint `9DC858229FC7DD38854AE2D88D81803C0EBFCD88`. Write `/etc/apt/sources.list.d/docker.sources` with `Types: deb`, `URIs: https://download.docker.com/linux/ubuntu`, `Suites: noble`, `Components: stable`, `Architectures: amd64`, and `Signed-By: /etc/apt/keyrings/docker.asc`; then install `docker-ce`, `docker-ce-cli`, `containerd.io`, `docker-buildx-plugin`, and `docker-compose-plugin`, and run `systemctl enable --now docker`.

`execute_plan()` enforces effective UID 0, explicit confirmation, a successful backup path, and sequential stop-on-error. It never executes the `HOST.REBOOT` action unless the caller supplied `--reboot`.

- [ ] **Step 4: Wire `host-prepare` CLI and wrapper**

The CLI grammar is:

```text
host-prepare plan [--target-user USER] [--memory-gib N] [--json PATH]
host-prepare apply [--target-user USER] [--memory-gib N] [--yes] [--reboot] [--json PATH]
```

Without `--yes`, print the action codes and require the exact response `APPLY`. Before adding the target user to `docker`, print `docker group grants root-equivalent daemon control` and require the separate response `ADD_DOCKER_GROUP`; otherwise retain `sudo docker` operation. `--yes` acknowledges the action plan but does not bypass the Docker-group-specific response.

```bash
#!/usr/bin/env bash
# bin/host-prepare
set -euo pipefail
exec "$(dirname "$0")/_dispatch" host-prepare "$@"
```

Run `chmod +x bin/host-prepare`.

- [ ] **Step 5: Run apply and CLI tests**

Run: `uv run pytest tests/unit/host/test_apply.py tests/cli/test_host_commands.py -q`

Expected: all tests pass; fake runner records ordered argv arrays and no real host command runs.

- [ ] **Step 6: Commit host apply support**

```bash
git add src/amd_ai/host/apply.py src/amd_ai/cli.py bin/host-prepare tests/unit/host/test_apply.py tests/cli/test_host_commands.py
git commit -m "feat: safely apply Ubuntu host preparation"
```

### Task 9: Implement post-reboot and container-probe verification

**Files:**
- Create: `src/amd_ai/host/verify.py`
- Modify: `src/amd_ai/cli.py`
- Create: `bin/host-verify`
- Test: `tests/unit/host/test_verify.py`
- Extend: `tests/cli/test_host_commands.py`

- [ ] **Step 1: Write failing verification tests**

```python
# tests/unit/host/test_verify.py
from amd_ai.host.verify import build_probe_argv, evaluate_post_reboot
from tests.unit.host.fakes import healthy_snapshot


def test_probe_uses_devices_and_actual_gids():
    argv = build_probe_argv(
        image="rocm-python:7.2.1-py3.12",
        device_gids={"/dev/kfd": 109, "/dev/dri/renderD128": 110},
    )
    assert argv.count("--device") == 2
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--group-add"] == ["109", "110"]
    assert "--privileged" not in argv
    assert "--ipc=host" not in argv


def test_kernel_gpu_error_blocks_verification():
    report = evaluate_post_reboot(
        healthy_snapshot(dmesg="amdgpu: MES failed to respond to msg=REMOVE_QUEUE")
    )
    assert report.status.value == "blocked"
    assert any(finding.code == "GPU.MES_TIMEOUT" for finding in report.findings)
```

- [ ] **Step 2: Run and verify missing verification code fails**

Run: `uv run pytest tests/unit/host/test_verify.py -q`

Expected: collection fails for `amd_ai.host.verify`.

- [ ] **Step 3: Implement host verification and probe command**

`evaluate_post_reboot()` reuses preflight policy and additionally requires the live TTM page count to equal the computed plan. It scans `dmesg` case-insensitively for these patterns and emits blocking findings:

```text
MES.*(timeout|failed to respond)
GPU reset begin
amdgpu.*page fault
failed to load firmware
ring .* timeout
```

`build_probe_argv()` returns this shape, with sorted unique GIDs:

```text
docker run --rm
  --device /dev/kfd
  --device /dev/dri
  --group-add <actual-kfd-gid>
  --group-add <actual-render-gid>
  rocm-python:7.2.1-py3.12
  /usr/local/bin/container-check --mode rocm --json -
```

The real argv is a flat Python list and never passes through a shell. Missing image returns `HOST.PROBE_IMAGE_MISSING`; missing device mapping returns `GPU.DEVICE_MAPPING`; `rocminfo` output without a line whose stripped value contains `gfx1151` returns `GPU.GFX1151_MISSING`.

- [ ] **Step 4: Wire CLI and wrapper**

`host-verify` accepts `--probe-image`, defaulting to `rocm-python:7.2.1-py3.12`, plus `--json PATH`. It returns 0 only when host checks and the container probe pass.

```bash
#!/usr/bin/env bash
# bin/host-verify
set -euo pipefail
exec "$(dirname "$0")/_dispatch" host-verify "$@"
```

Run `chmod +x bin/host-verify`.

- [ ] **Step 5: Run verification tests**

Run: `uv run pytest tests/unit/host/test_verify.py tests/cli/test_host_commands.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit verification support**

```bash
git add src/amd_ai/host/verify.py src/amd_ai/cli.py bin/host-verify tests/unit/host/test_verify.py tests/cli/test_host_commands.py
git commit -m "feat: verify rebooted gfx1151 host"
```

### Task 10: Document operations and run the host-plan gate

**Files:**
- Create: `docs/host-operations.md`
- Create: `README.md`
- Test: all host tests

- [ ] **Step 1: Write the operations document**

Document these exact sequences and safety boundaries:

```bash
./bin/host-preflight --json reports/preflight.json
sudo ./bin/host-prepare plan --target-user "$USER"
sudo ./bin/host-prepare apply --target-user "$USER"
sudo reboot
./bin/image-build rocm-python
./bin/host-verify --probe-image rocm-python:7.2.1-py3.12 --json reports/host-verify.json
```

Include: BIOS UMA target 0.5 GiB; TTM is a maximum rather than eager reservation; no automatic kernel rollback; backup location; no generic `dkms` removal; Docker-group security; `--reboot` opt-in; how to restore `/etc/modprobe.d/ttm.conf` and GRUB backups; and how to run read-only preflight on unsupported distributions.

- [ ] **Step 2: Run the complete host test suite**

Run:

```bash
uv run pytest tests/unit/host tests/cli/test_host_commands.py tests/test_version.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run static command checks**

Run:

```bash
bash -n bin/_dispatch bin/host-preflight bin/host-prepare bin/host-verify
uv run python -m compileall -q src/amd_ai
./bin/host-preflight --fixture-root tests/fixtures/host/healthy --json /tmp/amd-ai-preflight.json
```

Expected: all commands exit 0 and `/tmp/amd-ai-preflight.json` contains schema version 1.

- [ ] **Step 4: Commit host documentation**

```bash
git add docs/host-operations.md README.md
git commit -m "docs: add host preparation runbook"
```
