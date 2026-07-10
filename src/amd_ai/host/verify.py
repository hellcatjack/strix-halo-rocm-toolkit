from __future__ import annotations

import re
from dataclasses import dataclass

from amd_ai.host.models import HostSnapshot
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.ttm import MemoryConflict, compute_ttm_plan
from amd_ai.report import Finding, Report, Severity, Status
from amd_ai.runner import CommandResult, Runner


GPU_ERROR_PATTERNS = (
    (
        "GPU.MES_TIMEOUT",
        re.compile(r"MES.*(?:timeout|failed to respond)", re.IGNORECASE),
        "The kernel log contains a MES timeout or response failure",
    ),
    (
        "GPU.RESET",
        re.compile(r"GPU reset begin", re.IGNORECASE),
        "The kernel log contains a GPU reset",
    ),
    (
        "GPU.PAGE_FAULT",
        re.compile(r"amdgpu.*page fault", re.IGNORECASE),
        "The kernel log contains an amdgpu page fault",
    ),
    (
        "GPU.FIRMWARE",
        re.compile(r"failed to load firmware", re.IGNORECASE),
        "The kernel log contains a firmware loading failure",
    ),
    (
        "GPU.RING_TIMEOUT",
        re.compile(r"ring .* timeout", re.IGNORECASE),
        "The kernel log contains a GPU ring timeout",
    ),
)


@dataclass(frozen=True)
class ProbeOutcome:
    image: str
    image_id: str | None
    returncode: int | None
    output: str
    findings: tuple[Finding, ...]


def evaluate_post_reboot(snapshot: HostSnapshot) -> Report:
    preflight = evaluate_preflight(snapshot)
    findings = list(preflight.findings)
    statuses = [preflight.status]
    facts = dict(preflight.facts)

    try:
        ttm = compute_ttm_plan(
            mem_total_kib=snapshot.mem_total_kib,
            page_size=snapshot.page_size,
            dmi_memory_bytes=snapshot.dmi_memory_bytes,
        )
    except (MemoryConflict, ValueError) as error:
        findings.append(
            Finding(
                code="HOST.MEMORY_CONFLICT",
                severity=Severity.ERROR,
                summary="The TTM target cannot be computed safely",
                evidence=str(error),
                remediation="Resolve the memory discrepancy or use an explicit reviewed capacity.",
            )
        )
        statuses.append(Status.BLOCKED)
    else:
        facts["expected_ttm_pages_limit"] = ttm.pages_limit
        facts["nominal_memory_gib"] = ttm.nominal_gib
        if snapshot.ttm_pages_limit != ttm.pages_limit:
            findings.append(
                Finding(
                    code="HOST.TTM_MISMATCH",
                    severity=Severity.ERROR,
                    summary="The live TTM page limit does not match the AI Max target",
                    evidence=(
                        f"live={snapshot.ttm_pages_limit!r}, "
                        f"expected={ttm.pages_limit}"
                    ),
                    remediation="Apply the TTM plan, reboot, and run host-verify again.",
                )
            )
            statuses.append(Status.REBOOT_REQUIRED)

    if not snapshot.dmesg_available:
        findings.append(
            Finding(
                code="HOST.DMESG_UNAVAILABLE",
                severity=Severity.ERROR,
                summary="The current-boot kernel log could not be read",
                evidence=snapshot.dmesg or "dmesg returned no evidence",
                remediation="Grant reviewed dmesg access and rerun host-verify.",
            )
        )
        statuses.append(Status.BLOCKED)
    else:
        for code, pattern, summary in GPU_ERROR_PATTERNS:
            evidence = _first_matching_line(snapshot.dmesg, pattern)
            if evidence is None:
                continue
            findings.append(
                Finding(
                    code=code,
                    severity=Severity.ERROR,
                    summary=summary,
                    evidence=evidence,
                    remediation="Inspect the current-boot amdgpu log before running sustained GPU workloads.",
                )
            )
            statuses.append(Status.BLOCKED)

    return Report(
        command="host-verify",
        status=_status_with_precedence(statuses),
        generated_at=preflight.generated_at,
        facts=facts,
        findings=tuple(findings),
    )


def build_probe_argv(*, image: str, device_gids: dict[str, int]) -> list[str]:
    argv = ["docker", "run", "--rm"]
    if "/dev/kfd" in device_gids:
        argv.extend(("--device", "/dev/kfd"))
    if any(path.startswith("/dev/dri/render") for path in device_gids):
        argv.extend(("--device", "/dev/dri"))

    relevant_gids = {
        gid
        for path, gid in device_gids.items()
        if path == "/dev/kfd" or path.startswith("/dev/dri/render")
    }
    for gid in sorted(relevant_gids):
        argv.extend(("--group-add", str(gid)))
    argv.extend(
        (
            image,
            "/usr/local/bin/container-check",
            "--mode",
            "rocm",
            "--json",
            "-",
        )
    )
    return argv


def verify_host(snapshot: HostSnapshot, *, image: str, runner: Runner) -> Report:
    host_report = evaluate_post_reboot(snapshot)
    if host_report.status == Status.BLOCKED:
        outcome = ProbeOutcome(image, None, None, "", ())
    else:
        outcome = _run_container_probe(snapshot, image=image, runner=runner)

    findings = host_report.findings + outcome.findings
    status = Status.BLOCKED if outcome.findings else host_report.status
    facts = dict(host_report.facts)
    facts["probe"] = {
        "image": outcome.image,
        "image_id": outcome.image_id,
        "returncode": outcome.returncode,
        "output": outcome.output,
    }
    return Report(
        command="host-verify",
        status=status,
        generated_at=host_report.generated_at,
        facts=facts,
        findings=findings,
    )


def _run_container_probe(
    snapshot: HostSnapshot,
    *,
    image: str,
    runner: Runner,
) -> ProbeOutcome:
    inspect_args = (
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        image,
    )
    inspect_result = _run_optional(runner, inspect_args)
    if inspect_result.returncode != 0:
        return ProbeOutcome(
            image,
            None,
            None,
            inspect_result.stderr,
            (
                Finding(
                    code="HOST.PROBE_IMAGE_MISSING",
                    severity=Severity.ERROR,
                    summary="The requested ROCm probe image is unavailable",
                    evidence=inspect_result.stderr.strip() or image,
                    remediation="Build or load the exact probe image, then rerun host-verify.",
                ),
            ),
        )

    has_kfd = "/dev/kfd" in snapshot.device_gids
    has_render = any(
        path.startswith("/dev/dri/render") for path in snapshot.device_gids
    )
    if not (has_kfd and has_render):
        return ProbeOutcome(
            image,
            inspect_result.stdout.strip(),
            None,
            "",
            (
                Finding(
                    code="GPU.DEVICE_MAPPING",
                    severity=Severity.ERROR,
                    summary="The complete KFD and DRM device mapping is unavailable",
                    evidence=(f"kfd={has_kfd}, render={has_render}"),
                    remediation="Restore /dev/kfd and a /dev/dri/render* node before running the probe.",
                ),
            ),
        )

    probe_args = tuple(build_probe_argv(image=image, device_gids=snapshot.device_gids))
    probe_result = _run_optional(runner, probe_args)
    output = f"{probe_result.stdout}\n{probe_result.stderr}".strip()
    if probe_result.returncode != 0:
        return ProbeOutcome(
            image,
            inspect_result.stdout.strip(),
            probe_result.returncode,
            output,
            (
                Finding(
                    code="GPU.CONTAINER_PROBE_FAILED",
                    severity=Severity.ERROR,
                    summary="The ROCm container probe exited unsuccessfully",
                    evidence=output or f"exit code {probe_result.returncode}",
                    remediation="Inspect device permissions, the inbox driver, and the image ROCm userspace.",
                ),
            ),
        )
    if not any("gfx1151" in line.strip().lower() for line in output.splitlines()):
        return ProbeOutcome(
            image,
            inspect_result.stdout.strip(),
            probe_result.returncode,
            output,
            (
                Finding(
                    code="GPU.GFX1151_MISSING",
                    severity=Severity.ERROR,
                    summary="The container probe did not report gfx1151",
                    evidence=output or "probe output was empty",
                    remediation="Verify ROCm 7.2.1 userspace and the Radeon 8060S host driver.",
                ),
            ),
        )
    return ProbeOutcome(
        image,
        inspect_result.stdout.strip(),
        probe_result.returncode,
        output,
        (),
    )


def _run_optional(runner: Runner, argv: tuple[str, ...]) -> CommandResult:
    try:
        return runner.run(list(argv), check=False)
    except OSError as error:
        return CommandResult(argv, 127, "", str(error))


def _first_matching_line(text: str, pattern: re.Pattern[str]) -> str | None:
    for line in text.splitlines():
        if pattern.search(line):
            return line.strip()
    return None


def _status_with_precedence(statuses: list[Status]) -> Status:
    for status in (
        Status.BLOCKED,
        Status.REBOOT_REQUIRED,
        Status.CHANGE_REQUIRED,
        Status.UNVERIFIED,
    ):
        if status in statuses:
            return status
    return Status.PASS
