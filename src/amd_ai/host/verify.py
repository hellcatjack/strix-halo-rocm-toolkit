from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from amd_ai.host.models import HostSnapshot
from amd_ai.host.policy import evaluate_preflight
from amd_ai.report import Finding, Report, Severity, Status
from amd_ai.runner import CommandResult, Runner


GPU_ERROR_PATTERNS = (
    (
        "GPU.INIT_FATAL",
        re.compile(
            r"(?:Fatal error during GPU init|amdgpu:\s*probe of .* failed with error)",
            re.IGNORECASE,
        ),
        "The kernel log contains a fatal amdgpu initialization failure",
    ),
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
KERNEL_REQUIRED_PREFLIGHT_CODES = frozenset(
    {
        "HOST.UNSUPPORTED_OS",
        "HOST.UNSUPPORTED_ARCH",
        "HOST.OEM_617_REQUIRED",
        "HOST.OEM_617_CANDIDATE",
        "GPU.NOT_FOUND",
        "GPU.WRONG_DRIVER",
        "GPU.KFD_MISSING",
        "GPU.RENDER_MISSING",
    }
)


@dataclass(frozen=True)
class ProbeOutcome:
    image: str
    image_id: str | None
    returncode: int | None
    output: str
    findings: tuple[Finding, ...]


def evaluate_kernel_reboot(
    snapshot: HostSnapshot,
    *,
    display_manager_was_loaded: bool,
    display_manager_was_active: bool,
) -> Report:
    preflight = evaluate_preflight(snapshot)
    findings = list(preflight.findings)
    facts = dict(preflight.facts)
    blocked = any(
        finding.code in KERNEL_REQUIRED_PREFLIGHT_CODES
        for finding in preflight.findings
    )

    log_findings = _current_boot_gpu_findings(snapshot)
    findings.extend(log_findings)
    blocked = blocked or bool(log_findings)

    facts["kernel_checkpoint"] = {
        "display_manager_was_loaded": display_manager_was_loaded,
        "display_manager_was_active": display_manager_was_active,
        "display_manager_loaded": snapshot.display_manager_loaded,
        "display_manager_active": snapshot.display_manager_active,
    }
    if (display_manager_was_loaded or display_manager_was_active) and not (
        snapshot.display_manager_loaded and snapshot.display_manager_active
    ):
        findings.append(
            Finding(
                code="HOST.DISPLAY_MANAGER_INACTIVE",
                severity=Severity.ERROR,
                summary="The display manager did not recover after the kernel reboot",
                evidence=(
                    f"loaded={snapshot.display_manager_loaded}, "
                    f"active={snapshot.display_manager_active}"
                ),
                remediation=(
                    "Reboot, select the retained recovery kernel under Advanced "
                    "options for Ubuntu, and inspect the amdgpu current-boot log."
                ),
            )
        )
        blocked = True

    return Report(
        command="host-kernel-verify",
        status=Status.BLOCKED if blocked else Status.PASS,
        generated_at=preflight.generated_at,
        facts=facts,
        findings=tuple(findings),
    )


def evaluate_post_reboot(snapshot: HostSnapshot) -> Report:
    preflight = evaluate_preflight(snapshot)
    findings = list(preflight.findings)
    statuses = [preflight.status]
    facts = dict(preflight.facts)

    log_findings = _current_boot_gpu_findings(snapshot)
    findings.extend(log_findings)
    if log_findings:
        statuses.append(Status.BLOCKED)

    return Report(
        command="host-verify",
        status=_status_with_precedence(statuses),
        generated_at=preflight.generated_at,
        facts=facts,
        findings=tuple(findings),
    )


def build_probe_argv(
    *,
    image: str,
    device_gids: dict[str, int],
    docker_prefix: Sequence[str] = ("docker",),
) -> list[str]:
    argv = [*docker_prefix, "run", "--rm"]
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


def verify_host(
    snapshot: HostSnapshot,
    *,
    image: str,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
) -> Report:
    host_report = evaluate_post_reboot(snapshot)
    if host_report.status == Status.BLOCKED:
        outcome = ProbeOutcome(image, None, None, "", ())
    else:
        outcome = _run_container_probe(
            snapshot,
            image=image,
            runner=runner,
            docker_prefix=docker_prefix,
        )

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
    docker_prefix: Sequence[str],
) -> ProbeOutcome:
    inspect_args = (
        *docker_prefix,
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

    probe_args = tuple(
        build_probe_argv(
            image=image,
            device_gids=snapshot.device_gids,
            docker_prefix=docker_prefix,
        )
    )
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


def _current_boot_gpu_findings(snapshot: HostSnapshot) -> tuple[Finding, ...]:
    if not snapshot.dmesg_available:
        return (
            Finding(
                code="HOST.DMESG_UNAVAILABLE",
                severity=Severity.ERROR,
                summary="The current-boot kernel log could not be read",
                evidence=snapshot.dmesg or "dmesg returned no evidence",
                remediation="Grant reviewed dmesg access and rerun host verification.",
            ),
        )

    findings: list[Finding] = []
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
                remediation=(
                    "Inspect the current-boot amdgpu log before running sustained "
                    "GPU workloads."
                ),
            )
        )
    return tuple(findings)


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
