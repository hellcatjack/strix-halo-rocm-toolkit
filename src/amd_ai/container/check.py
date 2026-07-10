from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from amd_ai.image.profile import ProfileError, load_profile
from amd_ai.report import Finding, Report, Severity, Status
from amd_ai.runner import CommandResult, Runner, SubprocessRunner


TORCH_COMPONENTS = ("torch", "torchvision", "torchaudio", "triton")


def run_rocm_check(*, root: Path, runner: Runner, metadata_only: bool) -> Report:
    findings: list[Finding] = []
    version = _read_rocm_version(root)
    if version is None or not version.startswith("7.2.1"):
        findings.append(
            Finding(
                code="ROCM.VERSION",
                severity=Severity.ERROR,
                summary="The image does not report ROCm 7.2.1",
                evidence=version or "ROCm version metadata is missing",
                remediation="Rebuild from the locked ROCm 7.2.1 package set.",
            )
        )

    hipcc = _run_optional(runner, ("/opt/rocm/bin/hipcc", "--version"))
    if hipcc.returncode != 0:
        findings.append(
            Finding(
                code="ROCM.HIPCC",
                severity=Severity.ERROR,
                summary="hipcc metadata check failed",
                evidence=hipcc.stderr.strip() or f"exit code {hipcc.returncode}",
                remediation="Verify the locked HIP SDK installation.",
            )
        )

    kfd = root / "dev/kfd"
    render_nodes = sorted((root / "dev/dri").glob("renderD*"))
    rocminfo_output = "skipped (metadata-only)"
    rocminfo_architectures: tuple[str, ...] = ()
    if not metadata_only:
        if not kfd.exists():
            findings.append(
                Finding(
                    code="GPU.KFD_MISSING",
                    severity=Severity.ERROR,
                    summary="The container does not have /dev/kfd",
                    evidence="/dev/kfd is absent",
                    remediation="Start the container with --device /dev/kfd.",
                )
            )
        if not render_nodes:
            findings.append(
                Finding(
                    code="GPU.RENDER_MISSING",
                    severity=Severity.ERROR,
                    summary="The container has no DRM render node",
                    evidence="No /dev/dri/render* node is present",
                    remediation="Start the container with --device /dev/dri.",
                )
            )
        if kfd.exists() and render_nodes:
            rocminfo = _run_optional(runner, ("/opt/rocm/bin/rocminfo",))
            rocminfo_output = f"{rocminfo.stdout}\n{rocminfo.stderr}".strip()
            if rocminfo.returncode != 0:
                findings.append(
                    Finding(
                        code="ROCM.ROCINFO",
                        severity=Severity.ERROR,
                        summary="rocminfo failed inside the container",
                        evidence=rocminfo_output or f"exit code {rocminfo.returncode}",
                        remediation="Check device mappings, GIDs, and host amdgpu initialization.",
                    )
                )
            else:
                rocminfo_architectures = _rocminfo_architectures(rocminfo.stdout)
                if not any(
                    architecture.startswith("gfx1151")
                    for architecture in rocminfo_architectures
                ):
                    findings.append(
                        Finding(
                            code="ROCM.GFX1151_MISSING",
                            severity=Severity.ERROR,
                            summary="rocminfo did not discover a gfx1151 GPU agent",
                            evidence=", ".join(rocminfo_architectures) or "no GPU agents",
                            remediation="Check the host amdgpu driver and container device mappings.",
                        )
                    )

    facts = {
        "rocm_version": version,
        "hipcc": f"{hipcc.stdout}\n{hipcc.stderr}".strip(),
        "kfd": kfd.exists(),
        "render_nodes": [
            "/" + str(path.relative_to(root)) for path in render_nodes
        ],
        "rocminfo": rocminfo_output,
        "rocminfo_architectures": list(rocminfo_architectures),
        "metadata_only": metadata_only,
    }
    return Report(
        command="container-check",
        status=Status.BLOCKED if findings else Status.PASS,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        facts=facts,
        findings=tuple(findings),
    )


def public_version(full_version: str) -> str:
    return full_version.split("+", 1)[0]


def run_torch_check(
    *,
    root: Path,
    runner: Runner,
    metadata_only: bool,
    runtime: bool,
) -> Report:
    rocm_report = run_rocm_check(
        root=root,
        runner=runner,
        metadata_only=metadata_only,
    )
    findings = list(rocm_report.findings)
    facts = dict(rocm_report.facts)
    expected_versions = _load_expected_versions(root, findings, facts)
    modules: dict[str, object] = {}
    for name in TORCH_COMPONENTS:
        try:
            modules[name] = importlib.import_module(name)
        except Exception as error:
            findings.append(
                Finding(
                    code="TORCH.IMPORT",
                    severity=Severity.ERROR,
                    summary=f"Failed to import {name}",
                    evidence=f"{type(error).__name__}: {error}",
                    remediation="Rebuild the complete locked Torch profile.",
                )
            )

    for name, expected in expected_versions.items():
        module = modules.get(name)
        if module is None:
            continue
        full = str(getattr(module, "__version__", ""))
        public = public_version(full)
        facts[name] = {"full_version": full, "public_version": public}
        if public != expected:
            findings.append(
                Finding(
                    code="TORCH.VERSION",
                    severity=Severity.ERROR,
                    summary=f"Unexpected {name} public version",
                    evidence=f"expected={expected}, full={full or '<missing>'}",
                    remediation="Use a complete profile with the locked four-component versions.",
                )
            )

    torch = modules.get("torch")
    hip_version = ""
    if torch is not None:
        hip_version = str(getattr(getattr(torch, "version", None), "hip", "") or "")
        if not hip_version:
            findings.append(
                Finding(
                    code="TORCH.HIP_VERSION",
                    severity=Severity.ERROR,
                    summary="PyTorch does not report a ROCm HIP build",
                    evidence="torch.version.hip is empty",
                    remediation="Install the AMD ROCm wheel rather than a PyPI CPU wheel.",
                )
            )
    facts["torch_hip_version"] = hip_version

    manifest_status = "skipped (runtime)"
    if not runtime:
        manifest = root / "opt/amd-ai/torch-manifest.json"
        script = root / "opt/amd-ai/torch-manifest.py"
        result = _run_optional(
            runner,
            (sys.executable, str(script), "verify", str(manifest)),
        )
        manifest_status = f"{result.stdout}\n{result.stderr}".strip() or "pass"
        if result.returncode != 0:
            findings.append(
                Finding(
                    code="TORCH.MANIFEST",
                    severity=Severity.ERROR,
                    summary="Protected Torch distribution files changed",
                    evidence=manifest_status,
                    remediation="Rebuild the project from the verified PyTorch parent image.",
                )
            )
    facts["torch_manifest"] = manifest_status

    run_gpu = runtime or not metadata_only
    if run_gpu and torch is not None:
        _check_torch_gpu(torch, findings, facts)

    return Report(
        command="container-check",
        status=Status.BLOCKED if findings else Status.PASS,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        facts=facts,
        findings=tuple(findings),
    )


def _load_expected_versions(
    root: Path,
    findings: list[Finding],
    facts: dict[str, object],
) -> dict[str, str]:
    path = root / "opt/amd-ai/profile.env"
    try:
        profile = load_profile(path, allow_verified=True)
    except (OSError, ProfileError) as error:
        findings.append(
            Finding(
                code="TORCH.PROFILE",
                severity=Severity.ERROR,
                summary="The embedded Torch profile is invalid",
                evidence=f"{type(error).__name__}: {error}",
                remediation="Rebuild from a complete validated Torch profile.",
            )
        )
        return {}
    facts["torch_profile"] = {
        "id": profile.profile_id,
        "status": profile.status,
    }
    return {name: wheel.version for name, wheel in profile.wheels.items()}


def _check_torch_gpu(
    torch: object,
    findings: list[Finding],
    facts: dict[str, object],
) -> None:
    cuda = getattr(torch, "cuda")
    if not cuda.is_available():
        findings.append(
            Finding(
                code="TORCH.GPU_UNAVAILABLE",
                severity=Severity.ERROR,
                summary="PyTorch cannot access a ROCm GPU",
                evidence="torch.cuda.is_available() returned false",
                remediation="Check /dev mappings, GIDs, the host driver, and ROCm userspace.",
            )
        )
        return
    properties = cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    facts["gpu_architecture"] = architecture
    if not architecture.startswith("gfx1151"):
        findings.append(
            Finding(
                code="TORCH.GFX1151_MISSING",
                severity=Severity.ERROR,
                summary="PyTorch did not expose the Radeon 8060S architecture",
                evidence=architecture or "gcnArchName is empty",
                remediation="Use the qualified gfx1151 host and ROCm profile.",
            )
        )
        return
    try:
        functional = importlib.import_module("torch.nn.functional")
        torch.manual_seed(7)
        left_cpu = torch.randn((1024, 1024), dtype=torch.float32)
        right_cpu = torch.randn((1024, 1024), dtype=torch.float32)
        expected_mm = left_cpu @ right_cpu
        actual_mm = (left_cpu.half().cuda() @ right_cpu.half().cuda()).float().cpu()
        image_cpu = torch.randn((2, 4, 64, 64), dtype=torch.float32)
        kernel_cpu = torch.randn((8, 4, 3, 3), dtype=torch.float32)
        expected_conv = functional.conv2d(image_cpu, kernel_cpu, padding=1)
        actual_conv = functional.conv2d(
            image_cpu.half().cuda(),
            kernel_cpu.half().cuda(),
            padding=1,
        ).float().cpu()
        cuda.synchronize()
        matmul_error = (expected_mm - actual_mm).abs().max().item()
        conv_error = (expected_conv - actual_conv).abs().max().item()
        facts["gpu_smoke"] = {
            "device": cuda.get_device_name(0),
            "architecture": architecture,
            "matmul_max_error": matmul_error,
            "conv_max_error": conv_error,
        }
        if matmul_error > 0.2 or conv_error > 0.2:
            raise RuntimeError(
                f"FP16 error exceeded tolerance: matmul={matmul_error}, conv={conv_error}"
            )
    except Exception as error:
        findings.append(
            Finding(
                code="TORCH.GPU_OPERATION",
                severity=Severity.ERROR,
                summary="The synchronized GPU tensor operation failed",
                evidence=f"{type(error).__name__}: {error}",
                remediation="Inspect ROCm and kernel logs; CPU fallback is not accepted.",
            )
        )


def _rocminfo_architectures(output: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                re.findall(
                    r"^\s*Name:\s*(gfx[0-9a-f]+[^\s]*)\s*$",
                    output,
                    re.IGNORECASE | re.MULTILINE,
                )
            )
        )
    )


def _read_rocm_version(root: Path) -> str | None:
    for relative in ("opt/rocm/.info/version", "opt/rocm/.info/version-dev"):
        path = root / relative
        try:
            value = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError):
            continue
        if value:
            return value
    return None


def _run_optional(runner: Runner, argv: tuple[str, ...]) -> CommandResult:
    try:
        return runner.run(list(argv), check=False)
    except OSError as error:
        return CommandResult(argv, 127, "", str(error))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="container-check")
    parser.add_argument("--mode", choices=("rocm", "torch"), required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args(argv)
    if args.metadata_only and args.runtime:
        parser.error("--metadata-only and --runtime are mutually exclusive")
    runner = SubprocessRunner()
    if args.mode == "rocm":
        report = run_rocm_check(
            root=Path("/"),
            runner=runner,
            metadata_only=args.metadata_only,
        )
    else:
        report = run_torch_check(
            root=Path("/"),
            runner=runner,
            metadata_only=args.metadata_only,
            runtime=args.runtime,
        )
    if args.json_path == "-":
        print(json.dumps(report.to_dict(), sort_keys=True))
    elif args.json_path:
        report.write_json(Path(args.json_path))
    else:
        for finding in report.findings:
            print(f"[{finding.severity.value}] {finding.code}: {finding.summary}")
    return 0 if report.status == Status.PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
