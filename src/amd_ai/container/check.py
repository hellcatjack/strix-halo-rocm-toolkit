from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from amd_ai.report import Finding, Report, Severity, Status
from amd_ai.runner import CommandResult, Runner, SubprocessRunner


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

    facts = {
        "rocm_version": version,
        "hipcc": f"{hipcc.stdout}\n{hipcc.stderr}".strip(),
        "kfd": kfd.exists(),
        "render_nodes": [
            "/" + str(path.relative_to(root)) for path in render_nodes
        ],
        "rocminfo": rocminfo_output,
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
    if args.mode != "rocm":
        parser.error("torch mode is not available in this image")

    report = run_rocm_check(
        root=Path("/"),
        runner=SubprocessRunner(),
        metadata_only=args.metadata_only,
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

