import hashlib
import json
from pathlib import Path

import pytest

from amd_ai import __version__
from amd_ai.cli import main


def test_version_constant_and_cli(capsys):
    assert __version__ == "0.3.2"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "amd-ai 0.3.2"


def test_installer_only_release_keeps_stable_image_baseline() -> None:
    path = Path("profiles/releases/stable.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "4226d04bf995c9c253c6a978f08bdbb9466ccd47119f967ebd39f0c08b7bfe2d"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["release_id"] == "0.2.0"
    assert payload["base"]["manifest_digest"] == (
        "sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12"
    )
    assert payload["torch"]["manifest_digest"] == (
        "sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b"
    )


def test_oem_617_promotion_has_committed_qualification_evidence() -> None:
    evidence_root = Path("profiles/host/evidence")
    attestation_path = evidence_root / "6.17.0-1028-oem.json"
    report_path = evidence_root / "6.17.0-1028-oem-stable-gfx1151.json"
    report_digest = "856aac21f2667aa27563fe272c0d9c0857d7d45d7739382ad66f7b96134a7bc4"

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == report_digest
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert {
        result["name"] for result in report["results"] if result["passed"]
    } == set(report["required_checks"])

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation == {
        "gpu_driver": "amdgpu",
        "gpu_pci_id": "1002:1586",
        "host_preflight_status": "pass",
        "host_verify_status": "pass",
        "image_id": report["image_id"],
        "kernel": "6.17.0-1028-oem",
        "qualification_report": str(report_path),
        "qualification_report_sha256": report_digest,
        "schema_version": 1,
        "toolkit_revision": "b1c53876933ff0b1be02dfb948820767bf466779",
    }
    tested = json.loads(
        Path("profiles/host/tested-kernels.json").read_text(encoding="utf-8")
    )
    assert tested["kernels"] == [attestation["kernel"]]
    assert tested["source"] == str(attestation_path)
