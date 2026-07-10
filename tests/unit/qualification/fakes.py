from __future__ import annotations

from amd_ai.qualification.models import REQUIRED_CHECKS
from amd_ai.qualification.release import ReleaseInputs


def passing_release_inputs(
    *,
    failed_check: str | None = None,
    git_clean: bool = True,
    profile_status: str = "verified",
    qualification_image_id: str | None = None,
    embedded_locks_match: bool = True,
) -> ReleaseInputs:
    image_id = "sha256:" + "d" * 64
    results = [
        {
            "name": name,
            "passed": name != failed_check,
            "duration_seconds": 1.0,
            "details": {},
            "evidence": "",
        }
        for name in REQUIRED_CHECKS
    ]
    qualification = {
        "schema_version": 1,
        "profile_id": "stable-gfx1151",
        "profile_digest": "a" * 64,
        "image": "rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        "image_id": qualification_image_id or image_id,
        "gpu_arch": "gfx1151",
        "status": "blocked" if failed_check else "pass",
        "required_checks": list(REQUIRED_CHECKS),
        "results": results,
        "generated_at": "2026-07-09T12:00:00Z",
    }
    return ReleaseInputs(
        qualification=qualification,
        qualification_digest="b" * 64,
        profile_digest="a" * 64,
        design_digest="c" * 64,
        image_reference="rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        image_id=image_id,
        repo_digest="example.invalid/amd-ai@sha256:" + "e" * 64,
        image_labels={
            "org.amd-ai.profile.id": "rocm-7.2.1-py3.12-torch-2.9.1",
            "org.amd-ai.profile.status": profile_status,
            "org.amd-ai.rocm.version": "7.2.1",
            "org.amd-ai.torch.version": "2.9.1",
        },
        wheel_hashes={
            "torch": "1" * 64,
            "torchvision": "2" * 64,
            "torchaudio": "3" * 64,
            "triton": "4" * 64,
        },
        rocm_package_lock_digest="5" * 64,
        embedded_rocm_package_lock_digest=(
            "5" * 64 if embedded_locks_match else "8" * 64
        ),
        torch_profile_digest="9" * 64,
        embedded_torch_profile_digest=(
            "9" * 64 if embedded_locks_match else "0" * 64
        ),
        sbom_digest="6" * 64,
        git_revision="7" * 40,
        git_clean=git_clean,
        generated_at="2026-07-09T12:00:00Z",
    )
