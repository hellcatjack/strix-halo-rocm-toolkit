from __future__ import annotations

from amd_ai.doctor.checks import DoctorImageInspection, ProjectImageInspection
from amd_ai.installer.models import StableRelease


class FakeDoctorBackend:
    def __init__(self, release: StableRelease) -> None:
        labels = {
            "org.opencontainers.image.source": release.source_repository,
            "org.opencontainers.image.revision": release.source_revision,
            "org.amd-ai.rocm.version": release.rocm_version,
            "org.amd-ai.python.version": release.python_version,
        }
        self.images = {
            release.base.reference: DoctorImageInspection(
                release.base.config_digest, labels, (release.base.reference,)
            ),
            release.torch.reference: DoctorImageInspection(
                release.torch.config_digest,
                {
                    **labels,
                    "org.amd-ai.profile.id": release.torch_profile_id,
                    "org.amd-ai.profile.status": "verified",
                    "org.amd-ai.torch.version": release.torch_version,
                },
                (release.torch.reference,),
            ),
        }
        self.friendly = {
            "rocm-python:7.2.1-py3.12": release.base.config_digest,
            "rocm-pytorch:7.2.1-py3.12-torch2.9.1": release.torch.config_digest,
        }
        self.verify_errors: dict[str, str] = {}
        self.host_error: str | None = None
        self.gpu_error: str | None = None
        self.project = ProjectImageInspection(
            image_id="sha256:" + "8" * 64,
            fingerprint="f" * 64,
            changed=False,
            base_manifest_error=None,
        )
        self.overlay_error: str | None = None
        self.calls: list[tuple[str, ...]] = []
        self.mutations: list[tuple[str, ...]] = []

    def inspect_image(self, reference: str):
        self.calls.append(("inspect-image", reference))
        return self.images.get(reference)

    def inspect_friendly(self, reference: str):
        self.calls.append(("inspect-friendly", reference))
        return self.friendly.get(reference)

    def verify_image(self, release, image, kind: str):
        self.calls.append(("verify-image", kind, image.reference))
        return self.verify_errors.get(kind)

    def host_preflight(self):
        self.calls.append(("host-preflight",))
        return self.host_error

    def gpu_runtime(self, reference: str):
        self.calls.append(("gpu-runtime", reference))
        return self.gpu_error

    def inspect_project(self, config):
        self.calls.append(("inspect-project", config.image))
        return self.project

    def verify_effective_overlay(self, config, site_packages):
        self.calls.append(("verify-overlay", str(site_packages)))
        return self.overlay_error

    def pull(self, *args):
        self.mutations.append(("pull", *args))

    def tag(self, *args):
        self.mutations.append(("tag", *args))

    def rm(self, *args):
        self.mutations.append(("rm", *args))

    def build(self, *args):
        self.mutations.append(("build", *args))

    def quarantine(self, *args):
        self.mutations.append(("quarantine", *args))
