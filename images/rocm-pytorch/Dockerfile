# syntax=docker/dockerfile:1.7
ARG ROCM_PYTHON_BASE
FROM ${ROCM_PYTHON_BASE}

USER root

ARG PROFILE_ID
ARG PROFILE_STATUS
ARG ROCM_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TRITON_VERSION

COPY --from=profile-context /profile.env /opt/amd-ai/profile.env
COPY --from=profile-context /requirements.lock /opt/amd-ai/profile.requirements.lock
COPY images/common/torch-manifest.py /opt/amd-ai/torch-manifest.py
COPY src/amd_ai /opt/amd-ai/src/amd_ai
RUN install -d -m 0755 /opt/amd-ai/bin
COPY images/common/protected-pip /opt/amd-ai/bin/pip

RUN set -eux; \
    chmod 0755 /opt/amd-ai/bin/pip; \
    ln -s pip /opt/amd-ai/bin/pip3

RUN /opt/venv/bin/python - <<'PY'
import pip
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import parse_wheel_filename
from pip._vendor.packaging.version import Version

assert pip.__version__ == "24.0"
assert Requirement("demo>=1").name == "demo"
assert str(Version("1.0")) == "1.0"
assert str(parse_wheel_filename("demo-1.0-py3-none-any.whl")[0]) == "demo"
PY

RUN --mount=from=wheels,target=/wheels,ro \
    set -eux; \
    test -n "${PROFILE_ID}"; \
    test -n "${PROFILE_STATUS}"; \
    grep -Fqx "PROFILE_ID=${PROFILE_ID}" /opt/amd-ai/profile.env; \
    grep -Fqx "PROFILE_STATUS=${PROFILE_STATUS}" /opt/amd-ai/profile.env; \
    uv pip install \
      --python /opt/venv/bin/python \
      --no-index \
      --find-links /wheels \
      --require-hashes \
      --requirements /opt/amd-ai/profile.requirements.lock; \
    uv pip check --python /opt/venv/bin/python; \
    /opt/venv/bin/python /opt/amd-ai/torch-manifest.py create \
      /opt/amd-ai/torch-manifest.json; \
    /usr/local/bin/container-check --mode torch --metadata-only --json -; \
    find /opt/venv -xdev -type d -exec chmod go-w '{}' +

ENV PATH="/opt/amd-ai/bin:/opt/venv/bin:/opt/rocm/bin:${PATH}"

ARG VCS_REVISION=unknown
ARG IMAGE_SOURCE=unknown

LABEL org.amd-ai.profile.id="${PROFILE_ID}" \
      org.amd-ai.profile.status="${PROFILE_STATUS}" \
      org.amd-ai.rocm.version="${ROCM_VERSION}" \
      org.amd-ai.python.version="3.12" \
      org.amd-ai.torch.version="${TORCH_VERSION}" \
      org.opencontainers.image.revision="${VCS_REVISION}" \
      org.opencontainers.image.source="${IMAGE_SOURCE}"

USER developer
WORKDIR /workspace
