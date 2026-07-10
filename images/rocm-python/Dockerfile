# syntax=docker/dockerfile:1.7
ARG UBUNTU_BASE
ARG UV_IMAGE

FROM ${UV_IMAGE} AS uv
FROM ${UBUNTU_BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY --from=uv /uv /uvx /usr/local/bin/
COPY profiles/rocm/rocm.gpg profiles/rocm/rocm.gpg.sha256 /tmp/rocm-lock/
COPY profiles/rocm/7.2.1-packages.lock /tmp/rocm-lock/packages.lock

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -eux; \
    cd /tmp/rocm-lock; \
    sha256sum -c rocm.gpg.sha256; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates gnupg; \
    install -D -m 0644 rocm.gpg /etc/apt/keyrings/rocm.gpg; \
    printf '%s\n' \
      'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main' \
      'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu noble main' \
      > /etc/apt/sources.list.d/amd-rocm.list; \
    printf '%s\n' \
      'Package: *' \
      'Pin: release o=repo.radeon.com' \
      'Pin-Priority: 600' \
      > /etc/apt/preferences.d/rocm-pin-600; \
    apt-get update; \
    xargs -r apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl file git gnupg g++ make \
      ninja-build pciutils pkg-config procps python3.12 python3.12-dev \
      python3.12-venv unzip wget xz-utils zip \
      < /tmp/rocm-lock/packages.lock; \
    while IFS='=' read -r package version; do \
      test "$(dpkg-query -W -f='${Version}' "${package}")" = "${version}"; \
    done < /tmp/rocm-lock/packages.lock; \
    dpkg-query -W rocm-hip-sdk rocm-ml-sdk; \
    python3 -m venv /opt/venv; \
    /opt/rocm/bin/hipcc --version; \
    test -x /opt/rocm/bin/rocminfo; \
    uv --version; \
    rm -rf /var/lib/apt/lists/* /tmp/rocm-lock

COPY src/amd_ai /opt/amd-ai/src/amd_ai
COPY images/common/container-check /usr/local/bin/container-check

RUN set -eux; \
    existing_group="$(getent group 1000 | cut -d: -f1 || true)"; \
    if [[ -n "${existing_group}" ]]; then \
      [[ "${existing_group}" == developer ]] || groupmod --new-name developer "${existing_group}"; \
    else \
      groupadd --gid 1000 developer; \
    fi; \
    existing_user="$(getent passwd 1000 | cut -d: -f1 || true)"; \
    if [[ -n "${existing_user}" ]]; then \
      [[ "${existing_user}" == developer ]] || usermod --login developer --home /home/developer --move-home --shell /bin/bash "${existing_user}"; \
    else \
      useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash developer; \
    fi; \
    install -d -o developer -g developer -m 0755 /workspace; \
    chmod 0755 /usr/local/bin/container-check; \
    find /opt/venv -xdev -type d -exec chmod go-w '{}' +

ENV PATH="/opt/venv/bin:/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64" \
    ROCM_PATH="/opt/rocm" \
    HIP_PATH="/opt/rocm" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

ARG ROCM_VERSION=7.2.1
ARG VCS_REVISION=unknown
ARG IMAGE_SOURCE=unknown

LABEL org.amd-ai.rocm.version="${ROCM_VERSION}" \
      org.amd-ai.python.version="3.12" \
      org.opencontainers.image.revision="${VCS_REVISION}" \
      org.opencontainers.image.source="${IMAGE_SOURCE}"

USER developer
WORKDIR /workspace
CMD ["bash"]
