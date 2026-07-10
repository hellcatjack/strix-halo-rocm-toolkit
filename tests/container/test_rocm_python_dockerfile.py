from pathlib import Path


def test_rocm_python_is_clean_development_image():
    text = Path("images/rocm-python/Dockerfile").read_text(encoding="utf-8")
    lowered = text.lower().replace("no-torch", "")

    assert text.startswith("# syntax=docker/dockerfile:1.7")
    assert "ARG UBUNTU_BASE" in text
    assert "ARG IMAGE_SOURCE=unknown" in text
    assert "FROM ${UBUNTU_BASE}" in text
    assert "rocm-hip-sdk" in text
    assert "rocm-ml-sdk" in text
    assert "python3.12-venv" in text
    assert "cmake" in text and "ninja-build" in text and "g++" in text
    assert "python3 -m venv /opt/venv" in text
    assert "rm -rf /var/lib/apt/lists/*" in text
    assert "--mount=type=cache,target=/var/cache/apt,sharing=locked" in text
    assert "--mount=type=cache,target=/var/lib/apt/lists,sharing=locked" in text
    assert "FROM rocm/pytorch" not in text
    assert "torch" not in lowered
    assert 'org.opencontainers.image.source="${IMAGE_SOURCE}"' in text
    assert text.index("ARG VCS_REVISION=unknown") > text.index(
        "python3 -m venv /opt/venv"
    )
    assert text.index("ARG IMAGE_SOURCE=unknown") > text.index(
        "python3 -m venv /opt/venv"
    )


def test_rocm_python_has_one_venv_non_root_user_and_no_forced_caches():
    text = Path("images/rocm-python/Dockerfile").read_text(encoding="utf-8")

    assert 'ENV PATH="/opt/venv/bin:/opt/rocm/bin:${PATH}"' in text
    assert 'LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64"' in text
    assert "useradd" in text and "USER developer" in text
    assert "WORKDIR /workspace" in text
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in text
    for forbidden in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "COMFYUI",
        "MIOPEN_USER_DB_PATH",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        assert forbidden not in text


def test_ca_certificates_are_bootstrapped_before_https_rocm_sources():
    text = Path("images/rocm-python/Dockerfile").read_text(encoding="utf-8")

    bootstrap = "apt-get install -y --no-install-recommends ca-certificates gnupg"
    assert text.index(bootstrap) < text.index("https://repo.radeon.com/rocm/apt/7.2.1")


def test_build_does_not_run_device_dependent_rocminfo():
    text = Path("images/rocm-python/Dockerfile").read_text(encoding="utf-8")

    assert "/opt/rocm/bin/rocminfo --version" not in text
    assert "test -x /opt/rocm/bin/rocminfo" in text


def test_developer_identity_handles_ubuntu_base_uid_1000():
    text = Path("images/rocm-python/Dockerfile").read_text(encoding="utf-8")

    assert "getent group 1000" in text
    assert "groupmod --new-name developer" in text
    assert "getent passwd 1000" in text
    assert "usermod --login developer" in text
