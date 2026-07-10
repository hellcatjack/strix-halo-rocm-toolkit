import pytest

from amd_ai.image.lock import LockError, parse_package_lock
from amd_ai.image.rocm_lock import RESOLVER_SCRIPT, build_resolver_argv


def test_package_lock_requires_sorted_exact_versions():
    lock = parse_package_lock(
        "hipcc=7.2.1.70201-1\nrocm-core=7.2.1.70201-1\n"
    )

    assert lock == (
        ("hipcc", "7.2.1.70201-1"),
        ("rocm-core", "7.2.1.70201-1"),
    )


@pytest.mark.parametrize(
    "text",
    [
        "rocm-core\n",
        "rocm-core=\n",
        "rocm-core=7.2.1 70201\n",
        "ROCM-core=7.2.1\n",
        "rocm-core=7.2.1\nhipcc=7.2.1\n",
        "hipcc=7.2.1\nhipcc=7.2.1\n",
        "hipcc=7.2.1=bad\n",
    ],
)
def test_package_lock_rejects_unpinned_duplicate_or_unsorted_lines(text):
    with pytest.raises(LockError):
        parse_package_lock(text)


def test_resolver_container_keeps_stdin_open_for_bash_script(tmp_path):
    argv = build_resolver_argv(
        ubuntu_digest="ubuntu@sha256:" + "a" * 64,
        key_path=tmp_path / "rocm.gpg",
    )

    assert "--interactive" in argv
    assert argv[-2:] == ("bash", "-s")


def test_resolver_pins_the_official_amd_origin_above_ubuntu():
    assert "Pin: release o=repo.radeon.com" in RESOLVER_SCRIPT
    assert "Pin-Priority: 600" in RESOLVER_SCRIPT
