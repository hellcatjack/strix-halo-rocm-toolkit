import pytest

from amd_ai.host.verify import (
    build_probe_argv,
    evaluate_post_reboot,
    verify_host,
)
from amd_ai.report import Status
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import FakeRunner, healthy_snapshot


PROBE_IMAGE = "rocm-python:7.2.1-py3.12"


def finding_codes(report):
    return {finding.code for finding in report.findings}


def probe_runner(
    *,
    inspect_returncode=0,
    probe_returncode=0,
    output="gfx1151\n",
    docker_prefix=("docker",),
):
    inspect_args = (
        *docker_prefix,
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        PROBE_IMAGE,
    )
    run_args = tuple(
        build_probe_argv(
            image=PROBE_IMAGE,
            device_gids={"/dev/kfd": 109, "/dev/dri/renderD128": 110},
            docker_prefix=docker_prefix,
        )
    )
    return FakeRunner(
        {
            inspect_args: CommandResult(
                inspect_args,
                inspect_returncode,
                "sha256:probe\n" if inspect_returncode == 0 else "",
                "No such image" if inspect_returncode else "",
            ),
            run_args: CommandResult(
                run_args,
                probe_returncode,
                output,
                "probe failed" if probe_returncode else "",
            ),
        }
    )


def test_probe_uses_devices_and_actual_gids():
    argv = build_probe_argv(
        image=PROBE_IMAGE,
        device_gids={
            "/dev/kfd": 109,
            "/dev/dri/renderD128": 110,
            "/dev/dri/renderD129": 110,
        },
    )

    assert argv.count("--device") == 2
    assert [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--group-add"
    ] == ["109", "110"]
    assert "--privileged" not in argv
    assert "--ipc=host" not in argv
    assert argv[-5:] == [
        "/usr/local/bin/container-check",
        "--mode",
        "rocm",
        "--json",
        "-",
    ]


def test_verify_uses_configured_docker_prefix_for_all_probe_commands():
    docker_prefix = ("sudo", "-n", "docker")
    runner = probe_runner(docker_prefix=docker_prefix)

    report = verify_host(
        healthy_snapshot(kernel="6.14.0-1018-oem"),
        image=PROBE_IMAGE,
        runner=runner,
        docker_prefix=docker_prefix,
    )

    assert report.status == Status.PASS
    assert runner.calls
    assert all(call[:3] == docker_prefix for call in runner.calls)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("amdgpu: MES failed to respond to msg=REMOVE_QUEUE", "GPU.MES_TIMEOUT"),
        ("amdgpu: GPU reset begin!", "GPU.RESET"),
        ("amdgpu 0000:c5:00.0: page fault", "GPU.PAGE_FAULT"),
        ("amdgpu: failed to load firmware gc_11_5_1", "GPU.FIRMWARE"),
        ("amdgpu: ring gfx_0.0.0 timeout", "GPU.RING_TIMEOUT"),
    ],
)
def test_kernel_gpu_errors_block_verification(message, code):
    report = evaluate_post_reboot(healthy_snapshot(dmesg=message))

    assert report.status.value == "blocked"
    assert code in finding_codes(report)


def test_unavailable_dmesg_blocks_verification():
    report = evaluate_post_reboot(
        healthy_snapshot(dmesg="operation not permitted", dmesg_available=False)
    )

    assert report.status == Status.BLOCKED
    assert "HOST.DMESG_UNAVAILABLE" in finding_codes(report)


def test_live_ttm_must_match_the_computed_128_gib_target():
    report = evaluate_post_reboot(
        healthy_snapshot(kernel="6.14.0-1018-oem", ttm_pages_limit=1)
    )

    assert report.status.value == "reboot-required"
    assert "HOST.TTM_MISMATCH" in finding_codes(report)


def test_verified_kernel_and_successful_gfx1151_probe_pass():
    report = verify_host(
        healthy_snapshot(kernel="6.14.0-1018-oem"),
        image=PROBE_IMAGE,
        runner=probe_runner(),
    )

    assert report.status == Status.PASS
    assert report.facts["probe"]["image_id"] == "sha256:probe"


def test_missing_probe_image_is_blocking():
    report = verify_host(
        healthy_snapshot(kernel="6.14.0-1018-oem"),
        image=PROBE_IMAGE,
        runner=probe_runner(inspect_returncode=1),
    )

    assert report.status == Status.BLOCKED
    assert "HOST.PROBE_IMAGE_MISSING" in finding_codes(report)


def test_missing_device_mapping_does_not_start_container():
    runner = probe_runner()
    snapshot = healthy_snapshot(
        kernel="6.14.0-1018-oem",
        device_gids={"/dev/kfd": 109},
        current_group_ids=(109,),
    )

    report = verify_host(snapshot, image=PROBE_IMAGE, runner=runner)

    assert report.status == Status.BLOCKED
    assert "GPU.DEVICE_MAPPING" in finding_codes(report)
    assert not any(call[:2] == ("docker", "run") for call in runner.calls)


def test_probe_without_gfx1151_is_blocking():
    report = verify_host(
        healthy_snapshot(kernel="6.14.0-1018-oem"),
        image=PROBE_IMAGE,
        runner=probe_runner(output="gfx1100\n"),
    )

    assert report.status == Status.BLOCKED
    assert "GPU.GFX1151_MISSING" in finding_codes(report)
