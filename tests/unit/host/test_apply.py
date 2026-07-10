import json
import stat
from pathlib import Path

import pytest

from amd_ai.host.apply import (
    AMD_DEBUG_TOOLS_SHA256,
    ApplyError,
    ApplyRefused,
    backup_host_state,
    execute_plan,
    parse_gpg_fingerprints,
    ttm_input_text,
    verify_file_sha256,
)
from amd_ai.host.models import PlannedAction, PreparePlan
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import FakeRunner, healthy_snapshot


def action_plan(*actions):
    return PreparePlan(
        supported=True,
        target_user="customer",
        actions=tuple(actions),
        reboot_required=True,
    )


def backup_action():
    return PlannedAction(
        code="BACKUP.SNAPSHOT",
        summary="backup",
        argv=(),
        privileged=True,
    )


def test_backup_copies_config_and_records_commands(tmp_path):
    destination = tmp_path / "amd-ai"
    backup = backup_host_state(
        snapshot=healthy_snapshot(),
        destination=destination,
        root=Path("tests/fixtures/host/healthy"),
        runner=FakeRunner.healthy_target(),
        timestamp="20260709T120000Z",
    )

    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert (backup / "proc/cmdline").is_file()
    assert manifest["schema_version"] == 1
    assert [record["name"] for record in manifest["commands"]] == [
        "packages",
        "dkms",
        "kernel",
        "gpu",
        "docker",
    ]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "manifest.json").stat().st_mode) == 0o600


def test_execute_requires_root_and_confirmation():
    plan = action_plan(backup_action())

    with pytest.raises(ApplyRefused, match="root"):
        execute_plan(
            plan,
            FakeRunner(),
            effective_uid=1000,
            confirmed=True,
        )
    with pytest.raises(ApplyRefused, match="confirmation"):
        execute_plan(
            plan,
            FakeRunner(),
            effective_uid=0,
            confirmed=False,
        )


def test_ai_max_accepts_memory_warning_but_declines_reboot():
    assert ttm_input_text(nominal_gib=128, mem_total_kib=131015488) == "y\nn\n"
    assert ttm_input_text(nominal_gib=100, mem_total_kib=131015488) == "n\n"


def test_sha256_verification_stops_mismatched_wheel(tmp_path):
    wheel = tmp_path / "amd_debug_tools.whl"
    wheel.write_bytes(b"not the pinned wheel")

    with pytest.raises(ApplyError, match="SHA-256"):
        verify_file_sha256(wheel, AMD_DEBUG_TOOLS_SHA256)


def test_gpg_parser_requires_the_full_docker_fingerprint():
    listing = (
        "pub:-:4096:1:8D81803C0EBFCD88:1487788586:::-:::scESC::::::23::0:\n"
        "fpr:::::::::9DC858229FC7DD38854AE2D88D81803C0EBFCD88:\n"
    )

    assert parse_gpg_fingerprints(listing) == (
        "9DC858229FC7DD38854AE2D88D81803C0EBFCD88",
    )


def test_execute_disables_only_planned_source_after_backup(tmp_path):
    root = tmp_path / "root"
    source = root / "etc/apt/sources.list.d/rocm.list"
    source.parent.mkdir(parents=True)
    source.write_text(
        "deb https://repo.radeon.com/rocm/apt/6.4 noble main\n",
        encoding="utf-8",
    )
    action = PlannedAction(
        code="APT.DISABLE_OLD_ROCM_SOURCES",
        summary="disable source",
        argv=(),
        privileged=True,
        input_text=json.dumps(["etc/apt/sources.list.d/rocm.list"]),
    )

    result = execute_plan(
        action_plan(backup_action(), action),
        FakeRunner.backup_only(),
        effective_uid=0,
        confirmed=True,
        snapshot=healthy_snapshot(),
        root=root,
        backup_destination=tmp_path / "backups",
    )

    assert not source.exists()
    assert source.with_name("rocm.list.amd-ai-disabled").is_file()
    assert result.executed_codes == (
        "BACKUP.SNAPSHOT",
        "APT.DISABLE_OLD_ROCM_SOURCES",
    )


def test_ttm_memory_rounding_failure_uses_scoped_modprobe_fallback(tmp_path):
    root = tmp_path / "root"
    (root / "sys/module/ttm").mkdir(parents=True)
    ttm_args = ("/usr/local/bin/amd-ttm", "--set", "128")
    runner = FakeRunner.backup_only()
    runner.responses[ttm_args] = CommandResult(
        ttm_args,
        1,
        "",
        "requested memory exceeds available system memory",
    )
    initramfs_args = ("update-initramfs", "-u")
    runner.responses[initramfs_args] = CommandResult(
        initramfs_args,
        0,
        "",
        "",
    )
    action = PlannedAction(
        code="TTM.SET_AI_MAX",
        summary="set TTM",
        argv=ttm_args,
        privileged=True,
    )

    execute_plan(
        action_plan(backup_action(), action),
        runner,
        effective_uid=0,
        confirmed=True,
        snapshot=healthy_snapshot(),
        root=root,
        backup_destination=tmp_path / "backups",
    )

    config = root / "etc/modprobe.d/ttm.conf"
    assert config.read_text(encoding="utf-8") == (
        "options ttm pages_limit=33554432\n"
    )
    assert "amdgpu.gttsize" not in config.read_text(encoding="utf-8")
    assert initramfs_args in runner.calls


def test_ttm_unrelated_failure_never_writes_fallback(tmp_path):
    root = tmp_path / "root"
    (root / "sys/module/ttm").mkdir(parents=True)
    ttm_args = ("/usr/local/bin/amd-ttm", "--set", "128")
    runner = FakeRunner.backup_only()
    runner.responses[ttm_args] = CommandResult(
        ttm_args,
        1,
        "",
        "permission denied",
    )
    action = PlannedAction(
        code="TTM.SET_AI_MAX",
        summary="set TTM",
        argv=ttm_args,
        privileged=True,
    )

    with pytest.raises(ApplyError, match="amd-ttm failed"):
        execute_plan(
            action_plan(backup_action(), action),
            runner,
            effective_uid=0,
            confirmed=True,
            snapshot=healthy_snapshot(),
            root=root,
            backup_destination=tmp_path / "backups",
        )

    assert not (root / "etc/modprobe.d/ttm.conf").exists()


def test_unknown_internal_action_is_blocked_after_backup(tmp_path):
    unknown = PlannedAction(
        code="UNKNOWN.INTERNAL",
        summary="unknown",
        argv=(),
        privileged=True,
    )

    with pytest.raises(ApplyError, match="unknown internal action"):
        execute_plan(
            action_plan(backup_action(), unknown),
            FakeRunner.backup_only(),
            effective_uid=0,
            confirmed=True,
            snapshot=healthy_snapshot(),
            root=tmp_path / "root",
            backup_destination=tmp_path / "backups",
        )
