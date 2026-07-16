import json
import stat
from pathlib import Path

import pytest

from amd_ai.host.apply import (
    ApplyError,
    ApplyRefused,
    backup_host_state,
    execute_plan,
    parse_gpg_fingerprints,
)
from amd_ai.host.models import HostPlanPhase, PlannedAction, PreparePlan
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import FakeRunner, healthy_snapshot


def action_plan(*actions):
    return PreparePlan(
        phase=HostPlanPhase.TUNING,
        supported=True,
        target_user="customer",
        actions=tuple(actions),
        reboot_required=True,
    )


class DockerRepositoryRunner(FakeRunner):
    def run(self, args, *, check=True, input_text=None):
        result = super().run(args, check=check, input_text=input_text)
        if args[:2] == ["curl", "--fail"]:
            output = Path(args[args.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("docker signing key", encoding="utf-8")
        return result


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


def test_install_oem_617_rejects_candidate_drift_before_apt_install(tmp_path):
    update = ("apt-get", "update")
    policy = ("apt-cache", "policy", "linux-oem-6.17")
    install = (
        "apt-get",
        "install",
        "-y",
        "linux-oem-6.17",
        "linux-firmware",
    )
    runner = FakeRunner.backup_only()
    runner.responses[update] = CommandResult(update, 0, "", "")
    runner.responses[policy] = CommandResult(
        policy,
        0,
        "linux-oem-6.17:\n  Candidate: 6.18.0-1001.1\n",
        "",
    )
    action = PlannedAction(
        code="APT.INSTALL_OEM_617",
        summary="install kernel",
        argv=(),
        privileged=True,
    )

    with pytest.raises(ApplyError, match="6.17 candidate"):
        execute_plan(
            action_plan(backup_action(), action),
            runner,
            effective_uid=0,
            confirmed=True,
            snapshot=healthy_snapshot(),
            root=tmp_path / "root",
            backup_destination=tmp_path / "backups",
        )

    assert install not in runner.calls


@pytest.mark.parametrize(
    ("code", "install"),
    [
        (
            "DOCKER.INSTALL_BUILDX_PLUGIN",
            ("apt-get", "install", "-y", "docker-buildx-plugin"),
        ),
        (
            "DOCKER.INSTALL_UBUNTU_BUILDX",
            ("apt-get", "install", "-y", "docker-buildx"),
        ),
    ],
)
def test_buildx_repair_reprobes_runtime_and_plugin(tmp_path, code, install):
    root = tmp_path / "root"
    responses = dict(FakeRunner.backup_only().responses)
    update = ("apt-get", "update")
    runtime = ("docker", "version", "--format", "{{.Server.Version}}")
    buildx = ("docker", "buildx", "version")
    responses.update(
        {
            update: CommandResult(update, 0, "", ""),
            install: CommandResult(install, 0, "", ""),
            runtime: CommandResult(runtime, 0, "27.5.1\n", ""),
            buildx: CommandResult(buildx, 0, "github.com/docker/buildx v0.30.1\n", ""),
        }
    )
    if code == "DOCKER.INSTALL_BUILDX_PLUGIN":
        curl = (
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "https://download.docker.com/linux/ubuntu/gpg",
            "--output",
            str(root / "etc/apt/keyrings/docker.asc.amd-ai.tmp"),
        )
        gpg = (
            "gpg",
            "--batch",
            "--show-keys",
            "--with-colons",
            str(root / "etc/apt/keyrings/docker.asc.amd-ai.tmp"),
        )
        responses[curl] = CommandResult(curl, 0, "", "")
        responses[gpg] = CommandResult(
            gpg,
            0,
            "fpr:::::::::9DC858229FC7DD38854AE2D88D81803C0EBFCD88:\n",
            "",
        )
        runner = DockerRepositoryRunner(responses)
    else:
        runner = FakeRunner(responses)
    action = PlannedAction(code=code, summary="repair Buildx", argv=(), privileged=True)

    execute_plan(
        action_plan(backup_action(), action),
        runner,
        effective_uid=0,
        confirmed=True,
        snapshot=healthy_snapshot(),
        root=root,
        backup_destination=tmp_path / "backups",
    )

    assert install in runner.calls
    assert runtime in runner.calls
    assert buildx in runner.calls


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


@pytest.mark.parametrize(
    "code",
    ["TTM.SET_AI_MAX", "TTM.INSTALL_AMD_DEBUG_TOOLS"],
)
def test_removed_ttm_actions_are_never_dispatchable(tmp_path, code):
    action = PlannedAction(
        code=code,
        summary="removed",
        argv=(),
        privileged=True,
    )

    with pytest.raises(ApplyError, match="unknown internal action"):
        execute_plan(
            action_plan(backup_action(), action),
            FakeRunner.backup_only(),
            effective_uid=0,
            confirmed=True,
            snapshot=healthy_snapshot(),
            root=tmp_path / "root",
            backup_destination=tmp_path / "backups",
        )
