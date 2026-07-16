# Docker capability and OEM 6.17 host-safety design

## Problem

Two field failures expose separate assumptions in the current host workflow.

First, `container-check` uses only `docker run`, but `Docker.detect()` always
executes `docker buildx version`. A host with a working Docker daemon and no
Buildx plugin therefore fails before the ROCm container starts:

```text
container-check: command failed (1): docker buildx version:
docker: unknown command: docker buildx
```

The installer contributes to this failure because it records only the Docker
daemon version. If an existing Ubuntu `docker.io`, Snap, or manually installed
daemon responds, host preparation skips the Docker action without checking the
Buildx capability needed later by project builds.

Second, the host policy still treats `6.14.0-1018-oem` as the minimum and only
recorded kernel. A desktop system with Radeon 8060S device `1002:1586` and
subsystem `2014:801d` booted `6.14.0-1020-oem` to a black display. The same
machine recovered through an older kernel and then booted successfully with
`6.17.0-1028-oem`; `amdgpu`, `/dev/kfd`, `/dev/dri/renderD128`, and the desktop
were all available. The current installer also combines kernel installation,
TTM configuration, device groups, and Docker preparation in one host apply
stage, making a failed reboot unnecessarily hard to isolate.

The failed 6.14 boot's full internal kernel log is not available, so this
evidence does not assign one upstream bug as the definitive cause. It does
establish that the old toolkit baseline failed on this board while OEM 6.17
restored the required display and GPU interfaces.

## Decision

Release toolkit `v0.3.0` with two related safety changes:

1. Model Docker runtime and Docker Buildx as separate capabilities. Runtime
   checks require only the daemon; build operations require Buildx. Full host
   preparation repairs a missing Buildx installation without replacing a
   recognized existing Docker distribution.
2. Move the Ubuntu 24.04 host baseline to the OEM 6.17 branch. Install the
   branch-tracking `linux-oem-6.17` metapackage, record
   `6.17.0-1028-oem` as the qualified baseline, and separate kernel activation
   from TTM tuning with an intervening reboot and display/GPU checkpoint.

The stable ROCm 7.2.1 and PyTorch 2.9.1 images are unchanged. Their release ID,
manifest digests, layer contents, and public GHCR references remain valid and
are not republished for this toolkit release.

## Goals

- Let `container-check --runtime` run on every usable Docker daemon, including
  one without Buildx.
- Require Buildx only at commands that actually build, inspect build
  provenance, or prune BuildKit state.
- Repair missing Buildx through the package family matching Docker CE or
  Ubuntu `docker.io`.
- Produce actionable diagnostics for Docker runtime, Buildx, and package
  provenance independently.
- Install and boot an Ubuntu OEM 6.17 kernel before applying TTM changes.
- Preserve old kernels and provide physical-console GRUB recovery instructions
  before every kernel reboot.
- Stop before TTM changes or large image downloads when the new kernel cannot
  provide a healthy desktop and GPU stack.
- Resume existing installations without redownloading an already verified
  stable image.

## Non-goals

- Do not migrate a recognized Ubuntu `docker.io` installation to Docker CE or
  migrate Docker CE to `docker.io`.
- Do not auto-repair Snap or manually installed Docker distributions whose
  package ownership cannot be established safely.
- Do not require Buildx for image pulls, ordinary local image inspection, or
  container runtime checks that do not invoke BuildKit. Publication-specific
  `buildx imagetools` checks retain their Buildx dependency.
- Do not pin one exact 6.17 ABI forever or suppress Ubuntu security updates
  within the 6.17 OEM branch.
- Do not automatically downgrade a future OEM branch newer than 6.17.
- Do not remove old kernel packages, run `apt autoremove`, modify the GRUB
  default, or reboot without an explicit user action.
- Do not alter the ROCm 7.2.1, Python 3.12, PyTorch 2.9.1, protected overlay,
  per-project container, or model-cache policies.

## Docker capability model

### Runtime detection

Docker runtime detection succeeds when either direct Docker or the existing
fixed `sudo -n docker` fallback can execute:

```text
docker info --format {{.ServerVersion}}
```

It returns a runtime object containing the selected command prefix and daemon
version. It does not invoke Buildx. `container-check`, stable-image pull and
verification, container doctor probes, and other `docker run` paths use this
runtime-only contract.

### Buildx detection

The runtime object exposes a separate Buildx capability check based on:

```text
docker buildx version
```

Build commands call this check before doing expensive preparation. A missing
plugin raises a Buildx-specific error rather than reporting that Docker itself
is unavailable. The error includes the detected daemon version and directs an
installed user to the host repair command.

Buildx remains mandatory for:

- base and PyTorch image builds;
- per-project image builds;
- BuildKit provenance and SBOM operations;
- `buildx imagetools` publication checks;
- BuildKit pruning.

### Package provenance and repair

Host probing records the daemon version, Buildx version or failure, and one of
these package classifications:

| Classification | Evidence | Missing-Buildx action |
| --- | --- | --- |
| `docker-ce` | `docker-ce` or `docker-ce-cli` installed | install `docker-buildx-plugin` from the fingerprint-verified Docker repository |
| `ubuntu-docker-io` | `docker.io` installed | install Ubuntu's `docker-buildx` package |
| `mixed` | both package families installed | block automatic repair and report conflicting packages |
| `external` | daemon works but neither package family owns it | allow runtime commands; block automatic Buildx repair with manual guidance |
| `missing` | daemon probe fails | retain the existing fingerprint-verified Docker CE installation flow, including Buildx and Compose plugins |

Package repair never uninstalls or replaces the active daemon. After package
installation, the tool reruns both runtime and Buildx probes. A successful APT
command without a working `docker buildx version` is a failed repair.

The container-only installer mode performs the same capability inventory. It
allows release pull and GPU verification with runtime-only Docker, but stops
before a project build if Buildx is still absent.

## OEM 6.17 kernel policy

### Branch semantics

The supported target is the OEM 6.17 branch, not one immutable ABI:

- `6.17.0-1028-oem` is the recorded qualified baseline and receives `pass`
  when all other host checks pass.
- A different OEM kernel whose major/minor branch is 6.17 receives
  `unverified`; installation may continue only after post-reboot display,
  amdgpu, device, log, and stable Torch runtime checks pass.
- An OEM kernel older than 6.17, including every 6.14 build, receives
  `change-required` with an upgrade action.
- A generic or otherwise non-OEM kernel on the supported Ubuntu host receives
  `change-required` and is guided to the OEM 6.17 branch.
- An OEM branch newer than 6.17 is not automatically downgraded. It receives
  `unverified` and must pass the same runtime gates before ordinary use; it
  cannot be promoted as a release baseline without full qualification.

The static tested-kernel profile removes the 6.14 baseline and promotes
`6.17.0-1028-oem` only after the hardware release checks in this design pass.
Findings and documentation must no longer describe any 6.14 kernel as
supported.

### Package selection

The kernel action installs:

```text
linux-oem-6.17
linux-firmware
```

`linux-oem-6.17` supplies the matching image and headers through its declared
dependencies. The installer does not separately install the rolling
`linux-oem-24.04` metapackage because that name can transition to a future OEM
branch.

Before presenting an apply confirmation, the current APT cache must expose a
candidate for `linux-oem-6.17` whose version starts with `6.17.`. If no candidate
exists, the installer makes no host change and instructs the user to refresh
or repair the Ubuntu Noble repositories. During apply, after the protected APT
source cleanup and index refresh, the candidate is checked again immediately
before package installation. A changed or non-6.17 candidate stops the action.

### Kernel retention

The installer records the currently running kernel as the known recovery
kernel before applying changes. It never removes kernel images, invokes
`apt autoremove`, calls `grub-set-default`, or hides advanced GRUB entries.
Before requesting a reboot, terminal output and the JSON report show:

- current recovery kernel;
- installed target branch and candidate version;
- how to open **Advanced options for Ubuntu** and select the recovery kernel;
- the saved installer state and exact resume command.

## Two-checkpoint full workflow

The full installer stage order is expanded so kernel activation and host tuning
cannot share one unverified reboot boundary:

1. `BOOTSTRAP`
2. `HOST_PREFLIGHT`
3. `KERNEL_PLAN`
4. `KERNEL_CONFIRM`
5. `KERNEL_APPLY`
6. `KERNEL_REBOOT_PENDING`
7. `KERNEL_VERIFY`
8. `HOST_PLAN`
9. `HOST_CONFIRM`
10. `HOST_APPLY`
11. `REBOOT_PENDING`
12. `HOST_VERIFY`
13. `RELEASE_RESOLVE`
14. `IMAGE_PULL_OR_BUILD`
15. `IMAGE_VERIFY`
16. `PROJECT_INIT`
17. `PROJECT_VERIFY`
18. `COMPLETE`

### Kernel checkpoint

`KERNEL_PLAN` contains only changes needed before the new kernel boot:

- create a private host backup;
- disable only confirmed standalone ROCm 6.4 APT sources;
- remove only the existing verified ROCm 6.4 cleanup set, including a
  confirmed Radeon-origin `amdgpu-dkms`;
- refresh APT metadata;
- revalidate and install `linux-oem-6.17` and `linux-firmware`.

`KERNEL_CONFIRM` uses a kernel-specific confirmation. Existing approval for an
old combined host plan cannot authorize this new plan. `KERNEL_APPLY` never
reboots directly. If the machine is already running an acceptable OEM kernel
and no pre-boot package change is required, the reboot-pending stage is a
validated no-op.

`KERNEL_REBOOT_PENDING` requires a changed boot ID when kernel activation is
needed. A package installation alone is never treated as proof that the target
kernel is active.

### Kernel verification

`KERNEL_VERIFY` runs before TTM, Docker repair, image acquisition, or project
creation. It requires:

- a running OEM 6.17 kernel, or an explicitly unverified newer OEM branch;
- Radeon `1002:1586` bound to the inbox `amdgpu` driver;
- `/dev/kfd` and at least one `/dev/dri/renderD*` node;
- no fatal amdgpu initialization, firmware-load failure, ring timeout, MES
  timeout, page fault, or GPU-reset finding in the current boot log.

Desktop state is conditional. Before kernel apply, the probe records whether
the `display-manager` unit existed and was active. If it was active, kernel
verification requires it to be active after reboot. A headless machine without
an active display manager is not made to install or start one.

Any kernel-verification failure is blocking. The installer prints the physical
GRUB recovery procedure, report path, and resume command, then stops without
writing TTM configuration.

### Host-tuning checkpoint

Only after `KERNEL_VERIFY` passes may `HOST_PLAN` include:

- target-user device-group changes;
- Docker installation or matching Buildx repair;
- pinned `amd-debug-tools` installation;
- the approved AI Max TTM target and initramfs update.

The tuning plan gets its own backup and confirmation. A TTM change sets a
second reboot checkpoint. If the live value and persistent configuration
already satisfy the plan, no second reboot is requested. Docker and group-only
changes do not manufacture a TTM reboot requirement.

`HOST_VERIFY` validates the activated TTM value, target-user groups, Docker
runtime, Buildx when a project build is requested, device nodes, current boot
logs, and ROCm agent visibility. Stable PyTorch GPU execution remains mandatory
at `IMAGE_VERIFY`.

## State and upgrade migration

The changed stage order requires installer state schema version 3. Loading a
valid version-2 state performs an explicit migration rather than emitting a
completed-stage digest mismatch.

For full-mode states, migration preserves:

- mode, project path and project name;
- target user and per-project state location;
- stable release and immutable image references already recorded;
- report paths and Docker group authorization;
- project files and locally present Docker layers.

It does not preserve an old host-plan approval or claim that an old 6.14 host
checkpoint satisfies the new kernel policy. The migrated workflow restarts at
host preflight and derives kernel/tuning checkpoint completion from fresh
probes. A host already running a healthy 6.17 OEM kernel skips package and
kernel reboot actions. Existing image references are inspected and reused, so
the subsequent image stage does not pull layers already present.

Container-mode version-2 states keep their compatible stage position after
runtime and Buildx capabilities are reprobed. A runtime-only state can continue
through image verification but cannot pass a pending project-build stage
without Buildx.

Mode or project identity changes remain prohibited. Their messages identify the
state file and explain how to start a distinct per-project state; they do not
suggest deleting a valid state blindly.

## Reports and user-visible progress

Host and installer JSON facts add:

- `docker_runtime_version`;
- `docker_buildx_version` or its probe error;
- `docker_distribution`;
- running kernel and recovery kernel;
- target kernel branch and APT candidate;
- pre- and post-reboot display-manager state;
- separate kernel and tuning reboot requirements.

Progress labels explicitly distinguish:

- installing OEM 6.17;
- waiting for the kernel reboot;
- validating desktop and GPU initialization;
- applying host and TTM tuning;
- waiting for a tuning reboot;
- validating ROCm and stable PyTorch GPU execution.

Long APT, Docker pull, and Buildx commands continue through the existing live
progress reporter. A missing Buildx failure occurs before wheel downloads or a
project build starts and includes the matching package remediation.

## Error behavior

- Docker daemon failure: report runtime unavailable and the selected direct or
  sudo probe evidence.
- Buildx failure on runtime-only command: ignore it because the command has no
  Buildx dependency.
- Buildx failure on build command: stop with a Buildx-specific diagnostic.
- Mixed Docker packages: stop package mutation and list the conflicting
  packages.
- External Docker package source: permit runtime use, block automatic plugin
  installation, and provide manual verification commands.
- Missing or wrong `linux-oem-6.17` candidate: stop before kernel package
  mutation.
- Reboot into the previous kernel: report both actual and expected branches,
  then explain the GRUB selection; do not proceed to tuning.
- Display manager or amdgpu failure: stop at `KERNEL_VERIFY` and show rollback
  instructions.
- TTM failure: preserve the already verified kernel checkpoint and stop before
  image acquisition.

## Test strategy

### Unit tests

- Docker runtime detection succeeds when `docker info` works and
  `docker buildx` is absent.
- `container-check` issues `docker run` without probing Buildx.
- Every build, publish, and BuildKit-prune entry point requires Buildx before
  expensive work.
- Docker CE, Ubuntu `docker.io`, mixed, external, and missing classifications
  select the expected action or block reason.
- A package action is followed by fresh runtime and Buildx probes.
- `6.14.0-1020-oem` and generic kernels require change.
- `6.17.0-1028-oem` is recorded as qualified.
- a later 6.17 patch and a future OEM branch are unverified without automatic
  downgrade.
- `linux-oem-6.17` candidate parsing rejects absent and wrong-branch values.
- no TTM action can occur before a successful `KERNEL_VERIFY` checkpoint.
- old kernels are absent from every removal command.

### Workflow tests

- an old kernel follows both reboot checkpoints in the correct order;
- an already healthy 6.17 host skips the kernel reboot;
- a failed display manager, missing KFD, amdgpu fatal log, or unchanged boot ID
  blocks before tuning;
- an already correct TTM configuration skips the second reboot;
- version-2 full and container states migrate without digest errors;
- migrated installations reuse local immutable stable images and project data;
- interrupting and resuming at every new checkpoint does not replay completed
  mutation.

### Regression fixtures and hardware checks

Add a fixture matching the reported AXB35 hardware evidence:

```text
PCI device: 1002:1586
Subsystem: 2014:801d
failed kernel: 6.14.0-1020-oem
qualified kernel: 6.17.0-1028-oem
```

The non-hardware suite must pass without Docker or GPU access. Hardware release
verification on Radeon 8060S must then confirm the desktop remains usable,
`rocminfo` exposes `gfx1151`, PyTorch reports HIP availability, and a synchronized
GPU tensor operation succeeds. Absence of current-boot fatal amdgpu findings is
part of the release evidence.

## Documentation and release

Update the README and host operations manual to:

- make OEM 6.17 the first host prerequisite;
- explain the two possible manual reboots;
- show GRUB recovery before the first reboot instruction;
- distinguish Docker runtime from Buildx requirements;
- document package-specific Buildx repair;
- include a direct digest-pinned `docker run` GPU check that does not require
  Buildx;
- remove statements that 6.14 is supported;
- explain qualified versus unverified 6.17 patch kernels.

Publish the implementation as toolkit `v0.3.0`. Do not rebuild or retag the
stable ROCm/PyTorch images; retain their immutable manifest digests.

## Reference evidence

- Ubuntu lists `linux-oem-6.17` as the complete OEM 6.17 kernel and headers
  package: <https://packages.ubuntu.com/linux-oem>
- Ubuntu's Strix/Strix Halo display fix history includes fixes released on the
  6.17 OEM line: <https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2134488>
- The ROCm Ryzen compatibility matrix remains the authority for the userspace
  ROCm/PyTorch baseline: <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>
