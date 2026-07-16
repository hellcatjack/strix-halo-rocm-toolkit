# Read-Only GTT/TTM Diagnostics Design

**Date:** 2026-07-16

**Status:** Approved for implementation

## Objective

The toolkit must not tune, persist, or activate any host GTT/TTM setting. It
will retain read-only GTT/TTM facts because they are useful when diagnosing
Strix Halo GPU failures, but those facts cannot authorize a write or block an
otherwise healthy installation.

## Host Preparation

The host kernel phase remains responsible for the Ubuntu OEM 6.17 transition,
removal of confirmed legacy ROCm 6.4 packages, and removal of a confirmed
Radeon-origin `amdgpu-dkms` package.

After the kernel checkpoint passes, the platform phase may:

- install host utilities required for Docker and PCI diagnostics;
- install Docker when it is missing;
- repair Buildx with the package matching the installed Docker distribution;
- add the target user to required GPU device groups when explicitly approved.

The platform phase must never:

- install `amd-debug-tools` or invoke `amd-ttm`;
- create, replace, or remove `/etc/modprobe.d/ttm.conf`;
- write `ttm.pages_limit` or `amdgpu.gttsize`;
- update initramfs for a GTT/TTM change;
- request a reboot for GTT/TTM, Docker, Buildx, or group changes.

`--memory-gib` is removed from public and privileged host preparation
interfaces because it no longer controls a host mutation.

## Verification

The kernel checkpoint remains strict. It verifies the OEM kernel branch,
Radeon 8060S binding, KFD/render nodes, current-boot GPU errors, and conditional
desktop recovery before any platform mutation.

Final host verification retains preflight and current-boot GPU log checks, but
does not compute an expected TTM page limit and does not emit
`HOST.TTM_MISMATCH` or `HOST.MEMORY_CONFLICT`. The observed live TTM page limit,
kernel command line, and relevant TTM log lines remain report facts only.

## Installer Workflow

Full mode uses one reboot checkpoint:

1. Bootstrap and host preflight.
2. Kernel plan, independent confirmation, apply, reboot wait, and kernel
   verification.
3. Platform plan, independent confirmation, apply, and immediate host
   verification.
4. Release acquisition, image verification, and project creation.

The former second `REBOOT_PENDING` stage is removed from the full stage order.
Schema migration continues to accept the legacy `reboot_boot_id` field so old
state can be audited and migrated, but new workflows leave it unset. The
kernel reboot uses only `kernel_reboot_boot_id`.

Docker-group approval remains attached only to the platform phase. Kernel and
platform plan digests remain independent.

## Read-Only Memory Use

The pure memory normalization helper may remain for selecting a bounded
container shared-memory default. It has no host write capability and must not
be described as a GTT/TTM target.

Host probing may continue reading:

- `/sys/module/ttm/parameters/pages_limit`;
- `/sys/module/amdttm/parameters/pages_limit`;
- the current kernel command line;
- existing TTM configuration as backup or diagnostic evidence.

No probe result can silently re-enable the removed write path.

## Compatibility and Documentation

The CLI, README, installation guide, host operations guide, and examples must
state that the toolkit does not modify GTT/TTM. References to AI Max TTM
targets, `amd-ttm`, `--memory-gib`, TTM activation reboots, and TTM rollback of
tool-created files are removed or rewritten as read-only diagnostics.

Historical design and plan documents remain unchanged as implementation
history. Active user documentation and current implementation plans must match
this design.

## Tests and Acceptance Criteria

Automated tests must prove:

- no generated kernel or platform plan contains an action whose code starts
  with `TTM.`;
- no apply path can dispatch `amd-ttm`, write `ttm.conf`, or run
  `update-initramfs` for TTM;
- no public or privileged parser accepts `--memory-gib`;
- a mismatched or missing live TTM limit does not block final host verification;
- Docker-only, Buildx-only, and group-only platform plans do not request a
  reboot;
- full mode has only the kernel reboot checkpoint and never writes
  `reboot_boot_id`;
- read-only TTM facts remain present in host reports;
- the complete non-hardware suite passes, followed by the existing hardware
  qualification before promoting the OEM 6.17 kernel.
