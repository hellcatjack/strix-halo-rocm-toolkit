# Zero-to-Docker-Run README Design

## Objective

Restructure the existing README quick start into the single canonical path
from a new Ubuntu 24.04 host to an interactive project-derived container
started by standard `docker run`.

The README must not add a second quick start or repeat the complete direct-run
command in multiple sections. Managed project behavior remains available in
the detailed workflow chapters but is no longer the primary quick-start exit.

## Audience And Starting State

The primary reader has:

- a new Ubuntu 24.04.x AMD64 Desktop or Server installation;
- an AMD Ryzen AI Max+ 395 / Radeon 8060S;
- sudo access but not necessarily Docker, a suitable OEM kernel, or working GPU
  device permissions;
- no toolkit checkout, project directory, or local project image.

The path ends with Bash inside `video-lab:runtime`, started by raw Docker with
one read-write business-directory mount.

## Information Architecture

`快速开始` remains the only top-level quick-start section and contains seven
linear steps:

1. Confirm hardware, BIOS, sudo, network, and disk prerequisites.
2. Clone the toolkit from Huawei CodeArts, with GitHub as the fallback source.
3. Define `TOOLKIT` and `PROJECT`, then run the full installer with the project
   name `video-lab` and pulled release images.
4. Reboot only when instructed and rerun the identical installer command.
5. Confirm `/dev/kfd`, a render node, and the generated `video-lab:runtime`
   image.
6. Start the project-derived image with the one canonical direct Docker
   command.
7. Run a real Torch GPU matrix probe, then install ordinary Python packages
   into `.cache/python-site`.

Long explanations of installer state, managed overlays, dependency locking,
doctor/repair, and custom profiles are replaced with links to their existing
detailed sections.

## Canonical Direct Command

The complete direct-run command appears only in quick-start step 6:

```bash
RENDER_NODE="$(find /dev/dri -maxdepth 1 -type c -name 'renderD*' | sort | head -n 1)"

if [[ ! -c /dev/kfd || -z "$RENDER_NODE" ]]; then
  echo "未发现可用的 AMD GPU 设备节点，请先完成宿主机安装与重启。" >&2
else
  docker run --rm -it \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add "$(stat -c '%g' /dev/kfd)" \
    --group-add "$(stat -c '%g' "$RENDER_NODE")" \
    --user "$(id -u):$(id -g)" \
    --ipc=private \
    --shm-size=16g \
    --env HOME=/workspace \
    --env PIP_TARGET=/workspace/.cache/python-site \
    --env PYTHONPATH=/workspace/.cache/python-site:/workspace \
    --env PATH=/workspace/.cache/python-site/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --mount "type=bind,src=$PROJECT,dst=/workspace" \
    --workdir /workspace \
    --entrypoint /bin/bash \
    video-lab:runtime
fi
```

The command intentionally bypasses the project policy entrypoint. It uses the
same device mappings and numeric GID principle as the managed runner, but it
does not initialize an overlay, verify project freshness, protect Torch, or
run an automatic GPU gate.

## Quick-Start Installation Flow

The installation command remains:

```bash
./install.sh --mode full \
  --project-dir "$PROJECT" \
  --project-name video-lab \
  --image-source pull
```

The text explains only the user actions needed to continue:

- enter `INSTALL-KERNEL` for the reviewed kernel plan;
- reboot when `RESUME` requests it;
- rerun the identical command after reboot;
- enter `APPLY` for the reviewed platform plan;
- answer the Docker group prompt;
- wait for all installer stages to pass.

Detailed checkpoint and recovery semantics remain in `安装模式与自动化` and
are linked rather than duplicated.

## Direct-Run Detail Section

The existing `极简 Docker 直启` section is retained but shortened. It must
not repeat the canonical command. It will instead contain:

- a link back to quick-start step 6;
- an explanation of device mappings, numeric GIDs, UID/GID, shared memory,
  business-directory mount, PATH, `PIP_TARGET`, and entrypoint override;
- the exact SWR parent-image reference for users who do not need a derived
  image;
- the command boundary: no overlay, Torch protection, automatic verification,
  repair, or cross-host portability;
- guidance to use the managed project workflow when those guarantees matter.

The manual GPU probe appears only in quick-start step 7. The direct-run detail
section links back to it rather than duplicating the Python code.

## Managed Workflow Placement

`创建和运行项目` remains the canonical location for:

- `project init` and `project run`;
- `--dry-run`, build, and reuse behavior;
- policy entrypoint and automatic GPU verification;
- overlay and protected pip;
- project dependency locks and immutable parent identity.

The quick start mentions this workflow only as an optional safer alternative.

## Validation

The documentation change will be checked for:

- exactly one `快速开始` heading;
- exactly one complete `docker run --rm -it` command;
- no `project run` invocation inside quick start;
- Bash syntax for the canonical command;
- exact SWR manifest digest equality with `profiles/releases/stable.json`;
- presence of KFD, DRI, numeric group, UID/GID, 16 GiB shm, target pip, mount,
  workdir, and entrypoint arguments;
- absence of privileged mode, host IPC, extra capabilities, `PIP_USER`, and
  `PYTHONUSERBASE` in the command;
- `.cache` remaining excluded by the generated project `.dockerignore`;
- table-of-contents and section-link integrity;
- clean Markdown diff with no runtime source changes.

No GPU or Docker runtime test is required for this README-only restructure.

## Non-Goals

- No second quick-start path.
- No new installer, launcher, image, Dockerfile, or Python behavior.
- No generated shell script or Docker command utility.
- No Docker Compose or systemd unit.
- No cross-host deployment workflow.
- No restoration of overlay or Torch protection in direct mode.
- No duplication of the canonical direct command or GPU probe.
