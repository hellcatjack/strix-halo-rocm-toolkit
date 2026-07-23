# Simple Direct Docker Run Design

## Objective

Document an intentionally minimal, local-only way to start either the stable
PyTorch parent image or an existing project-derived image with standard
`docker run`. The user mounts one business directory, receives access to the
host Radeon GPU, and enters Bash.

This path is separate from `strix-halo-rocm project run`. It does not preserve
the managed overlay, protected pip workflow, project fingerprint gate, startup
Torch verification, automatic repair, or cross-host deployment behavior.

## Scope

The documentation will cover two image choices:

1. An existing project-derived image such as `video-lab:runtime`. Its policy
   entrypoint is deliberately bypassed with `--entrypoint /bin/bash`, while the
   dependencies installed into the image during its build remain available.
2. The stable public PyTorch parent image, referenced by its exact SWR digest.
   This image supplies ROCm, Python, and PyTorch but no project-specific
   dependencies.

The design requires no Dockerfile, image, entrypoint, launcher, installer, or
Python source changes.

## Runtime Contract

The host must already have a working `amdgpu` driver, `/dev/kfd`, at least one
`/dev/dri/renderD*` node, Docker access, and the selected image. The command
will:

- map `/dev/kfd` and `/dev/dri`;
- add the numeric GID of KFD and the first render node;
- run with the invoking user's UID and primary GID;
- use private IPC and 16 GiB shared memory;
- bind the business directory read-write at `/workspace`;
- use `/workspace` as both the working directory and home directory;
- place installed Python packages under `/workspace/.cache/python-site`;
- bypass `/usr/local/bin/project-entrypoint` with `/bin/bash`;
- avoid privileged mode, host IPC, extra capabilities, and relaxed seccomp.

The documented command is:

```bash
PROJECT="$(realpath "$HOME/ai-projects/video-lab")"
IMAGE="video-lab:runtime"
RENDER_NODE="$(find /dev/dri -maxdepth 1 -type c -name 'renderD*' | sort | head -n 1)"

test -c /dev/kfd
test -n "$RENDER_NODE"

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
  "$IMAGE"
```

For the public parent image, only `IMAGE` changes:

```bash
IMAGE="swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b"
```

## Python Package Persistence

The command intentionally removes `/opt/amd-ai/bin` from `PATH`, so the plain
pip installed in `/opt/venv` is selected instead of the protected pip wrapper.
`PIP_TARGET=/workspace/.cache/python-site` and the matching `PYTHONPATH` make a
normal command such as the following persist packages inside the single
business-directory mount. A venv does not support ordinary `pip --user`, so
this mode deliberately uses pip's target directory instead. The generated
project `.dockerignore` already excludes `.cache`, so these packages do not
enter later image builds:

```bash
pip install transformers
```

This mode permits packages in `.cache/python-site` to shadow or replace the
effective Torch stack. That behavior is explicitly accepted for this simple
mode. Users who need the verified Torch baseline must use the managed project
workflow.

## GPU Verification

GPU verification is manual rather than an entrypoint gate. The documentation
will include this probe:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())

x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU_OK:", y.device)
PY
```

The expected success signal is a non-empty HIP version, `available: True`, and
`GPU_OK: cuda:0`.

## Documentation Changes

README will receive a new `极简 Docker 直启` section near the existing image
references and project workflow. It will include:

- the host prerequisites;
- the project-derived image command;
- the exact SWR parent-image substitution;
- persistent user-site package installation;
- the manual GPU probe;
- a concise comparison with managed `project run`;
- troubleshooting for missing KFD/render nodes, Docker permissions, image
  lookup, and mounted-directory permissions.

The table of contents will link to the new section. No runtime tests are
required because the implementation changes documentation only; command
syntax and repository references will still be reviewed before publication.

## Explicit Non-Goals

- No generated launcher or deployment bundle.
- No cross-host portability promise.
- No automatic image build, pull, verification, or repair.
- No overlay initialization or transaction handling.
- No protection against replacing Torch, TorchVision, TorchAudio, or Triton.
- No automatic CPU-fallback prevention.
- No ComfyUI, model, cache, output, or credential mount conventions.
- No root or privileged container mode.
