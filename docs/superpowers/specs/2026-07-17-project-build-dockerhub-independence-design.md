# Project Build Docker Hub Independence Design

## Problem

China-host installation successfully pulls and verifies the stable Python and
PyTorch images from Huawei Cloud SWR, but project image creation fails before
the first build instruction. `templates/project/Dockerfile` declares
`# syntax=docker/dockerfile:1.7`, so BuildKit contacts Docker Hub to resolve the
external Dockerfile frontend. A Docker Hub timeout therefore blocks an
otherwise SWR-only installation.

## Decision

Make only the generated project image build independent of the external
Dockerfile frontend:

- remove the `docker/dockerfile:1.7` syntax directive from the project
  Dockerfile template;
- replace the BuildKit cache-mounted dependency installation step with a
  standard `RUN` instruction;
- retain the locked requirements, hash checking, Torch constraints, protected
  Torch manifest verification, non-root runtime user, labels, and entrypoint;
- leave the release-image Dockerfiles unchanged because they are used by the
  controlled image publication workflow, not by end-user project creation.

The project build may lose the dedicated uv cache mount between rebuilds. It
continues to reuse Docker layers when inputs are unchanged and continues to
inherit the shared immutable PyTorch parent image.

## Failure Handling

After this change, project image creation must not resolve
`docker.io/docker/dockerfile`. Any remaining failure should identify the
actual dependency source or project build instruction instead of a hidden
Dockerfile frontend dependency. Existing checkpoint behavior remains
unchanged, so rerunning the same installer command resumes at `PROJECT_INIT`.

## Verification

Add a static contract test that rejects an external syntax directive and cache
mount in the project template while preserving the locked uv installation and
Torch manifest verification. Run the focused project-template and project-build
tests, followed by the non-hardware regression suite. No real SWR or Docker Hub
download is required for local verification.
