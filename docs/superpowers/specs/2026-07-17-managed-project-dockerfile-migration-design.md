# Managed Project Dockerfile Migration Design

## Problem

The toolkit Dockerfile template no longer depends on the external
`docker/dockerfile:1.7` frontend, but existing projects contain a copy of the
template that was generated when the project was first initialized. Installer
resume detects the updated toolkit revision and replays `PROJECT_INIT`, yet the
existing-project path loads the project configuration and builds immediately
without refreshing any generated file. An untouched legacy project therefore
continues to contact Docker Hub.

## Decision

Add a narrowly scoped migration for the project Dockerfile:

- identify the untouched legacy Dockerfile by its exact SHA-256 digest,
  `9a347d016b40564cc950dc3aa76a798ab9792ba33c8aa82a40697b6660227373`;
- compare the project Dockerfile to the current toolkit template before every
  installer-managed build of an existing project;
- do nothing when the project already contains the current template;
- atomically replace the file with the current template only when its digest
  matches the known legacy digest;
- leave every unknown digest unchanged so user-customized Dockerfiles are never
  overwritten;
- preserve the project file mode and reject symlink migration.

This migration belongs to the installer existing-project path. The standalone
project initializer still refuses nonempty destinations, and the general
project builder continues to build whatever Dockerfile the project owns.

## Data Flow

During `ProductionInstallerActions.initialize_project`:

1. Bind and verify the selected immutable PyTorch parent.
2. If `amd-ai-project.toml` exists, load and validate the project configuration.
3. Inspect the project Dockerfile against the current template and legacy
   allowlist.
4. Replace only an exact legacy file, then calculate the normal build-context
   fingerprint.
5. Build or reuse the project image and continue the existing Torch manifest
   and overlay checks.

The migration occurs before fingerprinting, so a migrated project necessarily
receives a new derived image rather than reusing an image built from the old
Dockerfile.

## Failure Handling

Missing project Dockerfiles remain the responsibility of the existing project
build contract. A symlink at the managed Dockerfile path is rejected rather
than followed. Read, temporary-write, fsync, chmod, or atomic-replace failures
raise an actionable project initialization error before Docker is invoked.
Unknown regular-file digests are treated as user-owned customizations and are
not modified.

## Verification

Unit tests cover:

- exact legacy bytes migrate to the current template;
- an already-current template is unchanged;
- a one-byte user modification is preserved;
- a symlink is rejected;
- installer resume migrates before invoking the project build;
- migrated content no longer contains the external frontend or cache mount.

Run focused project and installer action tests, then the complete non-hardware
suite. Tests use temporary files and fake runners and perform no registry
downloads.
