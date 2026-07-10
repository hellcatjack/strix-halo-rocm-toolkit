# Per-project installer state design

## Problem

The installer currently defaults every invocation to:

```text
~/.local/state/strix-halo-rocm-toolkit/install-state.json
```

That file correctly protects resumable stage inputs, but it can describe only
one project and one install mode. After a full install for one project, a normal
container install for a second project loads the first project's state and
fails with either a `BOOTSTRAP` digest mismatch or `mode changed`. Requiring
every user to invent a `--state-path` conflicts with the intended one-command,
multi-project workflow.

## Decision

When `--state-path` is omitted, select a deterministic state file from the
normalized absolute project path. Preserve the old global state as a legacy
resume source only when it belongs to the requested project. An explicit
`--state-path` remains authoritative.

This is a focused compatibility change for toolkit `v0.2.2`. It does not change
the install-state schema, host policy, release manifest, or container images.

## State path

New implicit state files live under:

```text
~/.local/state/strix-halo-rocm-toolkit/projects/
```

The filename has this shape:

```text
<readable-project-basename>-<path-sha256-prefix>.json
```

The hash is calculated from the normalized absolute project path encoded as
UTF-8. The readable component is derived from the final path component,
restricted to ASCII letters, digits, dot, underscore, and hyphen, bounded in
length, and replaced with `project` when empty. The hash is the identity; the
readable component is diagnostic only.

For example, `/app/test/video-lab` receives a stable path similar to:

```text
~/.local/state/strix-halo-rocm-toolkit/projects/video-lab-b9bb64878f63.json
```

## Selection algorithm

State selection happens after the project directory has been supplied or
collected by the interactive prompt. The installer first acquires a fixed
toolkit coordination lock, then selects the project state and acquires its
state-file lock. The coordination lock remains held for the full workflow.

1. If the user supplied `--state-path`, normalize and use exactly that path.
2. Calculate the deterministic per-project path.
3. If that per-project state exists, use it.
4. Otherwise inspect the legacy global state.
5. If the legacy state is valid and its normalized `project_path` equals the
   requested project path, use the legacy state regardless of mode. This
   preserves both valid resumes and the existing mode-change protection.
6. If the valid legacy state belongs to another project, use the new
   per-project path.
7. If the legacy state exists but cannot be safely identified, select it and
   let existing corruption handling stop the run. Do not silently bypass
   ambiguous recovery evidence.
8. If no relevant state exists, use the new per-project path.

The selected path and whether it is explicit, per-project, or legacy are shown
as an informational installer status line.

## Compatibility and safety

- The completed `/app/test/rocmToolkit` installation continues using its
  existing global state without moving or rewriting it merely because the
  toolkit was upgraded.
- A new `/app/test/video-lab` container installation no longer reads the first
  project's valid global state and therefore starts a new workflow.
- If the same project is invoked with a different mode, the installer selects
  its existing state and reports the mode conflict instead of creating another
  state that bypasses the checkpoint.
- An existing per-project state always wins over an unrelated legacy state.
- Explicit state paths retain current digest, locking, permission, corruption,
  and resume semantics.
- State files remain mode `0600`; state directories remain mode `0700` subject
  to existing filesystem permissions.
- No state file is deleted, moved, or overwritten by selection alone.
- A decoded install state without a normalized `project_path` is corrupt and
  cannot execute stages.

Separate state files provide independent recovery histories. A fixed internal
coordination lock serializes workflows even when `--state-path` points outside
the default directory, so different projects cannot concurrently mutate host
packages, Docker, TTM configuration, images, or checkpoints. The selected
state's own lock still protects direct state access. The coordination lock is
not exposed as a CLI override.

## User workflow

After one full workstation installation, another project can be initialized
without state-management flags:

```bash
./install.sh \
  --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab \
  --image-source pull
```

`--state-path` remains documented as an advanced override for automation,
relocation, or operator-managed recovery.

## Error handling

Existing stage-input digest failures remain strict. Errors involving an
explicit or matching state should include the selected state path so the
operator can distinguish a true resume mismatch from accidental global-state
reuse. The installer must not suggest deleting a state file as an automatic
remediation.

## Tests

Automated tests cover:

- deterministic paths for normalized absolute project directories;
- safe filename handling and path-hash separation for equal basenames;
- explicit `--state-path` precedence;
- matching legacy-state reuse;
- unrelated valid legacy-state isolation;
- conservative handling of an unidentifiable legacy state;
- rejection and preservation of a decoded state with no project identity;
- existing per-project-state precedence;
- same-project mode mismatch protection;
- toolkit-wide coordination across different projects and explicit state
  locations;
- command-line and interactive project selection;
- unchanged checkpoint digest validation after state selection.

A production-path regression test will copy the current legacy full-install
state, request `/app/test/video-lab` in container mode, and verify that a new
per-project state is selected without modifying the copied legacy file.

## Documentation and release

README and installation documentation will describe automatic project state
isolation, legacy fallback, and the advanced override. Toolkit version becomes
`0.2.2`; stable image release ID `0.2.0` and both immutable image digests remain
unchanged.
