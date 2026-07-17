# SWR-First Stable Image Acquisition Design

## Goal

Make Huawei Cloud SWR the deterministic default registry for stable image
acquisition while retaining GHCR as an automatic acquisition fallback. Chinese
users must not need a region flag, registry login, or mutable image tag.

The initial public mirror is:

- Registry: `swr.cn-east-3.myhuaweicloud.com`
- Organization: `hellcat-home`
- Base repository: `strix-halo-rocm-python`
- PyTorch repository: `strix-halo-rocm-pytorch`

The SWR and GHCR repositories expose the same Docker Schema 2 manifests,
manifest digests, config digests, and compressed layers for stable release
`0.2.0`.

## Non-Goals

- Do not infer geography from IP address, locale, language, or timezone.
- Do not race registries or choose a source from transient latency.
- Do not change ROCm, Python, PyTorch, image contents, or stable release ID.
- Do not weaken exact-digest, config, label, or embedded-artifact verification.
- Do not download SWR image layers during development verification on the
  current US host.
- Do not automate future SWR publication credentials in this change.

## User Interface

Add this install option:

```text
--registry auto|swr|ghcr
```

`auto` is the default for both interactive and non-interactive installs:

1. Try the public Huawei SWR replica.
2. On acquisition failure only, try the public GHCR canonical source.
3. If both sources are unavailable, preserve the existing interactive local
   build offer. Non-interactive mode fails with both sanitized acquisition
   causes.

`--registry swr` and `--registry ghcr` force one source and disable registry
fallback. They are intended for deterministic testing, policy enforcement, and
incident diagnosis. A forced registry still preserves the existing interactive
local-build decision after acquisition failure.

`--image-source build` does not use a registry. Combining it with an explicit
non-`auto` registry is rejected as contradictory input.

The session plan and acquisition stage identify the policy and active source:

```text
PLAN     镜像仓库=auto（华为 SWR 优先，GHCR 回退）
DETAIL   尝试公开华为 SWR：swr.cn-east-3.myhuaweicloud.com/...
WARN     华为 SWR 获取失败，回退公开 GHCR：<sanitized cause>
DETAIL   已采用公开 GHCR：ghcr.io/...
```

No credential value or Docker authentication file content may appear in
terminal output or logs.

## Distribution Policy

Keep `profiles/releases/stable.json` as the canonical qualified release
identity. Its GHCR names, exact manifest/config digests, evidence digests, and
release ID remain unchanged.

Add a strict distribution policy at
`profiles/releases/registries.json`. It contains:

- schema version;
- deterministic default order `swr`, then `ghcr`;
- user-facing source labels;
- exact canonical-repository to mirror-repository mappings.

The policy maps repository names only. It never supplies or overrides a
manifest digest, config digest, artifact digest, version, or label. A resolved
replica reference is always:

```text
<mapped repository>@<canonical stable manifest digest>
```

The parser rejects duplicate keys, unknown keys, mutable tags, embedded
digests, malformed registry names, duplicate registry IDs, incomplete base or
PyTorch mappings, and a default order that does not end with the canonical
`ghcr` source.

For a custom stable manifest whose canonical repository is absent from the
policy:

- `auto` skips unsupported replicas and uses the canonical source;
- `ghcr` uses the canonical source when it is a valid GHCR reference;
- `swr` fails before any network operation with a clear unsupported-mapping
  error.

## Architecture

Create `amd_ai.installer.registry` with focused immutable records and pure
resolution functions:

- `RegistryPolicy` represents strict source order and repository mappings.
- `RegistrySource` represents one source ID, label, and mappings.
- `ReleaseCandidate` binds a source ID and label to a `StableRelease` copy whose
  base and PyTorch repository names have been replaced while all identity
  digests remain unchanged.
- `load_registry_policy()` parses the strict policy file.
- `resolve_release_candidates()` returns the ordered candidates for
  `auto|swr|ghcr`.

The existing release parser and verifier remain responsible for qualified
identity. Candidate resolution uses `dataclasses.replace()` to change only
`ReleaseImage.image`. `pull_and_verify_release()` therefore continues to verify
the exact reference selected for that attempt.

The installer workflow owns fallback so it can report attempts in real time:

1. Resolve the stable release.
2. Resolve registry candidates from the policy and CLI selection.
3. Estimate missing image data against candidates in order.
4. Pull and verify both base and PyTorch images from one candidate.
5. Catch `ReleaseAcquisitionError`, report it, and try the next candidate only
   when the selection is `auto`.
6. Propagate `ReleaseIdentityError` immediately without trying another source.
7. Persist the successful candidate's exact references.

If the base image succeeds and PyTorch acquisition fails, the next candidate
retries the pair. Docker content addressing reuses any already-present layers.
This keeps a release acquisition associated with one named source while
avoiding duplicate layer storage.

## Error Semantics

Fallback is allowed only for acquisition failures, including:

- DNS, connection, timeout, or TLS transport failure;
- anonymous authorization denied for a repository intended to be public;
- manifest or blob unavailable from the selected registry;
- Docker pull command failure before identity verification.

Fallback is forbidden for identity failures, including:

- selected exact RepoDigest missing after pull;
- manifest/config digest mismatch;
- OCI source, revision, ROCm, Python, profile, or PyTorch label mismatch;
- embedded lock or manifest artifact digest mismatch;
- malformed registry response after content is locally inspectable.

The final acquisition error lists attempted source IDs and sanitized causes in
order. It does not include command output beyond the existing bounded,
sanitized diagnostic policy.

## Disk Estimation

The current missing-layer estimator uses an exact remote manifest when the
image is not already local. Extend it to evaluate release candidates in the
same deterministic order:

- a successful preferred-source estimate is used;
- an acquisition-style failure in `auto` advances to the next source;
- explicit source selection reports the failure without probing another
  registry;
- local exact images continue to produce zero missing bytes without network
  access.

The source shown in the disk estimate must match the candidate used for that
estimate. Pull still re-evaluates availability because the network can change
between estimation and acquisition.

## Installer State Compatibility

Do not raise `STATE_SCHEMA_VERSION`.

Before image acquisition, `RELEASE_RESOLVE` may retain the canonical release
references. After a successful pull, `IMAGE_PULL_OR_BUILD` updates
`base_image_reference` and `torch_image_reference` from
`VerifiedReleaseImages`, so the state records the registry actually used.
Manifest and config digest fields are unchanged.

Registry preference is intentionally not part of a completed stage digest:

- an incomplete image stage may use a newly selected registry;
- a completed image stage already records immutable exact references and is
  never replayed merely because the default distribution policy changed;
- changing `--registry` after successful acquisition does not rewrite an
  existing project parent;
- existing states stopped at `IMAGE_PULL_OR_BUILD` can adopt SWR-first behavior
  without deleting state or downloaded layers.

The registry policy path is derived from `source_root` and is not user
configurable in this change. This keeps the one-click install surface small and
binds policy changes to reviewed toolkit source updates.

## Other Stable Pull Paths

Use the same resolver for user-facing recovery paths that may reacquire a
stable parent. A repair must prefer the project's already-recorded exact
registry reference when one exists. If recovery must resolve the canonical
release again, it uses the same SWR-first policy and the same acquisition versus
identity error distinction.

Maintainer publication remains GHCR-canonical. Release verification continues
to validate the canonical manifest and gains a separate policy validation test;
it does not perform a live SWR layer download as part of the ordinary test
suite.

## Testing

Follow test-first development. All registry behavior tests use fake Docker
interfaces or recording runners and must make zero real SWR network calls.

Required automated coverage:

- strict registry policy parsing and malformed-policy rejection;
- `auto` candidate order is SWR then GHCR;
- explicit `swr` and `ghcr` produce one candidate;
- unknown custom releases skip SWR in `auto`;
- default install pulls SWR exact references;
- SWR acquisition failure falls back to GHCR;
- SWR identity failure blocks without GHCR fallback;
- explicit source failure never tries the other registry;
- successful `VerifiedReleaseImages` replace canonical state references with
  the actual source references;
- existing completed states retain their recorded exact references;
- disk estimation follows the same candidate order and reports the chosen
  source;
- CLI validation, session plan text, resume behavior, and sanitized failure
  output;
- README and install guide commands for default and forced sources.

Verification on the current US host:

```bash
uv run pytest tests/unit/installer tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py -q
uv run pytest -m "not hardware and not container" -q
```

Do not run `docker pull swr.cn-east-3.myhuaweicloud.com/...` on the current US
host. Final China-network acceptance uses an external clean Docker host to run
the default installer and confirm that progress identifies SWR, both exact
images pull anonymously, and the runtime Torch GPU probe passes.

## Documentation

Update `README.md`, `docs/install.md`, and `docs/release-chain.md` to state:

- SWR is the default anonymous source and GHCR is the acquisition fallback;
- exact digest and full local identity verification remain mandatory;
- commands for `--registry swr` and `--registry ghcr`;
- identity errors never trigger source fallback;
- how to identify the chosen registry in installer progress and state;
- China-network acceptance steps without registry login.
