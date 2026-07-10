from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from amd_ai.image.profile import ProfileError, load_profile
from amd_ai.project.config import (
    IMAGE_ID_PATTERN,
    NAME_PATTERN,
    PROFILE_PATTERN,
    load_project_config,
)
from amd_ai.project.dependencies import (
    lock_project_dependencies,
    render_profile_constraints,
    render_torch_constraints,
)
from amd_ai.runner import Runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = REPOSITORY_ROOT / "templates/project"
STABLE_IMAGE = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"


class ProjectInitError(RuntimeError):
    pass


def initialize_project(
    *,
    name: str,
    destination: Path,
    base_profile: str,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
    template_root: Path = TEMPLATE_ROOT,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
) -> Path:
    if NAME_PATTERN.fullmatch(name) is None:
        raise ProjectInitError(f"invalid project name: {name!r}")
    if PROFILE_PATTERN.fullmatch(base_profile) is None:
        raise ProjectInitError(f"invalid base profile: {base_profile!r}")
    image = _resolve_base_image(base_profile, runner, docker_prefix)
    image_id = _inspect_image_id(image, runner, docker_prefix)

    destination = destination.resolve()
    created = not destination.exists()
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ProjectInitError(f"destination is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ProjectInitError(f"destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True, mode=0o755)
    destination.chmod(0o755)

    try:
        for source in sorted(template_root.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                continue
            target = destination / source.name
            shutil.copyfile(source, target)
            target.chmod(source.stat().st_mode & 0o777)

        config_path = destination / "amd-ai-project.toml"
        config_text = config_path.read_text(encoding="utf-8")
        replacements = {
            "name": name,
            "base_profile": base_profile,
            "image": f"{name}:runtime",
            "base_image": image_id,
            "base_digest": image_id,
        }
        for key, value in replacements.items():
            config_text = _replace_toml_string(config_text, key, value)
        config_path.write_text(config_text, encoding="utf-8")
        constraints = _base_constraints(
            base_profile=base_profile,
            image=image,
            destination=destination,
            runner=runner,
            docker_prefix=docker_prefix,
        )
        (destination / "torch-constraints.txt").write_text(
            constraints,
            encoding="utf-8",
        )
        lock_project_dependencies(destination)
        load_project_config(config_path)
        _apply_ownership(
            destination,
            owner_uid=_owner_id(owner_uid, "SUDO_UID", os.getuid()),
            owner_gid=_owner_id(owner_gid, "SUDO_GID", os.getgid()),
        )
    except Exception:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _base_constraints(
    *,
    base_profile: str,
    image: str,
    destination: Path,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> str:
    if base_profile == "stable":
        return render_torch_constraints(
            REPOSITORY_ROOT / "profiles/torch/stable.requirements.lock"
        )
    args = [
        *docker_prefix,
        "run",
        "--rm",
        image,
        "cat",
        "/opt/amd-ai/profile.env",
    ]
    result = runner.run(args, check=False)
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip()
        raise ProjectInitError(
            f"cannot read profile metadata from {image}: {evidence}"
        )
    temporary = destination / ".base-profile.env"
    try:
        temporary.write_text(result.stdout, encoding="utf-8")
        profile = load_profile(temporary, allow_verified=False)
    except (OSError, ProfileError) as error:
        raise ProjectInitError(f"invalid embedded base profile: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    if profile.profile_id != base_profile or profile.status != "experimental":
        raise ProjectInitError("embedded base profile ID or status does not match")
    return render_profile_constraints(profile)


def _resolve_base_image(
    base_profile: str,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> str:
    if base_profile == "stable":
        return STABLE_IMAGE
    args = [
        *docker_prefix,
        "image",
        "ls",
        "--filter",
        f"label=org.amd-ai.profile.id={base_profile}",
        "--format",
        "{{.Repository}}:{{.Tag}}",
    ]
    result = runner.run(args, check=False)
    if result.returncode != 0:
        raise ProjectInitError(
            f"cannot resolve base profile {base_profile}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    references = sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and "<none>" not in line
        }
    )
    if len(references) != 1:
        raise ProjectInitError(
            f"base profile {base_profile} must resolve to exactly one local image"
        )
    return references[0]


def _inspect_image_id(
    image: str,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> str:
    args = [*docker_prefix, "image", "inspect", "--format", "{{.Id}}", image]
    result = runner.run(args, check=False)
    image_id = result.stdout.strip()
    if result.returncode != 0 or IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        evidence = result.stderr.strip() or result.stdout.strip() or "image is missing"
        raise ProjectInitError(f"cannot inspect base image {image}: {evidence}")
    return image_id


def _replace_toml_string(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*.*$", re.MULTILINE)
    replacement = f"{key} = {json.dumps(value, ensure_ascii=True)}"
    rendered, count = pattern.subn(replacement, text)
    if count != 1:
        raise ProjectInitError(f"template must contain exactly one {key} assignment")
    return rendered


def _owner_id(explicit: int | None, environment_name: str, default: int) -> int:
    if explicit is not None:
        if explicit < 0:
            raise ProjectInitError(f"invalid owner ID: {explicit}")
        return explicit
    value = os.environ.get(environment_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ProjectInitError(f"invalid {environment_name}: {value!r}") from error
    if parsed < 0:
        raise ProjectInitError(f"invalid {environment_name}: {value!r}")
    return parsed


def _apply_ownership(root: Path, *, owner_uid: int, owner_gid: int) -> None:
    paths = [root, *sorted(root.iterdir(), key=lambda path: path.name)]
    for path in paths:
        stat = path.stat()
        if stat.st_uid == owner_uid and stat.st_gid == owner_gid:
            continue
        try:
            os.chown(path, owner_uid, owner_gid)
        except PermissionError as error:
            raise ProjectInitError(f"cannot set project ownership on {path}") from error
