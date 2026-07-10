from __future__ import annotations

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import (
        InvalidWheelFilename,
        canonicalize_name,
        parse_wheel_filename,
    )
    from packaging.version import Version
except ModuleNotFoundError:  # The verified image exposes packaging through pip.
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.utils import (  # type: ignore[no-redef]
        InvalidWheelFilename,
        canonicalize_name,
        parse_wheel_filename,
    )
    from pip._vendor.packaging.version import Version  # type: ignore[no-redef]


__all__ = (
    "InvalidRequirement",
    "InvalidWheelFilename",
    "Requirement",
    "Version",
    "canonicalize_name",
    "parse_wheel_filename",
)
