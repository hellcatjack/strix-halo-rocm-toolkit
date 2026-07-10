from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path


COMPONENTS = ("torch", "torchvision", "torchaudio", "triton")


def collect() -> dict[str, object]:
    components: dict[str, dict[str, str]] = {}
    imported: dict[str, object] = {}
    for name in COMPONENTS:
        try:
            distribution = importlib.metadata.distribution(name)
            module = importlib.import_module(name)
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str) or not module_file:
                raise RuntimeError("module has no __file__")
            components[name] = {
                "distribution_path": str(
                    Path(distribution.locate_file("")).resolve()
                ),
                "module_path": str(Path(module_file).resolve()),
                "version": distribution.version,
            }
            imported[name] = module
        except Exception as error:
            components[name] = {
                "error": f"{type(error).__name__}: {error}"
            }
    torch = imported.get("torch")
    hip = getattr(getattr(torch, "version", None), "hip", None)
    return {
        "schema_version": 1,
        "components": components,
        "torch_hip_version": hip,
    }


def main() -> int:
    print(json.dumps(collect(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
