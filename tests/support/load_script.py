import importlib.util
from pathlib import Path
from types import ModuleType


def load_script(path: str | Path) -> ModuleType:
    script = Path(path)
    spec = importlib.util.spec_from_file_location(script.stem.replace("-", "_"), script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

