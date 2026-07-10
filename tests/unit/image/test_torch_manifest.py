import json
from pathlib import Path

from tests.support.load_script import load_script


def test_manifest_detects_changed_distribution_file(tmp_path):
    module = load_script(Path("images/common/torch-manifest.py"))
    package = tmp_path / "torch"
    package.mkdir()
    binary = package / "libtorch.so"
    binary.write_bytes(b"first")
    manifest = tmp_path / "manifest.json"

    module.write_manifest({"torch": [binary]}, manifest)

    assert module.verify_manifest(manifest) == []
    binary.write_bytes(b"second")
    assert module.verify_manifest(manifest) == [f"changed: {binary}"]
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == 1


def test_manifest_detects_missing_file_and_unexpected_version(tmp_path, monkeypatch):
    module = load_script(Path("images/common/torch-manifest.py"))
    package = tmp_path / "triton"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    module.write_manifest(
        {"triton": [source]},
        manifest,
        versions={"triton": "3.5.1"},
    )
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "3.6.0")
    source.unlink()

    assert module.verify_manifest(manifest) == [
        "unexpected version: triton expected 3.5.1, got 3.6.0",
        f"missing: {source}",
    ]


def test_manifest_output_is_deterministic_and_sorted(tmp_path):
    module = load_script(Path("images/common/torch-manifest.py"))
    package = tmp_path / "torch"
    package.mkdir()
    second = package / "z.so"
    first = package / "a.py"
    second.write_bytes(b"z")
    first.write_bytes(b"a")
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"

    module.write_manifest({"torch": [second, first]}, one)
    module.write_manifest({"torch": [first, second]}, two)

    assert one.read_bytes() == two.read_bytes()

