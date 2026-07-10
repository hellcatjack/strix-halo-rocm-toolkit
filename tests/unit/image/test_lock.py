import hashlib
import io
import json
from pathlib import Path

import pytest

from amd_ai.image.lock import (
    DownloadError,
    download,
    hash_file,
    render_verified_profile,
    validate_wheelhouse_manifest,
    write_wheelhouse_manifest,
)


def test_hash_file_streams_and_returns_sha256(tmp_path):
    wheel = tmp_path / "torch.whl"
    wheel.write_bytes(b"amd-wheel-fixture")

    assert hash_file(wheel) == hashlib.sha256(b"amd-wheel-fixture").hexdigest()


def test_render_profile_adds_each_digest_in_component_order():
    source = Path("profiles/torch/stable.sources.env").read_text(encoding="utf-8")
    digests = {
        "torch": "a" * 64,
        "torchvision": "b" * 64,
        "torchaudio": "c" * 64,
        "triton": "d" * 64,
    }

    rendered = render_verified_profile(source, digests)

    assert rendered.index("TORCH_SHA256=") < rendered.index("TORCHVISION_SHA256=")
    assert "TRITON_SHA256=" + "d" * 64 in rendered
    assert render_verified_profile(source, digests) == rendered


def test_download_is_atomic_and_verifies_requested_digest(tmp_path, monkeypatch):
    content = b"locked wheel bytes"
    digest = hashlib.sha256(content).hexdigest()
    destination = tmp_path / "torch.whl"
    call = {}

    def open_fixture(request, *, timeout):
        call["timeout"] = timeout
        return io.BytesIO(content)

    monkeypatch.setattr("amd_ai.image.lock.urllib.request.urlopen", open_fixture)

    result = download("https://example.com/torch.whl", destination, digest)

    assert result == digest
    assert call == {"timeout": 60}
    assert destination.read_bytes() == content
    assert not destination.with_suffix(".part").exists()


def test_download_hash_mismatch_keeps_no_partial_or_destination(tmp_path, monkeypatch):
    destination = tmp_path / "torch.whl"
    monkeypatch.setattr(
        "amd_ai.image.lock.urllib.request.urlopen",
        lambda request, *, timeout: io.BytesIO(b"wrong"),
    )

    with pytest.raises(DownloadError, match="SHA-256"):
        download("https://example.com/torch.whl", destination, "a" * 64)

    assert not destination.exists()
    assert not destination.with_suffix(".part").exists()


def test_wheelhouse_manifest_detects_tampering(tmp_path):
    first = tmp_path / "a.whl"
    second = tmp_path / "b.whl"
    first.write_bytes(b"a")
    second.write_bytes(b"bb")

    manifest_path = write_wheelhouse_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [record["filename"] for record in payload["files"]] == ["a.whl", "b.whl"]
    assert validate_wheelhouse_manifest(tmp_path) == ()
    second.write_bytes(b"changed")
    assert validate_wheelhouse_manifest(tmp_path) == ("changed: b.whl",)
