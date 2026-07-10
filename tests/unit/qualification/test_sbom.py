from __future__ import annotations

from tests.support.load_script import load_script


def test_spdx_document_has_required_identity_and_stable_relationships():
    module = load_script("tools/generate-sbom.py")
    document = module.build_spdx(
        name="rocm-pytorch:stable",
        namespace="https://example.invalid/spdx/test",
        os_packages=[
            ("rocm-core", "7.2.1.70201-1"),
            ("rocm-core", "7.2.1.70201-1"),
        ],
        python_packages=[("torch", "2.9.1+rocm7.2.1")],
        created="2026-07-09T12:00:00Z",
    )

    assert document["spdxVersion"] == "SPDX-2.3"
    assert document["documentNamespace"] == "https://example.invalid/spdx/test"
    assert {package["name"] for package in document["packages"]} == {
        "rocm-core",
        "torch",
    }
    assert len(document["packages"]) == 2
    assert len(document["relationships"]) == 2
    assert all(package["filesAnalyzed"] is False for package in document["packages"])


def test_spdx_package_ids_are_deterministic_across_input_order():
    module = load_script("tools/generate-sbom.py")
    arguments = {
        "name": "image",
        "namespace": "https://example.invalid/spdx/order",
        "created": "2026-07-09T12:00:00Z",
    }

    first = module.build_spdx(
        **arguments,
        os_packages=[("b", "2"), ("a", "1")],
        python_packages=[],
    )
    second = module.build_spdx(
        **arguments,
        os_packages=[("a", "1"), ("b", "2")],
        python_packages=[],
    )

    assert first == second
