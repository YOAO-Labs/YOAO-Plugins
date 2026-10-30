from __future__ import annotations

import hashlib
import json

import pytest

from scripts import validate_managed_lifecycle as validator


def test_rapidocr_version_uses_distribution_metadata(monkeypatch) -> None:
    monkeypatch.setattr(validator, "distribution_version", lambda name: f"{name}-3.9.0")

    assert validator._rapidocr_version() == "rapidocr-3.9.0"


def test_approved_weights_require_exact_hash_manifest(tmp_path, monkeypatch) -> None:
    relative_paths = ["v5/det.onnx", "v5/cls.onnx", "v5/rec.onnx"]
    manifest = {}
    for relative_path in relative_paths:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = relative_path.encode()
        target.write_bytes(content)
        manifest[relative_path] = hashlib.sha256(content).hexdigest()
    monkeypatch.setenv("VALIDATION_WEIGHT_SHA256_JSON", json.dumps(manifest))

    records = validator._approved_weights(
        model_dir=tmp_path,
        relative_paths=relative_paths,
        approval_ref="models:approved",
    )

    assert {record["name"] for record in records} == {
        "det.onnx",
        "cls.onnx",
        "rec.onnx",
    }
    assert all(record["approval_ref"] == "models:approved" for record in records)
    assert str(tmp_path) not in str(records)


def test_approved_weights_reject_hash_drift(tmp_path, monkeypatch) -> None:
    weight = tmp_path / "v5/det.onnx"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"candidate")
    monkeypatch.setenv(
        "VALIDATION_WEIGHT_SHA256_JSON",
        json.dumps({"v5/det.onnx": "0" * 64}),
    )

    with pytest.raises(RuntimeError, match="approved SHA-256"):
        validator._approved_weights(
            model_dir=tmp_path,
            relative_paths=["v5/det.onnx"],
            approval_ref="models:approved",
        )


def test_full_pool_requires_nine_cuda_sessions() -> None:
    sessions = {
        component: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        for component in ("det", "cls", "rec")
    }
    snapshot = {
        "cap": 3,
        "current_size": 3,
        "session_count": 9,
        "gpu_resident": True,
        "device": "cuda",
        "provider": "CUDAExecutionProvider",
        "engines": {
            key: {
                "resident": True,
                "device": "cuda",
                "provider": "CUDAExecutionProvider",
                "sessions": {
                    component: list(providers)
                    for component, providers in sessions.items()
                },
            }
            for key in ("mobile", "server", "medium")
        },
    }

    validator._assert_full_gpu_pool(snapshot)

    snapshot["engines"]["medium"]["sessions"]["rec"] = ["CPUExecutionProvider"]
    with pytest.raises(AssertionError, match="runtime provider fallback.*medium.*rec"):
        validator._assert_full_gpu_pool(snapshot)


def test_failure_path_emits_strict_nonpassing_evidence() -> None:
    evidence = validator._failed_evidence(RuntimeError("local path must not leak"))

    assert evidence["schema_version"] == "1"
    assert evidence["passed"] is False
    assert "local path" not in str(evidence)


@pytest.fixture
def qualification_identity(monkeypatch, tmp_path):
    values = {
        "VALIDATION_BACKEND_REGISTRY_ID": "12345678-1234-4234-8234-123456789abc",
        "VALIDATION_GPU_RESOURCE_ID": "resource-1",
        "VALIDATION_GPU_UUID": "GPU-test",
        "VALIDATION_IMAGE_ID": "sha256:" + "a" * 64,
        "AAP_PLUGIN_OCI_ARTIFACT": "local/rapidocr@sha256:" + "b" * 64,
        "AAP_PLUGIN_ASSET_MANIFEST_SHA256": "c" * 64,
        "AAP_PLUGIN_REGISTRY_ID": "12345678-1234-4234-8234-123456789abc",
        "AAP_ROUTE_REGISTRY_ID": "12345678-1234-4234-8234-123456789abc",
        "AAP_PLUGIN_GPU_RESOURCE_ID": "resource-1",
        "AAP_PLUGIN_GPU_UUID": "GPU-test",
        "AAP_PLUGIN_DEVICE": "cuda:0",
        "AAP_PLUGIN_IMAGE_ID": "sha256:" + "a" * 64,
        "AAP_PLUGIN_ID": "rapidocr",
        "AAP_PLUGIN_VERSION": "0.1.0",
        "AAP_PLUGIN_TOKEN_FILE": str(tmp_path / "token"),
        "AAP_PLUGIN_DATA_DIR": str(tmp_path),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AAP_PLUGIN_QUALIFICATION_FILE", raising=False)
    monkeypatch.delenv("AAP_PLUGIN_QUALIFICATION_SHA256", raising=False)
    return values


@pytest.mark.parametrize("name", ["VALIDATION_BACKEND_REGISTRY_ID", "VALIDATION_GPU_RESOURCE_ID"])
def test_qualification_rejects_missing_registered_identity(qualification_identity, monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(RuntimeError, match=name):
        validator._deployment_identity()


@pytest.mark.parametrize("name", [
    "AAP_PLUGIN_REGISTRY_ID", "AAP_ROUTE_REGISTRY_ID", "AAP_PLUGIN_GPU_RESOURCE_ID",
    "AAP_PLUGIN_GPU_UUID", "AAP_PLUGIN_DEVICE", "AAP_PLUGIN_IMAGE_ID",
])
def test_qualification_rejects_mismatched_deployment_identity(qualification_identity, monkeypatch, name):
    monkeypatch.setenv(name, "different")
    with pytest.raises(RuntimeError, match="identities differ"):
        validator._deployment_identity()


@pytest.mark.asyncio
async def test_operator_route_binding_enforces_real_admission(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from aap_protocol_v2.routing_runtime import install_backend_routing_controller

    registry = "12345678-1234-4234-8234-123456789abc"
    secret = "route-test-secret-" + "x" * 32
    monkeypatch.setenv("AAP_ROUTE_REGISTRY_ID", registry)
    monkeypatch.setenv("AAP_ROUTE_CONTROL_TOKEN", secret)
    monkeypatch.delenv("AAP_ROUTE_CONTROL_TOKEN_FILE", raising=False)
    app = FastAPI()
    install_backend_routing_controller(app)

    @app.post("/predict")
    async def predict():
        return {"accepted": True}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/predict")).status_code == 503
        await validator._bind_operator_routes(client, registry_id=registry, route_secret=secret)
        assert (await client.post("/predict")).json() == {"accepted": True}
        client.event_hooks["request"].clear()
        assert (await client.post("/predict")).status_code == 401


def test_generated_fixture_must_match_approved_digest(tmp_path):
    with pytest.raises(RuntimeError, match="generated fixture does not match"):
        validator._approved_fixture(tmp_path, "0" * 64, "fixture:approved")

    expected = "2a8d693bbe24506ee718a13b2e6ac72b3fa8e20583634ac65317293e9d44bd07"
    path, record = validator._approved_fixture(tmp_path, expected, "fixture:approved")
    assert path.is_file()
    assert record["sha256"] == expected
    assert record["approval_ref"] == "fixture:approved"
