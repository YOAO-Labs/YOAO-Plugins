"""Validate RapidOCR's full composite-engine pool on the deployment GPU."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import os
import platform
import re
import secrets
import socket
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, redirect_stdout
from importlib.metadata import version as distribution_version
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import jwt
import uvicorn
from PIL import Image, ImageDraw
from aap_backend_runtime import (
    artifact_evidence,
    build_managed_lifecycle_evidence,
    exercise_lifecycle_fault_matrix,
)
from aap_lifecycle_contract import REQUIRED_CONTRACT_CHECKS, memory_cycle_evidence
from aap_protocol_v2.lifecycle import (
    AdmissionScope,
    AdmissionTokenClaims,
    GPU_ADMISSION_TOKEN_HEADER,
    GPU_GENERATION_HEADER,
    encode_ed25519_public_key,
    sign_admission_token,
)
from aap_protocol_v2.routing import (
    ROUTE_ADMISSION_AUDIENCE,
    ROUTE_ADMISSION_HEADER,
    ROUTE_ADMISSION_PROTOCOL,
    ROUTE_CONTROL_PATH,
    ROUTE_CONTROL_PROTOCOL,
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


CONTROL_SCOPES = {
    AdmissionScope.DRAIN,
    AdmissionScope.UNLOAD,
    AdmissionScope.RESUME,
    AdmissionScope.MODE,
    AdmissionScope.RESET,
}
POOL_TARGETS = (
    ("ocr-det", {"version": "v5", "size": "mobile", "lang": "universal"}),
    ("ocr-rec", {"version": "v5", "size": "server", "lang": "universal"}),
    ("ocr-e2e", {"version": "v6", "size": "medium", "lang": "universal"}),
)


def _rapidocr_version() -> str:
    return distribution_version("rapidocr")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _deployment_identity() -> tuple[str, str, str]:
    registry = _required_env("VALIDATION_BACKEND_REGISTRY_ID")
    resource = _required_env("VALIDATION_GPU_RESOURCE_ID")
    gpu_uuid = _required_env("VALIDATION_GPU_UUID")
    image_id = _required_env("VALIDATION_IMAGE_ID")
    artifact = _required_env("AAP_PLUGIN_OCI_ARTIFACT")
    asset_digest = _required_env("AAP_PLUGIN_ASSET_MANIFEST_SHA256")
    if (
        re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
        or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", artifact) is None
        or re.fullmatch(r"[0-9a-f]{64}", asset_digest) is None
    ):
        raise RuntimeError("qualification requires immutable image and manifest identities")
    if (
        _required_env("AAP_PLUGIN_REGISTRY_ID") != registry
        or _required_env("AAP_ROUTE_REGISTRY_ID") != registry
        or _required_env("AAP_PLUGIN_GPU_RESOURCE_ID") != resource
        or _required_env("AAP_PLUGIN_GPU_UUID") != gpu_uuid
        or _required_env("AAP_PLUGIN_DEVICE") != "cuda:0"
        or _required_env("AAP_PLUGIN_IMAGE_ID") != image_id
    ):
        raise RuntimeError("qualification and plugin deployment identities differ")
    for name in ("AAP_PLUGIN_ID", "AAP_PLUGIN_VERSION", "AAP_PLUGIN_TOKEN_FILE", "AAP_PLUGIN_DATA_DIR"):
        _required_env(name)
    if os.environ.get("AAP_PLUGIN_QUALIFICATION_FILE") or os.environ.get("AAP_PLUGIN_QUALIFICATION_SHA256"):
        raise RuntimeError("qualification cannot reuse an existing report")
    return registry, resource, gpu_uuid


@asynccontextmanager
async def _http_client(app: Any):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("qualification HTTP server exited before startup")
            await asyncio.sleep(0.01)
        bearer = Path(os.environ["AAP_PLUGIN_TOKEN_FILE"]).read_text().strip()
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {bearer}"},
            base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
            timeout=900,
        ) as client:
            yield client
    finally:
        server.should_exit = True
        try:
            await task
        finally:
            sock.close()


async def _bind_operator_routes(client: httpx.AsyncClient, *, registry_id: str, route_secret: str) -> None:
    key = secrets.token_bytes(32)
    command = {
        "schema": ROUTE_CONTROL_PROTOCOL, "command_id": str(uuid.uuid4()),
        "registry_id": registry_id, "authority_epoch": str(uuid.uuid4()),
        "retired_epoch": None, "challenge": secrets.token_hex(32),
        "action": "drain", "signing_key": None,
    }
    headers = {"Authorization": f"Bearer {route_secret}"}
    response = await client.post(ROUTE_CONTROL_PATH, json=command, headers=headers)
    response.raise_for_status()
    drained = response.json()
    if any(drained.get(field) != command[field] for field in ("registry_id", "command_id", "challenge", "authority_epoch")) or drained.get("phase") != "drained" or drained.get("active_requests") != 0:
        raise RuntimeError("operator route drain did not bind the reserved registry")
    command["action"] = "bind"
    command["signing_key"] = base64.urlsafe_b64encode(key).decode().rstrip("=")
    response = await client.post(ROUTE_CONTROL_PATH, json=command, headers=headers)
    response.raise_for_status()
    bound = response.json()
    if any(bound.get(field) != command[field] for field in ("registry_id", "command_id", "challenge", "authority_epoch")) or bound.get("backend_incarnation") != drained.get("backend_incarnation") or bound.get("phase") != "bound":
        raise RuntimeError("operator route bind did not match the drained backend")

    async def authorize(request: httpx.Request) -> None:
        if request.method == "GET":
            return
        now = int(time.time())
        claims = {
            "schema": ROUTE_ADMISSION_PROTOCOL, "aud": ROUTE_ADMISSION_AUDIENCE,
            "registry_id": registry_id, "backend_incarnation": bound["backend_incarnation"],
            "authority_epoch": command["authority_epoch"], "scope_id": str(uuid.uuid4()),
            "attempt_id": str(uuid.uuid4()), "operation": "gpu:unload" if request.url.path == "/unload" else "validation",
            "method": request.method, "path": request.url.path, "iat": now, "exp": now + 60,
        }
        request.headers[ROUTE_ADMISSION_HEADER] = jwt.encode(claims, key, algorithm="HS256", headers={"typ": "JWT"})

    client.event_hooks["request"].append(authorize)


def _gpu_query(fields: str) -> list[list[str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        [part.strip() for part in line.split(",")]
        for line in output.splitlines()
        if line.strip()
    ]


def _gpu_metadata(target_uuid: str | None) -> dict[str, Any]:
    rows = _gpu_query("uuid,memory.used,memory.total,driver_version")
    if target_uuid:
        matching = [row for row in rows if row[0] == target_uuid]
        if len(matching) != 1:
            raise RuntimeError(f"GPU UUID {target_uuid!r} is not visible")
        row = matching[0]
    elif len(rows) == 1:
        row = rows[0]
    else:
        raise RuntimeError(
            "VALIDATION_GPU_UUID is required when multiple GPUs are visible"
        )
    return {
        "uuid": row[0],
        "total_memory_mb": int(row[2]),
        "driver_version": row[3],
        "visible_device_count": len(rows),
    }


def _gpu_memory_mb(target_uuid: str) -> int:
    for row in _gpu_query("uuid,memory.used"):
        if row[0] == target_uuid:
            return int(row[1])
    raise RuntimeError(f"GPU UUID {target_uuid!r} is not visible")


def _gpu_memory_samples(target_uuid: str, *, count: int = 5) -> list[int]:
    samples = []
    for _ in range(count):
        samples.append(_gpu_memory_mb(target_uuid))
        time.sleep(0.5)
    return samples


def _gpu_compute_processes(target_uuid: str) -> list[dict[str, int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid, used = (part.strip() for part in line.split(",", 2))
        if gpu_uuid == target_uuid:
            processes.append(
                {"gpu_uuid": gpu_uuid, "pid": int(pid), "used_memory_mb": int(used)}
            )
    return processes


def _make_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "fixture.png"
    image = Image.new("RGB", (960, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 930, 290), outline="black", width=8)
    draw.text((90, 110), "AAP GPU LIFECYCLE 3090", fill="black")
    image.save(image_path)
    return image_path


def _approved_fixture(root: Path, expected: str, approval_ref: str) -> tuple[Path, dict[str, Any]]:
    path = _make_fixture(root)
    record = artifact_evidence(path, kind="fixture", approval_ref=approval_ref)
    if record["sha256"] != expected:
        raise RuntimeError("generated fixture does not match the approved SHA-256")
    return path, record


def _assert_unloaded(residency: dict[str, Any], snapshot: dict[str, Any]) -> None:
    assert residency["state"] == "unloaded"
    assert residency["gpu_loaded"] is False
    assert residency["evictable"] is False
    assert residency["active_requests"] == 0
    assert residency["builders"] == 0
    assert residency["borrowers"] == 0
    assert all(pool["resident"] is False for pool in residency["pools"].values())
    assert snapshot["current_size"] == 0
    assert snapshot["builders"] == 0
    assert snapshot["borrowers"] == 0
    assert snapshot["waiters"] == 0
    assert snapshot["cleanup_in_progress"] is False
    assert snapshot["cleanup_failed"] is False
    assert snapshot["gpu_resident"] is False
    assert snapshot["session_count"] == 0
    assert snapshot["engines"] == {}


def _assert_full_gpu_pool(snapshot: dict[str, Any]) -> None:
    if snapshot["cap"] != 3 or snapshot["current_size"] != 3:
        raise AssertionError(
            "RapidOCR validation requires all three representative engines resident"
        )
    if snapshot["session_count"] != 9 or len(snapshot["engines"]) != 3:
        raise AssertionError(
            "RapidOCR validation requires exactly nine owned ORT sessions"
        )
    if (
        snapshot["gpu_resident"] is not True
        or snapshot["device"] != "cuda"
        or snapshot["provider"] != "CUDAExecutionProvider"
    ):
        raise AssertionError("RapidOCR aggregate pool is not fully CUDA resident")
    for key, engine in snapshot["engines"].items():
        if (
            engine["resident"] is not True
            or engine["device"] != "cuda"
            or engine["provider"] != "CUDAExecutionProvider"
        ):
            raise AssertionError(f"RapidOCR engine {key!r} is not CUDA resident")
        if set(engine["sessions"]) != {"det", "cls", "rec"}:
            raise AssertionError(
                f"RapidOCR engine {key!r} does not own det/cls/rec sessions"
            )
        for component, providers in engine["sessions"].items():
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise AssertionError(
                    "RapidOCR runtime provider fallback detected for "
                    f"engine {key!r} component {component!r}: {providers!r}"
                )


def _assert_baseline_recovery(
    context: list[int], first: list[int], second: list[int], total_mb: int
) -> None:
    if abs(median(first) - median(second)) > 64:
        raise AssertionError("unloaded GPU baseline drifted between generations")
    slack_mb = max(512, int(total_mb * 0.02))
    if any(median(samples) > median(context) + slack_mb for samples in (first, second)):
        raise AssertionError("unloaded GPU memory did not return to context baseline")


def _approved_weights(
    *,
    model_dir: Path,
    relative_paths: list[str],
    approval_ref: str,
) -> list[dict[str, Any]]:
    try:
        manifest = json.loads(_required_env("VALIDATION_WEIGHT_SHA256_JSON"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("VALIDATION_WEIGHT_SHA256_JSON must be valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != set(relative_paths):
        raise RuntimeError("weight manifest must contain exactly the validated weights")
    records = []
    for relative_path in relative_paths:
        expected = manifest[relative_path]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise RuntimeError(f"invalid approved SHA-256 for {relative_path}")
        record = artifact_evidence(
            model_dir / relative_path,
            kind="weight",
            approval_ref=approval_ref,
        )
        if record["sha256"] != expected:
            raise RuntimeError(f"{relative_path} does not match the approved SHA-256")
        records.append(record)
    return records


async def _run() -> dict[str, Any]:
    registry_id, resource_id, requested_gpu_uuid = _deployment_identity()
    git_commit = _required_env("VALIDATION_GIT_COMMIT")
    image_id = _required_env("VALIDATION_IMAGE_ID")
    model_approval_ref = _required_env("VALIDATION_MODEL_APPROVAL_REF")
    fixture_approval_ref = _required_env("VALIDATION_FIXTURE_APPROVAL_REF")
    fixture_sha256 = _required_env("VALIDATION_FIXTURE_SHA256")
    if re.fullmatch(r"[0-9a-f]{64}", fixture_sha256) is None:
        raise RuntimeError("VALIDATION_FIXTURE_SHA256 must be a lowercase SHA-256 digest")

    private_root = tempfile.TemporaryDirectory(prefix="rapidocr-qualification-authority-")
    private_path = Path(private_root.name)
    private_key = Ed25519PrivateKey.generate()
    keyring_file = private_path / "gpu-keyring.json"
    keyring_file.write_text(json.dumps(
        {"validation": encode_ed25519_public_key(private_key.public_key())}
    ))
    keyring_file.chmod(0o600)
    route_secret = secrets.token_urlsafe(48)
    route_file = private_path / "route-token"
    route_file.write_text(route_secret)
    route_file.chmod(0o600)
    os.environ["AAP_PLUGIN_GPU_KEYRING_FILE"] = str(keyring_file)
    os.environ.pop("AAP_ROUTE_CONTROL_TOKEN", None)
    os.environ["AAP_ROUTE_CONTROL_TOKEN_FILE"] = str(route_file)
    os.environ["RAPIDOCR_BUILD_TIMEOUT"] = "300"
    os.environ["RAPIDOCR_IDLE_UNLOAD_SECONDS"] = "0"
    os.environ["RAPIDOCR_DEVICE"] = "gpu"

    app_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_root))
    import catalog  # noqa: PLC0415
    model_dir = Path(os.environ["AAP_PLUGIN_DATA_DIR"]) / "models"
    relative_paths = sorted({
        str(Path(path).relative_to(catalog.MODELS_DIR))
        for model_id, variants in POOL_TARGETS
        for path in (
            catalog.resolve(model_id, variants).det_path,
            catalog.resolve(model_id, variants).cls_path,
            catalog.resolve(model_id, variants).rec_path,
        )
    })
    approved_weights = _approved_weights(model_dir=model_dir, relative_paths=relative_paths, approval_ref=model_approval_ref)
    from plugin_assets import verify_assets  # noqa: PLC0415
    installed_weights = verify_assets(Path(os.environ["AAP_PLUGIN_DATA_DIR"]), os.environ["AAP_PLUGIN_ASSET_MANIFEST_SHA256"])
    if {(item["sha256"], item["size_bytes"]) for item in installed_weights} != {(item["sha256"], item["size_bytes"]) for item in approved_weights}:
        raise RuntimeError("installed assets do not match the exact approved qualifier weights")
    import main  # noqa: PLC0415


    validation_root = Path(tempfile.mkdtemp(prefix="rapidocr_lifecycle_validation_"))
    image_path, fixture = _approved_fixture(validation_root, fixture_sha256, fixture_approval_ref)
    gpu = _gpu_metadata(requested_gpu_uuid)
    target_uuid = gpu["uuid"]
    memory: dict[str, Any] = {"gpu_total_mb": gpu["total_memory_mb"]}
    if _gpu_compute_processes(target_uuid):
        raise RuntimeError("validation GPU is not isolated before CUDA init")
    contract_checks = {name: False for name in REQUIRED_CONTRACT_CHECKS}

    def token(
        scope: AdmissionScope,
        *,
        generation: str | None = None,
        control_epoch: str = "5",
        owner: str = "validation",
        operation: str = "validation",
        jti: str | None = None,
    ) -> str:
        control = scope in CONTROL_SCOPES
        claims = AdmissionTokenClaims(
            backend_registry_id=registry_id,
            gpu_resource_id=resource_id,
            boot_id=main._gpu_lifecycle.boot_id,  # type: ignore[union-attr] # noqa: SLF001
            generation=generation,
            control_epoch=control_epoch,
            scope=scope,
            jti=jti or f"validation-{scope.value}-{uuid.uuid4().hex}",
            exp=int(time.time()) + 1800,
            owner=owner if control else None,
            operation=operation if control else None,
        )
        return sign_admission_token(claims, private_key=private_key, kid="validation")

    def headers(scope: AdmissionScope, generation: str) -> dict[str, str]:
        return {
            GPU_GENERATION_HEADER: generation,
            GPU_ADMISSION_TOKEN_HEADER: token(scope, generation=generation),
        }

    async def checked_post(
        client: httpx.AsyncClient,
        path: str,
        body: dict[str, Any],
        request_headers: dict[str, str],
    ) -> dict[str, Any]:
        response = await client.post(path, json=body, headers=request_headers)
        if response.status_code >= 400:
            raise RuntimeError(
                f"{path} returned {response.status_code}: {response.text[:1000]}"
            )
        return response.json()

    try:
        async with main.lifespan(main.app):
            assert main._gpu_lifecycle is not None  # noqa: SLF001
            assert main._engine_pool is not None  # noqa: SLF001
            # Qualification-only authority: production startup remains unverified
            # until this exact deployment has passed and its evidence is approved.
            main._gpu_lifecycle = main.RapidOcrGpuLifecycle(
                main._engine_pool,
                verify_keyring=main.load_verify_keyring(keyring_file.read_text()),
                evictable_verified=True,
                **main.plugin_lifecycle_options(),
            )
            main._plugin_managed_lifecycle_verified = True
            async with _http_client(main.app) as client:
                await _bind_operator_routes(client, registry_id=registry_id, route_secret=route_secret)
                await checked_post(
                    client,
                    "/lifecycle/reset",
                    {"control_epoch": "1"},
                    {
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.RESET,
                            control_epoch="1",
                            operation="reset-1",
                        )
                    },
                )
                if _gpu_compute_processes(target_uuid):
                    raise RuntimeError(
                        "validation GPU is not isolated before CUDA init"
                    )
                gc.collect()
                memory["cold_context_baseline_mb"] = _gpu_memory_samples(target_uuid)
                await checked_post(
                    client, "/lifecycle/mode", {"gate": "enforce", "control_epoch": "2"},
                    {GPU_ADMISSION_TOKEN_HEADER: token(AdmissionScope.MODE, control_epoch="2", operation="prime-mode")},
                )
                # Prime all representative ORT roots once so the measured baseline
                # includes stable CUDA/cuDNN process state that survives releasing
                # every business session. Otherwise the first cycle incorrectly
                # counts that context as model memory which should be recoverable.
                for model_id, variants in POOL_TARGETS:
                    await checked_post(
                        client,
                        "/predict",
                        {
                            "tasks": [{"id": model_id, "file_path": str(image_path)}],
                            "context": {
                                "model_id": model_id,
                                "model_variants": variants,
                            },
                        },
                        headers(AdmissionScope.PREDICT, "1"),
                    )
                await main._engine_pool.unload_all(  # noqa: SLF001
                    reason="manual",
                    force_cleanup=True,
                )
                primed_snapshot = await main._engine_pool.snapshot()  # noqa: SLF001
                if (
                    primed_snapshot["current_size"] != 0
                    or primed_snapshot["session_count"] != 0
                    or primed_snapshot["engines"]
                ):
                    raise AssertionError(
                        "RapidOCR context priming did not release every ORT session"
                    )
                memory["context_baseline_mb"] = _gpu_memory_samples(target_uuid)
                # Return to legacy solely to reset the now-empty priming generation.
                await checked_post(
                    client, "/lifecycle/mode", {"gate": "legacy", "control_epoch": "3"},
                    {GPU_ADMISSION_TOKEN_HEADER: token(AdmissionScope.MODE, control_epoch="3", operation="prime-reset-mode")},
                )
                await checked_post(
                    client,
                    "/lifecycle/reset",
                    {"control_epoch": "4"},
                    {
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.RESET,
                            control_epoch="4",
                            operation="reset-4",
                        )
                    },
                )
                await checked_post(
                    client,
                    "/lifecycle/mode",
                    {"gate": "enforce", "control_epoch": "5"},
                    {
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.MODE,
                            control_epoch="5",
                            operation="mode-5",
                        )
                    },
                )
                setup = (await client.get("/setup")).json()
                if setup.get("registry_id") != registry_id or setup.get("gpu_resource_id") != resource_id or setup.get("physical_gpu_uuid") != requested_gpu_uuid:
                    raise AssertionError("native setup deployment identity differs")
                if "managed_lifecycle" not in setup:
                    raise AssertionError(
                        "managed lifecycle capability was not advertised"
                    )
                contract_checks["managed_lifecycle_advertised"] = True

                for model_id, variants in POOL_TARGETS:
                    await checked_post(
                        client,
                        "/predict",
                        {
                            "tasks": [{"id": model_id, "file_path": str(image_path)}],
                            "context": {
                                "model_id": model_id,
                                "model_variants": variants,
                            },
                        },
                        headers(AdmissionScope.PREDICT, "1"),
                    )
                contract_checks["real_inference"] = True
                first_snapshot = await main._engine_pool.snapshot()  # noqa: SLF001
                _assert_full_gpu_pool(first_snapshot)
                print(json.dumps({"event": "cycle_1_sessions", "snapshot": first_snapshot}, sort_keys=True), file=sys.stderr)
                if len(_gpu_compute_processes(target_uuid)) != 1:
                    raise RuntimeError("validator is not the sole GPU compute process")
                health = (await client.get("/health")).json()
                if health["compute"]["effective_provider"] != "CUDAExecutionProvider":
                    raise AssertionError(
                        f"RapidOCR did not remain on CUDA: {health['compute']}"
                    )
                if health["residency"]["evictable"] is not True:
                    raise AssertionError("resident RapidOCR pool was not evictable")
                contract_checks["provider_or_device_gpu"] = True
                memory["cycle_1_loaded_mb"] = _gpu_memory_samples(target_uuid, count=3)

                await checked_post(
                    client,
                    "/drain",
                    {"generation": "2"},
                    {
                        GPU_GENERATION_HEADER: "2",
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.DRAIN,
                            generation="2",
                            operation="cycle-1",
                        ),
                    },
                )
                first_unload = await checked_post(
                    client,
                    "/unload",
                    {"generation": "2"},
                    {
                        GPU_GENERATION_HEADER: "2",
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.UNLOAD,
                            generation="2",
                            operation="cycle-1",
                        ),
                    },
                )
                memory["cycle_1_unloaded_mb"] = _gpu_memory_samples(target_uuid)
                _assert_unloaded(
                    first_unload["residency"],
                    await main._engine_pool.snapshot(),  # noqa: SLF001
                )

                for model_id, variants in POOL_TARGETS:
                    await checked_post(
                        client,
                        "/warmup",
                        {"context": {"model_id": model_id, "model_variants": variants}},
                        headers(AdmissionScope.WARMUP, "3"),
                    )
                second_loaded_snapshot = await main._engine_pool.snapshot()  # noqa: SLF001
                _assert_full_gpu_pool(second_loaded_snapshot)
                print(json.dumps({"event": "cycle_2_sessions", "snapshot": second_loaded_snapshot}, sort_keys=True), file=sys.stderr)
                memory["cycle_2_loaded_mb"] = _gpu_memory_samples(target_uuid, count=3)
                workload_body = {
                    "context": {
                        "model_id": POOL_TARGETS[0][0],
                        "model_variants": POOL_TARGETS[0][1],
                    }
                }
                contract_checks.update(
                    await exercise_lifecycle_fault_matrix(
                        client=client,
                        lifecycle=main._gpu_lifecycle,  # noqa: SLF001
                        token=token,
                        workload_scope=AdmissionScope.WARMUP,
                        drain_scope=AdmissionScope.DRAIN,
                        unload_scope=AdmissionScope.UNLOAD,
                        workload_path="/warmup",
                        workload_body=workload_body,
                        current_generation="3",
                        stale_generation="1",
                        next_generation="4",
                        drain_owner="cycle-2",
                    )
                )
                second_unload = await checked_post(
                    client,
                    "/unload",
                    {"generation": "4"},
                    {
                        GPU_GENERATION_HEADER: "4",
                        GPU_ADMISSION_TOKEN_HEADER: token(
                            AdmissionScope.UNLOAD,
                            generation="4",
                            operation="cycle-2",
                        ),
                    },
                )
                final_health = (await client.get("/health")).json()
                memory["cycle_2_unloaded_mb"] = _gpu_memory_samples(target_uuid)
                final_snapshot = await main._engine_pool.snapshot()  # noqa: SLF001
                _assert_unloaded(final_health["residency"], final_snapshot)
                if second_unload["residency"]["gpu_loaded"] is not False:
                    raise AssertionError("managed unload did not clear GPU residency")
                contract_checks["full_cleanup"] = True

            _assert_baseline_recovery(
                memory["context_baseline_mb"],
                memory["cycle_1_unloaded_mb"],
                memory["cycle_2_unloaded_mb"],
                memory["gpu_total_mb"],
            )
            artifacts = _approved_weights(
                model_dir=model_dir,
                relative_paths=relative_paths,
                approval_ref=model_approval_ref,
            )
            artifacts.append(fixture)
            gpu["runtime_version"] = main._cuda_runtime  # noqa: SLF001
            cycles = [
                memory_cycle_evidence(
                    cycle=1,
                    generation="2",
                    context_samples_mb=memory["context_baseline_mb"],
                    loaded_samples_mb=memory["cycle_1_loaded_mb"],
                    unloaded_samples_mb=memory["cycle_1_unloaded_mb"],
                ),
                memory_cycle_evidence(
                    cycle=2,
                    generation="4",
                    context_samples_mb=memory["context_baseline_mb"],
                    loaded_samples_mb=memory["cycle_2_loaded_mb"],
                    unloaded_samples_mb=memory["cycle_2_unloaded_mb"],
                ),
            ]
            for cycle, snapshot in zip(cycles, (first_snapshot, second_loaded_snapshot)):
                cycle["session_observations"] = {
                    "engines": [
                        engine["sessions"][component]
                        for _, engine in sorted(snapshot["engines"].items())
                        for component in ("det", "cls", "rec")
                    ]
                }
            return build_managed_lifecycle_evidence(
                backend_name=os.environ["AAP_PLUGIN_ID"],
                deployment={
                    "git_commit": git_commit,
                    "image_id": image_id,
                    "runtime_versions": {
                        **main._plugin_runtime_versions(),  # noqa: SLF001
                        "plugin_id": os.environ["AAP_PLUGIN_ID"],
                        "plugin_version": os.environ["AAP_PLUGIN_VERSION"],
                        "artifact": os.environ["AAP_PLUGIN_OCI_ARTIFACT"],
                        "asset_manifest_sha256": os.environ["AAP_PLUGIN_ASSET_MANIFEST_SHA256"],
                        "registry_id": registry_id,
                        "gpu_resource_id": resource_id,
                        "qualification_authority": "operator-ephemeral",
                    },
                    "pool_topology": {
                        "registry_id": registry_id,
                        "gpu_resource_id": resource_id,
                        "pools": {"engines": {"cap": int(final_snapshot["cap"])}},
                        "session_counts": {"engines": 9},
                    },
                },
                artifacts=artifacts,
                gpu=gpu,
                cycles=cycles,
                contract_checks=contract_checks,
                final_residency=final_health["residency"],
                runtime_ephemera_clean=True,
            )
    finally:
        private_root.cleanup()
        shutil.rmtree(validation_root, ignore_errors=True)


def _failed_evidence(exc: Exception) -> dict[str, Any]:
    return build_managed_lifecycle_evidence(
        backend_name="rapidocr-backend",
        deployment={
            "git_commit": os.getenv("VALIDATION_GIT_COMMIT") or None,
            "image_id": os.getenv("VALIDATION_IMAGE_ID") or None,
            "runtime_versions": {"python": platform.python_version()},
            "pool_topology": {},
        },
        artifacts=[],
        gpu={
            "uuid": os.getenv("VALIDATION_GPU_UUID") or None,
            "total_memory_mb": None,
            "driver_version": None,
            "runtime_version": None,
            "visible_device_count": None,
        },
        cycles=[],
        contract_checks={name: False for name in REQUIRED_CONTRACT_CHECKS},
        final_residency=None,
        runtime_ephemera_clean=False,
        blockers=[
            {
                "code": "validation_exception",
                "message": f"validation failed with {type(exc).__name__}",
            }
        ],
    )


def main() -> None:
    try:
        with redirect_stdout(sys.stderr):
            evidence = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - emit fail-closed evidence before exit
        print(f"managed lifecycle validation failed: {exc}", file=sys.stderr)
        print(json.dumps(_failed_evidence(exc), ensure_ascii=False, sort_keys=True))
        raise SystemExit(1) from exc
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
