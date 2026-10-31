"""Retire real session owners through the native lifecycle and video pool."""
import asyncio
import time

import pytest
from aap_backend_runtime import TrackerSessionLost, TrackerSessionManager
from aap_protocol_v2.errors import LifecycleHTTPError
from aap_protocol_v2.lifecycle import AdmissionScope, AdmissionTokenClaims, LifecycleModeRequest, sign_admission_token
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gpu_lifecycle import GroundedSam2GpuLifecycle
from model_pool import ModelPool
from pool_domain import GroundedSam2Pools
from video_pool import VideoPool


class Tracker:
    device = "cpu"
    cleanup_uncertain = False

    def __init__(self):
        self.active_sessions = 0


def fixture():
    sessions = TrackerSessionManager(ttl_seconds=300)
    image = ModelPool(1, lambda *_: None, lambda: None)
    video = VideoPool(1, lambda _variant: Tracker(), lambda: None, idle_unload_seconds=0)
    private = Ed25519PrivateKey.generate()
    lifecycle = GroundedSam2GpuLifecycle(
        GroundedSam2Pools(image, video), verify_keyring={"current": private.public_key()},
        expected_registry_id="backend-1", expected_gpu_resource_id="node-a/GPU-1",
        evictable_verified=True, allow_legacy_workloads=False, retire_sessions=sessions.shutdown,
    )
    return sessions, video, lifecycle, private


def token(lifecycle, key, scope, *, generation=None, registry="backend-1", jti="token"):
    control = scope in {AdmissionScope.MODE, AdmissionScope.DRAIN}
    claims = AdmissionTokenClaims(
        backend_registry_id=registry, gpu_resource_id="node-a/GPU-1", boot_id=lifecycle.boot_id,
        control_epoch="1", generation=generation, scope=scope, jti=jti,
        owner="owner" if control else None, operation="retire" if control else None,
        exp=int(time.time()) + 300,
    )
    return sign_admission_token(claims, private_key=key, kid="current")


async def open_session(sessions, video, closed, *, opening=None, release=None):
    async def start(lease):
        lease.tracker.active_sessions += 1
        if opening is not None:
            opening.set()
            await release.wait()
        return object()

    async def close(lease, _state):
        lease.tracker.active_sessions -= 1
        closed.append("closed")

    return await sessions.start(binding=("owner",), borrow=lambda: video.borrow("tiny"), open_session=start, close_session=close)


async def promote(lifecycle, key):
    await lifecycle.set_mode(LifecycleModeRequest(gate="enforce", control_epoch="1"), token=token(lifecycle, key, AdmissionScope.MODE, jti="mode"))


def test_signed_retirement_releases_real_idle_session_and_rejects_invalid_identity():
    async def run():
        sessions, video, lifecycle, key = fixture()
        closed = []
        session = await open_session(sessions, video, closed)
        before = await video.snapshot()
        assert before["borrowers"] == 1 and before["active_sessions"] == 1
        with pytest.raises(LifecycleHTTPError):
            await lifecycle.terminal_drain_signed("1", generation_header="1", token=token(lifecycle, key, AdmissionScope.DRAIN, generation="1", registry="wrong", jti="invalid"))
        with pytest.raises(LifecycleHTTPError):
            await lifecycle.terminal_drain_signed("1", generation_header="1", token=token(lifecycle, Ed25519PrivateKey.generate(), AdmissionScope.DRAIN, generation="1", jti="bad-signature"))
        assert closed == [] and (await video.snapshot())["borrowers"] == 1
        result = await asyncio.wait_for(lifecycle.terminal_drain_signed("1", generation_header="1", token=token(lifecycle, key, AdmissionScope.DRAIN, generation="1", jti="valid")), 2)
        assert result["pool_ids"] == ["image", "video"] and closed == ["closed"]
        snapshot = await video.snapshot()
        assert snapshot["borrowers"] == 0 and snapshot["current_size"] == 0 and snapshot["active_sessions"] == 0
        with pytest.raises(TrackerSessionLost):
            await sessions.call(session, ("owner",), lambda _lease, _state: None)
        await lifecycle.shutdown()
    asyncio.run(run())


def test_signed_retirement_waits_for_admitted_start_and_real_future():
    async def run():
        sessions, video, lifecycle, key = fixture()
        await promote(lifecycle, key)
        operation = await lifecycle.begin_workload(AdmissionScope.PREDICT, generation_header="1", token=token(lifecycle, key, AdmissionScope.PREDICT, generation="1", jti="predict"))
        opening, release = asyncio.Event(), asyncio.Event()
        closed = []
        start = asyncio.create_task(open_session(sessions, video, closed, opening=opening, release=release))
        await opening.wait()
        terminal = asyncio.create_task(lifecycle.terminal_drain_signed("2", generation_header="2", token=token(lifecycle, key, AdmissionScope.DRAIN, generation="2", jti="drain")))
        await asyncio.sleep(0)
        assert not terminal.done() and closed == [] and (await video.snapshot())["borrowers"] == 1
        release.set()
        await start
        work = asyncio.get_running_loop().create_future()
        operation.track_future(work)
        close_operation = asyncio.create_task(operation.close())
        await asyncio.sleep(0)
        assert not terminal.done() and closed == []
        work.set_result(None)
        await close_operation
        proof = await asyncio.wait_for(terminal, 2)
        assert proof["unloaded"] and closed == ["closed"]
        snapshot = await video.snapshot()
        assert snapshot["borrowers"] == 0 and snapshot["current_size"] == 0 and snapshot["active_sessions"] == 0
        await lifecycle.shutdown()
    asyncio.run(run())


@pytest.mark.parametrize("decrement_first", [False, True])
def test_retirement_fails_closed_when_session_cleanup_fails(decrement_first):
    async def run():
        sessions, video, lifecycle, key = fixture()

        async def start(lease):
            lease.tracker.active_sessions += 1
            return object()

        async def close(lease, _state):
            if decrement_first:
                lease.tracker.active_sessions -= 1
            raise RuntimeError("tracker cleanup failed")

        await sessions.start(binding=("owner",), borrow=lambda: video.borrow("tiny"), open_session=start, close_session=close)
        with pytest.raises(LifecycleHTTPError):
            await asyncio.wait_for(lifecycle.terminal_drain_signed(
                "1", generation_header="1",
                token=token(lifecycle, key, AdmissionScope.DRAIN, generation="1", jti="drain-failure"),
            ), 1)
        residency = await lifecycle.residency()
        assert residency.state.value != "unloaded"

    asyncio.run(run())
