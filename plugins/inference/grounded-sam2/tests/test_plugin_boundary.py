import os
import subprocess
import sys
import textwrap


def test_cpu_plugin_http_and_terminal_retirement(tmp_path):
    token = "plugin-token-" + "x" * 48
    token_file = tmp_path / "token"
    token_file.write_text(token, encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    registry = "12345678-1234-4234-8234-123456789abc"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("AAP_", "GPU_LIFECYCLE_"))}
    env.update(
        AAP_PLUGIN_TOKEN_FILE=str(token_file), AAP_PLUGIN_DATA_DIR=str(data),
        AAP_PLUGIN_DEVICE="cpu", AAP_PLUGIN_ID="grounded-sam2",
        AAP_PLUGIN_VERSION="0.10.1", AAP_PLUGIN_REGISTRY_ID=registry,
        AAP_ROUTE_REGISTRY_ID=registry, CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="none",
        PREFETCH_SAM_VARIANTS="large", PREFETCH_DINO_VARIANTS="B",
    )
    code = textwrap.dedent('''
        import asyncio, os
        import main, predictor, video_predictor
        from fastapi.testclient import TestClient
        from pathlib import Path

        assert Path(predictor.CHECKPOINT_DIR) == Path(os.environ['AAP_PLUGIN_DATA_DIR']) / 'checkpoints'
        assert video_predictor.CHECKPOINT_DIR == predictor.CHECKPOINT_DIR
        main.init_perfhud_collectors = lambda: None
        main.shutdown_perfhud_collectors = lambda: None
        main.torch.cuda.is_available = lambda: False
        main.sample_perfhud = lambda: {'container_cpu_percent': 0, 'container_memory_percent': 0}
        attempted = []
        async def forbidden(*args, **kwargs):
            attempted.append(args)
            raise AssertionError('startup attempted asset download')
        asyncio.create_subprocess_exec = forbidden
        token = Path(os.environ['AAP_PLUGIN_TOKEN_FILE']).read_text()
        auth = {'Authorization': 'Bearer ' + token}
        with TestClient(main.app) as client:
            health = client.get('/health')
            assert health.status_code == 200, health.text
            assert health.json()['compute']['configured_device'] == 'cpu'
            assert client.get('/setup').status_code == 403
            setup = client.get('/setup', headers=auth)
            assert setup.status_code == 200, setup.text
            assert setup.json()['plugin_identity']['plugin_id'] == 'grounded-sam2'
            assert setup.json()['pool_topology'] == {'image': {'cap': 1}, 'video': {'cap': 1}}
            body = {'registry_id': os.environ['AAP_PLUGIN_REGISTRY_ID'], 'boot_id': health.json()['residency']['boot_id'], 'operation_id': 'retire-1'}
            response = client.post('/lifecycle/terminal-drain', json=body, headers=auth)
            assert response.status_code == 200, response.text
            assert response.json()['pool_ids'] == ['image', 'video']
            assert not attempted
    ''')
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
