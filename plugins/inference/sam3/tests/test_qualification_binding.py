import pytest

from scripts.validate_managed_lifecycle import _native_binding


@pytest.fixture
def native_env(monkeypatch, tmp_path):
    registry = '11111111-1111-4111-8111-111111111111'
    gpu = 'GPU-00000000-0000-0000-0000-000000000000'
    image = 'sha256:' + 'a' * 64
    values = {
        'VALIDATION_REGISTRY_ID': registry, 'AAP_PLUGIN_REGISTRY_ID': registry,
        'AAP_ROUTE_REGISTRY_ID': registry, 'VALIDATION_GPU_RESOURCE_ID': 'node/' + gpu,
        'AAP_PLUGIN_GPU_RESOURCE_ID': 'node/' + gpu, 'AAP_PLUGIN_GPU_UUID': gpu,
        'VALIDATION_GPU_UUID': gpu, 'VALIDATION_IMAGE_ID': image, 'AAP_PLUGIN_IMAGE_ID': image,
        'AAP_PLUGIN_ID': 'sam3', 'AAP_PLUGIN_VERSION': '0.10.0', 'AAP_PLUGIN_DEVICE': 'cuda:0',
        'AAP_PLUGIN_OCI_ARTIFACT': 'example/sam3@' + image,
        'AAP_PLUGIN_ASSET_MANIFEST_SHA256': 'b' * 64,
        'AAP_PLUGIN_TOKEN_FILE': str(tmp_path / 'token'), 'AAP_PLUGIN_DATA_DIR': str(tmp_path),
        'CUDA_VERSION': '12.8.0',
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('AAP_PLUGIN_QUALIFICATION_FILE', raising=False)
    monkeypatch.delenv('AAP_PLUGIN_QUALIFICATION_SHA256', raising=False)
    return values


def test_native_binding_uses_matching_installation_identity(native_env):
    registry, resource, plugin, version = _native_binding()
    assert (registry, resource, plugin, version) == (
        native_env['AAP_PLUGIN_REGISTRY_ID'], native_env['AAP_PLUGIN_GPU_RESOURCE_ID'], 'sam3', '0.10.0'
    )


@pytest.mark.parametrize('field', [
    'AAP_PLUGIN_REGISTRY_ID', 'AAP_ROUTE_REGISTRY_ID', 'AAP_PLUGIN_GPU_RESOURCE_ID',
    'AAP_PLUGIN_GPU_UUID', 'AAP_PLUGIN_IMAGE_ID', 'AAP_PLUGIN_DEVICE',
])
def test_native_binding_rejects_identity_drift(native_env, monkeypatch, field):
    monkeypatch.setenv(field, 'different')
    with pytest.raises(RuntimeError, match='identities differ'):
        _native_binding()


def test_native_binding_rejects_reusing_qualification(native_env, monkeypatch):
    monkeypatch.setenv('AAP_PLUGIN_QUALIFICATION_FILE', '/tmp/old-report')
    with pytest.raises(RuntimeError, match='cannot reuse'):
        _native_binding()
