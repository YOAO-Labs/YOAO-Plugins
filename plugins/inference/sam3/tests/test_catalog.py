from sam3_plugin.catalog import deployment_selections, descriptor
import pytest


def test_selections_cover_all_models_and_their_real_checkpoint(tmp_path):
    selections = {model: (variants, assets) for model, variants, assets in deployment_selections(tmp_path)}
    assert set(selections) == {
        'sam3-detection', 'sam3-segmentation', 'sam3-interactive-seg',
        'sam3-video-tracker', 'sam3-video-interactive-tracker',
    }
    for model, (variants, assets) in selections.items():
        assert variants == {'model_variant': 'sam3'}
        expected = 'checkpoints/sam3.1_multiplex.pt' if model == 'sam3-video-tracker' else 'checkpoints/sam3.pt'
        assert assets[0]['path'] == expected
        assert len(assets[0]['sha256']) == 64 and assets[0]['size_bytes'] > 0
    assert not list(tmp_path.iterdir())


def test_selection_refuses_unverified_checkpoint_identity(tmp_path, monkeypatch):
    catalog = descriptor()
    catalog['assets'][0]['sha256'] = None
    monkeypatch.setattr('sam3_plugin.catalog.descriptor', lambda: catalog)
    with pytest.raises(ValueError, match='Unverified published asset identity'):
        deployment_selections(tmp_path)
