import pretrained_manifest


def test_descriptor_covers_supported_variants_and_license_boundaries():
    descriptor = pretrained_manifest.descriptor()
    assert (descriptor['version'], descriptor['plugin_id'], descriptor['plugin_version']) == (1, 'rapidocr', '0.1.0')
    assets = {item['path']: item for item in descriptor['assets']}
    assert len(assets) == 13
    det = assets['PP-OCRv6/det/PP-OCRv6_det_small.onnx']
    assert det['size_bytes'] == 9929594
    assert det['sha256'] == '090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f'
    assert det['license']['status'] == 'verified' and det['license']['identifier'] == 'Apache-2.0'
    cls = assets['PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx']
    assert cls['license']['status'] == 'unverified' and cls['license']['identifier'] is None
    for model in ('ocr-det', 'ocr-rec', 'ocr-e2e'):
        entries = [item for item in (det, cls, assets['PP-OCRv6/rec/PP-OCRv6_rec_small.onnx']) if any(m['model_id'] == model and m['variants']['version'] == 'v6' and m['variants']['size'] == 'small' for m in item['models'])]
        assert len(entries) == 3
    assert descriptor['runtime_compatibility']['sessions_per_engine'] == ['det', 'cls', 'rec']
    assert all(a['size_bytes'] > 0 and a['source'].startswith('https://') and len(a['sha256']) == 64 for a in assets.values())
