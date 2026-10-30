"""Fixed externally visible variant and routing expectations."""
import pytest
import catalog

DET = [("v5", "mobile"), ("v5", "server"), ("v6", "tiny"), ("v6", "small"), ("v6", "medium")]
REC = [("v5", "mobile", "universal"), ("v5", "server", "universal"),
       ("v5", "mobile", "en"), ("v6", "tiny", "universal"),
       ("v6", "small", "universal"), ("v6", "medium", "universal")]

@pytest.mark.parametrize("version,size", DET)
def test_detection_variants(version, size):
    engine = catalog.resolve("ocr-det", {"version": version, "size": size})
    assert (engine.use_det, engine.use_cls, engine.use_rec) == (True, False, False)

@pytest.mark.parametrize("model,flags", [("ocr-rec", (False, True, True)), ("ocr-e2e", (True, True, True))])
@pytest.mark.parametrize("version,size,lang", REC)
def test_recognition_variants(model, flags, version, size, lang):
    engine = catalog.resolve(model, {"version": version, "size": size, "lang": lang})
    assert (engine.use_det, engine.use_cls, engine.use_rec) == flags
    assert engine.lang == lang

@pytest.mark.parametrize("model", ["ocr-rec", "ocr-e2e"])
@pytest.mark.parametrize("variants", [{"version": "v6", "size": "small", "lang": "en"}, {"version": "v5", "size": "server", "lang": "en"}, {"version": "v6", "size": "mobile", "lang": "universal"}])
def test_invalid_combinations_rejected(model, variants):
    with pytest.raises(ValueError):
        catalog.resolve(model, variants)
