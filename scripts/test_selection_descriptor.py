import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("selection_descriptor", Path(__file__).with_name("generate-selection-descriptor.py"))
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class SelectionDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, plugin):
        return generator.build(plugin, "1", "registry.example/plugin@sha256:" + "1" * 64, "2" * 64, self.root)

    def lookup(self, descriptor, model, **variants):
        return next(item for item in descriptor["selections"] if item["model_id"] == model and item["variants"] == variants)

    def test_libreyolo_uses_native_series_and_missing_asset_paths(self):
        result = self.build("libreyolo")
        self.assertEqual({item["variants"]["size"] for item in result["selections"]}, {"t", "s", "m", "c"})
        for size in ("t", "s", "m", "c"):
            entry = self.lookup(result, "detect", series="yolov9", size=size)
            self.assertFalse(entry["asset_free"])
            self.assertEqual([item["path"] for item in entry["assets"]], [f"assets/LibreYOLO9{size}.pt"])
            self.assertFalse((self.root / entry["assets"][0]["path"]).exists())

    def test_ocr_missing_small_remains_three_expected_assets(self):
        result = self.build("rapidocr")
        for model, variants in (("ocr-det", {"version": "v6", "size": "small"}), ("ocr-e2e", {"version": "v6", "size": "small", "lang": "universal"})):
            entry = self.lookup(result, model, **variants)
            self.assertFalse(entry["asset_free"])
            self.assertEqual(len(entry["assets"]), 3)
            self.assertTrue(all(not (self.root / item["path"]).exists() for item in entry["assets"]))

    def test_onnx_distinct_operation_dependencies(self):
        result = self.build("onnxtools")
        detector = self.lookup(result, "vehicle-detect")["assets"]
        classifier = self.lookup(result, "vehicle-attr-classify")["assets"]
        pipeline = self.lookup(result, "vehicle-attr")["assets"]
        self.assertEqual(len(detector), 1)
        self.assertEqual(len(classifier), 1)
        self.assertEqual(pipeline, sorted(detector + classifier, key=lambda item: item["path"]))

    def test_yolo_dependencies_and_prepared_precedence(self):
        result = self.build("yolo")
        def paths(model, series="yolo11", size="s"):
            return {a["path"] for a in self.lookup(result, model, series=series, size=size)["assets"]}
        self.assertEqual(paths("track"), paths("detect"))
        self.assertIn("checkpoints/weights/clip/ViT-B-32.pt", paths("detect-world", "yolo-worldv2"))
        self.assertIn("checkpoints/mobileclip_blt.ts", paths("detect-yoloe", "yoloe-11"))
        self.assertEqual(paths("exemplar-yoloe", "yoloe-11"), {"checkpoints/yoloe-11s-seg.pt"})
        item = self.lookup(result, "detect", series="yolo11", size="s")["assets"][0]
        cache = self.root / "model-assets"
        cache.mkdir()
        (cache / "assets.json").write_text(json.dumps({"version": 1, "assets": [{"filename": "yolo11s.pt", "sha256": item["sha256"], "size_bytes": item["size_bytes"]}]}))
        fallback = self.build("yolo")
        self.assertEqual(self.lookup(fallback, "detect", series="yolo11", size="s")["assets"][0]["path"], "model-assets/yolo11s.pt")
        (cache / (item["sha256"] + ".pt")).touch()
        prepared = self.build("yolo")
        self.assertEqual(self.lookup(prepared, "detect", series="yolo11", size="s")["assets"][0]["path"], "model-assets/" + item["sha256"] + ".pt")
        (cache / "assets.json").write_text(json.dumps({"version": 1, "assets": [{"filename": "yolo11s.pt", "sha256": "0" * 64, "size_bytes": item["size_bytes"]}]}))
        with self.assertRaisesRegex(ValueError, "differs"):
            self.build("yolo")

    def test_rejects_duplicate_selections_and_paths(self):
        groups = generator.selections_onnx(self.root)
        model, variants, assets = groups[0]
        for malformed in (groups + [groups[0]], [(model, variants, assets + assets)]):
            with self.subTest(malformed=malformed), patch.object(generator, "selections_onnx", return_value=malformed):
                with self.assertRaises(ValueError):
                    self.build("onnxtools")

    def test_rejects_noncanonical_and_symlink_escape(self):
        for path in ("../outside", "/absolute", "models//x", "models/./x", "models\\x"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                generator.safe_path(self.root, path)
        (self.root / "models").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.build("onnxtools")


if __name__ == "__main__":
    unittest.main()
