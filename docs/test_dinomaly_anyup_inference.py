"""Run: uv run --no-sync python -B docs/test_dinomaly_anyup_inference.py

Uses synthetic tensors and temporary images; never loads a Dinomaly checkpoint
or modifies the user's image/result directories.
"""

import csv
import runpy
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
NEW = runpy.run_path(str(ROOT / "docs/dinomaly_anyup_inference.py"))
OLD = runpy.run_path(str(ROOT / "dinomaly_anyup_inference.py"))
from anyup.layers.attention.attention_masking import compute_attention_mask


class SimilarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def compare(self, device):
        torch.manual_seed(73)
        model = NEW["AnyUp"](qk_dim=16, lfu_dim=16, num_heads=4).eval().to(device)
        for batch, size, chunk, window in [
            (1, (16, 16), 128, 0.1),
            (2, (18, 21), 17, 0.1),
            (2, (12, 16), 512, 0.0),
            (1, (12, 16), None, 0.1),
        ]:
            with self.subTest(device=device, batch=batch, size=size, chunk=chunk, window=window):
                model.cross_decode.window_ratio = window
                image = torch.randn(batch, 3, 16, 20, device=device)
                en = [torch.randn(batch, 8, 4, 5, device=device) for _ in range(2)]
                de = [x + torch.randn_like(x) * 0.4 for x in en]
                with torch.no_grad():
                    expected = OLD["compute_anomaly_map_anyup"](model, image, en, de, size, chunk)
                    actual = NEW["compute_anomaly_map_anyup"](model, image, en, de, size, chunk)
                self.assertEqual(actual.shape, (batch, 1, *size))
                self.assertTrue(actual.isfinite().all())
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

    def test_cpu_matches_original_including_batch_two(self):
        self.compare("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_original_including_batch_two(self):
        self.compare("cuda")

    def test_chunk_masks_exactly_match_original(self):
        for height, width, feat_h, feat_w, ratio in [(17, 23, 28, 28, 0.1), (32, 40, 7, 9, 0.25)]:
            expected = compute_attention_mask(height, width, feat_h, feat_w, ratio)
            windows = NEW["window2d"]((feat_h, feat_w), (height, width), ratio).reshape(-1, 4)
            actual = torch.cat([
                NEW["_attention_mask_chunk"](windows, start, min(start + 127, height * width),
                                               (feat_h, feat_w), torch.device("cpu"))
                for start in range(0, height * width, 127)
            ])
            self.assertTrue(torch.equal(actual, expected))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="relp-anyup-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "sample image.png"
        self.output = self.folder / "results"
        self.output.mkdir()
        self.assertTrue(cv2.imwrite(str(self.source), np.full((16, 20, 3), 120, dtype=np.uint8)))
        self.args = SimpleNamespace(input_dim=16, compare_baseline=False,
                                    upsample_sizes=[16, 32, 48], q_chunk_size=7, threshold=0.15)
        self.process = NEW["process_single_comparison"]

    @staticmethod
    def fake_model(image):
        return [torch.ones(1, 4, 2, 2)], [torch.ones(1, 4, 2, 2)]

    def statuses(self):
        path = self.output / "sample image_inference_status.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["method"]: row for row in csv.DictReader(handle)}

    def check_oom(self, failure_stage):
        old_paths = NEW["_result_paths"](self.output, self.source.stem, "AnyUp_32")
        for path in old_paths.values():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"previous run")
        unrelated = self.output / "unrelated.txt"
        unrelated.write_text("keep")
        tensor_refs = []
        liveness_at_cleanup = []

        def raise_oom(tensor):
            temporary = torch.ones(8)
            tensor_refs.extend([weakref.ref(tensor), weakref.ref(temporary)])
            raise torch.AcceleratorError("CUDA out of memory (simulated)")

        def compute(anyup, img, en, de, target_size, q_chunk_size):
            self.assertEqual(q_chunk_size, 7)
            tensor = torch.zeros(1, 1, *target_size)
            if target_size[0] == 32 and failure_stage == "compute":
                raise_oom(tensor)
            return tensor

        def gaussian(tensor):
            if tensor.shape[-1] == 32 and failure_stage == "gaussian":
                raise_oom(tensor)
            return tensor

        def cleanup(device):
            liveness_at_cleanup.append([ref() is not None for ref in tensor_refs])

        with patch.dict(self.process.__globals__, {
            "compute_anomaly_map_anyup": compute, "release_compute_memory": cleanup,
        }):
            self.process(self.source, self.fake_model, None, self.args, self.output,
                         torch.device("cpu"), gaussian)

        self.assertTrue(tensor_refs)
        self.assertTrue(all(not any(alive) for alive in liveness_at_cleanup))
        self.assertTrue(all(not path.exists() for path in old_paths.values()))
        self.assertEqual(unrelated.read_text(), "keep")
        statuses = self.statuses()
        self.assertEqual({method: row["status"] for method, row in statuses.items()},
                         {"AnyUp_16": "success", "AnyUp_32": "oom", "AnyUp_48": "success"})
        self.assertIn("out of memory", statuses["AnyUp_32"]["message"])
        for method in ("AnyUp_16", "AnyUp_48"):
            for path in NEW["_result_paths"](self.output, self.source.stem, method).values():
                self.assertIsNotNone(cv2.imread(str(path)))
        banner = cv2.imread(str(self.output / "sample image_comparison_grid_heatmaps.jpg"))
        self.assertEqual(banner.shape[:2], (448, 448 * 4))  # Includes the skipped panel.

    def test_compute_oom_clears_old_outputs_and_releases_traceback(self):
        self.check_oom("compute")

    def test_gaussian_oom_releases_computed_map_before_cleanup(self):
        self.check_oom("gaussian")

    def test_unreadable_image_records_failure(self):
        self.source.unlink()
        self.process(self.source, self.fake_model, None, self.args, self.output,
                     torch.device("cpu"), lambda tensor: tensor)
        self.assertTrue(all(row["status"] == "read_error" for row in self.statuses().values()))

    def test_failed_image_write_does_not_claim_success(self):
        self.args.upsample_sizes = [16]
        with patch.dict(self.process.__globals__, {
            "compute_anomaly_map_anyup": lambda *a, **kw: torch.zeros(1, 1, 16, 16),
            "release_compute_memory": lambda device: None,
        }), patch.object(cv2, "imwrite", return_value=False):
            with self.assertRaises(OSError):
                self.process(self.source, self.fake_model, None, self.args, self.output,
                             torch.device("cpu"), lambda tensor: tensor)
        self.assertEqual(self.statuses()["AnyUp_16"]["status"], "pending")


if __name__ == "__main__":
    unittest.main(verbosity=2)
