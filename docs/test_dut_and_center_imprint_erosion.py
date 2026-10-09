"""Geometry, pixel preservation, pairing, and safe output checks (no ML models)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "dut_and_center_imprint_erosion.py"
spec = importlib.util.spec_from_file_location("dut_erosion", SCRIPT)
erosion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(erosion)


class ErosionTests(unittest.TestCase):
    def test_rectangle_shrinks_exact_number_of_pixel_layers(self):
        mask = np.zeros((24, 32), np.uint8)
        mask[2:22, 4:28] = 255
        support, alpha = erosion.erode_dut_mask(mask, 3, feather_px=0)
        expected = np.zeros_like(mask)
        expected[5:19, 7:25] = 255
        np.testing.assert_array_equal(support, expected)
        self.assertEqual(alpha.dtype, np.float32)

    def test_euclidean_distance_at_diagonal_boundary(self):
        mask = np.full((40, 40), 255, np.uint8)
        mask[:20, :20] = 0
        support, _ = erosion.erode_dut_mask(mask, 2.5, feather_px=0)
        # Nearest background for (21, 21) is (19, 19), distance sqrt(8).
        self.assertEqual(support[21, 21], 255)
        self.assertEqual(support[20, 21], 0)  # sqrt(5)

    def test_canvas_edges_are_background_not_infinite_foreground(self):
        mask = np.full((20, 30), 255, np.uint8)
        support, _ = erosion.erode_dut_mask(mask, 2, feather_px=0)
        expected = np.zeros_like(mask)
        expected[2:-2, 2:-2] = 255
        np.testing.assert_array_equal(support, expected)

    def test_holes_are_not_enlarged_or_fabricated_by_default(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[2:38, 2:38] = 255
        mask[19:21, 19:21] = 0
        filled, _ = erosion.erode_dut_mask(mask, 3, feather_px=0)
        kept, _ = erosion.erode_dut_mask(mask, 3, feather_px=0, fill_holes=False)
        self.assertEqual(filled[19, 19], 0)
        self.assertEqual(filled[18, 19], 255)
        self.assertEqual(kept[18, 19], 0)
        self.assertFalse(np.any((filled > 0) & (mask == 0)))

    def test_feathering_preserves_interior_pixels_and_coordinates(self):
        mask = np.zeros((50, 60), np.uint8)
        mask[2:48, 3:57] = 255
        image = np.random.default_rng(7).integers(1, 256, (50, 60, 3), dtype=np.uint8)
        support, alpha = erosion.erode_dut_mask(mask, 4, feather_px=3)
        result = erosion.apply_alpha(image, alpha)
        np.testing.assert_array_equal(result[alpha == 1], image[alpha == 1])
        self.assertFalse(np.any(result[support == 0]))
        self.assertTrue(np.any((alpha > 0) & (alpha < 1)))
        self.assertEqual(result.shape, image.shape)

    def test_smoothing_stays_inside_available_input_support(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[10:50, 10:50] = 255
        mask[10:14, 30:32] = 0
        mask[7:10, 20:22] = 255
        support, _ = erosion.erode_dut_mask(mask, 2, smooth_sigma=2)
        self.assertFalse(np.any((support > 0) & (mask == 0)))

    def test_invalid_or_excessive_erosion_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            erosion.erode_dut_mask(np.zeros((10, 10), np.uint8), 1)
        with self.assertRaisesRegex(ValueError, "entire DUT"):
            erosion.erode_dut_mask(np.ones((10, 10), np.uint8), 100)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                erosion.erode_dut_mask(np.ones((10, 10), np.uint8), value)

    def make_dataset(self, root, mask_folder="mask"):
        image_path = root / "train/good" / "样例 (DUT).png"
        mask_path = root / mask_folder / "样例 (DUT)_mask.png"
        image = np.full((50, 60, 3), (80, 100, 120), np.uint8)
        mask = np.zeros((50, 60), np.uint8)
        mask[5:45, 5:55] = 255
        erosion.write_png(image_path, image)
        erosion.write_png(mask_path, mask)
        return image_path, mask_path

    def test_cli_unicode_pairing_ratio_png_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input", Path(temp) / "output"
            image_path, mask_path = self.make_dataset(source)
            before = (image_path.read_bytes(), mask_path.read_bytes())
            args = ["--input-root", str(source), "--output-root", str(output),
                    "--shrink-ratio", "0.1", "--feather-px", "0"]
            self.assertEqual(erosion.main(args + ["--dry-run"]), 0)
            self.assertFalse(output.exists())
            self.assertEqual(erosion.main(args), 0)
            mask = erosion.read_image(output / "masks" / mask_path.name, cv2.IMREAD_GRAYSCALE)
            self.assertEqual(np.count_nonzero(mask), 32 * 42)  # 4px on every side
            result = erosion.read_image(output / "train/good" / image_path.name, cv2.IMREAD_COLOR)
            self.assertEqual(result.shape, (50, 60, 3))
            np.testing.assert_array_equal(result[10, 10], [80, 100, 120])
            self.assertTrue((output / "processing_report.csv").is_file())
            self.assertTrue((output / "settings.json").is_file())
            self.assertTrue((output / "previews" / image_path.name).is_file())
            self.assertEqual(before, (image_path.read_bytes(), mask_path.read_bytes()))
            self.assertEqual(erosion.main(args), 1)

    def test_missing_mask_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input", Path(temp) / "output"
            _, mask_path = self.make_dataset(source, "masks")
            mask_path.unlink()
            self.assertEqual(erosion.main(["--input-root", str(source),
                                          "--output-root", str(output)]), 1)
            self.assertFalse(output.exists())

    def test_bad_dimensions_are_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input", Path(temp) / "output"
            _, mask_path = self.make_dataset(source)
            erosion.write_png(mask_path, np.ones((20, 20), np.uint8) * 255)
            self.assertEqual(erosion.main(["--input-root", str(source),
                                          "--output-root", str(output)]), 1)
            self.assertFalse((output / "train/good").exists())
            self.assertIn("size mismatch", (output / "processing_report.csv").read_text())

    def test_output_cannot_overlap_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input"
            self.make_dataset(source)
            for output in (source, source / "new", source.parent):
                with self.assertRaisesRegex(ValueError, "separate"):
                    erosion.validate_output(source, source / "mask", output.resolve())


if __name__ == "__main__":
    unittest.main()
