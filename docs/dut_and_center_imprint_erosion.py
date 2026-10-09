#!/usr/bin/env python3
"""Shrink already-centered DUT images using their saved binary segmentation masks.

Requires numpy and opencv-python(-headless). No DINO/SAM, GPU, resize, or
re-centering is needed. Coordinates and interior texture are preserved.

Examples (from the repository root):
    python docs/dut_and_center_imprint_erosion.py --dry-run
    python docs/dut_and_center_imprint_erosion.py --shrink-px 40 --limit 3
    python docs/dut_and_center_imprint_erosion.py --shrink-ratio 0.01

See dut_and_center_imprint_erosion.md for parameter units and caveats.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "center_train_imprint"


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def read_image(path, flags):
    """imdecode/fromfile also supports Unicode paths on Windows."""
    array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if array is None:
        raise ValueError(f"Cannot decode: {path}")
    return array


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError(f"Cannot encode: {path}")
    encoded.tofile(path)


def main_contour(mask):
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Input mask is empty")
    return binary, max(contours, key=cv2.contourArea)


def erode_dut_mask(mask, shrink_px, feather_px=2.0, smooth_sigma=0.0,
                   fill_holes=True):
    """Return uint8 support (0/255) and float32 alpha at the input resolution.

    Distance is measured to background pixel centers, using exact Euclidean
    distance. Retain d > shrink_px; the optional transition occupies the next
    feather_px pixels INWARD. All dimensions are source-mask pixels.

    Hole filling only repairs the geometry used to measure outer-edge distance.
    Final support is always a subset of the ORIGINAL mask: already removed
    image pixels cannot be recovered from centered JPGs.
    """
    for value in (shrink_px, feather_px, smooth_sigma):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Distances and sigma must be finite and nonnegative")
    binary, contour = main_contour(mask)
    geometry = np.zeros_like(binary)
    cv2.drawContours(geometry, [contour], -1, 1, cv2.FILLED)
    if not fill_holes:
        geometry &= binary

    # Explicit background is essential for objects touching the canvas boundary.
    pad = max(1, int(math.ceil(4 * smooth_sigma)))
    padded = cv2.copyMakeBorder(geometry, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    del geometry
    if smooth_sigma > 0:
        blurred = cv2.GaussianBlur(padded.astype(np.float32), (0, 0), smooth_sigma)
        padded = (blurred >= 0.5).astype(np.uint8)
        del blurred
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = distance[pad:-pad, pad:-pad].copy()
    del distance, padded
    alpha -= shrink_px
    if feather_px > 0:
        alpha /= feather_px
        np.clip(alpha, 0.0, 1.0, out=alpha)
        # Linear alpha is sufficient for a narrow antialiasing band.
    else:
        np.greater(alpha, 0, out=alpha)
    alpha *= binary
    support = (alpha > 0).astype(np.uint8) * 255
    if not np.any(support):
        raise ValueError("Erosion removed the entire DUT; reduce shrink distance/smoothing")
    return support, alpha


def apply_alpha(image, alpha):
    if image.shape[:2] != alpha.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image and alpha dimensions do not match")
    result = np.empty_like(image)
    # Process channels separately to bound peak memory for 8K inputs.
    for channel in range(3):
        result[:, :, channel] = np.rint(image[:, :, channel] * alpha).astype(np.uint8)
    return result


def plan_inputs(input_root, mask_dir, pattern="*", limit=None):
    image_dir = input_root / "train" / "good"
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if mask_dir is None:
        candidates = [input_root / name for name in ("masks", "mask")
                      if (input_root / name).is_dir()]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one masks/ or mask/ directory; use --mask-dir")
        mask_dir = candidates[0]
    if not mask_dir.is_dir():
        raise ValueError(f"Mask directory does not exist: {mask_dir}")
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    and p.match(pattern))
    if not images:
        raise ValueError(f"No images match {pattern!r} in {image_dir}")
    if limit is not None:
        images = images[:limit]
    index = {}
    for path in mask_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.casefold(), []).append(path)
    jobs, seen = [], set()
    for image in images:
        stem = image.stem.casefold()
        if stem in seen:
            raise ValueError(f"Duplicate image stem would overwrite a PNG: {image.stem}")
        seen.add(stem)
        candidates = index.get(stem + "_mask", []) + index.get(stem, [])
        if len(candidates) != 1:
            raise ValueError(f"Expected one matching mask, found {len(candidates)}: {image.name}")
        jobs.append((image, candidates[0]))
    return jobs, mask_dir


def validate_output(input_root, mask_dir, output_root):
    for source in (input_root.resolve(), mask_dir.resolve()):
        if output_root == source or output_root.is_relative_to(source) or source.is_relative_to(output_root):
            raise ValueError("Output must be separate from the input dataset and masks")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise ValueError(f"Output must be empty or nonexistent: {output_root}")


def write_preview(path, image, processed):
    h, w = image.shape[:2]
    size = (800, max(1, round(h * 800 / w)))
    panels = [cv2.resize(item, size, interpolation=cv2.INTER_AREA)
              for item in (image, processed)]
    preview = np.concatenate(panels, axis=1)
    preview = cv2.copyMakeBorder(preview, 40, 0, 0, 0, cv2.BORDER_CONSTANT)
    for x, label in ((10, "INPUT"), (810, "ERODED (same coordinates)")):
        cv2.putText(preview, label, (x, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
    write_png(path, preview)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mask-dir", type=Path, help="Auto-detect masks/ or mask/ by default")
    parser.add_argument("--output-root", type=Path, help="Default: INPUT_ROOT_erosion")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--shrink-px", "--erosion", type=nonnegative_float,
                       help="Inward distance in input-mask pixels (default: 40)")
    group.add_argument("--shrink-ratio", type=nonnegative_float,
                       help="Inward distance / shorter side of the main DUT bounding box")
    parser.add_argument("--feather-px", type=nonnegative_float, default=2.0,
                        help="Additional inward alpha transition in input pixels; 0 for hard edge")
    parser.add_argument("--smooth-sigma", type=nonnegative_float, default=0.0,
                        help="Optional Gaussian sigma for mask geometry, in input pixels")
    parser.add_argument("--keep-holes", action="store_true",
                        help="Also measure distance to holes (they will expand during erosion)")
    parser.add_argument("--pattern", default="*", help="Input filename glob, e.g. '*R5a-91.jpg'")
    parser.add_argument("--limit", type=int, help="Only process the first N matching images")
    parser.add_argument("--preview-count", type=nonnegative_int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check filename pairing/output paths only; do not decode or write files")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.shrink_ratio is not None and args.shrink_ratio >= 0.5:
        parser.error("--shrink-ratio must be less than 0.5")
    if args.shrink_px is None and args.shrink_ratio is None:
        args.shrink_px = 40.0
    return args


def main(argv=None):
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root.with_name(input_root.name + "_erosion")).resolve()
    try:
        jobs, mask_dir = plan_inputs(input_root, args.mask_dir, args.pattern, args.limit)
        validate_output(input_root, mask_dir, output_root)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(f"Input: {input_root}\nMasks: {mask_dir}\nOutput: {output_root}", flush=True)
    print(f"Matched {len(jobs)} images. Canvas dimensions and coordinates are preserved.", flush=True)
    if args.dry_run:
        for image_path, mask_path in jobs[:5]:
            print(f"  {image_path.name} -> {mask_path.name}")
        print("Dry run complete (file contents/dimensions not checked).")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(input_root=str(input_root), output_root=str(output_root),
                    mask_dir=str(mask_dir.resolve()), selected_images=len(jobs),
                    opencv_version=cv2.__version__, numpy_version=np.__version__)
    (output_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    columns = ["image", "mask", "output_image", "output_mask", "status", "error",
               "width", "height", "shrink_px", "input_pixels", "output_pixels", "removed_fraction"]
    failures = 0
    previews = 0
    with (output_root / "processing_report.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, (image_path, mask_path) in enumerate(jobs, 1):
            row = {"image": image_path.name, "mask": mask_path.name}
            image = mask = support = alpha = processed = None
            try:
                # Preserve stored raster orientation; masks use the same coordinates.
                image = read_image(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
                if image.shape[:2] != mask.shape:
                    raise ValueError(f"Image/mask size mismatch: {image.shape[:2]} vs {mask.shape}")
                binary, contour = main_contour(mask)
                _, _, bbox_w, bbox_h = cv2.boundingRect(contour)
                shrink = (args.shrink_ratio * min(bbox_w, bbox_h)
                          if args.shrink_ratio is not None else args.shrink_px)
                input_pixels = int(np.count_nonzero(binary))
                del binary, contour
                support, alpha = erode_dut_mask(mask, shrink, args.feather_px,
                                                args.smooth_sigma, not args.keep_holes)
                processed = apply_alpha(image, alpha)
                del alpha
                output_image = Path("train/good") / (image_path.stem + ".png")
                output_mask = Path("masks") / (image_path.stem + "_mask.png")
                write_png(output_root / output_image, processed)
                write_png(output_root / output_mask, support)
                if previews < args.preview_count:
                    write_preview(output_root / "previews" / (image_path.stem + ".png"), image, processed)
                    previews += 1
                output_pixels = int(np.count_nonzero(support))
                removed = 1.0 - output_pixels / input_pixels
                row.update(output_image=output_image.as_posix(), output_mask=output_mask.as_posix(),
                           status="ok", width=mask.shape[1], height=mask.shape[0],
                           shrink_px=shrink, input_pixels=input_pixels, output_pixels=output_pixels,
                           removed_fraction=removed)
                print(f"[{index}/{len(jobs)}] OK shrink={shrink:.2f}px removed={removed:.2%} {image_path.name}", flush=True)
            except (ValueError, OSError, cv2.error) as exc:
                failures += 1
                row.update(status="failed", error=str(exc))
                print(f"[{index}/{len(jobs)}] FAILED {image_path.name}: {exc}", flush=True)
            finally:
                # Drop large arrays before decoding the next 8K image.
                image = mask = support = alpha = processed = None
            writer.writerow(row)
            stream.flush()
    print(f"Finished: {len(jobs) - failures} succeeded, {failures} failed. Report: {output_root / 'processing_report.csv'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
