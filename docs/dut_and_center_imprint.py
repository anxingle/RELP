#!/usr/bin/env python3
"""
RELP Imprint Dataset Crop & Auto-Centering Script
Using Grounding DINO + SAM 2.1
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Robustly resolve RELP root directory using pathlib
script_dir = Path(__file__).resolve().parent

if (script_dir / "utils" / "wrappers").exists():
    relp_root = script_dir
elif (script_dir.parent / "utils" / "wrappers").exists():
    relp_root = script_dir.parent
else:
    relp_root = (script_dir / ".." / "RELP").resolve()

if str(relp_root) not in sys.path:
    sys.path.insert(0, str(relp_root))

from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
from utils.wrappers.sam2_wrapper import init_sam2, get_sam2_predictor
import utils.crop_ops as crop_ops

def parse_args():
    parser = argparse.ArgumentParser(description="Crop and auto-center images using Grounding DINO + SAM 2")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/an/workspace/imprint_datasets/train_imprint",
        help="Path to input images directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/an/workspace/imprint_datasets/center_train_imprint",
        help="Path to save cropped and centered images"
    )
    parser.add_argument(
        "--target_size",
        type=str,
        default="1280",
        help="Canvas target size: integer (e.g. 1280, 2048) or 'original' to keep max dimension"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding in pixels around tight mask bounds before canvas centering (default: 10)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="case . protective case . cover . object",
        help="Grounding DINO text prompt for detecting the product"
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.20,
        help="Grounding DINO box detection threshold (default: 0.20)"
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Also save corresponding binary masks to a 'masks' subdirectory"
    )
    parser.add_argument(
        "--erosion",
        type=int,
        nargs="?",
        const=10,
        default=0,
        help="Erode (shrink inward) the DUT mask by N pixels to eliminate boundary burrs (default: 10 if specified without value, 0 if omitted)"
    )
    return parser.parse_args()

def crop_and_center_image(image_bgr, mask, padding=10, target_size=1280, erosion=0):
    """
    Tightly crops the segmented object, removes background, and centers it on a black canvas.
    If erosion > 0, shrinks the DUT mask inward by N pixels using an elliptical kernel to eliminate edge burrs.
    """
    if erosion > 0:
        kernel_size = 2 * erosion + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.erode(mask, kernel)

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None

    h_img, w_img = image_bgr.shape[:2]
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Apply padding
    y_min_padded = max(0, y_min - padding)
    y_max_padded = min(h_img, y_max + padding + 1)
    x_min_padded = max(0, x_min - padding)
    x_max_padded = min(w_img, x_max + padding + 1)

    # Clean background (black-out non-object pixels)
    clean_image = image_bgr.copy()
    clean_image[mask == 0] = 0

    cropped_image = clean_image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    cropped_mask = mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

    h_crop, w_crop = cropped_image.shape[:2]
    if h_crop == 0 or w_crop == 0:
        return None, None

    # Resolve target canvas size
    if isinstance(target_size, str) and target_size.lower() == 'original':
        t_size = max(h_crop, w_crop)
    else:
        try:
            t_size = int(target_size)
        except (ValueError, TypeError):
            t_size = 1280

    # Scale calculation (preserving aspect ratio)
    scale = min(t_size / w_crop, t_size / h_crop)
    new_w = max(1, int(w_crop * scale))
    new_h = max(1, int(h_crop * scale))

    # Resize cropped object
    resized_img = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create black canvas and center the object
    canvas_img = np.zeros((t_size, t_size, 3), dtype=np.uint8)
    canvas_mask = np.zeros((t_size, t_size), dtype=np.uint8)

    x_offset = (t_size - new_w) // 2
    y_offset = (t_size - new_h) // 2

    canvas_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    canvas_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_mask

    return canvas_img, canvas_mask

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"❌ No valid images found in {input_dir}")
        return

    print("=" * 70)
    print("🚀 RELP Imprint Dataset Crop & Auto-Centering Pipeline")
    print(f"📂 Input directory:  {input_dir} ({len(image_paths)} images)")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Target size:      {args.target_size}x{args.target_size} square canvas")
    print(f"🔍 DINO Prompt:      '{args.prompt}' (threshold: {args.box_threshold})")
    if args.erosion > 0:
        print(f"✂️ DUT Erosion:      {args.erosion} pixels inward (Smooth edges / Anti-burr)")
    else:
        print(f"✂️ DUT Erosion:      Disabled (0 pixels)")
    print("=" * 70)

    # 1. Load Grounding DINO
    gd_weights_dir = relp_root / "weights" / "Grounding Dino"
    print(f"\n[1/3] Loading Grounding DINO model from {gd_weights_dir}...")
    gd_model = load_grounding_dino_model(model_id=str(gd_weights_dir), box_threshold=args.box_threshold)

    # 2. Load SAM 2.1
    sam_cfg = (relp_root / "weights" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml").resolve()
    sam_ckpt = (relp_root / "weights" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt").resolve()
    print(f"[2/3] Loading SAM 2.1 model from {sam_ckpt}...")
    sam_model = init_sam2(str(sam_cfg), str(sam_ckpt), device="cpu")
    sam_predictor = get_sam2_predictor(sam_model)

    if sam_predictor is None:
        print("❌ Error: Failed to initialize SAM 2 predictor.")
        return

    # 3. Process Images
    print(f"\n[3/3] Processing {len(image_paths)} images...")
    t_start = time.time()
    success_count = 0
    fallback_count = 0
    fail_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        filename = img_path.name
        t_img_start = time.time()

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[{idx}/{len(image_paths)}] ⚠️ Error reading {filename}, skipping.")
            fail_count += 1
            continue

        h_orig, w_orig = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Step A: Grounding DINO Detection
        detections = gd_model.predict(image_rgb, args.prompt)
        if not detections:
            detections = gd_model.predict(image_rgb, "object")

        if not detections:
            print(f"[{idx}/{len(image_paths)}] ⚠️ No detection for {filename}, using center fallback.")
            pad_x = int(w_orig * 0.05)
            pad_y = int(h_orig * 0.05)
            box = np.array([pad_x, pad_y, w_orig - pad_x, h_orig - pad_y])
            fallback_count += 1
        else:
            # 防呆过滤：剔除覆盖全图 (>85% 面积) 的台面背景假框，防止 SAM2 发生前景/背景内外反转
            valid_dets = [d for d in detections if ((d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])) / (w_orig * h_orig) < 0.85]
            if not valid_dets:
                valid_dets = detections
            valid_dets = sorted(valid_dets, key=lambda x: x["score"], reverse=True)
            box = np.array(valid_dets[0]["bbox"]).astype(int)

        # Step B: SAM 2 Segmentation
        sam_predictor.set_image(image_rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False
        )
        mask = (masks[0] > 0.0).astype(np.uint8) * 255

        # Step C: Tight Crop & Auto-Centering (with optional erosion)
        canvas_img, canvas_mask = crop_and_center_image(
            image_bgr,
            mask,
            padding=args.padding,
            target_size=args.target_size,
            erosion=args.erosion
        )

        if canvas_img is None:
            print(f"[{idx}/{len(image_paths)}] ❌ Failed to crop {filename}.")
            fail_count += 1
            continue

        # Save output image
        out_img_path = output_dir / filename
        cv2.imwrite(str(out_img_path), canvas_img)

        if args.save_masks:
            mask_out_path = mask_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), canvas_mask)

        dt_img = time.time() - t_img_start
        success_count += 1

        elapsed = time.time() - t_start
        fps = success_count / elapsed if elapsed > 0 else 0
        eta = (len(image_paths) - idx) / fps if fps > 0 else 0

        print(f"[{idx:>3}/{len(image_paths)}] ✅ {filename} ({w_orig}x{h_orig} -> {canvas_img.shape[1]}x{canvas_img.shape[0]}) in {dt_img:.2f}s | Speed: {fps:.2f} img/s | ETA: {eta:.0f}s")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("🎉 Processing Complete!")
    print(f"⏱️ Total time elapsed: {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"✅ Successfully processed: {success_count}/{len(image_paths)}")
    if fallback_count > 0:
        print(f"⚠️ Fallback detections:    {fallback_count}")
    if fail_count > 0:
        print(f"❌ Failed images:         {fail_count}")
    print(f"📁 Output saved to:       {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
