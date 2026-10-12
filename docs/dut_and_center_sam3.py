#!/usr/bin/env python3
"""用本地 SAM3 提取 DUT，再按 imprint 脚本的流程裁剪、等比缩放、居中。

从 RELP 根目录运行（PowerShell）：
    .venv/Scripts/python.exe -X utf8 -B docs/dut_and_center_sam3.py `
        --input_dir ../Dinomaly/data/20261011_T0 `
        --output_dir Result/center_sam3 --save_masks --save_debug --limit 4

算法：每张 RGB 图只编码一次；用简短 SAM3 文本提示找实例，筛除空 mask、
异常面积和非有限分数，再选择/合并。--prompt 可重复，按顺序尝试，第一次
得到合格 DUT mask 即停止。它不是 Grounding DINO 的点号分隔 prompt 语法。
默认 union 假设一张图只有一个 DUT，适合保护壳被拆成多个面板的情况；多 DUT
或只需主面板时用 --mask_selection best/largest，避免跨物体合并。

文本正常返回但没有合格 mask 时，用中心前景点和 SAM3 的多候选 mask 回退。
模型/API 异常会把该图标记为失败，不将程序错误当作文本漏检。此步骤假设图像
中心落在 DUT 上；面积/预测 IoU 筛选并不能证明它就是 DUT，回退也可能只
覆盖一个面板。报告标记需要复核的结果；不允许此假设时用 --fallback skip。
可通过 --fallback_point X Y 调整点位置，X/Y 是相对宽/高的 [0,1] 比例。

颜色：OpenCV 输入/裁剪/保存均为 BGR，只在送入 SAM3 时转一次 RGB。
mask 为原图尺寸 bool；几何处理统一转 uint8 0/255。画布 mask 用最近邻
缩放，与输出图像对齐。默认不腐蚀，以免删除边缘缺陷；--erosion 的单位为
原图像素，--padding 也是裁剪前的原图像素。original 指 padded crop 长边，
与 dut_and_center_imprint.py 一致，不表示保留原始相机画幅。

使用 utils/wrappers/sam3_wrapper.py 的本地权重/BPE，要求 CUDA + BF16；没有
Grounding DINO、SAM2 或 Detectron2 依赖，不下载模型。每张图 finally 清理
特征。默认保留输入文件名/扩展名；--output_format png 可避免再次 JPEG 压缩。
--save_masks 保存画布尺寸的 PNG mask，
--save_debug 另存原图 mask 和预览。processing_report.json 记录回退、候选、
裁剪坐标和缩放信息。输出目录不清空；比较不同参数时请使用不同输出目录。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


RELP_ROOT = Path(__file__).resolve().parents[1]
if str(RELP_ROOT) not in sys.path:
    sys.path.insert(0, str(RELP_ROOT))

from utils.wrappers.sam3_wrapper import DEFAULT_BPE_PATH, DEFAULT_CHECKPOINT_PATH, init_sam3


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def ratio(value):
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0,1]")
    return number


def canvas_size(value):
    return "original" if str(value).lower() == "original" else positive_int(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, default=RELP_ROOT.parent / "Dinomaly/data/20261011_T0")
    parser.add_argument("--output_dir", type=Path, default=RELP_ROOT / "Result/center_sam3")
    parser.add_argument("--target_size", type=canvas_size, default=1280, help="Positive square side or 'original'.")
    parser.add_argument("--padding", type=nonnegative_int, default=10, help="Padding before scaling, in source pixels.")
    parser.add_argument("--erosion", type=nonnegative_int, nargs="?", const=10, default=0,
                        help="Inward erosion in source pixels; flag alone means 10; default 0.")
    parser.add_argument("--prompt", action="append", help="Short SAM3 target phrase; repeat for ordered retries. Default: ipad textile case.")
    parser.add_argument("--confidence_threshold", type=ratio, default=0.5, help="SAM3 text instance confidence, not mask pixel threshold.")
    parser.add_argument("--mask_selection", choices=("union", "best", "largest"), default="union")
    parser.add_argument("--min_area_ratio", type=ratio, default=0.01, help="Minimum mask area / source image area.")
    parser.add_argument("--max_area_ratio", type=ratio, default=0.95, help="Maximum area ratio, including the final union.")
    parser.add_argument("--fallback", choices=("center", "skip"), default="center")
    parser.add_argument("--fallback_point", type=ratio, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"))
    parser.add_argument("--min_iou_score", type=ratio, default=0.5, help="Minimum predicted mask IoU for point fallback; no ground-truth comparison.")
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--bpe_path", type=Path, default=DEFAULT_BPE_PATH)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1; CPU/MPS unsupported.")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--output_format", choices=("original", "png"), default="original",
                        help="Keep input extension, or write lossless PNG for anomaly detection.")
    parser.add_argument("--limit", type=positive_int, help="Process the first N matching images after sorting.")
    parser.add_argument("--include", nargs="+", default=["*"], help="Filename glob patterns; nonrecursive. E.g. DSC01307.JPG DSC01373.JPG.")
    args = parser.parse_args(argv)
    args.prompt = args.prompt or ["ipad textile case"]
    if any(not prompt.strip() for prompt in args.prompt):
        parser.error("--prompt must not be empty")
    if args.min_area_ratio > args.max_area_ratio:
        parser.error("--min_area_ratio must not exceed --max_area_ratio")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Input and output directories must differ to protect source images")
    return args


def read_image(path):
    # imdecode/fromfile 支持 Windows 中文路径；仍按 OpenCV BGR 返回。
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def select_mask(masks, scores, image_shape, mode="best", min_area_ratio=0.01,
                max_area_ratio=0.95, min_score=0.0):
    """同步筛选 mask/score；分数含义由调用方在报告中注明。

    严格面积上限不会像旧脚本那样重新接纳全部背景候选；无合格结果时明确
    返回 None，交给回退/失败路径。通过 CLI 可调整接近满画面的 DUT 上限。
    """
    masks, scores = np.asarray(masks), np.asarray(scores)
    h, w = image_shape[:2]
    if masks.ndim != 3 or masks.shape[1:] != (h, w) or scores.shape != (len(masks),):
        raise ValueError("Expected masks NxHxW and scores N at original image resolution")
    if mode not in {"best", "largest", "union"}:
        raise ValueError(f"Unknown mask selection: {mode}")
    areas = np.count_nonzero(masks, axis=(1, 2))
    ratios = areas / (h * w)
    valid = np.flatnonzero((areas > 0) & np.isfinite(scores) & (scores >= min_score)
                           & (ratios >= min_area_ratio) & (ratios <= max_area_ratio))
    metadata = {"candidate_count": len(masks), "valid_indices": valid.tolist(),
                "candidate_scores": [float(s) if np.isfinite(s) else None for s in scores],
                "candidate_area_ratios": ratios.tolist(), "selection": mode}
    if not len(valid):
        metadata["rejection"] = "No mask passed the score and area limits"
        return None, metadata
    if mode == "union":
        selected = valid
        mask = np.any(masks[valid] > 0, axis=0)
    else:
        rank = scores[valid] if mode == "best" else areas[valid]
        selected = valid[[int(np.argmax(rank))]]
        mask = masks[selected[0]] > 0
    area_ratio = float(np.count_nonzero(mask) / (h * w))
    metadata.update(selected_indices=selected.tolist(), area_ratio=area_ratio)
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        metadata["rejection"] = "Combined mask exceeded the area limits"
        return None, metadata
    return mask, metadata


def segment_dut(predictor, image_rgb, prompts, selection="union", fallback="center",
                fallback_point=(0.5, 0.5), min_area_ratio=0.01, max_area_ratio=0.95,
                min_iou_score=0.5):
    """一张图只编码一次。返回原图 bool mask 或 None，以及可复核的决策记录。

    特征的生命周期由外层每图 finally 中 reset_image 管理。
    """
    predictor.set_image(image_rgb)
    attempts = []
    for prompt in prompts:
        result = predictor.predict_text(prompt)
        mask, decision = select_mask(result["masks"], result["scores"], image_rgb.shape,
                                     selection, min_area_ratio, max_area_ratio)
        decision.update(prompt=prompt, score_type="text_confidence")
        attempts.append(decision)
        if mask is not None:
            return mask, {"method": "text", "needs_review": False, "text_attempts": attempts}
    if fallback == "skip":
        return None, {"method": "no_mask", "needs_review": True, "text_attempts": attempts}
    h, w = image_rgb.shape[:2]
    point = np.array([[fallback_point[0] * (w - 1), fallback_point[1] * (h - 1)]], dtype=np.float32)
    # normalize_coords=True 的接口接受原图 XY 像素，并非已经归一化的点。
    masks, scores, low_res_logits = predictor.predict(
        point_coords=point, point_labels=np.array([1], dtype=np.int32),
        multimask_output=True, normalize_coords=True,
    )
    # 回退始终按预测 IoU 选候选，不能用文本选择模式或最大面积替代质量分数。
    mask, decision = select_mask(masks, scores, image_rgb.shape, "best", min_area_ratio,
                                 max_area_ratio, min_iou_score)
    del low_res_logits, masks
    decision.update(point_xy=point[0].tolist(), score_type="predicted_mask_iou")
    return mask, {"method": "point_fallback" if mask is not None else "no_mask",
                  "needs_review": True, "text_attempts": attempts, "fallback": decision}


def prepare_canvas(image_bgr, mask, padding=10, target_size=1280, erosion=0):
    """沿用 imprint 的裁剪/居中规则，附带记录原图→画布的几何信息。"""
    mask = np.asarray(mask)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or mask.shape != image_bgr.shape[:2]:
        raise ValueError("Expected BGR HxWx3 image and matching HxW mask")
    if padding < 0 or erosion < 0:
        raise ValueError("padding and erosion must be nonnegative")
    mask = (mask > 0).astype(np.uint8) * 255
    if erosion:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
        mask = cv2.erode(mask, kernel)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, None, None
    h, w = mask.shape
    x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
    x2, y2 = min(w, int(xs.max()) + padding + 1), min(h, int(ys.max()) + padding + 1)
    # 先 Tight Crop 再等比缩放，不把原始相机画幅拉伸成方形。
    cropped = image_bgr[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
    cropped[cropped_mask == 0] = 0
    crop_h, crop_w = cropped_mask.shape
    side = max(crop_h, crop_w) if str(target_size).lower() == "original" else int(target_size)
    if side <= 0:
        raise ValueError("target_size must be positive or 'original'")
    scale = min(side / crop_w, side / crop_h)
    new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized[resized_mask == 0] = 0  # 避免图像线性插值在二值 mask 外留下彩色光晕。
    offset_x, offset_y = (side - new_w) // 2, (side - new_h) // 2
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas_mask = np.zeros((side, side), dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    canvas_mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_mask
    geometry = {"crop_xyxy": [x1, y1, x2, y2], "resized_wh": [new_w, new_h],
                "offset_xy": [offset_x, offset_y], "canvas_side": side,
                "scale_xy": [new_w / crop_w, new_h / crop_h], "erosion_source_pixels": erosion}
    return canvas, canvas_mask, geometry


def crop_and_center_image(image_bgr, mask, padding=10, target_size=1280, erosion=0):
    """与原 imprint 脚本一致的二元返回接口，便于单独复用。"""
    image, mask, _ = prepare_canvas(image_bgr, mask, padding, target_size, erosion)
    return image, mask


def debug_outputs(output_dir, stem, image_bgr, mask):
    overlay = image_bgr.copy()
    overlay[mask] = (0.6 * image_bgr[mask] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    h, w = mask.shape
    scale = min(1.0, 1400 / max(h, w))
    preview = cv2.resize(overlay, (max(1, round(w * scale)), max(1, round(h * scale))))
    return [(output_dir / "source_masks" / f"{stem}_mask.png", mask.astype(np.uint8) * 255),
            (output_dir / "previews" / f"{stem}_overlay.jpg", preview)]


def write_outputs(output_dir, products, protected_sources=()):
    """先编码全部产物，再替换；失败时恢复旧文件，避免图片/mask 配对不完整。"""
    root = output_dir.resolve()
    protected = {Path(path).resolve() for path in protected_sources}
    stage = Path(tempfile.mkdtemp(prefix=".sam3-stage-", dir=root)).resolve()
    remove_stage = True
    try:
        # 清理临时目录前确认其位于用户指定的输出目录内。
        if stage.parent != root:
            raise ValueError("Staging directory escaped output directory")
        staged, backups, installed = [], [], []
        for index, (destination, array) in enumerate(products):
            destination = destination.resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Output file escaped output directory")
            if destination in protected:
                raise ValueError(f"Output would overwrite a source image: {destination}")
            temporary_file = stage / f"{index}{destination.suffix}"
            write_image(temporary_file, array)
            staged.append((temporary_file, destination))
        try:
            for index, (temporary_file, destination) in enumerate(staged):
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if not destination.is_file():
                        raise IsADirectoryError(destination)
                    backup = stage / f"backup-{index}"
                    destination.replace(backup)
                    backups.append((backup, destination))
                temporary_file.replace(destination)
                installed.append(destination)
        except Exception as commit_error:
            rollback_errors = []
            for destination in reversed(installed):
                try:
                    destination.unlink()
                except OSError as exc:
                    rollback_errors.append(str(exc))
            for backup, destination in reversed(backups):
                try:
                    backup.replace(destination)
                except OSError as exc:
                    rollback_errors.append(str(exc))
            if rollback_errors:
                # 无法恢复时保留备份，不能由临时目录清理继续删掉旧产物。
                remove_stage = False
                raise OSError(f"Output rollback failed; backups retained at {stage}: {rollback_errors}") from commit_error
            raise
    finally:
        if remove_stage and stage.parent == root:
            shutil.rmtree(stage)


def write_report(path, report):
    """完整写入临时 JSON 后替换，避免失败时截断旧报告或跟随文件符号链接。"""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".sam3-report-",
                                     suffix=".json", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        except Exception:
            stream.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    paths = sorted(p for p in args.input_dir.iterdir() if p.is_file()
                   and p.suffix.lower() in IMAGE_EXTENSIONS
                   and any(fnmatch.fnmatchcase(p.name.casefold(), pattern.casefold()) for pattern in args.include))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        print(f"No matching images in {args.input_dir}", file=sys.stderr)
        return 1
    names = [p.name if args.output_format == "original" else p.stem + ".png" for p in paths]
    stems = [p.stem.casefold() for p in paths]
    if len({name.casefold() for name in names}) != len(names) or (
        (args.save_masks or args.save_debug) and len(set(stems)) != len(stems)
    ):
        print("Output filenames would collide; use --include to select one source per stem", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SAM3 DUT extraction: {len(paths)} images; selection={args.mask_selection}; fallback={args.fallback}")
    print(f"Canvas={args.target_size}; padding={args.padding}; erosion={args.erosion} source pixels")
    predictor = init_sam3(args.checkpoint_path, bpe_path=args.bpe_path, device=args.device,
                          confidence_threshold=args.confidence_threshold)
    records = []
    started = time.perf_counter()
    try:
        for index, path in enumerate(paths, 1):
            record = {"input": str(path.resolve()), "filename": path.name, "status": "failed"}
            tick = time.perf_counter()
            try:
                image_bgr = read_image(path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mask, decision = segment_dut(
                    predictor, image_rgb, args.prompt, args.mask_selection, args.fallback,
                    args.fallback_point, args.min_area_ratio, args.max_area_ratio, args.min_iou_score,
                )
                record.update(decision, source_shape=list(image_bgr.shape))
                if mask is None:
                    raise ValueError("No DUT mask passed the configured filters")
                canvas, canvas_mask, geometry = prepare_canvas(
                    image_bgr, mask, args.padding, args.target_size, args.erosion,
                )
                if canvas is None:
                    raise ValueError("DUT mask became empty after erosion")
                outpath = args.output_dir / (path.name if args.output_format == "original" else path.stem + ".png")
                products = [(outpath, canvas)]
                if args.save_masks:
                    products.append((args.output_dir / "masks" / f"{path.stem}_mask.png", canvas_mask))
                if args.save_debug:
                    products.extend(debug_outputs(args.output_dir, path.stem, image_bgr, mask))
                write_outputs(args.output_dir, products, protected_sources=paths)
                record.update(status="success", output=str(outpath.resolve()), geometry=geometry)
                print(f"[{index}/{len(paths)}] {path.name}: {decision['method']} -> {canvas.shape[1]}x{canvas.shape[0]}")
            except Exception as exc:
                record["error"] = str(exc)
                print(f"[{index}/{len(paths)}] {path.name}: FAILED: {exc}", file=sys.stderr)
            finally:
                # 无论读取、模型、裁剪或写盘失败，都释放上一张图的 GPU 特征。
                try:
                    predictor.reset_image()
                except Exception as exc:
                    record.update(status="failed", cleanup_error=str(exc))
                    raise  # CUDA 清理失败时停止批次，但仍将当前记录写入报告。
                finally:
                    record["seconds"] = round(time.perf_counter() - tick, 3)
                    records.append(record)
    finally:
        succeeded = sum(r["status"] == "success" for r in records)
        fallback_count = sum(r.get("method") == "point_fallback" for r in records)
        report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "processed": len(records), "succeeded": succeeded, "failed": len(records) - succeeded,
                  "point_fallback_count": fallback_count, "seconds": round(time.perf_counter() - started, 3),
                  "images": records}
        report_path = args.output_dir / "processing_report.json"
        write_report(report_path, report)
    print(f"Done: {succeeded}/{len(records)} succeeded; {fallback_count} point fallbacks. Report: {report_path}")
    return 0 if succeeded == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
