#!/usr/bin/env python3
"""
Dinomaly (ViTill / DINOv3) + AnyUp Inference & Upsampling Parameter Comparison Tool
Specialized for Imprint and micro-defect detection analysis.
"""

# Imprint 测试集批量推理（手动执行）
# -----------------------------------------------------------------------------
# 在 WSL 终端中执行下列命令；复制时去掉每行开头的 "# "。
# 使用已准备好的 Python 3.12 uv 环境；--no-sync 避免运行时自动同步依赖。
# 本例选择仓库已有的 imprint_train_2000_v2/final_model.pth，
# 若需评估其他训练版本，只替换 --model_path。
#
# cd /home/an/ssd_work/RELP
# uv run --no-sync python docs/dinomaly_anyup_inference.py \
#   --model_path "/home/an/ssd_work/RELP/weights/imprint_train_2000_v2/final_model.pth" \
#   --backbone_weights "/mnt/c/Users/anxin/ssd_work/Dinomaly/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
#   --anyup_weights "/home/an/ssd_work/RELP/weights/anyup_paper.pth" \
#   --image_dir "/mnt/c/Users/anxin/ssd_work/Dinomaly/data/center_test_imprint" \
#   --output_dir "/home/an/ssd_work/RELP/Result/Imprint_tests_anyup" \
#   --input_dim 448 \
#   --upsample_sizes 160 320 448 640 \
#   --compare_baseline \
#   --save_csv \
#   --threshold 0.15 \
#   --device cpu
#
# 路径说明：脚本从自身位置定位 RELP 根目录及 anyup-main，无需设置 PYTHONPATH。
# Dinomaly 源码依次从 DINOMALY_ROOT 环境变量、RELP/Dinomaly、同级 Dinomaly
# 和本机 /mnt/c/Users/anxin/ssd_work/Dinomaly 查找；搬迁后可设置 DINOMALY_ROOT。
# 该源码目录须包含 models/uad.py 和定制的 dinov3/hub/backbones.py。
# 三个权重路径均显式指定；默认 AnyUp 相对权重路径也以 RELP 根目录为基准。
# --device cpu 可在无 CUDA 的环境运行，但批量推理耗时较长；
# 若当前 PyTorch 环境的 torch.cuda.is_available() 为 True，可改为 --device cuda。
# AnyUp 按查询分块计算特征、相似度及注意力掩膜，减少峰值显存占用。
# 完整 query 和最终异常图仍随输出像素数增长，不能保证 1280 在任意 24 GiB
# 环境下都能运行。--q_chunk_size 默认 2048，显存紧张时可降低为 128。
# 某分辨率 OOM 时会跳过并记录状态；本次请求的同名旧结果会在处理前清除。
#
# 输入范围：--image_dir 只扫描目录第一层，支持 jpg/jpeg/png/bmp（忽略大小写），
# 不递归扫描子目录。检查时该目录含 116 张 JPG，且文件主名没有重复。
# 不要同时传 --image_path，因为单图参数优先，会使 --image_dir 不生效。
#
# 输出目录会自动创建。每张成功处理的图片（例如 sample.jpg）会输出：
#   Result/Imprint_tests_anyup/sample_comparison_grid_heatmaps.jpg
#     横向拼图：原图 | Bilinear 基线热图 | AnyUp 160 | 320 | 448 | 640。
#     这就是参考文件 *_comparison_grid_heatmaps.jpg 对应的输出类型，
#     每个面板显示为 448x448；AnyUp 标签表示实际计算热图时的目标分辨率。
#   Result/Imprint_tests_anyup/sample_comparison_grid_overlays.jpg
#     对应的原图与热图叠加效果拼图。
#   Result/Imprint_tests_anyup/AnyUp_448/sample_AnyUp_448_heatmap.jpg
#     单独的热图（保存尺寸与原图相同），同目录还有 *_overlay.jpg、*_mask.png。
#     Baseline、AnyUp_160、AnyUp_320、AnyUp_640 子目录也会保存各自的结果。
# 全部成功处理后，应有 116 张 *_comparison_grid_heatmaps.jpg。
# 输出以输入文件主名命名：同主名不同扩展名会冲突，重复运行会覆盖同名结果。
# 每张输入另生成 *_inference_status.csv，记录本次各方案的 success/oom/pending
# 状态；未请求的分辨率目录不受影响，不代表本次结果。pending 表示尚未完成，
# 例如进程中途退出；读取失败记为 read_error。OOM 的拼图面板标记为 skipped。
# --threshold 只影响二值 mask；热图使用各自的 Min-Max 归一化着色，
# 因此不同图片/参数结果中的相同颜色不代表相同的绝对异常分数。
# --save_csv 为每个成功方案额外保存供 Dino_threshold_analyzer_GUI.py 使用的：
#   AnyUp_448/sample_AnyUp_448_anomaly_distance_raw.csv
#   AnyUp_448/sample_AnyUp_448_original.png
# CSV 是平滑后、按热图相同方式插值到输入图片尺寸的原始分数（无表头）。
# 不做 Min-Max 归一化、阈值过滤或黑背景置零；配套 PNG 为本次输入图片。
# GUI 的 Load CSV File 会自动匹配该 PNG。纯黑背景需要在 GUI 中排除；
# GUI 不会自动套用本脚本渲染时 RGB <= 5 的背景排除规则。
# 未开启 --save_csv 时不生成这些文件；重新运行会清除本次请求方案的旧导出。
# | 文件后缀 | 含义 | 用途 |
# |---|---|---|
# | `_comparison_grid_heatmaps.jpg` | 原图、Baseline、AnyUp 160/320/448/640 的热图横向拼接 | 快速比较不同参数，与你提供的参考图对应 |
# | `_comparison_grid_overlays.jpg` | 上述各方案的原图与热图叠加对比 | 判断高异常区域是否对应实际压痕 |
# | `_heatmap.jpg` | 单个方案的彩色异常热图 | 查看异常分布；蓝色较低，红色较高 |
# | `_overlay.jpg` | 原图与热图半透明叠加 | 对照纹理、边缘和缺陷位置 |
# | `_mask.png` | 按 `--threshold` 分割后的黑白掩膜 | 白色为达到异常阈值的区域，黑色为未达到阈值或被排除的背景 |
# -----------------------------------------------------------------------------

import sys
import time
import gc
import argparse
import csv
from os import environ
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 本文件位于 RELP/docs/，依赖与默认权重路径应从 RELP 根目录解析。
project_root = Path(__file__).resolve().parent.parent
anyup_dir = project_root / "anyup-main"
dinomaly_candidates = [
    Path(environ["DINOMALY_ROOT"]).expanduser()
    if environ.get("DINOMALY_ROOT") else None,
    project_root / "Dinomaly",
    project_root.parent / "Dinomaly",
    Path("/mnt/c/Users/anxin/ssd_work/Dinomaly"),
]
dinomaly_dir = next((p.resolve() for p in dinomaly_candidates
                     if p is not None
                     and (p / "models" / "uad.py").is_file()
                     and (p / "dinov3" / "hub" / "backbones.py").is_file()), None)
if dinomaly_dir is None:
    raise ImportError(
        "Dinomaly 源码未找到：请设置 DINOMALY_ROOT 为包含 models/uad.py "
        "和定制 dinov3/hub/backbones.py 的 Dinomaly 仓库目录。"
    )

for p in [project_root, anyup_dir, dinomaly_dir]:
    # sys.path 需要字符串；其余内部路径统一保留为 Path。
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anyup.model import AnyUp
from anyup.layers.attention.attention_masking import window2d
from anyup.utils.img import create_coordinate
from dinov3.hub.backbones import load_dinov3_model
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Dinomaly + AnyUp inference and compare different upsampling parameters"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained Dinomaly checkpoint (.pth)"
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        default=None,
        help="Path to a single image for multi-parameter side-by-side comparison"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Path to directory of images to process in batch"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("Result/dinomaly_anyup_experiments"),
        help="Directory to save comparison grids, heatmaps, and overlays"
    )
    parser.add_argument(
        "--upsample_sizes",
        nargs="+",
        type=int,
        default=[160, 320, 448, 640],
        help="List of AnyUp target resolutions to compare (e.g. 160 320 448 640 896)"
    )
    parser.add_argument(
        "--q_chunk_size",
        type=int,
        default=2048,
        help="Number of AnyUp queries per chunk (default: 2048; try 128 to reduce peak memory)"
    )
    parser.add_argument(
        "--compare_baseline",
        action="store_true",
        default=True,
        help="Include standard Bilinear interpolation (without AnyUp) as baseline in comparison"
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Export raw score CSVs and paired PNGs for Dino_threshold_analyzer_GUI"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Anomaly score threshold for binary defect mask (default: 0.15)"
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=448,
        help="Model input resolution (default: 448)"
    )
    parser.add_argument(
        "--backbone_weights",
        type=Path,
        default=None,
        help="Path to pretrained DINOv3 backbone (dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth)"
    )
    parser.add_argument(
        "--anyup_weights",
        type=Path,
        default=Path("weights/anyup_paper.pth"),
        help="Path to pretrained AnyUp weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device (default: cpu, recommended for AnyUp/DINO stability on Mac)"
    )
    args = parser.parse_args()
    if args.q_chunk_size <= 0:
        parser.error("--q_chunk_size must be positive")
    return args

def load_dinomaly_model(model_path, backbone_weights_path, device):
    print(f"[*] Loading DINOv3 backbone from: {backbone_weights_path}")
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = load_dinov3_model(
        "dinov3_vitb16",
        layers_to_extract_from=target_layers,
        pretrained_weight_path=backbone_weights_path
    )

    embed_dim, num_heads = 768, 12
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=nn.LayerNorm, attn=LinearAttention2
        ) for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
        mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder
    )

    print(f"[*] Loading trained Dinomaly checkpoint: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict):
        for k in ['model_state', 'model', 'state_dict']:
            if k in state_dict and isinstance(state_dict[k], dict):
                print(f"[*] Successfully unpacked model weights from key: '{k}'")
                state_dict = state_dict[k]
                break
    load_res = model.load_state_dict(state_dict, strict=False)
    print(f"[*] Checkpoint loaded (missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)})")
    model.to(device)
    model.eval()
    return model

def load_anyup_model(weights_path, device):
    print(f"[*] Loading AnyUp model from: {weights_path}")
    model = AnyUp()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    pad = (kernel_size - 1) // 2
    return lambda t: F.conv2d(t, gaussian_kernel.to(t.device), padding=pad)

def compute_anomaly_map_baseline(en_list, de_list, target_size=(448, 448)):
    """Standard bilinear interpolation without AnyUp (Legacy / Baseline)"""
    total_a_map = None
    count = 0
    for i in range(len(de_list)):
        if i >= len(en_list): break
        a_map = 1 - F.cosine_similarity(en_list[i], de_list[i])
        a_map = a_map.unsqueeze(1)
        a_map = F.interpolate(a_map, size=target_size, mode="bilinear", align_corners=True)
        total_a_map = a_map if total_a_map is None else total_a_map + a_map
        count += 1
    return (total_a_map / count) if count > 0 else total_a_map

def release_compute_memory(device):
    """Release GPU memory cached by previous large tensor operations."""
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _is_oom_error(exc):
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()

def _adaptive_pool_safe(tensor, size):
    """MPS has a broken adaptive_avg_pool2d for some sizes (PyTorch issue #96056)."""
    if tensor.device.type == "mps":
        return F.adaptive_avg_pool2d(tensor.cpu(), output_size=size).to(tensor.device)
    return F.adaptive_avg_pool2d(tensor, output_size=size)

def _attention_mask_chunk(windows, start, end, feature_size, device):
    """Match AnyUp's CPU window mask, materializing only the requested rows."""
    feat_h, feat_w = feature_size
    r0, r1, c0, c1 = windows[start:end].unbind(dim=1)
    rows = torch.arange(feat_h)
    cols = torch.arange(feat_w)
    row_ok = (rows >= r0[:, None]) & (rows < r1[:, None])
    col_ok = (cols >= c0[:, None]) & (cols < c1[:, None])
    allowed = (row_ok.unsqueeze(2) & col_ok.unsqueeze(1)).reshape(end - start, -1)
    return (~allowed).to(device)

def _anyup_pair_similarity(anyup, img_tensor, en_feat, de_feat, target_size, q_chunk_size):
    """Compute 1 - cosine_similarity of the AnyUp-upsampled en/de feature pair.

    A full AnyUp output is (B, 768, H, W); at 1280x1280 that is ~4.7 GiB per
    sample per feature map, before similarity temporaries. This replays AnyUp's
    inference in query chunks: the image branch (encoding, RoPE, queries) is
    computed once, each feature gets its own key/value (the key depends on the
    feature through key_features_encoder), and the attention output is turned
    into the similarity immediately. The calculation is mathematically equivalent
    to the original; floating-point rounding may depend on device and chunk size.
    Query storage, CPU window bounds and the output still scale with H * W.
    """
    out_h, out_w = target_size
    batch_size = img_tensor.shape[0]
    if en_feat.shape != de_feat.shape or en_feat.shape[0] != batch_size:
        raise ValueError("Image and en/de feature batches must match; en/de shapes must be equal")
    num_pixels = out_h * out_w
    if q_chunk_size is None:
        q_chunk_size = num_pixels
    if q_chunk_size <= 0:
        raise ValueError("q_chunk_size must be positive")

    # Image branch, identical to AnyUp.forward + AnyUp.upsample.
    enc = anyup.image_encoder(img_tensor)
    h = enc.shape[-2]
    coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
    enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
    enc = anyup.rope(enc, coords)
    enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)

    q = _adaptive_pool_safe(anyup.query_encoder(enc), (out_h, out_w))
    q = anyup.cross_decode.conv2d(q)
    q_flat = q.permute(0, 2, 3, 1).reshape(q.shape[0], out_h * out_w, -1)

    feat_h, feat_w = en_feat.shape[-2:]
    k_img = _adaptive_pool_safe(anyup.key_encoder(enc), (feat_h, feat_w))
    cross_attn = anyup.cross_decode.cross_attn
    num_heads = cross_attn.attention.num_heads

    # Per-feature keys/values, identical to AnyUp.upsample for each feature.
    kv = {}
    for name, feat in (("en", en_feat), ("de", de_feat)):
        k = torch.cat([k_img, anyup.key_features_encoder(F.normalize(feat, dim=1))], dim=1)
        k = anyup.aggregation(k)
        k_flat = k.permute(0, 2, 3, 1).reshape(k.shape[0], feat_h * feat_w, -1)
        v = feat.permute(0, 2, 3, 1).reshape(feat.shape[0], feat_h * feat_w, -1)
        kv[name] = (k_flat, cross_attn.norm_k(k_flat), v)

    # Keep only O(H*W) window bounds on CPU, not an H*W by feat_h*feat_w mask.
    # CPU bounds preserve the exact rounding used in AnyUp's original mask.
    windows = None
    if anyup.cross_decode.window_ratio > 0:
        windows = window2d(
            low_res=(feat_h, feat_w), high_res=(out_h, out_w),
            ratio=anyup.cross_decode.window_ratio,
        ).reshape(-1, 4)

    a_map = torch.empty((batch_size, 1, num_pixels), device=q.device, dtype=torch.float32)
    for start in range(0, num_pixels, q_chunk_size):
        end = min(start + q_chunk_size, num_pixels)
        mask_chunk = None
        if windows is not None:
            mask_chunk = _attention_mask_chunk(windows, start, end, (feat_h, feat_w), q.device)
            # Same 3D contiguous mask the model passes on to MultiheadAttention.
            mask_chunk = mask_chunk.unsqueeze(0).expand(q_flat.shape[0] * num_heads, -1, -1).contiguous()
        q_chunk = cross_attn.norm_q(q_flat[:, start:end, :])
        # Match CrossAttention.forward: pass the unnormalized key as MHA's value,
        # discard its projected output, and use attention weights on the features.
        u_en = torch.einsum("b i j, b j d -> b i d", cross_attn.attention(
            q_chunk, kv["en"][1], kv["en"][0], average_attn_weights=True, attn_mask=mask_chunk)[1], kv["en"][2])
        u_de = torch.einsum("b i j, b j d -> b i d", cross_attn.attention(
            q_chunk, kv["de"][1], kv["de"][0], average_attn_weights=True, attn_mask=mask_chunk)[1], kv["de"][2])
        a_map[:, 0, start:end] = 1 - F.cosine_similarity(u_en, u_de, dim=-1)
    return a_map.view(batch_size, 1, out_h, out_w)

def compute_anomaly_map_anyup(anyup, img_tensor, en_list, de_list, target_size=(448, 448), q_chunk_size=2048):
    """AnyUp guided feature super-resolution with chunked similarity and masks."""
    total_a_map = None
    count = 0
    for i in range(len(de_list)):
        if i >= len(en_list): break
        a_map = _anyup_pair_similarity(anyup, img_tensor, en_list[i], de_list[i], target_size, q_chunk_size)
        total_a_map = a_map if total_a_map is None else total_a_map + a_map
        count += 1
    return (total_a_map / count) if count > 0 else total_a_map

def render_heatmap_and_overlay(orig_bgr, a_map_np, threshold=0.15, alpha=0.45, mask_bg=True):
    """Generates colored heatmap, overlay image, and binary mask with proper Min-Max normalization"""
    h, w = orig_bgr.shape[:2]
    # Resize anomaly map back to original image dimensions for display
    a_map_resized = cv2.resize(a_map_np, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Proper Min-Max normalization across image dynamic range (JET: 0=Blue, 128=Green, 255=Red)
    min_val = float(np.min(a_map_resized))
    max_val = float(np.max(a_map_resized))
    denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
    norm_map = np.clip((a_map_resized - min_val) / denom * 255.0, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    
    # Mask pure black background (if present) so it doesn't glow blue/red
    is_bg = None
    if mask_bg:
        is_bg = (orig_bgr[:, :, 0] <= 5) & (orig_bgr[:, :, 1] <= 5) & (orig_bgr[:, :, 2] <= 5)
        heatmap_colored[is_bg] = 0

    # Semi-transparent overlay
    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
    if is_bg is not None:
        overlay[is_bg] = 0

    # Binary mask
    binary_mask = (a_map_resized >= threshold).astype(np.uint8) * 255
    if is_bg is not None:
        binary_mask[is_bg] = 0

    return heatmap_colored, overlay, binary_mask

def _result_paths(output_dir, stem, method):
    sub_dir = output_dir / method
    return {
        "overlay": sub_dir / f"{stem}_{method}_overlay.jpg",
        "heatmap": sub_dir / f"{stem}_{method}_heatmap.jpg",
        "mask": sub_dir / f"{stem}_{method}_mask.png",
    }

def _write_status(path, statuses):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "status", "message"])
        writer.writerows((method, row["status"], row["message"])
                         for method, row in statuses.items())
    temporary.replace(path)

def _save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save image: {path}")

def _analyzer_paths(output_dir, stem, method):
    base = output_dir / method / f"{stem}_{method}"
    return (base.with_name(base.name + "_anomaly_distance_raw.csv"),
            base.with_name(base.name + "_original.png"))

def _save_analyzer_data(output_dir, stem, method, orig_bgr, raw_map):
    """Export unnormalized scores on the same grid used by the rendered mask."""
    raw_map = np.asarray(raw_map)
    if raw_map.ndim != 2 or not np.isfinite(raw_map).all():
        raise ValueError("Analyzer export requires a finite 2D anomaly map")
    h, w = orig_bgr.shape[:2]
    aligned_map = cv2.resize(raw_map, (w, h), interpolation=cv2.INTER_LINEAR)
    csv_path, image_path = _analyzer_paths(output_dir, stem, method)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, aligned_map, delimiter=",", fmt="%.9g")
    _save_image(image_path, orig_bgr)
    print(f"✅ Saved analyzer CSV to: {csv_path}")

def process_single_comparison(image_path, model, anyup, args, output_dir, device, gaussian_fn):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    filename = image_path.name
    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = (["Baseline"] if args.compare_baseline else []) + [f"AnyUp_{size}" for size in args.upsample_sizes]
    statuses = {method: {"status": "pending", "message": ""} for method in methods}
    status_path = output_dir / f"{stem}_inference_status.csv"
    _write_status(status_path, statuses)
    # Remove only this image's requested generated outputs. An OOM, read failure
    # or interrupted run must not leave a previous run's files looking current.
    for method in methods:
        for path in _result_paths(output_dir, stem, method).values():
            path.unlink(missing_ok=True)
        for path in _analyzer_paths(output_dir, stem, method):
            path.unlink(missing_ok=True)
    for kind in ("overlays", "heatmaps"):
        (output_dir / f"{stem}_comparison_grid_{kind}.jpg").unlink(missing_ok=True)
    print(f"\n=======================================================")
    print(f"🔍 Analyzing Image: {filename}")
    print(f"=======================================================")

    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"Error loading {image_path}")
        for row in statuses.values():
            row.update(status="read_error", message=f"Cannot read {image_path}")
        _write_status(status_path, statuses)
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.input_dim, args.input_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 1. Forward pass through Dinomaly
    t0 = time.time()
    with torch.no_grad():
        en, de = model(img_tensor)
    dt_model = time.time() - t0
    print(f"[*] Dinomaly forward pass: {dt_model*1000:.1f}ms | Feats: {en[0].shape}")

    results = {}
    overlay_panels = []
    heatmap_panels = []

    # Prepare input thumbnail for side-by-side
    thumb_h, thumb_w = 448, 448
    thumb_input = cv2.resize(orig_bgr, (thumb_w, thumb_h))
    cv2.putText(thumb_input, "Input Image", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    overlay_panels.append(thumb_input.copy())
    heatmap_panels.append(thumb_input.copy())

    # 2. Baseline without AnyUp (Bilinear interpolation)
    if args.compare_baseline:
        t0 = time.time()
        with torch.no_grad():
            a_map_base = compute_anomaly_map_baseline(en, de, (args.input_dim, args.input_dim))
            a_map_base = gaussian_fn(a_map_base)[0, 0].cpu().numpy()
        dt_base = time.time() - t0
        heatmap_base, overlay_base, mask_base = render_heatmap_and_overlay(orig_bgr, a_map_base, threshold=args.threshold)
        
        col_ov = cv2.resize(overlay_base, (thumb_w, thumb_h))
        cv2.putText(col_ov, "Baseline (Bilinear)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(col_ov, f"{dt_base*1000:.1f}ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        overlay_panels.append(col_ov)

        col_hm = cv2.resize(heatmap_base, (thumb_w, thumb_h))
        cv2.putText(col_hm, "Baseline (Bilinear)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(col_hm, f"{dt_base*1000:.1f}ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        heatmap_panels.append(col_hm)

        results["Baseline"] = {"map": a_map_base, "heatmap": heatmap_base, "overlay": overlay_base, "mask": mask_base, "time": dt_base}

    # 3. AnyUp with different target upsampling resolutions
    for size in args.upsample_sizes:
        t0 = time.time()
        a_map_anyup = None
        oom_message = None
        try:
            with torch.no_grad():
                a_map_anyup = compute_anomaly_map_anyup(anyup, img_tensor, en, de, target_size=(size, size), q_chunk_size=args.q_chunk_size)
                a_map_anyup = gaussian_fn(a_map_anyup)[0, 0].cpu().numpy()
        except RuntimeError as exc:
            # A single oversized/foreign allocation should not abort the whole batch.
            if not _is_oom_error(exc):
                raise
            # Keep only text, not the exception/traceback holding GPU tensors.
            oom_message = str(exc)
        dt_anyup = time.time() - t0
        if oom_message is not None:
            # Also drop a computed map if smoothing or transfer caused the OOM.
            del a_map_anyup
            # The except block has ended, so its traceback has been released.
            release_compute_memory(device)
            statuses[f"AnyUp_{size}"].update(status="oom", message=oom_message)
            _write_status(status_path, statuses)
            print(f"  !! AnyUp @ {size} skipped: {oom_message}")
            skipped_panel = np.zeros_like(thumb_input)
            cv2.putText(skipped_panel, f"AnyUp @ {size}x{size}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(skipped_panel, "OOM - skipped", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            overlay_panels.append(skipped_panel.copy())
            heatmap_panels.append(skipped_panel.copy())
            continue
        release_compute_memory(device)
        heatmap_anyup, overlay_anyup, mask_anyup = render_heatmap_and_overlay(orig_bgr, a_map_anyup, threshold=args.threshold)
        
        col_ov = cv2.resize(overlay_anyup, (thumb_w, thumb_h))
        cv2.putText(col_ov, f"AnyUp @ {size}x{size}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(col_ov, f"{dt_anyup*1000:.1f}ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        overlay_panels.append(col_ov)

        col_hm = cv2.resize(heatmap_anyup, (thumb_w, thumb_h))
        cv2.putText(col_hm, f"AnyUp @ {size}x{size}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(col_hm, f"{dt_anyup*1000:.1f}ms", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        heatmap_panels.append(col_hm)

        results[f"AnyUp_{size}"] = {"map": a_map_anyup, "heatmap": heatmap_anyup, "overlay": overlay_anyup, "mask": mask_anyup, "time": dt_anyup}
        print(f"  -> AnyUp @ {size:<4} : {dt_anyup*1000:.1f}ms (Max anomaly score: {a_map_anyup.max():.3f})")

    # Combine into horizontal side-by-side comparison banners
    banner_overlay = np.hstack(overlay_panels)
    banner_ov_path = output_dir / f"{stem}_comparison_grid_overlays.jpg"
    _save_image(banner_ov_path, banner_overlay)
    print(f"✅ Saved overlay comparison banner to: {banner_ov_path}")

    banner_heatmap = np.hstack(heatmap_panels)
    banner_hm_path = output_dir / f"{stem}_comparison_grid_heatmaps.jpg"
    _save_image(banner_hm_path, banner_heatmap)
    print(f"✅ Saved pure heatmap comparison banner to: {banner_hm_path}")

    # Also save individual full-res overlays, heatmaps, and masks
    for k, v in results.items():
        sub_dir = output_dir / k
        sub_dir.mkdir(parents=True, exist_ok=True)
        for kind, path in _result_paths(output_dir, stem, k).items():
            _save_image(path, v[kind])
        if getattr(args, "save_csv", False):
            _save_analyzer_data(output_dir, stem, k, orig_bgr, v["map"])
        statuses[k].update(status="success", message="")
        _write_status(status_path, statuses)

    # Free GPU memory before the next image (AGENTS.md rule 5).
    del img_tensor, en, de
    release_compute_memory(device)

def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve backbone weight path
    if args.backbone_weights is None:
        candidates = [
            project_root / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            dinomaly_dir / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        ]
        backbone_weights = next((p for p in candidates if p.exists()), None)
        if not backbone_weights:
            print("Error: Could not locate dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth!")
            return
    else:
        backbone_weights = args.backbone_weights

    # 2. Resolve AnyUp weights
    anyup_path = args.anyup_weights
    if not anyup_path.is_absolute():
        anyup_path = project_root / anyup_path
    if not anyup_path.exists():
        print(f"Error: AnyUp weights not found at: {anyup_path}")
        return

    # 3. Load Models
    model = load_dinomaly_model(args.model_path, backbone_weights, device)
    anyup = load_anyup_model(anyup_path, device)
    gaussian_fn = get_gaussian_kernel(kernel_size=5, sigma=1.0)

    # 4. Resolve Images
    if args.image_path:
        images_to_run = [args.image_path]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images_to_run = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)
    else:
        # Default sample from train_imprint or test_imprint
        sample_cand = Path("/Users/an/workspace/imprint_datasets/test_imprint")
        if sample_cand.exists():
            images_to_run = sorted(p for p in sample_cand.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:5]
        else:
            print("Please specify --image_path or --image_dir")
            return

    print(f"\n[*] Processing {len(images_to_run)} images on device: {device}...")
    for img_p in images_to_run:
        process_single_comparison(img_p, model, anyup, args, output_dir, device, gaussian_fn)

    print("\n" + "=" * 60)
    print(f"🎉 All experiments completed! Check results in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
