import os
import sys
import time
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# If run from inside conf_gui, also add parent
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)

from utils import dinov3_utils

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dino_model_path = "weights/imprint_train_2000_v2/final_model.pth"
    test_image_path = "/Users/an/workspace/imprint_datasets/center_test_imprint/1_Imprint_Base Textile imprint Pictures To Minnie_20260630(Black)_Imprint_R437-P1-DK3 Dark-R1-m-2.jpg"

    print("==================================================")
    print("🚀 Running AnyUp Super-Resolution Comparison...")
    print(f"Device: {device}")
    print(f"Model: {dino_model_path}")
    print(f"Image: {test_image_path}")
    print("==================================================")

    dino_model = dinov3_utils.load_model(dino_model_path, device)
    anyup_model = dinov3_utils.load_anyup_upsampler(device)

    # 1. 严格对齐 Dinomaly 训练期预处理 (Resize 512 + CenterCrop 448)
    crop_size = 448
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 准备送入网络的原图裁剪版本 (作为展示用)
    unnorm_crop_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(crop_size),
    ])

    img_pil = Image.open(test_image_path).convert("RGB")
    t_img = transform(img_pil).unsqueeze(0).to(device)

    # 获取无归一化的原始局部裁切图 (BGR)
    raw_crop_bgr = cv2.cvtColor(np.array(unnorm_crop_transform(img_pil)), cv2.COLOR_RGB2BGR)

    out_base_dir = "Result/Comparison_Imprint_Fixed_v2"
    os.makedirs(out_base_dir, exist_ok=True)

    # 测试配置：原图 + Baseline + 3 个不同的 AnyUp 分辨率
    test_configs = [
        ("Baseline (Bilinear)", None),
        ("AnyUp @ 320x320", 320),
        ("AnyUp @ 448x448", 448),
        ("AnyUp @ 640x640", 640),
    ]

    # 初始化拼图容器
    heatmap_panels = []
    overlay_panels = []

    # --- Panel 0: Input Image ---
    panel_input_hm = raw_crop_bgr.copy()
    cv2.putText(panel_input_hm, "Input Image", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    heatmap_panels.append(panel_input_hm)

    panel_input_ov = raw_crop_bgr.copy()
    cv2.putText(panel_input_ov, "Input Image", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    overlay_panels.append(panel_input_ov)

    # 模型推理
    with torch.no_grad():
        en, de = dino_model(t_img)

        for label, target_size in test_configs:
            print(f"\n>>> Running for: {label}...")
            folder_name = label.replace(" ", "_").replace("@", "").replace("(", "").replace(")", "")
            sub_dir = os.path.join(out_base_dir, folder_name)
            os.makedirs(sub_dir, exist_ok=True)

            t0 = time.time()
            if target_size is None:
                # 经典双线性插值
                a_map, _ = dinov3_utils.cal_anomaly_maps(en, de, crop_size)
            else:
                # AnyUp 特征超分辨率上采样
                a_map, _ = dinov3_utils.cal_anomaly_maps_anyup(anyup_model, t_img, en, de, (target_size, target_size))
            dt_ms = (time.time() - t0) * 1000.0

            # 归一化并转为彩色热力图 (JET)
            a_arr = a_map[0, 0].cpu().numpy()
            a_norm = (a_arr - a_arr.min()) / (a_arr.max() - a_arr.min() + 1e-8)
            heatmap = cv2.applyColorMap((a_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # 统一尺寸为 448x448
            if heatmap.shape[:2] != (crop_size, crop_size):
                heatmap_448 = cv2.resize(heatmap, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            else:
                heatmap_448 = heatmap.copy()

            # 生成半透明 Overlay 融合图 (55% 原图 + 45% 热力图)
            overlay_448 = cv2.addWeighted(raw_crop_bgr, 0.55, heatmap_448, 0.45, 0)

            # 保存单项产物
            cv2.imwrite(os.path.join(sub_dir, "heatmap.png"), heatmap_448)
            cv2.imwrite(os.path.join(sub_dir, "overlay.png"), overlay_448)
            print(f"  -> Time: {dt_ms:.1f}ms | Saved to {sub_dir}/")

            # 制作拼图单项 Panel（打上标题与延迟标签）
            panel_hm = heatmap_448.copy()
            cv2.putText(panel_hm, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel_hm, f"{dt_ms:.1f}ms", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            heatmap_panels.append(panel_hm)

            panel_ov = overlay_448.copy()
            cv2.putText(panel_ov, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel_ov, f"{dt_ms:.1f}ms", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            overlay_panels.append(panel_ov)

    # ========================================================
    # 拼接并保存最终全景长图
    # ========================================================
    # 1. 纯热力图全景对比长图
    grid_heatmap = np.hstack(heatmap_panels)
    grid_heatmap_path = os.path.join(out_base_dir, "comparison_grid_heatmaps.jpg")
    cv2.imwrite(grid_heatmap_path, grid_heatmap)
    print(f"\n✅ 纯 Heatmap 对比长图已生成: {grid_heatmap_path}")

    # 2. 半透明 Overlay 叠加全景对比长图 (与业务标杆对齐)
    grid_overlay = np.hstack(overlay_panels)
    grid_overlay_path = os.path.join(out_base_dir, "comparison_grid_overlays.jpg")
    cv2.imwrite(grid_overlay_path, grid_overlay)
    print(f"✅ 半透明 Overlay 对比长图已生成: {grid_overlay_path}")

    print("\n" + "=" * 60)
    print("🎉 全部对比与长图拼接任务圆满完成！")
    print(f"📂 成果目录: {out_base_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
