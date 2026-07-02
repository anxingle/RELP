import utils.utils_general as ug
import pandas as pd
import os
import cv2
import numpy as np
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils import cv_ops, crop_ops

class CropStrategy(AnalysisStrategy):
    """
    极简的多实例切图策略 (Multi-Instance Crop Strategy)
    1. 拿原图过目标检测，找到 N 个物体的边界框。
    2. 用 SAM 为这 N 个框分别抠出精确的 Mask。
    3. 遍历这 N 个物体，使用各自的 Mask 保留自身、涂黑背景。
    4. 分别执行 Tight Crop 截断黑边。
    5. 根据 Crop_Output_Dim 进行等比缩放和 Padding（补充为正方形）。
    6. 分别保存，支持一张图切出 N 张完美画布图。
    """
    def execute(self) -> None:
        print(f">>> [CropStrategy] Executing for FM='{self.fm}'...")
        cfg_mgr = ConfigManager()
        
        # 1. 解析配置
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == self.fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        output_path_base = str(fm_row['Output_Path']) if fm_row is not None else "Result"

        flow_cfg = cfg_mgr.get_flow(self.fm)
        if not flow_cfg:
            print(f"Warning: No Flow configuration found for {self.fm}")
            return
            
        # 读取缩放尺寸 (默认给 Dino 用的可能需要方形，也可能是传统 Crop)
        target_size = None
        crop_output_dim = flow_cfg.crop_output_dim
        
        if crop_output_dim is not None and not pd.isna(crop_output_dim):
            if isinstance(crop_output_dim, str):
                crop_output_dim = crop_output_dim.strip()
                if crop_output_dim.startswith('[') and crop_output_dim.endswith(']'):
                    parts = crop_output_dim[1:-1].split(',')
                    if len(parts) > 0: target_size = int(parts[0].strip())
                else:
                    target_size = int(crop_output_dim)
            elif isinstance(crop_output_dim, (int, float)):
                target_size = int(crop_output_dim)
        
        is_dino_crop = 'dino' in str(self.fm).lower()
        if is_dino_crop and target_size is None:
             target_size = 1280 # Dino 必须要求有尺寸以构建方形
        
        print(f">>> [CropStrategy] Target Canvas Size parsed as: {target_size}")


        # 2. 解析模型权重
        weights_df = cfg_mgr.get_sheet('Weights')
        
        cfg_path_dut = None
        weights_path_dut = None
        sam_cfg = None
        sam_ckpt = None
        
        import utils.base_utils as bu
        def _norm(s): return bu._norm(s) if hasattr(bu, '_norm') else str(s).strip().lower().replace(" ", "_")
        
        candidates = [self.fm, _norm(self.fm), f"{self.product}_{self.generation}", f"{self.product} {self.generation}"]
        if self.fm.startswith('Crop_') or _norm(self.fm).startswith('crop_'):
            candidates.insert(0, 'Crop')
            candidates.insert(1, 'crop')
            
        first_col = weights_df.columns[0]
        for k in candidates:
            match = weights_df[weights_df[first_col] == k]
            if not match.empty:
                for _, row in match.iterrows():
                    var_name = str(row.get('variable_name', '')).strip()
                    path_val = row.get('relative_path')
                    if pd.notna(path_val) and var_name:
                        import os
                        p = os.path.abspath(os.path.join(os.getcwd(), path_val)) if not os.path.isabs(path_val) else path_val
                        if var_name == 'cfg_path_dut': cfg_path_dut = p
                        elif var_name == 'weights_path_dut': weights_path_dut = p
                        elif var_name == 'sam_config': sam_cfg = p
                        elif var_name == 'sam_weights': sam_ckpt = p



        if not sam_ckpt:
            sam_cfg = getattr(ug, 'sam_config_path', None)
            sam_ckpt = getattr(ug, 'sam_checkpoint_path', None)

        if not cfg_path_dut or not weights_path_dut:

            print("Error: DUT Object Detection weights missing!")
            return

        # 3. 加载模型
        print(">>> [CropStrategy] Loading models...")
        dut_predictor = ug.load_detectron2_model(cfg_path_dut, weights_path_dut, score_thresh=0.5)
        
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = ug._real_init_sam2(sam_cfg, sam_ckpt, device=device_str)
        sam2_predictor = ug.get_sam2_predictor(sam2_model)
        
        if not sam2_predictor:
            print("Error: SAM2 Initialization failed!")
            return

        # 4. 创建输出目录
        import pathlib
        input_dir = pathlib.Path(self.download_path)
        output_folder_name = f"{self.product}_{self.generation}_{self.fm}_Result"
        if output_path_base == "Result":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
            output_dir = pathlib.Path(project_root) / "Result" / output_folder_name
        else:
            output_dir = pathlib.Path(output_path_base) / output_folder_name
            
        crop_res_folder = output_dir / ("Dino_Crop_Result" if is_dino_crop else "Crop_Result")
        crop_res_folder.mkdir(parents=True, exist_ok=True)
        

        if not input_dir.exists():
            print(f"❌ Input path does not exist: {input_dir}")
            return
            
        # Recursive glob to find images
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(list(input_dir.rglob(ext)))
            image_paths.extend(list(input_dir.rglob(ext.upper())))
            
        print(f"📂 Found {len(image_paths)} images to process")
        # 5. 遍历图片进行多目标切图
        processed_count = 0
        for img_path in image_paths:
            print(f"\n--- Processing image: {img_path.name} ---")
            image = cv2.imread(str(img_path))
            if image is None: continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # (1) 原图找框
            tape_boxes, tape_classes = ug.detect_bounding_boxes(image, dut_predictor)
            if len(tape_boxes) == 0:
                print("No DUT objects found in image. Skipping.")
                continue
                
            print(f"Found {len(tape_boxes)} object bounds.")
            
            # (2) 交给 SAM 生成所有 Mask 列表
            sam2_predictor.set_image(image_rgb)
            all_detections = ug.extract_detections(
                sam2_predictor, image_rgb, tape_boxes, 
                tape_classes=tape_classes,
                mask_scaling_factor=1.0, # 如果需要可从 Excel 读
            )
            
            if not all_detections:
                print("SAM failed to extract any masks.")
                continue
                
            # 按面积从大到小排个序，方便命名
            all_detections_sorted = sorted(all_detections, key=lambda d: d["area"], reverse=True)
            print(f"SAM successfully segmented {len(all_detections_sorted)} instances.")


            # (3) 遍历每一个物体，单独处理
            for idx, det in enumerate(all_detections_sorted):
                print(f"  -> Processing Instance {idx+1}/{len(all_detections_sorted)} (Area: {det['area']:.0f})")
                
                # 准备当前物体的纯净版图像（背景涂黑）
                clean_image = image.copy()
                clean_image[det["mask"] == 0] = 0
                
                # [FEATURE REMOVE] - Add Defect Removal if specified in Flow
                if hasattr(ug, 'Feature_Scaling_Config') and getattr(ug, 'Feature_Scaling_Config'):
                    try:
                        clean_image, _ = ug.remove_defects(
                            clean_image, det["mask"], dut_predictor, sam2_predictor,
                            feature_scaling_config=ug.Feature_Scaling_Config,
                            class_names=None # Can add if needed
                        )
                    except Exception as e:
                        print(f"Warning: Feature_Remove failed for Instance {idx+1}: {e}")



                # 核心切割与缩放：引入正确的 June3 缩放逻辑 (Crop vs Dino Crop)
                import json
                
                crop_out_dim_raw = flow_cfg.crop_output_dim
                
                suffix = img_path.suffix
                if len(all_detections_sorted) == 1:
                    out_name = f"{img_path.stem}_crop{suffix}"
                else:
                    out_name = f"{img_path.stem}_Instance{idx+1}_crop{suffix}"
                out_path = crop_res_folder / out_name
                
                if 'dino' in str(self.fm).lower():
                    from core.pipeline import DefectDetectionPipeline
                    import utils.crop_ops as crop_ops
                    import json
                    
                    # Read Dino_Input_Dim
                    target_size = None
                    try:
                        dim_str = str(flow_cfg.dino_input_dim).replace("'", '"')
                        if dim_str and dim_str.lower() != 'nan':
                            if dim_str.startswith('['):
                                dim_list = json.loads(dim_str)
                                target_size = int(dim_list[0])
                            else:
                                target_size = int(float(dim_str))
                    except: pass
                    
                    if not target_size:
                        target_size = 1280 # Default fallback
                        
                    # Use the modern crop_and_resize with padding=10 for Dino requirement
                    cropped_img, _, _ = crop_ops.crop_and_resize(clean_image, det["mask"], padding=10, target_size=target_size)
                    if cropped_img is not None:
                        cv2.imwrite(str(out_path), cropped_img)
                        print(f"     ✅ Saved Modern Dino Crop to {out_path}")

                else:
                    import numpy as np
                    def process_crop_failure_mode(image, mask, crop_output_dim, output_path):
                        if mask is None or image is None: return
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) == 0: return 
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        img_crop = image[y_min:y_max+1, x_min:x_max+1]
                        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
                        img_crop = img_crop.copy()
                        img_crop[mask_crop == 0] = 0
                        h_curr, w_curr = img_crop.shape[:2]
                        target_max, target_width = None, None
                        
                        if isinstance(crop_output_dim, str):
                            crop_output_dim = crop_output_dim.strip()
                            if crop_output_dim.startswith('[') and crop_output_dim.endswith(']'):
                                 try:
                                     parts = crop_output_dim[1:-1].split(',')
                                     vals = [int(float(p.strip())) for p in parts if p.strip()]
                                     if len(vals) > 0: target_max = vals[0]
                                     if len(vals) > 1: target_width = vals[1]
                                 except: pass
                            else:
                                 try: target_max = int(float(crop_output_dim))
                                 except: pass
                        elif isinstance(crop_output_dim, (int, float)):
                            target_max = int(crop_output_dim)
                        elif isinstance(crop_output_dim, (list, tuple)):
                            if len(crop_output_dim) > 0: target_max = int(crop_output_dim[0])
                            if len(crop_output_dim) > 1: target_width = int(crop_output_dim[1])
                        
                        if target_max is None:
                            print("Warning: Invalid Crop_Output_Dim, saving original crop.")
                            cv2.imwrite(output_path, img_crop)
                            return

                        min_pad = 10 
                        max_curr = max(h_curr, w_curr)
                        scale = 1.0
                        available_max = target_max - (2 * min_pad)
                        if available_max < 1: available_max = 1
                        if max_curr > available_max: scale = available_max / max_curr
                        else: scale = 1.0
                        
                        new_h, new_w = int(h_curr * scale), int(w_curr * scale)
                        if new_h < 1: new_h = 1
                        if new_w < 1: new_w = 1
                        img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        
                        canvas_h = new_h + (2 * min_pad)
                        canvas_w = new_w + (2 * min_pad)
                        if target_width is not None and canvas_w < target_width:
                            canvas_w = target_width
                            
                        def make_multiple_16(val):
                            rem = val % 16
                            return val if rem == 0 else val + (16 - rem)
                            
                        canvas_h, canvas_w = make_multiple_16(canvas_h), make_multiple_16(canvas_w)
                        final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        x_off, y_off = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
                        y_end, x_end = min(y_off+new_h, canvas_h), min(x_off+new_w, canvas_w)
                        h_paste, w_paste = y_end - y_off, x_end - x_off
                        if h_paste > 0 and w_paste > 0:
                            final_canvas[y_off:y_end, x_off:x_end] = img_resized[:h_paste, :w_paste]
                        cv2.imwrite(output_path, final_canvas)

                    process_crop_failure_mode(clean_image, det["mask"], crop_out_dim_raw, str(out_path))
                    print(f"     ✅ Saved Crop (16x padded) to {out_path}")


            processed_count += 1
            
        print(f"\n>>> [CropStrategy] Execution Completed! Processed {processed_count} images.")

