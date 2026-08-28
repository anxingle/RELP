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
        output_path_base = str(fm_row['Output_Path']).strip() if fm_row is not None and pd.notna(fm_row.get('Output_Path')) else "Result"
        if output_path_base.lower() == 'nan': output_path_base = "Result"

        flow_cfg = cfg_mgr.get_flow(self.fm)
        if not flow_cfg:
            print(f"Warning: No Flow configuration found for {self.fm}")
            # Do not return here, we can fallback to default parameters for Crop Strategy
            target_size = None
            crop_out_dim_raw = None
        else:
            target_size = None
            crop_out_dim_raw = flow_cfg.crop_output_dim
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
        is_grounding_crop = 'groundingcrop' in str(self.fm).lower()
        if (is_dino_crop or is_grounding_crop) and target_size is None:
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
            
        if not weights_df.empty and len(weights_df.columns) > 0:
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
            import os
            sam_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
            sam_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt")

        if not cfg_path_dut or not weights_path_dut:
            cfg_path_dut = "mock"
            weights_path_dut = "mock" 

        # 3. 加载模型
        print(">>> [CropStrategy] Loading models...")
        if is_grounding_crop:
            from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
            import re
            
            grounding_prompt = "object"
            self.mask_scaling_factor = 1.0
            self.target_ratio = None
            self.top_k = None
            
            # GroundingCrop['elliptic object', ratio=1.3, Top_Oppt[1]]
            match = re.search(r"GroundingCrop\[(.*)\]", self.fm, re.IGNORECASE)
            if match:
                params_str = match.group(1)
                prompt_match = re.search(r"['\"]([^'\"]+)['\"]", params_str)
                if prompt_match:
                    grounding_prompt = prompt_match.group(1)
                
                ratio_match = re.search(r"ratio\s*=\s*([0-9.]+)", params_str, re.IGNORECASE)
                if ratio_match:
                    self.target_ratio = float(ratio_match.group(1))
                    
                top_match = re.search(r"Top(?:_Oppt)?\[(\d+)\]", params_str, re.IGNORECASE)
                if top_match:
                    self.top_k = int(top_match.group(1))
                    
            print(f"  -> Grounding DINO mode detected. Prompt: '{grounding_prompt}', Aspect Ratio Filter: {self.target_ratio}, Top-K Filter: {self.top_k}")
            
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
            local_model_path = os.path.join(project_root, "weights", "Grounding Dino")
            
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
                print(f"  -> Loading local Grounding DINO model from: {local_model_path}")
                dut_predictor = load_grounding_dino_model(model_id=local_model_path, box_threshold=0.25)
            else:
                print(f"  -> Loading Grounding DINO model from HuggingFace Hub: IDEA-Research/grounding-dino-tiny")
                dut_predictor = load_grounding_dino_model(box_threshold=0.25)
                
            self.grounding_prompt = grounding_prompt
        else:
            dut_predictor = ug.load_detectron2_model(cfg_path_dut, weights_path_dut, score_thresh=0.5)
            self.mask_scaling_factor = 1.0
        
        import torch
        # !!! 终极修复：绝对不要在 Mac 上使用 mps 跑 SAM2 !!!
        # MPS (Apple Silicon GPU) 跑 SAM2 会遭遇严重的精度截断 Bug，导致输出的 Mask 变成极为夸张的几何锯齿状（只剩几个多边形顶点）
        # 这是导致业务管线生成的 Mask 极其糟糕（夸张的狗牙、像被咬掉一块）的根本原因！
        # test_k11p 脚本之所以边缘完美，正是因为它显式硬编码了 device="cpu"！
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
            import numpy as np
            if is_grounding_crop:
                detections = dut_predictor.predict(image_rgb, text_prompt=self.grounding_prompt)
                
                # 按照置信度 (score) 从高到低排序，以支持 Top_Oppt[K] 的概率优先
                detections = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)
                
                tape_boxes = []
                tape_classes = []
                for d in detections:
                    if self.target_ratio is not None:
                        x1, y1, x2, y2 = d['bbox']
                        w, h = x2 - x1, y2 - y1
                        if w > 0 and h > 0:
                            aspect_ratio = max(w/h, h/w)
                            if aspect_ratio < self.target_ratio:
                                continue # 舍弃长宽比不满足要求的框
                                
                    tape_boxes.append(d['bbox'])
                    tape_classes.append(0)
                    
                    if self.top_k is not None and len(tape_boxes) >= self.top_k:
                        break
                        
                tape_boxes = np.array(tape_boxes) if tape_boxes else np.array([])
                tape_classes = np.array(tape_classes) if tape_classes else np.array([])
            else:
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
                mask_scaling_factor=self.mask_scaling_factor, # 如果需要可从 Excel 读
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
                
                # !!! 时光胶囊 !!!：捕捉经历下游除杂、膨胀之前的，原汁原味 SAM 紧凑边界
                import numpy as np
                pristine_y, pristine_x = np.where(det["mask"] > 0)
                pristine_y_min, pristine_y_max = np.min(pristine_y), np.max(pristine_y)
                pristine_x_min, pristine_x_max = np.min(pristine_x), np.max(pristine_x)
                det["pristine_bbox"] = (pristine_y_min, pristine_y_max, pristine_x_min, pristine_x_max)
                
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
                
                crop_out_dim_raw = flow_cfg.crop_output_dim if flow_cfg else None
                
                suffix = img_path.suffix
                if len(all_detections_sorted) == 1:
                    out_name = f"{img_path.stem}_crop{suffix}"
                else:
                    out_name = f"{img_path.stem}_Instance{idx+1}_crop{suffix}"
                out_path = crop_res_folder / out_name
                
                if is_dino_crop or is_grounding_crop:
                    from core.pipeline import DefectDetectionPipeline
                    import utils.crop_ops as crop_ops
                    import json
                    
                    # Read Dino_Input_Dim
                    target_size_save = None
                    try:
                        if flow_cfg:
                            dim_str = str(flow_cfg.dino_input_dim).replace("'", '"')
                            if dim_str and dim_str.lower() != 'nan':
                                if dim_str.startswith('['):
                                    dim_list = json.loads(dim_str)
                                    target_size_save = int(dim_list[0])
                                else:
                                    target_size_save = int(float(dim_str))
                    except: pass
                    
                    if not target_size_save:
                        target_size_save = 1280 # Default fallback
                        
                    # Use the modern crop_and_resize with padding=10 for Dino requirement
                    cropped_img, _, _ = crop_ops.crop_and_resize(clean_image, det["mask"], padding=10, target_size=target_size_save)
                    if cropped_img is not None:
                        cv2.imwrite(str(out_path), cropped_img)
                        print(f"     ✅ Saved Modern Dino/Grounding Crop to {out_path}")

                else:
                    import numpy as np
                    def process_crop_failure_mode(image, mask, crop_output_dim, output_path):
                        if mask is None or image is None: return
                        
                        # !!! 终极修复：绝对不要使用膨胀后的 mask 的边界去切图 !!!
                        # 使用物体最原始的 BBox 物理边界去切图。这不仅保证了100%切出原生检测的框，
                        # 更杜绝了所有由于膨胀导致边缘撑大，进而引发 Resize 时相位偏移（狗牙）的问题。
                        # 我们回退使用 det['box'] 作为切图和计数的唯一真相！
                        x_min, y_min, x_max, y_max = 0, 0, 0, 0
                        use_original_box = False
                        
                        if det is not None and "box" in det:
                            orig_box = det["box"]
                            orig_x_min, orig_y_min, orig_x_max, orig_y_max = map(int, orig_box)
                            img_h, img_w = image.shape[:2]
                            x_min = max(0, min(orig_x_min, img_w-1))
                            x_max = max(0, min(orig_x_max, img_w-1))
                            y_min = max(0, min(orig_y_min, img_h-1))
                            y_max = max(0, min(orig_y_max, img_h-1))
                            
                            if x_max > x_min and y_max > y_min:
                                use_original_box = True
                                
                        if not use_original_box:
                            # 降级方案
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

                        # 完全照搬 test_k11p_general_direct.py 的缩放和画布逻辑！
                        # 因为我们在上面已经强制使用原生长宽切出了 img_crop，所以这里的 w_curr 和 h_curr 就是完美的！
                        scale = min(target_max / w_curr, target_max / h_curr)
                        new_w, new_h = int(w_curr * scale), int(h_curr * scale)
                        if new_h < 1: new_h = 1
                        if new_w < 1: new_w = 1
                        
                        # 强制使用 INTER_LINEAR 获得平滑的抗锯齿边
                        img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                                
                        
                        
                        # 创建画布，不再使用 padding 和 16倍数 (除非业务流强行指定了 target_width)
                        canvas_h = target_max
                        canvas_w = target_width if target_width is not None else target_max
                        
                        final_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        x_off = (canvas_w - new_w) // 2
                        y_off = (canvas_h - new_h) // 2
                        
                        y_end, x_end = min(y_off+new_h, canvas_h), min(x_off+new_w, canvas_w)
                        h_paste, w_paste = y_end - y_off, x_end - x_off
                        if h_paste > 0 and w_paste > 0:
                            final_canvas[y_off:y_end, x_off:x_end] = img_resized[:h_paste, :w_paste]
                        
                        cv2.imwrite(output_path, final_canvas)
                        
                    process_crop_failure_mode(clean_image, det["mask"], crop_out_dim_raw, str(out_path))
                    print(f"     ✅ Saved Crop (16x padded) to {out_path}")


            processed_count += 1
            
        print(f"\n>>> [CropStrategy] Execution Completed! Processed {processed_count} images.")

