import os
import cv2
import pandas as pd
import numpy as np
import re
import torch
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
import utils.utils_general as ug
import utils.dinov3_utils as dinov3_utils
from core.pipeline import DefectDetectionPipeline

class GroundingSamStrategy(AnalysisStrategy):
    """
    Zero-Shot 究极形态：Grounding DINO (文本找框) + SAM2 (框出精细Mask) + Post Processing (在Mask内找缺陷)。
    """
    def execute(self) -> None:
        print(f">>> [ROUTER] Routing '{self.fm}' to Modern GroundingSAM implementation! 🚀")
        
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        # 1. 解析基础配置
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        if not os.path.isabs(self.download_path):
            self.download_path = os.path.abspath(os.path.join(os.getcwd(), self.download_path))
        
        # 2. 解析 Prompt
        flow_df = cfg_mgr.get_sheet('Flow')
        prompt_str = "defect."
        target_ratio = None
        self.top_k = None
        ops_to_run = []
        self.has_dino_suffix = False
        
        if not flow_df.empty:
            flow_row = flow_df[flow_df['Failure Mode'] == fm]
            if not flow_row.empty:
                defect_id_method = str(flow_row.iloc[0].get('Defect Identification', ''))
                # 解析方括号里的内容 GroundingSAM[elliptic camera.,ratio=1.3,Top[1]]-Dino
                match = re.search(r'GroundingSAM\[(.*)\]', defect_id_method, re.IGNORECASE)
                if match:
                    full_params = match.group(1).strip()
                    parts = full_params.split(',')
                    prompt_str = parts[0].strip().strip("'").strip('"')
                    for part in parts[1:]:
                        part_clean = part.strip().lower()
                        if 'ratio' in part_clean:
                            ratio_val_str = part_clean.split('=')[-1].strip()
                            if ':' in ratio_val_str:
                                try:
                                    num, den = ratio_val_str.split(':')
                                    target_ratio = float(num) / float(den)
                                except: pass
                            else:
                                try: target_ratio = float(ratio_val_str)
                                except: pass
                        elif 'top' in part_clean:
                            top_match = re.search(r"top(?:_oppt)?\[(\d+)\]", part_clean, re.IGNORECASE)
                            if top_match:
                                self.top_k = int(top_match.group(1))
                                
                print(f"  -> Grounding SAM mode detected. Prompt: '{prompt_str}', Aspect Ratio Filter: {target_ratio}, Top-K Filter: {self.top_k}")
                                
                from utils.utils_general import parse_defect_id_method
                parsed_id_ops = parse_defect_id_method(defect_id_method)
                dino_op = next((op for op in parsed_id_ops if 'dino' in op['name'].lower() and 'grounding' not in op['name'].lower()), None)
                if dino_op:
                    self.has_dino_suffix = True
                    print(f">>> [Strategy Info] Detected -Dino suffix in Defect Identification!")
                else:
                    self.has_dino_suffix = False
                    
                # Extract any intermediate ops (like Slicing) before Dino
                self.pre_dino_ops = [op for op in parsed_id_ops if 'grounding' not in op['name'].lower() and 'dino' not in op['name'].lower()]
                if self.pre_dino_ops:
                    print(f">>> [Strategy Info] Detected pre-Dino ops: {[op['name'] for op in self.pre_dino_ops]}")

                post_method = flow_row.iloc[0].get('Post Processing', '')
                if pd.notna(post_method) and str(post_method).lower() != 'nan':
                    from utils.utils_general import parse_defect_id_method
                    ops_to_run.extend(parse_defect_id_method(str(post_method)))
                        
        if not prompt_str.endswith('.'):
            prompt_str += '.'
            
        print(f">>> [Strategy Info] 解析出的自然语言 Prompt: '{prompt_str}'")

        # 3. 加载权重 (Grounding DINO + SAM2)
        weights_df = cfg_mgr.get_sheet('Weights')
        gdino_path = None
        
        if not weights_df.empty:
            matches = weights_df[weights_df.iloc[:, 0].astype(str).str.contains('Grounding', case=False, na=False)]
            if not matches.empty:
                from utils.base_utils import _norm
                path_col = next((c for c in weights_df.columns if _norm(c) in ('relativepath', 'relative_path', 'path')), None)
                if path_col and pd.notna(matches.iloc[0][path_col]):
                    gdino_path = str(matches.iloc[0][path_col]).strip()
        
        if gdino_path and not os.path.isabs(gdino_path):
            gdino_path = os.path.abspath(os.path.join(os.getcwd(), gdino_path))
        if not gdino_path or not os.path.exists(gdino_path):
            gdino_path = os.path.expanduser("~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
            
        print(f">>> [Strategy] Loading Grounding DINO model...")
        gdino_predictor = load_grounding_dino_model(model_id=gdino_path, box_threshold=0.25)
        
        print(f">>> [Strategy] Loading SAM2 model...")
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
        sam_cfg = weights.get('sam_config', os.path.join(os.getcwd(), "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"))
        sam_ckpt = weights.get('sam_weights', os.path.join(os.getcwd(), "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
        device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        sam2_model = ug._real_init_sam2(sam_cfg, sam_ckpt, device=device_str)
        sam2_predictor = ug.get_sam2_predictor(sam2_model)
        
        # 加载 DINOv3 (如果需要)
        self.dino_model = None
        self.upsampler = None
        dino_weight_path = weights.get('dino') or weights.get('Dino') or weights.get('dinov3_weights')
        if dino_weight_path:
            self.dino_model = dinov3_utils.load_model(dino_weight_path, torch.device(device_str))
            
        use_anyup = str(getattr(flow_row.iloc[0], 'Upsampling', '')).strip().lower() in ('yes', 'true', '1') if not flow_row.empty else False
        if use_anyup:
            self.upsampler = dinov3_utils.load_anyup_upsampler(torch.device(device_str))
            
            # 解析 Upsampling Definition
            upsampling_def = getattr(flow_row.iloc[0], 'Upsampling Definition', None) if not flow_row.empty else None
            utils_general.Upsampling_Target_Size = 448 # Default fallback value
            if upsampling_def and str(upsampling_def).strip().lower() != 'nan':
                try:
                    custom_size = int(float(upsampling_def))
                    if device_str == 'mps' and custom_size > 640:
                        print(f">>> [GroundingSamStrategy] DEBUG: MPS Device detected. Capping Upsampling Target Size from {custom_size} to 640.")
                        custom_size = 640
                    utils_general.Upsampling_Target_Size = custom_size
                    print(f">>> [GroundingSamStrategy] Global Upsampling Target Size set to: {utils_general.Upsampling_Target_Size}")
                except Exception as e:
                    print(f">>> [GroundingSamStrategy] Warning: Could not parse Upsampling Definition '{upsampling_def}': {e}")
            else:
                print(f">>> [GroundingSamStrategy] No Upsampling Definition found in Excel. Using default: 448")



            
            
        # 4. 解析结果保存目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        output_path_type = str(fm_row['Output_Path']).strip() if fm_row is not None and pd.notna(fm_row.get('Output_Path')) else 'Result'
        if output_path_type.lower() == 'nan': output_path_type = 'Result'
        res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result") if output_path_type == 'Result' else os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        base_inferred_pic_dir = os.path.join(res_path, 'Inferred Pic')
        os.makedirs(base_inferred_pic_dir, exist_ok=True)
        
        debug_mode = str(flow_row.iloc[0].get('Debug Mode', 'No')).strip().lower() == 'yes' if not flow_row.empty else False
        base_reference_dir = os.path.join(res_path, 'reference')
        if debug_mode:
            os.makedirs(base_reference_dir, exist_ok=True)
        
        import utils.filtering_ops as fo
        import utils.output_ops as output_ops
        
        parametric_results = []
        # Pre-load base filtering_df for performance, will dynamically swap inside loop if needed
        base_filtering_df = fo.load_and_filter_filtering_sheet("RELP_Configuration.xlsx", self.product, self.generation, fm, None, verbose=False)

        # 5. 执行推理
        processed_count = 0
        print(f"DEBUG: download_path is {self.download_path}")
        for root, dirs, files in os.walk(self.download_path):
            # Resolve relative folder structure
            try:
                rel_dir = os.path.relpath(root, self.download_path)
                if rel_dir == '.':
                    rel_dir = ''
            except ValueError:
                rel_dir = ''
                
            # Create subfolders for the current relative directory
            inferred_pic_dir = os.path.join(base_inferred_pic_dir, rel_dir)
            reference_dir = os.path.join(base_reference_dir, rel_dir)
            if rel_dir:
                os.makedirs(inferred_pic_dir, exist_ok=True)
                if debug_mode:
                    os.makedirs(reference_dir, exist_ok=True)

            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n  -> Processing {file} at {img_path} ...")
                
                # Dynamic Filtering Loading for Matte variants
                current_filtering_df = base_filtering_df
                if 'matte' in img_path.lower() or 'matte' in file.lower():
                    print(f"DEBUG: 'matte' keyword detected in path. Attempting to load specific '{fm} (matte)' filtering config...")
                    matte_filtering_df = fo.load_and_filter_filtering_sheet("RELP_Configuration.xlsx", self.product, self.generation, f"{fm} (matte)", None, verbose=False)
                    if not matte_filtering_df.empty:
                        print(f"DEBUG: Successfully loaded '{fm} (matte)' specific filtering configuration.")
                        current_filtering_df = matte_filtering_df
                    else:
                        print(f"DEBUG: '{fm} (matte)' specific configuration not found. Falling back to base '{fm}'.")
                
                image = cv2.imread(img_path)
                if image is None: continue
                
                # 5.1 Grounding DINO 找框
                results = gdino_predictor.predict(image, prompt_str)
                
                # 按照置信度 (score) 从高到低排序，以支持 Top_Oppt[K] 的概率优先
                results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
                
                filtered_results = []
                for res in results:
                    x1, y1, x2, y2 = res['bbox']
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        aspect_ratio = max(w/h, h/w)
                        if target_ratio is None or aspect_ratio >= target_ratio:
                            filtered_results.append(res)
                            if self.top_k is not None and len(filtered_results) >= self.top_k:
                                break
                
                print(f"     [GroundingDINO] Found {len(filtered_results)} valid BBoxes.")
                
                # 5.2 SAM2 抠图
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                sam2_predictor.set_image(image_rgb)
                
                sam_mask_total = np.zeros(image.shape[:2], dtype=np.uint8)
                boxes_to_draw = []
                
                for res in filtered_results:
                    box = res['bbox']
                    x1, y1, x2, y2 = map(int, box)
                    boxes_to_draw.append((x1, y1, x2, y2, res['class_name'], res['score']))
                    
                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=None, point_labels=None,
                        box=np.array([x1, y1, x2, y2]), multimask_output=False
                    )
                    
                    mask = masks[0]
                    if len(mask.shape) == 3: mask = mask[0]
                    
                    # Resize ONNX mask
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                    sam_mask_total[mask > 0] = 255
                
                # 输出 SAM 抠出的原图 (仅保留目标区域，其余全黑背景)
                if debug_mode:
                    sam_crop_display = np.zeros_like(image)
                    sam_crop_display[sam_mask_total > 0] = image[sam_mask_total > 0]
                    cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_crop.jpg"), sam_crop_display)

                # 输出单独的 SAM Mask (带红框参照) - 移动到 reference 文件夹
                if debug_mode:
                    sam_display = np.zeros_like(image)
                    sam_display[sam_mask_total > 0] = [255, 255, 255]
                    for x1, y1, x2, y2, cls_name, score in boxes_to_draw:
                        cv2.rectangle(sam_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_sam_mask.jpg"), sam_display)

                base_defect_mask = sam_mask_total
                if self.has_dino_suffix and self.dino_model:
                    
                    dino_contour_mask = sam_mask_total
                    if getattr(self, 'pre_dino_ops', None):
                        print(f"     [Pre-DINO] Applying spatial cropping ops: {[op['name'] for op in self.pre_dino_ops]}")
                        # We temporarily reuse the apply_filtering from filtering_ops
                        sliced_mask, _, _, _, _ = fo.apply_filtering(
                            sam_mask_total, image, self.pre_dino_ops, current_filtering_df,
                            None, img_path, contour_image=image,
                            adaptive_gaussian_config=None, visualization_overlay=None,
                            defect_id_method='GroundingSAM', mask_default=None
                        )
                        if sliced_mask is not None:
                            dino_contour_mask = sliced_mask

                    print(f"     [DINOv3] Running inference on SAM mask region...")
                    dino_th_df = cfg_mgr.get_sheet('Dino_TH')
                    current_dino_threshold = 0.5
                    if not dino_th_df.empty:
                        th_row = dino_th_df[dino_th_df['Failure Mode'] == fm]
                        if not th_row.empty:
                            row_dict = th_row.iloc[0]
                            # 兼容各种可能填写的列名: Dino_Low, Threshold, Dino Low, Low
                            for candidate_col in ['Dino_Low', 'Threshold', 'Dino Low', 'Low', 'low']:
                                if candidate_col in row_dict and pd.notna(row_dict[candidate_col]):
                                    try:
                                        current_dino_threshold = float(row_dict[candidate_col])
                                        print(f"     [DINOv3] Successfully loaded threshold from column '{candidate_col}': {current_dino_threshold}")
                                        break
                                    except Exception:
                                        pass
                            
                    dino_input_dim = flow_row.iloc[0].get('Dino_Input_Dim', None)
                            
                    dino_result = None
                    try:
                        dino_result = dinov3_utils.process_single_image_pipeline(
                            image_path=img_path,
                            model=self.dino_model,
                            upsampler=self.upsampler,
                            device=torch.device(device_str),
                            save_dir=reference_dir if debug_mode else res_path, 
                            super_resolution_factor=1.0,
                            contour_shrink_ratio=1.0,
                            transparency_threshold=current_dino_threshold,
                            flooding_enabled=False,
                            flooding_rgb=None,
                            filtering_df=current_filtering_df,
                            adaptive_gaussian_config=None,
                            dino_input_dim=dino_input_dim,
                            contour_image=dino_contour_mask,  
                            post_ops=[],
                            inferred_pic_dir=inferred_pic_dir if debug_mode else res_path
                        )
                    except Exception as e:
                        print(f">>> [DINOv3 Error] Failed to process image {file}: {e}")
                    if dino_result and len(dino_result) >= 5:
                        base_defect_mask = dino_result[4] # binary mask
                    else:
                        base_defect_mask = np.zeros(image.shape[:2], dtype=np.uint8)

                # 5.3 Post Processing (限制在缺陷 Mask 内部执行过滤！)
                if ops_to_run or self.has_dino_suffix:
                    print(f"     [PostProcessing] Applying filters strictly within defect mask: {ops_to_run}")
                    filter_save_dir = reference_dir if debug_mode else None
                    if ops_to_run:
                        filtered_defect_mask, _, _, _, _ = fo.apply_filtering(
                            base_defect_mask, image, ops_to_run, current_filtering_df,
                            filter_save_dir, img_path, contour_image=sam_mask_total,
                            adaptive_gaussian_config=None, visualization_overlay=None,
                            defect_id_method='GroundingSAM', mask_default=None
                        )
                    else:
                        filtered_defect_mask = base_defect_mask
                    
                    # === 真正的 Overlay 输出 (合并到原图上) ===
                    from utils.cv_ops import create_overlay_image
                    
                    # 检查是否有 DINO 生成的真实彩虹渐变热力图 (在 reference 目录下)
                    dino_ref_overlay_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_overlay.png")
                    dino_heatmap_loaded = None
                    if os.path.exists(dino_ref_overlay_path):
                        dino_heatmap_loaded = cv2.imread(dino_ref_overlay_path)
                    
                    if dino_heatmap_loaded is not None and sam_mask_total is not None and np.any(sam_mask_total):
                        if dino_heatmap_loaded.shape[:2] != image.shape[:2]:
                            dino_heatmap_loaded = cv2.resize(dino_heatmap_loaded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                        
                        overlay_display = image.copy()
                        # 将目标区域（SAM Mask 内部）融合展示 DINO 彩虹渐变热力图
                        sam_target_bool = sam_mask_total > 0
                        overlay_display[sam_target_bool] = dino_heatmap_loaded[sam_target_bool]
                        
                        # 如果有检测到的缺陷，画出缺陷外轮廓线（黄色细线），既美观又清晰标明异常区域
                        if filtered_defect_mask is not None and np.any(filtered_defect_mask):
                            defect_mask_bin = (filtered_defect_mask > 0).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(defect_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay_display, contours, -1, (0, 255, 255), 2)
                    elif filtered_defect_mask is not None and np.any(filtered_defect_mask):
                        mask_color = getattr(ug, 'Mask_Color', None)
                        if mask_color is not None:
                            color_rgb = mask_color['rgb']
                            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                            transparency = mask_color['transparency']
                        else:
                            color_bgr = (0, 0, 255) # default red
                            transparency = 0.4 # 半透明，避免纯色死板涂抹
                        overlay_display = create_overlay_image(image, filtered_defect_mask, color=color_bgr, transparency=transparency)
                    else:
                        overlay_display = image.copy()

                    if filtered_defect_mask is not None and np.any(filtered_defect_mask):
                        # --- 生成精确的 Crop 异常距离 CSV (确保坐标系与 1280x1280 空间对齐) ---
                        raw_csv_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_anomaly_distance_raw.csv")
                        if os.path.exists(raw_csv_path):
                            try:
                                # 1. 提取当前 SAM 掩码的 BBox (即送入 DINO 时的裁剪 BBox)
                                bx, by, bw, bh = cv2.boundingRect(sam_mask_total)
                                if max(bw, bh) > 0:
                                    defect_crop = filtered_defect_mask[by:by+bh, bx:bx+bw]
                                    scale = 1280.0 / max(bw, bh)
                                    new_w, new_h = int(bw * scale), int(bh * scale)
                                    defect_resized = cv2.resize(defect_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                                    y_start = (1280 - new_h) // 2
                                    x_start = (1280 - new_w) // 2
                                    mask_1280 = np.zeros((1280, 1280), dtype=np.uint8)
                                    mask_1280[y_start:y_start+new_h, x_start:x_start+new_w] = defect_resized
                                    
                                    # 读取原版 CSV 并通过 Mask 过滤，非缺陷区域全部置 0
                                    raw_dist = np.loadtxt(raw_csv_path, delimiter=",")
                                    crop_dist = np.where(mask_1280 > 0, raw_dist, 0.0)
                                    
                                    crop_csv_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_crop_anomaly_distance_raw.csv")
                                    np.savetxt(crop_csv_path, crop_dist.astype(np.float32), fmt="%.6f", delimiter=",")
                                    print(f"     [Result] Saved precise masked distance map to: {crop_csv_path}")
                            except Exception as e:
                                print(f"     [Warning] Failed to generate crop distance CSV: {e}")

                        # --- 生成 Parametric Output 数据 ---
                        try:
                            if not hasattr(self, 'defect_output_df'):
                                self.defect_output_df = cfg_mgr.get_sheet('Output')
                                print(f'DEBUG KEYS: {cfg_mgr.sheets_cache.keys()}')
                            # 提取 Defect Output Format
                            dof = 'individual'
                            if not self.defect_output_df.empty:
                                dof_row = self.defect_output_df[self.defect_output_df['Failure Mode'].astype(str).str.strip() == fm.strip()]
                                if not dof_row.empty:
                                    dof = str(dof_row.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                            print(f'DEBUG DOFFF 1: {dof}, is empty? {self.defect_output_df.empty}, columns: {self.defect_output_df.columns}')

                            sam_area = float(np.sum(sam_mask_total > 0)) if np.any(sam_mask_total) else 1.0
                            
                            extracted_results = output_ops.extract_parametric_results(
                                defect_mask=filtered_defect_mask,
                                original_img=image,
                                file_name=file,
                                root_dir=res_path, # Pass the Result folder path instead of image source path
                                defect_output_format=dof,
                                dut_area=sam_area,
                                defect_class='Defect',
                                output_config=ug.Output_Config if hasattr(ug, 'Output_Config') else [],
                                gray_scale_params=ug.Gray_Scale_Params if hasattr(ug, 'Gray_Scale_Params') else {},
                                filtering_df=current_filtering_df
                            )
                            # Inject subfolder path for output Excel
                            for res in extracted_results:
                                res['Dir'] = rel_dir if rel_dir else ''
                            parametric_results.extend(extracted_results)
                                
                        except Exception as e:
                            print(f">>> [Strategy Warning] Failed to calculate parametrics: {e}")
                    else:
                        overlay_display = image.copy()
                        parametric_results.append({
                            'Picture_Name': file, 'Defect_Class': 'Pass', 
                            'Dir': root, 'Filename': file
                        })
                        
                    cv2.imwrite(os.path.join(inferred_pic_dir, f"{os.path.splitext(file)[0]}_overlay.jpg"), overlay_display)
                    
                    # 保留原有的黑底图 (但放到 reference 供开发者看)
                    if debug_mode:
                        defect_display = np.zeros_like(image)
                        if filtered_defect_mask is not None and np.any(filtered_defect_mask):
                            defect_display[filtered_defect_mask > 0] = [0, 255, 0]
                        for x1, y1, x2, y2, cls_name, score in boxes_to_draw:
                            cv2.rectangle(defect_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_sam_filtered_defect.jpg"), defect_display)
                    
                processed_count += 1
                
        # --- 最终导出 Parametric Output Excel ---
        if parametric_results:
            if not hasattr(self, 'defect_output_df'):
                self.defect_output_df = cfg_mgr.get_sheet('Output')
                
            # 提取 Defect Output Format
            dof = 'individual'
            if not self.defect_output_df.empty:
                dof_row = self.defect_output_df[self.defect_output_df['Failure Mode'].astype(str).str.strip() == fm.strip()]
                if not dof_row.empty:
                    dof = str(dof_row.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
            print(f'DEBUG DOFFF 2: {dof}')
                    
            df = output_ops.build_parametric_output_df(
                data=parametric_results,
                failure_mode=fm,
                defect_output_format=dof,
            )
            if not df.empty:
                output_ops.parametric_output(
                    main_folder_path=res_path,
                    sub_folder_path="",
                    file_name="Parametric_Output.xlsx",
                    sheet_name='Sheet1',
                    df=df
                )
            print(f">>> [Strategy] Parametric Excel saved successfully with {len(parametric_results)} rows.")
            
        print(f">>> [Strategy] Execution Completed! Processed {processed_count} images.")
