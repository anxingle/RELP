import os
import cv2
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
import utils.utils_general as ug
import utils.filtering_ops as fo
import utils.output_ops as output_ops
import utils.dinov3_utils as dinov3_utils
from core.pipeline import DefectDetectionPipeline

class ComplexTextileStrategy(AnalysisStrategy):
    """
    为纺织品检测量身定制的高级复合视觉策略。
    1. 动态判断长宽比决定产品面(AB长条面 vs CD带按钮面)
    2. 提取Red Tag作为动态打分基准面积
    3. 在CD面识别按钮，过滤异常大框(反套娃)，对按钮周围1.3倍区域进行空间惩罚加权
    4. 动态Slicing切分有效判定区
    5. 使用DINOv3全局高清推断，局部加权算分。
    """
    def execute(self) -> None:
        print(f">>> [ROUTER] Routing '{self.fm}' to Complex Textile Strategy! 🚀")
        
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
            
        print(f">>> [Strategy] Loading Grounding DINO model...")
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        gdino_path = os.path.join(root_dir, "weights", "Grounding Dino")
        gdino_predictor = load_grounding_dino_model(model_id=gdino_path, box_threshold=0.25)
        
        print(f">>> [Strategy] Loading SAM2 model...")
        weights_df = cfg_mgr.get_sheet('Weights')
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
        sam_cfg = weights.get('sam_config', os.path.join(os.getcwd(), "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"))
        sam_ckpt = weights.get('sam_weights', os.path.join(os.getcwd(), "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): device_str = "mps"
        
        sam2_model = ug._real_init_sam2(sam_cfg, sam_ckpt, device=device_str)
        sam2_predictor = ug.get_sam2_predictor(sam2_model)
        
        print(f">>> [Strategy] Loading DINOv3 model for defect detection...")
        dino_weight_path = weights.get('dino') or weights.get('Dino') or weights.get('dinov3_weights')
        dino_model = dinov3_utils.load_model(dino_weight_path, torch.device(device_str))
        upsampler = None # Skip AnyUp for now
        
        # 结果保存路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        base_inferred_pic_dir = os.path.join(res_path, 'Inferred Pic')
        base_reference_dir = os.path.join(res_path, 'reference')
        os.makedirs(base_inferred_pic_dir, exist_ok=True)
        os.makedirs(base_reference_dir, exist_ok=True)
        
        filtering_df = fo.load_and_filter_filtering_sheet("RELP_Configuration.xlsx", self.product, self.generation, fm, None, verbose=False)
        parametric_results = []
        
        for root, dirs, files in os.walk(self.download_path):
            # Resolve relative folder structure to maintain nested directory hierarchy
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
                os.makedirs(reference_dir, exist_ok=True)

            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n{'='*50}\n  -> Processing {file}\n{'='*50}")
                
                image = cv2.imread(img_path)
                if image is None: continue
                img_h, img_w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # --- Step 1: Find Case & Determine Face Type ---
                print(">>> [Step 1] Locating main case object...")
                case_results = gdino_predictor.predict(image, "phone case in the center.")
                if not case_results:
                    print("     [Warning] No case found. Skipping.")
                    continue
                    
                # [Optimization] DINO can sometimes find large background objects (like the fixture).
                # The most robust way to find the main object is to take the box with the HIGHEST CONFIDENCE SCORE,
                # provided it meets the minimum aspect ratio requirement.
                ratio_threshold = 1.0
                valid_cases = []
                
                for res in case_results:
                    bx1, by1, bx2, by2 = map(int, res['bbox'])
                    bw, bh = bx2 - bx1, by2 - by1
                    if bw > 0 and bh > 0:
                        aspect_ratio = max(bw/bh, bh/bw)
                        if aspect_ratio >= ratio_threshold:
                            valid_cases.append(res)
                            
                if not valid_cases:
                    print("     [Warning] No valid case found passing the aspect ratio filter. Skipping.")
                    continue
                    
                # Pick the bounding box with the highest confidence score
                best_case = max(valid_cases, key=lambda x: x.get('score', 0))
                bx1, by1, bx2, by2 = map(int, best_case['bbox'])
                print(f"     [Result] Selected highest confidence box (Score: {best_case.get('score', 0):.3f})")
                
                sam2_predictor.set_image(image_rgb)
                masks, _, _ = sam2_predictor.predict(box=np.array([bx1, by1, bx2, by2]), multimask_output=False)
                case_mask = (masks[0] > 0).astype(np.uint8) * 255 if len(masks[0].shape)==2 else (masks[0][0] > 0).astype(np.uint8) * 255
                case_mask = cv2.resize(case_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # [BUGFIX] Sync with GUI / JSON Engine logic: morphological closing + fill poly
                # to avoid holes in the middle of the case or disconnected artifacts
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                case_mask = cv2.morphologyEx(case_mask, cv2.MORPH_CLOSE, kernel)
                case_mask = cv2.morphologyEx(case_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(case_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    print("     [Warning] SAM2 generated empty mask. Skipping.")
                    continue
                    
                largest_contour = max(contours, key=cv2.contourArea)
                clean_case_mask = np.zeros_like(case_mask)
                cv2.fillPoly(clean_case_mask, [largest_contour], 255)
                case_mask = clean_case_mask
                
                # Recalculate the overall bounding box of the combined mask
                x1, y1, box_w, box_h = cv2.boundingRect(largest_contour)
                x2, y2 = x1 + box_w, y1 + box_h
                ratio = box_w / box_h if box_h > 0 else 0
                
                # Dynamic Routing (Corrected threshold based on logic: AB~7, CD~4)
                # User correction: < 5.5 is AB face, > 5.5 is CD face
                is_cd_face = ratio > 5.5
                
                if is_cd_face:
                    face_name = "CD面 (带按钮)"
                else:
                    face_name = "A面 (无孔长直面)"
                    print(">>> [Step 1.5] Checking for charging hole to determine B face...")
                    hole_results = gdino_predictor.predict(image, "charging hole.")
                    valid_holes = []
                    
                    for res in hole_results:
                        hx1, hy1, hx2, hy2 = map(int, res['bbox'])
                        hcx, hcy = (hx1+hx2)/2, (hy1+hy2)/2
                        # Check if the center of the charging hole is inside the DUT bbox
                        if x1 <= hcx <= x2 and y1 <= hcy <= y2:
                            # Also check area ratio to avoid nested case bounding boxes
                            hole_w = hx2 - hx1
                            hole_h = hy2 - hy1
                            hole_area = hole_w * hole_h
                            dut_area = box_w * box_h
                            if dut_area > 0:
                                area_ratio = hole_area / dut_area
                                print(f"     [Debug] Potential hole found. Area ratio to DUT: {area_ratio*100:.2f}%")
                                if area_ratio < 0.2: # A charging hole shouldn't be larger than 20% of the case side
                                    valid_holes.append((area_ratio, [hx1, hy1, hx2, hy2]))
                                else:
                                    print(f"     [Warning] Dropped a false hole because it is too large ({area_ratio*100:.2f}% of DUT).")
                            
                    if valid_holes:
                        face_name = "B面 (带充电孔)"
                        print(f"     [Result] Found {len(valid_holes)} valid charging hole(s) within DUT bbox. Upgrading to: {face_name}")
                        
                        # Find the hole with the largest area
                        valid_holes.sort(key=lambda x: x[0], reverse=True)
                        largest_hole_box = valid_holes[0][1]
                        smaller_holes = [x[1] for x in valid_holes[1:]]
                        
                        print(f"     [Result] Largest charging hole (Ratio: {valid_holes[0][0]*100:.2f}%) uses Color Distance. {len(smaller_holes)} smaller holes use standard SAM2.")
                        
                        sam2_predictor.set_image(image_rgb)
                        final_hole_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        h_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        
                        # 1. Process Largest Hole with SAM2 + Color Distance
                        h_masks, _, _ = sam2_predictor.predict(box=np.array(largest_hole_box), multimask_output=False)
                        cur_h_mask = (h_masks[0] > 0).astype(np.uint8) * 255 if len(h_masks[0].shape)==2 else (h_masks[0][0] > 0).astype(np.uint8) * 255
                        cur_h_mask = cv2.resize(cur_h_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                        
                        # Apply Color Distance masking strictly inside the largest hole area to isolate the metal part
                        # RGB=[0,0,0], Prox=Near, Tolerance=52 based on user request
                        target_bgr = np.array([0, 0, 0], dtype=np.float32)
                        tolerance = 52.0
                        diff = image.astype(np.float32) - target_bgr
                        dist = np.linalg.norm(diff, axis=2)
                        valid_metal_pixels = (dist <= tolerance)
                        
                        # The final carved out area is ONLY where SAM says it's a hole AND the color matches the metal tolerance
                        metal_hole_mask = (cur_h_mask > 0) & valid_metal_pixels
                        
                        # Smooth and slightly dilate the hole mask to ensure edges are fully covered
                        metal_hole_mask_uint8 = metal_hole_mask.astype(np.uint8) * 255
                        metal_hole_mask_uint8 = cv2.morphologyEx(metal_hole_mask_uint8, cv2.MORPH_CLOSE, h_kernel)
                        metal_hole_mask_uint8 = cv2.dilate(metal_hole_mask_uint8, h_kernel, iterations=2)
                        
                        final_hole_mask = cv2.bitwise_or(final_hole_mask, metal_hole_mask_uint8)
                        
                        # 2. Process Smaller Holes with standard SAM2
                        for s_box in smaller_holes:
                            s_masks, _, _ = sam2_predictor.predict(box=np.array(s_box), multimask_output=False)
                            cur_s_mask = (s_masks[0] > 0).astype(np.uint8) * 255 if len(s_masks[0].shape)==2 else (s_masks[0][0] > 0).astype(np.uint8) * 255
                            cur_s_mask = cv2.resize(cur_s_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                            
                            cur_s_mask = cv2.morphologyEx(cur_s_mask, cv2.MORPH_CLOSE, h_kernel)
                            cur_s_mask = cv2.dilate(cur_s_mask, h_kernel, iterations=2)
                            
                            final_hole_mask = cv2.bitwise_or(final_hole_mask, cur_s_mask)
                        
                        # Mask out all holes from the main case_mask
                        case_mask[final_hole_mask > 0] = 0
                        print("     [Result] All charging hole features successfully carved out from Case Mask.")

                print(f"     [Result] Case W/H Ratio = {ratio:.2f}. Identified as: {face_name}")
                
                # --- Step 2: Dynamic Slicing based on Face Type ---
                slicing_param = "Y[0.0:1.0]"
                print(f">>> [Step 2] Applying full face (no truncation): {slicing_param}")
                slicing_op = [{'name': 'Slicing', 'params': [{'name': 'Y', 'params': [slicing_param[2:-1]], 'combine_mode': None}], 'combine_mode': None}]
                # --- DEBUG VISUALIZATION: Save the Slicing BBox ---
                debug_slice_img = image.copy()
                cv2.rectangle(debug_slice_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 6)
                cv2.putText(debug_slice_img, f"Case BBox (Ratio {ratio:.2f})", (int(x1), int(max(40, y1-20))), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_00_debug_slicing_box.jpg"), debug_slice_img)

                sliced_case_mask, _, _, _, _ = fo.apply_filtering(
                    case_mask, image, slicing_op, filtering_df, None, img_path, contour_image=case_mask
                )
                if sliced_case_mask is None: sliced_case_mask = case_mask

                # --- Step 3: Button Semantic Filtering & Penalty (Only CD Face) ---
                penalty_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                if is_cd_face:
                    print(">>> [Step 3] Detecting Buttons for Spatial Penalty...")
                    btn_results = gdino_predictor.predict(image, "press button on the case.")
                    img_area = img_w * img_h
                    valid_btns = 0
                    
                    # --- 使用原子算子进行防套娃与孤岛去重 ---
                    from utils.advanced_operators import op_isolated_anchor_filter, op_create_spatial_penalty_mask
                    final_bboxes = op_isolated_anchor_filter(
                        raw_results=btn_results, 
                        img_area=img_area, 
                        max_area_pct=0.05
                    )
                    
                    # --- 生成空间惩罚热区 ---
                    penalty_mask = op_create_spatial_penalty_mask(
                        bboxes=final_bboxes, 
                        img_w=img_w, 
                        img_h=img_h, 
                        expand_ratio=1.3
                    )
                    
                    valid_btns = len(final_bboxes)
                    
                    # --- DEBUG VISUALIZATION: Save the final buttons and penalty zones ---
                    debug_btn_img = image.copy()
                    for bx1, by1, bx2, by2, bw, bh in final_bboxes:
                        cv2.rectangle(debug_btn_img, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 4)
                        cv2.putText(debug_btn_img, "Button", (int(bx1), int(max(30, by1-10))), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                        
                        cx, cy = bx1 + bw/2, by1 + bh/2
                        new_w, new_h = bw * 1.3, bh * 1.3
                        nbx1, nby1 = int(cx - new_w/2), int(cy - new_h/2)
                        nbx2, nby2 = int(cx + new_w/2), int(cy + new_h/2)
                        cv2.rectangle(debug_btn_img, (max(0, nbx1), max(0, nby1)), (min(img_w, nbx2), min(img_h, nby2)), (0, 0, 255), 4)
                        cv2.putText(debug_btn_img, "Penalty Zone", (max(0, nbx1), min(img_h-10, nby2+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_01_debug_buttons_and_penalty.jpg"), debug_btn_img)
                    print(f"     [Result] Found {valid_btns} valid, anchor-matched buttons. Penalty zones generated via Operator.")
                else:
                    print(">>> [Step 3] AB face detected. Skipping button search.")

                # --- Step 4: Red Tag Calibration (Dynamic Reference Area) ---
                print(">>> [Step 4] Searching for 'red tag' for Dynamic Area Calibration...")
                tag_results = gdino_predictor.predict(image, "red tag.")
                dut_ref_area = float(np.sum(case_mask > 0)) # Default to case area
                ref_source = "Case BBox Mask"
                img_area = img_w * img_h
                if tag_results:
                    # --- DEBUG VISUALIZATION: Save the Red Tag ---
                    debug_tag_img = image.copy()
                    tx1, ty1, tx2, ty2 = tag_results[0]['bbox']
                    cv2.rectangle(debug_tag_img, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 0, 0), 6)
                    cv2.putText(debug_tag_img, "Red Tag Reference", (int(tx1), int(max(40, ty1-20))), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
                    cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_02_debug_red_tag.jpg"), debug_tag_img)
                    
                    # Take highest score tag
                    best_tag = tag_results[0]
                    tx1, ty1, tx2, ty2 = best_tag['bbox']
                    t_masks, _, _ = sam2_predictor.predict(box=np.array([tx1, ty1, tx2, ty2]), multimask_output=False)
                    tag_mask = (t_masks[0] > 0) if len(t_masks[0].shape)==2 else (t_masks[0][0] > 0)
                    t_area = float(np.sum(tag_mask))
                    # Basic sanity check
                    if t_area > 0 and t_area < img_area * 0.5:
                        dut_ref_area = t_area
                        ref_source = "Red Tag Semantic Mask"
                print(f"     [Result] Calibration source: {ref_source}. Area = {dut_ref_area} px.")

                # --- Step 5: DINOv3 Full Image Inference ---
                print(">>> [Step 5] DINOv3 HD Inspection (1280x1280)...")
                dino_th_df = cfg_mgr.get_sheet('Dino_TH')
                dino_threshold = 0.0 # 修改: 如果没找到设置，默认阈值应为 0
                if not dino_th_df.empty:
                    th_row = dino_th_df[dino_th_df['Failure Mode'] == fm]
                    if not th_row.empty:
                        row_dict = th_row.iloc[0]
                        for candidate_col in ['Dino_Low', 'Threshold', 'Dino Low', 'Low', 'low']:
                            if candidate_col in row_dict and pd.notna(row_dict[candidate_col]):
                                try:
                                    dino_threshold = float(row_dict[candidate_col])
                                    print(f"     [DINOv3] Successfully loaded threshold from column '{candidate_col}': {dino_threshold}")
                                    break
                                except Exception:
                                    pass
                
                # Fetch Dino_Input_Dim from Flow
                flow_df = cfg_mgr.get_sheet('Flow')
                flow_row = flow_df[flow_df['Failure Mode'] == fm]
                dino_dim = flow_row.iloc[0].get('Dino_Input_Dim', '[1280, 1280]') if not flow_row.empty else '[1280, 1280]'
                if pd.isna(dino_dim): dino_dim = '[1280, 1280]'

                dino_result = dinov3_utils.process_single_image_pipeline(
                    image_path=img_path,
                    model=dino_model,
                    upsampler=None,
                    device=torch.device(device_str),
                    save_dir=reference_dir,
                    super_resolution_factor=1.0,
                    contour_shrink_ratio=1.0,
                    transparency_threshold=dino_threshold,
                    flooding_enabled=False,
                    dino_input_dim=dino_dim,
                    # [CRITICAL] 必须传入完整的 case_mask！
                    # 因为 DINO 内部会根据这个 mask 的外接矩形 (BBox) 去裁剪原图。
                    # 如果传入 sliced_case_mask，原图就会被裁成只有一半高，导致送入 DINO 的比例严重失真放大。
                    contour_image=case_mask, 
                    post_ops=[]
                )
                
                # dino_result is a tuple: (heatmap_normalized, heatmap_transparent, heatmap_on_image, rgb_result_paths, binary_mask, mask_removed)
                if isinstance(dino_result, tuple) and len(dino_result) >= 5:
                    dino_heatmap_on_img = dino_result[2] # RGB image with heatmap overlay
                    dino_binary_mask = dino_result[4] # The binary defect mask
                else:
                    print("     [Error] DINOv3 inference failed or returned unexpected format.")
                    continue
                    
                if dino_binary_mask is None:
                    dino_binary_mask = np.zeros_like(sliced_case_mask)
                    
                # Apply Boolean AND with the sliced region to get the final valid defects
                final_defect_mask = cv2.bitwise_and(sliced_case_mask, dino_binary_mask)
                
                # --- NEW: Output Crop Anomaly Distance Raw CSV (1280x1280) ---
                print(">>> [Step 5.5] Generating crop_anomaly_distance_raw.csv...")
                csv_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_anomaly_distance_raw.csv")
                if os.path.exists(csv_path):
                    # 1. 提取当前掩码的 BBox (必须是原本送入 DINO 时的 BBox，即 case_mask 的 BBox)
                    bx, by, bw, bh = cv2.boundingRect(case_mask)
                    # 2. 从最终计算的真实坐标系提取 crop
                    defect_crop = final_defect_mask[by:by+bh, bx:bx+bw]
                    # 3. 按 DINO 逻辑缩放到 1280 容器大小
                    if max(bw, bh) > 0:
                        scale = 1280.0 / max(bw, bh)
                        new_w, new_h = int(bw * scale), int(bh * scale)
                        defect_resized = cv2.resize(defect_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        # 4. Padding 放入 1280x1280 画布
                        y_start = (1280 - new_h) // 2
                        x_start = (1280 - new_w) // 2
                        mask_1280 = np.zeros((1280, 1280), dtype=np.uint8)
                        mask_1280[y_start:y_start+new_h, x_start:x_start+new_w] = defect_resized
                        
                        # 5. 读取原版 CSV 并通过 Mask 过滤，将区域外像素全部归零
                        raw_dist = pd.read_csv(csv_path, header=None).values
                        crop_dist = np.where(mask_1280 > 0, raw_dist, 0.0)
                        
                        # Apply Spatial Penalty directly to the 1280x1280 space
                        if is_cd_face and penalty_mask is not None and np.any(penalty_mask):
                            penalty_crop = penalty_mask[by:by+bh, bx:bx+bw]
                            penalty_resized = cv2.resize(penalty_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            penalty_1280 = np.zeros((1280, 1280), dtype=np.uint8)
                            penalty_1280[y_start:y_start+new_h, x_start:x_start+new_w] = penalty_resized
                            
                            penalty_multiplier_val = 1.0 # 惩罚系数改为 1.0 (保持原始分数)
                            crop_dist = np.where(penalty_1280 > 0, crop_dist * penalty_multiplier_val, crop_dist)
                            print(f"     [Result] Applied Spatial Penalty (Multiplier: {penalty_multiplier_val}) to crop CSV directly.")
                        
                        # 6. 另存为新的 CSV 文件
                        crop_csv_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_crop_anomaly_distance_raw.csv")
                        pd.DataFrame(crop_dist).to_csv(crop_csv_path, index=False, header=False)
                        print(f"     [Result] Saved masked distance map to: {crop_csv_path}")

                # --- Step 6: Parametric Output ---
                print(">>> [Step 6] Compiling Parametric Results with Spatial Penalty...")
                output_df = cfg_mgr.get_sheet('Output')
                dof = 'individual'
                gray_scale_params_from_cfg = {}
                
                if not output_df.empty:
                    dof_row = output_df[output_df['Failure Mode'].astype(str).str.strip() == fm.strip()]
                    if not dof_row.empty:
                        row_data = dof_row.iloc[0]
                        dof = str(row_data.get('Defect Output Format', 'individual')).strip().lower()
                        
                        # Dynamically read Dino_Distance and binning thresholds from Excel
                        mode_val = row_data.get('Color Space Conversion')
                        bin_val = row_data.get('Color Space Binning')
                        if pd.notna(mode_val) and str(mode_val).strip() != '':
                            gray_scale_params_from_cfg = {
                                'Enabled': True,
                                'Mode': str(mode_val).strip(),
                                'Invert': str(row_data.get('Color Space Invertion', 'No')).strip(),
                                'Bining': str(bin_val).strip() if pd.notna(bin_val) else ''
                            }
                
                # Execute output_ops with Penalty Mask!
                extracted_results = output_ops.extract_parametric_results(
                    defect_mask=final_defect_mask, # Evaluate ONLY the real defects inside the valid slice
                    original_img=image,
                    file_name=file,
                    root_dir=res_path,
                    defect_output_format=dof,
                    dut_area=dut_ref_area, # USE DYNAMIC RED TAG AREA HERE
                    defect_class='Defect',
                    output_config=ug.Output_Config if hasattr(ug, 'Output_Config') else [],
                    gray_scale_params=gray_scale_params_from_cfg, # Use the parsed config from Excel
                    penalty_mask=None, # 已经直接在 crop CSV 里预乘了
                    penalty_multiplier=1.0 
                )
                
                if not extracted_results:
                    # 如果没有检测到任何缺陷，插入一条 0 面积/无缺陷的记录，确保两张图都会出现在 Excel 里
                    extracted_results.append({
                        'Picture_Name': file,
                        'Defect_Class': 'Defect',
                        'Area': 0,
                        'Defect Pct': '0.0000%',
                        'Dir': res_path,
                        'Filename': file
                    })
                    
                # 注入相对路径给 Excel 输出
                for res in extracted_results:
                    res['Dir'] = rel_dir if rel_dir else ''

                parametric_results.extend(extracted_results)
                
                # Output Visual Overlay
                # 修复: 从 reference 文件夹读取由 dinov3_utils 完美对齐并恢复坐标系后的 Heatmap 原图
                # 避免 dino_heatmap_on_img 因为内部 crop 导致直接 resize 时出现的错位拉伸和黑边。
                ref_heatmap_path = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_overlay.png")
                
                overlay_display = image.copy()
                
                if os.path.exists(ref_heatmap_path):
                    perfect_heatmap_bgr = cv2.imread(ref_heatmap_path)
                    
                    # 确保尺寸完全一致（理论上通过 restore_coords_func 恢复后是一致的）
                    if perfect_heatmap_bgr is not None and perfect_heatmap_bgr.shape[:2] == overlay_display.shape[:2]:
                        mask_bool = sliced_case_mask > 0
                        # 将有效切片区域替换为对齐后的热力图
                        overlay_display[mask_bool] = perfect_heatmap_bgr[mask_bool]
                        
                        # 如果有检测到的缺陷，画出缺陷外轮廓线（黄色细线），既美观又清晰标明异常区域
                        if final_defect_mask is not None and np.any(final_defect_mask):
                            defect_mask_bin = (final_defect_mask > 0).astype(np.uint8) * 255
                            contours_def, _ = cv2.findContours(defect_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay_display, contours_def, -1, (0, 255, 255), 2)
                    else:
                        print("     [Warning] Reference heatmap missing or shape mismatch. Falling back to simple overlay.")
                        from utils.cv_ops import create_overlay_image
                        overlay_display = create_overlay_image(image, sliced_case_mask, color=(0, 0, 255), transparency=1.0)
                else:
                    print("     [Warning] Reference heatmap not found. Falling back to simple overlay.")
                    from utils.cv_ops import create_overlay_image
                    overlay_display = create_overlay_image(image, sliced_case_mask, color=(0, 0, 255), transparency=1.0)

                # Draw penalty zone on overlay just for debug validation
                if is_cd_face and np.any(penalty_mask):
                    contours, _ = cv2.findContours(penalty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_display, contours, -1, (0, 255, 255), 4) # Yellow outline for Penalty
                
                cv2.imwrite(os.path.join(inferred_pic_dir, f"{os.path.splitext(file)[0]}_overlay.jpg"), overlay_display)
                
        # --- Final Parametric Output Excel ---
        if parametric_results:
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
            print(f">>> [Strategy] Parametric Excel saved successfully.")

