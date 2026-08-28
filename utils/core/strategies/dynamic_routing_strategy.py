import os
import cv2
import pandas as pd
import numpy as np
import torch
import json
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
import utils.utils_general as ug
import utils.filtering_ops as fo
import utils.output_ops as output_ops
import utils.dinov3_utils as dinov3_utils
from core.pipeline import DefectDetectionPipeline

class DynamicRoutingStrategy(AnalysisStrategy):
    """
    通用动态路由策略 (Dynamic Routing Strategy)
    通过读取外部 JSON 配置，彻底剥离原有的硬编码：
    1. 动态判断长宽比路由分支
    2. 动态读取提示词 (Prompts) 与切片参数 (Slicing Configs)
    3. 动态配置原子算子 (Atomic Operators) 的权重与阈值
    """
    def _load_json_config(self) -> dict:
        base_dir = os.path.join(os.path.dirname(__file__), 'configs')
        
        # 优先级 1: 精确代际匹配 (e.g., Textile_R692_bubble.json)
        exact_match = f"{self.product}_{self.generation}_{self.fm}.json".replace(" ", "")
        exact_path = os.path.join(base_dir, exact_match)
        
        # 优先级 2: 跨代际泛化匹配 (e.g., Textile_bubble.json)
        fallback_match = f"{self.product}_{self.fm}.json".replace(" ", "")
        fallback_path = os.path.join(base_dir, fallback_match)
        
        if os.path.exists(exact_path):
            print(f">>> [Config] Loaded EXACT match JSON: {exact_match}")
            with open(exact_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif os.path.exists(fallback_path):
            print(f">>> [Config] Exact match not found. Loaded FALLBACK JSON: {fallback_match}")
            with open(fallback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f">>> [Warning] Strategy config not found. Searched for {exact_match} and {fallback_match}. Using fallback defaults.")
            return {}

    def execute(self) -> None:
        print(f">>> [ROUTER] Routing '{self.fm}' to Dynamic Routing Strategy (JSON Configured)! 🚀")
        
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        # --- 0. 加载 JSON 动态配置 ---
        routing_cfg = self._load_json_config()
        
        # 提取配置 (Fallback机制确保稳定性)
        ratio_th = routing_cfg.get('routing', {}).get('aspect_ratio_threshold', 5.5)
        greater_branch = routing_cfg.get('routing', {}).get('greater_than_branch', 'CD_Face')
        
        slicing_cfg = routing_cfg.get('slicing', {})
        
        case_prompt = routing_cfg.get('semantic_targets', {}).get('case_prompt', 'object in the center.')
        button_prompt = routing_cfg.get('semantic_targets', {}).get('button_prompt', 'press button on the case.')
        tag_prompt = routing_cfg.get('semantic_targets', {}).get('calibration_tag_prompt', 'red tag.')
        elliptic_prompt = routing_cfg.get('semantic_targets', {}).get('elliptic_prompt', 'elliptic shape on the case.')
        
        op_max_area = routing_cfg.get('operator_params', {}).get('anchor_filter_max_area_pct', 0.05)
        op_expand_ratio = routing_cfg.get('operator_params', {}).get('penalty_expand_ratio', 1.3)
        op_multiplier = routing_cfg.get('operator_params', {}).get('penalty_multiplier', 2.0)
        
        # 1. 尝试从 JSON 的 infrastructure 块加载基础配置，如果失败则回退到 Excel (cfg_mgr)
        infra_cfg = routing_cfg.get('infrastructure', {})
        
        # 优先读取 JSON
        self.product = infra_cfg.get('product', self.product)
        self.generation = infra_cfg.get('generation', self.generation)
        self.download_path = infra_cfg.get('pic_path', self.download_path)
        
        # Fallback 到 Excel
        if not infra_cfg:
            fm_df = cfg_mgr.get_sheet('Failure Mode')
            fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
            self.product = str(fm_row['Product']) if fm_row is not None else self.product
            self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
            self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
            
        if not os.path.isabs(self.download_path):
            self.download_path = os.path.abspath(os.path.join(os.getcwd(), self.download_path))
            
        print(f">>> [Strategy] Loading Grounding DINO model...")
        gdino_path = os.path.expanduser("~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
        gdino_predictor = load_grounding_dino_model(model_id=gdino_path, box_threshold=0.25)
        
        print(f">>> [Strategy] Loading SAM2 model...")
        if infra_cfg and 'models' in infra_cfg:
            sam_cfg = os.path.join(os.getcwd(), infra_cfg['models'].get('sam2_config', "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"))
            sam_ckpt = os.path.join(os.getcwd(), infra_cfg['models'].get('sam2_checkpoint', "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
        else:
            weights_df = cfg_mgr.get_sheet('Weights')
            weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
            sam_cfg = weights.get('sam_config', os.path.join(os.getcwd(), "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"))
            sam_ckpt = weights.get('sam_weights', os.path.join(os.getcwd(), "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
            
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): device_str = "mps"
        
        sam2_model = ug._real_init_sam2(sam_cfg, sam_ckpt, device=device_str)
        sam2_predictor = ug.get_sam2_predictor(sam2_model)
        
        print(f">>> [Strategy] Loading DINOv3 model for defect detection...")
        if infra_cfg and 'models' in infra_cfg:
            dino_weight_path = os.path.join(os.getcwd(), infra_cfg['models'].get('dino_weights', ""))
        else:
            dino_weight_path = weights.get('dino') or weights.get('Dino') or weights.get('dinov3_weights')
            
        dino_model = dinov3_utils.load_model(dino_weight_path, torch.device(device_str))
        upsampler = None # Skip AnyUp for now
        
        # 结果保存路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        inferred_pic_dir_base = os.path.join(res_path, 'Inferred Pic')
        reference_dir_base = os.path.join(res_path, 'reference')
        os.makedirs(inferred_pic_dir_base, exist_ok=True)
        os.makedirs(reference_dir_base, exist_ok=True)
        
        filtering_df = fo.load_and_filter_filtering_sheet("RELP_Configuration.xlsx", self.product, self.generation, fm, None, verbose=False)
        parametric_results = []
        
        for root, dirs, files in os.walk(self.download_path):
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')]
            
            # --- Handle Relative Path for subfolders (e.g. AB_Face/bottom) ---
            rel_path = os.path.relpath(root, self.download_path)
            if rel_path == '.': rel_path = ''
            
            curr_ref_dir = os.path.join(reference_dir_base, rel_path)
            curr_inf_dir = os.path.join(inferred_pic_dir_base, rel_path)
            os.makedirs(curr_ref_dir, exist_ok=True)
            os.makedirs(curr_inf_dir, exist_ok=True)
            
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n{'='*50}\n  -> Processing {file}\n{'='*50}")
                
                image = cv2.imread(img_path)
                if image is None: continue
                img_h, img_w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # --- Step 1: Find Case & Determine Face Type ---
                print(">>> [Step 1] Locating main case object...")
                case_results = gdino_predictor.predict(image, case_prompt)
                if not case_results:
                    print("     [Warning] No case found. Skipping.")
                    continue
                    
                # Take largest box
                best_case = max(case_results, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                x1, y1, x2, y2 = best_case['bbox']
                case_w, case_h = x2 - x1, y2 - y1
                ratio = case_w / case_h if case_h > 0 else 0
                
                # Dynamic Routing using JSON config
                is_greater_branch = ratio > ratio_th
                face_name = routing_cfg.get('routing', {}).get('greater_than_branch', 'Greater_Branch') if is_greater_branch else routing_cfg.get('routing', {}).get('less_than_branch', 'Less_Branch')
                
                print(f"     [Result] Case W/H Ratio = {ratio:.2f}. Identified as: {face_name}")
                
                # Get Case Mask
                sam2_predictor.set_image(image_rgb)
                masks, _, _ = sam2_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                case_mask = (masks[0] > 0).astype(np.uint8) * 255 if len(masks[0].shape)==2 else (masks[0][0] > 0).astype(np.uint8) * 255
                case_mask = cv2.resize(case_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # --- Step 2: Straighten the Image ---
                print(">>> [Step 2] Correcting tilt angle and applying Square Canvas...")
                from utils.utils_general import DUT_Area_OBB_Cal
                from utils.crop_ops import apply_rotations, crop_and_resize
                
                # 按照标准 dino_strategy 流程：把原物体扣出来，涂黑背景
                extracted_image = image.copy()
                extracted_image[case_mask == 0] = 0 
                
                area, angle, cnt, pts, length, width = DUT_Area_OBB_Cal(case_mask)
                print(f"     [Result] OBB Angle = {angle:.2f}")
                
                # 旋转摆正
                image_rot, case_mask_rot, _ = apply_rotations(extracted_image, case_mask, angle)
                
                # 调用标准 crop_and_resize 制作送给 Dino 的正方形画布 (Square Canvas)
                try:
                    t_size = int(json.loads(dino_dim)[0]) if isinstance(dino_dim, str) else 1280
                except:
                    t_size = 1280
                    
                final_rotated_image, final_rotated_mask, _ = crop_and_resize(
                    image_rot, case_mask_rot, padding=0, target_size=t_size
                )
                
                if final_rotated_image is not None:
                    image = final_rotated_image
                    case_mask = final_rotated_mask
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # update RGB image as well
                img_h, img_w = image.shape[:2] # update dimensions
                print(f"     [Result] Object extracted, rotated, and placed on {t_size}x{t_size} canvas.")
                
                # 必须保存这幅标准的 "切好且摆正的图" 给 Dino
                tmp_dino_input_path = os.path.join(curr_ref_dir, f"{os.path.splitext(file)[0]}_rotated_cropped.jpg")
                cv2.imwrite(tmp_dino_input_path, image)

                # --- Step 3: Dynamic Slicing based on Face Type ---
                slicing_param = slicing_cfg.get(face_name, "Y[0.45:1.0]")
                print(f">>> [Step 3] Applying specific Slicing config for {face_name}: {slicing_param}")
                slicing_op = [{'name': 'Slicing', 'params': [{'name': 'Y', 'params': [slicing_param[2:-1]], 'combine_mode': None}], 'combine_mode': None}]
                # --- DEBUG VISUALIZATION: Save the Slicing BBox ---
                # Recalculate BBox after rotation
                y_idx, x_idx = np.where(case_mask > 0)
                if len(y_idx) > 0 and len(x_idx) > 0:
                    x1, y1 = np.min(x_idx), np.min(y_idx)
                    x2, y2 = np.max(x_idx), np.max(y_idx)
                    case_w, case_h = x2 - x1, y2 - y1
                else:
                    x1, y1, x2, y2 = 0, 0, img_w, img_h
                    case_h = img_h
                
                debug_slice_img = image.copy()
                cv2.rectangle(debug_slice_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 6)
                cv2.putText(debug_slice_img, f"Case BBox (Ratio {ratio:.2f})", (int(x1), int(max(40, y1-20))), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                # Draw the cut line
                slice_ratio = float(slicing_param.split(':')[0][2:]) if ':' in slicing_param else 0.45
                cut_y = int(y1 + case_h * slice_ratio)
                cv2.line(debug_slice_img, (int(x1), cut_y), (int(x2), cut_y), (255, 0, 255), 8)
                cv2.putText(debug_slice_img, f"Slice Line (Y {slice_ratio})", (int(x1), cut_y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
                cv2.imwrite(os.path.join(curr_ref_dir, f"{os.path.splitext(file)[0]}_00_debug_slicing_box.jpg"), debug_slice_img)

                sliced_case_mask, _, _, _, _ = fo.apply_filtering(
                    case_mask, image, slicing_op, filtering_df, None, img_path, contour_image=case_mask
                )
                if sliced_case_mask is None: sliced_case_mask = case_mask

                # --- Step 4: Semantic Filtering & Penalty (Greater Branch) ---
                penalty_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                if face_name == greater_branch:
                    print(f">>> [Step 4] Detecting Buttons for Spatial Penalty using prompt: '{button_prompt}'")
                    btn_results = gdino_predictor.predict(image, button_prompt)
                    img_area = img_w * img_h
                    valid_btns = 0
                    
                    # --- 使用原子算子进行防套娃与孤岛去重 ---
                    from utils.advanced_operators import op_isolated_anchor_filter, op_create_spatial_penalty_mask
                    final_bboxes = op_isolated_anchor_filter(
                        raw_results=btn_results, 
                        img_area=img_area, 
                        max_area_pct=op_max_area
                    )
                    
                    # --- 生成空间惩罚热区 ---
                    penalty_mask = op_create_spatial_penalty_mask(
                        bboxes=final_bboxes, 
                        img_w=img_w, 
                        img_h=img_h, 
                        expand_ratio=op_expand_ratio
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
                        cv2.putText(debug_btn_img, "Penalty x2", (max(0, nbx1), min(img_h-10, nby2+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    cv2.imwrite(os.path.join(curr_ref_dir, f"{os.path.splitext(file)[0]}_01_debug_buttons_and_penalty.jpg"), debug_btn_img)
                    print(f"     [Result] Found {valid_btns} valid, anchor-matched buttons. Penalty zones generated via Operator.")
                else:
                    print(">>> [Step 4] AB face detected. Skipping button search.")
                    # --- Step 5: B-Face Specific Filtering (Less Branch) ---
                    if 'bottom' in root.lower():
                        print(f">>> [Step 5] B-Face ('bottom' folder) detected. Searching for '{elliptic_prompt}' to exclude...")
                        ell_results = gdino_predictor.predict(image, elliptic_prompt)
                        if ell_results and len(ell_results) > 1:
                            # 按照面积从大到小排序，最大的通常是整个 case 本身，我们要排除掉它
                            ell_results = sorted(ell_results, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
                            ell_results.pop(0) # 弹出最大的 box
                            
                            best_ell = ell_results[0] # 获取排在第二的，即目标 elliptic shape
                            ex1, ey1, ex2, ey2 = best_ell['bbox']
                            
                            sam2_predictor.set_image(image_rgb)
                            e_masks, _, _ = sam2_predictor.predict(box=np.array([ex1, ey1, ex2, ey2]), multimask_output=False)
                            elliptic_mask = (e_masks[0] > 0).astype(np.uint8) * 255 if len(e_masks[0].shape)==2 else (e_masks[0][0] > 0).astype(np.uint8) * 255
                            
                            if elliptic_mask.shape[:2] != case_mask.shape[:2]:
                                elliptic_mask = cv2.resize(elliptic_mask, (case_mask.shape[1], case_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            # 从核心评价区（Sliced Mask）以及传给 DINO 的整体轮廓（Case Mask）中彻底挖掉这块区域
                            case_mask[elliptic_mask > 0] = 0
                            sliced_case_mask[elliptic_mask > 0] = 0
                            
                            # 保存 Debug 验证图片
                            debug_ell_img = image.copy()
                            cv2.rectangle(debug_ell_img, (int(ex1), int(ey1)), (int(ex2), int(ey2)), (0, 165, 255), 4)
                            cv2.putText(debug_ell_img, "Excluded Ellipse", (int(ex1), int(max(30, ey1-10))), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
                            cv2.imwrite(os.path.join(curr_ref_dir, f"{os.path.splitext(file)[0]}_01b_debug_elliptic.jpg"), debug_ell_img)
                            print("     [Result] Elliptic shape successfully identified and EXCLUDED from evaluation masks.")
                        else:
                            print("     [Warning] Could not find elliptic shape to exclude (or only found the case itself).")

                # --- Step 6: Red Tag Calibration (Dynamic Reference Area) ---
                print(f">>> [Step 6] Searching for calibration tag using prompt: '{tag_prompt}'")
                tag_results = gdino_predictor.predict(image, tag_prompt)
                dut_ref_area = float(np.sum(case_mask > 0)) # Default to case area
                ref_source = "Case BBox Mask"
                img_area = img_w * img_h
                if tag_results:
                    # --- DEBUG VISUALIZATION: Save the Red Tag ---
                    debug_tag_img = image.copy()
                    tx1, ty1, tx2, ty2 = tag_results[0]['bbox']
                    cv2.rectangle(debug_tag_img, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 0, 0), 6)
                    cv2.putText(debug_tag_img, "Red Tag Reference", (int(tx1), int(max(40, ty1-20))), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 6)
                    cv2.imwrite(os.path.join(curr_ref_dir, f"{os.path.splitext(file)[0]}_02_debug_red_tag.jpg"), debug_tag_img)
                    
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

                # --- Step 7: DINOv3 Full Image Inspection ---
                print(">>> [Step 7] DINOv3 HD Inspection (1280x1280)...")
                if infra_cfg and 'dino_threshold' in infra_cfg:
                    dino_threshold = infra_cfg.get('dino_threshold', 0.5)
                else:
                    dino_th_df = cfg_mgr.get_sheet('Dino_TH')
                    dino_threshold = 0.5
                    if not dino_th_df.empty:
                        th_row = dino_th_df[dino_th_df['Failure Mode'] == fm]
                        if not th_row.empty:
                            try: dino_threshold = float(th_row.iloc[0].get('Threshold', 0.5))
                            except: pass
                
                if infra_cfg and 'dino_input_dim' in infra_cfg:
                    dino_dim = infra_cfg.get('dino_input_dim', '[1280, 1280]')
                else:
                    # Fetch Dino_Input_Dim from Flow
                    flow_df = cfg_mgr.get_sheet('Flow')
                    flow_row = flow_df[flow_df['Failure Mode'] == fm]
                    dino_dim = flow_row.iloc[0].get('Dino_Input_Dim', '[1280, 1280]') if not flow_row.empty else '[1280, 1280]'
                    if pd.isna(dino_dim): dino_dim = '[1280, 1280]'

                # IMPORTANT: Slicing must ONLY apply to scoring, NOT to the DINO image crop/processing.
                # In Step 5, we MUST feed the FULL case mask to DINO to avoid cropping artifacts.
                dino_result = dinov3_utils.process_single_image_pipeline(
                    image_path=tmp_dino_input_path, # 使用标准的正方形画布路径
                    model=dino_model,
                    upsampler=None,
                    device=torch.device(device_str),
                    save_dir=curr_ref_dir,
                    super_resolution_factor=1.0,
                    contour_shrink_ratio=1.0,
                    transparency_threshold=dino_threshold,
                    flooding_enabled=False,
                    dino_input_dim=dino_dim,
                    contour_image=case_mask, 
                    post_ops=[]
                )

                # --- Step 6: Parametric Output ---
                print(">>> [Step 6] Compiling Parametric Results with Spatial Penalty...")
                if infra_cfg:
                    dof = infra_cfg.get('output_format', 'individual')
                else:
                    output_df = cfg_mgr.get_sheet('Output')
                    dof = 'individual'
                    if not output_df.empty:
                        dof_row = output_df[output_df['Failure Mode'].astype(str).str.strip() == fm.strip()]
                        if not dof_row.empty:
                            dof = str(dof_row.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                
                # Execute output_ops with Penalty Mask!
                extracted_results = output_ops.extract_parametric_results(
                    defect_mask=sliced_case_mask, # Evaluate only on the Sliced Mask
                    original_img=image,
                    file_name=file,
                    root_dir=rel_path, # 传入相对路径给 Excel 中的 Dir 字段
                    defect_output_format=dof,
                    dut_area=dut_ref_area, # USE DYNAMIC RED TAG AREA HERE
                    defect_class='Defect',
                    output_config=ug.Output_Config if hasattr(ug, 'Output_Config') else [],
                    gray_scale_params=ug.Gray_Scale_Params if hasattr(ug, 'Gray_Scale_Params') else {},
                    penalty_mask=penalty_mask if face_name == greater_branch else None,
                    penalty_multiplier=op_multiplier # Dynamic multiplier from JSON
                )
                
                # --- Step 7: Inject Human-Aligned Target Score ---
                scoring_cfg = routing_cfg.get('scoring', {})
                if scoring_cfg.get('method') == 'custom_weights':
                    custom_weights = scoring_cfg.get('weights', {})
                    for result in extracted_results:
                        custom_score = 0.0
                        for bin_name, weight in custom_weights.items():
                            custom_score += result.get(bin_name, 0.0) * float(weight)
                        result['AI_Target_Score'] = custom_score
                
                parametric_results.extend(extracted_results)
                
                # Output Visual Overlay
                # Retrieve the generated heatmap
                base_name = os.path.splitext(os.path.basename(tmp_dino_input_path))[0]
                
                # [致命BUG修正]: tmp_dino_input_path 的名字是 _rotated_cropped.jpg
                # 但是 Dino 内部由于种种后缀清理逻辑 (replace('_cropped', '') 等)，
                # 它最终保存下来的文件名竟然是 _rotated_overlay.png，而不是 _rotated_cropped_overlay.png！
                # 这种硬编码的后缀替换导致路径寻找失败。我们需要重新匹配实际生成的名字。
                dino_overlay_basename = base_name.replace('_cropped', '')
                dino_overlay_path = os.path.join(curr_ref_dir, f"{dino_overlay_basename}_overlay.png")
                
                # 重要：使用系统的统一接口输出最终的 Inferred Pic 结果 (仅保存单色掩码，供基础验证)
                import utils.cv_ops as cv_ops
                cv_ops.save_inferred_pic_overlay(
                    base_img=image, 
                    mask=sliced_case_mask, 
                    out_dir=curr_inf_dir, 
                    filename_base=os.path.splitext(file)[0], 
                    input_is_bgr=True
                )
                
                # 额外且必须：在 inferred_pic 里面保存一张基于真实 Heatmap 切片的终极叠加图！
                if os.path.exists(dino_overlay_path):
                    full_dino_overlay = cv2.imread(dino_overlay_path)
                    if full_dino_overlay is not None:
                        if full_dino_overlay.shape[:2] != image.shape[:2]:
                            full_dino_overlay = cv2.resize(full_dino_overlay, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                        
                        if sliced_case_mask.shape[:2] != image.shape[:2]:
                            sliced_case_mask = cv2.resize(sliced_case_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                        # 这次我们要用切片(sliced)后的 heatmap 覆盖原图，并且**直接覆盖掉**原来的 _overlay.jpg
                        # 因为复判人员需要看的就是带热力颜色的结果，而不是那张蓝色的图
                        overlay_display = image.copy()
                        mask_bool = sliced_case_mask > 0
                        
                        # OpenCV 读进来的 full_dino_overlay 是 BGR，直接覆盖给 BGR 的 overlay_display
                        overlay_display[mask_bool] = full_dino_overlay[mask_bool]
                        
                        # 额外画上 Penalty 框作为参考
                        if face_name == greater_branch and np.any(penalty_mask):
                            contours, _ = cv2.findContours(penalty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay_display, contours, -1, (0, 255, 255), 4)
                            
                        # 覆盖写入 _overlay.jpg，强制让终端展示的结果是包含热力图的！
                        final_overlay_path = os.path.join(curr_inf_dir, f"{os.path.splitext(file)[0]}_overlay.jpg")
                        cv2.imwrite(final_overlay_path, overlay_display)
                        print(f">>> [Strategy] SUCCESS! Sliced DINO Heatmap overlay perfectly saved as {final_overlay_path}")
                else:
                    print(f">>> [Strategy] ERROR! Could not find DINO overlay at {dino_overlay_path}")

                
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

