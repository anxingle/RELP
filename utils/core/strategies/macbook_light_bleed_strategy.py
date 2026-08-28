import os
import cv2
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model
import utils.utils_general as ug
import utils.filtering_ops as fo
import utils.output_ops as output_ops
from core.pipeline import DefectDetectionPipeline

class MacbookLightBleedStrategy(AnalysisStrategy):
    """
    针对 Macbook Light Bleed 定制的特定策略。
    1. 使用 Grounding DINO ("white screen") 寻找最大外轮廓作为 DUT 掩膜。
    2. 使用 Grounding DINO ("round object") 寻找参照物并计算面积作为评分基准。
    3. 将 DUT 掩膜传入 Filtering [Stencil-ROI] + [Global Threshold] 以提取真实漏光缺陷。
    4. 执行 Parametric Output.
    """
    def execute(self) -> None:
        print(f">>> [ROUTER] Routing '{self.fm}' to Macbook Light Bleed Strategy! 🚀")
        
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
        gdino_path = os.path.expanduser("~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
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
                os.makedirs(reference_dir, exist_ok=True)
                
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n{'='*50}\n  -> Processing {file}\n{'='*50}")
                
                image = cv2.imread(img_path)
                if image is None: continue
                img_h, img_w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # --- Step 1: Find DUT (white screen) ---
                print(">>> [Step 1] Locating DUT (white screen)...")
                screen_results = gdino_predictor.predict(image, "white screen.")
                if not screen_results:
                    print("     [Warning] No white screen found. Skipping.")
                    continue
                    
                # Take largest box for screen
                best_screen = max(screen_results, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                x1, y1, x2, y2 = best_screen['bbox']
                
                sam2_predictor.set_image(image_rgb)
                masks, _, _ = sam2_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                screen_mask = (masks[0] > 0).astype(np.uint8) * 255 if len(masks[0].shape)==2 else (masks[0][0] > 0).astype(np.uint8) * 255
                screen_mask = cv2.resize(screen_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Find maximum outer contour
                contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)
                    screen_mask = np.zeros_like(screen_mask)
                    cv2.drawContours(screen_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
                
                # --- DEBUG VISUALIZATION: Save Screen Mask ---
                cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_00_debug_screen_mask.jpg"), screen_mask)
                
                # --- Step 2: Find Reference Object (round object) ---
                print(">>> [Step 2] Locating Reference Object (round object)...")
                ref_results = gdino_predictor.predict(image, "round object.")
                dut_ref_area = float(np.sum(screen_mask > 0)) # Default to screen area
                
                if ref_results:
                    best_ref = max(ref_results, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))
                    rx1, ry1, rx2, ry2 = best_ref['bbox']
                    r_masks, _, _ = sam2_predictor.predict(box=np.array([rx1, ry1, rx2, ry2]), multimask_output=False)
                    ref_mask = (r_masks[0] > 0).astype(np.uint8) * 255 if len(r_masks[0].shape)==2 else (r_masks[0][0] > 0).astype(np.uint8) * 255
                    ref_mask = cv2.resize(ref_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    t_area = float(np.sum(ref_mask > 0))
                    if t_area > 0:
                        dut_ref_area = t_area
                        print(f"     [Result] Found round object. Area = {dut_ref_area} px.")
                        cv2.imwrite(os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_01_debug_ref_mask.jpg"), ref_mask)
                else:
                    print("     [Warning] No round object found. Using screen area as reference.")
                
                # --- Step 3: Defect Identification ---
                flow_df = cfg_mgr.get_sheet('Flow')
                flow_row = flow_df[flow_df['Failure Mode'] == fm].iloc[0] if not flow_df.empty else None
                method_str = str(flow_row.get('Defect Identification', '')) if flow_row is not None else ''
                
                print(f">>> [Step 3] Applying Defect Identification (Flow config: {method_str})...")
                
                # Parse operations from flow string
                filtering_ops = ug.parse_defect_id_method(method_str)
                if not filtering_ops:
                    print("     [Warning] No valid operations parsed from Flow. Falling back to Stencil-ROI + Adaptive Gaussian.")
                    filtering_ops = [
                        {'name': 'Stencil-ROI', 'params': [], 'combine_mode': None},
                        {'name': 'Adaptive Gaussian', 'params': [], 'combine_mode': None}
                    ]
                
                # Load Adaptive Gaussian config for this FM
                adaptive_gaussian_df = cfg_mgr.get_sheet('Adaptive Gaussian')
                ag_config = None
                if not adaptive_gaussian_df.empty:
                    from utils.base_utils import _norm
                    ag_rows = adaptive_gaussian_df[adaptive_gaussian_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm)]
                    if not ag_rows.empty:
                        ag_config = ag_rows.iloc[0].to_dict()
                
                final_defect_mask, _, _, _, _ = fo.apply_filtering(
                    screen_mask, image, filtering_ops, filtering_df, reference_dir, img_path, contour_image=screen_mask, adaptive_gaussian_config=ag_config
                )
                
                if final_defect_mask is None:
                    final_defect_mask = np.zeros_like(screen_mask)
                
                # --- Step 4: Parametric Output ---
                print(">>> [Step 4] Compiling Parametric Results...")
                output_df = cfg_mgr.get_sheet('Output')
                dof = 'individual'
                if not output_df.empty:
                    dof_row = output_df[output_df['Failure Mode'].astype(str).str.strip() == fm.strip()]
                    if not dof_row.empty:
                        dof = str(dof_row.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                
                extracted_results = output_ops.extract_parametric_results(
                    defect_mask=final_defect_mask,
                    original_img=image,
                    file_name=file,
                    root_dir=res_path,
                    defect_output_format=dof,
                    dut_area=float(np.sum(screen_mask > 0)), # FOR Defect Pct (absolute area)
                    defect_class='Light Bleed',
                    output_config=ug.Output_Config if hasattr(ug, 'Output_Config') else [],
                    gray_scale_params=ug.Gray_Scale_Params if hasattr(ug, 'Gray_Scale_Params') else {},
                    filtering_df=filtering_df
                )
                
                # We inject our specific reference area logic manually since we are completely pro-code
                # We also fix the Defect Pct bug where output_ops uses bounding box area instead of pixel counts in combined mode
                real_defect_pixels = float(np.sum(final_defect_mask > 0))
                dut_absolute_area = float(np.sum(screen_mask > 0))
                
                for res in extracted_results:
                    # 1. Override the Area and Defect_Area_Pixels to true mask pixels (bypassing contour external rectangle bug)
                    res['Area'] = real_defect_pixels
                    res['Defect_Area_Pixels'] = real_defect_pixels
                    
                    # 2. Re-calculate absolute Defect Pct correctly (Defect Pixels / DUT Mask Pixels)
                    if dut_absolute_area > 0:
                        res['Defect Pct'] = f"{round((real_defect_pixels / dut_absolute_area) * 100, 4)}%"
                        
                    # 3. Re-calculate Defect/Ref% correctly based on round object pixel area
                    res['Reference_BBox_Area'] = dut_ref_area
                    if dut_ref_area > 0:
                        res['Defect/Ref%'] = f"{round((real_defect_pixels / dut_ref_area) * 100, 2)}%"
                        
                    # 4. Inject subfolder path for output Excel
                    res['Dir'] = rel_dir if rel_dir else ''

                parametric_results.extend(extracted_results)
                
                # Output Visual Overlay
                from utils.cv_ops import create_overlay_image
                color_bgr = (0, 0, 255)
                transparency = 1.0
                if getattr(ug, 'Mask_Color', None) is not None:
                    c = ug.Mask_Color['rgb']
                    color_bgr = (c[2], c[1], c[0])
                    transparency = ug.Mask_Color['transparency']
                else:
                    # Fallback to light blue solid mask if config fails
                    color_bgr = (255, 165, 0) # BGR for Light Blue/Cyan
                    transparency = 1.0
                    
                # Force Light Blue (BGR: 255, 165, 0) as requested by user
                color_bgr = (255, 165, 0)
                transparency = 0.8
                    
                # To make tiny defects like Light Bleed visible, let's optionally dilate it slightly for the visualization overlay
                kernel = np.ones((5, 5), np.uint8)
                
                # Convert boolean mask to uint8 before passing to cv2.dilate
                if final_defect_mask.dtype == bool:
                    vis_mask_input = (final_defect_mask.astype(np.uint8) * 255)
                else:
                    vis_mask_input = final_defect_mask

                vis_mask = cv2.dilate(vis_mask_input, kernel, iterations=1)
                
                overlay_display = create_overlay_image(image, vis_mask, color=color_bgr, transparency=transparency)
                cv2.imwrite(os.path.join(inferred_pic_dir, f"{os.path.splitext(file)[0]}_overlay.jpg"), overlay_display)
                
        # --- Final Parametric Output Excel ---
        if parametric_results:
            # Respect the configuration from Output sheet, remove hardcoded override
            if hasattr(ug, 'Scoring_Config') and 'decay' not in [str(s).lower() for s in ug.Scoring_Config]:
                ug.Scoring_Config.append('Decay')
                
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
