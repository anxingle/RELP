import os
import cv2
import pandas as pd
import numpy as np
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils import utils_general
from utils import dinov3_utils
from utils import io_ops
from utils import output_ops

class DinoStrategy(AnalysisStrategy):
    """
    基于 Dino 模型的高级缺陷分析策略。
    完全重构版本：使用 6 步搭积木法，摆脱 9000 行面条代码。
    """
    
    def _resolve_models(self, cfg_mgr):
        import torch
        device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device_str)
        print(f">>> [DinoStrategy] Using device: {self.device}")

        fm = self.fm
        product = self.product
        generation = self.generation
        weights_df = cfg_mgr.get_sheet('Weights')
        flow_cfg = cfg_mgr.get_flow(fm)
        
        from utils.core.pipeline import DefectDetectionPipeline
        pipeline = DefectDetectionPipeline()
        weights = pipeline._resolve_model_weights(fm, product, generation, weights_df)
        
        cfg_path_dut = weights.get('cfg_path_dut')
        weights_path_dut = weights.get('weights_path_dut')
        
        utils_general.cfg_path_dut = cfg_path_dut
        utils_general.weights_path_dut = weights_path_dut
        
        self.dut_predictor = None
        if cfg_path_dut and weights_path_dut:
            print(f">>> [DinoStrategy] Loading DUT model from {weights_path_dut}")
            self.dut_predictor = utils_general.load_detectron2_model(cfg_path_dut, weights_path_dut, score_thresh=0.5)
            
        self.context['general_defect_key'] = None
        self.context['general_defect_save_vis'] = False
        self.context['defect_detection_debug'] = False
        
        gdd_val = getattr(flow_cfg, 'defect_detection_setting', None)
        if gdd_val is None:
            try:
                for k in flow_cfg.keys():
                    if k is not None and str(k).lower().replace(' ', '').replace('_', '') in ('generaldefectdetection', 'defectdetectionsetting'):
                        gdd_val = flow_cfg[k]
                        break
            except: pass
        if pd.notna(gdd_val) and str(gdd_val).strip() != '' and str(gdd_val).lower() != 'nan':
            raw_val = str(gdd_val).strip()
            if '[Save]' in raw_val or '[save]' in raw_val:
                self.context['general_defect_save_vis'] = True
                raw_val = str(raw_val) if raw_val is not None else ''
                raw_val = raw_val.replace('[Save]', '').replace('[save]', '').strip()
            self.context['general_defect_key'] = raw_val
            
        ddd_val = getattr(flow_cfg, 'defect_detection_debug', None)
        if ddd_val is None:
            try:
                for k in flow_cfg.keys():
                    if k is not None and str(k).lower().replace(' ', '').replace('_', '') == 'defectdetectiondebug':
                        ddd_val = flow_cfg[k]
                        break
            except: pass
        if pd.notna(ddd_val) and str(ddd_val).strip() != '' and str(ddd_val).lower() != 'nan':
            if str(ddd_val).strip().lower() in ('yes', 'true', '1'):
                self.context['defect_detection_debug'] = True

        self.general_defect_predictor = None
        gdd_key = self.context.get('general_defect_key')
        gd_cfg = None
        gd_weight = None
        
        if gdd_key:
            if not weights_df.empty:
                from utils.base_utils import _norm
                type_col = next((c for c in weights_df.columns if _norm(c) == 'type'), None)
                var_col = next((c for c in weights_df.columns if _norm(c) in ('variablename', 'variable_name')), None)
                path_col = next((c for c in weights_df.columns if _norm(c) in ('relativepath', 'relative_path')), None)
                if type_col and var_col and path_col:
                    matches = weights_df[weights_df[type_col].astype(str).apply(_norm) == _norm(gdd_key)]
                    for _, row in matches.iterrows():
                        var_name = _norm(row[var_col])
                        path_val = row[path_col]
                        if pd.notna(path_val):
                            if not os.path.isabs(path_val):
                                project_root = os.getcwd()
                                path_val = os.path.join(project_root, path_val)
                            if 'config' in var_name or 'cfg' in var_name:
                                gd_cfg = path_val
                            elif 'weight' in var_name:
                                gd_weight = path_val

        if not gd_cfg and not gd_weight:
            g_key = self.context.get('general_defect_key')
            if not g_key: g_key = 'General'
            general_weights = pipeline._resolve_model_weights(g_key, product, generation, weights_df)
            if not general_weights:
                general_weights = pipeline._resolve_model_weights('General', product, generation, weights_df)
            gd_cfg = general_weights.get('cfg_path') or general_weights.get('cfg_path_general')
            gd_weight = general_weights.get('weights_path') or general_weights.get('weights_path_general')
        if not gd_cfg and not gd_weight:
            crack_weights = pipeline._resolve_model_weights('Crack', product, generation, weights_df)
            gd_cfg = crack_weights.get('cfg_path_crack') or crack_weights.get('cfg_path')
            gd_weight = crack_weights.get('weights_path_crack') or crack_weights.get('weights_path')
            
        if gd_cfg and gd_weight and os.path.exists(gd_cfg) and os.path.exists(gd_weight):
            print(f">>> [DinoStrategy] Loading General Defect model from {gd_weight}")
            self.general_defect_predictor = utils_general.load_detectron2_model(gd_cfg, gd_weight, score_thresh=0.5)
            if hasattr(self.general_defect_predictor, 'cfg'):
                self.general_defect_predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        else:
            if self.context.get('general_defect_key'):
                print(f">>> [DinoStrategy] No specific General Defect weights found. Falling back to DUT.")
                self.general_defect_predictor = self.dut_predictor


        self.sam2_predictor = None
        sam_mode = 'Regular'
        cfg_mgr_ref = cfg_mgr if 'cfg_mgr' in locals() else self.cm
        fm_df = cfg_mgr_ref.get_sheet('Failure Mode')
        if not fm_df.empty:
            from utils.base_utils import _norm
            fm_matches = fm_df[fm_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm)]
            if not fm_matches.empty:
                fm_row = fm_matches.iloc[0]
                sm_cols = [c for c in fm_df.columns if _norm(c) in ('sammode', 'sam_mode')]
                if sm_cols and pd.notna(fm_row[sm_cols[0]]):
                    sam_mode = str(fm_row[sm_cols[0]]).strip()
        
        print(f">>> [Strategy] Resolved SAM Mode: {sam_mode}")
        
        sam_ckpt = getattr(utils_general, 'sam_checkpoint_path', None)
        sam_cfg = getattr(utils_general, 'sam_config_path', None)
        
        weights_df_ref = weights_df if 'weights_df' in locals() else self.cm.get_sheet('Weights')
        
        if not sam_ckpt or not sam_cfg:
            from utils.base_utils import _norm
            type_col = next((c for c in weights_df_ref.columns if _norm(c) in ('type', 'model_type')), None)
            vn_col = next((c for c in weights_df_ref.columns if _norm(c) in ('variablename', 'variable_name')), None)
            path_col = next((c for c in weights_df_ref.columns if _norm(c) in ('relativepath', 'relative_path')), None)
            
            sam_rows = weights_df_ref[weights_df_ref[weights_df_ref.columns[0]].astype(str).apply(_norm).isin(['sam', 'sam2'])]
            if not sam_rows.empty and type_col and vn_col and path_col:
                target_mode = sam_mode.lower()
                def is_onnx_row(r):
                    t_val = _norm(r.get(type_col, ''))
                    vn_val = _norm(r.get(vn_col, ''))
                    p_val = str(r.get(path_col, '')).lower()
                    return 'onnx' in t_val or 'onnx' in vn_val or 'onnx' in p_val

                filtered_sam = sam_rows[sam_rows.apply(is_onnx_row, axis=1)] if target_mode == 'onnx' else sam_rows[~sam_rows.apply(is_onnx_row, axis=1)]
                
                for _, row in filtered_sam.iterrows():
                    vn = _norm(row.get(vn_col))
                    path_val = row.get(path_col)
                    if pd.notna(path_val):
                        resolved_p = path_val if os.path.isabs(path_val) else os.path.join(os.getcwd(), path_val)
                        if 'checkpoint' in vn or 'weight' in vn or 'pt' in vn or 'onnx' in vn:
                            sam_ckpt = resolved_p
                        elif 'config' in vn or 'yaml' in vn:
                            sam_cfg = resolved_p

        try:
            if getattr(utils_general, 'sam2_model', None) is not None:
                sam2_model = utils_general.sam2_model
                if hasattr(sam2_model, 'predict'):
                    self.sam2_predictor = sam2_model
                else:
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print(f">>> [Strategy] Reusing globally initialized SAM2 model")
            elif sam_ckpt:
                import torch
                device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                utils_general.init_sam2(sam_cfg, sam_ckpt, device=device_str)
                sam2_model = utils_general.sam2_model
                if hasattr(sam2_model, 'predict'):
                    self.sam2_predictor = sam2_model
                else:
                    from sam2.sam2_image_predictor import SAM2ImagePredictor
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print(f">>> [Strategy] Loaded SAM2 model")
        except Exception as e:
            print(f">>> [Strategy] Failed to load SAM2: {e}")

        dino_weight_path = weights.get('dino') or weights.get('Dino') or weights.get('dinov3_weights')
        self.dino_model = None
        self.upsampler = None
        if dino_weight_path:
            try:
                print(f">>> [DinoStrategy] Loading Dino model from {dino_weight_path}")
                self.dino_model = dinov3_utils.load_model(dino_weight_path, self.device)
            except Exception as e:
                print(f">>> [DinoStrategy] Failed to load Dino model: {e}")
                
        use_anyup = str(getattr(flow_cfg, 'upsampling', '')).strip().lower() in ('yes', 'true', '1') if flow_cfg else False
        if use_anyup:
             try:
                 print(f">>> [DinoStrategy] Loading AnyUp upsampler...")
                 self.upsampler = dinov3_utils.load_anyup_model(self.device)
             except Exception as e:
                 pass
                 
        return flow_cfg

    def execute(self) -> None:
        print(">>> [Strategy] Executing DinoStrategy (Modern Pipeline)...")
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        from utils.base_utils import _norm
        fm_matches = fm_df[fm_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)]
        fm_row = fm_matches.iloc[0] if not fm_matches.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        output_path_type = str(fm_row['Output_Path']) if fm_row is not None else 'Result'
        if output_path_type == 'Result':
            self.res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        else:
            self.res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        flow_cfg = self._resolve_models(cfg_mgr)
        if not flow_cfg: return
            
        debug_mode = str(flow_cfg.debug_mode).strip().lower() in ('yes', 'true', '1') if flow_cfg.debug_mode else False
        self.context['is_debug_mode'] = debug_mode
        
        pca_val = getattr(flow_cfg, 'pca_alignment', 'No')
        if pca_val is None or str(pca_val).lower() == 'nan': pca_val = 'No'
        self.context['use_batch_alignment'] = str(pca_val).strip().lower() == 'yes'
        
        utils_general.Output_Config = ['Length']
        utils_general.Reference_Params = {'Length': {'R': 200.0}}
        utils_general.Gray_Scale_Params = {'Enabled': True, 'Mode': 'Gray Scale', 'Invert': 'Yes', 'Bining': '[0, 50, 90, 120, 150, 180, 210, 255]'}
        
        scaling_df = cfg_mgr.get_sheet('Scaling')
        utils_general.DUT_Scaling_Config = {}
        utils_general.Feature_Scaling_Config = {}
        
        import utils.filtering_ops as fo
        excel_path = "RELP_Configuration.xlsx" 
        filtering_df = fo.load_and_filter_filtering_sheet(excel_path, self.product, self.generation, fm, None, verbose=False)

        adaptive_gaussian_df = cfg_mgr.get_sheet('Adaptive Gaussian')
        current_adaptive_gaussian_config = None
        if not adaptive_gaussian_df.empty:
            ag_rows = adaptive_gaussian_df[adaptive_gaussian_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)]
            if not ag_rows.empty:
                if self.product:
                    prod_col = next((c for c in ag_rows.columns if _norm(c) == 'product'), None)
                    if prod_col:
                        prod_match = ag_rows[ag_rows[prod_col].astype(str).apply(_norm) == _norm(self.product)]
                        if not prod_match.empty: ag_rows = prod_match
                if self.generation:
                    gen_col = next((c for c in ag_rows.columns if _norm(c) == 'generation'), None)
                    if gen_col:
                        gen_match = ag_rows[ag_rows[gen_col].astype(str).apply(_norm) == _norm(self.generation)]
                        if not gen_match.empty: ag_rows = gen_match
                if not ag_rows.empty:
                    current_adaptive_gaussian_config = ag_rows.iloc[0].to_dict()
                    print(f">>> [DinoStrategy] Loaded Adaptive Gaussian config: {current_adaptive_gaussian_config}")
                
        dino_th_df = cfg_mgr.get_sheet('Dino_TH')
        current_dino_threshold = None
        flooding_enabled = False
        flooding_rgb = None
        if not dino_th_df.empty:
            th_rows = dino_th_df[(dino_th_df['Product'] == self.product) & (dino_th_df['Failure Mode'] == self.fm)]
            if not th_rows.empty:
                low = th_rows.iloc[0].get('Dino_Low', None)
                high = th_rows.iloc[0].get('Dino_High', None)
                low = float(low) if pd.notna(low) and str(low).strip() != '' else None
                high = float(high) if pd.notna(high) and str(high).strip() != '' else None
                if low is not None or high is not None:
                    current_dino_threshold = (low, high)
                    print(f">>> [DinoStrategy] Found Dino Threshold: {current_dino_threshold}")
                
                flood_str = str(th_rows.iloc[0].get('Flooding by TH', '')).strip().lower()
                if flood_str in ('yes', 'true', '1'):
                    flooding_enabled = True
                    flood_color = str(th_rows.iloc[0].get('Flooding RGB', '')).strip()
                    if flood_color:
                        try:
                            r, g, b = [int(x.strip()) for x in flood_color.split(',')]
                            flooding_rgb = (r, g, b)
                        except ValueError:
                            flooding_rgb = (0, 0, 0)
                    else:
                        flooding_rgb = (0, 0, 0)
                    print(f">>> [DinoStrategy] Flooding Enabled with RGB: {flooding_rgb}")
        
        download_path = os.path.join(project_root, self.download_path) if not os.path.isabs(self.download_path) else self.download_path
        res_path = self.res_path
        if not os.path.isabs(res_path): res_path = os.path.join(project_root, res_path)
             
        reference_dir = os.path.join(res_path, "reference") if debug_mode else res_path
        os.makedirs(reference_dir, exist_ok=True)
        inferred_pic_dir = os.path.join(res_path, "Inferred Pic") if debug_mode else res_path
        os.makedirs(inferred_pic_dir, exist_ok=True)

        if not os.path.exists(download_path): return
        
        parametric_results = []
        extracted_masks_dict = {}
        return_masks_only = self.context.get('return_masks_only', False)
        
        file_list_override = self.context.get('file_list')
        if file_list_override:
            print(f">>> [Strategy] Using provided file list ({len(file_list_override)} files).")
            files_to_process = file_list_override
        else:
            files_to_process = []
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')) and not file.startswith('.'):
                        files_to_process.append(os.path.join(root, file))
        
        # [CRITICAL PARAMS]: Extract Dino_Input_Dim from flow config early
        dino_input_dim_val = getattr(flow_cfg, 'dino_input_dim', None)
        dino_input_dim = None
        if dino_input_dim_val is not None and str(dino_input_dim_val).strip().lower() != 'nan':
            import ast
            try:
                dino_input_dim = ast.literal_eval(str(dino_input_dim_val))
            except Exception:
                dino_input_dim = str(dino_input_dim_val)
        self.context['dino_input_dim'] = dino_input_dim
        
        for image_path in files_to_process:
            file = os.path.basename(image_path)
            print(f"\n>>> [Strategy] Processing {file}...")
            
            # --- Handle Inherited Data ---
            inherited_data = self.context.get('inherited_data', {}).get(image_path)
            bbox_general_FM = None
            detected_labels = []
            
            suffix = os.path.splitext(file)[1]
            if suffix.lower() == '.heic': suffix = '.jpg'
            if debug_mode:
                path_to_process = os.path.join(reference_dir, f"{os.path.splitext(file)[0]}_rotated90_cropped{suffix}")
            else:
                path_to_process = os.path.join(res_path, f"temp_{os.path.splitext(file)[0]}{suffix}")
            
            if inherited_data:
                print(f">>> [Strategy] Using inherited mask and image from previous stage.")
                image = inherited_data.get('original_image')
                DUT_Corrected = inherited_data.get('dut')
                Contour_Corrected = inherited_data.get('contour')
                alignment_metadata = inherited_data.get('alignment')
                final_defect_mask = inherited_data.get('mask')
                dino_result = [None, None, None, None, final_defect_mask] if final_defect_mask is not None else None
            else:
                # [积木块 1 & 2 & 3 & 4]: 通用前置准备 (拿原图 -> 找框 -> SAM抠图 -> 对齐裁剪)
                image, DUT_Corrected, Contour_Corrected, alignment_metadata = self.prepare_dut_image(
                    image_path, self.dut_predictor, self.sam2_predictor
                )
                
                if DUT_Corrected is None:
                    print(f">>> [Strategy] Skipping {file}: DUT alignment failed.")
                    continue
                    
                # 无论是否 debug，我们必须把这幅 "切好且摆正的图" 存到磁盘。
                # 这是因为 dinov3_utils 是强耦合于文件路径和内部二次读图的。
                cv2.imwrite(path_to_process, DUT_Corrected)
    
                # [积木块 5]: 送给 Dino 进行异常推断
                if not self.dino_model:
                     print(f">>> [Strategy] No Dino model loaded, skipping inference.")
                     continue
                     
                print(f">>> [Strategy] Running Dino Inference on {file}...")
                contour_shrink_ratio = getattr(flow_cfg, 'contour_shrink_ratio', 1.0) or 1.0
                
                # 这一步极其关键！
                # 我们告诉 Dino: 你的输入图是 path_to_process，它的尺寸就是 DUT_Corrected 的尺寸。
                # 你的目标就是在这张图上找缺陷，而且绝对不要越过 Contour_Corrected 这个边界！
                dino_result = dinov3_utils.process_single_image_pipeline(
                    image_path=path_to_process, # Use the aligned/cropped image (DUT_Corrected) to match Contour_Corrected!
                    model=self.dino_model,
                    upsampler=self.upsampler,
                    device=self.device,
                    save_dir=reference_dir if debug_mode else res_path, 
                    super_resolution_factor=1.0,
                    contour_shrink_ratio=contour_shrink_ratio,
                    transparency_threshold=current_dino_threshold,
                    flooding_enabled=flooding_enabled,
                    flooding_rgb=flooding_rgb,
                    filtering_df=filtering_df,
                    adaptive_gaussian_config=current_adaptive_gaussian_config,
                    dino_input_dim=dino_input_dim,
                    contour_image=Contour_Corrected,  post_ops=[],
                    inferred_pic_dir=None
                )
                
                # [EcoRel / Defect Detection Setting] -> General Defect Detection Fallback
                general_defect_key = self.context.get('general_defect_key')
                if general_defect_key:
                    if self.general_defect_predictor is None:
                        print(f">>> [Strategy] Warning: General Defect Key '{general_defect_key}' is set but no predictor was loaded!")
                    else:
                        print(f">>> [Strategy] Running General Defect Detection (EcoRel fallback) for {file}...")
                        
                        save_vis = self.context.get('general_defect_save_vis') or self.context.get('defect_detection_debug')
                        gd_fname = f"{os.path.splitext(file)[0]}_FM_detected.jpg"
                        gd_path = os.path.join(reference_dir, gd_fname) if save_vis else None
                        
                        # Unpack properly
                        bbox_general_FM = utils_general.perform_general_defect_detection(
                            DUT_Corrected,
                            self.general_defect_predictor,
                            gd_path,
                            save_visualization=save_vis
                        )
                        print(f">>> [Strategy] Found {len(bbox_general_FM) if bbox_general_FM else 0} general defect bboxes")
    
                
                # [积木块 6]: 尺寸测量与 Excel 参数组装
                final_defect_mask = None
                if dino_result is not None and len(dino_result) >= 5:
                        final_defect_mask = dino_result[4] # binary_mask
                        
                # ---------------------------------------------------------------------
                # [NEW: General FM Scope Filtering (The physical crop after Dino)]
                # ---------------------------------------------------------------------
                matched_bboxes = []
                gfm_rules = self.context.get('general_fm_rules')
                
                if gfm_rules and final_defect_mask is not None:
                    from utils.general_fm_ops import apply_general_fm_scope
                    print(f">>> [Strategy] Applying General FM Scope Rules: {gfm_rules}")
                    
                    final_defect_mask, detected_labels, matched_bboxes = apply_general_fm_scope(
                        mask=final_defect_mask,
                        image=DUT_Corrected,
                        detector=self.general_defect_predictor or self.dut_predictor,
                        gfm_rules=gfm_rules
                    )
                    
                    # Update visualization bboxes for later rendering
                    bbox_general_FM = matched_bboxes if matched_bboxes else bbox_general_FM
                # ---------------------------------------------------------------------
                
            # If we are only extracting masks (Stage 1), save and continue
            if return_masks_only:
                extracted_masks_dict[image_path] = {
                    'original_image': image,
                    'dut': DUT_Corrected,
                    'contour': Contour_Corrected,
                    'alignment': alignment_metadata,
                    'mask': final_defect_mask
                }
                # Cleanup temp file
                if not debug_mode and not inherited_data:
                    try:
                        if os.path.exists(path_to_process): os.remove(path_to_process)
                    except: pass
                continue
                
            # Now handle +Filtering operations parsing from Defect Identification or Post Processing!
            ops_to_run = []
            
            # Check Defect Identification for "+Filtering[xxx]"
            method_col = next((c for c in fm_df.columns if _norm(c) in ('defectidentification', 'defect_identification')), None)
            ident_method = str(fm_row[method_col]).strip() if fm_row is not None and method_col else ''
            if '+filtering' in ident_method.lower():
                # Extract the filtering parts
                ident_parts = ident_method.split('+')
                for part in ident_parts:
                    if 'filtering[' in part.lower():
                        ops_to_run.extend(utils_general.parse_defect_id_method(part))
            
            # Check Post Processing
            post_method = getattr(flow_cfg, 'post_processing', '')
            if post_method and str(post_method).lower() != 'nan':
                ops_to_run.extend(utils_general.parse_defect_id_method(str(post_method)))
                
            if ops_to_run and final_defect_mask is not None and np.any(final_defect_mask):
                print(f">>> [Strategy] Applying additional filter ops on final mask: {ops_to_run}")
                import utils.filtering_ops as fo
                
                # Needs to be saved temporarily for filtering_ops
                if inherited_data:
                    cv2.imwrite(path_to_process, DUT_Corrected)
                    
                final_defect_mask, _, _, _, _ = fo.apply_filtering(
                    final_defect_mask, DUT_Corrected, ops_to_run, filtering_df,
                    reference_dir, path_to_process, contour_image=Contour_Corrected,
                    adaptive_gaussian_config=current_adaptive_gaussian_config,
                    visualization_overlay=None,
                    defect_id_method='Dino',
                    mask_default=Contour_Corrected
                )
                        
            if final_defect_mask is not None and np.any(final_defect_mask):
                    print(f">>> [Strategy] Defect found! Calculating dimensions...")
                    try:
                        mask_uint8 = (final_defect_mask.astype(np.uint8) * 255) if final_defect_mask.dtype == bool else final_defect_mask
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        debug_iou_data = [] # Store data for debug_iou image
                        
                        for idx, contour in enumerate(contours):
                            contour_area = cv2.contourArea(contour)
                            if contour_area < 5: # Filter out absolute noise
                                continue
                            
                            single_mask = np.zeros_like(mask_uint8)
                            cv2.drawContours(single_mask, [contour], -1, 255, -1)
                            
                            dims_res_single = output_ops.calculate_parametric_dimensions(
                            defect_cnt=contour,
                            dut_dims=DUT_Corrected.shape[:2],
                                ref_params=utils_general.Reference_Params,
                                output_config=utils_general.Output_Config,
                                contour_area_val=contour_area,
                                ref_area=1.0,
                                ref_found=True,
                                defect_mask=single_mask,
                                is_line_shape=True,
                                image=DUT_Corrected,
                                gray_scale_params=utils_general.Gray_Scale_Params,
                                output_dir=res_path,
                                image_filename=path_to_process,
                                detector=self.dut_predictor
                            )
                                
                            p_dict = {'Picture_Name': file, 'Defect_Class': 'Crack'}
                            
                            # Add 'Detected Defect' from EcoRel bounding boxes
                            print(f"DEBUG: Checking bbox_general_FM: {bool(bbox_general_FM)}")
                            current_defect_mode = None
                            
                            # Use labels from the new Scope Filter if available
                            if detected_labels:
                                current_defect_mode = ", ".join(detected_labels)
                                print(f"DEBUG: Selected '{current_defect_mode}' as Detected Defect from Scope Filter.")
                            elif bbox_general_FM:
                                dx, dy, dw, dh = cv2.boundingRect(contour)
                                cnt_area = dw * dh
                                if cnt_area > 0:
                                    best_overlap = 0
                                    for gdd in bbox_general_FM:
                                        if isinstance(gdd, dict):
                                            bbox = gdd['bbox']
                                            cls_name = gdd['class_name']
                                        else:
                                            bbox = gdd
                                            cls_name = "Component"
                                            
                                        if str(cls_name) == "0":
                                            cls_name = "Body"
                                            
                                        gx1, gy1, gx2, gy2 = map(int, bbox)
                                        
                                        # Simple intersection
                                        ix1 = max(dx, gx1)
                                        iy1 = max(dy, gy1)
                                        ix2 = min(dx + dw, gx2)
                                        iy2 = min(dy + dh, gy2)
                                        
                                        inter_w = max(0, ix2 - ix1)
                                        inter_h = max(0, iy2 - iy1)
                                        inter_area = inter_w * inter_h
                                        
                                        overlap_ratio = inter_area / cnt_area
                                        # DEBUG
                                        print(f"DEBUG(IoU): Crack bounding box {dx},{dy},{dw},{dh} overlaps with {cls_name} box {gx1},{gy1},{gx2},{gy2}. Ratio: {overlap_ratio:.3f}")
                                        
                                        # We need to filter out the entire "DUT" or "Background" box if it just matches everything.
                                        # In this case, "Class_0" is likely the entire Power Adapter body!
                                        # If a Crack is fully inside the main body (overlap = 1.0), we don't really want to label it as "Component" 
                                        # UNLESS there are no other more specific components.
                                        # Usually, we want to skip labeling if the box covers the whole image.
                                        
                                        if overlap_ratio >= 0.3 and overlap_ratio > best_overlap:
                                            # Optional: Ignore if it's the main DUT box (usually class 0 or covers > 80% of image)
                                            if (gx2 - gx1) * (gy2 - gy1) < (DUT_Corrected.shape[0] * DUT_Corrected.shape[1] * 0.9):
                                                best_overlap = overlap_ratio
                                                current_defect_mode = cls_name
                                                print(f"DEBUG: Selected '{current_defect_mode}' as Detected Defect.")
                                            
                            p_dict['Detected Defect'] = current_defect_mode if current_defect_mode else ''
                            
                            # Store for visualization
                            debug_iou_data.append({
                                'contour': contour,
                                'defect_mode': current_defect_mode
                            })
                            
                            p_dict.update(dims_res_single)
                            
                            # Curved Line Measurement
                            try:
                                from skimage.morphology import skeletonize
                                skeleton = skeletonize(single_mask // 255)
                                curved_len = np.sum(skeleton) * 0.0626
                                p_dict['Curved Line Measurement'] = curved_len
                            except ImportError:
                                pass
    
                            parametric_results.append(p_dict)
    
                    except Exception as e:
                        print(f">>> [Strategy] Error extracting parametric dimensions: {e}")
                        parametric_results.append({'Picture_Name': file, 'Defect_Class': 'Crack'})
                        
                    # --- Generate Debug IOU Image ---
                    if self.context.get('defect_detection_debug') or self.context.get('general_defect_save_vis'):
                        try:
                            debug_vis = DUT_Corrected.copy()
                            
                            # 1. Draw BBoxes
                            if bbox_general_FM:
                                for gdd in bbox_general_FM:
                                    if isinstance(gdd, dict):
                                        bbox = gdd['bbox']
                                        cls_name = gdd['class_name']
                                    else:
                                        bbox = gdd
                                        cls_name = "Component"
                                    if str(cls_name) == "0": cls_name = "Body"
                                        
                                    gx1, gy1, gx2, gy2 = map(int, bbox)
                                    cv2.rectangle(debug_vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                                    cv2.putText(debug_vis, cls_name, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                            # 2. Draw Contours and Labels
                            for idx, item in enumerate(debug_iou_data):
                                cnt = item['contour']
                                d_mode = item['defect_mode']
                                
                                # Green if matched inside a box, Red if unmatched
                                color = (0, 180, 0) if d_mode else (0, 0, 255)
                                cv2.drawContours(debug_vis, [cnt], -1, color, -1)
                                
                                # Label
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"])
                                    cY = int(M["m01"] / M["m00"])
                                else:
                                    x, y, w, h = cv2.boundingRect(cnt)
                                    cX, cY = x, y
                                cv2.putText(debug_vis, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                            debug_name = f"{os.path.splitext(file)[0]}_debug_iou.png"
                            cv2.imwrite(os.path.join(inferred_pic_dir, debug_name), debug_vis)
                            print(f">>> [Strategy] Saved Defect Detection Debug visualization: {debug_name}")
                        except Exception as e:
                            print(f">>> [Strategy] Error generating debug IOU image: {e}")
                    # --- End Generate Debug IOU Image ---
                    import utils.cv_ops as cv_ops
                    cv_ops.save_inferred_pic_overlay(DUT_Corrected, final_defect_mask, inferred_pic_dir, os.path.splitext(file)[0], input_is_bgr=True)
                            

                            
            else:
                parametric_results.append({'Picture_Name': file, 'Defect_Class': 'Pass'})
                
            # Cleanup temp file
            if not debug_mode and os.path.exists(path_to_process):
                 os.remove(path_to_process)

        
        if return_masks_only:
            return extracted_masks_dict
            
        if parametric_results:
            
            # Build and save DataFrame
            if 'Contour' not in getattr(utils_general, 'Output_Config', []):
                utils_general.Output_Config.append('Contour')
                
            df = output_ops.build_parametric_output_df(
                data=parametric_results,
                failure_mode=fm,
                defect_output_format=self.defect_output_df,
            )
            
            if 'Contour' in pd.DataFrame(parametric_results).columns and 'Contour' not in df.columns:
                df['Contour'] = pd.DataFrame(parametric_results)['Contour'].values
                
            if not df.empty:
                output_ops.parametric_output(
                    main_folder_path=res_path,
                    sub_folder_path="",
                    file_name="Parametric_Output.xlsx",

                    sheet_name='Sheet1',
                    df=df
                )

            print(f">>> [Strategy] Parametric Excel saved successfully with {len(parametric_results)} rows.")

        print(">>> [Strategy] Dino Execution Completed!")
