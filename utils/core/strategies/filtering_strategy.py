import os
import cv2
import numpy as np
import pandas as pd
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils import utils_general
from utils import io_ops
from utils import output_ops

class FilteringStrategy(AnalysisStrategy):
    """
    专门处理传统机器视觉 Filtering 语法的策略类。
    例如: Filtering[BG Subtraction]&&Filtering[Color Distance]
    """
    
    def __init__(self, context):
        super().__init__(context)
        self.cm = ConfigManager()

    def _resolve_models(self):
        """加载必要的模型 (DUT Predictor, SAM2)"""
        weights_df = self.cm.get_sheet('Weights')
        from core.pipeline import DefectDetectionPipeline
        pipeline = DefectDetectionPipeline()
        weights = pipeline._resolve_model_weights(self.fm, self.product, self.generation, weights_df)
        
        cfg_path_dut = weights.get('cfg_path_dut')
        weights_path_dut = weights.get('weights_path_dut')
        
        # Fallback to general DUT model if specific one is not found
        if not cfg_path_dut or not weights_path_dut:
            gen_weights = pipeline._resolve_model_weights('General', self.product, self.generation, weights_df)
            cfg_path_dut = gen_weights.get('cfg_path_dut')
            weights_path_dut = gen_weights.get('weights_path_dut')
        
        # 将路径注册到 utils_general 以兼容底层工具
        utils_general.cfg_path_dut = cfg_path_dut
        utils_general.weights_path_dut = weights_path_dut
        
        self.dut_predictor = None
        if cfg_path_dut and weights_path_dut:
            print(f">>> [FilteringStrategy] Loading DUT model from {weights_path_dut}")
            self.dut_predictor = utils_general.load_detectron2_model(cfg_path_dut, weights_path_dut, score_thresh=0.5)
        else:
             print(f">>> [FilteringStrategy] Warning: Could not resolve DUT model weights for FM='{self.fm}'")

        # 1.5 查找并加载 General Defect Predictor
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
                                
        if gd_cfg and gd_weight and os.path.exists(gd_cfg) and os.path.exists(gd_weight):
            print(f">>> [FilteringStrategy] Loading General Defect model from {gd_weight}")
            self.general_defect_predictor = utils_general.load_detectron2_model(gd_cfg, gd_weight, score_thresh=0.5)
            if hasattr(self.general_defect_predictor, 'cfg'):
                self.general_defect_predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        else:
            print(f">>> [FilteringStrategy] Warning: General Defect model not found for '{gdd_key}'. Falling back to DUT.")
            self.general_defect_predictor = self.dut_predictor # 兜底复用
        

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
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
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

    def execute(self) -> None:
        print(f">>> [FilteringStrategy] Executing traditional Filtering pipeline for '{self.fm}'...")
        
        # 1. 准备路径和配置
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        
        # Safely resolve paths
        download_path = self.download_path if self.download_path is not None else ""
        if not os.path.isabs(download_path):
            download_path = os.path.join(project_root, download_path)
            
        # Resolve Result path dynamically based on current fm and product
        if self.res_path:
            res_path = self.res_path
        else:
            # For a dispatched strategy, we must fetch the new output_path
            fm_df = self.cm.get_sheet('Failure Mode')
            from utils.base_utils import _norm
            fm_matches = fm_df[fm_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm)]
            output_path_type = 'Result'
            if not fm_matches.empty and pd.notna(fm_matches.iloc[0].get('Output_Path')):
                output_path_type = str(fm_matches.iloc[0]['Output_Path']).strip()
                if output_path_type.lower() == 'nan': output_path_type = 'Result'
                
            if output_path_type == 'Result':
                res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{self.fm}_{self.fm}_Result")
            else:
                res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{self.fm}_{self.fm}_Result")
        if not os.path.isabs(res_path):
            res_path = os.path.join(project_root, res_path)
            
        inferred_pic_dir = os.path.join(res_path, "Inferred Pic")
        os.makedirs(inferred_pic_dir, exist_ok=True)
        
        # 获取具体的 Flow 配置
        config_group = self.context.get('config_group_override')
        flow_df = self.cm.get_sheet('Flow')
        
        flow_matches = flow_df[
            (flow_df['Failure Mode'].astype(str).str.lower() == str(self.fm).lower())
        ]
        if config_group:
            from utils.base_utils import _norm
            flow_matches = flow_matches[flow_matches['Config_Group'].astype(str).apply(_norm) == _norm(config_group)]
            
        if flow_matches.empty:
            print(f">>> [FilteringStrategy] Error: Flow config not found for {self.fm} (Group: {config_group})")
            return
            
        flow_row = flow_matches.iloc[0]
        
        # 补丁：读取并下发 PCA_Alignment 标志给底层
        pca_val = flow_row.get('PCA_Alignment', 'No')
        if pca_val is None or str(pca_val).lower() == 'nan': pca_val = 'No'
        self.context['use_batch_alignment'] = str(pca_val).strip().lower() == 'yes'
        
        # Parse methods (e.g. Filtering[BG Subtraction]&&Filtering[Color Distance])
        method_str = str(flow_row.get('Defect Identification', '')).strip()
        ops_to_execute = utils_general.parse_defect_id_method(method_str)
        print(f">>> [FilteringStrategy] Parsed Ops: {ops_to_execute}")
        
        # Load Filtering sheet specifically for this FM
        from utils.filtering_ops import load_and_filter_filtering_sheet
        filtering_df = load_and_filter_filtering_sheet(
            config_path="RELP_Configuration.xlsx",
            product=self.product,
            generation=self.generation,
            failure_mode=self.fm,
            config_group=config_group,
            verbose=False
        )
        
        # Load Adaptive Gaussian config for this FM
        adaptive_gaussian_df = self.cm.get_sheet('Adaptive Gaussian')
        self.current_adaptive_gaussian_config = None
        if not adaptive_gaussian_df.empty:
            from utils.base_utils import _norm
            ag_rows = adaptive_gaussian_df[adaptive_gaussian_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm)]
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
                    self.current_adaptive_gaussian_config = ag_rows.iloc[0].to_dict()
                    print(f">>> [FilteringStrategy] Loaded Adaptive Gaussian config: {self.current_adaptive_gaussian_config}")

        self._resolve_models()
        
        # 删除旧的祖传硬编码，让其回退到 base_strategy 解析到的 Excel 配置
        # utils_general.Output_Config = ['Length']
        # utils_general.Reference_Params = {'Length': {'R': 200.0}}
        # utils_general.Gray_Scale_Params = {'Enabled': True, 'Mode': 'Gray Scale', 'Invert': 'Yes', 'Bining': '[0, 50, 90, 120, 150, 180, 210, 255]'}

        # 确定 Defect Output Format 和 Color Space 参数 (在循环外确定一次即可)
        dof = 'individual'
        if not self.defect_output_df.empty:
            dof_row = self.defect_output_df[self.defect_output_df['Failure Mode'].astype(str).str.strip() == self.fm.strip()]
            if not dof_row.empty:
                dof = str(dof_row.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                
                # 动态读取 Excel 里的 Color Space Conversion 和 Binning 参数
                mode_val = dof_row.iloc[0].get('Color Space Conversion')
                bin_val = dof_row.iloc[0].get('Color Space Binning')
                if pd.notna(mode_val) and str(mode_val).strip() != '':
                    utils_general.Gray_Scale_Params = {
                        'Enabled': True,
                        'Mode': str(mode_val).strip(),
                        'Invert': str(dof_row.iloc[0].get('Color Space Invertion', 'No')).strip(),
                        'Bining': str(bin_val).strip() if pd.notna(bin_val) else ''
                    }
                    print(f">>> [FilteringStrategy] Dynamically loaded Gray Scale Params: {utils_general.Gray_Scale_Params}")
                    
        # 准备动态读取 Reference
        reference_df = cfg_mgr.get_sheet('Reference')

        file_list = self.context.get('file_list', [])
        if not file_list:
            # 补丁：如果 Filtering 被直接配置在 Defect Identification（作为头部运行），则尝试主动读取文件夹
            if os.path.exists(download_path):
                print(f">>> [FilteringStrategy] No upstream file_list provided. Scanning local path: {download_path}")
                for root, dirs, files in os.walk(download_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')) and not file.startswith('.'):
                            file_list.append(os.path.join(root, file))
                            
            if not file_list:
                print(f">>> [FilteringStrategy] No files to process in {download_path}.")
                return
             
        parametric_results = []
             
        for image_path in file_list:
            file_name = os.path.basename(image_path)
            print(f"\n>>> [FilteringStrategy] Processing {file_name}...")
            
            # --- 动态 Reference 匹配 ---
            utils_general.Reference_Params.clear()
            if reference_df is not None and not reference_df.empty:
                try:
                    from utils.output_ops import match_reference_for_image
                    dynamic_ref = match_reference_for_image(
                        reference_df, file_name, self.product, self.generation, getattr(utils_general, 'Output_Config', []), 'DUT'
                    )
                    if dynamic_ref: utils_general.Reference_Params.update(dynamic_ref)
                except Exception as e:
                    print(f">>> [FilteringStrategy] Warning: Failed to match dynamic reference: {e}")
            
            # --- Handle Relative Path ---
            rel_path = os.path.relpath(os.path.dirname(image_path), download_path)
            if rel_path == '.': rel_path = ''
            
            curr_inf_dir = os.path.join(inferred_pic_dir, rel_path)
            os.makedirs(curr_inf_dir, exist_ok=True)
            curr_res_dir = os.path.join(res_path, rel_path)
            
            curr_ref_dir = os.path.join(res_path, "reference", rel_path)
            os.makedirs(curr_ref_dir, exist_ok=True)
            
            inherited_data = self.context.get('inherited_data', {}).get(image_path)
            if inherited_data:
                print(f">>> [FilteringStrategy] Using inherited mask and image from previous stage.")
                image = inherited_data.get('original_image')
                DUT_Corrected = inherited_data.get('dut')
                Contour_Corrected = inherited_data.get('contour')
                alignment_metadata = inherited_data.get('alignment')
                mask_filtered = inherited_data.get('mask')
                if mask_filtered is None:
                    h, w = DUT_Corrected.shape[:2]
                    mask_filtered = np.ones((h, w), dtype=np.uint8) * 255
            else:
                # [积木块 1 & 2 & 3]: 图像预处理与提取
                image, DUT_Corrected, Contour_Corrected, alignment_metadata = self.prepare_dut_image(
                    image_path, self.dut_predictor, self.sam2_predictor
                )
                
                if DUT_Corrected is None:
                    print(f">>> [FilteringStrategy] DUT alignment failed! Falling back to original raw image for: {file_name}")
                    DUT_Corrected = image.copy()
                    h, w = DUT_Corrected.shape[:2]
                    Contour_Corrected = np.ones((h, w), dtype=np.uint8) * 255
                    alignment_metadata = {}
                    
                mask_filtered = Contour_Corrected.copy()
            
            # [积木块 4]: 核心 Filtering 推理
            if mask_filtered.dtype != np.uint8:
                mask_filtered = (mask_filtered.astype(np.uint8) * 255)
            
            for op in ops_to_execute:
                op_name = op['name']
                if 'Filtering' in op_name:
                    op_params = op.get('params', [])
                    print(f"  -> Applying filter operation: {op_params}")
                    operations = utils_general.parse_filtering_operations(op_params)
                    
                    mask_filtered, extra_results, _, _, updated_contour = utils_general.apply_filtering(
                        mask_filtered=mask_filtered,
                        original_img=DUT_Corrected,
                        operations=operations,
                        filtering_df=filtering_df,
                        save_dir=curr_ref_dir,
                        image_path=image_path,
                        contour_image=Contour_Corrected,
                        adaptive_gaussian_config=self.current_adaptive_gaussian_config,
                        defect_id_method=self.fm,
                        mask_default=Contour_Corrected
                    )
                elif 'Adaptive Gaussian' in op_name or 'Global Threshold' in op_name or 'Global Threshhold' in op_name:
                    op_params = op.get('params', [])
                    print(f"  -> Applying {op_name} operation: {op_params}")
                    # Construct operation dict manually
                    actual_method = 'Global Threshold' if 'Global Thresh' in op_name else 'Adaptive Gaussian'
                    operations = [{'method': actual_method, 'key': '', 'output_params': op_params}]
                    
                    mask_filtered, extra_results, _, _, updated_contour = utils_general.apply_filtering(
                        mask_filtered=mask_filtered,
                        original_img=DUT_Corrected,
                        operations=operations,
                        filtering_df=filtering_df,
                        save_dir=curr_ref_dir,
                        image_path=image_path,
                        contour_image=Contour_Corrected,
                        adaptive_gaussian_config=self.current_adaptive_gaussian_config,
                        defect_id_method=self.fm,
                        mask_default=Contour_Corrected
                    )
            
            # [积木块 5]: 空间裁切 Scope Filter (General FM) - Skip if inherited!
            final_defect_mask = mask_filtered
            detected_labels = []
            matched_bboxes = []
            
            if not inherited_data:
                gfm_rules = self.context.get('general_fm_rules')
                if gfm_rules and final_defect_mask is not None:
                    from utils.general_fm_ops import apply_general_fm_scope
                    print(f">>> [FilteringStrategy] Applying General FM Scope Rules...")
                    
                    if self.general_defect_predictor is None:
                        print(f">>> [FilteringStrategy] Error: General Defect Predictor is required for Scope Rules but not loaded.")
                    else:
                        final_defect_mask, detected_labels, matched_bboxes = apply_general_fm_scope(
                            mask=final_defect_mask,
                            image=DUT_Corrected,
                            detector=self.general_defect_predictor,
                            gfm_rules=gfm_rules
                        )
                
            # [积木块 6]: 缺陷测算与输出
            if final_defect_mask is not None and np.any(final_defect_mask):
                print(f">>> [FilteringStrategy] Defect found in {file_name}!")
                mask_uint8 = (final_defect_mask.astype(np.uint8) * 255) if final_defect_mask.dtype == bool else final_defect_mask
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 画 debug 叠加图
                debug_vis = DUT_Corrected.copy()
                for cnt in contours:
                    if cv2.contourArea(cnt) < 5: continue
                    cv2.drawContours(debug_vis, [cnt], -1, (0, 0, 255), -1)
                
                if matched_bboxes:
                     for gdd in matched_bboxes:
                         gx1, gy1, gx2, gy2 = map(int, gdd['bbox'])
                         cv2.rectangle(debug_vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                         
                debug_path = os.path.join(curr_inf_dir, f"{os.path.splitext(file_name)[0]}_filtered_result.jpg")
                
                # 图片瘦身压缩逻辑：限制最大边长为 1920，且降低 JPEG 质量到 60
                h_vis, w_vis = debug_vis.shape[:2]
                max_dim = 1920
                if max(h_vis, w_vis) > max_dim:
                    scale_factor = max_dim / float(max(h_vis, w_vis))
                    debug_vis = cv2.resize(debug_vis, (int(w_vis * scale_factor), int(h_vis * scale_factor)), interpolation=cv2.INTER_AREA)
                
                cv2.imwrite(debug_path, debug_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                print(f"  -> Saved visualization to {debug_path}")
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 5: continue
                    
                    p_dict = {
                        'Picture_Name': file_name, 
                        'Defect_Class': 'Defect',
                        'Dir': rel_path,
                        'Filename': file_name
                    }
                    p_dict['Detected Defect'] = ", ".join(detected_labels) if detected_labels else ''
                    
                    single_mask = np.zeros_like(mask_uint8)
                    cv2.drawContours(single_mask, [cnt], -1, 255, -1)
                    
                    try:
                        dims_res_single = output_ops.calculate_parametric_dimensions(
                            defect_cnt=cnt,
                            dut_dims=DUT_Corrected.shape[:2],
                            ref_params=utils_general.Reference_Params,
                            output_config=utils_general.Output_Config,
                            contour_area_val=area,
                            ref_area=1.0,
                            ref_found=True,
                            defect_mask=single_mask,
                            is_line_shape=False,
                            image=DUT_Corrected,
                            gray_scale_params=utils_general.Gray_Scale_Params,
                            output_dir=curr_res_dir,
                            image_filename=image_path,
                            detector=self.dut_predictor
                        )
                        p_dict.update(dims_res_single)
                    except Exception as e:
                        print(f">>> [FilteringStrategy] Error calculating dimensions: {e}")
                    
                    parametric_results.append(p_dict)
            else:
                 print(f">>> [FilteringStrategy] No defect passed the filters for {file_name}.")
                 parametric_results.append({'Picture_Name': file_name, 'Defect_Class': 'Pass', 'Dir': rel_path, 'Filename': file_name})
                 
        if parametric_results:
            df = output_ops.build_parametric_output_df(
                data=parametric_results,
                failure_mode=self.fm,
                defect_output_format=dof
            )
            if not df.empty:
                output_ops.parametric_output(
                    main_folder_path=res_path,
                    sub_folder_path="",
                    file_name=f"Parametric_Output_{config_group}.xlsx",
                    sheet_name="Sheet1",
                    df=df
                )
        print(">>> [FilteringStrategy] Execution Completed!")
