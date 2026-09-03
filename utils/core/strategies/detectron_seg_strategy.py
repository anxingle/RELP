import os
import cv2
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.detectron_wrapper import load_detectron_model
from utils.wrappers.sam2_wrapper import init_sam2, get_sam2_predictor
from utils import cv_ops, output_ops, detectron_ops, utils_general
from utils.bbox_ops import sorting_bbox, scaling_up_bbox, bbox_centroid
from utils.output_ops import match_reference_for_image, SN_list_df
from utils.cv_ops import check_pic_SN_existance

def parse_silicon_path(folder_path, download_path, image_filename):
    relative_path = str(Path(folder_path).relative_to(download_path))
    if relative_path.count('/') == 0:
        relative_path = relative_path + '/' + relative_path
    
    dir_splited = relative_path.rsplit('/', 1)
    dir_1, dir_2 = dir_splited[0], dir_splited[1]
    
    project_code_parser = dir_2
    if dir_2.count('-') == 0:
        project_code_parser = dir_2 + '-' + dir_2
    project_code, face_name = project_code_parser.rsplit('-', 1)
    
    return {
        'Dir': relative_path,
        'Dir_1': dir_1,
        'Cube_Face': dir_2,
        'Project': project_code,
        'Detailed_Cube_Face': face_name,
        'Pic_Name': image_filename
    }

class DetectronSegStrategy(AnalysisStrategy):
    """
    硅胶脱层专项检测策略 (完全重构版)
    脱离旧大循环，实现了原生的 Detectron -> SAM2 -> 缺陷求交集 -> SAM2 缺陷细化的完美工业流。
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context)
        
    def execute(self) -> None:
        print(f">>> [Strategy] Executing DetectronSegStrategy for FM: {self.fm} ...")
        import numpy as np
        import utils.utils_general as utils_general
        
        cfg_mgr = ConfigManager()
        fm = self.fm
        product = self.product
        generation = self.generation
        
        # 1. 基础配置
        flow_cfg = cfg_mgr.get_flow(fm)
        if not flow_cfg: return
        
        debug_mode = str(flow_cfg.debug_mode).strip().lower() in ('yes', 'true', '1') if flow_cfg.debug_mode else False
        sn_read_mode = str(flow_cfg.sn_mapping).strip() if flow_cfg.sn_mapping else "From Excel"
        if sn_read_mode.lower() in ('from excel', 'excel'): sn_read_mode = "From Excel"
        elif sn_read_mode.lower() in ('from pic name', 'file_path', 'ocr'): sn_read_mode = "From Pic Name"
        
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        if not fm_df.empty:
            from utils.base_utils import _norm
            fm_row = fm_df[fm_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm)].iloc[0]
            if not self.download_path:
                self.download_path = fm_row.get('Pic_Path')
        
        # 获取 Weights
        weights_df = cfg_mgr.get_sheet('Weights')
        from core.pipeline import DefectDetectionPipeline
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, product, generation, weights_df)
        
        cfg_path = weights.get('cfg_path')
        weights_path = weights.get('weights_path')
        cfg_path_dut = weights.get('cfg_path_dut')
        weights_path_dut = weights.get('weights_path_dut')
        
        if not cfg_path or not weights_path:
            print(">>> [Strategy Error] Defect Model weights missing.")
            return
        if not cfg_path_dut or not weights_path_dut:
            print(">>> [Strategy Error] DUT Model weights missing.")
            return

        # 获取 SAM2 配置
        sam_cfg = weights.get('sam_config', getattr(utils_general, 'sam_config_path', None))
        sam_ckpt = weights.get('sam_weights', getattr(utils_general, 'sam_checkpoint_path', None))

        print(f">>> [Strategy] Loading Defect Predictor: {weights_path}")
        defect_predictor = load_detectron_model(cfg_path, weights_path)
        print(f">>> [Strategy] Loading DUT Predictor: {weights_path_dut}")
        dut_predictor = load_detectron_model(cfg_path_dut, weights_path_dut)
        
        print(f">>> [Strategy] Initializing SAM2: {sam_ckpt}")
        try:
            sam2_predictor = init_sam2(sam_cfg, sam_ckpt, device='cpu')
        except Exception as e:
            print(f">>> [Strategy Error] SAM2 load failed: {e}")
            return
            
        # 结果路径
        res_path = self.res_path
        if not res_path:
            import os
            res_path = os.path.join(os.getcwd(), 'Result', f"{product}_{generation}_{fm}_{fm}_Result")
            self.res_path = res_path
        os.makedirs(res_path, exist_ok=True)
        
        download_path = self.download_path
        if download_path and not os.path.isabs(download_path):
            import os
            download_path = os.path.join(os.getcwd(), download_path)
            
        if not download_path or not os.path.exists(download_path): 
            print(f">>> [Strategy Error] Download path missing or does not exist: {download_path}")
            return
        
        reference_df = cfg_mgr.get_sheet('Reference')
        
        # 开始遍历目录
        global_parametric_results = []
        for root, dirs, files in os.walk(download_path):
            valid_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not f.startswith('.')]
            if not valid_files: continue
            
            print(f"\n{'='*50}\n>>> [Strategy Detectron] Processing folder: {root}\n{'='*50}")
            
            # SN 匹配
            if sn_read_mode == "From Excel":
                file_to_exam = check_pic_SN_existance(root, 2)
            else:
                file_to_exam = check_pic_SN_existance(root, 1)
                
            # fallback if sn match failed
            if not file_to_exam:
                file_to_exam = [os.path.splitext(f)[0] for f in valid_files]
                
            for image_filename in file_to_exam:
                pic_file = os.path.join(root, image_filename + ".jpg")
                print(f"  -> Processing image: {pic_file}")
                
                image = cv_ops.load_and_validate_image(pic_file)
                if image is None: continue
                
                h, w = image.shape[:2]
                
                # --- 动态 Reference 匹配 ---
                utils_general.Reference_Params.clear()
                if reference_df is not None and not reference_df.empty:
                    try:
                        dynamic_ref = match_reference_for_image(
                            reference_df, pic_file, product, generation, getattr(utils_general, 'Output_Config', []), 'DUT'
                        )
                        if dynamic_ref: utils_general.Reference_Params.update(dynamic_ref)
                    except: pass
                    
                # --- 解析 SN ---
                SN_df = pd.DataFrame()
                if sn_read_mode == "From Excel":
                    sn_file = os.path.join(root, image_filename + ".xlsx")
                    if os.path.exists(sn_file):
                        SN_df = SN_list_df(pd.read_excel(sn_file))
                elif sn_read_mode == "From Pic Name":
                    SN_df = pd.DataFrame({"SN": [image_filename]})
                    
                # 解析硅胶特有路径
                path_info = parse_silicon_path(root, download_path, image_filename)
                
                # --- 1. 检测缺陷全图 Mask ---
                outputs = defect_predictor(image)
                instances = outputs["instances"]
                instances = instances[instances.scores >= 0.5]
                
                defect_mask = np.zeros((h, w), dtype=np.uint8)
                import numpy as np
                if len(instances) > 0 and instances.has("pred_masks"):
                    masks_tensor = instances.pred_masks.cpu().numpy()
                    for m in masks_tensor:
                        defect_mask[m > 0] = 255

                # --- 2. 检测 DUT 及非检测区 (Sub-features / Keep-out Zones) ---
                dut_outputs = dut_predictor(image)
                dut_instances = dut_outputs["instances"]
                dut_instances = dut_instances[dut_instances.scores >= 0.5]

                # --- 通用排除逻辑：Class ID == 0 为 DUT 主体；Class ID != 0 为非检测排除区 ---
                excluded_feature_boxes = []
                if dut_instances.has("pred_classes"):
                    classes_arr = dut_instances.pred_classes.cpu().numpy()
                    boxes_arr = dut_instances.pred_boxes.tensor.cpu().numpy()

                    # 收集所有非检测区 (Class ID != 0) 并从当前初始 defect_mask 中扣除置零
                    for cls_id, box in zip(classes_arr, boxes_arr):
                        if cls_id != 0:
                            excluded_feature_boxes.append(box)
                            x1, y1, x2, y2 = box.astype(int)
                            defect_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0

                    if len(excluded_feature_boxes) > 0:
                        print(
                            f"     [Strategy] Mask Subtraction: cleared {len(excluded_feature_boxes)} excluded sub-features (Class ID != 0) from defect mask.")

                    # 仅保留主待测物 (Class ID == 0) 用于后续的 SN 绑定与尺寸分析
                    dut_instances = dut_instances[dut_instances.pred_classes == 0]
                    print(
                        f"     [Strategy] Filtered to primary DUTs (Class ID == 0): remaining {len(dut_instances)} instances.")

                boxes = dut_instances.pred_boxes.tensor.cpu().numpy()
                boxes_coords = boxes[:, :4]
                
                # --- 处理多排物体 (Multi-row SN Mapping) ---
                from utils.bbox_ops import bbox_centroid
                from utils.output_ops import clustering_Kmeans
                
                final_boxes_with_sn = []
                
                if not SN_df.empty and SN_df.shape[1] > 1:
                    print(">>> [Strategy] Multi-row SN Mapping detected.")
                    from utils.bbox_ops import sorting_bbox, bbox_outlier_supression
                    
                    # Ensure bbox_pending_suppression has the full 8 columns (including center_y at index 7)
                    sorting_strategy, _, bbox_pending_suppression = sorting_bbox(boxes_coords, SN_df)
                    
                    _, _, _, _, bbox_clusters = clustering_Kmeans(bbox_pending_suppression, SN_df, 7)
                    sorted_keys = sorted(bbox_clusters.keys(), key=lambda k: np.mean(bbox_clusters[k][:, 7]))
                    
                    for row_idx, key in enumerate(sorted_keys):
                        sub_boxes_full = bbox_clusters[key]
                        if row_idx < SN_df.shape[1]:
                            sub_sn_series = SN_df.iloc[:, row_idx].dropna()
                        else:
                            sub_sn_series = pd.Series()
                            
                        # Apply outlier suppression if DUT qty > SN qty
                        if len(sub_sn_series) > 0 and len(sub_boxes_full) > len(sub_sn_series):
                            print(f"     [Strategy] Outlier suppression active for row {row_idx}: DUTs={len(sub_boxes_full)}, SNs={len(sub_sn_series)}")
                            sub_boxes_full = bbox_outlier_supression(sub_boxes_full, 4, sub_sn_series.to_frame(), None)
                            
                        # Retrieve the cleaned x1, y1, x2, y2 and sort by X axis (index 0)
                        sub_boxes = sub_boxes_full[:, :4]
                        sorted_sub_indices = sub_boxes[:, 0].argsort()
                        sub_boxes = sub_boxes[sorted_sub_indices]
                        
                        for item_idx, box in enumerate(sub_boxes):
                            sn_val = "Unknown"
                            if item_idx < len(sub_sn_series):
                                sn_val = str(sub_sn_series.iloc[item_idx]).strip()
                            final_boxes_with_sn.append((box, sn_val))
                else:
                    # 单排逻辑或者 shape[1] == 1 的多目标
                    from utils.bbox_ops import bbox_centroid, sorting_bbox, bbox_outlier_supression
                    
                    sorting_strategy, _, bbox_pending_suppression = sorting_bbox(boxes_coords, SN_df)
                    
                    # Evaluate SN quantity
                    SN_QTY = 0
                    if not SN_df.empty:
                        SN_QTY = len(SN_df) if isinstance(SN_df, pd.Series) else SN_df.count().sum()
                    elif sn_read_mode == "From Pic Name":
                        SN_QTY = 1
                        
                    # Apply outlier suppression if DUT qty > SN qty
                    if SN_QTY > 0 and len(bbox_pending_suppression) > SN_QTY:
                        print(f">>> [Strategy] Outlier suppression active (Single-row): DUTs={len(bbox_pending_suppression)}, SNs={SN_QTY}")
                        sn_for_suppression = SN_df if isinstance(SN_df, pd.DataFrame) else SN_df.to_frame()
                        bbox_pending_suppression = bbox_outlier_supression(bbox_pending_suppression, sorting_strategy, sn_for_suppression, None)
                        
                    boxes_coords = bbox_pending_suppression[:, :4]
                    
                    # 极其重要：如果是多行多列，但 Excel 只给了一列 SN，我们需要确保排序方式正确。
                    # 通常 sorting_strategy = 0 (按X排), 1 (按Y排)
                    if sorting_strategy == 0 or sorting_strategy == 1:
                        sorted_indices = boxes_coords[:, sorting_strategy].argsort()
                        boxes_coords = boxes_coords[sorted_indices]
                    else:
                        # 默认按 X 轴（从左到右）排序
                        sorted_indices = boxes_coords[:, 0].argsort()
                        boxes_coords = boxes_coords[sorted_indices]
                        
                    # 如果 sorting_strategy 返回 -1，并且确实有 10 个壳子但只有一列 SN，
                    # 那么纯按 X 轴排序会导致前排后排的物体交错匹配！
                    # 我们需要在这里执行一个简易的分排排序：先按 Y 大致分排，再在每排内按 X 排序。
                    if sorting_strategy == -1 and len(boxes_coords) > 1:
                        # 使用 y 坐标排序
                        y_sorted_indices = boxes_coords[:, 1].argsort()
                        boxes_coords = boxes_coords[y_sorted_indices]
                        
                        # 尝试将 y 坐标相近的分为同一排 (比如 y 差异小于 100 像素)
                        rows = []
                        current_row = [0]
                        for i in range(1, len(boxes_coords)):
                            if abs(boxes_coords[i, 1] - boxes_coords[current_row[0], 1]) < 100:
                                current_row.append(i)
                            else:
                                rows.append(current_row)
                                current_row = [i]
                        rows.append(current_row)
                        
                        # 每排内部按 X 轴排序
                        final_sorted_indices = []
                        for row in rows:
                            row_boxes = boxes_coords[row]
                            x_sorted_sub_indices = row_boxes[:, 0].argsort()
                            final_sorted_indices.extend([row[i] for i in x_sorted_sub_indices])
                            
                        boxes_coords = boxes_coords[final_sorted_indices]

                    for idx, box in enumerate(boxes_coords):
                        sn_val = "Unknown"
                        if sn_read_mode == "From Excel" and not SN_df.empty:
                            try:
                                # Fix the DataFrame retrieval. SN_list_df returns a Series if 1 column, or DataFrame.
                                # Let's get the values as a list to avoid iloc index mismatch 
                                # It's a DataFrame with shape (10, 1)
                                if isinstance(SN_df, pd.DataFrame):
                                    sn_val = str(SN_df.iloc[idx, 0]).strip()
                                else:
                                    sn_vals = SN_df.values.flatten()
                                    if idx < len(sn_vals):
                                        sn_val = str(sn_vals[idx]).strip()
                                print(f"DEBUG SN: idx={idx}, mapped_sn={sn_val}")
                            except Exception as e:
                                print(f"SN Parsing error: {e}")
                        elif sn_read_mode == "From Pic Name":
                            sn_val = image_filename
                        final_boxes_with_sn.append((box, sn_val))
                        
                parametric_results = []
                overall_BG = np.zeros((h, w), dtype=np.uint8)
                
                for box, DUT_SN in final_boxes_with_sn:
                    print(f"DEBUG LOOP: DUT_SN is {DUT_SN}")
                    box = scaling_up_bbox(np.array([box]), 1.05, 1.05, h, w)[0]
                    input_box = box.astype(int)
                    
                    # 3. 运行 SAM2 得到手机壳精确轮廓
                    if hasattr(sam2_predictor, 'set_image'):
                        sam2_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    else:
                        from utils.wrappers.sam2_wrapper import get_sam2_predictor
                        # wrap it first
                        real_predictor = get_sam2_predictor(sam2_predictor)
                        real_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        sam2_predictor = real_predictor
                    masks, scores, _ = sam2_predictor.predict(
                        point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False
                    )
                    
                    sorted_ind = np.argsort(scores)[::-1]
                    mask = masks[sorted_ind][0]
                    mask_uint8 = (mask > 0.0).astype(np.uint8) * 255
                    
                    # 计算 DUT 面积和 OBB (偏转角)
                    from utils.utils_general import DUT_Area_OBB_Cal
                    DUT_area, OBB_Angle, mask_contour, rect_points, DUT_length, DUT_width = DUT_Area_OBB_Cal(mask_uint8)
                    
                    # 4. 求交集获取当前 DUT 内的缺陷
                    SAM_yolo_mask = cv2.bitwise_and(mask_uint8, defect_mask)
                    
                    # 5. Quantify Defects
                    _, res_combined = cv2.threshold(SAM_yolo_mask, 110, 255, cv2.THRESH_BINARY)
                    defect_contours, _ = cv2.findContours(res_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    
                    defect_idx = 0
                    for cnt in defect_contours:
                        area = cv2.contourArea(cnt)
                        if area >= 25:
                            defect_idx += 1
                            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                            bbox_fine = np.array([x_c, y_c, x_c + w_c, y_c + h_c])
                            
                            # 缺陷二次细化 (SAM2)
                            fine_masks, fine_scores, _ = sam2_predictor.predict(
                                point_coords=None, point_labels=None, box=bbox_fine[None, :], multimask_output=False
                            )
                            fine_mask = fine_masks[np.argmax(fine_scores)]
                            fine_mask_uint8 = (fine_mask > 0.0).astype(np.uint8) * 255
                            
                            fine_seg_area, _, fine_seg_contour, _, _, _ = DUT_Area_OBB_Cal(fine_mask_uint8)
                            
                            # 防御性回退：如果算出的面积太离谱，退回粗略遮罩
                            area_ratio = fine_seg_area / area if area > 0 else 0
                            
                            # 从 Fine_Tuning 配置表获取阈值，回退到系统默认的 Acceptance_Low (0.99) 与 High (1.01)
                            import utils.utils_general as utils_general
                            acceptance_low = getattr(utils_general, 'Acceptance_Low', 0.99)
                            acceptance_high = getattr(utils_general, 'Acceptance_High', 1.01)
                            
                            # 因为 Box Mode 天生不精准，如果用户没有配置专门的低阈值，我们就开启宽松模式
                            if abs(acceptance_low - 0.99) < 0.001:
                                box_acceptance_low = 0.1
                                print(f"DEBUG: Using relaxed Box_Acceptance_Low: {box_acceptance_low} (Global default 0.99 detected)")
                                acceptance_low = box_acceptance_low
                            else:
                                print(f"DEBUG: Using configured Acceptance_Low: {acceptance_low}")

                            if acceptance_low <= area_ratio <= acceptance_high:
                                final_defect_cnt = fine_seg_contour if fine_seg_contour is not None else cnt
                                final_defect_area = fine_seg_area
                                print(f"DEBUG: Fine seg accepted (ratio {area_ratio:.2f})")
                            else:
                                final_defect_cnt = cnt
                                final_defect_area = area
                                print(f"DEBUG: Fine seg rejected (ratio {area_ratio:.2f} not in [{acceptance_low}, {acceptance_high}]), kept coarse mask")
                                
                            cv2.drawContours(overall_BG, [final_defect_cnt], -1, 255, cv2.FILLED)
                            
                            # 计算尺寸与比例
                            try:
                                rect = cv2.minAreaRect(final_defect_cnt)
                                _, size, _ = rect
                                d_len, d_wid = max(size), min(size)
                            except: d_len, d_wid = 0, 0
                            
                            p_dict = {
                                'Filename': image_filename,
                                'SN': DUT_SN,
                                'Defect': f"defect_{defect_idx}",
                                'Defect_Type': self.fm,
                                'Defect Pct': (final_defect_area / DUT_area * 100) if DUT_area > 0 else 0,
                                'Defect_Area': final_defect_area,
                                'Defect_Length': d_len,
                                'Defect_Width': d_wid,
                                'DUT_Area': DUT_area,
                                'OBB_Angle': OBB_Angle,
                                'DUT_Length': DUT_length,
                                'DUT_Width': DUT_width,
                            }
                            
                            # --- 物理尺寸换算 (Physical Dimensions Calculation) ---
                            import utils.utils_general as utils_general
                            from utils import output_ops
                            dims_res_single = output_ops.calculate_parametric_dimensions(
                                defect_cnt=final_defect_cnt,
                                dut_dims=(DUT_length, DUT_width),
                                ref_params=utils_general.Reference_Params,
                                output_config=getattr(utils_general, 'Output_Config', []),
                                contour_area_val=DUT_area,
                                ref_area=1.0,
                                ref_found=True,
                                defect_mask=None,
                                is_line_shape=False,
                                image=image,
                                gray_scale_params=getattr(utils_general, 'Gray_Scale_Params', None),
                                output_dir=None,
                                image_filename=image_filename,
                                detector=dut_predictor
                            )
                            p_dict.update(dims_res_single)
                            
                            # 获取当前图片的正确 Reference 显示字符串
                            ref_str, ref_status = output_ops.build_reference_info(utils_general.Reference_Params)
                            p_dict['Reference'] = ref_str
                            p_dict['Reference_Status'] = ref_status
                            
                            # 注入专属路径字段
                            p_dict.update(path_info)
                            
                            # 生成规范化 Contour 点阵或保存 npy (兼容老代码)
                            # 如果是 Combined 模式，需要把整体 Mask 存为 .npy
                            dof = 'individual'
                            if hasattr(self, 'defect_output_df') and not self.defect_output_df.empty:
                                out_match = self.defect_output_df[self.defect_output_df['Failure Mode'].astype(str).str.strip() == self.fm.strip()]
                                if not out_match.empty:
                                    dof = str(out_match.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                            if dof == 'combined':
                                # Note: the actual mask saving happens after the loop for the whole image
                                # For the p_dict, we assign a placeholder that will be popped by format_parametric_output_df
                                p_dict['Contour'] = "Combined_Mask.npy"
                            else:
                                if len(final_defect_cnt) > 0:
                                    # old logic passed oblique_bbox (which is rect_points from DUT_Area_OBB_Cal)
                                    from utils.cv_ops import normalize_contours
                                    try:
                                        norm_cnt = normalize_contours(rect_points, final_defect_cnt)
                                        # 确保保留 4 位小数
                                        if norm_cnt and len(norm_cnt) > 0:
                                            # norm_cnt 的结构是 [ [[x,y], [x,y]... ] ]
                                            # 但如果它把每个点拆成了单个元素，我们需要拍扁它。
                                            import numpy as np
                                            flat_pts = np.array(norm_cnt).reshape(-1, 2)
                                            formatted_cnt = [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in flat_pts]
                                            p_dict['Contour'] = str(formatted_cnt)
                                        else:
                                            p_dict['Contour'] = "[]" 
                                    except Exception as e:
                                        print(f"DEBUG EXCEPTION IN NORM: {e}")
                                        # Muted error for single-pixel artifacts
                                        points = [[int(pt[0][0]), int(pt[0][1])] for pt in final_defect_cnt]
                                        p_dict['Contour'] = str(points)
                            
                            parametric_results.append(p_dict)
                        global_parametric_results.append(p_dict)
                            
                    # 如果没有发现缺陷
                    if defect_idx == 0:
                        p_dict = {
                            'Filename': image_filename,
                            'SN': DUT_SN,
                            'Defect': "no_detection",
                            'Defect_Type': self.fm,
                            'Defect Pct': '0%',
                            'DUT_Area': DUT_area,
                            'OBB_Angle': OBB_Angle,
                            'DUT_Length': DUT_length,
                            'DUT_Width': DUT_width,
                        }
                        import utils.utils_general as utils_general
                        from utils import output_ops
                        ref_str, ref_status = output_ops.build_reference_info(utils_general.Reference_Params)
                        p_dict['Reference'] = ref_str
                        p_dict['Reference_Status'] = ref_status
                        
                        p_dict.update(path_info)
                        parametric_results.append(p_dict)
                        global_parametric_results.append(p_dict)

                # 7.2. 生成整图的红框展示图
                # --- 终极防御：强制从最终渲染图 overall_BG 中抠掉排除区，杜绝 SAM2 细化时的误反弹 ---
                for box in excluded_feature_boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    overall_BG[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0

                # 读取全局颜色配置
                import utils.utils_general as utils_general
                mask_color_cfg = getattr(utils_general, 'Mask_Color', None)
                color = (0, 162, 255) # Default orange-ish in RGB, wait OpenCV is BGR so (255, 162, 0)
                transparency = 0.5
                if mask_color_cfg and isinstance(mask_color_cfg, dict):
                    # Expecting RGB tuple
                    color = mask_color_cfg.get('rgb', (0, 162, 255))
                    transparency = mask_color_cfg.get('transparency', 0.5)
                # But create_overlay_image expects RGB tuple and converts it if needed? 
                # Let's just pass the color directly. Wait, create_overlay_image has default (0, 162, 255) which is user blue.
                # In the strategy it was hardcoded to color=(0, 0, 255). We should change it to use the default or config.
                
                overlay = cv_ops.create_overlay_image(image, overall_BG, color=color, transparency=transparency)
                
                # 恢复层级结构
                rel_path = os.path.relpath(root, download_path)
                if rel_path == '.':
                    rel_path = ''
                inferred_dir = os.path.join(self.res_path, "Inferred Pic", rel_path)
                os.makedirs(inferred_dir, exist_ok=True)
                cv2.imwrite(os.path.join(inferred_dir, f"{image_filename}_overlay.jpg"), overlay)
                
                # 如果是 Combined 模式，将 overall_BG 存为 .npy
                dof = 'individual'
                if hasattr(self, 'defect_output_df') and not self.defect_output_df.empty:
                    out_match = self.defect_output_df[self.defect_output_df['Failure Mode'].astype(str).str.strip() == self.fm.strip()]
                    if not out_match.empty:
                        dof = str(out_match.iloc[0].get('Defect Output Format', 'individual')).strip().lower()
                if dof == 'combined':
                    try:
                        import numpy as np
                        npy_path = os.path.join(self.res_path, f"{image_filename}_combined_final_mask.npy")
                        # 强转为 bool 或者 uint8 二值化存储
                        mask_to_save = (overall_BG > 0).astype(np.uint8) * 255
                        
                        # 兼容老系统的字典格式，以便 GUI 可以读取 Bbox 和尺寸信息做 Fit Content
                        save_dict = {'mask': mask_to_save}
                        
                        # 尝试从外部获取 bbox 并在原图上的原始物理边界框坐标
                        if len(final_boxes_with_sn) > 0:
                            # Use the bounding box of the whole mask as a fallback
                            nonzero = cv2.findNonZero(mask_to_save)
                            if nonzero is not None:
                                x, y, w, h = cv2.boundingRect(nonzero)
                                bbox_array = np.array([
                                    [x, y],
                                    [x + w, y],
                                    [x + w, y + h],
                                    [x, y + h]
                                ], dtype=np.float32)
                                save_dict['bbox'] = bbox_array
                                
                        # Save DUT dimensions to match legacy
                        save_dict['dut_dimensions'] = {'w': float(overall_BG.shape[1]), 'h': float(overall_BG.shape[0])}
                        
                        np.save(npy_path, save_dict)
                        print(f">>> [Strategy] Saved .npy dict mask to {npy_path} (Combined Mode)")
                        
                        # 把这行字典的 Contour 强行替换为真实的 npy 路径
                        for p_dict in parametric_results:
                            if p_dict.get('Contour') == 'Combined_Mask.npy':
                                p_dict['Contour'] = npy_path
                    except Exception as e:
                        print(f"Error saving .npy mask: {e}")
                    
        # --- 6. 生成图与表 (在遍历完整个文件夹后执行，否则会被覆盖) ---
        if global_parametric_results:
            df = pd.DataFrame(global_parametric_results)
            
            # 对接 Output_Ops 自动缩放引擎
            dof = self.defect_output_df if not self.defect_output_df.empty else pd.DataFrame()
            
            # 兼容旧代码，将 OBB_Angle 等专属字段传给 output_ops 让其保留
            # 强行注入 Contour，防止底层因为白名单没开把它扔掉
            if 'Contour' not in getattr(utils_general, 'Output_Config', []):
                utils_general.Output_Config.append('Contour')
                
            final_df = output_ops.build_parametric_output_df(
                data=df, failure_mode=fm, defect_output_format=dof, 
                gray_scale_params=getattr(utils_general, 'Gray_Scale_Params', {})
            )
            
            # 再次强制补回 Contour (防弹级)
            if 'Contour' in df.columns and 'Contour' not in final_df.columns:
                final_df['Contour'] = df['Contour'].values
            
            if not final_df.empty:
                output_ops.parametric_output(
                    main_folder_path=self.res_path, sub_folder_path="", file_name="Parametric_Output.xlsx",
                    sheet_name=f"{fm}_Result", df=final_df
                )
            print(f"  -> Generated final Parametric_Output.xlsx with {len(global_parametric_results)} rows.")

        print("\n>>> [Strategy] Detectron_Seg Execution Completed! <<<")
