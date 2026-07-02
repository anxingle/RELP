import os
import cv2
import pandas as pd
import numpy as np
import ast
from typing import Dict, Any
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.detectron_wrapper import load_detectron_model
from utils import cv_ops, utils_general, crop_ops
from utils import knn_utils
from utils.base_utils import _norm

def _norm_knn(s):
    if pd.isna(s): return ''
    return str(s).replace(' ', '_').lower().strip()

class KnnStrategy(AnalysisStrategy):
    """
    K近邻颜色聚类策略类 (Textile Discoloration 等缺陷)
    将原来 pipeline 中散落几百行的 KNN 解析和调用全部封装内聚。
    """
    def __init__(self, context: Dict[str, Any]):
        super().__init__(context)
        
    def execute(self) -> None:
        import cv2
        import numpy as np
        print(f">>> [Strategy] Executing KnnStrategy for FM: {self.fm}")
        
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        # 1. 基础配置
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        
        # 2. 解析方法 (如: KNN+Filtering[HSV]) 和 对齐设置
        flow_cfg = cfg_mgr.get_flow(fm)
        if not flow_cfg:
            print(f">>> [Strategy Error] 未在 Flow 找到 {fm} 对应的配置。")
            return
            
        defect_id_method = flow_cfg.defect_identification
        print(f">>> [Strategy] 识别方法: {defect_id_method}")
        
        # 将 Excel 中的 PCA_Alignment 映射给 use_batch_alignment (控制旋转)
        pca_val = getattr(flow_cfg, 'pca_alignment', 'No')
        use_batch_alignment = str(pca_val).strip().lower() in ('yes', 'true', '1')
        self.context['use_batch_alignment'] = use_batch_alignment
        print(f">>> [Strategy] 开启旋转对齐 (PCA_Alignment): {use_batch_alignment}")
        
        # 3. 加载权重并初始化模型 (Detectron 抠图用)
        weights_df = cfg_mgr.get_sheet('Weights')
        from core.pipeline import DefectDetectionPipeline
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
        
        dut_cfg = weights.get('cfg_path_dut') or weights.get('cfg_path')
        dut_weight = weights.get('weights_path_dut') or weights.get('weights_path')
        
        dut_predictor = None
        if dut_cfg and dut_weight:
            print(f">>> [Strategy] Loading DUT Predictor for Alignment: {dut_weight}")
            try:
                dut_predictor = load_detectron_model(dut_cfg, dut_weight, device='cpu')
            except Exception as e:
                print(f"Warning: Failed to load DUT Predictor: {e}")
        
        sam2_predictor = None 
        # 为了支持标准的 Detectron+SAM2 抠图对齐，我们加载全局 SAM2 权重
        sam_cfg = None
        sam_weight = None
        project_root = os.getcwd()
        for _, row in weights_df[weights_df['Product_Gen'] == 'SAM'].iterrows():
            var_name = str(row.get('variable_name', '')).lower()
            if 'config' in var_name or 'cfg' in var_name:
                sam_cfg = os.path.join(project_root, str(row['relative_path']))
            if 'weight' in var_name or 'checkpoint' in var_name or 'pt' in var_name:
                sam_weight = os.path.join(project_root, str(row['relative_path']))
                
        if sam_cfg and sam_weight:
            from utils.wrappers.sam2_wrapper import init_sam2
            print(f">>> [Strategy] Loading SAM2 Predictor for Alignment: {sam_weight}")
            try:
                # 传入 absolute paths
                sam2_predictor = init_sam2(sam_cfg, sam_weight, device='cpu')
            except Exception as e:
                print(f"Warning: Failed to load SAM2 Predictor: {e}")
                
        # 4. 解析结果路径
        output_path_type = str(fm_row['Output_Path']) if fm_row is not None else 'Result'
        if output_path_type == 'Result' or pd.isna(output_path_type):
            res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        else:
            res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        self.res_path = res_path
        
        # Debug 目录
        self.debug_dir = os.path.join(res_path, 'reference')
        is_debug_mode = str(flow_cfg.debug_mode).strip().lower() in ('yes', 'true', '1') if flow_cfg.debug_mode else False
        if is_debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # 5. 读取 KNN 专项配置 和 Scaling 干扰件移除配置
        knn_config_df = cfg_mgr.get_sheet('KNN')
        if knn_config_df.empty:
            print(f">>> [Strategy Error] 未能加载 KNN 配置表。")
            return
            
        scaling_df = cfg_mgr.get_sheet('Scaling')
        feature_scaling_config = {}
        if not scaling_df.empty:
            sc_match = scaling_df[
                (scaling_df['Product'].apply(_norm_knn) == _norm_knn(self.product)) & 
                (scaling_df['Generation'].astype(str).apply(_norm_knn) == _norm_knn(self.generation))
            ]
            if not sc_match.empty:
                sc_row = sc_match.iloc[0]
                for i in range(1, 4):  # 假设最多配置三个 Feature_Remove
                    f_name = sc_row.get(f'Feature_Remove_{i}')
                    f_factor = sc_row.get(f'Feature_Remove_Scaling_Factor_{i}')
                    if pd.notna(f_name) and pd.notna(f_factor):
                        feature_scaling_config[str(f_name).strip()] = float(f_factor)
                        
        # 兼容老代码的遗留调用：
        utils_general.Feature_Scaling_Config = feature_scaling_config
        setattr(utils_general, 'Feature_Scaling_Config', feature_scaling_config)
        print(f">>> [Strategy] 干扰件移除配置 (Feature Scaling): {feature_scaling_config}")
            
        # 6. 开始遍历图片
        if not os.path.exists(self.download_path):
            print(f">>> [Strategy] Download path invalid: {self.download_path}")
            return
            
        parametric_results = []
        for root, dirs, files in os.walk(self.download_path):
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.bmp')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n{'='*50}\n>>> [Strategy KNN] Processing {file} ...\n{'='*50}")
                
                # 6.1: 抠图对齐 (如果配了 dut 模型)
                if dut_predictor:
                    image, DUT_Corrected, Contour_Corrected, alignment_metadata = self.prepare_dut_image(
                        img_path, dut_predictor, sam2_predictor
                    )
                else:
                    # 如果没有 DUT 模型，那就全图跑
                    image = cv2.imread(img_path)
                    DUT_Corrected = image.copy()
                    h, w = image.shape[:2]
                    Contour_Corrected = np.ones((h, w), dtype=np.uint8) * 255
                    alignment_metadata = {}
                    
                if DUT_Corrected is None:
                    continue

                DUT_Corrected_RGB = cv2.cvtColor(DUT_Corrected, cv2.COLOR_BGR2RGB)
                
                # 6.2: 动态匹配 KNN 参数 (计算图片平均颜色来找 Excel 里的行)
                avg_r = np.mean(DUT_Corrected_RGB[:, :, 0])
                avg_g = np.mean(DUT_Corrected_RGB[:, :, 1])
                avg_b = np.mean(DUT_Corrected_RGB[:, :, 2])
                current_avg_rgb = np.array([avg_r, avg_g, avg_b])
                print(f">>> [Strategy] Calculated Avg RGB: {current_avg_rgb}")
                
                # 在 KNN 表中按 Product/FM 过滤
                filtered_df = knn_config_df[
                    (knn_config_df['Product'].apply(_norm_knn) == _norm_knn(self.product)) & 
                    (knn_config_df['Failure Mode'].apply(_norm_knn) == _norm_knn(self.fm))
                ]
                if filtered_df.empty:
                    filtered_df = knn_config_df
                    
                # 算欧氏距离，找最接近的颜色配置
                min_dist = float('inf')
                best_row = None
                for idx, row in filtered_df.iterrows():
                    try:
                        ref_r, ref_g, ref_b = float(row.get('R', 0)), float(row.get('G', 0)), float(row.get('B', 0))
                        ref_rgb = np.array([ref_r, ref_g, ref_b])
                        dist = np.linalg.norm(current_avg_rgb - ref_rgb)
                        if dist < min_dist:
                            min_dist = dist
                            best_row = row
                    except:
                        pass
                        
                if best_row is None:
                    print(">>> [Strategy Error] 无法匹配到对应的 KNN 颜色配置行。")
                    continue
                    
                print(f">>> [Strategy] Selected KNN Row: {best_row.to_dict()}")
                
                # 6.3 提取具体参数
                knn_k1, knn_sel1, knn_k2, knn_sel2 = 3, [3], 2, [1]
                target_rgb_knn = [0, 0, 0]
                defect_min_area = 0.01
                knn_dist_thresh = 0
                
                if 'KNN Setup' in best_row and pd.notna(best_row['KNN Setup']):
                    setup = ast.literal_eval(best_row['KNN Setup'])
                    if len(setup) >= 4:
                        knn_k1, knn_sel1, knn_k2, knn_sel2 = int(setup[0]), [int(setup[1])], int(setup[2]), [int(setup[3])]
                
                if 'TARGET_RGB' in best_row and pd.notna(best_row['TARGET_RGB']):
                    val = best_row['TARGET_RGB']
                    if isinstance(val, str): val = ast.literal_eval(val)
                    target_rgb_knn = list(val)
                    
                if 'DEFECT_MIN_AREA' in best_row and pd.notna(best_row['DEFECT_MIN_AREA']):
                    defect_min_area = float(best_row['DEFECT_MIN_AREA'])
                
                knn_resize_1st = 1.0
                knn_resize_2nd = 1.0
                knn_hole_detection_enabled = True
                knn_slicing_config = None
                
                if 'Resize' in best_row and pd.notna(best_row['Resize']):
                    import re
                    resize_str = str(best_row['Resize']).strip()
                    if '1st=' in resize_str or '2nd=' in resize_str:
                        fm1 = re.search(r'1st\s*=\s*([0-9.]+)', resize_str, re.IGNORECASE)
                        if fm1: knn_resize_1st = float(fm1.group(1))
                        fm2 = re.search(r'2nd\s*=\s*([0-9.]+)', resize_str, re.IGNORECASE)
                        if fm2: knn_resize_2nd = float(fm2.group(1))
                    else:
                        try:
                            knn_resize_1st = knn_resize_2nd = float(resize_str)
                        except ValueError:
                            pass
                            
                if 'Hole Detection' in best_row and pd.notna(best_row['Hole Detection']):
                    hd_str = str(best_row['Hole Detection']).strip().lower()
                    if hd_str == 'no':
                        knn_hole_detection_enabled = False
                
                if 'Slicing' in best_row and pd.notna(best_row['Slicing']):
                    import re
                    slicing_str = str(best_row['Slicing']).strip()
                    if slicing_str.lower() != 'no' and slicing_str != '':
                        knn_slicing_config = {'x_ranges': [], 'y_ranges': []}
                        s = slicing_str.strip('[]')
                        x_match = re.search(r'X\[(.*?)\]', s, re.IGNORECASE)
                        y_match = re.search(r'Y\[(.*?)\]', s, re.IGNORECASE)
                        
                        def parse_sr(m_str):
                            r = []
                            for p in m_str.replace(' ', '').split(','):
                                if ':' in p:
                                    try: r.append((float(p.split(':')[0]), float(p.split(':')[1])))
                                    except: pass
                            return r
                            
                        if x_match: knn_slicing_config['x_ranges'] = parse_sr(x_match.group(1))
                        if y_match: knn_slicing_config['y_ranges'] = parse_sr(y_match.group(1))
                        if not knn_slicing_config['x_ranges'] and not knn_slicing_config['y_ranges']:
                            knn_slicing_config = None
                            
                # Apply Slicing Mask
                DUT_Corrected_RGB_sliced = DUT_Corrected_RGB.copy()
                if knn_slicing_config:
                    h_s, w_s = DUT_Corrected_RGB.shape[:2]
                    mask_x = np.zeros((h_s, w_s), dtype=np.uint8)
                    mask_y = np.zeros((h_s, w_s), dtype=np.uint8)
                    if knn_slicing_config['x_ranges']:
                        for mr, mxr in knn_slicing_config['x_ranges']:
                            mask_x[:, int(w_s*mr):int(w_s*mxr)] = 255
                    if knn_slicing_config['y_ranges']:
                        for mr, mxr in knn_slicing_config['y_ranges']:
                            mask_y[int(h_s*mr):int(h_s*mxr), :] = 255
                            
                    if knn_slicing_config['x_ranges'] and knn_slicing_config['y_ranges']:
                        mask_keep = cv2.bitwise_and(mask_x, mask_y)
                    elif knn_slicing_config['x_ranges']: mask_keep = mask_x
                    else: mask_keep = mask_y
                    
                    mask_keep_3ch = cv2.cvtColor(mask_keep, cv2.COLOR_GRAY2RGB)
                    DUT_Corrected_RGB_sliced = cv2.bitwise_and(DUT_Corrected_RGB, mask_keep_3ch)
                    if Contour_Corrected is not None:
                        Contour_Corrected = cv2.bitwise_and(Contour_Corrected, mask_keep)
                        
                # 6.4 执行 KNN
                print(">>> [Strategy] Executing KNN algorithm...")
                filename_base = os.path.splitext(file)[0]
                
                # knn_utils 需要的是 cv2.findContours() 返回的那种 list of points，
                # 但如果我们传的是 numpy 的 mask (Contour_Corrected)，需要先转换一下。
                scaled_contour = None
                if Contour_Corrected is not None:
                    contours, _ = cv2.findContours(Contour_Corrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # 找最大的那个并确保格式正确 (N, 1, 2) dtype=int32，这样 fillPoly 才不会报错 (-215:Assertion failed)
                        scaled_contour = max(contours, key=cv2.contourArea)
                        scaled_contour = scaled_contour.reshape(-1, 1, 2).astype(np.int32)
                
                # KNN 要求将 ignore_mask 和相关参数匹配上
                knn_results, all_holes_rgb, anomaly_detection, selected_regions_mask = knn_utils.process_image_complete(
                    image_array=DUT_Corrected_RGB_sliced,
                    first_knn_cluster_qty=knn_k1,
                    second_knn_cluster_qty=knn_k2,
                    first_knn_selected=knn_sel1,
                    second_knn_selected=knn_sel2,
                    target_rgb=target_rgb_knn,
                    KNN_distance_threshold=knn_dist_thresh,
                    defect_min_area=defect_min_area,
                    rgb_distance_threshold=None,
                    input_scaled_contour=scaled_contour,  # 传入轮廓点，而不是 mask 矩阵
                    contour_shrink_ratio=1.0,
                    height_lower_ratio=0.0,  # 设为 0，意味着从顶部开始扫描，不进行 Y 轴裁切
                    output_dir=self.debug_dir if is_debug_mode else self.res_path, # 改为写入 reference/ 文件夹
                    image_name=filename_base,
                    resize_scale_1st=knn_resize_1st,
                    resize_scale_2nd=knn_resize_2nd,
                    enable_hole_detection=knn_hole_detection_enabled
                )
                
                # 决定使用哪个 Mask 作为基准（如果稀疏异常检测成功，优先用它）
                if selected_regions_mask is not None and cv2.countNonZero(selected_regions_mask) > 0:
                    print(">>> [Strategy] Using selected_regions_mask (Anomaly Detection) as primary defect mask.")
                    defect_mask_bin = selected_regions_mask
                elif all_holes_rgb is not None:
                    print(">>> [Strategy] Using all_holes_rgb as primary defect mask.")
                    if len(all_holes_rgb.shape) == 3:
                        gray = cv2.cvtColor(all_holes_rgb, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = all_holes_rgb
                    _, defect_mask_bin = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                else:
                    print(">>> [Strategy] KNN returned no valid mask.")
                    continue
                
                # 如果有后续过滤 (比如 Filtering[HSV]) 可以继续往下加逻辑
                if 'Filtering' in defect_id_method:
                    print(f">>> [Strategy] 发现后续 Filtering 过滤规则，开始执行...")
                    filtering_df = cfg_mgr.get_sheet('Filtering')
                    if not filtering_df.empty:
                        # 过滤出符合 Product 和 Failure Mode 的行
                        f_df = filtering_df[
                            (filtering_df['Product'].apply(_norm_knn) == _norm_knn(self.product)) & 
                            (filtering_df['Failure Mode'].apply(_norm_knn) == _norm_knn(self.fm))
                        ]
                        if f_df.empty: f_df = filtering_df
                        
                        # 解析操作，例如 "KNN+Filtering[HSV]"
                        import re
                        ops = []
                        match = re.search(r'Filtering\[(.*?)\]', defect_id_method, re.IGNORECASE)
                        if match:
                            params = [p.strip() for p in match.group(1).split(',')]
                            ops = utils_general.parse_filtering_operations(params)
                            
                        if ops:
                            defect_mask_bin, _, _, _, _ = utils_general.apply_filtering(
                                mask_filtered=defect_mask_bin > 127,  # 转成 boolean 传入
                                original_img=DUT_Corrected_RGB,
                                operations=ops,
                                filtering_df=f_df,
                                save_dir=self.debug_dir if is_debug_mode else self.res_path,
                                image_path=img_path,
                                contour_image=Contour_Corrected,
                                defect_id_method=self.fm,
                                mask_default=Contour_Corrected,
                                preferred_color_name=best_row.get('color_name')
                            )
                            # 转回 uint8
                            if defect_mask_bin.dtype == bool:
                                defect_mask_bin = (defect_mask_bin.astype(np.uint8) * 255)
                                
                # 与原始物件的轮廓做最后交集，防止溢出
                if Contour_Corrected is not None:
                    _, contour_mask = cv2.threshold(Contour_Corrected, 1, 255, cv2.THRESH_BINARY)
                    defect_mask_bin = cv2.bitwise_and(defect_mask_bin, contour_mask)
                
                print(f">>> [Strategy] Found final defect mask area: {cv2.countNonZero(defect_mask_bin)}")
                
                                # 保存叠加结果
                filename_base = os.path.splitext(file)[0]
                
                # --- [新增] 创建 Inferred Pic 文件夹 ---
                inferred_pic_dir = os.path.join(self.res_path, "Inferred Pic")
                os.makedirs(inferred_pic_dir, exist_ok=True)
                
                overlay = cv_ops.create_overlay_image(DUT_Corrected, defect_mask_bin, color=(0, 0, 255), transparency=0.5)
                out_path = os.path.join(inferred_pic_dir, f"{filename_base}_knn_overlay.jpg")
                cv2.imwrite(out_path, overlay)
                
                if is_debug_mode:
                    cv2.imwrite(os.path.join(self.debug_dir, f"{filename_base}_mask.jpg"), defect_mask_bin)
                    
                # --- [新增] 收集 Parametric 数据 ---
                contour_area_val = cv2.countNonZero(Contour_Corrected) if Contour_Corrected is not None else 1.0
                try:
                    from utils import output_ops
                    # In KNN, the binning image source needs to be passed correctly if Gray Scale is enabled
                    gray_params = getattr(utils_general, 'Gray_Scale_Params', {})
                    kwargs = {
                        'defect_cnt': None, 
                        'dut_dims': DUT_Corrected.shape[:2] if DUT_Corrected is not None else (1280, 1280),
                        'ref_params': getattr(utils_general, 'Reference_Params', {}),
                        'output_config': getattr(utils_general, 'Output_Config', []),
                        'contour_area_val': contour_area_val,
                        'ref_area': 1.0,
                        'ref_found': True,
                        'defect_mask': defect_mask_bin,
                        'is_line_shape': False,
                        'image': DUT_Corrected,
                        'gray_scale_params': gray_params
                    }
                    if gray_params.get('Enabled'):
                        # Pass the color image for binning as image argument to avoid img_bin_src TypeError in output_ops
                        kwargs['image'] = DUT_Corrected_RGB
                    
                    dims_res = output_ops.calculate_parametric_dimensions(**kwargs)
                    
                    defect_pct_val = 0
                    if contour_area_val > 0:
                        defect_pct_val = (cv2.countNonZero(defect_mask_bin) / contour_area_val) * 100
                        
                    p_dict = {
                        'Filename': file,
                        'Dir': self.download_path,
                        'Defect_Type': self.fm,
                        'Detected Defect': self.fm,
                        'Defect Pct': f"{round(defect_pct_val, 2)}%",
                        'Reference_Status': 'Found'
                    }
                    # 针对 Combined 模式，保存 .npy 并将路径填入 Contour
                    if str(self.defect_output_df).strip().lower() == 'combined' or str(getattr(self, 'defect_output_df', pd.DataFrame())).strip().lower() == 'combined' or True: # We just do it anyway and let format_parametric_output_df handle it or override it
                        try:
                            import numpy as np
                            npy_path = os.path.join(self.res_path, f"{filename_base}_combined_final_mask.npy")
                            np.save(npy_path, defect_mask_bin)
                            print(f">>> [Strategy] Saved .npy mask to {npy_path}")
                            # 在 Combined 模式下，旧代码把这个文件的路径塞进了 Contour 列
                            p_dict['Contour'] = npy_path
                        except Exception as e:
                            print(f"Error saving .npy mask: {e}")
                    else:
                        # 生成点坐标 Contour
                        try:
                            import cv2
                            import numpy as np
                            cnts_def, _ = cv2.findContours(defect_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if cnts_def:
                                largest_cnt = max(cnts_def, key=cv2.contourArea)
                                if len(largest_cnt) > 0:
                                    points = []
                                    for pt in largest_cnt:
                                        points.append([int(pt[0][0]), int(pt[0][1])])
                                    p_dict['Contour'] = str(points)
                        except Exception as e:
                            print(f"Error generating Contour: {e}")
                    p_dict.update(dims_res)
                    parametric_results.append(p_dict)
                except Exception as e:
                    print(f">>> [Strategy Error] Parametric calculation failed: {e}")
                    parametric_results.append({'Filename': file, 'Defect_Type': self.fm})
                    
        # --- [新增] 生成 Excel 表格 ---
        if parametric_results:
            try:
                from utils import output_ops
                # 尝试获取 Output 表格配置
                output_df = cfg_mgr.get_sheet('Output')
                defect_output_df = output_df if not output_df.empty else pd.DataFrame()
                
                df = output_ops.build_parametric_output_df(
                    data=parametric_results,
                    failure_mode=self.fm,
                    defect_output_format=defect_output_df
                )
                if not df.empty:
                    output_ops.parametric_output(
                        main_folder_path=self.res_path,
                        sub_folder_path="",
                        file_name="Parametric_Output.xlsx",
                        sheet_name='Sheet1',
                        df=df
                    )
                print(f">>> [Strategy] Parametric Excel saved successfully with {len(parametric_results)} rows.")
            except Exception as e:
                print(f">>> [Strategy Error] Failed to save Excel: {e}")

        print("\\n>>> [Strategy] KNN Strategy Execution Completed! <<<")

