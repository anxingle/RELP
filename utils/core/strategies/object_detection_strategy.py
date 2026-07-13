import os
import cv2
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.detectron_wrapper import load_detectron_model
from utils.detectron_ops import visualize_all_detections
import utils.utils_general as ug

class ObjectDetectionStrategy(AnalysisStrategy):
    """
    专门针对基础目标检测的策略。
    读取配置好的图片，用 Detectron2 跑框并保存结果。
    如果配置了 DUT Detectron 和 SAM 权重，则会先找出大物件并抠图涂黑背景，然后只在截取的局图上跑缺陷检测框。
    """
    def execute(self) -> None:
        print(f">>> [Strategy] Executing ObjectDetectionStrategy for FM: {self.fm}")
        
        # 1. 解析基础配置
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        
        # 2. 解析模型权重
        weights_df = cfg_mgr.get_sheet('Weights')
        flow_df = cfg_mgr.get_sheet('Flow')
        
        # 从 Flow 表中手动解析 Defect Detection Setting 
        general_defect_key = self.context.get('general_defect_key')
        if not general_defect_key and not flow_df.empty:
            from utils.base_utils import _norm
            dds_cols = [c for c in flow_df.columns if _norm(c) in ('defectdetectionsetting',)]
            if dds_cols:
                flow_row = flow_df[flow_df['Failure Mode'] == fm]
                if not flow_row.empty:
                    val = flow_row.iloc[0][dds_cols[0]]
                    if pd.notna(val) and str(val).lower() != 'nan':
                        raw_val = str(val).replace('[Save]', '').replace('[save]', '').strip()
                        general_defect_key = raw_val
                        self.context['general_defect_key'] = general_defect_key
        
        from core.pipeline import DefectDetectionPipeline
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
        
        cfg_path_dut = weights.get('cfg_path_dut')
        weights_path_dut = weights.get('weights_path_dut')
        # 尝试通过 Defect Detection Setting 寻找第二层缺陷权重
        general_defect_key = self.context.get('general_defect_key')
        cfg_path = None
        weights_path = None
        
        if general_defect_key:
            from utils.base_utils import _norm
            type_col = next((c for c in weights_df.columns if _norm(c) in ('type', 'model_type')), None)
            var_col = next((c for c in weights_df.columns if _norm(c) in ('variablename', 'variable_name')), None)
            path_col = next((c for c in weights_df.columns if _norm(c) in ('relativepath', 'relative_path')), None)
            
            if type_col and var_col and path_col:
                # 检查 Type 中是否包含 general_defect_key，因为可能存在 Screen Gen 这样的配置
                # 为了防止大小写或者空格的问题，使用 contains 或者 lower() == lower()
                matches = weights_df[weights_df[type_col].astype(str).apply(_norm) == _norm(general_defect_key)]
                
                # 回退：如果没精确匹配上，试试包含
                if matches.empty:
                     matches = weights_df[weights_df[type_col].astype(str).str.contains(str(general_defect_key), case=False, na=False)]
                     
                for _, row in matches.iterrows():
                    var_name = _norm(row[var_col])
                    path_val = row[path_col]
                    if pd.notna(path_val):
                        # 如果没有 product_gen 等限制，我们可以直接取
                        if not os.path.isabs(path_val):
                            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                            path_val = os.path.join(project_root, path_val)
                            
                        if 'config' in var_name or 'cfg' in var_name:
                            cfg_path = path_val
                        elif 'weight' in var_name or 'pth' in var_name:
                            weights_path = path_val
                            
        # 降级逻辑：如果只配置了一组权重（只有 DUT），
        # 我们依然启用 SAM 抠大图模式！因为目标就是“先把物体抠出来”。
        # 只要找到了任何一个检测器，我们就把它当做 DUT 找大件，扣出局部图后，如果没有专门的小图缺陷模型，就不跑小图的画框，直接保存纯净版。
        # 如果同时有俩模型，就在纯净版上继续跑第二层画框。
        use_crop_flow = True 
        
        if cfg_path_dut and weights_path_dut and cfg_path and weights_path:
            print(f">>> [Strategy Info] 检测到双重权重配置！启用 【DUT找大件 + SAM抠大图 + {general_defect_key} 局部二层画框】 模式。")
        else:
            print(">>> [Strategy Info] 仅检测到单组权重配置！启用 【DUT找大件 + SAM抠大图 (不进行二次画框)】 模式。")
            
        if not cfg_path_dut or not weights_path_dut:
            print(f">>> [Strategy Error] 未在 Weights 表中找到任何 {fm} 的 Detectron 权重配置 (既无 dut 也无 defect)。")
            return
            
        # 3. 加载模型
        try:
            print(f">>> [Strategy] Loading DUT detector. CFG: {cfg_path_dut}, WEIGHTS: {weights_path_dut}")
            
            # 使用更纯粹、更安全的数据集名称重置方式：
            # 在加载模型前，利用 Python 的 mock 机制拦截 MetadataCatalog 中 'defect_test' 的创建与获取，
            # 从而把 DUT 探测器注册的数据集名称和元数据彻底隔离到另一个独立的 Key 'dut_dataset_key' 中。
            from detectron2.data import MetadataCatalog
            original_get = MetadataCatalog.get
            
            def safe_get(name):
                # 只要是在加载 DUT 模型时发起的对 'defect_test' 的请求，我们都重定向到 'dut_dataset_key'
                if name == "defect_test":
                    return original_get("dut_dataset_key")
                return original_get(name)
                
            MetadataCatalog.get = safe_get
            
            # 安全加载，此时内部注册时会自动写入 'dut_dataset_key'，不会污染 'defect_test'
            dut_predictor = ug.load_detectron2_model(cfg_path_dut, weights_path_dut, score_thresh=0.5)
            
            # 加载完成后立即恢复原样，确保后续缺陷检测器（Defect Detector）正常获取并写入 'defect_test'
            MetadataCatalog.get = original_get
            
        except Exception as e:
            print(f">>> [Strategy Error] 加载 DUT Detectron 模型失败: {e}")
            return
            
        detector = None
        if cfg_path and weights_path:
            try:
                print(f">>> [Strategy] Loading Defect detector. CFG: {cfg_path}, WEIGHTS: {weights_path}")
                # 如果这个模型是 Instance Segmentation 模型（由于有 IS_CFG.pickle），
                # 兼容通用加载（使用 utils_general.load_detectron2_model 以防止 load_detectron_model 崩溃）
                if 'IS_CFG' in cfg_path:
                    detector = ug.load_detectron2_model(cfg_path, weights_path, score_thresh=0.5)
                else:
                    detector = load_detectron_model(cfg_path, weights_path, device='cpu')
            except Exception as e:
                print(f">>> [Strategy Error] 加载 Defect Detectron 模型失败: {e}")
                return
            
        # 加载 SAM2
        sam_cfg = weights.get('sam_config')
        sam_ckpt = weights.get('sam_weights')
        
        if not sam_ckpt:
            sam_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "weights/sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
            sam_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "weights/sam2/checkpoints/sam2.1_hiera_tiny.pt")

        print(f">>> [Strategy] Loading SAM2 model...")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = ug._real_init_sam2(sam_cfg, sam_ckpt, device=device_str)
        sam2_predictor = ug.get_sam2_predictor(sam2_model)
        if not sam2_predictor:
            print(">>> [Strategy Error] SAM2 Initialization failed!")
            return
            
        # 尝试加载 class_names
        class_names = None
        weights_dir = None
        if detector:
             weights_dir = os.path.dirname(weights_path)
        else:
             weights_dir = os.path.dirname(weights_path_dut)
        possible_dirs = [weights_dir, os.path.dirname(weights_dir), os.path.join(weights_dir, 'DUT'), os.path.join(os.path.dirname(weights_dir), 'DUT')]
        class_names = None
        
        for search_dir in possible_dirs:
            if class_names: break
            classes_txt_path = os.path.join(search_dir, 'classes.txt')
            if os.path.exists(classes_txt_path):
                with open(classes_txt_path, 'r', encoding='utf-8') as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                print(f">>> [Strategy] 从 {classes_txt_path} 加载了 {len(class_names)} 个类别名称: {class_names}")
                break
                
            classes_rtf_path = os.path.join(search_dir, 'classes.rtf')
            if os.path.exists(classes_rtf_path):
                import re
                try:
                    with open(classes_rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    text_part = text.split(r'\cf0')[-1].replace('}', '')
                    matches = re.findall(r'(\d+):\s*([^\\]+)', text_part)
                    if matches:
                        class_dict = {int(k): v.strip() for k, v in matches}
                        max_idx = max(class_dict.keys())
                        class_names = [class_dict.get(i, f"Class_{i}") for i in range(max_idx + 1)]
                        print(f">>> [Strategy] 从 {classes_rtf_path} 成功解析了 {len(class_names)} 个类别名称: {class_names}")
                        break
                except Exception as e:
                    pass
                    
        if not class_names:
            print(f">>> [Strategy Info] 未找到 classes.txt 或 classes.rtf，将使用默认类别名 (Class_0, Class_1...)。")
            
        # 4. 解析生成结果路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        if self.context.get('res_path'):
            res_path = self.context.get('res_path')
        else:
            output_path_type = str(fm_row['Output_Path']).strip() if fm_row is not None and pd.notna(fm_row.get('Output_Path')) else 'Result'
            if output_path_type.lower() == 'nan': output_path_type = 'Result'
            if output_path_type == 'Result':
                res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            else:
                res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        if not os.path.exists(res_path):
            os.makedirs(res_path)
            
        download_path = self.download_path
        if pd.isna(download_path) or not os.path.exists(download_path):
            print(f">>> [Strategy Error] Download path does not exist or empty: {download_path}")
            return
            
        print(f">>> [Strategy] Starting directory traversal on: {download_path}")
        
        # 5. 遍历并检测
        processed_count = 0
        for root, dirs, files in os.walk(download_path):
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n  -> Processing {file} ...")
                
                rel_path = os.path.relpath(root, self.download_path)
                if rel_path == '.': rel_path = ''
                curr_res_dir = os.path.join(res_path, rel_path)
                os.makedirs(curr_res_dir, exist_ok=True)
                
                image = cv2.imread(img_path)
                if image is None:
                    print(f"     [Error] Cannot read image {img_path}")
                    continue
                    
                if not use_crop_flow:
                    pass # 不再使用
                else:
                    # 新逻辑：强制执行 DUT -> SAM -> 局部图像
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # (1) 原图找大框
                    tape_boxes, tape_classes = ug.detect_bounding_boxes(image, dut_predictor)
                    if len(tape_boxes) == 0:
                        print("     No DUT objects found in image. Skipping.")
                        continue
                        
                    print(f"     Found {len(tape_boxes)} object bounds for cropping.")
                    
                    # (1.5) 读取配置表中的 Bbox_Expand_Ratio 
                    bbox_expand_ratio = 1.0
                    scaling_df = pd.read_excel('RELP_Configuration.xlsx', sheet_name='Scaling')
                    if not scaling_df.empty:
                        sc_row = scaling_df[scaling_df['Failure Mode'] == fm]
                        if not sc_row.empty:
                            # 1. 尝试找专门的 bbox_expand_ratio 字段
                            for col in scaling_df.columns:
                                if str(col).lower().replace(' ', '_') == 'bbox_expand_ratio':
                                    val = sc_row.iloc[0][col]
                                    if pd.notna(val):
                                        try: 
                                            bbox_expand_ratio = float(val)
                                            print(f"     [Config] Found Bbox_Expand_Ratio={bbox_expand_ratio} in Scaling sheet.")
                                        except: pass
                                    break
                                    
                    # 对框进行外扩计算
                    if bbox_expand_ratio != 1.0:
                        img_h, img_w = image.shape[:2]
                        expanded_boxes = []
                        for box in tape_boxes:
                            x1, y1, x2, y2 = box
                            w = x2 - x1
                            h = y2 - y1
                            cx = x1 + w / 2
                            cy = y1 + h / 2
                            new_w = w * bbox_expand_ratio
                            new_h = h * bbox_expand_ratio
                            new_x1 = max(0, cx - new_w / 2)
                            new_y1 = max(0, cy - new_h / 2)
                            new_x2 = min(img_w - 1, cx + new_w / 2)
                            new_y2 = min(img_h - 1, cy + new_h / 2)
                            expanded_boxes.append([new_x1, new_y1, new_x2, new_y2])
                        tape_boxes = np.array(expanded_boxes, dtype=np.float32)
                        print(f"     [Scaling] Expanded SAM input bounding boxes by {bbox_expand_ratio}x.")
                    
                    # (2) SAM抠出精确边界并涂黑背景
                    sam2_predictor.set_image(image_rgb)
                    all_detections = ug.extract_detections(
                        sam2_predictor, image_rgb, tape_boxes, 
                        tape_classes=tape_classes,
                        mask_scaling_factor=1.0, 
                    )
                    
                    if not all_detections:
                        print("     SAM failed to extract any masks.")
                        continue
                        
                    all_detections_sorted = sorted(all_detections, key=lambda d: d["area"], reverse=True)
                    # 仅保留面积最大的那个检测到的实例（DUT）
                    if len(all_detections_sorted) > 1:
                        print(f"     [Optimization] Found {len(all_detections_sorted)} DUT instances. Retaining only the largest one (Area: {all_detections_sorted[0]['area']:.0f}).")
                        all_detections_sorted = [all_detections_sorted[0]]
                    
                    # (3) 遍历切割并运行检测
                    for idx, det in enumerate(all_detections_sorted):
                        print(f"     -> Processing Instance {idx+1}/{len(all_detections_sorted)} (Area: {det['area']:.0f})")
                        
                        clean_image = image.copy()
                        clean_image[det["mask"] == 0] = 0
                        
                        orig_box = det["box"]
                        orig_x_min, orig_y_min, orig_x_max, orig_y_max = map(int, orig_box)
                        img_h, img_w = image.shape[:2]
                        
                        x_min = max(0, min(orig_x_min, img_w-1))
                        x_max = max(0, min(orig_x_max, img_w-1))
                        y_min = max(0, min(orig_y_min, img_h-1))
                        y_max = max(0, min(orig_y_max, img_h-1))
                        
                        img_crop = clean_image[y_min:y_max+1, x_min:x_max+1]
                        if img_crop.size == 0:
                            print("     [Error] Crop size is 0.")
                            continue
                            
                        # 在局部切割图上运行缺陷检测（如果有的话）
                        if detector:
                            debug_img, info = visualize_all_detections(img_crop, detector, class_names=class_names, score_thresh=0.5)
                            print(f"        Found {len(info)} defect objects on cropped instance.")
                            
                            # 解析统计数据用于 Parametric Output
                            if info:
                                output_df_sheet = cfg_mgr.get_sheet('Output')
                                dof = ''
                                if not output_df_sheet.empty:
                                    out_row = output_df_sheet[output_df_sheet['Failure Mode'] == self.fm]
                                    if not out_row.empty:
                                        dof = str(out_row.iloc[0].get('Defect Output Format', '')).strip().lower()

                                if dof == 'individual':
                                    # Output format requests individual rows for each defect
                                    data_list = []
                                    crop_h, crop_w = img_crop.shape[:2]
                                    for idx_d, d in enumerate(info):
                                        cls_name = d.get('class_name', 'Unknown_Defect')
                                        bbox = d.get('bbox', [0, 0, 0, 0])
                                        x1, y1, x2, y2 = bbox
                                        
                                        # Calculate normalized coordinates
                                        w = x2 - x1
                                        h = y2 - y1
                                        cx = x1 + w / 2.0
                                        cy = y1 + h / 2.0
                                        
                                        # Normalize relative to the cropped image
                                        cx_norm = cx / crop_w if crop_w > 0 else 0
                                        cy_norm = cy / crop_h if crop_h > 0 else 0
                                        w_norm = w / crop_w if crop_w > 0 else 0
                                        h_norm = h / crop_h if crop_h > 0 else 0
                                        
                                        # Generate Contour format (4 corners of the bbox in normalized coords)
                                        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
                                        x1_n = max(0.0, min(1.0, x1 / crop_w)) if crop_w > 0 else 0
                                        y1_n = max(0.0, min(1.0, y1 / crop_h)) if crop_h > 0 else 0
                                        x2_n = max(0.0, min(1.0, x2 / crop_w)) if crop_w > 0 else 0
                                        y2_n = max(0.0, min(1.0, y2 / crop_h)) if crop_h > 0 else 0
                                        
                                        contour_points = [
                                            [round(x1_n, 4), round(y1_n, 4)],
                                            [round(x2_n, 4), round(y1_n, 4)],
                                            [round(x2_n, 4), round(y2_n, 4)],
                                            [round(x1_n, 4), round(y2_n, 4)]
                                        ]
                                        
                                        row_data = {
                                            'Filename': file,
                                            'Dir': os.path.basename(os.path.dirname(img_path)),
                                            'Image Name': file,
                                            'Defect Pct': f"{(w_norm * h_norm) * 100:.2f}%", 
                                            'Defect_Type': cls_name,
                                            'Defect_ID': f"Instance_{idx+1}_Defect_{idx_d+1}",
                                            'Contour': str(contour_points),
                                            'cx': round(cx_norm, 4),
                                            'cy': round(cy_norm, 4),
                                            'w': round(w_norm, 4),
                                            'h': round(h_norm, 4)
                                        }
                                        data_list.append(row_data)
                                        
                                    df_temp = pd.DataFrame(data_list)
                                    print(f"        -> Recorded Parametric Output: {len(data_list)} individual defects with cx, cy, w, h columns.")
                                else:
                                    # Combined format (default)
                                    defect_counts = {}
                                    for d in info:
                                        cls_name = d.get('class_name', 'Unknown_Defect')
                                        defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1
                                    
                                    defect_summary = ", ".join([f"{k}:{v}" for k, v in defect_counts.items()])
                                    
                                    data_list = [{
                                        'Image Name': file,
                                        'Defect Pct': 1.0, 
                                        'Detected Defect': defect_summary
                                    }]
                                    df_temp = pd.DataFrame(data_list)
                                    print(f"        -> Recorded Parametric Output: {defect_summary}")
                                
                                from utils.output_ops import build_parametric_output_df, parametric_output
                                df_formatted = build_parametric_output_df(
                                    data=df_temp,
                                    failure_mode=self.fm,
                                    defect_output_format=output_df_sheet,
                                    gray_scale_params={},
                                    source_path=img_path,
                                    general_defect_key=self.context.get('general_defect_key')
                                )
                                parametric_output(
                                    main_folder_path=os.path.dirname(res_path),
                                    sub_folder_path=os.path.basename(res_path),
                                    file_name=f"{self.product}_{self.generation}_{self.fm}_Parametric_Output.xlsx",
                                    sheet_name="Parametric_Output",
                                    df=df_formatted
                                )
                        else:
                            debug_img = img_crop
                            print("        [No second model] Skipping secondary defect bbox detection, saving clean crop.")
                        
                        # 保存局部图
                        suffix = os.path.splitext(file)[1]
                        
                        # 判断后缀：画了框的叫 crop_bbox，没画框的叫 clean_crop
                        name_postfix = "crop_bbox" if detector else "clean_crop"
                        
                        if len(all_detections_sorted) == 1:
                            out_name = f"{os.path.splitext(file)[0]}_{name_postfix}{suffix}"
                        else:
                            out_name = f"{os.path.splitext(file)[0]}_Instance{idx+1}_{name_postfix}{suffix}"
                            
                        out_file_path = os.path.join(curr_res_dir, out_name)
                        cv2.imwrite(out_file_path, debug_img)

                processed_count += 1
                
        print(f">>> [Strategy] Execution Completed! Processed {processed_count} images. Results saved to {res_path}")
