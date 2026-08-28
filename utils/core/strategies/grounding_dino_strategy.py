import os
import cv2
import pandas as pd
import numpy as np
import re
from typing import Dict, Any

from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.grounding_dino_wrapper import load_grounding_dino_model

class GroundingDinoStrategy(AnalysisStrategy):
    """
    独立且零侵入的 Grounding DINO 策略层。
    完全根据 Flow 表中的配置来加载模型并执行自然语言检测。
    """
    def execute(self) -> None:
        print(f">>> [ROUTER] Routing '{self.fm}' to Modern GroundingDINO implementation! 🚀")
        
        cfg_mgr = ConfigManager()
        fm = self.fm
        
        # 1. 解析基础路径
        fm_df = cfg_mgr.get_sheet('Failure Mode')
        fm_row = fm_df[fm_df['Failure Mode'] == fm].iloc[0] if not fm_df.empty else None
        
        self.product = str(fm_row['Product']) if fm_row is not None else self.product
        self.generation = str(fm_row['Generation']) if fm_row is not None else self.generation
        self.download_path = str(fm_row['Pic_Path']) if fm_row is not None else self.download_path
        if not os.path.isabs(self.download_path):
            self.download_path = os.path.abspath(os.path.join(os.getcwd(), self.download_path))
        
        # 2. 解析 Prompt 和模型路径
        flow_df = cfg_mgr.get_sheet('Flow')
        prompt_str = "defect."
        target_ratio = None
        self.top_k = None
        
        if not flow_df.empty:
            flow_row = flow_df[flow_df['Failure Mode'] == fm]
            if not flow_row.empty:
                defect_id_method = str(flow_row.iloc[0].get('Defect Identification', ''))
                # 解析方括号里的内容 GroundingDino['Eclipictical',ratio = 2:1,Top[1]]
                match = re.search(r'\[(.*)\]', defect_id_method)
                if match:
                    full_params = match.group(1).strip()
                    parts = full_params.split(',')
                    prompt_str = parts[0].strip().strip("'").strip('"')
                    
                    # 提取额外的参数，如 ratio=2:1
                    for part in parts[1:]:
                        part_clean = part.strip().lower()
                        if 'ratio' in part_clean:
                            ratio_val_str = part_clean.split('=')[-1].strip()
                            if ':' in ratio_val_str:
                                try:
                                    num, den = ratio_val_str.split(':')
                                    target_ratio = float(num) / float(den)
                                except:
                                    pass
                            else:
                                try:
                                    target_ratio = float(ratio_val_str)
                                except:
                                    pass
                        elif 'top' in part_clean:
                            top_match = re.search(r"top(?:_oppt)?\[(\d+)\]", part_clean, re.IGNORECASE)
                            if top_match:
                                self.top_k = int(top_match.group(1))
                        
        if not prompt_str.endswith('.'):
            prompt_str += '.'
            
        print(f">>> [Strategy Info] 解析出的自然语言 Prompt: '{prompt_str}'")
        if target_ratio is not None:
            print(f">>> [Strategy Info] 要求 BBox 宽高比/高宽比过滤阈值: >= {target_ratio:.2f}")
        if self.top_k is not None:
            print(f">>> [Strategy Info] 要求 Top-K 概率过滤: 取前 {self.top_k} 个")

        # 从 Weights 提取路径
        weights_df = cfg_mgr.get_sheet('Weights')
        model_path = None
        
        # 匹配用户填写的 "Grounding Dino" 行
        if not weights_df.empty:
            matches = weights_df[weights_df.iloc[:, 0].astype(str).str.contains('Grounding', case=False, na=False)]
            if not matches.empty:
                # 找 path 列
                from utils.base_utils import _norm
                path_col = next((c for c in weights_df.columns if _norm(c) in ('relativepath', 'relative_path', 'path')), None)
                if path_col:
                    raw_path = matches.iloc[0][path_col]
                    if pd.notna(raw_path):
                        model_path = str(raw_path).strip()
        
        # 补全绝对路径
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.getcwd(), model_path))
            
        if not model_path or not os.path.exists(model_path):
            print(f">>> [Strategy Error] Weights 表中未找到有效的 Grounding Dino 模型路径: {model_path}")
            print("尝试使用默认缓存路径...")
            model_path = os.path.expanduser("~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71")
            if not os.path.exists(model_path):
                return
                
        # 3. 加载 Grounding Dino
        print(f">>> [Strategy] Loading Grounding DINO model from {model_path}")
        try:
            predictor = load_grounding_dino_model(model_id=model_path, box_threshold=0.25)
        except Exception as e:
            print(f">>> [Strategy Error] 模型加载失败: {e}")
            return
            
        # 4. 解析结果保存目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        output_path_type = str(fm_row['Output_Path']).strip() if fm_row is not None and pd.notna(fm_row.get('Output_Path')) else 'Result'
        if output_path_type.lower() == 'nan': output_path_type = 'Result'
        
        if output_path_type == 'Result':
            res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        else:
            res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        inferred_pic_dir = os.path.join(res_path, 'Inferred Pic')
        os.makedirs(inferred_pic_dir, exist_ok=True)
        
        if not os.path.exists(self.download_path):
            print(f">>> [Strategy Error] 图片路径不存在: {self.download_path}")
            return
            
        # 解析 Post Processing
        ops_to_run = []
        if not flow_df.empty:
            flow_row = flow_df[flow_df['Failure Mode'] == fm]
            if not flow_row.empty:
                post_method = flow_row.iloc[0].get('Post Processing', '')
                if pd.notna(post_method) and str(post_method).lower() != 'nan':
                    from utils.utils_general import parse_defect_id_method
                    ops_to_run.extend(parse_defect_id_method(str(post_method)))
        
        import utils.filtering_ops as fo
        excel_path = "RELP_Configuration.xlsx" 
        filtering_df = fo.load_and_filter_filtering_sheet(excel_path, self.product, self.generation, fm, None, verbose=False)
        
        # 5. 执行推理
        processed_count = 0
        for root, dirs, files in os.walk(self.download_path):
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"\n  -> Processing {file} ...")
                
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # ---------------------------------------------------------
                # 独立操作 1: Grounding DINO (仅处理原图，输出红框)
                # ---------------------------------------------------------
                results = predictor.predict(image, prompt_str)
                
                # 按照置信度 (score) 从高到低排序，以支持 Top_Oppt[K] 的概率优先
                results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
                
                filtered_results = []
                for res in results:
                    x1, y1, x2, y2 = res['bbox']
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        aspect_ratio = max(w/h, h/w)
                        if target_ratio is None or aspect_ratio >= target_ratio:
                            filtered_results.append((x1, y1, x2, y2, res['class_name'], res['score'], aspect_ratio))
                            if self.top_k is not None and len(filtered_results) >= self.top_k:
                                break
                            
                print(f"     Found {len(results)} matches for '{prompt_str}', {len(filtered_results)} passed filters (Ratio >= {target_ratio if target_ratio else 'N/A'}, Top {self.top_k if self.top_k else 'All'}).")
                
                # 画框输出图 (BBox Only)
                bbox_image = image.copy()
                for x1, y1, x2, y2, cls_name, score, aspect_ratio in filtered_results:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(bbox_image, f"{cls_name} {score:.2f} R:{aspect_ratio:.1f}", 
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                out_name = f"{os.path.splitext(file)[0]}_bbox.jpg"
                out_file_path = os.path.join(inferred_pic_dir, out_name)
                cv2.imwrite(out_file_path, bbox_image)
                                
                # ---------------------------------------------------------
                # 独立操作 2: Post Processing (仅处理原图，输出黑白 Mask)
                # ---------------------------------------------------------
                if ops_to_run:
                    print(f"     Applying Post Processing ops on pure original image: {ops_to_run}")
                    
                    full_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
                    
                    filtered_mask, _, _, _, _ = fo.apply_filtering(
                        full_mask, image, ops_to_run, filtering_df,
                        root, img_path, contour_image=image,
                        adaptive_gaussian_config=None,
                        visualization_overlay=None,
                        defect_id_method='GroundingDino',
                        mask_default=full_mask
                    )
                    
                    # 生成黑底白字的纯净掩码图 (GUI 风格)
                    mask_display = np.zeros_like(image)
                    if filtered_mask is not None and np.any(filtered_mask):
                        mask_display[filtered_mask > 0] = [255, 255, 255] # 纯白缺陷
                        
                    # 把红框印上去作为参照物
                    for x1, y1, x2, y2, cls_name, score, aspect_ratio in filtered_results:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(mask_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    
                    # 保存黑白掩码可视化图片
                    out_name_mask = f"{os.path.splitext(file)[0]}_hsv_mask.jpg"
                    out_file_path_mask = os.path.join(inferred_pic_dir, out_name_mask)
                    cv2.imwrite(out_file_path_mask, mask_display)
                    
                processed_count += 1
                
        print(f">>> [Strategy] Execution Completed! Processed {processed_count} images.")
        print(f"     Results saved to {inferred_pic_dir}")
