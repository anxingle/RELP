import os
import cv2
import pandas as pd
from typing import Dict, Any
from .base_strategy import AnalysisStrategy
from utils.config.config_manager import ConfigManager
from utils.wrappers.detectron_wrapper import load_detectron_model
from utils.detectron_ops import visualize_all_detections

class ObjectDetectionStrategy(AnalysisStrategy):
    """
    专门针对基础目标检测的策略。
    读取配置好的图片，用 Detectron2 跑框并保存结果。
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
        from core.pipeline import DefectDetectionPipeline
        weights = DefectDetectionPipeline()._resolve_model_weights(fm, self.product, self.generation, weights_df)
        
        # 取 cfg_path 和 weights_path 
        cfg_path = weights.get('cfg_path') or weights.get('cfg_path_dut')
        weights_path = weights.get('weights_path') or weights.get('weights_path_dut')
        
        if not cfg_path or not weights_path:
            print(f">>> [Strategy Error] 未在 Weights 表中找到 {fm} 对应的 detectron 权重配置。")
            return
            
        print(f">>> [Strategy] Model weights resolved. CFG: {cfg_path}, WEIGHTS: {weights_path}")
        
        # 3. 加载模型
        try:
            detector = load_detectron_model(cfg_path, weights_path, device='cpu')
        except Exception as e:
            print(f">>> [Strategy Error] 加载 Detectron 模型失败: {e}")
            return
            
        # 尝试加载 class_names (从 weights 同级目录下寻找 classes.txt 或 classes.rtf)
        weights_dir = os.path.dirname(weights_path)
        
        # 为了兼容 Mistral 的权重目录结构，我们尝试在 weights_dir, 它的父目录, 以及它的 DUT 子目录查找
        possible_dirs = [weights_dir, os.path.dirname(weights_dir), os.path.join(weights_dir, 'DUT'), os.path.join(os.path.dirname(weights_dir), 'DUT')]
        class_names = None
        
        for search_dir in possible_dirs:
            if class_names: break
                
            # 先找 txt
            classes_txt_path = os.path.join(search_dir, 'classes.txt')
            if os.path.exists(classes_txt_path):
                with open(classes_txt_path, 'r', encoding='utf-8') as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                print(f">>> [Strategy] 从 {classes_txt_path} 加载了 {len(class_names)} 个类别名称: {class_names}")
                break
                
            # 再找 rtf (Mac 经常保存为 rtf)
            classes_rtf_path = os.path.join(search_dir, 'classes.rtf')
            
            if os.path.exists(classes_rtf_path):
                import re
                try:
                    with open(classes_rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    # 粗暴解析 RTF: 找 "\cf0" 之后的纯文本，然后正则匹配 "0: 类别名"
                    text_part = text.split(r'\cf0')[-1].replace('}', '')
                    matches = re.findall(r'(\d+):\s*([^\\]+)', text_part)
                    
                    if matches:
                        class_dict = {int(k): v.strip() for k, v in matches}
                        max_idx = max(class_dict.keys())
                        class_names = [class_dict.get(i, f"Class_{i}") for i in range(max_idx + 1)]
                        print(f">>> [Strategy] 从 {classes_rtf_path} 成功解析了 {len(class_names)} 个类别名称: {class_names}")
                        break
                except Exception as e:
                    print(f">>> [Strategy Warning] 解析 RTF 失败: {e}")
                    
        if not class_names:
            print(f">>> [Strategy Info] 未找到 classes.txt 或 classes.rtf，将使用默认类别名 (Class_0, Class_1...)。")
            
        # 4. 解析生成结果路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path_type = str(fm_row['Output_Path']) if fm_row is not None else 'Result'
        if output_path_type == 'Result' or pd.isna(output_path_type):
            res_path = os.path.join(project_root, 'Result', f"{self.product}_{self.generation}_{fm}_{fm}_Result")
        else:
            res_path = os.path.join(output_path_type, f"{self.product}_{self.generation}_{fm}_{fm}_Result")
            
        if not os.path.exists(res_path):
            os.makedirs(res_path)
            
        download_path = self.download_path
        if pd.isna(download_path) or not os.path.exists(download_path):
            print(f">>> [Strategy] Download path does not exist or empty: {download_path}")
            return
            
        print(f">>> [Strategy] Starting directory traversal on: {download_path}")
        
        # 5. 遍历并检测
        for root, dirs, files in os.walk(download_path):
            valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) and not f.startswith('.')]
            for file in valid_files:
                img_path = os.path.join(root, file)
                print(f"  -> Processing {file} ...")
                
                # 读取图片
                image = cv2.imread(img_path)
                if image is None:
                    print(f"     [Error] Cannot read image {img_path}")
                    continue
                    
                # 运行检测与画框
                debug_img, info = visualize_all_detections(image, detector, class_names=class_names, score_thresh=0.5)
                
                print(f"     Found {len(info)} objects.")
                
                # 保存结果
                out_name = f"{os.path.splitext(file)[0]}_bbox.jpg"
                out_file_path = os.path.join(res_path, out_name)
                cv2.imwrite(out_file_path, debug_img)
                
        print(f">>> [Strategy] Execution Completed! Results saved to {res_path}")

