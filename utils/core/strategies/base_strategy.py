from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class AnalysisStrategy(ABC):
    """
    基础分析策略类，所有具体的缺陷分析方法都应继承此层。
    """
    def __init__(self, context: Dict[str, Any]):
        """
        :param context: 包含执行当前任务所需的全部上下文(ConfigManager, product, fm, paths等)
        """
        self.context = context
        self.fm = context.get('fm')
        self.product = context.get('product')
        self.generation = context.get('generation')
        self.download_path = context.get('download_path')
        self.res_path = context.get('res_path')
        self.defect_output_df = context.get('defect_output_df', pd.DataFrame())
        
        # --- 恢复 Scoring 与全局 Output 系统的解析 ---
        try:
            from utils.config.config_manager import ConfigManager
            from utils import output_ops, utils_general
            from utils.base_utils import _norm
            import numpy as np
            
            cfg_mgr = ConfigManager()
            output_df = cfg_mgr.get_sheet('Output')
            if self.defect_output_df.empty:
                self.defect_output_df = output_df
            
            # 初始化默认值防止报错
            utils_general.Output_Config = []
            utils_general.Gray_Scale_Params = {}
            utils_general.Reference_Params = {}
            
            if not output_df.empty:
                # Attempt matching with Generation first, fallback to without Generation
                out_match = output_df[
                    (output_df['Product'].astype(str).apply(_norm) == _norm(self.product)) & 
                    (output_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm)) &
                    (output_df['Generation'].astype(str).apply(_norm) == _norm(self.generation))
                ]
                if out_match.empty:
                    out_match = output_df[
                        (output_df['Product'].astype(str).apply(_norm) == _norm(self.product)) & 
                        (output_df['Failure Mode'].astype(str).apply(_norm) == _norm(self.fm))
                    ]
                if not out_match.empty:
                    out_row = out_match.iloc[0]
                    
                    # 1. 恢复 Scoring 列表
                    scoring_val = str(out_row.get('Scoring', '')).strip()
                    if scoring_val and scoring_val.lower() != 'nan':
                        utils_general.Scoring_Config = [s.strip() for s in scoring_val.split(',') if s.strip()]
                    else:
                        utils_general.Scoring_Config = []
                        
                    # 2. 恢复 Scoring Setting 字典
                    scoring_setting_val = str(out_row.get('Scoring Setting', '')).strip()
                    utils_general.Scoring_Setting = {}
                    if scoring_setting_val and scoring_setting_val.lower() != 'nan':
                        utils_general.Scoring_Setting = output_ops.parse_scoring_setting(scoring_setting_val)
                        
                    # 3. 兼容处理：如果 Scoring 列里写了类似 [max_weight=100] 的语法
                    if 'decay' in [s.lower() for s in utils_general.Scoring_Config] and '[' in scoring_val:
                        utils_general.Scoring_Setting.update(output_ops.parse_scoring_setting(scoring_val))
                        
                    # 4. 恢复 Output Config 列表
                    out_config_val = str(out_row.get('Parametrics', '')).strip()
                    if out_config_val and out_config_val.lower() != 'nan':
                        utils_general.Output_Config = [s.strip() for s in out_config_val.strip('[]').split(',') if s.strip()]
                    
                    # 4.5 恢复 Mask Color
                    mask_color_val = str(out_row.get('Mask Color', '')).strip()
                    if not mask_color_val or mask_color_val.lower() == 'nan':
                        mask_color_val = str(out_row.get('Mask_Color', '')).strip()
                        
                    utils_general.Mask_Color = None
                    if mask_color_val and mask_color_val.lower() != 'nan':
                        try:
                            import re
                            match = re.match(r'\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)%?\s*\]', mask_color_val)
                            if match:
                                r, g, b, transparency = match.groups()
                                utils_general.Mask_Color = {
                                    'rgb': (int(r), int(g), int(b)),
                                    'transparency': int(transparency) / 100.0
                                }
                                print(f">>> [Strategy Base] Resolved Mask Color: RGB{utils_general.Mask_Color['rgb']}, Trans: {int(transparency)}%")
                        except Exception as e:
                            print(f">>> [Strategy Base Warning] Failed to parse Mask Color '{mask_color_val}': {e}")
                            
                    # 5. 恢复 Gray_Scale_Params (分箱设置)
                    csc_val = str(out_row.get('Color Space Conversion', '')).strip()
                    inv_val = str(out_row.get('Color Space Invertion', '')).strip()
                    bin_val = str(out_row.get('Color Space Binning', '')).strip()
                    
                    if csc_val and csc_val.lower() != 'nan':
                        utils_general.Gray_Scale_Params = {
                            'Enabled': True,
                            'Mode': csc_val,
                            'Invert': inv_val if inv_val and inv_val.lower() != 'nan' else 'No',
                            'Bining': bin_val if bin_val and bin_val.lower() != 'nan' else None
                        }
                    else:
                        utils_general.Gray_Scale_Params = {}
                        
                    print(f">>> [Strategy Base] Resolved Scoring Config: {utils_general.Scoring_Config}")
                    print(f">>> [Strategy Base] Resolved Scoring Setting: {utils_general.Scoring_Setting}")
                    print(f">>> [Strategy Base] Resolved Gray Scale Params: {utils_general.Gray_Scale_Params}")
        except Exception as e:
            print(f">>> [Strategy Base Warning] Failed to resolve Scoring configs: {e}")

    @abstractmethod
    def execute(self) -> None:
        """
        执行具体的分析业务流（遍历文件、推理、输出等）
        """
        pass

    def prepare_dut_image(self, image_path: str, dut_predictor, sam2_predictor):
        """
        通用前置准备：拿原图 -> 找框 -> SAM抠图 -> 旋转对齐并返回
        """
        from utils import cv_ops
        from utils import utils_general
        import cv2

        # 1. Load original image
        image = cv_ops.load_and_validate_image(image_path)
        if image is None:
            print(f">>> [Strategy] Failed to load image: {image_path}")
            return None, None, None, None

        if dut_predictor is None:
            print(f">>> [Strategy] No DUT Predictor found for {self.fm}. Bypassing alignment and using raw image as DUT.")
            import numpy as np
            h, w = image.shape[:2]
            dummy_mask = np.ones((h, w), dtype=np.uint8) * 255
            return image, image.copy(), dummy_mask, {}

        # 2 & 3. Align and crop using the shared tool
        # We fetch configurations like batch_alignment from context or fallback to True
        use_batch_alignment = self.context.get('use_batch_alignment', True)
        mask_scaling_factor = self.context.get('mask_scaling_factor', 1.0)
        is_debug_mode = self.context.get('is_debug_mode', False)
        
        # Get target_size from context if available
        dino_input_dim = self.context.get('dino_input_dim', None)
        target_size = None
        if isinstance(dino_input_dim, list) and len(dino_input_dim) > 0:
            target_size = dino_input_dim[0]

        try:
            DUT_Corrected, Contour_Corrected, alignment_metadata = utils_general.perform_dut_alignment(
                image=image,
                dut_predictor=dut_predictor,
                sam2_predictor=sam2_predictor,
                use_batch_alignment=use_batch_alignment,
                mask_scaling_factor=mask_scaling_factor,
                feature_scaling_config=getattr(utils_general, 'Feature_Scaling_Config', None),
                is_debug_mode=is_debug_mode,
                product=self.product,
                target_size=target_size
            )
            
            return image, DUT_Corrected, Contour_Corrected, alignment_metadata

        except Exception as e:
            print(f">>> [Strategy] Error in prepare_dut_image for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return image, None, None, None
