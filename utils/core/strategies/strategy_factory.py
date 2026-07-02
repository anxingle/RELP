from typing import Dict, Any
from .base_strategy import AnalysisStrategy

class StrategyFactory:
    """
    策略工厂：根据 Excel 中配置的 defect_id_method 或 Failure Mode，
    返回负责执行的具体 Strategy 实例。
    """
    @staticmethod
    def get_strategy(method_name: str, fm_name: str, context: Dict[str, Any]) -> AnalysisStrategy:
        method_clean = str(method_name).strip().lower() if method_name is not None else ''
        fm_clean = str(fm_name).strip().lower() if fm_name is not None else ''



        # 1. 优先根据 Failure Mode (FM) 精准路由
        if fm_clean == 'canvas prepare' or fm_clean == 'k11p_defects':
            from .canvas_prepare_strategy import CanvasPrepareStrategy
            return CanvasPrepareStrategy(context)
            
        if fm_clean == 'crop' or fm_clean.startswith('crop_') or fm_clean.startswith('dino crop'):
            from .crop_strategy import CropStrategy
            return CropStrategy(context)

        # 2. 然后根据具体操作方法 (Defect Identification) 泛化路由
        if 'knn' in method_clean or 'knn' in fm_clean:
            from .knn_strategy import KnnStrategy
            return KnnStrategy(context)
            
        if 'dino' in method_clean and not fm_clean.startswith('dino crop'):
            from .dino_strategy import DinoStrategy
            return DinoStrategy(context)
            
        if 'detectron_seg' in method_clean or fm_clean == 'silicon delam':
            from .detectron_seg_strategy import DetectronSegStrategy
            return DetectronSegStrategy(context)
            
        if 'filtering' in method_clean:
            from .filtering_strategy import FilteringStrategy
            return FilteringStrategy(context)

        print(f"Warning: No specific strategy found for method='{method_name}', FM='{fm_name}'.")
        return None
