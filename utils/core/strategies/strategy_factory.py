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

        # 0. 拦截需要进行 General FM 派发的请求 (避免内部调用时产生无限递归，通过 general_fm_rules 判断)
        if not context.get('general_fm_rules'):
            from utils.config.config_manager import ConfigManager
            from utils.base_utils import _norm
            cm = ConfigManager()
            gfm_df = cm.get_sheet('General FM')
            if not gfm_df.empty and 'Failure Mode' in gfm_df.columns:
                gfm_matches = gfm_df[gfm_df['Failure Mode'].astype(str).apply(_norm) == _norm(fm_name)]
                if not gfm_matches.empty:
                    from .general_fm_dispatcher import GeneralFmDispatcher
                    return GeneralFmDispatcher(context)

        # 1. 优先根据 Failure Mode (FM) 精准路由
        if fm_clean == 'hiaa bleach':
            from .complex_bleach_strategy import ComplexBleachStrategy
            return ComplexBleachStrategy(context)
            
        if fm_clean == 'light bleed':
            from .macbook_light_bleed_strategy import MacbookLightBleedStrategy
            return MacbookLightBleedStrategy(context)

        if fm_clean == 'canvas prepare' or fm_clean == 'k11p_defects':
            from .canvas_prepare_strategy import CanvasPrepareStrategy
            return CanvasPrepareStrategy(context)
            
        if 'complex' in method_clean or 'complex_textile' in method_clean:
            # We want to use our handwritten ComplexTextileStrategy instead of DynamicRoutingStrategy
            if 'textile' in fm_clean or 'textile' in method_clean or fm_clean == 'bubble':
                from .complex_textile_strategy import ComplexTextileStrategy
                return ComplexTextileStrategy(context)
            else:
                from .dynamic_routing_strategy import DynamicRoutingStrategy
                return DynamicRoutingStrategy(context)
            
        if fm_clean == 'crop' or fm_clean.startswith('crop_') or fm_clean.startswith('dino crop') or fm_clean.startswith('groundingcrop'):
            from .crop_strategy import CropStrategy
            return CropStrategy(context)

        if 'object_detection' in method_clean or 'object detection' in method_clean or fm_clean == 'mistral_screen_defect':
            from .object_detection_strategy import ObjectDetectionStrategy
            return ObjectDetectionStrategy(context)

        # 2. 然后根据具体操作方法 (Defect Identification) 泛化路由
        if 'groundingdino' in method_clean or 'grounding dino' in method_clean:
            from .grounding_dino_strategy import GroundingDinoStrategy
            return GroundingDinoStrategy(context)

        if 'groundingsam' in method_clean or 'grounding sam' in method_clean:
            from .grounding_sam_strategy import GroundingSamStrategy
            return GroundingSamStrategy(context)

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
