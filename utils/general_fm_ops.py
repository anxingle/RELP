import cv2
import numpy as np
import pandas as pd

def image_matches_mapping(image_path: str, mapping_value: str) -> bool:
    """
    Check if image path matches any keyword in Mapping list.
    Mapping format: comma-separated values (e.g., "B1,B2,B3" or "[B1,B2,B3]")
    """
    if not mapping_value or str(mapping_value).lower() in ['nan', '']:
        return False
        
    mapping_str = str(mapping_value).strip()
    if mapping_str.startswith('[') and mapping_str.endswith(']'):
        mapping_str = mapping_str[1:-1]
        
    keywords = [k.strip() for k in mapping_str.split(',') if k.strip()]
    if not keywords:
        return False
        
    image_path_lower = image_path.lower()
    for kw in keywords:
        if kw.lower() in image_path_lower:
            return True
            
    return False

def apply_general_fm_scope(mask: np.ndarray, image: np.ndarray, detector, gfm_rules: list) -> tuple:
    """
    应用 General FM 的空间裁切与打标逻辑 (Scope Filtering)。
    
    :param mask: 原始大模型 (如 Dino) 输出的缺陷 Mask
    :param image: 对齐后的目标原图 (DUT_Corrected)
    :param detector: Detectron2 General Defect Predictor
    :param gfm_rules: General FM 表中提取出的规则列表
    :return: (裁切后的 Mask, 命中的部件名称列表, 可视化用的 BBox 列表)
    """
    if mask is None or not np.any(mask):
        return mask, [], []
        
    if detector is None:
        print(">>> [GeneralFmScope] Warning: No General Defect Predictor loaded. Skipping scope filtering.")
        return mask, [], []
        
    from utils.utils_general import perform_general_defect_detection
    
    # 1. 跑目标检测拿全图 BBox
    bboxes = perform_general_defect_detection(image, detector, None, False)
    if not bboxes:
         print(">>> [GeneralFmScope] No BBoxes found by detector. Defect mask will be zeroed out if Scope requires it.")
         return np.zeros_like(mask), [], []
         
    # 获取类别映射名称
    class_names = None
    if hasattr(detector, 'cfg'):
        from detectron2.data import MetadataCatalog
        if len(detector.cfg.DATASETS.TEST) > 0:
            dataset_name = detector.cfg.DATASETS.TEST[0]
            meta = MetadataCatalog.get(dataset_name)
            if hasattr(meta, 'thing_classes'):
                class_names = meta.thing_classes

    bbox_combined_mask = np.zeros_like(mask)
    detected_labels = set()
    matched_bboxes = []
    
    requires_bbox_scope = False
    
    # 2. 遍历规则，组装允许区域的 Mask
    for rule in gfm_rules:
        scope = str(rule.get('Scope', '')).strip().lower()
        target_defect_class = str(rule.get('Defect Class', '')).strip().lower()
        
        if scope == 'defect detection bbox':
            requires_bbox_scope = True
            
            for box_dict in bboxes:
                b_coords = box_dict['bbox']
                b_cls_name = str(box_dict.get('class_name', ''))
                b_cls_id = box_dict.get('class_id', -1)
                
                # 如果我们有传入的 class_names 并且 b_cls_name 还是个数字，尝试转换
                if class_names and b_cls_name.isdigit() and int(b_cls_name) < len(class_names):
                    b_cls_name = class_names[int(b_cls_name)]
                
                # 如果规则指定了 Defect Class，只有当检测框类别匹配时才算数
                # 兼容方案：如果没有 class_names 导致 b_cls_name 是数字，为了防止全部失效，我们默认放行。或者检查数字映射。
                is_match = False
                if not target_defect_class or target_defect_class == 'nan':
                    is_match = True
                elif target_defect_class == b_cls_name.lower():
                    is_match = True
                elif b_cls_name.isdigit(): # If we couldn't load metadata and it's just a digit, trust the model found a component.
                    is_match = True
                    print(f">>> [GeneralFmScope] Warning: Box class is digit '{b_cls_name}', treating as match for target '{target_defect_class}'.")

                if is_match:
                    gx1, gy1, gx2, gy2 = map(int, b_coords)
                    gx1, gx2 = max(0, gx1), min(mask.shape[1], gx2)
                    gy1, gy2 = max(0, gy1), min(mask.shape[0], gy2)
                    if gx2 > gx1 and gy2 > gy1:
                        bbox_combined_mask[gy1:gy2, gx1:gx2] = 255
                        detected_labels.add(b_cls_name)
                        matched_bboxes.append(box_dict)
                        
    # 3. 如果需要空间裁切，则执行按位与 (Intersection)
    if requires_bbox_scope:
        new_mask = cv2.bitwise_and(mask, bbox_combined_mask)
        print(f">>> [GeneralFmScope] Applied BBox Scope. Pixels before: {cv2.countNonZero(mask)}, after: {cv2.countNonZero(new_mask)}")
        return new_mask, list(detected_labels), matched_bboxes
    else:
        # 如果不是 Defect Detection BBOX，就原样返回
        return mask, [], []
