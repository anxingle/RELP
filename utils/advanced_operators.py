import cv2
import numpy as np
from typing import List, Tuple, Dict

def op_isolated_anchor_filter(
    raw_results: List[Dict], 
    img_area: int, 
    max_area_pct: float = 0.05,
    max_aspect_ratio: float = 15.0,
    iou_threshold: float = 0.3
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    [原子算子] 孤岛锚点重叠剔除算法 (Isolated Anchor Matching)
    
    业务场景：用于解决零样本视觉模型（如 Grounding DINO）在面对反光、倒角等复杂物理结构时，
    产生的“套娃（嵌套框）”和“幽灵重叠框”问题。
    
    机制：
    1. 物理尺寸初筛：剔除超过 max_area_pct 的巨大套娃框。
    2. 极端形变剔除：剔除长宽比过于夸张的细长条（通常是接缝或反光带）。
    3. 锚点定位：寻找在X轴上没有任何重叠的“孤立框 (Isolated Box)”，将其真实物理长宽比作为锚点基准。
    4. 重叠大对决：在发生重叠的框群中，计算谁的比例最接近锚点，抛弃其他伪影框。
    """
    # 1. 收集所有符合面积条件的框
    filtered_bboxes = []
    for br in raw_results:
        bx1, by1, bx2, by2 = br['bbox']
        bw, bh = bx2 - bx1, by2 - by1
        if (bw * bh) / img_area < max_area_pct:
            filtered_bboxes.append((bx1, by1, bx2, by2, bw, bh))
            
    final_bboxes = []
    if len(filtered_bboxes) == 0:
        return final_bboxes
        
    isolated_idx = -1
    overlapping_indices = []
    
    # 找出孤立的框和重叠的框 (X轴重叠判断)
    for i in range(len(filtered_bboxes)):
        is_overlapping = False
        for j in range(len(filtered_bboxes)):
            if i == j: continue
            ax1, _, ax2, _ = filtered_bboxes[i][:4]
            bx1, _, bx2, _ = filtered_bboxes[j][:4]
            ix1 = max(ax1, bx1)
            ix2 = min(ax2, bx2)
            iw = max(0, ix2 - ix1)
            # 如果水平重叠度超过 30%，认为它们挤在一起
            if iw > min(ax2-ax1, bx2-bx1) * iou_threshold:
                is_overlapping = True
                if i not in overlapping_indices: overlapping_indices.append(i)
                if j not in overlapping_indices: overlapping_indices.append(j)
                break
        if not is_overlapping:
            isolated_idx = i
            
    if isolated_idx != -1:
        # 存在明确的孤立按钮，将其作为长宽比锚点
        iso_box = filtered_bboxes[isolated_idx]
        iso_ratio = iso_box[4] / iso_box[5] if iso_box[5] > 0 else 1
        
        # 把所有没有重叠的孤立按钮都直接加入结果
        for i in range(len(filtered_bboxes)):
            if i not in overlapping_indices:
                final_bboxes.append(filtered_bboxes[i])
        
        # 在重叠簇中寻找与锚点比例最接近的真实按键框
        best_cluster_box = None
        best_ratio_diff = 999
        for idx in overlapping_indices:
            c_box = filtered_bboxes[idx]
            c_ratio = c_box[4] / c_box[5] if c_box[5] > 0 else 1
            diff = abs(c_ratio - iso_ratio)
            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_cluster_box = c_box
        if best_cluster_box:
            final_bboxes.append(best_cluster_box)
    else:
        # Fallback: 如果没有孤立点，按面积从大到小去重 (经典 IoU NMS)
        for i, box_a in enumerate(filtered_bboxes):
            ax1, ay1, ax2, ay2, aw, ah = box_a
            # 过滤极端形变
            if ah == 0 or aw == 0 or aw/ah > max_aspect_ratio or ah/aw > (max_aspect_ratio/3): 
                continue
            is_duplicate = False
            for j, box_b in enumerate(final_bboxes):
                bx1, by1, bx2, by2, bw, bh = box_b
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a, area_b = aw * ah, bw * bh
                union_area = area_a + area_b - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_threshold or inter_area > area_a * 0.8 or inter_area > area_b * 0.8:
                    is_duplicate = True
                    # 保留两者中面积较大的那个
                    if area_a > area_b: final_bboxes[j] = box_a
                    break
            if not is_duplicate: final_bboxes.append(box_a)
            
    return final_bboxes


def op_create_spatial_penalty_mask(
    bboxes: List[Tuple[float, float, float, float, float, float]], 
    img_w: int, 
    img_h: int, 
    expand_ratio: float = 1.3
) -> np.ndarray:
    """
    [原子算子] 生成空间惩罚热区掩码 (Spatial Penalty Mask Generation)
    
    业务场景：在检测出的目标框（如按键、接口）周围，由于装配公差和物理应力，
    往往是瑕疵高发区。此算子将目标框等比例外扩，生成一个用于分数加倍的区域掩码。
    """
    penalty_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for bx1, by1, bx2, by2, bw, bh in bboxes:
        cx, cy = bx1 + bw/2, by1 + bh/2
        new_w, new_h = bw * expand_ratio, bh * expand_ratio
        
        nbx1, nby1 = int(cx - new_w/2), int(cy - new_h/2)
        nbx2, nby2 = int(cx + new_w/2), int(cy + new_h/2)
        
        # 边界保护
        nbx1, nby1 = max(0, nbx1), max(0, nby1)
        nbx2, nby2 = min(img_w, nbx2), min(img_h, nby2)
        
        cv2.rectangle(penalty_mask, (nbx1, nby1), (nbx2, nby2), 255, -1)
        
    return penalty_mask


def op_relative_slicing(
    target_mask: np.ndarray,
    x_ranges: List[Tuple[float, float]] = None,
    y_ranges: List[Tuple[float, float]] = None,
    reference_contour: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [原子算子] 相对切片 (Relative BBox Slicing)
    
    业务场景：根据主体的真实Bounding Box坐标，精确切分出目标区域（如主体下半区 Y[0.45:1.0]）。
    此算子剥离了字符串解析逻辑，只处理纯矩阵的掩码切分。

    Args:
        target_mask: 需要被切分的瑕疵掩码 (H, W)
        x_ranges: X轴向的切分比例，格式如 [(0.0, 0.1), (0.9, 1.0)]
        y_ranges: Y轴向的切分比例，格式如 [(0.45, 1.0)]
        reference_contour: 参照主体轮廓掩码。如果提供，将计算该轮廓的BBox作为切片基准；
                           如果不提供，退化为相对全图分辨率的绝对切分。

    Returns:
        sliced_target: 切分后的目标掩码，dtype 与输入 target_mask 保持一致
        sliced_contour: 同步被切分的参考轮廓掩码，dtype 与输入 reference_contour 保持一致（若未提供参考轮廓则返回 None）
    """
    if target_mask is None:
        return None, None

    x_ranges = x_ranges or []
    y_ranges = y_ranges or []
    
    mask_u8 = (target_mask.astype(np.uint8) * 255) if target_mask.dtype == bool else target_mask.copy()
    h, w = mask_u8.shape[:2]
    
    bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, w, h
    
    # 如果提供了参考轮廓，则计算其 BBox 作为切片相对基准
    if reference_contour is not None:
        if len(reference_contour.shape) == 3:
            gray_contour = cv2.cvtColor(reference_contour, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray_contour, 1, 255, cv2.THRESH_BINARY)
        else:
            thresh = (reference_contour.astype(np.uint8) * 255) if reference_contour.dtype == bool else reference_contour
            
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            if bw > 0 and bh > 0:
                bbox_x, bbox_y, bbox_w, bbox_h = bx, by, bw, bh

    mask_x = np.zeros_like(mask_u8)
    mask_y = np.zeros_like(mask_u8)
    
    # 构造 X 轴掩码
    if x_ranges:
        for min_ratio, max_ratio in x_ranges:
            x_start = max(0, min(bbox_x + int(bbox_w * min_ratio), w))
            x_end = max(0, min(bbox_x + int(bbox_w * max_ratio), w))
            mask_x[:, x_start:x_end] = 255
            
    # 构造 Y 轴掩码
    if y_ranges:
        for min_ratio, max_ratio in y_ranges:
            y_start = max(0, min(bbox_y + int(bbox_h * min_ratio), h))
            y_end = max(0, min(bbox_y + int(bbox_h * max_ratio), h))
            mask_y[y_start:y_end, :] = 255
            
    # 合并 X 和 Y 掩码
    if x_ranges and y_ranges:
        mask_keep = cv2.bitwise_and(mask_x, mask_y)
    elif x_ranges:
        mask_keep = mask_x
    elif y_ranges:
        mask_keep = mask_y
    else:
        # 没有 ranges 时不发生截断
        mask_keep = np.ones_like(mask_u8) * 255
        
    sliced_target = cv2.bitwise_and(mask_u8, mask_keep)
    
    # 同步切分参考轮廓
    sliced_contour = None
    if reference_contour is not None:
        contour_u8 = (reference_contour.astype(np.uint8) * 255) if reference_contour.dtype == bool else reference_contour
        
        if mask_keep.shape[:2] != contour_u8.shape[:2]:
            mask_keep_resized = cv2.resize(mask_keep, (contour_u8.shape[1], contour_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_keep_resized = mask_keep
            
        if len(contour_u8.shape) == 3 and len(mask_keep_resized.shape) == 2:
            sliced_contour = cv2.bitwise_and(contour_u8, contour_u8, mask=mask_keep_resized)
        else:
            sliced_contour = cv2.bitwise_and(contour_u8, mask_keep_resized)
            
        # 恢复轮廓原来的 bool 类型
        if reference_contour.dtype == bool:
            sliced_contour = sliced_contour > 127
            
    # 恢复目标掩码原来的 bool 类型
    if target_mask.dtype == bool:
        sliced_target = sliced_target > 127
        
    return sliced_target, sliced_contour
