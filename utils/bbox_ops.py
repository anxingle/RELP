from utils.output_ops import clustering_Kmeans

import cv2
import numpy as np
import re

def create_bbox_mask(image_shape, bbox):
    """
    Create a binary mask from bounding box coordinates.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    
    return mask

def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate IOU between two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = map(int, bbox1)
    x1_2, y1_2, x2_2, y2_2 = map(int, bbox2)
    
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area <= 0: return 0.0
    return intersection_area / union_area

def calculate_defect_ratio_in_bbox(defect_mask, bbox):
    """
    Calculate the ratio of defect area within a BBOX.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = defect_mask.shape[:2]
    
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0: return 0.0
    
    defect_in_bbox = defect_mask[y1:y2, x1:x2]
    defect_area_in_bbox = cv2.countNonZero(defect_in_bbox)
    
    return defect_area_in_bbox / bbox_area

def filter_mask_by_bbox_iou(binary_mask, bbox_list, iou_threshold=0.2):
    """
    Filter binary mask to keep only defects that have defect ratio >= threshold in any BBOX.
    """
    if binary_mask is None or bbox_list is None or len(bbox_list) == 0:
        return binary_mask
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return binary_mask
    
    bboxes = []
    for bbox_item in bbox_list:
        if isinstance(bbox_item, dict):
            bbox = bbox_item['bbox']
        else:
            bbox = bbox_item
        bboxes.append(bbox)
    
    h, w = binary_mask.shape[:2]
    filtered_mask = np.zeros((h, w), dtype=np.uint8)
    
    kept_defects = 0
    total_defects = len(contours)
    
    for contour in contours:
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(defect_mask, [contour], -1, 255, -1)
        
        max_ratio = 0.0
        for idx, bbox in enumerate(bboxes):
            ratio = calculate_defect_ratio_in_bbox(defect_mask, bbox)
            if ratio > max_ratio:
                max_ratio = ratio
        
        if max_ratio >= iou_threshold:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
            kept_defects += 1
            # print(f"DEBUG: BBOX IOU - Kept defect with defect_ratio={max_ratio:.3f} (threshold={iou_threshold})")
    
    return filtered_mask

def parse_bbox_iou_scope(scope_value):
    """
    Parse BBOX IOU scope value to extract threshold.
    """
    if not scope_value: return False, 0.2
    scope_str = str(scope_value).strip().lower()
    
    if not scope_str.startswith('bbox iou'):
        return False, 0.2
    
    threshold = 0.2
    match = re.search(r'\[([0-9.]+)\]', scope_str)
    if match:
        try:
            threshold = float(match.group(1))
            if threshold < 0 or threshold > 1: threshold = 0.2
        except ValueError:
            threshold = 0.2
    
    return True, threshold

# --- Advanced Extracted Ops ---

import cv2
import numpy as np
import pandas as pd
import math

def order_points(pts):
    """
    Orders coordinates in the order: top-left, top-right, bottom-right, bottom-left.
    Args:
        pts: A list or numpy array of 4 points (x, y).
    Returns:
        Ordered numpy array of shape (4, 2).
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference,
    # whereas the bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def bbox_outlier_supression(bbox, sorting_strategy, df_SNlist, parsed_box_res):
    DUT_qty = len(bbox)
    sub_SN_QTY = df_SNlist.count()
    SN_qty = sub_SN_QTY.sum()
    bbox_to_remove = DUT_qty - SN_qty

    # if DUT is vertically stack-upped, switch bbox h/w positions to standardize drop-out action
    if sorting_strategy == 1:
        bbox[:, [4, 5]] = bbox[:, [5, 4]]
    cluster_to_drop = select_bbox2_dropout(bbox, df_SNlist, 4)

    print('Cluster_to_drop DD', cluster_to_drop)
    filtered_bbox = bbox_dropout(bbox, cluster_to_drop, 4)
    if len(cluster_to_drop) == bbox_to_remove:
        print('All outlier bboxes will be dropout in one batch')
        filtered_bbox = filtered_bbox[:, :4]
    elif len(cluster_to_drop) < bbox_to_remove:
        print('Outlier bboxes will be dropout in two batches')
        additional_bbox2_drop = bbox_to_remove - len(cluster_to_drop)
        if sorting_strategy == 1:
            filtered_bbox[:, [6, 7]] = bbox[:, [7, 6]]
        # sort bbox X/Y coordinates of centroids
        sorted_bbox_centroid = np.argsort(filtered_bbox[:, 6])
        sorted_filtered_bbox = filtered_bbox[sorted_bbox_centroid]
        # calculate centroids difference to prepare secondary dropout
        table_differences = np.diff(sorted_filtered_bbox[:, 6])
        # padding 1st element of diff list with 0
        padded_diff = np.insert(table_differences, 0, np.nan)
        ones_array = np.ones(5).reshape((-1, 1))
        print('table_differences', table_differences)
        tier2_outlier = select_bbox2_dropout(table_differences, ones_array, 0)
        print('tier2_outlier', tier2_outlier)
        tier2_outlier_list = tier2_outlier[0:additional_bbox2_drop]
        # print('tier2_outlier_list', tier2_outlier_list)
        extended_filtered_bbox = np.hstack((sorted_filtered_bbox, padded_diff.reshape(-1, 1)))
        filtered_bbox = bbox_dropout(extended_filtered_bbox, tier2_outlier_list, 8)
        filtered_bbox = filtered_bbox[:, :4]
        print('remainingLLL filtered_bbox', filtered_bbox)
    if filtered_bbox.shape[0] == SN_qty:
        print("Detected DUTs QTY matches SN Qty")
    elif filtered_bbox.shape[0] > SN_qty:
        # print("Detected DUTs QTY does not match SN Qty. Further action required")
        cluster_to_drop = select_bbox2_dropout(bbox, df_SNlist, 7)
        filtered_bbox = bbox_dropout(bbox, cluster_to_drop, 7)
        if filtered_bbox.shape[0] == SN_qty:
            print('Detected DUTs QTY matches SN Qty after 3rd try', filtered_bbox)
        else:
            print("try something else")
            return
    return filtered_bbox


def select_bbox2_dropout(ini_bbox, df_SN, bbox_sorting_dim):
    cluster1_data, cluster2_data, cluster1_mean, cluster2_mean, cluster_array = clustering_Kmeans(ini_bbox, df_SN,
                                                                                                  bbox_sorting_dim)
    cluster_to_drop = cluster1_data
    print('cluster1_data', cluster1_data)
    print('cluster2_data', cluster1_data)
    # print('select_bbox2_dropout ini_bbox', ini_bbox)

    if len(cluster1_data) / ini_bbox.shape[0] >= len(cluster2_data) / ini_bbox.shape[0]:
        arranging_strategy = "max"
    else:
        arranging_strategy = "min"
    if arranging_strategy == "max":
        cluster_to_drop = cluster2_data
    elif arranging_strategy == "min":
        cluster_to_drop = cluster1_data

    print('select_bbox2_dropout clustering', cluster_to_drop)
    return cluster_to_drop


def bbox_dropout(bbox, drop_list, column_index):
    mask = np.ones(len(bbox), dtype=bool)
    for value in drop_list:
        mask &= bbox[:, column_index] != value
    bbox_post_dropout = bbox[mask]
    # print('bbox_post_dropout', bbox_post_dropout)
    return bbox_post_dropout


def sorting_bbox(boxes_dino, df_SN):
    print('boxes_dino', boxes_dino)
    boxes_w = boxes_dino[:, 2] - boxes_dino[:, 0]
    avg_box_w = np.average(boxes_w)
    # print('avg_box_w', avg_box_w)
    boxes_h = boxes_dino[:, 3] - boxes_dino[:, 1]
    avg_box_h = np.average(boxes_h)
    # print('avg_box_h', avg_box_h)
    box_center_x = boxes_dino[:, 0] + 1 / 2 * boxes_w
    box_center_y = boxes_dino[:, 1] + 1 / 2 * boxes_h

    centers = [((x_min + x_max) / 2, (y_min + y_max) / 2) for x_min, y_min, x_max, y_max in boxes_dino]
    min_x = min([x_min for x_min, _, _, _ in boxes_dino])
    max_x = max([x_max for _, _, x_max, _ in boxes_dino])
    min_y = min([y_min for _, y_min, _, _ in boxes_dino])
    max_y = max([y_max for _, _, _, y_max in boxes_dino])
    whole_box_height = max_y - min_y
    whole_box_width = max_x - min_x
    sorting_strg = -1
    # if df_SN.shape[1] == 1:
    if whole_box_height < 1.5 * avg_box_h and whole_box_width > 1.5 * avg_box_w:
        sorting_strg = 0
        print("objects are organized horizontally")
    elif whole_box_width < 1.5 * avg_box_w and whole_box_height > 1.5 * avg_box_h:
        sorting_strg = 1
        print("objects are organized vertically")

    combined_array = np.column_stack((box_center_x, box_center_y, boxes_w, boxes_h))
    # element_diff_cal(box_center_x)
    box_center_matrix = np.column_stack((box_center_x, box_center_y))
    bbox_to_suppress = np.column_stack((boxes_dino, boxes_w, boxes_h, box_center_matrix))
    # print('box_center_matrix', box_center_matrix)
    return sorting_strg, combined_array, bbox_to_suppress


def bbox_centroid(bbox):
    x1, y1, x2, y2 = bbox
    center_x = x1 + (x2 - x1) * 0.5
    center_y = y1 + (y2 - y1) * 0.5
    return [[center_x, center_y]]


def calculate_intersection_area(mask, coords):
    polygon_mask = np.zeros(mask.shape, dtype=np.uint8)
    coords = coords.reshape((-1, 1, 2))
    cv2.fillPoly(polygon_mask, [coords], 255)
    intersection = cv2.bitwise_and(mask, polygon_mask)
    intersection_area = np.count_nonzero(intersection)
    return intersection_area


def scaling_up_bbox(obj_box, scalar_x, scalar_y, pic_h, pic_w):
    # bbox structured in xyxy format
    print('before scaling_up_bbox', obj_box)
    box_w = obj_box[:, 2] - obj_box[:, 0]
    box_h = obj_box[:, 3] - obj_box[:, 1]
    box_center_x = obj_box[:, 0] + 1 / 2 * box_w
    box_center_y = obj_box[:, 1] + 1 / 2 * box_h

    # 计算缩放后的宽度和高度
    scaled_w = scalar_x * box_w
    scaled_h = scalar_y * box_h

    # 根据中心点和缩放后的宽高计算新的边界
    scaled_x_left = np.clip(box_center_x - 1 / 2 * scaled_w, 0, pic_w)
    scaled_x_right = np.clip(box_center_x + 1 / 2 * scaled_w, 0, pic_w)
    scaled_y_up = np.clip(box_center_y - 1 / 2 * scaled_h, 0, pic_h)
    scaled_y_bottom = np.clip(box_center_y + 1 / 2 * scaled_h, 0, pic_h)

    # 确保左边界小于右边界，上边界小于下边界
    scaled_x_right = np.maximum(scaled_x_right, scaled_x_left + 1)
    scaled_y_bottom = np.maximum(scaled_y_bottom, scaled_y_up + 1)

    scaled_box = np.column_stack((scaled_x_left, scaled_y_up, scaled_x_right, scaled_y_bottom))
    scaled_box = np.array(scaled_box).astype(int)
    print('post scaling_up_bbox', scaled_box)

    return scaled_box


def adjust_and_scale_bbox(df, bbox_col, max_coords, scale_ratio):
    df_copy = df.copy()

    def process_bbox(bbox):
        # 确保bbox不超过最大坐标
        bbox[2] = min(bbox[2], max_coords[0])  # x2
        bbox[3] = min(bbox[3], max_coords[1])  # y2
        # 按比例缩放并转为整数
        scaled_bbox = np.array(bbox) * scale_ratio
        return scaled_bbox.astype(int).tolist()

    df_copy[bbox_col] = df_copy[bbox_col].apply(process_bbox)
    return df_copy


def bbox_extraction_pd(df, label2select, scale_factor, pic_h, pic_w):
    selected_df = df[df['label'] == label2select]
    if not selected_df.empty:
        # 提取bbox值并转换成二维ndarray
        selected_bboxes = np.array(selected_df['bbox'].apply(lambda x: np.array(x)))
        selected_bboxes_2d = np.vstack(selected_bboxes)
        # print("orig selected_bboxes_2d", selected_bboxes_2d)
        scaled_bboxes_2d = scaling_up_bbox(selected_bboxes_2d, scale_factor, scale_factor, pic_h, pic_w)
        # print("scaled selected_bboxes_2d", scaled_bboxes_2d)
        return scaled_bboxes_2d
    else:
        return np.zeros((1, 4))


def bbox_2darray(bbox_ori):
    # print("before bbox transformation", bbox_ori)
    bbox_trans = np.array([
        [bbox_ori[0], bbox_ori[1]],  # 左上角
        [bbox_ori[2], bbox_ori[1]],  # 右上角
        [bbox_ori[2], bbox_ori[3]],  # 右下角
        [bbox_ori[0], bbox_ori[3]]  # 左下角
    ])
    bbox_2d_int = bbox_trans.astype(np.int32)
    # 将轮廓转换为 (n, 1, 2) 的格式
    bbox_2d = bbox_2d_int.reshape(-1, 1, 2)
    return bbox_2d


def is_point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


