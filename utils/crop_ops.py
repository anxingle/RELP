
def remove_defects(final_rotated_image, final_rotated_mask, detector, sam2_predictor, feature_scaling_config=None, class_names=None):
    import cv2
    import numpy as np
    from utils.cv_ops import generate_sam2_mask
    
    final_output = final_rotated_image.copy()
    
    # Dilate mask slightly to avoid clipping the edges of the object
    kernel_dilate_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(final_rotated_mask, kernel_dilate_mask, iterations=1)
    
    final_output[dilated_mask == 0] = 0
    
    rotated_rgb = cv2.cvtColor(final_rotated_image, cv2.COLOR_BGR2RGB)

    outputs_rot = detector(rotated_rgb)
    instances_rot = outputs_rot["instances"]
    
    pred_classes = instances_rot.pred_classes.cpu().numpy()
    pred_boxes = instances_rot.pred_boxes.tensor.cpu().numpy()

    target_indices = []
    scaling_factors = []
    
    for i, cls_id in enumerate(pred_classes):
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        
        should_remove = False
        factor = 1.0
        
        # Check if it is a default defect (class 1)
        if cls_id == 1:
            should_remove = True
            # Check for override
            if feature_scaling_config and cls_name in feature_scaling_config:
                 factor = float(feature_scaling_config[cls_name])
                 print(f"DEBUG: Found override for {cls_name}: factor={factor}")
        
        # Check if it is in config (even if not class 1)
        elif feature_scaling_config and cls_name in feature_scaling_config:
            should_remove = True
            factor = float(feature_scaling_config[cls_name])
            print(f"DEBUG: Found config for {cls_name}: factor={factor}")
            
        if should_remove:
            target_indices.append(i)
            scaling_factors.append(factor)

    largest_defect_y = -1
    
    # [BUGFIX] 极其致命的遗漏：必须把当前正在处理的旋转后的图片重新喂给 SAM2！
    # 否则下面传入的 bbox 是基于旋转后图片的坐标，而 SAM2 脑子里还是原图，会导致抠出巨大且错位的黑块！
    sam2_predictor.set_image(rotated_rgb)
    
    # Pre-calculate internal mask for boundary check (defect removal)
    kernel_erode_internal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    internal_mask = cv2.erode(final_rotated_mask, kernel_erode_internal, iterations=1)

    for idx, factor in zip(target_indices, scaling_factors):
        box = pred_boxes[idx]
        cls_id = pred_classes[idx]
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        
        y1 = int(box[1])
        if cls_id == 1 and y1 > largest_defect_y:
            largest_defect_y = y1
            
        print(f"DEBUG: Processing Feature_Remove mask for {cls_name} with scaling factor {factor}")
        
        # In the old code, they used point+box predict. Here we use generate_sam2_mask which is slightly different
        # Let's strictly mimic the old morphological behaviors:
        
        # [BUGFIX]: The old logic used a center point + box for SAM2 to segment the camera rim accurately.
        # `generate_sam2_mask` in our utility doesn't use the point prompt, which sometimes causes SAM2 to segment the entire phone body instead of just the camera rim!
        x1, y1, x2, y2 = map(int, box)
        center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
        input_box = np.array([x1, y1, x2, y2])
        
        masks, _, _ = sam2_predictor.predict(
            point_coords=center,
            point_labels=np.array([1]),
            box=input_box[None, :],
            multimask_output=False
        )
        
        # Resize if necessary (ONNX output is sometimes smaller)
        h, w = rotated_rgb.shape[:2]
        if masks.shape[-1] != w or masks.shape[-2] != h:
             new_masks = []
             for m in masks:
                  if m.dtype != np.float32: m = m.astype(np.float32)
                  m_res = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                  new_masks.append(m_res)
             masks = np.array(new_masks)
             
        if len(masks) == 0:
            continue
            
        defect_sam_mask = (masks[0] > 0.5).astype(np.uint8) * 255
        
        # [BUGFIX]: Restore the old custom dilation scaling logic.
        # `generate_sam2_mask` does bounding-box scaling, not morphological dilation.
        if factor != 1.0:
             w_box = x2 - x1
             h_box = y2 - y1
             avg_dim = (w_box + h_box) / 2
             delta_pixels = int((avg_dim / 2) * (factor - 1.0))
             
             if delta_pixels > 0:
                 k_size = 2 * delta_pixels + 1
                 k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                 defect_sam_mask = cv2.dilate(defect_sam_mask, k, iterations=1)
             elif delta_pixels < 0:
                 delta_pixels = abs(delta_pixels)
                 if delta_pixels > 0:
                     k_size = 2 * delta_pixels + 1
                     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                     defect_sam_mask = cv2.erode(defect_sam_mask, k, iterations=1)


        # Always calculate defect properties for orientation check (feature-based)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        defect_mask_processed = cv2.morphologyEx(defect_sam_mask, cv2.MORPH_CLOSE, kernel_close)

        # Check if defect is internal (hole) or boundary (corner/edge)
        # Calculate overlap with internal mask
        overlap = cv2.bitwise_and(defect_sam_mask, internal_mask)
        overlap_count = cv2.countNonZero(overlap)
        defect_count = cv2.countNonZero(defect_sam_mask)
        
        # If overlap is small compared to defect area, it means the defect is largely outside the internal mask
        # [BUGFIX]: Only apply this boundary check for default defect (cls_id == 1) that is NOT explicitly overridden
        # If it is explicitly in feature_scaling_config (like Cam_Rim), we MUST remove it even if it's on the boundary.
        is_explicit_feature = feature_scaling_config and cls_name in feature_scaling_config
        
        if not is_explicit_feature and defect_count > 0 and (overlap_count / defect_count) < 0.8:
            print(f"DEBUG: Defect {cls_name} at boundary detected (ratio {overlap_count/defect_count:.2f}), skipping removal to preserve corners.")
            continue

        # If we decide to remove it, proceed with processing
        defect_sam_mask = defect_mask_processed # Use the closed mask

        # Fill holes inside the defect mask itself to make it solid
        contours, _ = cv2.findContours(defect_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_defect = max(contours, key=cv2.contourArea)
            filled_mask = np.zeros_like(defect_sam_mask)
            cv2.fillPoly(filled_mask, [largest_defect], 255)
            defect_sam_mask = filled_mask

        # Only apply default massive dilation if NO custom scaling was applied
        if factor == 1.0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            defect_sam_mask = cv2.dilate(defect_sam_mask, kernel_dilate, iterations=2)
            
        final_output[defect_sam_mask > 0] = 0
        
        # [THE REAL BUG]: The camera rim MUST be cut out of the mask for KNN.
        # If it's not cut out of the mask, KNN will see the black pixels we just created in final_output
        # as "dark green" or "gray" defects! 
        # BUT we must ONLY do this if we actually want it ignored by downstream algorithms.
        # The original code DOES NOT remove it from the mask, but original code didn't use `clean_image[mask==0]=0`
        # in the same way either. We must cut it out of the mask so KNN ignores it.
        final_rotated_mask[defect_sam_mask > 0] = 0
        
        print(f"Removed area of {cls_name} from output image AND mask.")
            
    return final_output, largest_defect_y

from .cv_ops import rotate_image_and_mask
from utils.detectron_ops import detect_bounding_boxes

from detectron2.data import MetadataCatalog

# --- Advanced Extracted Ops ---

import cv2
import numpy as np
import pandas as pd
import math

def resize_and_padding(image, scale_ratio):
    """
    将图片按比例缩放并放置在原尺寸的平均RGB背景中央
    """
    # 获取原始尺寸
    original_height, original_width = image.shape[:2]

    # 计算图片的平均RGB值
    mean_color = np.mean(image, axis=(0, 1))

    # 计算缩放后的尺寸
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # 缩放图片
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建平均RGB颜色的背景
    background = np.full((original_height, original_width, 3), mean_color, dtype=np.uint8)

    # 将缩放后的图片放置在背景中央
    background[0:new_height, 0:new_width] = resized_image
    max_coords = [new_width - 1, new_height - 1]
    print('max_coords', max_coords)
    return background, max_coords


def large_obj_cropping(image, rect_ratio=0.5):
    # Ensure image is grayscale for edge detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    
    # 1. Detect Edges/Lines
    # Using Canny on a mask (0/255) gives very sharp edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Use HoughLinesP to find line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=20)
    
    best_line = None
    max_len = 0
    border_thresh = 10 # Ignore lines within 10px of image border
    
    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate length
            length = np.hypot(x2-x1, y2-y1)
            
            # Check if line is on border
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if cx < border_thresh or cx > w - border_thresh or cy < border_thresh or cy > h - border_thresh:
                continue # Skip border lines
            
            if length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)

    if best_line is None:
        print("Warning: No valid reference edge found in large_obj_cropping. Using image bounds.")
        # Fallback to center crop or similar? For now, return None or full image box
        return np.array([[0,0], [w,0], [w,h], [0,h]]), 0, image

    # 2. Determine Inward Direction
    x1, y1, x2, y2 = best_line
    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3) # Draw best line in Red
    
    dx, dy = x2-x1, y2-y1
    norm = np.hypot(dx, dy)
    if norm == 0: norm = 1
    
    # Unit vector along line
    ux, uy = dx/norm, dy/norm
    
    # Normal vectors (perpendicular)
    # n1 = (-uy, ux), n2 = (uy, -ux)
    nx, ny = -uy, ux
    
    # Probe points to check which side is the mask (white/255)
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    probe_dist = 10
    
    # Check +Normal
    p1_x, p1_y = int(mid_x + nx * probe_dist), int(mid_y + ny * probe_dist)
    val1 = 0
    if 0 <= p1_x < w and 0 <= p1_y < h:
        val1 = gray[p1_y, p1_x]
        
    # Check -Normal
    p2_x, p2_y = int(mid_x - nx * probe_dist), int(mid_y - ny * probe_dist)
    val2 = 0
    if 0 <= p2_x < w and 0 <= p2_y < h:
        val2 = gray[p2_y, p2_x]
    
    # Determine correct normal (pointing INWARD to the object)
    final_nx, final_ny = nx, ny
    if val1 > 0: # p1 is inside
        final_nx, final_ny = nx, ny
    elif val2 > 0: # p2 is inside
        final_nx, final_ny = -nx, -ny
    else:
        # Fallback: try to verify with centroid if probe failed (e.g. thin edge)
        M = cv2.moments(gray)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Vector from mid to centroid
            vx, vy = cX - mid_x, cY - mid_y
            # Dot product with normal
            dot = vx * nx + vy * ny
            if dot > 0:
                final_nx, final_ny = nx, ny
            else:
                final_nx, final_ny = -nx, -ny

    # 3. Construct Crop Box
    rect_height = max_len * rect_ratio
    
    # Corners: 
    # c1, c2 are the line endpoints
    # c3, c4 are extruded endpoints
    
    c1 = (x1, y1)
    c2 = (x2, y2)
    c3 = (int(x2 + final_nx * rect_height), int(y2 + final_ny * rect_height))
    c4 = (int(x1 + final_nx * rect_height), int(y1 + final_ny * rect_height))
    
    corners = np.array([c1, c2, c3, c4], dtype=np.int32)
    
    # Draw result box on debug image
    cv2.polylines(debug_img, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    
    # Calculate Angle (of the reference line)
    angle = np.degrees(np.arctan2(dy, dx))
    
    return corners, angle, debug_img


def large_obj_cropping_v2(image, rect_ratio=0.5):
    """
    Generalized Large DUT cropping using Contour Analysis.
    1. Finds the object contour from the mask.
    2. Identifies the "Reference Edge" based on length and proximity to image borders.
       (User logic: "Average distance of edge to frame" - preferring edges placed against the boundary).
    3. Determines inward direction and constructs a crop box.
    """
    # Ensure image is single channel (mask)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Ensure binary mask for contour finding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    h, w = gray.shape[:2]
    
    # 1. Find Contours
    # Use RETR_EXTERNAL to get the outer boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found in large_obj_cropping_v2. Using image bounds.")
        return np.array([[0,0], [w,0], [w,h], [0,h]]), 0, image

    # Find largest contour by area (the object)
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify contour to polygon to get significant edges
    # Epsilon controls smoothness. 1% of arc length is a common heuristic.
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    pts = approx.reshape(-1, 2)
    num_pts = len(pts)
    
    best_segment = None
    max_score = -1
    
    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
    # Draw the approximated polygon for debug
    cv2.drawContours(debug_img, [approx], -1, (255, 255, 0), 2)

    # 2. Score Edges
    for i in range(num_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_pts] # Wrap around
        
        # Calculate Length
        length = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        
        # Skip very short segments (noise or corners)
        if length < 20: 
            continue
            
        # Calculate Midpoint
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        # Calculate Distance to Image Borders
        d_top = mid_y
        d_bottom = h - mid_y
        d_left = mid_x
        d_right = w - mid_x
        
        min_dist_border = min(d_top, d_bottom, d_left, d_right)
        
        # Score Logic:
        # We want to maximize Length and minimize Distance to Border.
        # Score = Length / (Distance + Epsilon)
        # Epsilon (e.g., 10) prevents division by zero and dampens the effect for very close edges.
        score = length / (min_dist_border + 10.0)
        
        if score > max_score:
            max_score = score
            best_segment = (p1, p2)

    if best_segment is None:
        print("Warning: No valid reference edge found in large_obj_cropping_v2. Using image bounds.")
        return np.array([[0,0], [w,0], [w,h], [0,h]]), 0, image

    # 3. Determine Inward Direction
    # (Same logic as before, but using the identified best_segment)
    p1, p2 = best_segment
    x1, y1 = p1
    x2, y2 = p2
    
    # Draw best edge in Red
    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    dx, dy = x2-x1, y2-y1
    norm = np.hypot(dx, dy)
    if norm == 0: norm = 1
    
    # Unit vector along line
    ux, uy = dx/norm, dy/norm
    
    # Normal vectors (perpendicular)
    nx, ny = -uy, ux
    
    # Probe points
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    probe_dist = 10
    
    # Check +Normal
    p_plus_x, p_plus_y = int(mid_x + nx * probe_dist), int(mid_y + ny * probe_dist)
    val_plus = 0
    if 0 <= p_plus_x < w and 0 <= p_plus_y < h:
        val_plus = gray[p_plus_y, p_plus_x] # Check mask value
        
    # Check -Normal
    p_minus_x, p_minus_y = int(mid_x - nx * probe_dist), int(mid_y - ny * probe_dist)
    val_minus = 0
    if 0 <= p_minus_x < w and 0 <= p_minus_y < h:
        val_minus = gray[p_minus_y, p_minus_x]
    
    final_nx, final_ny = nx, ny
    
    # Logic: Pick the side that has the mask (white pixel)
    if val_plus > 0: 
        final_nx, final_ny = nx, ny
    elif val_minus > 0:
        final_nx, final_ny = -nx, -ny
    else:
        # Fallback: Use Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            vx, vy = cX - mid_x, cY - mid_y
            dot = vx * nx + vy * ny
            if dot > 0:
                final_nx, final_ny = nx, ny
            else:
                final_nx, final_ny = -nx, -ny

    # 4. Construct Crop Box
    # Height based on the length of the reference edge
    # Note: We use the length of the *segment* we found, which approximates the object width along that edge.
    rect_height = norm * rect_ratio
    
    c1 = (x1, y1)
    c2 = (x2, y2)
    c3 = (int(x2 + final_nx * rect_height), int(y2 + final_ny * rect_height))
    c4 = (int(x1 + final_nx * rect_height), int(y1 + final_ny * rect_height))
    
    corners = np.array([c1, c2, c3, c4], dtype=np.int32)
    
    cv2.polylines(debug_img, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    return corners, angle, debug_img


def crop_by_bbox(image, bbox, scale_factor=1.0):
    """
    按照比例系数截取图像区域
    scale_factor: 缩放系数，>1 表示扩大范围，<1 表示缩小范围
    """
    if isinstance(bbox, (list, np.ndarray)):
        if isinstance(bbox[0], (list, np.ndarray)):
            x1, y1, x2, y2 = bbox[0]  # 如果是 [[x1, y1, x2, y2]] 格式
        else:
            x1, y1, x2, y2 = bbox  # 如果是 [x1, y1, x2, y2] 格式

    # 计算边界框的中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 计算原始宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 计算新的宽度和高度
    new_width = width * scale_factor
    new_height = height * scale_factor

    # 计算新的边界框坐标
    x1 = center_x - new_width / 2
    y1 = center_y - new_height / 2
    x2 = center_x + new_width / 2
    y2 = center_y + new_height / 2

    # 转换为整数类型
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    # 截取区域
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def calculate_min_area_rect_correction(mask):
    """
    Calculate the rotation correction needed to align the minAreaRect
    of the largest contour to the vertical axis.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    # Identify the four points
    p0, p1, p2, p3 = box
    
    # Calculate edge lengths
    d1 = np.linalg.norm(p0 - p1)
    d2 = np.linalg.norm(p1 - p2)
    
    # Select the longer edge vector
    if d1 > d2:
        v = p1 - p0
    else:
        v = p2 - p1
        
    # Calculate angle of the long edge relative to the vertical axis (y-axis)
    # Ideally, for a vertical object, this vector should be (0, 1) or (0, -1)
    # Angle w.r.t x-axis
    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    
    # We want the angle to be aligned with the nearest axis (0, 90, 180, 270)
    # This supports both Horizontal and Vertical objects without forcing Verticality.
    
    remainder = angle_deg % 90
    if remainder > 45:
        rot = 90 - remainder
    else:
        rot = -remainder
    
    return rot


def check_orientation_and_flip(image, mask, largest_defect_y=None, mask_to_sync=None):
    """
    Check if the object is upside down by analyzing the center of mass vs bounding box center.
    """
    flipped = False
    
    # Priority: Feature-based orientation check (if a large defect/hole was found)
    if largest_defect_y is not None:
        h, w = image.shape[:2]
        # If largest defect (hole) is in the bottom half, it's upside down
        # (Assuming the hole should be at the TOP)
        if largest_defect_y > h / 2:
            print(f"Orientation Check (Feature-based): Largest Hole at Y={largest_defect_y} (Bottom). Flipping 180.")
            image = cv2.rotate(image, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            if mask_to_sync is not None:
                mask_to_sync = cv2.rotate(mask_to_sync, cv2.ROTATE_180)
            flipped = True
        else:
            # print(f"Orientation Check (Feature-based): Largest Hole at Y={largest_defect_y} (Top). Correctly oriented.")
            pass
            
        if mask_to_sync is not None:
            return image, mask, mask_to_sync
        return image, mask

    # Fallback: Center of Mass logic
    # Calculate moments to find centroid
    M = cv2.moments(mask)
    if M["m00"] == 0:
        if mask_to_sync is not None:
            return image, mask, mask_to_sync
        return image, mask
        
    cY = int(M["m01"] / M["m00"])
    
    # Geometric center of the bounding box
    y_indices, _ = np.where(mask > 0)
    if len(y_indices) == 0:
        if mask_to_sync is not None:
            return image, mask, mask_to_sync
        return image, mask
        
    y_min, y_max = y_indices.min(), y_indices.max()
    geo_center_y = (y_min + y_max) // 2
    
    # Logic: The camera hole (empty space) makes the "heavy" part of the mask shift downwards 
    # if the hole is at the top. So Centroid Y > Geometric Center Y implies hole is at top.
    
    # Add threshold to prevent flipping for minor centroid differences
    threshold = 5 # Adjust based on user needs
    if cY < geo_center_y - threshold:
        # Hole is likely at the bottom (Upside down)
        print(f"Orientation Check (Center-of-Mass): Centroid ({cY}) < GeoCenter ({geo_center_y}) by {geo_center_y - cY} pixels. Flipping 180.")
        image = cv2.rotate(image, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        if mask_to_sync is not None:
            mask_to_sync = cv2.rotate(mask_to_sync, cv2.ROTATE_180)
        flipped = True
    else:
        # print(f"Orientation Check (Center-of-Mass): Centroid ({cY}) > GeoCenter ({geo_center_y}). Correctly oriented.")
        pass
        
    if mask_to_sync is not None:
        return image, mask, mask_to_sync
    return image, mask


def apply_rotations(image, mask, angle1):
    # print(f"DEBUG: apply_rotations input image shape: {image.shape}, mask shape: {mask.shape}, angle: {angle1}")
    # Single pass PCA rotation is usually sufficient if the angle calculation is correct.
    # rotation_deg1 = -(90 - angle1)
    
    # Align to nearest 90-degree axis (Horizontal or Vertical)
    # This prevents forcing horizontal objects to become vertical
    
    # [重构修复]: 恢复旧系统 utils_zee 中的绝对偏转计算逻辑
    # 注意 OBB 算出的角度可能高达 179 度，需要循环约束到 [-45, 45] 区间
    rotation_deg1 = angle1
    while abs(rotation_deg1) > 45:
        rotation_deg1 = rotation_deg1 - 90 if rotation_deg1 > 0 else rotation_deg1 + 90
        
    # 注意旧系统的 rotate_image_by_topleft 用的中心点是 TopLeft
    # 新系统统一收口到了原图中心转 (rotate_image_and_mask)
    img_s1, mask_s1 = rotate_image_and_mask(image, mask, rotation_deg1)
    # print(f"DEBUG: After 1st rotation (deg={rotation_deg1}): img_s1 shape: {img_s1.shape}")
    
    # Check if the result is actually vertical (height > width)
    # If not, rotate by another 90 degrees
    # h_s1, w_s1 = img_s1.shape[:2]
    
    # Calculate bounding box of the mask to get true object dimensions
    # y_idx, x_idx = np.where(mask_s1 > 0)
    # if len(y_idx) > 0:
    #     obj_h = y_idx.max() - y_idx.min()
    #     obj_w = x_idx.max() - x_idx.min()
    #     # print(f"DEBUG: Object dimensions after 1st rotation: H={obj_h}, W={obj_w}")
        
    #     if obj_w > obj_h:
    #         # print("DEBUG: Object is still horizontal (W > H). Applying extra 90 degree rotation.")
    #         img_s1, mask_s1 = rotate_image_and_mask(img_s1, mask_s1, 90)

    # Final correction using minAreaRect (box fitting)
    final_correction = calculate_min_area_rect_correction(mask_s1)
    # print(f"DEBUG: Final correction angle: {final_correction}")
    
    if abs(final_correction) > 0.1: 
        final_rotated_image, final_rotated_mask = rotate_image_and_mask(img_s1, mask_s1, final_correction)
        # print(f"DEBUG: After correction: final_rotated_image shape: {final_rotated_image.shape}")
    else:
        final_rotated_image, final_rotated_mask = img_s1, mask_s1

    return final_rotated_image, final_rotated_mask, (angle1, 0, 0)


def crop_and_resize(final_output, final_rotated_mask, padding=0, target_size=None):
    """
    1. Tight Crop: 消除原始背景，仅仅根据 mask 的真实上下左右边界裁剪，实现绝对纯净背景。
    2. Square Canvas (如果提供了 target_size): 将紧致裁剪出的物体等比缩放，然后放置在一个绝对纯黑的正方形中央。
    3. 坐标回溯: 提供足够的信息，让后续的掩码可以精准逆向贴回原图。
    """
    y_indices, x_indices = np.where(final_rotated_mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None, None

    # 1. 绝对紧致边界
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # 依然保留一点 padding 参数以防用户在某些特定模式（非 Dino）下想要留边
    y_min_padded = max(0, y_min - padding)
    y_max_padded = min(final_output.shape[0], y_max + padding + 1)
    x_min_padded = max(0, x_min - padding)
    x_max_padded = min(final_output.shape[1], x_max + padding + 1)
    
    # 获取原始紧致切图
    cropped_image = final_output[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    cropped_mask = final_rotated_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
    
    # 基础坐标数据
    crop_offset = {
        'x': int(x_min_padded),
        'y': int(y_min_padded),
        'w': int(x_max_padded - x_min_padded),
        'h': int(y_max_padded - y_min_padded),
        'scale': 1.0,
        'pad_x': 0,
        'pad_y': 0,
        'target_size': target_size
    }
    
    # 2. 如果不需要方图画布，直接返回
    if target_size is None:
        return cropped_image, cropped_mask, crop_offset
        
    # 3. 制作 Square Canvas (绝招复现)
    # 处理 target_size 可能为列表的情况 (e.g., [1280, 1280] or 1280)
    if isinstance(target_size, list) and len(target_size) > 0:
        t_size = target_size[0]
    else:
        try:
            t_size = int(target_size)
        except (ValueError, TypeError):
            t_size = 1280 # default fallback
            
    h, w = cropped_image.shape[:2]
    if h == 0 or w == 0:
        return cropped_image, cropped_mask, crop_offset
        
    # 等比缩放计算
    scale = min(t_size / w, t_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 缩放图像
    # [CRITICAL FIX] Restore exact Zee logic: Use INTER_LINEAR for resizing.
    # The user explicitly confirmed the smoothness of Zee's output which uses the default cv2.resize (INTER_LINEAR).
    resized_img = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 创造绝对纯黑正方形
    canvas_img = np.full((t_size, t_size, 3), 0, dtype=np.uint8)
    canvas_mask = np.full((t_size, t_size), 0, dtype=np.uint8)
    
    # 居中贴图计算
    x_offset = (t_size - new_w) // 2
    y_offset = (t_size - new_h) // 2
    
    # 贴上去
    canvas_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    canvas_mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_mask
    
    # 记录反推用的数学系数
    crop_offset['scale'] = scale
    crop_offset['pad_x'] = x_offset
    crop_offset['pad_y'] = y_offset
    
    return canvas_img, canvas_mask, crop_offset



def perform_dut_alignment(image, dut_predictor, sam2_predictor, 
                          use_batch_alignment=False, 
                          mask_scaling_factor=1.0,
                          feature_scaling_config=None,
                          is_debug_mode=False,
                          product=None,
                          target_size=None):
    import cv2
    import numpy as np
    from utils.detectron_ops import detect_bounding_boxes
    from utils.utils_general import extract_detections
    from utils.crop_ops import apply_rotations

    try:
        from detectron2.data import MetadataCatalog
    except ImportError:
        MetadataCatalog = None
    
    metadata = {}
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        thresh_val = 0.5
        if hasattr(dut_predictor, 'cfg'):
            thresh_val = dut_predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        
        print(f"DEBUG: Running detect_bounding_boxes with threshold {thresh_val}")
        tape_boxes, tape_classes = detect_bounding_boxes(image, dut_predictor)
        
        if len(tape_boxes) == 0:
            print("Warning: No DUT boxes detected.")
            return None, None, None
        
        sam2_predictor.set_image(image_rgb)
        
        import utils.utils_general as ug
        
        class_names = None
        if hasattr(dut_predictor, 'cfg') and MetadataCatalog is not None:
            if len(dut_predictor.cfg.DATASETS.TEST) > 0:
                dataset_name = dut_predictor.cfg.DATASETS.TEST[0]
                meta = MetadataCatalog.get(dataset_name)
                if hasattr(meta, 'thing_classes'):
                    class_names = meta.thing_classes

        # get DUT_Scaling_Config from utils_general where we injected it
        dut_scaling = getattr(ug, 'DUT_Scaling_Config', None)
        
        all_detections = extract_detections(
            sam2_predictor, image_rgb, tape_boxes, 
            tape_classes=tape_classes, mask_scaling_factor=mask_scaling_factor,
            dut_scaling_config=dut_scaling,
            class_names=class_names
        )
        
        if not all_detections:
            print("Warning: No DUT detections found after SAM2 segmentation.")
            return None, None, None
        
        largest = max(all_detections, key=lambda d: d["area"])
        
        # 恢复对原图的直接旋转！
        # 之前为了“恢复遮罩过滤”提前使用了 cv2.bitwise_and 把背景涂黑。
        # 这是一个致命错误，因为把背景涂黑后，再去调用 remove_defects（会二次调用 SAM2 去抠干扰件比如 Cam_Rim），
        # SAM2 会因为失去了真实的背景纹理和边界对比，导致无法正确抠出圆滑的相机圈，进而生成一个巨大的方形 Blob 掩码。
        # remove_defects 内部本身就有 final_output[dilated_mask == 0] = 0 的操作，所以不需要提前涂黑。
        clean_image = image.copy()
        
        # 1. 旋转纠正 (Rotation) - 修正：默认必须用OBB旋转，如果有PCA配置则用PCA
        angle_to_use = largest["angle"]
        if use_batch_alignment: # 表示启用了 PCA_Alignment 标志
            try:
                from utils.knn_utils import DUT_Area_OBB_Cal_knn
                _, pca_angle, _, _, _, _ = DUT_Area_OBB_Cal_knn(largest["mask"].astype(np.uint8) * 255)
                angle_to_use = pca_angle
                print(f"DEBUG: PCA_Alignment requested. Using PCA angle: {angle_to_use}")
            except Exception as e:
                print(f"Warning: PCA calculation failed, falling back to OBB angle {angle_to_use}. Error: {e}")
        else:
            print(f"DEBUG: Using standard OBB angle for rotation: {angle_to_use}")

        # 核心：无论如何都要旋转摆正！
        rotated_image, rotated_mask, _ = apply_rotations(
            clean_image, largest["mask"], angle_to_use
        )
        
        # [BUGFIX]: 在老代码中，remove_defects（挖除相机圈等干扰件）是发生在使用 crop_and_resize 裁切之前的！
        # 如果先 tight_crop 裁切并缩放，Detectron 会因为图片边缘没有 padding 以及分辨率改变，导致找框或者 SAM 抠图出现严重变形（变成巨大的黑块）。
        # 所以必须在原始旋转后的全尺寸图像上先执行 remove_defects！
        final_output_pre_crop, largest_defect_y = remove_defects(
            rotated_image, rotated_mask, dut_predictor, sam2_predictor,
            feature_scaling_config=feature_scaling_config,
            class_names=class_names
        )
        metadata['largest_defect_y'] = largest_defect_y
        
        # 2. Tight Crop & Square Canvas
        # 在完全清理完所有干扰件之后，再去进行紧致裁切和画布缩放。
        print(f"DEBUG: TIGHT CROP APPLIED on ROTATED image. Padding=0, Target_Size={target_size}")
        from utils.crop_ops import crop_and_resize
        final_rotated_image, final_rotated_mask, metadata_offset = crop_and_resize(
            final_output_pre_crop, rotated_mask, padding=0, target_size=target_size
        )
        
        if final_rotated_image is None:
            final_rotated_image = final_output_pre_crop
            final_rotated_mask = rotated_mask
        else:
            metadata['crop_offset'] = metadata_offset
        
        # 此时 final_rotated_mask 中已经被 remove_defects 打了洞，但在 crop_and_resize 中，
        # 如果是基于最大的外部边界（y_indices, x_indices = np.where(final_rotated_mask > 0)），
        # 内部被打的洞会自动被裁剪保留。这里为了安全起见，需要在 crop_and_resize 里确保内部的 0 也跟着被裁出来。
        # (目前的 crop_and_resize 代码是直接切取整个矩形块，内部的洞会自动被包含，所以非常安全)
        
        return final_rotated_image, final_rotated_mask, metadata
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Error in perform_dut_alignment: {e}")
        return None, None, None

def preprocess_alignment(path_to_process, result_folder, current_img_rgb, dut_predictor, sam2_predictor, model=None, is_debug_mode=False):
    import cv2
    import numpy as np
    import os
    try:
        from utils.crop_ops import perform_dut_alignment
    except: pass
    
    image = cv2.imread(path_to_process)
    if image is None:
        return None, None, None, path_to_process
        
    DUT_Corrected, Contour_Corrected, alignment_metadata = perform_dut_alignment(
        image, dut_predictor, sam2_predictor, use_batch_alignment=False, is_debug_mode=is_debug_mode
    )
    
    if DUT_Corrected is not None:
        if is_debug_mode and result_folder:
            ref_dir = os.path.join(result_folder, "reference")
            os.makedirs(ref_dir, exist_ok=True)
            fname = f"{os.path.splitext(os.path.basename(path_to_process))[0]}_rotated90_cropped.jpg"
            out_path = os.path.join(ref_dir, fname)
            cv2.imwrite(out_path, DUT_Corrected)
            return DUT_Corrected, Contour_Corrected, alignment_metadata, out_path
    
    return DUT_Corrected, Contour_Corrected, alignment_metadata, path_to_process
