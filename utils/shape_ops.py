import cv2
import numpy as np

def apply_shape_filtering(mask_filtered, asp_min, circ_min):
    contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_mask = np.zeros_like(mask_filtered)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == 0:
            continue
            
        # Aspect Ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w/h, h/w)
        
        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        if aspect_ratio >= asp_min and circularity >= circ_min:
            cv2.drawContours(out_mask, [cnt], -1, 255, -1)
            
    return out_mask

def check_pill_symmetry(mask_img):
    """
    Check if a given binary mask is a symmetric pill shape.
    Returns symmetry score (0-1) and perfect pill similarity (0-1).
    """
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        
    _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0
        
    cnt = max(contours, key=cv2.contourArea)
    actual_area = cv2.contourArea(cnt)
    if actual_area == 0: return 0, 0
        
    rect = cv2.minAreaRect(cnt)
    center, (w, h), angle = rect
    
    if h > w:
        w, h = h, w
        angle += 90
        
    if h == 0 or w == 0: return 0, 0
        
    theoretical_area = (w - h) * h + np.pi * (h / 2.0)**2
    pill_similarity = min(actual_area, theoretical_area) / max(1, max(actual_area, theoretical_area))
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask_img, M, (mask_img.shape[1], mask_img.shape[0]))
    
    x, y, rw, rh = cv2.boundingRect(cv2.findNonZero(rotated_mask))
    if rw == 0 or rh == 0: return 0, 0
    cropped = rotated_mask[y:y+rh, x:x+rw]
    
    left_half = cropped[:, :rw//2]
    right_half = cropped[:, rw//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    
    if left_half.shape[1] != right_half_flipped.shape[1]:
        min_w = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_w]
        right_half_flipped = right_half_flipped[:, :min_w]
        
    lr_overlap = cv2.bitwise_and(left_half, right_half_flipped)
    lr_symmetry_score = np.count_nonzero(lr_overlap) / max(1, np.count_nonzero(left_half))
    
    top_half = cropped[:rh//2, :]
    bottom_half = cropped[rh//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    
    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        min_h = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[:min_h, :]
        bottom_half_flipped = bottom_half_flipped[:min_h, :]
        
    tb_overlap = cv2.bitwise_and(top_half, bottom_half_flipped)
    tb_symmetry_score = np.count_nonzero(tb_overlap) / max(1, np.count_nonzero(top_half))
    
    return (lr_symmetry_score + tb_symmetry_score) / 2.0, pill_similarity
