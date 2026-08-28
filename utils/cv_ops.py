import os
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def create_overlay_image(base_img, mask, color=(0, 162, 255), transparency=0.8):
    """
    Creates an overlay image.
    :param base_img: Base image (RGB or BGR, but assumed consistent with output).
    :param mask: Mask (2D or 3D).
    :param color: Color tuple (default RGB 0, 162, 255).
    :param transparency: Transparency of the overlay layer (0.0 to 1.0). 
                         0.8 means 80% transparent (20% opacity).
                         Weight of overlay = 1 - transparency.
                         Weight of base = transparency.
    :return: Blended image.
    """
    if base_img is None:
        return None
        
    h, w = base_img.shape[:2]
    mask_vis = mask
    
    if mask_vis is None:
        return base_img.copy()

    if mask_vis.ndim == 3:
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2GRAY)
    if mask_vis.dtype == bool:
        mask_vis = (mask_vis.astype(np.uint8) * 255)
    if mask_vis.dtype != np.uint8:
        mask_vis = mask_vis.astype(np.uint8)
    
    # Resize mask if needed
    if mask_vis.shape[:2] != (h, w):
        mask_vis = cv2.resize(mask_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    color_layer = np.zeros_like(base_img)
    color_layer[mask_vis > 0] = color
    
    # Calculate weights based on transparency
    # User clarification: "Transparency 80% (i.e. mask is very strong)"
    # This implies Opacity 80% -> Overlay Weight 0.8, Base Weight 0.2
    # So we invert the previous logic interpretation
    # Now user requests 50% Opacity -> Overlay Weight 0.5, Base Weight 0.5
    
    alpha = transparency  # Overlay weight
    beta = 1.0 - transparency # Base weight
    
    # Ensure values are within [0, 1]
    alpha = max(0.0, min(1.0, alpha))
    beta = max(0.0, min(1.0, beta))
    
    blended = cv2.addWeighted(base_img, beta, color_layer, alpha, 0.0)
    
    overlay_res = base_img.copy()
    idx = (mask_vis > 0)
    if np.any(idx):
        overlay_res[idx] = blended[idx]
        
    return overlay_res


def save_inferred_pic_overlay(base_img, mask, out_dir, filename_base, suffix="", output_name=None, input_is_bgr=False):
    try:
        if base_img is None or mask is None or not out_dir or not filename_base:
            return False
        if input_is_bgr:
            base_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        else:
            base_rgb = base_img

        # Check for custom Mask Color from module-level global variable
        # Mask_Color is set in RELP3_main.py as utils_general.Mask_Color
        mask_color = globals().get('Mask_Color', None)
        if mask_color and isinstance(mask_color, dict):
            color = mask_color.get('rgb', (0, 162, 255))
            transparency = mask_color.get('transparency', 0.5)
            print(f"DEBUG: Using custom Mask Color: RGB{color}, Transparency: {transparency}")
        else:
            # Default: RGB 0, 162, 255 and Transparency 50% (Opacity 50%)
            color = (0, 162, 255)
            transparency = 0.5

        # Use unified creation function
        overlay_rgb = create_overlay_image(base_rgb, mask, color=color, transparency=transparency)
        
        if overlay_rgb is None:
            return False

        # Use original filename_base directly to avoid naming conflicts
        # The filename_base should already be clean (processed by caller)
        # Removing suffixes here can cause conflicts when different original files
        # share the same base name after suffix removal
        clean_fname = filename_base
        os.makedirs(out_dir, exist_ok=True)
        overlay_name = output_name if output_name else f"{clean_fname}_overlay{suffix}.jpg"
        
        # CRITICAL FIX: Reduce file size for large images
        h, w = overlay_rgb.shape[:2]
        MAX_DIM = 2048  # Cap max dimension to reduce file size
        
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            # Use INTER_NEAREST for overlay to preserve mask edges and avoid blurring
            overlay_rgb = cv2.resize(overlay_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            print(f"DEBUG: Resized overlay from ({h}, {w}) to ({new_h}, {new_w}) to reduce file size")
        
        # Save as JPG with quality 85 (good balance between quality and size)
        overlay_path = os.path.join(out_dir, overlay_name)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Log file size for debugging
        if os.path.exists(overlay_path):
            file_size_mb = os.path.getsize(overlay_path) / (1024 * 1024)
            print(f"DEBUG: Saved overlay {overlay_name} ({file_size_mb:.2f} MB), size: {overlay_rgb.shape}")
        
        return True
    except Exception:
        return False


def extract_hsv_config_from_filtering(filtering_df, require_hole_prefix=True):
    """Extract HSV configuration from filtering DataFrame
    
    Args:
        filtering_df: Filtering configuration DataFrame
        require_hole_prefix: If True, only accept Method values starting with 'Hole' (e.g., 'Hole_HSV')
    
    Returns:
        config: Dictionary with HSV configuration and optional resize_scale
    """
    config = {}
    try:
        if not filtering_df.empty:
            row = filtering_df.iloc[0]
            
            # Check Method column if require_hole_prefix is True
            method_col = None
            for col in filtering_df.columns:
                if str(col).lower().strip() == 'method':
                    method_col = col
                    break
            
            if require_hole_prefix and method_col:
                method_val = row.get(method_col)
                if pd.notna(method_val):
                    method_str = str(method_val).strip()
                    if not method_str.lower().startswith('hole'):
                        print(f"DEBUG: Method '{method_str}' does not start with 'Hole', skipping HSV extraction")
                        return config
                    else:
                        print(f"DEBUG: Method '{method_str}' starts with 'Hole', proceeding with HSV extraction")
            
            # First, try to find Method Setting column for Hole_HSV
            method_setting_col = None
            print(f"DEBUG: Looking for Method Setting column. Available columns: {list(filtering_df.columns)}")
            for col in filtering_df.columns:
                col_lower = str(col).lower().strip()
                print(f"DEBUG: Checking column: '{col}' -> '{col_lower}'")
                if 'method setting' in col_lower or col_lower == 'methodsetting':
                    method_setting_col = col
                    print(f"DEBUG: Found Method Setting column: '{col}'")
                    break
            
            if method_setting_col:
                method_setting_val = row.get(method_setting_col)
                print(f"DEBUG: Method Setting value: '{method_setting_val}' (type: {type(method_setting_val)})")
                if pd.notna(method_setting_val):
                    # Parse format like [(0,255), (16,255), (0,255), Size = 1/2]
                    val_str = str(method_setting_val).strip()
                    try:
                        import re
                        print(f"DEBUG: Attempting to parse Method Setting value: '{val_str}'")
                        
                        # Simplified parsing: find Size first, then parse HSV from the part before Size
                        # Format: [((0,255), (16,255), (0,255)), Size = 1/2] or [(0,255), (16,255), (0,255), Size = 1/2]
                        
                        # Step 1: Extract Size parameter
                        size_match = re.search(r'Size\s*=\s*(\d+)\s*/\s*(\d+)', val_str, re.IGNORECASE)
                        if size_match:
                            size_num, size_den = size_match.groups()
                            resize_scale = float(size_num) / float(size_den)
                            config['resize_scale'] = resize_scale
                            print(f"DEBUG: Found Size parameter: {size_num}/{size_den} = {resize_scale}")
                        
                        # Step 2: Extract HSV values
                        # Look for three (number,number) patterns
                        hsv_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\).*\(\s*(\d+)\s*,\s*(\d+)\s*\).*\(\s*(\d+)\s*,\s*(\d+)\s*\)', val_str)
                        if hsv_match:
                            h_min, h_max, s_min, s_max, v_min, v_max = hsv_match.groups()
                            config['HSV_H'] = f"{h_min}-{h_max}"
                            config['HSV_S'] = f"{s_min}-{s_max}"
                            config['HSV_V'] = f"{v_min}-{v_max}"
                            print(f"DEBUG: Parsed HSV: H=[{h_min}-{h_max}], S=[{s_min}-{s_max}], V=[{v_min}-{v_max}]")
                        else:
                            print(f"DEBUG: Could not find HSV values in: '{val_str}'")
                    except Exception as e:
                        print(f"DEBUG: Failed to parse Method Setting as HSV: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"DEBUG: Method Setting column not found in filtering_df columns: {list(filtering_df.columns)}")
            
            # Fallback: Look for HSV-related columns
            if not config:
                for col in filtering_df.columns:
                    col_lower = str(col).lower()
                    if 'hsv' in col_lower or 'hue' in col_lower or 'sat' in col_lower or 'val' in col_lower:
                        val = row.get(col)
                        if pd.notna(val):
                            config[col] = val
        
        print(f"DEBUG: Extracted HSV config: {config}")
    except Exception as e:
        print(f"Warning: Failed to extract HSV config: {e}")
    return config


def extract_gray_config_from_filtering(filtering_df):
    """Extract Gray configuration from filtering DataFrame"""
    config = {}
    try:
        if not filtering_df.empty:
            row = filtering_df.iloc[0]
            # Look for Gray-related columns
            for col in filtering_df.columns:
                col_lower = str(col).lower()
                if 'gray' in col_lower or 'grey' in col_lower:
                    val = row.get(col)
                    if pd.notna(val):
                        config[col] = val
        print(f"DEBUG: Extracted Gray config: {config}")
    except Exception as e:
        print(f"Warning: Failed to extract Gray config: {e}")
    return config


def apply_gray_filtering_simple(image_rgb, gray_config):
    """Apply Gray filtering to image"""
    try:
        # Convert to Gray
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Default: accept all
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        # Parse Gray range from config
        g_min, g_max = 0, 255
        
        for key, val in gray_config.items():
            try:
                if isinstance(val, str) and '-' in val:
                    parts = val.strip('[]').split('-')
                    g_min = int(parts[0])
                    g_max = int(parts[1])
            except:
                pass
        
        # Create Gray mask
        mask = cv2.inRange(gray, g_min, g_max)
        
        print(f"DEBUG: Gray filtering with range=[{g_min}-{g_max}]")
        return mask
        
    except Exception as e:
        print(f"Warning: Gray filtering failed: {e}")
        return np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255


def normalize_contours(rect_points, contours):
    """
    将轮廓坐标映射为相对于最小外接矩形(OBB)的 (0.0~1.0) 比例坐标。
    修复了原本数组形状越界报错、点阵排序不稳的问题。采用防弹级仿射变换。
    """
    import cv2
    import numpy as np
    
    # 防御：处理单层传入还是双层传入的情况 (contours 可能是一个单个的轮廓，或者是轮廓列表)
    if isinstance(contours, np.ndarray) and len(contours.shape) >= 2:
        contours = [contours]
        
    rect_points = np.array(rect_points, dtype=np.float32)
    if rect_points.shape != (4, 2):
        if len(rect_points) == 4 and len(rect_points[0]) == 2:
            rect_points = rect_points.reshape((4, 2))
        else:
            return contours # 无法处理

    # OpenCV 的标准的“透视变换法”
    # 1. 对矩形四个顶点进行排序 (左上, 右上, 右下, 左下)
    # 算质心
    center = np.mean(rect_points, axis=0)
    # 计算所有点到质心的角度并排序 (极坐标角度)
    angles = np.arctan2(rect_points[:, 1] - center[1], rect_points[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    pts_ordered = rect_points[sorted_idx] # 这个通常是: 左上, 右上, 右下, 左下 (如果是常规摆放)
    
    # 2. 计算宽和高
    # 边长 1
    width = np.linalg.norm(pts_ordered[1] - pts_ordered[0])
    # 边长 2
    height = np.linalg.norm(pts_ordered[3] - pts_ordered[0])
    
    if width < 1 or height < 1:
        return contours

    # 3. 构造理想的“归一化坐标系” (单位矩阵)
    # 我们希望把左上角变成 (0,0), 右下角变成 (1,1)
    pts_normalized = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    # 4. 获取透视变换矩阵 (3x3 矩阵)
    # 这一步直接将物理世界的倾斜矩形，完美拉伸映射到 (0,1) 的单位正方形上！
    try:
        matrix = cv2.getPerspectiveTransform(pts_ordered, pts_normalized)
    except:
        return contours

    normalized_contours = []
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
            
        try:
            # 确保轮廓是正确的形状 (N, 1, 2)
            cnt_arr = np.array(contour, dtype=np.float32)
            if len(cnt_arr.shape) == 2:
                cnt_arr = cnt_arr.reshape(-1, 1, 2)
            elif len(cnt_arr.shape) == 3 and cnt_arr.shape[2] != 2:
                continue

            # 使用 OpenCV 强大的 perspectiveTransform 一键完成旋转、缩放、偏移的计算！
            normalized_pts = cv2.perspectiveTransform(cnt_arr, matrix)
            
            # 由于可能出现超出框的毛刺，强行约束在 0-1 之间
            normalized_pts = np.clip(normalized_pts, 0.0, 1.0)
            
            # 转回列表并保留 4 位小数
            normalized = np.round(normalized_pts.reshape(-1, 2), decimals=4).tolist()
            normalized_contours.append(normalized)
            
        except Exception as e:
            # 如果某个轮廓异常，跳过它，防止影响整体
            print(f"DEBUG: Contours array shape error skipped: {e}")
            continue

    return normalized_contours


def cv_show(name, img):
    # window_title = str(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_show(image, window_title='Image Window'):
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mask_based_matting(image, mask):
    """""
    1. 尺寸匹配：
        - 图片和掩码的高度和宽度必须完全相同
        - 例如：如果图片是(864, 1280) ，掩码也必须是(864, 1280)
    2. 通道数匹配：
        - 如果图片是三通道(H, W, 3) ，掩码也需要是三通道
        - 如果掩码是单通道，需要用 cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)转换
    3. 数据类型匹配：
        - 两者必须是相同的数据类型，通常是 np.uint8
        - 掩码的值应该是二值化的（0或255）
    4. 数值范围：
        - 图片：0 - 255
        - 掩码：最好是二值化的（0表示背景，255表示前景）
    """""

    if len(mask.shape) == 2:  # 如果掩码是单通道
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3channel = mask
    # print("Image shape:", image.shape)
    # print("Mask shape:", mask_3channel.shape)
    # Ensure mask has same size as image
    if mask_3channel.shape[:2] != image.shape[:2]:
        mask_3channel = cv2.resize(mask_3channel, (image.shape[1], image.shape[0]))
    image = image.astype(np.uint8)
    mask_3channel = mask_3channel.astype(np.uint8)

    # Ensure mask values are either 0 or 255
    mask_3channel = np.where(mask_3channel > 0, 255, 0).astype(np.uint8)
    segmented_image = cv2.bitwise_and(image, mask_3channel)
    # cv2.imshow('segmented_image', segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return segmented_image


def contour_based_matting(image, contour):
    try:
        # 创建与原图同样大小的掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # 在掩码上填充轮廓
        if contour is not None and len(contour) > 0:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            # 如果图像是彩色的，需要将掩码转换为3通道
            if len(image.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # 使用掩码提取区域
            extracted_region = cv2.bitwise_and(image, mask)
            return extracted_region
        else:
            print("警告: contour_based_matting中的轮廓无效")
            return image  # 返回原始图像作为后备
    except Exception as e:
        print(f"警告: contour_based_matting处理中出错: {e}")
        return image  # 返回原始图像作为后备


def get_intersection_mask_and_contour(mask, polygon_bbox):
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    bbox_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    polygon_bbox = polygon_bbox.astype(np.int32)
    cv2.fillPoly(bbox_mask, [polygon_bbox], 255)
    intersection_mask = cv2.bitwise_and(mask, bbox_mask)
    # cv_show(intersection_mask, 'intersection_mask')

    img_array = np.array(intersection_mask)
    # 如果是3通道图像，转换为灰度图
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Fails to detect mask contour")
        return None

    main_contour = max(contours, key=cv2.contourArea)

    h, w = binary.shape[:2]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(result, [main_contour], -1, (0, 255, 0), 2)
    return intersection_mask, main_contour


def rotate_image_by_topleft(segmented_image, rect_points, angle):
    """
    以最小外接矩形的左上顶点为中心旋转图像区域，保持位置不变
    """
    # 确保输入图像不为空
    if segmented_image is None:
        raise ValueError("输入图像为空")

    # 确保rect_points是numpy数组
    rect_points = np.array(rect_points, dtype=np.float32)

    # 找到左上角顶点（x最小且y最小的点）
    top_points = rect_points[rect_points[:, 1].argsort()][:2]
    topleft = top_points[top_points[:, 0].argmin()]

    # 将topleft转换为float类型
    center = (float(topleft[0]), float(topleft[1]))

    # 计算旋转角度
    if abs(angle) > 45:
        rotation_angle = angle - 90 if angle > 0 else angle + 90
    else:
        rotation_angle = angle

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 使用原始图像尺寸进行旋转，保持位置不变
    h, w = segmented_image.shape[:2]
    rotated_image = cv2.warpAffine(segmented_image, rotation_matrix, (w, h))

    return rotated_image


def rotate_contour_by_topleft(contour, rect_points, angle):
    """
    以最小外接矩形的左上顶点为中心旋转轮廓

    参数:
    contour: 输入轮廓点集
    rect_points: 最小外接矩形的四个顶点坐标 shape为(4,2)的numpy数组
    angle: 当前矩形的偏转角度（度数）

    返回:
    rotated_contour: 旋转后的轮廓点集
    """
    # 确保输入contour不为空
    if contour is None or len(contour) == 0:
        raise ValueError("输入轮廓为空")

    # 确保contour是正确的格式
    contour = np.array(contour, dtype=np.float32)
    if len(contour.shape) == 2:
        contour = contour.reshape(-1, 1, 2)

    # 确保rect_points是numpy数组
    rect_points = np.array(rect_points, dtype=np.float32)

    # 找到左上角顶点（x最小且y最小的点）
    top_points = rect_points[rect_points[:, 1].argsort()][:2]
    topleft = top_points[top_points[:, 0].argmin()]

    # 将topleft转换为float类型
    center = (float(topleft[0]), float(topleft[1]))

    # 计算旋转角度
    if abs(angle) > 45:
        rotation_angle = angle - 90 if angle > 0 else angle + 90
    else:
        rotation_angle = angle

    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 重塑轮廓点集以便进行旋转变换
    contour_reshaped = contour.reshape(-1, 2)

    # 执行旋转变换
    ones = np.ones(shape=(len(contour_reshaped), 1))
    points_ones = np.hstack([contour_reshaped, ones])
    transformed_points = rotation_matrix.dot(points_ones.T).T

    # 将转换后的点集重新格式化为轮廓格式
    rotated_contour = transformed_points.reshape(-1, 1, 2)

    # 确保输出的轮廓点是整数类型
    rotated_contour = rotated_contour.astype(np.int32)

    # 验证输出轮廓的有效性
    if rotated_contour is None or len(rotated_contour) == 0:
        raise ValueError("旋转后的轮廓为空")

    return rotated_contour


def normalize_contours(rect_points, contours):
    """
    将contours从图片坐标系转换到以x1为原点的长方形归一化坐标系
    """
    # 提取长方形顶点
    x1 = rect_points[0]  # 左上角
    x2 = rect_points[1]  # 右上角
    x3 = rect_points[2]  # 右下角
    x4 = rect_points[3]  # 左下角

    # 计算长方形的方向向量
    v1 = x2 - x1  # x1x2向量（横轴）
    v2 = x4 - x1  # x1x4向量（竖轴）

    # 计算长方形的宽和高
    width = np.linalg.norm(v1)
    height = np.linalg.norm(v2)

    # 计算x1x2与水平方向的夹角
    angle = np.arctan2(v1[1], v1[0])  # 获取当前角度

    # 计算四种可能的旋转角度（0°、90°、180°、270°）相对于当前角度的差值
    possible_angles = [
        angle,  # 0度
        angle - np.pi / 2,  # 90度
        angle - np.pi,  # 180度
        angle - 3 * np.pi / 2,  # 270度
    ]

    # 将所有角度规范化到 [-pi, pi] 区间
    possible_angles = [((a + np.pi) % (2 * np.pi) - np.pi) for a in possible_angles]

    # 选择绝对值最小的角度作为旋转角度
    rotation_angle = min(possible_angles, key=abs)

    # 计算旋转矩阵的参数
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    # 构建旋转矩阵
    rotation_matrix = np.array([[cos_theta, sin_theta],
                                [-sin_theta, cos_theta]])

    # 对每个轮廓进行变换
    normalized_contours = []
    for contour in contours:
        # 将轮廓转换为n*2数组
        contour = contour.reshape(-1, 2)

        # 平移到以x1为原点
        translated = contour - x1

        # 旋转
        rotated = np.dot(translated, rotation_matrix.T)

        # 归一化
        normalized = rotated.copy()
        normalized[:, 0] /= width
        normalized[:, 1] /= height

        normalized = np.round(normalized, decimals=4)
        normalized_list = normalized.reshape(-1, 2).tolist()
        normalized_contours.append(normalized_list)

    return normalized_contours


def contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center_point = np.array([center_x, center_y]).astype(int)
    point = contour[0][0]
    points_array = np.array([
        [center_x, center_y],  # 中心点
        [point[0], point[1]]  # 轮廓上的点
    ])
    return points_array


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0, 6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def scaling_bbox_back2_ori_pic(obj_box, scalar_x, scalar_y, resized_h, resized_w, pic_h, pic_w):
    # bbox structured in xyxy format
    box_w = obj_box[:, 2] - obj_box[:, 0]
    box_h = obj_box[:, 3] - obj_box[:, 1]
    box_center_resized_x = obj_box[:, 0] + 1 / 2 * box_w
    box_center_resized_y = obj_box[:, 1] + 1 / 2 * box_h
    width_scaling_ratio = pic_w / resized_w
    height_scaling_ratio = pic_h / resized_h
    box_center_x = box_center_resized_x * width_scaling_ratio
    box_center_y = box_center_resized_y * height_scaling_ratio
    # enlarge bbox by certain ratio to compensate bbox training losses
    scaled_x_left = np.clip(box_center_x - 1 / 2 * scalar_x * box_w * width_scaling_ratio, a_min=0, a_max=box_center_x)
    scaled_x_right = np.clip(box_center_x + 1 / 2 * scalar_x * box_w * width_scaling_ratio, a_min=box_center_x,
                             a_max=pic_w)
    scaled_y_up = np.clip(box_center_y - 1 / 2 * scalar_y * box_h * height_scaling_ratio, a_min=0, a_max=box_center_y)
    scaled_y_bottom = np.clip(box_center_y + 1 / 2 * scalar_y * box_h * height_scaling_ratio, a_min=box_center_y,
                              a_max=pic_h)
    scaled_box = np.column_stack((scaled_x_left, scaled_y_up, scaled_x_right, scaled_y_bottom))
    # print('width_scaling_ratio', width_scaling_ratio)
    # print('height_scaling_ratio', height_scaling_ratio)
    return scaled_box


def check_pic_SN_existance(root_dir, required_duplicates):
    # 定义允许的文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    excel_extensions = {'.xlsx', '.xls', '.csv'}
    all_extensions = image_extensions | excel_extensions

    result_files = []

    # 遍历根目录下的所有文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"\nChecking folder: {dirpath}")
        print(f"File list: {filenames}")

        # 跳过隐藏文件夹
        if os.path.basename(dirpath).startswith('.'):
            print("Skipping hidden folder")
            continue

        # 获取当前文件夹中的所有合法文件
        valid_files = [
            os.path.splitext(f)
            for f in filenames
            if not f.startswith('.') and
               os.path.splitext(f)[1].lower() in all_extensions
        ]
        print(f"Valid files: {valid_files}")

        if not valid_files:
            print("No valid files found, skipping to next folder")
            continue

        # 只有当required_duplicates > 1时才检查文件类型共存
        if required_duplicates > 1:
            has_image = any(ext.lower() in image_extensions for _, ext in valid_files)
            has_excel = any(ext.lower() in excel_extensions for _, ext in valid_files)
            print(f"Checking file type coexistence: Image={has_image}, Excel={has_excel}")

            if not (has_image and has_excel):
                print("Required file types do not coexist, skipping to next folder")
                continue

        # 使用字典统计同名文件
        filename_counts = {}
        for base_name, _ in valid_files:
            filename_counts[base_name] = filename_counts.get(base_name, 0) + 1
        print(f"Filename counts: {filename_counts}")

        # 找出重复次数等于required_duplicates的文件名
        for filename, count in filename_counts.items():
            if count == required_duplicates:
                result_files.append(filename)
                print(f"Found matching file: {filename}")

    print(f"\nFinal result: {result_files}")
    return result_files


def selective_masking(img, masks_df, pic_name, save_path):
    mask = np.zeros_like(img)
    contours = masks_df['Contour'].to_numpy()
    cv2.drawContours(mask, contours, -1, (255, 0, 0), cv2.FILLED)
    alpha = 0.4
    overlay = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)
    maskpic_title = pic_name + '_defect.jpg'
    pic_save_path = os.path.join(save_path, maskpic_title)
    cv2.imshow('Overlay', overlay)
    cv2.imwrite(pic_save_path, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_lvl_cal(image, TH_lower, TH_upper):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将 BGR 转换为 HSV
    # 创建掩模
    mask = cv2.inRange(image, TH_lower, TH_upper)
    # 应用掩模
    blue_region = cv2.bitwise_and(image, image, mask=mask)
    blue_region = cv2.cvtColor(blue_region, cv2.COLOR_HSV2RGB)
    cv2.imshow('blue_region', blue_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blue_region


def color_detection(RBG):
    start_col = 'TH_R_Min'  # 从列'xx'开始
    num_cols = 3  # 连续三列
    color_dict = 'RELP_Configuration.xlsx'
    try:
        df = ConfigManager().get_sheet("color_dict")
    except Exception as e:
        print(f"Warning: Could not load color_dict from {color_dict}: {e}. Continuing with default or empty config.")
        df = pd.DataFrame()
    start_col_index = df.columns.get_loc(start_col)
    R = RBG[0]
    G = RBG[1]
    B = RBG[2]
    minimum = 10000
    for i in range(len(df)):
        d = abs(B - int(df.loc[i, "B"])) + abs(G - int(df.loc[i, "G"])) + abs(R - int(df.loc[i, "R"]))
        if (d <= minimum):
            minimum = d
            cname = df.loc[i, "color_name"]
            TH_lower = df.iloc[i, start_col_index:start_col_index + num_cols]
            TH_lower = TH_lower.infer_objects(copy=False)
            TH_lower = np.array(TH_lower.values).astype(int)
            TH_upper = df.iloc[i, start_col_index + num_cols:start_col_index + 2 * num_cols]
            TH_upper = TH_upper.infer_objects(copy=False)
            TH_upper = np.array(TH_upper.values).astype(int)
    return cname, TH_lower, TH_upper


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def contour_finding(contour_SAM):
    # 转换为二维数组
    mask = contour_SAM.squeeze(0)  # 形状变为 (3648, 5472)
    # 确保数组是连续的
    mask = np.ascontiguousarray(mask)
    # 将掩码转换为 8 位无符号整数
    mask = (mask * 255).astype(np.uint8)  # 将 1 转换为 255，0 保持不变
    # 显示掩码
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    fine_seg_area, _, seg_contour, _, _, _ = DUT_Area_OBB_Cal(mask)
    return seg_contour


def contour_scale(arr, scaler):
    for i in range(arr.shape[-1]):
        col_mean = np.mean(arr[:, :, i])
        arr[:, :, i] = np.where(arr[:, :, i] > col_mean, np.round((arr[:, :, i] - col_mean) * scaler + col_mean),
                                np.round(col_mean - (col_mean - arr[:, :, i]) * scaler))
    return (arr)


def load_and_validate_image(image_path):
    str_path = str(image_path)
    
    # Handle HEIC files
    if str_path.lower().endswith('.heic'):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            
            pil_image = Image.open(str_path)
            # Handle EXIF orientation
            try:
                from PIL import ImageOps
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                pass
                
            image = np.array(pil_image)
            
            # Convert PIL RGB/RGBA to OpenCV BGR
            if pil_image.mode == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif pil_image.mode == 'RGBA':
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                
            return image
        except ImportError:
            print(f"Error: Cannot read {image_path}. 'pillow-heif' is required for HEIC images.")
            print("Please install it: pip install pillow-heif")
            return None
        except Exception as e:
            print(f"Error reading HEIC file {image_path}: {e}")
            return None

    # Standard loading for other formats
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"️ Skipping invalid image: {image_path}")
        return None
    return image


def extract_class_masks(image, detector, target_class_names, score_thresh=0.5):
    """
    Extract masks from bounding boxes of specific target classes.
    Creates a binary mask where pixels inside the bounding boxes of target classes are 255.
    
    Args:
        image: BGR image (numpy array)
        detector: Detectron2 predictor or similar with inference capability
        target_class_names: List or set of target class names to extract (case-insensitive)
        score_thresh: Minimum confidence score to include
        
    Returns:
        combined_mask: Binary mask (numpy array) with 255 inside target class bboxes, 0 elsewhere
        target_detections: List of dicts containing detection info for target classes only
                         Each dict includes 'mask_pixels' if model provides masks
    """
    from detectron2.data import MetadataCatalog
    
    if isinstance(target_class_names, str):
        target_class_names = {target_class_names.lower()}
    else:
        target_class_names = set(c.lower() for c in target_class_names)
    
    h, w = image.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    target_detections = []
    
    # Run inference
    outputs = detector(image)
    instances = outputs["instances"]
    
    if len(instances) == 0:
        print(f"DEBUG: No detections found for mask extraction.")
        return combined_mask, target_detections
    
    # Get predictions
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    
    # Check if model has mask predictions
    has_masks = instances.has("pred_masks")
    if has_masks:
        pred_masks = instances.pred_masks.cpu().numpy()
    
    # Get class names
    class_names = None
    if hasattr(detector, 'cfg'):
        try:
            if len(detector.cfg.DATASETS.TEST) > 0:
                test_ds = detector.cfg.DATASETS.TEST[0]
                meta = MetadataCatalog.get(test_ds)
                if hasattr(meta, 'thing_classes'):
                    class_names = meta.thing_classes
        except Exception as e:
            print(f"DEBUG: Could not retrieve class names: {e}")
    
    # Process each detection
    for i in range(len(instances)):
        score = scores[i]
        if score < score_thresh:
            continue
        
        cls_id = pred_classes[i]
        box = boxes[i]
        
        # Get class name
        if class_names and cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"Class_{cls_id}"
        
        # Check if this class is in target classes
        if cls_name.lower() in target_class_names:
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Calculate actual mask pixels if masks are available
            mask_pixels = None
            if has_masks:
                # Get the mask for this instance
                instance_mask = pred_masks[i]
                # Resize mask to match bbox size if needed
                if instance_mask.shape[0] != (y2 - y1) or instance_mask.shape[1] != (x2 - x1):
                    instance_mask = cv2.resize(instance_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                # Count non-zero pixels in the mask
                mask_pixels = int(np.sum(instance_mask > 0))
                print(f"DEBUG: {cls_name} mask pixels: {mask_pixels}, bbox area: {(x2-x1)*(y2-y1)}")
            
            # Fill the bounding box area in the combined mask
            combined_mask[y1:y2, x1:x2] = 255
            
            det_info = {
                'class_id': int(cls_id),
                'class_name': cls_name,
                'score': float(score),
                'bbox': [x1, y1, x2, y2]
            }
            # Add mask_pixels if available
            if mask_pixels is not None:
                det_info['mask_pixels'] = mask_pixels
            
            target_detections.append(det_info)
            
            print(f"DEBUG: Extracted mask for {cls_name} (ID: {cls_id}), Box: [{x1}, {y1}, {x2}, {y2}]")
    
    print(f"DEBUG: Total target class detections: {len(target_detections)}, Mask pixels: {cv2.countNonZero(combined_mask)}")
    
    return combined_mask, target_detections


def generate_sam2_mask(sam2_predictor, image_rgb, box, mask_scaling_factor=1.0):
    import math
    x1, y1, x2, y2 = box
    
    # 彻底恢复 test_k11p 的原生 SAM 解析！
    # 强制抛弃额外的 Padding。因为 test_k11p 是无 padding 切图，所以这里还原为绝对原汁原味
    padding = 0
    h, w = image_rgb.shape[:2]
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    
    center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    input_box = np.array([x1, y1, x2, y2])

    masks, scores, _ = sam2_predictor.predict(
        point_coords=center,
        point_labels=np.array([1]),
        box=input_box[None, :],
        multimask_output=False
    )

    if len(masks) == 0:
        return None

    # Resize mask if necessary (e.g. ONNX model outputting low-res mask)
    # masks[0] is (H_mask, W_mask). SAM2/ONNX might return 256x256 logits.
    if masks.shape[-1] != w or masks.shape[-2] != h:
        # Resize to original image size
        mask_logits = masks[0]
        # Ensure mask_logits is float32 for resize
        if mask_logits.dtype != np.float32:
            mask_logits = mask_logits.astype(np.float32)
            
        # [CRITICAL FIX]: Must resize float logits FIRST, then threshold!
        # If we threshold first and then resize with INTER_NEAREST, we get jagged/staircase edges.
        # By resizing the float logits with INTER_LINEAR first, we preserve smooth interpolation,
        # then thresholding gives smooth edges - matching the exact behavior of test_k11p_general_direct.py
        mask_resized = cv2.resize(mask_logits, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask_resized > 0.5).astype(np.uint8) * 255
    else:
        # Assuming logits, threshold at 0.5 to match test_k11p_general_direct.py exactly
        mask = (masks[0] > 0.5).astype(np.uint8) * 255

    # Clean mask (Match cropping sample logic exactly)
    # The user explicitly verified that the exact sequence of 5x5 morphological operations
    # WITHOUT Gaussian blur produces the only "correct" SAM output.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes (Modify the native mask in place!)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    
    # [BUGFIX] Must create a clean mask containing ONLY the largest contour!
    # test_k11p script was operating on an image with only 1 instance so it didn't matter,
    # but SAM sometimes predicts small disconnected noise blobs in the background. 
    # If we don't erase them, the bounding box of the entire mask gets artificially inflated!
    clean_mask = np.zeros_like(mask)
    cv2.fillPoly(clean_mask, [largest_contour], 255) 
    mask_filled = clean_mask
    
    # Apply Mask Scaling ONLY if explicitly requested (factor != 1.0)
    if mask_scaling_factor != 1.0:
        w_box = x2 - x1
        h_box = y2 - y1
        avg_dim = (w_box + h_box) / 2
        delta_pixels = int((avg_dim / 2) * (mask_scaling_factor - 1.0))
        
        if delta_pixels > 0:
            k_size = 2 * delta_pixels + 1
            kernel_scale = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask_filled = cv2.dilate(mask_filled, kernel_scale, iterations=1)
        elif delta_pixels < 0:
            delta_pixels = abs(delta_pixels)
            if delta_pixels > 0:
                k_size = 2 * delta_pixels + 1
                kernel_scale = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                mask_filled = cv2.erode(mask_filled, kernel_scale, iterations=1)

    return mask_filled


def rotate_image_and_mask(image, mask, angle_deg):
    (h, w) = image.shape[:2]
    
    # Calculate new dimensions to avoid cropping
    angle_rad = np.radians(angle_deg)
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    
    # Add margin to prevent rounding errors or slight clipping
    # Significantly increased margin to 500 to absolutely rule out canvas clipping
    margin = 500
    nW = int(np.ceil(h * sin_a + w * cos_a)) + margin
    nH = int(np.ceil(h * cos_a + w * sin_a)) + margin
    
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Adjust translation
    M[0, 2] += (nW / 2.0) - center[0]
    M[1, 2] += (nH / 2.0) - center[1]

    # Rotate image with black background
    img_rot = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    # Rotate mask with black background
    mask_rot = cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)

    return img_rot, mask_rot


def get_color_name(rgb_tuple):
    """
    Returns a human-readable name for an RGB color.
    Simple approximation.
    """
    r, g, b = rgb_tuple
    if r < 30 and g < 30 and b < 30: return "Black"
    if r > 200 and g > 200 and b > 200: return "White"
    if r > 200 and g < 50 and b < 50: return "Red"
    if r < 50 and g > 200 and b < 50: return "Green"
    if r < 50 and g < 50 and b > 200: return "Blue"
    if r > 200 and g > 200 and b < 50: return "Yellow"
    if r < 50 and g > 200 and b > 200: return "Cyan"
    if r > 200 and g < 50 and b > 200: return "Magenta"
    return f"RGB({int(r)},{int(g)},{int(b)})"


def apply_rgb_filtering(original_img, mask_filtered, rgb_config_row, output_params, save_dir, image_path, contour_image=None):
    """
    Applies RGB Filtering to the mask.
    Refines mask_filtered by keeping only pixels that match the target RGB within tolerance.
    """
    import cv2
    import numpy as np
    import os
    import ast
    import pandas as pd
    
    rgb_result_paths = []
    
    try:
        print(f"DEBUG: Applying RGB Filtering...")
        
        # Parse Config Row
        target_rgb_str = rgb_config_row.get('rgb')
        tolerance = float(rgb_config_row.get('tolerance')) if pd.notna(rgb_config_row.get('tolerance')) else 0.0
        group_id = rgb_config_row.get('group_id')
        proximity = str(rgb_config_row.get('proximity')).strip().lower() if pd.notna(rgb_config_row.get('proximity')) else 'near'
        
        # Parse RGB
        target_rgb = None
        if isinstance(target_rgb_str, str):
            try:
                clean_str = target_rgb_str.strip().replace("'", "").replace('"', '')
                if clean_str.startswith('[') and clean_str.endswith(']'):
                    target_rgb = ast.literal_eval(clean_str)
                else:
                    target_rgb = [float(x) for x in clean_str.split(',')]
            except Exception:
                pass
        elif isinstance(target_rgb_str, (list, tuple)):
            target_rgb = target_rgb_str
        
        if target_rgb is None:
            print(f"Warning: Invalid RGB format {target_rgb_str}. Skipping RGB filter.")
            return mask_filtered, rgb_result_paths

        target_rgb = np.array(target_rgb)
        color_name = get_color_name(target_rgb)
        print(f"DEBUG: Target RGB: {target_rgb} ({color_name}), Tol: {tolerance}, Prox: {proximity}")

        # Calculate Distance
        # original_img is BGR (from cv2.imread), so we must flip target_rgb to BGR
        target_bgr = target_rgb[::-1]
        
        diff = original_img.astype(np.float32) - target_bgr.astype(np.float32)
        dist = np.linalg.norm(diff, axis=2)
        
        # Create Valid Mask based on proximity
        if proximity == 'far':
            valid_pixels = (dist > tolerance)
        else:
            valid_pixels = (dist <= tolerance)
            
        # Refine mask_filtered
        # Keep only pixels that were already in mask_filtered AND are valid
        mask_filtered_new = mask_filtered & valid_pixels
        
        # Check if anything changed
        # changed = np.any(mask_filtered != mask_filtered_new)
        # print(f"DEBUG: RGB Filtering changed mask: {changed}")
        
        mask_filtered = mask_filtered_new
        
        # Save Outputs if requested
        if output_params and save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
            fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
            suffix = f"_rgb_filtered"
            if group_id: suffix += f"_{group_id}"
            
            mask_u8 = mask_filtered.astype(np.uint8) * 255
            
            if 'pic' in output_params:
                masked_img_rgb = cv2.bitwise_and(original_img, original_img, mask=mask_u8)
                save_name = f"{fname_clean}{suffix}_pic.png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(masked_img_rgb, cv2.COLOR_RGB2BGR))
                print(f"Saved RGB filtered pic: {save_name}")
                rgb_result_paths.append((save_path, "RGB Filtered Pic"))

            if 'mask' in output_params or 'save' in [str(p).lower() for p in output_params]:
                save_name = f"{fname_clean}{suffix}_mask.png"
                cv2.imwrite(os.path.join(save_dir, save_name), mask_u8)
                print(f"Saved RGB filtered mask: {save_name}")

            if 'contour' in output_params:
                if contour_image is not None:
                    save_name = f"{fname_clean}{suffix}_contour.png"
                    cv2.imwrite(os.path.join(save_dir, save_name), contour_image)
                    print(f"Saved RGB filtered contour: {save_name}")

    except Exception as e:
        print(f"Error in apply_rgb_filtering: {e}")
        import traceback
        traceback.print_exc()

    return mask_filtered, rgb_result_paths


def apply_hsv_filtering(original_img, mask_filtered, hsv_config_row, output_params, save_dir, image_path, contour_image=None):
    """
    Applies HSV Filtering to the mask using ranges for H, S, V.
    hsv_config_row:
        {
            'ranges': [(H_min, H_max), (S_min, S_max), (V_min, V_max)],
            'color_name': Optional string used for naming/output grouping,
            'size': float (resize factor),
            'uniform_light': bool,
            'ul_kernel_size': int,
            'gamma': float,
            'contrast': float
        }
    """
    import cv2
    import numpy as np
    import os
    import ast
    import pandas as pd
    
    hsv_result_paths = []
    
    try:
        print("DEBUG: Applying HSV Filtering...")
        
        ranges = hsv_config_row.get('ranges')
        color_name = hsv_config_row.get('color_name')
        
        # Extended params
        size_scale = float(hsv_config_row.get('size', 1.0))
        uniform_light = hsv_config_row.get('uniform_light', False)
        ul_kernel_size = int(hsv_config_row.get('ul_kernel_size', 101))
        gamma = float(hsv_config_row.get('gamma', 1.0))
        contrast = float(hsv_config_row.get('contrast', 1.0))
        denoise = int(hsv_config_row.get('denoise', 0))
        roi_shrink = float(hsv_config_row.get('roi_shrink', 1.0))
        invert = hsv_config_row.get('invert', False)
        
        # Parse ranges if provided as string
        if isinstance(ranges, str):
            s = ranges.strip()
            try:
                ranges = ast.literal_eval(s)
            except Exception:
                # Try simple split format like "64-255,0-255,0-255"
                # Handle tuple string format manually if ast fails (e.g. "(64,255),(0,255),(0,255)")
                try:
                    # Remove outer brackets if present
                    s_clean = s.replace('[', '').replace(']', '')
                    # Split by closing parenthesis to get pairs
                    parts = s_clean.split('),')
                    parsed = []
                    for p in parts:
                        p = p.replace('(', '').replace(')', '').strip()
                        if ',' in p:
                            vals = p.split(',')
                            if len(vals) >= 2:
                                parsed.append((int(vals[0]), int(vals[1])))
                        elif '-' in p:
                             a, b = p.split('-', 1)
                             parsed.append((int(a.strip()), int(b.strip())))
                    
                    if len(parsed) == 3:
                        ranges = parsed
                except Exception:
                    pass
        
        if not isinstance(ranges, (list, tuple)):
            print(f"Warning: Invalid HSV ranges format (not list/tuple): {ranges}. Skipping HSV filter.")
            return mask_filtered, hsv_result_paths
        
        # Determine if multiple sets or single set
        is_multiple = False
        if len(ranges) > 0 and isinstance(ranges[0], (list, tuple)) and len(ranges[0]) == 3:
            # We have a list of sets of ranges e.g. [((H,H),(S,S),(V,V)), ((H,H),(S,S),(V,V))]
            is_multiple = True
            range_sets = ranges
        elif len(ranges) == 3:
            range_sets = [ranges]
        else:
            print(f"Warning: Invalid HSV ranges format (length check failed): {ranges}. Skipping HSV filter.")
            return mask_filtered, hsv_result_paths

        # Clamp ranges to valid HSV bounds used in project (0-255 for each slider per GUI)
        def clamp_pair(pair, low=0, high=255):
            try:
                a, b = int(pair[0]), int(pair[1])
            except Exception:
                a, b = 0, 255
            a = max(low, min(high, a))
            b = max(low, min(high, b))
            return a, b

        # --- Pre-processing Pipeline ---
        
        # 1. Resize
        img_to_process = original_img.copy()
        if size_scale != 1.0:
            h, w = img_to_process.shape[:2]
            new_w = int(w * size_scale)
            new_h = int(h * size_scale)
            img_to_process = cv2.resize(img_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"DEBUG: Resized image for HSV by {size_scale} (New size: {new_w}x{new_h})")
            
            # Also resize mask_filtered to match for later intersection
            if mask_filtered is not None:
                mask_u8_in = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
                mask_filtered_resized = cv2.resize(mask_u8_in, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_filtered_resized = None
        else:
            mask_filtered_resized = mask_filtered
        
        # 2. Gamma Correction
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img_to_process = cv2.LUT(img_to_process, table)
            print(f"DEBUG: Applied Gamma Correction: {gamma}")

        # 3. Contrast Correction
        if contrast != 1.0:
            img_to_process = cv2.convertScaleAbs(img_to_process, alpha=contrast, beta=0)
            print(f"DEBUG: Applied Contrast Correction: {contrast}")

        # 3.5 Denoise (Gaussian Blur)
        if denoise > 0:
            ksize = denoise * 2 + 1
            img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            print(f"DEBUG: Applied Denoise (Gaussian Blur): kernel size {ksize}")

        # Convert to HSV
        # Convert to HSV
        # If the input image is already BGR (like from cv2.imread), we can directly convert to HSV.
        # But if it's RGB, we should convert RGB->BGR->HSV.
        # By default, cv2.imread returns BGR.
        img_hsv = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2HSV)
        
        # 4. Uniform Light (Illumination Correction)
        if uniform_light:
            # Correct V and S channels
            h_ch, s_ch, v_ch = cv2.split(img_hsv)
            
            k_illum = ul_kernel_size
            if k_illum % 2 == 0: k_illum += 1
            
            # Correct V
            bg_v = cv2.GaussianBlur(v_ch, (k_illum, k_illum), 0)
            diff_v = cv2.subtract(v_ch, bg_v)
            v_corr = cv2.add(diff_v, 127)
            
            # Correct S
            bg_s = cv2.GaussianBlur(s_ch, (k_illum, k_illum), 0)
            diff_s = cv2.subtract(s_ch, bg_s)
            s_corr = cv2.add(diff_s, 127)
            
            img_hsv = cv2.merge([h_ch, s_corr, v_corr])
            print(f"DEBUG: Applied Uniform Light Correction (Kernel: {k_illum})")

        hsv_mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        
        for r_set in range_sets:
            h_min, h_max = clamp_pair(r_set[0])
            s_min, s_max = clamp_pair(r_set[1])
            v_min, v_max = clamp_pair(r_set[2])
            
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            
            current_mask = cv2.inRange(img_hsv, lower, upper)  # uint8 [0,255]
            hsv_mask = cv2.bitwise_or(hsv_mask, current_mask)
        
        if invert:
            hsv_mask = cv2.bitwise_not(hsv_mask)
            print("DEBUG: Applied Invert logic to HSV mask")
        
        # Refine mask_filtered by AND with hsv_mask
        # DEBUG: Save intermediate masks for verification


        # Count pixels before (using resized mask if applicable)
        pixels_before = np.count_nonzero(mask_filtered_resized) if mask_filtered_resized is not None else 0
        pixels_hsv = np.count_nonzero(hsv_mask)

        if mask_filtered_resized is not None:
            if mask_filtered_resized.dtype == bool:
                mask_filtered_new = mask_filtered_resized & (hsv_mask > 0)
            else:
                mask_filtered_new = cv2.bitwise_and(mask_filtered_resized, hsv_mask)
        else:
            mask_filtered_new = hsv_mask  # If no input mask, use HSV result directly
        
        # Count pixels after
        pixels_after = np.count_nonzero(mask_filtered_new)
        print(f"DEBUG: HSV Intersection Result: Before={pixels_before} px, HSV_Mask={pixels_hsv} px, After={pixels_after} px")
        
        # Resize back to original size if scaled
        if size_scale != 1.0:
            h_orig, w_orig = original_img.shape[:2]
            mask_u8_res = (mask_filtered_new.astype(np.uint8) * 255) if mask_filtered_new.dtype == bool else mask_filtered_new
            mask_filtered = cv2.resize(mask_u8_res, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            mask_filtered = mask_filtered_new

        # 5. ROI Shrink (Remove edge artifacts from Uniform Light)
        if roi_shrink < 1.0 and roi_shrink > 0:
            # Find the bounding box of all white pixels
            white_pixels = np.where(mask_filtered > 0)
            if len(white_pixels[0]) > 0:
                y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
                x_min, x_max = white_pixels[1].min(), white_pixels[1].max()
                
                # Calculate center and new dimensions
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min
                
                new_width = int(width * roi_shrink)
                new_height = int(height * roi_shrink)
                
                new_x_min = max(0, center_x - new_width // 2)
                new_x_max = min(mask_filtered.shape[1], center_x + new_width // 2)
                new_y_min = max(0, center_y - new_height // 2)
                new_y_max = min(mask_filtered.shape[0], center_y + new_height // 2)
                
                # Create mask for inner ROI
                roi_mask = np.zeros_like(mask_filtered)
                roi_mask[new_y_min:new_y_max, new_x_min:new_x_max] = 255
                
                # Apply mask to filter image
                mask_filtered = cv2.bitwise_and(mask_filtered, roi_mask)
                print(f"DEBUG: Applied ROI Shrink: ratio={roi_shrink}, new bbox=({new_x_min},{new_y_min})-({new_x_max},{new_y_max})")
        
        # Save outputs if requested
        if output_params and save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
            fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
            suffix = "_hsv_filtered"
            if color_name:
                suffix += f"_{str(color_name).replace(' ', '_')}"
            
            mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            
            if 'pic' in output_params:
                masked_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8)
                save_name = f"{fname_clean}{suffix}_pic.png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                print(f"Saved HSV filtered pic: {save_name}")
                hsv_result_paths.append((save_path, "HSV Filtered Pic"))
            
            if 'mask' in output_params or 'save' in [str(p).lower() for p in output_params]:
                save_name = f"{fname_clean}{suffix}_mask.png"
                cv2.imwrite(os.path.join(save_dir, save_name), mask_u8)
                print(f"Saved HSV filtered mask: {save_name}")
            
            if 'contour' in output_params and contour_image is not None:
                save_name = f"{fname_clean}{suffix}_contour.png"
                cv2.imwrite(os.path.join(save_dir, save_name), contour_image)
                print(f"Saved HSV filtered contour: {save_name}")
        
    except Exception as e:
        print(f"Error in apply_hsv_filtering: {e}")
        import traceback
        traceback.print_exc()
    
    return mask_filtered, hsv_result_paths


def _save_images_for_operations(ops, original_img, mask_filtered, contour_image, save_dir, image_path, step_suffix=""):
    if not ops or save_dir is None:
        return
    # Use original filename (without extension) to preserve full name for overlay naming
    # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
    fname = os.path.splitext(os.path.basename(image_path))[0]
    if mask_filtered is None or original_img is None:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if mask_filtered.dtype == bool:
        mask_u8 = (mask_filtered.astype(np.uint8) * 255)
    else:
        mask_u8 = mask_filtered
        try:
            max_val = float(np.max(mask_u8))
            if max_val <= 1.0:
                mask_u8 = (mask_u8.astype(np.float32) * 255.0).astype(np.uint8)
        except Exception:
            pass
    if mask_u8.ndim == 3:
        mask_u8 = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    h, w = original_img.shape[:2]
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_u8 > 0
    def next_step_index():
        try:
            existing = [f for f in os.listdir(save_dir) if f.startswith(fname + "_save_") and f.endswith(".png")]
            import re
            nums = []
            for f in existing:
                m = re.search(r"_step(\d+)\.png$", f)
                if m:
                    try:
                        nums.append(int(m.group(1)))
                    except:
                        pass
            if nums:
                return max(nums) + 1
            return 1
        except Exception:
            return 1
    current_step = next_step_index()
    for kind, with_contour in ops:
        if kind == 'legacy':
            continue
        step_str = f"_step{current_step}"
        if kind == 'mask':
            img = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
            name = f"{fname}_save_mask{step_str}.jpg"
        elif kind == 'overlay':
            img = cv2.bitwise_and(original_img, original_img, mask=mask_u8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            name = f"{fname}_save_overlay{step_str}.jpg"
        elif kind == 'complement_mask':
            inv = cv2.bitwise_not(mask_u8)
            img = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            name = f"{fname}_save_complement_mask{step_str}.jpg"
        elif kind == 'complement_overlay':
            inv = cv2.bitwise_not(mask_u8)
            img = cv2.bitwise_and(original_img, original_img, mask=inv)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            name = f"{fname}_save_complement_overlay{step_str}.jpg"
        else:
            continue
        if with_contour and contour_image is not None:
            print(f"DEBUG: Drawing contour on {name}, contour_image shape={contour_image.shape}")
            cimg = contour_image
            if cimg.shape[:2] != (h, w):
                cimg = cv2.resize(cimg, (w, h), interpolation=cv2.INTER_NEAREST)
            if cimg.ndim == 3:
                cgray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
            else:
                cgray = cimg
            cnts, _ = cv2.findContours((cgray > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"DEBUG: Found {len(cnts)} contours to draw")
            cv2.drawContours(img, cnts, -1, (0,255,0), 2)
        save_path = os.path.join(save_dir, name)
        result = cv2.imwrite(save_path, img)
        print(f"DEBUG: Saved {save_path}, success={result}")
        current_step += 1



def _add_legend_to_image(img, mask_info):
    import cv2
    import numpy as np
    
    # Create legend
    h, w = img.shape[:2]
    legend_height = 40 * len(mask_info) + 20
    legend_img = np.zeros((legend_height, w, 3), dtype=np.uint8)
    
    for i, info in enumerate(mask_info):
        color = info['color']
        text = info['config_name']
        y_pos = 30 + i * 40
        
        # Draw color box
        cv2.rectangle(legend_img, (20, y_pos - 15), (50, y_pos + 5), color, -1)
        # Draw text
        cv2.putText(legend_img, text, (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    # Stack vertically
    return np.vstack([img, legend_img])
