from pathlib import Path
import utils.scoring_utils as scoring_utils

import cv2
import numpy as np
import pandas as pd
import os
from utils.config.config_manager import ConfigManager
from utils.cv_ops import _save_images_for_operations, apply_hsv_filtering, apply_rgb_filtering


def parse_filtering_operations(params):
    """
    Parses a list of parameters (from Filtering[...]) into structured operations.
    Example: ['Area(5)', 'Pic', 'Save', 'RGB'] 
    -> [{'method': 'Area', 'key': '5', 'output_params': ['Pic', 'Save']}, {'method': 'RGB', 'key': '', 'output_params': []}]
    
    Handles complex cases like:
    ['Morph Top-Hat(Bright)', 'Save(Overlay+Contour, Complement_Overlay)']
    """
    operations = []
    current_op = None
    
    # Methods that start a new operation group
    known_methods = [
        'area', 'areafiltering',
        'rgb', 'rgbfiltering',
        'colordistance', 'color_distance', 'colordistancefiltering',  # Add Color Distance support
        'relativity', 'relativityfiltering',
        'adaptivegaussian', 'adaptive_gaussian', 'ag',
        'hsv', 'hsvfiltering',
        'shape', 'shapefiltering',
        'morphtophat', 'morphblackhat', 'morph_tophat', 'morph_blackhat',
        'globalthreshold', 'global_threshold', 'gt',
        'slicing', 'slice', 'axis', 'axis_slice',
        'bandroi', 'band_roI', 'band-roi',  # Add Band-ROI support
        'stencilroi', 'stencil_roI', 'stencil-roi',  # Add Stencil-ROI support
        'bgsubtraction', 'backgroundsubtraction', 'bg_subtraction', 'bg_sub',  # Add BG Subtraction support
        'odbp', 'weighteddiff', 'odbpweighteddiff'  # Add ODBP support
    ]
    
    # Helper to clean string
    def clean(s): return str(s).strip()
    def norm(s): return clean(s).lower().replace('_', '').replace(' ', '').replace('-', '')

    for p in params:
        p_str = clean(p)
        print(f"DEBUG: parse_filtering_params processing param: '{p_str}'")
        if not p_str:
            print(f"DEBUG: Empty param, skipping")
            continue
        
        # Check if it looks like Method(Key)
        method_candidate = p_str
        key_candidate = ""
        
        if '(' in p_str and p_str.endswith(')'):
            idx = p_str.find('(')
            method_candidate = p_str[:idx].strip()
            key_candidate = p_str[idx+1:-1].strip()
        
        # Special handling for Morph Top-Hat/Black-Hat
        norm_method = norm(method_candidate)
        print(f"DEBUG: method_candidate='{method_candidate}', norm_method='{norm_method}', key='{key_candidate}'")
        
        is_known_method = False
        if norm_method in known_methods:
            is_known_method = True
            print(f"DEBUG: Found known method: {norm_method}")
        elif 'morphtophat' in norm_method or 'morphblackhat' in norm_method:
             is_known_method = True
             # Normalize method name
             if 'morphtophat' in norm_method:
                 method_candidate = 'Morph Top-Hat'
             elif 'morphblackhat' in norm_method:
                 method_candidate = 'Morph Black-Hat'
        elif 'globalthreshold' in norm_method:
             is_known_method = True
             method_candidate = 'Global Threshold'
        elif 'odbp' in norm_method or 'weighteddiff' in norm_method:
             is_known_method = True
             method_candidate = 'ODBP'
        
        # Check if it's a Save operation (which is usually a parameter, NOT a new method start)
        if norm_method.startswith('save'):
            is_known_method = False
            print(f"DEBUG: Save operation detected, treating as parameter")

        print(f"DEBUG: is_known_method={is_known_method}, current_op={current_op}")
        
        if is_known_method:
            # Start new op
            current_op = {
                'method': method_candidate, 
                'key': key_candidate, 
                'output_params': []
            }
            operations.append(current_op)
        else:
            # It's a parameter (e.g. Pic, Save)
            # Add to current op if exists
            if current_op:
                current_op['output_params'].append(p_str)
            else:
                # Orphan param - might be a method without explicit call syntax
                # Try to interpret as method
                if norm_method in known_methods:
                    current_op = {
                        'method': method_candidate, 
                        'key': '', 
                        'output_params': []
                    }
                    operations.append(current_op)
                else:
                    print(f"DEBUG: Orphan parameter '{p_str}' ignored (no preceding method)")
    
    print(f"DEBUG: parse_filtering_operations parsed {len(operations)} operations from {params}")
    for i, op in enumerate(operations):
        print(f"  Op {i}: method='{op['method']}', key='{op['key']}', output_params={op['output_params']}")
                
    return operations


def apply_area_filtering(mask_filtered, contour_image, area_filtering_percentage, output_params, original_img, save_dir, image_path, visualization_overlay=None, enable_save_overlay=False, defect_id_method=None, step_suffix="", area_filtering_pixel=None):
    """
    Applies Area Filtering to remove defect blobs smaller than a percentage of the object area OR an absolute pixel count.
    area_filtering_pixel: Absolute pixel count threshold. If provided, overrides area_filtering_percentage.
    """
    import cv2
    import numpy as np
    import os
    
    area_filter_mask_to_remove = None
    rgb_result_paths = []
    
    # Logic extracted from dinov3_utils.py
    # Condition: Either Pixel or Percentage must be provided. Contour image needed for Percentage only.
    should_run = False
    if area_filtering_pixel is not None:
        should_run = True
    elif area_filtering_percentage is not None and contour_image is not None:
        should_run = True
        
    if should_run:
        try:
            min_defect_area = 0.0
            
            # Case 1: Absolute Pixel Threshold
            if area_filtering_pixel is not None:
                min_defect_area = float(area_filtering_pixel)
                print(f"DEBUG: Applying Area Filtering {step_suffix} (Absolute Pixel). Threshold: {min_defect_area} pixels")
                
            # Case 2: Percentage Threshold
            elif area_filtering_percentage is not None and contour_image is not None:
                print(f"DEBUG: Applying Area Filtering {step_suffix} (Percentage). Percentage: {area_filtering_percentage}%")
                
                # 1. Calculate Reference Area from Contour_Corrected
                if contour_image.shape[:2] != mask_filtered.shape:
                    contour_for_area = cv2.resize(contour_image, (mask_filtered.shape[1], mask_filtered.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    contour_for_area = contour_image

                # Area of object (non-zero pixels)
                object_area = cv2.countNonZero(contour_for_area)
                print(f"DEBUG: Object Area (Contour_Corrected): {object_area}")

                if object_area > 0:
                    # 2. Calculate Threshold Area
                    min_defect_area = (area_filtering_percentage / 100.0) * object_area
                    print(f"DEBUG: Calculated Min Defect Area Threshold: {min_defect_area:.2f} pixels")
                else:
                    print("DEBUG: Object Area is 0, skipping Area Filtering.")
                    min_defect_area = 0.0

            if min_defect_area > 0:
                # 3. Filter Connected Components in mask_filtered
                mask_u8 = mask_filtered.astype(np.uint8) * 255
                
                # Find contours of defects
                contours_defects, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create a mask of defects to REMOVE
                remove_mask = np.zeros_like(mask_u8)
                removed_count = 0
                
                for c in contours_defects:
                    c_area = cv2.contourArea(c)
                    if c_area < min_defect_area: # Filter out small noise
                         # print(f"DEBUG: Removing blob with area {c_area:.2f} < {min_defect_area:.2f}")
                         cv2.drawContours(remove_mask, [c], -1, 255, -1) # Mark for removal
                         removed_count += 1
                
                if removed_count > 0:
                    print(f"DEBUG: Removed {removed_count} defect blobs smaller than {min_defect_area:.2f}")
                    # Update mask_filtered
                    keep_mask = (remove_mask == 0)
                    mask_filtered = mask_filtered & keep_mask
                    
                    # Store remove mask
                    area_filter_mask_to_remove = remove_mask
                else:
                    print("DEBUG: No defect blobs removed by Area Filtering.")

            if output_params and save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                save_dir_area = save_dir

                # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')

                mask_u8_final = mask_filtered.astype(np.uint8) * 255
                if mask_u8_final.shape[:2] != original_img.shape[:2]:
                    mask_u8_final = cv2.resize(mask_u8_final, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                if 'pic' in output_params:
                    if visualization_overlay is not None:
                        masked_img_area = original_img.copy()
                        overlay_use = visualization_overlay
                        if overlay_use.shape[:2] != original_img.shape[:2]:
                            overlay_use = cv2.resize(overlay_use, (original_img.shape[1], original_img.shape[0]))
                        mask_bool = mask_u8_final > 0
                        masked_img_area[mask_bool] = overlay_use[mask_bool]
                        save_img = cv2.cvtColor(masked_img_area, cv2.COLOR_RGB2BGR)
                    else:
                        masked_img_area = np.zeros_like(original_img)
                        masked_img_area = cv2.bitwise_and(original_img, original_img, mask=mask_u8_final)
                        save_img = cv2.cvtColor(masked_img_area, cv2.COLOR_RGB2BGR)

                    save_name = f"{fname_clean}_area_filtered_pic{step_suffix}.png"
                    cv2.imwrite(os.path.join(save_dir_area, save_name), save_img)
                    print(f"Saved Area Filtering output: {save_name}")
                    rgb_result_paths.append((os.path.join(save_dir_area, save_name), "Area Filtered Pic"))

                if 'mask' in [str(p).lower() for p in output_params]:
                    save_name = f"{fname_clean}_area_filtered_mask{step_suffix}.png"
                    cv2.imwrite(os.path.join(save_dir_area, save_name), mask_u8_final)
                    print(f"Saved Area Filtering output: {save_name}")

                if 'save' in [str(p).lower() for p in output_params]:
                    use_overlay_for_save = False
                    if visualization_overlay is not None and enable_save_overlay:
                        if defect_id_method and 'dino' in str(defect_id_method).lower():
                            use_overlay_for_save = True

                    if use_overlay_for_save:
                        masked_img_vis = original_img.copy()
                        overlay_use = visualization_overlay
                        if overlay_use.shape[:2] != original_img.shape[:2]:
                            overlay_use = cv2.resize(overlay_use, (original_img.shape[1], original_img.shape[0]))

                        mask_bool = mask_u8_final > 0
                        masked_img_vis[mask_bool] = overlay_use[mask_bool]
                        save_img_vis = cv2.cvtColor(masked_img_vis, cv2.COLOR_RGB2BGR)

                        save_name = f"{fname_clean}_area_filtered_visualization.png"
                        cv2.imwrite(os.path.join(save_dir_area, save_name), save_img_vis)
                        print(f"Saved Area Filtering output (Visualization): {save_name}")

                if 'contour' in output_params:
                    contour_vis = np.zeros_like(original_img)
                    contours_final, _ = cv2.findContours(mask_u8_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_vis, contours_final, -1, (0, 255, 0), 1)
                    save_name = f"{fname_clean}_area_filtered_contour.png"
                    cv2.imwrite(os.path.join(save_dir_area, save_name), contour_vis)
                    print(f"Saved Area Filtering output: {save_name}")

        except Exception as e:
            print(f"Error applying Area Filtering: {e}")
            traceback.print_exc()
    elif area_filtering_percentage is not None and contour_image is None:
        print("Warning: Area Filtering requested but Contour_Corrected is missing. Skipping.")

    return mask_filtered, area_filter_mask_to_remove, rgb_result_paths


def apply_global_threshold_filtering(original_img, mask_filtered, gt_config, out_params, save_dir, image_path, contour_image=None):
    """
    Applies Global Threshold filtering to the mask.
    Similar to CV_GUI.py Global Threshold mode.
    
    gt_config:
        {
            'size': float (resize factor, e.g., 0.25 for 1/4),
            'threshold': int (0-255),
            'uniform_light': bool,
            'ul_kernel_size': int,
            'gamma': float,
            'contrast': float,
            'denoise': int (optional),
            'morph_open': int (optional),
            'morph_close': int (optional),
            'min_area': float (optional),
            'roi_shrink': float (optional)
        }
    """
    import cv2
    import numpy as np
    import os
    
    gt_result_paths = []
    
    try:
        print("DEBUG: Applying Global Threshold Filtering...")
        
        # Extract config parameters
        size_scale = float(gt_config.get('size', 1.0))
        threshold = int(gt_config.get('threshold', 127))
        uniform_light = gt_config.get('uniform_light', False)
        ul_kernel_size = int(gt_config.get('ul_kernel_size', 101))
        gamma = float(gt_config.get('gamma', 1.0))
        contrast = float(gt_config.get('contrast', 1.0))
        denoise = int(gt_config.get('denoise', 0))
        morph_open = int(gt_config.get('morph_open', 0))
        morph_close = int(gt_config.get('morph_close', 0))
        min_area = float(gt_config.get('min_area', 0))
        roi_shrink = float(gt_config.get('roi_shrink', 1.0))
        
        print(f"DEBUG: Global Threshold Params - Size:{size_scale}, Threshold:{threshold}, Uniform Light:{uniform_light}, UL Kernel:{ul_kernel_size}, Gamma:{gamma}, Contrast:{contrast}")
        
        # --- Pre-processing Pipeline ---
        
        # 1. Resize
        img_to_process = original_img.copy()
        if size_scale != 1.0:
            h, w = img_to_process.shape[:2]
            new_w = int(w * size_scale)
            new_h = int(h * size_scale)
            img_to_process = cv2.resize(img_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"DEBUG: Resized image for Global Threshold by {size_scale} (New size: {new_w}x{new_h})")
            
            # Also resize mask_filtered to match for later intersection
            if mask_filtered is not None:
                mask_u8_in = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
                mask_filtered_resized = cv2.resize(mask_u8_in, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_filtered_resized = None
        else:
            mask_filtered_resized = mask_filtered
        
        # Convert to Grayscale
        img_gray = cv2.cvtColor(img_to_process, cv2.COLOR_RGB2GRAY)
        
        # 2. Uniform Light (Illumination Correction)
        if uniform_light:
            k_illum = ul_kernel_size
            if k_illum % 2 == 0: k_illum += 1
            background = cv2.GaussianBlur(img_gray, (k_illum, k_illum), 0)
            diff = cv2.subtract(img_gray, background)
            img_gray = cv2.add(diff, 127)
            print(f"DEBUG: Applied Uniform Light Correction (Kernel: {k_illum})")
        
        # 3. Gamma Correction
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img_gray = cv2.LUT(img_gray, table)
            print(f"DEBUG: Applied Gamma Correction: {gamma}")
        
        # 4. Contrast Correction
        if contrast != 1.0:
            img_gray = cv2.convertScaleAbs(img_gray, alpha=contrast, beta=0)
            print(f"DEBUG: Applied Contrast Correction: {contrast}")
        
        # 5. Denoise (Gaussian Blur)
        if denoise > 0:
            ksize = denoise * 2 + 1
            img_gray = cv2.GaussianBlur(img_gray, (ksize, ksize), 0)
            print(f"DEBUG: Applied Denoise (Gaussian Blur): kernel size {ksize}")
        
        # 6. Global Threshold Binarization
        _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        print(f"DEBUG: Applied Global Threshold: {threshold}")
        
        # 7. Post-processing: Morphological Open (Remove Specks)
        if morph_open > 0:
            k_open = morph_open * 2 + 1
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            print(f"DEBUG: Applied Morph Open: kernel size {k_open}")
        
        # 8. Post-processing: Morphological Close (Fill Holes)
        if morph_close > 0:
            k_close = morph_close * 2 + 1
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            print(f"DEBUG: Applied Morph Close: kernel size {k_close}")
        
        # 9. Post-processing: Min Area Filter
        if min_area > 0:
            # Find all contours including holes (using RETR_TREE)
            cnts, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_clean = np.zeros_like(binary)
            if cnts and hierarchy is not None:
                for i, c in enumerate(cnts):
                    area = cv2.contourArea(c)
                    if area >= min_area:
                        # Check if this is a hole (child of another contour)
                        # hierarchy: [Next, Previous, First_Child, Parent]
                        # Parent = -1 means it's an external contour
                        # Parent >= 0 means it's a hole inside another contour
                        parent_idx = hierarchy[0][i][3]
                        if parent_idx >= 0:
                            # This is a hole (internal contour)
                            # Fill it (draw it as black to remove the hole)
                            cv2.drawContours(mask_clean, [c], -1, 0, -1)
                        else:
                            # This is an external contour, keep it
                            cv2.drawContours(mask_clean, [c], -1, 255, -1)
                    else:
                        # Area too small - remove this contour
                        # If it's a hole, fill it (make it solid)
                        # If it's external, remove it
                        parent_idx = hierarchy[0][i][3]
                        if parent_idx >= 0:
                            # Small hole - fill it (make it part of parent)
                            cv2.drawContours(mask_clean, [c], -1, 255, -1)
                        # else: small external contour - don't draw it (remains 0)
            binary = mask_clean
            print(f"DEBUG: Applied Min Area filter: {min_area}")
        
        # Count pixels before intersection
        pixels_before = np.count_nonzero(mask_filtered_resized) if mask_filtered_resized is not None else 0
        pixels_gt = np.count_nonzero(binary)
        
        # Intersect with input mask if provided
        if mask_filtered_resized is not None:
            if mask_filtered_resized.dtype == bool:
                mask_filtered_new = mask_filtered_resized & (binary > 0)
            else:
                mask_filtered_new = cv2.bitwise_and(mask_filtered_resized, binary)
        else:
            mask_filtered_new = binary
        
        pixels_after = np.count_nonzero(mask_filtered_new)
        print(f"DEBUG: Global Threshold Intersection Result: Before={pixels_before} px, GT_Mask={pixels_gt} px, After={pixels_after} px")
        
        # Resize back to original size if scaled
        if size_scale != 1.0:
            h_orig, w_orig = original_img.shape[:2]
            mask_u8_res = (mask_filtered_new.astype(np.uint8) * 255) if mask_filtered_new.dtype == bool else mask_filtered_new
            mask_filtered = cv2.resize(mask_u8_res, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            mask_filtered = mask_filtered_new
        
        # 10. ROI Shrink (Remove edge artifacts from Uniform Light)
        if roi_shrink < 1.0 and roi_shrink > 0:
            white_pixels = np.where(mask_filtered > 0)
            if len(white_pixels[0]) > 0:
                y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
                x_min, x_max = white_pixels[1].min(), white_pixels[1].max()
                
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
                
                roi_mask = np.zeros_like(mask_filtered)
                roi_mask[new_y_min:new_y_max, new_x_min:new_x_max] = 255
                
                mask_filtered = cv2.bitwise_and(mask_filtered, roi_mask)
                print(f"DEBUG: Applied ROI Shrink: ratio={roi_shrink}")
        
        # Save outputs if requested
        if out_params and save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
            suffix = "_gt_filtered"
            
            mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            
            if 'pic' in out_params:
                masked_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8)
                save_name = f"{fname_clean}{suffix}_pic.png"
                save_path = os.path.join(save_dir, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                print(f"Saved Global Threshold filtered pic: {save_name}")
                gt_result_paths.append((save_path, "Global Threshold Filtered Pic"))
            
            if 'mask' in out_params or 'save' in [str(p).lower() for p in out_params]:
                save_name = f"{fname_clean}{suffix}_mask.png"
                cv2.imwrite(os.path.join(save_dir, save_name), mask_u8)
                print(f"Saved Global Threshold filtered mask: {save_name}")
            
            if 'contour' in out_params and contour_image is not None:
                save_name = f"{fname_clean}{suffix}_contour.png"
                cv2.imwrite(os.path.join(save_dir, save_name), contour_image)
                print(f"Saved Global Threshold filtered contour: {save_name}")
        
    except Exception as e:
        print(f"Error in apply_global_threshold_filtering: {e}")
        import traceback
        traceback.print_exc()
    
    return mask_filtered, gt_result_paths


def apply_morph_tophat(original_img, mask_filtered, morph_config, out_params, save_dir, image_path, contour_image=None):
    """
    Applies Morphological Top-Hat (Bright) or Black-Hat (Dark) filtering.
    morph_config:
        {
            'mode': 'Top-Hat' or 'Black-Hat',
            'kernel_size': int,
            'threshold': int,
            'denoise': int,
            'open': int,
            'min_area': float
        }
    """
    import cv2
    import numpy as np
    import os
    import traceback

    morph_mask_to_keep = None
    
    if morph_config is not None:
        try:
            print("DEBUG: Applying Morphological Filtering Logic...")
            
            mode = morph_config.get('mode', 'Top-Hat')
            k_size = int(morph_config.get('kernel_size', 11))
            threshold = int(morph_config.get('threshold', 127))
            denoise = int(morph_config.get('denoise', 0))
            morph_open = int(morph_config.get('open', 0))
            morph_close = int(morph_config.get('close', 0))
            min_area = float(morph_config.get('min_area', 0))
            roi_shrink = float(morph_config.get('roi_shrink', 1.0))
            uniform_light = morph_config.get('uniform_light', False)
            ul_kernel_size = int(morph_config.get('ul_kernel_size', 101))
            
            # Ensure kernel size is odd
            if k_size < 3: k_size = 3
            if k_size % 2 == 0: k_size += 1
            
            print(f"DEBUG: Morph Params - Mode:{mode}, Kernel:{k_size}, Thresh:{threshold}, Denoise:{denoise}, Open:{morph_open}, Close:{morph_close}, MinArea:{min_area}, ROI Shrink:{roi_shrink}, UL:{uniform_light}, UL_Kernel:{ul_kernel_size}")
            
            # --- Process Each Component Separately ---
            print(f"DEBUG: Processing Morphological Filter per connected component...")
            
            # Initialize Combined Mask
            combined_mask = np.zeros_like(mask_filtered, dtype=bool)
            
            # Find Contours of the Input Mask
            input_mask_u8 = (mask_filtered.astype(np.uint8) * 255)
            contours_input, _ = cv2.findContours(input_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_input:
                for cnt in contours_input:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Apply ROI Shrink (if < 1.0)
                    if roi_shrink < 1.0 and roi_shrink > 0:
                        center_x = x + w / 2
                        center_y = y + h / 2
                        new_w = int(w * roi_shrink)
                        new_h = int(h * roi_shrink)
                        new_x = int(center_x - new_w / 2)
                        new_y = int(center_y - new_h / 2)
                        
                        # Update ROI coordinates (ensure within image bounds)
                        x = max(0, new_x)
                        y = max(0, new_y)
                        w = min(original_img.shape[1] - x, new_w)
                        h = min(original_img.shape[0] - y, new_h)
                        print(f"DEBUG: Applied ROI Shrink (ratio={roi_shrink}). ROI: {w}x{h}")
                    
                    # Add padding equal to half kernel size
                    pad = max(1, k_size // 2)
                    
                    # Calculate padded coordinates
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(original_img.shape[1], x + w + pad)
                    y2 = min(original_img.shape[0], y + h + pad)
                    
                    # Crop Image
                    img_crop = original_img[y1:y2, x1:x2]
                    
                    # Convert to Gray
                    img_gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
                    
                    # Uniform Light (Illumination Correction)
                    if uniform_light:
                        if ul_kernel_size % 2 == 0:
                            ul_kernel_size += 1
                        background_illum = cv2.GaussianBlur(img_gray_crop, (ul_kernel_size, ul_kernel_size), 0)
                        diff_illum = cv2.subtract(img_gray_crop, background_illum)
                        img_gray_crop = cv2.add(diff_illum, 127)
                    
                    # Denoise
                    if denoise > 0:
                        k_denoise = denoise * 2 + 1
                        img_gray_crop = cv2.GaussianBlur(img_gray_crop, (k_denoise, k_denoise), 0)
                    
                    # Morphological Operation
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    
                    if mode == 'Top-Hat':
                        # Top-Hat: Image - Opening(Image) -> Bright regions
                        morph_res = cv2.morphologyEx(img_gray_crop, cv2.MORPH_TOPHAT, kernel)
                    else:
                        # Black-Hat: Closing(Image) - Image -> Dark regions
                        morph_res = cv2.morphologyEx(img_gray_crop, cv2.MORPH_BLACKHAT, kernel)
                    
                    # Threshold
                    _, binary_crop = cv2.threshold(morph_res, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Post-processing: Open (Remove Specks)
                    if morph_open > 0:
                        k_open = morph_open * 2 + 1
                        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
                        binary_crop = cv2.morphologyEx(binary_crop, cv2.MORPH_OPEN, kernel_open)
                    
                    # Post-processing: Close (Fill Holes)
                    if morph_close > 0:
                        k_close = morph_close * 2 + 1
                        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
                        binary_crop = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, kernel_close)
                    
                    # Post-processing: Min Area
                    if min_area > 0:
                        cnts_crop, _ = cv2.findContours(binary_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        mask_crop_clean = np.zeros_like(binary_crop)
                        for c in cnts_crop:
                            if cv2.contourArea(c) >= min_area:
                                cv2.drawContours(mask_crop_clean, [c], -1, 255, -1)
                        binary_crop = mask_crop_clean

                    # Create Local Mask for Intersection (ROI)
                    roi_mask_local = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                    cv2.drawContours(roi_mask_local, [cnt], -1, 255, -1, offset=(-x1, -y1))
                    
                    # Intersection
                    binary_crop_bool = (binary_crop > 127)
                    local_result = binary_crop_bool & (roi_mask_local > 0)
                    
                    # Place back into combined_mask
                    combined_mask[y1:y2, x1:x2] |= local_result
            else:
                 print("DEBUG: No input ROIs found. Result empty.")
            
            # [BUGFIX] Only strictly intersect using global_constraint if ROI shrink was actually applied, AND
            # ensure the type matches exactly how it was initialized to avoid dtype poisoning.
            # Using bitwise_and on uint8 matrices is much safer and matches OpenCV's internal representation.
            
            # 1. Start with the raw morph combined mask
            morph_mask_u8 = (combined_mask.astype(np.uint8) * 255)

            # 2. Strict intersection with original contour (to completely slice off spilled borders)
            # The input_mask_u8 represents the PERFECT contour of the screen extracted by SAM.
            strictly_constrained_mask_u8 = cv2.bitwise_and(morph_mask_u8, input_mask_u8)

            # 3. Carefully update downstream tracking variables
            if mask_filtered.dtype == bool:
                mask_filtered = (strictly_constrained_mask_u8 > 127)
            else:
                mask_filtered = strictly_constrained_mask_u8
                
            morph_mask_to_keep = strictly_constrained_mask_u8
            
            print("DEBUG: Morphological Filter applied. Bitwise ROI intersection strictly enforced. mask_filtered updated.")
            
            # Save Output
            if out_params and save_dir:
                # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
                
                should_save = False
                for p in out_params:
                    if str(p).lower() == 'save':
                        should_save = True
                        break
                
                if should_save:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_mask = (mask_filtered.astype(np.uint8) * 255)
                    save_name = f"{fname_clean}_morph_{mode.lower().replace('-', '')}_mask.png"
                    cv2.imwrite(os.path.join(save_dir, save_name), save_mask)
                    print(f"Saved Morphological output: {save_name}")
            
        except Exception as e:
            print(f"Error executing Morphological Filter: {e}")
            traceback.print_exc()

    return mask_filtered, morph_mask_to_keep


def apply_filtering(mask_filtered, original_img, operations, filtering_df, save_dir, image_path, contour_image=None, adaptive_gaussian_config=None, visualization_overlay=None, defect_id_method=None, preferred_color_name=None, mask_default=None):
    """
    Centralized Filtering Function.
    Applies a sequence of filters (Area, RGB, Relativity, etc.) to the mask.
    
    Args:
        mask_filtered: Boolean mask (H, W)
        original_img: RGB image (H, W, 3)
        operations: List of parsed operations. Each op is dict:
                    {'method': 'Area'|'RGB'|..., 'key': str, 'output_params': list}
                    'key' corresponds to 'Method Setting' in Filtering sheet.
        filtering_df: DataFrame containing the Filter sheet rules (already filtered by FM/Product/Gen if possible, 
                      but we will match 'Method' and 'Method Setting' here).
        save_dir: Directory to save outputs
        image_path: Path to original image
        contour_image: Optional, for Area Filtering reference. Will be updated if Slicing is applied.
        adaptive_gaussian_config: Optional dict for Adaptive Gaussian configuration
        visualization_overlay: Optional image (H,W,3) to use as overlay for 'Pic' output (e.g. Dino heatmap).
                               If provided, 'Pic' will save [Original blended with Overlay] in masked regions.
        mask_default: Optional, default mask from DUT_Corrected/Contour_Corrected. 
                      If provided, all filtering operations will intersect with this mask.
    
    Returns:
        mask_filtered: Updated boolean mask
        extra_results: List of paths or metadata
        area_filter_mask_to_remove: Mask of pixels to remove from area filtering
        adaptive_gaussian_mask_to_keep: Mask of pixels to keep from adaptive gaussian
        contour_image: Updated contour image (may be modified by Slicing operation)
    """
    import pandas as pd
    import re
    
    extra_results = []
    
    if mask_filtered is None:
        return None, extra_results
    
    # --- NEW: Intersect with mask_default if provided ---
    # This ensures all filtering operations are constrained to the DUT region
    if mask_default is not None and mask_filtered is not None:
        # Ensure both masks are the same size
        if mask_default.shape[:2] != mask_filtered.shape[:2]:
            mask_default_resized = cv2.resize(mask_default, (mask_filtered.shape[1], mask_filtered.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_default_resized = mask_default
        
        # Convert to boolean for intersection
        if mask_filtered.dtype != bool:
            mask_filtered_bool = mask_filtered > 0
        else:
            mask_filtered_bool = mask_filtered
            
        if mask_default_resized.dtype != bool:
            mask_default_bool = mask_default_resized > 0
        else:
            mask_default_bool = mask_default_resized
        
        # Intersect: mask_filtered = mask_filtered AND mask_default
        mask_filtered = mask_filtered_bool & mask_default_bool
        
        # Convert back to uint8 if needed for consistency
        if mask_filtered.dtype == bool:
            mask_filtered = mask_filtered.astype(np.uint8) * 255
            
        print(f"DEBUG: Applied mask_default intersection. Pixels after intersect: {cv2.countNonZero(mask_filtered) if mask_filtered.dtype != bool else np.count_nonzero(mask_filtered)}")

    # Helper to normalize strings
    def _norm(s):
        return str(s).strip().lower().replace(' ', '').replace('_', '').replace('-', '')

    # Prepare Filtering DF Column Map
    col_map = {}
    if filtering_df is not None and not filtering_df.empty:
        # print(f"DEBUG: Filtering DF Columns: {filtering_df.columns.tolist()}")
        col_map = {c: _norm(c) for c in filtering_df.columns}
    else:
        print("DEBUG: Filtering DF is empty or None.")
    
    # Identify Columns
    method_col = None
    setting_col = None
    area_val_col = None
    threshold_col = None
    rgb_col = None
    tol_col = None
    prox_col = None
    group_col = None

    if filtering_df is not None and not filtering_df.empty:
        method_col = next((c for c in filtering_df.columns if col_map.get(c) == 'method'), None)
        setting_col = next((c for c in filtering_df.columns if col_map.get(c) in ('methodsetting', 'setting', 'key')), None)
        
        # Value Columns
        area_val_col = next((c for c in filtering_df.columns if col_map.get(c) in ('areafiltering%', 'areafiltering', 'area%', 'value')), None)
        threshold_col = next((c for c in filtering_df.columns if col_map.get(c) in ('threshold', 'thresh')), None)
        
        # RGB Columns
        rgb_col = next((c for c in filtering_df.columns if col_map.get(c) in ('rgb', 'rgbvalue', 'targetrgb')), None)
        tol_col = next((c for c in filtering_df.columns if col_map.get(c) in ('tolerance', 'tol')), None)
        prox_col = next((c for c in filtering_df.columns if col_map.get(c) in ('rgbproximity', 'proximity', 'prox')), None)
        group_col = next((c for c in filtering_df.columns if col_map.get(c) in ('targetedrgbgroup', 'groupid', 'group', 'rgbgroup', 'configgroup', 'config_group')), None)

    # Pre-process operations to handle format from parse_defect_id_method
    # The input 'operations' might be:
    # 1. [{'name': 'Filtering', 'params': ['Area(5)', 'Pic']}] -> Needs internal parsing
    # 2. [{'name': 'Area Filtering', 'params': ['Pic']}] -> Legacy/Transition format
    
    parsed_ops = []
    
    # Helper to check if name is a known method directly
    known_methods_set = {
        'area', 'areafiltering',
        'rgb', 'rgbfiltering',
        'colordistance', 'color_distance', 'colordistancefiltering',  # Add Color Distance support
        'relativity', 'relativityfiltering',
        'adaptivegaussian', 'adaptive_gaussian', 'ag',
        'hsv', 'hsvfiltering',
        'globalthreshold', 'global_threshold', 'gt',
        'bgsubtraction', 'backgroundsubtraction', 'bg_subtraction', 'bg_sub',
        'odbp', 'weighteddiff', 'odbpweighteddiff'
    }

    # Track operation sequence context
    # Initial context is the Defect Identification method (e.g., 'KNN-Dino')
    running_prev_op = str(defect_id_method) if defect_id_method else 'Dino'
    
    for op in operations:
        name = _norm(op.get('name', ''))
        params = op.get('params', [])
        
        # Determine the current operation's context (what came before it)
        # This context applies to ALL sub-operations derived from this operation block.
        current_context_op = running_prev_op
        
        if name == 'filtering':
            # This is the new syntax: Filtering[Area(5), Pic, Save]
            # Parse the params list to extract sub-operations
            print(f"DEBUG: Calling parse_filtering_params with params: {params}")
            sub_ops = parse_filtering_params(params)
            print(f"DEBUG: parse_filtering_params returned {len(sub_ops)} operations: {sub_ops}")
            
            # Inject context into sub_ops
            for sub_op in sub_ops:
                sub_op['previous_op_context'] = current_context_op
                
            parsed_ops.extend(sub_ops)
            
            # Update running_prev_op for the NEXT operation block
            # After this block, the previous op is 'Filtering'
            running_prev_op = 'Filtering'
        
        elif name in known_methods_set:
            # Legacy/Transition: Name is the method (e.g. 'Area Filtering')
            # Params are just output params (Pic, Save)
            op_dict = {
                'method': op.get('name'), # Keep original name for display/lookup
                'key': '', 
                'output_params': params,
                'previous_op_context': current_context_op
            }
            parsed_ops.append(op_dict)
            
            # Update running_prev_op
            running_prev_op = op.get('name')
            
        else:
            # Unknown outer wrapper, or maybe just pass it through if it has method/key structure?
            if 'method' in op:
                # If it already has method, assume it's a valid op object
                op['previous_op_context'] = current_context_op
                parsed_ops.append(op)
                running_prev_op = op.get('method')
            else:
                # Fallback: Treat name as method
                parsed_ops.append({
                    'method': op.get('name'),
                    'key': '',
                    'output_params': params,
                    'previous_op_context': current_context_op
                })
                running_prev_op = op.get('name')

    # Prepare return values for specific masks that dinov3_utils might need
    area_filter_mask_to_remove = None
    adaptive_gaussian_mask_to_keep = None
    
    print(f"DEBUG: apply_filtering parsed_ops: {parsed_ops}")

    # Store Pre-Adaptive Gaussian Mask to use as ROI reference for subsequent Relativity Filtering
    adaptive_gaussian_reference_mask = None
    save_step_counter = 0

    def save_with_counter(out_params, mask_to_save):
        nonlocal save_step_counter
        save_ops = _parse_save_operations(out_params)
        print(f"DEBUG: save_with_counter called with out_params={out_params}, save_ops={save_ops}, contour_image={'present' if contour_image is not None else 'None'}")
        if save_ops:
            save_step_counter += 1
            step_suffix = f"_step{save_step_counter}"
            print(f"DEBUG: Calling _save_images_for_operations with step_suffix={step_suffix}, contour_image={'present' if contour_image is not None else 'None'}")
            _save_images_for_operations(save_ops, original_img, mask_to_save, contour_image, save_dir, image_path, step_suffix)
        else:
            print(f"DEBUG: save_ops is empty, skipping save")

    for i, op in enumerate(parsed_ops):
        method = op.get('method', '').strip()
        key = str(op.get('key', '')).strip() # The value in () e.g. '5' or 'pct'
        out_params = op.get('output_params', [])
        
        # Retrieve context from operation
        op_context = op.get('previous_op_context', 'Other')

        
        print(f"DEBUG: Processing Filter Operation: Method='{method}', Key='{key}', Context='{op_context}'")
        
        # Determine if we should enable Save Overlay based on Previous Operation
        # User Rule: "Only if the immediately preceding operation is Dino, trigger _visualization.png"
        is_prev_op_dino = 'dino' in str(op_context).lower()
        
        # We control the behavior by:
        # 1. enable_save_overlay: Set to True if condition met (redundant but safe)
        # 2. defect_id_method: Pass a string containing 'Dino' if condition met, else 'Other'
        
        enable_save_overlay = is_prev_op_dino
        step_defect_id_context = "Dino" if is_prev_op_dino else "Other"
        
        # 1. Find matching rule in DataFrame
        matched_row = None
        if filtering_df is not None and not filtering_df.empty and method_col:
            # Debug: Print lookup intent
            # print(f"DEBUG: Looking for Method='{method}' in Filtering DF...")
            
            # Filter by Method
            df_methods = filtering_df[method_col].astype(str).apply(_norm)
            target_method = _norm(method)
            
            # For HSV filtering, require exact match to avoid partial matches
            if target_method == 'hsv':
                # Exact match for 'hsv'
                matches_by_method = filtering_df[df_methods == target_method]
                print(f"DEBUG: HSV filtering - looking for exact Method match: '{target_method}'")
            else:
                # For other methods, use contains (original behavior)
                matches_by_method = filtering_df[df_methods.str.contains(target_method, na=False)]
            
            # Optional: Restrict by RGB Proximity/Color first (original logic)
            if matches_by_method is not None and not matches_by_method.empty and preferred_color_name is not None:
                try:
                    f_cols_norm = {c: _norm(c) for c in filtering_df.columns}
                    prox_col_local = next((c for c in filtering_df.columns if f_cols_norm[c] in ('rgbproximity', 'proximity', 'color', 'colorname', 'rgbgroup', 'targetedrgbgroup')), None)
                    if prox_col_local:
                        prox_series_norm = matches_by_method[prox_col_local].astype(str).apply(_norm)
                        target_color_norm = _norm(preferred_color_name)
                        filtered_by_color = matches_by_method[prox_series_norm == target_color_norm]
                        if not filtered_by_color.empty:
                            matches_by_method = filtered_by_color
                            print(f"DEBUG: Filtering DF restricted by RGB Proximity/Color='{preferred_color_name}'. Rows: {len(matches_by_method)}")
                        else:
                            color_tokens = re.split(r'[^a-z]+', target_color_norm)
                            color_tokens = [t for t in color_tokens if t]
                            known_colors = {'green', 'blue', 'brown', 'gray', 'grey', 'purple', 'red', 'yellow', 'orange', 'black', 'white'}
                            base_color = next((t for t in color_tokens if t in known_colors), None)
                            if base_color == 'grey':
                                base_color = 'gray'
                            if base_color:
                                fuzzy = matches_by_method[prox_series_norm.str.contains(base_color, na=False)]
                                if not fuzzy.empty:
                                    matches_by_method = fuzzy
                                    print(f"DEBUG: Fuzzy RGB Proximity match using base color '{base_color}'. Rows: {len(matches_by_method)}")
                                else:
                                    print(f"Alarm: No fuzzy RGB Proximity match for base color '{base_color}'. Keeping method-only matches.")
                            else:
                                print(f"Alarm: No exact RGB Proximity match for '{preferred_color_name}'. Keeping method-only matches.")
                except Exception as e_color:
                    print(f"Warning: Failed to apply RGB Proximity restriction: {e_color}")
            
            if not matches_by_method.empty:
                is_pixel_mode = False
                try:
                    method_norm_for_pixel = _norm(method)
                    if method_norm_for_pixel in ('area', 'areafiltering'):
                        key_norm_for_pixel = _norm(key)
                        if key_norm_for_pixel:
                            if 'pixel' in key_norm_for_pixel or key_norm_for_pixel.endswith('px'):
                                is_pixel_mode = True
                except Exception:
                    is_pixel_mode = False
                # Filter by Setting (Key)
                if setting_col and key:
                    df_settings = matches_by_method[setting_col].astype(str).apply(_norm)
                    target_setting = _norm(key)
                    matches = matches_by_method[df_settings == target_setting]
                    
                    if not matches.empty:
                        matched_row = matches.iloc[0]
                        # print(f"DEBUG: Found matching rule in Filtering sheet.")
                    else:
                        method_norm = _norm(method)
                        if method_norm in ('hsv', 'hsvfiltering'):
                            print(f"Alarm: HSV Filtering 未找到匹配的 Method Setting='{key}'，将使用首条HSV规则")
                            matched_row = matches_by_method.iloc[0]
                        elif is_pixel_mode:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No exact Key match for '{key}'. Using first Area rule for Pixel mode.")
                        else:
                            print(f"DEBUG: No rule found for Key='{key}' in Method='{method}'")
                else:
                    method_norm = _norm(method)
                    if method_norm in ('hsv', 'hsvfiltering'):
                        if setting_col is None:
                            print("Alarm: HSV Filtering 过滤表缺少 Method Setting 列，将使用首条HSV规则")
                        # 若未提供 Key，但已按颜色/方法筛到行，直接使用首行，不再报警
                        matched_row = matches_by_method.iloc[0]
                    elif is_pixel_mode:
                        matched_row = matches_by_method.iloc[0]
                        print(f"DEBUG: No Method Setting column or key missing. Using first Area rule for Pixel mode.")
                    elif 'morphtophat' in method_norm or 'morphblackhat' in method_norm:
                        # For Morph Top-Hat/Black-Hat, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Morph rule from sheet.")
                            print(f"DEBUG: Morph Top-Hat - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Morph Top-Hat/Black-Hat in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'globalthreshold' in method_norm or 'global_threshold' in method_norm or method_norm == 'gt':
                        # For Global Threshold, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Global Threshold rule from sheet.")
                            print(f"DEBUG: Global Threshold - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Global Threshold in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'slicing' in method_norm or 'slice' in method_norm:
                        # For Slicing, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Slicing rule from sheet.")
                            print(f"DEBUG: Slicing - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Slicing in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'shape' in method_norm or 'shapefiltering' in method_norm:
                        # For Shape Filtering, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Shape rule from sheet.")
                            print(f"DEBUG: Shape Filtering - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Shape Filtering in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'bandroi' in method_norm or 'band' in method_norm:
                        # For Band-ROI, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Band-ROI rule from sheet.")
                            print(f"DEBUG: Band-ROI - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Band-ROI in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'stencilroi' in method_norm or 'stencil' in method_norm:
                        # For Stencil-ROI, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Stencil-ROI rule from sheet.")
                            print(f"DEBUG: Stencil-ROI - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Stencil-ROI in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'odbp' in method_norm or 'weighteddiff' in method_norm:
                        # For ODBP, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first ODBP rule from sheet.")
                            print(f"DEBUG: ODBP - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for ODBP in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'bgsubtraction' in method_norm or 'backgroundsubtraction' in method_norm:
                        # For BG Subtraction, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first BG Subtraction rule from sheet.")
                            print(f"DEBUG: BG Subtraction - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for BG Subtraction in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
                    elif 'colordistance' in method_norm or 'color_distance' in method_norm:
                        # For Color Distance, use first matching row if key is empty
                        if not matches_by_method.empty:
                            matched_row = matches_by_method.iloc[0]
                            print(f"DEBUG: No key provided for {method}. Using first Color Distance rule from sheet.")
                            print(f"DEBUG: Color Distance - Selected row data:\n{matched_row}")
                        else:
                            print(f"DEBUG: No matching rows found for Color Distance in filtering_df")
                            print(f"DEBUG: filtering_df has {len(filtering_df)} rows, method_col='{method_col}'")
                            if method_col:
                                print(f"DEBUG: Available methods in filtering_df: {filtering_df[method_col].unique().tolist()}")
            else:
                # Debug: Show available methods if not found
                unique_methods = df_methods.unique().tolist()
                print(f"DEBUG: No rule found for Method='{method}'. Available Methods in Sheet: {unique_methods}")
        else:
            if filtering_df is None or filtering_df.empty:
                print("Warning: Filtering DF is empty or None.")
            elif not method_col:
                print(f"Warning: Could not detect 'Method' column in Filtering sheet. Columns: {filtering_df.columns.tolist()}")

        
        # 2. Apply Logic based on Method
        
        # Determine Step Suffix for filename uniqueness (e.g. _step1, _step2)
        step_suffix = f"_step{i+1}"
        
        # --- AREA FILTERING ---
        if _norm(method) in ('area', 'areafiltering'):
            # Check if this is actually a Relativity request (Area(Relativity))
            is_relativity_redirect = False
            if 'relativity' in _norm(key):
                is_relativity_redirect = True

            # Check if this is a Pixel request (Area(Pixel) or Area(10px))
            is_pixel_mode = False
            pixel_val = None
            
            # 1. Check for explicit 'Pixel' key keyword or 'px' suffix in direct value
            key_norm = _norm(key)
            if 'pixel' in key_norm or key_norm.endswith('px'):
                is_pixel_mode = True
                # If key is like '10px', parse it directly
                if key_norm.endswith('px'):
                     try:
                         pixel_val = float(key_norm.replace('px', '').strip())
                     except: pass
                elif 'pixel' in key_norm and key_norm != 'pixel': # e.g. '10pixels'
                     try:
                         pixel_val = float(key_norm.replace('pixels', '').replace('pixel', '').strip())
                     except: pass

            pct_val = None
            if matched_row is not None and area_val_col:
                try:
                    val = matched_row.get(area_val_col)
                    val_float = float(val)
                    if is_pixel_mode:
                        pixel_val = val_float
                    else:
                        pct_val = val_float
                except Exception:
                    pass
            
            # Fallback to parsing key directly if not found in sheet
            if pct_val is None and pixel_val is None:
                try:
                    val_float = float(key)
                    # If we already decided it's pixel mode (e.g. key was just 'Pixel' but no sheet match? Unlikely. 
                    # But if key was '10', it defaults to pct unless is_pixel_mode is forced)
                    if is_pixel_mode:
                        pixel_val = val_float
                    else:
                        pct_val = val_float
                except ValueError:
                    pass
            
            if is_relativity_redirect:
                if pct_val is not None:
                    print(f"DEBUG: Redirecting Area(Relativity) to Relativity Filtering with TH={pct_val}")
                    
                    # Capture mask before Relativity
                    mask_before_relativity = mask_filtered.copy() if mask_filtered is not None else None
                    
                    mask_filtered = apply_relativity_filtering(
                        mask_filtered, pct_val, out_params, save_dir, image_path, 
                        reference_roi_mask=adaptive_gaussian_reference_mask,
                        original_img=original_img,
                        visualization_overlay=visualization_overlay,
                        enable_save_overlay=enable_save_overlay,
                        defect_id_method=step_defect_id_context
                    )
                    
                    # Calculate removed mask
                    if mask_before_relativity is not None and mask_filtered is not None:
                         # Ensure same type/shape
                         if mask_before_relativity.shape == mask_filtered.shape:
                             if mask_before_relativity.dtype == bool:
                                 removed = mask_before_relativity & (~mask_filtered)
                                 removed_uint8 = removed.astype(np.uint8) * 255
                             else:
                                 removed_uint8 = cv2.bitwise_and(mask_before_relativity, cv2.bitwise_not(mask_filtered))
                             
                             # Update area_filter_mask_to_remove
                             if area_filter_mask_to_remove is None:
                                 area_filter_mask_to_remove = removed_uint8
                             else:
                                 area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
                else:
                    print(f"Warning: Area(Relativity) requested but no threshold value found in Filtering sheet.")

            elif pct_val is not None or pixel_val is not None:
                mask_filtered, current_mask_to_remove, new_paths = apply_area_filtering(
                    mask_filtered, contour_image, pct_val, out_params, original_img, save_dir, image_path,
                    visualization_overlay=visualization_overlay,
                    enable_save_overlay=enable_save_overlay,
                    defect_id_method=step_defect_id_context,
                    step_suffix=step_suffix,
                    area_filtering_pixel=pixel_val
                )
                extra_results.extend(new_paths)
                save_with_counter(out_params, mask_filtered)
                
                # Accumulate mask to remove
                if current_mask_to_remove is not None:
                    if area_filter_mask_to_remove is None:
                        area_filter_mask_to_remove = current_mask_to_remove
                    else:
                        # Combine removal masks (bitwise OR of removed areas)
                        # mask_to_remove is 255 where removed.
                        area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, current_mask_to_remove)
            else:
                print(f"Warning: Could not determine Area Filtering Threshold (Pct or Pixel) for '{method}({key})'")

        # --- RGB FILTERING ---
        elif _norm(method) in ('rgb', 'rgbfiltering'):
            if matched_row is not None:
                rgb_config = {
                    'rgb': matched_row.get(rgb_col) if rgb_col else None,
                    'tolerance': matched_row.get(tol_col) if tol_col else None,
                    'proximity': matched_row.get(prox_col) if prox_col else None,
                    'group_id': matched_row.get(group_col) if group_col else None
                }
                
                # Capture mask before RGB to calculate what was removed
                mask_before_rgb = mask_filtered.copy() if mask_filtered is not None else None
                
                mask_filtered, new_paths = apply_rgb_filtering(
                    original_img, mask_filtered, rgb_config, out_params, save_dir, image_path, contour_image
                )
                extra_results.extend(new_paths)
                save_with_counter(out_params, mask_filtered)
                
                # Calculate removed mask
                if mask_before_rgb is not None and mask_filtered is not None:
                    # Ensure same type/shape
                    if mask_before_rgb.shape == mask_filtered.shape:
                        if mask_before_rgb.dtype == bool:
                            removed = mask_before_rgb & (~mask_filtered)
                            removed_uint8 = removed.astype(np.uint8) * 255
                        else:
                            removed_uint8 = cv2.bitwise_and(mask_before_rgb, cv2.bitwise_not(mask_filtered))
                            
                        # Update area_filter_mask_to_remove
                        if area_filter_mask_to_remove is None:
                            area_filter_mask_to_remove = removed_uint8
                        else:
                            area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
            else:
                print(f"Warning: No matching RGB rule found for '{method}({key})'")
        
        # --- COLOR DISTANCE FILTERING ---
        elif _norm(method) in ('colordistance', 'color_distance', 'colordistancefiltering'):
            # Color Distance uses Method Setting to parse config: [Prox = Near, RGB = (0,0,0), Tolerance = 206]
            if matched_row is not None:
                # Parse config from Method Setting column
                setting_val = matched_row.get(setting_col) if setting_col else None
                
                # Default values
                target_rgb = None
                tolerance = 0.0
                proximity = 'near'
                
                if setting_val is not None and pd.notna(setting_val):
                    setting_str = str(setting_val).strip()
                    print(f"DEBUG: Color Distance parsing Method Setting: '{setting_str}'")
                    
                    # Parse Prox = Near/Far
                    import re
                    prox_match = re.search(r'prox\s*=\s*(\w+)', setting_str, re.IGNORECASE)
                    if prox_match:
                        proximity = prox_match.group(1).lower()
                    
                    # Parse RGB = (0,0,0) or RGB = [0,0,0]
                    rgb_match = re.search(r'rgb\s*=\s*[\(\[]([^\)\]]+)[\)\]]', setting_str, re.IGNORECASE)
                    if rgb_match:
                        rgb_str = rgb_match.group(1)
                        try:
                            target_rgb = [float(x.strip()) for x in rgb_str.split(',')]
                        except:
                            print(f"Warning: Could not parse RGB values from '{rgb_str}'")
                    
                    # Parse Tolerance = 206
                    tol_match = re.search(r'tolerance\s*=\s*(\d+)', setting_str, re.IGNORECASE)
                    if tol_match:
                        tolerance = float(tol_match.group(1))
                
                if target_rgb is not None:
                    rgb_config = {
                        'rgb': target_rgb,
                        'tolerance': tolerance,
                        'proximity': proximity,
                        'group_id': matched_row.get(group_col) if group_col else None
                    }
                    
                    print(f"DEBUG: Color Distance config - RGB: {target_rgb}, Tolerance: {tolerance}, Proximity: {proximity}")
                    
                    # Capture mask before Color Distance to calculate what was removed
                    mask_before_cd = mask_filtered.copy() if mask_filtered is not None else None
                    
                    mask_filtered, new_paths = apply_rgb_filtering(
                        original_img, mask_filtered, rgb_config, out_params, save_dir, image_path, contour_image
                    )
                    extra_results.extend(new_paths)
                    save_with_counter(out_params, mask_filtered)
                    
                    # Calculate removed mask
                    if mask_before_cd is not None and mask_filtered is not None:
                        if mask_before_cd.shape == mask_filtered.shape:
                            if mask_before_cd.dtype == bool:
                                removed = mask_before_cd & (~mask_filtered)
                                removed_uint8 = removed.astype(np.uint8) * 255
                            else:
                                removed_uint8 = cv2.bitwise_and(mask_before_cd, cv2.bitwise_not(mask_filtered))
                                
                            if area_filter_mask_to_remove is None:
                                area_filter_mask_to_remove = removed_uint8
                            else:
                                area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
                else:
                    print(f"Warning: Color Distance requires RGB values in Method Setting. Format: [Prox = Near, RGB = (0,0,0), Tolerance = 206]")
            else:
                print(f"Warning: No matching Color Distance rule found for '{method}({key})'")
        
        # --- HSV FILTERING ---
        elif _norm(method) in ('hsv', 'hsvfiltering'):
            if matched_row is not None:
                # Method Setting (key) is used for matching.
                # The actual ranges should be in a value column to allow named configs (e.g. HSV(Red)).
                # We try 'RGB Code' (rgb_col) first, then 'Value' (area_val_col), then fallback to 'Method Setting' (setting_col).
                
                ranges_val = None
                
                # Debug HSV Columns
                val_rgb = matched_row.get(rgb_col) if rgb_col else None
                val_area = matched_row.get(area_val_col) if area_val_col else None
                val_setting = matched_row.get(setting_col) if setting_col else None
                print(f"DEBUG: HSV Config Lookup - RGB Col: '{val_rgb}', Area Val Col: '{val_area}', Setting Col: '{val_setting}'")

                # Revert Logic: Prioritize Area Value (Value Column), then Method Setting (Key)
                # Ignore RGB Code for HSV Ranges unless explicitly needed, but standard format is in Value col.
                
                if area_val_col and val_area is not None and str(val_area).strip() != '' and str(val_area).lower() != 'nan':
                     ranges_val = val_area
                
                if (ranges_val is None or str(ranges_val).strip() == '' or str(ranges_val).lower() == 'nan') and setting_col:
                     if val_setting is not None and str(val_setting).strip() != '' and str(val_setting).lower() != 'nan':
                         # Only use setting col if it looks like a range (list of tuples), not just a name like 'HSV(Red)'
                         s_val = str(val_setting).strip()
                         if s_val.startswith('[') and '(' in s_val:
                             ranges_val = val_setting
                
                # Fallback to RGB Col if still nothing (though unlikely for HSV)
                if (ranges_val is None or str(ranges_val).strip() == '' or str(ranges_val).lower() == 'nan') and rgb_col:
                     if val_rgb is not None and str(val_rgb).strip() != '' and str(val_rgb).lower() != 'nan':
                         ranges_val = val_rgb

                print(f"DEBUG: Selected HSV Ranges Value: '{ranges_val}'")
                
                # For HSV, reuse 'RGB Proximity' column to store color_name (per user spec)
                color_name_for_hsv = matched_row.get(prox_col) if prox_col else None
                
                # --- Parse Extended HSV Config ---
                # Format: [HSV=((0,179), (128,255), (130,255)), Size=1/2, Uniform Light=on, Gamma=1, Contrast=0.9, Min Area=25]
                hsv_config = {
                    'ranges': ranges_val,
                    'color_name': color_name_for_hsv,
                    'size': 1.0,
                    'uniform_light': False,
                    'ul_kernel_size': 101,
                    'gamma': 1.0,
                    'contrast': 1.0,
                    'min_area': 0
                }
                
                # Parse config from the ranges_val string if it contains extended params
                if ranges_val and isinstance(ranges_val, str):
                    config_str = str(ranges_val).strip()
                    print(f"DEBUG: Parsing HSV config string: {config_str[:100]}...")
                    
                    # Check if it's the simplified format: [((0,255), (16,255), (0,255)), Size=1/2]
                    # This format has HSV values directly without 'HSV=' prefix
                    import re
                    simplified_match = re.match(r'\[\s*\(\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*\)', config_str)
                    if simplified_match or (config_str.startswith('[') and config_str.count('(') >= 6 and 'hsv=' not in config_str.lower()):
                        print(f"DEBUG: Detected simplified HSV format (without HSV= prefix)")
                        try:
                            # Extract HSV values using regex
                            hsv_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\).*\(\s*(\d+)\s*,\s*(\d+)\s*\).*\(\s*(\d+)\s*,\s*(\d+)\s*\)', config_str)
                            if hsv_match:
                                h_min, h_max, s_min, s_max, v_min, v_max = hsv_match.groups()
                                hsv_config['ranges'] = ((int(h_min), int(h_max)), (int(s_min), int(s_max)), (int(v_min), int(v_max)))
                                print(f"DEBUG: Parsed simplified HSV ranges: {hsv_config['ranges']}")
                            
                            # Extract Size parameter
                            size_match = re.search(r'Size\s*=\s*(\d+)\s*/\s*(\d+)', config_str, re.IGNORECASE)
                            if size_match:
                                num, den = size_match.groups()
                                hsv_config['size'] = float(num) / float(den)
                                print(f"DEBUG: Parsed Size from simplified format: {num}/{den} = {hsv_config['size']}")
                        except Exception as e:
                            print(f"Warning: Failed to parse simplified HSV format: {e}")
                    
                    # Continue with extended format parsing if it has HSV= prefix or other params
                    elif 'hsv' in config_str.lower() or 'size' in config_str.lower() or 'gamma' in config_str.lower():
                        try:
                            # Remove outer brackets
                            if config_str.startswith('[') and config_str.endswith(']'):
                                config_str = config_str[1:-1]
                            
                            print(f"DEBUG: Config string after removing brackets: {config_str[:100]}...")
                            
                            # Smart split: use depth counter to handle nested parentheses
                            parts = []
                            buf = ""
                            depth = 0
                            for char in config_str:
                                if char == '(': depth += 1
                                elif char == ')': depth -= 1
                                elif char == ',' and depth == 0:
                                    parts.append(buf.strip())
                                    buf = ""
                                    continue
                                buf += char
                            if buf.strip(): parts.append(buf.strip())
                            
                            print(f"DEBUG: Parsed {len(parts)} parts: {parts}")
                            
                            for p in parts:
                                if '=' in p:
                                    k, v = p.split('=', 1)
                                    k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                    v_val = v.strip()
                                    
                                    try:
                                        if k_norm == 'hsv':
                                            # HSV ranges like ((0,179), (128,255), (130,255))
                                            try:
                                                import ast
                                                hsv_config['ranges'] = ast.literal_eval(v_val)
                                            except:
                                                print(f"Warning: Failed to parse HSV ranges from '{v_val}'")
                                                hsv_config['ranges'] = v_val
                                        elif k_norm in ('size', 'resize', 'scale'):
                                            try:
                                                val_f = 1.0
                                                if '/' in v_val:
                                                    num, den = v_val.split('/')
                                                    val_f = float(num) / float(den)
                                                else:
                                                    val_f = float(v_val)
                                                
                                                # Smart detection: if value > 5, treat as UL Kernel Size
                                                if val_f > 5.0:
                                                    hsv_config['ul_kernel_size'] = int(val_f)
                                                    print(f"DEBUG: Parsed 'Size={v_val}' as UL Kernel Size: {int(val_f)}")
                                                else:
                                                    hsv_config['size'] = val_f
                                                    print(f"DEBUG: Parsed 'Size={v_val}' as Resize Scale: {val_f}")
                                            except ValueError:
                                                pass
                                        elif k_norm in ('uniformlight', 'ul', 'uniform_light'):
                                            hsv_config['uniform_light'] = v_val.lower() in ('on', 'true', '1', 'yes')
                                        elif k_norm in ('ulkernelsize', 'ul_kernel', 'illum_kernel', 'ulkernel'):
                                            hsv_config['ul_kernel_size'] = int(float(v_val))
                                        elif k_norm in ('gamma',):
                                            hsv_config['gamma'] = float(v_val)
                                        elif k_norm in ('contrast',):
                                            hsv_config['contrast'] = float(v_val)
                                        elif k_norm in ('minarea', 'area', 'min_area'):
                                            hsv_config['min_area'] = float(v_val)
                                        elif k_norm in ('denoise', 'blur', 'gaussian'):
                                            hsv_config['denoise'] = int(float(v_val))
                                            print(f"DEBUG: Parsed 'Denoise={v_val}' as Denoise: {int(float(v_val))}")
                                        elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                            hsv_config['roi_shrink'] = float(v_val)
                                            print(f"DEBUG: Parsed 'ROI Shrink={v_val}' as ROI Shrink: {float(v_val)}")
                                    except ValueError:
                                        pass
                            
                            print(f"DEBUG: Parsed Extended HSV Config: {hsv_config}")
                            print(f"DEBUG: HSV ranges after parsing: {hsv_config.get('ranges')}")
                        except Exception as e:
                            print(f"Warning: Failed to parse extended HSV config: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Capture mask before HSV to compute removed pixels
                mask_before_hsv = mask_filtered.copy() if mask_filtered is not None else None
                
                mask_filtered, new_paths = apply_hsv_filtering(
                    original_img, mask_filtered, hsv_config, out_params, save_dir, image_path, contour_image
                )
                
                # Apply Min Area filtering if specified
                if hsv_config.get('min_area', 0) > 0 and mask_filtered is not None:
                    min_area = hsv_config['min_area']
                    mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
                    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    mask_clean = np.zeros_like(mask_u8)
                    for c in cnts:
                        if cv2.contourArea(c) >= min_area:
                            cv2.drawContours(mask_clean, [c], -1, 255, -1)
                    mask_filtered = mask_clean > 0
                    print(f"DEBUG: Applied HSV Min Area filter: {min_area}. Kept {cv2.countNonZero(mask_clean)} pixels.")
                
                extra_results.extend(new_paths)
                
                save_with_counter(out_params, mask_filtered)
                
                if mask_before_hsv is not None and mask_filtered is not None:
                    if mask_before_hsv.shape == mask_filtered.shape:
                        if mask_before_hsv.dtype == bool:
                            removed = mask_before_hsv & (~mask_filtered)
                            removed_uint8 = removed.astype(np.uint8) * 255
                        else:
                            removed_uint8 = cv2.bitwise_and(mask_before_hsv, cv2.bitwise_not(mask_filtered))
                        
                        if area_filter_mask_to_remove is None:
                            area_filter_mask_to_remove = removed_uint8
                        else:
                            area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
            else:
                print(f"Warning: No matching HSV rule found for '{method}({key})'")
        
        # --- RELATIVITY FILTERING ---
        elif _norm(method) in ('relativity', 'relativityfiltering'):
            relativity_th = None
            if matched_row is not None and area_val_col:
                 try:
                    relativity_th = float(matched_row.get(area_val_col))
                 except:
                    pass
            
            if relativity_th is None and key:
                try:
                    relativity_th = float(key)
                except ValueError:
                    pass
            
            if relativity_th is not None:
                # Capture mask before Relativity to calculate what was removed
                mask_before_relativity = mask_filtered.copy() if mask_filtered is not None else None
                
                mask_filtered = apply_relativity_filtering(
                    mask_filtered, relativity_th, out_params, save_dir, image_path,
                    reference_roi_mask=adaptive_gaussian_reference_mask,
                    visualization_overlay=visualization_overlay,
                    original_img=original_img,
                    enable_save_overlay=enable_save_overlay,
                    defect_id_method=step_defect_id_context
                )
                save_with_counter(out_params, mask_filtered)
                
                # Calculate removed mask (pixels that were 255/True and became 0/False)
                if mask_before_relativity is not None and mask_filtered is not None:
                    # Ensure same type/shape
                    if mask_before_relativity.shape == mask_filtered.shape:
                        # Removed = Before AND (NOT After)
                        # Works for boolean or uint8 (0/255)
                        if mask_before_relativity.dtype == bool:
                            removed = mask_before_relativity & (~mask_filtered)
                            removed_uint8 = removed.astype(np.uint8) * 255
                        else:
                            removed_uint8 = cv2.bitwise_and(mask_before_relativity, cv2.bitwise_not(mask_filtered))
                            
                        # Update area_filter_mask_to_remove
                        if area_filter_mask_to_remove is None:
                            area_filter_mask_to_remove = removed_uint8
                        else:
                            area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
            else:
                print(f"Warning: Could not determine Relativity Threshold for '{method}({key})'")

        # --- SHAPE FILTERING ---
        elif _norm(method) in ('shape', 'shapefiltering'):
            target_shape = None
            if matched_row is not None and setting_col:
                target_shape = matched_row.get(setting_col)
            
            if target_shape is None and key:
                target_shape = key
            
            if target_shape and _norm(target_shape) == 'line':
                # Get Aspect Ratio Threshold
                aspect_ratio_th = 3.0 # Default
                if matched_row is not None and threshold_col:
                     val = matched_row.get(threshold_col)
                     try:
                         val_float = float(val)
                         if val_float > 0:
                            aspect_ratio_th = val_float
                     except: pass

                # Check Update Mask Flag
                should_update_mask = False
                um_val_debug = "N/A"
                if matched_row is not None:
                     # Find 'Update Mask' column
                     f_cols_norm = {c: _norm(c) for c in filtering_df.columns}
                     update_mask_col = next((c for c in filtering_df.columns if f_cols_norm[c] in ('updatemask', 'update mask')), None)
                     
                     if update_mask_col:
                         um_val = str(matched_row.get(update_mask_col, '')).strip().lower()
                         um_val_debug = um_val
                         if um_val in ('yes', 'true', '1'):
                             should_update_mask = True
                             
                print(f"DEBUG: Applying Shape Filtering (Target: Line, Min Aspect Ratio: {aspect_ratio_th}, Update Mask: {should_update_mask}, Value: '{um_val_debug}')")
                
                # Capture mask before Shape Filtering
                mask_before_shape = mask_filtered.copy() if mask_filtered is not None else None
                if mask_filtered is not None and mask_filtered.dtype == bool:
                    mask_filtered_uint8 = (mask_filtered.astype('uint8') * 255)
                else:
                    mask_filtered_uint8 = mask_filtered
                pixels_before = cv2.countNonZero(mask_filtered_uint8) if mask_filtered_uint8 is not None else 0
                print(f"DEBUG: Shape Filtering Input Mask Pixels: {pixels_before}")
















                if mask_filtered is not None:
                    # Find contours
                    # Ensure mask is uint8 for findContours
                    # IMPORTANT: Use a copy because findContours might modify the source in some OpenCV versions
                    if mask_filtered.dtype == bool:
                        mask_for_contours = (mask_filtered.astype(np.uint8) * 255).copy()
                    else:
                        mask_for_contours = mask_filtered.copy()
                    
                    contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Create new mask
                    new_mask = np.zeros_like(mask_for_contours)
                    
                    kept_count = 0
                    for cnt in contours:
                        if is_line(cnt, min_aspect_ratio=aspect_ratio_th):
                            cv2.drawContours(new_mask, [cnt], -1, 255, -1) # 255 for uint8
                            kept_count += 1
                    
                    if should_update_mask:
                        # Ensure mask type consistency
                        if mask_filtered.dtype == bool:
                            mask_filtered = new_mask > 0
                        else:
                            mask_filtered = new_mask
                            
                        print(f"DEBUG: Shape Filtering kept {kept_count}/{len(contours)} contours (Mask UPDATED).")

                        # Calculate removed mask
                        if mask_before_shape is not None:
                             # Ensure same type/shape
                             if mask_before_shape.shape == mask_filtered.shape:
                                 if mask_before_shape.dtype == bool:
                                     removed = mask_before_shape & (~mask_filtered)
                                     removed_uint8 = removed.astype(np.uint8) * 255
                                 else:
                                     removed_uint8 = cv2.bitwise_and(mask_before_shape, cv2.bitwise_not(mask_filtered))
                                 
                                 # Update area_filter_mask_to_remove
                                 if area_filter_mask_to_remove is None:
                                     area_filter_mask_to_remove = removed_uint8
                                 else:
                                     area_filter_mask_to_remove = cv2.bitwise_or(area_filter_mask_to_remove, removed_uint8)
                    else:
                        print(f"DEBUG: Shape Filtering kept {kept_count}/{len(contours)} contours. Mask update SKIPPED (Handled in parametric output logic).")


            else:
                print(f"Warning: Shape Filtering requested but target shape '{target_shape}' not recognized (Only 'Line' supported).")

        # --- ADAPTIVE GAUSSIAN ---
        elif _norm(method) in ('adaptivegaussian', 'adaptive_gaussian', 'ag'):
            if adaptive_gaussian_config:
                print("DEBUG: Applying Adaptive Gaussian via centralized Filtering.")

                # Capture Mask BEFORE AG to serve as ROI Reference for subsequent Relativity Filtering
                if mask_filtered is not None:
                    adaptive_gaussian_reference_mask = mask_filtered.copy()
                    print("DEBUG: Captured Pre-AG Mask as Reference for Relativity.")

                mask_filtered, current_mask_to_keep = apply_adaptive_gaussian(
                    original_img, mask_filtered, adaptive_gaussian_config,
                    out_params, save_dir, image_path,
                    contour_image=contour_image
                )
                save_with_counter(out_params, mask_filtered)

                # Accumulate mask to keep
                # Adaptive Gaussian returns a 'keep' mask (boolean or uint8)
                # Note: apply_adaptive_gaussian returns (mask_filtered, mask_to_keep)
                if current_mask_to_keep is not None:
                    adaptive_gaussian_mask_to_keep = current_mask_to_keep
            else:
                print("Warning: Adaptive Gaussian requested but no config provided.")

        # --- MORPHOLOGICAL FILTERING (Top-Hat / Black-Hat) ---
        elif _norm(method) in ('morphtophat', 'morphblackhat', 'morph_tophat', 'morph_blackhat', 'morphtop-hat', 'morphblack-hat'):
            # Determine mode from method name
            mode = 'Top-Hat'
            if 'black' in _norm(method):
                mode = 'Black-Hat'
            
            # Default Config
            morph_config = {
                'mode': mode,
                'kernel_size': 15,
                'threshold': 127,
                'denoise': 0,
                'open': 0,
                'close': 0,
                'min_area': 0,
                'size': 1.0,
                'uniform_light': False,
                'ul_kernel_size': 101
            }
            
            # Try to find config in filtering_df if method matches
            print(f"DEBUG: Morph Top-Hat - matched_row is None: {matched_row is None}")
            if matched_row is not None:
                print(f"DEBUG: Morph Top-Hat - Found matched_row. Columns: {matched_row.index.tolist() if hasattr(matched_row, 'index') else 'N/A'}")
                # Parse config from Method Setting or Value column
                # Format: [Kernel Size = 7, Threshold = 29, Denoise = 1, Open=1, Min Area = 25]
                config_str = None
                if setting_col: 
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: Morph Top-Hat - Trying setting_col '{setting_col}': {config_str}")
                if (config_str is None or str(config_str).strip() == '') and area_val_col:
                    config_str = matched_row.get(area_val_col)
                    print(f"DEBUG: Morph Top-Hat - Trying area_val_col '{area_val_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip().replace('[', '').replace(']', '')
                        parts = s.split(',')
                        for p in parts:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('kernelsize', 'kernalsize', 'ksize', 'kernel', 'kernelsize'):
                                        morph_config['kernel_size'] = int(float(v_val))
                                    elif k_norm in ('threshold', 'thresh'):
                                        morph_config['threshold'] = int(float(v_val))
                                    elif k_norm in ('denoise',):
                                        morph_config['denoise'] = int(float(v_val))
                                    elif k_norm in ('open', 'morphopen'):
                                        morph_config['open'] = int(float(v_val))
                                    elif k_norm in ('close', 'morphclose'):
                                        morph_config['close'] = int(float(v_val))
                                    elif k_norm in ('minarea', 'area'):
                                        morph_config['min_area'] = float(v_val)
                                    elif k_norm in ('size', 'resize', 'scale'):
                                        if '/' in v_val:
                                            num, den = v_val.split('/')
                                            morph_config['size'] = float(num) / float(den)
                                        else:
                                            morph_config['size'] = float(v_val)
                                    elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                        morph_config['roi_shrink'] = float(v_val)
                                    elif k_norm in ('uniformlight', 'uniform_light', 'ul'):
                                        morph_config['uniform_light'] = v_val.lower() in ('on', 'true', '1', 'yes')
                                    elif k_norm in ('ulkernelsize', 'ul_kernel', 'ulkernel'):
                                        morph_config['ul_kernel_size'] = int(float(v_val))
                                except ValueError:
                                    pass
                        print(f"DEBUG: Parsed Morph Config from Sheet: {morph_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse Morph config string '{config_str}': {e}")

            print(f"DEBUG: Applying Morphological Filtering ({mode})...")
            
            # Create masked image (Overlay) as input for Morph Top-Hat
            mask_u8_for_input = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            masked_input_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8_for_input)
            
            # Handle Size parameter (Resize before processing)
            size_scale = float(morph_config.get('size', 1.0))
            if size_scale != 1.0:
                h, w = masked_input_img.shape[:2]
                new_w = int(w * size_scale)
                new_h = int(h * size_scale)
                masked_input_img = cv2.resize(masked_input_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_filtered_resized = cv2.resize(mask_u8_for_input, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask_filtered_input = mask_filtered_resized
                print(f"DEBUG: Resized input for Morph Top-Hat by {size_scale} (New size: {new_w}x{new_h})")
            else:
                mask_filtered_input = mask_filtered

            # Apply Morphological Filtering
            mask_filtered_result, _ = apply_morph_tophat(
                masked_input_img, mask_filtered_input, morph_config, out_params, save_dir, image_path, contour_image
            )
            
            # Resize result back to original size if scaled
            if size_scale != 1.0:
                h_orig, w_orig = original_img.shape[:2]
                if mask_filtered_result is not None:
                    mask_u8_res = (mask_filtered_result.astype(np.uint8) * 255) if mask_filtered_result.dtype == bool else mask_filtered_result
                    mask_filtered = cv2.resize(mask_u8_res, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_filtered = None
            else:
                mask_filtered = mask_filtered_result

            save_with_counter(out_params, mask_filtered)

        # --- SLICING (Axis-based Mask Cropping) ---
        elif _norm(method) in ('slicing', 'slice', 'axis', 'axis_slice'):
            # Default Config
            slicing_config = {
                'x_ranges': [],  # List of (min_ratio, max_ratio) tuples for X axis
                'y_ranges': [],  # List of (min_ratio, max_ratio) tuples for Y axis
                'invert': False  # If True, keep outside ranges; if False, keep inside ranges
            }
            
            print(f"DEBUG: Slicing - Attempting to find config...")
            
            # Try to find config in filtering_df if method matches
            if matched_row is not None:
                # Parse config from Method Setting or Value column
                # Format: [X[0:0.1, 0.9:1], Y[0.2:0.8]] or [X=0:0.1,0.9:1, Y=0.2:0.8]
                config_str = None
                if setting_col:
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: Slicing - Trying setting_col '{setting_col}': {config_str}")
                if (config_str is None or str(config_str).strip() == '') and area_val_col:
                    config_str = matched_row.get(area_val_col)
                    print(f"DEBUG: Slicing - Trying area_val_col '{area_val_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present (e.g., [X[0:0.1]] -> X[0:0.1])
                        # Only remove one layer of brackets from start and end
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Parse X and Y ranges
                        # Format examples:
                        # 1. X[0:0.1, 0.9:1]  or  X=0:0.1,0.9:1
                        # 2. Y[0.2:0.8]  or  Y=0.2:0.8
                        
                        import re
                        
                        # Pattern 1: X[...] or Y[...]
                        x_match = re.search(r'X\s*\[\s*([^\]]+)\s*\]', s, re.IGNORECASE)
                        y_match = re.search(r'Y\s*\[\s*([^\]]+)\s*\]', s, re.IGNORECASE)
                        
                        # Pattern 2: X=... or Y=...
                        if not x_match:
                            x_match = re.search(r'X\s*=\s*([\d.:,\s]+?)(?:,|\s+Y\s*|$)', s, re.IGNORECASE)
                        if not y_match:
                            y_match = re.search(r'Y\s*=\s*([\d.:,\s]+?)(?:,|$)', s, re.IGNORECASE)
                        
                        def parse_ranges(match_str):
                            """Parse range string like '0:0.1, 0.9:1' into list of tuples"""
                            ranges = []
                            if match_str:
                                parts = match_str.replace(' ', '').split(',')
                                for part in parts:
                                    if ':' in part:
                                        try:
                                            min_val, max_val = part.split(':')
                                            ranges.append((float(min_val), float(max_val)))
                                        except ValueError:
                                            pass
                            return ranges
                        
                        if x_match:
                            x_ranges_str = x_match.group(1)
                            slicing_config['x_ranges'] = parse_ranges(x_ranges_str)
                            print(f"DEBUG: Slicing - Parsed X ranges: {slicing_config['x_ranges']}")
                        
                        if y_match:
                            y_ranges_str = y_match.group(1)
                            slicing_config['y_ranges'] = parse_ranges(y_ranges_str)
                            print(f"DEBUG: Slicing - Parsed Y ranges: {slicing_config['y_ranges']}")
                        
                        print(f"DEBUG: Parsed Slicing Config: {slicing_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse Slicing config string '{config_str}': {e}")
                        import traceback
                        traceback.print_exc()
            
            # Apply Slicing
            if mask_filtered is not None and (slicing_config['x_ranges'] or slicing_config['y_ranges']):
                print(f"DEBUG: Applying Slicing...")

                h, w = mask_filtered.shape[:2]
                mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered

                # Create masks for X and Y ranges separately
                mask_x = np.zeros_like(mask_u8)
                mask_y = np.zeros_like(mask_u8)

                # Apply X ranges (column-based slicing)
                if slicing_config['x_ranges']:
                    for min_ratio, max_ratio in slicing_config['x_ranges']:
                        x_start = int(w * min_ratio)
                        x_end = int(w * max_ratio)
                        mask_x[:, x_start:x_end] = 255
                        print(f"DEBUG: Slicing - X range [{x_start}:{x_end}] ({min_ratio:.2f}:{max_ratio:.2f})")

                # Apply Y ranges (row-based slicing)
                if slicing_config['y_ranges']:
                    for min_ratio, max_ratio in slicing_config['y_ranges']:
                        y_start = int(h * min_ratio)
                        y_end = int(h * max_ratio)
                        mask_y[y_start:y_end, :] = 255
                        print(f"DEBUG: Slicing - Y range [{y_start}:{y_end}] ({min_ratio:.2f}:{max_ratio:.2f})")

                # Combine X and Y masks: intersection if both specified, union if only one
                if slicing_config['x_ranges'] and slicing_config['y_ranges']:
                    # Both X and Y specified: use intersection (AND)
                    mask_keep = cv2.bitwise_and(mask_x, mask_y)
                    print(f"DEBUG: Slicing - Using INTERSECTION of X and Y ranges")
                elif slicing_config['x_ranges']:
                    # Only X specified
                    mask_keep = mask_x
                else:
                    # Only Y specified
                    mask_keep = mask_y

                # Apply the mask (keep only pixels that are in the mask and in the keep region)
                mask_filtered = cv2.bitwise_and(mask_u8, mask_keep)

                # Also apply slicing to contour_image if available
                # This allows Defect Pct calculation to use sliced contour area
                if contour_image is not None:
                    # Ensure contour_image is uint8
                    if contour_image.dtype == bool:
                        contour_u8 = (contour_image.astype(np.uint8) * 255)
                    else:
                        contour_u8 = contour_image
                    
                    # Resize mask_keep to match contour_image if needed
                    if mask_keep.shape[:2] != contour_u8.shape[:2]:
                        mask_keep_resized = cv2.resize(mask_keep, (contour_u8.shape[1], contour_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_keep_resized = mask_keep
                    
                    # Apply slicing to contour
                    contour_sliced = cv2.bitwise_and(contour_u8, mask_keep_resized)
                    
                    # Update contour_image (preserve original dtype)
                    if contour_image.dtype == bool:
                        contour_image = contour_sliced > 127
                    else:
                        contour_image = contour_sliced
                    
                    contour_pixels_before = cv2.countNonZero(contour_u8) if contour_u8.dtype != bool else np.count_nonzero(contour_u8)
                    contour_pixels_after = cv2.countNonZero(contour_sliced) if contour_sliced.dtype != bool else np.count_nonzero(contour_sliced)
                    print(f"DEBUG: Slicing applied to contour. Contour pixels before: {contour_pixels_before}, after: {contour_pixels_after}")

                print(f"DEBUG: Slicing applied. Mask pixels before: {np.sum(mask_u8 > 0)}, after: {np.sum(mask_filtered > 0)}")
            else:
                if not slicing_config['x_ranges'] and not slicing_config['y_ranges']:
                    print(f"DEBUG: Slicing - No valid ranges found in config. Skipping.")
                else:
                    print(f"DEBUG: Slicing - mask_filtered is None. Skipping.")
            
            save_with_counter(out_params, mask_filtered)

        # --- GLOBAL THRESHOLD FILTERING ---
        elif _norm(method) in ('globalthreshold', 'global_threshold', 'gt'):
            # Default Config
            gt_config = {
                'size': 1.0,
                'threshold': 127,
                'uniform_light': False,
                'ul_kernel_size': 101,
                'gamma': 1.0,
                'contrast': 1.0,
                'denoise': 0,
                'morph_open': 0,
                'morph_close': 0,
                'min_area': 0,
                'roi_shrink': 1.0
            }
            
            # Try to find config in filtering_df if method matches
            print(f"DEBUG: Global Threshold - matched_row is None: {matched_row is None}")
            if matched_row is not None:
                print(f"DEBUG: Global Threshold - Found matched_row. Columns: {matched_row.index.tolist() if hasattr(matched_row, 'index') else 'N/A'}")
                # Parse config from Method Setting or Value column
                # Format: [Size = 1/4, Threshold = 45, Uniform Light = on, Kernel Size = 51, Gamma = 0.3, Contrast = 1.3]
                config_str = None
                if setting_col: 
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: Global Threshold - Trying setting_col '{setting_col}': {config_str}")
                if (config_str is None or str(config_str).strip() == '') and area_val_col:
                    config_str = matched_row.get(area_val_col)
                    print(f"DEBUG: Global Threshold - Trying area_val_col '{area_val_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Smart split: use depth counter to handle nested parentheses
                        parts = []
                        buf = ""
                        depth = 0
                        for char in s:
                            if char == '(': depth += 1
                            elif char == ')': depth -= 1
                            elif char == ',' and depth == 0:
                                parts.append(buf.strip())
                                buf = ""
                                continue
                            buf += char
                        if buf.strip(): parts.append(buf.strip())
                        
                        print(f"DEBUG: Global Threshold - Parsed {len(parts)} parts: {parts}")
                        
                        for p in parts:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('size', 'resize', 'scale'):
                                        if '/' in v_val:
                                            num, den = v_val.split('/')
                                            gt_config['size'] = float(num) / float(den)
                                        else:
                                            gt_config['size'] = float(v_val)
                                    elif k_norm in ('threshold', 'thresh'):
                                        gt_config['threshold'] = int(float(v_val))
                                    elif k_norm in ('uniformlight', 'ul', 'uniform_light'):
                                        gt_config['uniform_light'] = v_val.lower() in ('on', 'true', '1', 'yes')
                                    elif k_norm in ('kernelsize', 'kernalsize', 'ksize', 'kernel', 'kernelsize', 'ulkernelsize', 'ul_kernel', 'ulkernel'):
                                        gt_config['ul_kernel_size'] = int(float(v_val))
                                    elif k_norm in ('gamma',):
                                        gt_config['gamma'] = float(v_val)
                                    elif k_norm in ('contrast',):
                                        gt_config['contrast'] = float(v_val)
                                    elif k_norm in ('denoise', 'blur'):
                                        gt_config['denoise'] = int(float(v_val))
                                    elif k_norm in ('morphopen', 'open'):
                                        gt_config['morph_open'] = int(float(v_val))
                                    elif k_norm in ('morphclose', 'close'):
                                        gt_config['morph_close'] = int(float(v_val))
                                    elif k_norm in ('minarea', 'area', 'min_area'):
                                        gt_config['min_area'] = float(v_val)
                                    elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                        gt_config['roi_shrink'] = float(v_val)
                                except ValueError:
                                    pass
                        print(f"DEBUG: Parsed Global Threshold Config from Sheet: {gt_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse Global Threshold config string '{config_str}': {e}")
            
            print(f"DEBUG: Applying Global Threshold Filtering...")
            
            # Create masked image (Overlay) as input for Global Threshold
            mask_u8_for_input = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            masked_input_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8_for_input)
            
            # Apply Global Threshold Filtering
            mask_filtered, gt_paths = apply_global_threshold_filtering(
                masked_input_img, mask_filtered, gt_config, out_params, save_dir, image_path, contour_image
            )
            extra_results.extend(gt_paths)
            save_with_counter(out_params, mask_filtered)

        # --- BG SUBTRACTION FILTERING ---
        elif _norm(method) in ('bgsubtraction', 'backgroundsubtraction', 'bg_subtraction', 'bg_sub'):
            # Default Config
            bg_config = {
                'block_size': 53,
                'threshold': 13,
                'bg_method': 'Median',  # 'Median' or 'Gaussian'
                'blur': 0,
                'min_area': 0,
                'roi_shrink': 1.0
            }
            
            # Try to find config in filtering_df if method matches
            print(f"DEBUG: BG Subtraction - matched_row is None: {matched_row is None}")
            if matched_row is not None:
                print(f"DEBUG: BG Subtraction - Found matched_row. Columns: {matched_row.index.tolist() if hasattr(matched_row, 'index') else 'N/A'}")
                # Parse config from Method Setting or Value column
                # Format: [Block Size = 53, Threshold = 13, BG Method = Median, Blur = 0]
                config_str = None
                if setting_col: 
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: BG Subtraction - Trying setting_col '{setting_col}': {config_str}")
                if (config_str is None or str(config_str).strip() == '') and area_val_col:
                    config_str = matched_row.get(area_val_col)
                    print(f"DEBUG: BG Subtraction - Trying area_val_col '{area_val_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Parse key=value pairs
                        parts = s.split(',')
                        
                        for p in parts:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('blocksize', 'block_size', 'bsize', 'kernel', 'kernelsize'):
                                        bg_config['block_size'] = int(float(v_val))
                                    elif k_norm in ('threshold', 'thresh'):
                                        bg_config['threshold'] = int(float(v_val))
                                    elif k_norm in ('bgmethod', 'bg_method', 'method', 'backgroundmethod'):
                                        bg_config['bg_method'] = v_val.title()  # 'Median' or 'Gaussian'
                                    elif k_norm in ('blur', 'gaussianblur', 'bluramount'):
                                        bg_config['blur'] = int(float(v_val))
                                    elif k_norm in ('minarea', 'area', 'min_area'):
                                        bg_config['min_area'] = float(v_val)
                                    elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                        bg_config['roi_shrink'] = float(v_val)
                                except ValueError:
                                    pass
                        print(f"DEBUG: Parsed BG Subtraction Config from Sheet: {bg_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse BG Subtraction config string '{config_str}': {e}")
            
            print(f"DEBUG: Applying BG Subtraction Filtering...")
            
            # Create masked image (Overlay) as input for BG Subtraction
            mask_u8_for_input = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            masked_input_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8_for_input)
            
            # Apply BG Subtraction Filtering
            mask_filtered = apply_bg_subtraction_filtering(
                masked_input_img, mask_filtered, bg_config, out_params, save_dir, image_path, contour_image
            )
            save_with_counter(out_params, mask_filtered)

        # --- ODBP FILTERING ---
        elif _norm(method) in ('odbp', 'weighteddiff', 'odbpweighteddiff'):
            # Default Config
            odbp_config = {
                'median_kernel': 23,
                'threshold': 99,
                'alpha': 2.1,
                'beta': -1.2,
                'offset': 15,
                'invert': False,
                'blur': 0,
                'uniform_light': False,
                'ul_kernel_size': 101,
                'min_area': 0,
                'roi_shrink': 1.0
            }
            
            # Try to find config in filtering_df if method matches
            print(f"DEBUG: ODBP - matched_row is None: {matched_row is None}")
            if matched_row is not None:
                print(f"DEBUG: ODBP - Found matched_row. Columns: {matched_row.index.tolist() if hasattr(matched_row, 'index') else 'N/A'}")
                # Parse config from Method Setting or Value column
                # Format: [Median Kernel = 23, Threshold = 99, Alpha = 2.1, Beta = -1.2, Off = 15, Invert=Yes]
                config_str = None
                if setting_col: 
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: ODBP - Trying setting_col '{setting_col}': {config_str}")
                if (config_str is None or str(config_str).strip() == '') and area_val_col:
                    config_str = matched_row.get(area_val_col)
                    print(f"DEBUG: ODBP - Trying area_val_col '{area_val_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Parse key=value pairs
                        parts = s.split(',')
                        
                        for p in parts:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('mediankernel', 'median_kernel', 'kernel', 'kernelsize', 'blocksize'):
                                        odbp_config['median_kernel'] = int(float(v_val))
                                    elif k_norm in ('threshold', 'thresh'):
                                        odbp_config['threshold'] = int(float(v_val))
                                    elif k_norm in ('alpha', 'a'):
                                        odbp_config['alpha'] = float(v_val)
                                    elif k_norm in ('beta', 'b'):
                                        odbp_config['beta'] = float(v_val)
                                    elif k_norm in ('offset', 'off', 'gamma'):
                                        odbp_config['offset'] = float(v_val)
                                    elif k_norm in ('invert', 'inv', 'inverse'):
                                        odbp_config['invert'] = v_val.lower() in ('yes', 'true', '1', 'on', 'y')
                                    elif k_norm in ('blur', 'gaussianblur', 'bluramount'):
                                        odbp_config['blur'] = int(float(v_val))
                                    elif k_norm in ('uniformlight', 'uniform_light', 'ul'):
                                        odbp_config['uniform_light'] = v_val.lower() in ('yes', 'true', '1', 'on', 'y')
                                    elif k_norm in ('ulkernelsize', 'ul_kernel', 'ulkernel', 'illumkernel'):
                                        odbp_config['ul_kernel_size'] = int(float(v_val))
                                    elif k_norm in ('minarea', 'area', 'min_area'):
                                        odbp_config['min_area'] = float(v_val)
                                    elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                        odbp_config['roi_shrink'] = float(v_val)
                                except ValueError:
                                    pass
                        print(f"DEBUG: Parsed ODBP Config from Sheet: {odbp_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse ODBP config string '{config_str}': {e}")
            
            print(f"DEBUG: Applying ODBP Filtering...")
            
            # Create masked image (Overlay) as input for ODBP
            mask_u8_for_input = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
            masked_input_img = cv2.bitwise_and(original_img, original_img, mask=mask_u8_for_input)
            
            # Apply ODBP Filtering
            mask_filtered = apply_odbp_filtering(
                masked_input_img, mask_filtered, odbp_config, out_params, save_dir, image_path, contour_image
            )
            save_with_counter(out_params, mask_filtered)

        # --- BAND-ROI FILTERING ---
        elif _norm(method) in ('bandroi', 'band-roi', 'band_roi'):
            print(f"DEBUG: Applying Band-ROI Filtering...")
            
            # Default config
            band_roi_config = {
                'contour_th': 30,      # 二值化阈值（找轮廓用）
                'defect_th': 10,       # 缺陷灰度阈值
                'x_scale': 1.05,       # X方向缩放因子
                'y_scale': 1.05,       # Y方向缩放因子
                'original_x_scale': 1.0,  # 原始轮廓X方向缩放因子
                'original_y_scale': 1.0,  # 原始轮廓Y方向缩放因子
            }
            
            # Parse config from matched_row
            if matched_row is not None:
                config_str = None
                if setting_col:
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: Band-ROI - Trying setting_col '{setting_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Parse key=value pairs
                        # Format: Contour_TH=30, Defect_TH=10, x_scale=1.05, y_scale=1.1, original[x=1.01, y=1.01]
                        parts = s.split(',')
                        i = 0
                        while i < len(parts):
                            p = parts[i]
                            # Check for nested format: original[x=1.01, y=1.01]
                            if 'original[' in p.lower():
                                # Find the closing bracket
                                nested_str = p
                                while i < len(parts) - 1 and ']' not in nested_str:
                                    i += 1
                                    nested_str += ',' + parts[i]
                                
                                # Extract x and y values from original[x=..., y=...]
                                import re
                                original_match = re.search(r'original\[x=(\d+\.?\d*),\s*y=(\d+\.?\d*)\]', nested_str.lower())
                                if original_match:
                                    band_roi_config['original_x_scale'] = float(original_match.group(1))
                                    band_roi_config['original_y_scale'] = float(original_match.group(2))
                                    print(f"DEBUG: Parsed original scale: x={band_roi_config['original_x_scale']}, y={band_roi_config['original_y_scale']}")
                            elif '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('contourth', 'contour_th', 'threshold'):
                                        band_roi_config['contour_th'] = int(float(v_val))
                                    elif k_norm in ('defectth', 'defect_th', 'defectthreshold'):
                                        band_roi_config['defect_th'] = int(float(v_val))
                                    elif k_norm in ('xscale', 'x_scale'):
                                        band_roi_config['x_scale'] = float(v_val)
                                    elif k_norm in ('yscale', 'y_scale'):
                                        band_roi_config['y_scale'] = float(v_val)
                                except ValueError:
                                    pass
                            i += 1
                        print(f"DEBUG: Parsed Band-ROI Config: {band_roi_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse Band-ROI config string '{config_str}': {e}")
            
            # Also check if key contains parameters (for inline config like Band-ROI(Contour_TH=30))
            if key:
                try:
                    s = str(key).strip()
                    if '=' in s:
                        parts = s.split(',')
                        for p in parts:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                v_val = v.strip()
                                try:
                                    if k_norm in ('contourth', 'contour_th'):
                                        band_roi_config['contour_th'] = int(float(v_val))
                                    elif k_norm in ('defectth', 'defect_th'):
                                        band_roi_config['defect_th'] = int(float(v_val))
                                    elif k_norm in ('xscale', 'x_scale'):
                                        band_roi_config['x_scale'] = float(v_val)
                                    elif k_norm in ('yscale', 'y_scale'):
                                        band_roi_config['y_scale'] = float(v_val)
                                except ValueError:
                                    pass
                        print(f"DEBUG: Parsed Band-ROI Config from key: {band_roi_config}")
                except Exception as e:
                    print(f"Warning: Failed to parse Band-ROI config from key '{key}': {e}")
            
            # Apply Band-ROI Filtering
            mask_filtered = apply_band_roi_filtering(
                original_img, mask_filtered, band_roi_config, 
                out_params, save_dir, image_path, contour_image
            )
            save_with_counter(out_params, mask_filtered)

        # --- STENCIL-ROI FILTERING ---
        elif _norm(method) in ('stencilroi', 'stencil-roi', 'stencil_roi'):
            print(f"DEBUG: Applying Stencil-ROI Filtering...")
            
            # Default config
            stencil_roi_config = {
                'contour_th': 30,           # 二值化阈值（找轮廓用）
                'stencil_path': 'Pics/stencil/K700 stencil.jpg',  # 默认stencil路径
                'scale': 1.1,               # stencil缩放比例（等比例，向后兼容）
                'x_scale': None,            # X方向缩放比例（非等比例，优先于scale）
                'y_scale': None,            # Y方向缩放比例（非等比例，优先于scale）
                'alignment': 'bottom_aligned+X_centered',  # 对齐方式
                'original_x_scale': 1.0,    # 原始轮廓X方向缩放因子
                'original_y_scale': 1.0,    # 原始轮廓Y方向缩放因子
            }
            
            # Parse config from matched_row
            if matched_row is not None:
                config_str = None
                if setting_col:
                    config_str = matched_row.get(setting_col)
                    print(f"DEBUG: Stencil-ROI - Trying setting_col '{setting_col}': {config_str}")
                
                if config_str:
                    try:
                        s = str(config_str).strip()
                        # Remove outer brackets if present
                        if s.startswith('[') and s.endswith(']'):
                            s = s[1:-1]
                        
                        # Parse parameters
                        i = 0
                        while i < len(s):
                            # Skip whitespace and commas
                            while i < len(s) and s[i] in ' \t,':
                                i += 1
                            if i >= len(s):
                                break
                            
                            # Check for original[...] syntax
                            if s[i:].lower().startswith('original['):
                                match = re.search(r'original\[x=(\d+\.?\d*),\s*y=(\d+\.?\d*)\]', s[i:].lower())
                                if match:
                                    stencil_roi_config['original_x_scale'] = float(match.group(1))
                                    stencil_roi_config['original_y_scale'] = float(match.group(2))
                                    i += match.end()
                                else:
                                    i += 1
                            elif '=' in s[i:]:
                                # Find the key
                                eq_pos = s.find('=', i)
                                if eq_pos == -1:
                                    break
                                k = s[i:eq_pos].strip()
                                i = eq_pos + 1
                                
                                # Find the value (handle quotes)
                                v_val = ''
                                if i < len(s) and s[i] == "'":
                                    i += 1
                                    end_quote = s.find("'", i)
                                    if end_quote != -1:
                                        v_val = s[i:end_quote]
                                        i = end_quote + 1
                                elif i < len(s) and s[i] == '"':
                                    i += 1
                                    end_quote = s.find('"', i)
                                    if end_quote != -1:
                                        v_val = s[i:end_quote]
                                        i = end_quote + 1
                                else:
                                    # Read until comma or end
                                    end_pos = s.find(',', i)
                                    if end_pos == -1:
                                        end_pos = len(s)
                                    v_val = s[i:end_pos].strip()
                                    i = end_pos
                                
                                # Parse key-value pair
                                k_norm = k.lower().replace('_', '').replace('-', '').replace(' ', '')
                                try:
                                    if k_norm in ('contourth', 'contour_th', 'threshold'):
                                        stencil_roi_config['contour_th'] = int(float(v_val))
                                    elif k_norm in ('stencil', 'stencilpath', 'stencil_path'):
                                        # Remove surrounding quotes if present
                                        v_val_clean = v_val.strip("'\"")
                                        stencil_roi_config['stencil_path'] = v_val_clean
                                    elif k_norm in ('scale', 'stencilscale', 'stencil_scale'):
                                        stencil_roi_config['scale'] = float(v_val)
                                    elif k_norm in ('xscale', 'x_scale'):
                                        stencil_roi_config['x_scale'] = float(v_val)
                                    elif k_norm in ('yscale', 'y_scale'):
                                        stencil_roi_config['y_scale'] = float(v_val)
                                    elif k_norm in ('alignment', 'align'):
                                        stencil_roi_config['alignment'] = v_val
                                    elif k_norm in ('originalx', 'original_x'):
                                        stencil_roi_config['original_x_scale'] = float(v_val)
                                    elif k_norm in ('originaly', 'original_y'):
                                        stencil_roi_config['original_y_scale'] = float(v_val)
                                except ValueError:
                                    pass
                            else:
                                i += 1
                        print(f"DEBUG: Parsed Stencil-ROI Config: {stencil_roi_config}")
                    except Exception as e:
                        print(f"Warning: Failed to parse Stencil-ROI config string '{config_str}': {e}")
            
            # Apply Stencil-ROI Filtering
            mask_filtered = apply_stencil_roi_filtering(
                original_img, mask_filtered, stencil_roi_config,
                out_params, save_dir, image_path, contour_image
            )
            save_with_counter(out_params, mask_filtered)

        else:
            print(f"Warning: Unknown filtering method '{method}'")
            
        # No manual update of current_previous_op needed anymore, handled via op_context injection

    return mask_filtered, extra_results, area_filter_mask_to_remove, adaptive_gaussian_mask_to_keep, contour_image


def apply_band_roi_filtering(original_img, mask_filtered, band_roi_config, out_params, save_dir, image_path, contour_image=None):
    """
    Apply Band-ROI filtering: Find the largest contour, scale it, and detect defects in the ring region.
    
    Args:
        original_img: RGB image
        mask_filtered: Current mask (used as base/DUT mask)
        band_roi_config: Dict with 'contour_th', 'defect_th', 'x_scale', 'y_scale'
        out_params: Output parameters (e.g., ['Pic', 'Save'])
        save_dir: Directory to save outputs
        image_path: Path to original image
        contour_image: Optional contour image for reference
    
    Returns:
        Updated mask with defects detected in the ring region
    """
    import os
    import cv2
    import numpy as np
    
    print(f"DEBUG: Band-ROI Config: {band_roi_config}")
    
    # Extract parameters
    contour_th = band_roi_config.get('contour_th', 30)
    defect_th = band_roi_config.get('defect_th', 10)
    x_scale = band_roi_config.get('x_scale', 1.05)
    y_scale = band_roi_config.get('y_scale', 1.05)
    original_x_scale = band_roi_config.get('original_x_scale', 1.0)
    original_y_scale = band_roi_config.get('original_y_scale', 1.0)
    
    h, w = original_img.shape[:2]
    
    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_img
    
    # Find contours using grayscale thresholding (same as contour analysis.py)
    # This finds the actual object (screen) rather than using the full image boundary
    _, mask_base = cv2.threshold(gray, contour_th, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded mask
    contours, _ = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"DEBUG: Band-ROI - Found {len(contours)} contours after thresholding (threshold={contour_th})")
    
    if not contours:
        print("DEBUG: Band-ROI - No contours found in base mask")
        return mask_filtered
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    print(f"DEBUG: Band-ROI - Largest contour: area={largest_area}, bbox=({x},{y},{bw},{bh}), image_size=({w},{h})")
    print(f"DEBUG: Band-ROI - Contour coverage: {(largest_area/(w*h)*100):.1f}% of image")
    
    # Scale the contour
    def scale_contour(contour, img_shape, x_s, y_s):
        """Scale contour non-uniformly based on bounding box center"""
        x, y, bw, bh = cv2.boundingRect(contour)
        cx = x + bw / 2
        cy = y + bh / 2
        
        scaled = contour.copy().astype(np.float32)
        scaled[:, 0, 0] = cx + (scaled[:, 0, 0] - cx) * x_s
        scaled[:, 0, 1] = cy + (scaled[:, 0, 1] - cy) * y_s
        
        # Clip to image bounds
        ih, iw = img_shape[:2]
        scaled[:, 0, 0] = np.clip(scaled[:, 0, 0], 0, iw - 1)
        scaled[:, 0, 1] = np.clip(scaled[:, 0, 1], 0, ih - 1)
        
        return scaled.astype(np.int32)
    
    # Scale the original contour first (if original scale is not 1.0)
    if original_x_scale != 1.0 or original_y_scale != 1.0:
        original_scaled_contour = scale_contour(largest_contour, original_img.shape, original_x_scale, original_y_scale)
        print(f"DEBUG: Band-ROI - Applied original scale: x={original_x_scale}, y={original_y_scale}")
    else:
        original_scaled_contour = largest_contour
    
    # Then scale the already-scaled contour for the outer boundary
    scaled_contour = scale_contour(original_scaled_contour, original_img.shape, x_scale, y_scale)
    
    # Create inner and outer masks
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(mask_inner, [original_scaled_contour], 255)
    cv2.fillPoly(mask_outer, [scaled_contour], 255)
    
    # Ring region = outer - inner
    ring_mask = cv2.subtract(mask_outer, mask_inner)
    
    # Use entire ring region (no gray level filtering)
    defect_mask = (ring_mask > 0)
    defect_mask_u8 = defect_mask.astype(np.uint8) * 255
    
    # Count defect pixels
    defect_pixels = np.sum(defect_mask)
    ring_pixels = np.sum(ring_mask > 0)
    ratio = (defect_pixels / ring_pixels * 100) if ring_pixels > 0 else 0
    
    print(f"DEBUG: Band-ROI - Ring pixels: {ring_pixels}, Defect pixels: {defect_pixels}, Ratio: {ratio:.2f}%")
    
    # Save visualization if requested
    # Check if any param starts with 'Pic' or 'Save' (handles cases like 'Save(Overlay+Contour, Complement_Overlay)')
    has_pic = any(str(p).strip().lower().startswith('pic') for p in out_params) if out_params else False
    has_save = any(str(p).strip().lower().startswith('save') for p in out_params) if out_params else False
    if has_pic or has_save:
        # Create visualization
        vis_img = original_img.copy()
        
        # Draw contours
        cv2.drawContours(vis_img, [largest_contour], -1, (128, 128, 128), 2)  # Gray - original (before any scaling)
        cv2.drawContours(vis_img, [original_scaled_contour], -1, (255, 0, 0), 3)  # Red - after original scale
        cv2.drawContours(vis_img, [scaled_contour], -1, (0, 255, 0), 3)   # Green - after x_scale/y_scale
        
        # Highlight defect pixels
        vis_img[defect_mask] = vis_img[defect_mask] * 0.5 + np.array([0, 0, 255]) * 0.5  # Semi-transparent red
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(save_dir, f"{base_name}_band_roi_vis.png")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"DEBUG: Band-ROI - Saved visualization to {vis_path}")
        
        # Save defect mask
        mask_path = os.path.join(save_dir, f"{base_name}_band_roi_mask.png")
        cv2.imwrite(mask_path, defect_mask_u8)
        print(f"DEBUG: Band-ROI - Saved defect mask to {mask_path}")
        
        # --- NEW: Create Banded_ROI.jpg with ring region highlighted at 50% transparency ---
        # Create overlay with ring region highlighted
        banded_roi_img = original_img.copy()
        
        # Create a colored overlay for the ring region (e.g., yellow with 50% transparency)
        overlay = np.zeros_like(banded_roi_img)
        overlay[ring_mask > 0] = [255, 255, 0]  # Yellow color for ring region
        
        # Blend original image with overlay at 50% transparency
        alpha = 0.5
        banded_roi_img = cv2.addWeighted(banded_roi_img, 1 - alpha, overlay, alpha, 0)
        
        # Draw contours on top
        cv2.drawContours(banded_roi_img, [largest_contour], -1, (128, 128, 128), 2)  # Gray - original (before any scaling)
        cv2.drawContours(banded_roi_img, [original_scaled_contour], -1, (255, 0, 0), 3)  # Red - after original scale
        cv2.drawContours(banded_roi_img, [scaled_contour], -1, (0, 255, 0), 3)   # Green - after x_scale/y_scale
        
        # Save Banded_ROI.jpg
        banded_roi_path = os.path.join(save_dir, "Banded_ROI.jpg")
        cv2.imwrite(banded_roi_path, cv2.cvtColor(banded_roi_img, cv2.COLOR_RGB2BGR))
        print(f"DEBUG: Band-ROI - Saved Banded_ROI.jpg to {banded_roi_path}")
    
    # Return the defect mask combined with the input mask
    # The result is the intersection of the input mask (mask_filtered) and the defect mask in ring region
    # This ensures Band-ROI respects the input mask boundary (e.g., Contour_Corrected)
    if mask_filtered is not None and mask_filtered.shape[:2] == defect_mask_u8.shape[:2]:
        # Convert mask_filtered to uint8 if needed
        if mask_filtered.dtype == bool:
            mask_filtered_uint8 = (mask_filtered.astype(np.uint8) * 255)
        else:
            mask_filtered_uint8 = mask_filtered
        result_mask = cv2.bitwise_and(mask_filtered_uint8, defect_mask_u8)
        print(f"DEBUG: Band-ROI - Intersected with input mask. Result: {cv2.countNonZero(result_mask)} pixels")
    else:
        # Fallback: use defect_mask_u8 directly if input mask is not available or wrong size
        result_mask = defect_mask_u8
        print(f"DEBUG: Band-ROI - No valid input mask, using defect mask directly: {cv2.countNonZero(result_mask)} pixels")
    
    return result_mask


def apply_stencil_roi_filtering(original_img, mask_filtered, stencil_roi_config, out_params, save_dir, image_path, contour_image=None):
    """
    Apply Stencil-ROI filtering: Align DUT with stencil template and return ring region mask.
    
    Args:
        original_img: RGB image
        mask_filtered: Current mask (used as base/DUT mask)
        stencil_roi_config: Dict with 'contour_th', 'stencil_path', 'scale', 'alignment', 'original_x_scale', 'original_y_scale'
        out_params: Output parameters (e.g., ['Pic', 'Save'])
        save_dir: Directory to save outputs
        image_path: Path to original image
        contour_image: Optional contour image for reference
    
    Returns:
        Ring region mask between DUT and stencil contours
    """
    import os
    import cv2
    import numpy as np
    import re
    
    print(f"DEBUG: Stencil-ROI Config: {stencil_roi_config}")
    
    # Extract parameters
    debug_info = {
        'dut_original_area': None,
        'stencil_original_area': None,
        'stencil_scaled_area': None,
        'stencil_final_area': None,
        'ring_pixels': None
    }
    contour_th = stencil_roi_config.get('contour_th', 30)
    stencil_path = stencil_roi_config.get('stencil_path', 'Pics/stencil/K700 stencil.jpg')
    # Support both scale (legacy, equal) and x_scale/y_scale (non-equal scaling)
    x_scale_val = stencil_roi_config.get('x_scale')
    y_scale_val = stencil_roi_config.get('y_scale')
    if x_scale_val is not None and y_scale_val is not None:
        x_scale = float(x_scale_val)
        y_scale = float(y_scale_val)
        print(f"DEBUG: Stencil-ROI - Using non-equal scaling: x_scale={x_scale}, y_scale={y_scale}")
    else:
        scale = stencil_roi_config.get('scale', 1.1)
        x_scale = scale
        y_scale = scale
        print(f"DEBUG: Stencil-ROI - Using equal scaling: scale={scale}")
    alignment = stencil_roi_config.get('alignment', 'bottom_aligned+X_centered')
    original_x_scale = stencil_roi_config.get('original_x_scale', 1.0)
    original_y_scale = stencil_roi_config.get('original_y_scale', 1.0)
    
    h, w = original_img.shape[:2]
    
    # Convert to grayscale for contour detection
    # Note: original_img is in BGR format (from cv2.imread)
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_img
    
    # === Step 1: Find DUT contour ===
    _, binary = cv2.threshold(gray, contour_th, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("DEBUG: Stencil-ROI - No DUT contours found")
        return mask_filtered
    
    dut_contour = max(contours, key=cv2.contourArea)
    print(f"DEBUG: Stencil-ROI - Found DUT contour with area: {cv2.contourArea(dut_contour)}")
    
    # === Step 2: Load Stencil and find its contour ===
    if not os.path.exists(stencil_path):
        # Try relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stencil_path_alt = os.path.join(base_dir, stencil_path)
        if os.path.exists(stencil_path_alt):
            stencil_path = stencil_path_alt
        else:
            print(f"DEBUG: Stencil-ROI - Stencil not found: {stencil_path}")
            return mask_filtered
    
    stencil_img = cv2.imread(stencil_path)
    if stencil_img is None:
        print(f"DEBUG: Stencil-ROI - Failed to load stencil: {stencil_path}")
        return mask_filtered
    
    stencil_gray = cv2.cvtColor(stencil_img, cv2.COLOR_BGR2GRAY)
    _, stencil_binary = cv2.threshold(stencil_gray, 100, 255, cv2.THRESH_BINARY)
    stencil_contours, _ = cv2.findContours(stencil_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not stencil_contours:
        print("DEBUG: Stencil-ROI - No stencil contours found")
        return mask_filtered
    
    stencil_contour = max(stencil_contours, key=cv2.contourArea)
    print(f"DEBUG: Stencil-ROI - Found stencil contour with area: {cv2.contourArea(stencil_contour)}")
    
    # === Step 3: Get rotated bbox info for both contours ===
    def get_rotated_bbox_info(contour):
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        width, height = size
        rotation_angle = angle
        if width < height:
            rotation_angle = angle + 90
            width, height = height, width
        rect_horizontal = (center, (width, height), rotation_angle)
        bbox_points = cv2.boxPoints(rect_horizontal)
        return rect_horizontal, bbox_points, rotation_angle
    
    def rotate_contour_to_horizontal(contour, angle, center):
        rotation_angle = -angle
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        contour_reshaped = contour.reshape(-1, 2).astype(np.float32)
        contour_homogeneous = np.hstack([contour_reshaped, np.ones((contour_reshaped.shape[0], 1))])
        rotated_points = M @ contour_homogeneous.T
        rotated_contour = rotated_points.T.reshape(-1, 1, 2).astype(np.int32)
        return rotated_contour
    
    dut_rect, dut_bbox, dut_angle = get_rotated_bbox_info(dut_contour)
    stencil_rect, stencil_bbox, stencil_angle = get_rotated_bbox_info(stencil_contour)
    
    print(f"DEBUG: Stencil-ROI - DUT angle: {dut_angle:.2f}, Stencil angle: {stencil_angle:.2f}")
    
    # === Step 4: Rotate contours to horizontal ===
    dut_center = (int(dut_rect[0][0]), int(dut_rect[0][1]))
    stencil_center = (int(stencil_rect[0][0]), int(stencil_rect[0][1]))
    
    dut_contour_rotated = rotate_contour_to_horizontal(dut_contour, dut_angle, dut_center)
    
    # Flip DUT vertically to match stencil orientation
    dut_contour_rotated = dut_contour_rotated.astype(np.float32)
    dut_contour_rotated[:, 0, 1] = 2 * dut_center[1] - dut_contour_rotated[:, 0, 1]
    dut_contour_rotated = dut_contour_rotated.astype(np.int32)
    
    angle_diff = dut_angle - stencil_angle
    
    # Handle 180-degree equivalence (0° and 180° are the same for rectangles)
    # Normalize angle_diff to [-180, 180] range
    angle_diff = ((angle_diff + 180) % 360) - 180
    
    # If angle difference is close to 180° or -180°, treat it as 0° (equivalent directions)
    if abs(angle_diff) > 179:
        print(f"DEBUG: Stencil-ROI - Angle diff {angle_diff:.2f}° is near 180°, treating as 0° (180° equivalence)")
        angle_diff = 0.0
    
    print(f"DEBUG: Stencil-ROI - Angle difference: {angle_diff:.2f}°")
    stencil_contour_rotated = rotate_contour_to_horizontal(stencil_contour, angle_diff, stencil_center)
    
    # === Step 5: Scale and normalize stencil ===
    # Normalize stencil to DUT size first (because template and photo have different resolutions)
    # Then apply x_scale and y_scale (support non-equal scaling)
    stencil_h, stencil_w = stencil_img.shape[:2]
    normalize_scale = w / stencil_w  # Normalize based on width ratio
    total_x_scale = normalize_scale * x_scale
    total_y_scale = normalize_scale * y_scale
    
    # Scale stencil around its center
    M = cv2.moments(stencil_contour_rotated)
    if M["m00"] != 0:
        stencil_cx = int(M["m10"] / M["m00"])
        stencil_cy = int(M["m01"] / M["m00"])
    else:
        stencil_cx, stencil_cy = stencil_w // 2, stencil_h // 2
    
    stencil_contour_scaled = stencil_contour_rotated.astype(np.float32)
    # Apply non-equal scaling: x and y separately
    stencil_contour_scaled[:, 0, 0] = stencil_cx + (stencil_contour_scaled[:, 0, 0] - stencil_cx) * total_x_scale
    stencil_contour_scaled[:, 0, 1] = stencil_cy + (stencil_contour_scaled[:, 0, 1] - stencil_cy) * total_y_scale
    stencil_contour_scaled = stencil_contour_scaled.astype(np.int32)
    
    print(f"DEBUG: Stencil-ROI - Normalize scale: {normalize_scale:.2f}, Final x_scale: {total_x_scale:.2f}, Final y_scale: {total_y_scale:.2f}")
    
    # === Step 6: Align contours ===
    if alignment == 'bottom_aligned+X_centered':
        # Use axis-aligned bounding box for alignment
        dut_x, dut_y, dut_w, dut_h = cv2.boundingRect(dut_contour_rotated)
        dut_bottom_y = dut_y + dut_h
        dut_center_x = dut_x + dut_w / 2
        
        stencil_x, stencil_y, stencil_w, stencil_h = cv2.boundingRect(stencil_contour_scaled)
        stencil_bottom_y = stencil_y + stencil_h
        stencil_center_x = stencil_x + stencil_w / 2
        
        dy = dut_bottom_y - stencil_bottom_y
        dx = dut_center_x - stencil_center_x
        
        stencil_contour_final = stencil_contour_scaled.copy()
        stencil_contour_final[:, 0, 0] = stencil_contour_final[:, 0, 0] + dx
        stencil_contour_final[:, 0, 1] = stencil_contour_final[:, 0, 1] + dy
        stencil_contour_final = stencil_contour_final.astype(np.int32)
        
        # Verify and adjust alignment
        _, stencil_y_check, _, stencil_h_check = cv2.boundingRect(stencil_contour_final)
        stencil_bottom_check = stencil_y_check + stencil_h_check
        if stencil_bottom_check != dut_bottom_y:
            dy_adjust = dut_bottom_y - stencil_bottom_check
            stencil_contour_final[:, 0, 1] += dy_adjust
    else:
        stencil_contour_final = stencil_contour_scaled
    
    # === Step 7: Create ring mask ===
    # Use original contour areas for consistency with mask
    dut_area = cv2.contourArea(dut_contour)
    stencil_area = cv2.contourArea(stencil_contour_final)
    
    print(f"DEBUG: Stencil-ROI - DUT area: {dut_area:.1f}, Stencil area: {stencil_area:.1f}")
    
    # Create masks for DUT and Stencil
    # Use original dut_contour for mask to align with visualization on original_img
    mask_dut = np.zeros((h, w), dtype=np.uint8)
    mask_stencil = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_dut, [dut_contour], 255)  # Use original contour
    cv2.fillPoly(mask_stencil, [stencil_contour_final], 255)
    
    # Stencil ROI should NOT include DUT region
    # Only keep the part of Stencil that is outside DUT (Stencil - DUT)
    if stencil_area > dut_area:
        # Stencil is larger than DUT: ROI = Stencil - DUT (outer ring only)
        ring_mask = cv2.subtract(mask_stencil, mask_dut)
        print(f"DEBUG: Stencil-ROI - Stencil is OUTER, DUT is INNER. ROI = Stencil - DUT")
    else:
        # Stencil is smaller or equal to DUT: Stencil is inside DUT, no valid ROI
        # Return empty mask as ROI cannot include DUT region
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        print(f"DEBUG: Stencil-ROI - Stencil is INSIDE DUT. No valid ROI (empty mask returned)")
    
    # === Step 8: Combine with input mask ===
    if mask_filtered is not None and mask_filtered.shape[:2] == ring_mask.shape[:2]:
        if mask_filtered.dtype == bool:
            mask_filtered_uint8 = (mask_filtered.astype(np.uint8) * 255)
        else:
            mask_filtered_uint8 = mask_filtered
        result_mask = cv2.bitwise_and(mask_filtered_uint8, ring_mask)
    else:
        result_mask = ring_mask
    
    # === Step 9: Save visualization if requested ===
    has_pic = any(str(p).strip().lower().startswith('pic') for p in out_params) if out_params else False
    has_save = any(str(p).strip().lower().startswith('save') for p in out_params) if out_params else False
    
    # Always save visualization for Stencil-ROI regardless of pic/save parameters
    vis_img = original_img.copy()
    
    # Draw contours - use original contours for visualization on original image
    # DUT: use original dut_contour (detected from original_img)
    cv2.drawContours(vis_img, [dut_contour], -1, (255, 0, 0), 3)  # Blue - DUT (original)
    # Stencil: use stencil_contour_final (aligned to image coordinates)
    cv2.drawContours(vis_img, [stencil_contour_final], -1, (0, 255, 0), 3)  # Green - Stencil
    
    # Highlight ring region
    overlay = np.zeros_like(vis_img)
    overlay[ring_mask > 0] = [255, 255, 0]  # Yellow
    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(save_dir, f"{base_name}_stencil_roi_vis.jpg")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    print(f"DEBUG: Stencil-ROI - Saved visualization to {vis_path}")
    
    if has_pic or has_save:
        
        # Save mask
        mask_path = os.path.join(save_dir, f"{base_name}_stencil_roi_mask.jpg")
        cv2.imwrite(mask_path, ring_mask)
        print(f"DEBUG: Stencil-ROI - Saved mask to {mask_path}, shape={ring_mask.shape}, dtype={ring_mask.dtype}, pixels={np.sum(ring_mask > 0)}")
        
        # Save Stencil_ROI.jpg as color image with DUT contour (red) and ROI region (green/yellow tint)
        stencil_roi_path = os.path.join(save_dir, "Stencil_ROI.jpg")
        # Convert mask to BGR color image
        ring_mask_color = cv2.cvtColor(ring_mask, cv2.COLOR_GRAY2BGR)
        # Create a colored overlay for ROI region (light green/yellow)
        roi_overlay = np.zeros_like(ring_mask_color)
        roi_overlay[ring_mask > 0] = [0, 255, 255]  # Yellow/Cyan for ROI
        # Blend ROI overlay with mask
        ring_mask_color = cv2.addWeighted(ring_mask_color, 0.5, roi_overlay, 0.5, 0)
        # Draw Stencil contour in green (BGR: [0, 255, 0]) - draw first so DUT is on top
        cv2.drawContours(ring_mask_color, [stencil_contour_final], -1, (0, 255, 0), 3)
        # Draw DUT contour in red (BGR: [0, 0, 255]) - draw last so it's on top
        cv2.drawContours(ring_mask_color, [dut_contour_rotated], -1, (0, 0, 255), 3)
        cv2.imwrite(stencil_roi_path, ring_mask_color)
        print(f"DEBUG: Stencil-ROI - Saved Stencil_ROI.png with DUT contour (red) to {stencil_roi_path}, shape={ring_mask_color.shape}, pixels={np.sum(ring_mask > 0)}")
        
        # Save Contour_Debug.jpg - overlay with ROI region and DUT contour (red)
        if mask_filtered is not None:
            contour_debug_path = os.path.join(save_dir, f"{base_name}_contour_debug.jpg")
            # Create overlay like save_overlay: original_img masked by ring_mask
            contour_debug_img = cv2.bitwise_and(original_img, original_img, mask=ring_mask)
            contour_debug_img = cv2.cvtColor(contour_debug_img, cv2.COLOR_RGB2BGR)
            # Draw DUT contour in red (BGR: [0, 0, 255])
            # Need to use original dut_contour, not dut_contour_rotated, since original_img is not rotated
            cv2.drawContours(contour_debug_img, [dut_contour], -1, (0, 0, 255), 3)
            cv2.imwrite(contour_debug_path, contour_debug_img)
            print(f"DEBUG: Stencil-ROI - Saved Contour_Debug.png with DUT contour (red) to {contour_debug_path}")
    
    print(f"DEBUG: Stencil-ROI - Ring region pixels: {np.sum(ring_mask > 0)}")
    return result_mask


def parse_filtering_params(params):
    """
    Parses a list of parameters (from Filtering[...]) into structured operations.
    Example: ['Area(5)', 'Pic', 'Save', 'RGB'] 
    -> [{'method': 'Area', 'key': '5', 'output_params': ['Pic', 'Save']}, {'method': 'RGB', 'key': '', 'output_params': []}]
    """
    print(f"DEBUG: parse_filtering_params called with params: {params}, type: {type(params)}")
    operations = []
    current_op = None
    
    # Methods that start a new operation group
    known_methods = [
        'area', 'areafiltering',
        'rgb', 'rgbfiltering',
        'colordistance', 'color_distance', 'colordistancefiltering',  # Add Color Distance support
        'relativity', 'relativityfiltering',
        'adaptivegaussian', 'adaptive_gaussian', 'ag',
        'hsv', 'hsvfiltering',
        'shape', 'shapefiltering',
        'morphtophat', 'morphblackhat', 'morph_tophat', 'morph_blackhat',
        'globalthreshold', 'global_threshold', 'gt',
        'slicing', 'slice', 'axis', 'axis_slice',
        'bandroi', 'band_roi', 'band-roi', 'bandroi',  # Add Band-ROI support
        'stencilroi', 'stencil_roi', 'stencil-roi',  # Add Stencil-ROI support
        'bgsubtraction', 'backgroundsubtraction', 'bg_subtraction', 'bg_sub',  # Add BG Subtraction support
        'odbp', 'weighteddiff', 'odbpweighteddiff'  # Add ODBP support
    ]

    # Helper to clean string
    def clean(s): return str(s).strip()
    def norm(s): return clean(s).lower().replace('_', '').replace(' ', '').replace('-', '')

    for p in params:
        p_str = clean(p)
        if not p_str:
            continue
        
        # Check if it looks like Method(Key)
        method_candidate = p_str
        key_candidate = ""
        
        if '(' in p_str and p_str.endswith(')'):
            parts = p_str.split('(', 1)
            method_candidate = parts[0].strip()
            key_candidate = parts[1][:-1].strip() # Remove trailing )
        
        norm_method = norm(method_candidate)
        is_known_method = norm_method in known_methods
        
        # Special handling for Morph Top-Hat/Black-Hat
        if not is_known_method and ('morphtophat' in norm_method or 'morphblackhat' in norm_method):
            is_known_method = True
            # Normalize method name
            if 'morphtophat' in norm_method:
                method_candidate = 'Morph Top-Hat'
            elif 'morphblackhat' in norm_method:
                method_candidate = 'Morph Black-Hat'
        
        # Special handling for Global Threshold
        if not is_known_method and 'globalthreshold' in norm_method:
            is_known_method = True
            method_candidate = 'Global Threshold'
        
        # Special handling for BG Subtraction
        if not is_known_method and ('bgsubtraction' in norm_method or 'backgroundsubtraction' in norm_method):
            is_known_method = True
            method_candidate = 'BG Subtraction'
        
        # Special handling for ODBP
        if not is_known_method and ('odbp' in norm_method or 'weighteddiff' in norm_method):
            is_known_method = True
            method_candidate = 'ODBP'
        
        if is_known_method:
            # Start new op
            current_op = {
                'method': method_candidate, 
                'key': key_candidate, 
                'output_params': []
            }
            operations.append(current_op)
        else:
            # It's a parameter (e.g. Pic, Save)
            # Add to current op if exists
            if current_op:
                current_op['output_params'].append(p_str)
            else:
                # Orphan param? Maybe default to Area if none specified?
                # Or ignore.
                pass
                
    return operations


def apply_bg_subtraction_filtering(original_img, mask_filtered, bg_config, out_params, save_dir, image_path, contour_image=None):
    """
    Apply Background Subtraction filtering to detect defects by subtracting estimated background.
    
    Args:
        original_img: RGB image
        mask_filtered: Current mask (used as base/DUT mask)
        bg_config: Dict with 'block_size', 'threshold', 'bg_method', 'blur', 'min_area', 'roi_shrink'
        out_params: Output parameters (e.g., ['Pic', 'Save'])
        save_dir: Directory to save outputs
        image_path: Path to original image
        contour_image: Optional contour image for reference
    
    Returns:
        Updated mask with defects detected via background subtraction
    """
    import cv2
    import numpy as np
    import os
    
    print(f"DEBUG: BG Subtraction Config: {bg_config}")
    
    # Extract parameters
    block_size = bg_config.get('block_size', 53)
    threshold = bg_config.get('threshold', 13)
    bg_method = bg_config.get('bg_method', 'Median')  # 'Median' or 'Gaussian'
    blur_amount = bg_config.get('blur', 0)
    min_area = bg_config.get('min_area', 0)
    roi_shrink = bg_config.get('roi_shrink', 1.0)
    
    # Ensure block_size is odd and at least 3
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    
    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_img.copy()
    
    # Apply optional pre-blur
    if blur_amount > 0:
        ksize = blur_amount * 2 + 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        print(f"DEBUG: BG Subtraction - Applied Gaussian blur with kernel size {ksize}")
    
    # Apply ROI mask if provided
    if mask_filtered is not None:
        mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
        # Ensure same size
        if mask_u8.shape[:2] != gray.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Apply mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask_u8)
    else:
        gray_masked = gray
        mask_u8 = np.ones_like(gray) * 255
    
    # Estimate background
    if bg_method.lower() == 'median':
        # Use Median Blur (Robust to salt-and-pepper noise)
        bg = cv2.medianBlur(gray_masked, block_size)
        print(f"DEBUG: BG Subtraction - Estimated background using Median filter (block_size={block_size})")
    else:
        # Use Gaussian Blur (Default)
        bg = cv2.GaussianBlur(gray_masked, (block_size, block_size), 0)
        print(f"DEBUG: BG Subtraction - Estimated background using Gaussian filter (block_size={block_size})")
    
    # Calculate absolute difference: |Background - Original|
    # This detects both dark defects on bright BG and bright defects on dark BG
    diff = cv2.absdiff(bg, gray_masked)
    print(f"DEBUG: BG Subtraction - Calculated absolute difference")
    
    # Apply threshold to create binary mask
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    print(f"DEBUG: BG Subtraction - Applied threshold {threshold}, non-zero pixels: {cv2.countNonZero(binary)}")
    
    # Apply min_area filtering if specified
    if min_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(binary)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        binary = filtered_mask
        print(f"DEBUG: BG Subtraction - Applied min_area filter ({min_area}), remaining non-zero: {cv2.countNonZero(binary)}")
    
    # Apply ROI shrink if specified
    if roi_shrink < 1.0 and contour_image is not None:
        # This would require contour scaling logic similar to Band-ROI
        # For now, just log the parameter
        print(f"DEBUG: BG Subtraction - ROI shrink parameter {roi_shrink} noted (not implemented in this version)")
    
    # Convert to boolean mask
    result_mask = binary > 0
    
    # Save outputs if requested
    if 'pic' in [p.lower() for p in out_params] or 'save' in [p.lower() for p in out_params]:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save background image
        bg_path = os.path.join(save_dir, f"{fname}_bg_subtraction_bg.png")
        cv2.imwrite(bg_path, bg)
        print(f"DEBUG: BG Subtraction - Saved background image to {bg_path}")
        
        # Save difference image
        diff_path = os.path.join(save_dir, f"{fname}_bg_subtraction_diff.png")
        cv2.imwrite(diff_path, diff)
        print(f"DEBUG: BG Subtraction - Saved difference image to {diff_path}")
        
        # Save binary result
        binary_path = os.path.join(save_dir, f"{fname}_bg_subtraction_binary.png")
        cv2.imwrite(binary_path, binary)
        print(f"DEBUG: BG Subtraction - Saved binary result to {binary_path}")
    
    print(f"DEBUG: BG Subtraction - Completed, result mask pixels: {np.count_nonzero(result_mask)}")
    return result_mask


def apply_odbp_filtering(original_img, mask_filtered, odbp_config, out_params, save_dir, image_path, contour_image=None):
    """
    Apply ODBP (One-Dimensional Background Projection) filtering.
    Uses weighted difference between original image and median-filtered background.
    
    Formula: Result = Alpha * Original + Beta * Median_Background + Offset
    
    Args:
        original_img: RGB image
        mask_filtered: Current mask (used as base/DUT mask)
        odbp_config: Dict with 'median_kernel', 'threshold', 'alpha', 'beta', 'offset', 
                     'invert', 'blur', 'uniform_light', 'ul_kernel_size', 'min_area', 'roi_shrink'
        out_params: Output parameters (e.g., ['Pic', 'Save'])
        save_dir: Directory to save outputs
        image_path: Path to original image
        contour_image: Optional contour image for reference
    
    Returns:
        Updated mask with defects detected via ODBP
    """
    import cv2
    import numpy as np
    import os
    
    print(f"DEBUG: ODBP Config: {odbp_config}")
    
    # Extract parameters
    median_kernel = odbp_config.get('median_kernel', 23)
    threshold = odbp_config.get('threshold', 99)
    alpha = odbp_config.get('alpha', 2.1)
    beta = odbp_config.get('beta', -1.2)
    offset = odbp_config.get('offset', 15)
    invert = odbp_config.get('invert', False)
    blur_amount = odbp_config.get('blur', 0)
    uniform_light = odbp_config.get('uniform_light', False)
    ul_kernel_size = odbp_config.get('ul_kernel_size', 101)
    min_area = odbp_config.get('min_area', 0)
    roi_shrink = odbp_config.get('roi_shrink', 1.0)
    
    # Ensure median_kernel is odd and at least 3
    if median_kernel < 3:
        median_kernel = 3
    if median_kernel % 2 == 0:
        median_kernel += 1
    
    # Ensure ul_kernel_size is odd if enabled
    if uniform_light and ul_kernel_size % 2 == 0:
        ul_kernel_size += 1
    
    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_img.copy()
    
    # Apply ROI mask if provided
    if mask_filtered is not None:
        mask_u8 = (mask_filtered.astype(np.uint8) * 255) if mask_filtered.dtype == bool else mask_filtered
        # Ensure same size
        if mask_u8.shape[:2] != gray.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Apply mask
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask_u8)
    else:
        gray_masked = gray
        mask_u8 = np.ones_like(gray) * 255
    
    # --- Pre-processing: Uniform Light Correction ---
    if uniform_light:
        # Estimate background using Gaussian Blur (Low Pass Filter)
        background_illum = cv2.GaussianBlur(gray_masked, (ul_kernel_size, ul_kernel_size), 0)
        # Subtract background: Image - Background + 127
        # This centers the result around gray (127)
        diff_illum = cv2.subtract(gray_masked, background_illum)
        gray_processed = cv2.add(diff_illum, 127)
        print(f"DEBUG: ODBP - Applied uniform light correction (kernel={ul_kernel_size})")
    else:
        gray_processed = gray_masked
    
    # --- Pre-processing: Denoise ---
    if blur_amount > 0:
        ksize = blur_amount * 2 + 1
        gray_processed = cv2.GaussianBlur(gray_processed, (ksize, ksize), 0)
        print(f"DEBUG: ODBP - Applied Gaussian blur with kernel size {ksize}")
    
    # --- ODBP Core: Median Background Estimation ---
    # Estimate background using Median Blur (Robust to salt-and-pepper noise)
    bg_median = cv2.medianBlur(gray_processed, median_kernel)
    print(f"DEBUG: ODBP - Estimated background using Median filter (kernel={median_kernel})")
    
    # --- ODBP Core: Weighted Difference ---
    # Formula: Alpha * Original + Beta * Median_Background + Offset
    res_diff = cv2.addWeighted(gray_processed, alpha, bg_median, beta, offset)
    print(f"DEBUG: ODBP - Calculated weighted difference (alpha={alpha}, beta={beta}, offset={offset})")
    
    # --- Thresholding ---
    if invert:
        _, binary = cv2.threshold(res_diff, threshold, 255, cv2.THRESH_BINARY_INV)
        print(f"DEBUG: ODBP - Applied inverted threshold {threshold}")
    else:
        _, binary = cv2.threshold(res_diff, threshold, 255, cv2.THRESH_BINARY)
        print(f"DEBUG: ODBP - Applied threshold {threshold}")
    
    print(f"DEBUG: ODBP - Non-zero pixels after threshold: {cv2.countNonZero(binary)}")
    
    # --- Post-processing: Min Area Filter ---
    if min_area > 0:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(binary)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        binary = filtered_mask
        print(f"DEBUG: ODBP - Applied min_area filter ({min_area}), remaining non-zero: {cv2.countNonZero(binary)}")
    
    # --- ROI Shrink (if specified) ---
    if roi_shrink < 1.0 and contour_image is not None:
        print(f"DEBUG: ODBP - ROI shrink parameter {roi_shrink} noted (not implemented in this version)")
    
    # Convert to boolean mask
    result_mask = binary > 0
    
    # Save outputs if requested
    if 'pic' in [p.lower() for p in out_params] or 'save' in [p.lower() for p in out_params]:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save median background image
        bg_path = os.path.join(save_dir, f"{fname}_odbp_median_bg.png")
        cv2.imwrite(bg_path, bg_median)
        print(f"DEBUG: ODBP - Saved median background to {bg_path}")
        
        # Save weighted difference image
        diff_path = os.path.join(save_dir, f"{fname}_odbp_diff.png")
        cv2.imwrite(diff_path, res_diff)
        print(f"DEBUG: ODBP - Saved weighted difference to {diff_path}")
        
        # Save binary result
        binary_path = os.path.join(save_dir, f"{fname}_odbp_binary.png")
        cv2.imwrite(binary_path, binary)
        print(f"DEBUG: ODBP - Saved binary result to {binary_path}")
    
    print(f"DEBUG: ODBP - Completed, result mask pixels: {np.count_nonzero(result_mask)}")
    return result_mask


def apply_adaptive_gaussian(original_img, mask_filtered, adaptive_gaussian_config, out_params, save_dir, image_path, contour_image=None):
    """
    Applies Adaptive Gaussian Thresholding to the area defined by mask_filtered.
    contour_image: Optional mask of the DUT (DUT_Corrected) to exclude background noise.
    """
    import cv2
    import numpy as np
    import os
    import pandas as pd
    
    adaptive_gaussian_mask_to_keep = None
    
    if adaptive_gaussian_config is not None:
        try:
            print("DEBUG: Applying Adaptive Gaussian Logic...")
            
            # Helper to get param safely (case insensitive keys)
            def get_param(cfg, keys, default):
                for k in keys:
                    for ck in cfg.keys():
                        if str(ck).lower().replace(' ', '').replace('_', '') == k:
                            val = cfg[ck]
                            if pd.isna(val):
                                return default
                            if isinstance(val, str) and (val.strip() == '' or val.lower() == 'nan'):
                                return default
                            return val
                return default
            
            ag_block_size = int(get_param(adaptive_gaussian_config, ['blocksize'], 11))
            # FIX: 'c' or 'threshold' is for Adaptive Gaussian C parameter (NOT contrast adjustment)
            ag_c = float(get_param(adaptive_gaussian_config, ['c', 'threshold'], 2))
            ag_denoise = int(get_param(adaptive_gaussian_config, ['denoise'], 0))
            ag_clean = int(get_param(adaptive_gaussian_config, ['clean'], 0))
            ag_invert = str(get_param(adaptive_gaussian_config, ['invert'], 'False')).lower() in ('true', 'yes', '1')
            
            # New Param: Morphological Closing Kernel Size
            # Default to 0 (disabled). Good values: 3, 5, 7
            ag_morph_kernel = int(get_param(adaptive_gaussian_config, ['morphology', 'morph', 'closing', 'merge'], 0))
            
            # --- NEW: GUI-compatible preprocessing parameters ---
            ag_size = str(get_param(adaptive_gaussian_config, ['size'], 'Original'))
            ag_uniform_light = str(get_param(adaptive_gaussian_config, ['uniformlight', 'ul'], 'off')).lower() in ('on', 'true', 'yes', '1')
            ag_ul_kernel = int(get_param(adaptive_gaussian_config, ['ulkernel', 'ulkernel'], 63))
            ag_gamma = float(get_param(adaptive_gaussian_config, ['gamma'], 1.0))
            # FIX: Only use 'contrastcorrection' for image contrast adjustment, NOT 'contrast' (which is for AG C param)
            ag_contrast = float(get_param(adaptive_gaussian_config, ['contrastcorrection'], 1.0))
            ag_clahe = str(get_param(adaptive_gaussian_config, ['clahe'], 'False')).lower() in ('true', 'yes', '1')

            # Ensure block size is odd
            if ag_block_size < 3: ag_block_size = 3
            if ag_block_size % 2 == 0: ag_block_size += 1
            
            print(f"DEBUG: Adaptive Params - Block:{ag_block_size}, C:{ag_c}, Denoise:{ag_denoise}, Clean:{ag_clean}, Invert:{ag_invert}, Morph:{ag_morph_kernel}")
            print(f"DEBUG: GUI Params - Size:{ag_size}, UniformLight:{ag_uniform_light}, ULKernel:{ag_ul_kernel}, Gamma:{ag_gamma}, Contrast:{ag_contrast}, CLAHE:{ag_clahe}")
            
            # --- NEW: Apply Size (resize) preprocessing ---
            img_processed = original_img.copy()
            mask_processed = mask_filtered.copy()
            
            resize_map = {
                "original": 1.0,
                "1/2": 0.5,
                "1/4": 0.25,
                "1/8": 0.125
            }
            scale = resize_map.get(ag_size.lower(), 1.0)
            
            if scale != 1.0:
                h, w = img_processed.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_processed = cv2.resize(img_processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_processed = cv2.resize(mask_processed.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask_processed = mask_processed.astype(bool)
                print(f"DEBUG: Resized image to {new_w}x{new_h} (scale={scale})")
            
            # --- New Logic: Process Each Component Separately ---
            print(f"DEBUG: Processing Adaptive Gaussian per connected component...")
            
            # Initialize Combined Mask
            combined_mask = np.zeros_like(mask_processed, dtype=bool)
            
            # Find Contours of the Input Mask
            input_mask_u8 = (mask_processed.astype(np.uint8) * 255)
            contours_input, _ = cv2.findContours(input_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_input:
                for cnt in contours_input:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Add padding equal to half block size to ensure correct thresholding at boundaries
                    pad = max(1, ag_block_size // 2)
                    
                    # Calculate padded coordinates
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(img_processed.shape[1], x + w + pad)
                    y2 = min(img_processed.shape[0], y + h + pad)
                    
                    # Crop Image
                    img_crop = img_processed[y1:y2, x1:x2]
                    
                    # Convert to Gray
                    img_gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
                    
                    # --- NEW: Apply Uniform Light preprocessing ---
                    if ag_uniform_light:
                        k_illum = ag_ul_kernel
                        if k_illum % 2 == 0: k_illum += 1
                        background_illum = cv2.GaussianBlur(img_gray_crop, (k_illum, k_illum), 0)
                        diff_illum = cv2.subtract(img_gray_crop, background_illum)
                        img_gray_crop = cv2.add(diff_illum, 127)
                        print(f"DEBUG: Applied Uniform Light with kernel {k_illum}")
                    
                    # --- NEW: Apply CLAHE preprocessing ---
                    if ag_clahe:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        img_gray_crop = clahe.apply(img_gray_crop)
                        print("DEBUG: Applied CLAHE")
                    
                    # --- NEW: Apply Gamma preprocessing ---
                    if ag_gamma != 1.0:
                        invGamma = 1.0 / ag_gamma
                        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        img_gray_crop = cv2.LUT(img_gray_crop, table)
                        print(f"DEBUG: Applied Gamma correction {ag_gamma}")
                    
                    # --- NEW: Apply Contrast preprocessing ---
                    if ag_contrast != 1.0:
                        img_gray_crop = cv2.convertScaleAbs(img_gray_crop, alpha=ag_contrast, beta=0)
                        print(f"DEBUG: Applied Contrast adjustment {ag_contrast}")
                    
                    # Denoise
                    if ag_denoise > 0:
                        ksize = ag_denoise * 2 + 1
                        img_gray_crop = cv2.GaussianBlur(img_gray_crop, (ksize, ksize), 0)
                    
                    # Handle small crops
                    local_block_size = ag_block_size
                    if min(img_gray_crop.shape) <= local_block_size:
                        local_block_size = min(img_gray_crop.shape)
                        if local_block_size % 2 == 0: local_block_size -= 1
                    
                    # Adaptive Threshold
                    if min(img_gray_crop.shape) >= 3 and local_block_size >= 3:
                        ag_binary_crop = cv2.adaptiveThreshold(
                            img_gray_crop, 
                            255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 
                            local_block_size, 
                            ag_c
                        )
                    else:
                         ag_binary_crop = np.zeros_like(img_gray_crop)

                    # Clean (Morph Close)
                    if ag_clean > 0:
                        ksize = ag_clean * 2 + 1
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                        ag_binary_crop = cv2.morphologyEx(ag_binary_crop, cv2.MORPH_CLOSE, kernel)
                    
                    # Invert
                    if ag_invert:
                        ag_binary_crop = cv2.bitwise_not(ag_binary_crop)
                        
                    # Morphological Closing (Merge small fragments)
                    # Applied AFTER Invert if needed, but usually on the white foreground
                    if ag_morph_kernel > 0:
                        # Ensure kernel size is odd
                        ksize_morph = ag_morph_kernel
                        if ksize_morph % 2 == 0: ksize_morph += 1
                        
                        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_morph, ksize_morph))
                        ag_binary_crop = cv2.morphologyEx(ag_binary_crop, cv2.MORPH_CLOSE, kernel_morph)
                        # Optional: Dilate slightly to bridge larger gaps? No, Closing is safer.
                    
                    # Create Local Mask for Intersection (ROI)
                    roi_mask_local = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                    cv2.drawContours(roi_mask_local, [cnt], -1, 255, -1, offset=(-x1, -y1))
                    
                    # Intersection
                    ag_binary_crop_bool = (ag_binary_crop > 127)
                    local_result = ag_binary_crop_bool & (roi_mask_local > 0)
                    
                    # Place back into combined_mask
                    combined_mask[y1:y2, x1:x2] |= local_result
            else:
                 print("DEBUG: No input ROIs found. Result empty.")
            
            # --- Apply DUT Mask Constraint if provided ---
            # Need to resize contour_image to match processed image size if scale was applied
            contour_image_resized = contour_image
            if contour_image is not None and scale != 1.0:
                h_processed, w_processed = img_processed.shape[:2]
                contour_image_resized = cv2.resize(contour_image, (w_processed, h_processed), interpolation=cv2.INTER_NEAREST)
                print(f"DEBUG: Resized contour_image from {contour_image.shape[:2]} to {w_processed}x{h_processed}")
            
            if contour_image_resized is not None:
                print("DEBUG: Applying DUT Mask constraint to Adaptive Gaussian result.")
                # Ensure contour_image matches mask shape
                if contour_image_resized.shape[:2] == combined_mask.shape[:2]:
                    dut_mask_bool = (contour_image_resized > 0)
                    combined_mask = combined_mask & dut_mask_bool
                else:
                    print(f"Warning: DUT Mask shape {contour_image_resized.shape[:2]} mismatch with AG Mask {combined_mask.shape[:2]}. Skipping constraint.")

            # --- Resize mask back to original size if needed ---
            if scale != 1.0:
                h_orig, w_orig = original_img.shape[:2]
                combined_mask_uint8 = (combined_mask.astype(np.uint8) * 255)
                combined_mask_resized = cv2.resize(combined_mask_uint8, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                combined_mask = combined_mask_resized > 0
                print(f"DEBUG: Resized result mask back to original size {w_orig}x{h_orig}")

            # Update mask_filtered
            mask_filtered = combined_mask
            
            # Update mask_to_keep to reflect the final filtering result
            # This ensures that any subsequent logic (like annotation/calculation) uses the filtered result
            adaptive_gaussian_mask_to_keep = combined_mask
            
            print("DEBUG: Adaptive Gaussian applied. mask_filtered updated.")
            
            # Save Output
            if out_params and save_dir:
                # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')

                # Check if 'Save' is in out_params (case insensitive)
                should_save = False
                for p in out_params:
                    if str(p).lower() == 'save':
                        should_save = True
                        break
                
                if should_save:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_mask = (mask_filtered.astype(np.uint8) * 255)
                    save_name = f"{fname_clean}_adaptive_gaussian_mask.png"
                    cv2.imwrite(os.path.join(save_dir, save_name), save_mask)
                    print(f"Saved Adaptive Gaussian output: {save_name}")
            
        except Exception as e:
            print(f"Error executing Adaptive Gaussian: {e}")
            traceback.print_exc()

    return mask_filtered, adaptive_gaussian_mask_to_keep


def apply_relativity_filtering(mask_filtered, relativity_threshold, output_params, save_dir, image_path, reference_roi_mask=None, original_img=None, visualization_overlay=None, enable_save_overlay=False, defect_id_method=None):
    """
    Applies Relativity Filtering to mask_filtered.
    relativity_threshold: float (0.0 to 1.0) or percentage > 1.0
    reference_roi_mask: Optional mask to define ROIs. If provided, relativity is calculated per ROI from this mask.
    """
    import cv2
    import numpy as np
    import os
    import traceback

    relativity_mask_to_remove = None

    if relativity_threshold is not None:
        try:
            print("DEBUG: Applying Relativity Filtering...")
            
            # Parse Threshold
            relativity_th = 0.0
            if isinstance(relativity_threshold, (int, float)):
                relativity_th = float(relativity_threshold)
                if relativity_th > 1.0: # Assume percentage if > 1
                        relativity_th /= 100.0
            elif isinstance(relativity_threshold, str):
                val_str = relativity_threshold.strip().replace('%', '')
                try:
                    relativity_th = float(val_str)
                    if relativity_th > 1.0:
                        relativity_th /= 100.0
                    elif '%' in relativity_threshold:
                        relativity_th /= 100.0
                    else:
                         # Treat as percentage as per requirement
                         relativity_th /= 100.0
                except ValueError:
                    print(f"Warning: Invalid Relativity TH value: {relativity_threshold}. Defaulting to 0.")
                    relativity_th = 0.0
            
            print(f"DEBUG: Parsed Relativity TH: {relativity_th} ({relativity_th*100:.2f}%)")

            # Identify Input ROIs
            # If reference_roi_mask is provided (e.g. Pre-AG Mask), use it to find ROIs.
            # Otherwise, use the current mask_filtered (Post-AG Mask) as ROI source (Legacy/Self-Referential).
            
            roi_source_mask = reference_roi_mask if reference_roi_mask is not None else mask_filtered
            
            input_mask_u8 = (roi_source_mask.astype(np.uint8) * 255)
            contours_input, _ = cv2.findContours(input_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_input:
                print(f"DEBUG: Found {len(contours_input)} input ROIs for Relativity Filtering.")
                
                final_accumulated_mask = np.zeros_like(mask_filtered.astype(np.uint8))
                
                max_roi_debug = 10
                for idx, roi_cnt in enumerate(contours_input):
                    # Create mask for current ROI
                    roi_mask = np.zeros_like(input_mask_u8)
                    cv2.drawContours(roi_mask, [roi_cnt], -1, 255, -1)
                    
                    # Extract result within this ROI (mask_filtered is the input content)
                    mask_u8 = (mask_filtered.astype(np.uint8) * 255)
                    local_result = cv2.bitwise_and(mask_u8, mask_u8, mask=roi_mask)
                    
                    # Find components in this local result
                    contours_local, _ = cv2.findContours(local_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours_local:
                        # Calculate areas locally
                        areas_local = [cv2.contourArea(c) for c in contours_local]
                        max_area_local = max(areas_local)
                        area_th_local = max_area_local * relativity_th
                        
                        if idx < max_roi_debug:
                            print(f"DEBUG: ROI {idx+1}: Max Area={max_area_local}, Threshold={area_th_local}")
                        elif idx == max_roi_debug:
                            print(f"DEBUG: ROI log truncated. Total ROIs: {len(contours_input)}")
                        
                        # Filter locally
                        for i_c, c_local in enumerate(contours_local):
                            if areas_local[i_c] >= area_th_local:
                                cv2.drawContours(final_accumulated_mask, [c_local], -1, 255, -1)
                    else:
                        if idx < max_roi_debug:
                            print(f"DEBUG: ROI {idx+1}: No result found inside.")
                        elif idx == max_roi_debug:
                            print(f"DEBUG: ROI log truncated. Total ROIs: {len(contours_input)}")
                        
                # Update mask_filtered with the accumulated result
                mask_filtered = (final_accumulated_mask > 0)
                
            else:
                print("DEBUG: No input ROIs found in source mask. Skipping Relativity Filtering.")

            # Save Output
            if output_params and save_dir:
                 if not os.path.exists(save_dir):
                     os.makedirs(save_dir, exist_ok=True)
                 # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                 fname_clean = os.path.splitext(os.path.basename(image_path))[0].replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')

                 save_mask = (mask_filtered.astype(np.uint8) * 255)
                 
                 # Save Mask (Binary)
                 if 'mask' in [str(p).lower() for p in output_params]:
                     save_name = f"{fname_clean}_relativity_filtered_mask.png"
                     cv2.imwrite(os.path.join(save_dir, save_name), save_mask)
                     print(f"Saved Relativity Filtering output: {save_name}")
                 
                 # Determine if this is a Dino case (Defect Identification = Dino or contains Dino)
                 is_dino_case = defect_id_method and 'dino' in str(defect_id_method).lower()
                 
                 # Save Visualization (Overlay) for 'Save' param
                 if 'save' in [str(p).lower() for p in output_params]:
                     
                     use_overlay_for_save = False
                     if visualization_overlay is not None and enable_save_overlay and is_dino_case:
                         use_overlay_for_save = True

                     if use_overlay_for_save:
                         # Dino Case: Original Background + Overlay
                         masked_img_vis = original_img.copy()
                         overlay_use = visualization_overlay
                         if overlay_use.shape[:2] != original_img.shape[:2]:
                             overlay_use = cv2.resize(overlay_use, (original_img.shape[1], original_img.shape[0]))
                         mask_bool = save_mask > 0
                         masked_img_vis[mask_bool] = overlay_use[mask_bool]
                         save_img_vis = cv2.cvtColor(masked_img_vis, cv2.COLOR_RGB2BGR)

                         save_name = f"{fname_clean}_relativity_filtered_visualization.png"
                         cv2.imwrite(os.path.join(save_dir, save_name), save_img_vis)
                         print(f"Saved Relativity Filtering output (Visualization): {save_name}")
                     else:
                         # If condition not met, do NOT save visualization
                         pass

                 if 'pic' in output_params and original_img is not None:
                    # Support for Pic in Relativity
                    
                    if visualization_overlay is not None and is_dino_case:
                        # Dino Case: Original Background + Overlay
                        print("DEBUG: Relativity Pic with Overlay (Original Background)")
                        masked_img_rel = original_img.copy()

                        # Overlay logic
                        overlay_use = visualization_overlay
                        if overlay_use.shape[:2] != original_img.shape[:2]:
                            overlay_use = cv2.resize(overlay_use, (original_img.shape[1], original_img.shape[0]))
                        
                        mask_bool = save_mask > 0
                        masked_img_rel[mask_bool] = overlay_use[mask_bool]
                        
                        save_img = cv2.cvtColor(masked_img_rel, cv2.COLOR_RGB2BGR)
                    else:
                        # Fallback/Standard: Black Background
                        print("DEBUG: Relativity Pic without Overlay (Black Background)")
                        masked_img_rel = np.zeros_like(original_img)
                        masked_img_rel = cv2.bitwise_and(original_img, original_img, mask=save_mask)
                        save_img = cv2.cvtColor(masked_img_rel, cv2.COLOR_RGB2BGR)
                        
                    save_name_pic = f"{fname_clean}_relativity_filtered_pic.png"
                    cv2.imwrite(os.path.join(save_dir, save_name_pic), save_img)
                    print(f"Saved Relativity Filtering output: {save_name_pic}")

        except Exception as e:
            print(f"Error in Relativity Filtering: {e}")
            traceback.print_exc()

    return mask_filtered


def load_and_filter_filtering_sheet(config_path, product, generation, failure_mode, config_group=None, verbose=True):
    """
    Loads the Filtering sheet from RELP_Configuration.xlsx and filters by Product/Generation/Failure Mode/Config_Group.

    Args:
        config_path: Path to RELP_Configuration.xlsx
        product: Product name (e.g., 'Matt_Screen')
        generation: Generation (e.g., '2026')
        failure_mode: Failure mode (e.g., 'Minor_Wear')
        config_group: Config_Group value (e.g., 'B3'), optional
        verbose: Whether to print debug messages

    Returns:
        DataFrame: Filtered filtering rules, or empty DataFrame if no matches
    """
    import pandas as pd
    import os
    
    def _norm(s):
        return str(s).strip().lower().replace(' ', '').replace('_', '')
    
    def norm_gen(x):
        s = str(x).strip()
        if s.endswith('.0'): s = s[:-2]
        return _norm(s)
    
    if not os.path.exists(config_path):
        if verbose:
            print(f"DEBUG: Config file not found at {config_path}")
        return pd.DataFrame()
    
    try:
        # Find Filtering sheet
        xl = pd.ExcelFile(config_path)
        sheet_names = xl.sheet_names
        filtering_sheet_name = next((s for s in sheet_names if _norm(s).replace('_', '') in ('filtering', 'filteringrgb', 'rgbfiltering', 'filtering_rgb')), None)
        
        if not filtering_sheet_name:
            if verbose:
                print(f"DEBUG: Filtering sheet not found in config. Available sheets: {sheet_names}")
            return pd.DataFrame()
        
        # Load Filtering sheet
        rgb_filtering_df = ConfigManager().get_sheet(filtering_sheet_name)
        if verbose:
            print(f"DEBUG: Filtering sheet '{filtering_sheet_name}' loaded. Total rows: {len(rgb_filtering_df)}")
        
        if rgb_filtering_df.empty:
            if verbose:
                print("DEBUG: Filtering sheet is empty.")
            return pd.DataFrame()
        
        # Normalize columns
        rgb_cols = {c: _norm(c) for c in rgb_filtering_df.columns}
        p_col = next((c for c in rgb_filtering_df.columns if rgb_cols[c] == 'product' or 'product' in rgb_cols[c]), None)
        g_col = next((c for c in rgb_filtering_df.columns if rgb_cols[c] == 'generation' or 'generation' in rgb_cols[c]), None)
        f_col = next((c for c in rgb_filtering_df.columns if rgb_cols[c] in ('failuremode', 'failure_mode') or 'failuremode' in rgb_cols[c]), None)
        
        # 查找 Config_Group 列
        s_col = next((c for c in rgb_filtering_df.columns if rgb_cols[c] in ('configgroup', 'config_group')), None)
        
        if verbose:
            print(f"DEBUG: Filtering columns - Product: {p_col}, Generation: {g_col}, Failure Mode: {f_col}, Config_Group: {s_col}")
            print(f"DEBUG: Current Context - Product: '{product}', Generation: '{generation}', FM: '{failure_mode}', Config_Group: '{config_group}'")
        
        temp_df = rgb_filtering_df.copy()
        fail_reason = None
        
        # 1. Failure Mode Filter
        if f_col:
            fm_series_norm = temp_df[f_col].astype(str).apply(_norm)
            target_fm_norm = _norm(failure_mode)
            
            # Specific match
            temp_df_specific = temp_df[fm_series_norm == target_fm_norm]
            
            # Fuzzy match if exact fails
            if temp_df_specific.empty:
                def is_substring_match(sheet_val):
                    s_norm = _norm(sheet_val)
                    return s_norm in target_fm_norm if s_norm else False
                fuzzy_mask = temp_df[f_col].astype(str).apply(is_substring_match)
                temp_df_specific = temp_df[fuzzy_mask]
                if not temp_df_specific.empty and verbose:
                    print(f"DEBUG: Fuzzy FM match found. Target='{failure_mode}', Matched rows: {len(temp_df_specific)}")
            
            # Generic rules
            generic_fms = ['all', 'general', 'nan', '']
            temp_df_generic = temp_df[fm_series_norm.isin(generic_fms)]
            
            # Combine
            if not temp_df_specific.empty and not temp_df_generic.empty:
                temp_df = pd.concat([temp_df_specific, temp_df_generic])
                temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            elif not temp_df_specific.empty:
                temp_df = temp_df_specific
            elif not temp_df_generic.empty:
                temp_df = temp_df_generic
            else:
                temp_df = pd.DataFrame(columns=temp_df.columns)
                fail_reason = f"No rules matched Failure Mode: {failure_mode}"
            
            if not temp_df.empty and verbose:
                print(f"DEBUG: After FM filter: {len(temp_df)} rows")
        
        # 2. Product Filter
        if not temp_df.empty and p_col:
            prod_col_norm = temp_df[p_col].astype(str).apply(_norm)
            target_prod = _norm(product) if product else ''
            
            if target_prod:
                match_mask = (prod_col_norm == target_prod) | (prod_col_norm == 'nan') | (prod_col_norm == 'all') | (prod_col_norm == '')
            else:
                match_mask = (prod_col_norm == 'nan') | (prod_col_norm == 'all') | (prod_col_norm == '')
            
            temp_df = temp_df[match_mask]
            
            if temp_df.empty:
                fail_reason = f"No rules matched Product: {product} (after matching FM)"
                if verbose:
                    print(f"DEBUG: Product Filter FAILED. No matches for '{target_prod}' or wildcards.")
            elif verbose:
                print(f"DEBUG: After Product filter: {len(temp_df)} rows")
        
        # 3. Generation Filter
        if not temp_df.empty and g_col:
            gen_col_norm = temp_df[g_col].astype(str).apply(norm_gen)
            target_gen = norm_gen(generation) if generation else ''
            
            if target_gen:
                match_mask = (gen_col_norm == target_gen) | (gen_col_norm == 'nan') | (gen_col_norm == 'all') | (gen_col_norm == '')
            else:
                match_mask = (gen_col_norm == 'nan') | (gen_col_norm == 'all') | (gen_col_norm == '')
            
            temp_df = temp_df[match_mask]
            
            if temp_df.empty:
                fail_reason = f"No rules matched Generation: {generation} (after matching FM & Product)"
                if verbose:
                    print(f"DEBUG: Generation Filter FAILED. No matches for '{target_gen}' or wildcards.")
            elif verbose:
                print(f"DEBUG: After Generation filter: {len(temp_df)} rows")
        
        # 4. Config_Group Filter
        remaining_fallback_triggered = False
        if not temp_df.empty and config_group:
            if s_col:
                config_group_col_norm = temp_df[s_col].astype(str).apply(_norm)
                target_config_group = _norm(config_group)
                
                # Match specific config_group or wildcard (all/general/nan/empty)
                match_mask = (config_group_col_norm == target_config_group) | (config_group_col_norm == 'nan') | (config_group_col_norm == 'all') | (config_group_col_norm == 'general') | (config_group_col_norm == '')
                temp_df_filtered = temp_df[match_mask]
                
                if temp_df_filtered.empty:
                    # OPTIMIZATION: If no match for specific config_group, fallback to 'Remaining'
                    if target_config_group != _norm('Remaining'):
                        if verbose:
                            print(f"DEBUG: Config_Group '{config_group}' not found. Falling back to 'Remaining'...")
                        remaining_mask = (config_group_col_norm == _norm('Remaining')) | (config_group_col_norm == 'nan') | (config_group_col_norm == 'all') | (config_group_col_norm == 'general') | (config_group_col_norm == '')
                        temp_df = temp_df[remaining_mask]
                        remaining_fallback_triggered = True
                        if verbose:
                            if not temp_df.empty:
                                print(f"DEBUG: Fallback to 'Remaining' successful. {len(temp_df)} rules found.")
                            else:
                                print(f"DEBUG: Fallback to 'Remaining' also failed. No rules available.")
                    else:
                        temp_df = temp_df_filtered
                        fail_reason = f"No rules matched Config_Group: {config_group} (after matching FM, Product & Generation)"
                        if verbose:
                            print(f"DEBUG: Config_Group Filter FAILED. No matches for '{target_config_group}' or wildcards.")
                else:
                    temp_df = temp_df_filtered
                    if verbose:
                        print(f"DEBUG: After Config_Group filter: {len(temp_df)} rows")
            else:
                if verbose:
                    print(f"DEBUG: Config_Group column not found in Filtering sheet, skipping Config_Group filter")
        
        if not temp_df.empty:
            if verbose:
                print(f"DEBUG: Filtering enabled with {len(temp_df)} rules.")
            return temp_df
        else:
            if verbose:
                print(f"DEBUG: No matching filtering rules found. Reason: {fail_reason}")
            return pd.DataFrame()
            
    except Exception as e:
        if verbose:
            print(f"DEBUG: Error loading/filtering Filtering sheet: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def apply_filter_chain(image, filter_chain):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    final_mask = np.ones(gray_image.shape, dtype=np.uint8) * 255

    for f in filter_chain:
        method = f.get("method")
        params = f.get("params", {})

        if method == "Morph Top-Hat":
            ksize = params.get("Kernel Size", 5)
            tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, tophat_kernel)

            _, sparse_mask = cv2.threshold(tophat, params.get("Threshold", 10), 255, cv2.THRESH_BINARY)

            closing_ksize = params.get("Closing Kernel Size", 5)
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_ksize, closing_ksize))
            mask = cv2.morphologyEx(sparse_mask, cv2.MORPH_CLOSE, closing_kernel)

            if mask is not None:
                final_mask = cv2.bitwise_and(final_mask, mask)

    return final_mask


def run_defect_filtering_pipeline():
    print("--- Running Simplified Defect Filtering and Parametric Pipeline ---")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("ERROR: SAM2 modules not found.")
        return

    CONFIG_PATH = "RELP_Configuration.xlsx"

    # Load Failure Mode Configuration and check conditions
    print("\n🔍 Loading and checking Failure Mode Configuration...")
    import utils.utils_general as ug
    if not ug.load_and_check_failure_mode_config(CONFIG_PATH, sheet_name="Failure Mode"):
        print("\n❌ Pipeline execution aborted: Conditions not met.")
        return

    # Load defect filter configuration
    print("\n📋 Loading defect class configurations...")
    import utils.utils_general as ug
    DEFECT_FILTER_CONFIG, CLASS_NAMES = ug.load_defect_config(CONFIG_PATH, sheet_name="Mac_Classes")
    print(f"✅ Loaded {len(CLASS_NAMES)} defect classes from {CONFIG_PATH}")

    print("\n🔧 Loading models...")
    tape_detector = ug.load_detectron2_model(
        "weights/macbook/K11p/DUT/OD_CFG.pickle",
        "weights/macbook/K11p/DUT/model_final.pth"
    )
    defect_detector = ug.load_detectron2_model(
        "weights/General_FM/K11p/OD_CFG.pickle",
        "weights/General_FM/K11p/model_final.pth"
    )
    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(Path.cwd().parent / "sam2/checkpoints/sam2.1_hiera_tiny.pt"),
        device="cpu"
    )
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("✅ Models loaded.")

    # Set paths from configuration (you can also read these from Excel if needed)
    input_path = Path("Pics/macbook/K11p/B1")
    output_base = Path("Result")

    # Create subfolder: Product_Generation_OutputPath
    product = "Macbook_ML"
    generation = "K11p_ML"
    output_folder_name = f"{product}_{generation}_{output_base.name}"
    output_dir_base = output_base / output_folder_name

    output_dir_defects = output_dir_base / "defects"
    output_dir_parametric = output_dir_base

    print(f"\n📁 Input Path: {input_path}")
    print(f"📁 Output Base: {output_base}")
    print(f"📁 Output Folder: {output_dir_base}")
    print(f"📁 Defects Directory: {output_dir_defects}")

    # Create output directories
    output_dir_defects.mkdir(parents=True, exist_ok=True)
    output_dir_parametric.mkdir(parents=True, exist_ok=True)

    # Get image paths
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        return

    image_paths = sorted([p for p in input_path.iterdir()
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    if not image_paths:
        print(f"⚠️ No images found in {input_path}")
        return

    print(f"\n📂 Found {len(image_paths)} images to process")

    all_parametric_results = []
    parametric_cols_order = []

    for img_path in image_paths:
        print(f"\nProcessing image: {img_path.name}")
        centered_image, dut_contour = ug.process_image(
            img_path, sam2_predictor, tape_detector
        )
        if centered_image is None:
            print(" failed. Skipping.")
            continue

        outputs = defect_detector(centered_image)
        instances = outputs["instances"]
        overlay_image = centered_image.copy()
        final_combined_mask = np.zeros(centered_image.shape[:2], dtype=np.uint8)
        found_specific_defect = False

        if len(instances) > 0:
            pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()

            for i, (box, class_id) in enumerate(zip(pred_boxes, pred_classes)):
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"

                if class_name in DEFECT_FILTER_CONFIG and DEFECT_FILTER_CONFIG[class_name]["filter_chain"]:
                    found_specific_defect = True
                    x1, y1, x2, y2 = map(int, box)

                    cropped_image = centered_image[y1:y2, x1:x2]
                    if cropped_image.size == 0:
                        continue

                    filter_config = DEFECT_FILTER_CONFIG[class_name]
                    precise_mask = apply_filter_chain(cropped_image, filter_config.get("filter_chain", []))

                    if precise_mask is not None:
                        full_mask = np.zeros(centered_image.shape[:2], dtype=np.uint8)
                        full_mask[y1:y2, x1:x2] = precise_mask
                        overlay_image[full_mask > 0] = (255, 0, 255)
                        final_combined_mask = cv2.bitwise_or(final_combined_mask, full_mask)

        if found_specific_defect:
            output_path = output_dir_defects / f"{img_path.stem}_filtered_defects.jpg"
            cv2.imwrite(str(output_path), overlay_image)
            print(f"  ✅ Saved filtered defect result to {output_path}")

            parametric_row, p_cols = ug.generate_parametric_row(
                img_path.name, centered_image, final_combined_mask, dut_contour
            )
            all_parametric_results.append(parametric_row)

            if not parametric_cols_order:
                parametric_cols_order = ['Filename', 'Defect Pct'] + p_cols + ['Weighted Score']

    if all_parametric_results:
        print("\n💾 Saving parametric results to Excel...")
        df = pd.DataFrame(all_parametric_results)
        ordered_cols = ([col for col in parametric_cols_order if col in df.columns] +
                        [col for col in df.columns if col not in parametric_cols_order])
        df = df[ordered_cols]

        excel_path = output_dir_parametric / "parametric_output.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"  ✅ Parametric output saved to {excel_path}")
    else:
        print("\n⚠️ No parametric results to save.")

    print("\n✅ All tasks completed.")



# --- Helper Ops for Filtering ---
def is_line(contour, min_aspect_ratio=3.0):
    """
    Determines if a contour is a line (straight or curved) based on aspect ratio.
    """
    if contour is None or len(contour) < 5:
        return False
        
    try:
        # Min Area Rect for Aspect Ratio (Elongation)
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        
        if min(w, h) == 0: 
            return False
            
        aspect_ratio = max(w, h) / min(w, h)
        
        # Criteria: Just Elongation for now
        if aspect_ratio >= min_aspect_ratio:
            return True
            
        return False
    except Exception:
        return False


def _parse_save_operations(out_params):
    ops = []
    if not out_params:
        return ops
    def parse_single(token):
        res = []
        s = str(token).strip()
        sl = s.lower().replace(' ', '')
        if sl.startswith('save(') and s.endswith(')'):
            inner = s[s.find('(')+1:-1]
            # Support '+' as contour flag within each item, and commas to separate items
            # e.g. Save(Overlay+Contour, Complement_Overlay)
            parts = [x.strip() for x in inner.split(',') if x.strip()]
            if not parts:
                res.append(('mask', False))
            for item in parts:
                it = item.strip()
                itl = it.lower().replace(' ', '')
                # Only treat +Contour suffix as flag, don't let 'contour' change the base type
                has_contour = False
                if '+contour' in itl:
                    has_contour = True
                base = it.replace('+Contour', '').replace('+contour', '').strip()
                basel = base.lower().replace(' ', '')
                if basel in ('mask',):
                    res.append(('mask', has_contour))
                elif basel in ('overlay','overaly'):
                    res.append(('overlay', has_contour))
                elif basel in ('complement',):
                    res.append(('complement_overlay', has_contour))
                elif basel in ('complement_mask','complementmask','complement_mask'):
                    res.append(('complement_mask', has_contour))
                elif basel in ('complement_overlay','complementoveraly','complementoverlay','complement_overaly'):
                    res.append(('complement_overlay', has_contour))
        elif sl == 'save':
            res.append(('mask', False))
        return res
    # 1) Parse each token
    for p in out_params:
        ops.extend(parse_single(p))
    # 2) Attempt to reconstruct fragmented 'Save(...)' split across tokens
    try:
        combined = ",".join([str(x).strip() for x in out_params])
        if 'save(' in combined.lower():
            ops.extend(parse_single(combined))
    except Exception:
        pass
    # Deduplicate
    dedup = []
    seen = set()
    for k,c in ops:
        key = (k, bool(c))
        if key not in seen:
            seen.add(key)
            dedup.append((k, c))
    return dedup
    return ops


