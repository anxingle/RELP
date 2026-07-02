import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from PIL import Image



def detect_contour_holes(mask):
    """
    Detect holes using contour-based method
    
    Args:
        mask: Binary mask (uint8, 0 or 255)
    
    Returns:
        holes_mask: Binary mask of detected holes
    """
    try:
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Multi-scale hole detection
        kernel_sizes = [5, 8, 12]
        all_holes = np.zeros_like(binary)
        
        for kernel_size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            holes = cv2.bitwise_xor(closed, binary)
            all_holes = cv2.bitwise_or(all_holes, holes)
        
        # Filter small noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(all_holes, connectivity=8)
        filtered_holes = np.zeros_like(all_holes)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50:  # Minimum hole area
                filtered_holes[labels == i] = 255
        
        return filtered_holes
        
    except Exception as e:
        print(f"Error in detect_contour_holes: {e}")
        return np.zeros_like(mask)


def detect_distance_transform_holes(mask):
    """
    Detect holes using distance transform method
    
    Args:
        mask: Binary mask (uint8, 0 or 255)
    
    Returns:
        holes_mask: Binary mask of detected holes
        maxima_coords: List of (y, x) coordinates of hole centers
    """
    try:
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Get non-zero pixel coordinates
        pixel_coords = np.column_stack(np.where(binary > 0))
        
        if len(pixel_coords) == 0:
            return np.zeros_like(mask), []
        
        height, width = binary.shape
        
        # Create binary image
        binary_pixels = np.zeros((height, width), dtype=np.uint8)
        binary_pixels[pixel_coords[:, 0], pixel_coords[:, 1]] = 255
        
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_pixels, cv2.DIST_L2, 5)
        
        # Find local maxima
        kernel_size = 15
        local_max = cv2.dilate(dist_transform, np.ones((kernel_size, kernel_size), np.uint8))
        maxima_mask = (dist_transform == local_max) & (dist_transform > 5)
        
        maxima_coords = np.column_stack(np.where(maxima_mask))
        
        # Create holes mask
        holes_mask = np.zeros_like(mask)
        for y, x in maxima_coords:
            radius = int(dist_transform[y, x])
            cv2.circle(holes_mask, (x, y), radius, 255, -1)
        
        return holes_mask, maxima_coords.tolist()
        
    except Exception as e:
        print(f"Error in detect_distance_transform_holes: {e}")
        return np.zeros_like(mask), []


# Add sparsity detection parameters
SPARSITY_THRESHOLD_PERCENTILE = 65  # Sparsity threshold percentile
SPARSITY_MIN_AREA = 30  # Minimum anomaly region area
SPARSITY_MAX_AREA = 3000000  # Maximum anomaly region area
SPARSITY_DISPLAY_MIN_AREA = 10000  # Minimum area threshold for displaying anomaly regions

# Add sparsity KNN parameters
SPARSITY_KNN_K = 3  # Number of clusters for sparsity KNN
SPARSITY_KNN_CLUSTERS = 10  # Number of clusters for sparsity KNN
SPARSITY_KNN_CLUSTERS_SELECTION = [10]  # Select the sparsest clusters (usually the most distant clusters)

# Add contour smoothing parameters
CONTOUR_SMOOTH_POINTS = 50  # Number of points after contour smoothing, smaller values result in smoother contours
CONTOUR_SMOOTH_ENABLED = False  # Whether to enable contour smoothing
White_Presentation = False


def cv_show(name, img):
    # window_title = str(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_euclidean_distance_to_black_vectorized(pixels):
    """Vectorized calculation of Euclidean distance from pixels to black (0,0,0)"""
    black = np.array([0, 0, 0])
    return np.linalg.norm(pixels - black, axis=1)


def apply_greedy_connection_optimized(labels, coordinates, target_label=0):
    """Optimized greedy algorithm to connect pixels of target label"""
    if len(coordinates) == 0:
        return labels

    # Create coordinate to index mapping dictionary for improved lookup efficiency
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coordinates)}

    # Create label copy
    new_labels = labels.copy()

    # Get pixels with target label
    target_indices = np.where(labels == target_label)[0]

    # 8-neighborhood offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # For each target pixel, check its 8-neighborhood
    for idx in target_indices:
        y, x = coordinates[idx]

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            neighbor_key = (ny, nx)

            # Fast lookup for neighborhood pixels
            if neighbor_key in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor_key]

                # If neighborhood pixel is not target label, consider connecting
                if new_labels[neighbor_idx] != target_label:
                    # Simplified connection strategy: directly connect adjacent pixels
                    new_labels[neighbor_idx] = target_label

    return new_labels


def apply_second_knn_filter(pixels, coordinates, brightness_threshold_percentile=30, selected_clusters=None, second_knn_clusters=None):
    """Apply second KNN to filter dark pixels"""

    if len(pixels) == 0:
        return np.array([]), np.array([])

    # Calculate pixel brightness (using RGB average)
    brightness = np.mean(pixels, axis=1)

    # Use second KNN to cluster brightness
    print(f"Applying second KNN (K={second_knn_clusters}) to filter dark pixels...")
    brightness_reshaped = brightness.reshape(-1, 1)
    kmeans_brightness = KMeans(n_clusters=second_knn_clusters, random_state=42, n_init=10)
    brightness_labels = kmeans_brightness.fit_predict(brightness_reshaped)

    # Calculate average brightness for each cluster and sort
    cluster_means = []
    for i in range(second_knn_clusters):
        cluster_mask = brightness_labels == i
        if np.any(cluster_mask):
            cluster_means.append((i, np.mean(brightness[cluster_mask])))
        else:
            cluster_means.append((i, 0))

    # Sort clusters by brightness
    cluster_means.sort(key=lambda x: x[1])

    # Print cluster information
    for sorted_idx, (original_cluster_id, mean_brightness) in enumerate(cluster_means):
        cluster_mask = brightness_labels == original_cluster_id
        pixel_count = np.sum(cluster_mask)
        print(f"Second KNN cluster {sorted_idx + 1} (original ID {original_cluster_id}): {pixel_count} pixels, average brightness: {mean_brightness:.2f}")

    # Create mask based on selected clusters
    selected_mask = np.zeros(len(brightness_labels), dtype=bool)
    for gui_cluster_num in selected_clusters:
        sorted_idx = gui_cluster_num - 1  # GUI numbering starts from 1
        if 0 <= sorted_idx < second_knn_clusters:
            original_cluster_id = cluster_means[sorted_idx][0]
            cluster_mask = brightness_labels == original_cluster_id
            selected_mask |= cluster_mask
            print(f"Selected second KNN cluster {gui_cluster_num}: {np.sum(cluster_mask)} pixels")

    print(f"Second KNN filtering result - retained pixels: {np.sum(selected_mask)}, filtered pixels: {np.sum(~selected_mask)}")

    return pixels[selected_mask], coordinates[selected_mask]


def process_image_with_contour_detection(clustered_overlay_array, original_image_array, fast_mode=False, roi_start_y=0, target_rgb=None, full_original_image_array=None, defect_min_area=None, rgb_distance_threshold=None, scaled_contour=None, ignore_mask=None):
    """Optimized hole detection algorithm - improved processing speed"""
    try:
        # Directly use the passed numpy arrays
        image = cv2.cvtColor(clustered_overlay_array, cv2.COLOR_RGB2BGR)
        original_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)

        # Create mask for non-black pixels in original image
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        height, width = original_gray.shape
        original_mask = original_gray > 0

        # Erode the original mask ONLY at the outer boundaries to avoid edge artifacts
        # We want to preserve internal structures (like text) while ignoring the artificial ROI edge
        # Instead of eroding the mask directly (which thins everything including text),
        # we create a mask that only erodes from the image border inward.
        
        # 1. Create a full-white mask
        image_border_mask = np.zeros((height, width), dtype=np.uint8)
        # 2. Fill the ROI (original_mask) into it
        image_border_mask[original_mask] = 255
        
        # 3. Find contours of the ROI to identify the OUTER boundary
        roi_contours, _ = cv2.findContours(image_border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Create a mask that is only eroded from the outer contours
        filtering_mask = original_mask.copy().astype(np.uint8) * 255
        erosion_size = 5
        
        if roi_contours:
            # Find the largest contour area to set a threshold
            max_area = max(cv2.contourArea(c) for c in roi_contours) if roi_contours else 0
            
            # Draw black lines along the outer contours to "erode" only the boundary
            # Only erode large contours (likely the main object/ROI) to avoid thinning small details like text
            for contour in roi_contours:
                if cv2.contourArea(contour) > max_area * 0.1:  # Threshold: 10% of max area
                    cv2.drawContours(filtering_mask, [contour], -1, 0, thickness=erosion_size*2)
            
        # Convert back to boolean
        filtering_mask = filtering_mask > 0

        # Convert to grayscale and find all non-black pixels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_black_mask = gray > 0

        if np.sum(non_black_mask) == 0:
            print("No non-black pixels found in image")
            return

        # Get coordinates of all non-black pixels
        pixel_coords = np.column_stack(np.where(non_black_mask))
        print(f"Found {len(pixel_coords)} non-black pixels")

        # Create result images
        height, width = gray.shape
        result_image = image.copy()
        contour_only = np.zeros_like(image)

        if fast_mode:
            # Fast mode: use only the most effective detection methods
            print("Using fast mode for connected component detection...")

            # Method 1: Simplified density detection (single scale only)
            kernel_size = 8
            sigma = 3
            threshold = 15

            # Use vectorized operations to create density map
            density_map = np.zeros((height, width), dtype=np.float32)

            # Vectorized Gaussian kernel calculation
            y_coords, x_coords = pixel_coords[:, 0], pixel_coords[:, 1]

            # Batch processing to reduce loops
            for i in range(0, len(pixel_coords), 1000):  # Process in batches
                batch_coords = pixel_coords[i:i + 1000]
                for coord in batch_coords:
                    y, x = coord
                    y_start = max(0, y - kernel_size // 2)
                    y_end = min(height, y + kernel_size // 2 + 1)
                    x_start = max(0, x - kernel_size // 2)
                    x_end = min(width, x + kernel_size // 2 + 1)

                    if y_end > y_start and x_end > x_start:
                        gy, gx = np.ogrid[y_start - y:y_end - y, x_start - x:x_end - x]
                        gaussian = np.exp(-(gx * gx + gy * gy) / (2 * sigma * sigma))
                        density_map[y_start:y_end, x_start:x_end] += gaussian

            # Normalize and binarize
            if density_map.max() > 0:
                density_map = (density_map / density_map.max() * 255).astype(np.uint8)

            binary_map = (density_map > threshold).astype(np.uint8) * 255

            # Simplified morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

            # Apply ROI barrier if needed
            if roi_start_y > 0 and roi_start_y < height:
                # Draw a barrier line just above the ROI to close off any holes cut by the ROI boundary
                cv2.line(closed, (0, roi_start_y - 1), (width, roi_start_y - 1), 255, 1)

            # Apply virtual new border if scaled_contour is provided
            if scaled_contour is not None:
                cv2.drawContours(closed, [scaled_contour], -1, 255, 1)

            # Detect holes
            filled = closed.copy()
            h, w = filled.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, mask, (0, 0), 255)

            holes = cv2.bitwise_not(filled)
            holes = cv2.bitwise_or(holes, closed)
            all_holes = cv2.bitwise_xor(holes, closed)

            # Apply original mask to hole detection results
            all_holes = cv2.bitwise_and(all_holes, all_holes, mask=filtering_mask.astype(np.uint8))

            # ... existing code ...

        else:
            # Full mode: use all detection methods (original implementation)
            print("Using full mode for connected component detection...")

            # Method 1: Multi-scale density detection
            kernel_sizes = [5, 8, 12]
            sigmas = [2, 3, 4]
            density_thresholds = [10, 15, 20]

            all_holes = np.zeros((height, width), dtype=np.uint8)

            for kernel_size, sigma, threshold in zip(kernel_sizes, sigmas, density_thresholds):
                # Create density map for current scale
                density_map = np.zeros((height, width), dtype=np.float32)

                for coord in pixel_coords:
                    y, x = coord
                    y_start = max(0, y - kernel_size // 2)
                    y_end = min(height, y + kernel_size // 2 + 1)
                    x_start = max(0, x - kernel_size // 2)
                    x_end = min(width, x + kernel_size // 2 + 1)

                    local_h = y_end - y_start
                    local_w = x_end - x_start
                    if local_h > 0 and local_w > 0:
                        gy, gx = np.ogrid[y_start - y:y_end - y, x_start - x:x_end - x]
                        gaussian = np.exp(-(gx * gx + gy * gy) / (2 * sigma * sigma))
                        density_map[y_start:y_end, x_start:x_end] += gaussian

                # Normalize and binarize
                if density_map.max() > 0:
                    density_map = (density_map / density_map.max() * 255).astype(np.uint8)

                binary_map = (density_map > threshold).astype(np.uint8) * 255

                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

                # Apply ROI barrier if needed
                if roi_start_y > 0 and roi_start_y < height:
                    # Draw a barrier line just above the ROI to close off any holes cut by the ROI boundary
                    cv2.line(closed, (0, roi_start_y - 1), (width, roi_start_y - 1), 255, 1)

                # Apply virtual new border if scaled_contour is provided
                if scaled_contour is not None:
                    cv2.drawContours(closed, [scaled_contour], -1, 255, 1)

                # Detect holes for current scale
                filled = closed.copy()
                h, w = filled.shape
                mask = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(filled, mask, (0, 0), 255)

                holes = cv2.bitwise_not(filled)
                holes = cv2.bitwise_or(holes, closed)
                holes = cv2.bitwise_xor(holes, closed)

                # Apply original mask to current scale hole detection results
                holes = cv2.bitwise_and(holes, holes, mask=filtering_mask.astype(np.uint8))

                # Accumulate holes from all scales
                all_holes = cv2.bitwise_or(all_holes, holes)
                
            # Filter small defects if threshold is set
            if defect_min_area is not None and defect_min_area > 0:
                # Use connected components to filter small defects without filling holes
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(all_holes, connectivity=8)
                
                # Create a new mask for filtered holes
                filtered_holes = np.zeros_like(all_holes)
                
                # Iterate through labels (skip background 0)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= defect_min_area:
                        filtered_holes[labels == i] = 255
                
                # Update all_holes with the filtered result
                all_holes = filtered_holes

            # Method 2: Distance transform-based hole detection
            binary_pixels = np.zeros((height, width), dtype=np.uint8)
            binary_pixels[pixel_coords[:, 0], pixel_coords[:, 1]] = 255  # Vectorized assignment

            # Apply original mask to binary pixel map
            binary_pixels = cv2.bitwise_and(binary_pixels, binary_pixels, mask=filtering_mask.astype(np.uint8))

            # Distance transform
            dist_transform = cv2.distanceTransform(255 - binary_pixels, cv2.DIST_L2, 5)

            # Find local maxima (hole centers)
            kernel = np.ones((7, 7))
            dilated = cv2.dilate(dist_transform, kernel)
            local_maxima = (dist_transform == dilated) & (dist_transform > 3)

            # Apply original mask to local maxima
            local_maxima = local_maxima & filtering_mask

            # Generate hole contours from local maxima
            maxima_coords = np.column_stack(np.where(local_maxima))

            for coord in maxima_coords:
                y, x = coord
                radius = int(dist_transform[y, x])
                if radius > 2:
                    cv2.circle(result_image, (x, y), radius, (255, 0, 0), 2)
                    cv2.circle(contour_only, (x, y), radius, (255, 0, 0), 2)

            # Method 3: Contour detection
            contours, hierarchy = cv2.findContours(all_holes, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            hole_count = 0
            if hierarchy is not None:
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 20000:
                        cv2.drawContours(result_image, [contour], -1, (0, 0, 255), 2)
                        cv2.drawContours(contour_only, [contour], -1, (0, 0, 255), 2)

                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(result_image, f'H{hole_count}', (cx - 10, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            hole_count += 1

            # Method 4: Optimized Delaunay triangulation (reduced computation)
            if len(pixel_coords) > 3 and len(pixel_coords) < 10000:  # Limit point count
                from scipy.spatial import Delaunay

                # Sample points to reduce computation
                if len(pixel_coords) > 5000:
                    indices = np.random.choice(len(pixel_coords), 5000, replace=False)
                    sampled_coords = pixel_coords[indices]
                else:
                    sampled_coords = pixel_coords

                points = sampled_coords[:, [1, 0]].astype(np.float32)
                tri = Delaunay(points)

                # Only check triangles with larger areas
                for simplex in tri.simplices:
                    triangle = points[simplex]
                    area = 0.5 * abs((triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1]) -
                                     (triangle[2][0] - triangle[0][0]) * (triangle[1][1] - triangle[0][1]))

                    if area > 200:  # Increase threshold to reduce computation
                        center_x = int(np.mean(triangle[:, 0]))
                        center_y = int(np.mean(triangle[:, 1]))

                        # Use vectorized calculation for minimum distance
                        distances = np.sqrt(
                            (sampled_coords[:, 1] - center_x) ** 2 + (sampled_coords[:, 0] - center_y) ** 2)
                        min_dist = np.min(distances)

                        if min_dist > 10:
                            # Check if hole center is within non-black region of original image
                            if 0 <= center_y < height and 0 <= center_x < width and filtering_mask[center_y, center_x]:
                                cv2.circle(result_image, (center_x, center_y), 3, (0, 255, 0), 2)
                                cv2.circle(contour_only, (center_x, center_y), 3, (0, 255, 0), 2)

            print(f"Detected {hole_count} contour holes")
            print(f"Detected {len(maxima_coords)} distance transform holes")

        # Return numpy array results
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        contour_only_rgb = cv2.cvtColor(contour_only, cv2.COLOR_BGR2RGB)

        reconstructed_knn_rgb = None
        if not fast_mode:
            # Apply original mask to accumulated hole results
            all_holes = cv2.bitwise_and(all_holes, all_holes, mask=filtering_mask.astype(np.uint8))
            
            # Re-apply area filtering after masking (in case masking created small fragments)
            if defect_min_area is not None and defect_min_area > 0:
                # Use connected components to filter small defects without filling holes
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(all_holes, connectivity=8)
                
                # Create a new mask for filtered holes
                filtered_holes = np.zeros_like(all_holes)
                
                # Iterate through labels (skip background 0)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= defect_min_area:
                        filtered_holes[labels == i] = 255
                        
                all_holes = filtered_holes
            
            # Apply ignore_mask (flooding mask) if provided - REMOVE OVERLAPPING
            if ignore_mask is not None:
                # Resize ignore_mask to match current dimensions if needed
                if ignore_mask.shape[:2] != all_holes.shape[:2]:
                     ignore_mask_resized = cv2.resize(ignore_mask, (all_holes.shape[1], all_holes.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                     ignore_mask_resized = ignore_mask

                # Ensure ignore_mask is single channel
                if len(ignore_mask_resized.shape) == 3:
                    ignore_mask_resized = cv2.cvtColor(ignore_mask_resized, cv2.COLOR_BGR2RGB)
                    ignore_mask_resized = cv2.cvtColor(ignore_mask_resized, cv2.COLOR_RGB2GRAY)
                
                # Binarize ignore_mask
                _, ignore_mask_bin = cv2.threshold(ignore_mask_resized, 127, 255, cv2.THRESH_BINARY)
                
                # Apply Union logic with overlap removal (XOR)
                # "Union of holes and flooding mask, remove overlapping mask" -> A XOR B
                print("Applying ignore_mask to combine holes and flooding (Union - Intersection)...")
                all_holes = cv2.bitwise_xor(all_holes, ignore_mask_bin)

            all_holes_rgb = cv2.cvtColor(all_holes, cv2.COLOR_BGR2RGB)
            
            if target_rgb is not None:
                # Use correct original image (unmasked if available) for RGB calculations
                if full_original_image_array is not None:
                    # Use the full original image (unmasked)
                    full_original_bgr = cv2.cvtColor(full_original_image_array, cv2.COLOR_RGB2BGR)
                    full_original_gray = cv2.cvtColor(full_original_bgr, cv2.COLOR_BGR2GRAY)
                    base_mask_uint8 = (full_original_gray > 0).astype(np.uint8) * 255
                    img_for_calc = full_original_image_array # This is RGB
                else:
                    # Fallback to the masked image
                    base_mask_uint8 = original_mask.astype(np.uint8) * 255
                    img_for_calc = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # original_image is BGR

                # Calculate original object area
                original_object_area = cv2.countNonZero(base_mask_uint8)
                
                # --- RGB Detection (Pixel-wise) ---
                rgb_detected_mask = np.zeros_like(all_holes)
                rgb_defect_pixel_count = 0
                
                if rgb_distance_threshold is not None:
                    # Handle scalar or list threshold
                    if isinstance(rgb_distance_threshold, (list, tuple, np.ndarray)):
                        threshold_val = np.mean(rgb_distance_threshold)
                    else:
                        threshold_val = float(rgb_distance_threshold)
                        
                    # Vectorized processing for all hole pixels
                    mask_indices = np.where(all_holes > 0)
                    if len(mask_indices[0]) > 0:
                        # Extract RGB pixels
                        pixels = img_for_calc[mask_indices]
                        
                        # Calculate distances
                        target_arr = np.array(target_rgb)
                        dists = np.linalg.norm(pixels - target_arr, axis=1)
                        
                        # Filter by threshold
                        valid_idx = dists < threshold_val
                        
                        if np.any(valid_idx):
                            # Map distance to grayscale
                            visible_low = 50
                            if threshold_val > 0:
                                normalized_dist = np.minimum(dists[valid_idx] / threshold_val, 1.0)
                                gray_vals = 255 - normalized_dist * (255 - visible_low)
                                gray_vals = np.clip(gray_vals, visible_low, 255).astype(np.uint8)
                            else:
                                gray_vals = np.full(np.sum(valid_idx), 255, dtype=np.uint8)
                            
                            # Assign to mask
                            rgb_detected_mask[mask_indices[0][valid_idx], mask_indices[1][valid_idx]] = gray_vals
                            rgb_defect_pixel_count = np.sum(valid_idx)
                
                rgb_detected_mask_rgb = cv2.cvtColor(rgb_detected_mask, cv2.COLOR_GRAY2RGB) if rgb_distance_threshold is not None else None
                
                # --- Reconstruction Visualization ---
                # Use RGB_detected_mask as reconstruction mask if RGB detection is enabled
                if rgb_distance_threshold is not None:
                    reconstruction_mask = rgb_detected_mask
                else:
                    reconstruction_mask = all_holes
                
                # Create reconstructed KNN image
                reconstructed_knn = np.zeros_like(original_image)
                
                # Get coordinates of defects
                defect_coords = np.column_stack(np.where(reconstruction_mask > 0))
                
                if len(defect_coords) > 0:
                    # Convert target_rgb to BGR for distance calc (target_rgb is RGB from main)
                    target_bgr = np.array(target_rgb[::-1])
                    
                    # Get pixels from original image (original_image is BGR)
                    defect_pixels = original_image[defect_coords[:, 0], defect_coords[:, 1]]
                    
                    # Calculate distance
                    distances = np.linalg.norm(defect_pixels - target_bgr, axis=1)
                    
                    # Map distance to grayscale (smaller distance = brighter)
                    # Original: larger distance = brighter. New: smaller distance = brighter.
                    # distance 0 -> 255 (white), distance >= 255 -> 0 (black)
                    gray_values = 255 - np.clip(distances, 0, 255).astype(np.uint8)
                    
                    # Set pixels in reconstructed image
                    reconstructed_knn[defect_coords[:, 0], defect_coords[:, 1]] = np.column_stack([gray_values]*3)
                
                # Add original object outer contour
                if scaled_contour is not None:
                    # Use the passed Contour_Corrected (green)
                    cv2.drawContours(reconstructed_knn, [scaled_contour], -1, (0, 255, 0), 1)
                else:
                    contours_orig, _ = cv2.findContours(base_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(reconstructed_knn, contours_orig, -1, (0, 255, 0), 1)
                
                # Use RGB detected area for the ratio if RGB detection is enabled
                if rgb_distance_threshold is not None:
                     ratio = rgb_defect_pixel_count / original_object_area if original_object_area > 0 else 0
                else:
                     defect_area = cv2.countNonZero(reconstruction_mask)
                     ratio = defect_area / original_object_area if original_object_area > 0 else 0
                     
                # text = f"Defect Area% {ratio:.2%}"
                # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                # # Adjust text position to ensure it's not cut off (increased padding from 10 to 30)
                # text_x = max(10, width - text_size[0] - 30)
                # cv2.putText(reconstructed_knn, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                rgb_text = f"TARGET_RGB = {tuple(target_rgb)}"
                
                # Draw small square with target_rgb color to the left of the text
                rgb_text_size = cv2.getTextSize(rgb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                rgb_text_x = max(10, width - rgb_text_size[0] - 30)
                
                square_size = 20
                square_padding = 10
                square_x = rgb_text_x - square_size - square_padding
                square_y = 60 - square_size + 5 # Align with text baseline approx
                
                # Draw gray border
                cv2.rectangle(reconstructed_knn, (square_x, square_y), (square_x + square_size, square_y + square_size), (128, 128, 128), 1)
                # Fill with target_rgb (convert to BGR for OpenCV)
                target_bgr_fill = (int(target_rgb[2]), int(target_rgb[1]), int(target_rgb[0]))
                cv2.rectangle(reconstructed_knn, (square_x + 1, square_y + 1), (square_x + square_size - 1, square_y + square_size - 1), target_bgr_fill, -1)
                
                # Recalculate X for the second line in case it's longer

                rgb_text_size = cv2.getTextSize(rgb_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                rgb_text_x = max(10, width - rgb_text_size[0] - 30)
                cv2.putText(reconstructed_knn, rgb_text, (rgb_text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                reconstructed_knn_rgb = cv2.cvtColor(reconstructed_knn, cv2.COLOR_BGR2RGB)

            return result_image_rgb, contour_only_rgb, all_holes_rgb, reconstructed_knn_rgb, rgb_detected_mask_rgb
        else:
            return result_image_rgb, contour_only_rgb, None, None, None

    except Exception as e:
        print(f"Error occurred during hole detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def find_largest_contour(image_rgb):
    """Find the largest contour of objects in the original image - improved version for more continuous contours"""
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Find all non-black pixels
    non_black_mask = gray > 0

    if np.sum(non_black_mask) == 0:
        return None, 0, None

    # Convert to binary image
    binary = (non_black_mask * 255).astype(np.uint8)

    # Morphological operations to fill small gaps and make contours more continuous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # First closing operation to fill small gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Then opening operation to remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Use CHAIN_APPROX_NONE to get complete contour points instead of simplified version
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None, 0, None

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate contour area
    contour_area = cv2.contourArea(largest_contour)
    
    # Calculate bounding rectangle
    bounding_rect = cv2.boundingRect(largest_contour)

    print(f"Found largest contour, area: {contour_area}")
    print(f"Number of contour points: {len(largest_contour)}")
    print(f"Bounding box: x={bounding_rect[0]}, y={bounding_rect[1]}, w={bounding_rect[2]}, h={bounding_rect[3]}")

    return largest_contour, contour_area, bounding_rect


def get_contour_coordinates(contour, height, width):
    """Extract coordinate points from contour - improved version supporting denser contour points"""
    if contour is None:
        return np.array([])

    # Extract all points on the contour
    contour_points = contour.reshape(-1, 2)  # Convert to (N, 2) format

    # Optional: interpolate the contour to get denser points
    # Calculate contour perimeter
    perimeter = cv2.arcLength(contour, True)
    # Determine interpolation density based on perimeter
    epsilon = perimeter * 0.001  # Adjust this value to control contour fineness
    # Use polygon approximation while maintaining high precision
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)

    # If denser points are needed, interpolate between contour points
    dense_points = []
    for i in range(len(contour_points)):
        current_point = contour_points[i]
        next_point = contour_points[(i + 1) % len(contour_points)]

        # Add current point
        dense_points.append(current_point)

        # Calculate distance between two points
        distance = np.linalg.norm(next_point - current_point)

        # If distance is greater than threshold, interpolate between two points
        if distance > 2:  # Adjustable threshold
            num_interpolated = int(distance / 2)  # Insert one point every 2 pixels
            for j in range(1, num_interpolated):
                t = j / num_interpolated
                interpolated_point = current_point + t * (next_point - current_point)
                dense_points.append(interpolated_point.astype(int))

    contour_points = np.array(dense_points)

    # Convert to (y, x) format to match image coordinate system
    contour_coords = np.column_stack([contour_points[:, 1], contour_points[:, 0]])

    # Ensure coordinates are within image bounds
    contour_coords = contour_coords[
        (contour_coords[:, 0] >= 0) & (contour_coords[:, 0] < height) &
        (contour_coords[:, 1] >= 0) & (contour_coords[:, 1] < width)
        ]

    # Remove duplicate points
    contour_coords = np.unique(contour_coords, axis=0)

    print(f"Extracted contour coordinate points: {len(contour_coords)}")

    return contour_coords


def apply_knn_clustering_with_selection(pixels, coordinates, target_rgb, n_clusters, selected_clusters, KNN_distance_threshold):
    """Apply KNN clustering and select specific clusters (consistent with GUI logic)"""

    if len(pixels) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate Euclidean distance to target RGB
    target = np.array(target_rgb)
    distances = np.linalg.norm(pixels - target, axis=1)
    distances_reshaped = distances.reshape(-1, 1)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distances_reshaped)
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Sort clusters by distance
    cluster_distances = [(i, cluster_centers[i]) for i in range(n_clusters)]
    cluster_distances.sort(key=lambda x: x[1])

    # Create sorting mapping
    sorted_index_to_cluster_id = {}
    for sorted_idx, (original_cluster_id, distance) in enumerate(cluster_distances):
        sorted_index_to_cluster_id[sorted_idx] = original_cluster_id
        cluster_mask = cluster_labels == original_cluster_id
        pixel_count = np.sum(cluster_mask)
        print(f"Cluster {sorted_idx + 1} (original ID {original_cluster_id}): {pixel_count} pixels, average distance: {distance:.2f}")

    # Check if distance difference between first and last cluster meets threshold condition
    if len(cluster_distances) >= 2:
        first_cluster_distance = cluster_distances[0][1]  # Minimum distance
        last_cluster_distance = cluster_distances[-1][1]  # Maximum distance
        distance_diff = last_cluster_distance - first_cluster_distance
        
        print(f"First cluster distance: {first_cluster_distance:.2f}, last cluster distance: {last_cluster_distance:.2f}")
        print(f"Distance difference: {distance_diff:.2f}, threshold: {KNN_distance_threshold}")
        
        if distance_diff < KNN_distance_threshold:
            print(f"Distance difference {distance_diff:.2f} is less than threshold {KNN_distance_threshold}, skipping cluster selection")
            return np.array([]), np.array([]), np.array([])
    
    # Create mask based on selected clusters
    selected_mask = np.zeros(len(cluster_labels), dtype=bool)
    for gui_cluster_num in selected_clusters:
        sorted_idx = gui_cluster_num - 1  # GUI numbering starts from 1
        if 0 <= sorted_idx < n_clusters:
            original_cluster_id = sorted_index_to_cluster_id[sorted_idx]
            cluster_mask = cluster_labels == original_cluster_id
            selected_mask |= cluster_mask
            print(f"Selected cluster {gui_cluster_num}: {np.sum(cluster_mask)} pixels")

    # Return selected pixels and coordinates
    selected_pixels = pixels[selected_mask]
    selected_coordinates = coordinates[selected_mask]

    # Create unified labels for selected pixels
    binary_labels = np.ones(np.sum(selected_mask), dtype=int)

    return selected_pixels, selected_coordinates, binary_labels


def create_gui_style_cluster_image_with_distance_grayscale(coordinates, pixels, height, width):
    """Create grayscale cluster image based on RGB distance or full white display"""
    cluster_image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(coordinates) > 0:
        y_coords, x_coords = coordinates[:, 0], coordinates[:, 1]

        if White_Presentation:
            # All-white display mode
            for i in range(len(coordinates)):
                cluster_image[y_coords[i], x_coords[i]] = [255, 255, 255]
        else:
            # Grayscale value display mode based on distance
            # Calculate Euclidean distance from each pixel to black
            distances = calculate_euclidean_distance_to_black_vectorized(pixels)

            # Normalize distance to 0-255 range, closer distance means darker
            max_distance = np.max(distances)
            min_distance = np.min(distances)

            if max_distance > min_distance:
                # Normalize distance, smaller distance means smaller value (darker)
                normalized_distances = (distances - min_distance) / (max_distance - min_distance)
                grayscale_values = (normalized_distances * 255).astype(np.uint8)
            else:
                # If all distances are the same, use medium gray
                grayscale_values = np.full(len(distances), 128, dtype=np.uint8)

            # Set pixel values
            for i in range(len(coordinates)):
                gray_val = grayscale_values[i]
                cluster_image[y_coords[i], x_coords[i]] = [gray_val, gray_val, gray_val]

    return cluster_image



def create_overlay_image(image_rgb, coordinates, labels, height, width):
    """Create overlay image"""
    # Create result image
    result_image = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA format

    # Set colors for different clusters
    colors = {
        0: [255, 0, 0, 127],  # Pixels closer to black - red, 50% transparency
        1: [0, 255, 0, 127]  # Pixels farther from black - green, 50% transparency
    }

    # Vectorized color assignment
    for label in [0, 1]:
        mask = labels == label
        coords = coordinates[mask]
        if len(coords) > 0:
            result_image[coords[:, 0], coords[:, 1]] = colors[label]

    # Create overlay image
    original_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    original_rgba[:, :, :3] = image_rgb
    original_rgba[:, :, 3] = 255  # Fully opaque

    # Merge original and result images
    overlay = original_rgba.copy()

    # Alpha blending where clustering results exist
    mask = result_image[:, :, 3] > 0
    for c in range(3):  # RGB channels
        overlay[mask, c] = (original_rgba[mask, c] * (255 - result_image[mask, 3]) +
                            result_image[mask, c] * result_image[mask, 3]) // 255

    return overlay


def create_cluster_only_image(coordinates, labels, height, width):
    """Create pure clustering result image"""
    result_image = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA format

    colors = {
        0: [255, 0, 0, 127],  # Pixels closer to black - red, 50% transparency
        1: [0, 255, 0, 127]  # Pixels farther from black - green, 50% transparency
    }

    for label in [0, 1]:
        mask = labels == label
        coords = coordinates[mask]
        if len(coords) > 0:
            result_image[coords[:, 0], coords[:, 1]] = colors[label]

    return result_image



def create_gaussian_kernel(kernel_size, sigma):
    """Pre-compute Gaussian kernel"""
    k = kernel_size // 2
    y, x = np.ogrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    return kernel

def create_sparsity_detection_map(pixel_coords, height, width, original_mask=None, original_image=None, target_rgb=None, rgb_distance_threshold=None):
    """
    Create sparsity detection map showing sparse areas (optimized version)

    Args:
        pixel_coords: Pixel coordinate array [(y, x), ...]
        height, width: Image dimensions
        original_mask: Non-black pixel mask of original image
        original_image: Original RGB image
        target_rgb: Target RGB color
        rgb_distance_threshold: Threshold for RGB distance check

    Returns:
        sparsity_map: Sparsity detection map (0-255)
        sparsity_visualization: Visualization image (BGR)
        anomaly_regions: Anomaly region list
        anomaly_mask: Anomaly region mask
        selected_regions_mask: Mask of selected anomaly regions
        rgb_detected_mask: Mask of regions passing RGB distance check
    """
    print("\n=== Starting sparsity detection (optimized version) ===")
    import time
    start_time = time.time()

    # Create density map
    density_map = np.zeros((height, width), dtype=np.float32)

    # Multi-scale density calculation parameters
    kernel_sizes = [5, 8, 12, 16]  # Different neighborhood sizes
    sigmas = [2, 3, 4, 5]  # Corresponding Gaussian standard deviations
    weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different scales

    # Pre-compute all Gaussian kernels
    gaussian_kernels = []
    for kernel_size, sigma in zip(kernel_sizes, sigmas):
        kernel = create_gaussian_kernel(kernel_size, sigma)
        gaussian_kernels.append(kernel)
    
    print(f"Pre-computed Gaussian kernels completed, time: {time.time() - start_time:.3f}s")

    # Convert coordinates to numpy array for vectorized operations
    if len(pixel_coords) > 0:
        coords_array = np.array(pixel_coords)
        y_coords = coords_array[:, 0]
        x_coords = coords_array[:, 1]
    else:
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width, 3), dtype=np.uint8), [], np.zeros((height, width), dtype=np.uint8)

    # Optimized computation for each scale
    for i, (kernel_size, sigma, weight, gaussian_kernel) in enumerate(zip(kernel_sizes, sigmas, weights, gaussian_kernels)):
        scale_start_time = time.time()
        scale_density = np.zeros((height, width), dtype=np.float32)
        
        k = kernel_size // 2
        
        # Batch process coordinates, avoid individual loops
        for j in range(len(y_coords)):
            y, x = y_coords[j], x_coords[j]
            
            # Calculate boundaries
            y_start = max(0, y - k)
            y_end = min(height, y + k + 1)
            x_start = max(0, x - k)
            x_end = min(width, x + k + 1)
            
            # Calculate corresponding positions in Gaussian kernel
            ky_start = max(0, k - y)
            ky_end = ky_start + (y_end - y_start)
            kx_start = max(0, k - x)
            kx_end = kx_start + (x_end - x_start)
            
            if y_end > y_start and x_end > x_start and ky_end > ky_start and kx_end > kx_start:
                # Directly use pre-computed Gaussian kernel
                kernel_slice = gaussian_kernel[ky_start:ky_end, kx_start:kx_end]
                scale_density[y_start:y_end, x_start:x_end] += kernel_slice

        # Normalize current scale density map
        if scale_density.max() > 0:
            scale_density = scale_density / scale_density.max()

        # Weighted accumulation to total density map
        density_map += weight * scale_density
        
        print(f"Scale {i+1}/{len(kernel_sizes)} (kernel_size={kernel_size}) completed, time: {time.time() - scale_start_time:.3f}s")

    # Normalize final density map
    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    # Calculate sparsity map (inverse of density)
    sparsity_map = 1.0 - density_map

    # Apply original image mask (only show sparsity in non-black areas of original image)
    if original_mask is not None:
        sparsity_map = sparsity_map * original_mask.astype(np.float32)

    # Convert to 0-255 range
    sparsity_map_uint8 = (sparsity_map * 255).astype(np.uint8)

    # Create visualization image (using heatmap color mapping)
    sparsity_colored = cv2.applyColorMap(sparsity_map_uint8, cv2.COLORMAP_JET)

    # Optimized pixel point drawing - batch drawing
    sparsity_visualization = sparsity_colored.copy()
    if len(pixel_coords) > 0:
        # Batch draw white points, avoid individual loops
        for coord in pixel_coords[::max(1, len(pixel_coords)//1000)]:  # Sample drawing to avoid over-density
            y, x = coord
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(sparsity_visualization, (x, y), 1, (255, 255, 255), -1)

    total_time = time.time() - start_time
    print(f"Sparsity detection completed, total time: {total_time:.3f}s")
    print(f"Maximum sparsity value: {sparsity_map.max():.3f}")
    print(f"Average sparsity value: {sparsity_map.mean():.3f}")

    # Add anomaly region detection - using KNN method
    anomaly_regions, anomaly_mask, selected_regions_mask, rgb_detected_mask = detect_sparsity_anomalies_with_knn(
        sparsity_map,
        n_clusters=SPARSITY_KNN_CLUSTERS,
        selected_clusters=SPARSITY_KNN_CLUSTERS_SELECTION,
        min_area=SPARSITY_MIN_AREA,
        max_area=SPARSITY_MAX_AREA,
        original_image=original_image,
        target_rgb=target_rgb,
        rgb_distance_threshold=rgb_distance_threshold
    )

    return sparsity_map_uint8, sparsity_visualization, anomaly_regions, anomaly_mask, selected_regions_mask, rgb_detected_mask


def detect_sparsity_anomalies(sparsity_map, threshold_percentile=90, min_area=50, max_area=5000):
    """
    Automatically detect anomaly regions in sparsity map

    Args:
        sparsity_map: Sparsity map (0-1 range float32)
        threshold_percentile: Sparsity threshold percentile, regions above this value are considered anomalous
        min_area: Minimum anomaly region area
        max_area: Maximum anomaly region area

    Returns:
        anomaly_regions: List of anomaly regions
        anomaly_mask: Anomaly region mask
    """
    print(f"\n=== Starting anomaly region detection ===")

    # Calculate sparsity threshold
    threshold = np.percentile(sparsity_map[sparsity_map > 0], threshold_percentile)
    print(f"Sparsity threshold ({threshold_percentile}%): {threshold:.3f}")

    # Create binary mask
    anomaly_mask = (sparsity_map > threshold).astype(np.uint8)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anomaly_regions = []
    filtered_mask = np.zeros_like(anomaly_mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Only perform basic area range check to avoid noise regions that are too small or too large
        if area >= min_area and area <= max_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate center point
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate average sparsity of this region
            region_mask = np.zeros_like(anomaly_mask)
            cv2.fillPoly(region_mask, [contour], 1)
            avg_sparsity = np.mean(sparsity_map[region_mask == 1])

            anomaly_regions.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area,
                'avg_sparsity': avg_sparsity,
                'contour': contour
            })

            # Add to filtered mask
            cv2.fillPoly(filtered_mask, [contour], 1)

    # Sort by average sparsity (high to low)
    anomaly_regions.sort(key=lambda x: x['avg_sparsity'], reverse=True)

    print(f"Detected {len(anomaly_regions)} anomaly regions")
    # for i, region in enumerate(anomaly_regions):
        # print(f"Region {i + 1}: center({region['center'][0]}, {region['center'][1]}), "
        #       f"area={region['area']}, avg_sparsity={region['avg_sparsity']:.3f}")

    # Return all detected anomaly regions without area filtering here
    # Area filtering will be performed uniformly in the main function

    # Recreate filtered mask (containing all detected regions)
    display_filtered_mask = np.zeros_like(anomaly_mask)
    for region in anomaly_regions:
        cv2.fillPoly(display_filtered_mask, [region['contour']], 1)

    return anomaly_regions, display_filtered_mask


def detect_sparsity_anomalies_with_knn(sparsity_map, n_clusters=None, selected_clusters=None, min_area=50,
                                       max_area=5000, max_samples=50000, original_image=None, target_rgb=None, rgb_distance_threshold=None):
    """
    Use KNN clustering to detect anomaly regions in sparsity map (optimized version)

    Args:
        sparsity_map: Sparsity map (0-1 range float32)
        n_clusters: Number of KNN clusters
        selected_clusters: List of selected cluster numbers
        min_area: Minimum anomaly region area
        max_area: Maximum anomaly region area
        max_samples: Maximum sampling count for accelerating KMeans clustering
        original_image: Original RGB image for color verification
        target_rgb: Target RGB color for verification
        rgb_distance_threshold: Threshold for RGB distance

    Returns:
        anomaly_regions: List of anomaly regions
        anomaly_mask: Anomaly region mask
        selected_regions_mask: Mask of selected regions
        rgb_detected_mask: Mask of regions confirming to RGB target
    """
    import time
    start_time = time.time()
    
    # Initialize RGB detected mask
    rgb_detected_mask = np.zeros_like(sparsity_map, dtype=np.uint8)

    
    if n_clusters is None:
        n_clusters = SPARSITY_KNN_CLUSTERS
    if selected_clusters is None:
        selected_clusters = SPARSITY_KNN_CLUSTERS_SELECTION

    print(f"\n=== Starting KNN-based anomaly region detection (optimized version) ===")
    print(f"Using {n_clusters} clusters, selected clusters: {selected_clusters}")

    # Get all non-zero sparsity values and their coordinates
    non_zero_mask = sparsity_map > 0
    if not np.any(non_zero_mask):
        print("No sparse regions detected")
        return [], np.zeros_like(sparsity_map, dtype=np.uint8)

    # Extract sparsity values and coordinates
    sparsity_values = sparsity_map[non_zero_mask]
    sparsity_coords = np.column_stack(np.where(non_zero_mask))
    
    total_pixels = len(sparsity_values)
    print(f"Sparsity value range: {sparsity_values.min():.3f} - {sparsity_values.max():.3f}")
    print(f"Average sparsity: {sparsity_values.mean():.3f}")
    print(f"Total pixels: {total_pixels}")

    # Optimization 1: If data is too large, sample to accelerate KMeans
    if total_pixels > max_samples:
        print(f"Data too large ({total_pixels}), sampling to {max_samples} points for clustering")
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        sample_values = sparsity_values[sample_indices]
        
        # Cluster sampled data
        sample_reshaped = sample_values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample_reshaped)
        
        # Apply clustering results to all data
        sparsity_reshaped = sparsity_values.reshape(-1, 1)
        cluster_labels = kmeans.predict(sparsity_reshaped)
        cluster_centers = kmeans.cluster_centers_.flatten()
    else:
        # Perform KMeans clustering on sparsity values
        sparsity_reshaped = sparsity_values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sparsity_reshaped)
        cluster_centers = kmeans.cluster_centers_.flatten()

    clustering_time = time.time() - start_time
    print(f"Clustering completed, time: {clustering_time:.2f}s")

    # Sort clusters by sparsity (low to high)
    cluster_sparsity = [(i, cluster_centers[i]) for i in range(n_clusters)]
    cluster_sparsity.sort(key=lambda x: x[1])

    # Create sorting mapping
    sorted_index_to_cluster_id = {}
    for sorted_idx, (original_cluster_id, sparsity) in enumerate(cluster_sparsity):
        sorted_index_to_cluster_id[sorted_idx] = original_cluster_id
        cluster_mask = cluster_labels == original_cluster_id
        pixel_count = np.sum(cluster_mask)
        # print(f"Cluster {sorted_idx + 1} (original ID {original_cluster_id}): {pixel_count} pixels, average sparsity: {sparsity:.3f}")

    # Optimization 2: Vectorized cluster selection operation
    selected_mask = np.zeros(len(cluster_labels), dtype=bool)
    for gui_cluster_num in selected_clusters:
        sorted_idx = gui_cluster_num - 1  # GUI numbering starts from 1
        if 0 <= sorted_idx < n_clusters:
            original_cluster_id = sorted_index_to_cluster_id[sorted_idx]
            cluster_mask = cluster_labels == original_cluster_id
            selected_mask |= cluster_mask
            print(f"Selected cluster {gui_cluster_num}: {np.sum(cluster_mask)} pixels")

    mask_creation_time = time.time()
    print(f"Mask creation time: {mask_creation_time - start_time - clustering_time:.2f}s")

    # Optimization 3: Pre-filter coordinates to reduce subsequent processing
    selected_coords = sparsity_coords[selected_mask]
    
    if len(selected_coords) == 0:
        print("No selected pixels")
        return [], np.zeros_like(sparsity_map, dtype=np.uint8)

    # Create anomaly region mask - using more efficient method
    anomaly_mask = np.zeros_like(sparsity_map, dtype=np.uint8)
    
    # Optimization 4: Batch set pixel values
    if len(selected_coords) > 0:
        # Ensure coordinates are within valid range
        valid_coords = selected_coords[
            (selected_coords[:, 0] >= 0) & (selected_coords[:, 0] < sparsity_map.shape[0]) &
            (selected_coords[:, 1] >= 0) & (selected_coords[:, 1] < sparsity_map.shape[1])
        ]
        
        if len(valid_coords) > 0:
            anomaly_mask[valid_coords[:, 0], valid_coords[:, 1]] = 1

            # Optimization 5: Pre-compute morphological kernels
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Morphological operations to connect nearby anomaly points
            anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel_close)
            anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel_open)

    morphology_time = time.time()
    print(f"Morphological operations time: {morphology_time - mask_creation_time:.2f}s")

    # Find connected components
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anomaly_regions = []
    filtered_mask = np.zeros_like(anomaly_mask)

    # Optimization 6: Pre-filter contours to avoid processing too small regions
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append((contour, area))
    
    print(f"Area filtering: original {len(contours)} regions -> valid {len(valid_contours)} regions")

    # Optimization 7: Batch process valid contours
    for contour, area in valid_contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate center point
        center_x = x + w // 2
        center_y = y + h // 2

        # Optimization 8: More efficient region sparsity calculation
        # Create local mask instead of full image mask
        local_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Adjust contour coordinates to local coordinate system
        local_contour = contour - np.array([x, y])
        cv2.fillPoly(local_mask, [local_contour], 1)
        
        # Extract corresponding sparsity region
        sparsity_region = sparsity_map[y:y+h, x:x+w]
        
        # Calculate average sparsity
        masked_sparsity = sparsity_region[local_mask == 1]
        avg_sparsity = np.mean(masked_sparsity) if len(masked_sparsity) > 0 else 0

        # RGB Distance Check
        avg_rgb = None
        rgb_distance = None
        if original_image is not None and target_rgb is not None and rgb_distance_threshold is not None:
            # Extract RGB region
            rgb_region = original_image[y:y+h, x:x+w]
            masked_rgb = rgb_region[local_mask == 1]
            
            if len(masked_rgb) > 0:
                avg_rgb = np.mean(masked_rgb, axis=0)
                # Calculate Euclidean distance to target_rgb
                # target_rgb is typically list/tuple, convert to array
                target_arr = np.array(target_rgb)
                rgb_distance = np.linalg.norm(avg_rgb - target_arr)
                
                # If distance is less than threshold, add to RGB detected mask
                # Check if rgb_distance_threshold is a list (multi-channel?) or scalar
                # User said RGB_Distance is like [[50, 60, 70], ...].
                # Wait, if RGB_Distance is [50, 60, 70], does it mean per-channel threshold or one threshold for the vector?
                # User said "RGB_Distance 也是一个跟TARGET_RGB一样的二维数组比如，[50, 60,70]。里面的值 代表一个阈值"
                # "计算其与TARGET_RGB的距离 D。只有D小于这个阈值的部分才显示"
                # D is scalar (Euclidean distance). So the threshold should be scalar.
                # But RGB_Distance is [[50, 60, 70], ...]. This implies 3 values.
                # Maybe the user means separate thresholds for R, G, B channels?
                # Or maybe user means RGB_Distance corresponds to TARGET_RGB entries.
                # If TARGET_RGB = [[0,0,0], [240,142,59]], then RGB_Distance = [[50,60,70], [80,90,100]].
                # So for current TARGET_RGB [0,0,0], the threshold is [50,60,70].
                # But distance D is scalar. How to compare scalar D with [50,60,70]?
                # Maybe user means: if D < 50 (or some value from the list).
                # Or maybe user mistakenly provided 3 values for threshold.
                # Let's assume the user wants to use the *mean* of the threshold values, or maybe the threshold is per-channel?
                # "计算其与TARGET_RGB的距离 D" -> usually Euclidean distance of the vector.
                # "只有D小于这个阈值" -> implies single threshold.
                # But "RGB_Distance ... [50, 60, 70]" -> 3 values.
                # If I calculate per-channel distance? |R-Rt|, |G-Gt|, |B-Bt|.
                # If user says "Distance D", usually it's one value.
                # Let's look at the user prompt again: "计算其与TARGET_RGB的距离 D。只有D小于这个阈值的部分才显示"
                # And "RGB_Distance 也是一个跟TARGET_RGB一样的二维数组比如，[50, 60,70]"
                # If TARGET_RGB has 3 elements (colors), RGB_Distance has 3 elements (thresholds).
                # Example: TARGET_RGB = [[c1], [c2], [c3]]. RGB_Distance = [t1, t2, t3].
                # So for current color c1, we use threshold t1.
                # But user wrote `[50, 60, 70]` inside the array.
                # `TARGET_RGB = [[0, 0, 0], [240, 142, 59], [79, 171, 207]]`
                # `RGB_Distance = [[50, 60, 70], ...]`
                # Wait, `TARGET_RGB[0]` is `[0, 0, 0]`. `RGB_Distance[0]` is `[50, 60, 70]`.
                # This suggests `RGB_Distance[0]` is NOT a single scalar threshold.
                # Maybe user means per-channel thresholds?
                # Or maybe user just typed random numbers as example.
                # "里面的值 代表一个阈值，且跟TARGET_RGB的位置对应" -> Values represent A threshold (singular) corresponding to TARGET_RGB position.
                # If RGB_Distance corresponds to TARGET_RGB, and TARGET_RGB is a list of colors.
                # Maybe RGB_Distance is `[50, 60, 70]` where 50 corresponds to 1st color, 60 to 2nd, 70 to 3rd?
                # But user wrote `[[50, 60, 70], ...]`?
                # User prompt: "RGB_Distance 也是一个跟TARGET_RGB一样的二维数组比如，[50, 60,70]。"
                # If TARGET_RGB is `[[0,0,0], [240,142,59], [79,171,207]]`.
                # And RGB_Distance is `[50, 60, 70]`. Then it's a 1D array of scalars.
                # If user meant `[[50, 60, 70]]` as one element? No.
                # Let's assume user meant a list of scalars: `[50, 60, 70]`.
                # So for color 1, thresh=50. For color 2, thresh=60.
                # But I defined `RGB_Distance = [[50, 60, 70], ...]` in `main_RELP.py`.
                # If I used `[[50, 60, 70], [80, 90, 100], [110, 120, 130]]`, then for color 1, threshold is `[50, 60, 70]`.
                # This is confusing.
                # Let's check `main_RELP.py` again.
                # I added: `RGB_Distance = [[50, 60, 70], [80, 90, 100], [110, 120, 130]]`
                # And `TARGET_RGB = [[0, 0, 0], [240, 142, 59], [79, 171, 207]]`
                # So for `[0,0,0]`, threshold is `[50, 60, 70]`.
                # This implies per-channel threshold or something.
                # But user said "distance D".
                # If I take the mean of `[50, 60, 70]`, it's 60.
                # Or maybe user implies `RGB_Distance` IS `[50, 60, 70]` (1D array).
                # "RGB_Distance 也是一个跟TARGET_RGB一样的二维数组" -> 2D array like TARGET_RGB.
                # TARGET_RGB is (N, 3). So RGB_Distance is (N, 3).
                # So for each color, we have 3 threshold values.
                # If we calculate scalar distance D, we should probably compare D with... what?
                # Maybe the user wants to check if |R-Rt| < TR and |G-Gt| < TG and |B-Bt| < TB?
                # That would be 3 checks.
                # But user said "calculate its distance D to TARGET_RGB... only D < threshold".
                # If D is scalar, threshold must be scalar.
                # If threshold is vector `[50, 60, 70]`, maybe D is vector `[dR, dG, dB]`?
                # "calculate its distance D" usually means Euclidean.
                # I will calculate Euclidean distance D.
                # And I will use the *average* of the threshold values as the scalar threshold.
                # Or max?
                # Let's use average for now, it's safer. Or just the first value?
                # User example `[50, 60, 70]` might be for 3 different target colors?
                # "RGB_Distance 也是一个跟TARGET_RGB一样的二维数组比如，[50, 60,70]"
                # Maybe he meant `RGB_Distance = [50, 60, 70]` (1D) and TARGET_RGB has 3 colors?
                # If TARGET_RGB has 3 colors, then `[50, 60, 70]` matches one-to-one.
                # But he said "2D array".
                # If I look at `main_RELP.py`: `TARGET_RGB = [[0, 0, 0], [240, 142, 59], [79, 171, 207]]`.
                # If `RGB_Distance` matches this structure, it is `[[t1, t2, t3], [t4, t5, t6], [t7, t8, t9]]`.
                # If so, maybe I should check per-channel difference?
                # But user said "calculate distance D".
                # I'll implement: Calculate Euclidean distance D. Compare with `np.mean(rgb_distance_threshold)`.
                # This handles both cases (scalar or vector threshold).
                
                threshold_val = np.mean(rgb_distance_threshold)
                if rgb_distance < threshold_val:
                    cv2.fillPoly(rgb_detected_mask, [contour], 255)

        anomaly_regions.append({
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area,
            'avg_sparsity': avg_sparsity,
            'contour': contour,
            'avg_rgb': avg_rgb,
            'rgb_distance': rgb_distance
        })

        # Add to filtered mask
        cv2.fillPoly(filtered_mask, [contour], 1)

    contour_processing_time = time.time()
    print(f"Contour processing time: {contour_processing_time - morphology_time:.2f}s")

    # Sort by average sparsity (high to low)
    anomaly_regions.sort(key=lambda x: x['avg_sparsity'], reverse=True)

    total_time = time.time() - start_time
    print(f"KNN anomaly detection total time: {total_time:.2f}s")
    print(f"Detected {len(anomaly_regions)} anomaly regions")

    # Create binary mask for selected regions (white areas)
    selected_regions_mask = np.zeros_like(sparsity_map, dtype=np.uint8)
    
    # Fill selected anomaly regions with white (255)
    for region in anomaly_regions:
        contour = region['contour']
        cv2.fillPoly(selected_regions_mask, [contour], 255)
    
    print(f"Created binary mask containing {len(anomaly_regions)} regions")

    # print(f"Detected {len(anomaly_regions)} anomaly regions")
    # for i, region in enumerate(anomaly_regions):
    #     print(f"Region {i + 1}: center({region['center'][0]}, {region['center'][1]}), "
    #           f"area={region['area']}, average sparsity={region['avg_sparsity']:.3f}")

    return anomaly_regions, filtered_mask, selected_regions_mask, rgb_detected_mask


def save_selected_regions_mask(selected_regions_mask, output_dir, image_name):
    """
    Save binary mask of selected regions
    
    Args:
        selected_regions_mask: Binary mask of selected regions (0-255)
        output_dir: Output directory
        image_name: Image name
    
    Returns:
        mask_path: Path to saved mask file
    """
    if output_dir and image_name:
        mask_path = os.path.join(output_dir, f"{image_name}_selected_regions_mask.png")
        cv2.imwrite(mask_path, selected_regions_mask)
        print(f"Selected regions mask saved to: {mask_path}")
        return mask_path
    return None


def smooth_contour(contour, num_points=None):
    """
    Smooth contour to reduce edge fluctuations

    Args:
        contour: Original contour
        num_points: Number of points after smoothing, if None use global variable

    Returns:
        smoothed_contour: Smoothed contour
    """
    if num_points is None:
        num_points = CONTOUR_SMOOTH_POINTS

    if not CONTOUR_SMOOTH_ENABLED or len(contour) <= 3:
        return contour

    # Convert contour to float
    contour = contour.astype(np.float32)

    # Method 1: Use Gaussian filtering to smooth contour coordinates
    if len(contour) > 10:
        # Apply Gaussian filtering to x and y coordinates separately
        kernel_size = min(15, len(contour) // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create circular padded coordinate arrays
        x_coords = np.concatenate([contour[-kernel_size // 2:, 0, 0],
                                   contour[:, 0, 0],
                                   contour[:kernel_size // 2, 0, 0]])
        y_coords = np.concatenate([contour[-kernel_size // 2:, 0, 1],
                                   contour[:, 0, 1],
                                   contour[:kernel_size // 2, 0, 1]])

        # Apply Gaussian filtering
        x_smooth = cv2.GaussianBlur(x_coords.reshape(-1, 1), (kernel_size, 1), 0)
        y_smooth = cv2.GaussianBlur(y_coords.reshape(-1, 1), (kernel_size, 1), 0)

        # Extract middle part (remove padding)
        x_smooth = x_smooth[kernel_size // 2:-kernel_size // 2].flatten()
        y_smooth = y_smooth[kernel_size // 2:-kernel_size // 2].flatten()

        # Recombine contour
        smoothed_contour = np.column_stack([x_smooth, y_smooth])
        smoothed_contour = smoothed_contour.reshape(-1, 1, 2).astype(np.int32)

        # If need to reduce number of points, use uniform sampling
        if len(smoothed_contour) > num_points:
            indices = np.linspace(0, len(smoothed_contour) - 1, num_points, dtype=int)
            smoothed_contour = smoothed_contour[indices]

        return smoothed_contour

    return contour.astype(np.int32)


def filter_anomaly_regions_by_dynamic_area(anomaly_regions, contour_area, sparsity_area_threshold=0.005):
    """
    Filter anomaly regions based on dynamically calculated area threshold

    Args:
        anomaly_regions: List of anomaly regions
        contour_area: Contour area for calculating dynamic threshold
        sparsity_area_threshold: Area threshold ratio

    Returns:
        filtered_regions: Filtered list of anomaly regions
    """
    # Calculate dynamic area threshold
    if sparsity_area_threshold is None:
        dynamic_min_area = 0
    else:
        dynamic_min_area = contour_area * sparsity_area_threshold
    
    filtered_regions = []
    for region in anomaly_regions:
        if region['area'] >= dynamic_min_area:
            # Apply smoothing to contour
            if CONTOUR_SMOOTH_ENABLED:
                region['contour'] = smooth_contour(region['contour'])
            filtered_regions.append(region)

    print(f"Area filtering: Original {len(anomaly_regions)} regions -> Display {len(filtered_regions)} regions (area >= {dynamic_min_area:.1f})")

    return filtered_regions


def visualize_anomaly_regions(original_image, sparsity_map, anomaly_regions, anomaly_mask):
    """
    Visualize anomaly region detection results - only draw contours

    Args:
        original_image: Original image (BGR)
        sparsity_map: Sparsity map
        anomaly_regions: List of anomaly regions (already filtered)
        anomaly_mask: Anomaly region mask

    Returns:
        result_image: Visualization result image
    """
    # Create sparsity heatmap
    sparsity_uint8 = (sparsity_map * 255).astype(np.uint8)
    sparsity_colored = cv2.applyColorMap(sparsity_uint8, cv2.COLORMAP_JET)

    # Overlay sparsity information on original image
    alpha = 0.6
    result_image = cv2.addWeighted(original_image, 1 - alpha, sparsity_colored, alpha, 0)

    # Use the passed filtered anomaly regions directly, no additional filtering
    filtered_regions = anomaly_regions

    # Draw anomaly region contours (using smoothed contours)
    for i, region in enumerate(filtered_regions):
        # Draw using smoothed contours
        cv2.drawContours(result_image, [region['contour']], -1, (0, 0, 255), 2)

        # Add region labels
        center = region['center']
        label = f"A{i + 1}: {region['avg_sparsity']:.2f}"

        # Optional: Add labels near contours
        x, y, w, h = region['bbox']
        # Add text background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(result_image, (x, y - text_height - 15), (x + text_width + 10, y - 5), (0, 255, 255), -1)  # Yellow background
        cv2.putText(result_image, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)  # Black text

    return result_image





def generate_clustered_overlay(filtered_coordinates_final, filtered_pixels_final, contour_coords, height, width):
    """
    Generate clustered_overlay image

    Returns:
        final_overlay: Clustered_overlay image in RGB format
    """
    # Generate final clustered_overlay image
    final_overlay = np.zeros((height, width, 3), dtype=np.uint8)  # RGB format

    # First place non-contour points (using distance-based grayscale or all-white display)
    if len(filtered_coordinates_final) > 0:
        if White_Presentation:
            # All-white display mode
            for coord in filtered_coordinates_final:
                y, x = coord
                final_overlay[y, x] = [255, 255, 255]
        else:
            # Grayscale display mode based on distance
            # Calculate distances of filtered pixels and convert to grayscale
            filtered_distances = calculate_euclidean_distance_to_black_vectorized(filtered_pixels_final)
            max_distance = np.max(filtered_distances)
            min_distance = np.min(filtered_distances)

            if max_distance > min_distance:
                normalized_distances = (filtered_distances - min_distance) / (max_distance - min_distance)
                grayscale_values = (normalized_distances * 255).astype(np.uint8)
            else:
                grayscale_values = np.full(len(filtered_distances), 128, dtype=np.uint8)

            # Set grayscale values for non-contour points
            for i, coord in enumerate(filtered_coordinates_final):
                y, x = coord
                gray_val = grayscale_values[i]
                final_overlay[y, x] = [gray_val, gray_val, gray_val]

    # Then place contour points (using pure white)
    if len(contour_coords) > 0:
        for coord in contour_coords:
            y, x = coord
            final_overlay[y, x] = [255, 255, 255]  # Contour points remain pure white
        print(
            f"Total coordinates after adding contours: {len(filtered_coordinates_final) + len(contour_coords)} (non-contour: {len(filtered_coordinates_final)}, contour: {len(contour_coords)})")
    else:
        print("No contour coordinates, using only non-contour coordinates")

    return final_overlay

def process_image_complete(image_array, first_knn_cluster_qty, second_knn_cluster_qty,
                          first_knn_selected, second_knn_selected, target_rgb, KNN_distance_threshold,
                          sparsity_area_threshold=None, output_dir=None, image_name=None,
                          contour_shrink_ratio=None, height_lower_ratio=None, enable_sparsity=True, defect_min_area=None,
                          rgb_distance_threshold=None, ignore_mask=None, input_scaled_contour=None, resize_scale=None,
                          resize_scale_1st=1.0, resize_scale_2nd=1.0, enable_hole_detection=True):
    """
    Complete image processing function with all original features

    Args:
        image_array: Input image numpy array (RGB format)
        first_knn_cluster_qty: Number of clusters for first KNN
        second_knn_cluster_qty: Number of clusters for second KNN
        first_knn_selected: Selected cluster list for first KNN
        second_knn_selected: Selected cluster list for second KNN
        target_rgb: Target RGB value
        KNN_distance_threshold: KNN distance threshold
        sparsity_area_threshold: Sparsity area threshold coefficient for calculating minimum display area
        output_dir: Output directory (optional)
        image_name: Image name (optional)
        enable_sparsity: Whether to enable sparsity detection (default: True)
        defect_min_area: Minimum area threshold for defects (optional)
        ignore_mask: Mask of regions to ignore in hole detection (optional)
        input_scaled_contour: Pre-calculated scaled contour (optional)
        resize_scale: Resize scale factor (legacy, use resize_scale_1st and resize_scale_2nd instead)
        resize_scale_1st: Resize scale for first KNN (default: 1.0 = no resize)
        resize_scale_2nd: Resize scale for second KNN relative to first KNN result (default: 1.0 = no resize)
        enable_hole_detection: Whether to enable hole detection (default: True). If False, output last KNN cluster directly.

    Returns:
        dict: Dictionary containing all processing results
            - 'clustered_overlay': Final clustered overlay image
            - 'clustered_overlay_holes': Hole detection result image
            - 'clusters_only': Pure clustering result image
            - 'intermediate_overlay': Intermediate result image
            - 'contour_detection': Connected component detection result image
            - 'contour_only': Pure contour image
            - 'sparsity_map': Sparsity detection map
            - 'anomaly_detection': Anomaly region detection result image
            - 'anomaly_regions': Anomaly region information list
            - 'processing_info': Processing information
    """

    try:
        # Initialize result dictionary
        results = {
            'clustered_overlay': None,
            'clustered_overlay_holes': None,
            'reconstructed_knn': None,
            'clusters_only': None,
            'intermediate_overlay': None,
            'contour_detection': None,
            'contour_only': None,
            'sparsity_map': None,
            'anomaly_detection': None,
            'anomaly_regions': [],
            'processing_info': {}
        }

        # Ensure input is RGB format
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input image must be a 3D array in RGB format")

        image_rgb = image_array.copy()
        original_height, original_width = image_rgb.shape[:2]

        # Handle legacy resize_scale parameter
        if resize_scale is not None and 0 < resize_scale < 1.0:
            resize_scale_1st = resize_scale
            print(f"Legacy resize_scale detected: {resize_scale}, using as resize_scale_1st")

        # Apply resize for first KNN if specified
        first_knn_resized = False
        if resize_scale_1st < 1.0:
            new_height = int(original_height * resize_scale_1st)
            new_width = int(original_width * resize_scale_1st)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"First KNN: Resized image from {original_width}x{original_height} to {new_width}x{new_height} (scale: {resize_scale_1st})")
            first_knn_resized = True
            # Note: contour and other coordinates will be based on resized image
            # If original contour is provided, it should also be resized accordingly
            if input_scaled_contour is not None:
                input_scaled_contour = (input_scaled_contour * resize_scale_1st).astype(np.int32)

        height, width = image_rgb.shape[:2]

        processing_info = {
            'image_size': f"{width}x{height}",
            'original_size': f"{original_width}x{original_height}",
            'resize_scale_1st': resize_scale_1st,
            'resize_scale_2nd': resize_scale_2nd,
            'first_knn_clusters': first_knn_cluster_qty,
            'second_knn_clusters': second_knn_cluster_qty
        }

        # Step 1: Detect largest contour and calculate area
        print("Step 1: Detecting largest contour...")
        contour, contour_area, bounding_rect = find_largest_contour(image_rgb)
        contour_coords = get_contour_coordinates(contour, height, width) if contour is not None else []
        processing_info['contour_points'] = len(contour_coords)
        processing_info['contour_area'] = contour_area
        processing_info['bounding_rect'] = bounding_rect
        
        # Calculate display area threshold based on contour area
        calculated_min_display_area = contour_area * sparsity_area_threshold if sparsity_area_threshold else 0
        processing_info['calculated_min_display_area'] = calculated_min_display_area
        print(f"Maximum contour area: {contour_area}")
        print(f"Calculated minimum display area threshold: {calculated_min_display_area}")

        allowed_mask = np.ones((height, width), dtype=bool)
        scaled_contour = input_scaled_contour
        roi_start_y = 0
        
        if scaled_contour is not None:
            mask1 = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask1, [scaled_contour], 255)
            allowed_mask &= mask1.astype(bool)

        if scaled_contour is None and contour is not None and contour_shrink_ratio is not None and 0 < contour_shrink_ratio < 1:
            print(f"Applying contour shrink ratio: {contour_shrink_ratio}")
            pts = contour.reshape(-1, 2).astype(np.float32)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx = bounding_rect[0] + bounding_rect[2] / 2.0
                cy = bounding_rect[1] + bounding_rect[3] / 2.0
            center = np.array([cx, cy], dtype=np.float32)
            scaled = center + contour_shrink_ratio * (pts - center)
            scaled_contour = scaled.reshape(-1, 1, 2).astype(np.int32)
            mask1 = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask1, [scaled_contour], 255)
            allowed_mask &= mask1.astype(bool)
            processing_info['contour_shrink_ratio'] = contour_shrink_ratio

        if bounding_rect is not None and height_lower_ratio is not None and 0 <= height_lower_ratio <= 1:
            # If height_lower_ratio is 0, we do NOT apply any Y-axis restriction
            # If it is > 0, we restrict the analysis to [ratio, 1.0] of the object height
            if height_lower_ratio > 0:
                print(f"Applying height lower ratio: {height_lower_ratio}")
                x0, y0, bw, bh = bounding_rect
                # height_lower_ratio means we KEEP the range from [ratio, 1.0] of the object height
                # So if ratio=0.3, we start at 30% down from the top (y0)
                ys = y0 + int(bh * height_lower_ratio)
                ys = max(0, min(ys, height))
                roi_start_y = ys
                
                # When ratio > 0, we generally want to exclude the top part, but usually keep the bottom part open or until object end
                # However, to be safe and consistent with "ratio to 1", we can keep until the bottom of the image
                # This ensures we don't accidentally cut off anything below the object if the bounding box wasn't perfect
                ye = height 
                
                mask2 = np.zeros((height, width), dtype=np.uint8)
                # Only restrict Y range, use full width to be independent from bounding box width
                mask2[ys:ye, :] = 255
                allowed_mask &= mask2.astype(bool)
                processing_info['height_lower_ratio'] = height_lower_ratio

        # Apply allowed_mask to image_rgb to exclude regions from ALL subsequent processing
        # This ensures that excluded regions do not participate in KNN, hole detection, or sparsity analysis
        image_rgb[~allowed_mask] = [0, 0, 0]

        # Step 2: Find non-black pixels
        print("Step 2: Finding non-black pixels...")
        non_black_mask = ~np.all(image_rgb == [0, 0, 0], axis=2)
        non_black_coordinates = np.column_stack(np.where(non_black_mask))
        non_black_pixels = image_rgb[non_black_mask]

        if len(non_black_pixels) == 0:
            print("No non-black pixels found in image!")
            processing_info['error'] = "No non-black pixels found in image"
            results['processing_info'] = processing_info
            return results, None, None, None

        processing_info['non_black_pixels'] = len(non_black_pixels)
        print(f"Found {len(non_black_pixels)} non-black pixels.")

        # Step 3: Apply KNN clustering and select specific clusters
        print("Step 3: Applying KNN clustering...")
        selected_pixels, selected_coordinates, binary_labels = apply_knn_clustering_with_selection(
            non_black_pixels, non_black_coordinates, target_rgb, first_knn_cluster_qty, first_knn_selected, KNN_distance_threshold)

        if len(selected_pixels) == 0:
            print("No pixels selected by KNN!")
            processing_info['error'] = "No pixels selected - returning black images"
            results['processing_info'] = processing_info
            
            # Create black images for all outputs
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            black_mask = np.zeros((height, width), dtype=np.uint8)
            
            results['clustered_overlay'] = black_image
            results['intermediate_overlay'] = black_image
            results['sparsity_map'] = black_mask
            results['anomaly_detection'] = black_image
            results['anomaly_regions'] = []
            results['contour_area'] = 0
            results['clustered_overlay_holes'] = black_image
            results['selected_regions_mask'] = black_mask
            results['selected_clustering'] = black_image  # GUI style selected clustering
            
            # Save black images if output directory is provided
            if output_dir and image_name:
                # Save the black mask using the existing function
                save_selected_regions_mask(black_mask, output_dir, image_name)
                # Save black holes image to reference folder
                ref_dir = output_dir if os.path.basename(os.path.normpath(output_dir)) == 'reference' else os.path.join(output_dir, 'reference')
                os.makedirs(ref_dir, exist_ok=True)
                holes_output_path = os.path.join(ref_dir, f"{image_name}_KNN__clustered.png")
                Image.fromarray(black_image).save(holes_output_path)
            
            return results, black_image, black_image, black_mask

        processing_info['selected_pixels'] = len(selected_pixels)
        print(f"Selected {len(selected_pixels)} pixels after KNN.")

        # Step 4: Create intermediate overlay image
        print("Step 4: Creating intermediate overlay...")
        intermediate_overlay = create_gui_style_cluster_image_with_distance_grayscale(
            selected_coordinates, selected_pixels, height, width
        )
        results['intermediate_overlay'] = intermediate_overlay

        # Step 5: Apply resize for second KNN if specified, then apply second KNN filtering
        print("Step 5: Applying second KNN on first KNN selected pixels...")
        
        # Use the actual selected pixels from first KNN, not from intermediate_overlay
        # This ensures second KNN is applied only on the pixels selected by first KNN
        if len(selected_pixels) == 0:
            print("No pixels selected by first KNN for second KNN!")
            processing_info['error'] = "No pixels selected by first KNN for second KNN"
            results['processing_info'] = processing_info
            return results, None, None, None

        # Apply resize for second KNN if specified (relative to first KNN result)
        if resize_scale_2nd < 1.0:
            # Create a temporary image from selected pixels for resizing
            temp_image = np.zeros((height, width, 3), dtype=np.uint8)
            for i, coord in enumerate(selected_coordinates):
                y, x = coord
                temp_image[y, x] = selected_pixels[i]
            
            # Resize the image
            new_height_2nd = int(height * resize_scale_2nd)
            new_width_2nd = int(width * resize_scale_2nd)
            resized_image = cv2.resize(temp_image, (new_width_2nd, new_height_2nd), interpolation=cv2.INTER_AREA)
            print(f"Second KNN: Resized image from {width}x{height} to {new_width_2nd}x{new_height_2nd} (scale: {resize_scale_2nd})")
            
            # Extract pixels from resized image
            resized_mask = np.any(resized_image > 0, axis=2)
            resized_coords = np.column_stack(np.where(resized_mask))
            resized_pixels = resized_image[resized_mask]
            
            # Use resized pixels and coordinates for second KNN
            selected_pixels_for_2nd = resized_pixels
            selected_coordinates_for_2nd = resized_coords
            second_knn_height, second_knn_width = new_height_2nd, new_width_2nd
        else:
            # No resize for second KNN, use original selected pixels
            selected_pixels_for_2nd = selected_pixels
            selected_coordinates_for_2nd = selected_coordinates
            second_knn_height, second_knn_width = height, width

        processing_info['intermediate_pixels'] = len(selected_pixels_for_2nd)
        processing_info['second_knn_size'] = f"{second_knn_width}x{second_knn_height}"

        # Apply second KNN filtering on (possibly resized) first KNN selected pixels
        filtered_pixels_final_resized, filtered_coordinates_final_resized = apply_second_knn_filter(
            selected_pixels_for_2nd, selected_coordinates_for_2nd, selected_clusters=second_knn_selected, second_knn_clusters=second_knn_cluster_qty
        )

        # Step 5.5: If second KNN was resized, upscale results back to first KNN size
        if resize_scale_2nd < 1.0:
            print(f"Step 5.5: Upscaling second KNN results from {second_knn_width}x{second_knn_height} back to {width}x{height}...")
            # Create image from second KNN results
            second_knn_result_image = np.zeros((second_knn_height, second_knn_width, 3), dtype=np.uint8)
            if len(filtered_coordinates_final_resized) > 0:
                for i, coord in enumerate(filtered_coordinates_final_resized):
                    y, x = coord
                    second_knn_result_image[y, x] = filtered_pixels_final_resized[i]
            
            # Upscale to first KNN size
            upscaled_image = cv2.resize(second_knn_result_image, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Extract pixels from upscaled image
            upscaled_mask = np.any(upscaled_image > 0, axis=2)
            filtered_coordinates_final = np.column_stack(np.where(upscaled_mask))
            filtered_pixels_final = upscaled_image[upscaled_mask]
            print(f"Upscaled second KNN results: {len(filtered_pixels_final)} pixels at {width}x{height}")
        else:
            # No resize, use results directly
            filtered_pixels_final = filtered_pixels_final_resized
            filtered_coordinates_final = filtered_coordinates_final_resized

        if len(filtered_pixels_final) == 0:
            print("No pixels remaining after second KNN!")
            processing_info['error'] = "No pixels remaining after second KNN filtering - returning black images"
            results['processing_info'] = processing_info
            
            # Create black images for all outputs
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            black_mask = np.zeros((height, width), dtype=np.uint8)
            
            results['clustered_overlay'] = black_image
            results['intermediate_overlay'] = intermediate_overlay
            results['sparsity_map'] = black_mask
            results['anomaly_detection'] = black_image
            results['anomaly_regions'] = []
            results['contour_area'] = contour_area
            results['clustered_overlay_holes'] = black_image
            results['selected_regions_mask'] = black_mask
            results['selected_clustering'] = black_image  # GUI style selected clustering
            
            # Save black images if output directory is provided
            if output_dir and image_name:
                # Save the black mask using the existing function
                save_selected_regions_mask(black_mask, output_dir, image_name)
            
            return results, black_image, black_image, black_mask

        processing_info['filtered_pixels'] = len(filtered_pixels_final)
        print(f"Filtered {len(filtered_pixels_final)} pixels after second KNN.")

        # Step 6: Generate final clustered_overlay image
        print("Step 6: Generating final clustered overlay...")
        # If we shrank the contour or limited the height, the original contour is outside the ROI
        # So we should NOT draw the original contour as it would be confusing (appearing as a highlight in the excluded region)
        display_contour_coords = contour_coords
        if (contour_shrink_ratio is not None and contour_shrink_ratio < 1.0) or \
           (height_lower_ratio is not None and height_lower_ratio > 0.0):
            display_contour_coords = []

        final_overlay = generate_clustered_overlay(
            filtered_coordinates_final, filtered_pixels_final, display_contour_coords, height, width
        )
        results['clustered_overlay'] = final_overlay

        # Step 6.5: Create selected clustering image (white binary mask, similar to GUI's "Selected clustering")
        print("Step 6.5: Creating selected clustering image (GUI style)...")
        selected_clustering_image = np.zeros((height, width, 3), dtype=np.uint8)
        if len(filtered_coordinates_final) > 0:
            for coord in filtered_coordinates_final:
                y, x = coord
                selected_clustering_image[y, x] = [255, 255, 255]  # White, same as GUI
        results['selected_clustering'] = selected_clustering_image

        # Step 7: Create pure clustering results
        filtered_distances = calculate_euclidean_distance_to_black_vectorized(filtered_pixels_final)
        filtered_binary_labels = (filtered_distances > np.median(filtered_distances)).astype(int)
        cluster_result = create_cluster_only_image(filtered_coordinates_final, filtered_binary_labels, height, width)
        results['clusters_only'] = cluster_result

        # Step 8: Connected component detection and hole detection
        if output_dir and image_name:
            if enable_hole_detection:
                # Perform hole detection, directly pass numpy arrays
                contour_result, contour_only_result, holes_result, reconstructed_knn_result, rgb_detected_mask_from_holes = process_image_with_contour_detection(
                    final_overlay, image_rgb, fast_mode=False, roi_start_y=roi_start_y, target_rgb=target_rgb, full_original_image_array=image_array,
                    defect_min_area=defect_min_area, rgb_distance_threshold=rgb_distance_threshold, scaled_contour=scaled_contour, ignore_mask=ignore_mask
                )
            else:
                # Skip hole detection, use last KNN cluster directly
                print("Hole detection disabled. Using last KNN cluster directly.")
                contour_result = selected_clustering_image.copy()
                contour_only_result = selected_clustering_image.copy()
                holes_result = selected_clustering_image.copy()
                reconstructed_knn_result = None
                rgb_detected_mask_from_holes = selected_clustering_image.copy()

            # Save result images
            if contour_result is not None:
                ref_dir = output_dir if os.path.basename(os.path.normpath(output_dir)) == 'reference' else os.path.join(output_dir, 'reference')
                os.makedirs(ref_dir, exist_ok=True)
                
                # Save first KNN result (intermediate overlay) as white binary mask
                # This is the result after first KNN but before second KNN
                first_knn_white = np.zeros((height, width, 3), dtype=np.uint8)
                if len(selected_coordinates) > 0:
                    for coord in selected_coordinates:
                        y, x = coord
                        first_knn_white[y, x] = [255, 255, 255]
                first_knn_path = os.path.join(ref_dir, f"{image_name}_KNN__first.png")
                Image.fromarray(first_knn_white, 'RGB').save(first_knn_path)
                processing_info['first_knn_path'] = first_knn_path
                print(f"Saved first KNN result to: {first_knn_path}")
                
                # Save first KNN result with original RGB values (for debugging/comparison)
                first_knn_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                if len(selected_coordinates) > 0 and len(selected_pixels) > 0:
                    for i, coord in enumerate(selected_coordinates):
                        y, x = coord
                        # Ensure pixel values are valid uint8
                        rgb_val = selected_pixels[i]
                        if isinstance(rgb_val, (list, tuple, np.ndarray)) and len(rgb_val) >= 3:
                            first_knn_rgb[y, x] = [int(rgb_val[0]), int(rgb_val[1]), int(rgb_val[2])]
                first_knn_rgb_path = os.path.join(ref_dir, f"{image_name}_KNN__first_rgb.png")
                Image.fromarray(first_knn_rgb, 'RGB').save(first_knn_rgb_path)
                processing_info['first_knn_rgb_path'] = first_knn_rgb_path
                print(f"Saved first KNN result with original RGB to: {first_knn_rgb_path}")
                
                # Save second KNN result (final selected clustering) as white binary mask
                second_knn_path = os.path.join(ref_dir, f"{image_name}_KNN__second.png")
                Image.fromarray(selected_clustering_image, 'RGB').save(second_knn_path)
                processing_info['second_knn_path'] = second_knn_path
                print(f"Saved second KNN result to: {second_knn_path}")
                
                # Save second KNN result with original RGB values (for debugging/comparison)
                second_knn_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                if len(filtered_coordinates_final) > 0 and len(filtered_pixels_final) > 0:
                    for i, coord in enumerate(filtered_coordinates_final):
                        y, x = coord
                        rgb_val = filtered_pixels_final[i]
                        if isinstance(rgb_val, (list, tuple, np.ndarray)) and len(rgb_val) >= 3:
                            second_knn_rgb[y, x] = [int(rgb_val[0]), int(rgb_val[1]), int(rgb_val[2])]
                second_knn_rgb_path = os.path.join(ref_dir, f"{image_name}_KNN__second_rgb.png")
                Image.fromarray(second_knn_rgb, 'RGB').save(second_knn_rgb_path)
                processing_info['second_knn_rgb_path'] = second_knn_rgb_path
                print(f"Saved second KNN result with original RGB to: {second_knn_rgb_path}")
                
                # Save clustered_overlay_holes.png as _KNN__clustered.png in reference folder
                if holes_result is not None:
                    holes_output_path = os.path.join(ref_dir, f"{image_name}_KNN__clustered.png")
                    holes_pil = Image.fromarray(holes_result, 'RGB')
                    holes_pil.save(holes_output_path)
                    results['clustered_overlay_holes'] = holes_result
                    processing_info['holes_detection_path'] = holes_output_path

                # Save RGB detected mask if it exists (only when hole detection is enabled)
                if enable_hole_detection and rgb_detected_mask_from_holes is not None:
                    # If sparsity detection is also enabled and produced a mask, we might want to merge them
                    # But currently sparsity is disabled in main_RELP.py.
                    # Let's save this one.
                    rgb_mask_path = os.path.join(output_dir, f"{image_name}_RGB_detected.png")
                    rgb_pil = Image.fromarray(rgb_detected_mask_from_holes, 'RGB')
                    rgb_pil.save(rgb_mask_path)
                    results['rgb_detected_mask'] = rgb_detected_mask_from_holes
                    print(f"RGB detected mask saved to: {rgb_mask_path}")

                # Save reconstructed_knn.png - DISABLED as per user request
                if reconstructed_knn_result is not None:
                    # recon_output_path = os.path.join(output_dir, f"{image_name}_reconstructed_knn.png")
                    # recon_pil = Image.fromarray(reconstructed_knn_result, 'RGB')
                    # recon_pil.save(recon_output_path)
                    results['reconstructed_knn'] = reconstructed_knn_result
                    # processing_info['reconstructed_knn_path'] = recon_output_path

                # Save other results to results dictionary
                results['contour_detection'] = contour_result
                results['contour_only'] = contour_only_result
        else:
            # If no output directory provided, use simplified connected component detection
            contour_detection_result, contour_only_result = perform_contour_detection_on_image(final_overlay)
            results['contour_detection'] = contour_detection_result
            results['contour_only'] = contour_only_result

        # Step 9: Sparsity detection
        if enable_sparsity:
            original_non_black_mask = ~np.all(image_rgb == [0, 0, 0], axis=2)
            
            # Create an eroded mask to remove edge artifacts from sparsity detection
            # The sparsity calculation uses kernels up to size 16, so the density drops at the edges (artificial edges from ROI).
            # We erode the mask by a larger margin to ensure we ignore these edge artifacts.
            sparsity_erosion_size = 40
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sparsity_erosion_size, sparsity_erosion_size))
            sparsity_valid_mask = cv2.erode(original_non_black_mask.astype(np.uint8), erosion_kernel).astype(bool)
            
            clustered_non_black_mask = ~np.all(final_overlay == [0, 0, 0], axis=2)
            clustered_coords = np.column_stack(np.where(clustered_non_black_mask))

            processing_info['clustered_coords'] = len(clustered_coords)

            if len(clustered_coords) > 0:
                # Perform sparsity detection
                # Use sparsity_valid_mask to filter out edge artifacts
                sparsity_map, sparsity_visualization, anomaly_regions, anomaly_mask, selected_regions_mask, rgb_detected_mask_sparsity = create_sparsity_detection_map(
                    clustered_coords, height, width, sparsity_valid_mask,
                    original_image=image_rgb,
                    target_rgb=target_rgb,
                    rgb_distance_threshold=rgb_distance_threshold
                )

                # Use dynamic area threshold for one-time filtering
                filtered_anomaly_regions = filter_anomaly_regions_by_dynamic_area(
                    anomaly_regions, contour_area, sparsity_area_threshold
                )

                # Recreate binary mask based on filtered regions
                selected_regions_mask = np.zeros_like(sparsity_map, dtype=np.uint8)
                for region in filtered_anomaly_regions:
                    contour = region['contour']
                    cv2.fillPoly(selected_regions_mask, [contour], 255)

                results['sparsity_map'] = sparsity_visualization
                results['anomaly_regions'] = filtered_anomaly_regions
                results['selected_regions_mask'] = selected_regions_mask
                processing_info['anomaly_regions_count'] = len(filtered_anomaly_regions)
                processing_info['original_anomaly_regions_count'] = len(anomaly_regions)

                # Save binary mask
                if output_dir and image_name:
                    save_selected_regions_mask(selected_regions_mask, output_dir, image_name)

                # If anomaly regions are detected, generate visualization results
                if len(filtered_anomaly_regions) > 0:
                    clustered_overlay_bgr = cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR)
                    anomaly_visualization = visualize_anomaly_regions(
                        clustered_overlay_bgr,
                        sparsity_map / 255.0,
                        filtered_anomaly_regions,
                        anomaly_mask
                    )
                    # Convert back to RGB format
                    results['anomaly_detection'] = cv2.cvtColor(anomaly_visualization, cv2.COLOR_BGR2RGB)
                else:
                    processing_info['anomaly_note'] = "No obvious anomaly regions detected"
            else:
                processing_info['sparsity_note'] = "Not enough pixels in clustered_overlay for sparsity detection"
        else:
             processing_info['sparsity_note'] = "Sparsity detection disabled by user"

        results['processing_info'] = processing_info

        # Return key data in numpy format
        clustered_overlay_holes = results.get('clustered_overlay_holes')
        anomaly_detection = results.get('anomaly_detection')
        selected_regions_mask = results.get('selected_regions_mask')
        rgb_detected_mask = results.get('rgb_detected_mask')

        return results, clustered_overlay_holes, anomaly_detection, selected_regions_mask

    except Exception as e:
        print(f"CRITICAL ERROR in process_image_complete: {e}")
        import traceback
        traceback.print_exc()
        error_results = {
            'clustered_overlay': None,
            'clustered_overlay_holes': None,
            'reconstructed_knn': None,
            'clusters_only': None,
            'intermediate_overlay': None,
            'contour_detection': None,
            'contour_only': None,
            'sparsity_map': None,
            'anomaly_detection': None,
            'anomaly_regions': [],
            'selected_regions_mask': None,
            'rgb_detected_mask': None,
            'processing_info': {'error': f"Error occurred during processing: {str(e)}"}
        }
        return error_results, None, None, None


def perform_contour_detection_on_image(image_array):
    """
    Perform contour detection directly on image array, simulating core functionality of process_image_with_contour_detection
    
    Args:
        image_array: Input image array (RGB format)
    
    Returns:
        tuple: (contour_detection_result, contour_only_result)
    """
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
            
        # Binarization
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create contour detection result images
        contour_detection = image_array.copy()
        contour_only = np.zeros_like(image_array)
        
        if contours:
            # Draw contours on original image
            cv2.drawContours(contour_detection, contours, -1, (255, 0, 0), 2)
            # Create pure contour image
            cv2.drawContours(contour_only, contours, -1, (255, 255, 255), 2)
            
        return contour_detection, contour_only
        
    except Exception as e:
        # If error occurs, return original image and empty image
        return image_array.copy(), np.zeros_like(image_array)
