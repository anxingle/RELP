import re
from detectron2.data import MetadataCatalog


import pickle

import pandas as pd

import torch

import cv2
import numpy as np
from .cv_ops import load_and_validate_image

import os
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
except ImportError:
    pass

def detectron_mask(pic_dir, model_detectron, debug_mode=False, output_dir=None):
    print(f"DEBUG: detectron_mask called. debug_mode={debug_mode}, output_dir={output_dir}")
    img = load_and_validate_image(pic_dir)
    if img is None:
        return 0, 0, np.zeros((100, 100, 3), dtype=np.uint8)
    mask_height, mask_width = img.shape[:2]
    pic_cv = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    # Detectron2 prediction
    outputs = model_detectron(img)
    instances = outputs["instances"]

    # Filter by confidence threshold (default 50%)
    score_threshold = 0.5
    if instances.has("scores"):
        scores = instances.scores
        high_conf_indices = scores >= score_threshold
        instances = instances[high_conf_indices]
        print(f"DEBUG: Filtered instances with score < {score_threshold}. Remaining: {len(instances)}")

    if debug_mode and output_dir:
        try:
             ref_folder = os.path.join(output_dir, "reference")
             if not os.path.exists(ref_folder):
                 os.makedirs(ref_folder)
             
             from detectron2.utils.visualizer import Visualizer
             from detectron2.data import MetadataCatalog
             
             # Get metadata
             metadata = None
             if hasattr(model_detectron, 'cfg'):
                  if len(model_detectron.cfg.DATASETS.TEST) > 0:
                       metadata = MetadataCatalog.get(model_detectron.cfg.DATASETS.TEST[0])
                  elif len(model_detectron.cfg.DATASETS.TRAIN) > 0:
                       metadata = MetadataCatalog.get(model_detectron.cfg.DATASETS.TRAIN[0])
             
             # Visualize
             # img is BGR, Visualizer needs RGB
             v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
             out = v.draw_instance_predictions(instances.to("cpu"))
             
             # Save
             # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
             fname = os.path.splitext(os.path.basename(pic_dir))[0]
             save_path = os.path.join(ref_folder, f"{fname}_defect_seg_debug.png")
             
             # out.get_image() returns RGB, cv2.imwrite expects BGR
             cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
             print(f"DEBUG: Saved Detectron debug visualization to {save_path}")
        except Exception as e:
             print(f"Warning: Failed to save Detectron debug visualization: {e}")

    if len(instances) > 0 and instances.has("pred_masks"):
        # Get masks and classes
        masks = instances.pred_masks.cpu().numpy()  # [N, H, W]
        classes = instances.pred_classes.cpu().numpy()  # [N]

        # Find defect indices (class 0)
        defect_indices = np.where(classes == 0)[0]

        if len(defect_indices) > 0:
            # Combine all defect masks
            defect_masks = masks[defect_indices]
            combined_mask = np.any(defect_masks, axis=0).astype(np.uint8) * 255

            # Convert to 3-channel image
            pic_cv = cv2.merge([combined_mask, combined_mask, combined_mask])
            mask_height, mask_width = combined_mask.shape

    return mask_height, mask_width, pic_cv


def detectron2_predict_parsing(yolo_prediction, model, cfg_path_dut=None, weights_path_dut=None):
    prediction_df = pd.DataFrame()

    if cfg_path_dut is None:
        try:
            import utils.utils_general as ug
            cfg_path_dut = ug.cfg_path_dut
        except AttributeError:
            raise ValueError("cfg_path_dut is required for detectron2_predict_parsing")
    with open(cfg_path_dut, 'rb') as f:
        cfg_dut = pickle.load(f)
    if weights_path_dut is None:
        try:
            import utils.utils_general as ug
            weights_path_dut = getattr(ug, "weights_path_dut", None)
        except AttributeError:
            pass

    # Get class names from config instead of model
    if hasattr(cfg_dut, 'DATASETS'):
        try:
            # Try to get class names from metadata
            from detectron2.data import MetadataCatalog
            metadata = MetadataCatalog.get(cfg_dut.DATASETS.TRAIN[0])
            class_names = metadata.thing_classes
        except:
            # Fallback: create generic class names
            num_classes = cfg_dut.MODEL.ROI_HEADS.NUM_CLASSES
            class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        # Default fallback
        class_names = ["phone_case"]  # Adjust based on your classes

    # --- Start of RTF Loading Logic ---
    rtf_candidates = [
        os.path.join(os.path.dirname(cfg_path_dut), 'classes.rtf'), # Same dir as config
        os.path.join(os.path.dirname(weights_path_dut) if weights_path_dut else '', 'classes.rtf'), # Same dir as weights
        os.path.join(os.getcwd(), 'classes.rtf') # Current working dir
    ]
    
    loaded_map = {}
    for rtf_path in rtf_candidates:
        if rtf_path and os.path.exists(rtf_path):
            print(f"Loading class names from {rtf_path}")
            try:
                with open(rtf_path, 'r', errors='ignore') as f:
                    content = f.read()
                    import re
                    matches = re.findall(r'(\d+):\s*([a-zA-Z0-9_]+)', content)
                    for num_str, name in matches:
                        loaded_map[int(num_str)] = name
                break 
            except Exception as e:
                print(f"Error reading {rtf_path}: {e}")

    if loaded_map:
        max_id = max(loaded_map.keys())
        # Ensure list is long enough
        if len(class_names) <= max_id:
             class_names.extend([f"class_{i}" for i in range(len(class_names), max_id + 1)])
        
        for cid, cname in loaded_map.items():
            if cid < len(class_names):
                class_names[cid] = cname
    # --- End of RTF Loading Logic ---

    for result in yolo_prediction:
        for i, prediction in enumerate(result.boxes):
            sub_predict_df = pd.DataFrame()
            xyxy_list = prediction.xyxy[0].tolist()
            # print('xyxy', xyxy_list)
            xyxy_2d = np.array([xyxy_list])
            xyxy = xyxy_2d
            # print('resized xyxy', xyxy)
            x1, y1, x2, y2 = xyxy[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_array = np.array([center_x, center_y]).astype(int)
            center_df = pd.DataFrame({"bbox_center": [center_array]})
            xyxy = xyxy.flatten()
            xyxy = [int(float(item)) for item in xyxy]
            xyxy_df = pd.DataFrame({"bbox": [xyxy]})
            conf = prediction.conf.item()
            conf_df = pd.DataFrame({"conf": [conf]})
            cls = int(prediction.cls.item())

            if model == "feature":
                # You'll need to handle feature model similarly
                label = "feature_class"  # Adjust as needed
            elif model == "DUT":
                # Use class names from config
                if cls < len(class_names):
                    label = class_names[cls]
                else:
                    label = f"class_{cls}"

            label_df = pd.DataFrame({"label": [label]})
            cls_df = pd.DataFrame({"class_id": [cls]})
            sub_predict_df = pd.concat([xyxy_df, conf_df, center_df, label_df, cls_df], axis=1)
            prediction_df = pd.concat([prediction_df, sub_predict_df], axis=0).reset_index(drop=True)

    print("prediction_df", prediction_df)
    return prediction_df


def detectron_to_yolo_format(detectron_output, conf_threshold=0.25):
    class Results:
        def __init__(self):
            self.boxes = None

    class Boxes:
        def __init__(self, boxes_list, all_boxes_tensor):
            self.boxes_list = boxes_list
            self.xyxy = all_boxes_tensor  # Add xyxy attribute containing all boxes
            self.data = all_boxes_tensor  # Add data attribute for compatibility

        def __iter__(self):
            return iter(self.boxes_list)

        def numpy(self):
            return self.xyxy.numpy()

    class Box:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = torch.tensor([xyxy])
            self.conf = torch.tensor([conf])
            self.cls = torch.tensor([cls])

    instances = detectron_output["instances"]
    boxes_list = []
    all_boxes_data = []

    if len(instances) > 0 and instances.has("pred_boxes") and instances.has("scores") and instances.has(
            "pred_classes"):
        boxes = instances.pred_boxes.tensor.cpu()
        scores = instances.scores.cpu()
        classes = instances.pred_classes.cpu()

        for i in range(len(instances)):
            if scores[i] >= conf_threshold:
                xyxy = boxes[i].tolist()
                conf = scores[i].item()
                cls = classes[i].item()
                boxes_list.append(Box(xyxy, conf, cls))
                all_boxes_data.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])

    all_boxes_tensor = torch.tensor(all_boxes_data) if all_boxes_data else torch.tensor([])

    results_obj = Results()
    results_obj.boxes = Boxes(boxes_list, all_boxes_tensor)

    return [results_obj]


def width_lines_defect_detectron(
        input_image_folder: str,
        output_base_folder: str = 'bumper_lines_results',
        confidence_threshold: float = 0.3,
        image_extensions: tuple = ('.jpg', '.jpeg', '.png'),
        pixel_to_mm: float = 0.0626,
        number_of_width_lines: int = 3
):
    os.makedirs(output_base_folder, exist_ok=True)
    combined_excel = os.path.join(output_base_folder, "bumper_delam_width_measurements.xlsx")

    # Load Detectron2 config from pickle file
    cfg = get_cfg()

    # Load config from pickle file
    with open(cfg_path, 'rb') as f:
        cfg_loaded = pickle.load(f)
    cfg.merge_from_other_cfg(cfg_loaded)

    cfg.MODEL.WEIGHTS = weights_path_crack
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    # 设备选择：优先 CUDA → MPS → CPU
    device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg.MODEL.DEVICE = device_str

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Create predictor
    predictor = DefaultPredictor(cfg)

    input_path = Path(input_image_folder)
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {input_image_folder} or its subfolders")

    all_measurements = []

    # Process each image
    for img_file in image_files:
        try:
            rel_folder = img_file.parent.relative_to(input_path)
            folder_prefix = "_".join(rel_folder.parts) if rel_folder != Path(".") else ""
            image_display_name = f"{folder_prefix}/{img_file.name}" if folder_prefix else img_file.name

            relative_path = str(rel_folder / img_file.name).replace("\\", "/")
            product, build, config_val, color, test_item, unit_no, test_cycles, photo_position = parse_filename_info(
                relative_path)

            image = load_and_validate_image(str(img_file))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run Detectron2 inference
            outputs = predictor(image_rgb)
            instances = outputs["instances"].to("cpu")

            if not instances.has("pred_masks") or len(instances.pred_masks) == 0:
                continue

            masks = instances.pred_masks.numpy()  # Shape: (N, H, W)

            for defect_id in range(len(masks)):
                mask = masks[defect_id].astype(np.uint8) * 255
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) == 0:
                    continue

                top_y = y_coords.min()
                bottom_y = y_coords.max()
                line_ys = np.linspace(top_y, bottom_y, number_of_width_lines).astype(int)

                for line_idx, y in enumerate(line_ys):
                    y = max(0, min(y, mask.shape[0] - 1))
                    row = mask[y, :]
                    x_indices = np.where(row > 0)[0]

                    if len(x_indices) > 0:
                        width_px = x_indices[-1] - x_indices[0]
                        width_mm = width_px * pixel_to_mm
                    else:
                        width_px = 0

                    all_measurements.append({
                        'Image': image_display_name,
                        'Defect_ID': defect_id + 1,
                        'Line_Number': line_idx + 1,
                        'Width_Pixel': width_px,
                        'Width_mm': width_mm,
                        'Product': product,
                        'Build': build,
                        'Config': config_val,
                        'Color': color,
                        'Test_Item': test_item,
                        'Unit_No': unit_no,
                        'Test_Cycles': test_cycles,
                        'Photo_Position': photo_position
                    })

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    if all_measurements:
        df_combined = pd.DataFrame(all_measurements)

        if os.path.exists(combined_excel):
            try:
                existing_df = pd.read_excel(combined_excel)
                new_images = set(df_combined['Image'].unique())
                filtered_existing_df = existing_df[~existing_df['Image'].isin(new_images)]
                final_df = pd.concat([filtered_existing_df, df_combined], ignore_index=True)
                print(
                    f"Incremental update: removed {len(existing_df) - len(filtered_existing_df)} duplicate records, added {len(df_combined)} new records")
            except Exception as e:
                print(f"Error reading existing Excel file, creating new one: {e}")
                final_df = df_combined
        else:
            final_df = df_combined
            print(f"Creating new Excel file with {len(df_combined)} records")

        final_df.to_excel(combined_excel, index=False)
        print(f"Width measurements saved to: {combined_excel}")


def length_size_of_defect_detectron(
        input_image_folder: str,
        output_base_folder: str = 'bumper_lines_results',
        pixel_to_mm: float = 0.0626
):
    # Load Detectron2 config from pickle file
    cfg = get_cfg()

    # Load config from pickle file
    with open(cfg_path, 'rb') as f:
        cfg_loaded = pickle.load(f)
    cfg.merge_from_other_cfg(cfg_loaded)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Default confidence threshold
    # 设备选择：优先 CUDA → MPS → CPU
    device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg.MODEL.DEVICE = device_str

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Create output base directory
    os.makedirs(output_base_folder, exist_ok=True)

    results_list = []

    def calculate_metrics(masks, image_shape):
        metrics = []
        for mask in masks:
            # Convert mask tensor to numpy and resize
            mask_array = mask.astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_array, (image_shape[1], image_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized > 0).astype(np.uint8)

            # Label connected components
            labeled = label(binary_mask)
            regions = regionprops(labeled)

            if not regions:
                continue

            prop = max(regions, key=lambda r: r.area)

            length_px = prop.major_axis_length
            area_px = prop.area

            length_mm = round(length_px * pixel_to_mm, 2)
            area_mm2 = round(area_px * (pixel_to_mm ** 2), 2)

            metrics.append({
                "length_px": int(length_px),
                "length_mm": length_mm,
                "area_px": int(area_px),
                "area_mm2": area_mm2
            })
        return metrics

    for root, _, files in os.walk(input_image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(root, file)
                relative_img_path = os.path.relpath(img_path, input_image_folder)  # Full subfolder path

                print(f"Processing: {img_path}")

                # 解析文件路径信息
                image_identifier = relative_img_path.replace("\\", "/")  # Normalize path separators
                product, build, config_val, color, test_item, unit_no, test_cycles, photo_position = parse_filename_info(
                    image_identifier)

                try:
                    # Load image
                    image = load_and_validate_image(img_path)
                    if image is None:
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Run Detectron2 inference
                    outputs = predictor(image_rgb)
                    instances = outputs["instances"].to("cpu")

                    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
                        masks = instances.pred_masks.numpy()  # Shape: (N, H, W)
                        metrics_list = calculate_metrics(masks, image.shape[:2])

                        for idx, metrics in enumerate(metrics_list):
                            results_list.append({
                                "image_path": image_identifier,
                                "defect_id": idx + 1,
                                "length_px": metrics["length_px"],
                                "length_mm": metrics["length_mm"],
                                "area_px": metrics["area_px"],
                                "area_mm2": metrics["area_mm2"],
                                "Product": product,
                                "Build": build,
                                "Config": config_val,
                                "Color": color,
                                "Test_Item": test_item,
                                "Unit_No": unit_no,
                                "Test_Cycles": test_cycles,
                                "Photo_Position": photo_position
                            })
                    else:
                        results_list.append({
                            "image_path": image_identifier,
                            "defect_id": 0,
                            "length_px": 0,
                            "length_mm": 0.0,
                            "area_px": 0,
                            "area_mm2": 0.0,
                            "Product": product,
                            "Build": build,
                            "Config": config_val,
                            "Color": color,
                            "Test_Item": test_item,
                            "Unit_No": unit_no,
                            "Test_Cycles": test_cycles,
                            "Photo_Position": photo_position
                        })

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    results_list.append({
                        "image_path": image_identifier,
                        "defect_id": -1,  # Error flag
                        "length_px": None,
                        "length_mm": None,
                        "area_px": None,
                        "area_mm2": None,
                        "Product": product,
                        "Build": build,
                        "Config": config_val,
                        "Color": color,
                        "Test_Item": test_item,
                        "Unit_No": unit_no,
                        "Test_Cycles": test_cycles,
                        "Photo_Position": photo_position
                    })

    if results_list:
        df = pd.DataFrame(results_list)
        excel_path = os.path.join(output_base_folder, "Bumper_length_and_measurements.xlsx")

        if os.path.exists(excel_path):
            try:
                existing_df = pd.read_excel(excel_path)

                new_images = set(df['image_path'].unique())

                filtered_existing_df = existing_df[~existing_df['image_path'].isin(new_images)]

                final_df = pd.concat([filtered_existing_df, df], ignore_index=True)

                print(f"增量更新：移除了 {len(existing_df) - len(filtered_existing_df)} 条重复图片记录，添加了 {len(df)} 条新记录")
            except Exception as e:
                print(f"读取现有Excel文件时出错，将创建新文件: {e}")
                final_df = df
        else:
            final_df = df
            print(f"创建新的Excel文件，包含 {len(df)} 条记录")

        final_df.to_excel(excel_path, index=False)
        print(f"长度和面积测量结果已保存到: {excel_path}")
    else:
        print("No results to save.")

    return None


def bumper_defect_analysis_detectron(
        input_image_folder: str,
        output_base_folder: str = 'bumper_lines_results',
        confidence_threshold: float = 0.3,
        pixel_to_mm: float = 0.0626,
        number_of_width_lines: int = 3,
        Failure_Mode = "Bumper_Crack"
):
    # Load Detectron2 config from pickle file
    print(f"Loading crack cfg from: {cfg_path_crack}")
    with open(cfg_path_crack, 'rb') as f:
        cfg = pickle.load(f)  # use pickled cfg directly to avoid key mismatch

    # 覆盖运行时参数
    print(f"Setting weights: {weights_path_crack}")
    cfg.MODEL.WEIGHTS = weights_path_crack
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cpu"

    # Create predictor
    predictor = DefaultPredictor(cfg)
    print("Created predictor", predictor )


    # Create output base directory
    os.makedirs(output_base_folder, exist_ok=True)

    # Create masked results directory
    masked_output_folder = os.path.join(output_base_folder, "Inferred Pic")
    os.makedirs(masked_output_folder, exist_ok=True)

    # Lists for both types of measurements
    width_measurements = []
    length_area_measurements = []

    def calculate_width_metrics(masks, classes, image_shape, image_display_name, product, build, config_val,
                                color, test_item, unit_no, test_cycles, photo_position):
        """Calculate width measurements at multiple horizontal lines"""
        measurements = []

        for defect_id in range(len(masks)):
            mask = (masks[defect_id] > 0).astype(np.uint8) * 255
            class_id = classes[defect_id] if classes is not None else -1
            
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                continue

            top_y = y_coords.min()
            bottom_y = y_coords.max()
            line_ys = np.linspace(top_y, bottom_y, number_of_width_lines).astype(int)
            print("Width lines process", line_ys )

            for line_idx, y in enumerate(line_ys):
                y = max(0, min(y, mask.shape[0] - 1))
                row = mask[y, :]
                x_indices = np.where(row > 0)[0]

                if len(x_indices) > 0:
                    width_px = x_indices[-1] - x_indices[0]
                    width_mm = width_px * pixel_to_mm
                else:
                    width_px = 0
                    width_mm = 0.0

                measurements.append({
                    'Image': image_display_name,
                    'Defect_ID': defect_id + 1,
                    'Class_ID': int(class_id),
                    'Line_Number': line_idx + 1,
                    'Width_Pixel': width_px,
                    'Width_mm': width_mm,
                    'Product': product,
                    'Build': build,
                    'Config': config_val,
                    'Color': color,
                    'Test_Item': test_item,
                    'Unit_No': unit_no,
                    'Test_Cycles': test_cycles,
                    'Photo_Position': photo_position
                })
        return measurements

    def calculate_length_area_metrics(masks, classes, image_shape):
        """Calculate length, width, area measurements and contours"""
        metrics = []
        h, w = image_shape
        # Define full image rectangle for normalization
        full_image_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        for i, mask in enumerate(masks):
            class_id = classes[i] if classes is not None else -1
            
            mask_array = mask.astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_array, (w, h), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized > 0).astype(np.uint8)

            labeled = label(binary_mask)
            regions = regionprops(labeled)

            if not regions:
                continue

            prop = max(regions, key=lambda r: r.area)

            length_px = prop.major_axis_length
            width_px = prop.minor_axis_length
            area_px = prop.area

            length_mm = round(length_px * pixel_to_mm, 2)
            width_mm = round(width_px * pixel_to_mm, 2)
            area_mm2 = round(area_px * (pixel_to_mm ** 2), 2)
            
            # Extract and normalize contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            normalized_cnt = []
            if contours:
                # Find the contour corresponding to the largest region
                # (Or just take the largest contour)
                c_areas = [cv2.contourArea(c) for c in contours]
                idx = int(np.argmax(c_areas))
                cnt = contours[idx]
                
                # Normalize
                try:
                    normalized_cnt = normalize_contours(full_image_rect, [cnt])[0]
                except Exception as e:
                    print(f"Error normalizing contour: {e}")
                    normalized_cnt = []

            metrics.append({
                "length_px": int(length_px),
                "length_mm": length_mm,
                "width_px": int(width_px),
                "width_mm": width_mm,
                "area_px": int(area_px),
                "area_mm2": area_mm2,
                "Contour": str(normalized_cnt),
                "Class_ID": int(class_id)
            })
        return metrics

    def save_segmentation_result(image, masks, output_path):
        """Save segmentation visualization"""
        if len(masks) == 0:
            cv2.imwrite(output_path, image)
            return

        # Create a combined mask
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            mask_resized = cv2.resize((mask > 0).astype(np.uint8) * 255,
                                      (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, mask_resized)

        # Create overlay
        overlay = image.copy()
        overlay[combined_mask > 0] = [0, 255, 0]  # Green overlay for segmentation

        # Blend original image with overlay
        alpha = 0.6
        result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

        # Save result
        cv2.imwrite(output_path, result)

    input_path = Path(input_image_folder)
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif')]

    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {input_image_folder} or its subfolders")

    # Process each image
    for img_file in image_files:
        try:
            rel_folder = img_file.parent.relative_to(input_path)
            folder_prefix = "_".join(rel_folder.parts) if rel_folder != Path(".") else ""
            image_display_name = f"{folder_prefix}/{img_file.name}" if folder_prefix else img_file.name
            print("image_display_name", image_display_name)
            # Create corresponding output path in masked folder
            relative_img_path = str(rel_folder / img_file.name).replace("\\", "/")
            relative_dir = str(rel_folder).replace("\\", "/") if str(rel_folder) != "." else ""
            masked_output_dir = os.path.join(masked_output_folder, relative_dir)
            os.makedirs(masked_output_dir, exist_ok=True)
            masked_output_path = os.path.join(masked_output_dir, img_file.name)

            product, build, config_val, color, test_item, unit_no, test_cycles, photo_position = parse_filename_info(
                relative_img_path)

            # Load image
            image = load_and_validate_image(img_file)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run Detectron2 inference
            outputs = predictor(image_rgb)
            print("Detectron2 inference outputs process..")
            instances = outputs["instances"].to("cpu")
            print("Detectron2 inference instances process..")


            if not instances.has("pred_masks") or len(instances.pred_masks) == 0:
                # Save original image if no masks found
                cv2.imwrite(masked_output_path, image)

                # Add zero measurements for both types
                width_measurements.append({
                    'Image': image_display_name,
                    'Defect_ID': 0,
                    'Class_ID': -1,
                    'Line_Number': 1,
                    'Width_Pixel': 0,
                    'Width_mm': 0.0,
                    'Product': product,
                    'Build': build,
                    'Config': config_val,
                    'Color': color,
                    'Test_Item': test_item,
                    'Unit_No': unit_no,
                    'Test_Cycles': test_cycles,
                    'Photo_Position': photo_position
                })

                length_area_measurements.append({
                    "image_path": relative_img_path,
                    "Image_Name": image_display_name,
                    "defect_id": 0,
                    "Class_ID": -1,
                    "length_px": 0,
                    "length_mm": 0.0,
                    "width_px": 0,
                    "width_mm": 0.0,
                    "area_px": 0,
                    "area_mm2": 0.0,
                    "Contour": "[]",
                    "Product": product,
                    "Build": build,
                    "Config": config_val,
                    "Color": color,
                    "Test_Item": test_item,
                    "Unit_No": unit_no,
                    "Test_Cycles": test_cycles,
                    "Photo_Position": photo_position
                })
                continue

            masks = instances.pred_masks.numpy()
            if instances.has("pred_classes"):
                classes = instances.pred_classes.numpy()
            else:
                classes = None

            refined = []
            for defect_id in range(len(masks)):
                m = (masks[defect_id] > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if contours:
                    areas = [cv2.contourArea(c) for c in contours]
                    idx = int(np.argmax(areas))
                    coarse = contours[idx]
                    coarse_area = float(areas[idx])
                    
                    # Resolve Segmentation Mode from Global SAM_Input
                    seg_mode = "Point" # Default
                    input_data = None
                    
                    if SAM_Input is not None:
                        clean_sam_input = str(SAM_Input).strip().strip("'").strip('"').upper()
                        if clean_sam_input in ["POINTS", "POINT"]:
                            seg_mode = "Point"
                        elif clean_sam_input in ["BBOX", "BOX"]:
                            seg_mode = "Box"
                        elif clean_sam_input in ["CONTOUR", "CNTS"]:
                            seg_mode = "Contour"
                        else:
                            seg_mode = "Contour" # Fallback to Contour if unknown
                    
                    # Prepare input data based on mode
                    if seg_mode == "Point":
                        pts = contour_centroid(coarse)
                        input_data = pts
                    elif seg_mode == "Box":
                        x, y, w, h = cv2.boundingRect(coarse)
                        input_data = np.array([x, y, x + w, y + h])
                    elif seg_mode == "Contour":
                        input_data = coarse
                        
                    print(f"DEBUG: bumper_defect calling fine_seg with mode='{seg_mode}'")
                    refined_cnt = fine_seg(image_rgb, input_data, coarse_area, coarse, seg_mode)
                    
                    refined_mask = np.zeros_like(m)
                    try:
                        cv2.drawContours(refined_mask, [refined_cnt], -1, 255, cv2.FILLED)
                        refined.append((refined_mask > 0).astype(np.uint8))
                    except Exception:
                        refined.append((m > 0).astype(np.uint8))
                else:
                    refined.append((m > 0).astype(np.uint8))
            masks = np.stack(refined, axis=0)

            # Save segmentation result
            save_segmentation_result(image, masks, masked_output_path)

            # Calculate width measurements
            width_results = calculate_width_metrics(
                masks, classes, image.shape[:2], image_display_name, product, build, config_val,
                color, test_item, unit_no, test_cycles, photo_position
            )
            width_measurements.extend(width_results)

            # Calculate length/area measurements
            length_area_results = calculate_length_area_metrics(masks, classes, image.shape[:2])

            if length_area_results:
                for idx, metrics in enumerate(length_area_results):
                    length_area_measurements.append({
                        "image_path": relative_img_path,
                        "Image_Name": image_display_name,
                        "defect_id": idx + 1,
                        "Class_ID": metrics.get("Class_ID", -1),
                        "length_px": metrics["length_px"],
                        "length_mm": metrics["length_mm"],
                        "width_px": metrics["width_px"],
                        "width_mm": metrics["width_mm"],
                        "area_px": metrics["area_px"],
                        "area_mm2": metrics["area_mm2"],
                        "Contour": metrics["Contour"],
                        "Product": product,
                        "Build": build,
                        "Config": config_val,
                        "Color": color,
                        "Test_Item": test_item,
                        "Unit_No": unit_no,
                        "Test_Cycles": test_cycles,
                        "Photo_Position": photo_position
                    })
            else:
                # Add zero measurements if no valid regions found
                length_area_measurements.append({
                    "image_path": relative_img_path,
                    "Image_Name": image_display_name,
                    "defect_id": 0,
                    "Class_ID": -1,
                    "length_px": 0,
                    "length_mm": 0.0,
                    "width_px": 0,
                    "width_mm": 0.0,
                    "area_px": 0,
                    "area_mm2": 0.0,
                    "Contour": "[]",
                    "Product": product,
                    "Build": build,
                    "Config": config_val,
                    "Color": color,
                    "Test_Item": test_item,
                    "Unit_No": unit_no,
                    "Test_Cycles": test_cycles,
                    "Photo_Position": photo_position
                })

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Save width measurements
    if width_measurements:
        df_width = pd.DataFrame(width_measurements)
        width_excel = os.path.join(output_base_folder, "bumper_delam_width_measurements.xlsx")

        if os.path.exists(width_excel):
            try:
                existing_df = pd.read_excel(width_excel)
                new_images = set(df_width['Image'].unique())
                filtered_existing_df = existing_df[~existing_df['Image'].isin(new_images)]
                final_df = pd.concat([filtered_existing_df, df_width], ignore_index=True)
                print(
                    f"Width measurements - Incremental update: removed {len(existing_df) - len(filtered_existing_df)} duplicate records, added {len(df_width)} new records")
            except Exception as e:
                print(f"Error reading existing width Excel file, creating new one: {e}")
                final_df = df_width
        else:
            final_df = df_width
            print(f"Creating new width Excel file with {len(df_width)} records")

        final_df.to_excel(width_excel, index=False)
        print(f"Width measurements saved to: {width_excel}")

    # Save length/area measurements
    if length_area_measurements:
        df_length_area = pd.DataFrame(length_area_measurements)
        length_area_excel = os.path.join(output_base_folder, "Bumper_length_and_measurements.xlsx")

        if os.path.exists(length_area_excel):
            try:
                existing_df = pd.read_excel(length_area_excel)
                new_images = set(df_length_area['image_path'].unique())
                filtered_existing_df = existing_df[~existing_df['image_path'].isin(new_images)]
                final_df = pd.concat([filtered_existing_df, df_length_area], ignore_index=True)
                print(
                    f"Length/Area measurements - Incremental update: removed {len(existing_df) - len(filtered_existing_df)} duplicate records, added {len(df_length_area)} new records")
            except Exception as e:
                print(f"Error reading existing length/area Excel file, creating new one: {e}")
                final_df = df_length_area
        else:
            final_df = df_length_area
            print(f"Creating new length/area Excel file with {len(df_length_area)} records")

        final_df.to_excel(length_area_excel, index=False)
        print(f"Length/Area measurements saved to: {length_area_excel}")

    # Generate parametric_out.xlsx
    if length_area_measurements:
        try:
            para_df = pd.DataFrame(length_area_measurements)
            
            # 1. Generate standard 'Defect' column (e.g. defect_1)
            para_df['Defect'] = para_df['defect_id'].apply(lambda x: f"defect_{x}" if x > 0 else "no_detection")
            
            # 2. Map SN from Unit_No if available, otherwise use image name or index
            if 'Unit_No' in para_df.columns:
                para_df['SN'] = para_df['Unit_No']
            else:
                para_df['SN'] = para_df['Image_Name'] # Fallback
            
            # 3. Rename columns to match Standard Format (Pixels -> Defect_*, mm -> abs_defect_*)
            # Standard format: Defect_Area (px), abs_defect_Area (mm)
            para_df.rename(columns={
                'Image_Name': 'Pic_Name',
                'area_px': 'Defect_Area',
                'length_px': 'Defect_Length',
                'width_px': 'Defect_Width',
                'area_mm2': 'abs_defect_Area', 
                'length_mm': 'abs_defect_Length',
                'width_mm': 'abs_defect_Width'
            }, inplace=True)
            
            # 4. Add other standard columns if missing
            para_df['Defect_Type'] = Failure_Mode
            
            # 5. Select and Order columns to match standard output as closely as possible
            # Standard often has: SN, Defect, Defect_Area, Contour, Defect_Length, Defect_Width, DUT_..., Ratio_..., abs_...
            cols_ordered = [
                'SN', 'Defect', 'Defect_Type', 
                'Defect_Area', 'Contour', 'Defect_Length', 'Defect_Width',
                'abs_defect_Area', 'abs_defect_Length', 'abs_defect_Width',
                'Pic_Name', 'Product', 'Build', 'Config', 'Color', 'Test_Item', 'Test_Cycles', 'Photo_Position'
            ]
            
            # Filter valid columns
            valid_cols = [c for c in cols_ordered if c in para_df.columns]
            final_para_df = para_df[valid_cols]
            
            parametric_excel = os.path.join(output_base_folder, "parametric_out.xlsx")
            final_para_df.to_excel(parametric_excel, index=False)
            print(f"Parametric output saved to: {parametric_excel}")
            
        except Exception as e:
            print(f"Error generating parametric_out.xlsx: {e}")

    # Print where masked results are saved
    print(f"Segmentation results saved to: {masked_output_folder}")

    return None


def load_detectron2_model(cfg_path: str, weights_path: str, score_thresh=0.7):
    try:
        if cfg_path.lower().endswith(('.pkl', '.pickle')):
             with open(cfg_path, 'rb') as f:
                cfg = pickle.load(f)
        else:
             cfg = get_cfg()
             cfg.merge_from_file(cfg_path)
    except Exception as e:
         print(f"Error loading config from {cfg_path}: {e}")
         return None

    cfg.MODEL.WEIGHTS = weights_path
    print(f"obj_det_model weight loaded from: {weights_path}")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cpu"
    
    # --- Class Name Registration from classes.rtf ---
    try:
        # Determine dataset name to register classes to
        dataset_name = None
        if hasattr(cfg, 'DATASETS'):
            if len(cfg.DATASETS.TEST) > 0:
                dataset_name = cfg.DATASETS.TEST[0]
            elif len(cfg.DATASETS.TRAIN) > 0:
                dataset_name = cfg.DATASETS.TRAIN[0]
        
        if dataset_name:
            # Look for classes.rtf in config or weights directory
            search_dirs = [
                os.path.dirname(cfg_path),
                os.path.dirname(weights_path)
            ]
            
            print(f"DEBUG: Looking for classes.rtf in: {search_dirs}")
            
            rtf_path = None
            for d in search_dirs:
                if d and os.path.isdir(d):
                    p = os.path.join(d, 'classes.rtf')
                    if os.path.exists(p):
                        rtf_path = p
                        break
            
            if rtf_path:
                print(f"Loading class names from {rtf_path}")
                with open(rtf_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Debug content
                # print(f"DEBUG: RTF Content: {content[:200]}") # Only print first 200 chars
                
                # Regex to find "ID: Name" patterns
                # Matches: "0: Case" or "1: Cam_Rim"
                matches = re.findall(r'(\d+):\s*([a-zA-Z0-9_]+)', content)
                
                print(f"DEBUG: Regex found matches: {matches}")

                if matches:
                    # Sort by ID
                    matches.sort(key=lambda x: int(x[0]))
                    
                    # Create list of names, filling gaps if necessary
                    max_id = int(matches[-1][0])
                    class_names = ["Unknown"] * (max_id + 1)
                    
                    for cid_str, name in matches:
                        class_names[int(cid_str)] = name
                    
                    # Register to MetadataCatalog
                    # Force overwrite if exists
                    MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
                    print(f"Registered class names for '{dataset_name}': {class_names}")
                else:
                    print("Warning: classes.rtf found but no 'ID: Name' pattern matched.")
            else:
                 print("Warning: classes.rtf not found. Class names will be missing.")
        else:
             print("Warning: Could not determine dataset name from config. Class names not registered.")

    except Exception as e:
        print(f"Error registering class names: {e}")
    # -----------------------------------------------

    return DefaultPredictor(cfg)


def detect_bounding_boxes(image_rgb, detector):
    outputs = detector(image_rgb)
    instances = outputs["instances"]
    
    # DEBUG: Print all detections
    print(f"DEBUG: detect_bounding_boxes found {len(instances)} total instances.")
    
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    
    class_names = None
    if hasattr(detector, 'cfg'):
        try:
             # Attempt to retrieve class names from the first test dataset
             if len(detector.cfg.DATASETS.TEST) > 0:
                 test_ds = detector.cfg.DATASETS.TEST[0]
                 class_names = MetadataCatalog.get(test_ds).thing_classes
        except Exception as e:
             print(f"DEBUG: Could not retrieve class names: {e}")

    for i in range(len(instances)):
        cls_id = pred_classes[i]
        score = scores[i]
        box = boxes[i]
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else "Unknown"
        print(f"  -> Instance {i}: Class ID={cls_id} ({cls_name}), Score={score:.4f}, Box={box}")

    tape_mask = instances.pred_classes == 0
    tape_boxes = instances.pred_boxes[tape_mask].tensor.cpu().numpy()
    tape_classes = instances.pred_classes[tape_mask].cpu().numpy()
    
    print(f"DEBUG: Filtering for Class ID 0. Found {len(tape_boxes)} matching boxes.")
    
    return tape_boxes, tape_classes


def visualize_all_detections(image, detector, class_names=None, score_thresh=0.5):
    """
    Visualize all detections from a detector on an image.
    Draws bounding boxes with class names and confidence scores.
    
    Args:
        image: BGR image (numpy array)
        detector: Detectron2 predictor or similar with inference capability
        class_names: Optional list of class names. If None, will try to get from detector
        score_thresh: Minimum confidence score to display
        
    Returns:
        debug_img: Image with all detections visualized
        detections_info: List of dicts containing detection info
    """
    import cv2
    import numpy as np
    from detectron2.data import MetadataCatalog
    
    debug_img = image.copy()
    detections_info = []
    
    # Run inference
    if detector is None:
        print("DEBUG: Detector is None. Cannot visualize.")
        return debug_img, detections_info
    
    outputs = detector(image)
    instances = outputs["instances"]
    
    if len(instances) == 0:
        print("DEBUG: No detections found for visualization.")
        return debug_img, detections_info
    
    # Get predictions
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    
    # Get class names if not provided
    if class_names is None and hasattr(detector, 'cfg'):
        try:
            if len(detector.cfg.DATASETS.TEST) > 0:
                test_ds = detector.cfg.DATASETS.TEST[0]
                class_names = MetadataCatalog.get(test_ds).thing_classes
        except Exception as e:
            print(f"DEBUG: Could not retrieve class names: {e}")
    
    # Color map for different classes
    colors = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    # Image dimensions for relative scaling
    img_h, img_w = debug_img.shape[:2]
    
    # Calculate a dynamic base scale relative to image size
    # Assuming a 1080p image looks good with font_scale 0.4
    base_dim = max(img_w, img_h)
    dynamic_scale = max(0.3, min(1.0, (base_dim / 1920.0) * 0.4))
    dynamic_thickness = max(1, int(dynamic_scale * 2.5))
    
    # Draw each detection
    for i in range(len(instances)):
        score = scores[i]
        if score < score_thresh:
            continue
            
        cls_id = pred_classes[i]
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name
        if class_names and cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"Class_{cls_id}"
        
        # Select color based on class
        color = colors[cls_id % len(colors)]
        
        # Draw bounding box (Thinner line: 1 instead of 2)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, dynamic_thickness)
        
        # Prepare label text
        label = f"{cls_name}: {score:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Box dimensions
        box_w = x2 - x1
        box_h = y2 - y1
        
        # We want the text to not be absurdly large relative to the box
        # Try base dynamic scale, but shrink it if it's wider than the box (allow up to 2x box width for very thin boxes)
        font_scale = dynamic_scale
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, dynamic_thickness)
        
        # If text is much wider than the box, try to shrink it a bit (but not smaller than 0.25)
        if text_w > box_w * 2 and box_w > 20:
            font_scale = max(0.25, font_scale * (box_w * 1.5 / text_w))
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, dynamic_thickness)
            
        # Avoid drawing text out of image bounds at the top
        text_y = max(y1, text_h + 5)
        
        # Draw filled rectangle for text background (Smaller padding)
        cv2.rectangle(debug_img, (x1, text_y - text_h - 4), (x1 + text_w + 4, text_y + baseline - 2), color, -1)
        
        # Draw text
        cv2.putText(debug_img, label, (x1 + 2, text_y - 2), 
                   font, font_scale, (255, 255, 255), dynamic_thickness, cv2.LINE_AA)
        
        # Store detection info
        detections_info.append({
            'class_id': int(cls_id),
            'class_name': cls_name,
            'score': float(score),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    # Add summary text
    summary = f"Total detections: {len(detections_info)}"
    cv2.putText(debug_img, summary, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(debug_img, summary, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return debug_img, detections_info


def get_detected_class_names(image, detector, score_thresh=0.5):
    """
    Get all detected class names from an image using the detector.
    
    Args:
        image: BGR image (numpy array)
        detector: Detectron2 predictor or similar with inference capability
        score_thresh: Minimum confidence score to include
        
    Returns:
        detected_classes: Set of detected class names (lowercase for matching)
        class_name_to_id: Dictionary mapping class names to class IDs
    """
    from detectron2.data import MetadataCatalog
    
    detected_classes = set()
    class_name_to_id = {}
    
    # Run inference
    if detector is None:
        print("DEBUG: Detector is None. Cannot visualize.")
        return debug_img, detections_info
    
    outputs = detector(image)
    instances = outputs["instances"]
    
    if len(instances) == 0:
        return detected_classes, class_name_to_id
    
    # Get predictions
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    
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
    
    # Collect detected class names
    for i in range(len(instances)):
        score = scores[i]
        if score < score_thresh:
            continue
            
        cls_id = pred_classes[i]
        if class_names and cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = f"Class_{cls_id}"
        
        detected_classes.add(cls_name.lower())
        if cls_name not in class_name_to_id:
            class_name_to_id[cls_name] = []
        class_name_to_id[cls_name].append(int(cls_id))
    
    return detected_classes, class_name_to_id


