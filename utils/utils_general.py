from .output_ops import clustering_Kmeans, SN_list_df, format_parametric_output_df, resolve_sn_from_excel, build_parametric_output_df, build_reference_info, get_output_columns, parametric_output, detect_reference_bbox_area, filter_reference_by_detected_classes, populate_reference_params_from_row, match_reference_by_detected_classes, match_reference_for_image, calculate_parametric_dimensions, format_parametric_dict, parse_scoring_setting, exponential_decay_weights, generate_parametric_row
from utils.config.config_manager import ConfigManager
from utils.wrappers.sam2_wrapper import init_sam2 as _real_init_sam2, get_sam2_predictor, SAM2OnnxPredictor
from utils.wrappers.detectron_wrapper import load_detectron_model
sam2_model = None
from .io_ops import radar_download, radar_upload, unzip, zip_folder, remove_MACOSX, move_file, move_file_keepfolder, delete_empty_folders, path_creator, Count_Subfolders, create_folder, two_steps_path_join, extract_zip_without_macosx, extract_config_group_from_path, extract_product_side_from_path

from .cv_ops import create_overlay_image, save_inferred_pic_overlay, extract_hsv_config_from_filtering, extract_gray_config_from_filtering, apply_gray_filtering_simple, normalize_contours, cv_show, mask_based_matting, contour_based_matting, get_intersection_mask_and_contour, rotate_image_by_topleft, rotate_contour_by_topleft, contour_centroid, show_mask, scaling_bbox_back2_ori_pic, check_pic_SN_existance, selective_masking, gray_lvl_cal, color_detection, show_masks, contour_finding, contour_scale, load_and_validate_image, extract_class_masks, generate_sam2_mask, rotate_image_and_mask, get_color_name, apply_rgb_filtering, apply_hsv_filtering, _save_images_for_operations
from .filtering_ops import parse_filtering_operations, apply_area_filtering, apply_global_threshold_filtering, apply_morph_tophat, apply_filtering, apply_band_roi_filtering, apply_stencil_roi_filtering, parse_filtering_params, apply_bg_subtraction_filtering, apply_odbp_filtering, apply_adaptive_gaussian, apply_relativity_filtering, load_and_filter_filtering_sheet, apply_filter_chain, run_defect_filtering_pipeline, is_line, _parse_save_operations
from .detectron_ops import detectron_mask, detectron2_predict_parsing, detectron_to_yolo_format, width_lines_defect_detectron, length_size_of_defect_detectron, bumper_defect_analysis_detectron, load_detectron2_model, detect_bounding_boxes, visualize_all_detections, get_detected_class_names

from .crop_ops import resize_and_padding, large_obj_cropping, large_obj_cropping_v2, crop_by_bbox, calculate_min_area_rect_correction, check_orientation_and_flip, apply_rotations, crop_and_resize, perform_dut_alignment, preprocess_alignment
from .bbox_ops import order_points, bbox_outlier_supression, select_bbox2_dropout, bbox_dropout, sorting_bbox, bbox_centroid, calculate_intersection_area, scaling_up_bbox, adjust_and_scale_bbox, bbox_extraction_pd, bbox_2darray, is_point_in_bbox


import os
# Enable MPS fallback immediately
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import PIL.Image
import traceback
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import pickle
from detectron2.engine import DefaultPredictor
from PIL import Image
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sklearn.cluster import KMeans
import re
try:
    import radarclient
except ImportError:
    radarclient = None
from datetime import datetime, timedelta
import zipfile
import shutil
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from pathlib import Path
import pillow_heif
Scoring_mode = "Yes"
segement_mode = "Box"
Large_DUT = "No"
Large_DUT_Height_Ratio = 0.5
excel_Pic_paired_FM = ['Delam']
single_Pic_FM = ['Bumper_Crack']
Large_DUT_FM = ['None']
device = "mps"
Failure_Mode = None
download_mode = None
Mainfolder_path = None
silicone_radar_number = None
bumper_radar_number = None
Acceptance_Low = 0.99
Acceptance_High = 1.01
Dino_Threshold_Low = None
Dino_Threshold_High = None
Flooding_By_TH = "No"
Flooding_RGB = None
SAM_Input = None
Upsampling_Target_Size = None # Global variable for custom upsampling size
PCA_Alignment = "No"
Output_Config = []
Reference_Params = {}
DUT_Scaling_Config = {}
Feature_Scaling_Config = {}
Gray_Scale_Params = {}
Reference_BBox_Area = None  # Store Reference bbox area for Area_Ref calculation











class SAM2OnnxPredictor:
    def __init__(self, model_dir, device='cpu'):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is not installed. Please install it to use ONNX models.")

        self.device = device
        
        # Suppress ONNX Runtime warnings
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error level (suppresses Warnings)
        
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == 'mps':
            # MPS support in ORT is limited, default to CPU/CoreML
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

        files = os.listdir(model_dir)
        enc_file = next((f for f in files if 'encoder' in f and f.endswith('.onnx')), None)
        dec_file = next((f for f in files if 'decoder' in f and f.endswith('.onnx')), None)

        if not enc_file or not dec_file:
            raise FileNotFoundError(f"Could not find encoder/decoder ONNX files in {model_dir}")

        print(f"Loading ONNX Encoder: {enc_file}")
        self.encoder = ort.InferenceSession(os.path.join(model_dir, enc_file), sess_options, providers=providers)
        
        print(f"Loading ONNX Decoder: {dec_file}")
        self.decoder = ort.InferenceSession(os.path.join(model_dir, dec_file), sess_options, providers=providers)
        
        self.input_size = (1024, 1024)
        self.original_size = None
        self.features = None
        
        # Determine input names dynamically
        self.enc_input_name = self.encoder.get_inputs()[0].name
        
        # Determine encoder output indices dynamically
        enc_outputs = self.encoder.get_outputs()
        self.enc_output_indices = {}
        for i, out in enumerate(enc_outputs):
            if 'image_embed' in out.name: self.enc_output_indices['image_embeddings'] = i
            elif 'high_res_feats_0' in out.name: self.enc_output_indices['high_res_feats_0'] = i
            elif 'high_res_feats_1' in out.name: self.enc_output_indices['high_res_feats_1'] = i
        
        dec_inputs = self.decoder.get_inputs()
        self.dec_input_names = {}
        for inp in dec_inputs:
            name = inp.name
            if 'image_embed' in name: self.dec_input_names['image_embeddings'] = name
            elif 'high_res_feats_0' in name: self.dec_input_names['high_res_feats_0'] = name
            elif 'high_res_feats_1' in name: self.dec_input_names['high_res_feats_1'] = name
            elif 'point_coord' in name: self.dec_input_names['point_coords'] = name
            elif 'point_label' in name: self.dec_input_names['point_labels'] = name
            elif 'has_mask_input' in name: self.dec_input_names['has_mask_input'] = name
            elif 'mask_input' in name: self.dec_input_names['mask_input'] = name
            elif 'orig_im_size' in name: self.dec_input_names['orig_im_size'] = name

    def _preprocess_image(self, image):
        target_size = 1024
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(image, (new_w, new_h))
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - mean) / std
        
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        img_padded = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        img_chw = img_padded.transpose(2, 0, 1)
        img_batch = img_chw[None, ...]
        return img_batch.astype(np.float32), scale

    def set_image(self, image):
        self.original_size = image.shape[:2]
        img_input, self.scale = self._preprocess_image(image)
        
        # Run encoder
        outputs = self.encoder.run(None, {self.enc_input_name: img_input})
        
        # Map outputs dynamically
        self.features = {}
        if self.enc_output_indices:
            self.features['image_embeddings'] = outputs[self.enc_output_indices['image_embeddings']]
            self.features['high_res_feats_0'] = outputs[self.enc_output_indices['high_res_feats_0']]
            self.features['high_res_feats_1'] = outputs[self.enc_output_indices['high_res_feats_1']]
        else:
            # Fallback to X-AnyLabeling order if names don't match
            # high_res_feats_0, high_res_feats_1, image_embed
            self.features['high_res_feats_0'] = outputs[0]
            self.features['high_res_feats_1'] = outputs[1]
            self.features['image_embeddings'] = outputs[2]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        if self.features is None:
            raise RuntimeError("No image set. Call set_image() first.")

        coords = []
        labels = []
        
        # Process box
        if box is not None:
            box = np.array(box).reshape(-1, 4)
            for b in box:
                coords.append([b[0], b[1]])
                coords.append([b[2], b[3]])
                labels.append(2) # Top-Left
                labels.append(3) # Bottom-Right
        
        # Process points
        if point_coords is not None:
            point_coords = np.array(point_coords).reshape(-1, 2)
            point_labels = np.array(point_labels).reshape(-1)
            for i, p in enumerate(point_coords):
                coords.append(p)
                labels.append(point_labels[i])
        
        if not coords:
             return np.array([]), np.array([]), np.array([])

        coords_np = np.array(coords, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        
        # Scale coords
        coords_np *= self.scale
        
        # Add batch dim
        coords_batch = coords_np[None, :, :]
        labels_batch = labels_np[None, :]
        
        # Prepare other inputs
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        orig_im_size = np.array(self.original_size, dtype=np.int32)[None, :]
        
        inputs = {
            self.dec_input_names.get('image_embeddings'): self.features['image_embeddings'],
            self.dec_input_names.get('high_res_feats_0'): self.features['high_res_feats_0'],
            self.dec_input_names.get('high_res_feats_1'): self.features['high_res_feats_1'],
            self.dec_input_names.get('point_coords'): coords_batch,
            self.dec_input_names.get('point_labels'): labels_batch,
            self.dec_input_names.get('mask_input'): mask_input,
            self.dec_input_names.get('has_mask_input'): has_mask_input,
            self.dec_input_names.get('orig_im_size'): orig_im_size
        }
        
        # Remove None keys (if mapping failed)
        inputs = {k: v for k, v in inputs.items() if k is not None}
        
        # Run decoder
        outputs = self.decoder.run(None, inputs)
        masks = outputs[0] # (B, M, H, W)
        scores = outputs[1] # (B, M)
        
        if not multimask_output:
            best_idx = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best_idx, :, :][:, None, :, :]
            scores = scores[np.arange(scores.shape[0]), best_idx][:, None]
        
        # Post-process masks: Unpad and Resize
        # 256x256 corresponds to 1024x1024 input (padded)
        orig_h, orig_w = self.original_size
        
        # Calculate valid region in the 256x256 mask
        # self.scale = 1024 / max(orig_h, orig_w)
        # new_h = orig_h * scale, new_w = orig_w * scale
        new_h = int(orig_h * self.scale)
        new_w = int(orig_w * self.scale)
        
        # The content in 1024x1024 occupies [0:new_h, 0:new_w]
        # In 256x256 space, it occupies [0:valid_h, 0:valid_w]
        # Ratio is 256/1024 = 0.25
        valid_h = int(new_h * 0.25)
        valid_w = int(new_w * 0.25)
        
        # Ensure at least 1 pixel
        valid_h = max(1, valid_h)
        valid_w = max(1, valid_w)
        
        final_masks = []
        # masks[0] shape is (M, 256, 256)
        for m in masks[0]:
            # Crop valid region
            m_crop = m[:valid_h, :valid_w]
            
            # Resize to original size
            if m_crop.dtype != np.float32:
                m_crop = m_crop.astype(np.float32)
            
            m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            final_masks.append(m_orig)
            
        return np.array(final_masks), scores[0], None

def old_init_sam2(config_path, checkpoint_path, device='cpu'):
    global sam2_model
    try:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
             print(f"Warning: SAM2 checkpoint not found at {checkpoint_path}")
             return
        
        # Check if checkpoint is ONNX directory
        if os.path.isdir(checkpoint_path) or (checkpoint_path.endswith('.onnx')):
            # If path points to file, use its directory
            if os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.dirname(checkpoint_path)
                
            print(f"Initializing SAM2 ONNX from: {checkpoint_path}")
            try:
                sam2_model = SAM2OnnxPredictor(checkpoint_path, device)
                print("SAM2 ONNX Predictor initialized.")
                return
            except Exception as e:
                print(f"Failed to init ONNX predictor: {e}")
                print("Falling back to standard init (will likely fail if ONNX files are not PyTorch checkpoints)...")

        if not config_path or not os.path.exists(config_path):
             print(f"Warning: SAM2 config not found at {config_path}")
             return
        
        # Hydra workaround: switch cwd to config root and pass relative path
        print(f"Initializing SAM2 with Config: {config_path}, Checkpoint: {checkpoint_path}")
        original_cwd = os.getcwd()
        
        # Determine config directory and relative name
        # If 'configs' is in path, use parent of 'configs' as root
        if 'configs' in config_path:
             idx = config_path.rfind('configs')
             config_dir = config_path[:idx]
             config_name = config_path[idx:]
        else:
             config_dir = os.path.dirname(config_path)
             config_name = os.path.basename(config_path)
        
        try:
            print(f"Switching directory to {config_dir} for Hydra compatibility...")
            os.chdir(config_dir)
            print(f"Calling build_sam2 with config: {config_name}")
            sam2_model = build_sam2(config_name, checkpoint_path, device=device)
            print("SAM2 model initialized.")
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"Warning: Could not initialize SAM2 model: {e}")
        sam2_model = None

def old_get_sam2_predictor(model):
    """
    Helper to get a SAM2 predictor.
    If model is already a predictor (ONNX), return it.
    If model is a PyTorch model, wrap it in SAM2ImagePredictor.
    """
    if model is None:
        return None
    
    # Check for ONNX predictor (duck typing)
    if hasattr(model, 'predict') and hasattr(model, 'set_image'):
        return model
        
    # Assume PyTorch model
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(model)

def old_load_detectron_model(cfg_path, weights_path, device='cpu'):
    """
    Helper to load Detectron2 model from config and weights.
    """
    try:
        if not cfg_path or not weights_path:
            print("Error: cfg_path or weights_path is missing.")
            return None
            
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
            
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        
        predictor = DefaultPredictor(cfg)
        print(f"Loaded Detectron2 model from {weights_path}")
        return predictor
    except Exception as e:
        print(f"Error loading Detectron2 model: {e}")
        return None


def parse_defect_id_method(method_str):
    """
    Parses the defect identification method string.
    Format: Operation[param1, param2]-Operation2[param]+Operation3&&Operation4
    Example: Dino[pic, contour]-KNN[mask] or Dino[...]+KNN[...] or Filtering[HSV]&&KNN
    Returns: List of dictionaries [{'name': 'Dino', 'params': ['pic', 'contour'], 'combine_mode': None}, ...]
    combine_mode: None (start), '-' (sequential/independent), '+' (intersection with previous), '&&' (union with previous)
    """
    operations = []
    if not method_str:
        return operations
    
    # Filter out 'nan' strings (from empty Excel cells)
    if str(method_str).strip().lower() in ('nan', 'none', ''):
        return operations
    
    # DEBUG: Print raw input
    print(f"DEBUG parse_defect_id_method: Raw input = '{method_str}'")
    print(f"DEBUG parse_defect_id_method: Type = {type(method_str)}")
    print(f"DEBUG parse_defect_id_method: Length = {len(str(method_str))}")
    print(f"DEBUG parse_defect_id_method: Repr = {repr(str(method_str))}")
        
    # Split by '+', '-', or '&&' at top-level only (ignore inside [] or ())
    parts = []
    buf = ""
    depth_bracket = 0
    depth_paren = 0
    i = 0
    method_str_len = len(str(method_str))
    while i < method_str_len:
        ch = str(method_str)[i]
        if ch == '[':
            depth_bracket += 1
        elif ch == ']':
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        
        # Check for '&&' operator
        if ch == '&' and i + 1 < method_str_len and str(method_str)[i + 1] == '&' and depth_bracket == 0 and depth_paren == 0:
            if buf.strip():
                parts.append(buf)
            parts.append('&&')
            buf = ""
            i += 2  # Skip both '&' characters
            continue
        
        # Check for '+' or '-' operator
        if ch in ('+', '-') and depth_bracket == 0 and depth_paren == 0:
            if buf.strip():
                parts.append(buf)
            parts.append(ch)
            buf = ""
            i += 1
            continue
        
        buf += ch
        i += 1
    
    if buf.strip():
        parts.append(buf)
    
    current_combine_mode = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part in ('+', '-', '&&'):
            current_combine_mode = part
            continue
        
        if part.startswith('[') and part.endswith(']'):
            part = part[1:-1].strip()
            
        # Check for brackets: Operation[param1, param2] or Operation[Nested[param]]
        # Allow alphanumeric, underscores, and spaces in operation name
        # Use [^\[]+ to match anything until the opening bracket
        match = re.match(r"([^\[]+)(?:\[(.*)\])?", part)
        if match:
            op_name = match.group(1).strip()
            params_str = match.group(2)
            params = []
            if (not params_str) and op_name.lower().startswith('save(') and op_name.endswith(')'):
                inner = op_name[op_name.find('(') + 1:-1]
                op_name = 'Save'
                if inner.strip():
                    params = [p.strip() for p in inner.split(',') if p.strip()]
            elif params_str:
                # Parse nested operations recursively
                # Check if param contains nested brackets (e.g., Filtering[Morph Top-Hat])
                if '[' in params_str and ']' in params_str:
                    # This is a nested operation, parse it recursively
                    nested_ops = parse_defect_id_method(params_str)
                    params = nested_ops
                else:
                    # Split params by comma, respecting parentheses (e.g. Save(A, B))
                    raw_tokens = []
                    buf = ""
                    depth = 0
                    for char in params_str:
                        if char == '(':
                            depth += 1
                            buf += char
                        elif char == ')':
                            depth = max(0, depth - 1)
                            buf += char
                        elif char == ',' and depth == 0:
                            raw_tokens.append(buf.strip())
                            buf = ""
                        else:
                            buf += char
                    if buf.strip():
                        raw_tokens.append(buf.strip())
                    
                    params = []
                    for tok in raw_tokens:
                        params.append(tok)
            operations.append({
                'name': op_name, 
                'params': params,
                'combine_mode': current_combine_mode
            })
        else:
            # Fallback
            operations.append({
                'name': part, 
                'params': [],
                'combine_mode': current_combine_mode
            })
    
    # DEBUG: Print parsed result
    print(f"DEBUG parse_defect_id_method: Parsed {len(operations)} operations: {operations}")
            
    return operations




"""
Project: Silicone & Bumper phone case Defect Analysis
Base: Detectron2 & SAM2.1
Coder: Zeeshan Hyder
Dated: September 05,2025
Updated: September 16,2025
Updated: September 25, 2025
Updated: September 30, 2025
Updated: NOV 7,2025
"""



def perform_general_defect_detection(image, predictor, output_path, save_visualization=False, prompt=None):
    if image is None or predictor is None:
        return None

    # Handle Grounding DINO predictor
    if hasattr(predictor, "predict") and type(predictor).__name__ == "GroundingDinoPredictor":
        if prompt is None:
            # We try to use the predictor's stored prompt if the caller didn't pass one.
            if hasattr(predictor, "default_prompt") and predictor.default_prompt:
                prompt = predictor.default_prompt
            else:
                print("Warning: GroundingDino requires a prompt but none was given. Defaulting to 'defect.'")
                prompt = "defect." 
            
        try:
            results = predictor.predict(image, prompt)
            
            if save_visualization and output_path:
                vis_img = image.copy()
                for res in results:
                    box = [int(v) for v in res['bbox']]
                    cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"{res['class_name']}:{res['score']:.2f}", 
                              (box[0], max(0, box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(output_path, vis_img)
                print(f"Saved Grounding DINO Detection result to: {output_path}")
                
            return results
        except Exception as e:
            print(f"Error in Grounding DINO Detection: {e}")
            return []

    # Handle standard Detectron2 predictor
    from detectron2.utils.visualizer import Visualizer, ColorMode
    try:
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        
        # Visualize
        if save_visualization:
            try:
                # Use default ColorMode to keep original image
                v = Visualizer(image[:, :, ::-1],
                               metadata=MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0] if len(predictor.cfg.DATASETS.TEST)>0 else "__unused"), 
                               scale=1.0
                )
                
                # Draw instance predictions (BBox + Labels)
                out = v.draw_instance_predictions(instances)
                cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
                print(f"Saved General Defect Detection result to: {output_path}")
            except Exception as viz_err:
                print(f"Error in General Defect Visualization: {viz_err}")
        
        # Prepare results with class names
        results = []
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
        
        # Extract class names carefully to avoid crash
        class_names = []
        try:
             if len(predictor.cfg.DATASETS.TEST) > 0:
                 meta = MetadataCatalog.get(predictor.cfg.DATASETS.TEST[0])
                 if hasattr(meta, 'thing_classes'):
                     class_names = meta.thing_classes
        except Exception as e:
             pass
             
        for box, cls_idx in zip(boxes, classes):
            results.append({
                'bbox': box.tolist(), # Convert to list to avoid tensor serialization issues later
                'class_name': class_names[cls_idx] if (class_names and cls_idx < len(class_names)) else str(cls_idx),
                'class_id': int(cls_idx)
            })
            
        return results
        
    except Exception as e:
        print(f"Error in General Defect Detection: {e}")
        return []

























# def Max_likelihood_Sorting():










def seg_point_transition(bbox, x_ratio, y_ratio):
    x1, y1, x2, y2 = bbox
    point_x = x1 + (x2 - x1) * x_ratio
    point_y = y1 + (y2 - y1) * y_ratio
    center_x = x1 + (x2 - x1) * 0.5
    center_y = y1 + (y2 - y1) * 0.5
    # return np.array([[point_x, point_y],[center_x, center_y]])
    return [[point_x, point_y], [center_x, center_y]]














def seg_with_yolo_bbox(image, boxes, defect_masks, df_SN, defects_dataframe, segement_mode, res_path, rel_path,
                       pic_name, overall_BG,Failure_Mode):
    #print('df_SN in seg_with_yolo_bbox', df_SN)
    image_copy = image
    if isinstance(image, str):
        image = load_and_validate_image(image)
        if image is None:
            return defects_dataframe, overall_BG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"DEBUG: inside seg_with_yolo_bbox, sam2_model is {sam2_model}")
    predictor = get_sam2_predictor(sam2_model)
    print(f"DEBUG: predictor is {predictor}")
    predictor.set_image(image)


    index = 0
    sorting_strategy, bbox_matrix, bbox_suppression = sorting_bbox(boxes, df_SN)
    # print('multiline DUT bbox', bbox_suppression)

    # input sorting_strategy as index to sort bbox by x/y
    if sorting_strategy < 2:
        sorted_indices = boxes[:, sorting_strategy].argsort()
    elif sorting_strategy == 7:
        print('seperate DUT by lines')
        clustering_Kmeans(bbox_suppression, df_SN, sorting_strategy)

    boxes = boxes[sorted_indices]
    print('boxes after sorting', boxes)
    lvl2_defects_dataframe = pd.DataFrame()
    # masks, scores = SAM_Predict(image, boxes, 1)
    # -----------
    # mask_BG_cnt = np.zeros(image.shape, dtype=np.uint8)
    for index_SN, box in enumerate(boxes):
        print('debug')
        DUT_SN = df_SN.iloc[index_SN, 0]
        input_box = box.astype(int)
        if segement_mode == "Point":
            seg_point = seg_point_transition(box, 0.5, 0.1)
            seg_point = np.array(seg_point).astype(int)
            print('seg_point', seg_point)
            masks, scores, _ = predictor.predict(
                point_coords=seg_point,
                point_labels=[1, 1],
                box=None,
                multimask_output=False,
            )
        elif segement_mode == "Box":
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
        
        # Resize if necessary (ONNX)
        # Check if resizing is still needed (in case predict didn't do it, though it should now)
        h, w = image.shape[:2]
        if masks.shape[-1] != w or masks.shape[-2] != h:
             print(f"DEBUG: Resizing mask from {masks.shape} to {w}x{h}")
             new_masks = []
             for m in masks:
                  if m.dtype != np.float32: m = m.astype(np.float32)
                  m_res = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                  new_masks.append(m_res)
             masks = np.array(new_masks)

        plt.figure(figsize=(8, 8))
        # plot each DUT after SAM
        # plt.imshow(image)
        show_mask(masks[0], plt.gca())
        show_box(input_box, plt.gca())
        plt.axis = ('off')
        print('number of masks', len(masks))
        # authentic plot each DUT after SAM
        # plt.show()
        # 分别保存分割好的mask
        # -----------
        DUT_Info = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 确保 mask 是布尔类型
            mask = mask > 0
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            # 转换为 uint8 (0 或 255)，不取反
            mask = mask.astype(np.uint8) * 255
            DUT_area, OBB_Angle, mask_contour, leaned_bbox, DUT_length, DUT_width = DUT_Area_OBB_Cal(mask)
            
            # Additional High-Precision Alignment if requested
            if PCA_Alignment == 'Yes':
                try:
                    # 1. PCA Rotation
                    _, pca_angle, _, _, _, _ = DUT_Area_OBB_Cal_knn(mask)
                    
                    # 2. Apply Rotations (PCA based)
                    final_img, final_mask, _ = apply_rotations(image, mask, pca_angle)
                    
                    # 3. Orientation Check
                    final_img, final_mask = check_orientation_and_flip(final_img, final_mask)
                    
                    # 4. Crop and Resize
                    cropped_img, cropped_mask, _ = crop_and_resize(final_img, final_mask)
                    
                    if cropped_img is not None:
                        # Construct save path
                        try:
                            base_dir = Mainfolder_path
                        except NameError:
                            base_dir = os.path.dirname(res_path) # Fallback attempt
                            
                        aligned_root = str(base_dir) + "_Aligned"
                        
                        # Handle relative path structure
                        rel_p = str(rel_path)
                        save_dir = os.path.join(aligned_root, rel_p)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                            
                        # Save with DUT_SN to handle multiple objects
                        # If pic_name has no extension, add .jpg
                        # If multiple objects, ensure uniqueness
                        save_name = f"{pic_name}_{DUT_SN}.jpg"
                        if len(boxes) == 1:
                             save_name = f"{pic_name}.jpg"
                             
                        cv2.imwrite(os.path.join(save_dir, save_name), cropped_img)
                        # print(f"Saved aligned DUT to {os.path.join(save_dir, save_name)}")
                except Exception as e:
                    print(f"Error in PCA_Alignment processing: {e}")

            print('leaned_bbox', leaned_bbox)
            partial_cutout_contour = []
            oblique_4corner_coords = []
            large_obj_angle = 0
            if Large_DUT == "Yes":
                oblique_4corner_coords, large_obj_angle, large_cropped_pic = large_obj_cropping_v2(mask, rect_ratio=Large_DUT_Height_Ratio)
                _, partial_cutout_contour = get_intersection_mask_and_contour(mask, oblique_4corner_coords)
                # cv_show(large_cropped_pic, 'large_cropped_pic')
                large_obj_area = calculate_intersection_area(mask, oblique_4corner_coords)
                DUT_area = large_obj_area
                OBB_Angle = large_obj_angle
                print('large DUT_area', DUT_area)
                print('large OBB_Angle', OBB_Angle)
            segmented_img = mask_based_matting(image, mask)
            orthodox_segement = rotate_image_by_topleft(segmented_img, leaned_bbox, OBB_Angle)

            mask_BG = np.zeros(image.shape, dtype=np.uint8)
            # cv2.drawContours(mask_BG, mask_contour, -1, (255, 255, 255), 1)
            if Large_DUT != "Yes":
                rotated_contour = rotate_contour_by_topleft(mask_contour, leaned_bbox, OBB_Angle)
            else:
                rotated_contour = rotate_contour_by_topleft(partial_cutout_contour, oblique_4corner_coords,
                                                            large_obj_angle)
            cv2.drawContours(mask_BG, [rotated_contour], -1, (0, 255, 0), 1)

            DUT_Info.append([DUT_area, OBB_Angle, DUT_length, DUT_width])
            # Only use the current DUT info for the current mask
            current_DUT_Info_df = pd.DataFrame([DUT_Info[-1]], columns=['DUT_Area', 'OBB_Angle', 'DUT_Length', 'DUT_Width'])
            print('current_DUT_Info_df', current_DUT_Info_df)
            SAM_yolo_mask = cv2.bitwise_and(mask, defect_masks)
            # cv_show(SAM_yolo_mask, 'combo')
            SAM_yolo_mask = SAM_yolo_mask.astype(np.uint8)
            input_box_centroid = np.array(bbox_centroid(input_box)).astype(np.int32)
            print('input_box centroid', input_box_centroid)
            # Determine seg_mode and input_data based on Large_DUT and SAM_Input
            current_oblique_bbox = leaned_bbox
            final_seg_mode = "Box"
            final_input_data = input_box
            
            if Large_DUT == "Yes":
                current_oblique_bbox = oblique_4corner_coords
                # Default for Large_DUT is Point
                final_seg_mode = "Point"
                final_input_data = input_box_centroid
            
            # Allow SAM_Input to override the segmentation mode
            print(f"*** DEBUG: SAM_Input global value in seg_with_yolo_bbox is: '{SAM_Input}' ***")
            
            if SAM_Input is not None:
                # Robust cleaning of SAM_Input
                clean_sam_input = str(SAM_Input).strip().strip("'").strip('"').upper()
                print(f"*** DEBUG: Cleaned SAM_Input: '{clean_sam_input}' (Original: '{SAM_Input}') ***")

                if clean_sam_input in ["POINTS", "POINT"]:
                    final_seg_mode = "Point"
                    final_input_data = input_box_centroid
                    print("*** DEBUG: Mode set to POINT via SAM_Input ***")
                elif clean_sam_input in ["BBOX", "BOX"]:
                    final_seg_mode = "Box"
                    final_input_data = input_box
                    print("*** DEBUG: Mode set to BOX via SAM_Input ***")
                elif clean_sam_input in ["CONTOUR", "CNTS"]:
                    final_seg_mode = "Contour"
                    final_input_data = None # Will be handled inside quantify_defects
                    print("*** DEBUG: Mode set to CONTOUR via SAM_Input ***")
                else:
                    print(f"*** DEBUG: SAM_Input '{SAM_Input}' did not match any known mode. Defaulting to CONTOUR. ***")
                    final_seg_mode = "Contour"
                    final_input_data = None # Will be handled inside quantify_defects
            
            print(f"*** DEBUG: Final seg_mode passed to quantify_defects: {final_seg_mode} ***")
            sub_defects_res, defect_contours = quantify_defects(SAM_yolo_mask, DUT_SN, final_input_data, current_DUT_Info_df,
                                                                    image_copy, current_oblique_bbox, final_seg_mode)
            if Scoring_mode == "Yes":
                for cnt in defect_contours:
                    rotated_defect_contour = rotate_contour_by_topleft(cnt, leaned_bbox, OBB_Angle)
                    if len(rotated_defect_contour) > 0:
                        # print('rotated_defect_contour', rotated_defect_contour)
                        #     # 获取第一个点，并确保其形状为 (1, 2)
                        first_point = [rotated_defect_contour[0]]
                        #     # 将第一个点添加到轮廓的末尾以闭合轮廓
                        rotated_defect_contour = np.vstack([rotated_defect_contour, first_point])
                    # # 确保轮廓数据类型是 np.int32
                    rotated_defect_contour = rotated_defect_contour.astype(np.int32)
                    rotated_defect_contour = rotated_defect_contour.reshape(-1, 2)  # 变成 (125, 2)
                    rotated_defect_contour = rotated_defect_contour.reshape(1, -1, 2)  # 变成 (1, 125, 2)
                    # 如果画contour用下面的命令
                    # cv2.drawContours(mask_BG, rotated_defect_contour, -1, (255, 255, 255), cv2.FILLED)
                    defect_cutout = contour_based_matting(orthodox_segement, rotated_defect_contour)
                    # cv_show(defect_cutout, 'defect_cutout')
                    mask_BG = cv2.bitwise_or(mask_BG, defect_cutout)

                    # cv2.imshow('defect_cutout', defect_cutout)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                # cv_show(mask_BG, 'Individual_DUT')
                cropped_img = crop_by_bbox(mask_BG, box, 1.3)
                # cv_show(cropped_img, 'cropped individual_DUT')
                rel_path_str = str(rel_path)
                non_cube_face_name, cube_face_name = rel_path_str.rsplit('/', 1)
                DUT_path_list = ["Scoring", str(non_cube_face_name), DUT_SN]
                # print('DUT_path_list', DUT_path_list)
                DUT_path = path_creator(res_path, DUT_path_list)
                print('DUT_path', DUT_path)
                os.makedirs(DUT_path, exist_ok=True)
                pic_saving_path = os.path.join(DUT_path, f"{cube_face_name}.jpg")
                # print('cube_face_name', cube_face_name)
                cv2.imwrite(pic_saving_path, cropped_img)
                cv2.drawContours(overall_BG, defect_contours, -1, (255, 255, 255), cv2.FILLED)

            # print('sub_defects_res', sub_defects_res)
            lvl2_defects_dataframe = pd.concat([lvl2_defects_dataframe, sub_defects_res], axis=0)
            # print('lvl2_defects_dataframe', lvl2_defects_dataframe)
            # plt.savefig('pic-{}.png'.format(i + index))
            # plt.show()
            index += 1
    # cv_show(overall_BG, 'mask_BG_cnt')
    
    # Unified Overlay Logic
    # Convert overall_BG (accumulated contours) to mask
    if overall_BG.ndim == 3:
        mask_final = cv2.cvtColor(overall_BG, cv2.COLOR_BGR2GRAY)
    else:
        mask_final = overall_BG
        
    # Use create_overlay_image with standard Blue color
    # Note: image is BGR here (from cv2.imread in RELP3_main.py or similar)
    # So we must use BGR color (255, 162, 0) to get Blue.
    # User requested RGB (0, 162, 255).
    # BGR equivalent: (255, 162, 0).
    # User requested Opacity 50%
    remasked_image = create_overlay_image(image, mask_final, color=(255, 162, 0), transparency=0.5)
    
    # remasked_image = cv2.addWeighted(image, 1, overall_BG, 0.8, 0)
    # cv_show(remasked_image, 'remasked_image')
    maskedPic_path_list = ["Inferred Pic", rel_path_str]
    maskedPic_path = path_creator(res_path, maskedPic_path_list)
    os.makedirs(maskedPic_path, exist_ok=True)
    pic_saving_path = os.path.join(maskedPic_path, f"{pic_name}.jpg")
    cv2.imwrite(pic_saving_path, remasked_image)

    plt.close('all')
    defects_dataframe = pd.concat([defects_dataframe, lvl2_defects_dataframe], axis=0)
    defects_dataframe = defects_dataframe.reset_index(drop=True)
    print('re-indexed defects_dataframe', defects_dataframe)
    return defects_dataframe, overall_BG






# def rotate_contour_by_topleft(contour, rect_points, angle):
#     """
#     以最小外接矩形的左上顶点为中心旋转轮廓
#
#     参数:
#     contour: 输入轮廓点集
#     rect_points: 最小外接矩形的四个顶点坐标 shape为(4,2)的numpy数组
#     angle: 当前矩形的偏转角度（度数）
#
#     返回:
#     rotated_contour: 旋转后的轮廓点集
#     """
#     import numpy as np
#     import cv2
#
#     if contour is None or len(contour) == 0:
#         raise ValueError("输入轮廓为空")
#
#     # 处理不规则形状的轮廓
#     try:
#         contour = np.array(contour, dtype=np.float32)
#     except ValueError:
#         try:
#             points = []
#             for point in contour:
#                 if isinstance(point, np.ndarray):
#                     points.append(point.flatten()[:2])
#                 else:
#                     points.append(point[0])
#             contour = np.array(points, dtype=np.float32)
#         except Exception as e:
#             print(f"Contour shape: {np.array(contour).shape}")
#             print(f"Contour type: {type(contour)}")
#             raise ValueError(f"无法处理的轮廓格式: {e}")
#
#     # 确保形状正确
#     if len(contour.shape) == 2:
#         contour = contour.reshape(-1, 1, 2)
#
#     # 增加点的密度
#     points = contour.reshape(-1, 2)
#     dense_points = []
#
#     for i in range(len(points) - 1):
#         p1, p2 = points[i], points[i + 1]
#         # 计算两点之间的距离
#         dist = np.linalg.norm(p2 - p1)
#         # 根据距离动态调整插值点数量
#         num_points = max(20, int(dist / 5))
#
#         # 对于底边部分增加更多的点
#         if p1[1] > points[:, 1].max() - 50 and p2[1] > points[:, 1].max() - 50:
#             num_points *= 2
#
#         interp_points = np.linspace(p1, p2, num_points)
#         dense_points.extend(interp_points)
#
#     dense_contour = np.array(dense_points, dtype=np.float32).reshape(-1, 1, 2)
#
#     rect_points = np.array(rect_points, dtype=np.float32)
#     top_points = rect_points[rect_points[:, 1].argsort()][:2]
#     topleft = top_points[top_points[:, 0].argmin()]
#     center = (float(topleft[0]), float(topleft[1]))
#
#     if abs(angle) > 45:
#         rotation_angle = angle - 90 if angle > 0 else angle + 90
#     else:
#         rotation_angle = angle
#
#     rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
#
#     contour_reshaped = dense_contour.reshape(-1, 2)
#     ones = np.ones(shape=(len(contour_reshaped), 1))
#     points_ones = np.hstack([contour_reshaped, ones])
#     transformed_points = rotation_matrix.dot(points_ones.T).T
#
#     rotated_contour = transformed_points.reshape(-1, 1, 2)
#     rotated_contour = rotated_contour.astype(np.int32)
#
#     return rotated_contour




def quantify_defects(whole_mask, SN, bbox, DUT_info, img, oblique_bbox, seg_mode, area_threshold=25):
    print('input seg_mode', seg_mode)
    print('input bbox', bbox)
    image = cv2.cvtColor(whole_mask, cv2.COLOR_BGRA2GRAY)
    _, res_combined = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(res_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print('contours in quantify_defects', contours)
    defect_metrics = []
    SN_df = pd.DataFrame()
    DEFECT_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    counting_idx = 0
    whole_defects_df = None
    indexed_defect = None
    refined_contours = []
    if contours and not all(cv2.contourArea(cnt) < area_threshold for cnt in contours):
        for Defects_index, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print("area found in quantifying is ", area)
            if area >= area_threshold:
                counting_idx = counting_idx + 1
                indexed_defect = 'defect_' + str(counting_idx)
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_fine = np.array([x, y, x + w, y + h])
                print('bbox_fine', bbox_fine)
                cnt_centorid = contour_centroid(cnt)
                cnt_centorid = np.array(cnt_centorid).astype(int)
                print('cnt_centorid', cnt_centorid)
                bbox_fine_centroid = bbox_centroid(bbox_fine)
                bbox_fine_centroid = np.array(bbox_fine_centroid).astype(int)
                # use SAM to segment defect, comparing with area of coarse seg
                try:
                    # Decide input based on seg_mode (which respects SAM_Input)
                    if seg_mode == "Point":
                         cnt = fine_seg(img, cnt_centorid, area, cnt, seg_mode)
                    elif seg_mode == "Box":
                         cnt = fine_seg(img, bbox_fine, area, cnt, seg_mode)
                    elif seg_mode == "Contour":
                         cnt = fine_seg(img, cnt, area, cnt, seg_mode)
                    # Fallback logic
                    elif Failure_Mode == "Scuff" or Large_DUT == "Yes":
                        cnt = fine_seg(img, cnt_centorid, area, cnt, seg_mode)
                    else:
                        cnt = fine_seg(img, bbox_fine, area, cnt, seg_mode)
                except Exception as e:
                    print(f"警告: 在fine_seg调用中出错: {e}")
                    # 保持原始轮廓不变，继续处理
                
                refined_contours.append(cnt)

                # Calculate Defect Length and Width using minAreaRect (Pixels)
                # Must be done BEFORE normalization to get pixel dimensions
                d_rect = cv2.minAreaRect(cnt)
                d_size = d_rect[1]
                defect_length = max(d_size)
                defect_width = min(d_size)

                cnt = normalize_contours(oblique_bbox, cnt)
                cnt = np.array(cnt, dtype=np.float32)

                # print('normalize_contours', cnt)
                defect_metrics.append([area, cnt, defect_length, defect_width])
                metrics_df = pd.DataFrame(defect_metrics, columns=['Defect_Area', 'Contour', 'Defect_Length', 'Defect_Width'])
                sn_df = pd.DataFrame([SN], columns=['SN'])
                SN_df = pd.concat([SN_df, sn_df], axis=0, ignore_index=True)
                defect_df = pd.DataFrame([indexed_defect], columns=['Defect'])
                DEFECT_df = pd.concat([DEFECT_df, defect_df], axis=0, ignore_index=True)
                whole_defects_df_0 = pd.concat([SN_df, DEFECT_df, metrics_df], axis=1)
                DUT_repeated = pd.concat([DUT_info] * len(whole_defects_df_0), ignore_index=True)
                whole_defects_df = pd.concat([whole_defects_df_0, DUT_repeated], axis=1)
                # print('defect_metrics.append', defect_metrics)


    else:
        indexed_defect = 'no_detection'
        x, y, w, h = 0, 0, 0, 0
        area = 0
        defect_length = 0
        defect_width = 0
        # defect_metrics.append([area, np.array([[[0, 0]]], dtype=np.int32)])
        defect_metrics.append([area, [[[0, 0]]], defect_length, defect_width])
        # defect_metrics = bbox + defect_metrics
        # print('defect_metrics all', defect_metrics)
        metrics_df = pd.DataFrame(defect_metrics,
                                  columns=['Defect_Area', 'Contour', 'Defect_Length', 'Defect_Width'])
        sn_df = pd.DataFrame([SN], columns=['SN'])
        SN_df = pd.concat([SN_df, sn_df], axis=0, ignore_index=True)
        defect_df = pd.DataFrame([indexed_defect], columns=['Defect'])
        DEFECT_df = pd.concat([DEFECT_df, defect_df], axis=0, ignore_index=True)
        whole_defects_df_0 = pd.concat([SN_df, DEFECT_df, metrics_df], axis=1)
        DUT_repeated = pd.concat([DUT_info] * len(whole_defects_df_0), ignore_index=True)
        whole_defects_df = pd.concat([whole_defects_df_0, DUT_repeated], axis=1)
    
    final_contours = refined_contours if refined_contours else contours
    return whole_defects_df, final_contours




def fine_seg(image, bbox, coarse_area, coarse_contour, segement_mode):
    if isinstance(image, str):
        image = load_and_validate_image(image)
        if image is None:
            return coarse_contour
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if sam2_model is None:
        print("Warning: SAM2 model not initialized. Skipping fine segmentation and using coarse contour.")
        return coarse_contour

    print(f"DEBUG: inside seg_with_yolo_bbox, sam2_model is {sam2_model}")
    predictor = get_sam2_predictor(sam2_model)
    print(f"DEBUG: predictor is {predictor}")
    predictor.set_image(image)

    
    if not isinstance(bbox, list):
        input_box = bbox.astype(int)
    else:
        input_box = bbox

    if segement_mode == "Point":
        # Ensure input_box is properly formatted for Point mode
        # If it is a single point (1,2) or (2,), we might want to add context or just use it.
        # User log showed 2 points.
        print(f'*** DEBUG: fine_seg (Point Mode) input: {input_box} ***')
        
        # If input_box is single point, ensure it is (N, 2)
        if len(input_box.shape) == 1:
            input_box = input_box.reshape(1, 2)
            
        # If we need 2 points (e.g. centroid + something else), ensure logic here matches expectation.
        # For now, we trust the input `bbox` passed from quantify_defects.
        
        # Assuming point_labels match number of points. 
        # If input_box has 2 points, we need 2 labels.
        current_labels = np.ones(len(input_box), dtype=np.int32)
        
        masks, scores, _ = predictor.predict(
            point_coords=input_box,
            point_labels=current_labels,
            box=None,
            multimask_output=False,
        )
    elif segement_mode == "Box":
        print(f'*** DEBUG: fine_seg (Box Mode) input: {input_box} ***')
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
    elif segement_mode == "Contour":
        print(f'*** DEBUG: fine_seg (Contour Mode) raw input shape: {input_box.shape} ***')
        # Sample points from contour
        contour_points = input_box.reshape(-1, 2)
        # Downsample if too many points to avoid excessive memory usage/noise
        # Increased limit from 20 to 128 to better preserve shape details for complex defects (e.g. Cracks)
        target_points = 128
        if len(contour_points) > target_points:
            indices = np.linspace(0, len(contour_points) - 1, target_points, dtype=int)
            points = contour_points[indices]
        else:
            points = contour_points
        
        print(f'*** DEBUG: fine_seg (Contour Mode) used points: {len(points)} ***')
        labels = np.ones(len(points), dtype=np.int32)
        
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=False,
        )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind] # Fix bug: Sort scores to match masks
    
    print('fine_mask_shape', masks.shape)
    # if len(masks.shape) == 3 and masks.shape[0] == 1:
    #     masks = np.squeeze(masks, axis=0)
    # cv_show(masks, 'fine_mask_sha  pe')
    plt.title("Fine Segmentation Results", fontsize=16)  # 添加标题
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    plt.figure(figsize=(8, 8))
    # show_mask(masks[0], plt.gca()) # visualization might crash if mask size mismatch, skipping
    # plt.imshow(image)
    plt.axis = ('off')
    # plt.show()
    fine_seg_area = 0
    fine_seg_contour = None
    plt.close('all')
    
    h, w = image.shape[:2]
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Resize mask if necessary (ONNX compatibility)
        if mask.shape[-1] != w or mask.shape[-2] != h:
             if mask.dtype != np.float32:
                  mask = mask.astype(np.float32)
             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # mask = mask + 255
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        try:
            # 确保 mask 是布尔类型 (处理 SAM2 返回的 float 0.0/1.0)
            mask = mask > 0.0
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            # 转换为 uint8 (0 或 255)，不取反
            mask = mask.astype(np.uint8) * 255
            # mask = np.nan_to_num(mask, nan=0.0, posinf=255.0, neginf=0.0)
            # mask = np.clip(mask, 0, 255)
            # mask = mask.astype(np.uint8)
            # cv_show(mask, 'fine_seg')
            fine_seg_area, _, fine_seg_contour, _, _, _ = DUT_Area_OBB_Cal(mask)
            if fine_seg_contour is not None:
                extracted_region = contour_based_matting(image, fine_seg_contour)
            else:
                print("警告: 未能提取有效轮廓，跳过contour_based_matting")
                extracted_region = None
        except Exception as e:
            print(f"警告: 在fine_seg处理中出错: {e}")
            fine_seg_area = 0
            fine_seg_contour = None
            extracted_region = None
        # cv2.imshow("extracted_region Image", extracted_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if fine_seg_contour is not None and len(fine_seg_contour) > 0:
            try:
                x_f, y_f, w_f, h_f = cv2.boundingRect(fine_seg_contour)
            except Exception as e:
                print(f"警告: 计算边界矩形时出错: {e}")
                x_f, y_f, w_f, h_f = 0, 0, 0, 0
        else:
            print("警告: fine_seg_contour无效，使用默认边界矩形")
            x_f, y_f, w_f, h_f = 0, 0, 0, 0
        if segement_mode == "Box":
            try:
                area_ratio = fine_seg_area / coarse_area if coarse_area > 0 else 0
                width_ratio = w_f / (bbox[3] - bbox[1]) if (bbox[3] - bbox[1]) > 0 else 0
                
                # Relaxed threshold logic for Box mode (Aligned with Contour mode relaxation)
                Box_Acceptance_Low = 0.1
                current_low = Acceptance_Low
                
                # Check if Acceptance_Low is at default strict value (0.99)
                if abs(Acceptance_Low - 0.99) < 0.001: 
                     current_low = Box_Acceptance_Low
                     print(f"DEBUG: Using relaxed Box_Acceptance_Low: {current_low} (Global default 0.99 detected)")
                else:
                     print(f"DEBUG: Using configured Acceptance_Low: {current_low}")
                
                # Also relax width_ratio check (allow slight expansion or exact match)
                # Original logic: width_ratio < 1 (Strictly smaller)
                # New logic: width_ratio < 1.5 (Allow reasonable expansion)
                
                # Check if we should enforce Acceptance_High
                # If Acceptance_High is close to default 1.01, we might want to be strict IF not in relaxed mode?
                # But user explicitly asked why it's not working.
                # If user manually set Acceptance_Low (not 0.99), we should respect both Low and High.
                
                enforce_high = True
                if abs(Acceptance_Low - 0.99) < 0.001:
                    # Default strict mode -> relaxed internal logic
                    # In this case, we usually ignore High limit to allow expansion
                    enforce_high = False
                
                if area_ratio > current_low:
                    if enforce_high and area_ratio >= Acceptance_High:
                        coarse_contour = coarse_contour
                        print(f"keep to coarse segment result (Box mode) - Ratio {area_ratio:.2f} exceeds High limit {Acceptance_High}")
                    else:
                        if area_ratio >= Acceptance_High:
                             print(f"WARNING: Ratio {area_ratio:.2f} exceeds High limit {Acceptance_High}, but keeping result (Relaxed Mode).")
                        
                        coarse_contour = fine_seg_contour
                        print(f"contour updated to fine segment result (Box mode) - Ratio {area_ratio:.2f} accepted > {current_low}")
                else:
                    coarse_contour = coarse_contour
                    # print("keep to coarse segment result")
                    print(f"keep to coarse segment result (Box mode) - Ratio {area_ratio:.2f} too low (<= {current_low})")
            except Exception as e:
                print(f"警告: 在比较区域比例时出错: {e}")
                coarse_contour = coarse_contour # Fallback
            # else:
            #    coarse_contour = coarse_contour
            #    print("keep to coarse segment result")
        elif segement_mode == "Contour":
            ratio = fine_seg_area / coarse_area if coarse_area > 0 else 0
            print(f'DEBUG: Contour Mode - Fine Area: {fine_seg_area:.2f}, Coarse Area: {coarse_area:.2f}, Ratio: {ratio:.2f}')
            print(f'DEBUG: Thresholds - Low: {Acceptance_Low}, High: {Acceptance_High}')
            
            # Relaxed threshold for Contour mode to allow refinement (e.g. thinning of cracks)
            # Default Acceptance_Low (0.99) is too strict for shape refinement
            Contour_Acceptance_Low = 0.1 
            
            enforce_high = True
            if abs(Acceptance_Low - 0.99) < 0.001:
                # Default strict mode -> relaxed internal logic
                # In this case, we usually ignore High limit to allow expansion
                enforce_high = False

            if ratio > Contour_Acceptance_Low:
                if enforce_high and ratio >= Acceptance_High:
                     coarse_contour = coarse_contour
                     print(f"keep to coarse segment result (Contour mode) - Ratio {ratio:.2f} exceeds High limit {Acceptance_High}")
                else:
                    if ratio >= Acceptance_High:
                         print(f"WARNING: Ratio {ratio:.2f} exceeds High limit {Acceptance_High}, but keeping result (Relaxed Mode).")
                
                    coarse_contour = fine_seg_contour
                    print(f"contour updated to fine segment result (Contour mode) - Ratio {ratio:.2f} accepted > {Contour_Acceptance_Low}")
            else:
                coarse_contour = coarse_contour
                print(f"keep to coarse segment result (Contour mode) - Ratio {ratio:.2f} too low (<= {Contour_Acceptance_Low})")
        elif segement_mode == "Point":
            ratio = fine_seg_area / coarse_area if coarse_area > 0 else 0
            print(f'fine_seg_area/coarse_area: {ratio}')
            
            # Relaxed threshold logic for Point mode
            Point_Acceptance_Low = 0.1
            current_low = Acceptance_Low
            
            enforce_high = True
            if abs(Acceptance_Low - 0.99) < 0.001: 
                 current_low = Point_Acceptance_Low
                 enforce_high = False
                 print(f"DEBUG: Using relaxed Point_Acceptance_Low: {current_low} (Global default 0.99 detected)")
            else:
                 print(f"DEBUG: Using configured Acceptance_Low: {current_low}")
            
            if ratio > current_low:
                if enforce_high and ratio >= Acceptance_High:
                     coarse_contour = coarse_contour
                     print(f"keep to coarse segment result (Point mode) - Ratio {ratio:.2f} exceeds High limit {Acceptance_High}")
                else:
                    if ratio >= Acceptance_High:
                         print(f"WARNING: Ratio {ratio:.2f} exceeds High limit {Acceptance_High}, but keeping result (Relaxed Mode).")
                
                    coarse_contour = fine_seg_contour
                    print(f"contour updated to fine segment result (Point mode) - Ratio {ratio:.2f} accepted > {current_low}")
            else:
                coarse_contour = coarse_contour
                print(f"keep to coarse segment result (Point mode) - Ratio {ratio:.2f} too low (<= {current_low})")
    return coarse_contour


def DUT_Area_OBB_Cal(DUT):
    _, img_gray = cv2.threshold(DUT, 110, 255, cv2.THRESH_BINARY)
    if len(img_gray.shape) == 3:  # 检查是否为三通道图像
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    largest_contour = None
    angle = 0
    rect_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])  # 默认值，避免未定义错误

    # OBB detection
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            largest_contour = contour
            max_area = area

    if largest_contour is not None:
        try:
            rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rect
            length = max(size)
            width = min(size)
            print("OBB angel is ", angle)
            rect_points = cv2.boxPoints(rect)
            rect_points = rect_points.astype(int)
            print('rect_points', rect_points)

            if len(rect_points) != 4:
                print("警告: 未能正确检测到长方形的四个顶点，使用默认值")
                rect_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
                length, width = 0.0, 0.0
            else:
                # 对顶点进行排序
                rect_points = rect_points.reshape(4, 2)
                center = rect_points.mean(axis=0)
                angles = np.arctan2(rect_points[:, 1] - center[1],
                                    rect_points[:, 0] - center[0])
                sorted_idx = np.argsort(angles)
                rect_points = rect_points[sorted_idx]
        except Exception as e:
            print(f"警告: 计算最小外接矩形时出错: {e}")
            rect_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
            length, width = 0.0, 0.0
    else:
        print("警告: 未找到有效轮廓，使用默认rect_points值")
        length, width = 0.0, 0.0

    return max_area, round(angle, 1), largest_contour, rect_points, length, width


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

















def sort_columns_with_numbers(df):
    columns = df.columns.tolist()
    pattern = re.compile(r'(\d+)')
    sorted_columns = sorted(columns, key=lambda x: tuple(map(int, pattern.findall(x))))
    return df[sorted_columns]


def multirows_DUT_handling(bbox_clusters, df_SN, bbox_index, defect_mask, mask_h, mask_w, defects_quantify,
                           resized_image, pic_name, relative_path, image_filename, parsed_boxes,Failure_Mode):
    # calculate mean for subgroups
    group_defect_df = pd.DataFrame()
    means = {key: bbox_clusters[key][:, bbox_index].mean() for key in bbox_clusters}
    sorted_keys = sorted(bbox_clusters.keys(), key=lambda k: np.mean(bbox_clusters[k][:, bbox_index]))
    overrall_mask_BG_cnt = np.zeros(defect_mask.shape, dtype=np.uint8)
    # 创建一个包含排序后的键值对的列表
    sorted_arrays = [(key, bbox_clusters[key]) for key in sorted_keys]
    # 迭代排序后的列表
    for index, (key, sub_bbox_array) in enumerate(sorted_arrays):
        print("here we go, multirows_DUT_handling")
        print('sub_bbox_array in multirow', sub_bbox_array)
        subgroup_SN = df_SN.iloc[:, index]
        subgroup_SN_df = subgroup_SN.to_frame()
        if len(df_SN.iloc[:, index]) < sub_bbox_array.shape[0]:
            boxes_post_dropout = bbox_outlier_supression(sub_bbox_array, 4, subgroup_SN_df, parsed_boxes)
            print('boxes_post_dropout', boxes_post_dropout)
        else:
            boxes_post_dropout = sub_bbox_array
        scaled_boxes = scaling_up_bbox(boxes_post_dropout, 1.05, 1.05, mask_h, mask_w)
        defects_quantify, overrall_mask_BG_cnt = seg_with_yolo_bbox(resized_image, scaled_boxes, defect_mask,
                                                                    subgroup_SN_df, defects_quantify, "Box", pic_name,
                                                                    relative_path, image_filename, overrall_mask_BG_cnt,Failure_Mode)
        # defects_quantify = pd.concat([defects_quantify, group_defect_df], axis=0)
        # print('defects_quantify', defects_quantify)
    return defects_quantify


def normalize_corrdinates(pic_DF):
    normalize_df = pd.DataFrame()
    ori_column = pic_DF.columns
    pic_df = pic_DF.to_numpy()
    boxes_w = pic_df[:, 4] - pic_df[:, 2]
    boxes_h = pic_df[:, 5] - pic_df[:, 3]
    defect_x1 = (pic_df[:, 7] - pic_df[:, 2]) / boxes_w
    defect_y1 = (pic_df[:, 8] - pic_df[:, 3]) / boxes_h
    defect_x2 = (pic_df[:, 9] - pic_df[:, 2]) / boxes_w
    defect_y2 = (pic_df[:, 10] - pic_df[:, 3]) / boxes_h
    pic_df = np.hstack((pic_df, defect_x1.reshape(-1, 1), defect_y1.reshape(-1, 1), defect_x2.reshape(-1, 1),
                        defect_y2.reshape(-1, 1)))
    # normalized_contour = np.zeros((pic_df.shape[0], 1, 2))
    for row_id, cnt in enumerate(pic_df):
        # print('haha', cnt[11])
        contours = cnt[11]
        y = np.zeros_like(contours)
        # print('待减数shape', pic_df[row_id, 2].shape)
        array_0 = np.full((contours.shape[0], 1), pic_df[row_id, 2])
        array_1 = np.full((contours.shape[0], 1), pic_df[row_id, 3])
        boxes_w_array = np.full((contours.shape[0], 1), boxes_w[row_id])
        boxes_h_array = np.full((contours.shape[0], 1), boxes_h[row_id])
        y[:, :, 0] = (contours[:, :, 0] - array_0) / boxes_w_array * 300
        y[:, :, 1] = (contours[:, :, 1] - array_1) / boxes_h_array * 1280
        pic_list = list(pic_df[row_id][:2]) + list(pic_df[row_id][6:11]) + list(pic_df[row_id][12:])
        pic_list.append(y)
        normalize_sub = pd.DataFrame([pic_list],
                                     columns=['SN', 'Defect', 'Defect_Area', 'D_x1', 'D_y1', 'D_x2', 'D_y2', 'DUT_Area',
                                              'OBB_Angle', 'defect_x1', 'defect_y1', 'defect_x2', 'defect_y2',
                                              'Contour'])
        normalize_df = pd.concat([normalize_df, normalize_sub], axis=0).reset_index(drop=True)
    return normalize_df




def heatmap_creation(img, df, pic_name, save_path):
    height, width = img.shape[:2]

    count_map = np.zeros_like(img, dtype=np.uint8)
    count_map = np.zeros((height, width), dtype=np.uint8)
    contours_OBB = df[['OBB_Angle', 'Contour']]
    contours = df['Contour'].to_numpy()
    rotate_angle = df['OBB_Angle'].median()

    for contour in contours:
        # create a mask of original pic size
        mask = np.zeros((height, width), dtype=np.uint8)
        # use fillPoly to flood polygons
        cv2.fillPoly(mask, [contour], 1)
        # dilating contours to include adjacent pixels
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # 将掩码与 count_map 相加
        count_map[mask == 1] += 1

    rotate_M_1 = cv2.getRotationMatrix2D((width // 2, height // 2), -(90 - rotate_angle), 1)
    count_map = cv2.warpAffine(count_map, rotate_M_1, (width, height))
    heatmap_normalized = cv2.normalize(count_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Unified Blue Color (RGB 0, 162, 255) -> BGR (255, 162, 0)
    norm_gray = heatmap_normalized.astype(float) / 255.0
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    # BGR: (255, 162, 0)
    heatmap[:, :, 0] = (norm_gray * 255).astype(np.uint8) # Blue
    heatmap[:, :, 1] = (norm_gray * 162).astype(np.uint8) # Green
    heatmap[:, :, 2] = 0                                  # Red
    
    # Overlay heatmap on original image if needed, but this function saves heatmap only
    # If we want to overlay it on original image:
    # remasked_heatmap = cv2.addWeighted(img, 0.2, heatmap, 0.8, 0)
    # But function signature says "heatmap_creation", and saves "heatmap".
    # Previous code saved COLORMAP_JET applied to count_map.
    # So saving just the heatmap (blue-scale) is correct per request?
    # Or should it be overlaid?
    # The user request "A类缺陷叠加图统一用一个函数实现" refers to overlay.
    # This function creates a heatmap from contours.
    # If this is for visualization, maybe overlay is better?
    # The previous code saved `heatmap` which was just the color map.
    # So I will keep saving just the heatmap but in Blue scale.
    
    pic_name_header = os.path.splitext(pic_name)[0]
    heatmap_title = pic_name_header + '_heatmap.jpg'
    pic_save_path = os.path.join(save_path, heatmap_title)
    print('pic_save_path', pic_save_path)
    cv2.imshow('Heatmap', heatmap)
    cv2.imwrite(pic_save_path, heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



























def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))




input_point = np.array([[1700, 1000]])
input_label = np.array([1])


def SAM_Predict_Simplified(image, boxes, scale_factor=1):
    # input_label = np.array([1])
    new_layer = np.zeros_like(image)
    new_layer1 = np.zeros_like(image)
    image = np.array(image)
    if isinstance(image, str):
        image = load_and_validate_image(image)
        if image is None:
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"DEBUG: inside seg_with_yolo_bbox, sam2_model is {sam2_model}")
    predictor = get_sam2_predictor(sam2_model)
    print(f"DEBUG: predictor is {predictor}")
    predictor.set_image(image)

    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    for index_SN, box in enumerate(boxes):
        input_box = box.astype(int)
        print('sam box', input_box)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        
        # Resize if necessary (ONNX)
        h, w = image.shape[:2]
        if masks.shape[-1] != w or masks.shape[-2] != h:
             new_masks = []
             for m in masks:
                  if m.dtype != np.float32: m = m.astype(np.float32)
                  m_res = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                  new_masks.append(m_res)
             masks = np.array(new_masks)
             
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

    return masks, scores


def SAM_Predict(image, boxes, scale_factor=1):
    # input_label = np.array([1])
    new_layer = np.zeros_like(image)
    new_layer1 = np.zeros_like(image)
    image = np.array(image)
    if isinstance(image, str):
        image = load_and_validate_image(image)
        if image is None:
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"DEBUG: inside seg_with_yolo_bbox, sam2_model is {sam2_model}")
    predictor = get_sam2_predictor(sam2_model)
    print(f"DEBUG: predictor is {predictor}")
    predictor.set_image(image)

    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    for index_SN, box in enumerate(boxes):
        input_box = box.astype(int)
        print('sam box', input_box)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        
        # Resize if necessary (ONNX)
        h, w = image.shape[:2]
        if masks.shape[-1] != w or masks.shape[-2] != h:
             print(f"DEBUG: Resizing mask from {masks.shape} to {w}x{h}")
             new_masks = []
             for m in masks:
                  if m.dtype != np.float32: m = m.astype(np.float32)
                  m_res = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                  new_masks.append(m_res)
             masks = np.array(new_masks)

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        plt.title("Segmentation Results", fontsize=16)  # 添加标题
        # for mask in masks:
        #     show_mask(mask, plt.gca(), random_color=True)
        # scores = scores[sorted_ind]
        # logits = logits[sorted_ind]
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
        plt.figure(figsize=(8, 8))
        # IMPORTANT plot each DUT after SAM
        plt.imshow(image)
        contour_ori = contour_finding(masks)
        contour_scaled = contour_scale(contour_ori, scale_factor)
        # print('contour_ori xx', contour_ori)
        cv2.drawContours(new_layer, [contour_scaled], -1, (255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, new_layer)
        # IMPORTANT plot each DUT after SAM
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.axis = ('off')

        x1, y1, x2, y2 = box.astype(int)
        contour_box = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32)
        # print('x1 in sam', x1)
        # height, width, _ = result.shape
        # # 确定裁剪区域
        # crop_x1 = max(0, x1)
        # crop_y1 = max(0, y1)
        # crop_x2 = min(width, x2)
        # crop_y2 = min(height, y2)
        # # 裁剪图像
        # cropped_pic = result[crop_y1:crop_y2, crop_x1:crop_x2]
        # cv2.imshow('cropping Result', cropped_pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.drawContours(new_layer1, contour_box, -1, (255, 255, 255), thickness=cv2.FILLED)
        result1 = cv2.bitwise_and(result, new_layer1)
        # cv2.imshow('Result1', result1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # result2 = cv2.bitwise_and(image, new_layer)

        mask = (masks[0] * 255).astype(np.uint8)
        new_layer[mask > 0] = image[mask > 0]
    return result1, contour_scaled









def feature_removal(image, contour_nd, bbox_nd=np.empty((3, 4), dtype=float)):
    contour_2remove, _ = SAM_Predict(image, contour_nd, 1.1)
    print("removing features in rectangular shape")
    if bbox_nd is not None and len(bbox_nd) > 0:
        for bbox in bbox_nd:
            bbox_2d = bbox_2darray(bbox)
            # print("bbox_2d", bbox_2d)
            cv2.drawContours(contour_2remove, [bbox_2d], -1, (255, 255, 255), thickness=cv2.FILLED)
            result = cv2.bitwise_and(image, contour_2remove)
            # cv2.imshow('Result', result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        # 如果 bbox_nd 为空，直接返回原始图像
        result = cv2.bitwise_and(image, contour_2remove)
    return result






def simple_feature_filter(DUT_df, SN_df):
    result_df = pd.DataFrame()
    print('simple_feature DUT_df', DUT_df)
    qty = SN_df.shape[0]
    result_df = DUT_df.nlargest(qty, 'conf')
    result_df.reset_index(drop=True, inplace=True)
    filtered_box = result_df.loc[0, "bbox"]
    filtered_box = np.array([filtered_box])
    print('filtered_box', filtered_box)
    return filtered_box


def feature_filter(DUT_df, feat_df):
    file_path = 'RELP_Configuration.xlsx'
    try:
        feat_list = ConfigManager().get_sheet("feature")
    except Exception as e:
        print(f"Warning: Could not load feature from {file_path}: {e}. Continuing with empty config.")
        feat_list = pd.DataFrame()
    print('feat_list', feat_list)
    feat_df['SN'] = None
    for i, row_pred in feat_df.iterrows():
        point = row_pred['bbox_center']
        for j, row_dut in DUT_df.iterrows():
            bbox = row_dut['DUT_bbox']
            if is_point_in_bbox(point, bbox):
                feat_df.at[i, 'SN'] = row_dut['SN']
                break
    print('feat_df', feat_df)
    # 初始化一个空的 DataFrame 用于存储结果
    result_df = pd.DataFrame()
    # 遍历 feat_list 中的每个特征
    for index, row in feat_list.iterrows():
        feature_name = row['feature_name']
        qty = row['qty']
        # 从 feat_df 中选择对应的特征
        feature_df = feat_df[feat_df['label'] == feature_name]
        # 按照 SN 分组，并在每个分组中按照 conf 从高到低排序
        grouped = feature_df.groupby('SN', group_keys=False).apply(
            lambda x: x.sort_values('conf', ascending=False).head(qty))
        # 将选中的行合并到结果 DataFrame 中
        result_df = pd.concat([result_df, grouped])
    # 重置索引
    result_df.reset_index(drop=True, inplace=True)
    print('filtered', result_df)
    return result_df


def feature_location_retrieval(feat_df, feature, DUT_SN):
    result_df = feat_df[(feat_df['SN'] == DUT_SN) & (feat_df['label'] == feature)]
    feat_bbox = result_df.loc[0, 'bbox']
    feat_buttom = feat_bbox[-1]
    # print('feature_location_retrieval', feat_buttom)
    return feat_buttom


""" --------------------zee-----------------------------------"""


def target_labelz(DUT_df, SN_df, target_label='class_0'):
    print('simple_feature DUT_df', DUT_df)

    filtered_DUT = DUT_df[DUT_df['label'] == target_label]

    if filtered_DUT.empty:
        print(f"No detections found for label: {target_label}")
        return np.array([])  # or return None or handle as needed

    qty = SN_df.shape[0]

    result_df = filtered_DUT.nlargest(qty, 'conf')
    result_df.reset_index(drop=True, inplace=True)

    filtered_box = result_df.loc[0, "bbox"]
    filtered_box = np.array([filtered_box])

    print('filtered_box', filtered_box)
    return filtered_box









""" --------zee--------"""










def parse_filename_info(filename):

    product = "Unknown"
    build = "Unknown"
    config = "Unknown"
    color = "Unknown"
    test_item = "Unknown"
    unit_no = "Unknown"
    test_cycles = "Unknown"
    photo_position = "Unknown"

    # 处理完整路径格式的文件名
    if '/' in filename:
        path_parts = filename.split('/')

        # 从主文件夹名称提取信息
        if len(path_parts) > 0:
            main_folder = path_parts[0]

            # 解析主文件夹名称，使用下划线分割
            folder_parts = main_folder.split('_')

            if len(folder_parts) >= 4:
                # 提取product信息 (M120)
                product = folder_parts[0]

                # 提取build信息 (DVT)
                build = folder_parts[1]

                # 提取config信息 (SOP4 3hr)
                config = folder_parts[2]

                # 提取color信息 (Gray24)
                color = folder_parts[3]

                # 提取test_item信息 (Backwards CIC)
                if len(folder_parts) >= 5:
                    test_item = folder_parts[4]
                    # 如果test_item包含括号，提取括号前的内容
                    if '(' in test_item:
                        test_item = test_item.split('(')[0].strip()

        # 从子目录中提取unit_no
        if len(path_parts) > 1:
            subfolder = path_parts[1]  # 22 (索引1)
            if subfolder.isdigit():
                unit_no = subfolder

        # 从文件名中提取test_cycles和photo_position
        if len(path_parts) > 3:
            filename_part = path_parts[-1]  # 22-200c-2.jpg
            # 格式: 22-200c-2.jpg
            file_parts = filename_part.split('-')
            if len(file_parts) >= 2:
                # 提取test_cycles (200c)，支持大小写不敏感匹配
                cycles_match = re.search(r'(\d+[cC])', file_parts[1])
                if cycles_match:
                    # 统一转换为小写输出
                    test_cycles = cycles_match.group(1).lower()

                # 提取photo_position (2)
                if len(file_parts) >= 3:
                    position_match = re.search(r'(\d+)', file_parts[2])
                    if position_match:
                        photo_position = position_match.group(1)

    return product, build, config, color, test_item, unit_no, test_cycles, photo_position




#===========================================







def init_sam2(config_path, checkpoint_path, device='cpu'):
    global sam2_model
    sam2_model = _real_init_sam2(config_path, checkpoint_path, device)

def extract_detections(sam2_predictor, image_rgb, boxes, tape_classes=None, mask_scaling_factor=1.0, dut_scaling_config=None, class_names=None):
    from utils.cv_ops import generate_sam2_mask
    import cv2
    import numpy as np
    
    all_detections = []
    
    if len(boxes) == 0:
        return all_detections
        
    for i, box in enumerate(boxes):
        class_id = tape_classes[i] if tape_classes is not None and i < len(tape_classes) else -1
        class_name = class_names[class_id] if class_names and 0 <= class_id < len(class_names) else f"Class_{class_id}"
        
        # Determine actual scaling factor based on class
        actual_scaling = mask_scaling_factor
        if dut_scaling_config and isinstance(dut_scaling_config, dict):
            # Check for specific class scaling
            for k, v in dut_scaling_config.items():
                if isinstance(k, str) and k.startswith('Feature_Remove_') and not k.startswith('Feature_Remove_Scaling_'):
                    feat_class = str(v).lower()
                    if feat_class == str(class_name).lower():
                        # Find corresponding scaling factor
                        idx = k.split('_')[-1]
                        scale_key = f"Feature_Remove_Scaling_Factor_{idx}"
                        if scale_key in dut_scaling_config:
                            try:
                                actual_scaling = float(dut_scaling_config[scale_key])
                            except:
                                pass
                        break
            
            # Check for generic DUT scaling if it's class 0
            if 'DUT_Class_Name' in dut_scaling_config and str(class_name).lower() == str(dut_scaling_config['DUT_Class_Name']).lower():
                 if 'DUT_Scaling_Factor' in dut_scaling_config:
                      try:
                          actual_scaling = float(dut_scaling_config['DUT_Scaling_Factor'])
                      except:
                          pass
            elif class_id == 0 and 'DUT_Scaling_Factor' in dut_scaling_config:
                 try:
                      actual_scaling = float(dut_scaling_config['DUT_Scaling_Factor'])
                 except:
                      pass
                      
        mask = generate_sam2_mask(sam2_predictor, image_rgb, box, actual_scaling)
        if mask is None:
            continue
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 10:
            try:
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[-1]
                if rect[1][0] < rect[1][1]:
                    angle += 90
                # 标准化角度到 (-45, 45] 区间，避免 90/180 度翻转
                while angle > 45:
                    angle -= 90
                while angle <= -45:
                    angle += 90
            except:
                angle = 0
                
            all_detections.append({
                "box": box,
                "class_id": class_id,
                "class_name": class_name,
                "mask": mask,
                "contour": largest_contour,
                "area": area,
                "angle": angle
            })
            
    return all_detections
