from utils.base_utils import _norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import shutil

# Ensure Dinomaly is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Now dinov3_utils.py is directly inside utils/, so we need to go up one level to get to the project root
project_root = os.path.dirname(current_dir)
dinomaly_path = os.path.join(project_root, 'Dinomaly')
if dinomaly_path not in sys.path:
    sys.path.append(dinomaly_path)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from functools import partial
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dinov3.hub.backbones import load_dinov3_model
# from dataset import get_data_transforms # Unused
from torchvision import transforms
import pandas as pd
from utils.utils_general import apply_filtering
from utils import utils_general

# Modern AnyUp Integration
anyup_path = os.path.join(project_root, 'Dinomaly', 'anyup-main')
if anyup_path not in sys.path:
    sys.path.append(anyup_path)

try:
    from anyup.model import AnyUp
except ImportError as e:
    AnyUp = None
    print(f"Warning: Could not import AnyUp model. Error: {e}")

def load_anyup_upsampler(device):
    if AnyUp is None:
        print("AnyUp module not imported. Skipping AnyUp loading.")
        return None
    print(">>> Loading Modern AnyUp Model...")
    try:
        model = AnyUp()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(os.path.dirname(current_dir), 'weights')
        weight_path = os.path.join(weights_dir, 'anyup_paper.pth')
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"AnyUp weights loaded from {weight_path}")
        else:
            print(f"AnyUp weights not found at {weight_path}")
            return None
            
        model = model.to(device)
        # MPS 芯片在使用 FP16 时容易在 RoPE 或 Attention 中发生类型冲突导致崩溃。
        # 因此，在 MPS 设备上保持 FP32 运行，依靠分块和尺寸限制来防爆显存。
        if device.type == 'mps':
            print("Running AnyUp model on MPS using FP32 (Half precision disabled to avoid type mismatch)...")
            # model = model.half() # Disabled FP16 for MPS
        elif device.type == 'cuda':
            print("Converting AnyUp model to FP16 for CUDA memory optimization...")
            model = model.half()
            
        model.eval()
        return model
    except Exception as e:
        print(f"AnyUp loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def cal_anomaly_maps_anyup(upsampler, img_tensor, en_list, de_list, target_size):
    total_a_map = None
    count = 0
    import torch.nn.functional as F
    
    for i in range(len(de_list)):
        if i >= len(en_list): break
        
        feat_en = en_list[i]
        feat_de = de_list[i]
        
        # Device/Dtype sync
        up_device = next(upsampler.parameters()).device
        up_dtype = next(upsampler.parameters()).dtype
        
        if i == 0:
            print(f"DEBUG: AnyUp is currently computing on Device: [{up_device}] with Precision: [{up_dtype}]")
        
        c_img = img_tensor.to(up_device)
        c_en = feat_en.to(up_device)
        c_de = feat_de.to(up_device)
        
        if up_dtype == torch.float16:
            c_img, c_en, c_de = c_img.half(), c_en.half(), c_de.half()
        else: # float32 fallback (which MPS now uses)
            c_img, c_en, c_de = c_img.float(), c_en.float(), c_de.float()
            
        # Native Chunking Inference
        u_en = upsampler(c_img, c_en, output_size=target_size, q_chunk_size=128)
        u_de = upsampler(c_img, c_de, output_size=target_size, q_chunk_size=128)
        
        a_map = 1 - F.cosine_similarity(u_en, u_de)
        a_map = torch.unsqueeze(a_map, dim=1).float() # Keep map in float32
        
        if total_a_map is None:
            total_a_map = a_map
        else:
            total_a_map += a_map
        count += 1
        
    anomaly_map = total_a_map / count if count > 0 else total_a_map
    return anomaly_map.to(img_tensor.device), []


try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# Visualization settings
TRANSPARENCY_THRESHOLD = 0.2

def get_gaussian_kernel(kernel_size=5, sigma=4):
    """Generate Gaussian kernel"""
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    def gaussian_filter_func(x):
        return torch.nn.functional.conv2d(x, kernel.to(x.device), padding=kernel_size//2)
    
    return gaussian_filter_func

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def cvt2heatmap(gray):
    # Unified Blue Color (RGB 0, 162, 255) -> BGR (255, 162, 0)
    # Map gray intensity to Blue intensity
    norm_gray = gray.astype(float) / 255.0
    heatmap = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    # BGR: (255, 162, 0)
    heatmap[:, :, 0] = (norm_gray * 255).astype(np.uint8) # Blue
    heatmap[:, :, 1] = (norm_gray * 162).astype(np.uint8) # Green
    heatmap[:, :, 2] = 0                                  # Red
    return heatmap

def show_cam_on_image(img, anomaly_map):
    # Unified Transparency: 50% Opacity
    # User clarification: "Transparency 50%"
    # This implies Opacity 50% -> Overlay Weight 0.5, Base Weight 0.5
    
    # anomaly_map should be the colored heatmap (3 channels)
    if anomaly_map.ndim == 2:
        # If single channel, convert to Blue heatmap first
        anomaly_map = cvt2heatmap(anomaly_map)
    
    # Resize if needed
    if anomaly_map.shape[:2] != img.shape[:2]:
        anomaly_map = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]))
        
    cam = cv2.addWeighted(img, 0.5, anomaly_map, 0.5, 0)
    return cam

def cal_anomaly_maps(fs_list, ft_list, out_size=224):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        # Use bicubic interpolation for smoother heatmaps (reduce blockiness/squares)
        a_map = F.interpolate(a_map, size=out_size, mode='bicubic', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def load_model(model_path, device):
    """Load trained DinoV3 model"""
    encoder_name = 'dinov3_vitb16'

    # Resolve absolute path for encoder backbone weights robustly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    candidates = [
        os.path.join(project_root, 'weights', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
        os.path.join(project_root, 'Dinomaly', 'weights', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
        os.path.join(os.getcwd(), 'weights', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
        'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    ]
    encoder_weight = None
    for p in candidates:
        if os.path.exists(p):
            encoder_weight = p
            break
    if encoder_weight is None:
        print("Warning: DinoV3 encoder backbone weight not found in common locations. Proceeding without explicit path.")
        encoder_weight = candidates[-1]

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    # Use DinoV3 encoder
    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                pretrained_weight_path=encoder_weight)
    
    # DinoV3 base model parameters
    if 'vits' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'vitb' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'vitl' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("Architecture not in vits, vitb, vitl.")
    
    bottleneck = []
    decoder = []
    
    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)
    
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)
    
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, 
                   target_layers=target_layers, mask_neighbor_size=0, 
                   fuse_layer_encoder=fuse_layer_encoder, 
                   fuse_layer_decoder=fuse_layer_decoder)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights: {model_path}")
    else:
        raise FileNotFoundError(f"Model weight file not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    return model

def read_dino_threshold_config(config_path='Anomaly_config.xlsx'):
    """Read Dino_TH sheet from config file to get transparency threshold range."""
    # Try finding in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Potential paths to check
    paths_to_check = [
        config_path, # Current working directory
        os.path.join(script_dir, config_path), # Script directory
        os.path.join(os.path.dirname(script_dir), 'Detectron_Dino', config_path) # Detectron_Dino directory
    ]
    
    print(f"DEBUG: Looking for {config_path} in: {paths_to_check}")

    for path in paths_to_check:
        if os.path.exists(path):
            print(f"DEBUG: Found config file at {path}")
            try:
                df = pd.read_excel(path, sheet_name='Dino_TH')
                print(f"DEBUG: Successfully read Dino_TH sheet from {path}")
                print(f"DEBUG: Columns: {df.columns.tolist()}")
                
                if 'Dino_Low' in df.columns and 'Dino_High' in df.columns:
                     if not df.empty:
                         low = df['Dino_Low'].iloc[0]
                         high = df['Dino_High'].iloc[0]
                         print(f"DEBUG: Raw values - Low: {low}, High: {high}")
                         
                         if pd.isna(low) and pd.isna(high):
                             print("DEBUG: Both values are NaN. Using default range [0.2, max_val].")
                             return (0.2, None)
                         elif pd.notna(low) and pd.isna(high):
                             print(f"DEBUG: Low={low}, High=NaN. Using range [{low}, max_val].")
                             return (float(low), None)
                         elif pd.notna(low) and pd.notna(high):
                             print(f"Loaded Dino_TH from {path}: Low={low}, High={high}")
                             return (float(low), float(high))
                         else:
                             print(f"DEBUG: Values are partially NaN (Low={low}, High={high}). Skipping.")
                     else:
                         print("DEBUG: DataFrame is empty. Using default range [0.2, max_val].")
                         return (0.2, None)
                else:
                    print("DEBUG: Missing columns Dino_Low or Dino_High")
            except Exception as e:
                print(f"Note: Could not read Dino_TH from {path}: {e}")
                # pass # Don't just pass, let's see the error
    
    print("DEBUG: Failed to load Dino threshold from config.")
    return None

# Global variable to cache color dataframe
COLOR_DF = None

def get_color_name(rgb):
    """
    Get human readable color name from RGB tuple using Manhattan distance
    and a CSV database of colors, matching Live Image Color Detection.py logic.
    
    Args:
        rgb: tuple/list/array of (R, G, B)
    
    Returns:
        str: Closest color name
    """
    global COLOR_DF
    
    # Load CSV if not loaded
    if COLOR_DF is None:
        try:
            # Look for colors.csv in the same directory as this script or Detectron_Dino
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, "colors.csv"),
                os.path.join(os.path.dirname(script_dir), "Detectron_Dino", "colors.csv"),
                "colors.csv"
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path:
                index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
                COLOR_DF = pd.read_csv(csv_path, header=None, names=index)
                print(f"DEBUG: Loaded color database from {csv_path}")
            else:
                print("WARNING: colors.csv not found. Falling back to basic colors.")
        except Exception as e:
            print(f"ERROR: Failed to load colors.csv: {e}")

    # Fallback to basic colors if CSV loading failed
    if COLOR_DF is None:
        colors = {
            'Black': (0, 0, 0),
            'White': (255, 255, 255),
            'Red': (255, 0, 0),
            'Green': (0, 255, 0),
            'Blue': (0, 0, 255),
            'Yellow': (255, 255, 0),
            'Cyan': (0, 255, 255),
            'Magenta': (255, 0, 255),
            'Orange': (255, 165, 0),
            'Purple': (128, 0, 128),
            'Brown': (165, 42, 42),
            'Gray': (128, 128, 128),
            'Light Gray': (211, 211, 211),
            'Dark Gray': (169, 169, 169)
        }
        
        rgb_arr = np.array(rgb)
        min_dist = float('inf')
        closest_name = 'Unknown'
        
        for name, value in colors.items():
            dist = np.linalg.norm(rgb_arr - np.array(value))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

    # Use Manhattan distance with CSV data
    R, G, B = rgb
    
    # Vectorized calculation for speed
    # d = abs(B - df['B']) + abs(G - df['G']) + abs(R - df['R'])
    d = (np.abs(B - COLOR_DF['B']) + 
         np.abs(G - COLOR_DF['G']) + 
         np.abs(R - COLOR_DF['R']))
    
    min_index = d.idxmin()
    closest_name = COLOR_DF.loc[min_index, 'color_name']
    
    return closest_name

def generate_transparent_heatmap(gray, threshold=TRANSPARENCY_THRESHOLD):
    """
    Generate transparent heatmap where values below threshold are transparent.
    
    Args:
        gray: Normalized grayscale anomaly map (0-1)
        threshold: Threshold below which pixels become transparent. 
                   Can be float or tuple (low, high).
    """
    # Create colormap
    heatmap = cv2.applyColorMap(np.uint8(gray * 255), cv2.COLORMAP_JET)
    
    # Create alpha channel
    alpha = np.zeros_like(gray)
    
    if isinstance(threshold, (list, tuple)):
        low, high = threshold
        mask_indices = (gray >= low) & (gray <= high)
        if high > low:
            # Linear mapping for range
            alpha[mask_indices] = (gray[mask_indices] - low) / (high - low)
        else:
            alpha[mask_indices] = 1.0
    else:
        # Values below threshold will have 0 alpha (fully transparent)
        # Values above threshold:
        # If threshold is very low (<0.2), we want to make them MORE visible quickly.
        # Standard: alpha = (val - th) / (1 - th) -> Linear ramp starting from 0 at threshold
        
        mask_indices = gray > threshold
        
        if threshold < 0.2:
            # Boost visibility for low thresholds
            # Instead of linear ramp from 0, jump to a base visibility (e.g., 0.3)
            # and ramp up from there.
            # alpha = 0.3 + 0.7 * (val - th) / (1 - th)
            # This ensures even values slightly above threshold are visible
            normalized_val = (gray[mask_indices] - threshold) / (1 - threshold + 1e-6)
            alpha[mask_indices] = 0.3 + 0.7 * normalized_val
        else:
            alpha[mask_indices] = (gray[mask_indices] - threshold) / (1 - threshold)
    
    # Apply alpha to heatmap
    # Convert heatmap to BGRA
    heatmap_bgra = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
    heatmap_bgra[:, :, 3] = np.uint8(alpha * 255)
    
    return heatmap_bgra

def show_cam_on_image_transparent(img, anomaly_map, threshold=TRANSPARENCY_THRESHOLD):
    """
    Overlay transparent heatmap on image.
    
    Args:
        img: Original image (uint8)
        anomaly_map: Normalized anomaly map (0-1)
        threshold: Threshold for transparency
    """
    # Generate heatmap with transparency
    heatmap_colored = cv2.applyColorMap(np.uint8(anomaly_map * 255), cv2.COLORMAP_JET)
    # Keep BGR to match OpenCV image format
    # heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create mask for blending
    mask = np.zeros_like(anomaly_map)
    
    if isinstance(threshold, (list, tuple)):
        low, high = threshold
        mask_indices = (anomaly_map >= low) & (anomaly_map <= high)
        if high > low:
            mask[mask_indices] = (anomaly_map[mask_indices] - low) / (high - low)
        else:
            mask[mask_indices] = 1.0
    else:
        # Same logic as above: boost visibility for low thresholds
        mask_indices = anomaly_map > threshold
        if threshold < 0.2:
             normalized_val = (anomaly_map[mask_indices] - threshold) / (1 - threshold + 1e-6)
             mask[mask_indices] = 0.3 + 0.7 * normalized_val
        else:
             mask[mask_indices] = (anomaly_map[mask_indices] - threshold) / (1 - threshold)
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask]*3, axis=-1)
    
    # Blend: result = img * (1 - alpha) + heatmap * alpha
    overlay = img.astype(np.float32) * (1 - mask_3ch) + heatmap_colored.astype(np.float32) * mask_3ch
    
    return np.uint8(np.clip(overlay, 0, 255))

def find_largest_contour(image_rgb):
    """Find the largest contour of objects in the original image"""
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
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Use CHAIN_APPROX_NONE to get complete contour points
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
    return largest_contour, contour_area, bounding_rect

def get_allowed_mask(image, contour_shrink_ratio=None, height_lower_ratio=None):
    """Generate allowed mask based on contour shrink and height crop"""
    h, w = image.shape[:2]
    allowed_mask = np.ones((h, w), dtype=bool)
    
    contour, _, bounding_rect = find_largest_contour(image)
    
    if contour is not None and contour_shrink_ratio is not None and 0 < contour_shrink_ratio < 1:
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
        mask1 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask1, [scaled_contour], 255)
        allowed_mask &= mask1.astype(bool)
        print(f"Applied contour shrink ratio: {contour_shrink_ratio}")

    if bounding_rect is not None and height_lower_ratio is not None and 0 <= height_lower_ratio <= 1:
        if height_lower_ratio > 0:
            x0, y0, bw, bh = bounding_rect
            ys = y0 + int(bh * height_lower_ratio)
            ys = max(0, min(ys, h))
            ye = h 
            
            mask2 = np.zeros((h, w), dtype=np.uint8)
            mask2[ys:ye, :] = 255
            allowed_mask &= mask2.astype(bool)
            print(f"Applied height lower ratio: {height_lower_ratio} (start y: {ys})")

    return allowed_mask

def check_and_save_intermediate(op_keyword, post_ops, mask, original_img, save_dir, filename_base):
    """
    Checks if 'Save' follows 'op_keyword' in post_ops, OR if 'Save' is in the parameters of 'op_keyword'.
    If so, generates a red overlay and saves it.
    """
    if not post_ops:
        return

    should_save = False
    # Check if 'Save' follows op_keyword OR is in params
    for i, op in enumerate(post_ops):
        # Fuzzy match op_keyword in operation name
        if op_keyword.lower() in op['name'].lower():
            # Check 1: 'Save' as the NEXT operation
            if i + 1 < len(post_ops):
                if post_ops[i+1]['name'].lower() == 'save':
                    should_save = True
            
            # Check 2: 'Save' in parameters of CURRENT operation
            # params is a list of strings
            if op.get('params'):
                for param in op['params']:
                    if 'save' in param.lower():
                        should_save = True
            
            if should_save:
                break
    
    if should_save:
        print(f"DEBUG: 'Save' requested after {op_keyword}. Generating intermediate result...")
        
        # Ensure mask is boolean/binary and same size as original_img
        if mask.shape[:2] != original_img.shape[:2]:
             mask_for_overlay = cv2.resize(mask.astype(np.uint8), (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
             mask_for_overlay = mask.astype(np.uint8)
        
        # Create Overlay
        heatmap_on_image = original_img.copy()
        
        # Define Red Color (RGB: 255, 0, 0)
        red_layer = np.zeros_like(original_img)
        red_layer[:] = [255, 0, 0] 
        
        # Alpha blending
        alpha = 0.3 
        
        # Indices where mask is active
        mask_indices = (mask_for_overlay > 0)
        
        if np.any(mask_indices):
            roi_orig = original_img[mask_indices]
            roi_red = red_layer[mask_indices]
            
            blended = cv2.addWeighted(roi_orig, 0.7, roi_red, 0.3, 0)
            heatmap_on_image[mask_indices] = blended
            
        # Clean filename base
        fname_clean = filename_base.replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
        
        save_name = f"{fname_clean}_after_{op_keyword.replace(' ', '_')}.png"
        save_path = os.path.join(save_dir, save_name)
        
        cv2.imwrite(save_path, cv2.cvtColor(heatmap_on_image, cv2.COLOR_RGB2BGR))
        print(f"Saved intermediate result to: {save_path}")




def process_single_image_pipeline(image_path, model, upsampler, device, save_dir, 
                                   super_resolution_factor=2.0,
                                   contour_shrink_ratio=None,
                                   height_lower_ratio=None,
                                   transparency_threshold=None,
                                   filtering_df=None,
                                   output_params=None,
                                   contour_image=None,
                                   flooding_enabled=False,
                                   flooding_rgb=None,
                                   dino_input_dim=None,
                                   force_stretch=False,
                                   adaptive_gaussian_config=None,
                                   post_ops=None,
                                   # Legacy args (kept for compatibility but largely unused if post_ops is used)
                                   rgb_output_params=None,
                                   adaptive_gaussian_output_params=None,
                                   defect_id_method=None,
                                   inferred_pic_dir=None,
                                   image_numpy=None
                                   ):
    """Process single image - supports super-resolution upsampling (AnyUp) and mask filtering"""
    # Data preprocessing
    # image_size = 512
    # crop_size = 448
    # data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    
    # Load image first to get original size
    if image_numpy is not None:
        image = Image.fromarray(image_numpy)
        if image.mode != 'RGB':
             image = image.convert('RGB')
    else:
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError:
            # 兼容截断或损坏图像：使用 cv2 读取并转为 PIL Image
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                raise
            image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    full_input_image = np.array(image) # Keep copy of full input for restoration
    full_input_h, full_input_w = full_input_image.shape[:2]
    original_img = np.array(image) # RGB numpy array for flooding
    orig_w, orig_h = image.size
    
    # Variables for coordinate restoration
    crop_off_x = 0
    crop_off_y = 0
    tight_crop_w = orig_w
    tight_crop_h = orig_h
    
    # Calculate inference size maintaining aspect ratio
    # Longest side should be around crop_size (448), but ensure multiple of patch_size (16)
    patch_size = 16
    
    target_max = None # Initialize target_max safely
    is_list_input = False # Initialize is_list_input safely (used later for heatmap logic)

    # --- Tight BBox Crop (Applied to ALL modes for alignment) ---
    # Always crop to tight bounding box first to ensure heatmap alignment with object
    # This ensures both Dino_Input_Dim and default modes produce aligned heatmaps
    try:
        bbox = None
        # Prioritize using contour_image for bbox if available (most reliable)
        if contour_image is not None:
            y_idx, x_idx = np.where(contour_image > 0)
            if len(y_idx) > 0 and len(x_idx) > 0:
                bbox = (np.min(x_idx), np.min(y_idx), np.max(x_idx) - np.min(x_idx) + 1, np.max(y_idx) - np.min(y_idx) + 1)
        
        # Fallback to image content if no contour or empty contour
        if bbox is None:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            x, y, w, h = cv2.boundingRect(thresh)
            if w > 0 and h > 0:
                bbox = (x, y, w, h)
        
        # Apply Crop if valid bbox found
        if bbox:
            x, y, w, h = bbox
            # Store offsets for restoration
            crop_off_x = x
            crop_off_y = y
            tight_crop_w = w
            tight_crop_h = h
            
            # Crop Image
            image_np = np.array(image)
            image_crop = image_np[y:y+h, x:x+w]
            
            # Apply Mask if contour available (Black out background)
            if contour_image is not None:
                # Crop Contour
                contour_crop = contour_image[y:y+h, x:x+w]
                
                if len(image_crop.shape) == 3:
                    mask_3ch = cv2.merge([contour_crop, contour_crop, contour_crop])
                    image_crop = cv2.bitwise_and(image_crop, mask_3ch)
                else:
                    image_crop = cv2.bitwise_and(image_crop, contour_crop)
                
                # Update contour_image to cropped version
                contour_image = contour_crop
            
            image = Image.fromarray(image_crop)
            orig_w, orig_h = image.size
            
            print(f"DEBUG: Re-cropped input to tight bbox: {w}x{h} (Stripped padding & Masked background)")
    except Exception as e:
        print(f"Warning: Failed to re-crop image to tight bbox: {e}")

    # --- Dino Input Dim Logic (User Request) ---
    if dino_input_dim is not None:
        print(f"DEBUG: Applying Dino_Input_Dim: {dino_input_dim}")

        # Parse Dino_Input_Dim (Reuse Crop_Output_Dim logic)
        target_max = None
        target_width = None
        is_list_input = False # Flag to distinguish between scalar and list input
        
        dim_str = str(dino_input_dim).strip()
        if dim_str.startswith('[') and dim_str.endswith(']'):
             is_list_input = True
             try:
                 parts = dim_str[1:-1].split(',')
                 vals = [int(float(p.strip())) for p in parts if p.strip()]
                 if len(vals) > 0: target_max = vals[0]
                 if len(vals) > 1: target_width = vals[1]
             except:
                 pass
        else:
             try:
                 target_max = int(float(dim_str))
             except:
                 pass
        
        # Initialize padding variables
        active_pad_left = 0
        active_pad_right = 0
        active_pad_top = 0
        active_pad_bottom = 0

        if target_max is not None:
            # Ensure target_max is 16x aligned (Dino Requirement)
            rem = target_max % 16
            if rem != 0:
                target_max = target_max - rem
                print(f"DEBUG: Adjusted target_max to {target_max} for 16x alignment.")

            if is_list_input:
                # User Request: List Input = Fit to Size (AR Preserved) + Pad
                container_h = target_max
                container_w = target_width if target_width is not None else target_max
                
                # Align container to 16x
                if container_w % 16 != 0: container_w = (container_w // 16) * 16
                if container_h % 16 != 0: container_h = (container_h // 16) * 16
                
                print(f"DEBUG: Dino List Input. Fitting image into {container_w}x{container_h} (Preserving AR + Padding).")
                
                # Calculate Scale to fit
                scale = min(container_w / orig_w, container_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                # Ensure at least 1px
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                
                # Resize Content
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image_resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # Update Image object
                original_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(original_img)
                orig_w, orig_h = new_w, new_h
                
                # Resize contour if present
                if contour_image is not None:
                     contour_image = cv2.resize(contour_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            elif force_stretch:
                # Strategy B: Force Stretch to Square (target_max x target_max)
                # This ignores aspect ratio and fills the entire target dimension
                print(f"DEBUG: Force_Stretch=True (Strategy B). Resizing input from {orig_w}x{orig_h} to {target_max}x{target_max} (Ignoring Aspect Ratio).")
                print(f"DEBUG: Force_Stretch Memory: Before={orig_w}x{orig_h}, After={target_max}x{target_max}")
                
                new_w = target_max
                new_h = target_max
                
                # Resize image directly
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image_resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Update Image object
                original_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(original_img)
                orig_w, orig_h = new_w, new_h
                
                # Resize contour if present
                if contour_image is not None:
                     contour_image = cv2.resize(contour_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                     print("DEBUG: Resized contour_image to match Force_Stretch (Strategy B).")
            else:
                # Strategy A: Maintain Aspect Ratio (Default) - Mar19 Version
                # Force Stretch/Upscale to fit target_max while maintaining Aspect Ratio
                # User Feedback: "Proportion looks wrong" -> Do not stretch to square, preserve aspect ratio.
                # User Feedback: "Too much black border" -> Scale to max possible size within target_max.
                
                max_curr = max(orig_w, orig_h)
                min_pad = 16 # Ensure at least 16 pixels padding on all sides (User Request: 10px -> 16px aligned)
                
                # Available size for CONTENT is target_max - 2*min_pad
                available_max = target_max - (min_pad * 2)
                if available_max < 16: available_max = target_max # Fallback if target_max is too small
                
                # Calculate Scale to fit
                scale = available_max / max_curr
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                # Ensure we don't exceed available_max (approx) to keep padding
                if new_w > available_max: new_w = available_max
                if new_h > available_max: new_h = available_max
                
                # NOTE: We do NOT force 16x alignment here for Scalar Input.
                # We want to allow padding to handle the alignment in the Container logic.
                # This ensures we have margins as requested.
                
                # Ensure we don't exceed target_max (Absolute Limit)
                if new_w > target_max: new_w = target_max
                if new_h > target_max: new_h = target_max
                
                # Resize image directly
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # Use INTER_LINEAR or INTER_CUBIC for upscaling
                image_resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Update Image object
                original_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(original_img)
                orig_w, orig_h = new_w, new_h
                
                # Resize contour if present
                if contour_image is not None:
                     contour_image = cv2.resize(contour_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                     print("DEBUG: Resized contour_image to match Dino_Input_Dim (Aspect Ratio Preserved).")
                
                print(f"DEBUG: Scaled input for Dino (Dino_Input_Dim): {new_w}x{new_h} (Aspect Ratio Preserved)")
            
        else:
             print("Warning: Failed to parse Dino_Input_Dim, using default logic.")
             # Fallback to default logic
             base_size = 448
             scale = base_size / max(orig_w, orig_h)
             new_w = int(orig_w * scale)
             new_h = int(orig_h * scale)
             new_w = round(new_w / patch_size) * patch_size
             new_h = round(new_h / patch_size) * patch_size
             new_w = max(new_w, patch_size)
             new_h = max(new_h, patch_size)

    else:
        # Default Logic (Original)
        base_size = 640  # Changed from 448 to 640 for better detail preservation
        scale = base_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Round to nearest multiple of patch_size
        new_w = round(new_w / patch_size) * patch_size
        new_h = round(new_h / patch_size) * patch_size
        
        # Ensure at least one patch
        new_w = max(new_w, patch_size)
        new_h = max(new_h, patch_size)

    # Use calculated dimensions (16x aligned).
    # Force Square Padding (Required by ViTill / uad.py model architecture)
    # The model assumes square feature maps (side = sqrt(tokens)).
    
    # If target_max is set (from Dino_Input_Dim), use it as the container size
    # This ensures that even if content is smaller (due to margin), we still output target_max x target_max
    
    # New Logic: 
    # - If List input: Force pad to [target_max, target_width (or target_max)]
    # - If Scalar input: Do NOT force pad, keep aspect ratio (max_dim = content size)
    
    if target_max is not None:
        if is_list_input:
            max_dim_h = target_max
            max_dim_w = target_width if target_width is not None else target_max
        else:
            # Scalar input:
            # 1. Long side -> target_max (e.g. 896) to ensure overall height/width matches input
            # 2. Short side -> Align to 16x (padding added if needed)
            
            def align16(v):
                return (v + 15) // 16 * 16

            if new_h >= new_w:
                max_dim_h = target_max
                max_dim_w = align16(new_w)
            else:
                max_dim_w = target_max
                max_dim_h = align16(new_h)
    else:
        # Default behavior: Maintain aspect ratio (no square forcing)
        # This ensures heatmap alignment with object by avoiding Y-axis stretching
        max_dim_h = new_h
        max_dim_w = new_w
    
    # Save content dimensions before padding (for later cropping)
    content_w = new_w
    content_h = new_h
    
    # Calculate Center Padding
    total_pad_w = max_dim_w - new_w
    total_pad_h = max_dim_h - new_h
    
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    
    # Save active padding for post-processing crop
    active_pad_left = pad_left
    active_pad_right = pad_right
    active_pad_top = pad_top
    active_pad_bottom = pad_bottom
    
    print(f"Original size: {orig_w}x{orig_h}, Resized: {new_w}x{new_h}, Square Input: {max_dim_w}x{max_dim_h}")
    print(f"Padding: Left={pad_left}, Right={pad_right}, Top={pad_top}, Bottom={pad_bottom}")
    
    # Create transform with padding (0 if disabled)
    data_transform = transforms.Compose([
        transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for mask (nearest neighbor resizing + padding)
    mask_transform = transforms.Compose([
        transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0),
        transforms.ToTensor()
    ])
    
    # Generate mask for filtering if parameters provided
    # Need cv2 image for mask generation
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    allowed_mask = get_allowed_mask(image_cv2, contour_shrink_ratio, height_lower_ratio)
    
    # Prepare mask for later application
    mask_processed = None
    if contour_shrink_ratio is not None or height_lower_ratio is not None:
        # Convert mask to PIL to apply same transforms as input image (cropping etc)
        mask_pil = Image.fromarray((allowed_mask * 255).astype(np.uint8))
        # Apply transform
        mask_tensor = mask_transform(mask_pil)
        if isinstance(mask_tensor, torch.Tensor):
            mask_processed = mask_tensor.squeeze().cpu().numpy()
        else:
            mask_processed = np.array(mask_tensor)
        # Ensure binary
        mask_processed = (mask_processed > 0.5).astype(np.float32)

    # Also incorporate DUT mask (contour_image/DUT_Corrected) if available
    # This ensures anomalies outside the DUT are ignored
    if contour_image is not None:
        # Convert to PIL
        if isinstance(contour_image, np.ndarray):
             # Ensure 0-255 range
             c_mask_pil = Image.fromarray(contour_image.astype(np.uint8))
        else:
             c_mask_pil = contour_image
             
        # Apply transform (Resize + Pad) to match model input
        c_mask_tensor = mask_transform(c_mask_pil)
        
        if isinstance(c_mask_tensor, torch.Tensor):
            c_mask_processed = c_mask_tensor.squeeze().cpu().numpy()
        else:
            c_mask_processed = np.array(c_mask_tensor)
            
        # Handle potential channel dimension if mask is 3-channel
        if len(c_mask_processed.shape) > 2:
            c_mask_processed = c_mask_processed[0]
            
        # Ensure binary
        c_mask_processed = (c_mask_processed > 0.5).astype(np.float32)
        
        # Combine
        if mask_processed is not None:
            mask_processed = mask_processed * c_mask_processed
        else:
            mask_processed = c_mask_processed
            
        print("DEBUG: Incorporated DUT_Corrected mask into anomaly map filtering.")

    img_tensor = data_transform(image).unsqueeze(0).to(device)
    print(f"DEBUG: Dino input tensor shape: {img_tensor.shape} (should be [1, 3, 1280, 1280] for [1280, 1280] input)")
    
    # Gaussian Smoothing Kernel
    # Reverted to kernel_size=5, sigma=1 per user request.
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1)
    
    with torch.no_grad():
        # Model inference
        output = model(img_tensor)
        en, de = output[0], output[1]

        # If Upsampler (AnyUp) is available, upsample encoder features
        upsampler_used = False
        if upsampler is not None:
            base_h, base_w = img_tensor.shape[-2], img_tensor.shape[-1]
            if hasattr(utils_general, 'Upsampling_Target_Size') and utils_general.Upsampling_Target_Size is not None:
                super_h = super_w = utils_general.Upsampling_Target_Size
                current_scale_factor = super_h / base_h
            else:
                super_h = int(base_h * super_resolution_factor)
                super_w = int(base_w * super_resolution_factor)
                current_scale_factor = super_resolution_factor
                
            target_size = (super_h, super_w)
            print(f"Using upsampler ({upsampler.__class__.__name__}) for super-resolution upsampling: {base_h}x{base_w} -> {super_h}x{super_w} (Scale: {current_scale_factor:.2f})")
            
            try:
                print("🚀 >>> AnyUp Process STARTED: Upsampling encoder features... <<< 🚀")
                anomaly_map, a_map_list = cal_anomaly_maps_anyup(upsampler, img_tensor, en, de, target_size)
                print("✅ >>> AnyUp Process SUCCESSFUL: Super-resolution process finished. <<< ✅")
                upsampler_used = True
            except Exception as e:
                print(f"❌ >>> AnyUp Process FAILED: {e}. Falling back to low-res features! <<< ❌")
                anomaly_map, a_map_list = cal_anomaly_maps(en, de, (img_tensor.shape[-2], img_tensor.shape[-1]))
                upsampler_used = False
        else:
            anomaly_map, a_map_list = cal_anomaly_maps(en, de, (img_tensor.shape[-2], img_tensor.shape[-1]))
            upsampler_used = False
        

# Apply Gaussian smoothing
        anomaly_map = gaussian_kernel(anomaly_map)
        
        # Process anomaly map
        current_anomaly_map = anomaly_map[0, 0].cpu().numpy()
        
        # Check if we should keep the full padded heatmap (when Dino_Input_Dim is a list like [1280, 1280])
        # This matches the Dino Crop behavior - keep the full square output
        keep_full_heatmap = is_list_input and target_max is not None
        
        if keep_full_heatmap:
            # When Dino_Input_Dim is a list (e.g., [1280, 1280]), keep the full 1280x1280 heatmap
            # Resize anomaly map to match the padded input dimensions if needed
            if current_anomaly_map.shape != (max_dim_h, max_dim_w):
                print(f"Resizing anomaly map from {current_anomaly_map.shape} to padded dimensions {max_dim_h}x{max_dim_w}")
                current_anomaly_map = cv2.resize(current_anomaly_map, (max_dim_w, max_dim_h), interpolation=cv2.INTER_LINEAR)
                if mask_processed is not None:
                    mask_processed = cv2.resize(mask_processed, (max_dim_w, max_dim_h), interpolation=cv2.INTER_NEAREST)
            
            # Update content dimensions to match the full padded size
            content_h = max_dim_h
            content_w = max_dim_w
            print(f"DEBUG: Keeping full heatmap {max_dim_h}x{max_dim_w} (Dino_Input_Dim is list)")
        else:
            # Original logic: Crop padding (Remove black borders)
            # Note: pad values need to be scaled if SR is used
            scale_factor = (utils_general.Upsampling_Target_Size / base_h) if (hasattr(utils_general, 'Upsampling_Target_Size') and utils_general.Upsampling_Target_Size is not None and upsampler_used) else (super_resolution_factor if upsampler_used else 1.0)
            
            c_pad_top = int(active_pad_top * scale_factor)
            c_pad_bottom = int(active_pad_bottom * scale_factor)
            c_pad_left = int(active_pad_left * scale_factor)
            c_pad_right = int(active_pad_right * scale_factor)
            
            # User Requirement: Keep at least 16px margin (16x aligned) in the final image
            # Scale margin for SR
            keep_margin = int(16 * scale_factor)
            
            # Adjust crop values to preserve margin
            c_pad_top = max(0, c_pad_top - keep_margin)
            c_pad_bottom = max(0, c_pad_bottom - keep_margin)
            c_pad_left = max(0, c_pad_left - keep_margin)
            c_pad_right = max(0, c_pad_right - keep_margin)
            
            # Current anomaly map is (H, W) corresponding to Padded Input
            h_curr, w_curr = current_anomaly_map.shape
            
            # Ensure cropping indices are valid
            c_pad_bottom = max(0, c_pad_bottom)
            c_pad_right = max(0, c_pad_right)
            
            # Crop Center
            end_h = h_curr - c_pad_bottom
            end_w = w_curr - c_pad_right
            
            if end_h > c_pad_top and end_w > c_pad_left:
                current_anomaly_map = current_anomaly_map[c_pad_top : end_h, c_pad_left : end_w]
                print(f"DEBUG: Cropped anomaly map from {h_curr}x{w_curr} to {current_anomaly_map.shape} (Margin={keep_margin})")
            
            # Crop mask if exists (mask was also padded)
            if mask_processed is not None:
                 h_mask, w_mask = mask_processed.shape
                 
                 # Apply same logic to mask
                 keep_margin_mask = 16
                 
                 m_pad_top = max(0, active_pad_top - keep_margin_mask)
                 m_pad_bottom = max(0, active_pad_bottom - keep_margin_mask)
                 m_pad_left = max(0, active_pad_left - keep_margin_mask)
                 m_pad_right = max(0, active_pad_right - keep_margin_mask)
                 
                 end_h_mask = h_mask - m_pad_bottom
                 end_w_mask = w_mask - m_pad_right
                 
                 if end_h_mask > m_pad_top and end_w_mask > m_pad_left:
                     mask_processed = mask_processed[m_pad_top : end_h_mask, m_pad_left : end_w_mask]
            
            # Calculate target dimensions for resizing anomaly map (matching the eventual cropped original_img)
            # This ensures binary_mask and object_mask have matching dimensions
            #            
            # Step 1: Resize anomaly map to padded input dimensions
            if current_anomaly_map.shape != (max_dim_h, max_dim_w):
                print(f"Resizing anomaly map from {current_anomaly_map.shape} to padded dimensions {max_dim_h}x{max_dim_w}")
                current_anomaly_map = cv2.resize(current_anomaly_map, (max_dim_w, max_dim_h), interpolation=cv2.INTER_LINEAR)
                
                if mask_processed is not None:
                    mask_processed = cv2.resize(mask_processed, (max_dim_w, max_dim_h), interpolation=cv2.INTER_NEAREST)
            
            # Step 2: Crop to remove padding and match original_img dimensions
            # The padding amounts in the padded image:
            # - Top/Bottom: active_pad_top, active_pad_bottom
            # - Left/Right: active_pad_left, active_pad_right
            # We want to crop exactly these padding regions to get back to content_h x content_w
            
            # Calculate crop boundaries (remove all padding)
            crop_top = active_pad_top
            crop_bottom = max_dim_h - active_pad_bottom
            crop_left = active_pad_left
            crop_right = max_dim_w - active_pad_right
            
            # Apply crop
            if crop_bottom > crop_top and crop_right > crop_left:
                current_anomaly_map = current_anomaly_map[crop_top:crop_bottom, crop_left:crop_right]
                print(f"Cropped anomaly map from {max_dim_h}x{max_dim_w} to {current_anomaly_map.shape[0]}x{current_anomaly_map.shape[1]}")
                
                if mask_processed is not None:
                    mask_processed = mask_processed[crop_top:crop_bottom, crop_left:crop_right]
            
            # Verify dimensions match content_h x content_w
            if current_anomaly_map.shape != (content_h, content_w):
                print(f"Warning: Anomaly map shape {current_anomaly_map.shape} does not match expected ({content_h}, {content_w})")
                # Fallback: resize to match
                current_anomaly_map = cv2.resize(current_anomaly_map, (content_w, content_h), interpolation=cv2.INTER_LINEAR)
                if mask_processed is not None:
                    mask_processed = cv2.resize(mask_processed, (content_w, content_h), interpolation=cv2.INTER_NEAREST)

        # Apply allowed mask to anomaly map
        if mask_processed is not None:
            # mask_processed is now resized to orig_size
            # current_anomaly_map is also orig_size
            if mask_processed.shape != current_anomaly_map.shape:
                mask_resized = cv2.resize(mask_processed, 
                                        (current_anomaly_map.shape[1], current_anomaly_map.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_processed
                
            current_anomaly_map = current_anomaly_map * mask_resized
            print("Applied mask filtering to anomaly map")

        # Save raw anomaly distances to CSV before generating heatmap
        try:
            if save_dir:
                # User request: anomaly_distance_raw.csv should be in reference folder (save_dir)
                csv_dir = save_dir
                    
                os.makedirs(csv_dir, exist_ok=True)
                # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                fname = os.path.splitext(os.path.basename(image_path))[0]
                csv_path = os.path.join(csv_dir, f"{fname}_anomaly_distance_raw.csv")
                np.savetxt(csv_path, current_anomaly_map.astype(np.float32), fmt="%.6f", delimiter=",")
                print(f"Saved anomaly distance CSV: {csv_path}")
            else:
                 print("DEBUG: save_dir is None, skipping anomaly distance CSV.")
        except Exception as e:
            print(f"Failed to save anomaly distance CSV: {e}")

        heatmap_normalized = min_max_norm(current_anomaly_map)
        
        # Keep a copy of the raw normalized heatmap for full visualization (Unfiltered)
        heatmap_normalized_raw = heatmap_normalized.copy()
        
        # Read config for threshold
        # If transparency_threshold is provided as a tuple (low, high), it overrides internal config reading
        if isinstance(transparency_threshold, (tuple, list)) and len(transparency_threshold) == 2:
            dino_threshold = transparency_threshold
            print(f"DEBUG: Using provided transparency_threshold as Dino Threshold: {dino_threshold}")
        elif isinstance(transparency_threshold, (float, int)):
            # 如果传入的是像 0.0 这样的单一浮点数（来自 fallback），我们也把它视为 raw_low 阈值
            dino_threshold = (float(transparency_threshold), None)
            print(f"DEBUG: Using provided scalar transparency_threshold as Dino Threshold: {dino_threshold}")
        else:
            dino_threshold = read_dino_threshold_config()
        
        # Determine actual threshold values
        if dino_threshold is not None:
             # Calculate Raw Min/Max for conversion
             raw_min = current_anomaly_map.min()
             raw_max = current_anomaly_map.max()
             print(f"DEBUG: Raw Anomaly Score Range: [{raw_min:.4f}, {raw_max:.4f}]")
             
             low_raw, high_raw = dino_threshold
             
             # Handle None high (default to raw_max)
             if high_raw is None:
                  high_raw = raw_max
             
             print(f"DEBUG: Using RAW threshold from config: {low_raw} - {high_raw}")
             
             # Convert RAW thresholds to Normalized thresholds for Visualization
             if raw_max > raw_min:
                 low_norm = (low_raw - raw_min) / (raw_max - raw_min)
                 high_norm = (high_raw - raw_min) / (raw_max - raw_min)
             else:
                 low_norm = 0.0
                 high_norm = 1.0
             
             current_threshold = (low_norm, high_norm)
             print(f"DEBUG: Converted to Normalized threshold (for visualization): {low_norm:.4f} - {high_norm:.4f}")
             
             # For Annotation/Calculation, use RAW threshold directly to avoid normalization artifacts
             # We do NOT update dino_threshold to normalized values here, 
             # we keep it as raw values but pass the normalized tuple to visualization functions.
        else:
            current_threshold = transparency_threshold if transparency_threshold is not None else TRANSPARENCY_THRESHOLD
            print(f"DEBUG: Using default Normalized threshold: {current_threshold}")

        # --- Flooding Logic (New) ---
        if flooding_enabled and flooding_rgb is not None:
            print(f"DEBUG: Flooding enabled. RGB: {flooding_rgb}")
            
            # Determine flooding mask (inverse of valid region)
            # Use RAW values if available for precision
            if dino_threshold is not None:
                low_raw, high_raw = dino_threshold
                if high_raw is None: high_raw = float('inf')
                
                # Regions OUTSIDE [low, high] are flooded
                flooding_mask = (current_anomaly_map < low_raw) | (current_anomaly_map > high_raw)
                print(f"DEBUG: Calculated flooding mask using RAW thresholds: < {low_raw} or > {high_raw}")
            else:
                # Fallback to normalized threshold (usually just a lower bound)
                # current_threshold is normalized (0-1)
                if isinstance(current_threshold, (list, tuple)):
                    low, high = current_threshold
                    flooding_mask = (heatmap_normalized < low) | (heatmap_normalized > high)
                else:
                    # If only single threshold (e.g. 0.2), usually we treat > 0.2 as valid.
                    # So < 0.2 is flooded.
                    flooding_mask = heatmap_normalized < current_threshold
                print(f"DEBUG: Calculated flooding mask using Normalized thresholds")

            # Apply Contour Constraint (Contour_Corrected) if available
            if contour_image is not None:
                print("DEBUG: Applying Contour_Corrected constraint to flooding mask")
                # Ensure contour_image matches original_img dimensions
                if contour_image.shape[:2] != original_img.shape[:2]:
                    contour_resized = cv2.resize(contour_image, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    contour_resized = contour_image
                
                # Assume contour_image is binary (0/255) or similar
                # Only flood INSIDE the contour
                object_mask_bool = contour_resized > 0
                flooding_mask = flooding_mask & object_mask_bool
            else:
                print("DEBUG: No Contour_Corrected provided for flooding constraint")

            # Apply flooding
            flooded_img = original_img.copy() # RGB
            
            # Apply color
            flooded_img[flooding_mask] = flooding_rgb
            
            # Save flooded image
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
                filename = os.path.splitext(os.path.basename(image_path))[0]
                # Clean filename to remove suffixes
                filename = filename.replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
                flooded_path = os.path.join(save_dir, f"{filename}_flooded.png")
                
                # Save as BGR for OpenCV
                cv2.imwrite(flooded_path, cv2.cvtColor(flooded_img, cv2.COLOR_RGB2BGR))
                print(f"Saved flooded image to: {flooded_path}")
                
                # Add to rgb_result_paths so it's tracked
                if 'rgb_result_paths' not in locals():
                    rgb_result_paths = []
                rgb_result_paths.append((flooded_path, "Flooding Result"))
            else:
                print("DEBUG: save_dir is None, skipping flooded image save.")

            # Return early as requested ("Step 3 not needed")
            # Return 5 values to match unpacking in RELP3_main.py: 
            # heatmap_normalized, heatmap_transparent, heatmap_on_image, rgb_result_paths, binary_mask
            # NOTE: User requested RGB Filtering to run AFTER Flooding.
            # Removing early return to allow subsequent processing.
            # return None, None, None, [flooded_path], None
        
        # Process original image (Moved up for Area Filtering output support)
        # Use original image for overlay to keep resolution
        original_img = np.array(image) # RGB
        
        # Check if we should keep the full padded image (when Dino_Input_Dim is a list like [1280, 1280])
        keep_full_image = is_list_input and target_max is not None
        
        if not keep_full_image:
            # Crop original_img to remove padding
            h_orig, w_orig = original_img.shape[:2]
            # Use active_pad variables
            # Note: image (PIL) was padded, so original_img (np.array(image)) contains padding.
            
            # User Requirement: Keep at least 16px margin (16x aligned) in the final image
            keep_margin_orig = 16
            
            o_pad_top = max(0, active_pad_top - keep_margin_orig)
            o_pad_bottom = max(0, active_pad_bottom - keep_margin_orig)
            o_pad_left = max(0, active_pad_left - keep_margin_orig)
            o_pad_right = max(0, active_pad_right - keep_margin_orig)
            
            end_h_orig = h_orig - o_pad_bottom
            end_w_orig = w_orig - o_pad_right
            
            # SCALAR MODE: original_img is likely CONTENT only (no padding yet).
            # If so, cropping it is WRONG unless it was padded.
            # However, in SCALAR mode, image (PIL) goes into data_transform which PADS it for Tensor.
            # But original_img = np.array(image) at line 1551 comes from 'image'.
            # If 'image' was NOT padded (scalar mode uses data_transform to pad), then original_img is clean.
            #
            # BUT: In Strategy A (Scalar) and B (Force Stretch), we update 'image' to be the RESIZED content.
            # We do NOT pad 'image' itself in Scalar mode.
            # So original_img is indeed clean content.
            #
            # THEREFORE: This cropping block is actually DANGEROUS for Scalar Mode if active_pad > 0
            # because it crops the CONTENT.
            #
            # FIX: Only crop if we know original_img has padding. 
            # In current logic, original_img does NOT have padding in Scalar Mode.
            # So we should SKIP this block or ensure pads are 0.
            # But active_pad variables ARE set (for Tensor padding).
            #
            # CONCLUSION: In Scalar Mode, original_img is already "Tight Crop" + "Resize".
            # It matches the "Content" area.
            # The Heatmap will be cropped to match this Content area (lines 1291-1327).
            # So original_img should NOT be touched.
            
            if is_list_input:
                 # In List Input mode (Old logic) we might have padded.
                 # But in New Logic, we pad 'original_img' below if keep_full_image is True.
                 # If keep_full_image is False (unlikely for List input), we might need this.
                 pass
            else:
                 # Scalar Mode: original_img is pure content. Do NOT crop.
                 print("DEBUG: Scalar Mode - Keeping original_img as is (No padding to remove).")
                 pass

            # Legacy Code Block (Commented out / Guarded to prevent wrong cropping in Scalar Mode)
            # if end_h_orig > o_pad_top and end_w_orig > o_pad_left:
            #      original_img = original_img[o_pad_top : end_h_orig, o_pad_left : end_w_orig]
            #      print(f"DEBUG: Cropped original_img from {h_orig}x{w_orig} to {original_img.shape} (Margin={keep_margin_orig})")
        else:
            print(f"DEBUG: Keeping full original_img {original_img.shape[0]}x{original_img.shape[1]} (Dino_Input_Dim is list)")
            
            # Apply Padding to original_img to match the Full Container Size (for Visualization)
            # This ensures Overlay works correctly (Heatmap is Full Container Size)
            # And 'active_pad' variables remain valid for coordinate restoration.
            if original_img.shape[0] != max_dim_h or original_img.shape[1] != max_dim_w:
                 print(f"DEBUG: Padding original_img from {original_img.shape[:2]} to {max_dim_h}x{max_dim_w} for visualization.")
                 original_img = cv2.copyMakeBorder(original_img, 
                                                 active_pad_top, active_pad_bottom, 
                                                 active_pad_left, active_pad_right, 
                                                 cv2.BORDER_CONSTANT, value=[0,0,0])

        # --- Simple Threshold Filtering ---
        # Modified to match DinoV3 behavior (conceptually) by removing strict filtering
        # This ensures low-confidence anomalies ("Blue Blocks") are not removed.
        
        if isinstance(current_threshold, (list, tuple)):
            low, high = current_threshold
            mask_filtered = (heatmap_normalized >= low) & (heatmap_normalized <= high)
        else:
            mask_filtered = heatmap_normalized > current_threshold

        # --- NEW: Centralized Filtering Logic ---
        # Replaces individual Area/Adaptive/RGB logic
        
        # Initialize masks for annotation (so they exist even if filtering is skipped or doesn't set them)
        area_filter_mask_to_remove = None
        adaptive_gaussian_mask_to_keep = None

        # Generate Full Heatmap Overlay for Visualization (used in 'Pic' output of filtering)
        # This ensures that if we save a 'Pic' inside filtering, it shows the heatmap overlay on the original image.
        # We generate it using the current heatmap and original_img.
        
        # 1. Resize heatmap to match original image (critical for overlay alignment)
        if heatmap_normalized.shape[:2] != original_img.shape[:2]:
             # Use INTER_LINEAR for sharper results (user request)
             heatmap_resized_for_vis = cv2.resize(heatmap_normalized, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
             heatmap_resized_for_vis = heatmap_normalized

        # 2. Create Solid Heatmap (JET)
        heatmap_solid_vis = cv2.applyColorMap(np.clip(255 * heatmap_resized_for_vis, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_solid_vis = cv2.cvtColor(heatmap_solid_vis, cv2.COLOR_BGR2RGB) # Ensure RGB for blending with original_img (RGB)

        # 3. Create Overlay with fixed 70% transparency (30% opacity of heatmap)
        # User request: "Load on original image with 70% transparency" -> 0.7 Original + 0.3 Heatmap
        heatmap_overlay_full = cv2.addWeighted(original_img, 0.7, heatmap_solid_vis, 0.3, 0)


        # --- Save AnyUp Overlay ---
        if upsampler_used:
             try:
                 base_name = os.path.splitext(os.path.basename(image_path))[0]
                 
                 # Save to Inferred Pic explicitly (ignoring all other conditions)
                 if inferred_pic_dir and os.path.exists(inferred_pic_dir):
                     inferred_path = os.path.join(inferred_pic_dir, f"{base_name}_overlay_anyup.png")
                     success = cv2.imwrite(inferred_path, cv2.cvtColor(heatmap_overlay_full, cv2.COLOR_RGB2BGR))
                     if success:
                         print(f"DEBUG: Saved AnyUp overlay to Inferred Pic: {inferred_path}")
                 
                 # Also save to reference dir
                 if save_dir and os.path.exists(save_dir):
                     ref_path_local = os.path.join(save_dir, f"{base_name}_overlay_anyup.png")
                     success_local = cv2.imwrite(ref_path_local, cv2.cvtColor(heatmap_overlay_full, cv2.COLOR_RGB2BGR))
                     if success_local:
                         print(f"DEBUG: Saved AnyUp overlay to reference dir: {ref_path_local}")
             except Exception as e:
                 print(f"Warning: Failed to save AnyUp overlay: {e}")
                 

        if post_ops is not None and len(post_ops) > 0:
            print(f"DEBUG: Applying Centralized Filtering with ops: {post_ops}")
            
            # Use filtering_df passed from main (formerly rgb_filtering_df)
            # Unpack 4 values now
            # Pass contour_image as mask_default to ensure all filtering is constrained to DUT region
            mask_filtered, new_paths, area_filter_mask_to_remove, adaptive_gaussian_mask_to_keep, _ = apply_filtering(
                mask_filtered, original_img, post_ops, filtering_df,
                save_dir, image_path, contour_image=contour_image,
                adaptive_gaussian_config=adaptive_gaussian_config,
                visualization_overlay=heatmap_overlay_full,
                defect_id_method=defect_id_method if defect_id_method else 'Dino',
                mask_default=contour_image
            )
            
            if 'rgb_result_paths' not in locals(): rgb_result_paths = []
            rgb_result_paths.extend(new_paths)

        # Check for 'Save' after Filtering (if explicitly requested as a standalone op, handled in apply_filtering)
        # But if 'Save' was part of the last filtering op params, it's handled there.
        # This fallback is for legacy or mixed usage if needed.

        heatmap_normalized = heatmap_normalized * mask_filtered
        
        # Use new transparent heatmap generation
        # heatmap_transparent = generate_transparent_heatmap(heatmap_normalized, threshold=current_threshold)
        
        if heatmap_normalized.shape != (original_img.shape[0], original_img.shape[1]):
            # Use INTER_LINEAR for sharper results (user request)
            heatmap_normalized = cv2.resize(heatmap_normalized, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Generate overlay image
        # Logic: If centralized filtering was applied (post_ops present)
        # OR legacy methods were applied (fallback)
        
        use_filtered_overlay = False
        if (post_ops and len(post_ops) > 0) or adaptive_gaussian_config is not None:
            use_filtered_overlay = True
        
        if use_filtered_overlay:
            print("DEBUG: Generating Filtered Dino Overlay (70% Transparency) for Post-Processing Result.")
            
            # Ensure mask is boolean/binary and same size as original_img
            if mask_filtered.shape != (original_img.shape[0], original_img.shape[1]):
                 mask_for_overlay = cv2.resize(mask_filtered.astype(np.uint8), (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                 mask_for_overlay = mask_filtered.astype(np.uint8)
            
            # [修复]: 之前这里强行生成了一个 [0, 162, 255] (橙色/蓝色) 的纯色色块！
            # 这是导致所有 Dino 的热力图全部变成了纯蓝色 Mask 的罪魁祸首！
            # 我们应该直接使用上面已经算好的 heatmap_solid_vis (这是由 JET 色图生成的真实热力颜色)
            
            # 使用真实的热力颜色进行混合
            # We want: Final = Original * 0.7 + Heatmap * 0.3
            blended_heatmap = cv2.addWeighted(original_img, 0.7, heatmap_solid_vis, 0.3, 0)
            
            heatmap_on_image = original_img.copy()
            
            mask_indices = (mask_for_overlay > 0)
            if np.any(mask_indices):
                heatmap_on_image[mask_indices] = blended_heatmap[mask_indices]
                
            heatmap_transparent = heatmap_on_image # For consistency if used elsewhere
            
        else:
            # Fallback to original heatmap overlay for raw/unfiltered cases
            # Note: show_cam_on_image_transparent was already called to create heatmap_overlay_full
            # But if we didn't enter the filtering block, we might rely on re-generating it or using it.
            # Actually we generated heatmap_overlay_full above unconditionally.
            heatmap_on_image = heatmap_overlay_full
            heatmap_transparent = generate_transparent_heatmap(heatmap_normalized, threshold=current_threshold)
        
        # Annotation if config is used (Dino_TH present)
        if dino_threshold is not None:
            print(f"DEBUG: Annotating with threshold {dino_threshold}")
            low_raw, high_raw = dino_threshold
            if high_raw is None: high_raw = float('inf')

            # Calculate binary mask based on RAW threshold directly
            # This ensures we respect the raw values exactly as requested
            binary_mask = ((current_anomaly_map >= low_raw) & (current_anomaly_map <= high_raw)).astype(np.uint8) * 255
            
            # Calculate object area (non-black pixels in original image)
            gray_temp = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            # Use a slightly higher threshold to ignore JPEG artifacts in black background
            _, object_mask = cv2.threshold(gray_temp, 10, 255, cv2.THRESH_BINARY)
            if object_mask.shape[:2] != binary_mask.shape[:2]:
                object_mask = cv2.resize(object_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            if object_mask.dtype != np.uint8:
                object_mask = object_mask.astype(np.uint8)
            
            # Ensure binary_mask is strictly within the object to avoid background noise
            binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=object_mask)
            
            # Save pre-filtering mask for calculating removed parts
            binary_mask_pre_filter = binary_mask.copy()
            
            # --- NEW: Apply Area Filtering to Annotation Binary Mask ---
            if area_filter_mask_to_remove is not None:
                print("DEBUG: Applying Area Filtering Mask to Annotation/Calculated Mask.")
                # area_filter_mask_to_remove is 255 where we want to remove
                # binary_mask is (H, W). area_filter_mask_to_remove is (H, W).
                # Ensure sizes match (should match as they are both from padded output logic, but good to check)
                if binary_mask.shape == area_filter_mask_to_remove.shape:
                    keep_mask_annot = (area_filter_mask_to_remove == 0).astype(np.uint8) * 255
                    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=keep_mask_annot)
                else:
                    print(f"Warning: Shape mismatch for Area Filtering in Annotation: {binary_mask.shape} vs {area_filter_mask_to_remove.shape}")
            
            # --- NEW: Apply Adaptive Gaussian Mask to Annotation Binary Mask ---
            if adaptive_gaussian_mask_to_keep is not None:
                print("DEBUG: Applying Adaptive Gaussian Mask to Annotation/Calculated Mask.")
                # adaptive_gaussian_mask_to_keep is boolean (True = Keep)
                # Ensure dimensions match
                if binary_mask.shape == adaptive_gaussian_mask_to_keep.shape:
                    keep_mask_ag = adaptive_gaussian_mask_to_keep.astype(np.uint8) * 255
                    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=keep_mask_ag)
                else:
                    print(f"DEBUG: Adaptive Gaussian Mask size mismatch. Skipping application to Annotation Mask. {binary_mask.shape} vs {adaptive_gaussian_mask_to_keep.shape}")
            
            object_area = cv2.countNonZero(object_mask)
            
            # Calculate defect area
            defect_area = cv2.countNonZero(binary_mask)
            ratio = defect_area / object_area if object_area > 0 else 0
            
            print(f"DEBUG: Object Area: {object_area}, Defect Area: {defect_area}, Ratio: {ratio}")

            # Draw object contour (Green)
            contours_obj, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heatmap_on_image, contours_obj, -1, (0, 255, 0), 1)
            
            # Draw Defect Contours (Yellow) - DISABLED per user request
            # In RGB, Yellow is (255, 255, 0)
            # contours_defect, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(heatmap_on_image, contours_defect, -1, (255, 255, 0), 1)
            
            # Calculate object bounding box area for ratio
            object_bbox_area = 0
            if contours_obj:
                # Find largest contour by area
                largest_obj_contour = max(contours_obj, key=cv2.contourArea)
                x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(largest_obj_contour)
                object_bbox_area = w_obj * h_obj
                
                # Draw Object Bounding Box (Red) - User Requested
                cv2.rectangle(heatmap_on_image, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), (255, 0, 0), 2)
                print(f"DEBUG: Drawn Object BBox: {x_obj},{y_obj},{w_obj},{h_obj}")
            
            # Calculate defect area
            defect_area = cv2.countNonZero(binary_mask)
            
            # Ratio = Defect Area / Object Area (Pixel Count)
            # Use object_area (non-zero pixels) instead of object_bbox_area for more accurate defect percentage
            ratio = defect_area / object_area if object_area > 0 else 0
            
            print(f"DEBUG: Object Area: {object_area}, Defect Area: {defect_area}, Ratio: {ratio}")

            # Draw text
            text = f"Defect Area% {ratio:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            width = heatmap_on_image.shape[1]
            text_x = max(10, width - text_size[0] - 30)
            cv2.putText(heatmap_on_image, text, (text_x, 30), font, font_scale, color, thickness)
            
            print(f"Annotated defect area: {ratio:.2%} (Threshold: {low_raw}-{high_raw})")
        else:
            print("DEBUG: Skipping annotation because dino_threshold is None")
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Get original filename (without extension) for overlay naming
            # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
            original_filename = os.path.splitext(os.path.basename(image_path))[0]
            # Clean filename for other uses (remove suffixes)
            filename = original_filename.replace('_rotated90_cropped', '').replace('_rotated90', '').replace('_cropped', '').replace('_Corrected', '')
            
            if upsampler is not None and super_resolution_factor > 1.0:
                suffix = f"_sr{super_resolution_factor}x"
            else:
                suffix = "_upsampled" if upsampler is not None else ""
        
        # cv2.imwrite(os.path.join(save_dir, f"{filename}_original{suffix}.png"), 
        #            cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        
        # Save Standard Solid Heatmap (Blue Background) - For raw visualization
        # Use unfiltered data to show full distribution including noise
        # CRITICAL FIX: Ensure heatmap matches original_img dimensions for proper alignment
        if save_dir:
            heatmap_to_save = heatmap_normalized_raw.copy()
            if heatmap_to_save.shape != (original_img.shape[0], original_img.shape[1]):
                heatmap_to_save = cv2.resize(heatmap_to_save, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap_solid = cv2.applyColorMap(np.clip(255 * heatmap_to_save, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # The 'original_img' here is NOT the raw image! It is the padded 1280x1280 image fed to DINO!
            # We MUST use restore_coords_func so that this heatmap actually strips the padding and reverts to the tight bbox!
            if 'restore_coords_func' in locals():
                b = restore_coords_func(heatmap_solid[:,:,0])
                g = restore_coords_func(heatmap_solid[:,:,1])
                r = restore_coords_func(heatmap_solid[:,:,2])
                if b is not None and g is not None and r is not None:
                    heatmap_solid = cv2.merge([b, g, r])
                    
            cv2.imwrite(os.path.join(save_dir, f"{filename}_heatmap{suffix}.png"), heatmap_solid)
        
        if upsampler is not None and super_resolution_factor > 1.0:
            print(f"Processed resolution: {super_h}×{super_w}, Output resolution: {orig_h}×{orig_w}")
        else:
            print(f"Enhanced results saved to: {save_dir}")
        
        # --- Post-processing: RGB Filtering (Legacy Removed) ---
        # Logic moved to apply_filtering in utils_general.py
        if 'rgb_result_paths' not in locals():
            rgb_result_paths = []


        # -------------------------------------------------------------------

        # Custom Output Params Handling
        if output_params:
             print(f"DEBUG: Processing custom output params: {output_params}")
             
             # Calculate binary_mask if not already done
             if 'binary_mask' not in locals():
                 low, high = current_threshold if isinstance(current_threshold, (list, tuple)) else (current_threshold, 1.0)
                 binary_mask = ((heatmap_normalized >= low) & (heatmap_normalized <= high)).astype(np.uint8) * 255
             
             # Use os.path.splitext to handle filenames with dots (e.g., "0.08")
             filename_base = os.path.splitext(os.path.basename(image_path))[0]
             
             if upsampler is not None and super_resolution_factor > 1.0:
                 suffix = f"_sr{super_resolution_factor}x"
             else:
                 suffix = "_upsampled" if upsampler is not None else ""

             if 'pic' in output_params and save_dir:
                 # Save masked image (Original pixels where mask is white)
                 masked_img = cv2.bitwise_and(original_img, original_img, mask=binary_mask)
                 cv2.imwrite(os.path.join(save_dir, f"{filename_base}_pic{suffix}.png"), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                 print(f"Saved custom output: pic")
                 
             if 'mask' in output_params and save_dir:
                 cv2.imwrite(os.path.join(save_dir, f"{filename_base}_mask{suffix}.png"), binary_mask)
                 print(f"Saved custom output: mask")
                 
             if 'contour' in output_params and save_dir:
                if contour_image is not None:
                    cv2.imwrite(os.path.join(save_dir, f"{filename_base}_contour{suffix}.png"), contour_image)
                    print(f"Saved custom output: contour")
                else:
                    print("Warning: 'contour' requested but contour_image is None.")

       # Ensure binary_mask exists for return
        if 'binary_mask' not in locals():
             low, high = current_threshold if isinstance(current_threshold, (list, tuple)) else (current_threshold, 1.0)
             # Handle None threshold
             if low is None: low = 0.5
             binary_mask = ((heatmap_normalized >= low) & (heatmap_normalized <= high)).astype(np.uint8) * 255
             
             if 'binary_mask_pre_filter' not in locals():
                 binary_mask_pre_filter = binary_mask.copy()
        
        # Calculate mask_removed (pixels present in pre-filter but removed in final)
        if 'binary_mask_pre_filter' in locals() and binary_mask is not None:
            # Ensure sizes match
            if binary_mask_pre_filter.shape == binary_mask.shape:
                mask_removed = cv2.subtract(binary_mask_pre_filter, binary_mask)
            else:
                mask_removed = None
        else:
            mask_removed = None

        # --- Coordinate Restoration Logic ---
        # Restore binary_mask AND mask_removed to full_input_image coordinates (DUT_Corrected)
        # This ensures that the returned mask aligns with bbox_general_FM in RELP3_main.py
        try:
            if 'binary_mask' in locals() and binary_mask is not None:
                # 1. Remove Padding (that was left in original_img)
                # We need to calculate how much padding is in binary_mask
                
                # Recalculate remaining padding (same logic as used to create original_img)
                # active_pad_... are the total pads added to make it square
                # o_pad_... are the pads removed to get back to content (with margin)
                
                # Default to 0 if variables not found (safety)
                ap_t = active_pad_top if 'active_pad_top' in locals() else 0
                ap_b = active_pad_bottom if 'active_pad_bottom' in locals() else 0
                ap_l = active_pad_left if 'active_pad_left' in locals() else 0
                ap_r = active_pad_right if 'active_pad_right' in locals() else 0
                
                op_t = o_pad_top if 'o_pad_top' in locals() else 0
                op_b = o_pad_bottom if 'o_pad_bottom' in locals() else 0
                op_l = o_pad_left if 'o_pad_left' in locals() else 0
                op_r = o_pad_right if 'o_pad_right' in locals() else 0

                h_curr, w_curr = binary_mask.shape[:2]
                
                if not keep_full_heatmap and (h_curr == content_h and w_curr == content_w):
                    # In scalar mode, the padding was completely removed from the anomaly map
                    # so binary_mask is exactly content_h x content_w with NO padding.
                    y_start = 0
                    y_end = h_curr
                    x_start = 0
                    x_end = w_curr
                else:
                    # Legacy logic or keep_full_heatmap mode
                    rem_pad_top = max(0, ap_t - op_t)
                    rem_pad_bottom = max(0, ap_b - op_b)
                    rem_pad_left = max(0, ap_l - op_l)
                    rem_pad_right = max(0, ap_r - op_r)
                    
                    y_start = rem_pad_top
                    y_end = h_curr - rem_pad_bottom
                    x_start = rem_pad_left
                    x_end = w_curr - rem_pad_right
                
                # Helper function to restore coordinates
                def restore_coords_func(mask_input):
                    # CRITICAL FIX for misalignment:
                    # tight_crop_w and tight_crop_h are the size of the SAM crop BEFORE Dino_Input_Dim resizing.
                    # When Dino_Input_Dim resizes the input, it may pad OR just scale.
                    # For scalar mode (Aspect Ratio Preserved), we scaled by `scale` without forcing square pad inside the content area.
                    # So mask_content (which is content_w x content_h) exactly corresponds to tight_crop_w x tight_crop_h scaled by `scale`.
                    # Resizing mask_content directly to (tight_crop_w, tight_crop_h) is mathematically perfect.
                    # IF there is a misalignment, it means either:
                    # 1. crop_off_x/crop_off_y is wrong
                    # 2. padding was stripped incorrectly
                    # 3. OpenCV resize vs Numpy shape mismatch.
                    if mask_input is None: return None
                    print(f"DEBUG: restore_coords_func: ")
                    print(f"  - mask_input.shape (Dino Output): {mask_input.shape}")
                    print(f"  - full_input_image.shape (DUT_Corrected): {full_input_image.shape}")
                    print(f"  - crop_offset (x,y): {crop_off_x}, {crop_off_y}")
                    print(f"  - tight_crop dims (w,h): {tight_crop_w}, {tight_crop_h}")
                    print(f"  - resize coordinates: y={y_start}:{y_end}, x={x_start}:{x_end}")
                    if y_end > y_start and x_end > x_start:
                        mask_content = mask_input[y_start:y_end, x_start:x_end]
                        
                        # 2. Resize to tight_crop size (Reverses the scaling)
                        # tight_crop_w/h are the dimensions of the crop from the original image
                        # The tight_crop_w/h is the dimension BEFORE we applied `scale = available_max / max_curr`
                        # The mask_content is the dimension AFTER `new_w = int(orig_w * scale)` and `new_h = int(orig_h * scale)`
                        # So by resizing from `mask_content` (which is roughly `content_w` x `content_h`) back to `tight_crop_w` x `tight_crop_h`,
                        # we are perfectly reversing the aspect-ratio-preserving scaling we did earlier.
                        print(f"DEBUG: Resizing mask_content {mask_content.shape} to tight_crop {tight_crop_w}x{tight_crop_h}")
                        mask_restored_crop = cv2.resize(mask_content.astype(np.uint8), (tight_crop_w, tight_crop_h), interpolation=cv2.INTER_NEAREST)
                        
                        # 3. Place into full image
                        # full_input_image was saved at start
                        final_mask = np.zeros((full_input_h, full_input_w), dtype=np.uint8)
                        
                        # Handle boundaries
                        y1 = crop_off_y
                        y2 = crop_off_y + tight_crop_h
                        x1 = crop_off_x
                        x2 = crop_off_x + tight_crop_w
                        
                        # Ensure we fit in full image
                        y2 = min(y2, full_input_h)
                        x2 = min(x2, full_input_w)
                        
                        # Resize mask_restored_crop if needed (if tight_crop vars were slightly off or rounding)
                        target_h = y2 - y1
                        target_w = x2 - x1
                        
                        if target_h > 0 and target_w > 0:
                            if mask_restored_crop.shape[0] != target_h or mask_restored_crop.shape[1] != target_w:
                                mask_restored_crop = cv2.resize(mask_restored_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                            
                            final_mask[y1:y2, x1:x2] = mask_restored_crop
                            return final_mask
                    return None

                # Restore both masks
                binary_mask = restore_coords_func(binary_mask)
                mask_removed = restore_coords_func(mask_removed)
                
                if binary_mask is not None:
                    # x1/y1 are local to restore_coords_func, use crop_off_x/y from outer scope
                    print(f"DEBUG: Restored binary_mask to full image size: {full_input_w}x{full_input_h} (Offset: {crop_off_x},{crop_off_y})")
                else:
                    print("Warning: Target size for mask restoration is 0. Returning original.")
            else:
                # If binary_mask is None, mask_removed is also None (or irrelevant)
                pass
                    
        except Exception as e:
            print(f"Error during coordinate restoration: {e}")
            import traceback
            traceback.print_exc()

        # -------------------------------------------------------------
        # 移除向 Inferred Pic 输出 _overlay_upsampled.jpg 掩码图的遗留逻辑。
        # 现代架构中，最终的二值化轮廓掩码（_overlay.jpg）已由 Strategy 层
        # 直接调用 cv_ops 负责生成并统一管理。此处只需在 save_dir 备份一张
        # 原生渐变热力图（_overlay.png），作为算法调试参考即可。
        # -------------------------------------------------------------
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            heatmap_on_image_to_save = heatmap_on_image
            if 'restore_coords_func' in locals() and heatmap_on_image.shape[:2] != full_input_image.shape[:2]:
                b = restore_coords_func(heatmap_on_image[:,:,0])
                g = restore_coords_func(heatmap_on_image[:,:,1])
                r = restore_coords_func(heatmap_on_image[:,:,2])
                if b is not None and g is not None and r is not None:
                    heatmap_on_image_to_save = cv2.merge([b, g, r])
                    
            overlay_save_path = os.path.join(save_dir, f"{filename}_overlay{suffix}.png")
            cv2.imwrite(
                overlay_save_path,
                cv2.cvtColor(heatmap_on_image_to_save, cv2.COLOR_RGB2BGR)
            )
            # 兼容性保证：如果带有 _upsampled 等后缀，额外多存一份标准 _overlay.png，确保各 Strategy 均能命中
            if suffix:
                cv2.imwrite(
                    os.path.join(save_dir, f"{filename}_overlay.png"),
                    cv2.cvtColor(heatmap_on_image_to_save, cv2.COLOR_RGB2BGR)
                )

        return heatmap_normalized, heatmap_transparent, heatmap_on_image, rgb_result_paths, binary_mask, mask_removed
