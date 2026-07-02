import argparse
import os
# Enable MPS fallback for ops not implemented on MPS (e.g. upsample_bicubic2d)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
import sys
use_local_modules = str(os.getenv("RELP3_USE_LOCAL_MODULES", "")).strip().lower() in ("1", "true", "yes", "on")

# Add custom module paths
current_dir = os.path.dirname(os.path.abspath(__file__))
detectron_path = os.path.join(current_dir, 'detectron_all')
sam2_path = os.path.join(current_dir, 'weights', 'sam2')

if detectron_path not in sys.path:
    sys.path.append(detectron_path)

if sam2_path not in sys.path:
    sys.path.append(sam2_path)

if use_local_modules:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    detectron_path = os.path.join(current_dir, "detectron_all")
    sam2_path = os.path.join(current_dir, "weights", "sam2")
    if detectron_path not in sys.path:
        sys.path.append(detectron_path)
    if sam2_path not in sys.path:
        sys.path.append(sam2_path)
import pandas as pd
import cv2
import numpy as np
import torch
import re
import gc  # OPTIMIZATION: Import garbage collector for memory management
import PIL.Image
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

import os
import cv2
import numpy as np
import gc
import pandas as pd


from utils import utils_general
try:
    from utils import scoring_utils
except ImportError:
    scoring_utils = None
    print("Warning: scoring_utils is not available")
    
try:
    from utils import knn_utils
except ImportError:
    knn_utils = None
    print("Warning: knn_utils is not available")



from utils.bbox_ops import calculate_bbox_iou, create_bbox_mask, calculate_defect_ratio_in_bbox, filter_mask_by_bbox_iou, parse_bbox_iou_scope
from utils.wrappers.detectron_wrapper import load_detectron_model
from utils.wrappers.sam2_wrapper import init_sam2, get_sam2_predictor

class DefectDetectionPipeline:
    def __init__(self):
        pass

    def _resolve_model_weights(self, fm, product, generation, weights_df):
        import pandas as pd
        import os
        from utils.base_utils import _norm
        
        result = {}
        if weights_df is None or weights_df.empty:
            return result
            
        def _resolve_rel_or_abs(val):
            if pd.isna(val): return None
            s = str(val).strip()
            if s == '': return None
            if os.path.isabs(s): return s
            return os.path.abspath(os.path.join(os.getcwd(), s))

        # Build candidate keys in order of precedence
        key_candidates = []
        if product and generation and fm:
            key_candidates.append(f"{product}_{generation}_{fm}")
            key_candidates.append(f"{product} {generation} {fm}")
            key_candidates.append(f"{product}-{generation}-{fm}")
            key_candidates.append(f"{product}{generation}{fm}")
        if product and generation:
            key_candidates.append(f"{product}_{generation}")
            key_candidates.append(f"{product} {generation}")
            key_candidates.append(f"{product}-{generation}")
        if product:
            key_candidates.append(f"{product}")
        if fm:
            key_candidates.append(f"{fm}")
        key_candidates.append("General")

        cols = list(weights_df.columns)
        if not cols: return result
        
        first_col = cols[0]
        type_col = next((c for c in cols if _norm(c) == 'type'), None)
        vn_col = next((c for c in cols if _norm(c) in ('variablename', 'variable_name')), None)
        path_col = next((c for c in cols if _norm(c) in ('relativepath', 'relative_path', 'path')), None)

        if not type_col or not vn_col or not path_col:
            return result

        df_keys = weights_df[first_col].astype(str).apply(_norm)
        target_keys = [_norm(k) for k in key_candidates if k]

        # Order matches by priority of the target_keys
        matches = weights_df[df_keys.isin(target_keys)]
        
        if not matches.empty:
            # Sort by target key priority (earlier keys = higher priority)
            matches_list = []
            for k in target_keys:
                m = weights_df[df_keys == k]
                if not m.empty:
                    matches_list.append(m)
            if matches_list:
                matches = pd.concat(matches_list)

            for _, row in matches.iterrows():
                t_val = _norm(row.get(type_col))
                vn_val = _norm(row.get(vn_col))
                raw_path = row.get(path_col)
                p_val = _resolve_rel_or_abs(raw_path)
                
                if not p_val: continue
                
                # Check for DINO first because its Type can be 'Defect' but variable_name is 'Dino'
                if 'dino' in t_val or 'dino' in vn_val:
                    if ('weight' in vn_val or 'pth' in vn_val or 'dino' in vn_val) and 'dino' not in result: result['dino'] = p_val
                elif 'dut' in t_val:
                    if 'cfg' in vn_val and 'cfg_path_dut' not in result: result['cfg_path_dut'] = p_val
                    if 'weight' in vn_val and 'weights_path_dut' not in result: result['weights_path_dut'] = p_val
                elif 'defect' in t_val or 'detectron' in t_val or 'seg' in t_val:
                    if 'cfg' in vn_val and 'cfg_path' not in result: result['cfg_path'] = p_val
                    if 'weight' in vn_val and 'weights_path' not in result: result['weights_path'] = p_val
                elif 'sam' in t_val:
                    if ('config' in vn_val or 'cfg' in vn_val) and 'sam_config' not in result: result['sam_config'] = p_val
                    if ('weight' in vn_val or 'checkpoint' in vn_val or 'pth' in vn_val) and 'sam_weights' not in result: result['sam_weights'] = p_val
        return result

    def run(self):
        run()

def run():

    print("Starting RELP3 Main Pipeline...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RELP3 Main Pipeline')
    parser.add_argument('--pic_path', type=str, help='Override Pic_Path from configuration')
    parser.add_argument('--product', type=str, help='Filter by Product')
    parser.add_argument('--generation', type=str, help='Filter by Generation')
    parser.add_argument('--failure_mode', type=str, help='Filter by Failure Mode')
    parser.add_argument('--downloading_radar', type=str, help='Downloading Radar ID for Auto Run')
    
    # Use parse_known_args to avoid issues if other args are passed or if run without args
    args, unknown = parser.parse_known_args()
    
    override_pic_path = args.pic_path
    filter_product = args.product
    filter_generation = args.generation
    filter_failure_mode = args.failure_mode
    downloading_radar = args.downloading_radar
    
    if override_pic_path:
        print(f"DEBUG: Overriding Pic_Path with: {override_pic_path}")
    if filter_product or filter_generation or filter_failure_mode:
        print(f"DEBUG: Filtering execution for Product='{filter_product}', Gen='{filter_generation}', FM='{filter_failure_mode}'")

    config_path = os.path.join(os.getcwd(), 'RELP_Configuration.xlsx')
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Helper functions defined at top
    def _norm(s):
        s = str(s)
        res = ''.join(ch.lower() if ch.isalnum() else '_' for ch in s).strip('_')
        while '__' in res:
            res = res.replace('__', '_')
        return res

    def resolve_path(path_val):
        if pd.isna(path_val) or path_val == '':
            return None
        path_str = str(path_val).strip()
        if not os.path.isabs(path_str):
            return os.path.abspath(path_str)
        return path_str

    def _resolve_rel_or_abs(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == '':
            return None
        if os.path.isabs(s):
            return s
        return os.path.abspath(os.path.join(os.getcwd(), s))

    def set_from_row(row, col_names, attr):
        for cn in col_names:
            if cn in row.index:
                p = _resolve_rel_or_abs(row.get(cn))
                if p:
                    setattr(utils_general, attr, p)
                    return True
        return False

    try:
        fm_df = pd.read_excel(config_path, sheet_name='Failure Mode')
        flow_df = pd.read_excel(config_path, sheet_name='Flow')
        weights_df = pd.read_excel(config_path, sheet_name='Weights')
        print(f"DEBUG: Failure Mode sheet loaded. Rows: {len(fm_df)}, Columns: {list(fm_df.columns)}")
        print(f"DEBUG: Flow sheet loaded. Rows: {len(flow_df)}, Columns: {list(flow_df.columns)}")

        try:
            # Robustly load Adaptive Gaussian sheet
            xl_temp = pd.ExcelFile(config_path)
            ag_sheet_name = next((s for s in xl_temp.sheet_names if _norm(s).replace('_', '') == 'adaptivegaussian'), None)
            if ag_sheet_name:
                adaptive_gaussian_df = pd.read_excel(config_path, sheet_name=ag_sheet_name)
                print(f"DEBUG: Adaptive Gaussian sheet loaded successfully as '{ag_sheet_name}'. Rows: {len(adaptive_gaussian_df)}")
            else:
                adaptive_gaussian_df = pd.DataFrame()
                print(f"DEBUG: Adaptive Gaussian sheet not found. Available sheets: {xl_temp.sheet_names}")
        except Exception as e:
            print(f"Warning: Error loading Adaptive Gaussian sheet: {e}")
            adaptive_gaussian_df = pd.DataFrame()
        
        # --- Load General FM sheet (for Defect Detection BBOX scope) ---
        general_fm_df = pd.DataFrame()
        try:
            xl_temp = pd.ExcelFile(config_path)
            general_fm_sheet_name = next((s for s in xl_temp.sheet_names if _norm(s).replace('_', '') == 'generalfm'), None)
            if general_fm_sheet_name:
                general_fm_df = pd.read_excel(config_path, sheet_name=general_fm_sheet_name)
                print(f"DEBUG: General FM sheet loaded successfully as '{general_fm_sheet_name}'. Rows: {len(general_fm_df)}, Columns: {list(general_fm_df.columns)}")
            else:
                print(f"DEBUG: General FM sheet not found. Available sheets: {xl_temp.sheet_names}")
        except Exception as e:
            print(f"Warning: Error loading General FM sheet: {e}")
            general_fm_df = pd.DataFrame()

        # --- Resolve SAM2 Weights (Global) ---
        # 1. Check Default Paths (Priority 1: "This Address" / Default Folder)
        default_sam_root = os.path.join(os.getcwd(), 'weights', 'sam2')
        default_sam_checkpoints = os.path.join(default_sam_root, 'checkpoints')
        
        # Candidate paths for weights and config
        sam_weight_candidates = [
            os.path.join(default_sam_checkpoints, 'sam2.1_hiera_tiny.pt'), # Screenshot structure
            os.path.join(default_sam_root, 'sam2.1_hiera_tiny.pt') # Simplified structure
        ]
        sam_config_candidates = [
            os.path.join(default_sam_checkpoints, 'configs', 'sam2.1', 'sam2.1_hiera_t.yaml'), # Screenshot structure
            os.path.join(default_sam_root, 'sam2.1_hiera_t.yaml') # Simplified structure
        ]
        
        found_sam_weight = next((p for p in sam_weight_candidates if os.path.exists(p)), None)
        found_sam_config = next((p for p in sam_config_candidates if os.path.exists(p)), None)
        
        if found_sam_weight and found_sam_config:
            utils_general.sam_checkpoint_path = found_sam_weight
            utils_general.sam_config_path = found_sam_config
            print(f"Global SAM2 configuration found in default path: {utils_general.sam_config_path}, {utils_general.sam_checkpoint_path}")
        else:
            # 2. If not found in default paths, check Weights sheet (Priority 2)
            try:
                 cols = weights_df.columns
                 col_norm_map = {c: _norm(c) for c in cols}
                 
                 # Strategy A: Horizontal (Product_Gen=SAM, then columns sam_weights etc.)
                 first_col = cols[0]
                 
                 df_keys_norm = weights_df[first_col].astype(str).apply(_norm)
                 sam_keywords = ['sam', 'sam2', 'segment_anything', 'general', 'global']
                 sam_row_idx = df_keys_norm[df_keys_norm.isin(sam_keywords)].index
                 
                 if len(sam_row_idx) > 0:
                      # Check if it is Vertical format (screenshot style)
                      # Vertical: Product_Gen=SAM, Type=SAM2, variable_name=weight, relative_path=...
                      vn_col = next((c for c in cols if col_norm_map[c] in ('variable_name', 'variable', 'variablename')), None)
                      rp_col = next((c for c in cols if col_norm_map[c] in ('relative_path', 'path', 'value', 'relativepath')), None)
                      type_col = next((c for c in cols if col_norm_map[c] in ('type', 'model_type')), None)
                      
                      if vn_col and rp_col:
                          # Iterate all rows matching SAM
                          sam_rows = weights_df.loc[sam_row_idx]
                          
                          for _, row in sam_rows.iterrows():
                              vn_val = _norm(row.get(vn_col))
                              path_val = row.get(rp_col)
                              type_val = _norm(row.get(type_col)) if type_col else ''
                              
                              # Global Init: Default to Regular only (Skip Onnx) to avoid mixing or default-locking to Onnx
                              if 'onnx' in type_val or 'onnx' in vn_val or 'onnx' in str(path_val).lower():
                                  continue

                              if 'weight' in vn_val or 'checkpoint' in vn_val:
                                  utils_general.sam_checkpoint_path = _resolve_rel_or_abs(path_val)
                              elif 'config' in vn_val or 'cfg' in vn_val:
                                  utils_general.sam_config_path = _resolve_rel_or_abs(path_val)
                      else:
                          # Horizontal format fallback
                          sam_row = weights_df.loc[sam_row_idx[0]]
                          set_from_row(sam_row, ['sam_config', 'sam_cfg', 'sam2_config', 'sam2_cfg', 'config', 'cfg'], 'sam_config_path')
                          set_from_row(sam_row, ['sam_weights', 'sam_checkpoint', 'sam_pt', 'sam2_weights', 'sam2_checkpoint', 'weights', 'checkpoint'], 'sam_checkpoint_path')
                 
                 if utils_general.sam_config_path and utils_general.sam_checkpoint_path:
                      print(f"Global SAM2 configuration resolved from Weights sheet: {utils_general.sam_config_path}, {utils_general.sam_checkpoint_path}")
                      
                      # Initialize SAM2 model globally
                      # try:
                      #     device_str = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu")
                      #     utils_general.init_sam2(utils_general.sam_config_path, utils_general.sam_checkpoint_path, device=device_str)
                      # except Exception as e:
                      #     print(f"Warning: Failed to initialize SAM2 model: {e}")
                      print("DEBUG: Global SAM2 Init skipped to allow per-row SAM Mode override.")
                 else:
                      print("Warning: SAM2 config or checkpoint path missing after Weights sheet check.")

            except Exception as e:
                 print(f"Warning: Error resolving global SAM weights from Excel: {e}")
        
        try:
            fine_tuning_df = pd.read_excel(config_path, sheet_name='Fine_Tuning')
        except Exception:
            fine_tuning_df = pd.DataFrame()

        # Load Reference Sheet
        try:
            reference_df = pd.read_excel(config_path, sheet_name='Reference')
            print("DEBUG: Reference sheet loaded successfully.")
        except Exception:
            reference_df = pd.DataFrame()
            print("DEBUG: Reference sheet not found or empty.")

        # Load Scaling Sheet (New)
        try:
            scaling_df = pd.read_excel(config_path, sheet_name='Scaling')
            print("DEBUG: Scaling sheet loaded successfully.")
        except Exception:
            scaling_df = pd.DataFrame()
            print("DEBUG: Scaling sheet not found or empty.")

        # Read Dino_TH if it exists
        try:
             # User mentioned adding Dino_TH, we assume it's a new sheet
             dino_th_df = pd.read_excel(config_path, sheet_name='Dino_TH')
             print("DEBUG: Dino_TH sheet loaded successfully.")
        except Exception:
             # Fallback: maybe it's in the Failure Mode sheet or another sheet? 
             # For now, initialize empty DF if not found
             print("DEBUG: Dino_TH sheet not found, trying to read columns from Failure Mode sheet if present.")
             dino_th_df = pd.DataFrame()

        try:
             # Try to find sheet name robustly
             xl = pd.ExcelFile(config_path)
             sheet_names = xl.sheet_names
             # Added 'filtering' to the accepted sheet names
             rgb_sheet_name = next((s for s in sheet_names if _norm(s).replace('_', '') in ('filtering', 'filteringrgb', 'rgbfiltering', 'filtering_rgb')), None)
             
             if rgb_sheet_name:
                 rgb_filtering_df = pd.read_excel(config_path, sheet_name=rgb_sheet_name)
                 print(f"DEBUG: Filtering sheet loaded successfully (found as '{rgb_sheet_name}').")
             else:
                 print(f"DEBUG: Filtering sheet not found. Available sheets: {sheet_names}")
                 rgb_filtering_df = pd.DataFrame()
        except Exception as e:
             print(f"DEBUG: Error loading Filtering sheet: {e}") 
             rgb_filtering_df = pd.DataFrame()

        # Load Output sheet
        try:
             output_sheet_name = next((s for s in sheet_names if _norm(s).replace('_', '') == 'output'), None)
             if output_sheet_name:
                 output_sheet_df = pd.read_excel(config_path, sheet_name=output_sheet_name)
                 print(f"DEBUG: Output sheet loaded successfully as '{output_sheet_name}'. Rows: {len(output_sheet_df)}")
             else:
                 output_sheet_df = pd.DataFrame()
                 print(f"DEBUG: Output sheet not found. Available sheets: {sheet_names}")
        except Exception as e:
            print(f"Warning: Error loading Output sheet: {e}")
            output_sheet_df = pd.DataFrame()

    except Exception as e:
        print(f"Error reading configuration: {e}")
        return

    # Clean Failure Modes
    # Robust column lookup for Failure Mode
    fm_cols_norm = {c: _norm(c) for c in fm_df.columns}
    fm_col_name = next((c for c in fm_df.columns if fm_cols_norm[c] in ('failuremode', 'failure_mode')), None)
    
    if not fm_col_name:
        print("Error: 'Failure Mode' column not found in Failure Mode sheet.")
        return

    fms = fm_df[fm_col_name].dropna().astype(str).str.strip()
    fms = fms[fms != ""]
    unique_fms = fms.unique()
    print(f"DEBUG: Unique Failure Modes found: {list(unique_fms)}")

    # Helper functions
    # Moved helper functions to the top of run() to avoid NameError

    for fm in unique_fms:
        # Filter by command line arguments if provided
        if filter_failure_mode and _norm(fm) != _norm(filter_failure_mode):
            continue
            
        print(f"\n{'='*30}\nProcessing Failure Mode: {fm}\n{'='*30}")
        
        # Reset Flooding Defaults
        utils_general.Flooding_By_TH = "No"
        utils_general.Flooding_RGB = None
        utils_general.Reference_Params = {} # Clear Reference Params for new Failure Mode
        utils_general.Output_Config = [] # Clear Output Config
        utils_general.Gray_Scale_Params = {} # Clear Gray Scale Params for new Failure Mode
        utils_general.Scoring_Config = []
        
        # Get Config Rows
        fm_rows = fm_df[fm_df[fm_col_name] == fm]
        if fm_rows.empty:
            continue
        
        # Filter fm_rows by Product/Generation if provided
        if filter_product:
            # Robustly get Product column
            fm_cols_norm = {c: _norm(c) for c in fm_df.columns}
            prod_col = next((c for c in fm_df.columns if fm_cols_norm[c] == 'product'), None)
            if prod_col:
                fm_rows = fm_rows[fm_rows[prod_col].astype(str).apply(_norm) == _norm(filter_product)]
        
        if filter_generation:
            fm_cols_norm = {c: _norm(c) for c in fm_df.columns}
            gen_col = next((c for c in fm_df.columns if fm_cols_norm[c] == 'generation'), None)
            if gen_col:
                def norm_gen_filter(x):
                    s = str(x).strip()
                    if s.endswith('.0'): s = s[:-2]
                    return _norm(s)
                fm_rows = fm_rows[fm_rows[gen_col].apply(norm_gen_filter) == _norm(filter_generation)]
        
        if fm_rows.empty:
            print(f"DEBUG: No matching rows in Failure Mode sheet for FM='{fm}' after filtering.")
            continue

        fm_row = fm_rows.iloc[0]
        
        # Robust lookup in Flow sheet
        flow_cols_norm_lookup = {c: _norm(c) for c in flow_df.columns}
        flow_fm_col = next((c for c in flow_df.columns if flow_cols_norm_lookup[c] in ('failuremode', 'failure_mode')), None)
        
        if flow_fm_col:
            flow_rows = flow_df[flow_df[flow_fm_col] == fm]
            print(f"DEBUG: Matching Flow rows for FM='{fm}': {len(flow_rows)} rows found")
            
            # Extract method from Defect Identification column for better routing
            method_name = None
            if not flow_rows.empty:
                defect_id_col = next((c for c in flow_df.columns if flow_cols_norm_lookup[c] in ('defect_identification', 'defectidentification')), None)
                if defect_id_col:
                    method_name = str(flow_rows.iloc[0].get(defect_id_col, ''))

            # ---> STRANGLER FIG PATTERN ROUTER <---
            from utils.core.strategies.strategy_factory import StrategyFactory
            
            # Fetch Failure Mode row to extract product and generation
            fm_row_for_context = fm_df[fm_df[fm_col_name] == fm].iloc[0] if not fm_df[fm_df[fm_col_name] == fm].empty else None
            strat_product = str(fm_row_for_context.get('Product', '')) if fm_row_for_context is not None else ''
            strat_generation = str(fm_row_for_context.get('Generation', '')) if fm_row_for_context is not None else ''
            strat_pic_path = str(fm_row_for_context.get('Pic_Path', '')) if fm_row_for_context is not None else ''
            
            # Override from command line if provided
            if override_pic_path:
                strat_pic_path = override_pic_path
            
            strat_context = {
                'fm': fm,
                'product': strat_product,
                'generation': strat_generation,
                'download_path': strat_pic_path,
                'ConfigManager': None
            }
            strategy = StrategyFactory.get_strategy(method_name, fm, strat_context)
            if strategy:
                print(f"🚀 [ROUTER] Routing '{fm}' to Modern Strategy implementation! 🚀")
                strategy.execute()
                continue
            else:
                print(f"❌ [ROUTER] Failure Mode '{fm}' with method '{method_name}' is not supported by any Modern Strategy. Legacy loop has been removed. Please add a Strategy.")
                continue

if __name__ == "__main__":
    CONFIG_PATH = "RELP_Configuration.xlsx"

    if load_and_check_failure_mode_config(CONFIG_PATH, sheet_name="Failure Mode"):
        run_defect_filtering_pipeline()
    else:
        run()
