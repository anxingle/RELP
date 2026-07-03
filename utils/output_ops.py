


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

def _norm(s):
    """Normalize string for case-insensitive comparison."""
    if s is None:
        return ""
    return str(s).strip().lower().replace('_', '').replace(' ', '')


def clustering_Kmeans(bbox, SN_DF, sorting_strategy):
    print('clustering_Kmeans bbox', bbox)
    if bbox.ndim == 1:
        bbox = bbox.reshape((-1, 1))
    # if SN_DF.ndim == 1:
    #     SN_DF = SN_DF.reshape((-1,1))
    sub_bbox = bbox[:, sorting_strategy]
    print('sub_bbox', sub_bbox)
    n_bins = int(np.log2(len(sub_bbox)) + 1)
    counts, bin_edges = np.histogram(sub_bbox, bins=n_bins)
    print('histogram count', counts)
    # counts, bin_edges = np.histogram(sub_bbox, bins= 2)
    # peaks, _ = find_peaks(counts)
    peaks = np.where((counts > np.mean(counts)) | (counts >= 1))[0]
    print('peaks', peaks)
    cluster_qty = max(2, SN_DF.shape[1])
    # if more than 1 peak found, might be binomial distribution
    if len(peaks) > 1:
        print("bbox have a bimodal distribution")
        # K-Means clustering
        kmeans = KMeans(n_clusters=cluster_qty, random_state=42).fit(sub_bbox.reshape(-1, 1))
        labels = kmeans.labels_

        clusters = {i: [] for i in range(kmeans.n_clusters)}
        # 根据簇标签分组
        for point, label in zip(bbox, labels):
            clusters[label].append(point)
        # 将列表转换为NumPy数组
        clusters_arrays = {i: np.array(cluster) for i, cluster in clusters.items()}

        cluster1_data = bbox[labels == 0]
        cluster1_data = cluster1_data[:, sorting_strategy]
        cluster2_data = bbox[labels == 1]
        cluster2_data = cluster2_data[:, sorting_strategy]
        cluster1_mean = np.mean(cluster1_data)
        cluster2_mean = np.mean(cluster2_data)
    else:
        print("Data does not have a bimodal distribution")
        cluster1_data = sub_bbox
        cluster2_data = sub_bbox
        cluster1_mean = np.mean(cluster1_data)
        cluster2_mean = np.mean(cluster2_data)
        # clusters_arrays = bbox
        clusters_arrays = {0: bbox}
    return cluster1_data, cluster2_data, cluster1_mean, cluster2_mean, clusters_arrays


def SN_list_df(df):
    columns = df.columns.tolist()
    column_2drop = [f for f in columns if 'SN' not in f]
    df_dropped = df.drop(columns=column_2drop)
    print('df_dropped', df_dropped)
    df_dropDuplicate = df_dropped.T.drop_duplicates().T
    print('df_dropDuplicate', df_dropDuplicate)
    SN_df_width = df_dropDuplicate.shape[1]
    if SN_df_width > 1:
        # 导入utils_general中的sort_columns_with_numbers功能
        from utils.utils_general import sort_columns_with_numbers
        df_dropDuplicate = sort_columns_with_numbers(df_dropDuplicate)
    return df_dropDuplicate


def format_parametric_output_df(df, failure_mode, defect_output_format=None):

    try:
        df = df.copy()
        if 'Pic_Name' in df.columns and 'Filename' not in df.columns:
            df['Filename'] = df['Pic_Name']
        df['Defect_Type'] = failure_mode
        if 'Defect_Area' in df.columns and 'DUT_Area' in df.columns and 'Defect Pct' not in df.columns:
            df['Defect Pct'] = (df['Defect_Area'] / df['DUT_Area']) * 100.0
        elif 'Defect_Area' in df.columns and 'DUT_Area' not in df.columns:
            # Physical area available (mm²), rename to Defect Area for display
            df = df.rename(columns={'Defect_Area': 'Defect Area'})
            
        import utils.utils_general as utils_general
        # 如果 df 中已经有这些列，说明是从字典列表构建并且已经附带了各自正确的 Reference，
        # 则不再用全局变量一刀切地覆盖。
        if 'Reference' not in df.columns:
            ref_str, ref_status = build_reference_info(utils_general.Reference_Params)
            df['Reference'] = ref_str
            df['Reference_Status'] = ref_status
        elif 'Reference_Status' not in df.columns:
            _, ref_status = build_reference_info(utils_general.Reference_Params)
            df['Reference_Status'] = ref_status
        # Rename ratios
        if 'Ratio_Length' in df.columns and 'Length_Ratio' not in df.columns:
            df = df.rename(columns={'Ratio_Length': 'Length_Ratio'})
        if 'Ratio_Width' in df.columns and 'Width_Ratio' not in df.columns:
            df = df.rename(columns={'Ratio_Width': 'Width_Ratio'})
        dof = str(defect_output_format).strip().lower()
        if dof == 'combined':
            drop_cols = [
                'Defect_ID',
                # 'Contour',  # 必须保留！因为在 Combined 模式下，这里面存的是 .npy 的路径
                'Length_Ratio',
                'Width_Ratio',
                'Ratio_Length',
                'Ratio_Width',
                'Curved Line Length'
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        # Defect_ID from 'Defect' column if available
        if 'Defect_ID' not in df.columns:
            if 'Defect' in df.columns:
                import re
                ids = df['Defect'].astype(str).str.extract(r'(\d+)')
                df['Defect_ID'] = ids[0].astype('Int64')
                # Fill missing ids with group index per Filename
                if df['Defect_ID'].isna().any():
                    df['Defect_ID'] = df.groupby(df.get('Filename', pd.Series(range(len(df))))).cumcount()
            else:
                df['Defect_ID'] = range(len(df))
        return df
    except Exception as e:
        print(f"Warning: format_parametric_output_df failed: {e}")
        return df


def resolve_sn_from_excel(source_path):
    try:
        if not source_path:
            return None
        if os.path.isdir(source_path):
            return None
        base = os.path.splitext(os.path.basename(source_path))[0]
        sn_file = os.path.join(os.path.dirname(source_path), f"{base}.xlsx")
        if not os.path.exists(sn_file):
            return None
        sn_list = pd.read_excel(sn_file)
        sn_df = SN_list_df(sn_list)
        values = [v for v in sn_df.values.flatten().tolist() if pd.notna(v)]
        if not values:
            return None
        if len(values) == 1:
            return str(values[0])
        return "_".join(str(v) for v in values)
    except Exception as e:
        print(f"Warning: resolve_sn_from_excel failed: {e}")
        return None


def build_parametric_output_df(data, failure_mode, defect_output_format,
                               gray_scale_params=None, source_path=None, general_defect_key=None):
    """
    Build the final Parametric Output DataFrame with unified column selection.

    Args:
        data: List of dicts or DataFrame containing parametric results
        failure_mode: Failure mode name
        defect_output_format: 'Combined' or 'Individual'
        gray_scale_params: Gray Scale parameters dict (optional)
        source_path: Source file path (optional)
        general_defect_key: General Defect Detection key (optional), used to determine if 'Detected Defect' column should be added

    Returns:
        DataFrame with selected columns in proper order
    """
    try:
        # 1. Build DataFrame from data
        if isinstance(data, list):
            import utils.utils_general as utils_general
            formatted = []
            for item in data:
                if isinstance(item, dict):
                    formatted_item = format_parametric_dict(item.copy(), source_path, utils_general.Reference_Params, defect_output_format)
                    formatted.append(formatted_item)
            # DEBUG: Check all keys in formatted items
            all_keys = set()
            curved_line_count = 0
            for item in formatted:
                all_keys.update(item.keys())
                if 'Curved Line Length' in item:
                    curved_line_count += 1
            print(f"DEBUG: build_parametric_output_df - Total unique keys in formatted data: {len(all_keys)}")
            print(f"DEBUG: build_parametric_output_df - Items with 'Curved Line Length': {curved_line_count}/{len(formatted)}")
            print(f"DEBUG: build_parametric_output_df - All keys: {sorted(all_keys)}")
            print(f"DEBUG: build_parametric_output_df - GrayScale keys: {[k for k in all_keys if 'GrayScale' in k]}")
            print(f"DEBUG: build_parametric_output_df - Area_Ref keys: {[k for k in all_keys if 'Ref' in k or 'Area' in k]}")
            df = pd.DataFrame(formatted)
            print(f"DEBUG: build_parametric_output_df - DataFrame columns after creation: {df.columns.tolist()}")
            print(f"DEBUG: build_parametric_output_df - GrayScale columns in DataFrame: {[c for c in df.columns if 'GrayScale' in c]}")
            print(f"DEBUG: build_parametric_output_df - Curved Line Length in DataFrame: {'Curved Line Length' in df.columns}")
            print(f"DEBUG: build_parametric_output_df - Defect/Ref% in DataFrame: {'Defect/Ref%' in df.columns}")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            return pd.DataFrame()

        # 2. Apply basic formatting
        print(f"DEBUG: Before format_parametric_output_df - Curved Line Length in df: {'Curved Line Length' in df.columns}")
        df = format_parametric_output_df(df, failure_mode, defect_output_format)
        print(f"DEBUG: After format_parametric_output_df - Curved Line Length in df: {'Curved Line Length' in df.columns}")

        # 3. Add SN if needed
        sn_read_mode = globals().get('SN_Read_Mode')
        if 'SN' not in df.columns and str(sn_read_mode).strip().lower() == 'from excel':
            sn_val = resolve_sn_from_excel(source_path)
            if sn_val:
                df['SN'] = sn_val

        # 4. Ensure base columns exist
        base_cols = ['Filename', 'Dir', 'Defect_Type', 'Defect Pct', 'Reference', 'Reference_Status']
        for col in base_cols:
            if col not in df.columns:
                df[col] = ''

        # 4.5 Ensure Detected Defect column exists if Defect Detection Setting is configured
        # Detected Defect is now directly created in RELP3_main.py
        if general_defect_key and 'Detected Defect' not in df.columns:
            # If Detected Defect column doesn't exist, create empty column
            df['Detected Defect'] = ''
            print(f"DEBUG: Created empty 'Detected Defect' column. Defect Detection Setting: {general_defect_key}")

        # 5. Apply Decay Weighting if configured
        import utils.utils_general as utils_general
        scoring_setting = getattr(utils_general, 'Scoring_Setting', {})
        scoring_config = getattr(utils_general, 'Scoring_Config', [])
        advanced_scoring = getattr(utils_general, 'Advanced_Scoring', '')
        
        print(f"DEBUG: build_parametric_output_df - Scoring_Setting: {scoring_setting}")
        print(f"DEBUG: build_parametric_output_df - Scoring_Config: {scoring_config}")
        
        # Get scoring parameters (moved outside 'decay' check for Weighted Score processing)
        multiplier = scoring_setting.get('multiplier', 1.0)
        multiplier_column = scoring_setting.get('multiplier_column', None)
        roi_coefficient = scoring_setting.get('roi_coefficient', 1.0)
        use_new_score = scoring_setting.get('use_new_score', False)
        
        print(f"DEBUG: build_parametric_output_df - multiplier: {multiplier}, multiplier_column: {multiplier_column}, roi_coefficient: {roi_coefficient}, use_new_score: {use_new_score}")
        
        if 'decay' in [str(s).lower() for s in scoring_config] and scoring_setting:
            try:
                import utils.scoring_utils as scoring_utils
                
                # Find bin columns
                bin_cols = [c for c in df.columns if '[' in str(c) and not '_C[' in str(c) 
                           and not str(c).startswith('Weight[')]
                
                # Get scoring parameters for decay weighting
                max_weight = scoring_setting.get('max_weight', 1.0)
                decay_rate = scoring_setting.get('decay_rate', 0.5)
                reverse = scoring_setting.get('reverse', True)
                min_weight = scoring_setting.get('min_weight', None)
                use_defect_pct = scoring_setting.get('defect_pct', True)
                
                if bin_cols:
                    n_bins = len(bin_cols)

                    # Generate weights and calculate scores
                    weights = scoring_utils.exponential_decay_weights(n_bins, max_weight, decay_rate, reverse, min_weight=min_weight)
                    weighted_scores = []
                    
                    for idx, row in df.iterrows():
                        # Check if this is a no_detection record (Defect_ID = -1)
                        defect_id = row.get('Defect_ID', None)
                        if defect_id == -1:
                            # For no_detection records, set Defect Pct to 0% and Weighted Score to 0
                            df.at[idx, 'Defect Pct'] = '0%'
                            weighted_scores.append(0)
                            continue
                        
                        weighted_sum = sum(float(row.get(col, 0) or 0) * weights[i] for i, col in enumerate(bin_cols))
                        
                        if use_defect_pct:
                            defect_pct = row.get('Defect Pct', 0)
                            if pd.notna(defect_pct):
                                try:
                                    pct_val = float(str(defect_pct).replace('%', ''))
                                    # Always divide by 100 to convert percentage to decimal
                                    # Defect Pct is always stored as percentage (e.g., "3.97%" means 3.97% = 0.0397)
                                    pct_val = pct_val / 100.0
                                    weighted_sum *= pct_val
                                except:
                                    pass
                        
                        # Handle multiplier_column (e.g., multiplier=[Area_Ref, 100])
                        # When multiplier_column is set, multiply by the column value (converted to decimal)
                        if multiplier_column:
                            # Map parameter name to actual column name
                            column_mapping = {
                                'Area_Ref': 'Defect/Ref%',
                                'Area_Ref%': 'Defect/Ref%',
                                'defect_ref': 'Defect/Ref%',
                                'defect_over_ref': 'Defect/Ref%'
                            }
                            actual_column = column_mapping.get(multiplier_column, multiplier_column)
                            
                            col_value = row.get(actual_column, row.get(multiplier_column, 0))
                            if pd.notna(col_value) and col_value != 0:
                                try:
                                    # Convert percentage to decimal (e.g., "27.91%" -> 0.2791)
                                    col_val_str = str(col_value).replace('%', '')
                                    col_val_decimal = float(col_val_str) / 100.0
                                    weighted_sum *= multiplier * col_val_decimal
                                    print(f"DEBUG: Applied multiplier_column={multiplier_column} (col={actual_column}), value={col_value}, decimal={col_val_decimal}, multiplier={multiplier}")
                                except Exception as e:
                                    print(f"DEBUG: Failed to apply multiplier_column={multiplier_column}, value={col_value}, error: {e}")
                            else:
                                print(f"DEBUG: multiplier_column={multiplier_column} (col={actual_column}) value is {col_value}, skipping")
                        else:
                            weighted_sum *= multiplier * roi_coefficient
                        
                        weighted_scores.append(weighted_sum)
                    
                    df['Weighted Score'] = weighted_scores
                    
                    # Add weight columns with correct format Weight[low-high]
                    import re
                    print(f"DEBUG: Adding weight columns for {len(bin_cols)} bins")
                    print(f"DEBUG: bin_cols = {bin_cols}")
                    for i, col in enumerate(bin_cols):
                        # Match pattern like [0.0-30.0] or [0-30] - support float values
                        range_match = re.search(r'\[(\d+\.?\d*-\d+\.?\d*)\]', str(col))
                        if range_match:
                            weight_name = f"Weight[{range_match.group(1)}]"
                            df[weight_name] = round(weights[i], 2)
                            print(f"DEBUG: Added {weight_name} = {round(weights[i], 2)}")
                        else:
                            print(f"DEBUG: No range match for column: {col}")
                    print(f"DEBUG: DataFrame columns after adding weights: {[c for c in df.columns if c.startswith('Weight[')]}")
                    
                    # Comparison mode
                    if advanced_scoring and advanced_scoring.lower() == 'comparison':
                        complement_scores = scoring_utils.calculate_complement_score(bin_cols, weights, df, use_defect_pct=use_defect_pct)
                        df['Complement Weighted Score'] = complement_scores
                        df['Weighted Score Diff'] = df['Complement Weighted Score'] - df['Weighted Score']
                        
            except Exception as e:
                print(f"Warning: Failed to apply Decay Weighting: {e}")
        
        # Process Weighted Score if multiplier or roi_coefficient is set (outside 'decay' check)
        # Apply if: use_new_score is True, OR multiplier != 1.0, OR roi_coefficient != 1.0
        # Note: If 'decay' mode was used, multiplier and roi_coefficient are already applied above
        is_decay_mode = 'decay' in [str(s).lower() for s in scoring_config] and scoring_setting
        if 'Weighted Score' in df.columns:
            print(f"DEBUG: Weighted Score column found. Sample value: {df['Weighted Score'].iloc[0] if len(df) > 0 else 'N/A'}, type: {df['Weighted Score'].dtype}")
            # Only apply multiplier/roi_coefficient here if NOT in decay mode (already applied in decay block)
            if not is_decay_mode and (use_new_score or multiplier != 1.0 or roi_coefficient != 1.0):
                original_values = df['Weighted Score'].copy()
                df['Weighted Score'] = df['Weighted Score'].apply(
                    lambda x: float(x) * multiplier * roi_coefficient if pd.notna(x) else 0
                )
                print(f"DEBUG: Weighted Score updated with multiplier={multiplier}, roi_coefficient={roi_coefficient}")
                print(f"DEBUG: Weighted Score before: {original_values.iloc[0] if len(original_values) > 0 else 'N/A'}, after: {df['Weighted Score'].iloc[0] if len(df) > 0 else 'N/A'}")
            elif is_decay_mode:
                print(f"DEBUG: Skipping additional multiplier application - already applied in decay mode")
        else:
            print(f"DEBUG: Weighted Score column NOT found in DataFrame. Available columns: {df.columns.tolist()}")

        # 6. Use unified column selection
        columns_to_keep = get_output_columns(df.columns.tolist(), defect_output_format, gray_scale_params)
        
        # 7. Ensure all selected columns exist
        for col in columns_to_keep:
            if col not in df.columns:
                df[col] = ''
        
        # 8. Return DataFrame with selected columns
        return df[columns_to_keep]
        
    except Exception as e:
        print(f"Warning: build_parametric_output_df failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()



# --- 修复依赖补充的函数 ---

def build_reference_info(ref_params):
    ref_items = []
    status_items = []
    present_keys = []
    for key in ['Length', 'Width', 'Area', 'Ratio_Pixel']:
        if key in ref_params:
            r = ref_params[key].get('R', None)
            if r is not None:
                try:
                    rf = float(r)
                    if rf.is_integer():
                        ref_items.append(f"{key}-{int(rf)}")
                    else:
                        ref_items.append(f"{key}-{rf}")
                except Exception:
                    ref_items.append(f"{key}-{r}")
                present_keys.append(key)

    if len(present_keys) == 2:
        if 'Length' in present_keys:
            main_key = 'Length'
        elif 'Width' in present_keys:
            main_key = 'Width'
        else:
            main_key = 'Area'
    else:
        main_key = None

    for key in present_keys:
        if main_key and key == main_key:
            status_items.append(f"{key}(Main)")
        else:
            status_items.append(key)

    ref_str = '_'.join(ref_items) if ref_items else ''
    status_str = '_'.join(status_items) if status_items else 'Not Found'
    return ref_str, status_str


def get_output_columns(all_columns, defect_output_format, gray_scale_params=None):
    """
    统一决定 Parametric_Output.xlsx 输出哪些列
    
    Args:
        all_columns: DataFrame 中所有可用的列名列表
        defect_output_format: 'Combined' 或 'Individual'
        gray_scale_params: Gray Scale 参数字典，包含 'Mode' 等信息
    
    Returns:
        应该保留的列名列表（按顺序）
    """
    # 1. 基础列（始终保留，按顺序）
    base_columns = [
        'Filename', 'Dir', 'Defect_Type', 'Defect Pct',
        'Reference', 'Reference_Status', 'SN', 'Defect',
        'Factual_Area', 'Factual_Length', 'Factual_Width'
    ]
    
    # 1.5 Individual 模式添加 Defect ID 列
    if str(defect_output_format).lower() == 'individual':
        base_columns.insert(3, 'Defect_ID')  # 在 Defect_Type 后插入 Defect_ID
    
    # 1.5.5 如果存在 Contour 列且不需要被 drop（在 Individual 或强制保留时），将其放在 Defect Pct 之后
    if 'Contour' in all_columns:
        try:
            defect_pct_idx = base_columns.index('Defect Pct')
            base_columns.insert(defect_pct_idx + 1, 'Contour')
        except ValueError:
            base_columns.append('Contour')
            
    # 1.6 如果存在 Detected Defect 列（Defect Detection Setting 配置时），添加到 Defect Pct 之后
    if 'Detected Defect' in all_columns:
        # 找到 Defect Pct 的索引，在其后插入 Detected Defect
        try:
            defect_pct_idx = base_columns.index('Defect Pct')
            base_columns.insert(defect_pct_idx + 1, 'Detected Defect')
        except ValueError:
            # Defect Pct 不在列表中，添加到末尾
            base_columns.append('Detected Defect')
    
    # 1.7 如果存在 Shape 列（Shape Filtering 配置时），添加到 Defect Pct 之后
    if 'Shape' in all_columns:
        # 找到 Defect Pct 的索引，在其后插入 Shape
        try:
            defect_pct_idx = base_columns.index('Defect Pct')
            base_columns.insert(defect_pct_idx + 1, 'Shape')
        except ValueError:
            # Defect Pct 不在列表中，添加到末尾
            base_columns.append('Shape')
    
    # 1.8 如果存在 Area_Ref 相关列（Defect/Ref% 等），添加到 Defect Pct 之后
    area_ref_fields = ['Defect/Ref%', 'Defect_Area_Pixels', 'Reference_BBox_Area']
    for field in area_ref_fields:
        if field in all_columns:
            try:
                defect_pct_idx = base_columns.index('Defect Pct')
                base_columns.insert(defect_pct_idx + 1, field)
            except ValueError:
                base_columns.append(field)
    
    # 2. 根据 Gray Scale 模式定义列组
    # Note: Weighted Score 和 Complement Weighted Score 会在最后统一添加
    mode_column_groups = {
        'Neighboring Gray_Scale': [
            # 基础统计
            'Neighboring_Inner_Mean', 'Neighboring_Inner_Median', 'Neighboring_Inner_Std',
            'Neighboring_Outer_Mean', 'Neighboring_Outer_Median', 'Neighboring_Outer_Std',
            'Neighboring_Diff_Mean', 'Neighboring_Diff_Median', 'Neighboring_Diff_Std',
            # 组件统计
            'Neighboring_CompDiff_Mean', 'Neighboring_CompDiff_Median',
            'Neighboring_CompDiff_Min', 'Neighboring_CompDiff_Max',
            'Neighboring_CompDiff_Std', 'Neighboring_CompDiff_Count',
            # 派生指标（当设置了 min_size 时）
            'Defect_above_TH'
        ],
        'HSV': [
            'HSV_Hue_Mean', 'HSV_Hue_Median', 'HSV_Hue_Std',
            'HSV_Saturation_Mean', 'HSV_Saturation_Median', 'HSV_Saturation_Std',
            'HSV_Value_Mean', 'HSV_Value_Median', 'HSV_Value_Std'
        ],
        'Gray Scale': [
            'GrayScale_Mean', 'GrayScale_Median', 'GrayScale_Std',
            'GrayScale_Min', 'GrayScale_Max'
        ],
        'Ring Comparison': [
            'Ring_Inner_Mean', 'Ring_Inner_Median', 'Ring_Inner_Std',
            'Ring_Outer_Mean', 'Ring_Outer_Median', 'Ring_Outer_Std',
            'Ring_Diff_Mean', 'Ring_Diff_Median', 'Ring_Diff_Std',
            'Ring_Pct_Mean', 'Ring_Pct_Median', 'Ring_Pct_Std'
        ]
    }
    
    # 3. Combined Mode 要排除的列（仅在 Combined 模式下排除）
    combined_exclude = [
        'Defect_ID', 'Contour',
        'Ratio_Length', 'Ratio_Width',
        'Length_Ratio', 'Width_Ratio'
        # Note: 'Curved Line Length' 不在此列表中，因为它在 Individual 模式下应该保留
    ]
    
    # 4. 构建结果列列表
    result = []
    
    # 添加基础列（如果存在）
    for col in base_columns:
        if col in all_columns:
            result.append(col)
    
    # 添加模式相关的列
    mode = None
    if gray_scale_params and isinstance(gray_scale_params, dict):
        mode = gray_scale_params.get('Mode')
    
    if mode and mode in mode_column_groups:
        # 添加模式特定的列（如 Gray Scale 统计列）
        for col in mode_column_groups[mode]:
            if col in all_columns and col not in result:
                result.append(col)
        # 同时保留其他非排除列（如 Length, Width, Shape 等）
        for col in all_columns:
            if col not in result and col not in combined_exclude:
                result.append(col)
    else:
        # 如果没有匹配的模式，保留所有非基础列（向后兼容）
        for col in all_columns:
            if col not in result and col not in combined_exclude:
                result.append(col)
    
    # 4.5 调整 Curved Line Length 的位置，放在 Length 后面
    if 'Curved Line Length' in result:
        # 先移除 Curved Line Length
        result.remove('Curved Line Length')
        # 找到 Length 的索引
        if 'Length' in result:
            length_idx = result.index('Length')
            # 在 Length 后面插入 Curved Line Length
            result.insert(length_idx + 1, 'Curved Line Length')
        else:
            # 如果没有 Length，添加到末尾
            result.append('Curved Line Length')
    
    # 5. Combined Mode 排除特定列
    if str(defect_output_format).lower() == 'combined':
        # Combined 模式下排除特定列，包括 Curved Line Length
        combined_only_exclude = combined_exclude + ['Curved Line Length']
        result = [c for c in result if c not in combined_only_exclude]

    # 6. 添加通道分箱列（如 HSV_H[50-80], HSV_S[50-80], HSV_V[50-80] 等）
    bin_columns = [c for c in all_columns if '[' in c and ']' in c and not c.startswith('Weight[')]
    for col in bin_columns:
        if col not in result:
            result.append(col)

    # DEBUG: Check if Curved Line Length is in result
    if 'Curved Line Length' in result:
        print(f"DEBUG: get_output_columns - 'Curved Line Length' is in result")
    elif 'Curved Line Length' in all_columns:
        print(f"DEBUG: get_output_columns - 'Curved Line Length' is in all_columns but NOT in result")
    else:
        print(f"DEBUG: get_output_columns - 'Curved Line Length' is NOT in all_columns")

    # 7. 添加 Decay Weighting 相关列（如果存在）
    weight_columns = [c for c in all_columns if c.startswith('Weight[')]
    for col in weight_columns:
        if col not in result:
            result.append(col)

    # 8. 添加 Weighted Score（最后）
    # 首先确保 Weighted Score 和 Complement Weighted Score 不在 result 中（如果之前被添加了，先移除）
    score_columns = ['Weighted Score', 'Complement Weighted Score', 'Weighted Score Diff']
    for col in score_columns:
        if col in result:
            result.remove(col)
    
    # 然后按顺序添加到最后
    for col in score_columns:
        if col in all_columns:
            result.append(col)

    return result


import os

from utils.io_ops import create_folder

from utils.base_utils import _norm

import pandas as pd
import numpy as np
from utils.io_ops import extract_product_side_from_path
from utils.crop_ops import check_orientation_and_flip, apply_rotations, calculate_min_area_rect_correction

import math
from skimage.morphology import skeletonize
Reference_Params = {}
from utils.config.config_manager import ConfigManager

import cv2

def parametric_output(main_folder_path, sub_folder_path, file_name, sheet_name, df):
    if main_folder_path is None:
        print(f"Error: parametric_output received None for main_folder_path (file_name={file_name}). Skipping output.")
        return
    res_folder = create_folder(main_folder_path, sub_folder_path)
    res_full_path = os.path.join(res_folder, file_name)
    
    final_df = df
    
    if os.path.exists(res_full_path):
        try:
            with pd.ExcelFile(res_full_path) as xls:
                if sheet_name in xls.sheet_names:
                    existing_df = ConfigManager().get_sheet(sheet_name)
                    print(f"DEBUG: Found existing sheet '{sheet_name}' with columns: {existing_df.columns.tolist()}")
                    
                    # Align columns: Concatenate allows aligning by column name automatically
                    final_df = pd.concat([existing_df, df], ignore_index=True)
                else:
                    print(f"DEBUG: Sheet '{sheet_name}' not found in existing file.")
        except Exception as e:
            print(f"Warning: Could not read existing Excel file: {e}. Overwriting.")

    # Clean up: Remove 'Area' if 'Reference_Area' exists in the final combined dataframe
    if 'Reference_Area' in final_df.columns and 'Area' in final_df.columns:
        print("DEBUG: Removing duplicate 'Area' column from final output.")
        final_df = final_df.drop(columns=['Area'])

    print(f"DEBUG: Writing to {res_full_path}, Columns: {final_df.columns.tolist()}")
    
    # Write the full dataframe (overwriting the sheet)
    # Note: openpyxl 'overlay' mode with header=False is tricky for column alignment.
    # It's safer to rewrite the sheet if we want to guarantee column consistency.
    
    # If the file exists, we need to preserve other sheets
    if os.path.exists(res_full_path):
        with pd.ExcelWriter(res_full_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            final_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(res_full_path, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name=sheet_name, index=False)


def detect_reference_bbox_area(image, detector, score_thresh=0.5):
    """
    Detect Reference class (e.g., 'Ref', 'Reference') and calculate its bbox area.
    
    Args:
        image: BGR image (numpy array)
        detector: Detectron2 predictor or similar with inference capability
        score_thresh: Minimum confidence score to include
        
    Returns:
        ref_bbox_area: Area of Reference bbox in pixels, or None if not found
        ref_det: Detection dict for Reference, or None if not found
    """
    from detectron2.data import MetadataCatalog
    
    # Run inference
    outputs = detector(image)
    instances = outputs["instances"]
    
    if len(instances) == 0:
        print("DEBUG: detect_reference_bbox_area - No detections found.")
        return None, None
    
    # Get predictions
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    
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
    
    # Find Reference class
    ref_bbox_area = None
    ref_det = None
    
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
        
        # Check if this is Reference class
        if cls_name.lower() in ['ref', 'reference']:
            ref_bbox_area = (x2 - x1) * (y2 - y1)
            ref_det = {
                'class_id': int(cls_id),
                'class_name': cls_name,
                'score': float(score),
                'bbox': [x1, y1, x2, y2]
            }
            print(f"DEBUG: Found Reference class - {cls_name}, bbox area: {ref_bbox_area}px")
            break
    
    if ref_bbox_area is None:
        print("DEBUG: No Reference class found in detections.")
    
    return ref_bbox_area, ref_det


def filter_reference_by_detected_classes(reference_df, detected_classes, product=None, generation=None, product_side=None):
    """
    Filter Reference DataFrame by detected class names.
    Only returns rows where the 'Reference' column value matches one of the detected class names.

    Args:
        reference_df: Reference sheet DataFrame
        detected_classes: Set of detected class names (lowercase)
        product: Optional product name to also filter by
        generation: Optional generation to also filter by
        product_side: Optional product side to filter by (e.g., 'R360_Side1', 'Side1')

    Returns:
        filtered_df: Filtered DataFrame or empty DataFrame if no match
    """
    if reference_df.empty or not detected_classes:
        return pd.DataFrame()

    ref_cols_norm = {c: _norm(c) for c in reference_df.columns}

    # Find Reference column
    ref_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'reference'), None)
    if not ref_col:
        print("DEBUG: No 'Reference' column found in Reference sheet.")
        return pd.DataFrame()

    # Filter by detected classes (case-insensitive)
    mask = reference_df[ref_col].astype(str).str.lower().isin(detected_classes)
    filtered_df = reference_df[mask]

    print(f"DEBUG: Filtered Reference by detected classes {detected_classes}: {len(filtered_df)} rows found.")

    # Further filter by Product if provided
    if product:
        prod_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'product'), None)
        if prod_col:
            prod_mask = filtered_df[prod_col].astype(str).apply(_norm) == _norm(product)
            filtered_df = filtered_df[prod_mask]
            print(f"DEBUG: Further filtered by Product '{product}': {len(filtered_df)} rows found.")

    # Further filter by Generation if provided
    if generation:
        gen_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'generation'), None)
        if gen_col:
            gen_mask = filtered_df[gen_col].astype(str).apply(_norm) == _norm(generation)
            filtered_df = filtered_df[gen_mask]
            print(f"DEBUG: Further filtered by Generation '{generation}': {len(filtered_df)} rows found.")

    # Further filter by Product_Side if provided
    if product_side:
        product_side_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'productside'), None)
        if product_side_col:
            # Try exact match first
            side_mask = filtered_df[product_side_col].astype(str).apply(_norm) == _norm(product_side)
            filtered_by_side = filtered_df[side_mask]

            # If no exact match, try partial match (e.g., 'R360_Side1' matches 'Side1')
            if filtered_by_side.empty and '_' in product_side:
                # Extract just the Side part (e.g., 'Side1' from 'R360_Side1')
                side_only = product_side.split('_')[-1]
                side_mask = filtered_df[product_side_col].astype(str).apply(_norm) == _norm(side_only)
                filtered_by_side = filtered_df[side_mask]

            if not filtered_by_side.empty:
                filtered_df = filtered_by_side
                print(f"DEBUG: Further filtered by Product_Side '{product_side}': {len(filtered_df)} rows found.")
            else:
                print(f"DEBUG: Product_Side '{product_side}' not found in Reference sheet, keeping previous filter results.")

    return filtered_df


def populate_reference_params_from_row(reference_df, ref_row, output_config):
    """
    Populate Reference_Params dictionary from a reference row.
    
    Args:
        reference_df: Reference sheet DataFrame
        ref_row: Reference row (Series)
        output_config: List of output parameters to populate
        
    Returns:
        reference_params: Dictionary of reference parameters
    """
    reference_params = {}
    
    ref_cols_norm = {c: _norm(c) for c in reference_df.columns}
    
    # Get Reference Type
    ref_type_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'reference'), None)
    if ref_type_col:
        ref_type_val = ref_row.get(ref_type_col)
        if pd.notna(ref_type_val):
            reference_params['Ref_Type'] = str(ref_type_val).strip()
            print(f"DEBUG: Reference Type found: {reference_params['Ref_Type']}")
    
    # Populate parameters
    for param in output_config:
        if param == 'Contour':
            continue
        
        # Match columns like "Area" and "Area_Adjustment"
        p_norm = _norm(param)
        r_col = next((c for c in reference_df.columns if _norm(c) == p_norm), None)
        a_col = next((c for c in reference_df.columns if _norm(c) == _norm(f"{param}_Adjustment") or _norm(c) == _norm(f"{param}_Adjustement") or _norm(c) == _norm(f"{param}_Adjuste")), None)
        
        print(f"DEBUG: Looking for Param '{param}' (norm='{p_norm}'). Found Cols: R='{r_col}', A='{a_col}'")
        
        param_dict = {}
        if r_col:
            val = ref_row.get(r_col)
            if pd.notna(val):
                try:
                    param_dict['R'] = float(val)
                except:
                    pass
        if a_col:
            val = ref_row.get(a_col)
            if pd.notna(val):
                try:
                    param_dict['A'] = float(val)
                except:
                    pass
        
        if param_dict:
            reference_params[param] = param_dict
            print(f"DEBUG: Reference Params for {param}: {param_dict}")
    
    return reference_params


def match_reference_by_detected_classes(reference_df, detected_classes, product=None, generation=None, output_config=None, product_side=None):
    """
    Universal function to match Reference sheet rows by detected class names.
    This function can be used by all flows (OD_Ref, Dino, KNN, Filtering, etc.)

    Matching Logic:
    1. Filter Reference rows where 'Reference' column matches one of detected_classes
    2. Further filter by Product (if provided)
    3. Further filter by Generation (if provided)
    4. Further filter by Product_Side (if provided)
    5. Return the first matched row's Reference_Params

    Args:
        reference_df: Reference sheet DataFrame
        detected_classes: Set of detected class names (lowercase) or single class name string
        product: Optional product name to filter
        generation: Optional generation to filter
        output_config: List of output parameters to populate (default: Reference_Params from utils_general)
        product_side: Optional product side to filter by (e.g., 'R360_Side1', 'Side1')

    Returns:
        tuple: (matched_reference_params, matched_class_name)
            - matched_reference_params: Dictionary of reference parameters or empty dict if no match
            - matched_class_name: The matched class name or None
    """
    if reference_df.empty:
        print("DEBUG: Reference DataFrame is empty.")
        return {}, None

    # Convert single class name to set
    if isinstance(detected_classes, str):
        detected_classes = {detected_classes.lower()}
    elif not isinstance(detected_classes, set):
        detected_classes = set(detected_classes)

    if not detected_classes:
        print("DEBUG: No detected classes provided.")
        return {}, None

    print(f"DEBUG: Matching Reference for detected classes: {detected_classes}")

    # Use global Output_Config if not provided
    if output_config is None:
        output_config = Reference_Params if 'Reference_Params' in globals() else []

    # Filter by detected classes
    filtered_df = filter_reference_by_detected_classes(
        reference_df=reference_df,
        detected_classes=detected_classes,
        product=product,
        generation=generation,
        product_side=product_side
    )

    if filtered_df.empty:
        print(f"DEBUG: No Reference row matched detected classes {detected_classes}")
        return {}, None

    # Use the first matched row
    ref_row = filtered_df.iloc[0]

    # Get the matched class name
    ref_cols_norm = {c: _norm(c) for c in reference_df.columns}
    ref_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'reference'), None)
    matched_class_name = None
    if ref_col:
        matched_class_name = str(ref_row.get(ref_col, '')).strip()
        print(f"DEBUG: Matched Reference row by class: {matched_class_name}")

    # Populate Reference_Params
    reference_params = populate_reference_params_from_row(
        reference_df=reference_df,
        ref_row=ref_row,
        output_config=output_config if output_config else []
    )

    print(f"DEBUG: Populated Reference_Params: {reference_params}")

    return reference_params, matched_class_name


def match_reference_for_image(reference_df, image_path, product, generation, output_config, ref_type='DUT'):
    """
    根据图片路径匹配 Reference 参数。
    支持从图片路径中提取 Product_Side 信息，并根据 Product_Side 筛选 Reference 行。

    匹配逻辑：
    1. 从图片路径中提取 Product_Side（如 'R360_Side1'）
    2. 先尝试用 Product_Side 过滤 Reference 行
    3. 如果 Product_Side 匹配不到，则回退到只用 Product + Generation 匹配
    4. 如果 Reference sheet 中没有 Product_Side 列，则使用现有的逻辑

    Args:
        reference_df: Reference sheet DataFrame
        image_path: 图片的完整路径
        product: Product 名称
        generation: Generation 名称
        output_config: 输出参数列表
        ref_type: Reference 类型（如 'DUT', 'Pixel_Ratio' 等）

    Returns:
        dict: Reference 参数字典，如果没有匹配到则返回空字典
    """
    if reference_df.empty:
        print("DEBUG: Reference DataFrame is empty.")
        return {}

    # 从图片路径中提取 Product_Side
    product_side = extract_product_side_from_path(image_path)
    if product_side:
        print(f"DEBUG: Extracted Product_Side='{product_side}' from image path: {image_path}")

    ref_cols_norm = {c: _norm(c) for c in reference_df.columns}

    # 检查 Reference sheet 是否有 Product_Side 列
    has_product_side_col = any(ref_cols_norm[c] == 'productside' for c in reference_df.columns)

    # 第一步：尝试用 Product_Side 过滤（如果提供了 Product_Side 且 Reference sheet 有该列）
    if product_side and has_product_side_col:
        print(f"DEBUG: Trying to match Reference with Product_Side='{product_side}'")

        # 先按 Product 过滤
        ref_prod_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'product'), None)
        if ref_prod_col:
            ref_row_df = reference_df[reference_df[ref_prod_col].astype(str).apply(_norm) == _norm(product)]

            # 再按 Generation 过滤
            ref_gen_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'generation'), None)
            if ref_gen_col and not ref_row_df.empty and generation:
                ref_row_df = ref_row_df[ref_row_df[ref_gen_col].astype(str).apply(_norm) == _norm(generation)]

            # 最后按 Product_Side 过滤
            if not ref_row_df.empty:
                product_side_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'productside'), None)
                if product_side_col:
                    # 尝试精确匹配
                    side_mask = ref_row_df[product_side_col].astype(str).apply(_norm) == _norm(product_side)
                    filtered_by_side = ref_row_df[side_mask]

                    # 如果精确匹配不到，尝试部分匹配（如 'R360_Side1' 匹配 'Side1'）
                    if filtered_by_side.empty and '_' in product_side:
                        side_only = product_side.split('_')[-1]
                        side_mask = ref_row_df[product_side_col].astype(str).apply(_norm) == _norm(side_only)
                        filtered_by_side = ref_row_df[side_mask]

                    if not filtered_by_side.empty:
                        ref_row = filtered_by_side.iloc[0]
                        print(f"DEBUG: Found Reference row with Product_Side='{product_side}'")

                        # 填充 Reference 参数
                        reference_params = populate_reference_params_from_row(
                            reference_df=reference_df,
                            ref_row=ref_row,
                            output_config=output_config if output_config else []
                        )
                        print(f"DEBUG: Reference Params with Product_Side: {reference_params}")
                        return reference_params
                    else:
                        print(f"DEBUG: No Reference row found with Product_Side='{product_side}', falling back to Product+Generation match")

    # 第二步：回退到只用 Product + Generation 匹配（现有逻辑）
    print(f"DEBUG: Matching Reference with Product='{product}', Generation='{generation}'")

    ref_prod_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'product'), None)
    if ref_prod_col:
        ref_row_df = reference_df[reference_df[ref_prod_col].astype(str).apply(_norm) == _norm(product)]

        ref_gen_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'generation'), None)
        if ref_gen_col and not ref_row_df.empty and generation:
            gen_match = ref_row_df[ref_row_df[ref_gen_col].astype(str).apply(_norm) == _norm(generation)]
            if not gen_match.empty:
                ref_row_df = gen_match

        if not ref_row_df.empty:
            # 在通用回退时，优先选择没有配置 Product_Side 的那一行（即真正的通用行）
            product_side_col = next((c for c in reference_df.columns if ref_cols_norm[c] == 'productside'), None)
            if product_side_col:
                # 过滤出 Product_Side 为 NaN 或为空的行
                generic_mask = ref_row_df[product_side_col].isna() | (ref_row_df[product_side_col].astype(str).str.strip() == '') | (ref_row_df[product_side_col].astype(str).str.lower() == 'nan')
                if generic_mask.any():
                    ref_row_df = ref_row_df[generic_mask]
                    
            ref_row = ref_row_df.iloc[0]
            print(f"DEBUG: Found Reference row for Product='{product}', Generation='{generation}' (Fallback Generic)")

            reference_params = populate_reference_params_from_row(
                reference_df=reference_df,
                ref_row=ref_row,
                output_config=output_config if output_config else []
            )
            print(f"DEBUG: Reference Params (fallback): {reference_params}")
            return reference_params

    print(f"DEBUG: No Reference row found for Product='{product}', Generation='{generation}'")
    return {}


def calculate_parametric_dimensions(defect_cnt, dut_dims, ref_params, output_config, contour_area_val, ref_area, ref_found, defect_mask=None, is_line_shape=False, image=None, gray_scale_params=None, filtering_df=None, mask_default=None, output_dir=None, image_filename=None, detector=None):
    """
    Calculates parametric ratios and absolute dimensions for a defect.
    
    Args:
        defect_cnt: Defect contour (numpy array) or defect dimensions tuple (len, wid).
        dut_dims: Tuple (dut_len, dut_wid) in pixels.
        ref_params: Dictionary of reference parameters (utils_general.Reference_Params).
        output_config: List of output config keys (utils_general.Output_Config).
        contour_area_val: Area of the DUT contour (pixels) for fallback scaling.
        ref_area: Reference area value for fallback scaling.
        ref_found: Boolean indicating if reference data was found.
        defect_mask: Optional binary mask of the defect (used for Curved Line and Gray Scale).
        is_line_shape: Boolean indicating if the defect shape is classified as 'Line'.
        image: Original image (numpy array, BGR) for Gray Scale analysis.
        gray_scale_params: Dict containing 'Conversion', 'Invert', 'Bining'.
        output_dir: Optional output directory for saving reference images.
        image_filename: Optional original image filename for naming reference images.
        filtering_df: Optional DataFrame containing Filtering rules to apply secondary filtering.
        mask_default: Optional DUT mask for complement calculation in Comparison mode.
        detector: Optional detector (e.g., Detectron2 predictor) to detect Reference class for Area_Ref calculation.
        
    Returns:
        dict: Dictionary containing 'Ratio_Length', 'Ratio_Width', 'Length', 'Width', 'Reference_Length', 'Reference_Width', etc.
    """
    if gray_scale_params:
         print(f"DEBUG: calculate_parametric_dimensions called with gray_scale_params: {gray_scale_params}")
    else:
         print(f"DEBUG: calculate_parametric_dimensions called with gray_scale_params: None or Empty")
    results = {}
    output_config = output_config or []
    
    # 1. Determine Defect Dimensions
    defect_len = 0
    defect_wid = 0
    
    if isinstance(defect_cnt, np.ndarray):
        # It's a contour
        if len(defect_cnt) > 0:
            try:
                _, (w_def, h_def), _ = cv2.minAreaRect(defect_cnt)
                defect_len = max(w_def, h_def)
                defect_wid = min(w_def, h_def)
            except: pass
    elif isinstance(defect_cnt, (tuple, list)) and len(defect_cnt) == 2:
        # It's already dimensions (len, wid)
        # We assume input is (len, wid) as max/min
        v1, v2 = defect_cnt
        defect_len = max(v1, v2)
        defect_wid = min(v1, v2)

    dut_len, dut_wid = dut_dims
    dut_area = contour_area_val if contour_area_val > 0 else (dut_len * dut_wid)

    ref_len_real = 0.0
    ref_wid_real = 0.0
    ref_area_real = 0.0

    has_len = 'Length' in ref_params
    has_wid = 'Width' in ref_params
    has_area = 'Area' in ref_params
    has_ratio_pixel = 'Ratio_Pixel' in ref_params

    if has_len:
        ref_entry = ref_params['Length']
        r_val = float(ref_entry.get('R', 0))
        adj = float(ref_entry.get('A', 1.0))
        ref_len_real = r_val * adj
    if has_wid:
        ref_entry = ref_params['Width']
        r_val = float(ref_entry.get('R', 0))
        adj = float(ref_entry.get('A', 1.0))
        ref_wid_real = r_val * adj
    if has_area:
        ref_entry = ref_params['Area']
        r_val = float(ref_entry.get('R', 0))
        adj = float(ref_entry.get('A', 1.0))
        ref_area_real = r_val * adj
        
    ratio_pixel_val = 0.0
    if has_ratio_pixel:
        ref_entry = ref_params['Ratio_Pixel']
        r_val = float(ref_entry.get('R', 0))
        adj = float(ref_entry.get('A', 1.0))
        ratio_pixel_val = r_val * adj

    if has_len:
        main_dim = 'Length'
    elif has_wid:
        main_dim = 'Width'
    elif has_area:
        main_dim = 'Area'
    else:
        main_dim = None

    main_scale = 0.0
    if has_ratio_pixel and ratio_pixel_val > 0:
        main_scale = ratio_pixel_val
    elif main_dim == 'Length' and ref_len_real > 0 and dut_len > 0:
        main_scale = ref_len_real / dut_len
    elif main_dim == 'Width' and ref_wid_real > 0 and dut_wid > 0:
        main_scale = ref_wid_real / dut_wid
    elif main_dim == 'Area' and ref_area_real > 0 and dut_area > 0:
        main_scale = np.sqrt(ref_area_real / dut_area)

    if main_scale > 0:
        if not has_len and dut_len > 0:
            ref_len_real = dut_len * main_scale
            has_len = ref_len_real > 0
        if not has_wid and dut_wid > 0:
            ref_wid_real = dut_wid * main_scale
            has_wid = ref_wid_real > 0
        if not has_area and dut_area > 0:
            ref_area_real = dut_area * (main_scale ** 2)
            has_area = ref_area_real > 0

    if not has_area and has_len and has_wid and ref_len_real > 0 and ref_wid_real > 0:
        ref_area_real = ref_len_real * ref_wid_real
        has_area = True
    if not has_wid and has_len and has_area and ref_len_real > 0:
        ref_wid_real = ref_area_real / ref_len_real
        has_wid = ref_wid_real > 0
    if not has_len and has_wid and has_area and ref_wid_real > 0:
        ref_len_real = ref_area_real / ref_wid_real
        has_len = ref_len_real > 0

    # ===============================
    # 新增: 实际物理面积 (Factual Area) 的换算
    # ===============================
    # 精准获取缺陷的像素面积，而不是简单地用外接矩形的长乘宽
    true_pixel_area = 0
    if defect_mask is not None:
        true_pixel_area = cv2.countNonZero(defect_mask)
    elif isinstance(defect_cnt, np.ndarray) and len(defect_cnt) > 0:
        true_pixel_area = cv2.contourArea(defect_cnt)
    else:
        true_pixel_area = defect_len * defect_wid

    if main_scale > 0:
        results['Factual_Area'] = float(true_pixel_area * (main_scale ** 2))
    elif has_area and ref_area_real > 0 and dut_area > 0:
        # 如果只有面积参考值，根据总面积比换算
        results['Factual_Area'] = float(true_pixel_area * (ref_area_real / dut_area))
        
    # 为长度和宽度也添加明确的 Factual 别名
    if 'Length' in results:
        results['Factual_Length'] = results['Length']
    elif main_scale > 0:
        results['Factual_Length'] = defect_len * main_scale
        
    if 'Width' in results:
        results['Factual_Width'] = results['Width']
    elif main_scale > 0:
        results['Factual_Width'] = defect_wid * main_scale

    # 2. Calculate Ratios (Always)
    if dut_len > 0:
        results['Ratio_Length'] = defect_len / dut_len
    else:
        results['Ratio_Length'] = 0

    if dut_wid > 0:
        results['Ratio_Width'] = defect_wid / dut_wid
    else:
        results['Ratio_Width'] = 0

    # 3. Calculate Absolute Dimensions (if Reference exists)
    
    # --- Length ---
    if 'Length' in ref_params or 'Length' in output_config:
        if ref_len_real > 0 and dut_len > 0:
            l_ratio = defect_len / dut_len
            results['Length'] = l_ratio * ref_len_real
            results['Reference_Length'] = ref_len_real
        elif dut_area > 0 and ref_area_real > 0:
            scale = np.sqrt(ref_area_real / dut_area)
            results['Length'] = defect_len * scale
            results['Reference_Length'] = dut_len * scale

    # --- Width ---
    if 'Width' in ref_params or 'Width' in output_config:
        if ref_wid_real > 0 and dut_wid > 0:
            w_ratio = defect_wid / dut_wid
            results['Width'] = w_ratio * ref_wid_real
            results['Reference_Width'] = ref_wid_real
        elif dut_area > 0 and ref_area_real > 0:
            scale = np.sqrt(ref_area_real / dut_area)
            results['Width'] = defect_wid * scale
            results['Reference_Width'] = dut_wid * scale

    # --- Curved Line Measurement ---
    # Check if any key resembling 'Curved Line Measurement' is in output_config
    curved_key = next((k for k in output_config if 'curved' in k.lower() and 'line' in k.lower()), None)
    
    # Only proceed if requested AND shape is Line
    if curved_key and is_line_shape:
        curved_len_px = 0
        
        # Prepare Mask for Skeletonization
        mask_for_skel = None
        if defect_mask is not None:
             mask_for_skel = defect_mask
        elif isinstance(defect_cnt, np.ndarray) and len(defect_cnt) > 0:
             # Draw contour to create mask
             # We need canvas size. Use bounding rect as approximation + padding
             x, y, w, h = cv2.boundingRect(defect_cnt)
             mask_for_skel = np.zeros((h, w), dtype=np.uint8)
             # Shift contour
             cnt_shifted = defect_cnt - [x, y]
             cv2.drawContours(mask_for_skel, [cnt_shifted], -1, 255, -1)
             
        if mask_for_skel is not None:
            try:
                # Skeletonize expects boolean or 0/1
                skel = skeletonize(mask_for_skel > 0)
                curved_len_px = np.sum(skel)
            except Exception as e:
                print(f"Error in Curved Line Skeletonization: {e}")
        
        if curved_len_px > 0:
            # Determine Scale Factor
            scale_factor = 0
            
            # Priority 1: Use Length Reference Scale
            if ref_len_real > 0 and dut_len > 0:
                scale_factor = ref_len_real / dut_len
                
            # Priority 2: Use Width Reference Scale
            elif ref_wid_real > 0 and dut_wid > 0:
                scale_factor = ref_wid_real / dut_wid
            
            # Priority 3: Use Area Fallback Scale
            elif dut_area > 0 and ref_area_real > 0:
                scale_factor = np.sqrt(ref_area_real / dut_area)
            
            if scale_factor > 0:
                results[curved_key] = curved_len_px * scale_factor
            else:
                # If no reference, maybe output pixel value or ratio?
                # For now, if no reference is found, we can't give physical units.
                # But user asked for Ratio in general. 
                # Let's add Ratio_Curved_Line if requested? 
                # The prompt implies adding a column "Curved Line Measurement" based on parametric_out.
                # We will output absolute if ref exists, else maybe pixels?
                # Let's just output 0 or pixels if no ref.
                results[curved_key] = curved_len_px # Pixels if no scale
                results[curved_key + '_Pixels'] = curved_len_px

    # --- Color Space Measurement (formerly Gray Scale) ---
    print(f"DEBUG: Processing Gray Scale Params: {gray_scale_params}")
    print(f"DEBUG: Dims Gray Scale Params: {gray_scale_params}")
    if gray_scale_params and image is not None and defect_mask is not None:
        try:
            mode = gray_scale_params.get('Mode', 'Gray Scale')
            advanced_scoring = globals().get('Advanced_Scoring', '')
            
            # 1. Prepare ROI
            x, y, w, h = cv2.boundingRect(defect_mask)
            defect_mask_total_px = cv2.countNonZero(defect_mask)
            print(f"DEBUG: ROI extraction - defect_mask total pixels={defect_mask_total_px}, bounding box=({x},{y},{w},{h})")
            if w > 0 and h > 0:
                roi = image[y:y+h, x:x+w]
                mask_roi = defect_mask[y:y+h, x:x+w]
                mask_roi_px = cv2.countNonZero(mask_roi)
                print(f"DEBUG: ROI extraction - mask_roi pixels={mask_roi_px} (should equal defect_mask total)")
                
                # Create complement mask for Advanced Scoring = Comparison
                complement_mask_roi = None
                if advanced_scoring and advanced_scoring.lower() == 'comparison':
                    # Create complement mask: DUT area MINUS defect area
                    # Use mask_default if available to properly identify DUT area in ROI
                    if mask_default is not None:
                        # Extract mask_default ROI
                        mask_default_roi = mask_default[y:y+h, x:x+w]
                        # Complement = (DUT area in ROI) - (defect area)
                        # mask_default_roi contains DUT area (255), defect area is also 255 in mask_roi
                        # So: complement = mask_default_roi AND (NOT mask_roi)
                        complement_mask_roi = cv2.subtract(mask_default_roi, mask_roi)
                    else:
                        # Fallback: Assume entire ROI is DUT area (may include background)
                        complement_mask_roi = np.ones_like(mask_roi, dtype=np.uint8) * 255
                        complement_mask_roi = cv2.subtract(complement_mask_roi, mask_roi)
                        print(f"DEBUG: Created complement mask without mask_default (fallback). ROI shape: {mask_roi.shape}, complement pixels: {cv2.countNonZero(complement_mask_roi)}")
                
                # Parsing Binning Params common to both
                bining_str = gray_scale_params.get('Bining', '')
                bins = []
                if bining_str:
                    try:
                        clean_bins = str(bining_str).replace('[', '').replace(']', '').strip()
                        if clean_bins:
                            # Check for new format: "start-end, step=N" or "start-end, step=N"
                            # Support both integer and float values
                            import re
                            range_step_match = re.match(r'([\d.]+)\s*-\s*([\d.]+)\s*,?\s*step\s*=\s*([\d.]+)', clean_bins, re.IGNORECASE)
                            if range_step_match:
                                start_val = float(range_step_match.group(1))
                                end_val = float(range_step_match.group(2))
                                step_val = float(range_step_match.group(3))
                                # Generate bins: [start, start+step, start+2*step, ..., end]
                                bins = []
                                current = start_val
                                while current < end_val:
                                    bins.append(current)
                                    current += step_val
                                # Ensure end_val is included
                                if not bins or bins[-1] != end_val:
                                    bins.append(end_val)
                                print(f"DEBUG: Generated bins from range-step format: {bins}")
                            else:
                                # Legacy format: comma-separated values
                                bins = [float(x.strip()) for x in clean_bins.split(',') if x.strip()]
                    except Exception as e:
                        print(f"Error parsing Bining string '{bining_str}': {e}")
                        bins = []

                # DEBUG: Print Bining info
                print(f"DEBUG: Color Space Measurement. Bining Str: '{bining_str}', Parsed Bins: {bins}")

                # Determine Scale Factor for defect region
                # UNIFIED: Always use defect_mask pixel count as denominator
                # This ensures binning results represent the distribution within the defect
                # and sum to 100% across all bins
                defect_area_scale = 0
                scale_mode = "Percent"
                
                if len(bins) >= 2:
                    # Use defect_mask_total_px as denominator (defect pixel count)
                    if defect_mask_total_px > 0:
                        defect_area_scale = 100.0 / defect_mask_total_px
                        scale_mode = "Percentage of Defect Pixels (%)"
                
                # Calculate complement area scale if Advanced Scoring is Comparison
                complement_area_scale = defect_area_scale
                
                print(f"DEBUG: Color Space Calculation. Mode: {scale_mode}, Defect Scale: {defect_area_scale:.6f}, Ref Found: {ref_found}, Defect Pixels: {defect_mask_total_px}")
                
                if defect_area_scale == 0:
                     print("Warning: Defect Area Scale is 0. No binning results will be generated.")
                
                # --- Gray Scale Mode ---
                if mode == 'Gray Scale':
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    target_img = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_roi)
                    
                    invert_param = gray_scale_params.get('Invert', '')
                    if str(invert_param).lower() in ('yes', 'true', '1'):
                        gray_roi_inv = cv2.bitwise_not(gray_roi)
                        target_img = cv2.bitwise_and(gray_roi_inv, gray_roi_inv, mask=mask_roi)
                    
                    if defect_area_scale > 0:
                        for i in range(len(bins) - 1):
                            low = bins[i]
                            high = bins[i+1]
                            # Use left-closed, right-open interval [low, high)
                            # For integer pixel values, this means [low, high-1]
                            # Use high - 0.1 to exclude the exact high value
                            high_exclusive = high - 0.1 if high > low else high
                            range_mask = cv2.inRange(target_img, int(low), int(high_exclusive))
                            final_bin_mask = cv2.bitwise_and(range_mask, mask_roi)
                            px_count = cv2.countNonZero(final_bin_mask)
                            val = px_count * defect_area_scale
                            results[f"GrayScale[{low}-{high}]"] = val
                            
                            # Calculate complement mask values if Advanced Scoring is Comparison
                            if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_mask_roi is not None:
                                complement_target_img = cv2.bitwise_and(gray_roi, gray_roi, mask=complement_mask_roi)
                                if str(invert_param).lower() in ('yes', 'true', '1'):
                                    complement_target_img = cv2.bitwise_and(gray_roi_inv, gray_roi_inv, mask=complement_mask_roi)
                                complement_range_mask = cv2.inRange(complement_target_img, int(low), int(high_exclusive))
                                complement_final_bin_mask = cv2.bitwise_and(complement_range_mask, complement_mask_roi)
                                complement_px_count = cv2.countNonZero(complement_final_bin_mask)
                                complement_val = complement_px_count * complement_area_scale  # Use complement-specific scale
                                results[f"GrayScale_C[{low}-{high}]"] = complement_val

                # --- HSV Mode ---
                elif str(mode).upper().startswith('HSV'):
                    # Force black background for HSV calculation to match Save(Overlay) visual
                    # This eliminates edge artifacts from original background
                    masked_roi_for_hsv = cv2.bitwise_and(roi, roi, mask=mask_roi)
                    hsv_roi = cv2.cvtColor(masked_roi_for_hsv, cv2.COLOR_BGR2HSV)
                    
                    # Create complement HSV ROI if Advanced Scoring is Comparison
                    complement_hsv_roi = None
                    if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_mask_roi is not None:
                        complement_masked_roi = cv2.bitwise_and(roi, roi, mask=complement_mask_roi)
                        complement_hsv_roi = cv2.cvtColor(complement_masked_roi, cv2.COLOR_BGR2HSV)
                    
                    # --- Secondary Filtering (Strict Enforcement) ---
                    # DISABLED: Binning should be performed on the final mask without additional filtering
                    # to ensure accurate distribution across the full range of pixel values
                    # The filtering was already applied during the Defect Identification phase
                    print(f"DEBUG: Skipping secondary filtering for binning - using final mask as-is")

                    invert_param = gray_scale_params.get('Invert', '')
                    should_invert = str(invert_param).lower() in ('yes', 'true', '1')
                    
                    # Split channels
                    h, s, v = cv2.split(hsv_roi)
                    
                    # Split complement channels if needed
                    complement_h, complement_s, complement_v = None, None, None
                    if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_hsv_roi is not None:
                        complement_h, complement_s, complement_v = cv2.split(complement_hsv_roi)
                    
                    if should_invert:
                        # Invert H: (h + 90) % 180 (Complementary color in 0-179 space)
                        h = (h.astype(int) + 90) % 180
                        h = h.astype(np.uint8)
                        # Invert S, V: 255 - val
                        s = cv2.bitwise_not(s)
                        v = cv2.bitwise_not(v)
                        
                        # Invert complement channels if needed
                        if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_h is not None:
                            complement_h = (complement_h.astype(int) + 90) % 180
                            complement_h = complement_h.astype(np.uint8)
                            complement_s = cv2.bitwise_not(complement_s)
                            complement_v = cv2.bitwise_not(complement_v)
                    
                    # Parse brackets: HSV[HS] -> H and S
                    import re
                    match = re.search(r'HSV\[(.*?)\]', mode, re.IGNORECASE)
                    
                    # Apply mask to all channels (to ensure background is 0)
                    h = cv2.bitwise_and(h, h, mask=mask_roi)
                    s = cv2.bitwise_and(s, s, mask=mask_roi)
                    v = cv2.bitwise_and(v, v, mask=mask_roi)
                    
                    # Apply mask to complement channels if needed
                    if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_h is not None:
                        complement_h = cv2.bitwise_and(complement_h, complement_h, mask=complement_mask_roi)
                        complement_s = cv2.bitwise_and(complement_s, complement_s, mask=complement_mask_roi)
                        complement_v = cv2.bitwise_and(complement_v, complement_v, mask=complement_mask_roi)
                    
                    # Increase threshold to filter out dark/low-saturation noise (artifacts at edges)
                    # Default was > 0, now > 10 to be safer against compression noise
                    min_val = 30
                    min_sat = 20
                    non_black_mask = ((s > min_sat) & (v > min_val)).astype(np.uint8) * 255
                    
                    # Create non-black mask for complement if needed
                    complement_non_black_mask = None
                    if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_s is not None:
                        complement_non_black_mask = ((complement_s > min_sat) & (complement_v > min_val)).astype(np.uint8) * 255
                    
                    target_channels = {}
                    complement_target_channels = {}
                    is_combined = False
                    
                    use_h_inrange = False
                    if match:
                        comp_str = match.group(1).upper()
                        if comp_str == 'H':
                            use_h_inrange = True
                            target_channels = {'H': h}
                            if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_h is not None:
                                complement_target_channels = {'H': complement_h}
                        else:
                            comps = []
                            complement_comps = []
                            if 'H' in comp_str: 
                                comps.append(('H', h))
                                if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_h is not None:
                                    complement_comps.append(('H', complement_h))
                            if 'S' in comp_str: 
                                comps.append(('S', s))
                                if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_s is not None:
                                    complement_comps.append(('S', complement_s))
                            if 'V' in comp_str: 
                                comps.append(('V', v))
                                if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_v is not None:
                                    complement_comps.append(('V', complement_v))
                            
                            if comps:
                                is_combined = True
                                sum_sq = np.zeros_like(h, dtype=np.float32)
                                names = ""
                                for name, img in comps:
                                    names += name
                                    sum_sq += img.astype(np.float32) ** 2
                                
                                magnitude = np.sqrt(sum_sq / len(comps))
                                target_channels[names] = magnitude
                                
                                # Calculate complement magnitude if needed
                                if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_comps:
                                    complement_sum_sq = np.zeros_like(complement_h, dtype=np.float32)
                                    complement_names = ""
                                    for name, img in complement_comps:
                                        complement_names += name
                                        complement_sum_sq += img.astype(np.float32) ** 2
                                    
                                    complement_magnitude = np.sqrt(complement_sum_sq / len(complement_comps))
                                    complement_target_channels[complement_names] = complement_magnitude
                    else:
                        # Legacy/Default HSV (All separated)
                        # Use dict keys 'H', 'S', 'V' to match previous output format "HSV_H[...]"
                        target_channels = {'H': h, 'S': s, 'V': v}
                        if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_h is not None:
                            complement_target_channels = {'H': complement_h, 'S': complement_s, 'V': complement_v}
                    
                    if defect_area_scale > 0:
                        for ch_name, ch_data in target_channels.items():
                            total_bin_sum = 0.0
                            print(f"DEBUG: Color Space Processing Channel '{ch_name}' with {len(bins)-1} bins.")
                            
                            # Use full mask_roi for consistent calculation (without non_black_mask filtering)
                            # This ensures denominator and numerator are consistent
                            total_mask_px = cv2.countNonZero(mask_roi)
                            print(f"DEBUG: Total Defect Pixels in ROI: {total_mask_px}")
                            
                            for i in range(len(bins) - 1):
                                low = bins[i]
                                high = bins[i+1]
                                
                                # OpenCV inRange is inclusive [low, upper]
                                # So we set upper = high - 1 to make it [low, high)
                                # Exception: If it's the very last bin, we might want to include the max value (e.g. 255)
                                upper_val = high - 1 if i < len(bins) - 2 else high
                                
                                if ch_name == 'H' and use_h_inrange:
                                    lower = np.array([low, 0, 0], dtype=np.uint8)
                                    upper = np.array([upper_val, 255, 255], dtype=np.uint8)
                                    bin_mask = cv2.inRange(hsv_roi, lower, upper)
                                    final_bin_mask = cv2.bitwise_and(bin_mask, mask_roi)
                                elif is_combined:
                                    # ch_data is float magnitude
                                    # Binning: half-open intervals to avoid double counting
                                    # And must be inside mask
                                    if i < len(bins) - 2:
                                        bin_mask = (ch_data >= low) & (ch_data < high)
                                    else:
                                        bin_mask = (ch_data >= low) & (ch_data <= high)
                                    bin_mask = bin_mask & (mask_roi > 0)
                                    final_bin_mask = bin_mask.astype(np.uint8) * 255
                                else:
                                    # ch_data is uint8 image
                                    # Binning: half-open intervals to avoid double counting
                                    if i < len(bins) - 2:
                                        bin_mask = (ch_data >= low) & (ch_data < high)
                                    else:
                                        bin_mask = (ch_data >= low) & (ch_data <= high)
                                    final_bin_mask = (bin_mask.astype(np.uint8) * 255)
                                    final_bin_mask = cv2.bitwise_and(final_bin_mask, mask_roi)
                                
                                px_count = cv2.countNonZero(final_bin_mask)
                                # Use defect_area_scale for consistent binning (same as Gray Scale mode)
                                val = px_count * defect_area_scale
                                results[f"HSV_{ch_name}[{low}-{high}]"] = val
                                total_bin_sum += val
                                print(f"DEBUG: Bin HSV_{ch_name}[{low}-{high}] - px_count={px_count}, val={val:.6f}, scale={defect_area_scale:.6f}")
                                
                                # Calculate complement mask values if Advanced Scoring is Comparison
                                if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_target_channels:
                                    complement_ch_data = complement_target_channels.get(ch_name)
                                    if complement_ch_data is not None:
                                        if ch_name == 'H' and use_h_inrange and complement_hsv_roi is not None:
                                            complement_bin_mask = cv2.inRange(complement_hsv_roi, lower, upper)
                                            complement_final_bin_mask = cv2.bitwise_and(complement_bin_mask, complement_mask_roi)
                                            if complement_non_black_mask is not None:
                                                complement_final_bin_mask = cv2.bitwise_and(complement_final_bin_mask, complement_non_black_mask)
                                        elif is_combined:
                                            # ch_data is float magnitude
                                            if i < len(bins) - 2:
                                                complement_bin_mask = (complement_ch_data >= low) & (complement_ch_data < high)
                                            else:
                                                complement_bin_mask = (complement_ch_data >= low) & (complement_ch_data <= high)
                                            complement_bin_mask = complement_bin_mask & (complement_mask_roi > 0)
                                            complement_final_bin_mask = complement_bin_mask.astype(np.uint8) * 255
                                            if complement_non_black_mask is not None:
                                                complement_final_bin_mask = cv2.bitwise_and(complement_final_bin_mask, complement_non_black_mask)
                                        else:
                                            # ch_data is uint8 image
                                            if i < len(bins) - 2:
                                                complement_bin_mask = (complement_ch_data >= low) & (complement_ch_data < high)
                                            else:
                                                complement_bin_mask = (complement_ch_data >= low) & (complement_ch_data <= high)
                                            complement_final_bin_mask = (complement_bin_mask.astype(np.uint8) * 255)
                                            complement_final_bin_mask = cv2.bitwise_and(complement_final_bin_mask, complement_mask_roi)
                                            if complement_non_black_mask is not None:
                                                complement_final_bin_mask = cv2.bitwise_and(complement_final_bin_mask, complement_non_black_mask)
                                        
                                        complement_px_count = cv2.countNonZero(complement_final_bin_mask)
                                        complement_val = complement_px_count * complement_area_scale  # Use complement-specific scale
                                        results[f"HSV_{ch_name}_C[{low}-{high}]"] = complement_val
                            
                            print(f"DEBUG: Channel '{ch_name}' Total Bin Sum: {total_bin_sum:.4f} (Mode: {scale_mode})")

                # --- Dino Distance Mode ---
                elif str(mode).upper().startswith('DINO_DISTANCE') or str(mode).upper().startswith('DINO DISTANCE'):
                    print(f"DEBUG: Calculating Dino Distance Distribution ({mode})...")
                    
                    # Load Dino distance map from CSV file
                    # The CSV file should be in the same directory as the image, named {base_name}_anomaly_distance_raw.csv
                    dino_dist_map = None
                    if image_filename is not None:
                        import os
                        base_name = os.path.splitext(os.path.basename(image_filename))[0]
                        # Keep the original base_name including suffixes like _rotated90_cropped
                        # The CSV file naming convention matches the image file naming
                        # Dino saves CSV as {fname}_anomaly_distance_raw.csv (without _cropped prefix)
                        
                        # Try to find the CSV file in output_dir or reference_dir
                        csv_candidates = []
                        if output_dir is not None:
                            csv_candidates.append(os.path.join(output_dir, f"{base_name}_anomaly_distance_raw.csv"))
                            csv_candidates.append(os.path.join(output_dir, "reference", f"{base_name}_anomaly_distance_raw.csv"))
                        
                        # Also check in the image's directory
                        img_dir = os.path.dirname(image_filename)
                        if img_dir:
                            csv_candidates.append(os.path.join(img_dir, f"{base_name}_anomaly_distance_raw.csv"))
                            csv_candidates.append(os.path.join(img_dir, "reference", f"{base_name}_anomaly_distance_raw.csv"))
                        
                        for csv_path in csv_candidates:
                            if os.path.exists(csv_path):
                                try:
                                    dino_dist_map = np.loadtxt(csv_path, delimiter=",")
                                    print(f"DEBUG: Loaded Dino distance map from {csv_path}, shape={dino_dist_map.shape}")
                                    break
                                except Exception as e:
                                    print(f"DEBUG: Failed to load Dino distance map from {csv_path}: {e}")
                    
                    if dino_dist_map is None:
                        print(f"WARNING: Could not find Dino distance CSV file for {image_filename}")
                    else:
                        # Align distance map to ROI coordinates
                        # The distance map is typically for the full image or DUT_Corrected
                        # We need to crop/resize it to match the ROI (defect bounding box)
                        
                        # Get the bounding box of the defect within the full image
                        # x, y are the top-left coordinates of the ROI within the full image
                        x, y, w, h = cv2.boundingRect(defect_mask)
                        
                        # Resize distance map to match the original image size if needed
                        if dino_dist_map.shape[0] != roi.shape[0] or dino_dist_map.shape[1] != roi.shape[1]:
                            print(f"DEBUG: Resizing Dino distance map from {dino_dist_map.shape} to {roi.shape[:2]}")
                            dino_dist_map_resized = cv2.resize(dino_dist_map.astype(np.float32), (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            dino_dist_map_resized = dino_dist_map
                        
                        # Extract ROI from distance map
                        # Note: x, y are relative to the full image, but dino_dist_map might be for DUT_Corrected
                        # We need to handle the coordinate transformation
                        
                        # For simplicity, assume the distance map is aligned with the input image
                        # and extract the corresponding region
                        if y + h <= dino_dist_map_resized.shape[0] and x + w <= dino_dist_map_resized.shape[1]:
                            dino_roi = dino_dist_map_resized[y:y+h, x:x+w]
                        else:
                            # If coordinates are out of bounds, use the full distance map
                            print(f"DEBUG: ROI coordinates out of bounds, using full distance map")
                            dino_roi = dino_dist_map_resized
                        
                        # Apply mask to distance map
                        # Ensure mask_roi and dino_roi have the same size
                        if dino_roi.shape[:2] != mask_roi.shape[:2]:
                            dino_roi = cv2.resize(dino_roi, (mask_roi.shape[1], mask_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                        
                        # Convert to float for processing
                        dino_roi_float = dino_roi.astype(np.float32)
                        
                        # Apply mask: set non-mask pixels to 0
                        dino_masked = np.where(mask_roi > 0, dino_roi_float, 0)
                        
                        # For Dino_Distance mode, calculate denominator based on Dino distance map within the mask
                        # Denominator = pixels >= first bin threshold in the Dino distance map (after applying mask)
                        # This gives the distribution within the selected defect mask
                        first_bin_threshold = bins[0] if bins else 0.1
                        denominator_px = np.sum(dino_masked >= first_bin_threshold)
                        print(f"DEBUG: Dino_Distance denominator - first_bin_threshold={first_bin_threshold}, denominator_px={denominator_px} (within defect mask)")
                        
                        if denominator_px > 0:
                            defect_area_scale = 100.0 / denominator_px
                            scale_mode = "Percentage of Defect Pixels >= Threshold (%)"
                            print(f"DEBUG: Dino_Distance scale calculation - defect_area_scale={defect_area_scale:.6f} (100/{denominator_px})")
                            
                            total_bin_sum = 0.0
                            for i in range(len(bins) - 1):
                                low = bins[i]
                                high = bins[i+1]
                                
                                # Create mask for this bin range
                                # Use half-open intervals like HSV mode
                                if i < len(bins) - 2:
                                    bin_mask = (dino_masked >= low) & (dino_masked < high)
                                else:
                                    bin_mask = (dino_masked >= low) & (dino_masked <= high)
                                
                                # Convert to uint8 mask
                                bin_mask_uint8 = (bin_mask.astype(np.uint8)) * 255
                                
                                # Count pixels in this bin
                                px_count = cv2.countNonZero(bin_mask_uint8)
                                val = px_count * defect_area_scale
                                results[f"Dino_Dist[{low:.4f}-{high:.4f}]"] = val
                                total_bin_sum += val
                                
                                print(f"DEBUG: Dino_Dist[{low:.4f}-{high:.4f}] - px_count={px_count}, val={val:.6f}")
                            
                            print(f"DEBUG: Dino Distance Total Bin Sum: {total_bin_sum:.4f} (Mode: {scale_mode})")
                        else:
                            print(f"WARNING: No Dino pixels >= {first_bin_threshold} in defect mask, skipping binning")

                # --- Morphological Mode ---
                elif str(mode).upper().startswith('MORPH'):
                    print(f"DEBUG: Calculating Morphological Distribution ({mode})...")
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Initialize all morph params with defaults
                    morph_params = {
                        'kernel_size': 11,
                        'threshold': 0,  # No threshold for color space measurement (we want raw response)
                        'denoise': 0,
                        'open': 0,
                        'close': 0,
                        'min_area': 0,
                        'size': 1.0,  # No resize
                        'uniform_light': False,
                        'ul_kernel_size': 101,
                        'roi_shrink': 1.0
                    }
                    
                    # Parse inline params from gray_scale_params (user specified in Color Space Conversion)
                    for k, v in gray_scale_params.items():
                        k_norm = str(k).lower().replace(' ', '').replace('_', '')
                        try:
                            if k_norm in ('kernelsize', 'ksize', 'kernel'):
                                morph_params['kernel_size'] = int(float(v))
                            elif k_norm in ('threshold', 'thresh'):
                                morph_params['threshold'] = int(float(v))
                            elif k_norm in ('denoise',):
                                morph_params['denoise'] = int(float(v))
                            elif k_norm in ('open', 'morphopen'):
                                morph_params['open'] = int(float(v))
                            elif k_norm in ('close', 'morphclose'):
                                morph_params['close'] = int(float(v))
                            elif k_norm in ('minarea', 'area'):
                                morph_params['min_area'] = float(v)
                            elif k_norm in ('size', 'resize', 'scale'):
                                if '/' in str(v):
                                    num, den = str(v).split('/')
                                    morph_params['size'] = float(num) / float(den)
                                else:
                                    morph_params['size'] = float(v)
                            elif k_norm in ('uniformlight', 'uniform_light', 'ul'):
                                morph_params['uniform_light'] = str(v).lower() in ('on', 'true', '1', 'yes')
                            elif k_norm in ('ulkernelsize', 'ul_kernel', 'ulkernel'):
                                morph_params['ul_kernel_size'] = int(float(v))
                            elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                morph_params['roi_shrink'] = float(v)
                        except (ValueError, TypeError):
                            pass
                    
                    # Try to reuse from Filtering configuration (only for params not already set inline)
                    if filtering_df is not None:
                        try:
                            # Determine if we're looking for Top-Hat or Black-Hat
                            is_black_hat = 'black' in str(mode).lower()
                            target_method = 'Morph Black-Hat' if is_black_hat else 'Morph Top-Hat'
                            
                            print(f"DEBUG: Trying to reuse {target_method} config from filtering_df...")
                            
                            # Find matching row in filtering_df
                            method_col = None
                            setting_col = None
                            
                            # Find column names
                            for col in filtering_df.columns:
                                col_norm = str(col).lower().replace(' ', '').replace('_', '')
                                if col_norm in ('method', 'targetedrgbgroupmethod'):
                                    method_col = col
                                if col_norm in ('methodsetting', 'setting', 'targetmethodsetting'):
                                    setting_col = col
                            
                            if method_col is not None:
                                # Normalize method names in filtering_df for matching
                                def _norm_method(val):
                                    if pd.isna(val):
                                        return ''
                                    return str(val).lower().replace(' ', '').replace('_', '').replace('-', '')
                                
                                filtering_df_norm = filtering_df.copy()
                                filtering_df_norm['__method_norm__'] = filtering_df[method_col].apply(_norm_method)
                                
                                target_norm = target_method.lower().replace(' ', '').replace('_', '').replace('-', '')
                                matches = filtering_df_norm[filtering_df_norm['__method_norm__'].str.contains(target_norm, na=False)]
                                
                                if not matches.empty:
                                    matched_row = matches.iloc[0]
                                    config_str = None
                                    if setting_col is not None:
                                        config_str = matched_row.get(setting_col)
                                    
                                    if config_str and pd.notna(config_str):
                                        print(f"DEBUG: Found {target_method} config: {config_str}")
                                        # Parse config string like [Kernel Size = 17, Threshold = 1, ...]
                                        s = str(config_str).strip().replace('[', '').replace(']', '')
                                        parts = s.split(',')
                                        for p in parts:
                                            if '=' in p:
                                                k, v = p.split('=', 1)
                                                k_norm = k.strip().lower().replace(' ', '').replace('_', '')
                                                v_val = v.strip()
                                                try:
                                                    if k_norm in ('kernelsize', 'kernalsize', 'ksize', 'kernel', 'kernelsize'):
                                                        morph_params['kernel_size'] = int(float(v_val))
                                                        print(f"DEBUG: Reused Kernel Size from Filtering: {morph_params['kernel_size']}")
                                                    elif k_norm in ('threshold', 'thresh'):
                                                        # Note: For color space measurement, we typically don't threshold
                                                        # But we store it for reference
                                                        pass
                                                    elif k_norm in ('denoise',):
                                                        morph_params['denoise'] = int(float(v_val))
                                                    elif k_norm in ('open', 'morphopen'):
                                                        morph_params['open'] = int(float(v_val))
                                                    elif k_norm in ('close', 'morphclose'):
                                                        morph_params['close'] = int(float(v_val))
                                                    elif k_norm in ('minarea', 'area'):
                                                        morph_params['min_area'] = float(v_val)
                                                    elif k_norm in ('size', 'resize', 'scale'):
                                                        if '/' in v_val:
                                                            num, den = v_val.split('/')
                                                            morph_params['size'] = float(num) / float(den)
                                                        else:
                                                            morph_params['size'] = float(v_val)
                                                    elif k_norm in ('uniformlight', 'uniform_light', 'ul'):
                                                        morph_params['uniform_light'] = v_val.lower() in ('on', 'true', '1', 'yes')
                                                    elif k_norm in ('ulkernelsize', 'ul_kernel', 'ulkernel'):
                                                        morph_params['ul_kernel_size'] = int(float(v_val))
                                                    elif k_norm in ('roishrink', 'roi_shrink', 'shrink'):
                                                        morph_params['roi_shrink'] = float(v_val)
                                                except (ValueError, TypeError):
                                                    pass
                        except Exception as e:
                            print(f"DEBUG: Error trying to reuse Filtering config: {e}")
                    
                    # Ensure kernel sizes are odd
                    if morph_params['kernel_size'] < 3: 
                        morph_params['kernel_size'] = 3
                    if morph_params['kernel_size'] % 2 == 0: 
                        morph_params['kernel_size'] += 1
                    if morph_params['ul_kernel_size'] % 2 == 0:
                        morph_params['ul_kernel_size'] += 1
                    
                    k_size = morph_params['kernel_size']
                    
                    # Apply Size (resize) if specified
                    working_roi = gray_roi.copy()
                    if morph_params['size'] != 1.0 and morph_params['size'] > 0:
                        new_w = int(working_roi.shape[1] * morph_params['size'])
                        new_h = int(working_roi.shape[0] * morph_params['size'])
                        working_roi = cv2.resize(working_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        print(f"DEBUG: Resized ROI by {morph_params['size']} for Morph analysis. New size: {new_w}x{new_h}")
                    
                    # Apply Uniform Light if enabled
                    if morph_params['uniform_light']:
                        background = cv2.GaussianBlur(working_roi, (morph_params['ul_kernel_size'], morph_params['ul_kernel_size']), 0)
                        diff_illum = cv2.subtract(working_roi, background)
                        working_roi = cv2.add(diff_illum, 127)
                        print(f"DEBUG: Applied Uniform Light correction with kernel {morph_params['ul_kernel_size']}")
                    
                    # Apply Denoise if specified
                    if morph_params['denoise'] > 0:
                        k_denoise = morph_params['denoise'] * 2 + 1
                        working_roi = cv2.GaussianBlur(working_roi, (k_denoise, k_denoise), 0)
                        print(f"DEBUG: Applied Denoise with kernel {k_denoise}")
                    
                    # Apply Morphological operation
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                    
                    morph_type = cv2.MORPH_TOPHAT
                    morph_name = "TopHat"
                    if 'black' in str(mode).lower():
                        morph_type = cv2.MORPH_BLACKHAT
                        morph_name = "BlackHat"
                    
                    # NOTE: For Color Space Conversion, we do NOT apply Open/Close before Top-Hat
                    # because we want to measure the raw Top-Hat response.
                    # Open/Close are used in Filtering for mask post-processing, not for measurement.
                    if morph_params['open'] > 0 or morph_params['close'] > 0:
                        print(f"DEBUG: Note: Open/Close operations are skipped for Color Space Measurement (they are for Filtering mask post-processing only)")
                    
                    print(f"DEBUG: Applying Morph {morph_name} with Kernel Size: {k_size}")
                    
                    # Apply Morph to processed ROI
                    target_img = cv2.morphologyEx(working_roi, morph_type, kernel)
                    
                    # Resize back to original size if needed
                    if morph_params['size'] != 1.0 and morph_params['size'] > 0:
                        target_img = cv2.resize(target_img, (gray_roi.shape[1], gray_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    print(f"DEBUG: Morph {morph_name} applied. Output range: [{target_img.min()}, {target_img.max()}]")
                    
                    # Binning
                    total_bin_sum = 0
                    for i in range(len(bins) - 1):
                        low = bins[i]
                        high = bins[i+1]
                        
                        # --- Defect Region ---
                        # Apply Defect Mask to Morph Result
                        masked_target = cv2.bitwise_and(target_img, target_img, mask=mask_roi)
                        
                        # Count pixels in bin range
                        # Use strictly > low to avoid 0-background if low=0? 
                        # Usually bins start at 0. If low=0, we include 0.
                        # But masked areas are 0. So we must ensure we only count inside mask_roi.
                        # bitwise_and(..., mask_roi) ensures we only look at defect pixels.
                        # However, if defect pixel has morph value 0, and low=0, it gets counted. This is correct.
                        
                        range_mask = cv2.inRange(masked_target, low, high)
                        final_bin_mask = cv2.bitwise_and(range_mask, mask_roi)
                        
                        # Handle potential high-bound exclusivity consistency
                        if i < len(bins) - 2:
                            # [low, high)
                            # inRange is inclusive [low, high]. We need to exclude high if strictly <
                            # But cv2.inRange doesn't support open interval.
                            # We can subtract high values? Or just accept overlap/inconsistency.
                            # Standard HSV logic used: (ch_data >= low) & (ch_data < high) via numpy.
                            # Let's use numpy for consistency if performance allows, or accept inRange.
                            # HSV used numpy. Let's use numpy here too for consistency.
                            pass 
                        
                        # Numpy approach for binning (Consistent with HSV)
                        # Extract pixels inside mask
                        valid_pixels = target_img[mask_roi > 0]
                        if i < len(bins) - 2:
                            count = np.sum((valid_pixels >= low) & (valid_pixels < high))
                        else:
                            count = np.sum((valid_pixels >= low) & (valid_pixels <= high))
                            
                        val = count * defect_area_scale
                        results[f"Morph_{morph_name}[{low}-{high}]"] = val
                        total_bin_sum += val
                        
                        # --- Complement Region ---
                        if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_mask_roi is not None:
                            valid_pixels_c = target_img[complement_mask_roi > 0]
                            if i < len(bins) - 2:
                                count_c = np.sum((valid_pixels_c >= low) & (valid_pixels_c < high))
                            else:
                                count_c = np.sum((valid_pixels_c >= low) & (valid_pixels_c <= high))
                            
                            complement_val = count_c * complement_area_scale
                            results[f"Morph_{morph_name}_C[{low}-{high}]"] = complement_val
                    
                    print(f"DEBUG: Morph {morph_name} Total Bin Sum: {total_bin_sum:.4f}")

                # --- Ring Comparison Mode ---
                elif str(mode).upper().startswith('RING'):
                    print(f"DEBUG: Calculating Ring Comparison...")
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Parse Params
                    ring_width = 10  # Default ring width
                    metric_type = 'diff'  # Default: 'diff' (absolute diff), 'ratio' (ratio)
                    
                    for k, v in gray_scale_params.items():
                        k_norm = str(k).lower().replace(' ', '').replace('_', '')
                        if k_norm in ('ringwidth', 'width', 'ring'):
                            try:
                                ring_width = int(float(v))
                            except: pass
                        elif k_norm in ('metrictype', 'metric', 'type'):
                            metric_type = str(v).lower()
                    
                    # Ensure ring width is valid
                    if ring_width < 1: ring_width = 10
                    if ring_width > 50: ring_width = 50
                    
                    print(f"DEBUG: Ring Comparison - Width:{ring_width}, Metric:{metric_type}")
                    
                    # Create ring mask (dilated mask minus original mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
                    mask_dilated = cv2.dilate(mask_roi, kernel)
                    ring_mask = cv2.subtract(mask_dilated, mask_roi)
                    
                    # Debug: Print overall statistics
                    inner_pixels_all = gray_roi[mask_roi > 0]
                    ring_pixels_all = gray_roi[ring_mask > 0]
                    print(f"DEBUG: Ring Comparison - Inner pixels: {len(inner_pixels_all)}, Ring pixels: {len(ring_pixels_all)}")
                    if len(inner_pixels_all) > 0:
                        print(f"DEBUG: Inner - Min: {np.min(inner_pixels_all)}, Max: {np.max(inner_pixels_all)}, Mean: {np.mean(inner_pixels_all):.2f}")
                    if len(ring_pixels_all) > 0:
                        print(f"DEBUG: Ring - Min: {np.min(ring_pixels_all)}, Max: {np.max(ring_pixels_all)}, Mean: {np.mean(ring_pixels_all):.2f}")
                    
                    # Check if Binning is enabled
                    use_binning = len(bins) >= 2
                    
                    if use_binning:
                        print(f"DEBUG: Ring Comparison with Binning. Bins: {bins}")
                        total_bin_sum = 0
                        
                        for i in range(len(bins) - 1):
                            low = bins[i]
                            high = bins[i+1]
                            
                            # Get pixels in current bin from inner region (mask)
                            inner_pixels = gray_roi[mask_roi > 0]
                            if len(inner_pixels) > 0:
                                if i < len(bins) - 2:
                                    inner_bin_mask = (inner_pixels >= low) & (inner_pixels < high)
                                else:
                                    inner_bin_mask = (inner_pixels >= low) & (inner_pixels <= high)
                                inner_bin_pixels = inner_pixels[inner_bin_mask]
                            else:
                                inner_bin_pixels = np.array([])
                            
                            # Get pixels in current bin from outer ring
                            ring_pixels = gray_roi[ring_mask > 0]
                            if len(ring_pixels) > 0:
                                if i < len(bins) - 2:
                                    ring_bin_mask = (ring_pixels >= low) & (ring_pixels < high)
                                else:
                                    ring_bin_mask = (ring_pixels >= low) & (ring_pixels <= high)
                                ring_bin_pixels = ring_pixels[ring_bin_mask]
                            else:
                                ring_bin_pixels = np.array([])
                            
                            # Calculate statistics for this bin
                            inner_mean = np.mean(inner_bin_pixels) if len(inner_bin_pixels) > 0 else 0
                            inner_std = np.std(inner_bin_pixels) if len(inner_bin_pixels) > 0 else 0
                            outer_mean = np.mean(ring_bin_pixels) if len(ring_bin_pixels) > 0 else 0
                            outer_std = np.std(ring_bin_pixels) if len(ring_bin_pixels) > 0 else 0
                            
                            # Calculate diff for this bin
                            if metric_type == 'ratio':
                                diff_mean = inner_mean / (outer_mean + 1e-6) if outer_mean > 0 else 0
                            else:
                                diff_mean = abs(inner_mean - outer_mean)
                            
                            # Calculate percentage of pixels in this bin
                            inner_bin_count = len(inner_bin_pixels)
                            inner_total = len(inner_pixels) if len(inner_pixels) > 0 else 1
                            bin_pct = (inner_bin_count / inner_total) * 100.0
                            
                            # Calculate Weighted_Diff = Diff * Pct^3 / 10000
                            # This further reduces the impact of small areas with high differences
                            weighted_diff = diff_mean * (bin_pct ** 3) / 10000.0
                            
                            # Store results for this bin (only Inner, Outer, Diff, Pct, Weighted_Diff)
                            results[f'Ring_Inner[{low}-{high}]'] = inner_mean
                            results[f'Ring_Outer[{low}-{high}]'] = outer_mean
                            results[f'Ring_Diff[{low}-{high}]'] = diff_mean
                            results[f'Ring_Pct[{low}-{high}]'] = bin_pct
                            results[f'Weighted_Diff[{low}-{high}]'] = weighted_diff
                            
                            total_bin_sum += weighted_diff
                            
                            print(f"DEBUG: Ring Bin[{low}-{high}] - Inner:{inner_mean:.2f}, Outer:{outer_mean:.2f}, Diff:{diff_mean:.2f}, Pct:{bin_pct:.2f}%, Weighted_Diff:{weighted_diff:.2f}")
                        
                        print(f"DEBUG: Ring Comparison Total Bin Sum: {total_bin_sum:.4f}")
                        
                        # Also calculate overall statistics (without binning)
                        inner_pixels_all = gray_roi[mask_roi > 0]
                        ring_pixels_all = gray_roi[ring_mask > 0]
                        inner_mean_all = np.mean(inner_pixels_all) if len(inner_pixels_all) > 0 else 0
                        outer_mean_all = np.mean(ring_pixels_all) if len(ring_pixels_all) > 0 else 0
                        if metric_type == 'ratio':
                            diff_mean_all = inner_mean_all / (outer_mean_all + 1e-6) if outer_mean_all > 0 else 0
                        else:
                            diff_mean_all = abs(inner_mean_all - outer_mean_all)
                        
                        results['Ring_Inner_Mean'] = inner_mean_all
                        results['Ring_Outer_Mean'] = outer_mean_all
                        results['Ring_Diff_Mean'] = diff_mean_all
                        
                    else:
                        # Original non-binning implementation
                        # Calculate statistics for inner region (mask)
                        inner_pixels = gray_roi[mask_roi > 0]
                        inner_mean = np.mean(inner_pixels) if len(inner_pixels) > 0 else 0
                        inner_std = np.std(inner_pixels) if len(inner_pixels) > 0 else 0
                        inner_median = np.median(inner_pixels) if len(inner_pixels) > 0 else 0
                        
                        # Calculate statistics for outer ring
                        ring_pixels = gray_roi[ring_mask > 0]
                        outer_mean = np.mean(ring_pixels) if len(ring_pixels) > 0 else 0
                        outer_std = np.std(ring_pixels) if len(ring_pixels) > 0 else 0
                        outer_median = np.median(ring_pixels) if len(ring_pixels) > 0 else 0
                        
                        # Calculate comparison metrics
                        if metric_type == 'ratio':
                            diff_mean = inner_mean / (outer_mean + 1e-6)
                            diff_median = inner_median / (outer_median + 1e-6)
                        else:  # 'diff' or default
                            diff_mean = abs(inner_mean - outer_mean)
                            diff_median = abs(inner_median - outer_median)
                        
                        diff_std = abs(inner_std - outer_std)
                        
                        # Store results
                        results['Ring_Inner_Mean'] = inner_mean
                        results['Ring_Inner_Median'] = inner_median
                        results['Ring_Inner_Std'] = inner_std
                        results['Ring_Outer_Mean'] = outer_mean
                        results['Ring_Outer_Median'] = outer_median
                        results['Ring_Outer_Std'] = outer_std
                        results['Ring_Diff_Mean'] = diff_mean
                        results['Ring_Diff_Median'] = diff_median
                        results['Ring_Diff_Std'] = diff_std
                        
                        print(f"DEBUG: Ring Comparison - Inner Mean:{inner_mean:.2f}, Outer Mean:{outer_mean:.2f}, Diff:{diff_mean:.2f}")
                    
                    # --- Complement Region (if Comparison mode) ---
                    if advanced_scoring and advanced_scoring.lower() == 'comparison' and complement_mask_roi is not None:
                        complement_mask_dilated = cv2.dilate(complement_mask_roi, kernel)
                        complement_ring_mask = cv2.subtract(complement_mask_dilated, complement_mask_roi)
                        
                        if use_binning:
                            # Binning for complement
                            for i in range(len(bins) - 1):
                                low = bins[i]
                                high = bins[i+1]
                                
                                comp_inner_pixels = gray_roi[complement_mask_roi > 0]
                                if len(comp_inner_pixels) > 0:
                                    if i < len(bins) - 2:
                                        comp_inner_bin_mask = (comp_inner_pixels >= low) & (comp_inner_pixels < high)
                                    else:
                                        comp_inner_bin_mask = (comp_inner_pixels >= low) & (comp_inner_pixels <= high)
                                    comp_inner_bin_pixels = comp_inner_pixels[comp_inner_bin_mask]
                                else:
                                    comp_inner_bin_pixels = np.array([])
                                
                                comp_ring_pixels = gray_roi[complement_ring_mask > 0]
                                if len(comp_ring_pixels) > 0:
                                    if i < len(bins) - 2:
                                        comp_ring_bin_mask = (comp_ring_pixels >= low) & (comp_ring_pixels < high)
                                    else:
                                        comp_ring_bin_mask = (comp_ring_pixels >= low) & (comp_ring_pixels <= high)
                                    comp_ring_bin_pixels = comp_ring_pixels[comp_ring_bin_mask]
                                else:
                                    comp_ring_bin_pixels = np.array([])
                                
                                comp_inner_mean = np.mean(comp_inner_bin_pixels) if len(comp_inner_bin_pixels) > 0 else 0
                                comp_outer_mean = np.mean(comp_ring_bin_pixels) if len(comp_ring_bin_pixels) > 0 else 0
                                
                                if metric_type == 'ratio':
                                    comp_diff_mean = comp_inner_mean / (comp_outer_mean + 1e-6) if comp_outer_mean > 0 else 0
                                else:
                                    comp_diff_mean = abs(comp_inner_mean - comp_outer_mean)
                                
                                results[f'Ring_C_Inner_Mean[{low}-{high}]'] = comp_inner_mean
                                results[f'Ring_C_Outer_Mean[{low}-{high}]'] = comp_outer_mean
                                results[f'Ring_C_Diff_Mean[{low}-{high}]'] = comp_diff_mean
                                
                                comp_inner_bin_count = len(comp_inner_bin_pixels)
                                comp_inner_total = len(comp_inner_pixels) if len(comp_inner_pixels) > 0 else 1
                                comp_bin_pct = (comp_inner_bin_count / comp_inner_total) * 100.0
                                results[f'Ring_C_Bin_Pct[{low}-{high}]'] = comp_bin_pct
                        else:
                            # Original non-binning for complement
                            comp_inner_pixels = gray_roi[complement_mask_roi > 0]
                            comp_inner_mean = np.mean(comp_inner_pixels) if len(comp_inner_pixels) > 0 else 0
                            comp_inner_std = np.std(comp_inner_pixels) if len(comp_inner_pixels) > 0 else 0
                            comp_inner_median = np.median(comp_inner_pixels) if len(comp_inner_pixels) > 0 else 0
                            
                            comp_ring_pixels = gray_roi[complement_ring_mask > 0]
                            comp_outer_mean = np.mean(comp_ring_pixels) if len(comp_ring_pixels) > 0 else 0
                            comp_outer_std = np.std(comp_ring_pixels) if len(comp_ring_pixels) > 0 else 0
                            comp_outer_median = np.median(comp_ring_pixels) if len(comp_ring_pixels) > 0 else 0
                            
                            if metric_type == 'ratio':
                                comp_diff_mean = comp_inner_mean / (comp_outer_mean + 1e-6)
                                comp_diff_median = comp_inner_median / (comp_outer_median + 1e-6)
                            else:
                                comp_diff_mean = abs(comp_inner_mean - comp_outer_mean)
                                comp_diff_median = abs(comp_inner_median - comp_outer_median)
                            
                            comp_diff_std = abs(comp_inner_std - comp_outer_std)
                            
                            results['Ring_C_Inner_Mean'] = comp_inner_mean
                            results['Ring_C_Inner_Median'] = comp_inner_median
                            results['Ring_C_Inner_Std'] = comp_inner_std
                            results['Ring_C_Outer_Mean'] = comp_outer_mean
                            results['Ring_C_Outer_Median'] = comp_outer_median
                            results['Ring_C_Outer_Std'] = comp_outer_std
                            results['Ring_C_Diff_Mean'] = comp_diff_mean
                            results['Ring_C_Diff_Median'] = comp_diff_median
                            results['Ring_C_Diff_Std'] = comp_diff_std
                            
                            print(f"DEBUG: Ring Comparison (Complement) - Inner Mean:{comp_inner_mean:.2f}, Outer Mean:{comp_outer_mean:.2f}, Diff:{comp_diff_mean:.2f}")

                # --- Neighboring Gray Scale Mode ---
                elif str(mode).upper().startswith('NEIGHBORING'):
                    print(f"DEBUG: Calculating Neighboring Gray Scale...")
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Parse Params
                    # Support both single value (e.g., kernel=3) and range (e.g., kernel=[10,12])
                    kernel_inner = 0   # Inner radius (0 means starting from defect edge)
                    kernel_outer = 3   # Outer radius (default 3)
                    min_size = 0       # Minimum component size (area in pixels), 0 means no filtering
                    save_neighbor_img = True  # Default: generate neighbor image
                    
                    for k, v in gray_scale_params.items():
                        k_norm = str(k).lower().replace(' ', '').replace('_', '')
                        if k_norm in ('kernel', 'kernelsize', 'ksize', 'kernel_size'):
                            try:
                                v_str = str(v).strip()
                                # Check if it's a range format [min, max] or (min, max)
                                if (v_str.startswith('[') and v_str.endswith(']')) or \
                                   (v_str.startswith('(') and v_str.endswith(')')):
                                    # Parse range like [10,12] or (10, 12)
                                    inner_content = v_str[1:-1].strip()
                                    parts = [p.strip() for p in inner_content.split(',')]
                                    if len(parts) >= 2:
                                        kernel_inner = int(float(parts[0]))
                                        kernel_outer = int(float(parts[1]))
                                    elif len(parts) == 1:
                                        # Single value in brackets, treat as outer radius
                                        kernel_outer = int(float(parts[0]))
                                else:
                                    # Single value format
                                    kernel_outer = int(float(v_str))
                                    kernel_inner = 0
                            except Exception as e:
                                print(f"DEBUG: Failed to parse kernel value '{v}': {e}, using default")
                                kernel_inner = 0
                                kernel_outer = 3
                        elif k_norm in ('minsize', 'min_size', 'minarea', 'min_area', 'size'):
                            try:
                                min_size = int(float(v))
                            except:
                                min_size = 0
                        elif k_norm in ('neighborimg', 'neighbor_img', 'neighboringimg', 'neighboring_img', 'pic', 'saveimg', 'save_img'):
                            # Control whether to generate neighbor image
                            v_str = str(v).lower().strip()
                            if v_str in ('off', 'false', '0', 'no', 'disable'):
                                save_neighbor_img = False
                            elif v_str in ('on', 'true', '1', 'yes', 'enable'):
                                save_neighbor_img = True
                    
                    # Ensure kernel sizes are valid
                    if kernel_inner < 0: kernel_inner = 0
                    if kernel_outer < 1: kernel_outer = 3
                    if kernel_outer > 51: kernel_outer = 51
                    if min_size < 0: min_size = 0
                    if kernel_inner >= kernel_outer:
                        kernel_inner = max(0, kernel_outer - 1)
                    
                    # Ensure odd kernel sizes for morphological operations
                    kernel_inner_size = kernel_inner * 2 + 1 if kernel_inner > 0 else 0
                    kernel_outer_size = kernel_outer * 2 + 1
                    
                    print(f"DEBUG: Neighboring Gray Scale - Inner Radius:{kernel_inner}, Outer Radius:{kernel_outer}, Min Size:{min_size}")
                    
                    # Create neighboring mask (dilated with outer kernel minus dilated with inner kernel)
                    # This creates a ring/annulus region at distance [kernel_inner, kernel_outer]
                    kernel_outer_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_outer_size, kernel_outer_size))
                    mask_dilated_outer = cv2.dilate(mask_roi, kernel_outer_elem)
                    
                    if kernel_inner > 0:
                        kernel_inner_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_inner_size, kernel_inner_size))
                        mask_dilated_inner = cv2.dilate(mask_roi, kernel_inner_elem)
                        neighbor_mask = cv2.subtract(mask_dilated_outer, mask_dilated_inner)
                    else:
                        # Original behavior: outer dilated minus original mask
                        neighbor_mask = cv2.subtract(mask_dilated_outer, mask_roi)
                    
                    # Debug: Print overall statistics
                    inner_pixels_all = gray_roi[mask_roi > 0]
                    neighbor_pixels_all = gray_roi[neighbor_mask > 0]
                    print(f"DEBUG: Neighboring Gray Scale - Inner pixels: {len(inner_pixels_all)}, Neighbor pixels: {len(neighbor_pixels_all)}")
                    if len(inner_pixels_all) > 0:
                        print(f"DEBUG: Inner - Min: {np.min(inner_pixels_all)}, Max: {np.max(inner_pixels_all)}, Mean: {np.mean(inner_pixels_all):.2f}")
                    if len(neighbor_pixels_all) > 0:
                        print(f"DEBUG: Neighbor - Min: {np.min(neighbor_pixels_all)}, Max: {np.max(neighbor_pixels_all)}, Mean: {np.mean(neighbor_pixels_all):.2f}")
                    
                    # Calculate statistics for inner region (mask)
                    inner_pixels = gray_roi[mask_roi > 0]
                    inner_mean = np.mean(inner_pixels) if len(inner_pixels) > 0 else 0
                    inner_std = np.std(inner_pixels) if len(inner_pixels) > 0 else 0
                    inner_median = np.median(inner_pixels) if len(inner_pixels) > 0 else 0
                    
                    # Calculate statistics for neighboring region
                    neighbor_pixels = gray_roi[neighbor_mask > 0]
                    neighbor_mean = np.mean(neighbor_pixels) if len(neighbor_pixels) > 0 else 0
                    neighbor_std = np.std(neighbor_pixels) if len(neighbor_pixels) > 0 else 0
                    neighbor_median = np.median(neighbor_pixels) if len(neighbor_pixels) > 0 else 0
                    
                    # Calculate comparison metrics
                    diff_mean = abs(inner_mean - neighbor_mean)
                    diff_median = abs(inner_median - neighbor_median)
                    diff_std = abs(inner_std - neighbor_std)
                    
                    # Store results
                    results['Neighboring_Inner_Mean'] = inner_mean
                    results['Neighboring_Inner_Median'] = inner_median
                    results['Neighboring_Inner_Std'] = inner_std
                    results['Neighboring_Outer_Mean'] = neighbor_mean
                    results['Neighboring_Outer_Median'] = neighbor_median
                    results['Neighboring_Outer_Std'] = neighbor_std
                    results['Neighboring_Diff_Mean'] = diff_mean
                    results['Neighboring_Diff_Median'] = diff_median
                    results['Neighboring_Diff_Std'] = diff_std
                    
                    print(f"DEBUG: Neighboring Gray Scale - Inner Mean:{inner_mean:.2f}, Neighbor Mean:{neighbor_mean:.2f}, Diff:{diff_mean:.2f}")
                    
                    # --- Per-Component Difference Analysis (Optimized) ---
                    # Find connected components (each separate red region)
                    # Use connectivity=4 to only consider physically connected pixels (not diagonal)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=4)
                    
                    # num_labels includes background (label 0), so actual components = num_labels - 1
                    print(f"DEBUG: Found {num_labels - 1} connected components")
                    
                    # OPTIMIZATION: Batch process all components at once
                    # Pre-filter by min_size using stats (much faster than creating masks)
                    component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (index 0)
                    valid_component_ids = np.where(component_areas >= min_size)[0] + 1  # +1 because labels start from 1
                    skipped_count = (num_labels - 1) - len(valid_component_ids)
                    
                    if skipped_count > 0:
                        print(f"DEBUG: Skipped {skipped_count} components below min_size={min_size}")
                    
                    if len(valid_component_ids) == 0:
                        component_data = []
                    else:
                        # OPTIMIZATION: Use scipy.ndimage for efficient batch processing
                        from scipy import ndimage
                        
                        # Create dilated version of entire labels image once
                        labels_dilated = cv2.dilate(labels.astype(np.uint16), kernel_outer_elem)
                        
                        # Calculate inner mean for all components at once using ndimage.mean
                        # This is much faster than looping
                        inner_means = ndimage.mean(gray_roi, labels, valid_component_ids)
                        
                        # Calculate neighbor means using the dilated labels
                        # Create neighbor labels (dilated - original)
                        neighbor_labels = labels_dilated.copy()
                        neighbor_labels[labels > 0] = 0  # Zero out original defect pixels
                        
                        # Calculate mean for each component's neighbor region
                        neighbor_means = ndimage.mean(gray_roi, neighbor_labels, valid_component_ids)
                        
                        # Get areas from stats
                        component_areas = stats[valid_component_ids, cv2.CC_STAT_AREA]
                        
                        # Calculate diffs
                        diffs = np.abs(inner_means - neighbor_means)
                        
                        # Filter out components with no neighbors (neighbor_mean = 0 or nan)
                        valid_mask = ~np.isnan(neighbor_means) & (neighbor_means > 0)
                        
                        component_data = []
                        processed_count = 0
                        
                        for i, label_id in enumerate(valid_component_ids):
                            if not valid_mask[i]:
                                continue
                                
                            component_area = component_areas[i]
                            component_mean = inner_means[i]
                            neighbor_mean_comp = neighbor_means[i]
                            diff = diffs[i]
                            
                            component_data.append((diff, component_area))
                            processed_count += 1
                            
                            # Only print component info for first few and every 50th
                            if processed_count <= 5 or processed_count % 50 == 0:
                                print(f"DEBUG: Component {label_id} - Area:{component_area}px, Inner:{component_mean:.2f}, Outer:{neighbor_mean_comp:.2f}, Diff:{diff:.2f}")
                        
                        print(f"DEBUG: Processed {processed_count} components above min_size={min_size}")
                    
                    # Calculate total ROI area (DUT area after ROI Shrink and Slicing)
                    # dut_area is the total DUT area, mask_roi is the defect mask within DUT
                    total_roi_area = dut_area
                    total_mask_pixels = np.sum(mask_roi > 0)  # For reference: actual defect pixels
                    
                    # Debug: Print detailed area info
                    print(f"DEBUG: Total ROI Area (DUT area): {total_roi_area}, Total Defect Pixels: {total_mask_pixels}")
                    
                    if len(component_data) > 0:
                        # Unpack data
                        component_diffs = np.array([d[0] for d in component_data])
                        component_areas = np.array([d[1] for d in component_data])
                        total_mask_area = np.sum(component_areas)
                        
                        # Store component diff statistics
                        results['Neighboring_CompDiff_Mean'] = np.mean(component_diffs)
                        results['Neighboring_CompDiff_Median'] = np.median(component_diffs)
                        results['Neighboring_CompDiff_Min'] = np.min(component_diffs)
                        results['Neighboring_CompDiff_Max'] = np.max(component_diffs)
                        results['Neighboring_CompDiff_Std'] = np.std(component_diffs)
                        results['Neighboring_CompDiff_Count'] = len(component_diffs)
                        
                        # --- Defect above TH: proportion of mask area above min_size ---
                        if total_roi_area > 0:
                            defect_above_th = (total_mask_area / total_roi_area) * 100.0
                        else:
                            defect_above_th = 0.0
                        results['Defect_above_TH'] = defect_above_th
                        
                        # --- New Score: sum of (component_area_ratio * neighboring_diff) ---
                        # Calculate area ratio for each component relative to total ROI
                        if total_roi_area > 0:
                            area_ratios = component_areas / total_roi_area
                        else:
                            area_ratios = np.zeros_like(component_areas, dtype=float)
                        
                        # Calculate weighted score: sum(area_ratio * diff)
                        new_score = np.sum(area_ratios * component_diffs)
                        results['Weighted Score'] = new_score
                        
                        print(f"DEBUG: Component Diff Stats - Mean:{np.mean(component_diffs):.2f}, Median:{np.median(component_diffs):.2f}, Min:{np.min(component_diffs):.2f}, Max:{np.max(component_diffs):.2f}, Count:{len(component_diffs)}, TotalArea:{total_mask_area}")
                        print(f"DEBUG: Defect_above_TH: {defect_above_th:.2f}%, Weighted Score: {new_score:.4f}")
                    else:
                        # No components above min_size
                        results['Neighboring_CompDiff_Mean'] = 0
                        results['Neighboring_CompDiff_Median'] = 0
                        results['Neighboring_CompDiff_Min'] = 0
                        results['Neighboring_CompDiff_Max'] = 0
                        results['Neighboring_CompDiff_Std'] = 0
                        results['Neighboring_CompDiff_Count'] = 0
                        results['Defect_above_TH'] = 0.0
                        results['Weighted Score'] = 0.0
                        print(f"DEBUG: No components above min_size={min_size}. Defect_above_TH: 0%, Weighted Score: 0")
                        
                        # --- Binning Distribution of Component Differences (Area-Weighted) ---
                        if len(bins) >= 2:
                            print(f"DEBUG: Calculating area-weighted component difference distribution with bins: {bins}")
                            
                            if total_mask_area > 0:
                                for i in range(len(bins) - 1):
                                    low = bins[i]
                                    high = bins[i+1]
                                    
                                    # Find components in this bin
                                    if i < len(bins) - 2:
                                        in_bin = (component_diffs >= low) & (component_diffs < high)
                                    else:
                                        in_bin = (component_diffs >= low) & (component_diffs <= high)
                                    
                                    # Calculate total area of components in this bin
                                    area_in_bin = np.sum(component_areas[in_bin])
                                    
                                    # Calculate percentage based on area (not count)
                                    percentage = (area_in_bin / total_mask_area) * 100.0
                                    results[f'Neighboring_CompDiffDist[{low}-{high}]'] = percentage
                                    
                                    comp_count = np.sum(in_bin)
                                    print(f"DEBUG: CompDiff Bin[{low}-{high}]: {int(comp_count)} components, Area:{area_in_bin}px ({percentage:.2f}%)")
                    
                    # --- Generate Reference Image (only if save_neighbor_img is True) ---
                    if save_neighbor_img:
                        try:
                            # OPTIMIZATION: Skip detailed annotation if too many components
                            # Reuse already computed labels, stats, centroids from earlier
                            # valid_component_ids already contains components above min_size
                            
                            valid_component_count = len(valid_component_ids)
                            
                            # If too many components, skip detailed annotation and just save basic image
                            MAX_COMPONENTS_FOR_ANNOTATION = 100
                            if valid_component_count > MAX_COMPONENTS_FOR_ANNOTATION:
                                print(f"DEBUG: Too many components ({valid_component_count}) for detailed annotation. Skipping component-level visualization.")
                                # Create simple visualization without per-component annotation
                                neighbor_vis = np.zeros_like(gray_roi)
                                neighbor_vis[neighbor_mask > 0] = gray_roi[neighbor_mask > 0]
                                neighbor_color = cv2.cvtColor(neighbor_vis, cv2.COLOR_GRAY2BGR)
                                # Just draw red for all defects and green for all neighbors
                                neighbor_color[mask_roi > 0] = [0, 0, 255]  # Red for masks
                                neighbor_color[neighbor_mask > 0] = [0, 255, 0]  # Green for neighbors
                            else:
                                # Create a visualization of the neighboring pixels
                                neighbor_vis = np.zeros_like(gray_roi)
                                neighbor_vis[neighbor_mask > 0] = gray_roi[neighbor_mask > 0]
                                
                                # Create a color visualization
                                neighbor_color = cv2.cvtColor(neighbor_vis, cv2.COLOR_GRAY2BGR)
                                
                                # Batch process - draw all red masks at once
                                neighbor_color[mask_roi > 0] = [0, 0, 255]  # Red for all masks
                                
                                # Pre-calculate dilated labels for efficient neighbor lookup
                                labels_dilated = cv2.dilate(labels.astype(np.uint16), kernel_outer_elem)
                                
                                # Process only components that pass min_size filter (use valid_component_ids)
                                for label_id in valid_component_ids:
                                    component_area = stats[label_id, cv2.CC_STAT_AREA]
                                    
                                    # Get centroid for text placement
                                    cx, cy = int(centroids[label_id][0]), int(centroids[label_id][1])
                                    
                                    # Use pre-dilated labels to find neighbors
                                    comp_mask_bool = (labels == label_id)
                                    neighbor_mask_bool = (labels_dilated == label_id) & (labels != label_id)
                                    
                                    # Only draw green where neighbor_mask is valid (within the ring/annulus) and not on other defects
                                    valid_neighbor = neighbor_mask_bool & (neighbor_mask > 0) & (mask_roi == 0)
                                    neighbor_color[valid_neighbor] = [0, 255, 0]  # Green for neighbors
                                    
                                    # Calculate diff value for annotation
                                    component_pixels = gray_roi[comp_mask_bool]
                                    neighbor_pixels_comp = gray_roi[neighbor_mask_bool]
                                    if len(component_pixels) == 0 or len(neighbor_pixels_comp) == 0:
                                        continue
                                    
                                    component_mean = np.mean(component_pixels)
                                    neighbor_mean_comp = np.mean(neighbor_pixels_comp)
                                    diff = abs(component_mean - neighbor_mean_comp)
                                    
                                    # Draw text with very small font
                                    text = f"{diff:.1f}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.3  # Very small font
                                    thickness = 1
                                    
                                    # Get text size to center it
                                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                                    text_x = cx - text_width // 2
                                    text_y = cy + text_height // 2
                                    
                                    # Draw black background for better visibility
                                    cv2.putText(neighbor_color, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
                                    # Draw white text
                                    cv2.putText(neighbor_color, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                            
                            # Save to reference folder (use output_dir if provided, fallback to default)
                            if output_dir is not None:
                                ref_folder = os.path.join(output_dir, "reference")
                            else:
                                ref_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference")
                            
                            if not os.path.exists(ref_folder):
                                os.makedirs(ref_folder)
                            
                            # Generate unique filename based on image_filename if available
                            if image_filename:
                                base_name = os.path.splitext(os.path.basename(image_filename))[0]
                                if kernel_inner > 0:
                                    ref_filename = f"{base_name}_Neighboring_GrayScale_k{kernel_inner}-{kernel_outer}.png"
                                else:
                                    ref_filename = f"{base_name}_Neighboring_GrayScale_k{kernel_outer}.png"
                            else:
                                if kernel_inner > 0:
                                    ref_filename = f"Neighboring_GrayScale_k{kernel_inner}-{kernel_outer}.png"
                                else:
                                    ref_filename = f"Neighboring_GrayScale_k{kernel_outer}.png"
                            
                            ref_path = os.path.join(ref_folder, ref_filename)
                            cv2.imwrite(ref_path, neighbor_color)
                            print(f"DEBUG: Saved Neighboring Gray Scale reference image to {ref_path}")
                        except Exception as e:
                            print(f"DEBUG: Failed to save Neighboring Gray Scale reference image: {e}")
                    else:
                        print(f"DEBUG: Skipping Neighboring Gray Scale image generation (save_neighbor_img={save_neighbor_img})")

        except Exception as e:
            print(f"Error in Color Space Measurement: {e}")

    # --- Calculate Defect/Ref% if Area_Ref is requested ---
    # Use global Reference_BBox_Area if available, or try to detect Reference using detector
    global Reference_BBox_Area
    if 'Area_Ref' in output_config or 'area_ref' in [k.lower() for k in output_config]:
        # Try to detect Reference if Reference_BBox_Area is not set and detector is available
        ref_bbox_area = Reference_BBox_Area
        if ref_bbox_area is None and detector is not None and image is not None:
            print("DEBUG: Area_Ref - Trying to detect Reference class using detector...")
            ref_bbox_area, _ = detect_reference_bbox_area(image, detector, score_thresh=0.5)
            if ref_bbox_area is not None:
                print(f"DEBUG: Area_Ref - Auto-detected Reference bbox area: {ref_bbox_area}px")
        
        if ref_bbox_area is not None and ref_bbox_area > 0:
            # Calculate defect area from defect_mask if available
            if defect_mask is not None:
                defect_area_pixels = cv2.countNonZero(defect_mask)
                defect_ref_pct = (defect_area_pixels / ref_bbox_area) * 100
                results['Defect/Ref%'] = f"{round(defect_ref_pct, 2)}%"
                results['Defect_Area_Pixels'] = defect_area_pixels
                results['Reference_BBox_Area'] = ref_bbox_area
                print(f"DEBUG: Area_Ref calculation - Defect Area: {defect_area_pixels}px, Ref BBox Area: {ref_bbox_area}px, Defect/Ref%: {defect_ref_pct:.2f}%")
            else:
                results['Defect/Ref%'] = "N/A (no defect mask)"
                results['Reference_BBox_Area'] = ref_bbox_area
                print(f"DEBUG: Area_Ref requested but no defect mask available")
        else:
            results['Defect/Ref%'] = "N/A (no reference bbox)"
            results['Reference_BBox_Area'] = 0
            print(f"DEBUG: Area_Ref requested but no reference bbox area available")

    print(f"DEBUG: calculate_parametric_dimensions returning keys: {list(results.keys())}")
    print(f"DEBUG: results has Defect_above_TH: {'Defect_above_TH' in results}")
    print(f"DEBUG: results has Weighted Score: {'Weighted Score' in results}")
    return results


def format_parametric_dict(param_dict, path, ref_params, defect_output_format):
    dir_val = param_dict.get('Dir', '')
    if not dir_val:
        try:
            dir_val = os.path.dirname(path)
        except Exception:
            dir_val = ""
    dof = str(defect_output_format).strip().lower()
    is_combined = dof == 'combined'
    
    # 如果字典里已经自带了正确的 Reference（比如 DetectronSegStrategy 自己计算的），就不要用最后残留的全局变量去覆盖
    if 'Reference' not in param_dict:
        ref_str, ref_status = build_reference_info(ref_params)
        param_dict['Reference'] = ref_str
        param_dict['Reference_Status'] = ref_status
        
    param_dict['Dir'] = dir_val
    if 'filename' in param_dict:
        param_dict['Filename'] = param_dict.pop('filename')
    if 'defect pct' in param_dict:
        param_dict['Defect Pct'] = param_dict.pop('defect pct')
    if is_combined:
        if 'Defect_ID' in param_dict:
            param_dict.pop('Defect_ID', None)
        if 'Contour' in param_dict:
            param_dict.pop('Contour', None)
        if 'Ratio_Length' in param_dict:
            param_dict.pop('Ratio_Length', None)
        if 'Ratio_Width' in param_dict:
            param_dict.pop('Ratio_Width', None)
        if 'Length_Ratio' in param_dict:
            param_dict.pop('Length_Ratio', None)
        if 'Width_Ratio' in param_dict:
            param_dict.pop('Width_Ratio', None)
        if 'Curved Line Length' in param_dict:
            param_dict.pop('Curved Line Length', None)
    else:
        if 'Ratio_Length' in param_dict:
            param_dict['Length_Ratio'] = param_dict.pop('Ratio_Length')
        if 'Ratio_Width' in param_dict:
            param_dict['Width_Ratio'] = param_dict.pop('Ratio_Width')
    # Rename Curved Line Measurement to Curved Line Length
    curved_line_found = False
    for k in list(param_dict.keys()):
        kl = str(k).lower()
        if ('curved' in kl) and ('line' in kl) and k != 'Curved Line Length':
            param_dict['Curved Line Length'] = param_dict.pop(k)
            curved_line_found = True
            print(f"DEBUG: Renamed '{k}' to 'Curved Line Length', value: {param_dict['Curved Line Length']}")
            break
    
    if not curved_line_found:
        # Only log once per image to reduce verbosity
        pass  # Silently skip - this is expected for most defects
    
    # Remove Source column as requested
    if 'Source' in param_dict:
        param_dict.pop('Source', None)
    
    # Keep Area_Ref related fields if they exist
    # Defect/Ref%, Defect_Area_Pixels, Reference_BBox_Area are preserved
    area_ref_fields = ['Defect/Ref%', 'Defect_Area_Pixels', 'Reference_BBox_Area']
    has_area_ref = any(field in param_dict for field in area_ref_fields)
    if has_area_ref:
        print(f"DEBUG: format_parametric_dict - Preserving Area_Ref fields: {[f for f in area_ref_fields if f in param_dict]}")
    
    return param_dict


def parse_scoring_setting(setting_str):
    params = {
        'max_weight': 1.0,
        'decay_rate': 0.5,
        'reverse': True,
        'min_weight': None,
        'defect_pct': True,
        'multiplier': 1.0,
        'multiplier_column': None,
        'use_ring_pct': False,
        'roi_coefficient': 1.0,
        'use_new_score': False
    }
    
    if not setting_str or not isinstance(setting_str, str):
        return params
        
    try:
        clean_str = setting_str.strip()
        if clean_str.startswith('['): clean_str = clean_str[1:]
        if clean_str.endswith(']'): clean_str = clean_str[:-1]
        clean_str = clean_str.strip()
        
        import re
        multiplier_match = re.search(r'multiplier\s*=\s*\[([^\]]+)\]', clean_str)
        if multiplier_match:
            multiplier_content = multiplier_match.group(1).strip()
            multiplier_parts = [p.strip() for p in multiplier_content.split(',')]
            if len(multiplier_parts) >= 2:
                params['multiplier_column'] = multiplier_parts[0]
                try:
                    params['multiplier'] = float(multiplier_parts[1])
                except ValueError:
                    pass
            clean_str = clean_str.replace(multiplier_match.group(0), '')
            
        parts = clean_str.split(',')
        for part in parts:
            if '=' not in part:
                continue
            key, val = part.split('=', 1)
            key = key.strip().lower()
            val = val.strip()
            
            if key == 'max_weight':
                params['max_weight'] = float(val)
            elif key == 'decay_rate':
                params['decay_rate'] = float(val)
            elif key == 'reverse':
                params['reverse'] = val.lower() in ('true', 'yes', '1')
            elif key == 'min_weight':
                params['min_weight'] = float(val)
            elif key == 'defect_pct':
                params['defect_pct'] = val.lower() in ('true', 'yes', '1')
            elif key == 'use_ring_pct':
                params['use_ring_pct'] = val.lower() in ('true', 'yes', '1')
            elif key == 'roi_coefficient':
                params['roi_coefficient'] = float(val)
            elif key == 'use_new_score':
                params['use_new_score'] = val.lower() in ('true', 'yes', '1')
            elif key == 'multiplier' and not multiplier_match:
                try:
                    params['multiplier'] = float(val)
                except ValueError:
                    pass
    except Exception as e:
        print(f"Error parsing scoring settings: {e}")
        
    return params


def exponential_decay_weights(n_bins, max_weight=1.0, decay_rate=0.5, reverse=True):
    indices = np.arange(n_bins)
    if reverse:
        weights = max_weight * np.exp(-decay_rate * (n_bins - 1 - indices))
    else:
        weights = max_weight * np.exp(-decay_rate * indices)
    return weights.tolist()


def generate_parametric_row(image_name, dut_image, combined_defect_mask, dut_contour):
    row = {'Filename': image_name}
    total_dut_area = cv2.contourArea(dut_contour)
    total_defect_area = cv2.countNonZero(combined_defect_mask)
    row['Defect Pct'] = (total_defect_area / total_dut_area * 100) if total_dut_area > 0 else 0

    gray_dut = cv2.cvtColor(dut_image, cv2.COLOR_BGR2GRAY)
    defect_pixels = gray_dut[combined_defect_mask > 0]
    bins = np.arange(0, 256, 30)
    bins[-1] = 256

    hist, _ = np.histogram(defect_pixels, bins=bins)
    bin_percentages = (hist / total_defect_area * 100) if total_defect_area > 0 else np.zeros_like(hist)

    bin_cols, weight_cols = [], []
    for i, pct in enumerate(bin_percentages):
        col_name = f'GrayScale[{bins[i]}-{bins[i + 1]}]'
        row[col_name] = pct
        bin_cols.append(col_name)

    scoring_params = parse_scoring_setting(
        "[max_weight=100, decay_rate=0.75, reverse=True, defect_pct=True, multiplier=[Area_Ref, 1000]]")
    weights = exponential_decay_weights(len(bin_cols), scoring_params['max_weight'],
                                        scoring_params['decay_rate'], scoring_params['reverse'])

    weighted_score = sum(pct * w for pct, w in zip(bin_percentages, weights))
    if scoring_params['defect_pct']:
        weighted_score *= row['Defect Pct'] / 100.0
    if scoring_params['multiplier_column'] == 'Area_Ref':
        weighted_score *= scoring_params['multiplier']

    row['Weighted Score'] = weighted_score
    for i, w in enumerate(weights):
        weight_col = f'Weight[{bins[i]}-{bins[i + 1]}]'
        weight_cols.append(weight_col)
        row[weight_col] = w

    return row, bin_cols + weight_cols


