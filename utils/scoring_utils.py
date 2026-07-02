from utils.base_utils import _norm
import os
import numpy as np
import cv2
import re

try:
    from utils import dinov3_utils
except ImportError:
    try:
        import dinov3_utils
    except ImportError:
        dinov3_utils = None


def calculate_complement_score(bin_cols, weights, df, defect_pct_col='Defect Pct', use_defect_pct=True):
    """
    计算补集的加权分数
    
    参数:
        bin_cols: 原始分箱列名列表
        weights: 权重列表
        df: 包含分箱值的数据框
        defect_pct_col: 缺陷百分比列名
        use_defect_pct: 是否乘以缺陷百分比 (True/False)
    
    返回:
        complement_scores: 补集加权分数列表
        complement_bin_values: 补集分箱值字典
    """
    complement_scores = []
    complement_bin_values = {}
    
    for idx, row in df.iterrows():
        weighted_sum = 0.0
        
        for i, bin_col in enumerate(bin_cols):
            # 查找对应的补集列
            complement_col = bin_col.replace('[', '_C[')
            bin_val = row.get(complement_col, 0)
            if bin_val is not None:
                weighted_sum += float(bin_val) * weights[i]
        
        # 乘以缺陷百分比 (如果启用)
        if use_defect_pct:
            defect_pct = row.get(defect_pct_col, 0)
            if defect_pct is not None:
                defect_pct_str = str(defect_pct).strip()
                has_percent = False
                if defect_pct_str.endswith('%'):
                    defect_pct_str = defect_pct_str[:-1]
                    has_percent = True
                try:
                    pct_value = float(defect_pct_str)
                    if has_percent or pct_value > 1:
                        pct_value = pct_value / 100.0
                    final_score = weighted_sum * pct_value
                except ValueError:
                    final_score = weighted_sum
            else:
                final_score = weighted_sum
        else:
            final_score = weighted_sum
        
        complement_scores.append(final_score)
    
    return complement_scores


def exponential_decay_weights(n_bins, max_weight=1.0, decay_rate=0.5, reverse=True, normalize=False, min_weight=None):
    """
    生成指数衰减的权重序列
    
    参数:
        n_bins: 分箱数量（产生的权重个数）
        max_weight: 最大权重值（第一个分箱的权重）
        decay_rate: 衰减率，越大衰减越快（建议 0.1 ~ 2.0）
        reverse: True 表示 S 越小权重越大（反向衰减）
        normalize: 是否归一化（使权重和为1），默认 False
        min_weight: 最小权重值（最后一个分箱的权重），如果提供则自动计算 decay_rate
    
    返回:
        weights: 权重列表
    """
    # 如果提供了 min_weight，自动计算 decay_rate
    if min_weight is not None and n_bins > 1:
        # 公式: min_weight = max_weight * exp(-decay_rate * (n_bins - 1))
        # 解得: decay_rate = -ln(min_weight / max_weight) / (n_bins - 1)
        if max_weight > 0 and min_weight > 0:
            decay_rate = -np.log(min_weight / max_weight) / (n_bins - 1)
            print(f"DEBUG: Auto-calculated decay_rate={decay_rate:.4f} from max_weight={max_weight}, min_weight={min_weight}, n_bins={n_bins}")
    
    indices = np.arange(n_bins)
    
    if reverse:
        # S 越小（索引小）权重越大
        weights = max_weight * np.exp(-decay_rate * (n_bins - 1 - indices))
    else:
        # S 越大（索引大）权重越大
        # 确保第一个分箱的权重为max_weight
        weights = max_weight * np.exp(-decay_rate * indices)
        # 调整权重，使第一个分箱的权重为max_weight
        if n_bins > 0:
            weights[0] = max_weight
    
    # 可选归一化
    if normalize:
        weights = weights / np.sum(weights)
    
    return weights.tolist()


def parse_scoring_setting(setting_str):
    """
    解析 Scoring Setting 字符串，提取参数
    格式: [max_weight=100, decay_rate=2, reverse=True]
    或: [max_weight=100, min_weight=10, reverse=True]  # 自动计算 decay_rate
    或: [max_weight=100, decay_rate=2, defect_pct=False]  # 不乘以 Defect Pct
    或: [max_weight=100, decay_rate=0.75, reverse=True, defect_pct=True, multiplier=[Area_Ref, 100]]
        # multiplier=[Area_Ref, 100] 表示: score = 当前分数 * 100 * (Defect/Ref%)
    """
    params = {
        'max_weight': 1.0,
        'decay_rate': 0.5,
        'reverse': True,
        'min_weight': None,  # 新增参数
        'defect_pct': True,  # 默认乘以 Defect Pct
        'multiplier': 1.0,   # 新增参数: multiplier (默认 1.0)
        'multiplier_column': None,  # 新增参数: multiplier 关联的列名
        'use_ring_pct': False,  # 新增参数: 使用 Ring_Diff * Ring_Pct (默认 False)
        'roi_coefficient': 1.0,  # 新增参数: ROI 系数 (默认 1.0)
        'use_new_score': False   # 新增参数: 是否使用 New_Score 列 (默认 False)
    }
    
    if not setting_str or not isinstance(setting_str, str):
        return params
    
    try:
        # 移除外层方括号（只移除首尾的单个方括号，保留中间的）
        clean_str = setting_str.strip()
        if clean_str.startswith('['):
            clean_str = clean_str[1:]
        if clean_str.endswith(']'):
            clean_str = clean_str[:-1]
        clean_str = clean_str.strip()
        print(f"DEBUG: parse_scoring_setting - Original setting_str: '{setting_str}'")
        print(f"DEBUG: parse_scoring_setting - Cleaned string: '{clean_str}'")
        
        # 处理 multiplier=[Area_Ref, 100] 这种格式
        import re
        # 先提取 multiplier 的特殊格式
        multiplier_match = re.search(r'multiplier\s*=\s*\[([^\]]+)\]', clean_str)
        print(f"DEBUG: parse_scoring_setting - multiplier_match: {multiplier_match}")
        if multiplier_match:
            multiplier_content = multiplier_match.group(1).strip()
            print(f"DEBUG: parse_scoring_setting - multiplier_content: '{multiplier_content}'")
            multiplier_parts = [p.strip() for p in multiplier_content.split(',')]
            print(f"DEBUG: parse_scoring_setting - multiplier_parts: {multiplier_parts}")
            if len(multiplier_parts) >= 2:
                params['multiplier_column'] = multiplier_parts[0]  # 如 'Area_Ref'
                params['multiplier'] = float(multiplier_parts[1])  # 如 100
                print(f"DEBUG: parse_scoring_setting - Set multiplier_column={params['multiplier_column']}, multiplier={params['multiplier']}")
        
        # 解析普通键值对
        pairs = re.findall(r'(\w+)\s*=\s*([^,\[]+)', clean_str)
        
        for key, val in pairs:
            key = key.strip().lower()
            val = val.strip()
            
            if key == 'max_weight':
                params['max_weight'] = float(val)
            elif key == 'min_weight':
                params['min_weight'] = float(val)
            elif key == 'decay_rate':
                params['decay_rate'] = float(val)
            elif key == 'reverse':
                params['reverse'] = val.lower() in ('true', 'yes', '1', 'on')
            elif key == 'defect_pct':
                params['defect_pct'] = val.lower() in ('true', 'yes', '1', 'on')
            elif key == 'multiplier' and not params['multiplier_column']:
                # 只有在没有解析到列表格式时才设置普通数值
                params['multiplier'] = float(val)
            elif key == 'use_ring_pct':
                params['use_ring_pct'] = val.lower() in ('true', 'yes', '1', 'on')
            elif key == 'roi_coefficient':
                params['roi_coefficient'] = float(val)
            elif key == 'use_new_score':
                params['use_new_score'] = val.lower() in ('true', 'yes', '1', 'on')
            elif key == 'new_score':
                params['use_new_score'] = val.lower() in ('true', 'yes', '1', 'on')
    except Exception as e:
        print(f"Warning: Failed to parse Scoring Setting '{setting_str}': {e}")
    
    return params


def _normalize_map(dist_map):
    if dist_map is None or dist_map.size == 0:
        return None
    a_min = float(np.min(dist_map))
    a_max = float(np.max(dist_map))
    if a_max - a_min <= 0:
        return np.zeros_like(dist_map, dtype=np.float32)
    if dinov3_utils is not None and hasattr(dinov3_utils, "min_max_norm"):
        return dinov3_utils.min_max_norm(dist_map)
    return (dist_map - a_min) / (a_max - a_min)


def _resolve_distance_csv(reference_dir, base_candidates):
    if not reference_dir:
        return None
    for base in base_candidates:
        if not base:
            continue
        cand = os.path.join(reference_dir, f"{base}_anomaly_distance_raw.csv")
        if os.path.exists(cand):
            return cand
    return None


def _heatmap_from_norm(norm_map):
    norm_map = np.clip(norm_map, 0.0, 1.0)
    return cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)


def generate_dino_distance_overlay(dut_bgr, mask_filtered, reference_dir, base_name, path_to_process=None):
    if dut_bgr is None or mask_filtered is None:
        return None
    if reference_dir is None:
        return None

    base_candidates = []
    if base_name:
        base_candidates.append(base_name)
    if path_to_process:
        base_candidates.append(os.path.basename(path_to_process).split('.')[0])

    csv_path = _resolve_distance_csv(reference_dir, base_candidates)
    if not csv_path:
        return None

    try:
        dist_map = np.loadtxt(csv_path, delimiter=",")
    except Exception:
        return None

    if mask_filtered.dtype == bool:
        mask = mask_filtered
    else:
        mask = mask_filtered > 0

    norm_map = _normalize_map(dist_map)
    if norm_map is None:
        return None
    heatmap = _heatmap_from_norm(norm_map)
    target_h, target_w = heatmap.shape[:2]
    if mask.shape[:2] != dut_bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (dut_bgr.shape[1], dut_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if mask.shape[:2] != (target_h, target_w) or dut_bgr.shape[:2] != (target_h, target_w):
        gray = cv2.cvtColor(dut_bgr, cv2.COLOR_BGR2GRAY) if dut_bgr.ndim == 3 else dut_bgr
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        if w > 0 and h > 0:
            dut_crop = dut_bgr[y:y + h, x:x + w]
            mask_crop = mask[y:y + h, x:x + w]
            if dut_crop.shape[:2] != (target_h, target_w):
                dut_crop = cv2.resize(dut_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if mask_crop.shape[:2] != (target_h, target_w):
                mask_crop = cv2.resize(mask_crop.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
            dut_bgr = dut_crop
            mask = mask_crop
        else:
            dut_bgr = cv2.resize(dut_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0

    base = (dut_bgr.astype(np.float32) * 0.5).astype(np.uint8)
    overlay = base.copy()
    overlay[mask] = heatmap[mask]

    os.makedirs(reference_dir, exist_ok=True)
    out_path = os.path.join(reference_dir, f"{base_name}_dino_overaly.png")
    cv2.imwrite(out_path, overlay)
    return out_path


def calculate_local_contrast_coefficient(image, mask, kernel_size=11, mode='tophat', 
                                         local_window_size=21, contrast_metric='michelson',
                                         bins=None, return_full_stats=False):
    """
    计算Morph Top-Hat/Black-Hat区域的局部对比度系数 (Local Contrast Coefficient)
    
    该函数量化Morph Top-Hat/Black-Hat检测到的区域与周围局部背景的对比度差异，
    类似于Color Space Binning对HSV的分箱统计，但专门针对形态学操作后的局部对比度。
    
    参数:
        image: 输入图像 (H, W, 3) BGR或(H, W)灰度图
        mask: 二值掩码，True/255表示缺陷区域
        kernel_size: Morph Top-Hat/Black-Hat的结构元素大小
        mode: 'tophat'(亮区域) 或 'blackhat'(暗区域)
        local_window_size: 局部窗口大小，用于计算背景参考
        contrast_metric: 对比度度量方法
            - 'michelson': Michelson对比度 (Imax - Imin) / (Imax + Imin)
            - 'rms': RMS对比度 (标准差/均值)
            - 'weber': Weber对比度 (ΔI / I_background)
            - 'cv': 变异系数 (标准差/均值)
        bins: 分箱边界列表，如[0, 50, 100, 150, 200, 255]
              如果为None，则自动生成5个等宽分箱
        return_full_stats: 是否返回完整的统计信息
    
    返回:
        dict: 包含以下键值:
            - 'contrast_score': 总体对比度分数 (0-1，越高表示对比度越强)
            - 'bin_distribution': 各分箱的像素分布(百分比)
            - 'mean_contrast': 平均对比度
            - 'max_contrast': 最大对比度
            - 'contrast_map': 对比度热力图 (如果return_full_stats=True)
            - 'morph_response': Morph操作响应图 (如果return_full_stats=True)
    """
    import cv2
    import numpy as np
    
    if image is None or mask is None:
        return None
    
    # 确保掩码是布尔类型
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 确保kernel_size为奇数
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 应用Morphological操作
    if mode.lower() in ['tophat', 'top-hat', 'bright']:
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        mode_name = 'TopHat'
    else:
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mode_name = 'BlackHat'
    
    # 确保local_window_size为奇数
    if local_window_size < 3:
        local_window_size = 3
    if local_window_size % 2 == 0:
        local_window_size += 1
    
    # 计算局部背景均值 (使用morphological opening/closing估计背景)
    if mode.lower() in ['tophat', 'top-hat', 'bright']:
        # 对于TopHat，opening估计背景
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    else:
        # 对于BlackHat，closing估计背景
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 计算局部对比度
    contrast_map = np.zeros_like(gray, dtype=np.float32)
    
    if contrast_metric == 'michelson':
        # Michelson对比度: (I_defect - I_bg) / (I_defect + I_bg + eps)
        # 对于TopHat: morph_res = I - I_bg，所以 I = morph_res + I_bg
        eps = 1e-6
        I_defect = morph_res.astype(np.float32) + background.astype(np.float32)
        I_bg = background.astype(np.float32)
        contrast_map = np.abs(I_defect - I_bg) / (I_defect + I_bg + eps)
        
    elif contrast_metric == 'weber':
        # Weber对比度: ΔI / I_bg
        eps = 1e-6
        contrast_map = morph_res.astype(np.float32) / (background.astype(np.float32) + eps)
        
    elif contrast_metric == 'rms':
        # RMS对比度: 使用局部标准差
        # 计算局部均值
        local_mean = cv2.blur(gray.astype(np.float32), (local_window_size, local_window_size))
        # 计算局部平方均值
        local_sq_mean = cv2.blur((gray.astype(np.float32))**2, (local_window_size, local_window_size))
        # 局部标准差
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        # RMS对比度
        eps = 1e-6
        contrast_map = local_std / (local_mean + eps)
        
    elif contrast_metric == 'cv':
        # 变异系数: 标准差/均值
        local_mean = cv2.blur(gray.astype(np.float32), (local_window_size, local_window_size))
        local_sq_mean = cv2.blur((gray.astype(np.float32))**2, (local_window_size, local_window_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        eps = 1e-6
        contrast_map = local_std / (local_mean + eps)
        
    else:
        # 默认使用归一化的morph响应作为对比度
        morph_max = np.max(morph_res)
        if morph_max > 0:
            contrast_map = morph_res.astype(np.float32) / morph_max
        else:
            contrast_map = morph_res.astype(np.float32)
    
    # 只考虑掩码区域内的对比度
    masked_contrast = contrast_map[mask_bool]
    
    if len(masked_contrast) == 0:
        return {
            'contrast_score': 0.0,
            'bin_distribution': {},
            'mean_contrast': 0.0,
            'max_contrast': 0.0,
            'std_contrast': 0.0
        }
    
    # 计算总体对比度分数 (使用掩码区域内对比度的均值)
    mean_contrast = float(np.mean(masked_contrast))
    max_contrast = float(np.max(masked_contrast))
    std_contrast = float(np.std(masked_contrast))
    
    # 对比度分数: 结合均值和标准差，标准化到0-1
    contrast_score = min(1.0, mean_contrast + 0.5 * std_contrast)
    
    # 分箱统计
    if bins is None:
        # 自动生成5个等宽分箱
        bins = [0, 51, 102, 153, 204, 255]
    
    bin_distribution = {}
    total_pixels = len(masked_contrast)
    
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]
        
        # 对于最后一箱，包含上限
        if i == len(bins) - 2:
            count = np.sum((masked_contrast >= low) & (masked_contrast <= high))
        else:
            count = np.sum((masked_contrast >= low) & (masked_contrast < high))
        
        percentage = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0
        bin_distribution[f"LCC[{low}-{high}]"] = float(percentage)
    
    result = {
        'contrast_score': contrast_score,
        'bin_distribution': bin_distribution,
        'mean_contrast': mean_contrast,
        'max_contrast': max_contrast,
        'std_contrast': std_contrast,
        'total_pixels': total_pixels,
        'mode': mode_name,
        'contrast_metric': contrast_metric,
        'kernel_size': kernel_size
    }
    
    if return_full_stats:
        result['contrast_map'] = contrast_map
        result['morph_response'] = morph_res
        result['background'] = background
    
    return result


def calculate_morph_contrast_bins(image, mask, kernel_size=11, mode='tophat',
                                  bins=None, use_adaptive_bins=True):
    """
    计算Morph Top-Hat/Black-Hat的对比度分箱统计 (类似于Color Space Binning)
    
    该函数提供类似于HSV Color Space Binning的接口，用于Morphological操作的对比度分析。
    
    参数:
        image: 输入图像 (H, W, 3) BGR或(H, W)灰度图
        mask: 二值掩码，True/255表示缺陷区域
        kernel_size: Morph Top-Hat/Black-Hat的结构元素大小
        mode: 'tophat'(亮区域) 或 'blackhat'(暗区域)
        bins: 分箱边界列表，如[0, 50, 100, 150, 200, 255]
              如果为None且use_adaptive_bins=True，则根据数据自适应生成
        use_adaptive_bins: 是否根据morph响应自适应生成分箱
    
    返回:
        dict: 包含各分箱的像素数量和百分比，格式与Color Space Measurement兼容
    """
    import cv2
    import numpy as np
    
    if image is None or mask is None:
        return {}
    
    # 确保掩码是布尔类型
    if mask.dtype != bool:
        mask_bool = mask > 0
    else:
        mask_bool = mask
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 确保kernel_size为奇数
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # 应用Morphological操作
    if mode.lower() in ['tophat', 'top-hat', 'bright']:
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        mode_name = 'TopHat'
    else:
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mode_name = 'BlackHat'
    
    # 提取掩码区域内的morph响应值
    masked_morph = morph_res[mask_bool]
    
    if len(masked_morph) == 0:
        return {}
    
    # 自适应分箱或固定分箱
    if bins is None:
        if use_adaptive_bins:
            # 根据数据范围自适应生成5个分箱
            min_val = float(np.min(masked_morph))
            max_val = float(np.max(masked_morph))
            if max_val > min_val:
                step = (max_val - min_val) / 5
                bins = [min_val + i * step for i in range(6)]
            else:
                bins = [0, 51, 102, 153, 204, 255]
        else:
            bins = [0, 51, 102, 153, 204, 255]
    
    # 计算分箱统计
    results = {}
    total_pixels = len(masked_morph)
    
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]
        
        # 对于最后一箱，包含上限
        if i == len(bins) - 2:
            count = np.sum((masked_morph >= low) & (masked_morph <= high))
        else:
            count = np.sum((masked_morph >= low) & (masked_morph < high))
        
        percentage = (count / total_pixels * 100.0) if total_pixels > 0 else 0.0
        results[f"Morph_{mode_name}[{low:.1f}-{high:.1f}]"] = percentage
    
    # 添加统计信息
    results[f"Morph_{mode_name}_Mean"] = float(np.mean(masked_morph))
    results[f"Morph_{mode_name}_Max"] = float(np.max(masked_morph))
    results[f"Morph_{mode_name}_Std"] = float(np.std(masked_morph))
    results[f"Morph_{mode_name}_TotalPixels"] = total_pixels
    
    return results


def score_morph_contrast(image, mask, kernel_size=11, mode='tophat',
                         weight_high_contrast=1.0, decay_rate=0.5):
    """
    对Morph Top-Hat/Black-Hat区域进行加权评分
    
    类似于Color Space Binning的加权评分，对高对比度区域给予更高权重。
    
    参数:
        image: 输入图像
        mask: 缺陷区域掩码
        kernel_size: Morph结构元素大小
        mode: 'tophat' 或 'blackhat'
        weight_high_contrast: 高对比度区域的权重
        decay_rate: 权重衰减率
    
    返回:
        float: 加权评分 (0-1)
    """
    results = calculate_morph_contrast_bins(image, mask, kernel_size, mode)
    
    if not results:
        return 0.0
    
    # 提取分箱值
    bin_keys = [k for k in results.keys() if k.startswith(f"Morph_{mode.capitalize()}[")]
    if not bin_keys:
        return 0.0
    
    # 按分箱边界排序
    def extract_range(key):
        import re
        match = re.search(r'\[(.+)-(.+)\]', key)
        if match:
            return float(match.group(1))
        return 0
    
    bin_keys.sort(key=extract_range)
    
    # 获取分箱值
    bin_values = [results[k] for k in bin_keys]
    
    # 生成指数衰减权重 (高对比度分箱权重更高)
    n_bins = len(bin_values)
    weights = exponential_decay_weights(n_bins, max_weight=weight_high_contrast, 
                                        decay_rate=decay_rate, reverse=False)
    
    # 计算加权分数
    weighted_sum = sum(v * w for v, w in zip(bin_values, weights))
    total_weight = sum(weights)
    
    if total_weight > 0:
        score = weighted_sum / total_weight / 100.0  # 归一化到0-1
    else:
        score = 0.0
    
    return min(1.0, score)


def morph_contrast_binning(original_img, morph_mask, kernel_size=11, n_bins=5, threshold=0, 
                           mode='tophat', denoise=0):
    """
    对 Morph Top-Hat/Black-Hat 找到的缺陷区域进行局部对比度分箱
    
    类似于 Color Space Binning，但针对 Morphological 响应值进行分箱
    
    参数:
        original_img: 原始图像 (RGB 或 BGR)
        morph_mask: Morph Top-Hat 找到的缺陷 mask (bool 或 uint8)
        kernel_size: Morphological 操作的核大小
        n_bins: 分箱数量（默认 5，对应 S1-S5）
        threshold: 阈值，低于此值的响应设为 0
        mode: 'tophat' (亮缺陷) 或 'blackhat' (暗缺陷)
        denoise: 高斯模糊核大小 (0 表示不降噪)
    
    返回:
        bin_values: 各分箱的像素数量列表 [bin1_count, bin2_count, ..., bin5_count]
        bin_percentages: 各分箱的百分比列表
        contrast_map: 原始对比度响应图 (可用于可视化)
        bin_map: 每个像素所属的分箱索引图
    """
    import numpy as np
    import cv2
    
    if original_img is None or morph_mask is None:
        return None, None, None, None
    
    # 确保 mask 是 bool 类型
    if morph_mask.dtype != bool:
        mask = morph_mask > 0
    else:
        mask = morph_mask
    
    # 如果没有缺陷区域，返回全 0
    if not np.any(mask):
        return [0] * n_bins, [0.0] * n_bins, np.zeros_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.uint8)
    
    # 确保图像为灰度图
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_img.copy()
    
    # 降噪处理
    if denoise > 0:
        k_denoise = denoise * 2 + 1
        gray = cv2.GaussianBlur(gray, (k_denoise, k_denoise), 0)
    
    # 确保核大小为奇数
    k_size = max(3, kernel_size)
    if k_size % 2 == 0:
        k_size += 1
    
    # 执行 Morphological 操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    
    if mode.lower() == 'blackhat':
        # Black-Hat: Closing - Image -> 暗区域
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        # Top-Hat: Image - Opening -> 亮区域
        morph_res = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 应用阈值
    if threshold > 0:
        morph_res = np.where(morph_res >= threshold, morph_res, 0)
    
    # 只保留 mask 区域内的响应值
    contrast_map = morph_res.astype(np.float32)
    contrast_map[~mask] = 0
    
    # 获取 mask 区域内的响应值
    masked_values = contrast_map[mask]
    
    if len(masked_values) == 0:
        return [0] * n_bins, [0.0] * n_bins, contrast_map, np.zeros_like(mask, dtype=np.uint8)
    
    # 计算分箱边界（基于 mask 区域内的实际响应值范围）
    min_val = float(np.min(masked_values))
    max_val = float(np.max(masked_values))
    
    # 如果所有值相同，直接返回
    if max_val <= min_val:
        bin_values = [0] * n_bins
        bin_values[0] = int(np.sum(mask))
        total = sum(bin_values)
        bin_percentages = [100.0 * v / total if total > 0 else 0.0 for v in bin_values]
        bin_map = np.zeros_like(mask, dtype=np.uint8)
        bin_map[mask] = 1  # 全部放入第一个 bin
        return bin_values, bin_percentages, contrast_map, bin_map
    
    # 创建分箱边界（等宽分箱）
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # 对每个像素进行分箱
    bin_map = np.zeros_like(mask, dtype=np.uint8)
    bin_values = [0] * n_bins
    
    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        
        # 最后一个 bin 包含最大值
        if i == n_bins - 1:
            in_bin = (contrast_map >= low) & (contrast_map <= high) & mask
        else:
            in_bin = (contrast_map >= low) & (contrast_map < high) & mask
        
        bin_map[in_bin] = i + 1  # 1-indexed
        bin_values[i] = int(np.sum(in_bin))
    
    # 计算百分比
    total_pixels = sum(bin_values)
    bin_percentages = [100.0 * v / total_pixels if total_pixels > 0 else 0.0 for v in bin_values]
    
    print(f"DEBUG: Morph Contrast Binning - Range: [{min_val:.2f}, {max_val:.2f}], Bins: {bin_values}, Percentages: {[f'{p:.2f}%' for p in bin_percentages]}")
    
    return bin_values, bin_percentages, contrast_map, bin_map


def visualize_morph_bins(original_img, bin_map, color_map='jet'):
    """
    可视化 Morph Contrast Binning 结果
    
    参数:
        original_img: 原始图像
        bin_map: 分箱索引图 (来自 morph_contrast_binning)
        color_map: OpenCV colormap 名称
    
    返回:
        visualization: 彩色可视化图像
    """
    import numpy as np
    import cv2
    
    if bin_map is None or original_img is None:
        return None
    
    # 获取 colormap
    cmap_dict = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'viridis': cv2.COLORMAP_VIRIDIS
    }
    cmap = cmap_dict.get(color_map.lower(), cv2.COLORMAP_JET)
    
    # 归一化 bin_map 到 0-255
    max_bin = np.max(bin_map)
    if max_bin > 0:
        normalized = (bin_map.astype(np.float32) / max_bin * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(bin_map, dtype=np.uint8)
    
    # 应用 colormap
    colored = cv2.applyColorMap(normalized, cmap)
    
    # 只在有分箱的区域显示颜色
    mask = bin_map > 0
    visualization = original_img.copy()
    if len(visualization.shape) == 2:
        visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2RGB)
    elif visualization.shape[2] == 3 and isinstance(visualization[0,0,0], np.uint8):
        # 假设是 RGB，转换为 BGR 用于 OpenCV
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    visualization[mask] = colored[mask]
    
    return visualization
