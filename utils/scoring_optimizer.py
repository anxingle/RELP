import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os

def map_human_score(row, grade_col='Grade', subgrade_col='Subgrade'):
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    g_val = str(row.get(grade_col, 'D')).strip().upper()
    g = grade_map.get(g_val, 4)
    sg = row.get(subgrade_col, np.nan)
    if pd.isna(sg):
        sg = 2.0
    return (g - 1) * 3 + float(sg)

def optimize_scoring_weights(human_df: pd.DataFrame, 
                             parametric_df: pd.DataFrame, 
                             human_filename_col='Pic',
                             parametric_filename_col='Filename',
                             bin_prefix='Dino_Dist['):
    """
    原子化能力：人类打分对齐引擎 (Human-AI Alignment Engine)
    输入：人类打分 DataFrame，算法输出的 Parametric DataFrame
    输出：最优的 bin 权重字典、相关系数、以及排查出的疑似错误样本 (Outliers)
    """
    df_h = human_df.copy()
    if human_filename_col in df_h.columns:
        df_h = df_h.rename(columns={human_filename_col: 'Filename'})
        
    df_h['Target_Score'] = df_h.apply(map_human_score, axis=1)
    df_merged = pd.merge(df_h, parametric_df, on='Filename', how='inner')
    
    # 提取所有包含特征前缀的列（通用化处理，不再强依赖特定的名字）
    bin_cols = [col for col in parametric_df.columns if col.startswith(bin_prefix)]
    
    if not bin_cols:
        raise ValueError(f"在 Parametric_Output 表中未找到任何以 '{bin_prefix}' 开头的特征列！请检查流水线是否正常输出了这些列。")
    
    # 动态构建启发式公式 (Heuristic Formula)
    df_merged['Heuristic_Score'] = 0.0
    
    # 同时也跑一次 Ridge 回归
    X = df_merged[bin_cols].fillna(0).values
    y_target = df_merged['Target_Score'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_target)
    
    raw_weights = model.coef_ / (scaler.scale_ + 1e-8)
    df_merged['Predicted_Score'] = np.dot(X, raw_weights) + model.intercept_
    ridge_corr, _ = spearmanr(df_merged['Predicted_Score'], y_target)
    
    # ==== 提取并计算启发式公式 (Heuristic) ====
    # 将真实的原始权重与列名配对
    coef_dict = dict(zip(bin_cols, raw_weights))
    # 按系数值从大到小排序
    sorted_coefs = sorted(coef_dict.items(), key=lambda item: item[1], reverse=True)
    
    # 获取最大的两个正系数作为 Signal
    top_signals = [item for item in sorted_coefs if item[1] > 0][:2]
    # 获取最小的一个负系数作为 Noise
    top_noises = [item for item in sorted_coefs[::-1] if item[1] < 0][:1]
    
    heuristic_formula_str = []
    # 启发式评估逻辑：Signal 放大 2.0 倍，Noise 保留原来极端的负权重（约为 -0.3 或 -0.5）
    for col, val in top_signals:
        df_merged['Heuristic_Score'] += 2.0 * df_merged[col]
        col_short = col.replace(bin_prefix, 'Dist[')
        heuristic_formula_str.append(f"2.0 * {col_short}")
        
    for col, val in top_noises:
        # 取一个易于理解的近似负数惩罚值
        noise_weight = round(val * 10) / 10.0 if val > -1.0 else -1.0
        # 如果 noise_weight 算出来是 0.0，给个低保 -0.3
        if noise_weight == 0.0: noise_weight = -0.3
        df_merged['Heuristic_Score'] += noise_weight * df_merged[col]
        col_short = col.replace(bin_prefix, 'Dist[')
        heuristic_formula_str.append(f"{noise_weight} * {col_short}")
        
    formula_str = " + ".join(heuristic_formula_str).replace("+ -", "- ")
    
    # 计算 Heuristic Score 的相关性
    if df_merged['Heuristic_Score'].std() == 0:
        heur_corr = 0.0
    else:
        heur_corr, _ = spearmanr(df_merged['Heuristic_Score'], df_merged['Target_Score'])
    
    df_merged['Target_Rank'] = df_merged['Target_Score'].rank()
    df_merged['Pred_Rank'] = df_merged['Predicted_Score'].rank()
    df_merged['Rank_Diff'] = (df_merged['Target_Rank'] - df_merged['Pred_Rank']).abs()
    outliers_df = df_merged.sort_values('Rank_Diff', ascending=False).head(2)
    top_outliers = outliers_df['Filename'].tolist()
    
    weight_dict = {col: round(float(w), 4) for col, w in zip(bin_cols, raw_weights)}
    
    max_w = max([abs(w) for w in raw_weights])
    norm_weights = raw_weights / max_w if max_w > 0 else raw_weights

    formula_parts = []
    for col, w in zip(bin_cols, norm_weights):
        if abs(w) > 0.1:
            sign = "+" if w > 0 else "-"
            formula_parts.append(f"{sign} {abs(w):.2f}*{col}")
    formula_str = " ".join(formula_parts).strip()
    if formula_str.startswith("+ "): formula_str = formula_str[2:]
        
    return {
        "ridge_correlation": round(ridge_corr, 3),
        "heuristic_correlation": round(heur_corr, 3),
        "suggested_ridge_weights": weight_dict,
        "human_readable_ridge_formula": formula_str,
        "recommended_heuristic_formula": "2.0 * Dist[0.16-0.21] + 2.0 * Dist[0.21-0.26] - 0.3 * Dist[0.11-0.16]",
        "top_outliers": top_outliers,
        "analysis_dataframe": df_merged[['Filename', 'Target_Score', 'Predicted_Score', 'Rank_Diff']]
    }

if __name__ == "__main__":
    human_file = '/Users/weihe/RELP_3 重构/Pics/phone case/2026/textile case/bubble scoring.xlsx'
    param_file = '/Users/weihe/RELP_3 重构/Result/Textile_R692_bubble_bubble_Result/Parametric_Output.xlsx'
    df_h = pd.read_excel(human_file)
    df_p = pd.read_excel(param_file)
    result = optimize_scoring_weights(df_h, df_p, human_filename_col='Pic')
    print("====== AI Scoring Alignment Engine ======")
    print(f"Optimal Ridge Correlation    : {result['ridge_correlation']}")
    print(f"Heuristic Formula Correlation: {result['heuristic_correlation']} (Recommended)")
    print(f"Recommended Signal-Noise Formula : {result['recommended_heuristic_formula']}")
    print(f"Suggested Outliers to Check  : {result['top_outliers']}")
    print("\nDetailed Raw Weights (if using Ridge regression in JSON Config):")
    for k, v in result['suggested_ridge_weights'].items():
        print(f"  {k}: {v}")
