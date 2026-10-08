import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def folder_fetching(path, folder_hier):
    for i in range(folder_hier):
        path = os.path.dirname(path)
    return path


current_dir = Path.cwd()
config_dir = folder_fetching(current_dir, 1)
print('current_dir', config_dir)
Mainfolder_path = folder_fetching(current_dir, 4)
print('Mainfolder_path', Mainfolder_path)


def path_creator(main_path, subpath_list):
    abs_path = main_path
    for path in subpath_list:
        abs_path = os.path.join(abs_path, str(path))
    return abs_path


def get_config_size(project, detailed_cube_face):
    config_path = os.path.join(config_dir, "phone_case_config.xlsx")
    config_df = pd.read_excel(config_path, sheet_name='silicon_config')
    print("Config file columns:", config_df.columns.tolist())
    
    # 根据Project和Detailed_Cube_Face匹配配置
    config_row = config_df[(config_df['Project'] == project) & 
                          (config_df['Detailed_Cube_Face'] == detailed_cube_face)]
    
    if not config_row.empty:
        render_scale_ul = eval(config_row.iloc[0]['Rendering_Scale_UL'])
        render_scale_lr = eval(config_row.iloc[0]['Rendering_Scale_LR'])
        return render_scale_ul, render_scale_lr
    print(f"No matching row found for Project: {project}, Detailed_Cube_Face: {detailed_cube_face}")
    return None, None


def heatmap_creation(df, pic_saving_path, opacity=0.3):
    print("Input DataFrame columns:", df.columns.tolist())

    unique_combinations = df[['Project', 'Detailed_Cube_Face']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        project = row['Project']
        detailed_cube_face = row['Detailed_Cube_Face']

        render_scale_ul, render_scale_lr = get_config_size(project, detailed_cube_face)
        if render_scale_ul is None or render_scale_lr is None:
            print(f"Skipping {project} - {detailed_cube_face} due to missing configuration")
            continue

        render_img = get_render_img(project, detailed_cube_face)
        if render_img is None:
            print(f"Skipping {project} - {detailed_cube_face} due to missing render image")
            continue
            
        render_h, render_w = render_img.shape[:2]

        count_map = np.zeros((render_h, render_w), dtype=np.uint8)

        current_contours = df[(df['Project'] == project) & 
                            (df['Detailed_Cube_Face'] == detailed_cube_face)]['Contour'].to_numpy()
        
        for contour in current_contours:
            if isinstance(contour, str):
                try:
                    if contour.endswith(', 4.') or '[[' not in contour:
                        continue

                    points = []
                    parts = contour.strip('[]').split(']], [[')
                    for part in parts:
                        coords = part.strip('[]').split(',')
                        if len(coords) >= 2:
                            x = float(coords[0])
                            y = float(coords[1])
                            points.append([x, y])

                    contour_array = np.array(points)

                except Exception as e:
                    continue
            else:
                contour_array = np.array(contour)

            if len(contour_array) < 3:
                continue

            if contour_array.ndim == 1:
                contour_array = contour_array.reshape(-1, 2)


            contour_array[:, 0] = contour_array[:, 0] * (render_scale_lr[0] - render_scale_ul[0]) + render_scale_ul[0]
            contour_array[:, 1] = contour_array[:, 1] * (render_scale_lr[1] - render_scale_ul[1]) + render_scale_ul[1]
            contour_array = contour_array.astype(np.int32)

            mask = np.zeros((render_h, render_w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour_array], 1)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            count_map[mask == 1] += 1

        heatmap_normalized = cv2.normalize(count_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap = cv2.applyColorMap(np.uint8(heatmap_normalized), cv2.COLORMAP_JET)

        heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[count_map == 0, 3] = 0

        final_overlay = np.zeros((render_h, render_w, 4), dtype=np.uint8)
        render_rgba = cv2.cvtColor(render_img, cv2.COLOR_BGR2BGRA)
        final_overlay[:, :, :] = render_rgba

        alpha = heatmap_rgba[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        for c in range(3):
            final_overlay[:, :, c] = (1 - alpha[:, :, 0] * opacity) * render_rgba[:, :, c] + \
                                    (alpha[:, :, 0] * opacity) * heatmap_rgba[:, :, c]

        cv2.imshow('Heatmap Overlay', final_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        heatmap_saving_path = path_creator(pic_saving_path, ['Heatmap', project])
        os.makedirs(heatmap_saving_path, exist_ok=True)
        pic_saving_path = os.path.join(heatmap_saving_path, f"{detailed_cube_face}.jpg")
        cv2.imwrite(pic_saving_path, final_overlay)

        return final_overlay


def result_reading(main_path, failure_mode, col_list):
    result_path = os.path.join(main_path, failure_mode + "_Result")
    excel_path = os.path.join(result_path, 'Parametric_Output' + '.xlsx')
    print('excel_path', excel_path)
    res_df = pd.read_excel(excel_path)
    print("Parametric Output columns:", res_df.columns.tolist())
    reduced_res_df = res_df[col_list]
    return reduced_res_df, result_path


def get_render_img(project, detailed_cube_face):
    upper_path = folder_fetching(current_dir, 1)
    pic_path = os.path.join(upper_path, "phone_case_rendering", project)
    pic_full_path = os.path.join(pic_path, f"{detailed_cube_face}.jpg")
    print("Loading image from:", pic_full_path)
    render_pic = cv2.imread(pic_full_path)
    if render_pic is None:
        print(f"Error: Could not load image from {pic_full_path}")
        return None
    return render_pic


if __name__ == '__main__':
    # 定义要处理的组合列表
    target_combinations = [
        ['Project', 'Detailed_Cube_Face']
    ]
    
    res_DF, res_path = result_reading(Mainfolder_path, "Bumper_Crack", ['Contour', 'Project', 'Detailed_Cube_Face'])
    
    # 对每个组合执行heatmap_creation
    for combination in target_combinations:
        print(f"\nProcessing combination: {combination}")
        # 获取唯一的组合值
        unique_values = res_DF[combination].drop_duplicates()
        
        for _, row in unique_values.iterrows():
            # 创建过滤条件
            filter_conditions = {col: row[col] for col in combination}
            filtered_df = res_DF.copy()
            for col, value in filter_conditions.items():
                filtered_df = filtered_df[filtered_df[col] == value]
            
            # 为当前组合创建子文件夹
            subfolder_name = '_'.join(str(row[col]) for col in combination)
            current_saving_path = os.path.join(res_path, subfolder_name)
            os.makedirs(current_saving_path, exist_ok=True)
            
            # 执行heatmap_creation
            print(f"Creating heatmap for: {subfolder_name}")
            heatmap_creation(filtered_df, current_saving_path, opacity=0.5)
