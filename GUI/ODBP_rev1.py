import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove
import matplotlib as mpl
import copy
import math
import os
import glob
scaler = 0.985

# Multi-dimension template matching
def tem_matching(main_pic):
    template = cv2.imread("/Users/weihe/Downloads/ODBP cam.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 30, 50)
    # cv_show('template', template)
    (tH, tW) = template.shape[:2]
    image2 = copy.deepcopy(main_pic)
    gray = cv2.cvtColor(main_pic, cv2.COLOR_BGR2GRAY)
    found = None
    for scale in np.linspace(2, 0.1, 10):
        (H, W) = gray.shape[:2]
        resized = cv2.resize(gray, (int(W * scale), int(H * scale)))
        r = gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 30, 50)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r * 1), int(maxLoc[1] * r * 1))
    (endX, endY) = (int((maxLoc[0] + tW) * r * 1.15), int((maxLoc[1] + tH) * r * 1))
    cv2.rectangle(image2, (startX, startY), (endX, endY), (0, 0, 255), 2)
    camcut_out = np.array([[startX, startY], [endX, startY], [endX, endY], [startX, endY]])
    return camcut_out

def score(img, scaled_pic, mask_ROI, reduction_factor, substracted_area, str_caption):
    draw_img = img.copy()
    if reduction_factor ==1:
        total_area = cv2.contourArea(mask_ROI) * reduction_factor
    elif reduction_factor !=1:
        total_area = substracted_area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, [mask_ROI], -1, (127), thickness=cv2.FILLED)
    res_ROI = cv2.bitwise_and(scaled_pic, scaled_pic, mask=mask)
    # cv_show("scoring", res)
    contours, _ = cv2.findContours(res_ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
    print(len(filtered_contours))
    defect_area = 0
    for C in filtered_contours:
        defect_area += cv2.contourArea(C)
    # print("defect area", defect_area)
    contour_ROI = cv2.drawContours(draw_img, filtered_contours, -1, (0, 255, 0), 3)
    cv_show(str(str_caption), contour_ROI)
    return defect_area, total_area

def create_mask(ori_pic, contour, masks, color):
    draw_img = ori_pic.copy()
    pic_to_mask = masks.copy()
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(draw_img.shape[:2], dtype=np.uint8)
    # print('mask', mask.shape)
    mask = cv2.drawContours(pic_to_mask, [contour], -1, (color), thickness=cv2.FILLED)
    res = cv2.bitwise_and(pic_to_mask, pic_to_mask, mask=mask)
    cv_show("Masked_Mask", res)
    return res

def flip_pic(img, pic_processed, cam_contour):
    (iH, iW) = img.shape[:2]
    if np.max(cam_contour[:, 1]) > iH/4:
        pic_processed = cv2.flip(pic_processed, -1)
    return pic_processed

def pic_stats(img, contour_found):
    draw_img = img.copy()
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(draw_img.shape[:2], dtype=np.uint8)
    tl_x = np.min(contour_found[:, 0])
    tl_y = np.min(contour_found[:, 1])
    br_x = np.max(contour_found[:, 0])
    br_y = np.max(contour_found[:, 1])
    dut_w = br_x - tl_x
    dut_h = br_y - tl_y
    dut_info = np.int_(np.array([tl_x, tl_y, br_x, br_y, dut_w, dut_h]))
    # print('dut_info', dut_info)
    return dut_info

def Locate_keyTP(img, contour_found, cam_contour, x_min, x_max, y_min, y_max):
    flip_pic(img, contour_found, cam_contour)
    dut_stats = pic_stats(img, contour_found)
    rec_contour = create_rec(dut_stats, x_min, x_max, y_min, y_max)
    return rec_contour

def Locate_bumper(img, contour_found, key_contour, TP_contour, gap):
    dut_info = pic_stats(img, contour_found)
    bumper_contour = [
    [dut_info[0], dut_info[1]], [dut_info[0], dut_info[3]], [key_contour[3][0]-gap, dut_info[3]],
     [key_contour[3][0] - gap, key_contour[0][1]-gap], [key_contour[1][0] + gap, key_contour[0][1]-gap],
     [key_contour[1][0] + gap, dut_info[3]], [dut_info[2], dut_info[3]], [dut_info[2], dut_info[1]],
     [TP_contour[1][0] + gap, dut_info[1]], [TP_contour[2][0] + gap, TP_contour[2][1] + gap],
      [TP_contour[3][0] - gap, TP_contour[2][1] + gap], [TP_contour[0][0] - gap, dut_info[1]]
     ]
    bumper_contour = np.array(bumper_contour)
    return bumper_contour

def create_rec(dut_stats, x_min, x_max, y_min, y_max):
    tl = [dut_stats[0] + dut_stats[4] * x_min, dut_stats[1] + dut_stats[5] * y_min]
    tr = [dut_stats[0] + dut_stats[4] * x_max, dut_stats[1] + dut_stats[5] * y_min]
    bl = [dut_stats[0] + dut_stats[4] * x_min, dut_stats[1] + dut_stats[5] * y_max]
    br = [dut_stats[0] + dut_stats[4] * x_max, dut_stats[1] + dut_stats[5] * y_max]
    rec_created = np.int_(np.array([tl,tr, br, bl]))
    return rec_created

def draw_contour(img_BG, contour_draw, str_caption):
    draw_img = img_BG.copy()
    contours, _ = cv2.findContours(contour_draw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
    print(len(filtered_contours))
    contour_to_draw = cv2.drawContours(draw_img, filtered_contours, -1, (0, 255, 0), 3)
    cv_show(str(str_caption), contour_to_draw)
    return contour_to_draw

def contour_scale(arr, scaler):
    for i in range(arr.shape[-1]):
        col_mean = np.mean(arr[:, :, i])
        arr[:, :, i] = np.where(arr[:, :, i] > col_mean, np.round((arr[:, :, i] - col_mean)*scaler +col_mean),
                              np.round(col_mean - (col_mean - arr[:, :, i])*scaler))
    return(arr)

def convert_3d_to_2d(data):
    rows = len(data) * len(data[0])
    cols = len(data[0][0])
    flattened_data = [[0] * cols for _ in range(rows)]
    for i, dim1 in enumerate(data):
        for j, dim2 in enumerate(dim1):
            for k, value in enumerate(dim2):
                flattened_data[i * len(dim1) + j][k] = value
    return flattened_data

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    for i in range(rows):
        tmp = cols - len(imgArray[i])
        for j in range(tmp):
            img = np.zeros((imgArray[0][0].shape[0], imgArray[0][0].shape[1]), dtype='uint8')
            imgArray[i].append(img)

    if rows >= 2:
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]

    else:
        width = imgArray[0].shape[1]
        height = imgArray[0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def pic_labeling(path_to_process):
    pic_list = (glob.glob(path_to_process + "/*.jpeg"))
    print('list', pic_list)
    score_overall = pd.DataFrame()
    for pic in pic_list:
        ROI_val = pd.DataFrame()
        pic_name = os.path.basename(pic)
        pic_name = str(pic_name).split(".")[0]
        print('pic_name', pic_name)
        img=cv2.imread(pic)
        obj = remove(img)
        cv_show("BG revmoved", obj)
        img_adap = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

        # 应用掩码
        res_med=cv2.medianBlur(img_adap,49)
        res_out_med=cv2.medianBlur(img_adap,49)

        # 用滤波的方法处理图片。核心算法为灰度图-中值滤波图。
        # 屏幕内圈处理
        res_in_diff = cv2.addWeighted(img_adap, 2.2, res_med, -1.14, 0)
        # cv_show("Diff_in", res_in_diff)
        _, thres_in_low = cv2.threshold(res_in_diff, 40,127, cv2.THRESH_BINARY)
        _, thres_in_high = cv2.threshold(res_in_diff, 150,255, cv2.THRESH_BINARY)

        _, img_in = cv2.threshold(res_in_diff, 185,196, cv2.THRESH_BINARY)
        # cv_show('img_in', img_in)

        GaussianBlur_2=cv2.GaussianBlur(img_in,(5,5),1)
        # cv_show("GaussianBlur_2", GaussianBlur_2)
        medianBlur_in=cv2.medianBlur(GaussianBlur_2,5)
        # cv_show("medianBlur_in", medianBlur_in)

        kernel = np.ones((3,3), np.uint8)
        medianBlur_in = cv2.morphologyEx(medianBlur_in, cv2.MORPH_CLOSE, kernel, iterations=1)
        # cv_show('medianBlur_in', medianBlur_in)

        # 屏幕外圈处理
        res_out_diff = cv2.addWeighted(img_adap,  -1.3, res_out_med, 1.45, 0)
        _, img_out = cv2.threshold(res_out_diff, 8, 26, cv2.THRESH_BINARY_INV)
        # cv_show("img_out", img_out)

        GaussianBlur_out =cv2.GaussianBlur(img_out,(5,5),1)
        medianBlur_out=cv2.medianBlur(GaussianBlur_out,3)
        # cv_show("medianBlur_out", medianBlur_out)

        kernel = np.ones((3,3), np.uint8)
        res_combined = cv2.addWeighted(medianBlur_in,  1, medianBlur_out,5, 0)
        # res_combined = cv2.medianBlur(res_combined,3)
        _, res_combined = cv2.threshold(res_combined, 110, 255, cv2.THRESH_BINARY)

        draw_img = obj.copy()
        contours, _ = cv2.findContours(res_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
        print(len(filtered_contours))
        contour_order = sorted(filtered_contours, key=cv2.contourArea, reverse=True)  # 已轮廓区域面积进行排序
        pad_area = cv2.contourArea(contour_order[1])

        draw_img = img.copy()

        contour_2d = np.array(contour_order[1].tolist())
        mask = np.zeros(draw_img.shape[:2], dtype=np.uint8)
        # print(mask.shape)
        mask = cv2.drawContours(mask, [contour_2d], -1, (127), thickness=cv2.FILLED)
        res = cv2.bitwise_and(img, img, mask=mask)

        #Scale the mask and refine ROI
        draw_img = img.copy()
        contour_scaled = contour_scale(contour_2d, scaler)
        mask1 = np.zeros(draw_img.shape[:2], dtype=np.uint8)
        mask1 = cv2.drawContours(mask1, [contour_scaled], -1, (127), thickness=cv2.FILLED)
        res1 = cv2.bitwise_and(res_combined, res_combined, mask=mask1)
        res2 = res1.copy()

        draw_img = img.copy()
        contours, _ = cv2.findContours(res1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
        print(len(filtered_contours))
        contour_updated = cv2.drawContours(draw_img, filtered_contours, -1, (0, 255, 0), 3)
        cv_show("with_camera", contour_updated)

        cut_out = tem_matching(img)
        cam_cut = create_mask(img, cut_out, res1, 0)
        contour_nocam = draw_contour(img, cam_cut, "cam_cut")
        cv2.imwrite(path_to_process + "/labeled_pic/"+ pic_name+'_l.jpeg', contour_nocam)

        #Create masks for key, trackpad and bumper
        contour_iter1 = np.array(convert_3d_to_2d(contour_order[1]))
        # print('contour_order[1]', contour_iter1)
        key_mask = Locate_keyTP(img, contour_iter1, cam_cut, 0.05, 0.95, 0.35, 1)
        ROI_key = create_mask(img, key_mask, res1, 127)

        TP_mask = Locate_keyTP(img, contour_iter1, cam_cut, 0.25, 0.75, 0, 0.2)
        ROI_TP = create_mask(img, TP_mask, res1, 60)

        bumper_mask = Locate_bumper(img, contour_iter1, key_mask, TP_mask, 5)
        ROI_bumper = create_mask(img, bumper_mask, res1, 180)

        # Score ROIs
        key_ROI  = score(img, cam_cut, key_mask, 1, 0, "key area")
        TP_ROI  = score(img, cam_cut, TP_mask, 1, 0, "TP area")
        sub_area = pad_area - key_ROI[1]-TP_ROI[1]
        bumper_ROI  = score(img, cam_cut, bumper_mask, 0.99, sub_area, "bumper area")
        ROI_val = np.array([key_ROI, TP_ROI, bumper_ROI]).flatten()
        ROI_val = ROI_val.reshape(1,6)
        column2 = ["Key defect", "Key area", "TP defect", "TP area", "bumper defect", "bumper area"]
        df_info = pd.DataFrame()
        df_info['SN'] = [pic_name]
        # print('df_info', df_info)
        ROI_val = pd.DataFrame(ROI_val, columns=column2)
        ROI_val = pd.concat([df_info, ROI_val], axis=1)
        score_overall = pd.concat([score_overall, ROI_val], ignore_index=True)
        # print('score_overall', score_overall)
    return score_overall
# -------------
pic_path = '/Users/weihe/Downloads/ODBP'

if not os.path.exists(pic_path+"/labeled_pic/"):
    os.makedirs(pic_path+"/labeled_pic/")
if not os.path.exists(pic_path+"/scores/"):
    os.makedirs(pic_path + "/scores/")

ROI_scores = pic_labeling(pic_path)
ROI_scores.to_excel(pic_path + "/scores/"+ 'ODBP Scores.xlsx', index=False)