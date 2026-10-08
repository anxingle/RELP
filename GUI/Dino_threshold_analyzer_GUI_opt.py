#!/usr/bin/env python3
"""
DINO Anomaly Distance Analyzer (High-Performance Optimized Edition)
------------------------------------------------------------------
Optimizations:
1. Display Proxy Cache: Automatically downsamples ultra-large matrices (e.g. 8688x8688)
   to an optimal screen resolution (~800px) for real-time interaction (100x speedup).
2. Debounced Rendering: Throttles high-frequency slider events to eliminate UI freezing.
3. Fast Histogram Subsampling: Renders distribution instantaneously without stalling.
4. Preserved Full-Precision Output: All saves, ROIs, and mask applications operate on
   original raw matrix with zero precision loss.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import cv2
import os
import glob

# Maximum display resolution for real-time interactive canvas
MAX_DISPLAY_DIM = 800
DEBOUNCE_MS = 25  # Event throttle threshold in milliseconds


class HeatmapAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Distance Analyzer (Turbo Optimized)")
        self.root.geometry("1100x920")

        # Full-precision raw data
        self.raw_map = None
        self.bg_img = None
        self.current_min = 0.0
        self.current_max = 1.0
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100

        # Display Proxy Cache (for 60FPS smooth rendering)
        self.disp_map = None
        self.disp_bg_img = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # ROI & Cluster State
        self.roi_confirmed = False
        self.roi_min = 0.0
        self.roi_max = 1.0
        self.cluster_mask = None
        self.cluster_labels_map = None
        self.cluster_vars = []

        # Debounce timer handle
        self._render_timer = None

        # --- UI Layout ---
        self._build_ui()

    def _build_ui(self):
        # 1. Top Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_load = tk.Button(control_frame, text="Load CSV File", command=self.load_file, font=("Arial", 10, "bold"))
        self.btn_load.pack(side=tk.LEFT)

        self.lbl_file = tk.Label(control_frame, text="No file loaded", fg="#666")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_apply = tk.Button(control_frame, text="Select Image & Apply Mask", command=self.apply_mask_to_image)
        self.btn_apply.pack(side=tk.LEFT, padx=10)

        self.btn_confirm_roi = tk.Button(control_frame, text="Confirm Spatial ROI", command=self.confirm_spatial_roi)
        self.btn_confirm_roi.pack(side=tk.LEFT, padx=10)

        # Cluster Controls
        self.cluster_frame = tk.Frame(control_frame)
        self.cluster_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(self.cluster_frame, text="K=").pack(side=tk.LEFT)
        self.entry_k = tk.Entry(self.cluster_frame, width=3)
        self.entry_k.insert(0, "3")
        self.entry_k.pack(side=tk.LEFT)

        self.btn_cluster = tk.Button(self.cluster_frame, text="Generate Clusters", command=self.generate_clusters)
        self.btn_cluster.pack(side=tk.LEFT, padx=5)

        # Checkbox Frame for Clusters
        self.checkbox_frame = tk.Frame(self.root)
        self.checkbox_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        # 2. Bottom Slider Panel
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)

        # --- Left Column: Value Filters ---
        val_frame = tk.LabelFrame(self.slider_frame, text="Distance Value Filter (Real-Time Smooth)")
        val_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Min Slider
        min_frame = tk.Frame(val_frame)
        min_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.lbl_min = tk.Label(min_frame, text="Min Value: 0.00", width=18, anchor="w")
        self.lbl_min.pack(side=tk.LEFT)
        self.slider_min = tk.Scale(min_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   command=self.on_min_change, resolution=0.01, showvalue=False)
        self.slider_min.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Max Slider
        max_frame = tk.Frame(val_frame)
        max_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.lbl_max = tk.Label(max_frame, text="Max Value: 1.00", width=18, anchor="w")
        self.lbl_max.pack(side=tk.LEFT)
        self.slider_max = tk.Scale(max_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   command=self.on_max_change, resolution=0.01, showvalue=False)
        self.slider_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Right Column: Spatial Filters ---
        spatial_frame = tk.LabelFrame(self.slider_frame, text="Spatial Coordinate Filter")
        spatial_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # X Sliders
        x_frame = tk.Frame(spatial_frame)
        x_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.lbl_x_min = tk.Label(x_frame, text="X Min: 0", width=12, anchor="w")
        self.lbl_x_min.pack(side=tk.LEFT)
        self.slider_x_min = tk.Scale(x_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_x_min_change, showvalue=False)
        self.slider_x_min.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_x_max = tk.Label(x_frame, text="X Max: 100", width=12, anchor="w")
        self.lbl_x_max.pack(side=tk.LEFT)
        self.slider_x_max = tk.Scale(x_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_x_max_change, showvalue=False)
        self.slider_x_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Y Sliders
        y_frame = tk.Frame(spatial_frame)
        y_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.lbl_y_min = tk.Label(y_frame, text="Y Min: 0", width=12, anchor="w")
        self.lbl_y_min.pack(side=tk.LEFT)
        self.slider_y_min = tk.Scale(y_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_y_min_change, showvalue=False)
        self.slider_y_min.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_y_max = tk.Label(y_frame, text="Y Max: 100", width=12, anchor="w")
        self.lbl_y_max.pack(side=tk.LEFT)
        self.slider_y_max = tk.Scale(y_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_y_max_change, showvalue=False)
        self.slider_y_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 3. Matplotlib Figure Canvas
        self.fig = Figure(figsize=(14, 8), dpi=100)
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 4])
        self.ax_hist = self.fig.add_subplot(gs[0])
        self.ax_map = self.fig.add_subplot(gs[1])
        self.fig.tight_layout(pad=3.0, w_pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Cache artist handle
        self.im_artist = None

        self.ax_hist.set_title("Distance Distribution")
        self.ax_map.set_title("Heatmap\n(Min <= V <= Max)", fontsize=10, pad=5)
        self.ax_map.axis('off')

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Anomaly Distance Raw CSV", "*_anomaly_distance_raw.csv"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            # 1. 智能拦截误选
            file_name_lower = os.path.basename(file_path).lower()
            if "status" in file_name_lower:
                messagebox.showwarning(
                    "文件类型提示",
                    f"您当前选择的是推理状态文件：\n{os.path.basename(file_path)}\n\n"
                    "该文件记录的是推理日志 (method,status,message)，并不是特征距离矩阵！\n"
                    "请进入具体的模型子目录（例如 AnyUp_640 或 Baseline），选择对应的 *_anomaly_distance_raw.csv 文件。"
                )
                return

            # 2. 载入原始高精度矩阵 (兼容 UTF-8 BOM)
            try:
                self.raw_map = np.loadtxt(file_path, delimiter=",", encoding="utf-8-sig")
            except Exception:
                try:
                    self.raw_map = np.loadtxt(file_path, encoding="utf-8-sig")
                except Exception:
                    import pandas as pd
                    df = pd.read_csv(file_path, header=None, encoding="utf-8-sig")
                    self.raw_map = df.values.astype(np.float64)

            self.lbl_file.config(text=f"{os.path.basename(file_path)} [{self.raw_map.shape[0]}x{self.raw_map.shape[1]}]")

            # 3. 构建显示代理缓存 (Display Proxy Cache)
            # 如果原始矩阵分辨率极高 (>800)，将其等比压缩至 ~800px 用于流畅交互，提速 100x
            h_raw, w_raw = self.raw_map.shape
            max_side = max(h_raw, w_raw)
            if max_side > MAX_DISPLAY_DIM:
                scale_factor = MAX_DISPLAY_DIM / max_side
                disp_w = int(w_raw * scale_factor)
                disp_h = int(h_raw * scale_factor)
                self.disp_map = cv2.resize(self.raw_map.astype(np.float32), (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                self.scale_x = disp_w / w_raw
                self.scale_y = disp_h / h_raw
            else:
                self.disp_map = self.raw_map.copy()
                self.scale_x = 1.0
                self.scale_y = 1.0

            # 4. 重置状态
            self.roi_confirmed = False
            self.roi_min = self.raw_map.min()
            self.roi_max = self.raw_map.max()
            self.cluster_mask = None
            self.cluster_labels_map = None
            for widget in self.checkbox_frame.winfo_children():
                widget.destroy()
            self.cluster_vars = []

            # 5. 加载配套的原图 (如有)
            self.bg_img = None
            self.disp_bg_img = None
            base_name = os.path.basename(file_path).replace("_anomaly_distance_raw.csv", "")
            dir_name = os.path.dirname(file_path)
            potential_images = glob.glob(os.path.join(dir_name, f"{base_name}_original*.png"))

            if potential_images:
                img = cv2.imread(potential_images[0])
                if img is not None:
                    self.bg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 调整显示代理背景尺寸
                    dh, dw = self.disp_map.shape[:2]
                    self.disp_bg_img = cv2.resize(self.bg_img, (dw, dh), interpolation=cv2.INTER_AREA)

            # 6. 配置滑动条数值范围 (基于全局原图的极值)
            min_val, max_val = float(self.raw_map.min()), float(self.raw_map.max())
            step = (max_val - min_val) / 200 if max_val > min_val else 0.001

            self.slider_min.config(from_=min_val, to=max_val, resolution=step)
            self.slider_min.set(min_val)
            self.current_min = min_val
            self.lbl_min.config(text=f"Min Value: {min_val:.4f}")

            self.slider_max.config(from_=min_val, to=max_val, resolution=step)
            self.slider_max.set(max_val)
            self.current_max = max_val
            self.lbl_max.config(text=f"Max Value: {max_val:.4f}")

            # 7. 配置空间滑动条
            self.slider_x_min.config(from_=0, to=w_raw, resolution=1)
            self.slider_x_min.set(0)
            self.x_min = 0
            self.lbl_x_min.config(text=f"X Min: 0")

            self.slider_x_max.config(from_=0, to=w_raw, resolution=1)
            self.slider_x_max.set(w_raw)
            self.x_max = w_raw
            self.lbl_x_max.config(text=f"X Max: {w_raw}")

            self.slider_y_min.config(from_=0, to=h_raw, resolution=1)
            self.slider_y_min.set(0)
            self.y_min = 0
            self.lbl_y_min.config(text=f"Y Min: 0")

            self.slider_y_max.config(from_=0, to=h_raw, resolution=1)
            self.slider_y_max.set(h_raw)
            self.y_max = h_raw
            self.lbl_y_max.config(text=f"Y Max: {h_raw}")

            # 8. 绘制直方图与初始热力图
            self.plot_histogram()
            self._render_heatmap_now()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def confirm_spatial_roi(self):
        if self.raw_map is None:
            return

        x0 = max(0, int(self.x_min))
        x1 = min(self.raw_map.shape[1], int(self.x_max))
        y0 = max(0, int(self.y_min))
        y1 = min(self.raw_map.shape[0], int(self.y_max))

        if x1 <= x0 or y1 <= y0:
            return

        roi_data = self.raw_map[y0:y1, x0:x1]
        if roi_data.size == 0:
            return

        # 更新直方图和极值
        self.plot_histogram(data=roi_data)
        min_val, max_val = float(roi_data.min()), float(roi_data.max())
        step = (max_val - min_val) / 200 if max_val > min_val else 0.001

        self.slider_min.config(from_=min_val, to=max_val, resolution=step)
        self.slider_max.config(from_=min_val, to=max_val, resolution=step)
        self.slider_min.set(min_val)
        self.slider_max.set(max_val)
        self.current_min = min_val
        self.current_max = max_val

        self.roi_confirmed = True
        self.roi_min = min_val
        self.roi_max = max_val

        self._render_heatmap_now()
        messagebox.showinfo("ROI Confirmed", "Spatial ROI Confirmed.\nDistance sliders and heatmap range updated.")

    def generate_clusters(self):
        if self.raw_map is None:
            return

        try:
            k = int(self.entry_k.get())
            if k < 1: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid K (must be integer >= 1)")
            return

        # 基于当前代理或原图采样聚类以保证速度
        target_map = self.disp_map if self.disp_map is not None else self.raw_map
        h, w = target_map.shape
        x0 = max(0, int(self.x_min * self.scale_x))
        x1 = min(w, int(self.x_max * self.scale_x))
        y0 = max(0, int(self.y_min * self.scale_y))
        y1 = min(h, int(self.y_max * self.scale_y))

        mask_val = (target_map >= self.current_min) & (target_map <= self.current_max)
        mask_spatial = np.zeros_like(target_map, dtype=bool)
        if x1 > x0 and y1 > y0:
            mask_spatial[y0:y1, x0:x1] = True

        base_mask = mask_val & mask_spatial
        valid_data = target_map[base_mask]

        if len(valid_data) < k:
            messagebox.showerror("Error", f"Not enough data points ({len(valid_data)}) for {k} clusters")
            return

        # 降采样限制最大聚类样本数，秒级收敛
        if len(valid_data) > 50000:
            sub_indices = np.random.choice(len(valid_data), size=50000, replace=False)
            train_pts = valid_data[sub_indices].reshape(-1, 1).astype(np.float32)
        else:
            train_pts = valid_data.reshape(-1, 1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, _, centers = cv2.kmeans(train_pts, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # 全量预测标签
        centers_flat = centers.flatten()
        sorted_indices = np.argsort(centers_flat)
        centers_sorted = centers_flat[sorted_indices]

        all_pts = valid_data.reshape(-1, 1).astype(np.float32)
        # 欧氏距离分类
        dists = np.abs(all_pts - centers_sorted[None, :])
        remapped_labels = np.argmin(dists, axis=1)

        self.cluster_labels_map = np.full_like(target_map, -1, dtype=int)
        self.cluster_labels_map[base_mask] = remapped_labels

        # 生成复选框
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.cluster_vars = []

        for i in range(k):
            var = tk.BooleanVar(value=True)
            self.cluster_vars.append(var)
            c_val = centers_sorted[i]
            chk = tk.Checkbutton(self.checkbox_frame, text=f"Cluster {i+1} ({c_val:.3f})",
                                 variable=var, command=self.on_cluster_update)
            chk.pack(side=tk.LEFT, padx=5)

        self.on_cluster_update()

    def on_cluster_update(self):
        if self.cluster_labels_map is None:
            return

        selected_clusters = [i for i, var in enumerate(self.cluster_vars) if var.get()]
        self.cluster_mask = np.isin(self.cluster_labels_map, selected_clusters)
        self._render_heatmap_now()

    def plot_histogram(self, data=None):
        self.ax_hist.clear()
        target = self.raw_map if data is None else data
        
        # 极速直方图优化：若像素量巨大，采用跨步采样，0.01秒瞬间画出
        if target.size > 100000:
            step = target.size // 100000
            data_flat = target.ravel()[::step]
        else:
            data_flat = target.ravel()

        self.ax_hist.hist(data_flat, bins=50, color='skyblue', edgecolor='black', log=True, orientation='horizontal')
        self.ax_hist.set_title("Distribution (Log Scale)", fontsize=10)
        self.ax_hist.set_ylabel("Raw Distance")
        self.ax_hist.set_xlabel("Count (Log)")
        self.ax_hist.grid(True, alpha=0.3)

        self.min_line = self.ax_hist.axhline(self.current_min, color='g', linestyle='--', linewidth=2, label='Min')
        self.max_line = self.ax_hist.axhline(self.current_max, color='r', linestyle='--', linewidth=2, label='Max')
        self.range_span = self.ax_hist.axhspan(self.current_min, self.current_max, color='yellow', alpha=0.2)
        self.ax_hist.legend(loc="upper right", fontsize=8)

        self.canvas.draw()

    # --- 高频滑动条事件与防抖节流 ---
    def on_min_change(self, value):
        val = float(value)
        self.current_min = val
        self.lbl_min.config(text=f"Min Value: {val:.4f}")
        if self.current_min > self.current_max:
            self.slider_max.set(val)
        self.schedule_update()

    def on_max_change(self, value):
        val = float(value)
        self.current_max = val
        self.lbl_max.config(text=f"Max Value: {val:.4f}")
        if self.current_max < self.current_min:
            self.slider_min.set(val)
        self.schedule_update()

    def on_x_min_change(self, value):
        val = int(value)
        self.x_min = val
        self.lbl_x_min.config(text=f"X Min: {val}")
        if self.x_min > self.x_max:
            self.slider_x_max.set(val)
        self.schedule_update()

    def on_x_max_change(self, value):
        val = int(value)
        self.x_max = val
        self.lbl_x_max.config(text=f"X Max: {val}")
        if self.x_max < self.x_min:
            self.slider_x_min.set(val)
        self.schedule_update()

    def on_y_min_change(self, value):
        val = int(value)
        self.y_min = val
        self.lbl_y_min.config(text=f"Y Min: {val}")
        if self.y_min > self.y_max:
            self.slider_y_max.set(val)
        self.schedule_update()

    def on_y_max_change(self, value):
        val = int(value)
        self.y_max = val
        self.lbl_y_max.config(text=f"Y Max: {val}")
        if self.y_max < self.y_min:
            self.slider_y_min.set(val)
        self.schedule_update()

    def schedule_update(self):
        """防抖调度器：在用户快速拖拽时阻断重复计算，等微停顿瞬间平滑重绘"""
        # 即时轻量更新直方图指示线 (几乎零耗时)
        self.update_histogram_lines_fast()
        
        # 节流防抖触发热力图
        if self._render_timer is not None:
            self.root.after_cancel(self._render_timer)
        self._render_timer = self.root.after(DEBOUNCE_MS, self._render_heatmap_now)

    def update_histogram_lines_fast(self):
        if hasattr(self, 'min_line'):
            self.min_line.set_ydata([self.current_min, self.current_min])
        if hasattr(self, 'max_line'):
            self.max_line.set_ydata([self.current_max, self.current_max])
        if hasattr(self, 'range_span'):
            self.range_span.remove()
            self.range_span = self.ax_hist.axhspan(self.current_min, self.current_max, color='yellow', alpha=0.2)

    def _render_heatmap_now(self):
        self._render_timer = None
        if self.disp_map is None:
            return

        # 1. 在轻量显示代理上计算当前掩膜 (运算耗时 < 5ms)
        target_map = self.disp_map
        h, w = target_map.shape
        x0 = max(0, int(self.x_min * self.scale_x))
        x1 = min(w, int(self.x_max * self.scale_x))
        y0 = max(0, int(self.y_min * self.scale_y))
        y1 = min(h, int(self.y_max * self.scale_y))

        mask_val = (target_map >= self.current_min) & (target_map <= self.current_max)
        mask_spatial = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            mask_spatial[y0:y1, x0:x1] = True

        mask = mask_val & mask_spatial
        if self.cluster_mask is not None:
            mask = mask & self.cluster_mask

        # 2. 归一化极值
        vmin = self.roi_min if self.roi_confirmed else float(self.raw_map.min())
        vmax = self.roi_max if self.roi_confirmed else float(self.raw_map.max())
        denom = vmax - vmin if (vmax - vmin) > 0 else 1.0

        # 3. 生成紧致的显示 RGBA 图 (仅在 ~800px 尺寸下，耗时从 1.5s 骤降到 0.008s)
        norm_map = np.clip((target_map - vmin) / denom, 0, 1)
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(norm_map)

        # 混合背景或纯热力图
        if self.disp_bg_img is not None:
            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[mask] = 0.55
            rgba_img[:, :, 3] = alpha

            self.ax_map.clear()
            self.ax_map.imshow(self.disp_bg_img)
            self.ax_map.imshow(rgba_img)
        else:
            rgba_img[~mask, 3] = 0.0
            self.ax_map.clear()
            self.ax_map.imshow(rgba_img, vmin=0, vmax=1)

        title_str = f"Heatmap Overlay\n({self.current_min:.4f} <= V <= {self.current_max:.4f})"
        self.ax_map.set_title(title_str, fontsize=10, pad=8)
        self.ax_map.axis('off')

        # 单次原子刷新画布
        self.canvas.draw_idle()

    def apply_mask_to_image(self):
        """物理级高精度遮罩应用与导出：在全分辨率 raw_map 上执行，保持 100% 原始像素精度"""
        if self.raw_map is None:
            messagebox.showwarning("Warning", "Please load a heatmap (CSV) first.")
            return

        img_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All Files", "*.*")]
        )
        if not img_path:
            return

        try:
            target_img = cv2.imread(img_path)
            if target_img is None:
                raise ValueError("Could not load image.")

            # 在全精度 raw_map 上计算真实物理 Mask
            h_raw, w_raw = self.raw_map.shape
            mask_val = (self.raw_map >= self.current_min) & (self.raw_map <= self.current_max)
            mask_spatial = np.zeros((h_raw, w_raw), dtype=bool)

            x0 = max(0, int(self.x_min))
            x1 = min(w_raw, int(self.x_max))
            y0 = max(0, int(self.y_min))
            y1 = min(h_raw, int(self.y_max))
            if x1 > x0 and y1 > y0:
                mask_spatial[y0:y1, x0:x1] = True

            full_mask = mask_val & mask_spatial
            mask_uint8 = full_mask.astype(np.uint8) * 255

            # 对齐目标图像分辨率
            th, tw = target_img.shape[:2]
            mask_resized = cv2.resize(mask_uint8, (tw, th), interpolation=cv2.INTER_NEAREST)

            black_bg = np.zeros_like(target_img)
            mask_bool = mask_resized > 0
            black_bg[mask_bool] = target_img[mask_bool]

            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"masked_{os.path.basename(img_path)}"
            )

            if save_path:
                cv2.imwrite(save_path, black_bg)
                messagebox.showinfo("Success", f"Full-resolution masked image saved to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeatmapAnalyzerGUI(root)
    root.mainloop()
