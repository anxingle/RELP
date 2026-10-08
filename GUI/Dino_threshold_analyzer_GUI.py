#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import os
import glob

class HeatmapAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Distance Analyzer")
        self.root.geometry("1000x900")

        # Data variables
        self.raw_map = None
        self.bg_img = None
        self.normalized_map = None
        self.current_min = 0.0
        self.current_max = 1.0
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 100
        
        # New variables for ROI and Clustering
        self.roi_confirmed = False
        self.roi_min = 0.0
        self.roi_max = 1.0
        self.cluster_mask = None
        self.cluster_labels_map = None
        self.cluster_vars = []
        
        # --- UI Layout ---
        
        # 1. Top Control Panel
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.btn_load = tk.Button(control_frame, text="Load CSV File", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT)
        
        self.lbl_file = tk.Label(control_frame, text="No file loaded")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_apply = tk.Button(control_frame, text="Select Image & Apply Mask", command=self.apply_mask_to_image)
        self.btn_apply.pack(side=tk.LEFT, padx=10)

        # Confirm ROI Button
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
        self.checkbox_frame = tk.Frame(root)
        self.checkbox_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 2. Bottom Slider Panel
        self.slider_frame = tk.Frame(root)
        self.slider_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)
        
        # --- Left Column: Value Filters ---
        val_frame = tk.LabelFrame(self.slider_frame, text="Distance Value Filter")
        val_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Min Slider
        min_frame = tk.Frame(val_frame)
        min_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.lbl_min = tk.Label(min_frame, text="Min Value: 0.00", width=15)
        self.lbl_min.pack(side=tk.LEFT)
        self.slider_min = tk.Scale(min_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   command=self.on_min_change, resolution=0.01)
        self.slider_min.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Max Slider
        max_frame = tk.Frame(val_frame)
        max_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.lbl_max = tk.Label(max_frame, text="Max Value: 1.00", width=15)
        self.lbl_max.pack(side=tk.LEFT)
        self.slider_max = tk.Scale(max_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   command=self.on_max_change, resolution=0.01)
        self.slider_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Right Column: Spatial Filters ---
        spatial_frame = tk.LabelFrame(self.slider_frame, text="Spatial Coordinate Filter")
        spatial_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # X Sliders
        x_frame = tk.Frame(spatial_frame)
        x_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.lbl_x_min = tk.Label(x_frame, text="X Min: 0", width=10)
        self.lbl_x_min.pack(side=tk.LEFT)
        self.slider_x_min = tk.Scale(x_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_x_min_change)
        self.slider_x_min.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.lbl_x_max = tk.Label(x_frame, text="X Max: 100", width=10)
        self.lbl_x_max.pack(side=tk.LEFT)
        self.slider_x_max = tk.Scale(x_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_x_max_change)
        self.slider_x_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Y Sliders
        y_frame = tk.Frame(spatial_frame)
        y_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.lbl_y_min = tk.Label(y_frame, text="Y Min: 0", width=10)
        self.lbl_y_min.pack(side=tk.LEFT)
        self.slider_y_min = tk.Scale(y_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_y_min_change)
        self.slider_y_min.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.lbl_y_max = tk.Label(y_frame, text="Y Max: 100", width=10)
        self.lbl_y_max.pack(side=tk.LEFT)
        self.slider_y_max = tk.Scale(y_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_y_max_change)
        self.slider_y_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 3. Matplotlib Figure (Middle Content)
        # Use GridSpec to customize layout ratios
        from matplotlib.gridspec import GridSpec
        
        self.fig = Figure(figsize=(14, 8), dpi=100)
        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 4])
        
        # Subplots: Left for Histogram (Vertical), Right for Heatmap
        self.ax_hist = self.fig.add_subplot(gs[0])
        self.ax_map = self.fig.add_subplot(gs[1])
        
        # Adjust padding
        self.fig.tight_layout(pad=3.0, w_pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial empty state
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
            # 前置智能拦截：防止用户误选状态日志文件 (如 inference_status.csv)
            file_name_lower = os.path.basename(file_path).lower()
            if "status" in file_name_lower:
                messagebox.showwarning(
                    "文件类型提示",
                    f"您当前选择的是推理状态文件：\n{os.path.basename(file_path)}\n\n"
                    "该文件记录的是推理日志 (method,status,message)，而不是特征距离矩阵！\n"
                    "请进入具体的模型子目录（例如 AnyUp_640 或 Baseline），选择对应的 *_anomaly_distance_raw.csv 文件。"
                )
                return

            # Load CSV (使用 utf-8-sig 自动清除 Windows UTF-8 BOM 标记 \ufeff)
            try:
                self.raw_map = np.loadtxt(file_path, delimiter=",", encoding="utf-8-sig")
            except Exception:
                try:
                    self.raw_map = np.loadtxt(file_path, encoding="utf-8-sig") # try space delimiter
                except Exception:
                    import pandas as pd
                    df = pd.read_csv(file_path, header=None, encoding="utf-8-sig")
                    self.raw_map = df.values.astype(np.float64)
                
            self.lbl_file.config(text=os.path.basename(file_path))
            
            # Reset ROI/Cluster state
            self.roi_confirmed = False
            self.roi_min = self.raw_map.min()
            self.roi_max = self.raw_map.max()
            self.cluster_mask = None
            self.cluster_labels_map = None
            for widget in self.checkbox_frame.winfo_children():
                widget.destroy()
            self.cluster_vars = []
            
            # Normalize for visualization (0-1)
            min_val, max_val = self.raw_map.min(), self.raw_map.max()
            if max_val - min_val > 0:
                self.normalized_map = (self.raw_map - min_val) / (max_val - min_val)
            else:
                self.normalized_map = np.zeros_like(self.raw_map)

            # Try to load corresponding image for overlay
            self.bg_img = None
            base_name = os.path.basename(file_path).replace("_anomaly_distance_raw.csv", "")
            dir_name = os.path.dirname(file_path)
            
            # Look for original image patterns in the result folder
            potential_images = glob.glob(os.path.join(dir_name, f"{base_name}_original*.png"))
            
            if potential_images:
                # Pick the first match
                img_path = potential_images[0]
                img = cv2.imread(img_path)
                if img is not None:
                    self.bg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize if mismatch
                    if self.bg_img.shape[:2] != self.raw_map.shape:
                        self.bg_img = cv2.resize(self.bg_img, (self.raw_map.shape[1], self.raw_map.shape[0]))
            
            # Setup Slider Range based on raw values
            step = (max_val - min_val) / 200 if max_val > min_val else 0.01
            
            self.slider_min.config(from_=min_val, to=max_val, resolution=step)
            self.slider_min.set(min_val)
            self.current_min = min_val
            
            self.slider_max.config(from_=min_val, to=max_val, resolution=step)
            self.slider_max.set(max_val)
            self.current_max = max_val

            # Setup Spatial Sliders
            h, w = self.raw_map.shape
            
            self.slider_x_min.config(from_=0, to=w, resolution=1)
            self.slider_x_min.set(0)
            self.x_min = 0
            
            self.slider_x_max.config(from_=0, to=w, resolution=1)
            self.slider_x_max.set(w)
            self.x_max = w

            self.slider_y_min.config(from_=0, to=h, resolution=1)
            self.slider_y_min.set(0)
            self.y_min = 0
            
            self.slider_y_max.config(from_=0, to=h, resolution=1)
            self.slider_y_max.set(h)
            self.y_max = h
            
            # Plot Histogram
            self.plot_histogram()
            
            # Initial Heatmap Update
            self.update_heatmap()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def confirm_spatial_roi(self):
        if self.raw_map is None:
            return
            
        # 1. Get ROI bounds
        x0 = max(0, int(self.x_min))
        x1 = min(self.raw_map.shape[1], int(self.x_max))
        y0 = max(0, int(self.y_min))
        y1 = min(self.raw_map.shape[0], int(self.y_max))
        
        if x1 <= x0 or y1 <= y0:
            return

        # 2. Extract Data
        roi_data = self.raw_map[y0:y1, x0:x1]
        
        if roi_data.size == 0:
            return

        # 3. Update Histogram
        self.plot_histogram(data=roi_data)
        
        # 4. Update Distance Sliders Range
        min_val, max_val = roi_data.min(), roi_data.max()
        step = (max_val - min_val) / 200 if max_val > min_val else 0.01
        
        self.slider_min.config(from_=min_val, to=max_val, resolution=step)
        self.slider_max.config(from_=min_val, to=max_val, resolution=step)
        
        self.slider_min.set(min_val)
        self.slider_max.set(max_val)
        self.current_min = min_val
        self.current_max = max_val
        
        # 5. Set Flag and Re-normalize Heatmap
        self.roi_confirmed = True
        self.roi_min = min_val
        self.roi_max = max_val
        
        self.update_heatmap()
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
            
        # Get data based on Current Mask (Spatial + Distance)
        mask = self.get_combined_mask()
        # Temporarily ignore existing cluster mask if re-generating?
        # Actually get_combined_mask includes cluster_mask. 
        # But when generating new clusters, we should probably ignore the OLD cluster mask.
        # So let's manually compute mask without cluster_mask.
        
        # 1. Value Mask
        mask_val = (self.raw_map >= self.current_min) & (self.raw_map <= self.current_max)
        
        # 2. Spatial Mask
        h, w = self.raw_map.shape
        mask_spatial = np.zeros_like(self.raw_map, dtype=bool)
        x0 = max(0, int(self.x_min))
        x1 = min(w, int(self.x_max))
        y0 = max(0, int(self.y_min))
        y1 = min(h, int(self.y_max))
        if x1 > x0 and y1 > y0:
            mask_spatial[y0:y1, x0:x1] = True
            
        base_mask = mask_val & mask_spatial
        valid_data = self.raw_map[base_mask]
        
        if len(valid_data) < k:
             messagebox.showerror("Error", f"Not enough data points ({len(valid_data)}) for {k} clusters")
             return
             
        # K-Means
        # Reshape for cv2.kmeans: (N, 1) float32
        data_pts = valid_data.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data_pts, k, None, criteria, 10, flags)
        
        # Map labels back
        self.cluster_labels_map = np.full_like(self.raw_map, -1, dtype=int)
        self.cluster_labels_map[base_mask] = labels.flatten()
        
        # Generate Checkboxes
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.cluster_vars = []
        
        # Sort clusters by center value
        centers_flat = centers.flatten()
        sorted_indices = np.argsort(centers_flat)
        
        # Remap labels so 0 is lowest value, K-1 is highest
        rank_map = {old_idx: new_rank for new_rank, old_idx in enumerate(sorted_indices)}
        remapped_labels_map = np.full_like(self.cluster_labels_map, -1)
        
        # Optimize remapping
        # We only care about valid pixels
        valid_indices = self.cluster_labels_map != -1
        current_labels = self.cluster_labels_map[valid_indices]
        new_labels = np.array([rank_map[l] for l in current_labels])
        remapped_labels_map[valid_indices] = new_labels
        
        self.cluster_labels_map = remapped_labels_map
        
        # Create Checkboxes
        for i in range(k):
            var = tk.BooleanVar(value=True)
            self.cluster_vars.append(var)
            c_val = centers_flat[sorted_indices[i]]
            chk = tk.Checkbutton(self.checkbox_frame, text=f"Cluster {i+1} ({c_val:.3f})", 
                                 variable=var, command=self.on_cluster_update)
            chk.pack(side=tk.LEFT, padx=5)
            
        self.on_cluster_update()

    def on_cluster_update(self):
        if self.cluster_labels_map is None:
            return
            
        selected_clusters = [i for i, var in enumerate(self.cluster_vars) if var.get()]
        
        # Update cluster mask
        self.cluster_mask = np.isin(self.cluster_labels_map, selected_clusters)
        
        self.update_heatmap()

    def plot_histogram(self, data=None):
        self.ax_hist.clear()
        if data is None:
            data_flat = self.raw_map.flatten()
        else:
            data_flat = data.flatten()
        
        # Orientation horizontal: y-axis is values, x-axis is count
        n, bins, patches = self.ax_hist.hist(data_flat, bins=50, color='skyblue', edgecolor='black', log=True, orientation='horizontal')
        self.ax_hist.set_title("Distance Distribution (Log Scale)")
        self.ax_hist.set_ylabel("Raw Distance")
        self.ax_hist.set_xlabel("Count (Log)")
        self.ax_hist.grid(True, alpha=0.3)
        
        # Add horizontal lines/region for threshold
        self.min_line = self.ax_hist.axhline(self.current_min, color='g', linestyle='--', linewidth=2, label='Min')
        self.max_line = self.ax_hist.axhline(self.current_max, color='r', linestyle='--', linewidth=2, label='Max')
        self.range_span = self.ax_hist.axhspan(self.current_min, self.current_max, color='yellow', alpha=0.2)
        self.ax_hist.legend()
        
        self.canvas.draw()

    def on_min_change(self, value):
        val = float(value)
        self.current_min = val
        self.lbl_min.config(text=f"Min Threshold: {val:.4f}")
        
        # Ensure Min <= Max
        if self.current_min > self.current_max:
             self.slider_max.set(val) # Push max up
        
        if self.raw_map is not None:
            self.update_histogram_lines()
            self.update_heatmap()

    def on_max_change(self, value):
        val = float(value)
        self.current_max = val
        self.lbl_max.config(text=f"Max Value: {val:.4f}")
        
        # Ensure Max >= Min
        if self.current_max < self.current_min:
             self.slider_min.set(val) # Push min down
        
        if self.raw_map is not None:
            self.update_histogram_lines()
            self.update_heatmap()

    def on_x_min_change(self, value):
        val = int(value)
        self.x_min = val
        self.lbl_x_min.config(text=f"X Min: {val}")
        
        if self.x_min > self.x_max:
            self.slider_x_max.set(val)
            
        if self.raw_map is not None:
            self.update_heatmap()

    def on_x_max_change(self, value):
        val = int(value)
        self.x_max = val
        self.lbl_x_max.config(text=f"X Max: {val}")
        
        if self.x_max < self.x_min:
            self.slider_x_min.set(val)
            
        if self.raw_map is not None:
            self.update_heatmap()

    def on_y_min_change(self, value):
        val = int(value)
        self.y_min = val
        self.lbl_y_min.config(text=f"Y Min: {val}")
        
        if self.y_min > self.y_max:
            self.slider_y_max.set(val)
            
        if self.raw_map is not None:
            self.update_heatmap()

    def on_y_max_change(self, value):
        val = int(value)
        self.y_max = val
        self.lbl_y_max.config(text=f"Y Max: {val}")
        
        if self.y_max < self.y_min:
            self.slider_y_min.set(val)
            
        if self.raw_map is not None:
            self.update_heatmap()

    def update_histogram_lines(self):
        if hasattr(self, 'min_line'):
            self.min_line.set_ydata([self.current_min, self.current_min])
        if hasattr(self, 'max_line'):
            self.max_line.set_ydata([self.current_max, self.current_max])
        
        # Update shaded region
        if hasattr(self, 'range_span'):
            self.range_span.remove()
        self.range_span = self.ax_hist.axhspan(self.current_min, self.current_max, color='yellow', alpha=0.2)
        
        self.canvas.draw()

    def get_combined_mask(self):
        # 1. Value Mask
        mask_val = (self.raw_map >= self.current_min) & (self.raw_map <= self.current_max)
        
        # 2. Spatial Mask
        h, w = self.raw_map.shape
        mask_spatial = np.zeros_like(self.raw_map, dtype=bool)
        
        # Handle bounds safely
        x0 = max(0, int(self.x_min))
        x1 = min(w, int(self.x_max))
        y0 = max(0, int(self.y_min))
        y1 = min(h, int(self.y_max))
        
        if x1 > x0 and y1 > y0:
            mask_spatial[y0:y1, x0:x1] = True
            
        final_mask = mask_val & mask_spatial
        
        if self.cluster_mask is not None:
            final_mask = final_mask & self.cluster_mask
            
        return final_mask

    def update_heatmap(self):
        self.ax_map.clear()
        
        # Create mask for RANGE + SPATIAL
        mask = self.get_combined_mask()
        
        # Determine normalization range
        if self.roi_confirmed:
            vmin, vmax = self.roi_min, self.roi_max
        else:
            vmin, vmax = self.raw_map.min(), self.raw_map.max()
            
        # Create visualization
        if self.bg_img is not None:
            # Draw background image
            self.ax_map.imshow(self.bg_img)
            
            # Overlay Heatmap
            cmap = plt.get_cmap('jet')
            
            # Dynamic normalization
            denom = vmax - vmin if (vmax - vmin) > 0 else 1.0
            norm_map = (self.raw_map - vmin) / denom
            norm_map = np.clip(norm_map, 0, 1)
            
            rgba_img = cmap(norm_map)
            
            # Set alpha: 0 if outside range, 0.5 if inside
            alpha = np.zeros_like(self.raw_map)
            alpha[mask] = 0.5 
            
            rgba_img[:, :, 3] = alpha
            
            self.ax_map.imshow(rgba_img)
            
        else:
            # No background image
            masked_map = np.ma.masked_where(~mask, self.raw_map)
            self.ax_map.imshow(masked_map, cmap='jet', vmin=vmin, vmax=vmax)
            
        # Update title with current range to show overlap status clearly
        title_str = f"Heatmap\n({self.current_min:.4f} <= V <= {self.current_max:.4f})"
        self.ax_map.set_title(title_str, fontsize=10, pad=10)
        self.ax_map.axis('off')
        
        self.canvas.draw()

    def apply_mask_to_image(self):
        if self.raw_map is None:
            messagebox.showwarning("Warning", "Please load a heatmap (CSV) first.")
            return

        # 1. Select Target Image
        img_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"), ("All Files", "*.*")]
        )
        if not img_path:
            return
            
        try:
            # 2. Load Image
            target_img = cv2.imread(img_path)
            if target_img is None:
                raise ValueError("Could not load image.")
            
            # 3. Create Mask
            mask = self.get_combined_mask()
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # 4. Resize mask to match target image
            h, w = target_img.shape[:2]
            mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 5. Create Black Background and Apply Mask
            black_bg = np.zeros_like(target_img)
            mask_bool = mask_resized > 0
            black_bg[mask_bool] = target_img[mask_bool]
            
            # 6. Save Result
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"masked_{os.path.basename(img_path)}"
            )
            
            if save_path:
                cv2.imwrite(save_path, black_bg)
                messagebox.showinfo("Success", f"Saved masked image to:\n{save_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeatmapAnalyzerGUI(root)
    root.mainloop()
