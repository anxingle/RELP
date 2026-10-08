import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

class KNNClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Clustering Visualization")
        self.root.geometry("1400x900")
        
        # 初始化变量
        self.image_path = "Masked_Pic/dino ab.jpg"  # 默认图片路径
        self.original_image = None
        self.processed_image = None
        self.non_black_pixels = None
        self.non_black_coordinates = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.n_clusters = 2
        self.target_rgb = [0, 0, 0]  # 默认目标RGB值
        self.selected_clusters = set()
        
        self.setup_ui()
        self.load_default_image()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # RGB控制
        rgb_frame = ttk.LabelFrame(control_frame, text="RGB Target", padding="5")
        rgb_frame.grid(row=0, column=0, padx=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Label(rgb_frame, text="R:").grid(row=0, column=0, padx=2)
        self.r_var = tk.StringVar(value="0")
        self.r_entry = ttk.Entry(rgb_frame, textvariable=self.r_var, width=5)
        self.r_entry.grid(row=0, column=1, padx=2)
        
        ttk.Label(rgb_frame, text="G:").grid(row=0, column=2, padx=2)
        self.g_var = tk.StringVar(value="0")
        self.g_entry = ttk.Entry(rgb_frame, textvariable=self.g_var, width=5)
        self.g_entry.grid(row=0, column=3, padx=2)
        
        ttk.Label(rgb_frame, text="B:").grid(row=0, column=4, padx=2)
        self.b_var = tk.StringVar(value="0")
        self.b_entry = ttk.Entry(rgb_frame, textvariable=self.b_var, width=5)
        self.b_entry.grid(row=0, column=5, padx=2)
        
        # KNN簇数控制
        knn_frame = ttk.LabelFrame(control_frame, text="KNN Clusters", padding="5")
        knn_frame.grid(row=0, column=1, padx=(0, 10), sticky=(tk.W, tk.E))
        
        ttk.Label(knn_frame, text="Clusters:").grid(row=0, column=0, padx=2)
        self.clusters_var = tk.StringVar(value="2")
        self.clusters_entry = ttk.Entry(knn_frame, textvariable=self.clusters_var, width=5)
        self.clusters_entry.grid(row=0, column=1, padx=2)
        
        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Process", command=self.process_image).grid(row=0, column=1, padx=2)
        
        # Clustering Selection面板
        self.selection_frame = ttk.LabelFrame(main_frame, text="Clustering Selection", padding="10")
        self.selection_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 图像显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图形，使用更合适的尺寸
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 6))
        self.fig.patch.set_facecolor('white')
        
        # 创建三个子图
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # 设置子图标题，使用更小的字体
        self.ax1.set_title("Picture after clustering", fontsize=10, pad=5)
        self.ax2.set_title("Selected clustering", fontsize=10, pad=5)
        self.ax3.set_title("Picture minus selected clustering", fontsize=10, pad=5)
        
        # 调整子图间距
        self.fig.tight_layout(pad=1.5)
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.load_default_image()
            
    def load_default_image(self):
        try:
            # 读取图像
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Cannot load image: {self.image_path}")
                return
            
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.process_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            
    def calculate_euclidean_distance(self, pixels, target_rgb):
        """计算像素点到目标RGB的欧式距离"""
        target = np.array(target_rgb)
        return np.linalg.norm(pixels - target, axis=1)
        
    def process_image(self):
        if self.original_image is None:
            return
            
        try:
            # 获取参数
            self.target_rgb = [int(self.r_var.get()), int(self.g_var.get()), int(self.b_var.get())]
            self.n_clusters = int(self.clusters_var.get())
            
            # 找到所有非全黑像素
            height, width = self.original_image.shape[:2]
            non_black_mask = ~np.all(self.original_image == [0, 0, 0], axis=2)
            self.non_black_coordinates = np.column_stack(np.where(non_black_mask))
            self.non_black_pixels = self.original_image[non_black_mask]
            
            if len(self.non_black_pixels) == 0:
                messagebox.showwarning("Warning", "No non-black pixels found")
                return
                
            # 计算到目标RGB的欧式距离
            distances = self.calculate_euclidean_distance(self.non_black_pixels, self.target_rgb)
            distances_reshaped = distances.reshape(-1, 1)
            
            # KMeans聚类
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(distances_reshaped)
            self.cluster_centers = kmeans.cluster_centers_.flatten()
            
            # 更新Clustering Selection界面
            self.update_selection_interface()
            
            # 更新图像显示
            self.update_images()
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Processing error: {str(e)}")
            
    def update_selection_interface(self):
        # 清除现有的选择框
        for widget in self.selection_frame.winfo_children():
            widget.destroy()
            
        # 根据距离排序簇
        cluster_distances = [(i, self.cluster_centers[i]) for i in range(self.n_clusters)]
        cluster_distances.sort(key=lambda x: x[1])  # 按距离排序
        
        self.cluster_vars = {}
        
        # 创建选择框
        for idx, (cluster_id, distance) in enumerate(cluster_distances):
            frame = ttk.Frame(self.selection_frame)
            frame.grid(row=0, column=idx, padx=10, pady=5)
            
            # 标签（按距离排序，1是最小距离）
            label = ttk.Label(frame, text=f"{idx + 1}", font=("Arial", 12, "bold"))
            label.pack()
            
            # 复选框
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(frame, variable=var, command=self.on_selection_change)
            checkbox.pack()
            
            self.cluster_vars[cluster_id] = var
            
    def on_selection_change(self):
        # 更新选中的簇
        self.selected_clusters = set()
        for cluster_id, var in self.cluster_vars.items():
            if var.get():
                self.selected_clusters.add(cluster_id)
                
        # 更新图像显示
        self.update_images()
        
    def create_cluster_image(self, selected_only=False):
        """创建聚类结果图像"""
        height, width = self.original_image.shape[:2]
        result_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 定义颜色
        if selected_only:
            # 第二个图：选中的簇用白色显示
            colors = [[255, 255, 255]] * 8  # 所有簇都用白色
        else:
            # 第一个图：不同簇用不同颜色
            colors = [
                [255, 0, 0],    # 红色
                [0, 255, 0],    # 绿色
                [0, 0, 255],    # 蓝色
                [255, 255, 0],  # 黄色
                [255, 0, 255],  # 洋红
                [0, 255, 255],  # 青色
                [128, 128, 128], # 灰色
                [255, 128, 0],  # 橙色
            ]
        
        for i, coord in enumerate(self.non_black_coordinates):
            cluster_id = self.cluster_labels[i]
            
            if selected_only and cluster_id not in self.selected_clusters:
                continue
                
            if not selected_only or cluster_id in self.selected_clusters:
                y, x = coord
                color_idx = cluster_id % len(colors)
                result_image[y, x] = colors[color_idx]
                
        return result_image
        
    def create_masked_original_image(self):
        """创建使用第二个图作为mask的原图"""
        result_image = self.original_image.copy()
        
        if not self.selected_clusters:
            return result_image
            
        # 对于原图中的每个非黑像素点，检查是否在选中的簇中
        for i, coord in enumerate(self.non_black_coordinates):
            cluster_id = self.cluster_labels[i]
            if cluster_id in self.selected_clusters:
                # 如果该像素点属于选中的簇（在图2中非全黑），则在第三个图中设为黑色
                y, x = coord
                result_image[y, x] = [0, 0, 0]
                
        return result_image
        
    def calculate_aspect_ratio_size(self, original_height, original_width, fixed_height=4):
        """根据原图比例计算合适的显示尺寸"""
        aspect_ratio = original_width / original_height
        calculated_width = fixed_height * aspect_ratio
        return fixed_height, calculated_width
        
    def update_images(self):
        if self.cluster_labels is None:
            return
            
        # 清除之前的图像
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # 图1：聚类后的图像
        cluster_image = self.create_cluster_image(selected_only=False)
        self.ax1.imshow(cluster_image)
        self.ax1.set_title("Picture after clustering", fontsize=10, pad=5)
        self.ax1.axis('off')
        
        # 图2：选中的簇（白色显示）
        selected_image = self.create_cluster_image(selected_only=True)
        self.ax2.imshow(selected_image)
        self.ax2.set_title("Selected clustering", fontsize=10, pad=5)
        self.ax2.axis('off')
        
        # 图3：使用第二个图作为mask的原图
        masked_original = self.create_masked_original_image()
        self.ax3.imshow(masked_original)
        self.ax3.set_title("Picture minus selected clustering", fontsize=10, pad=5)
        self.ax3.axis('off')
        
        # 调整布局并刷新画布
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
        self.canvas.flush_events()

def main():
    root = tk.Tk()
    app = KNNClusteringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()