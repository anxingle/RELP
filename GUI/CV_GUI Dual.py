import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

class BinarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CV GUI Tool")
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path1 = None
        self.image_path2 = None
        self.cv_image1 = None
        self.gray_image1 = None
        self.cv_image2 = None
        self.gray_image2 = None
        self.processed_image1 = None
        self.processed_image2 = None
        self.threshold_val = 127
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main Layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel (Top)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Split into two rows for better visibility
        row1_frame = ttk.Frame(control_frame)
        row1_frame.pack(fill=tk.X, pady=2)
        
        self.row2_frame_ref = ttk.Frame(control_frame)
        self.row2_frame_ref.pack(fill=tk.X, pady=2)
        
        # --- Row 1 Controls ---
        
        ttk.Button(row1_frame, text="Load Image 1", command=self.load_image1).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1_frame, text="Load Image 2", command=self.load_image2).pack(side=tk.LEFT, padx=5)
        
        # Method Selection
        ttk.Label(row1_frame, text="Method:").pack(side=tk.LEFT, padx=(5, 5))
        self.method_var = tk.StringVar(value="Global Threshold")
        self.method_combo = ttk.Combobox(
            row1_frame, 
            textvariable=self.method_var, 
            values=["Global Threshold", "Adaptive Gaussian", "Background Subtraction", "Color Filter", "HSV Detector"],
            state="readonly",
            width=18
        )
        self.method_combo.pack(side=tk.LEFT, padx=5)
        self.method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        # HSV Controls (Hidden initially)
        self.hsv_frame = ttk.LabelFrame(control_frame, text="HSV Detector Settings", padding="10")
        
        # HSV Variables
        self.h_min_var = tk.IntVar(value=0)
        self.h_max_var = tk.IntVar(value=179)
        self.s_min_var = tk.IntVar(value=0)
        self.s_max_var = tk.IntVar(value=255)
        self.v_min_var = tk.IntVar(value=0)
        self.v_max_var = tk.IntVar(value=255)
        
        self.hsv_labels = {}

        # Use Grid layout for better alignment
        # Columns: [Label, Slider, Value] x 3 (H, S, V) ? No, too wide.
        # Let's try:
        # Row 0: H Min Label, Slider, Value, S Min Label, Slider, Value, V Min Label, Slider, Value
        # Row 1: H Max Label, Slider, Value, S Max Label, Slider, Value, V Max Label, Slider, Value
        
        def add_hsv_slider_grid(parent, label_text, var, v_min, v_max, key, row, col_start):
            # Label
            ttk.Label(parent, text=label_text, width=6).grid(row=row, column=col_start, padx=(10, 2), pady=5, sticky="e")
            # Slider
            s = ttk.Scale(parent, from_=v_min, to=v_max, orient=tk.HORIZONTAL, variable=var, command=self.on_slider_change, length=200)
            s.grid(row=row, column=col_start+1, padx=2, pady=5)
            # Value Label
            l = ttk.Label(parent, text=str(var.get()), width=4)
            l.grid(row=row, column=col_start+2, padx=2, pady=5, sticky="w")
            self.hsv_labels[key] = l

        # Min Row (Row 0)
        add_hsv_slider_grid(self.hsv_frame, "H Min:", self.h_min_var, 0, 255, 'h_min', 0, 0)
        add_hsv_slider_grid(self.hsv_frame, "S Min:", self.s_min_var, 0, 255, 's_min', 0, 3)
        add_hsv_slider_grid(self.hsv_frame, "V Min:", self.v_min_var, 0, 255, 'v_min', 0, 6)
        
        # Max Row (Row 1)
        add_hsv_slider_grid(self.hsv_frame, "H Max:", self.h_max_var, 0, 255, 'h_max', 1, 0)
        add_hsv_slider_grid(self.hsv_frame, "S Max:", self.s_max_var, 0, 255, 's_max', 1, 3)
        add_hsv_slider_grid(self.hsv_frame, "V Max:", self.v_max_var, 0, 255, 'v_max', 1, 6)

        # RGB Filter Controls (Hidden initially)
        self.rgb_frame = ttk.Frame(row1_frame)
        
        # Proximity
        ttk.Label(self.rgb_frame, text="Prox:").pack(side=tk.LEFT)
        self.prox_var = tk.StringVar(value="Near")
        self.prox_combo = ttk.Combobox(
            self.rgb_frame,
            textvariable=self.prox_var,
            values=["Near", "Far"],
            state="readonly",
            width=5
        )
        self.prox_combo.pack(side=tk.LEFT, padx=2)
        self.prox_combo.bind("<<ComboboxSelected>>", lambda e: self.update_binarization())
        
        # RGB Inputs
        ttk.Label(self.rgb_frame, text="R:").pack(side=tk.LEFT, padx=(5,0))
        self.r_var = tk.StringVar(value="0")
        self.r_entry = ttk.Entry(self.rgb_frame, textvariable=self.r_var, width=4)
        self.r_entry.pack(side=tk.LEFT)
        self.r_entry.bind('<Return>', lambda e: self.update_binarization())
        self.r_entry.bind('<FocusOut>', lambda e: self.update_binarization())
        
        ttk.Label(self.rgb_frame, text="G:").pack(side=tk.LEFT, padx=(2,0))
        self.g_var = tk.StringVar(value="0")
        self.g_entry = ttk.Entry(self.rgb_frame, textvariable=self.g_var, width=4)
        self.g_entry.pack(side=tk.LEFT)
        self.g_entry.bind('<Return>', lambda e: self.update_binarization())
        self.g_entry.bind('<FocusOut>', lambda e: self.update_binarization())
        
        ttk.Label(self.rgb_frame, text="B:").pack(side=tk.LEFT, padx=(2,0))
        self.b_var = tk.StringVar(value="0")
        self.b_entry = ttk.Entry(self.rgb_frame, textvariable=self.b_var, width=4)
        self.b_entry.pack(side=tk.LEFT)
        self.b_entry.bind('<Return>', lambda e: self.update_binarization())
        self.b_entry.bind('<FocusOut>', lambda e: self.update_binarization())

        # Block Size Slider (Hidden initially)
        self.block_frame = ttk.Frame(row1_frame)
        self.block_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.block_frame, text="Block Size:").pack(side=tk.LEFT)
        self.block_var = tk.IntVar(value=11)
        self.block_slider = ttk.Scale(
            self.block_frame, from_=3, to=255, orient=tk.HORIZONTAL, 
            variable=self.block_var, command=self.on_slider_change, length=150
        )
        self.block_slider.pack(side=tk.LEFT, padx=5)
        self.block_label = ttk.Label(self.block_frame, text="11")
        self.block_label.pack(side=tk.LEFT)

        # Threshold/C Slider
        self.thresh_frame = ttk.Frame(row1_frame)
        self.thresh_frame.pack(side=tk.LEFT, padx=5)
        self.thresh_label_txt = tk.StringVar(value="Threshold:")
        ttk.Label(self.thresh_frame, textvariable=self.thresh_label_txt).pack(side=tk.LEFT)
        
        self.slider_var = tk.IntVar(value=127)
        self.slider = ttk.Scale(
            self.thresh_frame, 
            from_=0, 
            to=255, 
            orient=tk.HORIZONTAL, 
            variable=self.slider_var,
            command=self.on_slider_change,
            length=200
        )
        self.slider.pack(side=tk.LEFT, padx=5)
        
        # Value Label
        self.value_label = ttk.Label(self.thresh_frame, text="127")
        self.value_label.pack(side=tk.LEFT, padx=5)

        # --- Row 2 Controls ---

        # --- Denoise (Blur) Control ---
        self.blur_frame = ttk.Frame(self.row2_frame_ref)
        self.blur_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.blur_frame, text="Denoise:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=0)
        self.blur_slider = ttk.Scale(
            self.blur_frame, from_=0, to=10, orient=tk.HORIZONTAL, 
            variable=self.blur_var, command=self.on_slider_change, length=100
        )
        self.blur_slider.pack(side=tk.LEFT, padx=5)
        
        # Add Label for Denoise
        self.blur_label = ttk.Label(self.blur_frame, text="0")
        self.blur_label.pack(side=tk.LEFT, padx=2)
        
        # --- Clean Specks (Morph Open) Control ---
        self.morph_frame = ttk.Frame(self.row2_frame_ref)
        self.morph_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.morph_frame, text="Clean:").pack(side=tk.LEFT)
        self.morph_var = tk.IntVar(value=0)
        self.morph_slider = ttk.Scale(
            self.morph_frame, from_=0, to=5, orient=tk.HORIZONTAL, 
            variable=self.morph_var, command=self.on_slider_change, length=100
        )
        self.morph_slider.pack(side=tk.LEFT, padx=5)
        
        # Add Label for Clean
        self.morph_label = ttk.Label(self.morph_frame, text="0")
        self.morph_label.pack(side=tk.LEFT, padx=2)
        
        # --- Invert Control ---
        self.invert_var = tk.BooleanVar(value=False)
        self.invert_check = ttk.Checkbutton(
            self.row2_frame_ref, 
            text="Invert", 
            variable=self.invert_var, 
            command=lambda: self.on_slider_change(None)
        )
        self.invert_check.pack(side=tk.LEFT, padx=10)
        
        # Initial UI State Update
        self.update_ui_state()

        # Image Display Area (Two images side-by-side, each with Original/Selected/Unselected)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        left_column = ttk.Frame(image_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        right_column = ttk.Frame(image_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        self.left_panel1 = ttk.LabelFrame(left_column, text="Original Image 1")
        self.left_panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.left_panel1.pack_propagate(False)
        self.lbl_original1 = ttk.Label(self.left_panel1, anchor="center")
        self.lbl_original1.pack(expand=True)
        
        self.middle_panel1 = ttk.LabelFrame(left_column, text="Selected 1")
        self.middle_panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.middle_panel1.pack_propagate(False)
        self.lbl_result1 = ttk.Label(self.middle_panel1, anchor="center")
        self.lbl_result1.pack(expand=True)
        
        self.right_panel1 = ttk.LabelFrame(left_column, text="Unselected 1")
        self.right_panel1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.right_panel1.pack_propagate(False)
        self.lbl_unselected1 = ttk.Label(self.right_panel1, anchor="center")
        self.lbl_unselected1.pack(expand=True)
        
        self.left_panel2 = ttk.LabelFrame(right_column, text="Original Image 2")
        self.left_panel2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.left_panel2.pack_propagate(False)
        self.lbl_original2 = ttk.Label(self.left_panel2, anchor="center")
        self.lbl_original2.pack(expand=True)
        
        self.middle_panel2 = ttk.LabelFrame(right_column, text="Selected 2")
        self.middle_panel2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.middle_panel2.pack_propagate(False)
        self.lbl_result2 = ttk.Label(self.middle_panel2, anchor="center")
        self.lbl_result2.pack(expand=True)
        
        self.right_panel2 = ttk.LabelFrame(right_column, text="Unselected 2")
        self.right_panel2.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        self.right_panel2.pack_propagate(False)
        self.lbl_unselected2 = ttk.Label(self.right_panel2, anchor="center")
        self.lbl_unselected2.pack(expand=True)
        
    def load_image1(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path1 = file_path
            img = cv2.imread(file_path)
            if img is not None:
                self.cv_image1 = img
                self.gray_image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_image(rgb_image, self.lbl_original1)
                self.update_binarization()

    def load_image2(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path2 = file_path
            img = cv2.imread(file_path)
            if img is not None:
                self.cv_image2 = img
                self.gray_image2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_image(rgb_image, self.lbl_original2)
                self.update_binarization()

    def on_method_change(self, event):
        self.update_ui_state()
        self.update_binarization()

    def update_ui_state(self):
        method = self.method_var.get()
        
        # Reset ranges based on method
        self.hsv_frame.pack_forget() # Default hidden

        if method == "Global Threshold":
            self.block_frame.pack_forget()
            self.rgb_frame.pack_forget()
            self.thresh_label_txt.set("Threshold:")
            self.slider.configure(from_=0, to=255)
            # Restore reasonable default if out of range
            if self.slider_var.get() < 0: self.slider_var.set(127)
            self.thresh_frame.pack(side=tk.LEFT, padx=5) # Ensure threshold is shown
            
        elif method == "Adaptive Gaussian":
            self.rgb_frame.pack_forget()
            self.block_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            self.thresh_label_txt.set("C (Constant):")
            self.slider.configure(from_=-50, to=50) # C value range
            # Set default C to roughly 2
            self.slider_var.set(2)
            self.thresh_frame.pack(side=tk.LEFT, padx=5)
            
        elif method == "Background Subtraction":
            self.rgb_frame.pack_forget()
            self.block_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            self.thresh_label_txt.set("Diff Threshold:")
            self.slider.configure(from_=0, to=255)
            self.slider_var.set(20) # Default diff threshold
            self.thresh_frame.pack(side=tk.LEFT, padx=5)

        elif method == "Color Filter":
            self.block_frame.pack_forget()
            self.rgb_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            self.thresh_label_txt.set("Tolerance:")
            self.slider.configure(from_=0, to=255)
            # Default tolerance
            if self.slider_var.get() == 127: self.slider_var.set(30)
            self.thresh_frame.pack(side=tk.LEFT, padx=5)

        elif method == "HSV Detector":
            self.block_frame.pack_forget()
            self.rgb_frame.pack_forget()
            self.thresh_frame.pack_forget() # Hide generic threshold slider
            # Show HSV Frame between row1 and row2 (or just pack it)
            # Since control_frame is the parent, and row1/row2 are packed.
            # We want it visible. We can pack it after row1_frame.
            # But update_ui_state is called repeatedly.
            # Pack order matters.
            # If we pack it, it goes to the bottom.
            # We can pack it before row2_frame if possible.
            self.hsv_frame.pack(fill=tk.X, pady=2, before=self.row2_frame_ref)
            # We need reference to row2_frame.
            # I need to save row2_frame as self.row2_frame in setup_ui first.


    def on_slider_change(self, event):
        # Update labels
        self.value_label.config(text=str(int(self.slider_var.get())))
        self.blur_label.config(text=str(int(self.blur_var.get())))
        self.morph_label.config(text=str(int(self.morph_var.get())))
        
        # Update HSV labels if they exist
        if hasattr(self, 'hsv_labels'):
            for key, lbl in self.hsv_labels.items():
                if key == 'h_min': lbl.config(text=str(self.h_min_var.get()))
                elif key == 'h_max': lbl.config(text=str(self.h_max_var.get()))
                elif key == 's_min': lbl.config(text=str(self.s_min_var.get()))
                elif key == 's_max': lbl.config(text=str(self.s_max_var.get()))
                elif key == 'v_min': lbl.config(text=str(self.v_min_var.get()))
                elif key == 'v_max': lbl.config(text=str(self.v_max_var.get()))

        # Ensure block size is odd
        block_val = int(self.block_var.get())
        if block_val % 2 == 0:
            block_val += 1
            self.block_var.set(block_val)
        self.block_label.config(text=str(block_val))
        
        self.update_binarization()
            
    def update_binarization(self):
        if self.cv_image1 is None and self.cv_image2 is None:
            return
        for idx in (1, 2):
            cv_img = getattr(self, f"cv_image{idx}", None)
            gray_img = getattr(self, f"gray_image{idx}", None)
            if cv_img is None:
                continue
            old_cv = getattr(self, "cv_image", None)
            old_gray = getattr(self, "gray_image", None)
            old_lbl_res = getattr(self, "lbl_result", None)
            old_lbl_uns = getattr(self, "lbl_unselected", None)
            old_processed = getattr(self, "processed_image", None)
            self.cv_image = cv_img
            self.gray_image = gray_img
            if idx == 1:
                self.lbl_result = self.lbl_result1
                self.lbl_unselected = self.lbl_unselected1
            else:
                self.lbl_result = self.lbl_result2
                self.lbl_unselected = self.lbl_unselected2
            try:
                self._update_binarization_single()
                if idx == 1:
                    self.processed_image1 = getattr(self, "processed_image", None)
                else:
                    self.processed_image2 = getattr(self, "processed_image", None)
            finally:
                self.cv_image = old_cv
                self.gray_image = old_gray
                self.lbl_result = old_lbl_res
                self.lbl_unselected = old_lbl_uns
                self.processed_image = old_processed

    def _update_binarization_single(self):
        method = self.method_var.get()
        val = int(self.slider_var.get())
        
        binary = None

        if method == "Color Filter":
            img_to_process = self.cv_image.copy()
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
            
            try:
                r_val = int(self.r_var.get())
                g_val = int(self.g_var.get())
                b_val = int(self.b_var.get())
                target_rgb = np.array([r_val, g_val, b_val], dtype=np.float32)
            except ValueError:
                target_rgb = np.array([0, 0, 0], dtype=np.float32)
            
            diff = img_rgb.astype(np.float32) - target_rgb
            dist = np.linalg.norm(diff, axis=2)
            
            proximity = self.prox_var.get()
            if proximity == "Far":
                binary = np.where(dist > val, 255, 0).astype(np.uint8)
            else:
                binary = np.where(dist <= val, 255, 0).astype(np.uint8)

            gray_orig = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            _, non_black_mask = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY)
            binary = cv2.bitwise_and(binary, non_black_mask)

        elif method == "HSV Detector":
            img_to_process = self.cv_image.copy()
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            img_hsv = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2HSV)
            
            h_min = self.h_min_var.get()
            h_max = self.h_max_var.get()
            s_min = self.s_min_var.get()
            s_max = self.s_max_var.get()
            v_min = self.v_min_var.get()
            v_max = self.v_max_var.get()
            
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            binary = cv2.inRange(img_hsv, lower, upper)

        else:
            if self.gray_image is None:
                return
            img_to_process = self.gray_image.copy()
            
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            if method == "Global Threshold":
                _, binary = cv2.threshold(img_to_process, val, 255, cv2.THRESH_BINARY)
                
            elif method == "Adaptive Gaussian":
                block_size = int(self.block_var.get())
                if block_size < 3:
                    block_size = 3
                if block_size % 2 == 0:
                    block_size += 1
                
                binary = cv2.adaptiveThreshold(
                    img_to_process, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    block_size, 
                    val
                )
                
            elif method == "Background Subtraction":
                block_size = int(self.block_var.get())
                if block_size < 3:
                    block_size = 3
                if block_size % 2 == 0:
                    block_size += 1
                
                bg = cv2.GaussianBlur(img_to_process, (block_size, block_size), 0)
                diff = cv2.subtract(bg, img_to_process)
                _, binary = cv2.threshold(diff, val, 255, cv2.THRESH_BINARY)
        
        if binary is None:
            return

        morph_amount = self.morph_var.get()
        if morph_amount > 0:
            ksize = morph_amount * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if self.invert_var.get():
            binary = cv2.bitwise_not(binary)

        self.processed_image = binary
        
        self.display_image(binary, self.lbl_result)
        
        if hasattr(self, "lbl_unselected"):
             unselected = cv2.bitwise_not(binary)
             
             if method == "Color Filter":
                 gray_orig = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                 _, non_black_mask = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY)
                 unselected = cv2.bitwise_and(unselected, non_black_mask)
                 
             self.display_image(unselected, self.lbl_unselected)
        
    def display_image(self, cv_img, label_widget):
        h, w = cv_img.shape[:2]
        
        # 使用 Label 所在的面板尺寸作为约束，避免图片本身尺寸反过来影响布局
        container = label_widget.master
        try:
            container.update_idletasks()
        except Exception:
            pass
        
        max_w = container.winfo_width()
        max_h = container.winfo_height()
        if max_w <= 1 or max_h <= 1:
            max_w = 400
            max_h = 220
        
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Choose interpolation method based on scaling
        # If downscaling (scale < 1), use INTER_AREA to avoid aliasing/Moiré patterns (which look like blocks)
        # If upscaling (scale > 1), use INTER_NEAREST to show sharp pixels (pixel-level view)
        if scale < 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_NEAREST
            
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interpolation)
        
        # Convert to PIL Image
        if len(resized.shape) == 2: # Grayscale
            img = Image.fromarray(resized)
        else: # RGB
            img = Image.fromarray(resized)
            
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Keep reference to avoid garbage collection
        label_widget.imgtk = imgtk
        label_widget.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = BinarizationApp(root)
    root.mainloop()
