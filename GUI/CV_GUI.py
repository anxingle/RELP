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
        self.image_path = None
        self.original_cv_image = None # Store full resolution
        self.cv_image = None     # Current (potentially resized) BGR image
        self.gray_image = None   # Grayscale image
        self.processed_image = None # Binary image
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
        
        # Load Button
        ttk.Button(row1_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        # Resize Selection
        ttk.Label(row1_frame, text="Size:").pack(side=tk.LEFT, padx=(5, 2))
        self.resize_var = tk.StringVar(value="Original")
        self.resize_combo = ttk.Combobox(
            row1_frame, 
            textvariable=self.resize_var, 
            values=["Original", "1/2", "1/4", "1/8"],
            state="readonly",
            width=8
        )
        self.resize_combo.pack(side=tk.LEFT, padx=2)
        self.resize_combo.bind("<<ComboboxSelected>>", self.apply_resize)

        # Method Selection
        ttk.Label(row1_frame, text="Method:").pack(side=tk.LEFT, padx=(5, 5))
        self.method_var = tk.StringVar(value="Global Threshold")
        self.method_combo = ttk.Combobox(
            row1_frame, 
            textvariable=self.method_var, 
            values=[
                "Global Threshold", 
                "Adaptive Gaussian", 
                "Background Subtraction", 
                "ODBP (Weighted Diff)", 
                "Combined", 
                "Color Filter (HSV)", 
                "HSV Detector", 
                "Morph Top-Hat (Bright)", 
                "Morph Black-Hat (Dark)"
            ],
            state="readonly",
            width=25
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
        self.lbl_block_title = ttk.Label(self.block_frame, text="Block Size:")
        self.lbl_block_title.pack(side=tk.LEFT)
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

        # ODBP Weighted Difference Controls
        self.odbp_frame = ttk.Frame(row1_frame)
        
        # Alpha (Original Image Weight)
        ttk.Label(self.odbp_frame, text="Alpha:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=2.2)
        self.alpha_slider = ttk.Scale(self.odbp_frame, from_=-5.0, to=5.0, orient=tk.HORIZONTAL, variable=self.alpha_var, command=self.on_slider_change, length=60)
        self.alpha_slider.pack(side=tk.LEFT, padx=2)
        self.alpha_label = ttk.Label(self.odbp_frame, text="2.2")
        self.alpha_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Beta (Background Weight)
        ttk.Label(self.odbp_frame, text="Beta:").pack(side=tk.LEFT)
        self.beta_var = tk.DoubleVar(value=-1.14)
        self.beta_slider = ttk.Scale(self.odbp_frame, from_=-5.0, to=5.0, orient=tk.HORIZONTAL, variable=self.beta_var, command=self.on_slider_change, length=60)
        self.beta_slider.pack(side=tk.LEFT, padx=2)
        self.beta_label = ttk.Label(self.odbp_frame, text="-1.14")
        self.beta_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Gamma (Offset)
        ttk.Label(self.odbp_frame, text="Off:").pack(side=tk.LEFT)
        self.offset_var = tk.DoubleVar(value=0)
        self.offset_slider = ttk.Scale(self.odbp_frame, from_=-255, to=255, orient=tk.HORIZONTAL, variable=self.offset_var, command=self.on_slider_change, length=60)
        self.offset_slider.pack(side=tk.LEFT, padx=2)
        self.offset_label = ttk.Label(self.odbp_frame, text="0")
        self.offset_label.pack(side=tk.LEFT)

        # BG Subtraction Method Toggle (Gaussian vs Median)
        self.bg_method_frame = ttk.Frame(row1_frame)
        self.bg_method_var = tk.StringVar(value="Gaussian")
        ttk.Label(self.bg_method_frame, text="BG Method:").pack(side=tk.LEFT)
        self.bg_method_combo = ttk.Combobox(
            self.bg_method_frame,
            textvariable=self.bg_method_var,
            values=["Gaussian", "Median"],
            state="readonly",
            width=10
        )
        self.bg_method_combo.pack(side=tk.LEFT, padx=2)
        self.bg_method_combo.bind("<<ComboboxSelected>>", lambda e: self.update_binarization())

        # --- Lower Controls (Filters) ---
        # Split into two rows for better spacing
        self.blur_frame = ttk.Frame(self.row2_frame_ref)
        self.blur_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.filter_row2_frame = ttk.Frame(self.row2_frame_ref)
        self.filter_row2_frame.pack(fill=tk.X, pady=(5, 5))

        # --- Row 1: Pre-processing (Enhance) ---
        # CLAHE
        self.clahe_var = tk.IntVar(value=0)
        self.clahe_check = ttk.Checkbutton(self.blur_frame, text="CLAHE (Enhance)", variable=self.clahe_var, command=lambda: self.on_slider_change(None))
        self.clahe_check.pack(side=tk.LEFT, padx=(0, 5))

        # Uniform Light
        self.illum_var = tk.IntVar(value=0)
        self.illum_check = ttk.Checkbutton(
            self.blur_frame, 
            text="Uniform Light", 
            variable=self.illum_var, 
            command=self.toggle_illum_slider
        )
        self.illum_check.pack(side=tk.LEFT, padx=(0, 5))
        
        # Uniform Light Kernel Size (Hidden initially)
        self.illum_k_var = tk.IntVar(value=101)
        # Create a frame to hold the label and slider together
        self.illum_k_frame = ttk.Frame(self.blur_frame)
        # We pack this frame when needed, inside it we have label and slider
        ttk.Label(self.illum_k_frame, text="UL Kernel:").pack(side=tk.LEFT)
        self.illum_k_slider = ttk.Scale(
            self.illum_k_frame, from_=3, to=501, orient=tk.HORIZONTAL, 
            variable=self.illum_k_var, command=self.on_slider_change, length=100
        )
        self.illum_k_slider.pack(side=tk.LEFT)
        self.illum_k_label = ttk.Label(self.illum_k_frame, text="101")
        self.illum_k_label.pack(side=tk.LEFT)

        # Gamma Correction
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.gamma_label_title = ttk.Label(self.blur_frame, text="Gamma:")
        self.gamma_label_title.pack(side=tk.LEFT, padx=(10, 2))
        self.gamma_slider = ttk.Scale(
            self.blur_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, 
            variable=self.gamma_var, command=self.on_slider_change, length=100
        )
        self.gamma_slider.pack(side=tk.LEFT)
        self.gamma_label = ttk.Label(self.blur_frame, text="1.0")
        self.gamma_label.pack(side=tk.LEFT)
        
        # Contrast Correction (New)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_label_title = ttk.Label(self.blur_frame, text="Contrast:")
        self.contrast_label_title.pack(side=tk.LEFT, padx=(10, 2))
        self.contrast_slider = ttk.Scale(
            self.blur_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, 
            variable=self.contrast_var, command=self.on_slider_change, length=100
        )
        self.contrast_slider.pack(side=tk.LEFT)
        self.contrast_label = ttk.Label(self.blur_frame, text="1.0")
        self.contrast_label.pack(side=tk.LEFT)

        # --- Row 2: Post-processing (Denoise/Clean) ---
        ttk.Label(self.filter_row2_frame, text="Denoise:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=0)
        self.blur_slider = ttk.Scale(self.filter_row2_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.blur_var, command=self.on_slider_change)
        self.blur_slider.pack(side=tk.LEFT, padx=5)
        self.blur_label = ttk.Label(self.filter_row2_frame, text="0")
        self.blur_label.pack(side=tk.LEFT)

        # Bilateral (Anti-Moiré)
        self.bilateral_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.filter_row2_frame, text="Bilateral (Moiré)", variable=self.bilateral_var, command=lambda: self.on_slider_change(None)).pack(side=tk.LEFT, padx=(10, 2))

        # Median (Salt&Pepper)
        self.median_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.filter_row2_frame, text="Median (Salt&Pepper)", variable=self.median_filter_var, command=lambda: self.on_slider_change(None)).pack(side=tk.LEFT, padx=(2, 5))

        # Morphological Operations (Clean)
        # Split into Open (Remove Specks) and Close (Fill Holes)
        
        # Open (Remove Specks)
        ttk.Label(self.filter_row2_frame, text="Open (Specks):").pack(side=tk.LEFT, padx=(10, 0))
        self.morph_open_var = tk.IntVar(value=0)
        self.morph_open_slider = ttk.Scale(self.filter_row2_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.morph_open_var, command=self.on_slider_change)
        self.morph_open_slider.pack(side=tk.LEFT, padx=5)
        self.morph_open_label = ttk.Label(self.filter_row2_frame, text="0")
        self.morph_open_label.pack(side=tk.LEFT)

        # Close (Fill Holes)
        ttk.Label(self.filter_row2_frame, text="Close (Holes):").pack(side=tk.LEFT, padx=(10, 0))
        self.morph_close_var = tk.IntVar(value=0)
        self.morph_close_slider = ttk.Scale(self.filter_row2_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=self.morph_close_var, command=self.on_slider_change)
        self.morph_close_slider.pack(side=tk.LEFT, padx=5)
        self.morph_close_label = ttk.Label(self.filter_row2_frame, text="0")
        self.morph_close_label.pack(side=tk.LEFT)
        
        ttk.Label(self.filter_row2_frame, text="Min Area:").pack(side=tk.LEFT, padx=(10, 0))
        self.area_var = tk.DoubleVar(value=0)
        self.area_slider = ttk.Scale(self.filter_row2_frame, from_=0, to=5000, orient=tk.HORIZONTAL, variable=self.area_var, command=self.on_slider_change)
        self.area_slider.pack(side=tk.LEFT, padx=5)
        
        # Entry for Min Area manual input
        self.area_entry = ttk.Entry(self.filter_row2_frame, width=5, textvariable=self.area_var)
        self.area_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.area_entry.bind('<Return>', self.on_slider_change)
        self.area_entry.bind('<FocusOut>', self.on_slider_change)
        
        # Keep hidden label for compatibility with on_slider_change
        self.area_label = ttk.Label(self.filter_row2_frame, text="0")

        # ROI Shrink Ratio (for Uniform Light edge artifact removal)
        ttk.Label(self.filter_row2_frame, text="ROI Shrink:").pack(side=tk.LEFT, padx=(10, 0))
        self.roi_shrink_var = tk.DoubleVar(value=1.0)
        self.roi_shrink_slider = ttk.Scale(
            self.filter_row2_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL, 
            variable=self.roi_shrink_var, command=self.on_slider_change, length=80
        )
        self.roi_shrink_slider.pack(side=tk.LEFT, padx=5)
        self.roi_shrink_label = ttk.Label(self.filter_row2_frame, text="1.0")
        self.roi_shrink_label.pack(side=tk.LEFT, padx=(0, 5))

        self.save_original_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.filter_row2_frame, text="Save Original", variable=self.save_original_var).pack(side=tk.LEFT, padx=(10, 0))

        # Invert
        self.invert_var = tk.BooleanVar(value=False)
        self.invert_check = ttk.Checkbutton(
            self.filter_row2_frame, 
            text="Invert", 
            variable=self.invert_var, 
            command=lambda: self.on_slider_change(None)
        )
        self.invert_check.pack(side=tk.LEFT, padx=10)
        
        # Initial UI State Update
        self.update_ui_state()

        # Image Display Area (Split into Left/Middle/Right)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Original)
        self.left_panel = ttk.LabelFrame(image_frame, text="Original Image")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        self.lbl_original = ttk.Label(self.left_panel)
        self.lbl_original.pack(fill=tk.BOTH, expand=True)
        
        # Middle Panel (Selected - formerly Binarized)
        self.middle_panel = ttk.LabelFrame(image_frame, text="Selected")
        self.middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.lbl_result = ttk.Label(self.middle_panel)
        self.lbl_result.pack(fill=tk.BOTH, expand=True)
        
        # Right Panel (Unselected - New)
        self.right_panel = ttk.LabelFrame(image_frame, text="Unselected")
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        self.lbl_unselected = ttk.Label(self.right_panel)
        self.lbl_unselected.pack(fill=tk.BOTH, expand=True)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            # Read image using OpenCV
            loaded_img = cv2.imread(file_path)
            if loaded_img is not None:
                self.original_cv_image = loaded_img
                self.apply_resize()

    def apply_resize(self, event=None):
        if self.original_cv_image is None: return
        
        resize_map = {
            "Original": 1.0,
            "1/2": 0.5,
            "1/4": 0.25,
            "1/8": 0.125
        }
        scale_txt = self.resize_var.get()
        scale = resize_map.get(scale_txt, 1.0)
        
        if scale == 1.0:
            self.cv_image = self.original_cv_image.copy()
        else:
            h, w = self.original_cv_image.shape[:2]
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Use INTER_AREA for downscaling for better quality
            self.cv_image = cv2.resize(self.original_cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Convert to Grayscale for binarization
        self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        
        # Display Original Image (Resized version)
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.display_image(rgb_image, self.lbl_original)
        
        # Initial Processing
        self.update_binarization()

    def on_method_change(self, event):
        self.update_ui_state()
        self.update_binarization()

    def toggle_illum_slider(self):
        """Toggle visibility of Uniform Light Kernel slider"""
        if self.illum_var.get():
            # Find the widget that should be AFTER the slider
            # It's the "Gamma:" label.
            children = self.blur_frame.winfo_children()
            target = None
            for child in children:
                if isinstance(child, ttk.Label) and child.cget("text") == "Gamma:":
                    target = child
                    break
            
            if target:
                self.illum_k_frame.pack(side=tk.LEFT, padx=(5, 0), before=target)
            else:
                self.illum_k_frame.pack(side=tk.LEFT, padx=(5, 0))
        else:
            self.illum_k_frame.pack_forget()
        
        self.on_slider_change(None)

    def update_ui_state(self):
        method = self.method_var.get()
        
        # Reset ranges based on method
        self.hsv_frame.pack_forget() # Default hidden
        self.odbp_frame.pack_forget()
        self.block_frame.pack_forget()
        self.rgb_frame.pack_forget()
        self.thresh_frame.pack_forget()
        self.bg_method_frame.pack_forget()
        
        # Hide all lower controls by default
        self.blur_frame.pack_forget()
        self.filter_row2_frame.pack_forget()

        # Helper to show lower controls
        def show_lower_controls(show_pre=True, show_post=True):
            if show_pre:
                self.blur_frame.pack(fill=tk.X, pady=(5, 0))
            else:
                self.blur_frame.pack_forget()
                
            if show_post:
                self.filter_row2_frame.pack(fill=tk.X, pady=(5, 5))
            else:
                self.filter_row2_frame.pack_forget()

        if method == "Global Threshold":
            self.thresh_label_txt.set("Threshold:")
            self.slider.configure(from_=0, to=255)
            # Restore reasonable default if out of range
            if self.slider_var.get() < 0: self.slider_var.set(127)
            self.thresh_frame.pack(side=tk.LEFT, padx=5) # Ensure threshold is shown
            show_lower_controls(show_pre=True, show_post=True)
            
        elif method in ["Adaptive Gaussian", "Background Subtraction", "Combined (BG Sub & Median)", "Morph Top-Hat (Bright)", "Morph Black-Hat (Dark)"]:
            self.thresh_label_txt.set("Threshold:")
            self.slider.configure(from_=0, to=255)
            # Restore previous slider value if valid
            
            self.thresh_frame.pack(side=tk.LEFT, padx=5)
            
            if method in ["Combined (BG Sub & Median)", "Morph Top-Hat (Bright)", "Morph Black-Hat (Dark)"]:
                self.lbl_block_title.config(text="Kernel Size:")
            else:
                self.lbl_block_title.config(text="Block Size:")
            
            if self.block_var.get() < 3: self.block_var.set(49)
            
            if self.thresh_frame.winfo_manager():
                self.block_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            else:
                self.block_frame.pack(side=tk.LEFT, padx=5)
            
            # Show BG Method toggle for Background Subtraction
            if method == "Background Subtraction":
                self.bg_method_frame.pack(side=tk.LEFT, padx=5)

            # Morph modes use Denoise (Blur) but NOT CLAHE/Illum/Gamma/Contrast in current logic
            if method in ["Morph Top-Hat (Bright)", "Morph Black-Hat (Dark)"]:
                show_lower_controls(show_pre=False, show_post=True)
            else:
                show_lower_controls(show_pre=True, show_post=True)

        elif method == "Color Filter (HSV)":
            self.thresh_label_txt.set("Tolerance:")
            self.slider.configure(from_=0, to=255)
            # Default tolerance
            if self.slider_var.get() == 127: self.slider_var.set(30)
            self.thresh_frame.pack(side=tk.LEFT, padx=5)
            if self.thresh_frame.winfo_manager():
                self.rgb_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            else:
                self.rgb_frame.pack(side=tk.LEFT, padx=5)
            # Color Filter uses Denoise and Illumination Correction, but not CLAHE/Gamma/Contrast usually?
            # Actually code uses Denoise and Illumination Correction.
            # It does NOT use CLAHE, Gamma, Contrast.
            # It DOES use Invert (for unselected display logic).
            # It does NOT use Clean/Morph or Min Area in the main logic block (lines 614+).
            # Wait, lines 614+ is just the binarization logic.
            # The post-processing (Clean, Min Area, Invert) is applied AFTER binarization (lines 998+).
            # So Post-processing controls ARE relevant.
            # Pre-processing controls: Denoise and Illumination are used. CLAHE/Gamma/Contrast are NOT used in Color Filter block.
            # So we should hide CLAHE, Gamma, Contrast for Color Filter.
            # But `show_lower_controls` is all-or-nothing for rows.
            # Let's refine `show_lower_controls` or just hide specific widgets.
            show_lower_controls(show_pre=True, show_post=True)
            # Hide irrelevant pre-processing
            self.clahe_var.set(0) # Disable
            # We can't easily hide individual widgets inside the packed frame without repacking everything.
            # But we can disable them or just leave them visible if they *could* be useful (e.g. Gamma on color image before filtering?)
            # The code for Color Filter (lines 614+) explicitly implements Denoise and Illumination.
            # It does NOT implement Gamma/Contrast/CLAHE.
            # So those sliders do nothing. We should hide them.
            
            # To hide specific widgets in `blur_frame`:
            # We need to unpack them.
            # This suggests we should repack `blur_frame` content dynamically.
            self.repack_blur_frame(show_clahe=False, show_illum=True, show_gamma=False, show_contrast=False)


        elif method == "HSV Detector":
            self.hsv_frame.pack(fill=tk.X, pady=2, before=self.row2_frame_ref)
            # HSV Detector uses Denoise, Gamma, Contrast, Illumination, CLAHE.
            # So all pre-processing is relevant.
            show_lower_controls(show_pre=True, show_post=True)
            self.repack_blur_frame(show_clahe=True, show_illum=True, show_gamma=True, show_contrast=True)

        elif method == "ODBP (Weighted Diff)":
            self.thresh_label_txt.set("Threshold:")
            self.slider.configure(from_=0, to=255)
            self.thresh_frame.pack(side=tk.LEFT, padx=5)
            
            self.lbl_block_title.config(text="Median Kernel:")
            if self.block_var.get() < 3: self.block_var.set(49)
            
            if self.thresh_frame.winfo_manager():
                self.block_frame.pack(side=tk.LEFT, padx=5, before=self.thresh_frame)
            else:
                self.block_frame.pack(side=tk.LEFT, padx=5)
            
            self.odbp_frame.pack(side=tk.LEFT, padx=5)
            
            # ODBP uses Denoise (Blur).
            # It does NOT use CLAHE, Illumination, Gamma, Contrast in its block (lines 772+).
            # So hide Pre-processing row except maybe Denoise?
            # Denoise is in Row 2 (Post-processing frame) in the UI setup, but logically it's pre-processing.
            # Wait, UI Setup:
            # Row 1 (blur_frame): CLAHE, Uniform Light, Gamma, Contrast.
            # Row 2 (filter_row2_frame): Denoise (Blur), Bilateral, Median, Clean (Morph), Min Area, Save, Invert.
            
            # ODBP code uses:
            # - Denoise (Blur) -> Row 2
            # - Median Kernel (Block Size) -> Row 1 Control
            # - Alpha/Beta/Offset -> Row 1 Control
            # - Threshold -> Row 1 Control
            
            # It does NOT use CLAHE, Illumination, Gamma, Contrast.
            # So Row 1 (blur_frame) should be HIDDEN or empty.
            show_lower_controls(show_pre=False, show_post=True)

        else:
            # Default for others (Global Threshold, Adaptive, etc.)
            # They use the "Grayscale processing for other methods" block (lines 897+).
            # This block uses: Illumination, CLAHE, Gamma, Contrast, Denoise (Blur/Bilateral/Median).
            # So ALL controls are relevant.
            show_lower_controls(show_pre=True, show_post=True)
            self.repack_blur_frame(show_clahe=True, show_illum=True, show_gamma=True, show_contrast=True)

    def repack_blur_frame(self, show_clahe=True, show_illum=True, show_gamma=True, show_contrast=True):
        # Clear current packing
        for child in self.blur_frame.winfo_children():
            child.pack_forget()
            
        # Repack based on flags
        if hasattr(self, 'clahe_check') and show_clahe: self.clahe_check.pack(side=tk.LEFT, padx=(0, 5))
        if hasattr(self, 'illum_check') and show_illum: 
            self.illum_check.pack(side=tk.LEFT, padx=(0, 5))
            # Handle slider visibility if checked
            if self.illum_var.get():
                self.illum_k_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        if hasattr(self, 'gamma_label') and show_gamma:
            self.gamma_label_title.pack(side=tk.LEFT, padx=(10, 2))
            self.gamma_slider.pack(side=tk.LEFT)
            self.gamma_label.pack(side=tk.LEFT)
            
        if hasattr(self, 'contrast_label') and show_contrast:
            self.contrast_label_title.pack(side=tk.LEFT, padx=(10, 2))
            self.contrast_slider.pack(side=tk.LEFT)
            self.contrast_label.pack(side=tk.LEFT)

    def on_slider_change(self, event):
        # Update labels
        self.value_label.config(text=str(int(self.slider_var.get())))
        self.blur_label.config(text=str(int(self.blur_var.get())))
        
        if hasattr(self, 'morph_open_label'):
            self.morph_open_label.config(text=str(int(self.morph_open_var.get())))
        if hasattr(self, 'morph_close_label'):
            self.morph_close_label.config(text=str(int(self.morph_close_var.get())))
            
        self.area_label.config(text=str(int(self.area_var.get())))
        
        if hasattr(self, 'alpha_label'):
            self.alpha_label.config(text=f"{self.alpha_var.get():.1f}")
            self.beta_label.config(text=f"{self.beta_var.get():.1f}")
            self.offset_label.config(text=f"{self.offset_var.get():.0f}")
        
        # Update Gamma Label
        if hasattr(self, 'gamma_var'):
            self.gamma_label.config(text=f"{self.gamma_var.get():.1f}")
            
        # Update Contrast Label
        if hasattr(self, 'contrast_var'):
            self.contrast_label.config(text=f"{self.contrast_var.get():.1f}")

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
        
        # Ensure Uniform Light Kernel is odd
        if hasattr(self, 'illum_k_var'):
            k_val = int(self.illum_k_var.get())
            if k_val % 2 == 0:
                k_val += 1
                self.illum_k_var.set(k_val)
            self.illum_k_label.config(text=str(k_val))

        # Update ROI Shrink Label
        if hasattr(self, 'roi_shrink_var'):
            self.roi_shrink_label.config(text=f"{self.roi_shrink_var.get():.2f}")

        if self.gray_image is not None:
            self.update_binarization()
            
    def preprocess_channel(self, img, skip_denoise=False):
        """
        Apply Illumination Correction, CLAHE, and Denoise to a single channel image.
        """
        if img is None: return None
        out = img.copy()
        
        # 1. Illumination Correction
        if hasattr(self, 'illum_var') and self.illum_var.get():
            k_illum = 51
            kernel_illum = cv2.getStructuringElement(cv2.MORPH_RECT, (k_illum, k_illum))
            background_illum = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_illum)
            diff_illum = cv2.subtract(out, background_illum)
            out = cv2.add(diff_illum, 127)

        # 2. CLAHE
        if hasattr(self, 'clahe_var') and self.clahe_var.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            out = clahe.apply(out)
            
        # 3. Denoise (Blur)
        if not skip_denoise:
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                if hasattr(self, 'bilateral_var') and self.bilateral_var.get():
                    d = blur_amount * 2 + 1
                    sigma = blur_amount * 15
                    out = cv2.bilateralFilter(out, d, sigma, sigma)
                else:
                    ksize = blur_amount * 2 + 1
                    out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        
        return out

    def update_binarization(self):
        if self.cv_image is None:
            return
            
        method = self.method_var.get()
        val = int(self.slider_var.get())
        
        binary = None

        if method == "Color Filter (HSV)":
            # 1. Pre-processing: Denoise (Color)
            img_to_process = self.cv_image.copy()
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            # --- Pre-processing: Illumination Correction (Uniform Light) ---
            if hasattr(self, 'illum_var') and self.illum_var.get():
                # Correct V and S channels in HSV, then convert back to BGR
                img_hsv_temp = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img_hsv_temp)
                
                # Use Gaussian Blur for background estimation (smoother than Morph Open)
                k_illum = 101 # Default Large kernel
                if hasattr(self, 'illum_k_var'):
                    k_illum = int(self.illum_k_var.get())
                if k_illum % 2 == 0: k_illum += 1
                
                # Correct V
                bg_v = cv2.GaussianBlur(v, (k_illum, k_illum), 0)
                diff_v = cv2.subtract(v, bg_v)
                v_corr = cv2.add(diff_v, 127)
                
                # Correct S (Saturation)
                bg_s = cv2.GaussianBlur(s, (k_illum, k_illum), 0)
                diff_s = cv2.subtract(s, bg_s)
                s_corr = cv2.add(diff_s, 127)
                
                img_hsv_corr = cv2.merge([h, s_corr, v_corr])
                img_to_process = cv2.cvtColor(img_hsv_corr, cv2.COLOR_HSV2BGR)

            # Convert to RGB (OpenCV is BGR)
            img_rgb = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
            
            # Parse Target RGB
            try:
                r_val = int(self.r_var.get())
                g_val = int(self.g_var.get())
                b_val = int(self.b_var.get())
                target_rgb = np.array([r_val, g_val, b_val], dtype=np.float32)
            except ValueError:
                target_rgb = np.array([0, 0, 0], dtype=np.float32)
            
            # Calculate Distance
            # Convert images to float32 to avoid overflow/underflow during subtraction
            diff = img_rgb.astype(np.float32) - target_rgb
            dist = np.linalg.norm(diff, axis=2)
            
            # Apply Threshold based on Proximity
            proximity = self.prox_var.get()
            if proximity == "Far":
                # Select pixels > Tolerance (Far from color)
                binary = np.where(dist > val, 255, 0).astype(np.uint8)
            else:
                # Near (Default): Select pixels <= Tolerance (Close to color)
                binary = np.where(dist <= val, 255, 0).astype(np.uint8)

            # Exclude originally black pixels (Rule: If originally full black, it is invalid/unselected)
            # Use original image (before blur) to check for black
            gray_orig = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            # Create mask where pixels are NOT black (> 0)
            _, non_black_mask = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY)
            # Apply AND: Keep only selected pixels that were also NOT black
            binary = cv2.bitwise_and(binary, non_black_mask)

        elif method == "HSV Detector":
            # 1. Pre-processing: Denoise (Color)
            img_to_process = self.cv_image.copy()
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            # --- Pre-processing: Gamma Correction ---
            if hasattr(self, 'gamma_var'):
                gamma = self.gamma_var.get()
                if gamma != 1.0:
                    invGamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    img_to_process = cv2.LUT(img_to_process, table)

            # --- Pre-processing: Contrast Correction ---
            if hasattr(self, 'contrast_var'):
                contrast = self.contrast_var.get()
                if contrast != 1.0:
                    # Alpha = contrast, Beta = 0
                    img_to_process = cv2.convertScaleAbs(img_to_process, alpha=contrast, beta=0)

            # Convert to HSV
            img_hsv = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2HSV)
            
            # --- Pre-processing: Illumination Correction (Uniform Light) ---
            if hasattr(self, 'illum_var') and self.illum_var.get():
                # Correct V and S channels
                h, s, v = cv2.split(img_hsv)
                
                # Use Gaussian Blur for background estimation
                k_illum = 101 # Default Large kernel
                if hasattr(self, 'illum_k_var'):
                    k_illum = int(self.illum_k_var.get())
                if k_illum % 2 == 0: k_illum += 1
                
                # Correct V
                bg_v = cv2.GaussianBlur(v, (k_illum, k_illum), 0)
                diff_v = cv2.subtract(v, bg_v)
                v_corr = cv2.add(diff_v, 127)
                
                # Correct S
                bg_s = cv2.GaussianBlur(s, (k_illum, k_illum), 0)
                diff_s = cv2.subtract(s, bg_s)
                s_corr = cv2.add(diff_s, 127)
                
                # Update img_hsv with corrected channels so inRange uses it
                img_hsv = cv2.merge([h, s_corr, v_corr])

            # --- Pre-processing: CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
            if hasattr(self, 'clahe_var') and self.clahe_var.get():
                h, s, v = cv2.split(img_hsv)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                v_clahe = clahe.apply(v)
                img_hsv = cv2.merge([h, s, v_clahe])
            
            # Get Thresholds
            h_min = self.h_min_var.get()
            h_max = self.h_max_var.get()
            s_min = self.s_min_var.get()
            s_max = self.s_max_var.get()
            v_min = self.v_min_var.get()
            v_max = self.v_max_var.get()
            
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            
            binary = cv2.inRange(img_hsv, lower, upper)

        elif method == "ODBP (Weighted Diff)":
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # Pre-processing
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
                
            k_size = int(self.block_var.get())
            if k_size < 3: k_size = 3
            if k_size % 2 == 0: k_size += 1
            
            # Dynamic Weighted Difference
            alpha = self.alpha_var.get()
            beta = self.beta_var.get()
            offset = self.offset_var.get()
            
            res_med = cv2.medianBlur(img_to_process, k_size)
            
            # Formula: Alpha * Img + Beta * Med + Offset
            res_diff = cv2.addWeighted(img_to_process, alpha, res_med, beta, offset)
            
            _, binary = cv2.threshold(res_diff, val, 255, cv2.THRESH_BINARY)

        elif method == "Background Subtraction":
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # Pre-processing
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
                
            block_size = int(self.block_var.get())
            if block_size < 3: block_size = 3
            if block_size % 2 == 0: block_size += 1
            
            # Check BG Method
            bg_method = self.bg_method_var.get()
            
            if bg_method == "Median":
                # Estimate background using Median Blur (Robust to salt-and-pepper noise)
                bg = cv2.medianBlur(img_to_process, block_size)
            else:
                # Estimate background using Gaussian Blur (Default)
                bg = cv2.GaussianBlur(img_to_process, (block_size, block_size), 0)
            
            # Calculate difference: Background - Original (Assuming dark defects on bright BG)
            # Or absdiff to catch both
            # Let's stick to subtraction to match BG Sub behavior, but maybe AbsDiff is safer?
            # User wants "circles". If circles are white on black, then BG - Img is 0. Img - BG is positive.
            # Standard BG Sub implemented below uses: diff = cv2.subtract(bg, img_to_process)
            # This implies Dark Defects.
            # IF the user sees circles with BG Sub, maybe they are dark?
            # Let's provide ABSOLUTE difference for this new robust mode to be safe.
            diff = cv2.absdiff(bg, img_to_process)
            
            _, binary = cv2.threshold(diff, val, 255, cv2.THRESH_BINARY)

        elif method == "Combined (BG Sub & Median)":
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # Pre-processing
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            k_size = int(self.block_var.get())
            if k_size < 3: k_size = 3
            if k_size % 2 == 0: k_size += 1
            
            # 1. Background Subtraction (Gaussian)
            bg_gauss = cv2.GaussianBlur(img_to_process, (k_size, k_size), 0)
            # Using absdiff to be safer and match potential user intent
            diff_gauss = cv2.absdiff(bg_gauss, img_to_process) 
            _, bin_gauss = cv2.threshold(diff_gauss, val, 255, cv2.THRESH_BINARY)
            
            # 2. Median Difference (ODBP Formula)
            res_med = cv2.medianBlur(img_to_process, k_size)
            res_in_diff = cv2.addWeighted(img_to_process, 2.2, res_med, -1.14, 0)
            _, bin_med = cv2.threshold(res_in_diff, val, 255, cv2.THRESH_BINARY)
            
            # 3. Intersection
            binary = cv2.bitwise_and(bin_gauss, bin_med)

        elif method == "Morph Top-Hat (Bright)":
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # Pre-processing
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
                
            k_size = int(self.block_var.get())
            if k_size < 3: k_size = 3
            if k_size % 2 == 0: k_size += 1
            
            # Top-Hat: Image - Opening(Image). 
            # Removes background (Opening) and keeps bright features smaller than kernel.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            tophat = cv2.morphologyEx(img_to_process, cv2.MORPH_TOPHAT, kernel)
            
            _, binary = cv2.threshold(tophat, val, 255, cv2.THRESH_BINARY)
            
        elif method == "Morph Black-Hat (Dark)":
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # Pre-processing
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
                
            k_size = int(self.block_var.get())
            if k_size < 3: k_size = 3
            if k_size % 2 == 0: k_size += 1
            
            # Black-Hat: Closing(Image) - Image.
            # Removes background (Closing) and keeps dark features smaller than kernel.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            blackhat = cv2.morphologyEx(img_to_process, cv2.MORPH_BLACKHAT, kernel)
            
            _, binary = cv2.threshold(blackhat, val, 255, cv2.THRESH_BINARY)

        else:
            # Grayscale processing for other methods
            if self.gray_image is None: return
            img_to_process = self.gray_image.copy()
            
            # --- -1. Pre-processing: Illumination Correction (Uniform Light) ---
            if hasattr(self, 'illum_var') and self.illum_var.get():
                # Estimate background using Gaussian Blur (Low Pass Filter)
                # This is faster and often smoother than Morph Opening for illumination
                k_illum = 101 # Default Large kernel
                if hasattr(self, 'illum_k_var'):
                    k_illum = int(self.illum_k_var.get())
                if k_illum % 2 == 0: k_illum += 1
                background_illum = cv2.GaussianBlur(img_to_process, (k_illum, k_illum), 0)
                
                # Subtract background: Image - Background + 127
                # This centers the result around gray (127)
                diff_illum = cv2.subtract(img_to_process, background_illum)
                img_to_process = cv2.add(diff_illum, 127)
                
            # --- 0. Pre-processing: CLAHE (Enhance weak features) ---
            if hasattr(self, 'clahe_var') and self.clahe_var.get():
                # ClipLimit=2.0, TileGridSize=(8,8) is standard.
                # We can make it stronger if needed, but fixed is usually okay.
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_to_process = clahe.apply(img_to_process)

            # --- 0.5. Pre-processing: Gamma Correction ---
            if hasattr(self, 'gamma_var'):
                gamma = self.gamma_var.get()
                if gamma != 1.0:
                    invGamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    img_to_process = cv2.LUT(img_to_process, table)

            # --- 0.6. Pre-processing: Contrast Correction ---
            if hasattr(self, 'contrast_var'):
                contrast = self.contrast_var.get()
                if contrast != 1.0:
                    # Alpha = contrast, Beta = 0
                    img_to_process = cv2.convertScaleAbs(img_to_process, alpha=contrast, beta=0)
            
            # 1. Pre-processing: Denoise (Grayscale)
            blur_amount = self.blur_var.get()
            if blur_amount > 0:
                ksize = blur_amount * 2 + 1
                
                # Median Filter (Best for Salt-and-Pepper noise)
                if hasattr(self, 'median_filter_var') and self.median_filter_var.get():
                    img_to_process = cv2.medianBlur(img_to_process, ksize)
                
                # Bilateral Filter (Anti-Moiré, preserves edges)
                elif self.bilateral_var.get():
                    # d: Diameter of each pixel neighborhood (e.g. 5, 9, ...)
                    # sigmaColor: Filter sigma in the color space (larger -> mix disparate colors)
                    # sigmaSpace: Filter sigma in the coordinate space (larger -> influence from farther pixels)
                    # We map 'blur_amount' (0-10) to reasonable parameters.
                    d = ksize 
                    sigmaColor = blur_amount * 15 # e.g. 15 to 150
                    sigmaSpace = blur_amount * 15 # e.g. 15 to 150
                    img_to_process = cv2.bilateralFilter(img_to_process, d, sigmaColor, sigmaSpace)
                else:
                    # Standard Gaussian Blur
                    img_to_process = cv2.GaussianBlur(img_to_process, (ksize, ksize), 0)
            
            # 2. Binarization
            if method == "Global Threshold":
                _, binary = cv2.threshold(img_to_process, val, 255, cv2.THRESH_BINARY)
                
            elif method == "Adaptive Gaussian":
                block_size = int(self.block_var.get())
                if block_size < 3: block_size = 3
                if block_size % 2 == 0: block_size += 1
                
                # C is the value subtracted from the mean/weighted sum
                binary = cv2.adaptiveThreshold(
                    img_to_process, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    block_size, 
                    val
                )
        
        if binary is None:
            return

        # 3. Post-processing: Clean Specks (Morphological Opening)
        morph_open_amount = self.morph_open_var.get()
        if morph_open_amount > 0:
            ksize = morph_open_amount * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            # Use MORPH_OPEN (Erosion then Dilation) to remove small white noise (specks)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 3.5. Post-processing: Fill Holes (Morphological Closing)
        morph_close_amount = self.morph_close_var.get()
        if morph_close_amount > 0:
            ksize = morph_close_amount * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            # Use MORPH_CLOSE (Dilation then Erosion) to fill small black holes inside objects
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 4. Area Filtering (Remove small connected components)
        min_area = self.area_var.get()
        if min_area > 0:
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create a new mask for valid contours
            mask = np.zeros_like(binary)
            for cnt in contours:
                if cv2.contourArea(cnt) >= min_area:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            binary = mask

        # 4.5. ROI Shrink (Remove edge artifacts from Uniform Light)
        roi_shrink_ratio = self.roi_shrink_var.get()
        if roi_shrink_ratio < 1.0 and roi_shrink_ratio > 0:
            # Find the bounding box of all white pixels
            white_pixels = np.where(binary > 0)
            if len(white_pixels[0]) > 0:
                y_min, y_max = white_pixels[0].min(), white_pixels[0].max()
                x_min, x_max = white_pixels[1].min(), white_pixels[1].max()
                
                # Calculate center and new dimensions
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                width = x_max - x_min
                height = y_max - y_min
                
                new_width = int(width * roi_shrink_ratio)
                new_height = int(height * roi_shrink_ratio)
                
                new_x_min = max(0, center_x - new_width // 2)
                new_x_max = min(binary.shape[1], center_x + new_width // 2)
                new_y_min = max(0, center_y - new_height // 2)
                new_y_max = min(binary.shape[0], center_y + new_height // 2)
                
                # Create mask for inner ROI
                roi_mask = np.zeros_like(binary)
                roi_mask[new_y_min:new_y_max, new_x_min:new_x_max] = 255
                
                # Apply mask to binary image
                binary = cv2.bitwise_and(binary, roi_mask)

        # 5. Invert if requested
        if self.invert_var.get():
            binary = cv2.bitwise_not(binary)

        self.processed_image = binary
        
        # Display Result (Selected)
        self.display_image(binary, self.lbl_result)
        
        # Calculate and Display Unselected (Inverted Mask)
        if hasattr(self, 'lbl_unselected'):
             # Start with inverted selection (potential unselected candidates)
             unselected = cv2.bitwise_not(binary)
             
             # If using Color Filter, apply the same rule: Originally black pixels must stay black
             if method == "Color Filter":
                 # Use existing non_black_mask if calculated above, or recalculate
                 # We need to make sure 'non_black_mask' is available here.
                 # It's defined inside the if block above. Let's recalculate to be safe and clean.
                 gray_orig = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
                 _, non_black_mask = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY)
                 
                 # Apply mask to unselected as well
                 unselected = cv2.bitwise_and(unselected, non_black_mask)
                 
             self.display_image(unselected, self.lbl_unselected)
        
    def display_image(self, cv_img, label_widget):
        # Resize image to fit the label/window while maintaining aspect ratio
        h, w = cv_img.shape[:2]
        
        # Get widget dimensions (approximate if not yet drawn)
        # We'll use a fixed max size for simplicity or dynamic if possible
        max_w = 380  # Reduced to fit 3 panels
        max_h = 600
        
        scale = min(max_w/w, max_h/h)
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
