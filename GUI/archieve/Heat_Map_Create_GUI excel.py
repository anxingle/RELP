import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image, ImageTk
import threading

class HeatMapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heatmap Generator")
        self.root.geometry("1000x800")
        
        # Variables
        self.base_image_path = tk.StringVar()
        self.data_source_path = tk.StringVar()
        self.data_source_type = tk.StringVar(value="file") # 'file' or 'folder'
        self.merge_data_var = tk.BooleanVar(value=False)
        self.opacity_var = tk.DoubleVar(value=0.6)
        
        self.click_points = [] # Stores (x, y) tuples in ORIGINAL image coordinates
        self.scale_ul = None # (x, y) in ORIGINAL image coordinates
        self.scale_lr = None # (x, y) in ORIGINAL image coordinates
        self.dragging_point_index = None
        
        self.original_image = None # CV2 image (BGR)
        self.display_image = None # PIL Image for display (Resized)
        self.display_scale = 1.0 # Scale factor (Display / Original)
        
        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure Grid Layout for 7:3 Split
        main_frame.columnconfigure(0, weight=7) # Left (Image)
        main_frame.columnconfigure(1, weight=3) # Right (Controls + Log)
        main_frame.rowconfigure(0, weight=1)
        
        # --- Left Panel: Image Canvas ---
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Canvas Setup
        v_scroll = ttk.Scrollbar(left_panel, orient=tk.VERTICAL)
        h_scroll = ttk.Scrollbar(left_panel, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(left_panel, bg="gray", xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)
        
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # --- Right Panel: Controls & Log ---
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # 2. Log & Action (Bottom of Right Panel)
        bottom_frame = ttk.Frame(right_panel)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)
        
        # Log Area (Left side of bottom frame)
        log_group = ttk.LabelFrame(bottom_frame, text="Log", padding="5")
        log_group.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_group, height=6, width=30) # Reduced height
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_group, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set)
        
        # Run Button & Save Button
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Use Grid layout for equal sizing
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.rowconfigure(0, weight=1)
        
        self.run_btn = tk.Button(btn_frame, text="Generate\nHeatmap", command=self.start_processing,
                                 font=("Arial", 14, "bold"),
                                 bg="#90EE90", # Light Green
                                 highlightbackground="#90EE90", # For Mac
                                 relief="solid", borderwidth=2)
        self.run_btn.grid(row=0, column=0, sticky="nsew", padx=(0, 5), ipadx=10)
        
        # Save Button: Transparent bg (System default), Orange Text, Orange Border
        self.save_btn = tk.Button(btn_frame, text="Save\nResult", command=self.prompt_save_result, state="disabled",
                                 font=("Arial", 14, "bold"),
                                 bg="white", 
                                 fg="black", # Black Text
                                 disabledforeground="gray", # Gray text when disabled
                                 highlightbackground="#FFA500", # Orange Border (Mac)
                                 highlightthickness=2,
                                 relief="flat", borderwidth=0) # Flat relief, border via highlightthickness
        # Note: On standard Tkinter, bg cannot be fully transparent, but omitting it uses system default.
        # On Mac 'highlightbackground' controls the border color effectively for flat buttons or 'solid' relief.
        
        self.save_btn.grid(row=0, column=1, sticky="nsew", ipadx=10)

        # 1. Settings Frame (Top of Right Panel)
        control_frame = ttk.LabelFrame(right_panel, text="Settings", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Base Image
        ttk.Label(control_frame, text="Base Image:").pack(anchor="w", pady=(5, 0))
        img_entry_frame = ttk.Frame(control_frame)
        img_entry_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(img_entry_frame, textvariable=self.base_image_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(img_entry_frame, text="Browse", command=self.browse_image).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Data Source
        ttk.Label(control_frame, text="Data Source:").pack(anchor="w", pady=(5, 0))
        data_entry_frame = ttk.Frame(control_frame)
        data_entry_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(data_entry_frame, textvariable=self.data_source_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="File", command=lambda: self.browse_data("file")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Folder", command=lambda: self.browse_data("folder")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Options
        ttk.Checkbutton(control_frame, text="Merge files", variable=self.merge_data_var, command=self.on_merge_change).pack(anchor="w", pady=5)
        
        opacity_frame = ttk.Frame(control_frame)
        opacity_frame.pack(fill=tk.X, pady=5)
        ttk.Label(opacity_frame, text="Opacity:").pack(side=tk.LEFT)
        self.opacity_scale = ttk.Scale(opacity_frame, from_=0.1, to=1.0, variable=self.opacity_var, orient=tk.HORIZONTAL)
        self.opacity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Coord Info
        self.coord_label = ttk.Label(control_frame, text="Select range on image", wraplength=280)
        self.coord_label.pack(fill=tk.X, pady=5)
        
        # 3. Filters Frame (Middle of Right Panel - Takes remaining space)
        self.filter_frame_container = ttk.LabelFrame(right_panel, text="Filters", padding="5")
        self.filter_frame_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Filter Header with Refresh Button
        filter_header_frame = ttk.Frame(self.filter_frame_container)
        filter_header_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(filter_header_frame, text="Filter Options").pack(side=tk.LEFT)
        ttk.Button(filter_header_frame, text="Refresh", command=self.trigger_scan_filters, width=8).pack(side=tk.RIGHT)
        
        # Scrollable Filter Area
        self.filter_canvas = tk.Canvas(self.filter_frame_container)
        self.filter_scrollbar = ttk.Scrollbar(self.filter_frame_container, orient="vertical", command=self.filter_canvas.yview)
        self.filter_scrollable_frame = ttk.Frame(self.filter_canvas)
        
        self.filter_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.filter_canvas.configure(
                scrollregion=self.filter_canvas.bbox("all")
            )
        )
        
        self.filter_canvas.create_window((0, 0), window=self.filter_scrollable_frame, anchor="nw")
        self.filter_canvas.configure(yscrollcommand=self.filter_scrollbar.set)
        
        self.filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        self.filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def trigger_scan_filters(self):
        # UI update to show loading
        for widget in self.filter_scrollable_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.filter_scrollable_frame, text="Loading filters...", foreground="blue").pack(anchor="w", padx=5, pady=5)
        
        threading.Thread(target=self.scan_data_filters, daemon=True).start()

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.base_image_path.set(path)
            self.load_image(path)

    def browse_data(self, mode):
        if mode == "file":
            path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
            if path:
                self.data_source_path.set(path)
                self.data_source_type.set("file")
                self.log(f"Selected File: {os.path.basename(path)}")
                self.trigger_scan_filters()
        else:
            path = filedialog.askdirectory()
            if path:
                self.data_source_path.set(path)
                self.data_source_type.set("folder")
                # Count files
                count = 0
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file == "parametric_out.xlsx":
                            count += 1
                self.log(f"Selected Folder: {path}")
                self.log(f"Found {count} parametric_out.xlsx files")
                self.trigger_scan_filters()

    def load_image(self, path):
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                raise ValueError("Cannot read image")
            
            # Update idle tasks to ensure canvas dimensions are available
            self.root.update_idletasks()
            
            # Get current canvas dimensions or fallback
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w < 100: canvas_w = 800
            if canvas_h < 100: canvas_h = 500
            
            # Add margin
            max_w = canvas_w - 20
            max_h = canvas_h - 20
            
            h, w = self.original_image.shape[:2]
            
            scale_w = max_w / w
            scale_h = max_h / h
            self.display_scale = min(scale_w, scale_h, 1.0) # Downscale only
            
            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            
            img_resized = cv2.resize(self.original_image, (new_w, new_h))
            
            # Convert to RGB for display
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            self.display_image = Image.fromarray(img_rgb)
            
            # Reset clicks
            self.click_points = []
            self.scale_ul = None
            self.scale_lr = None
            self.update_coord_label()
            
            self.redraw_canvas()
            self.log(f"Image loaded. Original: {w}x{h}, Display: {new_w}x{new_h} (Scale: {self.display_scale:.2f})")
            
        except Exception as e:
            self.log(f"Failed to load image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def redraw_canvas(self):
        if self.display_image:
            self.tk_img = ImageTk.PhotoImage(self.display_image)
            self.canvas.config(scrollregion=(0, 0, self.tk_img.width(), self.tk_img.height()))
            
            # Calculate centered position
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w = self.tk_img.width()
            img_h = self.tk_img.height()
            
            x_pos = max(0, (canvas_w - img_w) // 2)
            y_pos = max(0, (canvas_h - img_h) // 2)
            
            self.canvas.delete("all") # Clear previous
            self.canvas.create_image(x_pos, y_pos, image=self.tk_img, anchor="nw")
            
            # Draw points and rect (Convert Original -> Display + Offset)
            for i, pt in enumerate(self.click_points):
                x = int(pt[0] * self.display_scale) + x_pos
                y = int(pt[1] * self.display_scale) + y_pos
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white", width=2)
                self.canvas.create_text(x, y-15, text=str(i+1), fill="red", font=("Arial", 12, "bold"))
            
            if len(self.click_points) == 2:
                p1 = self.click_points[0]
                p2 = self.click_points[1]
                
                # Determine rect bounds in display coords
                x1 = int(p1[0] * self.display_scale) + x_pos
                y1 = int(p1[1] * self.display_scale) + y_pos
                x2 = int(p2[0] * self.display_scale) + x_pos
                y2 = int(p2[1] * self.display_scale) + y_pos
                
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, dash=(5, 5))

    def get_image_coords(self, event):
        if self.display_image is None:
            return None, None, None, None

        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w = self.tk_img.width()
        img_h = self.tk_img.height()
        
        x_pos = max(0, (canvas_w - img_w) // 2)
        y_pos = max(0, (canvas_h - img_h) // 2)
        
        rel_x = cx - x_pos
        rel_y = cy - y_pos
        
        ox = int(rel_x / self.display_scale)
        oy = int(rel_y / self.display_scale)
        
        h, w = self.original_image.shape[:2]
        ox = max(0, min(ox, w-1))
        oy = max(0, min(oy, h-1))
        
        return ox, oy, rel_x, rel_y

    def on_canvas_click(self, event):
        ox, oy, rx, ry = self.get_image_coords(event)
        if ox is None: return

        self.dragging_point_index = None
        hit_threshold = 15 # pixels
        
        for i, pt in enumerate(self.click_points):
            px = pt[0] * self.display_scale
            py = pt[1] * self.display_scale
            dist = ((rx - px)**2 + (ry - py)**2)**0.5
            if dist < hit_threshold:
                self.dragging_point_index = i
                return

        if len(self.click_points) >= 2:
            self.click_points = [] 
            
        self.click_points.append((ox, oy))
        self.update_selection()
        self.redraw_canvas()

    def on_canvas_drag(self, event):
        if self.dragging_point_index is None: return
        ox, oy, _, _ = self.get_image_coords(event)
        if ox is None: return
        
        self.click_points[self.dragging_point_index] = (ox, oy)
        self.update_selection()
        self.redraw_canvas()

    def on_canvas_release(self, event):
        self.dragging_point_index = None

    def update_selection(self):
        if len(self.click_points) == 2:
            p1 = self.click_points[0]
            p2 = self.click_points[1]
            self.scale_ul = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            self.scale_lr = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            self.update_coord_label()

    def update_coord_label(self):
        if self.scale_ul and self.scale_lr:
            self.coord_label.config(text=f"Selected Range: UL {self.scale_ul} - LR {self.scale_lr} (Original Pixels)")
        else:
            self.coord_label.config(text="Please click 2 points on the image to define the heatmap range (Not Selected)")

    def log(self, message):
        def _log():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, _log)

    def start_processing(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load a base image!")
            return
        if not self.data_source_path.get():
            messagebox.showwarning("Warning", "Please select a data source!")
            return
        if self.scale_ul is None or self.scale_lr is None:
            messagebox.showwarning("Warning", "Please select 2 points on the image!")
            return
            
        self.run_btn.config(state="disabled")
        threading.Thread(target=self.run_generation, daemon=True).start()

    def display_generation_result(self, final_img):
        try:
            self.final_heatmap_img = final_img
            
            # Resize for display
            final_h, final_w = final_img.shape[:2]
            new_w = int(final_w * self.display_scale)
            new_h = int(final_h * self.display_scale)
            
            final_img_resized = cv2.resize(final_img, (new_w, new_h))
            final_img_rgb = cv2.cvtColor(final_img_resized, cv2.COLOR_BGR2RGB)
            self.display_image = Image.fromarray(final_img_rgb)
            self.redraw_canvas()
            
            self.log("=== Generation Done ===")
            
            # Force UI update so the image is visible
            self.root.update()
            
            # Enable save button
            if hasattr(self, 'save_btn'):
                self.save_btn.config(state="normal")
            
        except Exception as e:
            self.log(f"Error displaying result: {e}")
            messagebox.showerror("Error", f"Error displaying result: {e}")
        finally:
             self.run_btn.config(state="normal")

    def prompt_save_result(self):
        if not hasattr(self, 'final_heatmap_img') or self.final_heatmap_img is None:
            return

        # Determine default save path
        base_name = os.path.splitext(os.path.basename(self.base_image_path.get()))[0]
        default_filename = f"{base_name}_heatmap_result.jpg"
        
        data_source = self.data_source_path.get()
        if self.data_source_type.get() == "folder":
            default_dir = data_source
        else:
            default_dir = os.path.dirname(data_source)
            
        default_path = os.path.join(default_dir, default_filename)
        
        # Ask user
        msg = f"Heatmap generated successfully.\n\nDo you want to save it to the default location?\n{default_path}\n\nSelect 'Yes' to save there.\nSelect 'No' to choose another location.\nSelect 'Cancel' to discard."
        
        choice = messagebox.askyesnocancel("Save Heatmap", msg)
        
        if choice is None: # Cancel
            self.log("Save cancelled by user.")
            return
            
        if choice: # Yes -> Default
            save_path = default_path
        else: # No -> Save As
            save_path = filedialog.asksaveasfilename(
                initialdir=default_dir,
                initialfile=default_filename,
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")]
            )
            
        if save_path:
            try:
                cv2.imwrite(save_path, self.final_heatmap_img)
                self.log(f"Saved to: {save_path}")
                messagebox.showinfo("Success", f"Saved to:\n{save_path}")
            except Exception as e:
                self.log(f"Error saving file: {e}")
                messagebox.showerror("Error", f"Failed to save file: {e}")
        else:
            self.log("Save cancelled.")

    def on_merge_change(self):
        if self.data_source_path.get():
             self.trigger_scan_filters()

    def load_data_internal(self, force_merge=False):
        data_path = self.data_source_path.get()
        is_folder = self.data_source_type.get() == "folder"
        dfs = []
        
        if is_folder:
            file_list = []
            self.log("Searching for files...")
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file == "parametric_out.xlsx":
                        file_list.append(os.path.join(root, file))
            
            if not file_list:
                    raise ValueError("No parametric_out.xlsx files found in the selected folder.")
            
            self.log(f"Found {len(file_list)} files. Reading data...")

            if self.merge_data_var.get() or force_merge:
                for i, fp in enumerate(file_list):
                    try:
                        if i % 10 == 0:
                            self.log(f"Reading file {i+1}/{len(file_list)}")
                        df = pd.read_excel(fp)
                        dfs.append(df)
                    except Exception as e:
                        self.log(f"Error reading {os.path.basename(fp)}: {e}")
            else:
                try:
                    df = pd.read_excel(file_list[0])
                    dfs.append(df)
                except Exception as e:
                    raise ValueError(f"Error reading file: {e}")
        else:
            try:
                df = pd.read_excel(data_path)
                dfs.append(df)
            except Exception as e:
                raise ValueError(f"Error reading file: {e}")
        
        if not dfs:
            raise ValueError("No valid data loaded")
            
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df

    def scan_data_filters(self):
        try:
            self.log("Scanning data for filters...")
            # Always force merge during scan to get all potential filters
            try:
                df = self.load_data_internal(force_merge=True)
                self.cached_df = df
            except Exception as e:
                self.log(f"Error loading data for filters: {e}")
                self.root.after(0, self.update_filter_ui, [], [])
                return
            
            # Helper to get unique values including NA
            def get_unique_with_na(col_name):
                if col_name not in df.columns:
                    return []
                # Fill NA/Empty with 'NA'
                temp_series = df[col_name].fillna('NA').astype(str)
                # Handle empty strings if any
                temp_series = temp_series.replace('', 'NA')
                unique_vals = sorted(temp_series.unique().tolist())
                return unique_vals

            modes = get_unique_with_na('Defect Mode')
            shapes = get_unique_with_na('Shape')
            
            self.root.after(0, self.update_filter_ui, modes, shapes)
            self.log(f"Scan complete. Found {len(modes)} modes, {len(shapes)} shapes.")
            
        except Exception as e:
            self.log(f"Scan info error: {e}")
            self.root.after(0, self.update_filter_ui, [], []) 

    def update_filter_ui(self, modes, shapes):
        # Clear existing widgets
        for widget in self.filter_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.filter_vars = {}
        
        def on_filter_change(*args):
             # Trigger refresh if data is already loaded/generated
             if hasattr(self, 'cached_df') and self.cached_df is not None:
                 threading.Thread(target=self.refresh_heatmap, daemon=True).start()

        # Defect Mode
        if modes:
            ttk.Label(self.filter_scrollable_frame, text="Defect Mode", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5, 2), padx=5)
            self.filter_vars['Defect Mode'] = {}
            for m in modes:
                var = tk.BooleanVar(value=True)
                var.trace_add("write", on_filter_change) # Add trace
                self.filter_vars['Defect Mode'][m] = var
                ttk.Checkbutton(self.filter_scrollable_frame, text=str(m), variable=var).pack(anchor="w", padx=15)
            
            ttk.Separator(self.filter_scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5, padx=5)

        # Shape
        if shapes:
            ttk.Label(self.filter_scrollable_frame, text="Shape", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5, 2), padx=5)
            self.filter_vars['Shape'] = {}
            for s in shapes:
                var = tk.BooleanVar(value=True)
                var.trace_add("write", on_filter_change) # Add trace
                self.filter_vars['Shape'][s] = var
                ttk.Checkbutton(self.filter_scrollable_frame, text=str(s), variable=var).pack(anchor="w", padx=15)
        
        if not modes and not shapes:
             ttk.Label(self.filter_scrollable_frame, text="No filters available").pack(anchor="w", padx=5, pady=5)
             
        # Force update layout
        self.filter_canvas.update_idletasks()
        self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all")) 

    def refresh_heatmap(self):
        # Lightweight version of run_generation that assumes data is loaded
        if not hasattr(self, 'cached_df') or self.cached_df is None:
            return
        
        try:
             # Use the common logic
             self.run_generation_logic(self.cached_df)
        except Exception as e:
             self.log(f"Refresh error: {e}")

    def run_generation(self):
        try:
            self.log("=== Start Generation ===")
            
            # 1. Get Data
            if hasattr(self, 'cached_df') and self.cached_df is not None:
                full_df = self.cached_df # Use existing cache
                self.log("Using cached data.")
            else:
                self.log("Loading data...")
                full_df = self.load_data_internal()
                self.cached_df = full_df # Cache it
                
            self.run_generation_logic(full_df)
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {str(e)}"))
            self.root.after(0, lambda: self.run_btn.config(state="normal"))

    def run_generation_logic(self, full_df_input):
        # Work on a copy
        full_df = full_df_input.copy()
        
        self.log(f"Total Rows: {len(full_df)}")
        
        if 'Contour' not in full_df.columns:
            raise ValueError("Data missing 'Contour' column")
        
        # Fill NA for filtering purposes
        if 'Defect Mode' in full_df.columns:
                full_df['Defect Mode'] = full_df['Defect Mode'].fillna('NA').astype(str).replace('', 'NA')
        
        if 'Shape' in full_df.columns:
                full_df['Shape'] = full_df['Shape'].fillna('NA').astype(str).replace('', 'NA')
            
        # 2. Filter Data
        if hasattr(self, 'filter_vars'):
            # Filter Defect Mode
            if 'Defect Mode' in self.filter_vars and 'Defect Mode' in full_df.columns:
                selected_modes = [k for k, v in self.filter_vars['Defect Mode'].items() if v.get()]
                if len(selected_modes) < len(self.filter_vars['Defect Mode']):
                    full_df = full_df[full_df['Defect Mode'].isin(selected_modes)]
            
            # Filter Shape
            if 'Shape' in self.filter_vars and 'Shape' in full_df.columns:
                selected_shapes = [k for k, v in self.filter_vars['Shape'].items() if v.get()]
                if len(selected_shapes) < len(self.filter_vars['Shape']):
                    full_df = full_df[full_df['Shape'].isin(selected_shapes)]
        
        self.log(f"Rows after filtering: {len(full_df)}")
        
        if len(full_df) == 0:
             # raise ValueError("No data left after filtering!")
             self.log("Warning: No data left after filtering!")
            
        # 3. Generate Heatmap
        opacity = self.opacity_var.get()
        
        final_img = self.generate_heatmap_logic(
            self.original_image.copy(),
            full_df['Contour'].values,
            self.scale_ul,
            self.scale_lr,
            opacity
        )
        
        # 4. Show Result (Schedule on Main Thread)
        self.root.after(0, self.display_generation_result, final_img)


    def create_custom_colormap(self):
        """
        Create a custom colormap:
        0 (Low Freq):   User Blue (R0, G162, B255) -> BGR(255, 162, 0)
        128 (Mid Freq): Yellow    (R255, G255, B0) -> BGR(0, 255, 255)
        255 (High Freq): Red      (R255, G0, B0)   -> BGR(0, 0, 255)
        """
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        
        # Color Stops (BGR)
        c_low = np.array([255, 162, 0])   # User Blue
        c_mid = np.array([0, 255, 255])   # Yellow
        c_high = np.array([0, 0, 255])    # Red
        
        # Segment 1: 0 -> 128 (Low to Mid)
        for i in range(128):
            t = i / 128.0
            color = (1 - t) * c_low + t * c_mid
            lut[i, 0] = color.astype(np.uint8)
            
        # Segment 2: 128 -> 255 (Mid to High)
        for i in range(128, 256):
            t = (i - 128) / 127.0
            color = (1 - t) * c_mid + t * c_high
            lut[i, 0] = color.astype(np.uint8)
            
        return lut

    def generate_heatmap_logic(self, render_img, contours, scale_ul, scale_lr, opacity):
        render_h, render_w = render_img.shape[:2]
        count_map = np.zeros((render_h, render_w), dtype=np.float32)
        
        valid_contours = 0
        
        # Pre-calculate ROI dimensions
        roi_w = scale_lr[0] - scale_ul[0]
        roi_h = scale_lr[1] - scale_ul[1]
        
        self.log(f"Mapping contours to ROI: {scale_ul} -> {scale_lr} (w={roi_w}, h={roi_h})")
        
        for contour in contours:
            contour_array = None
            if isinstance(contour, str):
                try:
                    # Clean and parse string representation
                    # Remove surrounding brackets and split by ']], [['
                    clean = contour.strip()
                    if clean.startswith('['): clean = clean[1:]
                    if clean.endswith(']'): clean = clean[:-1]
                    
                    # Handle empty or malformed
                    if not clean: continue
                    
                    # Simple parsing strategy: remove all brackets and split by comma
                    # This handles [[x,y], [x,y]] and [x,y, x,y] formats
                    clean = clean.replace('[', '').replace(']', '')
                    parts = clean.split(',')
                    
                    if len(parts) < 2: continue
                    
                    points = []
                    for i in range(0, len(parts), 2):
                        if i+1 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i+1])
                                points.append([x, y])
                            except ValueError:
                                pass
                                
                    contour_array = np.array(points)
                except Exception as e:
                    # self.log(f"Parse error: {e}")
                    continue
            else:
                contour_array = np.array(contour)
            
            if contour_array is None or len(contour_array) < 3:
                continue
                
            if contour_array.ndim == 1:
                contour_array = contour_array.reshape(-1, 2)
            
            # Coordinate Mapping
            contour_mapped = np.zeros_like(contour_array)
            
            # Logic from Heat_Map_Creator.py:
            # contour_array[:, 0] = contour_array[:, 0] * (render_scale_lr[0] - render_scale_ul[0]) + render_scale_ul[0]
            # contour_array[:, 1] = contour_array[:, 1] * (render_scale_lr[1] - render_scale_ul[1]) + render_scale_ul[1]
            
            contour_mapped[:, 0] = contour_array[:, 0] * roi_w + scale_ul[0]
            contour_mapped[:, 1] = contour_array[:, 1] * roi_h + scale_ul[1]
            
            contour_final = contour_mapped.astype(np.int32)
            
            # Draw mask
            mask = np.zeros((render_h, render_w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour_final], 1)
            
            # Dilate
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            count_map[mask == 1] += 1
            valid_contours += 1
            
        self.log(f"Processed {valid_contours} contours")
            
        # Generate Heatmap
        if valid_contours == 0:
            return render_img
            
        # Normalize: Map [1, max] to [0, 255] (Blue -> Red)
        max_val = count_map.max()
        if max_val <= 1:
            # If max count is 1, everything is low frequency. 
            # User wants Blue (slightly darker) for low frequency.
            # Jet 0 is Dark Blue.
            heatmap_normalized = np.zeros_like(count_map, dtype=np.uint8)
            # If we want distinct Blue for count=1
            heatmap_normalized[count_map > 0] = 0 # Blue
        else:
            # Linear mapping: 1 -> 0, max -> 255
            # val_norm = (val - 1) / (max - 1) * 255
            count_float = count_map.astype(np.float32)
            norm_float = (count_float - 1) / (max_val - 1) * 255
            norm_float = np.clip(norm_float, 0, 255)
            heatmap_normalized = norm_float.astype(np.uint8)

        # Use Custom Colormap
        custom_lut = self.create_custom_colormap()
        
        # Convert single-channel normalized heatmap to 3-channel BGR
        # This is required because cv2.LUT with a 3-channel LUT requires a 3-channel input
        # (or 1-channel LUT with 1-channel input).
        # By converting to BGR, we get (v, v, v) for each pixel.
        # Channel 0 (B) uses LUT channel 0.
        # Channel 1 (G) uses LUT channel 1.
        # Channel 2 (R) uses LUT channel 2.
        heatmap_normalized_bgr = cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2BGR)
        
        heatmap = cv2.LUT(heatmap_normalized_bgr, custom_lut)
        
        heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
        # Make zero-count areas transparent
        heatmap_rgba[count_map == 0, 3] = 0
        
        render_rgba = cv2.cvtColor(render_img, cv2.COLOR_BGR2BGRA)
        final_overlay = render_rgba.copy()
        
        # Alpha blending
        alpha = heatmap_rgba[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        for c in range(3):
            final_overlay[:, :, c] = (1 - alpha[:, :, 0] * opacity) * render_rgba[:, :, c] + \
                                    (alpha[:, :, 0] * opacity) * heatmap_rgba[:, :, c]
                                    
        return final_overlay.astype(np.uint8)

if __name__ == '__main__':
    root = tk.Tk()
    app = HeatMapGUI(root)
    root.mainloop()
