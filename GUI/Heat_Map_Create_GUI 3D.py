import sys
import os
import numpy as np
import pandas as pd
import cv2
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox, 
                             QSlider, QTextEdit, QGroupBox, QScrollArea, QSplitter, QMessageBox,
                             QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPalette, QColor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# --- Worker Thread for Data Loading ---
class DataLoader(QThread):
    finished = pyqtSignal(object) # Returns loaded dataframe
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, path, source_type, file_type, merge):
        super().__init__()
        self.path = path
        self.source_type = source_type
        self.file_type = file_type
        self.merge = merge

    def run(self):
        try:
            dfs = []
            if self.source_type == "folder":
                file_list = []
                for root, dirs, files in os.walk(self.path):
                    for file in files:
                        if self.file_type == "Excel":
                            if file == "parametric_out.xlsx":
                                file_list.append(os.path.join(root, file))
                        else: # NPY
                            if file.endswith(".npy"):
                                file_list.append(os.path.join(root, file))
                
                if not file_list:
                    raise ValueError("No matching files found.")
                
                self.log.emit(f"Found {len(file_list)} files.")
                
                files_to_read = file_list if self.merge else [file_list[0]]
                
                for i, fp in enumerate(files_to_read):
                    if i % 10 == 0: self.log.emit(f"Reading {i+1}/{len(files_to_read)}")
                    df = self.read_file(fp)
                    dfs.append(df)
            else:
                df = self.read_file(self.path)
                dfs.append(df)

            if not dfs:
                raise ValueError("No valid data loaded")
                
            full_df = pd.concat(dfs, ignore_index=True)
            self.finished.emit(full_df)
            
        except Exception as e:
            self.error.emit(str(e))

    def read_file(self, fp):
        if self.file_type == "Excel":
            return pd.read_excel(fp)
        else:
            data = np.load(fp, allow_pickle=True)
            return self._process_npy_data(data)

    def _process_npy_data(self, data):
        # Logic from original script
        mask = None
        bbox = None
        if data.shape == () and data.dtype == 'O':
            item = data.item()
            if isinstance(item, dict):
                mask = item.get('mask')
                bbox = item.get('bbox')
                if mask is not None:
                    mask = (mask > 0).astype(np.uint8)
        elif data.ndim == 2:
             mask = (data > 0).astype(np.uint8)
             
        if mask is None:
             return pd.DataFrame({'Contour': list(data)})

        h, w = mask.shape
        ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = ret[0] if len(ret) == 2 else ret[1]
        
        normalized_contours = []
        for cnt in contours:
            cnt = cnt.reshape(-1, 2).astype(np.float32)
            cnt[:, 0] /= w
            cnt[:, 1] /= h
            normalized_contours.append(cnt)
            
        return pd.DataFrame({'Contour': normalized_contours})

class HeatMap3DGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeatMap 3D Generator")
        self.resize(1500, 800)  # Increased width for 3-column layout
        
        # Dark Theme
        self.apply_dark_theme()
        
        # Main Widget & Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- Left Panel 1 (Load & Filter) ---
        left_panel_1 = QWidget()
        left_layout_1 = QVBoxLayout(left_panel_1)
        splitter.addWidget(left_panel_1)
        
        # --- Left Panel 2 (View & Heatmap Settings) ---
        left_panel_2 = QWidget()
        left_layout_2 = QVBoxLayout(left_panel_2)
        splitter.addWidget(left_panel_2)
        
        # === Left Panel 1 Content ===
        
        # 1. STL Selection
        grp_stl = QGroupBox("3D Model")
        l_stl = QVBoxLayout(grp_stl)
        btn_stl = QPushButton("Load STL File")
        btn_stl.clicked.connect(self.load_stl_dialog)
        self.lbl_stl_path = QLabel("No file selected")
        self.lbl_stl_path.setWordWrap(True)
        
        l_stl.addWidget(btn_stl)
        l_stl.addWidget(self.lbl_stl_path)
        left_layout_1.addWidget(grp_stl)
        
        # 2. Data Source
        grp_data = QGroupBox("Heatmap Data Source")
        l_data = QVBoxLayout(grp_data)
        
        h_type = QHBoxLayout()
        h_type.addWidget(QLabel("Type:"))
        self.combo_file_type = QComboBox()
        self.combo_file_type.addItems(["Excel", "NPY"])
        h_type.addWidget(self.combo_file_type)
        l_data.addLayout(h_type)
        
        h_btns = QHBoxLayout()
        btn_file = QPushButton("Select File")
        btn_file.clicked.connect(lambda: self.load_data_dialog("file"))
        btn_folder = QPushButton("Select Folder")
        btn_folder.clicked.connect(lambda: self.load_data_dialog("folder"))
        h_btns.addWidget(btn_file)
        h_btns.addWidget(btn_folder)
        l_data.addLayout(h_btns)
        
        self.chk_merge = QCheckBox("Merge all files in folder")
        l_data.addWidget(self.chk_merge)
        
        self.lbl_data_path = QLabel("No data selected")
        self.lbl_data_path.setWordWrap(True)
        l_data.addWidget(self.lbl_data_path)
        left_layout_1.addWidget(grp_data)
        
        # 3. Filter Options
        grp_filter = QGroupBox("Filters")
        l_filter = QVBoxLayout(grp_filter)
        
        btn_refresh = QPushButton("Refresh Filters")
        btn_refresh.clicked.connect(self.refresh_filters)
        l_filter.addWidget(btn_refresh)
        
        self.scroll_filters = QScrollArea()
        self.scroll_filters.setWidgetResizable(True)
        self.widget_filters = QWidget()
        self.layout_filters = QVBoxLayout(self.widget_filters)
        self.scroll_filters.setWidget(self.widget_filters)
        l_filter.addWidget(self.scroll_filters)
        
        left_layout_1.addWidget(grp_filter, 1) # Stretch
        
        # === Left Panel 2 Content ===

        # 3.5 Standard Views
        grp_views = QGroupBox("Standard Views")
        l_views = QGridLayout(grp_views)
        
        views = [
            ("Top (+Z)", 0, 0, 1, 0, 1, 0),    ("Bottom (-Z)", 0, 0, -1, 0, 1, 0),
            ("Front (-Y)", 0, -1, 0, 0, 0, 1), ("Back (+Y)", 0, 1, 0, 0, 0, 1),
            ("Left (-X)", -1, 0, 0, 0, 0, 1),  ("Right (+X)", 1, 0, 0, 0, 0, 1)
        ]
        
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        
        for (name, vx, vy, vz, ux, uy, uz), (r, c) in zip(views, positions):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, x=vx, y=vy, z=vz, ux=ux, uy=uy, uz=uz: self.set_camera_view(x, y, z, ux, uy, uz))
            l_views.addWidget(btn, r, c)
            
        # Lock View Checkbox
        self.chk_lock_view = QCheckBox("Lock Object Orientation")
        self.chk_lock_view.setToolTip("Lock the object orientation to prevent accidental rotation when adjusting the box.")
        self.chk_lock_view.toggled.connect(self.toggle_view_lock)
        l_views.addWidget(self.chk_lock_view, 3, 0, 1, 2)
        
        left_layout_2.addWidget(grp_views)

        # 4. Heatmap Settings
        grp_settings = QGroupBox("Settings")
        l_settings = QVBoxLayout(grp_settings)
        
        # Selection Mode
        h_sel = QHBoxLayout()
        self.chk_select_mode = QCheckBox("Select Projection Area")
        self.chk_select_mode.toggled.connect(self.toggle_selection_mode)
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        h_sel.addWidget(self.chk_select_mode)
        h_sel.addWidget(self.btn_clear_sel)
        l_settings.addLayout(h_sel)
        
        self.lbl_selection = QLabel("No area selected (Using full model)")
        self.lbl_selection.setWordWrap(True)
        l_settings.addWidget(self.lbl_selection)

        h_opacity = QHBoxLayout()
        h_opacity.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(60)
        self.slider_opacity.valueChanged.connect(lambda: self.lbl_opacity.setText(f"{self.slider_opacity.value()}%"))
        h_opacity.addWidget(self.slider_opacity)
        self.lbl_opacity = QLabel("60%")
        h_opacity.addWidget(self.lbl_opacity)
        l_settings.addLayout(h_opacity)
        
        self.chk_multi_overlay = QCheckBox("Multi-Overlay Mode")
        self.chk_multi_overlay.setChecked(True)
        l_settings.addWidget(self.chk_multi_overlay)
        
        self.btn_generate = QPushButton("Generate Heatmap on 3D")
        self.btn_generate.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_generate.clicked.connect(self.generate_heatmap)
        l_settings.addWidget(self.btn_generate)
        
        left_layout_2.addWidget(grp_settings)
        
        # Log (Moved to bottom of Panel 2)
        self.txt_log = QTextEdit()
        self.txt_log.setMaximumHeight(200) # Increased height a bit
        self.txt_log.setReadOnly(True)
        left_layout_2.addWidget(self.txt_log)
        
        left_layout_2.addStretch() # Push everything up
        
        # --- Right Panel (VTK) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        
        # Adjust Splitter factors (Panel 1 : Panel 2 : 3D View) AFTER adding all widgets
        # Ratio 2:2:6
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 6)
        
        # Explicitly set sizes to enforce the 2:2:6 ratio initially
        # Total width 1500 -> 300 : 300 : 900
        splitter.setSizes([300, 300, 900])
        
        # Grid Layout for 3D View + Sliders
        # Layout:
        # [       ] [ H Pan ]
        # [ V Pan ] [  3D   ]
        # [       ] [ Zoom  ]
        
        gl_right = QGridLayout()
        right_layout.addLayout(gl_right)
        
        # 1. Horizontal Pan Slider (Top)
        self.slider_pan_h = QSlider(Qt.Horizontal)
        self.slider_pan_h.setRange(0, 1000)
        self.slider_pan_h.setValue(500)
        self.slider_pan_h.setToolTip("Pan Left/Right")
        self.slider_pan_h.valueChanged.connect(self.pan_horizontal)
        self.slider_pan_h.sliderReleased.connect(lambda: self.reset_pan_slider(self.slider_pan_h, 'h'))
        gl_right.addWidget(self.slider_pan_h, 0, 1)
        
        # 2. Vertical Pan Slider (Left)
        self.slider_pan_v = QSlider(Qt.Vertical)
        self.slider_pan_v.setRange(0, 1000)
        self.slider_pan_v.setValue(500)
        self.slider_pan_v.setToolTip("Pan Up/Down")
        self.slider_pan_v.valueChanged.connect(self.pan_vertical)
        self.slider_pan_v.sliderReleased.connect(lambda: self.reset_pan_slider(self.slider_pan_v, 'v'))
        gl_right.addWidget(self.slider_pan_v, 1, 0)
        
        # 3. VTK Widget (Center)
        self.vtkWidget = QVTKRenderWindowInteractor(right_panel)
        gl_right.addWidget(self.vtkWidget, 1, 1)
        
        # 4. Zoom Controls (Bottom)
        h_zoom = QHBoxLayout()
        h_zoom.addWidget(QLabel("Zoom:"))
        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(20, 500) # 0.2x to 5.0x
        self.slider_zoom.setValue(100) # 1.0x
        self.slider_zoom.setToolTip("Zoom Level (0.2x - 5.0x)")
        self.slider_zoom.valueChanged.connect(self.set_zoom_level)
        # self.slider_zoom.sliderPressed.connect(self.on_slider_pressed) # Removed
        # self.slider_zoom.sliderReleased.connect(self.on_slider_released) # Removed
        h_zoom.addWidget(self.slider_zoom)
        self.lbl_zoom_val = QLabel("1.0x")
        self.lbl_zoom_val.setFixedWidth(40)
        h_zoom.addWidget(self.lbl_zoom_val)
        
        gl_right.addLayout(h_zoom, 2, 1)
        
        # Make VTK expand
        gl_right.setColumnStretch(1, 1)
        gl_right.setRowStretch(1, 1)
        
        # VTK Setup
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        # Set Interactor Style to Trackball Camera (Mouse Rotate/Pan/Zoom)
        self.style_camera = vtk.vtkInteractorStyleTrackballCamera()
        
        # Locked Style: Allow Zoom (Right/Scroll) and Pan (Middle/Shift+Left), but block Rotate (Left)
        # We inherit from vtkInteractorStyleTrackballCamera but override OnLeftButtonDown to do nothing
        class LockedInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
            def OnLeftButtonDown(self):
                pass
            def OnLeftButtonUp(self):
                pass
            def OnMouseMove(self):
                pass
            def OnMiddleButtonDown(self): pass
            def OnMiddleButtonUp(self): pass
            def OnRightButtonDown(self): pass
            def OnRightButtonUp(self): pass
            def OnMouseWheelForward(self): pass # Disable wheel zoom
            def OnMouseWheelBackward(self): pass # Disable wheel zoom
                
        self.style_locked = LockedInteractorStyle()
        
        # Also disable wheel on the main camera style if we want "Only Slider"
        # We can create a custom style for the main camera too.
        class CustomCameraStyle(vtk.vtkInteractorStyleTrackballCamera):
             def __init__(self, owner):
                 super().__init__()
                 self.owner = owner
             def OnLeftButtonDown(self):
                 if getattr(self.owner, "suppress_camera_events", False):
                     return
                 super().OnLeftButtonDown()
             def OnLeftButtonUp(self):
                 if getattr(self.owner, "suppress_camera_events", False):
                     return
                 super().OnLeftButtonUp()
             def OnMouseMove(self):
                 if getattr(self.owner, "suppress_camera_events", False):
                     return
                 super().OnMouseMove()
             def OnMouseWheelForward(self): pass
             def OnMouseWheelBackward(self): pass
        
        self.style_camera = CustomCameraStyle(self)
        
        self.interactor.SetInteractorStyle(self.style_camera)
        
        # Add Axes Actor
        self.axes = vtk.vtkAxesActor()
        self.axes.SetTotalLength(20, 20, 20)
        self.axes.SetShaftTypeToCylinder()
        self.axes.SetCylinderRadius(0.02)
        self.renderer.AddActor(self.axes)

        # Data State
        self.stl_path = None
        self.last_slider_val = 100 # Track last slider value for relative zoom
        
        # Pan State
        self.last_pan_h = 500
        self.last_pan_v = 500
        
        self.data_path = None
        self.data_source_type = None # 'file' or 'folder'
        self.cached_df = None
        self.filter_checkboxes = {} # {'Defect Mode': {val: QCheckBox}, ...}
        self._restore_style_timer = None
        self.suppress_camera_events = False
        
        # Selection State (per-face)
        # face key: 'posX','negX','posY','negY','posZ','negZ'
        self.active_face = None
        self.face_states = {}  # face_key -> {'point_widgets': [], 'outline_actor': None, 'outline_source': None, 'heatmap_actors': []}
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        
        # Coordinate Overlay Labels (Text Actors)
        self.txt_actor_p1 = None
        self.txt_actor_p2 = None
        self.txt_actor_rot = None
        
        # ROI Widget
        self.outline_source = None # Replaces plane_widget
        self.outline_actor = None
        self.heatmap_actors = []
        self.selection_actor = None
        
        # Add Observer for Picking
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click, 1.0) # Priority 1.0
        
        # Add Observer for Camera Interaction (Sync Zoom + Update Rotation)
        # ModifiedEvent on Interactor is not useful for Camera changes. Use InteractionEvent.
        self.interactor.AddObserver("InteractionEvent", self.on_interaction) 

        self.log("Application started.")
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

    def log(self, msg):
        self.txt_log.append(msg)
        print(msg)

    def set_camera_view(self, vx, vy, vz, ux, uy, uz):
        camera = self.renderer.GetActiveCamera()
        
        # Update active face by view direction
        def face_key_from(vx, vy, vz):
            if vx == 1: return 'posX'
            if vx == -1: return 'negX'
            if vy == 1: return 'posY'
            if vy == -1: return 'negY'
            if vz == 1: return 'posZ'
            if vz == -1: return 'negZ'
            return 'posZ'
        self.active_face = face_key_from(vx, vy, vz)
        self.log(f"Active Face: {self.active_face}")
        
        # Ensure face state entry exists
        if self.active_face not in self.face_states:
            self.face_states[self.active_face] = {'point_widgets': [], 'outline_actor': None, 'outline_source': None, 'heatmap_actors': []}
        
        # Toggle interaction of per-face widgets/outline
        for fk, st in self.face_states.items():
            active = (fk == self.active_face)
            for hw in st.get('point_widgets', []):
                # Keep widgets enabled for visibility; gate interaction only
                try:
                    hw.SetEnabled(1)
                except Exception:
                    try:
                        hw.EnabledOn()
                    except Exception:
                        pass
                # Disable/enable event processing to avoid interaction on非激活面
                try:
                    hw.SetProcessEvents(1 if active else 0)
                except Exception:
                    pass
                # Ensure representation stays visible when切面
                try:
                    rep = hw.GetRepresentation()
                    rep.SetVisibility(1)
                except Exception:
                    pass
            if st.get('outline_actor'):
                # Keep all outlines visible; only gate interaction by face
                st['outline_actor'].SetVisibility(1)
        
        # Calculate distance to keep object in view
        # We can just reset camera, but we need to set position first
        # Current focal point is likely 0,0,0 if centered
        
        focal_point = camera.GetFocalPoint()
        dist = camera.GetDistance()
        
        new_pos = (
            focal_point[0] + vx * dist,
            focal_point[1] + vy * dist,
            focal_point[2] + vz * dist
        )
        
        camera.SetPosition(new_pos)
        camera.SetViewUp(ux, uy, uz)
        
        # Ensure we are looking at the object
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        
        # Reset Zoom Slider
        camera = self.renderer.GetActiveCamera()
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(100)
        self.last_slider_val = 100 # Reset tracker
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom_val.setText("1.0x")
        
        self.log(f"Camera View Set: ({vx}, {vy}, {vz})")

    def toggle_view_lock(self, checked):
        if checked:
            self.interactor.SetInteractorStyle(self.style_locked)
            self.log("View Locked: Object orientation is fixed.")
        else:
            self.interactor.SetInteractorStyle(self.style_camera)
            self.log("View Unlocked: Object can be rotated.")

    def toggle_selection_mode(self, checked):
        if checked:
            self.log("Selection Mode ON: Click on model to select points.")
            # We don't disable trackball, we just intercept events in on_left_click
        else:
            self.log("Selection Mode OFF.")

    def set_zoom_level(self, val):
        if self.last_slider_val is None:
             self.last_slider_val = 100
             
        # Calculate ratio of change relative to LAST slider position
        # This provides smooth, continuous zoom regardless of camera state
        if self.last_slider_val == 0: self.last_slider_val = 1 # Safety
        
        ratio = val / float(self.last_slider_val)
        
        # Update tracker
        self.last_slider_val = val
        self.lbl_zoom_val.setText(f"{val/100.0:.1f}x")
        
        camera = self.renderer.GetActiveCamera()
        if not camera: return
        
        if camera.GetParallelProjection():
            # For Parallel, Zoom() changes the ParallelScale (Inverse relationship)
            # Zoom(factor) => Scale = Scale / factor
            # So if ratio > 1 (Zoom In), Scale decreases. Correct.
            camera.Zoom(ratio)
        else:
            # For Perspective, Dolly() moves the camera
            # Dolly(factor) => Distance = Distance / factor
            # So if ratio > 1 (Zoom In), Distance decreases. Correct.
            camera.Dolly(ratio)
            
        self.renderer.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        
        # Also update rotation label in case user wants to see it, 
        # though Zoom doesn't change rotation usually.
        self.update_rotation_label()

    def reset_pan_slider(self, slider, axis):
        """Reset pan slider to center (500) to allow infinite panning."""
        slider.blockSignals(True)
        slider.setValue(500)
        slider.blockSignals(False)
        if axis == 'h':
            self.last_pan_h = 500
        else:
            self.last_pan_v = 500

    def pan_horizontal(self, val):
        camera = self.renderer.GetActiveCamera()
        if not camera: return
        
        delta = val - self.last_pan_h
        self.last_pan_h = val
        
        if delta == 0: return
        
        # Calculate Panning Vector (Left/Right)
        # Right = Cross(DirectionOfProjection, ViewUp)
        dop = np.array(camera.GetDirectionOfProjection())
        up = np.array(camera.GetViewUp())
        right = np.cross(dop, up)
        right = right / np.linalg.norm(right) # Normalize
        
        # Scale factor based on distance/parallel scale
        if camera.GetParallelProjection():
            scale = camera.GetParallelScale()
        else:
            scale = camera.GetDistance()
            
        # Heuristic sensitivity: 0.1% of distance per tick?
        # Sensitivity 1/1000 of view scale per unit?
        # Let's try 0.002
        move_vec = right * delta * scale * 0.002
        
        # Move Camera (Position and FocalPoint)
        # -delta moves Left, +delta moves Right?
        # Usually Scroll Right -> View moves Right (Camera moves Left)?
        # Or Scroll Right -> Look Right (Camera moves Right)?
        # Standard Scrollbar: Scroll Right -> Viewport moves Right -> Content moves Left.
        # So Camera moves Right.
        
        pos = np.array(camera.GetPosition())
        fp = np.array(camera.GetFocalPoint())
        
        camera.SetPosition(pos + move_vec)
        camera.SetFocalPoint(fp + move_vec)
        
        self.vtkWidget.GetRenderWindow().Render()

    def pan_vertical(self, val):
        camera = self.renderer.GetActiveCamera()
        if not camera: return
        
        delta = val - self.last_pan_v
        self.last_pan_v = val
        
        if delta == 0: return
        
        # Calculate Panning Vector (Up/Down)
        # Up is ViewUp (orthogonalized?)
        # ViewUp is not always orthogonal to DOP, but usually is.
        # Better to recompute true Up = Cross(Right, DOP)
        dop = np.array(camera.GetDirectionOfProjection())
        up_approx = np.array(camera.GetViewUp())
        right = np.cross(dop, up_approx)
        up = np.cross(right, dop)
        up = up / np.linalg.norm(up)
        
        if camera.GetParallelProjection():
            scale = camera.GetParallelScale()
        else:
            scale = camera.GetDistance()
            
        move_vec = up * delta * scale * 0.002
        
        # Move Camera
        # Slider Up (val increases) -> Move Camera Up?
        # Standard Scrollbar Down -> View moves Down -> Content moves Up.
        # But this is a Slider, not a Scrollbar.
        # Joystick Logic: Push Up -> Move Up.
        # So +delta -> +Up.
        
        pos = np.array(camera.GetPosition())
        fp = np.array(camera.GetFocalPoint())
        
        camera.SetPosition(pos + move_vec)
        camera.SetFocalPoint(fp + move_vec)
        
        self.vtkWidget.GetRenderWindow().Render()

    def clear_selection(self):
        # Clear selection for active face only
        if not self.active_face:
            self.log("No active face to clear.")
            return
        state = self.face_states.get(self.active_face, None)
        if not state:
            self.log("Nothing to clear for current face.")
            return
        # Clear Point Widgets
        for pw in state.get('point_widgets', []):
            try:
                pw.EnabledOff()
            except Exception:
                try:
                    pw.Off()
                except Exception:
                    pass
        state['point_widgets'] = []
        # Clear Outline
        if state.get('outline_actor'):
            self.renderer.RemoveActor(state['outline_actor'])
            state['outline_actor'] = None
            state['outline_source'] = None
        if getattr(self, 'selection_actor', None):
            self.renderer.RemoveActor(self.selection_actor)
            self.selection_actor = None
        self.lbl_selection.setText("Selection cleared for current face. Pick 2 points (Diagonal) to define ROI.")
        self.vtkWidget.GetRenderWindow().Render()
        self.log(f"Selection cleared for {self.active_face}.")

    def clear_heatmaps(self):
        """Remove all generated heatmap actors."""
        if not self.heatmap_actors:
            return
            
        for actor in self.heatmap_actors:
            self.renderer.RemoveActor(actor)
        self.heatmap_actors = []
        self.vtkWidget.GetRenderWindow().Render()
        self.log("Previous heatmaps cleared.")

    def create_arrow_glyph(self, scale=1.0):
        # 1. Cone (Arrow Head)
        cone = vtk.vtkConeSource()
        cone.SetHeight(0.4 * scale)
        cone.SetRadius(0.2 * scale)
        cone.SetResolution(32)
        cone.SetDirection(0, 0, -1)
        cone.SetCenter(0, 0, 0.2 * scale) # Tip at 0,0,0
        cone.Update()
        
        # 2. Cylinder (Arrow Shaft)
        cyl = vtk.vtkCylinderSource()
        cyl.SetHeight(0.6 * scale)
        cyl.SetRadius(0.08 * scale)
        cyl.SetResolution(32)
        cyl.Update()
        
        # Cylinder aligns Y by default. Rotate to Z.
        t_cyl = vtk.vtkTransform()
        t_cyl.Translate(0, 0, 0.7 * scale)
        t_cyl.RotateX(90) 
        
        tf_cyl = vtk.vtkTransformPolyDataFilter()
        tf_cyl.SetInputConnection(cyl.GetOutputPort())
        tf_cyl.SetTransform(t_cyl)
        tf_cyl.Update()
        
        # 3. Combine
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(cone.GetOutput())
        appender.AddInputData(tf_cyl.GetOutput())
        appender.Update()
        
        return appender.GetOutput()

    def create_arrow_glyph_oriented(self, normal=(0, 0, 1), scale=1.0):
        """
        Create an arrow glyph oriented so that its axis aligns with 'normal'.
        The base arrow is along +Z; compute rotation from +Z to normal.
        """
        base = self.create_arrow_glyph(scale)
        nx, ny, nz = normal
        import math
        # Normalize
        n = math.sqrt(nx*nx + ny*ny + nz*nz)
        if n == 0:
            nx, ny, nz = 0, 0, 1
            n = 1.0
        nx, ny, nz = nx/n, ny/n, nz/n
        # Axis-angle from z=(0,0,1) to n
        z = np.array([0.0, 0.0, 1.0])
        v = np.array([nx, ny, nz])
        dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        axis = np.cross(z, v)
        axis_norm = np.linalg.norm(axis)
        t = vtk.vtkTransform()
        if axis_norm > 1e-6 and angle_deg != 0.0:
            axis = axis / axis_norm
            t.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])
        elif angle_deg > 179.999:
            # Opposite to Z; rotate 180 around X
            t.RotateWXYZ(180, 1, 0, 0)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(base)
        tf.SetTransform(t)
        tf.Update()
        return tf.GetOutput()

    def on_left_click(self, obj, event):
        if not self.chk_select_mode.isChecked():
            # Pass event to next observer (InteractorStyle)
            self.interactor.GetInteractorStyle().OnLeftButtonDown()
            return
        
        # Determine or create active face state
        if not self.active_face:
            camera = self.renderer.GetActiveCamera()
            dop = np.array(camera.GetDirectionOfProjection())
            axis = int(np.argmax(np.abs(dop)))
            sign = 1 if dop[axis] >= 0 else -1
            self.active_face = ['posX','posY','posZ'][axis] if sign > 0 else ['negX','negY','negZ'][axis]
        state = self.face_states.setdefault(self.active_face, {'point_widgets': [], 'outline_actor': None, 'outline_source': None, 'heatmap_actors': []})
        
        # If已经有2个点，标记为需要自动清理以开始新的ROI（更友好）
        has_full_roi = len(state['point_widgets']) >= 2

        click_pos = self.interactor.GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        
        pos = self.picker.GetPickPosition()
        
        if self.picker.GetCellId() != -1:
            # 若当前面已有完整ROI，则在第一次点击时自动清理该面的选择，然后继续记录新点
            if has_full_roi:
                self.clear_selection()
                state = self.face_states.setdefault(self.active_face, {'point_widgets': [], 'outline_actor': None, 'outline_source': None, 'heatmap_actors': []})
            if len(state['point_widgets']) == 0 and not (getattr(self, "chk_multi_overlay", None) and self.chk_multi_overlay.isChecked()):
                # Clear only this face's overlays
                for actor in state.get('heatmap_actors', []):
                    self.renderer.RemoveActor(actor)
                state['heatmap_actors'] = []
            self.log(f"Picked: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Calculate scale based on model bounds
            bounds = self.actor.GetBounds()
            diag = np.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
            scale = diag * 0.05 # 5% of diagonal
            
            # Add Draggable Handle Widget with Arrow Glyph
            hw = vtk.vtkHandleWidget()
            hw.SetInteractor(self.interactor)
            
            rep = vtk.vtkPolygonalHandleRepresentation3D()
            rep.SetWorldPosition(pos)
            # Oriented arrow normal per-face
            face_normals = {
                'posZ': (0, 0, 1),
                'negZ': (0, 0, -1),
                'posY': (0, 1, 0),
                'negY': (0, -1, 0),
                'posX': (1, 0, 0),
                'negX': (-1, 0, 0),
            }
            nrm = face_normals.get(self.active_face or 'posZ', (0, 0, 1))
            # Use mirrored normal so the whole glyph points opposite to face normal
            nrm = (-nrm[0], -nrm[1], -nrm[2])
            rep.SetHandle(self.create_arrow_glyph_oriented(nrm, scale))
            # Face-specific color to区分不同面
            face_colors = {
                'posZ': (0, 1, 0),     # Green
                'negZ': (1, 1, 0),     # Yellow
                'posY': (0, 1, 1),     # Cyan
                'negY': (1, 0, 1),     # Magenta
                'posX': (0, 0.7, 1),   # Light Blue
                'negX': (1, 0, 0),     # Red
            }
            c = face_colors.get(self.active_face or 'posZ', (0, 1, 0))
            rep.GetProperty().SetColor(*c)
            
            hw.SetRepresentation(rep)
            hw.EnabledOn()
            
            # Add Observer for Interaction (Drag)
            hw.AddObserver("InteractionEvent", self.on_point_drag)
            # Lock rotation during handle drag to avoid 'follow' rotation on release
            hw.AddObserver("StartInteractionEvent", self.on_point_drag_start)
            hw.AddObserver("EndInteractionEvent", self.on_point_drag_end)
            
            state['point_widgets'].append(hw)
            
            if len(state['point_widgets']) == 2:
                self.create_roi_box()
                # Get current positions (might be slightly different from picked due to precision or if they were already moved - though here they are fresh)
                if hasattr(state['point_widgets'][0], 'GetRepresentation'):
                    p1 = state['point_widgets'][0].GetRepresentation().GetWorldPosition()
                    p2 = state['point_widgets'][1].GetRepresentation().GetWorldPosition()
                else:
                    p1 = state['point_widgets'][0].GetPosition()
                    p2 = state['point_widgets'][1].GetPosition()
                
                self.lbl_selection.setText(f"P1: ({p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}) | P2: ({p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f})")
                self.log(f"Region defined. P1: {p1}, P2: {p2}")
                
                # Auto-disable selection mode to allow immediate dragging
                self.chk_select_mode.setChecked(False)
            
            self.vtkWidget.GetRenderWindow().Render()
        else:
             self.log("Clicked outside model.")

    def on_interaction(self, obj, event):
        """Unified handler for interaction events."""
        self.update_rotation_label()

    def update_rotation_label(self):
        camera = self.renderer.GetActiveCamera()
        if not camera:
            return
            
        # Get Azimuth and Elevation
        # Note: VTK's Azimuth/Elevation are relative to the camera's initial position/view up.
        # However, for user feedback, absolute orientation (View Plane Normal) might be more useful.
        # Or we can just calculate angles based on the Direction of Projection.
        
        # Method 1: Get Orientation (Roll, Pitch, Yaw) - returns tuple
        # orientation = camera.GetOrientation()
        
        # Method 2: Calculate from Direction of Projection (DOP)
        dop = camera.GetDirectionOfProjection()
        # Convert to spherical coordinates or just show vector?
        # Azimuth: angle in XY plane (atan2(dy, dx))
        # Elevation: angle from Z axis (acos(dz))
        
        import math
        azimuth = math.degrees(math.atan2(dop[1], dop[0]))
        elevation = math.degrees(math.asin(dop[2]))
        
        # Normalize azimuth to 0-360 or -180 to 180
        
        # Position label
        width, height = self.vtkWidget.GetRenderWindow().GetSize()
        y_pos_rot = height - 120 # Below UR label (which is at height-90)
        
        text = f"Cam Rot: Azimuth {azimuth:.1f}, Elevation {elevation:.1f}"
        
        if self.txt_actor_rot is None:
            self.txt_actor_rot = vtk.vtkTextActor()
            self.txt_actor_rot.GetTextProperty().SetFontSize(14)
            self.txt_actor_rot.GetTextProperty().SetFontFamilyToArial()
            self.txt_actor_rot.GetTextProperty().SetBold(0)
            self.txt_actor_rot.GetTextProperty().SetColor(1, 1, 0) # Yellow for distinction
            self.renderer.AddActor(self.txt_actor_rot)
            
        self.txt_actor_rot.SetInput(text)
        self.txt_actor_rot.SetPosition(20, y_pos_rot)
        # We don't call Render() here to avoid infinite loops if this is called during Render
        # But for Interactor events, it should be fine. 
        # Actually, ModifiedEvent on Interactor might be too frequent.
        # Better to just update the actor and let the next Render cycle pick it up?
        # But if we are dragging, we want it to update.
        # The Interactor handles the render loop.

    def update_coordinate_labels(self, p1, p2):
        # Helper to create/update text actors
        def create_or_update_text(actor, text, x, y, color=(0, 1, 0)):
            if actor is None:
                actor = vtk.vtkTextActor()
                actor.GetTextProperty().SetFontSize(14) # Smaller font
                actor.GetTextProperty().SetColor(color)
                actor.GetTextProperty().SetBold(1)
                actor.GetTextProperty().SetFontFamilyToArial() # Clean sans-serif
                self.renderer.AddActor(actor)
            actor.SetInput(text)
            actor.SetPosition(x, y)
            return actor

        # Position text at top-left
        # Stack them vertically to avoid clipping on the right side
        # P1 (UL) at (20, height - 60)
        # P2 (UR) at (20, height - 90)
        
        width, height = self.vtkWidget.GetRenderWindow().GetSize()
        
        y_pos_ul = height - 60 
        y_pos_ur = height - 90 # 30px below UL
        
        # P1 (UL - Upper Left area)
        self.txt_actor_p1 = create_or_update_text(
            self.txt_actor_p1, 
            f"UL: ({p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f})", 
            20, y_pos_ul, 
            color=(0, 1, 0) # Green
        )
        
        # P2 (UR - Upper Right area) - Now stacked below UL
        self.txt_actor_p2 = create_or_update_text(
            self.txt_actor_p2, 
            f"UR: ({p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f})", 
            20, y_pos_ur, 
            color=(0, 1, 0) # Green
        )
        
        # Initialize Rotation Label if not already
        if self.txt_actor_rot is None:
             self.update_rotation_label()

    def on_point_drag(self, obj, event):
        # Callback when any point widget is dragged
        state = self.face_states.get(self.active_face or "", None)
        if state and len(state.get('point_widgets', [])) == 2 and state.get('outline_source'):
            if hasattr(state['point_widgets'][0], 'GetRepresentation'):
                p1 = state['point_widgets'][0].GetRepresentation().GetWorldPosition()
                p2 = state['point_widgets'][1].GetRepresentation().GetWorldPosition()
            else:
                p1 = state['point_widgets'][0].GetPosition()
                p2 = state['point_widgets'][1].GetPosition()
            
            # Update Outline Box
            state['outline_source'].SetBounds(
                min(p1[0], p2[0]), max(p1[0], p2[0]),
                min(p1[1], p2[1]), max(p1[1], p2[1]),
                min(p1[2], p2[2]), max(p1[2], p2[2])
            )
            
            # Update Label
            self.lbl_selection.setText(f"P1: ({p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}) | P2: ({p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f})")
            
            # Update 3D Overlay Labels
            self.update_coordinate_labels(p1, p2)
            
            self.vtkWidget.GetRenderWindow().Render()

    def on_point_drag_start(self, obj, event):
        if self._restore_style_timer:
            self._restore_style_timer.stop()
            self._restore_style_timer = None
        self.suppress_camera_events = True

    def on_point_drag_end(self, obj, event):
        def restore():
            self.suppress_camera_events = False
        self._restore_style_timer = QTimer(self)
        self._restore_style_timer.setSingleShot(True)
        def force_release_and_restore():
            try:
                x, y = self.interactor.GetEventPosition()
            except Exception:
                x, y = 0, 0
            try:
                ctrl = self.interactor.GetControlKey()
                shift = self.interactor.GetShiftKey()
            except Exception:
                ctrl = 0
                shift = 0
            try:
                self.interactor.SetEventInformationFlipY(x, y, ctrl, shift, chr(0), 0, None)
            except Exception:
                try:
                    self.interactor.SetEventInformation(x, y, ctrl, shift, chr(0), 0, None)
                except Exception:
                    pass
            try:
                self.interactor.InvokeEvent("LeftButtonReleaseEvent")
            except Exception:
                pass
            restore()
        self._restore_style_timer.timeout.connect(force_release_and_restore)
        self._restore_style_timer.start(150)

    def create_roi_box(self):
        # Create Outline Box
        state = self.face_states.get(self.active_face, None)
        if not state or len(state.get('point_widgets', [])) < 2:
            return
        if hasattr(state['point_widgets'][0], 'GetRepresentation'):
            p1 = state['point_widgets'][0].GetRepresentation().GetWorldPosition()
            p2 = state['point_widgets'][1].GetRepresentation().GetWorldPosition()
        else:
            p1 = state['point_widgets'][0].GetPosition()
            p2 = state['point_widgets'][1].GetPosition()
        
        state['outline_source'] = vtk.vtkOutlineSource()
        state['outline_source'].SetBounds(
            min(p1[0], p2[0]), max(p1[0], p2[0]),
            min(p1[1], p2[1]), max(p1[1], p2[1]),
            min(p1[2], p2[2]), max(p1[2], p2[2])
        )
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(state['outline_source'].GetOutputPort())
        
        state['outline_actor'] = vtk.vtkActor()
        state['outline_actor'].SetMapper(mapper)
        # Match outline color with face arrow color
        face_colors = {
            'posZ': (0, 1, 0),
            'negZ': (1, 1, 0),
            'posY': (0, 1, 1),
            'negY': (1, 0, 1),
            'posX': (0, 0.7, 1),
            'negX': (1, 0, 0),
        }
        oc = face_colors.get(self.active_face or 'posZ', (1, 0, 0))
        state['outline_actor'].GetProperty().SetColor(*oc)
        state['outline_actor'].GetProperty().SetLineWidth(2.0)
        
        self.renderer.AddActor(state['outline_actor'])
        
        # Initial Overlay Labels
        self.update_coordinate_labels(p1, p2)
        
        self.vtkWidget.GetRenderWindow().Render()

    def load_stl_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select STL", "", "STL Files (*.stl)")
        if path:
            self.load_stl(path)

    def load_stl(self, path):
        self.stl_path = path
        self.lbl_stl_path.setText(os.path.basename(path))
        
        # Read STL
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        
        # Center the model
        bounds = reader.GetOutput().GetBounds()
        center_x = (bounds[1] + bounds[0]) / 2.0
        center_y = (bounds[3] + bounds[2]) / 2.0
        center_z = (bounds[5] + bounds[4]) / 2.0
        
        transform = vtk.vtkTransform()
        transform.Translate(-center_x, -center_y, -center_z)
        
        self.transform_filter = vtk.vtkTransformPolyDataFilter()
        self.transform_filter.SetInputConnection(reader.GetOutputPort())
        self.transform_filter.SetTransform(transform)
        self.transform_filter.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.transform_filter.GetOutputPort())
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(self.axes) # Re-add axes
        self.renderer.AddActor(self.actor)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        
        # Reset Zoom Slider
        camera = self.renderer.GetActiveCamera()
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(100)
        self.slider_zoom.blockSignals(False)
        self.lbl_zoom_val.setText("1.0x")
        
        self.log(f"Loaded STL: {path} (Centered)")

    def load_data_dialog(self, mode):
        if mode == "file":
            ftype = self.combo_file_type.currentText()
            filt = "Excel Files (*.xlsx)" if ftype == "Excel" else "NPY Files (*.npy)"
            path, _ = QFileDialog.getOpenFileName(self, f"Select {ftype}", "", filt)
            if path:
                self.data_path = path
                self.data_source_type = "file"
                self.lbl_data_path.setText(os.path.basename(path))
                self.refresh_filters()
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if path:
                self.data_path = path
                self.data_source_type = "folder"
                self.lbl_data_path.setText(path)
                self.refresh_filters()

    def refresh_filters(self):
        if not self.data_path: return
        
        self.log("Scanning data...")
        self.btn_generate.setEnabled(False)
        
        # Start Thread
        self.loader = DataLoader(self.data_path, self.data_source_type, 
                                 self.combo_file_type.currentText(), self.chk_merge.isChecked())
        self.loader.finished.connect(self.on_data_loaded)
        self.loader.error.connect(lambda e: self.log(f"Error: {e}"))
        self.loader.log.connect(self.log)
        self.loader.start()

    def on_data_loaded(self, df):
        self.cached_df = df
        self.btn_generate.setEnabled(True)
        self.log(f"Data Loaded. {len(df)} rows.")
        
        # Update Filter UI
        # Clear old
        while self.layout_filters.count():
            child = self.layout_filters.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        self.filter_checkboxes = {}
        
        def add_category(name, col):
            if col not in df.columns: return
            vals = sorted(df[col].fillna('NA').astype(str).unique())
            lbl = QLabel(name)
            lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
            self.layout_filters.addWidget(lbl)
            
            self.filter_checkboxes[col] = {}
            for v in vals:
                if not v: v = "NA"
                chk = QCheckBox(v)
                chk.setChecked(True)
                chk.stateChanged.connect(self.generate_heatmap)
                self.layout_filters.addWidget(chk)
                self.filter_checkboxes[col][v] = chk
                
        add_category("Defect Mode", "Defect Mode")
        add_category("Shape", "Shape")
        self.layout_filters.addStretch()

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

    def generate_heatmap(self, *args):
        if self.cached_df is None or not self.actor:
            QMessageBox.warning(self, "Warning", "Please load both STL and Data first.")
            return

        # 1. Filter Data
        df = self.cached_df.copy()
        for col, checkboxes in self.filter_checkboxes.items():
            if col in df.columns:
                selected = [val for val, chk in checkboxes.items() if chk.isChecked()]
                df[col] = df[col].fillna('NA').astype(str).replace('', 'NA')
                df = df[df[col].isin(selected)]
                
        if len(df) == 0:
            QMessageBox.warning(self, "Warning", "No data after filtering.")
            return
            
        if not (hasattr(self, "chk_multi_overlay") and self.chk_multi_overlay.isChecked()):
            self.clear_heatmaps()
            
        self.log(f"Generating heatmap from {len(df)} contours...")
        
        # 1.1 Pre-scan for Coordinate Normalization
        all_points = []
        
        # Debug: Print first contour to understand format
        if len(df) > 0:
            first_c = df['Contour'].iloc[0]
            self.log(f"Debug: First contour type: {type(first_c)}")
            self.log(f"Debug: First contour raw: {str(first_c)[:100]}...")

        import ast
        for idx, contour in enumerate(df['Contour'].values):
            if isinstance(contour, str):
                 try:
                    # Try ast.literal_eval first (safest for python-style lists)
                    points_list = ast.literal_eval(contour)
                    pts = np.array(points_list).reshape(-1, 2)
                    if pts.size > 0:
                        all_points.append(pts)
                 except Exception as e_ast:
                     # Fallback to manual parsing
                     try:
                        # Robust parsing for both comma and space separated values
                        clean = contour.replace('[', ' ').replace(']', ' ').replace('\n', ' ')
                        # Replace commas with spaces to handle mixed formats
                        clean = clean.replace(',', ' ')
                        parts = clean.split()
                        points = [float(p) for p in parts if p.strip()]
                        pts = np.array(points).reshape(-1, 2)
                        if pts.size > 0:
                            all_points.append(pts)
                     except Exception as e:
                         if idx == 0: self.log(f"Parse error (idx 0): {e}") 
                         continue
            else:
                try:
                    pts = np.array(contour)
                    if pts.ndim == 1: pts = pts.reshape(-1, 2)
                    if pts.size > 0:
                        all_points.append(pts)
                except: continue
                
        if not all_points:
            self.log("Error: No valid contour points found.")
            return

        # Check range
        min_val = np.min([p.min() for p in all_points if len(p) > 0])
        max_val_coord = np.max([p.max() for p in all_points if len(p) > 0])
        
        self.log(f"Contour Data Range: Min={min_val:.4f}, Max={max_val_coord:.4f}")
        
        need_normalization = False
        norm_w, norm_h = 1.0, 1.0
        
        if max_val_coord > 1.05: # Allow small epsilon for 1.0
            self.log("Detected pixel coordinates (Max > 1.0). Auto-normalizing...")
            need_normalization = True
            # Estimate dimensions from max values
            # This is a guess if we don't have image size. 
            # We assume the max coordinate represents the extent.
            # Or we can ask user? For now, let's normalize by max extent found.
            
            # Find max x and max y separately
            max_x = np.max([p[:, 0].max() for p in all_points if len(p) > 0])
            max_y = np.max([p[:, 1].max() for p in all_points if len(p) > 0])
            
            # Add a small buffer? Or assume nearest power of 2? 
            # Usually images are 1920, 1024, 2048 etc.
            # Let's just normalize to the data bounds for now.
            norm_w = max_x if max_x > 0 else 1.0
            norm_h = max_y if max_y > 0 else 1.0
            self.log(f"Normalizing by Max X={norm_w}, Max Y={norm_h}")

        # 2. Generate Heatmap Image (Texture)
        # Texture size
        tex_w, tex_h = 1024, 1024
        count_map = np.zeros((tex_h, tex_w), dtype=np.float32)
        
        valid_cnt = 0
        
        for contour_array in all_points:
            if contour_array is None or len(contour_array) < 3: continue
            
            # Map normalized [0,1] to texture size [0, tex_w/h]
            pts = contour_array.copy()
            
            if need_normalization:
                pts[:, 0] /= norm_w
                pts[:, 1] /= norm_h
                
            pts[:, 0] *= tex_w
            pts[:, 1] *= tex_h
            pts = pts.astype(np.int32)
            
            mask = np.zeros((tex_h, tex_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            # Dilation removed as requested
            count_map[mask == 1] += 1
            valid_cnt += 1
            
        self.log(f"Processed {valid_cnt} contours.")
        
        # Normalize and Colorize (Using Custom Logic from 2D version)
        max_val = count_map.max()
        if max_val <= 1:
            # If max count is 1, everything is low frequency. 
            heatmap_normalized = np.zeros_like(count_map, dtype=np.uint8)
            heatmap_normalized[count_map > 0] = 0 
        else:
            # Linear mapping: 1 -> 0, max -> 255
            count_float = count_map.astype(np.float32)
            norm_float = (count_float - 1) / (max_val - 1) * 255
            norm_float = np.clip(norm_float, 0, 255)
            heatmap_normalized = norm_float.astype(np.uint8)
            
        norm_map = heatmap_normalized # Alias for compatibility
        
        # Use Custom Colormap
        custom_lut = self.create_custom_colormap()
        
        # Convert single-channel normalized heatmap to 3-channel BGR for LUT
        heatmap_normalized_bgr = cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2BGR)
        
        heatmap_color = cv2.LUT(heatmap_normalized_bgr, custom_lut)
        
        # Note: In 2D version, we do NOT set the background to black.
        # We rely on alpha channel for transparency.
        # Setting background to black [0,0,0] causes dark/black halos during interpolation.
        # Leaving it as the LUT output (Blue) ensures that any color bleeding is consistent with the heatmap edges.
        
        # Add Alpha Channel
        # Opacity slider
        user_opacity = self.slider_opacity.value() / 100.0
        
        # Create RGBA
        b, g, r = cv2.split(heatmap_color)
        
        # Alpha: 0 where count is 0, else user_opacity * 255
        alpha = np.zeros_like(norm_map)
        alpha[count_map > 0] = int(255 * user_opacity)
        
        # Ensure Border is Transparent (1px padding)
        # This prevents smearing when using ClampToEdge
        alpha[0, :] = 0
        alpha[-1, :] = 0
        alpha[:, 0] = 0
        alpha[:, -1] = 0
        
        heatmap_rgba = cv2.merge([r, g, b, alpha]) # VTK expects RGB(A)
        
        # Convert to VTK Texture
        h, w, _ = heatmap_rgba.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(w, h, 1)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
        
        # Copy data
        # Flip Y because VTK texture coords origin is bottom-left, images are top-left
        heatmap_rgba = cv2.flip(heatmap_rgba, 0)
        
        vtk_arr = vtk.vtkUnsignedCharArray()
        vtk_arr.SetNumberOfComponents(4)
        vtk_arr.SetArray(heatmap_rgba.ravel(), w*h*4, 1)
        vtk_image.GetPointData().SetScalars(vtk_arr)
        
        texture = vtk.vtkTexture()
        texture.SetInputData(vtk_image)
        # Disable Interpolation to remove artifacts and keep lines sharp
        texture.InterpolateOff() 
        texture.RepeatOff() # Don't repeat
        texture.EdgeClampOn() # Clamp to edge (which is transparent)
        
        # Apply Texture Map to Plane (Project from Z axis)
        # We need to project UVs onto the actor
        bounds = self.actor.GetBounds()
        
        # Create a new actor for this heatmap overlay
        overlay_mapper = vtk.vtkPolyDataMapper()
        
        # We need to apply TextureMapToPlane to the geometry
        # But wait, if we modify the input geometry (Shared PolyData), it affects all actors sharing it?
        # vtkPolyDataMapper takes an input connection.
        # vtkTextureMapToPlane produces a NEW PolyData with TCoords.
        
        plane = vtk.vtkTextureMapToPlane()
        plane.SetInputConnection(self.transform_filter.GetOutputPort()) # Use original geometry source
        
        # 2. Determine Projection Plane based on Point Widgets and Active Face
        proj_normal = (0, 0, 1)
        
        state = self.face_states.get(self.active_face or "", None)
        if state and len(state.get('point_widgets', [])) == 2:
            if hasattr(state['point_widgets'][0], 'GetRepresentation'):
                p1 = state['point_widgets'][0].GetRepresentation().GetWorldPosition()
                p2 = state['point_widgets'][1].GetRepresentation().GetWorldPosition()
            else:
                p1 = state['point_widgets'][0].GetPosition()
                p2 = state['point_widgets'][1].GetPosition()
            
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            zmin, zmax = min(z1, z2), max(z1, z2)
            
            # Determine Projection Plane and Normal from active face key（含正负方向）
            fk = self.active_face or 'posZ'
            if fk == 'posZ':  # Top
                plane.SetOrigin(xmin, ymin, zmax)
                plane.SetPoint1(xmax, ymin, zmax)
                plane.SetPoint2(xmin, ymax, zmax)
                proj_normal = (0, 0, 1)
            elif fk == 'negZ':  # Bottom
                plane.SetOrigin(xmin, ymin, zmin)
                plane.SetPoint1(xmax, ymin, zmin)
                plane.SetPoint2(xmin, ymax, zmin)
                proj_normal = (0, 0, -1)
            elif fk == 'posY':  # Back (+Y)
                plane.SetOrigin(xmin, ymax, zmin)
                plane.SetPoint1(xmax, ymax, zmin)
                plane.SetPoint2(xmin, ymax, zmax)
                proj_normal = (0, 1, 0)
            elif fk == 'negY':  # Front (-Y)
                plane.SetOrigin(xmin, ymin, zmin)
                plane.SetPoint1(xmax, ymin, zmin)
                plane.SetPoint2(xmin, ymin, zmax)
                proj_normal = (0, -1, 0)
            elif fk == 'posX':  # Right (+X)
                plane.SetOrigin(xmax, ymin, zmin)
                plane.SetPoint1(xmax, ymax, zmin)
                plane.SetPoint2(xmax, ymin, zmax)
                proj_normal = (1, 0, 0)
            else:  # negX Left (-X)
                plane.SetOrigin(xmin, ymin, zmin)
                plane.SetPoint1(xmin, ymax, zmin)
                plane.SetPoint2(xmin, ymin, zmax)
                proj_normal = (-1, 0, 0)
            
            self.log(f"Projection Face {self.active_face}: Normal {proj_normal}")
            
        else:
            # Default Global Bounds (XY Plane)
            plane.SetOrigin(bounds[0], bounds[2], bounds[5]) # Bottom-Left-Top
            plane.SetPoint1(bounds[1], bounds[2], bounds[5]) # Bottom-Right-Top
            plane.SetPoint2(bounds[0], bounds[3], bounds[5]) # Top-Left-Top
            self.log("Projection: Full XY Plane")
        
        plane.Update()
        
        # 3. Filter Backfaces and Side-faces (Normal-based Filtering)
        # This prevents "streaking" on side faces and "print-through" on back faces.
        
        # Clip geometry to Box Widget (if active for current face) to limit spatial extent
        clip_filter = None
        if state and len(state.get('point_widgets', [])) == 2:
            # Create a box implicit function for clipping
            if hasattr(state['point_widgets'][0], 'GetRepresentation'):
                p1 = state['point_widgets'][0].GetRepresentation().GetWorldPosition()
                p2 = state['point_widgets'][1].GetRepresentation().GetWorldPosition()
            else:
                p1 = state['point_widgets'][0].GetPosition()
                p2 = state['point_widgets'][1].GetPosition()
            
            # Add padding to box to ensure surface points are included, especially for flat surfaces
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            margin = max(dist * 0.1, 0.5) # 10% margin or at least 0.5 units
            
            box = vtk.vtkBox()
            box.SetBounds(
                min(p1[0], p2[0]) - margin, max(p1[0], p2[0]) + margin,
                min(p1[1], p2[1]) - margin, max(p1[1], p2[1]) + margin,
                min(p1[2], p2[2]) - margin, max(p1[2], p2[2]) + margin
            )
            
            # Use vtkClipPolyData to clip OUTSIDE the box
            clip_filter = vtk.vtkClipPolyData()
            clip_filter.SetInputConnection(plane.GetOutputPort()) # Clip AFTER texture mapping? No, clip before or after?
            # If we clip after, UVs are already generated.
            # ClipPolyData clips cells.
            clip_filter.SetClipFunction(box)
            clip_filter.InsideOutOn() # Keep INSIDE
            clip_filter.Update()
            
        # First, ensure we have normals
        final_connection = clip_filter.GetOutputPort() if clip_filter else plane.GetOutputPort()
        
        normal_filter = vtk.vtkPolyDataNormals()
        normal_filter.SetInputConnection(final_connection)
        normal_filter.ComputePointNormalsOn()
        # Enable Splitting to handle sharp edges correctly
        # This duplicates vertices at corners, ensuring Side Face vertices are separate from Front Face vertices
        # This prevents "streaking" where a shared vertex has a valid UV but the next one is masked.
        normal_filter.SplittingOn()
        normal_filter.SetFeatureAngle(45.0) 
        normal_filter.Update()
        
        poly_data = normal_filter.GetOutput()
        
        # Use Numpy to filter TCoords based on Normal dot product
        try:
            from vtk.util import numpy_support
            
            normals_array = poly_data.GetPointData().GetNormals()
            tcoords_array = poly_data.GetPointData().GetTCoords()
            
            if normals_array is not None and tcoords_array is not None:
                normals = numpy_support.vtk_to_numpy(normals_array)
                tcoords = numpy_support.vtk_to_numpy(tcoords_array)
                
                # Normalize projection vector
                pn = np.array(proj_normal)
                norm_pn = np.linalg.norm(pn)
                if norm_pn > 0:
                    pn = pn / norm_pn
                
                # Calculate Dot Product: Normal . Projection (should be > 0 for facing)
                # However, Projection Direction is usually "Look Direction".
                # If we project *along* Z (0,0,1), we are looking *towards* Z?
                # vtkTextureMapToPlane projects *onto* the plane.
                # If plane normal is Z, then we want faces with Normal ~ Z.
                
                dots = np.dot(normals, pn)
                
                # Threshold: 
                # 1.0 = Perfectly facing
                # 0.0 = Perpendicular (Side)
                # -1.0 = Back facing
                # We want to mask out anything < threshold
                
                # Set TCoords to (0,0) [Transparent] for masked points
                # Increased threshold to 0.2 to strictly filter out side faces and prevent "streaking"
                mask = dots < 0.2 
                tcoords[mask] = [0.0, 0.0]
                
                tcoords_array.Modified()
                
        except ImportError:
            self.log("Warning: vtk.util.numpy_support not found. Normal filtering skipped.")
        except Exception as e:
            self.log(f"Error in normal filtering: {e}")

        overlay_mapper.SetInputData(poly_data)
        
        overlay_actor = vtk.vtkActor()
        overlay_actor.SetMapper(overlay_mapper)
        overlay_actor.SetTexture(texture)
        
        # Avoid Z-fighting by shifting slightly?
        # Or rely on transparency?
        # A small shift along normal is good practice.
        # But we don't know the normal easily here without complexity.
        # VTK handles coincident topology if we set ResolveCoincidentTopology.
        
        overlay_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        overlay_mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(0, -1) # Push overlay forward
        
        # Ensure lighting doesn't mess up the colors (optional)
        overlay_actor.GetProperty().SetAmbient(1.0)
        overlay_actor.GetProperty().SetDiffuse(0.0)
        overlay_actor.GetProperty().SetSpecular(0.0)
        
        self.renderer.AddActor(overlay_actor)
        self.heatmap_actors.append(overlay_actor)
        # Track per-face overlay
        if self.active_face:
            st = self.face_states.setdefault(self.active_face, {'point_widgets': [], 'outline_actor': None, 'outline_source': None, 'heatmap_actors': []})
            st.setdefault('heatmap_actors', []).append(overlay_actor)
        
        self.vtkWidget.GetRenderWindow().Render()
        self.log("Heatmap overlay added.")

        # Update label but keep selection active for adjustment
        self.lbl_selection.setText("Heatmap Generated. Adjust Box and Regenerate, or Clear Selection.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeatMap3DGUI()
    window.show()
    sys.exit(app.exec_())
