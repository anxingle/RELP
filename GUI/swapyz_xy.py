import sys
import os
import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                             QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
                             QHeaderView, QListWidget, QListWidgetItem, QSizePolicy, QMessageBox,QStyleFactory)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon,QPalette, QColor
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import csv
import numpy as np

class STLViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D Drop Angle Analyzer")

        self.setStyleSheet("""
            QWidget {
                background-color: #333333;  
            }
            QPushButton {
                background-color: #444444;  
                color: #FFFFFF;             
                border: 1px solid #555555; 
                padding: 5px;
                border-radius: 5px;       
            }
            QPushButton:pressed {
                background-color: #666666;  
            }
            QPushButton:hover {
                background-color: #555555;  
            }
        """)

        # Set initial window size with the desired aspect ratio
        initial_width = 950
        initial_height = 700
        self.resize(initial_width, initial_height)  # Set initial size

        # Define the aspect ratio (e.g., 4:3)
        self.aspect_ratio = initial_width / initial_height

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create central widget layout
        self.central_widget_layout = QHBoxLayout()
        self.main_layout.addLayout(self.central_widget_layout)

        # Create left layout for controls and table
        self.left_layout = QVBoxLayout()
        self.central_widget_layout.addLayout(self.left_layout, 3)

        # Create right layout for VTK render window
        self.right_layout = QVBoxLayout()
        self.central_widget_layout.addLayout(self.right_layout, 5)

        # Set up the VTK render window
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.right_layout.addWidget(self.vtkWidget)

        # Add file selection buttons
        self.file_selection_layout = QHBoxLayout()
        self.model_button = QPushButton("Select 3D Model")
        self.model_button.clicked.connect(self.select_model_file)
        self.background_button = QPushButton("Select Image Folder")
        self.background_button.clicked.connect(self.select_background_file)
        # Add Reset button to reset all sliders
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_values)

        self.file_selection_layout.addWidget(self.model_button)
        self.file_selection_layout.addWidget(self.background_button)
        self.file_selection_layout.addWidget(self.reset_button)
        self.left_layout.addLayout(self.file_selection_layout)

        # Add control sliders and input boxes for rotations
        self.rotation_layout = QVBoxLayout()
        self.alpha_slider, self.alpha_input, self.alpha_label = self.create_slider_with_input("Rx", -1800, 1800)
        self.beta_slider, self.beta_input, self.beta_label = self.create_slider_with_input("Ry", -1800, 1800)
        self.gamma_slider, self.gamma_input, self.gamma_label = self.create_slider_with_input("Rz", -1800, 1800)

        self.rotation_layout.addLayout(self.create_slider_layout(self.gamma_slider, self.gamma_label, self.gamma_input))
        self.rotation_layout.addLayout(self.create_slider_layout(self.beta_slider, self.beta_label, self.beta_input))
        self.rotation_layout.addLayout(self.create_slider_layout(self.alpha_slider, self.alpha_label, self.alpha_input))
        self.left_layout.addLayout(self.rotation_layout)

        # Add control sliders without input boxes for position and scale
        self.position_layout = QHBoxLayout()
        self.pos_x_slider, self.posx_input, self.pos_x_label = self.create_slider_with_input("Pos X", -200, 200)
        self.pos_y_slider, self.posy_input, self.pos_y_label = self.create_slider_with_input("Pos Y", -200, 200)
        self.scale_slider, self.scale_input, self.scale_label = self.create_slider_with_input("Scale", 0, 15,5)
        self.position_layout.addLayout(self.create_slider_layout(self.pos_x_slider, self.pos_x_label, self.posx_input))
        self.position_layout.addLayout(self.create_slider_layout(self.pos_y_slider, self.pos_y_label, self.posy_input))
        self.position_layout.addLayout(self.create_slider_layout(self.scale_slider, self.scale_label, self.scale_input))
        self.left_layout.addLayout(self.position_layout)

        # Add table for recording transformations
        self.table_layout = QHBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Rz", "Ry", "Rx"])

        # Set the resize mode for columns
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        # Ensure the table expands to fill available space
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add the table to the layout
        self.table_layout.addWidget(self.table)

        # Add the table layout to the left layout
        self.left_layout.addLayout(self.table_layout)

        # Create and set up the image preview list
        self.image_preview_list = QListWidget()
        self.image_preview_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.image_preview_list.setMinimumWidth(100)
        self.image_preview_list.itemClicked.connect(self.load_selected_image)

        # Add the label and the list widget to the layout
        image_preview_layout = QVBoxLayout()
        image_preview_layout.addWidget(self.image_preview_list)

        self.table_layout.addLayout(image_preview_layout, 1)
        self.table_layout.addWidget(self.table, 4)

        self.table.setStyleSheet("""
            QHeaderView::section {
                background-color: #2d2d2d;
                color: white;
                padding: 4px;
                border: none;
            }
            QTableWidget {
                gridline-color: #2d2d2d;
                background-color: #2d2d2d;
                color: white;
            }
            QTableWidget::item {
                border: none;
            }
        """)

        # Apply consistent styling to QListWidget
        self.image_preview_list.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                color: white;
                border: none;
            }
        """)

        # Record, Delete, and Export buttons layout
        self.buttons_layout = QHBoxLayout()
        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.record_transformation)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_transformation)
        self.export_button = QPushButton("Export to CSV")
        self.export_button.clicked.connect(self.export_to_csv)

        self.buttons_layout.addWidget(self.record_button)
        self.buttons_layout.addWidget(self.delete_button)
        self.buttons_layout.addWidget(self.export_button)
        self.left_layout.addLayout(self.buttons_layout)

        # Create a horizontal layout to contain both labels
        footer_layout = QHBoxLayout()

        # Create the left-aligned label
        left_label = QLabel("   Ecosystem Reliability  |  By fred_liu2@apple.com")
        left_label.setAlignment(Qt.AlignLeft)
        left_label.setStyleSheet("background-color: transparent; color: #f0f0f0; font-size: 12px;")

        # Create the right-aligned label
        right_label = QLabel("Ver. 3.1  |  License valid till 12/31/2030")
        right_label.setAlignment(Qt.AlignRight)
        right_label.setStyleSheet("background-color: transparent; color: #f0f0f0; font-size: 12px;")

        # Add labels to the horizontal layout
        footer_layout.addWidget(left_label)
        footer_layout.addStretch()  # This will push the second label to the right
        footer_layout.addWidget(right_label)

        # Add the horizontal layout to the main layout
        self.main_layout.addLayout(footer_layout)

        # Set up VTK
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.stl_file_path = ""
        self.background_image_path = ""
        self.actor = None
        self.transform = vtk.vtkTransform()

        self.vtkWidget.GetRenderWindow().Render()
        self.vtkWidget.GetRenderWindow().GetInteractor().Initialize()
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(None)

        # Add XYZ axes to the renderer
        self.axes = vtk.vtkAxesActor()
        self.axes.SetTotalLength(10, 10, 10)
        self.axes.SetShaftTypeToCylinder()
        self.axes.SetCylinderRadius(0.02)
        self.axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(vtk.vtkTextProperty())
        self.axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(vtk.vtkTextProperty())
        self.axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(vtk.vtkTextProperty())
        self.renderer.AddActor(self.axes)

        # Add small axes in the corner
        self.add_corner_axes()

        self.reset_camera()

        self.start_app()

    def reset_values(self):
        # Reset all sliders and input boxes to their default values
        self.gamma_slider.setValue(0)
        self.beta_slider.setValue(0)
        self.alpha_slider.setValue(0)

        self.pos_x_slider.setValue(0)
        self.pos_y_slider.setValue(0)
        self.scale_slider.setValue(50)  # Scale default value is 5.0, multiplied by 10 for the slider

        # Update transformations
        self.update_transform()

    def swap_yz_axes(self):

        swap_transform = vtk.vtkTransform()
        swap_transform.Identity()

        swap_transform.RotateWXYZ(180, 0, 1, 0)
        swap_transform.RotateWXYZ(-90, 1, 0, 0)

        for actor in self.renderer.GetActors():
            actor.SetUserTransform(swap_transform)

    def resizeEvent(self, event):
        # Get the new size of the window
        new_width = event.size().width()
        new_height = event.size().height()

        # Adjust the window size to maintain the aspect ratio
        self.resize(new_width, new_height)
    def create_slider_with_input(self, label_text, min_val, max_val, init_val=0, has_input=True):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val * 10)
        slider.setMaximum(max_val * 10)
        slider.setValue(init_val * 10)
        value_label = QLabel(f"{label_text}:")

        if has_input:
            input_box = QLineEdit(f"{init_val:.1f}")
            slider.valueChanged.connect(
                lambda: self.update_transform_from_slider(slider, input_box, value_label, label_text))
            input_box.returnPressed.connect(
                lambda: self.update_transform_from_input(input_box, slider, value_label, label_text))
            return slider, input_box, value_label
        else:
            slider.valueChanged.connect(
                lambda: self.update_transform_from_slider_no_input(slider, value_label, label_text))
            return slider, value_label

    def create_slider_layout(self, slider, value_label, input_box=None):
        layout = QVBoxLayout()
        layout.addWidget(slider)
        if input_box:
            sublayout = QHBoxLayout()
            sublayout.addWidget(value_label)
            sublayout.addWidget(input_box)
            layout.addLayout(sublayout)
        return layout

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.renderer.GetActiveCamera().SetPosition(0, 0, 315)  # Set the camera position
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.vtkWidget.GetRenderWindow().Render()

    def update_transform_from_slider(self, slider, input_box, value_label, label_text):
        value = slider.value() / 100.0
        input_box.setText(f"{value:.2f}")
        value_label.setText(f"{label_text}:")
        self.update_transform()

    def update_transform_from_slider_no_input(self, slider, value_label, label_text):
        value = slider.value() / 10.0
        value_label.setText(f"{label_text}:")
        self.update_transform()

    def update_transform_from_slider_o(self, slider, input_box, value_label, label_text):
        value = slider.value() / 10.0
        input_box.setText(f"{value:.1f}")
        value_label.setText(f"{label_text}: {value:.1f}")
        self.update_transform()

    def update_transform_from_slider_no_input_o(self, slider, value_label, label_text):
        value = slider.value() / 10.0
        value_label.setText(f"{label_text}: {value:.1f}")
        self.update_transform()

    def update_transform_from_input(self, input_box, slider, value_label, label_text):
        try:
            value = float(input_box.text())
            slider.setValue(int(value*100))
            value_label.setText(f"{label_text}:")
            self.update_transform()
        except ValueError:
            # Handle the case where the input is not a valid float
            pass

    def update_transform(self, initial=False):
        if not self.actor or not self.transform:
            return
        # Get Euler angles from sliders
        alpha = self.alpha_slider.value() / 100.0  # X-Rot
        beta = self.beta_slider.value() / 100.0  # Y-Rot
        gamma = self.gamma_slider.value() / 100.0  # Z-Rot

        # Get position and scale values from sliders
        pos_x = self.pos_x_slider.value() / 20.0
        pos_y = self.pos_y_slider.value() / 20.0
        scale = self.scale_slider.value() / 100.0

        # Start with the identity transformation
        self.transform.Identity()

        # Apply scaling
        self.transform.Scale(scale, scale, scale)

        # Apply translation
        self.transform.Translate(pos_x, 0,pos_y, )

        # Apply rotations around the fixed world coordinate system
        # self.transform.RotateZ(-gamma)
        # self.transform.RotateY(-beta)
        # self.transform.RotateX(-alpha)

        self.transform.RotateWXYZ(gamma, 0, 0, 1)  # Rotate around global Z axis
        self.transform.RotateWXYZ(beta, 0, 1, 0)  # Rotate around global Y axis
        self.transform.RotateWXYZ(alpha, 1, 0, 0)  # Rotate around global X axis

        # Get the rotation matrix
        matrix = vtk.vtkMatrix4x4()
        self.transform.GetMatrix(matrix)

        # Update input boxes with the actual Euler angles
        self.beta_input.setText(f"{beta:.2f}")
        self.gamma_input.setText(f"{gamma:.2f}")
        self.alpha_input.setText(f"{alpha:.2f}")

        # Update axes transform
        self.axes.SetUserTransform(self.transform)

        if initial:
            # Center the 3D model in the background image initially
            bounds = [0] * 6
            self.actor.GetBounds(bounds)
            center_x = (bounds[1] + bounds[0]) / 2.0
            center_y = (bounds[3] + bounds[2]) / 2.0
            center_z = (bounds[5] + bounds[4]) / 2.0
            self.transform.Translate(-center_x, -center_y, -center_z)

            # Adjust the background image size based on the model size
            model_width = bounds[1] - bounds[0]
            model_height = bounds[3] - bounds[2]
            background_scale_factor = 50  # Scale the background to be larger than the model
            self.background_width = model_width * background_scale_factor
            self.background_height = model_height * background_scale_factor

        self.vtkWidget.GetRenderWindow().Render()

    def select_model_file(self):
        options = QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select 3D Model File", "", "STL Files (*.stl);;All Files (*)",
                                                   options=options)
        if file_path:
            self.stl_file_path = file_path
            self.load_files(load_model=True, load_background=False)

    def select_background_file(self):
        if not self.stl_file_path:
            self.show_message_box(
                "Load 3D Model First",
                "Please load the 3D model files first, thank you.",
                QMessageBox.Warning
            )
            return

        options = QFileDialog.ReadOnly
        folder_path = QFileDialog.getExistingDirectory(self, "Select Background Image Folder", options=options)
        if folder_path:
            self.preview_folder(folder_path)

    def preview_folder(self, folder_path):
        self.image_preview_list.clear()

        image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'gif']
        for file_name in os.listdir(folder_path):
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, os.path.join(folder_path, file_name))
                pixmap = QPixmap(os.path.join(folder_path, file_name)).scaled(50, 50,
                                                                              Qt.KeepAspectRatio)
                item.setIcon(QIcon(pixmap))
                self.image_preview_list.addItem(item)
    def load_selected_image(self, item):
        self.background_image_path = item.data(Qt.UserRole)
        print(f"Selected background image from folder: {self.background_image_path}")
        self.load_files(load_model=False, load_background=True)

    def load_files(self, load_model=True, load_background=True, swap_axes=False):
        if load_model:
            if not self.stl_file_path or not os.path.exists(self.stl_file_path):
                print(f"STL file not found: {self.stl_file_path}")
                return

            # Clear previous actors
            self.renderer.RemoveAllViewProps()

            # Load STL file
            self.reader = vtk.vtkSTLReader()
            self.reader.SetFileName(self.stl_file_path)
            self.reader.Update()

            # Create 3D model actor
            self.actor = vtk.vtkActor()
            self.transform = vtk.vtkTransform()

            # Apply initial scale to make the model visible
            initial_scale = 0.05  # 1/20 of the background size
            self.transform.Scale(initial_scale, initial_scale, initial_scale)

            self.update_transform(initial=True)
            self.transform_filter = vtk.vtkTransformPolyDataFilter()
            self.transform_filter.SetInputConnection(self.reader.GetOutputPort())
            self.transform_filter.SetTransform(self.transform)
            self.transform_filter.Update()

            self.mapper = vtk.vtkPolyDataMapper()
            self.mapper.SetInputConnection(self.transform_filter.GetOutputPort())
            self.actor.SetMapper(self.mapper)

            self.renderer.AddActor(self.actor)

            # Move the model in the Z direction to avoid overlap with the background
            # self.actor.SetPosition(0, 0, 20)  # Adjust the Z position as needed

            # Reset camera and clipping range
            self.renderer.ResetCamera()
            self.renderer.GetActiveCamera().SetClippingRange(0.1, 10000.0)

            # Swap Y and Z axes if requested
            self.swap_yz_axes()

        if load_background:
            if not self.background_image_path or not os.path.exists(self.background_image_path):
                print(f"Background image file not found: {self.background_image_path}")
                return

            # Remove previous background image actor
            if hasattr(self, 'background_actor') and self.background_actor:
                self.renderer.RemoveActor(self.background_actor)

            # Set background image as texture
            self.set_background_image(self.renderer, self.background_image_path)

        self.vtkWidget.GetRenderWindow().Render()

    def set_background_image(self, renderer, image_path):
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Background image file not found: {image_path}")
            return

        # Load the background image using vtkImageReader2Factory
        image_reader_factory = vtk.vtkImageReader2Factory()
        image_reader = image_reader_factory.CreateImageReader2(image_path)
        if not image_reader:
            print(f"Unsupported image format: {image_path}")
            return

        image_reader.SetFileName(image_path)
        image_reader.Update()

        # Check if the image is loaded correctly
        image_data = image_reader.GetOutput()
        if image_data is None:
            print(f"Failed to load image: {image_path}")
            return

        # Get image dimensions to maintain aspect ratio
        width, height = image_data.GetDimensions()[0:2]
        aspect_ratio = width / height

        # Create a texture from the image
        texture = vtk.vtkTexture()
        texture.SetInputConnection(image_reader.GetOutputPort())

        # Rotate the texture by 180 degrees
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(180, 0, 0, 1)  # Rotate 180 degrees around the Z-axis
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)

        # Create a plane to map the texture onto, maintaining aspect ratio
        plane_source = vtk.vtkPlaneSource()

        # Adjust the plane size to fill the widget while maintaining aspect ratio
        widget_aspect_ratio = self.background_width / self.background_height
        if (widget_aspect_ratio > 1):
            scale_factor = self.background_width / width
        else:
            scale_factor = self.background_height / height

        plane_width = width * scale_factor
        plane_height = height * scale_factor

        plane_source.SetOrigin(-plane_width / 2, -plane_height / 2, 0.0)
        plane_source.SetPoint1(plane_width / 2, -plane_height / 2, 0.0)
        plane_source.SetPoint2(-plane_width / 2, plane_height / 2, 0.0)

        transform_filter.SetInputConnection(plane_source.GetOutputPort())

        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputConnection(transform_filter.GetOutputPort())

        self.background_actor = vtk.vtkActor()
        self.background_actor.SetMapper(plane_mapper)
        self.background_actor.SetTexture(texture)
        self.background_actor.SetPosition(0, 0, -50)  # Ensure the background plane is behind the model

        renderer.AddActor(self.background_actor)
        renderer.ResetCamera()
        renderer.GetActiveCamera().ParallelProjectionOn()
        self.renderer.GetActiveCamera().SetClippingRange(0.1, 10000.0)

    def add_corner_axes(self):
        # Create a renderer for the corner axes
        corner_renderer = vtk.vtkRenderer()
        corner_renderer.SetViewport(0.8, 0.8, 1.0, 1.0)  # Position the corner axes in the top right corner
        corner_renderer.SetLayer(1)
        corner_renderer.InteractiveOff()

        # Create the axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2, 2, 2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.05)

        # 对 corner axes 也做同样的 swap_yz_axes
        swap_transform = vtk.vtkTransform()
        swap_transform.RotateWXYZ(190, 0, 1, 0)
        swap_transform.RotateWXYZ(-98, 1, 0, 0)
        axes.SetUserTransform(swap_transform)

        # Set text properties for the axes labels
        text_property = vtk.vtkTextProperty()
        text_property.SetFontSize(10)
        text_property.SetColor(1.0, 1.0, 1.0)  # White text color

        axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(text_property)
        axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(text_property)
        axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(text_property)

        # Add the axes to the corner renderer
        corner_renderer.AddActor(axes)

        # Add the corner renderer to the render window
        self.vtkWidget.GetRenderWindow().SetNumberOfLayers(2)
        self.vtkWidget.GetRenderWindow().AddRenderer(corner_renderer)

    def record_transformation(self):
        if not self.actor:
            return
        background_name = os.path.basename(self.background_image_path) if self.background_image_path else "N/A"
        alpha = self.alpha_slider.value()/100
        beta = self.beta_slider.value()/100
        gamma = self.gamma_slider.value()/100

        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(background_name))
        self.table.setItem(row_position, 1, QTableWidgetItem(str(gamma)))
        self.table.setItem(row_position, 2, QTableWidgetItem(str(beta)))
        self.table.setItem(row_position, 3, QTableWidgetItem(str(alpha)))

    def delete_transformation(self):
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            self.table.removeRow(selected_row)

    def export_to_csv(self):
        options = QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Background Image", "Rz", "Ry", "Rx"])

                for row in range(self.table.rowCount()):
                    row_data = []
                    for column in range(self.table.columnCount()):
                        item = self.table.item(row, column)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

    def start_app(self):
        expiration_date = datetime.date(2030, 12, 31)
        # Get today's date
        today = datetime.date.today()
        # Calculate the number of days left
        days_left = (expiration_date - today).days

        if days_left < 0:
            # The license has expired
            self.show_message_box(
                "License Expired",
                "The license for this app has expired. Please contact the developer.",
                QMessageBox.Critical
            )
            sys.exit(0)
        elif days_left <= 10:
            # The license will expire soon
            self.show_message_box(
                "License Expiring Soon",
                f"The license for this app will expire in {days_left} days. Please contact the developer.",
                QMessageBox.Warning
            )

    def show_message_box(self, title, message, icon):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Set custom stylesheet
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #333;
                color: white;
                font-size: 14px;
            }
            QLabel {
                color: white;
                font-size: 14px;
                qproperty-alignment: AlignCenter;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                padding: 8px;
                font-size: 14px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #777;
            }
            QPushButton:pressed {
                background-color: #999;
            }
        """)

        # Adjust the size and move the message box to center
        msg_box.adjustSize()
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - msg_box.width()) // 2
        y = (screen_geometry.height() - msg_box.height()) // 2
        msg_box.move(x, y)

        msg_box.exec_()
def apply_dark_theme(app):
    # Set Fusion style
    app.setStyle(QStyleFactory.create("macos"))

    # Create a dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(dark_palette)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = STLViewer()
    window.show()
    sys.exit(app.exec_())
