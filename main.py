import sys
import os
import numpy as np
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QProgressBar,
    QLineEdit, QLabel, QMessageBox, QListWidget, QInputDialog
)
from PyQt6.QtGui import QAction
from pyvistaqt import QtInteractor
import pyvista as pv
import matplotlib.pyplot as plt
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, QLabel, QPushButton

class GrayscaleDialog(QDialog):
    def __init__(self, filename, initial_value, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(f"Set Grey Scale for '{os.path.basename(filename)}'")

        self.value = initial_value

        # Set the initial size of the dialog window
        self.resize(400, 200)  # Set the desired width and height

        layout = QVBoxLayout(self)
        
        # Add slider
        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(initial_value)
        self.slider.valueChanged.connect(self.update_line_edit)
        layout.addWidget(self.slider)

        # Add line edit
        self.line_edit = QLineEdit(str(initial_value))
        self.line_edit.setValidator(QtGui.QIntValidator(0, 255))
        self.line_edit.textChanged.connect(self.update_slider)
        layout.addWidget(self.line_edit)

        # Add buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def update_slider(self, text):
        if text.isdigit():
            self.slider.setValue(int(text))

    def update_line_edit(self, value):
        self.line_edit.setText(str(value))

    def get_value(self):
        return self.slider.value()

class SlicingWorker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, models, model_filenames, layer_thickness, grayscale_values):
        super().__init__()
        self.models = models
        self.model_filenames = model_filenames
        self.layer_thickness = layer_thickness
        self.grayscale_values = grayscale_values

    def run(self):
        try:
            slices = {file_name: [] for file_name in self.model_filenames}  # Store slices for each model

            # Slice each model
            for model_index, (model, file_name) in enumerate(zip(self.models, self.model_filenames)):
                z_min, z_max = model.bounds[4], model.bounds[5]
                z_values = np.arange(z_min, z_max, self.layer_thickness)

                for z in z_values:
                    slice_ = model.slice(normal=[0, 0, 1], origin=[0, 0, z])
                    if slice_.n_points > 0:
                        slices[file_name].append(slice_)

                # Emit progress update signal
                self.update_progress.emit(model_index + 1)

            # Emit finished signal with slices result
            self.finished.emit(slices)
        except Exception as e:
            self.error.emit(str(e))


class ModelSlicerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.setWindowTitle("Model Slicer")
        self.setGeometry(100, 100, 1000, 600)

        # Create the menu bar
        self.create_menu()

        # Main widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        # Left layout for rendering window
        self.create_rendering_area(main_layout)

        # Right layout for controls
        self.create_control_panel(main_layout)

        # Initialize model variable
        self.models = []  # 修改为列表以存储多个模型
        self.model_filenames = []
        self.grayscale_values = []  # 存储每个模型的灰度值

    def create_menu(self):
        """Creates the menu bar and adds Load Models and Reset actions."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        load_model_action = QAction("Load Models", self)
        load_model_action.triggered.connect(self.load_models)
        file_menu.addAction(load_model_action)

        # Add Reset action
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_application)
        file_menu.addAction(reset_action)
        
    def reset_application(self):
        """Resets the application to its initial state."""
        # Clear models and slices
        self.models.clear()
        self.model_filenames.clear()
        self.grayscale_values.clear()
        self.slices = {}

        # Clear the plotter and re-add fixed elements
        self.plotter.clear()
        self.setup_fixed_elements()

        # Clear the model names list widget
        self.model_names_list.clear()

        # Hide progress bar
        self.progress_bar.setVisible(False)

        # Reset camera
        self.plotter.reset_camera()

        QMessageBox.information(self, "Info", "Application has been reset.")


    def create_rendering_area(self, main_layout):
        """Creates the rendering area for displaying the 3D model."""
        render_layout = QVBoxLayout()
        self.plotter = QtInteractor(self.central_widget)
        render_layout.addWidget(self.plotter.interactor)
        main_layout.addLayout(render_layout, stretch=5)

        # Initialize plotter
        self.setup_plotter()

    def setup_plotter(self):
        """Sets up the initial plotter configuration, including background, axes, and grid."""
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.set_background(color='grey')

        # Add grid floor similar to Blender
        x = np.arange(-192, 193, 1)  # Ensure inclusive of boundaries
        y = np.arange(-108, 109, 1)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        grid = pv.StructuredGrid(x, y, z)
        self.plotter.add_mesh(grid, color='black', style='wireframe', line_width=0.5)

        # Create and add x-axis line
        x_axis = pv.Line([-192, 0, 0], [192, 0, 0])
        self.plotter.add_mesh(x_axis, color='red', line_width=5)

        # Create and add y-axis line
        y_axis = pv.Line([0, -108, 0], [0, 108, 0])
        self.plotter.add_mesh(y_axis, color='green', line_width=5)

    def create_control_panel(self, main_layout):
        """Creates the control panel with buttons and inputs."""
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # Model names list widget
        self.model_names_list = QListWidget()
        self.model_names_list.itemClicked.connect(self.on_model_name_clicked)
        self.model_names_list.itemDoubleClicked.connect(self.on_model_name_double_clicked)
        control_layout.addWidget(self.model_names_list)

        # Horizontal layout for label and input
        thickness_layout = QHBoxLayout()
        
        # Layer thickness input and label
        self.layer_thickness_label = QLabel("Layer Thickness:")
        thickness_layout.addWidget(self.layer_thickness_label)

        self.layer_thickness_input = QLineEdit("0.2")
        thickness_layout.addWidget(self.layer_thickness_input)

        control_layout.addLayout(thickness_layout)

        # Scale input and label
        scale_layout = QHBoxLayout()
        self.scale_label = QLabel("Image Scale:")
        scale_layout.addWidget(self.scale_label)

        self.scale_input = QLineEdit("1.0")  # Default scale is 1.0
        scale_layout.addWidget(self.scale_input)

        control_layout.addLayout(scale_layout)

        # "Slice Model" button
        self.slice_model_button = QPushButton("Slice Model")
        self.slice_model_button.clicked.connect(self.slice_model)
        control_layout.addWidget(self.slice_model_button)

        # "Save Slices" button
        self.save_slices_button = QPushButton("Save Slices as Images")
        self.save_slices_button.clicked.connect(self.save_slices_as_images)
        control_layout.addWidget(self.save_slices_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        main_layout.addLayout(control_layout, stretch=1)

    def load_models(self):
        """Loads multiple 3D model files and displays them in the rendering window."""
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Open Model Files", "", "STL Files (*.stl);;All Files (*)"
        )

        if file_names:
            self.models.clear()  # 清空现有模型列表
            self.model_filenames.clear()  # 清空现有文件名列表
            self.grayscale_values.clear()  # 清空现有灰度值列表

            for file_name in file_names:
                model = pv.read(file_name)
                self.models.append(model)
                self.model_filenames.append(file_name)
                self.grayscale_values.append(255)  # 默认灰度值为255

            # Display all models and reset the plotter
            self.refresh_plotter()

            # Update model names list widget
            self.model_names_list.clear()
            self.model_names_list.addItems([os.path.basename(name) for name in self.model_filenames])

            self.plotter.reset_camera()
            self.plotter.show()

    def on_model_name_clicked(self, item):
        """Handle clicking on a model name to highlight the model."""
        selected_index = self.model_names_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.models):
            return

        # Highlight the selected model
        self.highlight_model(selected_index)

    def on_model_name_double_clicked(self, item):
        """Handle double-clicking on a model name to set grayscale value."""
        selected_index = self.model_names_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.models):
            return

        # Set grayscale value for the selected model
        self.set_grayscale_value(selected_index)

    def highlight_model(self, selected_index):
        """Highlight the selected model in the plotter."""
        # Save current camera position
        camera_position = self.plotter.camera_position

        # Re-setup plotter to maintain fixed elements without resetting the camera
        self.setup_plotter()

        # Display all models with highlight on the selected one
        for index, model in enumerate(self.models):
            if index == selected_index:
                self.plotter.add_mesh(model, color='yellow', opacity=0.8)  # Highlight selected model
            else:
                self.plotter.add_mesh(model, color='lightblue', opacity=0.5)

        # Restore previous camera position
        self.plotter.camera_position = camera_position

        # Update the plotter
        self.plotter.show()



    def set_grayscale_value(self, index):
        """Set the grayscale value for the model at the given index."""
        if index < 0 or index >= len(self.models):
            return

        # Create and show grayscale dialog
        dialog = GrayscaleDialog(self.model_filenames[index], self.grayscale_values[index], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.grayscale_values[index] = dialog.get_value()
            QMessageBox.information(self, "Grayscale Value Set", 
                                    f"Grayscale value for {os.path.basename(self.model_filenames[index])} set to {self.grayscale_values[index]}.")


    def slice_model(self):
        """Slices all loaded models based on the specified layer thickness."""
        if not self.models:
            QMessageBox.warning(self, "Warning", "Please load models first!")
            return

        try:
            layer_thickness = float(self.layer_thickness_input.text())
            if layer_thickness <= 0:
                raise ValueError("Layer thickness must be positive.")
        except ValueError as e:
            QMessageBox.warning(
                self, "Invalid Input", f"Please enter a valid positive number for layer thickness. Error: {e}"
            )
            return

        # Configure progress bar
        self.progress_bar.setMaximum(len(self.models))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Save current camera position
        self.camera_position = self.plotter.camera_position

        # Create and start the slicing worker thread
        self.slicing_worker = SlicingWorker(self.models, self.model_filenames, layer_thickness, self.grayscale_values)
        self.slicing_worker.update_progress.connect(self.progress_bar.setValue)
        self.slicing_worker.finished.connect(self.on_slicing_finished)
        self.slicing_worker.error.connect(self.on_slicing_error)
        self.slicing_worker.start()

    def setup_fixed_elements(self):
        """Sets up the fixed elements like grid, axes, and background."""
        # Add grid floor similar to Blender
        x = np.arange(-192, 193, 1)  # Ensure inclusive of boundaries
        y = np.arange(-108, 109, 1)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        grid = pv.StructuredGrid(x, y, z)
        self.plotter.add_mesh(grid, color='black', style='wireframe', line_width=0.5)

        # Create and add x-axis line
        x_axis = pv.Line([-192, 0, 0], [192, 0, 0])
        self.plotter.add_mesh(x_axis, color='red', line_width=5)

        # Create and add y-axis line
        y_axis = pv.Line([0, -108, 0], [0, 108, 0])
        self.plotter.add_mesh(y_axis, color='green', line_width=5)

    def on_slicing_finished(self, slices):
        """Handle the completion of slicing."""
        self.slices = slices  # Store the slices in the main window for further processing or saving

        # Determine the global z_min and z_max across all slices for color normalization
        z_min = min(model.bounds[4] for model in self.models)
        z_max = max(model.bounds[5] for model in self.models)

        # Create a colormap for the gradient
        norm = plt.Normalize(z_min, z_max)
        cmap = plt.get_cmap('viridis')  # Use a colormap for the gradient

        # Add all slices to the plotter for visualization with gradient color
        all_slices = pv.MultiBlock()  # MultiBlock to store all slices for visualization
        for file_name, slice_list in slices.items():
            for slice_ in slice_list:
                # Calculate the average z value of the slice to determine the color
                avg_z = np.mean(slice_.points[:, 2])

                # Get the RGB color from the colormap based on the average z value
                color = cmap(norm(avg_z))[:3]  # Get RGB components
                float_color = [c for c in color]  # Directly use colormap RGB in 0-1 range

                # Apply color as scalars to the slice
                slice_.point_data['color'] = np.tile(float_color, (slice_.n_points, 1))

                # Append the colored slice to the MultiBlock
                all_slices.append(slice_)

        self.plotter.clear()  # Clear existing plot before adding new elements and slices

        # Re-add fixed elements like grid, axes, and background
        self.setup_fixed_elements()

        # Plot slices with color data
        self.plotter.add_mesh(all_slices, scalars='color', rgb=True, opacity=0.7)

        # Restore previous camera position
        self.plotter.camera_position = self.camera_position

        QMessageBox.information(self, "Info", "Slicing completed successfully.")
        self.progress_bar.setVisible(False)
        self.plotter.show()



    def on_slicing_error(self, message):
        """Handle errors during slicing."""
        QMessageBox.critical(self, "Error", f"An error occurred during slicing: {message}")
        self.progress_bar.setVisible(False)

    def refresh_plotter(self):
        """Refreshes the plotter to display all models with the fixed elements."""
        self.setup_plotter()
        for model in self.models:
            self.plotter.add_mesh(model, color='lightblue', opacity=0.5)
        self.plotter.reset_camera()
        self.plotter.show()

    def save_slices_as_images(self):
        """Save slices as 4K images using OpenCV, allowing for separate or combined saving."""
        if not self.slices:
            QMessageBox.warning(self, "Warning", "Please slice the models first!")
            return

        # Ask user if they want to save slices separately or combined
        choice = QMessageBox.question(
            self, "Save Option",
            "Would you like to save the slices for each model separately or combined?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if choice == QMessageBox.StandardButton.Yes:
            self.save_slices_separately()
        else:
            self.save_slices_combined()

    def save_slices_separately(self):
        """Save slices for each model in separate folders."""
        directory = QFileDialog.getExistingDirectory(self, "Select Base Directory to Save Images")
        if not directory:
            return

        # Calculate global bounds across all models
        global_x_min = min(model.bounds[0] for model in self.models)
        global_x_max = max(model.bounds[1] for model in self.models)
        global_y_min = min(model.bounds[2] for model in self.models)
        global_y_max = max(model.bounds[3] for model in self.models)
        
        # Set z_min to 0 and find global z_max across all models
        z_min = 0
        global_z_max = max(model.bounds[5] for model in self.models)

        global_x_range = global_x_max - global_x_min
        global_y_range = global_y_max - global_y_min

        # Calculate global scale factors to maintain aspect ratio
        image_width, image_height = 3840, 2160
        global_x_scale = image_width / global_x_range
        global_y_scale = image_height / global_y_range

        # Use the smaller scale to maintain aspect ratio
        global_scale = min(global_x_scale, global_y_scale)

        try:
            # Get user-defined scale from input and apply it
            user_scale = float(self.scale_input.text())
            if user_scale <= 0:
                raise ValueError("Scale must be positive.")
            global_scale *= user_scale
        except ValueError as e:
            QMessageBox.warning(
                self, "Invalid Input", f"Please enter a valid positive number for image scale. Error: {e}"
            )
            return

        # Calculate the global offset to center all models
        global_x_offset = (image_width - global_x_range * global_scale) / 2
        global_y_offset = (image_height - global_y_range * global_scale) / 2

        # Configure progress bar for saving
        num_slices = int(np.ceil(global_z_max / float(self.layer_thickness_input.text())))
        self.progress_bar.setMaximum(len(self.models) * num_slices)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Save images for each model separately
        for model_index, (model, file_name) in enumerate(zip(self.models, self.model_filenames)):
            model_name = os.path.splitext(os.path.basename(file_name))[0]
            model_dir = os.path.join(directory, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Use unified z-values across all models
            layer_thickness = float(self.layer_thickness_input.text())
            z_values = np.arange(z_min, global_z_max, layer_thickness)

            # Slice each model across the global z range
            for i, z in enumerate(z_values):
                # Create a black canvas for the 4K image
                image = np.zeros((2160, 3840), dtype=np.uint8)

                # Slice the model at the current z-value
                slice_ = model.slice(normal=[0, 0, 1], origin=[0, 0, z])

                # Check if the slice has lines to draw
                if slice_.n_points > 0:
                    lines = slice_.extract_geometry().lines
                    points = slice_.points

                    if lines.size > 0:
                        line_indices = lines.reshape(-1, 3)[:, 1:]

                        # Dictionary to track which points connect to which others
                        connections = {i: set() for i in range(len(points))}

                        for idx_pair in line_indices:
                            connections[idx_pair[0]].add(idx_pair[1])
                            connections[idx_pair[1]].add(idx_pair[0])

                        # Find all contours by following connections
                        contours = []
                        visited = set()

                        def follow_contour(start_point):
                            contour = []
                            stack = [start_point]
                            while stack:
                                point = stack.pop()
                                if point not in visited:
                                    visited.add(point)
                                    contour.append(point)
                                    stack.extend(connections[point] - visited)
                            return contour

                        for idx, conn in connections.items():
                            if idx not in visited and len(conn) > 0:
                                contour = follow_contour(idx)
                                if len(contour) > 2:
                                    # Convert contour points to image coordinates
                                    contour_points = [
                                        (
                                            int((points[pt][0] - global_x_min) * global_scale + global_x_offset),
                                            int((global_y_max - points[pt][1]) * global_scale + global_y_offset)
                                        ) for pt in contour
                                    ]
                                    contours.append(np.array(contour_points, dtype=np.int32))

                        # Fill each contour with the grayscale value for the current model
                        cv2.fillPoly(image, contours, color=int(self.grayscale_values[model_index]))

                # Save image for the current slice index
                slice_image_path = os.path.join(model_dir, f"slice_{i:03d}.png")
                cv2.imwrite(slice_image_path, image)

                self.progress_bar.setValue(self.progress_bar.value() + 1)
                QtWidgets.QApplication.processEvents()  # Update UI

        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Info", f"Slices saved in separate folders under {directory}")

    def save_slices_combined(self):
        """Save slices by overlaying slices from all models."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")
        if not directory:
            return

        # Calculate global bounds across all models
        global_x_min = min(model.bounds[0] for model in self.models)
        global_x_max = max(model.bounds[1] for model in self.models)
        global_y_min = min(model.bounds[2] for model in self.models)
        global_y_max = max(model.bounds[3] for model in self.models)
        
        # Set z_min to 0 and find global z_max across all models
        z_min = 0
        global_z_max = max(model.bounds[5] for model in self.models)

        global_x_range = global_x_max - global_x_min
        global_y_range = global_y_max - global_y_min

        # Calculate global scale factors to maintain aspect ratio
        image_width, image_height = 3840, 2160
        global_x_scale = image_width / global_x_range
        global_y_scale = image_height / global_y_range

        # Use the smaller scale to maintain aspect ratio
        global_scale = min(global_x_scale, global_y_scale)

        try:
            # Get user-defined scale from input and apply it
            user_scale = float(self.scale_input.text())
            if user_scale <= 0:
                raise ValueError("Scale must be positive.")
            global_scale *= user_scale
        except ValueError as e:
            QMessageBox.warning(
                self, "Invalid Input", f"Please enter a valid positive number for image scale. Error: {e}"
            )
            return

        # Calculate the global offset to center all models
        global_x_offset = (image_width - global_x_range * global_scale) / 2
        global_y_offset = (image_height - global_y_range * global_scale) / 2

        # Use a unified range of z-values across all models
        layer_thickness = float(self.layer_thickness_input.text())
        z_values = np.arange(z_min, global_z_max, layer_thickness)
        num_slices = len(z_values)

        # Configure progress bar for saving
        self.progress_bar.setMaximum(num_slices)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Save images for each slice index
        for i, z in enumerate(z_values):
            # Create a black canvas for the 4K image
            image = np.zeros((2160, 3840), dtype=np.uint8)

            for model_index, (model, file_name) in enumerate(zip(self.models, self.model_filenames)):
                # Slice the model at the current z-value
                slice_ = model.slice(normal=[0, 0, 1], origin=[0, 0, z])

                # Check if the slice has lines to draw
                if slice_.n_points > 0:
                    lines = slice_.extract_geometry().lines
                    points = slice_.points

                    if lines.size > 0:
                        line_indices = lines.reshape(-1, 3)[:, 1:]

                        # Dictionary to track which points connect to which others
                        connections = {i: set() for i in range(len(points))}

                        for idx_pair in line_indices:
                            connections[idx_pair[0]].add(idx_pair[1])
                            connections[idx_pair[1]].add(idx_pair[0])

                        # Find all contours by following connections
                        contours = []
                        visited = set()

                        def follow_contour(start_point):
                            contour = []
                            stack = [start_point]
                            while stack:
                                point = stack.pop()
                                if point not in visited:
                                    visited.add(point)
                                    contour.append(point)
                                    stack.extend(connections[point] - visited)
                            return contour

                        for idx, conn in connections.items():
                            if idx not in visited and len(conn) > 0:
                                contour = follow_contour(idx)
                                if len(contour) > 2:
                                    # Convert contour points to image coordinates
                                    contour_points = [
                                        (
                                            int((points[pt][0] - global_x_min) * global_scale + global_x_offset),
                                            int((global_y_max - points[pt][1]) * global_scale + global_y_offset)
                                        ) for pt in contour
                                    ]
                                    contours.append(np.array(contour_points, dtype=np.int32))

                        # Fill each contour with the grayscale value for the current model
                        cv2.fillPoly(image, contours, color=int(self.grayscale_values[model_index]))

            # Save image for the current slice index
            slice_image_path = os.path.join(directory, f"slice_{i:03d}.png")
            cv2.imwrite(slice_image_path, image)

            self.progress_bar.setValue(self.progress_bar.value() + 1)
            QtWidgets.QApplication.processEvents()  # Update UI

        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Info", f"Slices saved in {directory}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ModelSlicerApp()
    window.show()
    sys.exit(app.exec())
