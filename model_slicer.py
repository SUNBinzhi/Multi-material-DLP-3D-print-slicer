import os
import cv2
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication
from plotter_setup import setup_plotter  # Import the setup_plotter function

def slice_model(app):
    """Slices all loaded models based on the specified layer thickness."""
    if not app.models:
        QMessageBox.warning(app, "Warning", "Please load models first!")
        return

    try:
        layer_thickness = float(app.layer_thickness_input.text())
        if layer_thickness <= 0:
            raise ValueError("Layer thickness must be positive.")
    except ValueError as e:
        QMessageBox.warning(
            app, "Invalid Input", f"Please enter a valid positive number for layer thickness. Error: {e}"
        )
        return

    # 显示进度提示框
    QMessageBox.information(app, "Info", "Starting slicing process. Please wait...")

    # Configure progress bar
    app.progress_bar.setMaximum(len(app.models))
    app.progress_bar.setValue(0)
    app.progress_bar.setVisible(True)

    # Re-setup plotter to maintain fixed elements
    setup_plotter(app.plotter)

    app.slices = {file_name: [] for file_name in app.model_filenames}  # Store slices for each model
    all_slices = pv.MultiBlock()  # MultiBlock to store all slices

    # Slice each model
    for model, file_name in zip(app.models, app.model_filenames):
        z_min, z_max = model.bounds[4], model.bounds[5]
        z_values = np.arange(z_min, z_max, layer_thickness)

        # Slice and collect each z value with gradient color
        norm = plt.Normalize(z_min, z_max)
        cmap = plt.get_cmap('viridis')

        for z in z_values:
            slice_ = model.slice(normal=[0, 0, 1], origin=[0, 0, z])
            if slice_.n_points > 0:
                app.slices[file_name].append(slice_)  # Store the slice

                # Calculate the color based on the z position
                color = cmap(norm(z))[:3]  # Get RGB color from colormap
                color = [int(c * 255) for c in color]  # Convert to 0-255 scale

                # Convert color to float range 0-1 for PyVista
                float_color = [c / 255.0 for c in color]

                # Assign color to slice
                colors = np.array([float_color] * slice_.n_points)
                slice_.point_data['colors'] = colors

                # Append the colored slice to the MultiBlock
                all_slices.append(slice_)

        app.progress_bar.setValue(app.progress_bar.value() + 1)
        QApplication.processEvents()  # Update UI

    # Add all slices to the plotter
    app.plotter.add_mesh(all_slices, scalars='colors', opacity=0.7, rgb=True)

    app.progress_bar.setVisible(False)
    app.plotter.reset_camera()
    app.plotter.show()

    # 完成提示
    QMessageBox.information(app, "Info", "Slicing process completed successfully!")

def save_slices_as_images(app):
    """Save slices as 4K images using OpenCV, allowing for separate or combined saving."""
    if not app.slices:
        QMessageBox.warning(app, "Warning", "Please slice the models first!")
        return

    # Ask user if they want to save slices separately or combined
    choice = QMessageBox.question(
        app, "Save Option",
        "Would you like to save the slices for each model separately or combined?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if choice == QMessageBox.StandardButton.Yes:
        save_slices_separately(app)
    else:
        save_slices_combined(app)

def save_slices_separately(app):
    """Save slices for each model in separate folders."""
    directory = QFileDialog.getExistingDirectory(app, "Select Base Directory to Save Images")
    if not directory:
        return

    # Calculate global bounds across all models
    global_x_min = min(model.bounds[0] for model in app.models)
    global_x_max = max(model.bounds[1] for model in app.models)
    global_y_min = min(model.bounds[2] for model in app.models)
    global_y_max = max(model.bounds[3] for model in app.models)
    
    # Set z_min to 0 and find global z_max across all models
    z_min = 0
    global_z_max = max(model.bounds[5] for model in app.models)

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
        user_scale = float(app.scale_input.text())
        if user_scale <= 0:
            raise ValueError("Scale must be positive.")
        global_scale *= user_scale
    except ValueError as e:
        QMessageBox.warning(
            app, "Invalid Input", f"Please enter a valid positive number for image scale. Error: {e}"
        )
        return

    # Calculate the global offset to center all models
    global_x_offset = (image_width - global_x_range * global_scale) / 2
    global_y_offset = (image_height - global_y_range * global_scale) / 2

    # Configure progress bar for saving
    num_slices = int(np.ceil(global_z_max / float(app.layer_thickness_input.text())))
    app.progress_bar.setMaximum(len(app.models) * num_slices)
    app.progress_bar.setValue(0)
    app.progress_bar.setVisible(True)

    # Save images for each model separately
    for model_index, (model, file_name) in enumerate(zip(app.models, app.model_filenames)):
        model_name = os.path.splitext(os.path.basename(file_name))[0]
        model_dir = os.path.join(directory, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Use unified z-values across all models
        layer_thickness = float(app.layer_thickness_input.text())
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
                    cv2.fillPoly(image, contours, color=int(app.grayscale_values[model_index]))

            # Save image for the current slice index
            slice_image_path = os.path.join(model_dir, f"slice_{i:03d}.png")
            cv2.imwrite(slice_image_path, image)

            app.progress_bar.setValue(app.progress_bar.value() + 1)
            QApplication.processEvents()  # Update UI

    app.progress_bar.setVisible(False)
    QMessageBox.information(app, "Info", f"Slices saved in separate folders under {directory}")

def save_slices_combined(app):
    """Save slices by overlaying slices from all models."""
    directory = QFileDialog.getExistingDirectory(app, "Select Directory to Save Images")
    if not directory:
        return

    # Calculate global bounds across all models
    global_x_min = min(model.bounds[0] for model in app.models)
    global_x_max = max(model.bounds[1] for model in app.models)
    global_y_min = min(model.bounds[2] for model in app.models)
    global_y_max = max(model.bounds[3] for model in app.models)
    
    # Set z_min to 0 and find global z_max across all models
    z_min = 0
    global_z_max = max(model.bounds[5] for model in app.models)

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
        user_scale = float(app.scale_input.text())
        if user_scale <= 0:
            raise ValueError("Scale must be positive.")
        global_scale *= user_scale
    except ValueError as e:
        QMessageBox.warning(
            app, "Invalid Input", f"Please enter a valid positive number for image scale. Error: {e}"
        )
        return

    # Calculate the global offset to center all models
    global_x_offset = (image_width - global_x_range * global_scale) / 2
    global_y_offset = (image_height - global_y_range * global_scale) / 2

    # Use a unified range of z-values across all models
    layer_thickness = float(app.layer_thickness_input.text())
    z_values = np.arange(z_min, global_z_max, layer_thickness)
    num_slices = len(z_values)

    # Configure progress bar for saving
    app.progress_bar.setMaximum(num_slices)
    app.progress_bar.setValue(0)
    app.progress_bar.setVisible(True)

    # Save images for each slice index
    for i, z in enumerate(z_values):
        # Create a black canvas for the 4K image
        image = np.zeros((2160, 3840), dtype=np.uint8)

        for model_index, (model, file_name) in enumerate(zip(app.models, app.model_filenames)):
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
                        cv2.fillPoly(image, contours, color=int(app.grayscale_values[model_index]))

        # Save image for the current slice index
        slice_image_path = os.path.join(directory, f"slice_{i:03d}.png")
        cv2.imwrite(slice_image_path, image)

        app.progress_bar.setValue(app.progress_bar.value() + 1)
        QApplication.processEvents()  # Update UI

    app.progress_bar.setVisible(False)
    QMessageBox.information(app, "Info", f"Slices saved in {directory}")
