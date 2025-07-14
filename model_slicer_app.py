import os
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar,
    QLineEdit, QLabel, QListWidget, QInputDialog, QMessageBox
)
from PyQt6.QtGui import QAction
from pyvistaqt import QtInteractor
from plotter_setup import setup_plotter
from model_loader import load_models
from model_slicer import slice_model, save_slices_as_images
import concurrent.futures

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

        # Initialize ThreadPoolExecutor for background tasks
        self.executor = concurrent.futures.ThreadPoolExecutor()

    def create_menu(self):
        """Creates the menu bar and adds a Load Model action."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        load_model_action = QAction("Load Models", self)  # 修改菜单名称
        load_model_action.triggered.connect(self.load_models)  # 更新为新的加载方法
        file_menu.addAction(load_model_action)

    def create_rendering_area(self, main_layout):
        """Creates the rendering area for displaying the 3D model."""
        render_layout = QVBoxLayout()
        self.plotter = QtInteractor(self.central_widget)
        render_layout.addWidget(self.plotter.interactor)
        main_layout.addLayout(render_layout, stretch=5)

        # Initialize plotter
        setup_plotter(self.plotter)

    def create_control_panel(self, main_layout):
        """Creates the control panel with buttons and inputs."""
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)

        # Model names list widget
        self.model_names_list = QListWidget()
        self.model_names_list.itemClicked.connect(self.on_model_name_clicked)
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
        self.models, self.model_filenames, self.grayscale_values = load_models(self)
        self.refresh_plotter()

    def refresh_plotter(self):
        """Refreshes the plotter to display all models with the fixed elements."""
        setup_plotter(self.plotter)
        for model in self.models:
            self.plotter.add_mesh(model, color='lightblue', opacity=0.5)
        self.plotter.reset_camera()
        self.plotter.show()

    def on_model_name_clicked(self, item):
        """Handle clicking on a model name to highlight the model and then set grayscale value."""
        selected_index = self.model_names_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.models):
            return

        # Highlight the selected model first
        self.highlight_model(selected_index)

        # Then set grayscale value for the selected model
        self.set_grayscale_value(selected_index)

    def highlight_model(self, selected_index):
        """Highlight the selected model in the plotter."""
        # Re-setup plotter to maintain fixed elements
        setup_plotter(self.plotter)

        # Display all models with highlight on the selected one
        for index, model in enumerate(self.models):
            if index == selected_index:
                self.plotter.add_mesh(model, color='yellow', opacity=0.8)  # Highlight selected model
            else:
                self.plotter.add_mesh(model, color='lightblue', opacity=0.5)

        self.plotter.reset_camera()
        self.plotter.show()

    def set_grayscale_value(self, index):
        """Set the grayscale value for the model at the given index."""
        if index < 0 or index >= len(self.models):
            return

        # Prompt for grayscale value
        value, ok = QInputDialog.getInt(
            self, "Set Grayscale Value",
            f"Set grayscale value for {self.model_filenames[index]} (0-255):",
            min=0, max=255, value=self.grayscale_values[index]
        )

        if ok:
            self.grayscale_values[index] = value
            QMessageBox.information(self, "Grayscale Value Set", f"Grayscale value for {self.model_filenames[index]} set to {value}.")

    def slice_model(self):
        """Slices all loaded models based on the specified layer thickness."""
        # Run slicing in a separate thread
        self.executor.submit(slice_model, self)

    def save_slices_as_images(self):
        """Save slices as 4K images using OpenCV, allowing for separate or combined saving."""
        save_slices_as_images(self)
