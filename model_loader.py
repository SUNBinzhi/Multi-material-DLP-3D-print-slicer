import os
from tkinter import messagebox
import pyvista as pv
from PyQt6.QtWidgets import QFileDialog
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_models(self):
    """Loads multiple 3D model files and displays them in the rendering window."""
    try:
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Open Model Files", "", "STL Files (*.stl);;All Files (*)"
        )

        models = []
        model_filenames = []
        grayscale_values = []

        if file_names:
            models.clear()  # 清空现有模型列表
            model_filenames.clear()  # 清空现有文件名列表
            grayscale_values.clear()  # 清空现有灰度值列表

            for file_name in file_names:
                model = pv.read(file_name)
                models.append(model)
                model_filenames.append(file_name)
                grayscale_values.append(255)  # 默认灰度值为255

            # Update model names list widget
            self.model_names_list.clear()
            self.model_names_list.addItems([os.path.basename(name) for name in model_filenames])

            self.plotter.reset_camera()
            self.plotter.show()

        return models, model_filenames, grayscale_values

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        messagebox.critical(self, "Error", f"Failed to load models: {str(e)}")
        return [], [], []

# def load_models(self):
#     """Loads multiple 3D model files and displays them in the rendering window."""
#     file_names, _ = QFileDialog.getOpenFileNames(
#         self, "Open Model Files", "", "STL Files (*.stl);;All Files (*)"
#     )

#     models = []
#     model_filenames = []
#     grayscale_values = []

#     if file_names:
#         models.clear()  # 清空现有模型列表
#         model_filenames.clear()  # 清空现有文件名列表
#         grayscale_values.clear()  # 清空现有灰度值列表

#         for file_name in file_names:
#             model = pv.read(file_name)
#             models.append(model)
#             model_filenames.append(file_name)
#             grayscale_values.append(255)  # 默认灰度值为255

#         # Update model names list widget
#         self.model_names_list.clear()
#         self.model_names_list.addItems([os.path.basename(name) for name in model_filenames])

#         self.plotter.reset_camera()
#         self.plotter.show()

#     return models, model_filenames, grayscale_values
