"""

[x]1. TRAVERSE Data/Dataset/test/y_0/PATIENTID.h5
[x]2. GET PATIENT IDS FROM FILE NAMES
[x]3. LIST ALL THE PATIENT IDS ON THE UI (scrollable widget)
[x]4. ADD A LOAD BUTTON TO LOAD A PATIENT'S CONTENT
[x]5. SET SOME SORT OF TITLE ON THE UI FOR PATIENT ID
[x]6. DISPLAY THE LOADED DATA ON THE GRAPH
[x]7. LOAD THE MODEL FROM THE DIR Results/BEST_RESNET_00.h5
[x]8. USE THE PREDICT BUTTON AND DISPLAY THE CLASSIFICATION MADE FOR THE PATIENT
[x]9. MAP THE LABELS INTO CORRESPONDING ONES
[x]10.LOAD DIFFERENT MODELS FROM A DIRECTORY
[ ]11.LOAD DIFFERENT WINDOW SIZED MODELS (DATA NEEDS TO BE LOADED ACCORDING TO THAT) PROBABLY WE NEED TO MAKE ADJUSTABLE BINNING ?
[ ]12.CREATE SOME SORT OF SAVE THE PATIENT RESULT TO CSV?

"""

import sys
import h5py
import sklearn
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from pathlib import Path
from itertools import chain
import Modules.constants as constants
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QWidget,
    QLabel,
    QListWidget,
    QHBoxLayout,
    QMessageBox,
)


tf.config.set_visible_devices([], "GPU")  # disables gpu


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG-GUI")
        self.resize(900, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.left_panel = QVBoxLayout()
        self.layout.addLayout(self.left_panel, 1)

        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search...")

        self.patient_list = QListWidget()
        self.patient_list.setAlternatingRowColors(True)

        self.search_bar.textChanged.connect(self.filter_list)

        self.left_panel.addWidget(self.search_bar)
        self.left_panel.addWidget(self.patient_list)
        self.load_patient_ids()
        self.patient_list.currentItemChanged.connect(self.load_patient_data)

        self.right_panel = QVBoxLayout()
        self.layout.addLayout(self.right_panel, 3)

        self.model_options = QComboBox()
        self.model_paths = {}

        self.dropdown_label = QLabel("Loaded Model: --")
        self.right_panel.addWidget(self.model_options)
        self.right_panel.addWidget(self.dropdown_label)

        self.model_dir = QLabel(f"Model Directory: --, --")
        self.right_panel.addWidget(self.model_dir)

        self.load_model_files(Path(constants.MODELS_DIR))

        self.model_options.currentTextChanged.connect(self.load_selected_model)

        self.plot_graph = pg.PlotWidget()
        self.right_panel.addWidget(self.plot_graph)

        self.evaluate_button = QPushButton("Evaluate Patient Data")
        self.evaluate_button.clicked.connect(self.evaluate_patient_data)
        self.right_panel.addWidget(self.evaluate_button)

        self.prediction_label = QLabel("Prediction: --, --")
        self.true_label = QLabel("True Label: --, --")
        self.right_panel.addWidget(self.prediction_label)
        self.right_panel.addWidget(self.true_label)

        self.X_test = None
        self.true_label_value = None
        self.model = None
        self.selected_patient = None

        self.load_selected_model()

    def load_model_files(self, folder_path: Path):
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"\n[ !! ] Directory not found: {folder_path}")
            return
        model_files = chain(folder_path.glob("*.keras"), folder_path.glob("*.h5"))
        for model_file in model_files:
            filename = model_file.stem
            self.model_options.addItem(filename)
            self.model_paths[filename] = model_file

    def load_selected_model(self):
        self.prediction_label.setText("Prediction: --, --")
        selected_file = self.model_options.currentText()
        full_path = self.model_paths.get(selected_file)

        if full_path:
            try:
                self.model = tf.keras.models.load_model(full_path)
                self.dropdown_label.setText(f"Loaded Model: {selected_file}")
                self.model_dir.setText(f"Model Directory: {full_path}")
                print(f"[ OK ] Model loaded from: {full_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading model: {e}")
        else:
            self.dropdown_label.setText("Loaded Model: --")
            QMessageBox.warning(self, "Warning", "Model file not found.")

    def load_patient_ids(self):
        DATA_PATH = constants.DATASET / "test"
        if not DATA_PATH.exists():
            print("\n [ !! ] Dataset directory not found!")
            return

        patient_files = []
        for y_dir in ["y_0", "y_1", "y_2", "y_3"]:
            y_path = DATA_PATH / y_dir
            if y_path.exists():
                patient_files.extend(list(y_path.glob("*.h5")))

        patient_files.sort(key=lambda f: f.stem)
        self.all_patients = [file.stem for file in patient_files]

        self.update_patient_list(self.all_patients)

    def load_patient_data(self):
        self.prediction_label.setStyleSheet("color:black, font-weight:regular")

        selected_item = self.patient_list.currentItem()

        self.selected_patient = selected_item.text()

        patient_file = self.selected_patient + ".h5"
        patient_label = None

        for label in range(4):
            data_dir = Path(f"Data/Dataset/test/y_{label}")
            file_path = data_dir / patient_file

            if file_path.exists():
                patient_label = label
                break

        if patient_label is None:
            QMessageBox.critical(
                self,
                "[ Error! ]",
                f"\n [ !! ] Patient file {patient_file} not found in any directory!",
            )
            return

        with h5py.File(file_path, "r") as h5_file:
            dataset = next(iter(h5_file.values()))
            self.X_test = dataset[:]

            self.true_label_value = patient_label
        """
            Scaling applied here just to demonstrate better visuals on the graph.
        """
        self.plot_graph.clear()
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_values = np.arange(self.X_test.shape[0]).reshape(-1, 1)
        x_values = scaler.fit_transform(x_values).flatten()

        y_values = self.X_test[:, 0].reshape(-1, 1)
        y_values = scaler.fit_transform(y_values).flatten()
        self.plot_graph.setYRange(-1, 1)

        self.plot_graph.plot(x_values, y_values, pen="g", width=1.5, name="ECG Signal")
        self.label_map = {0: "AFIB", 1: "GSVT", 2: "SB", 3: "SR"}
        f = self.label_map.get(self.true_label_value, "Unknown")
        self.true_label.setText(
            f"True Label: {f}, True Label Class: {self.true_label_value}"
        )
        self.prediction_label.setText("Prediction: --, --")

    def evaluate_patient_data(self):
        if self.X_test is None:
            QMessageBox.warning(self, "Warning", "No patient data loaded!")
            return

        if self.model is None:
            QMessageBox.critical(self, "Error", "Model not loaded!")
            return

        try:
            patient_data = np.expand_dims(self.X_test, axis=0)
            self.patient_data = patient_data
            prediction = self.model.predict(patient_data)
            predicted_class = np.argmax(prediction)
            self.predicted_class_value = predicted_class

            f = self.label_map.get(predicted_class, "Unknown")

            self.prediction_label.setText(
                f"Prediction: {f}, Prediction Class: {predicted_class}"
            )
            print(f"Prediction: {f}, Prediction Class: {predicted_class}")
            if predicted_class == self.true_label_value:
                self.prediction_label.setStyleSheet("color:green; font-weight:regular;")
            elif predicted_class != self.true_label_value:
                self.prediction_label.setStyleSheet("color:red; font-weight:regular;")
            else:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {e}")

    def update_patient_list(self, patient_ids):
        self.patient_list.clear()
        for pid in patient_ids:
            self.patient_list.addItem(pid)

    def filter_list(self):
        search = self.search_bar.text().lower()
        filtered = [pid for pid in self.all_patients if search in pid.lower()]
        self.update_patient_list(filtered)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
