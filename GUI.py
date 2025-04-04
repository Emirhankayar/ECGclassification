"""

1. TRAVERSE Data/Dataset/test/y_0/PATIENTID.h5
2. GET PATIENT IDS FROM FILE NAMES
3. LIST ALL THE PATIENT IDS ON THE UI (scrollable widget)
4. ADD A LOAD BUTTON TO LOAD A PATIENT'S CONTENT
5. SET SOME SORT OF TITLE ON THE UI FOR PATIENT ID
6. DISPLAY THE LOADED DATA ON THE GRAPH
7. LOAD THE MODEL FROM THE DIR Results/BEST_RESNET_00.h5
8. USE THE PREDICT BUTTON AND DISPLAY THE CLASSIFICATION MADE FOR THE PATIENT
9. MAP THE LABELS INTO CORRESPONDING ONES

"""

import sys
import h5py
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QListWidget,
    QHBoxLayout,
    QMessageBox,
)

tf.config.set_visible_devices([], "GPU")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG-GUI")
        self.resize(900, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.patient_list = QListWidget()
        self.layout.addWidget(self.patient_list, 1)
        self.load_patient_ids()

        self.right_panel = QVBoxLayout()
        self.layout.addLayout(self.right_panel, 3)

        self.plot_graph = pg.PlotWidget()
        self.right_panel.addWidget(self.plot_graph)

        self.load_button = QPushButton("Load Patient Data")
        self.load_button.clicked.connect(self.load_patient_data)
        self.right_panel.addWidget(self.load_button)

        self.evaluate_button = QPushButton("Evaluate Patient Data")
        self.evaluate_button.clicked.connect(self.evaluate_patient_data)
        self.right_panel.addWidget(self.evaluate_button)

        self.prediction_label = QLabel("Prediction: ")
        self.true_label = QLabel("True Label: ")
        self.right_panel.addWidget(self.prediction_label)
        self.right_panel.addWidget(self.true_label)

        self.X_test = None
        self.true_label_value = None
        self.model = None
        self.selected_patient = None

        self.load_model()

    def load_model(self):
        try:
            print("\n [   ] Loading model...")
            self.model = tf.keras.models.load_model("Results/BEST_RESNET_03.h5")
            print("\n [ X ] Model loaded successfully!")
        except Exception as e:
            print(f"\n [ ! ] Error loading model: {e}")

    def load_patient_ids(self):
        DATA_PATH = Path("Data/Dataset/test")
        if not DATA_PATH.exists():
            print("\n [ ! ] Dataset directory not found!")
            return

        patient_files = []
        for y_dir in ["y_0", "y_1", "y_2", "y_3"]:
            y_path = DATA_PATH / y_dir
            if y_path.exists():
                patient_files.extend(list(y_path.glob("*.h5")))

        # Sort the patient files alphabetically by the filename
        patient_files.sort(key=lambda f: f.stem)  # Sorting by the stem (patient ID)

        self.patient_list.clear()
        for file in patient_files:
            self.patient_list.addItem(file.stem)

    def load_patient_data(self):
        selected_item = self.patient_list.currentItem()
        if not selected_item:
            QMessageBox.warning(
                self, "\n [ Warning! ]", "\n [ ! ] Please select a patient ID first."
            )
            return

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
                f"\n [ ! ] Patient file {patient_file} not found in any directory!",
            )
            return

        with h5py.File(file_path, "r") as h5_file:
            dataset = next(iter(h5_file.values()))
            self.X_test = dataset[:]

            self.true_label_value = patient_label

        self.plot_graph.clear()
        x_values = np.arange(self.X_test.shape[0])
        y_values = self.X_test[:, 0]
        self.plot_graph.plot(x_values, y_values, pen="b", name="ECG Signal")

        self.true_label.setText(f"True Label: {self.true_label_value}")

    def evaluate_patient_data(self):
        if self.X_test is None:
            QMessageBox.warning(self, "Warning", "No patient data loaded!")
            return

        if self.model is None:
            QMessageBox.critical(self, "Error", "Model not loaded!")
            return

        try:
            patient_data = np.expand_dims(self.X_test, axis=0)
            prediction = self.model.predict(patient_data)
            predicted_class = np.argmax(prediction)

            self.prediction_label.setText(f"Prediction: {predicted_class}")
            print(f"Prediction: {predicted_class}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
