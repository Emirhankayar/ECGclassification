"""

[x]1. TRAVERSE Data/Dataset/test/0/PATIENTID.h5
[x]2. GET PATIENT IDS FROM FILE NAMES
[x]3. LIST ALL THE PATIENT IDS ON THE UI (scrollable widget)
[x]4. ADD A LOAD BUTTON TO LOAD A PATIENT'S CONTENT
[x]5. SET SOME SORT OF TITLE ON THE UI FOR PATIENT ID
[x]6. DISPLAY THE LOADED DATA ON THE GRAPH
[x]7. LOAD THE MODEL FROM THE DIR Results/BEST_RESNET_00.h5
[x]8. USE THE PREDICT BUTTON AND DISPLAY THE CLASSIFICATION MADE FOR THE PATIENT
[x]9. MAP THE LABELS INTO CORRESPONDING ONES
[x]10.LOAD DIFFERENT MODELS FROM A DIRECTORY
[NOT NECESSARY MAYBE WHEN WE ADD ML ?]11.LOAD DIFFERENT WINDOW SIZED MODELS (DATA NEEDS TO BE LOADED ACCORDING TO THAT) PROBABLY WE NEED TO MAKE ADJUSTABLE BINNING ?
[X]12.CREATE SOME SORT OF SAVE THE PATIENT RESULT TO CSV?

# TODO
[ ]13. Add grad cam to the model to highlight areas and then display them on the current pyqtgraph
[ ]14. After all will be done refactor the code prepare it for deployment
Maybe use flask?


"""

import pathlib
import sklearn
import itertools
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tensorflow as tf
import Modules.constants as constants
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QMainWindow,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QWidget,
    QLabel,
    QFileDialog,
    QListWidget,
    QHBoxLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)


tf.config.set_visible_devices([], "GPU")  # disable GPU


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG-GUI")
        self.resize(900, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # LEFT PANEL
        self.left_panel = QVBoxLayout()
        self.layout.addLayout(self.left_panel, 1)
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search patient id...")

        self.patient_list = QListWidget()
        self.patient_list.setAlternatingRowColors(True)

        self.search_bar.textChanged.connect(self.update_search_list)
        self.left_panel.addWidget(self.search_bar)
        self.left_panel.addWidget(self.patient_list)
        self.load_patient_list()
        self.patient_list.currentItemChanged.connect(self.load_patient_data)

        # Bottom Split: Controls
        """
        IF LAYOUT IS QH = Horizontal layout
        IF LAYOUT IS QV = Vertical layout 
        imagine this as a grid that has a shape
        """
        self.left_bottom_split = QHBoxLayout()
        self.left_panel.addLayout(self.left_bottom_split)

        # LEFT CONTROLS LEFT PANEL
        self.predicted_rhythm = None
        font = QFont("FontAwesome")
        font.setPointSize(20)
        self.left_controls = QHBoxLayout()
        self.left_bottom_split.addLayout(self.left_controls, 2)

        """
        PLEASE DO NOT MODIFY THE BELOW 3 BUTTONS EMPTY QUOTES
        THEY CONTAIN GLYPHS THAT DOES NOT APPEAR IN THE EDITOR UI
        RESOURCE OF GLYPHS IS FONTAWESOME
        """
        self.button_add = QPushButton("")
        self.button_remove = QPushButton("")
        self.button_save = QPushButton("")
        self.button_add.setToolTip("Add Patient")
        self.button_remove.setToolTip("Remove Patient")
        self.button_save.setToolTip("Save Patient")
        self.button_add.setFont(font)
        self.button_remove.setFont(font)
        self.button_save.setFont(font)
        self.left_controls.addWidget(self.button_add)
        self.left_controls.addWidget(self.button_remove)
        self.left_controls.addWidget(self.button_save)
        self.button_save.setEnabled(False)
        self.button_save.clicked.connect(self.save_data)
        self.button_add.clicked.connect(self.add_data)
        self.button_remove.clicked.connect(self.rm_data)

        # RIGHT PANEL
        self.right_panel = QVBoxLayout()
        self.layout.addLayout(self.right_panel, 3)

        # Top: ECG plot
        self.plot_graph = pg.PlotWidget()
        self.right_panel.addWidget(self.plot_graph)

        # Bottom Split: Controls (Left + Right)
        self.right_bottom_split = QHBoxLayout()
        self.right_panel.addLayout(self.right_bottom_split)

        # LEFT CONTROLS
        self.left_controls = QVBoxLayout()
        self.right_bottom_split.addLayout(self.left_controls, 2)

        self.model_options = QComboBox()
        self.model_paths = {}
        self.dropdown_label = QLabel("Loaded Model: --")
        self.left_controls.addWidget(self.model_options)
        self.left_controls.addWidget(self.dropdown_label)

        self.model_dir = QLabel(f"Model Directory: --, --")
        self.left_controls.addWidget(self.model_dir)
        self.load_model_files(pathlib.Path(constants.MODELS_DIR))
        self.model_options.currentTextChanged.connect(self.load_selected_model)

        self.evaluate_button = QPushButton("Evaluate Patient Data")
        self.evaluate_button.clicked.connect(self.evaluate_patient_data)
        self.left_controls.addWidget(self.evaluate_button)

        self.prediction_label = QLabel("Prediction: --, --")
        self.true_label = QLabel("True Label: --, --")
        self.left_controls.addWidget(self.prediction_label)
        self.left_controls.addWidget(self.true_label)

        # RIGHT CONTROLS
        self.right_controls = QVBoxLayout()
        self.right_bottom_split.addLayout(self.right_controls, 2)
        self.diagnostics_table = QTableWidget()
        self.diagnostics_table.setFixedHeight(250)
        self.diagnostics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.diagnostics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.diagnostics_table.setStyleSheet("padding: 5px; color: #444;")
        self.load_diagnostics_button = QPushButton("Load Patient Diagnostics")
        self.load_diagnostics_button.clicked.connect(self.load_patient_diagnostics)
        self.right_controls.addWidget(self.load_diagnostics_button)
        self.right_controls.addWidget(self.diagnostics_table)

        self.X_test = None
        self.true_label_value = None
        self.model = None
        self.selected_patient = None
        self.extra_patient_info = {}
        self.previous_patient = None

        self.load_selected_model()

    def load_patient_list(self):
        if hasattr(self, "all_patients") and self.all_patients:
            return

        DATA_PATH = constants.DATASET / "test"
        if not DATA_PATH.exists():
            print("\n [ !! ] Dataset directory not found!")
            return

        self.subdirs = [d.name for d in DATA_PATH.iterdir() if d.is_dir()]
        patient_files = sorted(
            (
                file
                for dir in self.subdirs
                for file in (DATA_PATH / dir).glob("*.csv")
                if (DATA_PATH / dir).exists()
            ),
            key=lambda f: f.stem,
        )

        self.all_patients = [file.stem for file in patient_files]
        self.update_patient_list(self.all_patients)

    def update_patient_list(self, patient_ids):
        self.patient_list.clear()
        for pid in patient_ids:
            self.patient_list.addItem(pid)

    def update_search_list(self):
        search = self.search_bar.text().lower()
        filtered = [pid for pid in self.all_patients if search in pid.lower()]
        self.update_patient_list(filtered)

    def load_patient_data(self):
        self.prediction_label.setStyleSheet("color:black; font-weight:regular")
        selected_item = self.patient_list.currentItem()
        if not selected_item:
            return

        self.selected_patient = selected_item.text()
        patient_file = self.selected_patient + ".csv"
        patient_label = None
        if self.selected_patient != self.previous_patient:
            self.diagnostics_table.setRowCount(0)
            self.diagnostics_table.setColumnCount(0)
            self.load_diagnostics_button.setEnabled(True)
            self.load_diagnostics_button.setText("Load Patient Diagnostics")
            self.load_diagnostics_button.setToolTip(
                "Click to load diagnostics for this patient"
            )

        for label in range(4):
            data_dir = pathlib.Path(f"Data/Dataset/test/{label}")
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

        try:
            df = pd.read_csv(file_path, header=None)
            self.X_test = df.to_numpy()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV data: {e}")
            return

        self.true_label_value = patient_label

        self.plot_graph.clear()
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_values = np.arange(self.X_test.shape[0]).reshape(-1, 1)
        x_values = scaler.fit_transform(x_values).flatten()

        y_values = self.X_test[:, 0].reshape(-1, 1)
        y_values = scaler.fit_transform(y_values).flatten()

        self.plot_graph.setYRange(-1.5, 1.5)
        self.plot_graph.plot(x_values, y_values, pen="g", width=1.5, name="ECG Signal")

        self.label_map = {0: "AFIB", 1: "GSVT", 2: "SB", 3: "SR"}
        f = self.label_map.get(self.true_label_value, "Unknown")
        self.true_label.setText(
            f"True Label: {f}, True Label Class: {self.true_label_value}"
        )
        self.prediction_label.setText("Prediction: --, --")

    def load_patient_diagnostics(self):
        if self.selected_patient is None:
            QMessageBox.warning(self, "Warning", "No patient selected!")
            return

        try:
            diagnostics_df = pd.read_excel("Data/DiagnosticsTesting.xlsx")
            patient_row = diagnostics_df[
                diagnostics_df["FileName"] == self.selected_patient
            ]

            if not patient_row.empty:
                extra_info = patient_row.drop(columns=["FileName"]).iloc[0]
                self.extra_patient_info = extra_info.to_dict()

                attributes = list(self.extra_patient_info.keys())
                values = list(self.extra_patient_info.values())

                self.diagnostics_table.setRowCount(0)
                self.diagnostics_table.setColumnCount(len(attributes))

                self.diagnostics_table.setHorizontalHeaderLabels(attributes)

                self.diagnostics_table.setRowCount(1)
                for col, value in enumerate(values):
                    self.diagnostics_table.setItem(0, col, QTableWidgetItem(str(value)))

                self.load_diagnostics_button.setText("Refresh Patient Diagnostics")
                self.load_diagnostics_button.setToolTip(
                    "Click to refresh diagnostics for this patient"
                )

                self.previous_patient = self.selected_patient
            else:
                self.extra_patient_info = {}
                self.diagnostics_table.setRowCount(0)
                self.diagnostics_table.setColumnCount(0)
                QMessageBox.warning(
                    self, "Warning", "No diagnostics information found."
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Warning", f"Failed to load diagnostics info: {e}"
            )
            self.extra_patient_info = {}
            self.diagnostics_table.setRowCount(0)
            self.diagnostics_table.setColumnCount(0)  # Clear table content

    def load_model_files(self, folder_path: pathlib.Path):
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"\n[ !! ] Directory not found: {folder_path}")
            return
        model_files = itertools.chain(
            folder_path.glob("*.keras"), folder_path.glob("*.h5")
        )
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

            self.predicted_rhythm = f
            self.button_save.setEnabled(True)

            if predicted_class == self.true_label_value:
                self.prediction_label.setStyleSheet("color:green; font-weight:regular;")
            elif predicted_class != self.true_label_value:
                self.prediction_label.setStyleSheet("color:red; font-weight:regular;")
            else:
                pass

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {e}")

    def add_data(self):
        choice, _ = QFileDialog.getOpenFileName(
            self, "Select Patient File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if choice:
            patient_name = pathlib.Path(choice).stem
            if patient_name not in self.all_patients:
                self.all_patients.append(patient_name)
                self.update_patient_list(self.all_patients)
        elif not patient_name:
            directory = QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                new_patients = []
                for patient_file in pathlib.Path(directory).glob("*.csv"):
                    patient_name = patient_file.stem
                    if patient_name not in self.all_patients:
                        new_patients.append(patient_name)

                if new_patients:
                    self.all_patients.extend(new_patients)
                    self.update_patient_list(self.all_patients)

        else:
            return False

    def rm_data(self):
        selected_item = self.patient_list.currentItem()
        if selected_item:
            patient_name = selected_item.text()

            if patient_name in self.all_patients:
                self.all_patients.remove(patient_name)
                self.update_patient_list(self.all_patients)
                print(f"Removed patient: {patient_name}")
            else:
                QMessageBox.warning(self, "Warning", "Patient not found in list.")
        else:
            QMessageBox.warning(self, "Warning", "No patient selected for removal!")

    def save_data(self):
        if self.selected_patient is None:
            QMessageBox.warning(self, "Warning", "No patient selected!")
            return

        if not self.extra_patient_info:
            QMessageBox.warning(self, "Warning", "No diagnostic data loaded!")
            return

        try:
            diagnostics_df = pd.read_excel("Data/DiagnosticsTesting.xlsx")

            patient_row = diagnostics_df[
                diagnostics_df["FileName"] == self.selected_patient
            ]

            if patient_row.empty:
                QMessageBox.warning(
                    self, "Warning", "No diagnostics data found for this patient!"
                )
                return

            diagnostics_df.loc[
                diagnostics_df["FileName"] == self.selected_patient, "Rhythm"
            ] = self.predicted_rhythm

            diagnostics_df.to_excel("Data/DiagnosticsTesting.xlsx", index=False)
            QMessageBox.information(self, "Success", "Prediction saved successfully!")

            self.load_diagnostics_button.setText("Refresh Patient Diagnostics")
            self.load_diagnostics_button.setToolTip(
                "Click to refresh diagnostics for this patient"
            )

            self.diagnostics_table.setRowCount(0)
            self.diagnostics_table.setColumnCount(0)
            self.load_patient_diagnostics()

            self.button_save.setEnabled(False)
            self.previous_patient = self.selected_patient

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save prediction: {e}")


"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.showFullScreen()
    #    window.show()
    sys.exit(app.exec_())
"""
