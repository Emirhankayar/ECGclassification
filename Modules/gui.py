import itertools
import pathlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import sklearn as sk
import tensorflow as tf
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)
import Modules.constants as constants

tf.config.set_visible_devices([], "GPU")  # disable GPU


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MATE-GUI")
        self.resize(900, 600)

        # MAIN WRAPPER
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # LEFT PANEL
        self.l_panel = QVBoxLayout()
        self.layout.addLayout(self.l_panel, 1)

        # LEFT PANEL SEARCH BAR
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search patient id...")
        self.search_bar.textChanged.connect(self.update_search_list)
        self.l_panel.addWidget(self.search_bar)

        # LEFT PANEL LIST
        self.patient_list = QListWidget()
        self.patient_list.setAlternatingRowColors(True)
        self.l_panel.addWidget(self.patient_list)
        self.update_patient_list([])
        self.patient_list.currentItemChanged.connect(self.load_patient_data)

        # LEFT PANEL CONTROLS (ADD,REMOVE,SAVE)
        self.predicted_rhythm = None
        font = QFont("FontAwesome")
        font.setPointSize(20)
        self.l_panel_l_ctrl = QHBoxLayout()
        self.l_panel.addLayout(self.l_panel_l_ctrl, 2)
        """
        PLEASE DO NOT MODIFY THE BELOW 3 BUTTONS EMPTY QUOTES
        THEY CONTAIN GLYPHS THAT DOES NOT APPEAR IN THE EDITOR UI
        RESOURCE OF GLYPHS IS FONTAWESOME
        """
        self.btn_add = QPushButton("")
        self.btn_remove = QPushButton("")
        self.btn_save = QPushButton("")
        self.btn_add.setToolTip("Add Patient")
        self.btn_remove.setToolTip("Remove Patient")
        self.btn_save.setToolTip("Save Patient")
        self.btn_add.setFont(font)
        self.btn_remove.setFont(font)
        self.btn_save.setFont(font)
        self.l_panel_l_ctrl.addWidget(self.btn_add)
        self.l_panel_l_ctrl.addWidget(self.btn_remove)
        self.l_panel_l_ctrl.addWidget(self.btn_save)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_data)
        self.btn_add.clicked.connect(self.add_data)
        self.btn_remove.clicked.connect(self.rm_data)

        # RIGHT PANEL
        self.r_panel = QVBoxLayout()
        self.layout.addLayout(self.r_panel, 3)

        # RIGHT PANEL PLOT
        self.plot_graph = pg.PlotWidget()
        self.r_panel.addWidget(self.plot_graph)

        # RIGHT PANEL CONTROLS RIGHT/LEFT
        self.r_bot_split = QHBoxLayout()
        self.r_panel.addLayout(self.r_bot_split)
        self.r_panel_l_ctrl = QHBoxLayout()
        self.r_bot_split.addLayout(self.r_panel_l_ctrl, 2)

        # RIGHT PANEL LEFT CONTROLS
        self.r_panel_ll_ctrl = QVBoxLayout()
        self.r_panel_lr_ctrl = QVBoxLayout()
        self.r_panel_l_ctrl.addLayout(self.r_panel_ll_ctrl)
        self.r_panel_l_ctrl.addLayout(self.r_panel_lr_ctrl)

        self.model_options = QComboBox()
        self.model_paths = {}
        self.dropdown_label = QLabel("Loaded Model: --")
        self.r_panel_ll_ctrl.addWidget(self.model_options)
        self.r_panel_ll_ctrl.addWidget(self.dropdown_label)

        self.model_dir = QLabel(f"Model Directory: --, --")
        self.r_panel_ll_ctrl.addWidget(self.model_dir)
        self.load_model_files(pathlib.Path(constants.MODELS_DIR))
        self.model_options.currentTextChanged.connect(self.load_selected_model)

        self.btn_eval = QPushButton("Evaluate Patient Data")
        self.btn_eval.clicked.connect(self.evaluate_patient_data)
        self.r_panel_lr_ctrl.addWidget(self.btn_eval)

        self.prediction_label = QLabel("Prediction: --, --")
        self.true_label = QLabel("True Label Class: --, --\nTrue Label: --, --")
        self.r_panel_lr_ctrl.addWidget(self.prediction_label)
        self.r_panel_lr_ctrl.addWidget(self.true_label)

        # RIGHT PANEL RIGHT CONTROLS
        self.r_panel_r_ctrl = QVBoxLayout()
        self.r_bot_split.addLayout(self.r_panel_r_ctrl, 1)

        # DEFAULTS
        self.X_test = None
        self.all_patients = []
        self.patient_dir_map = {}
        self.true_label_value = None
        self.model = None
        self.selected_patient = None
        self.info_diag = {}
        self.previous_patient = None
        self.mounted_dirs = []
        self.diagnostics_df = None
        self.load_diag_file()
        self.draw_diag_grid()
        self.load_selected_model()

    def load_patient_list(self, dir_path: pathlib.Path):
        label_path = dir_path / "Label_Map.xlsx"
        if not label_path.exists():
            QMessageBox.critical(self, "Error", f"Label map not found at {label_path}")
            return

        # TODO REMOVE PATIENT FROM THE LIST IF DIR DOES NOT EXIST
        label_df = pd.read_excel(label_path)
        self.label_df = label_df
        self.all_patients = label_df["FileName"].astype(str).tolist()
        self.all_patients = sorted(set(self.all_patients))
        self.update_patient_list(self.all_patients)

    def update_patient_list(self, patient_ids):
        self.patient_list.clear()
        for pid in patient_ids:
            self.patient_list.addItem(pid)

    def update_search_list(self):
        search = self.search_bar.text().lower()
        filtered = [pid for pid in self.all_patients if search in pid.lower()]
        self.update_patient_list(filtered)

    def update_label_data_on_change(self):
        patient_row = self.label_df[self.label_df["FileName"] == self.selected_patient]
        self.true_label_value = int(patient_row["Rhythm"].iloc[0])
        self.label_map = {
            0: "Atrial Fibrillation (AFIB)",
            1: "Generic Supraventricular Tachycardia (GSVT)",
            2: "Sinus Bradycardia (SB)",
            3: "Sinus Rhythm (SR)",
        }
        label_text = self.label_map.get(self.true_label_value, "Unknown")
        self.true_label.setText(
            f"True Label Class: {self.true_label_value}\nTrue Label: {label_text}"
        )

    def load_patient_data(self):
        self.prediction_label.setStyleSheet("color:black; font-weight:regular")
        selected_item = self.patient_list.currentItem()
        if not selected_item:
            return

        self.selected_patient = selected_item.text()
        (
            self.info_diag.clear()
            if self.selected_patient != selected_item.text()
            else None
        )

        patient_file = f"{self.selected_patient}.csv"

        if self.selected_patient not in self.patient_dir_map:
            QMessageBox.critical(
                self,
                "Error",
                f"Directory for patient '{self.selected_patient}' not found.",
            )
            return

        data_path = self.patient_dir_map[self.selected_patient] / patient_file
        if not data_path.exists():
            QMessageBox.critical(
                self, "Error", f"Patient data file not found at {data_path}"
            )
            return

        df = pd.read_csv(data_path, header=None)
        self.X_test = df.to_numpy()

        self.update_label_data_on_change()

        # FANCY UP THE PLOT
        self.plot_graph.clear()
        scaler = sk.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_values = np.arange(self.X_test.shape[0]).reshape(-2, 2)
        x_values = scaler.fit_transform(x_values).flatten()
        y_values = self.X_test[:, 0].reshape(-1, 1)
        y_values = scaler.fit_transform(y_values).flatten()

        self.plot_graph.setYRange(-1.5, 1.5)
        self.plot_graph.plot(x_values, y_values, pen="g", width=1.5, name="ECG Signal")

        self.prediction_label.setText("Prediction: --, --")
        self.load_patient_diag()

    def add_data(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Patient Directory")
        if not directory:
            return

        dir_path = pathlib.Path(directory)

        if dir_path in self.mounted_dirs:
            QMessageBox.warning(
                self,
                "Duplicate Directory",
                f"The directory '{dir_path}' has already been added.",
            )
            return

        patient_files = list(dir_path.glob("*.csv"))
        if not patient_files:
            QMessageBox.warning(
                self,
                "Warning",
                "No .csv files found in the directory. Please provide a directory with .csv files.",
            )
            return

        for file in patient_files:
            patient_name = file.stem
            self.all_patients.append(patient_name)
            self.patient_dir_map[patient_name] = dir_path

        self.mounted_dirs.append(dir_path)
        self.load_patient_list(dir_path)
        print(f"[ OK ] Loaded {len(patient_files)} patients from {dir_path}")

    def draw_diag_grid(self):
        if hasattr(self, "diag_grid_widget") and self.diag_grid_widget:
            return

        self.diag_grid_widget = QWidget()
        self.diag_grid_layout = QGridLayout()
        self.diag_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.diag_grid_layout.setSpacing(15)

        self.diag_labels = {}
        cols = 3

        for i, col_name in enumerate(self.diagnostics_columns):
            label = QLabel(f"<b>{col_name}</b>: --")
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignLeft)
            row = i // cols
            col = i % cols
            self.diag_grid_layout.addWidget(label, row, col)
            self.diag_labels[col_name] = label

        self.diag_grid_widget.setLayout(self.diag_grid_layout)
        self.r_panel_r_ctrl.addWidget(self.diag_grid_widget)

    def load_diag_file(self):
        try:
            self.diagnostics_df = pd.read_excel("Data/DiagnosticsTesting.xlsx")
            self.diagnostics_columns = list(self.diagnostics_df.columns)
            self.diagnostics_columns.remove("FileName")
        except Exception as e:
            QMessageBox.warning(
                self, "Warning", f"Failed to load diagnostics file: {e}"
            )
            self.diagnostics_df = pd.DataFrame()
            self.diagnostics_columns = []

    def load_patient_diag(self):
        if self.selected_patient is None:
            QMessageBox.warning(self, "Warning", "No patient selected!")
            return

        self.draw_diag_grid()

        patient_row = self.diagnostics_df[
            self.diagnostics_df["FileName"] == self.selected_patient
        ]

        if not patient_row.empty:
            info = patient_row.drop(columns=["FileName"]).iloc[0]
            for col_name in self.diagnostics_columns:
                value = info.get(col_name, "--")
                display_value = "--" if pd.isna(value) else value
                self.diag_labels[col_name].setText(
                    f"<b>{col_name}</b>: {display_value}"
                )

            self.info_diag = info.to_dict()
            rhythm_value = patient_row["Rhythm"].iloc[0]
            self.btn_eval.setEnabled(pd.isna(rhythm_value))

    def load_model_files(self, dir_path: pathlib.Path):
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"\n[ !! ] Directory not found: {dir_path}")
            return
        model_files = itertools.chain(dir_path.glob("*.keras"), dir_path.glob("*.h5"))
        for model_file in model_files:
            filename = model_file.stem
            self.model_options.addItem(filename)
            self.model_paths[filename] = model_file

    def load_selected_model(self):
        self.prediction_label.setText("Prediction: --, --")
        selected_file = self.model_options.currentText()
        full_path = self.model_paths.get(selected_file)
        self.model = tf.keras.models.load_model(full_path)
        self.dropdown_label.setText(f"Loaded Model: {selected_file}")
        self.model_dir.setText(f"Model Directory: {full_path}")

    def min_max_normalize(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data, dtype=np.float32)
        return ((data - min_val) / (max_val - min_val)).astype(np.float32)

    def evaluate_patient_data(self):
        if self.X_test is None or self.model is None:
            QMessageBox.warning(self, "Warning", "No patient data/Model loaded!")
            return

        patient_data = np.expand_dims(self.X_test, axis=0)
        patient_data = self.min_max_normalize(patient_data[0])
        patient_data = np.expand_dims(patient_data, axis=0)

        self.patient_data = patient_data
        prediction = self.model.predict(patient_data)
        predicted_class = np.argmax(prediction)
        self.predicted_class_value = predicted_class

        f = self.label_map.get(predicted_class, "Unknown")

        self.prediction_label.setText(
            f"Prediction: {f},\nPrediction Class: {predicted_class}"
        )

        self.predicted_rhythm = f
        self.btn_save.setEnabled(True)

        color = "green" if predicted_class == self.true_label_value else "red"
        self.prediction_label.setStyleSheet(f"color:{color}; font-weight:regular;")

    def save_data(self):
        if self.selected_patient is None:
            QMessageBox.warning(self, "Warning", "No patient selected!")
            return

        if self.predicted_rhythm is None:
            QMessageBox.warning(self, "Warning", "No prediction made!")
            return

        idx = self.diagnostics_df.index[
            self.diagnostics_df["FileName"] == self.selected_patient
        ]
        if idx.empty:
            QMessageBox.warning(
                self, "Warning", "No diagnostics data found for this patient!"
            )
            return

        self.diagnostics_df.loc[idx, "Rhythm"] = self.predicted_rhythm

        self.diagnostics_df.to_excel("Data/DiagnosticsTesting.xlsx", index=False)
        self.load_patient_diag()

        QMessageBox.information(self, "Success", "Prediction saved successfully!")

        self.btn_save.setEnabled(False)
        self.previous_patient = self.selected_patient

    def reset_ui(self):
        self.update_patient_list([])
        self.patient_list.clear()
        self.search_bar.clear()
        self.plot_graph.clear()
        self.prediction_label.setText("Prediction: --, --")
        self.prediction_label.setStyleSheet("color:black; font-weight:regular;")
        self.true_label.setText("True Label: --, --")
        self.btn_save.setEnabled(False)
        self.X_test = self.selected_patient = self.previous_patient = (
            self.true_label_value
        ) = None
        self.info_diag = {}
        if hasattr(self, "diag_labels"):
            for label in self.diag_labels.values():
                label.setText("")
                label.hide()

    def rm_data(self):
        if not self.mounted_dirs:
            QMessageBox.information(self, "Info", "No directories to remove.")
            return

        removed_patients = []
        for dir_path in self.mounted_dirs:
            for pid, path in list(self.patient_dir_map.items()):
                if path == dir_path:
                    removed_patients.append(pid)
                    self.patient_dir_map.pop(pid, None)
                    if pid in self.all_patients:
                        self.all_patients.remove(pid)

        self.mounted_dirs.clear()
        self.reset_ui()

        QMessageBox.information(
            self, "Unmounted", f"Removed {len(removed_patients)} patients and reset UI."
        )
