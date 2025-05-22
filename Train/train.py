import collections
import os
import sys
import imblearn
import keras_tuner
import numpy as np
import sklearn
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("Modules"))))
import Modules.ds_loader as ds_loader

dataset_loader = ds_loader.DatasetLoader(
    xlsx_path="../Data/Label_Map.xlsx", data_dir="../Data/Internal_Dataset"
)
X, y = dataset_loader.load_data()
X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)
print("Unique classes in y:", np.unique(y_train))
print("Datatype:", (X_train.dtype), (y_train.dtype))
print(f"NaNs in X: {np.isnan(X_train).sum()}")
print(f"Infs in X: {np.isinf(X_train).sum()}")
print(f"Class distribution of training before SMOTE: {collections.Counter(y_train)}")
print(f"Class distribution of validation: {collections.Counter(y_val)}")
print(f"Class distribution of test: {collections.Counter(y_test)}")

print(
    f"Before scaling - X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}"
)


def min_max_normalize(*datasets):
    normalized_datasets = []
    for data in datasets:
        norm_data = []
        for sample in data:
            min_val = np.min(sample)
            max_val = np.max(sample)
            if max_val - min_val == 0:
                norm_sample = np.zeros_like(sample, dtype=np.float32)
            else:
                norm_sample = (sample - min_val) / (max_val - min_val)
            norm_data.append(norm_sample)
        normalized_datasets.append(np.stack(norm_data, axis=0).astype(np.float32))
    return normalized_datasets


X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
print(f"Min and Max of X_train: {np.min(X_train)}, {np.max(X_train)}")
print(f"Min and Max of X_val: {np.min(X_val)}, {np.max(X_val)}")
print(f"Min and Max of X_test: {np.min(X_test)}, {np.max(X_test)}")

print(f"\n\n[ ii ] Applying oversampling via SMOTE")

X_train_flat = X_train.reshape((X_train.shape[0], -1))
smote = imblearn.over_sampling.SMOTE(random_state=42)
X_resampled, y_train = smote.fit_resample(X_train_flat, y_train)
X_train = X_resampled.reshape((-1, *X_train.shape[1:]))
print(f"Class distribution after SMOTE: {collections.Counter(y_train)}")
INPUT_SIZE, FILTERS = 500, 64
RDIR = f"../src/Results/RES_{INPUT_SIZE}_{FILTERS}"
MDIR = f"../RES_{INPUT_SIZE}_{FILTERS}.keras"
CDIR = f"../C_RES_{INPUT_SIZE}_{FILTERS}.keras"
CVDIR = f"../RES_{INPUT_SIZE}_CV.keras"

from baseline import ResnetTuner

tuner = keras_tuner.BayesianOptimization(
    ResnetTuner(),
    objective="val_accuracy",
    max_trials=100,
    overwrite=False,
    directory=RDIR,
    project_name=f"RES_{INPUT_SIZE}_{FILTERS}",
)


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CDIR, monitor="val_accuracy", save_best_only=True, verbose=1
)

tuner.search(
    X_train,
    y_train,
    epochs=300,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping, model_checkpoint],
)

print(tuner.results_summary())

model = tuner.get_best_models(num_models=1)[0]
print(model.summary())


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\n{best_hps.values}")

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_pred = model.predict(X_test)

if y_pred.shape[1] == 1:
    y_pred_binary = (y_pred > 0.5).astype(int)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
else:
    y_pred_binary = np.argmax(y_pred, axis=1)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred, multi_class="ovr")

print("Classification Report (Test Data):")
print(sklearn.metrics.classification_report(y_test, y_pred_binary))
print(f"AUC: {auc}")

y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred, axis=1)

print("Classification Report (Train Data):")
print(sklearn.metrics.classification_report(y_train, y_train_pred))
model.save(MDIR)
