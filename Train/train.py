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
    xlsx_path="../Data/Label_Map.xlsx", data_dir="../Data/ECGDataDenoised"
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

print("\n[ ii ] Applying MinMax scaling to the dataset...")
scaler = sklearn.preprocessing.MinMaxScaler()

X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

print(
    f"Before scaling - X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}"
)

X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_val_reshaped = scaler.transform(X_val_reshaped)
X_test_reshaped = scaler.transform(X_test_reshaped)

X_train = X_train_reshaped.reshape(X_train.shape[0], *X_train.shape[1:])
X_val = X_val_reshaped.reshape(X_val.shape[0], *X_val.shape[1:])
X_test = X_test_reshaped.reshape(X_test.shape[0], *X_test.shape[1:])

print(f"Min and Max of X_train: {np.min(X_train)}, {np.max(X_train)}")
print(f"Min and Max of X_val: {np.min(X_val)}, {np.max(X_val)}")
print(f"Min and Max of X_test: {np.min(X_test)}, {np.max(X_test)}")

print(f"\n\n[ ii ] Applying oversampling via SMOTE")

X_train_flat = X_train.reshape((X_train.shape[0], -1))
smote = imblearn.over_sampling.SMOTE(random_state=42)
X_resampled, y_train = smote.fit_resample(X_train_flat, y_train)
X_train = X_resampled.reshape((-1, *X_train.shape[1:]))
print(f"Class distribution after SMOTE: {collections.Counter(y_train)}")
INPUT_SIZE, LAYERS = 500, 3
RDIR = f"../src/Results/RES_{INPUT_SIZE}_{LAYERS}"
MDIR = f"../RES_{INPUT_SIZE}_{LAYERS}.keras"
CDIR = f"../C_RES_{INPUT_SIZE}_{LAYERS}.keras"
CVDIR = f"../RES_{INPUT_SIZE}_CV.keras"

from baseline import ResnetTuner

# model = ResnetTuner()
# model = model.build_model()
# print(model.summary())

tuner = keras_tuner.BayesianOptimization(
    ResnetTuner(),
    objective="val_accuracy",
    max_trials=100,
    overwrite=False,
    directory=RDIR,
    project_name=f"RES_{INPUT_SIZE}_{LAYERS}",
)


lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CDIR, monitor="val_accuracy", save_best_only=True, verbose=1
)

tuner.search(
    X_train,
    y_train,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping, model_checkpoint],
)

print(tuner.results_summary())

model = tuner.get_best_models(num_models=1)[0]
print(model.summary())


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\n{best_hps.values}")


"""history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping, model_checkpoint],
)
"""
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
