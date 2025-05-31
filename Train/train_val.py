import collections
import gc
import os
import pathlib
import pickle
import sys
import imblearn
import keras_tuner
import numpy as np
import sklearn
import tensorflow as tf
import train_constants as c
from Train.model import Resnet, ResnetTuner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("Modules"))))
import Modules.ds_loader as ds_loader

dataset_loader = ds_loader.DatasetLoader(
    xlsx_path="../Data/Internal_Dataset/Label_Map.xlsx",
    data_dir="../Data/Internal_Dataset",
)

X, y = dataset_loader.load_data()

X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True
)
print(f"Class distribution of training: {collections.Counter(y_train)}")
print(f"Class distribution of validation: {collections.Counter(y_val)}")
print(f"Class distribution of test: {collections.Counter(y_test)}")


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


tuner = keras_tuner.RandomSearch(
    ResnetTuner(),
    objective="val_loss",
    max_trials=100,
    overwrite=False,
    directory=c.RDIR,
    project_name=f"RES_{c.INPUT_SIZE}_{c.FILTERS}",
)

tuner.reload()

best_trials = tuner.oracle.get_best_trials(num_trials=c.NUM_MODELS)

for trial in best_trials:
    print(f"Trial ID: {trial.trial_id}, Score: {trial.score}")
    print("Hyperparameters:", trial.hyperparameters.values)

best_hps_list = tuner.get_best_hyperparameters(num_trials=c.NUM_MODELS)

results_dir = c.RDIR / f"RES_{c.INPUT_SIZE}_{c.FILTERS}"
results_dir.mkdir(parents=True, exist_ok=True)
all_model_results = []

print(f"{'='*80}")
print(f"PHASE 1: TRAINING FOR ALL {c.NUM_MODELS} MODELS")
print(f"{'='*80}")
for model_idx in range(c.NUM_MODELS):
    individual_model_file = results_dir / f"model_{model_idx + 1}.pkl"

    if individual_model_file.exists():
        print(
            f"Model {model_idx + 1} results already exist at {individual_model_file}. Skipping..."
        )
        with open(individual_model_file, "rb") as f:
            existing_result = pickle.load(f)
            all_model_results.append(existing_result)
        continue

    print(f"\n{'='*60}")
    print(f"TRAINING MODEL {model_idx + 1}/{c.NUM_MODELS}")
    print(f"{'='*60}")

    best_hps = best_hps_list[model_idx]
    print(f"\nHyperparameters for Model {model_idx + 1}:")
    for param in best_hps.space:
        print(f"  {param.name}: {best_hps.get(param.name)}")

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gc.collect()

    X_tr, X_val_fold = min_max_normalize(X_train, X_val)

    orig_shape = X_tr.shape[1:]
    X_tr_flat = X_tr.reshape((X_tr.shape[0], -1))
    smote = imblearn.over_sampling.SMOTE(random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_flat, y_train)
    X_tr = X_tr_res.reshape((-1, *orig_shape)).astype(np.float32)
    y_tr = y_tr_res
    X_tr, y_tr = sklearn.utils.shuffle(X_tr, y_tr, random_state=42)

    print(
        f"Model {model_idx + 1} - Training samples: {X_tr.shape[0]}, Validation samples: {X_val_fold.shape[0]}"
    )
    print(f"Class distribution after SMOTE: {collections.Counter(y_tr)}")

    model_builder = Resnet()
    model = model_builder.build_model(
        c_units=best_hps.get("f_units"),
        p_size=best_hps.get("p_size"),
        m_convolutions=best_hps.get("m_convolutions"),
        n_convolutions=best_hps.get("n_convolutions"),
        coefficient=best_hps.get("coefficient"),
        k_units_1=best_hps.get("kernel_size_init"),
        k_units_2=best_hps.get("kernel_size_res"),
        d_units_1=best_hps.get("dense_units_1"),
        dropout_1=best_hps.get("dropout_1"),
        d_units_2=best_hps.get("dense_units_2"),
        dropout_2=best_hps.get("dropout_2"),
        learning_rate=best_hps.get("learning_rate"),
        weight_decay=best_hps.get("weight_decay"),
        )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{c.CDIR}_model_{model_idx + 1}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val_fold, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[lr_scheduler, early_stopping, model_checkpoint],
        verbose=1,
    )

    val_loss, val_accuracy = model.evaluate(X_val_fold, y_val, verbose=0)
    print(f"Model {model_idx + 1} Validation Accuracy: {val_accuracy:.4f}")

    model.save(f"{c.CVDIR}_model_{model_idx + 1}.keras", include_optimizer=False)
    print(f"Saved model {model_idx + 1} with Accuracy: {val_accuracy:.4f}")

    print(f"\n=== Model {model_idx + 1} Training Results ===")
    print(f"Final validation accuracy: {val_accuracy:.4f}")

    model_result = {
        "model_idx": model_idx + 1,
        "hyperparameters": {
            param.name: best_hps.get(param.name) for param in best_hps.space
        },
        "val_accuracy": float(val_accuracy),
        "history": history,
    }

    with open(individual_model_file, "wb") as f:
        pickle.dump(model_result, f)

    print(f"✅ Saved Model {model_idx + 1} results to: {individual_model_file}")

    print(f"\nModel {model_idx + 1} training completed!")

print(f"\n{'='*80}")
print(f"PHASE 1 COMPLETED: ALL {c.NUM_MODELS} MODELS TRAINED")
print(f"{'='*80}")
