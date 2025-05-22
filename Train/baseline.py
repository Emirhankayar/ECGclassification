import gc

import keras_tuner
import tensorflow as tf


# 1-D convolutional ResNet model
# https://pmc.ncbi.nlm.nih.gov/articles/PMC10128986/#sec012
class ResnetTuner(keras_tuner.HyperModel):
    def residual_block(
        self, x, f_units, k_units, p_size, n_convolutions, name_prefix="res"
    ):
        # shortcut = x
        for i in range(n_convolutions):
            x = tf.keras.layers.Conv1D(
                filters=f_units,
                kernel_size=k_units,
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                name=f"{name_prefix}_conv_{i+1}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_{i+2}")(x)
            x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_{i+2}")(x)

        """
        if shortcut.shape[-1] != f_units:
            shortcut = tf.keras.layers.Conv1D(
                filters=f_units,
                kernel_size=1,
                padding="same",
                kernel_initializer="he_normal",
                name=f"{name_prefix}_conv_skip",
            )(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_")(x)
        """
        x = tf.keras.layers.MaxPooling1D(
            pool_size=p_size, strides=2, name=f"{name_prefix}_mp"
        )(x)
        return x

    def build(self, hp):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        gc.collect()
        tf.keras.backend.clear_session()

        # HYPERPARAMS
        f_units = hp.Choice("f_units", [64])
        p_size = 5
        n_convolutions = hp.Int("n_convolutions", min_value=1, max_value=5, step=1)
        # n_convolutions = 3
        # m_convolutions = 4
        m_convolutions = hp.Int("m_convolutions", min_value=1, max_value=5, step=1)
        inputs = tf.keras.layers.Input(shape=(500, 12))

        # FIRST CONVOLUTION
        x = tf.keras.layers.Conv1D(
            filters=f_units,
            kernel_size=3,
            strides=2,
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(name=f"bn_1")(x)
        x = tf.keras.layers.ReLU(name=f"relu_1")(x)
        # RESIDUALS
        for i in range(m_convolutions):
            x = self.residual_block(
                x,
                f_units=f_units,
                k_units=3,
                p_size=p_size,
                n_convolutions=n_convolutions,
                name_prefix=f"res_{i+1}",
            )

        # CLASSIFIER
        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(
            hp.Choice(f"dense_units_1", [64, 32]),
            activation="relu",
            name=f"fc_1",
        )(x)
        x = tf.keras.layers.Dropout(
            hp.Float(f"dropout_1", min_value=0.25, max_value=0.5, step=0.05),
            name=f"dropout_1",
        )(x)
        x = tf.keras.layers.Dense(
            hp.Choice(f"dense_units_2", [64, 32]),
            activation="relu",
            name=f"fc_2",
        )(x)
        x = tf.keras.layers.Dropout(
            hp.Float(f"dropout_2", min_value=0.25, max_value=0.5, step=0.05),
            name=f"dropout_2",
        )(x)

        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)

        model = tf.keras.models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float(
                    "learning_rate", min_value=1e-5, max_value=9e-4, sampling="log"
                ),
                weight_decay=hp.Float(
                    "weight_decay", min_value=1e-6, max_value=1e-2, sampling="log"
                ),
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            batch_size=hp.Choice("batch_size", [32]),
            *args,
            **kwargs,
        )


class Resnet:
    def __init__(self, input_shape=(500, 12)):
        self.input_shape = input_shape

    def residual_block(
        self, x, c_units, k_units, p_size, n_convolutions, name_prefix="res"
    ):
        # shortcut = x

        for i in range(n_convolutions):
            x = tf.keras.layers.Conv1D(
                filters=c_units,
                kernel_size=k_units,
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                name=f"{name_prefix}_conv_{i+1}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn_{i+2}")(x)
            x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_{i+2}")(x)
        """
        if shortcut.shape[-1] != c_units:
            shortcut = tf.keras.layers.Conv1D(
                filters=c_units,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_initializer="he_normal",
                name=f"{name_prefix}_conv_skip",
            )(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        """
        x = tf.keras.layers.MaxPooling1D(
            pool_size=p_size, strides=2, name=f"{name_prefix}_mp"
        )(x)
        return x

    def build_model(
        self,
        m_convolutions=4,
        n_convolutions=3,
        c_units=128,
        p_size=2,
        d_units_1=128,
        dropout_1=0.3,
        d_units_2=64,
        dropout_2=0.3,
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        gc.collect()
        tf.keras.backend.clear_session()

        inputs = tf.keras.Input(shape=self.input_shape, name="ecg_sig")

        x = tf.keras.layers.Conv1D(
            filters=c_units,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            name="initial_conv",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(name=f"bn_1")(x)
        x = tf.keras.layers.ReLU(name=f"relu_1")(x)

        # Residual blocks
        for i in range(m_convolutions):
            x = self.residual_block(
                x,
                c_units=c_units,
                k_units=3,
                p_size=p_size,
                n_convolutions=n_convolutions,
                name_prefix=f"res_{i+1}",
            )

        x = tf.keras.layers.Flatten(name="flatten")(x)

        x = tf.keras.layers.Dense(d_units_1, activation="relu", name="fc_1")(x)
        x = tf.keras.layers.Dropout(dropout_1, name="drop_1")(x)

        x = tf.keras.layers.Dense(d_units_2, activation="relu", name="fc_2")(x)
        x = tf.keras.layers.Dropout(dropout_2, name="drop_2")(x)

        outputs = tf.keras.layers.Dense(4, activation="softmax", dtype="float32")(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
