import gc
import keras_tuner
import tensorflow as tf


# 1-D convolutional ResNet model
class Resnet(keras_tuner.HyperModel):
    def block(self, x, f_units=64, k_units=3, name_prefix="block"):
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_1")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=3, strides=2, name=f"{name_prefix}_pool_1"
        )(x)

        # C1
        x = tf.keras.layers.Conv1D(
            filters=f_units,
            kernel_size=k_units,
            padding="same",
            name=f"{name_prefix}_conv_1",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_2")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2, name=f"{name_prefix}_pool_2"
        )(x)

        # C2
        x = tf.keras.layers.Conv1D(
            filters=f_units,
            kernel_size=k_units,
            padding="same",
            name=f"{name_prefix}_conv_2",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_3")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2, name=f"{name_prefix}_pool_3"
        )(x)
        return x

    def build(self, hp):
        gc.collect()
        tf.keras.backend.clear_session()

        # HP
        f_units = hp.Choice("f_units", [32, 64])
        k_units = hp.Choice("k_units", [6, 8, 10, 12, 14, 16])
        d_units = hp.Choice("d_units", [32, 64, 128, 256])
        dropout_0 = hp.Float("dropout_0", min_value=0.2, max_value=0.5, step=0.05)
        dropout_1 = hp.Float("dropout_1", min_value=0.2, max_value=0.5, step=0.05)

        # INPUT LAYER
        inputs = tf.keras.Input(shape=(2500, 12), name="ecg_sig")

        # FIRST CONVOLUTION
        x = tf.keras.layers.Conv1D(
            filters=f_units, kernel_size=k_units, padding="same"
        )(inputs)

        # BLOCKS
        x = self.block(x, f_units, k_units // 2, name_prefix="block")

        # CLASSIFIER
        x = tf.keras.layers.Flatten(name="flat")(x)
        x = tf.keras.layers.Dense(d_units, activation="relu", name="fc_1")(x)
        x = tf.keras.layers.Dropout(dropout_0, name="drop_1")(x)
        x = tf.keras.layers.Dense(d_units // 2, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_1, name="drop_2")(x)

        # OUTPUT
        outputs = tf.keras.layers.Dense(4, activation="softmax", name="output")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-3),
            weight_decay=hp.Choice("weight_decay", [1e-3, 1e-4, 1e-5, 0.0]),
        )
        model.compile(
            optimizer=optimizer,
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
