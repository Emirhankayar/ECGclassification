import gc
import tensorflow as tf
import keras_tuner


# 1-D convolutional ResNet model
# https://pmc.ncbi.nlm.nih.gov/articles/PMC10128986/#sec012
class ResnetTuner(keras_tuner.HyperModel):
    def residual_block(self, x, f_units, k_units, name_prefix="res"):
        shortcut = x

        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_1")(x)

        x = tf.keras.layers.Conv1D(
            filters=f_units,
            kernel_size=k_units,
            strides=1,
            padding="same",
            name=f"{name_prefix}_conv_1",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_2")(x)

        x = tf.keras.layers.Conv1D(
            filters=f_units,
            kernel_size=k_units,
            strides=1,
            padding="same",
            name=f"{name_prefix}_conv_2",
            dtype=x.dtype,
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_3")(x)

        if shortcut.shape[-1] != f_units:
            shortcut = tf.keras.layers.Conv1D(
                filters=f_units,
                kernel_size=1,
                padding="same",
                name=f"{name_prefix}_conv_skip",
            )(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        return x

    def build(self, hp):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        gc.collect()
        tf.keras.backend.clear_session()
        # HYPERPARAMS
        f_units = hp.Choice("f_units", [8,16,32,64])
        dropout = hp.Float("dropout_0", min_value=0.2, max_value=0.5, step=0.05)
        p_size = hp.Choice("p_size", [2,3,4,5,10])
        # INPUT LAYER
        inputs = tf.keras.Input(shape=(500, 12), name="ecg_sig")

        # FIRST CONVOLUTION
        x = tf.keras.layers.Conv1D(filters=f_units,kernel_size=hp.Choice("k_units_0", [1, 3 , 5, 7]),strides=1,padding="same",)(inputs)

        # RESIDUALS
        x = self.residual_block(
            x,
            f_units,
            k_units=hp.Choice("k_units_1", [1, 3, 5, 7]),
            name_prefix="res_1",
        )
        x = tf.keras.layers.MaxPooling1D(pool_size=p_size, strides=2, name="max_pool_1")(x)

        x = self.residual_block(x,f_units * 2,k_units=hp.Choice("k_units_2", [1, 3, 5, 7]),name_prefix="res_2",)
        x = tf.keras.layers.MaxPooling1D(pool_size=p_size, strides=2, name="max_pool_2")(x)

        x = self.residual_block(x, f_units * 4, k_units=hp.Choice("k_units_3", [1, 3, 5, 7]), name_prefix="res_3", )
        x = tf.keras.layers.MaxPooling1D(pool_size=p_size, strides=2, name="max_pool_3")(x)

        # CLASSIFIER
        x = tf.keras.layers.GlobalAveragePooling1D(name="glob_avg_pool_1")(x)
        x = tf.keras.layers.Dense(f_units * 4, activation="relu", name="fc_1")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

        # OUTPUT
        outputs = tf.keras.layers.Dense(4, activation="softmax", name="output")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-3),
            nesterov=True,
            #weight_decay=hp.Choice("weight_decay", [1e-3, 1e-4, 1e-5, 0.0]),
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


class Resnet:
    def __init__(
        self,
        input_shape=(2500, 12),
        num_classes=4,
        f_units=256,
        d_units_0=64,
        dropout_0=0.5,
        learning_rate=0.001,
        weight_decay=1e-4,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.f_units = f_units
        self.d_units_0 = d_units_0
        self.dropout_0 = dropout_0
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def residual_block(self, x, c_units, k_units, name_prefix="res"):
        shortcut = x
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)

        x = tf.keras.layers.Conv1D(
            filters=c_units,
            kernel_size=k_units,
            strides=1,
            padding="same",
            name=f"{name_prefix}_conv_1",
        )(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

        x = tf.keras.layers.Conv1D(
            filters=c_units,
            kernel_size=k_units,
            strides=1,
            padding="same",
            name=f"{name_prefix}_conv_2",
        )(x)
        x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu_3")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn3")(x)

        if shortcut.shape[-1] != c_units:
            shortcut = tf.keras.layers.Conv1D(
                filters=c_units,
                kernel_size=1,
                strides=1,
                padding="same",
                name=f"{name_prefix}_conv_skip",
            )(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        return x

    def build_model(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        gc.collect()
        tf.keras.backend.clear_session()

        inputs = tf.keras.Input(shape=self.input_shape, name="ecg_sig")

        x = tf.keras.layers.Conv1D(
            filters=self.f_units,
            kernel_size=5,
            strides=2,
            padding="same",
            name="initial_conv",
        )(inputs)

        x = self.residual_block(x, self.f_units, k_units=3, name_prefix="res_1")
        x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name="max_pool_1")(x)

        x = self.residual_block(x, self.f_units//2, k_units=3, name_prefix="res_2")
        x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name="max_pool_2")(x)

        x = self.residual_block(x//2, self.f_units//4, k_units=3, name_prefix="res_3")
        x = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name="max_pool_3")(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(self.f_units // 4, activation="relu", name="fc_1")(x)
        x = tf.keras.layers.Dropout(self.dropout_0, name="drop_1")(x)

        outputs = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="output", dtype="float32"
        )(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
