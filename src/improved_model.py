import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

IMG_SIZE = 128
NUM_CLASSES = 6


def conv_block(filters, kernel_size, stride=2):
    return [
        layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(0.1)
        ),
        layers.BatchNormalization(),
        layers.PReLU()
    ]


def create_improved_model():

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # First convolution
    x = layers.Conv2D(
        32,
        (5, 5),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    # 1x1 convolution
    x = layers.Conv2D(
        32,
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    # Residual block 1
    shortcut = x

    x = layers.Conv2D(
        64,
        (5, 5),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(
        64,
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)

    # Match shortcut dimensions
    shortcut = layers.Conv2D(
        64,
        (1, 1),
        strides=2,
        padding="same",
        use_bias=False
    )(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.PReLU()(x)

    # Residual block 2
    shortcut = x

    x = layers.Conv2D(
        128,
        (5, 5),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    x = layers.Conv2D(
        128,
        (1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(
        128,
        (1, 1),
        strides=2,
        padding="same",
        use_bias=False
    )(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.PReLU()(x)

    # Final feature extraction
    x = layers.Conv2D(
        256,
        (5, 5),
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(0.1)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)

    # Global feature representation
    x = layers.GlobalAveragePooling2D()(x)

    # Classification
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
