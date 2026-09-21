from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128
NUM_CLASSES = 6


def create_improved_model():

    model = models.Sequential([

        # Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 1),
            kernel_regularizer=regularizers.l2(0.0001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizers.l2(0.0001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_regularizer=regularizers.l2(0.0001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification
        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.0001)
        ),

        layers.Dropout(0.5),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
