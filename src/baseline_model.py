from tensorflow.keras import layers, models

IMG_SIZE = 128
NUM_CLASSES = 6


def create_baseline_model():

    model = models.Sequential([
        
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 1)
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
