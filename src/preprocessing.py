import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32

DATASET_DIR = "../dataset"


def create_data_generators():

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_data = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    validation_data = validation_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_data, validation_data
