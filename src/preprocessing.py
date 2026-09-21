from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32

TRAIN_PATH = "dataset/NEU-DET/train/images"
VALIDATION_PATH = "dataset/NEU-DET/validation/images"


def create_data_generators():

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_data = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    validation_data = validation_datagen.flow_from_directory(
        VALIDATION_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, validation_data
