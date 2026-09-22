import tensorflow as tf

from preprocessing import create_data_generators
from baseline_model import create_baseline_model
from improved_model import create_improved_model


EPOCHS = 30


def train_baseline():

    train_data, validation_data = create_data_generators()

    model = create_baseline_model()

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS
    )

    model.save("baseline_surface_defect_cnn.keras")

    return model, history


def train_improved():

    train_data, validation_data = create_data_generators()

    model = create_improved_model()

    # Paper-style learning-rate schedule:
    # decrease learning rate by 0.8 every 3 epochs
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr:
        lr * 0.8 if epoch > 0 and epoch % 3 == 0 else lr
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        callbacks=[lr_scheduler]
    )

    model.save("improved_surface_defect_cnn.keras")

    return model, history


if __name__ == "__main__":

    print("Training baseline CNN...")

    baseline_model, baseline_history = train_baseline()

    print("Training SurfNet-based CNN...")

    improved_model, improved_history = train_improved()

    print("Training completed successfully!")
