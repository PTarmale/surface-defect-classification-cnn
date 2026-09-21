import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from preprocessing import IMG_SIZE


def predict_image(
    image_path,
    model_path,
    class_names
):

    # Load trained model
    model = load_model(model_path)

    # Load image
    image = Image.open(image_path).convert("L")

    # Resize image
    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # Convert to array
    image_array = np.array(image) / 255.0

    # Add batch and channel dimensions
    image_input = np.expand_dims(
        image_array,
        axis=0
    )

    image_input = np.expand_dims(
        image_input,
        axis=-1
    )

    # Prediction
    prediction = model.predict(
        image_input,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )

    print("Predicted class:", predicted_class)
    print(
        "Confidence: {:.2f}%".format(
            confidence
        )
    )

    return predicted_class, confidence


if __name__ == "__main__":

    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ]

    image_path = "sample_image.jpg"

    model_path = (
        "baseline_surface_defect_cnn.keras"
    )

    predict_image(
        image_path,
        model_path,
        class_names
    )
