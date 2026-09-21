import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import create_data_generators


def evaluate_model(model, validation_data):

    validation_data.reset()

    predictions = model.predict(
        validation_data,
        verbose=1
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_classes = validation_data.classes

    class_names = list(
        validation_data.class_indices.keys()
    )

    accuracy = accuracy_score(
        true_classes,
        predicted_classes
    )

    precision = precision_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_classes,
        predicted_classes,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(
        true_classes,
        predicted_classes
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        report,
        cm
    )


if __name__ == "__main__":

    # Load validation data
    train_data, validation_data = create_data_generators()

    # -----------------------------
    # Evaluate Baseline CNN
    # -----------------------------

    print("\nEvaluating Baseline CNN...")

    baseline_model = load_model(
        "baseline_surface_defect_cnn.keras"
    )

    baseline_results = evaluate_model(
        baseline_model,
        validation_data
    )

    print("\nBaseline CNN Classification Report:")
    print(baseline_results[4])

    # -----------------------------
    # Evaluate Improved CNN
    # -----------------------------

    print("\nEvaluating Improved CNN...")

    improved_model = load_model(
        "improved_surface_defect_cnn.keras"
    )

    improved_results = evaluate_model(
        improved_model,
        validation_data
    )

    print("\nImproved CNN Classification Report:")
    print(improved_results[4])

    # -----------------------------
    # Model Comparison
    # -----------------------------

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],

        "Baseline CNN": [
            baseline_results[0],
            baseline_results[1],
            baseline_results[2],
            baseline_results[3]
        ],

        "Improved CNN": [
            improved_results[0],
            improved_results[1],
            improved_results[2],
            improved_results[3]
        ]
    })

    print("\nModel Comparison:")
    print(comparison)

    # Save comparison
    comparison.to_csv(
        "model_comparison.csv",
        index=False
    )

    # Save reports
    with open(
        "baseline_classification_report.txt",
        "w"
    ) as file:
        file.write(baseline_results[4])

    with open(
        "improved_classification_report.txt",
        "w"
    ) as file:
        file.write(improved_results[4])

    print("\nEvaluation completed successfully!")
    print("Results saved successfully!")
