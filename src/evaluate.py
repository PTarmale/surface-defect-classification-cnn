import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import create_data_generators
from baseline_model import create_baseline_model
from improved_model import create_improved_model


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
        average="weighted"
    )

    recall = recall_score(
        true_classes,
        predicted_classes,
        average="weighted"
    )

    f1 = f1_score(
        true_classes,
        predicted_classes,
        average="weighted"
    )

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )

    cm = confusion_matrix(
        true_classes,
        predicted_classes
    )

    return accuracy, precision, recall, f1, report, cm


if __name__ == "__main__":

    train_data, validation_data = create_data_generators()

    print("\nEvaluating Baseline CNN...")

    baseline_model = create_baseline_model()

    baseline_model.load_weights(
        "baseline_surface_defect_cnn.keras"
    )

    baseline_results = evaluate_model(
        baseline_model,
        validation_data
    )

    print("\nBaseline Classification Report:")
    print(baseline_results[4])

    print("\nEvaluating Improved CNN...")

    improved_model = create_improved_model()

    improved_model.load_weights(
        "improved_surface_defect_cnn.keras"
    )

    improved_results = evaluate_model(
        improved_model,
        validation_data
    )

    print("\nImproved Classification Report:")
    print(improved_results[4])

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

    comparison.to_csv(
        "model_comparison.csv",
        index=False
    )

    print("\nEvaluation completed successfully!")
