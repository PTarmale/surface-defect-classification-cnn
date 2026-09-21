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
    # Evaluate Final Modified CNN
    # -----------------------------

    print("\nEvaluating Final Modified CNN...")

    final_model = load_model(
        "final_surface_defect_cnn.keras"
    )

    final_results = evaluate_model(
        final_model,
        validation_data
    )

    print("\nFinal Modified CNN Classification Report:")
    print(final_results[4])

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

        "Final Modified CNN": [
            final_results[0],
            final_results[1],
            final_results[2],
            final_results[3]
        ]
    })

    print("\nModel Comparison:")
    print(comparison)

    # Save comparison
    comparison.to_csv(
        "final_model_comparison.csv",
        index=False
    )

    # Save reports
    with open(
        "baseline_classification_report.txt",
        "w"
    ) as file:
        file.write(baseline_results[4])

    with open(
        "final_classification_report.txt",
        "w"
    ) as file:
        file.write(final_results[4])

    # Save confusion matrices
    np.savetxt(
        "baseline_confusion_matrix.csv",
        baseline_results[5],
        delimiter=",",
        fmt="%d"
    )

    np.savetxt(
        "final_confusion_matrix.csv",
        final_results[5],
        delimiter=",",
        fmt="%d"
    )

    print("\nEvaluation completed successfully!")
    print("Results saved successfully!")
