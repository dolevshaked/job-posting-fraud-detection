import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import json

def save_confusion_matrix_plot(y_true, y_pred, output_path):
    """
    save confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Confusion matrix plot saved to:", output_path)


def save_roc_curve_plot(model, X_test, y_test, output_path, model_name="Model"):
    """
    save ROC curve for the best model.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} AUC = {auc_score:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("ROC curve plot saved to:", output_path)


def save_model_comparison_table(model_results, output_path):
    """
    Save model comparison results as CSV.
    """
    results_df = pd.DataFrame(model_results)
    results_df.to_csv(output_path, index=False)



def save_model_comparison_results(model_comp_results, output_path):
    """
    save model comparison results to a text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(model_comp_results, indent=4, default=str))

    print("Model comparison results saved to:", output_path)


def evaluate_model(model, X_test, y_test):
    """
    evaluate a selected model on the test set.
    ARGS:
        model: the fitted model to evaluate
        X_test: the test set features
        y_test: the test set target
    RETURNS:
        results: a dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        results["roc_auc"] = roc_auc_score(y_test, y_proba)

    print("Evaluation results on test set:")
    for metric_name, value in results.items():
        print(f"{metric_name}: {value:.3f}")

    return results