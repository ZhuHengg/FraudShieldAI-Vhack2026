import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_predictions(y_true: np.ndarray, iso_predictions: np.ndarray, df_filtered: pd.DataFrame, out_plots: str, out_results: str) -> pd.DataFrame:
    """
    Evaluates the model by generating classification reports, a confusion matrix heatmap, 
    risk score distributions, and saving top 10 anomalies.
    """
    print("=" * 60)
    print("STEP 6: EVALUATION")
    print("=" * 60)

    # 1. Classification report
    report_text = classification_report(
        y_true, iso_predictions,
        target_names=["Legitimate", "Fraud"],
    )
    print(report_text)
    
    # Save Report
    report_path = os.path.join(out_results, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")

    # 2. Confusion matrix heatmap
    cm = confusion_matrix(y_true, iso_predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Isolation Forest — Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = os.path.join(out_plots, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_plot_path}")

    # 3. Risk score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df_filtered.loc[df_filtered["isFraud"] == 0, "iso_risk_score"],
        bins=80, alpha=0.6, label="Legitimate", color="#2196F3",
    )
    ax.hist(
        df_filtered.loc[df_filtered["isFraud"] == 1, "iso_risk_score"],
        bins=80, alpha=0.7, label="Fraud", color="#F44336",
    )
    ax.axvline(70, color="orange", ls="--", lw=1.2, label="Approve / Flag boundary (70)")
    ax.axvline(92, color="red",    ls="--", lw=1.2, label="Flag / Block boundary (92)")
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Risk Score Distribution by Class")
    ax.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(out_plots, "risk_score_distribution.png")
    fig.savefig(dist_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_plot_path}")

    # 4. Top 10 most anomalous transactions
    top10 = df_filtered.nlargest(10, "iso_risk_score")
    print("\nTop 10 Most Anomalous Transactions:")
    print(top10.to_string(index=False))

    top10_path = os.path.join(out_results, "top10_anomalies.csv")
    top10.to_csv(top10_path, index=False)
    print(f"Saved: {top10_path}")

    return top10
