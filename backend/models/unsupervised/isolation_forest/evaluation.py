import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_predictions(y_true: np.ndarray, iso_predictions: np.ndarray, df_results: pd.DataFrame, df_model: pd.DataFrame, all_features: list, out_plots: str, out_results: str) -> pd.DataFrame:
    """
    Evaluates the model by generating classification reports, a confusion matrix heatmap, 
    risk score distributions, a feature correlation heatmap, and returning the tier summary.
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
        df_results.loc[df_results["is_fraud"] == 0, "iso_risk_score"],
        bins=80, alpha=0.6, label="Legitimate", color="#2196F3",
    )
    ax.hist(
        df_results.loc[df_results["is_fraud"] == 1, "iso_risk_score"],
        bins=80, alpha=0.7, label="Fraud", color="#F44336",
    )
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Risk Score Distribution by Class")
    ax.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(out_plots, "risk_score_distribution.png")
    fig.savefig(dist_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_plot_path}")

    # 4. Feature Correlation Heatmap
    # Combine features with target and risk score for correlation analysis
    corr_df = df_model[all_features].copy()
    corr_df['is_fraud'] = y_true
    corr_df['iso_risk_score'] = df_results['iso_risk_score']
    
    corr = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
        square=True, linewidths=.5, annot=True, fmt=".2f", 
        annot_kws={"size": 8}, cbar_kws={"shrink": .5}, ax=ax
    )
    
    ax.set_title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    corr_plot_path = os.path.join(out_plots, "correlation_heatmap.png")
    fig.savefig(corr_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {corr_plot_path}")

    # 5. Top anomalies
    top10 = df_results.nlargest(10, "iso_risk_score")
    top10_path = os.path.join(out_results, "top10_anomalies.csv")
    top10.to_csv(top10_path, index=False)
    print(f"Saved: {top10_path}")

    return top10
