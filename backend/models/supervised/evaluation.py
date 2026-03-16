import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

def tune_threshold(
    y_val: np.ndarray,
    val_risk_scores: np.ndarray,
    out_plots: str
) -> tuple:
    """
    Finds the optimal decision threshold by maximising F1 score
    on the validation set using the Precision-Recall curve.

    val_risk_scores is 0-100 scale.
    This is the ONLY legitimate source of threshold values.
    Returns best_threshold, best_precision, best_recall, best_f1
    """
    print("\n" + "=" * 60)
    print("STEP 6: THRESHOLD TUNING (VALIDATION SET)")
    print("=" * 60)

    precisions, recalls, thresholds = precision_recall_curve(
        y_val,
        val_risk_scores
    )

    f1_scores = (
        2 * (precisions * recalls) /
        (precisions + recalls + 1e-10)
    )

    best_idx       = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1_scores[best_idx])

    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Precision:         {best_precision:.2%}")
    print(f"Recall:            {best_recall:.2%}")
    print(f"F1 Score:          {best_f1:.4f}")

    print(f"\nPrecision at different recall targets (validation):")
    for target_recall in [0.95, 0.90, 0.85, 0.80, 0.70]:
        idx = np.argmin(np.abs(recalls - target_recall))
        if idx < len(thresholds):
            print(f"  Recall={target_recall:.0%}: "
                  f"precision={precisions[idx]:.2%}  "
                  f"threshold={thresholds[idx]:.2f}  "
                  f"F1={f1_scores[idx]:.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        recalls, precisions,
        color='steelblue', linewidth=2,
        label='Precision-Recall curve'
    )
    ax.scatter(
        best_recall, best_precision,
        color='red', s=150, zorder=5,
        label=(
            f'Optimal: threshold={best_threshold:.1f}\n'
            f'P={best_precision:.2%} | '
            f'R={best_recall:.2%} | '
            f'F1={best_f1:.3f}'
        )
    )
    ax.axhline(
        0.50, color='orange', linestyle='--',
        alpha=0.7, label='Precision=50% reference'
    )
    ax.axhline(
        0.20, color='grey', linestyle='--',
        alpha=0.5, label='Baseline precision=20%'
    )
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve\n(Threshold tuned on Validation Set)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pr_plot_path = os.path.join(out_plots, 'precision_recall_curve.png')
    fig.savefig(pr_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {pr_plot_path}")

    return best_threshold, best_precision, best_recall, best_f1


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: np.ndarray,
    df_results: pd.DataFrame,
    all_features: list,
    out_plots: str,
    out_results: str
):
    print("=" * 60)
    print("STEP 8: EVALUATION (TEST SET)")
    print("=" * 60)

    # Classification report
    report_text = classification_report(
        y_true, predictions,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
    print(report_text)

    report_path = os.path.join(out_results, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Supervised LightGBM Pipeline\n")
        f.write("=" * 50 + "\n\n")
        f.write("LEAKAGE CONTROLS APPLIED:\n")
        f.write("  Split:                  train(64%) / val(16%) / test(20%)\n")
        f.write("  recipient_risk_score:   computed from train only\n")
        f.write("  StandardScaler:         fitted on original train (pre-SMOTE)\n")
        f.write("  SMOTE:                  applied to train only\n")
        f.write("  Threshold tuning:       empirically via PR curve on val set\n")
        f.write("                          optimal_threshold = argmax(F1)\n")
        f.write("                          NOT manually set\n")
        f.write("  Final evaluation:       test set only\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_text)
    print(f"Saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    cm_percent = cm / cm.sum() * 100
    
    # Custom annotations combining count and percentage
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    
    # Calculate precision, recall, f1, fpr manually
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics_str = f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}"
    
    ax.set_xlabel(f"Predicted\n\n{metrics_str}")
    ax.set_ylabel("Actual")
    ax.set_title("LightGBM — Confusion Matrix\n(Evaluated on Held-Out Test Set)")
    plt.tight_layout()
    cm_plot_path = os.path.join(out_plots, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_plot_path}")

    # ROC curve
    fpr_roc, tpr_roc, _ = roc_curve(y_true, df_results["lgb_risk_score"] / 100.0)
    roc_auc = auc(fpr_roc, tpr_roc)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_roc, tpr_roc, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(out_plots, "roc_curve.png")
    fig.savefig(roc_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {roc_plot_path}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Add AUC to report.
    with open(report_path, "a") as f:
        f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")

    # Risk score distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df_results.loc[df_results["is_fraud"] == 0, "lgb_risk_score"],
        bins=80, alpha=0.6, label="Legitimate", color="steelblue", density=True
    )
    ax.hist(
        df_results.loc[df_results["is_fraud"] == 1, "lgb_risk_score"],
        bins=80, alpha=0.7, label="Fraud", color="red", density=True
    )
    
    approve_t = df_results['lgb_risk_score'][df_results['risk_tier'] == 'Approve'].max(skipna=True)
    flag_t = df_results['lgb_risk_score'][df_results['risk_tier'] == 'Flag'].max(skipna=True)
    
    # Get thresholds safely from dataframe distribution map if possible, 
    # Or ideally, directly from the lgb_model thresholds, here we pull from max bound.
    if pd.isna(flag_t): # defaults in case of empty sets
         flag_t = approve_t + (100-approve_t) * 0.5 

    if not pd.isna(approve_t):
        ax.axvline(approve_t, color='orange', linestyle='--', lw=2, label=f'Approve Threshold: {approve_t:.2f} (Empirically Tuned)')
    if not pd.isna(flag_t):
        ax.axvline(flag_t, color='black', linestyle='--', lw=2, label=f'Flag Threshold: {flag_t:.2f}')
    
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_ylabel("Density")
    ax.set_title("LightGBM Risk Score Distribution by Class (Test Set)")
    ax.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(out_plots, "risk_score_distribution.png")
    fig.savefig(dist_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_plot_path}")


    # Top 10 frauds
    top10 = df_results.nlargest(10, "lgb_risk_score")
    top10_path = os.path.join(out_results, "top10_fraud.csv")
    top10.to_csv(top10_path, index=False)
    print(f"Saved: {top10_path}")

    # Save all predicted frauds
    predicted_frauds = df_results[df_results['lgb_prediction'] == 1]
    frauds_path = os.path.join(out_results, "predicted_frauds.csv")
    predicted_frauds.to_csv(frauds_path, index=False)
    print(f"Saved {len(predicted_frauds):,} predicted frauds to: {frauds_path}")

def plot_feature_importance(importance_series: pd.Series, out_plots: str):
    """
    Native LightGBM importances — no SHAP here.
    Horizontal bar chart. Color Top 3 red, 4-8 orange, rest steelblue.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = []
    for i in range(len(importance_series)):
        if i < 3:
            colors.append('red')
        elif i < 8:
            colors.append('orange')
        else:
            colors.append('steelblue')

    importance_series.sort_values(ascending=True).plot(
        kind='barh', ax=ax, color=colors[::-1]
    )
    
    ax.set_xlabel('Importance Score')
    ax.set_title('LightGBM Feature Importance')
    plt.tight_layout()
    
    feat_plot_path = os.path.join(out_plots, 'feature_importance.png')
    fig.savefig(feat_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {feat_plot_path}")
