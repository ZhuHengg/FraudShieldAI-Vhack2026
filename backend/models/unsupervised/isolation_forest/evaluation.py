import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve   # CHANGED: added precision_recall_curve
)

# CHANGED: added tune_threshold function
def tune_threshold(
    y_val: np.ndarray,
    val_risk_scores: np.ndarray,
    out_plots: str
) -> tuple:
    """
    Finds the optimal decision threshold by maximising F1 score
    on the validation set using the Precision-Recall curve.

    Called BEFORE evaluation on test set.
    Threshold is never derived from test data.

    Returns:
        best_threshold  — score cutoff that maximises F1
        best_precision  — precision at optimal threshold
        best_recall     — recall at optimal threshold
        best_f1         — F1 score at optimal threshold
    """
    print("\n" + "=" * 60)
    print("STEP 7: THRESHOLD TUNING (VALIDATION SET)")
    print("=" * 60)

    # Compute PR curve on validation set only
    # val_risk_scores are 0-100 normalized anomaly scores
    # used as probability proxy — higher = more suspicious
    precisions, recalls, thresholds = precision_recall_curve(
        y_val,
        val_risk_scores
    )

    # Compute F1 at every threshold point
    f1_scores = (
        2 * (precisions * recalls) /
        (precisions + recalls + 1e-10)
    )

    # Find threshold that maximises F1
    best_idx       = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1_scores[best_idx])

    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Precision:         {best_precision:.2%}")
    print(f"Recall:            {best_recall:.2%}")
    print(f"F1 Score:          {best_f1:.4f}")

    # Show precision at different recall targets
    # Useful for understanding the tradeoff before committing
    print(f"\nPrecision at different recall targets (validation):")
    for target_recall in [0.95, 0.90, 0.85, 0.80, 0.70]:
        idx = np.argmin(np.abs(recalls - target_recall))
        if idx < len(thresholds):
            print(f"  Recall={target_recall:.0%}: "
                  f"precision={precisions[idx]:.2%}  "
                  f"threshold={thresholds[idx]:.2f}  "
                  f"F1={f1_scores[idx]:.3f}")

    # Plot PR curve
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
    iso_predictions: np.ndarray,
    df_results: pd.DataFrame,
    df_model: pd.DataFrame,
    all_features: list,
    out_plots: str,
    out_results: str
) -> pd.DataFrame:
    """
    Evaluates the model by generating classification reports, a confusion matrix heatmap,
    risk score distributions, a feature correlation heatmap, and returning the tier summary.
    """
    # CHANGED: step number bumped from 7 to 8 — threshold tuning is now step 7
    print("=" * 60)
    print("STEP 8: EVALUATION (TEST SET)")
    print("=" * 60)

    # 1. Classification report
    report_text = classification_report(
        y_true, iso_predictions,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
    print(report_text)

    report_path = os.path.join(out_results, "classification_report.txt")
    with open(report_path, "w") as f:
        # CHANGED: document leakage controls in report header
        f.write("Isolation Forest — Synthetic ASEAN E-Wallet Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write("LEAKAGE CONTROLS APPLIED:\n")
        f.write("  Split:               train(64%) / val(16%) / test(20%)\n")
        f.write("  recipient_risk_score: computed from train only\n")
        f.write("  Clip bounds:          computed from train only\n")
        f.write("  StandardScaler:       fitted on train only\n")
        f.write("  MinMaxScaler:         fitted on train only\n")
        f.write("  Threshold tuning:     validation set only\n")
        f.write("  Final evaluation:     test set only\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_text)
    print(f"Saved: {report_path}")

    # 2. Confusion matrix
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
    ax.set_title("Isolation Forest — Confusion Matrix\n(Evaluated on Held-Out Test Set)")
    plt.tight_layout()
    cm_plot_path = os.path.join(out_plots, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_plot_path}")

    # 3. ROC Curve & AUC Score
    fpr, tpr, _ = roc_curve(y_true, df_results["iso_risk_score"] / 100.0)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
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

    # Update report with AUC
    with open(report_path, "a") as f:
        f.write(f"\nROC AUC Score: {roc_auc:.4f}\n")

    # 4. Risk score distribution
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
    ax.set_title("Risk Score Distribution by Class (Test Set)")
    ax.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(out_plots, "risk_score_distribution.png")
    fig.savefig(dist_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {dist_plot_path}")

    # 5. Feature correlation heatmap
    corr_df = df_model[all_features].copy()
    corr_df['is_fraud']      = y_true
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
    ax.set_title("Feature Correlation Heatmap (Test Set)", fontsize=16)
    plt.tight_layout()
    corr_plot_path = os.path.join(out_plots, "correlation_heatmap.png")
    fig.savefig(corr_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {corr_plot_path}")

    # 6. Top anomalies
    top10 = df_results.nlargest(10, "iso_risk_score")
    top10_path = os.path.join(out_results, "top10_anomalies.csv")
    top10.to_csv(top10_path, index=False)
    print(f"Saved: {top10_path}")

    # 7. Save all predicted frauds
    predicted_frauds = df_results[df_results['iso_prediction'] == 1]
    frauds_path = os.path.join(out_results, "predicted_frauds.csv")
    predicted_frauds.to_csv(frauds_path, index=False)
    print(f"Saved {len(predicted_frauds):,} predicted frauds to: {frauds_path}")

    return top10


def analyze_bias(
    iso_forest,
    X_test: np.ndarray,
    all_features: list,
    out_plots: str,
    out_results: str
):
    # unchanged — no modifications needed
    print("\n" + "=" * 60)
    print("STEP 9: BIAS DETECTION")
    print("=" * 60)

    split_counts = np.zeros(len(all_features))
    for i, tree in enumerate(iso_forest.estimators_):
        feature_mapping = iso_forest.estimators_features_[i]
        feature_indices = tree.tree_.feature
        for idx in feature_indices:
            if idx >= 0:
                original_idx = feature_mapping[idx]
                split_counts[original_idx] += 1

    split_freq = pd.Series(
        split_counts / split_counts.sum(),
        index=all_features
    ).sort_values(ascending=False)

    print("=== FEATURE SPLIT FREQUENCY ===")
    print(split_freq.round(4))

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors1 = [
        'red' if v > 0.15 else 'orange' if v > 0.08
        else 'steelblue' for v in split_freq.values
    ]
    ax1.barh(split_freq.index, split_freq.values, color=colors1)
    ax1.axvline(0.15, color='red', linestyle='--',
                label='Danger threshold (>15%)')
    ax1.axvline(0.08, color='orange', linestyle='--',
                label='Warning threshold (>8%)')
    ax1.set_xlabel('Split Frequency')
    ax1.set_title('Feature Split Frequency\n'
                  '(Red = model over-relying on this feature)')
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(out_plots, 'feature_split_frequency.png'), dpi=150)
    plt.close(fig1)

    total_depths = np.zeros(len(all_features))
    depth_counts = np.zeros(len(all_features))

    for i, tree in enumerate(iso_forest.estimators_):
        n_nodes        = tree.tree_.node_count
        feature_idx    = tree.tree_.feature
        children_left  = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature_mapping = iso_forest.estimators_features_[i]

        node_depth = np.zeros(n_nodes, dtype=int)
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id],  depth + 1))
                stack.append((children_right[node_id], depth + 1))

        for node_id in range(n_nodes):
            feat = feature_idx[node_id]
            if feat >= 0:
                original_idx = feature_mapping[feat]
                total_depths[original_idx] += node_depth[node_id]
                depth_counts[original_idx] += 1

    mean_depths = pd.Series(
        np.where(depth_counts > 0, total_depths / depth_counts, np.nan),
        index=all_features
    ).sort_values(ascending=True)

    print("\n=== MEAN SPLIT DEPTH ===")
    print(mean_depths.round(2))

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors2 = [
        'red' if v < 3.0 else 'orange' if v < 5.0
        else 'steelblue' for v in mean_depths.values
    ]
    ax2.barh(mean_depths.index, mean_depths.values, color=colors2)
    ax2.axvline(3.0, color='red', linestyle='--',
                label='Danger threshold (depth < 3)')
    ax2.axvline(5.0, color='orange', linestyle='--',
                label='Warning threshold (depth < 5)')
    ax2.set_xlabel('Mean Split Depth')
    ax2.set_title('Mean Split Depth Per Feature\n'
                  '(Red = dominates early splits)')
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(out_plots, 'feature_mean_depth.png'), dpi=150)
    plt.close(fig2)

    print("\n=== PERMUTATION IMPORTANCE ===")
    if len(X_test) > 50000:
        indices = np.random.choice(len(X_test), 50000, replace=False)
        X_eval = X_test[indices]
    else:
        X_eval = X_test

    baseline_scores = -iso_forest.decision_function(X_eval)
    baseline_mean   = baseline_scores.mean()

    permutation_impact = {}
    for i, feature in enumerate(all_features):
        X_permuted       = X_eval.copy()
        X_permuted[:, i] = np.random.permutation(X_eval[:, i])
        permuted_scores  = -iso_forest.decision_function(X_permuted)
        impact = abs(permuted_scores.mean() - baseline_mean)
        permutation_impact[feature] = impact
        print(f"  {feature}: {impact:.6f}")

    perm_series = pd.Series(
        permutation_impact
    ).sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors3 = [
        'red' if v > perm_series.max() * 0.5
        else 'orange' if v > perm_series.max() * 0.2
        else 'steelblue' for v in perm_series.values
    ]
    ax3.barh(perm_series.index, perm_series.values, color=colors3)
    ax3.set_xlabel('Score Change When Feature Shuffled')
    ax3.set_title('Permutation Importance\n'
                  '(Red = model collapses without this feature)')
    plt.tight_layout()
    fig3.savefig(
        os.path.join(out_plots, 'permutation_importance.png'), dpi=150
    )
    plt.close(fig3)

    diagnosis_path = os.path.join(out_results, "bias_diagnosis.txt")
    with open(diagnosis_path, "w") as f:
        f.write("=== BIAS DIAGNOSIS ===\n")
        f.write("\nFeatures to WATCH (split freq > 8%):\n")
        f.write(split_freq[split_freq > 0.08].to_string())
        f.write("\n\nFeatures dominating EARLY splits (mean depth < 5):\n")
        f.write(mean_depths[mean_depths < 5.0].to_string())
        f.write("\n\nFeatures model CANNOT live without (top 3 permutation):\n")
        f.write(perm_series.head(3).to_string())
    print(f"\nSaved: {diagnosis_path}")

    return split_freq, mean_depths, perm_series