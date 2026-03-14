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
    print("STEP 7: EVALUATION (TEST SET)")
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

    # 6. Save ALL predicted frauds
    predicted_frauds = df_results[df_results['iso_prediction'] == 1]
    frauds_path = os.path.join(out_results, "predicted_frauds.csv")
    predicted_frauds.to_csv(frauds_path, index=False)
    print(f"Saved ALL predicted frauds ({len(predicted_frauds)} records) to: {frauds_path}")

    return top10

def analyze_bias(iso_forest, X_test: np.ndarray, all_features: list, out_plots: str, out_results: str):
    """
    Analyzes model bias using three methods: Split Frequency, Mean Split Depth, and Permutation Importance.
    """
    print("\n" + "=" * 60)
    print("STEP 8: BIAS DETECTION")
    print("=" * 60)

    # ── METHOD 1: Feature Split Frequency ──
    split_counts = np.zeros(len(all_features))
    for i, tree in enumerate(iso_forest.estimators_):
        # Map back to original indices
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
    colors1 = ['red' if v > 0.15 else 'orange' if v > 0.08 else 'steelblue' for v in split_freq.values]
    ax1.barh(split_freq.index, split_freq.values, color=colors1)
    ax1.axvline(0.15, color='red', linestyle='--', label='Danger threshold (>15%)')
    ax1.axvline(0.08, color='orange', linestyle='--', label='Warning threshold (>8%)')
    ax1.set_xlabel('Split Frequency')
    ax1.set_title('Feature Split Frequency\n(Red = model over-relying on this feature)')
    ax1.legend()
    plt.tight_layout()
    sf_plot_path = os.path.join(out_plots, 'feature_split_frequency.png')
    fig1.savefig(sf_plot_path, dpi=150)
    plt.close(fig1)

    # ── METHOD 2: Mean Split Depth ──
    total_depths  = np.zeros(len(all_features))
    depth_counts  = np.zeros(len(all_features))

    for i, tree in enumerate(iso_forest.estimators_):
        n_nodes        = tree.tree_.node_count
        feature_idx    = tree.tree_.feature
        children_left  = tree.tree_.children_left
        children_right = tree.tree_.children_right
        # Map back to original indices
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
    colors2 = ['red' if v < 3.0 else 'orange' if v < 5.0 else 'steelblue' for v in mean_depths.values]
    ax2.barh(mean_depths.index, mean_depths.values, color=colors2)
    ax2.axvline(3.0, color='red', linestyle='--', label='Danger threshold (depth < 3)')
    ax2.axvline(5.0, color='orange', linestyle='--', label='Warning threshold (depth < 5)')
    ax2.set_xlabel('Mean Split Depth')
    ax2.set_title('Mean Split Depth Per Feature\n(Red = dominates early splits)')
    ax2.legend()
    plt.tight_layout()
    md_plot_path = os.path.join(out_plots, 'feature_mean_depth.png')
    fig2.savefig(md_plot_path, dpi=150)
    plt.close(fig2)

    # ── METHOD 3: Permutation Importance ──
    print("\n=== PERMUTATION IMPORTANCE ===")
    
    # Subsample for speed if test set is large
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
        print(f"  {feature}: impact = {impact:.6f}")

    perm_series = pd.Series(permutation_impact).sort_values(ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors3 = ['red' if v > perm_series.max() * 0.5 else 'orange' if v > perm_series.max() * 0.2 else 'steelblue' for v in perm_series.values]
    ax3.barh(perm_series.index, perm_series.values, color=colors3)
    ax3.set_xlabel('Score Change When Feature Shuffled')
    ax3.set_title('Permutation Importance\n(Red = model collapses without this feature)')
    plt.tight_layout()
    pi_plot_path = os.path.join(out_plots, 'permutation_importance.png')
    fig3.savefig(pi_plot_path, dpi=150)
    plt.close(fig3)

    # ── FINAL DIAGNOSIS ──
    diagnosis_path = os.path.join(out_results, "bias_diagnosis.txt")
    with open(diagnosis_path, "w") as f:
        f.write("=== BIAS DIAGNOSIS ===\n")
        f.write("\nFeatures to WATCH (split freq > 8%):\n")
        f.write(split_freq[split_freq > 0.08].to_string())
        f.write("\n\nFeatures dominating EARLY splits (mean depth < 5):\n")
        f.write(mean_depths[mean_depths < 5.0].to_string())
        f.write("\n\nFeatures model CANNOT live without (top 3 permutation):\n")
        f.write(perm_series.head(3).to_string())
    
    print(f"\nSaved diagnosis to: {diagnosis_path}")
    return split_freq, mean_depths, perm_series
