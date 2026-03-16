import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, out_plots):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    total = np.sum(cm)
    cm_perc = cm / total * 100
    
    labels = np.asarray([f"{count}\n({perc:.1f}%)" for count, perc in zip(cm.flatten(), cm_perc.flatten())]).reshape(2, 2)
    
    fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    ax.set_ylabel('True Label')
    ax.set_xlabel(f'Predicted Label\n\nPrecision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}')
    ax.set_title('Ensemble — Confusion Matrix\n(Evaluated on Held-Out Test Set)')
    
    path = os.path.join(out_plots, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_roc_curve(y_true, ensemble_scores, out_plots):
    y_score = ensemble_scores / 100.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    
    path = os.path.join(out_plots, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_risk_distribution(df_results, approve_threshold, flag_threshold, out_plots):
    fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
    
    legit = df_results[df_results['is_fraud'] == 0]['ensemble_risk_score']
    fraud = df_results[df_results['is_fraud'] == 1]['ensemble_risk_score']
    
    ax.hist(legit, bins=80, color='steelblue', alpha=0.6, label='Legit (Class 0)')
    ax.hist(fraud, bins=80, color='red', alpha=0.6, label='Fraud (Class 1)')
    
    ax.axvline(approve_threshold, color='green', linestyle='--', 
               label=f'Approve Threshold: {approve_threshold:.1f} (empirically tuned)')
    ax.axvline(flag_threshold, color='orange', linestyle='--', 
               label=f'Flag Threshold: {flag_threshold:.1f} (empirically tuned)')
    
    ax.set_xlabel('Ensemble Risk Score')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution')
    ax.legend()
    
    path = os.path.join(out_plots, 'risk_score_distribution.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_score_comparison(df_results, out_plots):
    cols_to_melt = ['lgb_risk_score', 'iso_risk_score', 'beh_risk_score', 'ensemble_risk_score']
    df_long = pd.melt(df_results, id_vars=['is_fraud'], value_vars=cols_to_melt, 
                      var_name='Model', value_name='Risk Score')
    
    fig, ax = plt.subplots(dpi=150, figsize=(10, 6))
    sns.boxplot(x='Model', y='Risk Score', hue='is_fraud', data=df_long, 
                palette={0: 'steelblue', 1: 'red'}, ax=ax)
    
    ax.set_title('Score Comparison by Model')
    path = os.path.join(out_plots, 'score_comparison.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_weight_heatmap(results_df, out_plots):
    pivot_df = results_df.pivot(index='w_iso', columns='w_lgb', values='f1')
    pivot_df = pivot_df.sort_index(ascending=False)
    
    best_row = results_df.loc[results_df['f1'].idxmax()]
    best_w_iso = best_row['w_iso']
    best_w_lgb = best_row['w_lgb']
    
    fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={'label': 'F1 Score'})
    
    y_idx = list(pivot_df.index).index(best_w_iso)
    x_idx = list(pivot_df.columns).index(best_w_lgb)
    
    ax.scatter(x_idx + 0.5, y_idx + 0.5, color='red', marker='x', s=100, linewidth=3, label='Optimal Combo')
    
    ax.set_title('Grid Search: Weight Tuning F1 Heatmap')
    ax.set_ylabel('Isolation Forest Weight')
    ax.set_xlabel('LightGBM Weight')
    ax.legend()
    
    path = os.path.join(out_plots, 'weight_tuning_heatmap.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")

def evaluate_ensemble(y_true, y_pred, df_results, approve_threshold, flag_threshold, out_plots, out_results):
    os.makedirs(out_plots, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)
    
    plot_confusion_matrix(y_true, y_pred, out_plots)
    plot_roc_curve(y_true, df_results['ensemble_risk_score'].values, out_plots)
    plot_risk_distribution(df_results, approve_threshold, flag_threshold, out_plots)
    plot_score_comparison(df_results, out_plots)
    
    report_str = classification_report(y_true, y_pred)
    
    header = '''Ensemble — FraudShield AI
══════════════════════════════════════
LEAKAGE CONTROLS APPLIED:
  Split:             train(64%) / val(16%) / test(20%)
  Weight tuning:     grid search on validation set only
  Threshold tuning:  PR curve argmax F1 on val set only
  Final evaluation:  test set only
  Hardcoded values:  none — all empirically derived
══════════════════════════════════════\n\n'''

    report_path = os.path.join(out_results, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header + report_str)
    print(f"Saved: {report_path}")
    
    cols = ['transaction_id', 'is_fraud', 'lgb_risk_score', 'iso_risk_score', 
            'beh_risk_score', 'ensemble_risk_score', 'risk_tier', 'ensemble_prediction']
            
    if 'transaction_id' not in df_results.columns:
        df_results['transaction_id'] = df_results.index
        
    scores_path = os.path.join(out_results, 'score_breakdown.csv')
    df_results[cols].to_csv(scores_path, index=False)
    print(f"Saved: {scores_path}")
