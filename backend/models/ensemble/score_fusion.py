import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

class ScoreFusion:
    def __init__(self):
        # Weights start as None
        # Must be set via tune_weights() empirically
        # NEVER hardcode weights
        self.w_lgb  = None
        self.w_iso  = None
        self.w_beh  = None

        # Thresholds start as None
        # Must be set via tune_threshold() empirically
        self.approve_threshold = None
        self.flag_threshold    = None

    def fuse(self, lgb_scores, iso_scores, beh_scores):
        """
        Combines three 0-100 score arrays into single fused score.
        tune_weights() must be called before fuse().
        All inputs must be on 0-100 scale.
        """
        if self.w_lgb is None:
            raise RuntimeError(
                "Weights not set. Call tune_weights() first."
            )
        return (
            np.array(lgb_scores) * self.w_lgb +
            np.array(iso_scores) * self.w_iso +
            np.array(beh_scores) * self.w_beh
        )

    def tune_weights(self, y_val,
                     lgb_val, iso_val, beh_val,
                     step=0.10):
        """
        Grid search over weight combinations on VALIDATION SET ONLY.
        Finds combination that maximises F1 score.
        All inputs on 0-100 scale.

        Grid: weights from 0.1 to 0.8 in steps of 0.10
        Constraint: w_lgb + w_iso + w_beh = 1.0

        Saves weight_tuning_heatmap.png showing F1 per combination.
        Returns best_weights dict, best_f1, results dataframe.
        """
        best_f1      = 0
        best_weights = None
        results      = []

        weight_options = np.arange(0.1, 0.9, step).round(1)

        for w_lgb, w_iso in product(weight_options, weight_options):
            w_beh = round(1.0 - w_lgb - w_iso, 1)
            if w_beh < 0.1 or w_beh > 0.8:
                continue

            fused = (
                np.array(lgb_val) * w_lgb +
                np.array(iso_val) * w_iso +
                np.array(beh_val) * w_beh
            )

            # Find best threshold for this weight combination
            precisions, recalls, thresholds = precision_recall_curve(
                y_val, fused
            )
            f1s = (2 * precisions * recalls /
                  (precisions + recalls + 1e-10))
            best_f1_here = f1s.max()

            results.append({
                'w_lgb': w_lgb,
                'w_iso': w_iso,
                'w_beh': w_beh,
                'f1':    best_f1_here
            })

            if best_f1_here > best_f1:
                best_f1      = best_f1_here
                best_weights = {
                    'w_lgb': w_lgb,
                    'w_iso': w_iso,
                    'w_beh': w_beh
                }

        self.w_lgb = best_weights['w_lgb']
        self.w_iso = best_weights['w_iso']
        self.w_beh = best_weights['w_beh']

        print(f"\n=== OPTIMAL WEIGHTS (validation set) ===")
        print(f"LightGBM:    {self.w_lgb:.2f}")
        print(f"IsoForest:   {self.w_iso:.2f}")
        print(f"Behavioural: {self.w_beh:.2f}")
        print(f"Best F1:     {best_f1:.4f}")

        return best_weights, best_f1, pd.DataFrame(results)

    def tune_threshold(self, y_val, fused_val_scores, out_plots):
        """
        Finds optimal approve threshold on VALIDATION SET ONLY.
        Uses PR curve to maximise F1 on fused scores.
        fused_val_scores on 0-100 scale.
        Saves precision_recall_curve.png.
        Returns best_threshold, precision, recall, f1.
        """
        precisions, recalls, thresholds = precision_recall_curve(
            y_val, fused_val_scores
        )
        f1_scores      = (2 * precisions * recalls /
                         (precisions + recalls + 1e-10))
        best_idx       = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])
        best_precision = float(precisions[best_idx])
        best_recall    = float(recalls[best_idx])
        best_f1        = float(f1_scores[best_idx])

        self.approve_threshold = best_threshold
        self.flag_threshold    = best_threshold + (
            (100 - best_threshold) * 0.5
        )

        print(f"\n=== OPTIMAL THRESHOLD (validation set) ===")
        print(f"Threshold:  {best_threshold:.2f}")
        print(f"Precision:  {best_precision:.2%}")
        print(f"Recall:     {best_recall:.2%}")
        print(f"F1:         {best_f1:.4f}")

        # Plot PR curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(recalls, precisions, color='steelblue',
                linewidth=2, label='Precision-Recall curve')
        ax.scatter(
            best_recall, best_precision,
            color='red', s=150, zorder=5,
            label=(f'Optimal: threshold={best_threshold:.1f}\n'
                   f'P={best_precision:.2%} | '
                   f'R={best_recall:.2%} | '
                   f'F1={best_f1:.3f}')
        )
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Ensemble Precision-Recall Curve\n'
                     '(Threshold tuned on Validation Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_plots, 'precision_recall_curve.png'),
            dpi=150
        )
        plt.close(fig)
        print(f"Saved: {out_plots}/precision_recall_curve.png")

        return best_threshold, best_precision, best_recall, best_f1

    def get_decision(self, score: float) -> str:
        """Maps 0-100 fused score to APPROVE/FLAG/BLOCK."""
        if self.approve_threshold is None:
            raise RuntimeError(
                "Call tune_threshold() before get_decision()"
            )
        if score < self.approve_threshold:
            return 'Approve'
        elif score < self.flag_threshold:
            return 'Flag'
        else:
            return 'Block'

    def get_decisions(self, scores) -> np.ndarray:
        """Batch decision mapping."""
        return np.array([self.get_decision(s) for s in scores])

    def save_config(self, path: str,
                    val_precision: float,
                    val_recall: float,
                    val_f1: float):
        """
        Saves all empirically tuned values to ensemble_config.json.
        All values derived from validation set — never hardcoded.
        """
        config = {
            'weights': {
                'lgb': self.w_lgb,
                'iso': self.w_iso,
                'beh': self.w_beh
            },
            'thresholds': {
                'optimal_threshold': self.approve_threshold,
                'flag_threshold':    self.flag_threshold
            },
            'validation_metrics': {
                'precision': val_precision,
                'recall':    val_recall,
                'f1':        val_f1
            },
            'notes': {
                'weights_source':   'grid search on validation set',
                'threshold_source': 'PR curve argmax F1 on val set',
                'scale':            '0-100 for all model scores',
                'hardcoded_values': 'none — all empirically derived'
            }
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble config saved: {path}")