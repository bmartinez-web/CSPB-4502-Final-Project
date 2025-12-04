"""
Evaluation utilities: metrics, calibration, plots.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

def compute_metrics(y_true, y_prob, y_pred):
    metrics = {}
    metrics["auroc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["auprc"] = average_precision_score(y_true, y_prob)
    metrics["f1@0.5"] = f1_score(y_true, y_pred)
    metrics["precision@0.5"] = precision_score(y_true, y_pred)
    metrics["recall@0.5"] = recall_score(y_true, y_pred)
    metrics["brier"] = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics

def reliability_plot(y_true, y_prob, out_path: str):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", linestyle="-", label="Model")
    plt.plot([0,1], [0,1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Curve")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def pr_curve(y_true, y_prob, out_path: str):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def roc_curve_plot(y_true, y_prob, out_path: str):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
