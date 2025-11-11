import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve, auc,
                             brier_score_loss, matthews_corrcoef, confusion_matrix)

def compute_metrics(y_true, y_pred, y_proba=None):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['precision'] = precision_score(y_true, y_pred, zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, zero_division=0)
    m['f1'] = f1_score(y_true, y_pred, zero_division=0)
    m['mcc'] = matthews_corrcoef(y_true, y_pred)
    if y_proba is not None:
        try:
            m['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            m['roc_auc'] = None
        p, r, _ = precision_recall_curve(y_true, y_proba)
        m['pr_auc'] = auc(r, p)
        m['brier'] = brier_score_loss(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    m.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return m
