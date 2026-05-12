"""Metricas multilabel pedidas por la pauta del laboratorio."""

import numpy as np
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_multilabel_metrics(y_true, y_pred):
    """Calcula las métricas multilabel pedidas por la pauta.
    
    y_true, y_pred: np.ndarray shape (N, K), valores 0/1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    return {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
