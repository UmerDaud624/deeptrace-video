import traceback
import numpy as np

try:
    from src.evaluation.metrics import compute_metrics, compute_eer
    print("import OK")

    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    probs  = np.array([0.05, 0.1, 0.08, 0.92, 0.88, 0.95, 0.79, 0.85])

    metrics = compute_metrics(labels, probs)
    print("AUC :", metrics["auc_roc"])
    print("EER :", metrics["eer"])
    print("F1  :", metrics["f1"])
    print("TP/FP/TN/FN:", metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])
except Exception:
    traceback.print_exc()